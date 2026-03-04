"""
ANÁLISIS ÓPTICO DE VÍDEO TÉRMICO — Streamlit v5
Cambios respecto a v4:
  - BiLSTM numpy puro REEMPLAZADO por FukuzonoLSTM (PyTorch):
      BiLSTM + SelfAttention + cabeza de regresión
      Feature engineering de 14 variables (Fukuzono ratio, rolling stats, etc.)
      Predicción AUTORREGRESIVA hasta 1/v → 0
  - Walk-Forward Validation ELIMINADO
  - Modelo Híbrido ELIMINADO
  - STL visual MANTENIDO
"""

import warnings, tempfile, os
import numpy as np
import cv2
import streamlit as st
from skimage.morphology import remove_small_objects
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ─── PÁGINA ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Flujo Óptico Térmico",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  section[data-testid="stSidebar"] { background-color: #0d1117; }
  .stApp { background-color: #0d1117; color: #cdd9e5; }
  h1, h2, h3 { color: #58a6ff; font-family: monospace; }
  p, label, .stMarkdown { color: #cdd9e5; }
  .block-container { padding-top: 1.5rem; }
  div[data-testid="metric-container"] { background:#161b22; border-radius:8px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ─── PALETA ──────────────────────────────────────────────────────────────────
BG     = "#0d1117"
BG2    = "#161b22"
BORDER = "#30363d"
CYAN   = "#58a6ff"
ORANGE = "#ff8c00"
GREEN  = "#3fb950"
PINK   = "#ff6b9d"
YELLOW = "#f0e130"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(color="#cdd9e5"),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    legend=dict(bgcolor=BG2, bordercolor=BORDER),
    margin=dict(l=60, r=20, t=50, b=50),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════════════════
#  FLUJO ÓPTICO — sin cambios
# ════════════════════════════════════════════════════════════════════════════

def to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_smoke_mask(img_gray, blur_k=21, texture_thresh=3.5, dark_thresh=80):
    h, w = img_gray.shape
    if blur_k % 2 == 0: blur_k += 1
    big_k = blur_k * 3
    if big_k % 2 == 0: big_k += 1
    blurred  = cv2.GaussianBlur(img_gray, (blur_k, blur_k), 0)
    blurred2 = cv2.GaussianBlur(img_gray, (big_k,  big_k),  0)
    local_var = cv2.absdiff(blurred, blurred2).astype(np.float32)
    smoke = (img_gray < dark_thresh) & (local_var < texture_thresh)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    s  = smoke.astype(np.uint8)
    s  = cv2.morphologyEx(s, cv2.MORPH_CLOSE, k)
    s  = cv2.morphologyEx(s, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    sb = s.astype(bool)
    return remove_small_objects(sb, min_size=max(50, int(h * w * 0.005)))


def detect_motion_opencv(g1, g2, smoke, diff_thresh=12.0, min_area=200):
    t1 = cv2.GaussianBlur(g1, (11, 11), 2.0)
    t2 = cv2.GaussianBlur(g2, (11, 11), 2.0)
    diff = cv2.absdiff(t1, t2).astype(np.float32)
    m  = (diff > diff_thresh) & ~smoke
    mu = m.astype(np.uint8)
    mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mu = cv2.morphologyEx(mu, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return remove_small_objects(mu.astype(bool), min_size=min_area), diff


def _warp_flow(img, flow):
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _compute_gradients(img):
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return Ix, Iy


def _correlation_volume(f1, f2, radius=4):
    H, W, C = f1.shape
    d = 2 * radius + 1
    f1n = f1 / (np.linalg.norm(f1, axis=2, keepdims=True) + 1e-8)
    f2n = f2 / (np.linalg.norm(f2, axis=2, keepdims=True) + 1e-8)
    corr = np.zeros((H, W, d * d), np.float32)
    idx = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(f2n, dy, axis=0), dx, axis=1)
            corr[..., idx] = np.sum(f1n * shifted, axis=2)
            idx += 1
    return corr


def _flow_raft_lite(g1, g2, iters=12, corr_radius=4, corr_levels=4,
                    update_iters=6, alpha_smooth=0.5, feature_channels=32,
                    downsample_factor=4):
    h0, w0 = g1.shape
    ds = max(1, downsample_factor)
    h, w = h0 // ds, w0 // ds
    if h < 8 or w < 8:
        ds = 1; h, w = h0, w0

    i1 = cv2.resize(g1.astype(np.float32), (w, h)) / 255.0
    i2 = cv2.resize(g2.astype(np.float32), (w, h)) / 255.0

    def extract_features(img, n_ch):
        feats = [img[..., np.newaxis]]
        sigmas = np.linspace(0.5, 3.0, max(1, n_ch - 1))
        for s in sigmas:
            k = max(3, int(4 * s) | 1)
            feats.append(cv2.GaussianBlur(img, (k, k), s)[..., np.newaxis])
        arr = np.concatenate(feats, axis=2).astype(np.float32)
        return arr[:, :, :n_ch]

    n_ch = max(2, feature_channels // 8)
    feat1 = extract_features(i1, n_ch)
    feat2 = extract_features(i2, n_ch)

    def build_corr_pyramid(f1, f2, n_levels, radius):
        pyr = []
        for lv in range(n_levels):
            scale = 0.5 ** lv
            if lv == 0:
                c = _correlation_volume(f1, f2, radius)
            else:
                f2_down = cv2.resize(f2, (max(1, int(f2.shape[1]*scale)),
                                          max(1, int(f2.shape[0]*scale))))
                f2_down = np.stack([cv2.resize(f2_down[:,:,c_],
                                               (f2.shape[1], f2.shape[0]))
                                    for c_ in range(f2_down.shape[2])], axis=2)
                c = _correlation_volume(f1, f2_down, radius)
            pyr.append(c)
        return pyr

    corr_pyr = build_corr_pyramid(feat1, feat2,
                                   min(corr_levels, 3), min(corr_radius, 3))
    Ix, Iy = _compute_gradients(i1)
    flow = np.zeros((h, w, 2), np.float32)

    for iteration in range(iters):
        warped2 = _warp_flow(i2, flow)
        It = (warped2 - i1).astype(np.float32)
        feat2_warped = extract_features(warped2, n_ch)
        corr_now = _correlation_volume(feat1, feat2_warped, min(corr_radius, 3))
        conf = np.max(corr_now, axis=2)
        conf = cv2.GaussianBlur(conf, (5, 5), 1.0)
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        denom = Ix**2 + Iy**2 + 1e-6
        df_x  = -It * Ix / denom
        df_y  = -It * Iy / denom
        lr_iter = 1.0 / (update_iters + 1)
        flow[..., 0] = flow[..., 0] + lr_iter * conf * df_x
        flow[..., 1] = flow[..., 1] + lr_iter * conf * df_y
        if alpha_smooth > 0:
            ks = max(3, min(11, int(alpha_smooth * 10) | 1))
            flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (ks, ks), alpha_smooth)
            flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (ks, ks), alpha_smooth)
        max_disp = min(h, w) * 0.3
        flow = np.clip(flow, -max_disp, max_disp)

    if ds > 1:
        flow_up = np.stack([
            cv2.resize(flow[..., c_], (w0, h0), interpolation=cv2.INTER_LINEAR) * ds
            for c_ in range(2)
        ], axis=2)
    else:
        flow_up = flow

    return flow_up.astype(np.float32)


def _flow_dis(g1, g2,
              preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
              finest_scale=1, grad_desc_iters=25,
              variational_refinement_iters=5,
              variational_refinement_alpha=20.0,
              variational_refinement_gamma=10.0,
              variational_refinement_delta=5.0,
              use_mean_normalization=True,
              use_spatial_propagation=True):
    dis = cv2.DISOpticalFlow_create(preset)
    dis.setFinestScale(finest_scale)
    dis.setGradientDescentIterations(grad_desc_iters)
    dis.setVariationalRefinementIterations(variational_refinement_iters)
    dis.setVariationalRefinementAlpha(variational_refinement_alpha)
    dis.setVariationalRefinementGamma(variational_refinement_gamma)
    dis.setVariationalRefinementDelta(variational_refinement_delta)
    dis.setUseMeanNormalization(use_mean_normalization)
    dis.setUseSpatialPropagation(use_spatial_propagation)
    return dis.calc(g1, g2, None)


def _flow_lk_pyramid(g1, g2, win_size=21, max_level=4, max_corners=500,
                     quality_level=0.01, min_distance=7, block_size=7,
                     back_threshold=1.0, eigen_threshold=1e-4):
    h, w = g1.shape
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p0 = cv2.goodFeaturesToTrack(g1, maxCorners=max_corners,
                                  qualityLevel=quality_level,
                                  minDistance=min_distance, blockSize=block_size)
    flow_dense = np.zeros((h, w, 2), np.float32)
    if p0 is None or len(p0) == 0:
        return flow_dense
    lk_params = dict(winSize=(win_size, win_size), maxLevel=max_level,
                     criteria=criteria, minEigThreshold=eigen_threshold)
    p1, st, err = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, **lk_params)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, p1, None, **lk_params)
    fb_err = np.linalg.norm(p0r - p0, axis=2).squeeze()
    good = (st.squeeze() == 1) & (st2.squeeze() == 1) & (fb_err < back_threshold)
    pts0 = p0[good].reshape(-1, 2)
    pts1 = p1[good].reshape(-1, 2)
    if len(pts0) < 4:
        return flow_dense
    disps = pts1 - pts0
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    gxy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts0)
        K = min(8, len(pts0))
        dists, idxs = tree.query(gxy, k=K, workers=1)
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        flow_x = (weights * disps[idxs, 0]).sum(axis=1).reshape(h, w)
        flow_y = (weights * disps[idxs, 1]).sum(axis=1).reshape(h, w)
    except ImportError:
        flow_x = np.zeros(h * w, np.float32)
        flow_y = np.zeros(h * w, np.float32)
        for pi, gp in enumerate(gxy):
            d = np.linalg.norm(pts0 - gp, axis=1)
            ni = np.argmin(d)
            flow_x[pi] = disps[ni, 0]
            flow_y[pi] = disps[ni, 1]
        flow_x = flow_x.reshape(h, w)
        flow_y = flow_y.reshape(h, w)
    flow_dense[..., 0] = flow_x
    flow_dense[..., 1] = flow_y
    return flow_dense


def _flow_farneback(g1, g2, pyr_scale=0.5, levels=5, winsize=15,
                    iterations=3, poly_n=7, poly_sigma=1.5, use_gaussian=True):
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian else 0
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
        flags=flags)


def compute_optical_flow(g1, g2, motion, smoke, algo="RAFT-lite", params=None):
    if params is None:
        params = {}
    valid = motion & ~smoke
    if algo == "RAFT-lite":
        flow = _flow_raft_lite(g1, g2, **params)
    elif algo == "DIS":
        preset_map = {
            "Ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "Fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "Medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        p = params.copy()
        p["preset"] = preset_map.get(p.get("preset", "Medium"),
                                      cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = _flow_dis(g1, g2, **p)
    elif algo == "LK-Pyramid":
        flow = _flow_lk_pyramid(g1, g2, **params)
    elif algo == "Farneback":
        flow = _flow_farneback(g1, g2, **params)
    else:
        flow = _flow_farneback(g1, g2)
    if flow.shape[:2] != g1.shape[:2]:
        flow = np.stack([
            cv2.resize(flow[..., c], (g1.shape[1], g1.shape[0]))
            for c in range(2)
        ], axis=2)
    flow[~valid] = np.nan
    return flow.astype(np.float32), valid


def mean_displacement(flow, valid):
    fc = flow[valid]
    if len(fc) == 0: return 0.0
    return float(np.nanmean(np.hypot(fc[:, 0], fc[:, 1])))


def flow_to_hsv_color(flow):
    fv = flow.copy(); fv[np.isnan(fv)] = 0
    mag, ang = cv2.cartToPolar(fv[..., 0], fv[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ════════════════════════════════════════════════════════════════════════════
#  PROCESAMIENTO DE SEÑAL — sin cambios
# ════════════════════════════════════════════════════════════════════════════

def apply_outlier_filter(signal, method="IQR", iqr_k=1.5, zscore_thr=3.0,
                          clip_min=None, clip_max=None, replace="interpolate"):
    s = signal.astype(float).copy()
    n = len(s)
    if n < 4:
        return s
    if method == "IQR":
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        mask_out = (s < lo) | (s > hi)
    elif method == "Z-score":
        mu, sigma = s.mean(), s.std() + 1e-12
        mask_out = np.abs((s - mu) / sigma) > zscore_thr
    elif method == "Rango manual":
        lo = clip_min if clip_min is not None else s.min()
        hi = clip_max if clip_max is not None else s.max()
        mask_out = (s < lo) | (s > hi)
    else:
        mask_out = np.zeros(n, dtype=bool)
    if not mask_out.any():
        return s
    s[mask_out] = np.nan
    if replace == "interpolate":
        idx = np.arange(n)
        good = ~mask_out
        if good.sum() >= 2:
            s = np.interp(idx, idx[good], s[good])
        else:
            s = np.where(np.isnan(s), np.nanmedian(s), s)
    elif replace == "mediana":
        med = np.nanmedian(s)
        s = np.where(np.isnan(s), med, s)
    else:
        s = np.where(np.isnan(s), 0.0, s)
    return s


def apply_signal_processing(signal, method, window=5, polyorder=2,
                              cutoff=0.1, fourier_terms=10, block_size=5):
    n = len(signal)
    if n < 4:
        return signal.copy()
    if method == "Sin filtro (raw)":
        return signal.copy()
    elif method == "Media móvil":
        w = max(3, min(window, n // 2 * 2 - 1))
        return uniform_filter1d(signal.astype(float), size=w)
    elif method == "Savitzky-Golay":
        w = max(5, min(window | 1, n if n % 2 == 1 else n - 1))
        p = min(polyorder, w - 1)
        return savgol_filter(signal.astype(float), window_length=w, polyorder=p)
    elif method == "Butterworth LP (quitar ruido)":
        nyq = 0.5
        cutf = max(0.01, min(cutoff, 0.49))
        b, a = butter(2, cutf / nyq, btype='low')
        return filtfilt(b, a, signal.astype(float))
    elif method == "Solo tendencia (regresión polinómica)":
        x = np.arange(n)
        p = min(polyorder, n - 1)
        coeffs = np.polyfit(x, signal.astype(float), p)
        return np.polyval(coeffs, x)
    elif method == "FFT denoise":
        fft_vals = np.fft.rfft(signal.astype(float))
        freqs = np.fft.rfftfreq(n)
        mask = np.abs(freqs) < cutoff
        fft_vals[~mask] = 0
        return np.fft.irfft(fft_vals, n=n)
    elif method == "Serie de Fourier (reconstrucción)":
        s = signal.astype(float)
        terms = max(1, min(fourier_terms, n // 2))
        fft_v = np.fft.rfft(s)
        fft_filt = np.zeros_like(fft_v)
        fft_filt[:min(terms + 1, len(fft_v))] = fft_v[:min(terms + 1, len(fft_v))]
        return np.fft.irfft(fft_filt, n=n)
    elif method == "Promedio por bloques":
        bs = max(2, min(block_size, n // 2))
        s = signal.astype(float)
        out = s.copy()
        for i in range(0, n, bs):
            blk = s[i:i + bs]
            out[i:i + bs] = blk.mean()
        return out
    return signal.copy()


def stl_decompose(signal, timestamps, period=None):
    s = signal.astype(float).copy()
    n = len(s)
    if n < 8:
        return None
    if period is None or period < 2:
        s_norm = s - s.mean()
        autocorr = np.correlate(s_norm, s_norm, mode='full')[n-1:]
        autocorr /= (autocorr[0] + 1e-12)
        search = autocorr[2:n//2]
        if len(search) > 2:
            peaks = []
            for i in range(1, len(search)-1):
                if search[i] > search[i-1] and search[i] > search[i+1]:
                    peaks.append((search[i], i+2))
            if peaks:
                period = max(peaks, key=lambda x: x[0])[1]
            else:
                period = max(4, n // 8)
        else:
            period = max(4, n // 8)
    period = max(2, min(period, n // 2))
    hw = period // 2
    trend = np.full(n, np.nan)
    for i in range(n):
        lo_ = max(0, i - hw)
        hi_ = min(n, i + hw + 1)
        trend[i] = s[lo_:hi_].mean()
    trend = np.interp(np.arange(n),
                      np.where(~np.isnan(trend))[0],
                      trend[~np.isnan(trend)])
    detrended = s - trend
    seasonal_pattern = np.zeros(period)
    counts = np.zeros(period)
    for i in range(n):
        ph = i % period
        seasonal_pattern[ph] += detrended[i]
        counts[ph] += 1
    counts = np.maximum(counts, 1)
    seasonal_pattern /= counts
    seasonal_pattern -= seasonal_pattern.mean()
    seasonal = np.array([seasonal_pattern[i % period] for i in range(n)])
    residual = s - trend - seasonal
    return {"observed": s, "trend": trend, "seasonal": seasonal,
            "residual": residual, "period": period, "n": n}


def build_decomposition_figure(decomp, title, y_label="Valor", timestamps=None):
    if decomp is None:
        return None
    ts = timestamps if timestamps is not None else np.arange(decomp["n"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observado", "Tendencia", "Estacional", "Residuo"],
                        vertical_spacing=0.07,
                        row_heights=[0.28, 0.24, 0.24, 0.24])
    colors = [CYAN, ORANGE, GREEN, PINK]
    keys   = ["observed", "trend", "seasonal", "residual"]
    for row, (key, color) in enumerate(zip(keys, colors), start=1):
        fig.add_trace(go.Scatter(x=ts, y=decomp[key], mode="lines",
            name=key.capitalize(),
            line=dict(color=color, width=1.8), showlegend=False), row=row, col=1)
        fig.update_yaxes(title_text=key[:3].capitalize(),
            tickfont=dict(size=9, color="#8b949e"),
            gridcolor=BORDER, row=row, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=4, col=1, gridcolor=BORDER, color="#8b949e")
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"Descomposición STL — {title}", font=dict(color=CYAN, size=13)),
        height=580)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#cdd9e5", size=11)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  FUKUZONO LSTM — PyTorch (nuevo modelo)
# ════════════════════════════════════════════════════════════════════════════

N_FEATURES_FUKU = 14


def rolling_slope_np(arr, window):
    slopes = np.full(len(arr), np.nan)
    x = np.arange(window, dtype=np.float32)
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        slopes[i] = np.polyfit(x, y, 1)[0]
    return slopes


def robust_scale(arr):
    """Escala robusta (mediana / IQR) sin sklearn."""
    med = np.nanmedian(arr)
    q1  = np.nanpercentile(arr, 25)
    q3  = np.nanpercentile(arr, 75)
    iqr = q3 - q1 + 1e-8
    return (arr - med) / iqr


def engineer_features_fuku(inv_v_arr, disp_arr, vel_arr, t_days_arr,
                            window_short=10, window_long=25):
    """
    Construye las 14 features del modelo Fukuzono a partir de arrays numpy.
    Maneja series cortas (window > n) degradando a window=max(2, n).
    """
    n = len(inv_v_arr)
    inv_v  = inv_v_arr.astype(np.float64)
    disp   = disp_arr.astype(np.float64)
    vel    = vel_arr.astype(np.float64)
    t_days = t_days_arr.astype(np.float64)

    ws = min(window_short, max(2, n - 1))
    wl = min(window_long,  max(2, n - 1))

    dt = np.gradient(t_days)
    dt = np.where(np.abs(dt) < 1e-8, 1e-8, dt)
    d_inv_v  = np.gradient(inv_v, dt)
    d2_inv_v = np.gradient(d_inv_v, dt)

    ratio      = np.where(np.abs(d_inv_v) > 1e-8, inv_v / d_inv_v, np.nan)
    t_F_approx = t_days - ratio

    # Rolling stats via stride tricks
    def _roll_mean_std(x, w):
        pad = np.pad(x, (w - 1, 0), mode='edge')
        wins = np.lib.stride_tricks.sliding_window_view(pad, w)
        return wins.mean(axis=1), wins.std(axis=1) + 1e-8

    roll_mean, roll_std = _roll_mean_std(inv_v, ws)
    sl_short = rolling_slope_np(inv_v, ws)
    sl_long  = rolling_slope_np(inv_v, wl)

    inv_v_max    = np.maximum.accumulate(inv_v)
    pct_from_max = (inv_v_max - inv_v) / (inv_v_max + 1e-8)
    ratio_roll   = inv_v / (roll_mean + 1e-8)
    t_elap       = t_days / (t_days[-1] + 1e-8)

    inv_v_norm = robust_scale(inv_v)
    disp_norm  = robust_scale(disp)
    vel_norm   = robust_scale(vel)

    t_F_valid = np.where(np.isfinite(t_F_approx), t_F_approx,
                         np.nanmedian(t_F_approx) if np.any(np.isfinite(t_F_approx)) else 0.0)
    t_F_norm  = robust_scale(t_F_valid)

    inv_v_log  = np.log1p(np.clip(inv_v, 0, None))
    inv_v_log /= (np.max(inv_v_log) + 1e-8)

    d1n  = np.clip(d_inv_v  / (np.nanstd(d_inv_v)  + 1e-8), -5, 5)
    d2n  = np.clip(d2_inv_v / (np.nanstd(d2_inv_v) + 1e-8), -5, 5)
    ssn  = np.where(np.isfinite(sl_short),
                    np.clip(sl_short / (np.nanstd(sl_short) + 1e-8), -5, 5), 0.0)
    sln  = np.where(np.isfinite(sl_long),
                    np.clip(sl_long  / (np.nanstd(sl_long)  + 1e-8), -5, 5), 0.0)
    rstn = roll_std  / (np.nanmax(roll_std)  + 1e-8)
    rmn  = roll_mean / (np.nanmax(roll_mean) + 1e-8)

    feats = np.column_stack([
        inv_v_norm, inv_v_log, d1n, d2n, t_F_norm,
        ssn, sln, rmn, rstn, ratio_roll,
        pct_from_max, disp_norm, vel_norm, t_elap
    ]).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = F.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(2)
        return (x * w).sum(dim=1), w.squeeze(2)


class FukuzonoLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES_FUKU, hidden_dim=128,
                 n_layers=2, dropout=0.40, bidirectional=True):
        super().__init__()
        self.n_dirs = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.lstm_norm = nn.LayerNorm(hidden_dim * self.n_dirs)
        self.attention  = SelfAttention(hidden_dim * self.n_dirs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.n_dirs, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        xp = self.input_proj(x)
        lo, _ = self.lstm(xp)
        lo = self.lstm_norm(lo)
        ctx, attn = self.attention(lo)
        return self.classifier(ctx).squeeze(-1), ctx


def train_fukuzono(inv_v_arr, disp_arr, vel_arr, t_days_arr,
                   seq_len=20, hidden_dim=128, n_layers=2,
                   dropout=0.30, bidirectional=True,
                   lr=1e-3, epochs=300, patience=30,
                   reg_epochs=200, reg_lr=1e-3,
                   step_hours=24.0, max_steps=120,
                   stop_threshold=0.5,
                   progress_cb=None):
    """
    Entrena FukuzonoLSTM completo desde cero (sin pesos previos).
    Devuelve: pred_train, future_inv_v, future_ts_offsets, metrics, history_loss
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n = len(inv_v_arr)
    feats_all = engineer_features_fuku(inv_v_arr, disp_arr, vel_arr, t_days_arr)

    seq_len = min(seq_len, max(3, n - 2))

    # Construir ventanas
    X_list, y_list = [], []
    for i in range(n - seq_len):
        X_list.append(feats_all[i: i + seq_len])
        y_list.append(inv_v_arr[i + seq_len])

    if len(X_list) < 4:
        raise ValueError(f"Serie demasiado corta ({n} puntos). Necesitas al menos {seq_len + 4}.")

    X_all = torch.tensor(np.array(X_list), dtype=torch.float32).to(DEVICE)
    y_all = torch.tensor(np.array(y_list), dtype=torch.float32).to(DEVICE)

    y_mean = float(y_all.mean())
    y_std  = float(y_all.std()) + 1e-8
    y_scaled = (y_all - y_mean) / y_std

    # Split train/val
    split = max(2, int(len(X_all) * 0.85))
    X_tr, y_tr = X_all[:split], y_scaled[:split]
    X_val, y_val = X_all[split:], y_scaled[split:]

    # ── Modelo completo (entrenamiento end-to-end) ──
    model = FukuzonoLSTM(n_features=N_FEATURES_FUKU, hidden_dim=hidden_dim,
                         n_layers=n_layers, dropout=dropout,
                         bidirectional=bidirectional).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(10, epochs//8), gamma=0.6)
    loss_fn   = nn.HuberLoss(delta=0.5)

    history_loss = []
    best_val = np.inf
    best_state = None
    wait = 0

    total_ep = epochs + reg_epochs

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds, _ = model(X_tr)
        loss = loss_fn(preds, y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        history_loss.append(float(loss))

        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                vp, _ = model(X_val)
                val_loss = float(loss_fn(vp, y_val))
        else:
            val_loss = float(loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if progress_cb:
            progress_cb((ep + 1) / total_ep * 0.6)

    if best_state:
        model.load_state_dict(best_state)

    # ── Cabeza de regresión fine-tuning (congelar BiLSTM) ──
    for p in model.lstm.parameters():
        p.requires_grad = False
    for p in model.input_proj.parameters():
        p.requires_grad = False

    reg_head = nn.Sequential(
        nn.Linear(hidden_dim * (2 if bidirectional else 1), 64),
        nn.GELU(),
        nn.Linear(64, 1)
    ).to(DEVICE)

    opt_reg = torch.optim.Adam(
        list(reg_head.parameters()) + list(model.attention.parameters()),
        lr=reg_lr)

    for ep2 in range(reg_epochs):
        model.eval()
        reg_head.train()
        opt_reg.zero_grad()
        with torch.no_grad():
            _, emb = model(X_tr)
        preds_reg = reg_head(emb).squeeze(-1)
        loss_reg = loss_fn(preds_reg, y_tr)
        loss_reg.backward()
        nn.utils.clip_grad_norm_(reg_head.parameters(), 1.0)
        opt_reg.step()
        history_loss.append(float(loss_reg))
        if progress_cb:
            progress_cb(0.6 + (ep2 + 1) / reg_epochs * 0.35)

    # ── Predicciones en train ──
    model.eval()
    reg_head.eval()
    with torch.no_grad():
        _, emb_all = model(X_all)
        preds_all_sc = reg_head(emb_all).squeeze(-1)
    preds_all_real = (preds_all_sc.cpu().numpy() * y_std + y_mean)

    pred_train = np.full(n, np.nan)
    for i, pv in enumerate(preds_all_real):
        pred_train[i + seq_len] = max(0.0, pv)

    # ── Loop autorregresivo ──
    buf_inv_v = list(inv_v_arr.astype(float))
    buf_disp  = list(disp_arr.astype(float))
    buf_vel   = list(vel_arr.astype(float))
    buf_t     = list(t_days_arr.astype(float))

    future_inv_v = []
    future_t_off = []  # offsets en días desde el último punto real

    dt_mean = float(np.mean(np.diff(t_days_arr))) if len(t_days_arr) > 1 else step_hours / 24.0
    step_days = step_hours / 24.0

    with torch.no_grad():
        for step in range(max_steps):
            inv_v_buf = np.array(buf_inv_v)
            disp_buf  = np.array(buf_disp)
            vel_buf   = np.array(buf_vel)
            t_buf     = np.array(buf_t)

            feats_buf = engineer_features_fuku(inv_v_buf, disp_buf, vel_buf, t_buf)
            window = torch.tensor(
                feats_buf[-seq_len:], dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            _, emb = model(window)
            pred_sc = float(reg_head(emb).squeeze())
            next_inv_v = max(0.0, pred_sc * y_std + y_mean)

            next_t   = buf_t[-1] + step_days
            next_vel = 1.0 / (next_inv_v + 1e-8)
            next_disp = buf_disp[-1] + next_vel * step_days

            future_inv_v.append(next_inv_v)
            future_t_off.append(next_t - t_days_arr[-1])

            buf_inv_v.append(next_inv_v)
            buf_vel.append(next_vel)
            buf_disp.append(next_disp)
            buf_t.append(next_t)

            if next_inv_v <= stop_threshold:
                break

    if progress_cb:
        progress_cb(1.0)

    # Métricas
    vm = ~np.isnan(pred_train)
    real_seg = inv_v_arr[vm]
    pred_seg = pred_train[vm]
    mae  = float(np.mean(np.abs(real_seg - pred_seg)))
    rmse = float(np.sqrt(np.mean((real_seg - pred_seg) ** 2)))
    mape = float(np.mean(np.abs((real_seg - pred_seg) / (np.abs(real_seg) + 1e-8))) * 100)
    ss_r = np.sum((real_seg - pred_seg) ** 2)
    ss_t = np.sum((real_seg - real_seg.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-12))

    n_params = sum(p.numel() for p in model.parameters()) + \
               sum(p.numel() for p in reg_head.parameters())

    metrics = dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2,
                   n_params=n_params, epochs_run=len(history_loss),
                   best_val_loss=best_val,
                   seq_len=seq_len, hidden_dim=hidden_dim,
                   n_layers=n_layers, bidirectional=bidirectional)

    return pred_train, np.array(future_inv_v), np.array(future_t_off), metrics, history_loss


# ════════════════════════════════════════════════════════════════════════════
#  FIGURAS PLOTLY
# ════════════════════════════════════════════════════════════════════════════

def build_velocity_figure(timestamps, displacements, processed_disp,
                           frame_range, method_name):
    ts  = np.array(timestamps)
    raw = np.array(displacements)
    pro = np.array(processed_disp)
    fa, fb = frame_range
    mask = np.zeros(len(ts), dtype=bool)
    mask[fa:fb] = True
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=raw, mode="lines", name="Raw",
        line=dict(color=CYAN, width=1.2, dash="dot"), opacity=0.5))
    fig.add_trace(go.Scatter(x=ts[mask], y=pro[mask], mode="lines",
        name=f"Procesada ({method_name})", line=dict(color=ORANGE, width=2.5)))
    if fa > 0:
        fig.add_trace(go.Scatter(x=ts[:fa], y=raw[:fa], mode="lines",
            name="Fuera del rango", line=dict(color="#444", width=1), showlegend=False))
    if fb < len(ts):
        fig.add_trace(go.Scatter(x=ts[fb:], y=raw[fb:], mode="lines",
            line=dict(color="#444", width=1), showlegend=False))
    if mask.any():
        idx_max = np.argmax(pro[mask])
        xm = ts[mask][idx_max]; ym = pro[mask][idx_max]
        fig.add_annotation(x=xm, y=ym,
            text=f"max {ym:.2f} px<br>t={xm:.1f}s",
            showarrow=True, arrowhead=2, arrowcolor=ORANGE,
            font=dict(color=ORANGE, size=10), bgcolor=BG2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Velocidad óptica media vs. Tiempo", font=dict(color=CYAN, size=14)),
        xaxis_title="Tiempo (s)", yaxis_title="px / frame", height=380)
    return fig


def build_inverse_velocity_figure(ts_iv, inv_vel_raw, inv_vel_proc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_iv, y=inv_vel_raw, mode="lines", name="1/v Raw",
        line=dict(color="#888", width=1, dash="dot"), opacity=0.5))
    fig.add_trace(go.Scatter(x=ts_iv, y=inv_vel_proc, mode="lines",
        name="1/v Procesada (→ LSTM)", line=dict(color=PINK, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,157,0.07)"))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Velocidad Inversa (1/v) — proxy de lentitud / estasis",
                   font=dict(color=CYAN, size=14)),
        xaxis_title="Tiempo (s)", yaxis_title="1 / (px/frame)", height=380)
    return fig


def compute_inv_vel(disp, timestamps):
    d      = np.abs(np.diff(disp.astype(np.float64)))
    iv     = 1.0 / (d + 1e-9)
    finite = iv[np.isfinite(iv)]
    p99    = float(np.percentile(finite, 99)) if len(finite) > 1 else float(iv.max())
    iv     = np.clip(iv, 0.0, p99)
    ts_iv  = np.array(timestamps[1:], dtype=np.float64)
    return ts_iv, iv


def build_fukuzono_figure(ts_real, inv_v_real, pred_train,
                           future_t_off, future_inv_v,
                           metrics, history_loss,
                           stop_threshold=0.5):
    """Figura principal Fukuzono: histórico + ajuste + proyección autorregresiva."""
    ts_fut = ts_real[-1] + future_t_off
    rmse   = metrics["RMSE"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32],
                        subplot_titles=["Serie real + proyección autorregresiva FukuzonoLSTM",
                                        "Curva de pérdida"])

    # Serie real
    fig.add_trace(go.Scatter(x=ts_real, y=inv_v_real, mode="lines",
        name="1/v real", line=dict(color=PINK, width=2.2)), row=1, col=1)

    # Reconstrucción train
    vm = ~np.isnan(pred_train)
    if vm.any():
        fig.add_trace(go.Scatter(x=ts_real[vm], y=pred_train[vm], mode="lines",
            name="Ajuste LSTM (train)", line=dict(color=YELLOW, width=1.8, dash="dash")),
            row=1, col=1)

    # Banda ±RMSE
    if len(future_inv_v) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([ts_fut, ts_fut[::-1]]),
            y=np.concatenate([future_inv_v + rmse, (future_inv_v - rmse)[::-1]]),
            fill="toself", fillcolor="rgba(63,185,80,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"±1 RMSE ({rmse:.3f})", showlegend=True), row=1, col=1)

        # Proyección futura
        fig.add_trace(go.Scatter(x=ts_fut, y=future_inv_v, mode="lines+markers",
            name="Proyección autorregresiva",
            line=dict(color=GREEN, width=2.2), marker=dict(size=5)), row=1, col=1)

        # Punto de falla proyectado
        fig.add_vline(x=float(ts_fut[-1]), line_dash="dot", line_color="red",
                      annotation_text=f"Falla proyectada<br>t={ts_fut[-1]:.1f}s",
                      annotation_font_color="red", row=1, col=1)

    # Línea de corte
    fig.add_hline(y=stop_threshold, line_dash="dot", line_color=BORDER,
                  annotation_text=f"umbral={stop_threshold}", row=1, col=1)

    # Separación histórico / proyección
    fig.add_vline(x=float(ts_real[-1]), line_dash="dot", line_color=BORDER, row=1, col=1)

    # Curva de pérdida
    fig.add_trace(go.Scatter(y=history_loss, mode="lines", name="Loss",
        line=dict(color=CYAN, width=1.8), showlegend=True), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="FukuzonoLSTM — Predicción Autorregresiva (1/v → 0)",
                   font=dict(color=CYAN, size=14)), height=440)
    fig.update_yaxes(title_text="1/v", row=1, col=1, gridcolor=BORDER)
    fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1, gridcolor=BORDER)
    fig.update_yaxes(title_text="Loss (log)", type="log", row=1, col=2, gridcolor=BORDER)
    fig.update_xaxes(title_text="Epoch", row=1, col=2, gridcolor=BORDER)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#cdd9e5", size=11)
    return fig


def build_excel_displacement_figure(series_list, processed_list, method_name, range_list):
    COLORS_RAW  = [CYAN, GREEN, "#c678dd", "#e5c07b", "#56b6c2", "#e06c75"]
    COLORS_PROC = [ORANGE, PINK, "#ff6b9d", "#ffd700", "#00bcd4", "#ff5722"]
    fig = go.Figure()
    for idx, (s, proc, rng) in enumerate(zip(series_list, processed_list, range_list)):
        ts  = s["timestamps"]
        raw = s["displacements"]
        fa, fb = rng
        mask = np.zeros(len(ts), dtype=bool)
        mask[fa:fb] = True
        cr = COLORS_RAW[idx % len(COLORS_RAW)]
        cp = COLORS_PROC[idx % len(COLORS_PROC)]
        lbl = s["name"].replace(".xlsx", "").replace(".xls", "")
        fig.add_trace(go.Scatter(x=ts, y=raw, mode="lines", name=f"{lbl} raw",
            line=dict(color=cr, width=1, dash="dot"), opacity=0.45))
        fig.add_trace(go.Scatter(x=ts[mask], y=proc[mask], mode="lines",
            name=f"{lbl} ({method_name})", line=dict(color=cp, width=2.4)))
        if mask.any():
            im = int(np.argmax(np.abs(proc[mask])))
            xm = ts[mask][im]; ym = proc[mask][im]
            fig.add_annotation(x=xm, y=ym, text=f"max {ym:.4f}<br>{lbl}",
                showarrow=True, arrowhead=2, arrowcolor=cp,
                font=dict(color=cp, size=9), bgcolor=BG2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Desplazamiento vs. Tiempo", font=dict(color=CYAN, size=14)),
        xaxis_title="Tiempo (s desde inicio)", yaxis_title="Desplazamiento (mm)", height=420)
    return fig


def build_excel_inv_vel_figure(inv_seg_only_list):
    COLORS = [PINK, "#c678dd", GREEN, ORANGE, CYAN, "#e5c07b"]
    fig = go.Figure()
    all_proc_vals = []
    for ts_iv, inv_proc, inv_raw, _, _, _, _ in inv_seg_only_list:
        finite = inv_proc[np.isfinite(inv_proc) & (inv_proc > 0)]
        if len(finite): all_proc_vals.extend(finite.tolist())
    global_cap = float(np.percentile(all_proc_vals, 95)) if all_proc_vals else None
    for idx, (ts_iv, inv_proc, inv_raw, _, _, _, lbl) in enumerate(inv_seg_only_list):
        c = COLORS[idx % len(COLORS)]
        ir_disp = np.where(inv_raw > (global_cap * 3 if global_cap else np.inf), np.nan, inv_raw)
        fig.add_trace(go.Scatter(x=ts_iv, y=ir_disp, mode="lines", name=f"{lbl} raw",
            line=dict(color=c, width=1, dash="dot"), opacity=0.35))
        fig.add_trace(go.Scatter(x=ts_iv, y=inv_proc, mode="lines",
            name=f"{lbl} procesada", line=dict(color=c, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.07)"))
    ymax = global_cap * 1.15 if global_cap else None
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Velocidad Inversa 1/|Δdisp| — proxy de estasis",
                   font=dict(color=CYAN, size=14)),
        xaxis_title="Tiempo (s)", yaxis_title="1 / |Δdesplazamiento|", height=400)
    if ymax:
        fig.update_yaxes(range=[0, ymax])
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  CARGA DE FRAMES / EXCEL
# ════════════════════════════════════════════════════════════════════════════

def load_frames_from_images(uploaded_files, assumed_fps=1.0):
    if len(uploaded_files) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes para calcular flujo óptico.")
    sorted_files = sorted(uploaded_files, key=lambda f: f.name)
    frames = []
    failed = []
    for i, uf in enumerate(sorted_files):
        raw = uf.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            failed.append(uf.name); continue
        frames.append((i / assumed_fps, img))
    if failed:
        st.warning(f"No se pudieron leer {len(failed)} imagen(es): {', '.join(failed[:5])}")
    if len(frames) < 2:
        raise ValueError("Menos de 2 imágenes válidas cargadas.")
    dur = frames[-1][0]
    return frames, dur, assumed_fps


def extract_frames_from_video(video_bytes, n_frames):
    cap = None; tmp_path = None
    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(video_bytes); tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened(): break
        cap.release(); os.unlink(tmp_path); cap = None
    if cap is None:
        raise RuntimeError("No se pudo abrir el video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur   = total / fps
    idxs  = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frm = cap.read()
        if ret: frames.append((idx / fps, frm))
    cap.release()
    try: os.unlink(tmp_path)
    except: pass
    return frames, dur, fps


def parse_excel_series(uploaded_file):
    import pandas as pd
    import io
    raw = uploaded_file.read()
    df = None
    for engine in ["openpyxl", "xlrd"]:
        try:
            df = pd.read_excel(io.BytesIO(raw), engine=engine, header=0)
            break
        except Exception:
            continue
    if df is None:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=0)
    if df is None or len(df) < 3:
        raise ValueError("El archivo no tiene suficientes filas.")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["fecha", "date", "time", "tiempo", "datetime"]):
            date_col = c; break
    if date_col is None:
        date_col = df.columns[0]
    disp_col = None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["desplaz", "displace", "mm", "deform", "mm)"]):
            disp_col = c; break
    if disp_col is None:
        for c in df.columns:
            if c != date_col:
                try:
                    pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="raise")
                    disp_col = c; break
                except Exception:
                    continue
    if disp_col is None:
        raise ValueError("No se encontró columna de desplazamiento.")
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    valid = dates.notna()
    dates = dates[valid].reset_index(drop=True)
    disp_raw = df[disp_col][valid].astype(str).str.replace(",", ".").str.strip()
    displacements = pd.to_numeric(disp_raw, errors="coerce").fillna(0.0).values.astype(np.float64)
    if len(dates) < 3:
        raise ValueError(f"Solo {len(dates)} filas válidas tras parseo.")
    t0 = dates.iloc[0]
    timestamps = np.array([(d - t0).total_seconds() for d in dates], dtype=np.float64)
    return {"name": uploaded_file.name, "timestamps": timestamps,
            "displacements": displacements, "dates": dates,
            "df": df[valid].reset_index(drop=True),
            "date_col": date_col, "disp_col": disp_col}


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def _init_session_state():
    defaults = dict(
        cache_key=None, frames=None, duration_sec=None, fps=None,
        timestamps=None, displacements=None, analysis_done=False,
        lstm_result=None,
        excel_series=None, excel_cache_key=None, excel_lstm_results=None,
        excel_processing_key=None, vid_processing_key=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    _init_session_state()

    st.title("Análisis de Flujo Óptico — Cámara Térmica")
    st.markdown(
        "Sube un **video**, **imágenes** o una **serie temporal Excel** para comenzar. "
        "Ajusta todos los parámetros en el panel lateral."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Parámetros generales")
        n_frames    = st.slider("Número de frames (solo modo video)", 5, 200, 30, 5)
        assumed_fps = st.slider("FPS asumido (modo imágenes)", 0.1, 60.0, 1.0, 0.1)
        diff_thr    = st.slider("Umbral movimiento",  2.0, 60.0, 12.0, 0.5)
        dark_thr    = st.slider("Oscuridad humo",     20, 150, 80, 2)
        tex_thr     = st.slider("Tratamiento Material en Suspensión", 0.5, 20.0, 3.5, 0.25)
        flow_step   = st.slider("Paso vectores (quiver)", 6, 40, 18, 2)
        show_pairs  = st.checkbox("Mostrar panel por cada par", value=False)

        st.divider()
        st.header("Algoritmo de Flujo Optico")
        flow_algo = st.selectbox("Algoritmo", [
            "RAFT-lite", "DIS", "LK-Pyramid", "Farneback"], index=0)

        flow_min_area = st.slider("Area minima movimiento (px)", 10, 1000, 200, 10)

        if flow_algo == "RAFT-lite":
            st.markdown("**RAFT-lite**")
            raft_iters        = st.slider("Iteraciones refinamiento", 2, 30, 12, 1)
            raft_corr_radius  = st.slider("Radio correlacion (r)", 1, 8, 4, 1)
            raft_corr_levels  = st.slider("Niveles piramide correlacion", 1, 4, 3, 1)
            raft_update_iters = st.slider("Pasos update por nivel", 1, 12, 6, 1)
            raft_alpha_smooth = st.slider("Suavizado espacial (alpha TV)", 0.0, 2.0, 0.5, 0.1)
            raft_feat_ch      = st.slider("Canales features", 4, 64, 16, 4)
            raft_ds           = st.slider("Factor downsample (velocidad)", 1, 8, 4, 1)
            flow_params = dict(iters=raft_iters, corr_radius=raft_corr_radius,
                               corr_levels=raft_corr_levels, update_iters=raft_update_iters,
                               alpha_smooth=raft_alpha_smooth, feature_channels=raft_feat_ch,
                               downsample_factor=raft_ds)
        elif flow_algo == "DIS":
            st.markdown("**DIS — Dense Inverse Search**")
            dis_preset       = st.selectbox("Preset", ["Ultrafast", "Fast", "Medium"], index=2)
            dis_finest_scale = st.slider("Finest scale", 0, 3, 1, 1)
            dis_gd_iters     = st.slider("Gradient descent iters", 5, 100, 25, 5)
            dis_var_iters    = st.slider("Variational refinement iters", 0, 20, 5, 1)
            dis_var_alpha    = st.slider("Var. refinement alpha", 1.0, 50.0, 20.0, 1.0)
            dis_var_gamma    = st.slider("Var. refinement gamma", 1.0, 30.0, 10.0, 1.0)
            dis_var_delta    = st.slider("Var. refinement delta", 0.1, 20.0, 5.0, 0.5)
            dis_mean_norm    = st.checkbox("Usar normalizacion de media", value=True)
            dis_spatial_prop = st.checkbox("Usar propagacion espacial", value=True)
            flow_params = dict(preset=dis_preset, finest_scale=dis_finest_scale,
                               grad_desc_iters=dis_gd_iters,
                               variational_refinement_iters=dis_var_iters,
                               variational_refinement_alpha=dis_var_alpha,
                               variational_refinement_gamma=dis_var_gamma,
                               variational_refinement_delta=dis_var_delta,
                               use_mean_normalization=dis_mean_norm,
                               use_spatial_propagation=dis_spatial_prop)
        elif flow_algo == "LK-Pyramid":
            st.markdown("**Lucas-Kanade Piramidal**")
            lk_win_size    = st.slider("Tamano ventana LK", 5, 51, 21, 2)
            lk_max_level   = st.slider("Niveles piramide", 1, 6, 4, 1)
            lk_max_corners = st.slider("Max corners (Shi-Tomasi)", 50, 2000, 500, 50)
            lk_quality     = st.select_slider("Calidad minima corners",
                              options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                              value=0.01, format_func=lambda x: f"{x:.3f}")
            lk_min_dist    = st.slider("Distancia minima entre corners (px)", 2, 30, 7, 1)
            lk_block_size  = st.slider("Block size Shi-Tomasi", 3, 21, 7, 2)
            lk_back_thresh = st.slider("Umbral backward consistency (px)", 0.1, 5.0, 1.0, 0.1)
            lk_eigen_thr   = st.select_slider("Umbral eigenvalue minimo",
                              options=[1e-5, 1e-4, 5e-4, 1e-3, 1e-2],
                              value=1e-4, format_func=lambda x: f"{x:.0e}")
            flow_params = dict(win_size=lk_win_size, max_level=lk_max_level,
                               max_corners=lk_max_corners, quality_level=lk_quality,
                               min_distance=lk_min_dist, block_size=lk_block_size,
                               back_threshold=lk_back_thresh, eigen_threshold=lk_eigen_thr)
        else:
            st.markdown("**Farneback**")
            pyr_scale  = st.slider("pyr_scale",  0.1, 0.9, 0.5, 0.05)
            levels     = st.slider("levels",     1, 10, 5, 1)
            winsize    = st.slider("winsize",    5, 51, 15, 2)
            iterations = st.slider("iterations", 1, 10, 3, 1)
            poly_n     = st.selectbox("poly_n", [5, 7], index=1)
            poly_sigma = st.slider("poly_sigma", 0.5, 3.0, 1.5, 0.1)
            fb_gaussian= st.checkbox("Filtro Gaussiano", value=True)
            flow_params = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                               iterations=iterations, poly_n=int(poly_n),
                               poly_sigma=poly_sigma, use_gaussian=fb_gaussian)

        st.divider()
        st.header("Procesamiento de señal")
        signal_method = st.selectbox("Método", [
            "Sin filtro (raw)", "Media móvil", "Savitzky-Golay",
            "Butterworth LP (quitar ruido)", "Solo tendencia (regresión polinómica)",
            "FFT denoise", "Serie de Fourier (reconstrucción)", "Promedio por bloques"])

        signal_window = 7; signal_polyord = 3; signal_cutoff = 0.15
        signal_fourier_terms = 10; signal_block_size = 5

        if signal_method == "Media móvil":
            signal_window = st.slider("Ventana (muestras)", 3, 101, 7, 2)
        elif signal_method == "Savitzky-Golay":
            signal_window  = st.slider("Ventana SG (impar)", 5, 101, 11, 2)
            signal_polyord = st.slider("Orden polinomio", 1, 5, 3, 1)
        elif signal_method == "Butterworth LP (quitar ruido)":
            signal_cutoff = st.slider("Frecuencia de corte [0–0.5]", 0.01, 0.49, 0.15, 0.01)
        elif signal_method == "Solo tendencia (regresión polinómica)":
            signal_polyord = st.slider("Grado del polinomio", 1, 8, 3, 1)
        elif signal_method == "FFT denoise":
            signal_cutoff = st.slider("Umbral de frecuencia [0–0.5]", 0.01, 0.49, 0.15, 0.01)
        elif signal_method == "Serie de Fourier (reconstrucción)":
            signal_fourier_terms = st.slider("Armónicos a conservar", 1, 100, 10, 1)
        elif signal_method == "Promedio por bloques":
            signal_block_size = st.slider("Tamaño de bloque (N muestras)", 2, 200, 5, 1)

        st.markdown("---")
        st.markdown("**Filtro de Outliers**")
        out_active_sb = st.checkbox("Activar filtro de outliers", key="out_active_sb", value=False)
        out_order_sb = "Antes del suavizado"
        out_method_sb = "IQR"; out_replace_sb = "interpolate"
        out_iqr_k_sb = 1.5; out_zscore_sb = 3.0; out_cmin_sb = None; out_cmax_sb = None
        if out_active_sb:
            out_order_sb  = st.radio("Aplicar",
                              ["Antes del suavizado", "Después del suavizado"],
                              horizontal=True, key="out_order_sb")
            out_method_sb = st.selectbox("Método outliers",
                              ["IQR", "Z-score", "Rango manual"], key="out_method_sb")
            out_replace_sb = st.selectbox("Reemplazar outlier con",
                               ["interpolate", "mediana", "NaN→0"], key="out_replace_sb")
            if out_method_sb == "IQR":
                out_iqr_k_sb = st.slider("Factor IQR (k)", 0.5, 5.0, 1.5, 0.1, key="iqr_sb")
            elif out_method_sb == "Z-score":
                out_zscore_sb = st.slider("Umbral Z-score", 1.0, 6.0, 3.0, 0.1, key="zscore_sb")
            else:
                out_cmin_sb = st.number_input("Valor mínimo", value=0.0, key="cmin_sb")
                out_cmax_sb = st.number_input("Valor máximo", value=999.0, key="cmax_sb")

        st.markdown("---")
        st.markdown("**Descomposición STL**")
        stl_show_disp = st.checkbox("STL — Desplazamiento", key="stl_show_disp", value=False)
        stl_show_inv  = st.checkbox("STL — Velocidad Inversa", key="stl_show_inv", value=False)
        stl_period = None
        if stl_show_disp or stl_show_inv:
            stl_period_auto = st.checkbox("Período automático", key="stl_period_auto", value=True)
            if not stl_period_auto:
                stl_period = st.slider("Período estacional (muestras)", 2, 200, 10, 1,
                                        key="stl_period_val")

        st.divider()
        st.header("Filtro por frames")
        frame_filter_placeholder = st.empty()

        st.divider()
        st.header("FukuzonoLSTM — Hiperparámetros")
        lstm_seq_len  = st.slider("Seq length (ventana historia)", 5, 60, 20, 1)
        lstm_hidden   = st.select_slider("Hidden dim", options=[32, 64, 128, 256], value=128)
        lstm_layers   = st.slider("Capas LSTM", 1, 4, 2, 1)
        lstm_dropout  = st.slider("Dropout", 0.0, 0.6, 0.30, 0.05)
        lstm_bidir    = st.checkbox("Bidireccional", value=True)
        lstm_lr       = st.select_slider("Learning rate (BiLSTM)",
                          options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3],
                          value=1e-3, format_func=lambda x: f"{x:.0e}")
        lstm_epochs   = st.slider("Epochs BiLSTM", 50, 500, 200, 25)
        lstm_patience = st.slider("Early stopping patience", 10, 80, 30, 5)
        reg_epochs    = st.slider("Epochs cabeza de regresión", 50, 500, 200, 25)
        reg_lr        = st.select_slider("Learning rate (cabeza)",
                          options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
                          value=1e-3, format_func=lambda x: f"{x:.0e}")
        lstm_step_h   = st.slider("Paso autorregresivo (horas)", 1, 168, 24, 1)
        lstm_max_steps= st.slider("Pasos máximos autorregresivos", 10, 300, 120, 10)
        lstm_stop_thr = st.slider("Umbral de parada 1/v (falla)", 0.01, 10.0, 0.5, 0.01)

    st.markdown("---")
    input_mode = st.radio("Fuente de entrada",
        ["Video", "Imágenes (frames directos)", "Serie Temporal Excel"], horizontal=True)

    uploaded_file   = None
    uploaded_imgs   = None
    uploaded_excels = None

    if input_mode == "Video":
        uploaded_file = st.file_uploader("Selecciona un video (.mp4 / .avi / .mov / .mkv)",
                                          type=["mp4", "avi", "mov", "mkv"])
    elif input_mode == "Imágenes (frames directos)":
        st.info("Sube las imágenes en **orden temporal**. Se ordenarán por nombre.")
        uploaded_imgs = st.file_uploader("Selecciona imágenes (múltiples)",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
            accept_multiple_files=True)
    else:
        st.info("Sube uno o más archivos Excel con columnas **Fecha** y **Desplazamiento (mm)**.")
        uploaded_excels = st.file_uploader("Selecciona archivos Excel (.xlsx / .xls)",
            type=["xlsx", "xls"], accept_multiple_files=True)

    # ════════════════════════════════════════════════════════════════════════
    #  MODO EXCEL
    # ════════════════════════════════════════════════════════════════════════
    if input_mode == "Serie Temporal Excel":
        if not uploaded_excels or len(uploaded_excels) == 0:
            st.info("Sube al menos un archivo Excel para comenzar.")
            return

        excel_cache_key = "_".join(sorted(f.name for f in uploaded_excels))
        processing_key = (
            f"{excel_cache_key}|{signal_method}|{signal_window}|{signal_polyord}|"
            f"{signal_cutoff}|{signal_fourier_terms}|{signal_block_size}|"
            f"{out_active_sb}|{out_order_sb if out_active_sb else ''}|"
            f"{out_method_sb if out_active_sb else ''}|"
            f"{out_iqr_k_sb if out_active_sb else ''}|"
            f"{out_zscore_sb if out_active_sb else ''}"
        )

        if st.session_state["excel_cache_key"] != excel_cache_key:
            series_list = []
            errors = []
            with st.spinner("Leyendo archivos Excel..."):
                for uf in uploaded_excels:
                    try:
                        s = parse_excel_series(uf)
                        series_list.append(s)
                    except Exception as e:
                        errors.append(f"{uf.name}: {e}")
            if errors:
                for err in errors:
                    st.warning(f"Error leyendo {err}")
            if not series_list:
                st.error("No se pudo leer ningún archivo Excel válido."); return
            st.session_state["excel_series"] = series_list
            st.session_state["excel_cache_key"] = excel_cache_key
            st.session_state["excel_lstm_results"] = None

        series_list = st.session_state["excel_series"]
        if not series_list:
            st.error("No hay series cargadas."); return

        if st.session_state.get("excel_processing_key") != processing_key:
            st.session_state["excel_lstm_results"] = None
            st.session_state["excel_processing_key"] = processing_key

        st.success(f"{len(series_list)} serie(s) cargadas correctamente.")
        with st.expander("Vista previa de las series", expanded=True):
            import pandas as pd
            cols_prev = st.columns(min(len(series_list), 3))
            for idx, s in enumerate(series_list):
                with cols_prev[idx % 3]:
                    st.markdown(f"**{s['name']}**")
                    st.caption(
                        f"{len(s['timestamps'])} puntos  |  "
                        f"Duración: {s['timestamps'][-1]/3600:.2f} h  |  "
                        f"Rango disp: [{s['displacements'].min():.4f}, "
                        f"{s['displacements'].max():.4f}] mm")
                    df_show = pd.DataFrame({
                        "Fecha": s["dates"].dt.strftime("%d-%m-%Y %H:%M"),
                        "Desplaz. (mm)": np.round(s["displacements"], 6)})
                    st.dataframe(df_show.head(8), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Rango de datos y Procesamiento de Señal")

        range_list = []
        for idx, s in enumerate(series_list):
            n_pts = len(s["timestamps"])
            lbl = s["name"].replace(".xlsx","").replace(".xls","")
            rng = st.slider(f"Rango de puntos — {lbl}",
                min_value=0, max_value=max(1, n_pts - 1),
                value=(0, max(1, n_pts - 1)),
                key=f"excel_range_{idx}")
            range_list.append(rng)

        import pandas as pd

        proc_final_list   = []
        inv_seg_only_list = []

        for idx, (s, rng) in enumerate(zip(series_list, range_list)):
            raw_full = s["displacements"].astype(float)
            ts_full  = s["timestamps"]
            fa_e, fb_e = rng[0], min(rng[1] + 1, len(raw_full))
            ts_seg   = ts_full[fa_e:fb_e]
            raw_seg  = raw_full[fa_e:fb_e]
            lbl_e    = s["name"].replace(".xlsx","").replace(".xls","")

            if out_active_sb and out_order_sb == "Antes del suavizado":
                seg_step1 = apply_outlier_filter(raw_seg, method=out_method_sb,
                    iqr_k=out_iqr_k_sb, zscore_thr=out_zscore_sb,
                    clip_min=out_cmin_sb, clip_max=out_cmax_sb, replace=out_replace_sb)
            else:
                seg_step1 = raw_seg.copy()

            seg_step2 = apply_signal_processing(seg_step1, method=signal_method,
                window=signal_window, polyorder=signal_polyord, cutoff=signal_cutoff,
                fourier_terms=signal_fourier_terms, block_size=signal_block_size)

            if out_active_sb and out_order_sb == "Después del suavizado":
                seg_final_disp = apply_outlier_filter(seg_step2, method=out_method_sb,
                    iqr_k=out_iqr_k_sb, zscore_thr=out_zscore_sb,
                    clip_min=out_cmin_sb, clip_max=out_cmax_sb, replace=out_replace_sb)
            else:
                seg_final_disp = seg_step2.copy()

            with st.expander(f"Vista señal procesada — {lbl_e}", expanded=(idx == 0)):
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=ts_seg, y=raw_seg, name="Raw",
                    line=dict(color=CYAN, width=1, dash="dot"), opacity=0.5))
                fig_t.add_trace(go.Scatter(x=ts_seg, y=seg_final_disp,
                    name=f"Procesada ({signal_method})", line=dict(color=ORANGE, width=2)))
                fig_t.update_layout(**PLOTLY_LAYOUT,
                    title=dict(text=f"Desplazamiento procesado — {lbl_e}", font=dict(color=CYAN)),
                    xaxis_title="t (s)", yaxis_title="mm", height=240)
                st.plotly_chart(fig_t, use_container_width=True, key=f"excel_sig_proc_{idx}")

                if stl_show_disp:
                    decomp_d = stl_decompose(seg_final_disp, ts_seg, period=stl_period)
                    if decomp_d:
                        fig_dd = build_decomposition_figure(decomp_d,
                            title=f"Desplazamiento — {lbl_e}", y_label="mm", timestamps=ts_seg)
                        if fig_dd:
                            st.plotly_chart(fig_dd, use_container_width=True,
                                            key=f"excel_stl_disp_{idx}")

            proc_full = raw_full.copy()
            proc_full[fa_e:fb_e] = seg_final_disp
            proc_final_list.append(proc_full)

            ts_iv_e, inv_seg_final = compute_inv_vel(seg_final_disp, ts_seg)
            _,       inv_seg_raw   = compute_inv_vel(raw_seg, ts_seg)

            inv_seg_only_list.append((ts_iv_e, inv_seg_final, inv_seg_raw,
                                      seg_final_disp, raw_seg, ts_seg,
                                      s["name"].replace(".xlsx","").replace(".xls","")))

            if stl_show_inv:
                with st.expander(f"STL Velocidad Inversa — {lbl_e}", expanded=False):
                    decomp_i = stl_decompose(inv_seg_final, ts_seg, period=stl_period)
                    if decomp_i:
                        fig_di = build_decomposition_figure(decomp_i,
                            title=f"Vel. Inversa — {lbl_e}", y_label="1/|Δmm|",
                            timestamps=ts_seg)
                        if fig_di:
                            st.plotly_chart(fig_di, use_container_width=True,
                                            key=f"excel_stl_inv_{idx}")
                        st.caption(f"Período: {decomp_i['period']} muestras")

        st.divider()
        st.subheader("Desplazamiento vs. Tiempo — todas las series")
        fig_disp = build_excel_displacement_figure(series_list, proc_final_list,
                                                    signal_method, range_list)
        st.plotly_chart(fig_disp, use_container_width=True, key="excel_disp_all")

        st.subheader("Velocidad Inversa — todas las series")
        fig_inv = build_excel_inv_vel_figure(inv_seg_only_list)
        st.plotly_chart(fig_inv, use_container_width=True, key="excel_inv_all")

        st.subheader("Resumen estadístico")
        stats_rows = []
        for (ts_iv_i, inv_proc_i, inv_raw_i, disp_proc_i, disp_raw_i, ts_full_i, lbl_i), proc, s in zip(
                inv_seg_only_list, proc_final_list, series_list):
            stats_rows.append({
                "Archivo": s["name"],
                "Puntos (rango)": len(disp_proc_i),
                "Disp. media (mm)": f"{disp_proc_i.mean():.6f}",
                "Disp. max (mm)": f"{np.abs(disp_proc_i).max():.6f}",
                "Disp. std (mm)": f"{disp_proc_i.std():.6f}",
                "Vel. inv. media": f"{inv_proc_i.mean():.4f}",
                "Duración (h)": f"{(ts_full_i[-1]-ts_full_i[0])/3600:.2f}" if len(ts_full_i) > 1 else "—",
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

        # ── FukuzonoLSTM para Excel ──────────────────────────────────────────
        st.divider()
        st.subheader("Predicción FukuzonoLSTM — Series Temporales Excel")

        lstm_excel_btn = st.button("Entrenar FukuzonoLSTM en todas las series",
                                    type="primary", key="lstm_excel_btn")

        if lstm_excel_btn:
            results = {}
            for idx, (ts_iv_i, inv_proc_i, inv_raw_i, disp_proc_i, disp_raw_i,
                       ts_full_i, lbl_raw) in enumerate(inv_seg_only_list):

                lbl = series_list[idx]["name"]
                n_min = lstm_seq_len + 4
                if len(inv_proc_i) < n_min:
                    st.warning(f"{lbl}: serie demasiado corta ({len(inv_proc_i)} pts). Saltando.")
                    continue

                st.markdown(f"##### Entrenando: `{lbl}`")
                prog = st.progress(0, text=f"{lbl} — entrenando...")

                # Construir vel a partir de inv_v (1/inv_v)
                vel_i  = 1.0 / (inv_proc_i + 1e-8)
                disp_i = disp_proc_i[1:]  # alinear con inv_v (len-1)
                t_days_i = ts_iv_i / 86400.0  # convertir segundos a días

                # Alinear todo al mismo tamaño
                _n = min(len(inv_proc_i), len(disp_i), len(t_days_i))
                inv_i_ = inv_proc_i[:_n]
                disp_i_ = disp_i[:_n]
                vel_i_  = vel_i[:_n]
                t_i_    = t_days_i[:_n]

                try:
                    pred_tr, fut_inv, fut_t_off, metr, hloss = train_fukuzono(
                        inv_i_, disp_i_, vel_i_, t_i_,
                        seq_len=lstm_seq_len,
                        hidden_dim=lstm_hidden, n_layers=lstm_layers,
                        dropout=lstm_dropout, bidirectional=lstm_bidir,
                        lr=lstm_lr, epochs=lstm_epochs, patience=lstm_patience,
                        reg_epochs=reg_epochs, reg_lr=reg_lr,
                        step_hours=lstm_step_h, max_steps=lstm_max_steps,
                        stop_threshold=lstm_stop_thr,
                        progress_cb=lambda f, _p=prog, _l=lbl: _p.progress(f, text=f"{_l} — {f*100:.0f}%"))
                    prog.progress(1.0, text=f"{lbl} — completado ✓")
                    results[lbl] = (pred_tr, fut_inv, fut_t_off, metr, hloss, ts_iv_i[:_n], inv_i_)
                except Exception as e:
                    prog.empty()
                    st.error(f"Error entrenando {lbl}: {e}")

            st.session_state["excel_lstm_results"] = results if results else None

        xlr = st.session_state["excel_lstm_results"]
        if xlr:
            for res_idx, (lbl, entry) in enumerate(xlr.items()):
                pred_tr, fut_inv, fut_t_off, metr, hloss, ts_s, iv_s = entry
                st.markdown(f"#### {lbl}")

                # Convertir offsets de días a segundos para el eje X
                fut_t_sec = ts_s[-1] + fut_t_off * 86400.0

                fig_main = build_fukuzono_figure(
                    ts_s, iv_s, pred_tr,
                    fut_t_sec - ts_s[-1],   # offsets en segundos
                    fut_inv, metr, hloss, lstm_stop_thr)
                # Ajustar x para que sea absoluto
                fig_main = build_fukuzono_figure(
                    ts_s, iv_s, pred_tr,
                    fut_t_off * 86400.0, fut_inv, metr, hloss, lstm_stop_thr)

                st.plotly_chart(fig_main, use_container_width=True,
                                key=f"excel_fuku_{res_idx}_{lbl[:20]}")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("MAE",  f"{metr['MAE']:.5f}")
                mc2.metric("RMSE", f"{metr['RMSE']:.5f}")
                mc3.metric("MAPE", f"{metr['MAPE']:.2f}%")
                mc4.metric("R²",   f"{metr['R2']:.4f}")

                if len(fut_inv) > 0:
                    t_falla_s = ts_s[-1] + fut_t_off[-1] * 86400.0
                    st.info(
                        f"**Falla proyectada:** offset {fut_t_off[-1]:.1f} días desde el último dato  |  "
                        f"Pasos autorregresivos: {len(fut_inv)}  |  "
                        f"1/v final: {fut_inv[-1]:.4f}")

                with st.expander(f"Detalle modelo y pronóstico — {lbl}"):
                    st.markdown(f"""
| Parámetro | Valor |
|-----------|-------|
| Seq length | {metr['seq_len']} |
| Hidden dim | {metr['hidden_dim']} |
| Capas LSTM | {metr['n_layers']} |
| Bidireccional | {metr['bidirectional']} |
| Parámetros totales | {metr['n_params']:,} |
| Epochs ejecutados | {metr['epochs_run']} |
| Mejor val loss | {metr['best_val_loss']:.6f} |
""")
                    _dt = pd.DataFrame({
                        "Paso": list(range(1, len(fut_inv) + 1)),
                        "Offset (días)": np.round(fut_t_off, 2),
                        "1/v proyectado": np.round(fut_inv, 4),
                        "v estimada (1/día)": np.round(1.0 / (fut_inv + 1e-8), 6),
                    })
                    st.dataframe(_dt, use_container_width=True, hide_index=True)
        return

    # ════════════════════════════════════════════════════════════════════════
    #  MODOS VIDEO / IMÁGENES
    # ════════════════════════════════════════════════════════════════════════

    run_btn = st.button("Analizar", type="primary")

    nothing_uploaded = (
        (input_mode == "Video" and uploaded_file is None) or
        (input_mode == "Imágenes (frames directos)" and
         (uploaded_imgs is None or len(uploaded_imgs) == 0))
    )
    if nothing_uploaded:
        st.info("Sube un archivo y pulsa **Analizar** para comenzar.")
        return

    if input_mode == "Video":
        cache_key = f"video_{uploaded_file.name}_{n_frames}"
    else:
        names = "_".join(sorted(f.name for f in uploaded_imgs))
        cache_key = f"imgs_{names}_{len(uploaded_imgs)}_{assumed_fps}"

    if st.session_state["cache_key"] != cache_key:
        if input_mode == "Video":
            video_bytes = uploaded_file.read()
            with st.spinner(f"Extrayendo {n_frames} frames del video..."):
                try:
                    frames, dur, fps = extract_frames_from_video(video_bytes, n_frames)
                except Exception as e:
                    st.error(f"Error leyendo video: {e}"); return
        else:
            with st.spinner(f"Cargando {len(uploaded_imgs)} imágenes..."):
                try:
                    frames, dur, fps = load_frames_from_images(uploaded_imgs,
                                                                assumed_fps=assumed_fps)
                except Exception as e:
                    st.error(f"Error cargando imágenes: {e}"); return

        st.session_state.update({
            "cache_key": cache_key, "frames": frames,
            "duration_sec": dur, "fps": fps,
            "analysis_done": False, "timestamps": None,
            "displacements": None, "lstm_result": None,
        })

    frames = st.session_state["frames"]
    dur    = st.session_state["duration_sec"]
    fps    = st.session_state["fps"]

    if not frames or len(frames) < 2:
        st.error("No se obtuvieron suficientes frames."); return

    st.success(
        f"{len(frames)} frames  |  Duración: {dur:.1f}s  |  FPS equiv.: {fps:.1f}"
        + ("  |  Fuente: imágenes" if input_mode == "Imágenes (frames directos)" else ""))

    if input_mode == "Imágenes (frames directos)" and len(frames) > 0:
        with st.expander(f"Vista previa de los {len(frames)} frames cargados", expanded=False):
            max_preview = min(len(frames), 12)
            cols = st.columns(min(max_preview, 6))
            for ci, fi in enumerate(range(0, max_preview)):
                t_fi, img_fi = frames[fi]
                with cols[ci % 6]:
                    st.image(cv2.cvtColor(img_fi, cv2.COLOR_BGR2RGB),
                             caption=f"#{fi+1}  t={t_fi:.2f}s",
                             use_container_width=True)
            if len(frames) > max_preview:
                st.caption(f"... y {len(frames) - max_preview} frames más.")

    # ── ROI ───────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Área de Interés (ROI)")

    _ref_t, _ref_img = frames[0]
    _H, _W = _ref_img.shape[:2]

    roi_key = f"roi_{st.session_state['cache_key']}"
    if roi_key not in st.session_state:
        st.session_state[roi_key] = (0, 0, _W, _H)

    col_sliders, col_preview = st.columns([1, 2])

    with col_sliders:
        st.markdown("**Posición y tamaño del ROI**")
        roi_x = st.slider("Origen X", min_value=0, max_value=_W - 2,
            value=st.session_state[roi_key][0], step=1, key="roi_x")
        roi_y = st.slider("Origen Y", min_value=0, max_value=_H - 2,
            value=st.session_state[roi_key][1], step=1, key="roi_y")
        roi_w = st.slider("Ancho del ROI", min_value=2, max_value=_W - roi_x,
            value=min(st.session_state[roi_key][2], _W - roi_x), step=1, key="roi_w")
        roi_h = st.slider("Alto del ROI", min_value=2, max_value=_H - roi_y,
            value=min(st.session_state[roi_key][3], _H - roi_y), step=1, key="roi_h")

        if st.button("Resetear ROI (imagen completa)"):
            st.session_state[roi_key] = (0, 0, _W, _H)
            st.session_state["roi_x"] = 0
            st.session_state["roi_y"] = 0
            st.session_state["roi_w"] = _W
            st.session_state["roi_h"] = _H

        st.session_state[roi_key] = (roi_x, roi_y, roi_w, roi_h)

        st.markdown(f"""
| Campo | Valor |
|-------|-------|
| Origen X | {roi_x} px |
| Origen Y | {roi_y} px |
| Ancho | {roi_w} px |
| Alto | {roi_h} px |
| Área ROI | {roi_w * roi_h:,} px² |
| Cobertura | {roi_w * roi_h / (_W * _H) * 100:.1f}% |
""")

    with col_preview:
        st.markdown("**Vista previa — ROI sobre frame de referencia**")
        _preview = cv2.cvtColor(_ref_img.copy(), cv2.COLOR_BGR2RGB)
        _overlay = _preview.copy()
        _overlay[:roi_y, :] = (_overlay[:roi_y, :] * 0.35).astype(np.uint8)
        _overlay[roi_y+roi_h:, :] = (_overlay[roi_y+roi_h:, :] * 0.35).astype(np.uint8)
        _overlay[roi_y:roi_y+roi_h, :roi_x] = (
            _overlay[roi_y:roi_y+roi_h, :roi_x] * 0.35).astype(np.uint8)
        _overlay[roi_y:roi_y+roi_h, roi_x+roi_w:] = (
            _overlay[roi_y:roi_y+roi_h, roi_x+roi_w:] * 0.35).astype(np.uint8)
        cv2.rectangle(_overlay, (roi_x, roi_y),
                      (roi_x + roi_w - 1, roi_y + roi_h - 1),
                      (255, 140, 0), thickness=max(2, _W // 150))
        _corner_len = max(8, min(roi_w, roi_h) // 6)
        _th = max(3, _W // 100)
        for cx, cy in [(roi_x, roi_y), (roi_x+roi_w-1, roi_y),
                       (roi_x, roi_y+roi_h-1), (roi_x+roi_w-1, roi_y+roi_h-1)]:
            dx = 1 if cx == roi_x else -1
            dy = 1 if cy == roi_y else -1
            cv2.line(_overlay, (cx, cy), (cx + dx*_corner_len, cy), (255,255,255), _th)
            cv2.line(_overlay, (cx, cy), (cx, cy + dy*_corner_len), (255,255,255), _th)
        st.image(_overlay, use_container_width=True,
                 caption=f"ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
        _crop_preview = cv2.cvtColor(
            _ref_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], cv2.COLOR_BGR2RGB)
        st.image(_crop_preview, use_container_width=True, caption="Recorte ROI")

    def apply_roi(img_bgr):
        return img_bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    with frame_filter_placeholder.container():
        n_pairs = len(frames) - 1
        fr_range = st.slider("Rango de pares a incluir (frames A→B)",
            min_value=0, max_value=max(1, n_pairs - 1),
            value=(0, max(1, n_pairs - 1)), key="frame_range_slider")

    frame_range = fr_range

    # ── Análisis completo ─────────────────────────────────────────────────────
    if run_btn:
        timestamps, displacements = [], []
        total_pairs = len(frames) - 1
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bar = st.progress(0, text="Procesando pares...")
        for i in range(total_pairs):
            t1, img1 = frames[i]; t2, img2 = frames[i + 1]
            img1 = apply_roi(img1); img2 = apply_roi(img2)
            g1 = to_gray(img1); g2 = to_gray(img2)
            smoke        = detect_smoke_mask(g2, dark_thresh=dark_thr, texture_thresh=tex_thr)
            motion, diff = detect_motion_opencv(g1, g2, smoke, diff_thresh=diff_thr,
                                                min_area=flow_min_area)
            flow, valid  = compute_optical_flow(g1, g2, motion, smoke,
                                                algo=flow_algo, params=flow_params)
            disp = mean_displacement(flow, valid)
            timestamps.append((t1 + t2) / 2)
            displacements.append(disp)

            if show_pairs:
                h, w = g1.shape; px = h * w
                fc   = flow[valid]
                mm   = float(np.nanmean(np.hypot(fc[:,0], fc[:,1]))) if len(fc) else 0.0
                with st.expander(
                    f"Par {i+1}/{total_pairs}  t={t1:.2f}s → {t2:.2f}s  disp={disp:.2f}px",
                    expanded=(i == 0)):
                    BG_ = "#0d1117"; TEXT_ = "#e6edf3"
                    fig_p, axes_p = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG_)
                    axes_p = axes_p.flatten()
                    def rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    def hsvc(f):
                        fv = f.copy(); fv[np.isnan(fv)] = 0
                        mag_,ang_ = cv2.cartToPolar(fv[...,0],fv[...,1])
                        h_ = np.zeros((*f.shape[:2],3),np.uint8)
                        h_[...,0]=ang_*180/np.pi/2; h_[...,1]=255
                        h_[...,2]=cv2.normalize(mag_,None,0,255,cv2.NORM_MINMAX)
                        return cv2.cvtColor(cv2.cvtColor(h_,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2RGB)
                    tkw = dict(color=TEXT_, fontsize=10, fontweight="bold", pad=5)
                    for ax_ in axes_p:
                        ax_.set_xticks([]); ax_.set_yticks([])
                        for sp in ax_.spines.values(): sp.set_edgecolor("#30363d")
                    axes_p[0].imshow(rgb(img1)); axes_p[0].set_title(f"[A] t={t1:.2f}s", **tkw)
                    axes_p[1].imshow(rgb(img2)); axes_p[1].set_title(f"[B] t={t2:.2f}s", **tkw)
                    axes_p[2].imshow(rgb(img2))
                    ov_ = np.zeros((*smoke.shape,4),np.float32); ov_[smoke]=[1.,.5,0.,.5]
                    axes_p[2].imshow(ov_)
                    axes_p[2].set_title("[C] Máscara humo", **tkw)
                    axes_p[3].imshow(diff, cmap="hot", vmin=0, vmax=80)
                    cnts_,_ = cv2.findContours(motion.astype(np.uint8),cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    for cnt_ in cnts_:
                        pts_ = cnt_[:,0,:]
                        axes_p[3].plot(np.append(pts_[:,0],pts_[0,0]),
                                       np.append(pts_[:,1],pts_[0,1]),
                                       color="#00ff88",linewidth=1.2,alpha=0.9)
                    axes_p[3].set_title("[D] Diferencia + movimiento", **tkw)
                    axes_p[4].imshow(hsvc(flow))
                    axes_p[4].set_title("[E] Flujo óptico (color=dir)", **tkw)
                    axes_p[5].imshow(rgb(img2))
                    gh,gw = flow.shape[:2]; yc_,xc_ = np.mgrid[0:gh:flow_step,0:gw:flow_step]
                    fx_,fy_ = flow[yc_,xc_,0], flow[yc_,xc_,1]
                    vm_ = ~np.isnan(fx_) & ~np.isnan(fy_)
                    vm_ &= np.hypot(np.where(np.isnan(fx_),0,fx_),
                                    np.where(np.isnan(fy_),0,fy_)) > 0.5
                    if vm_.any():
                        axes_p[5].quiver(xc_[vm_],yc_[vm_],fx_[vm_],fy_[vm_],
                                         color="#00e5ff",scale=None,scale_units="xy",
                                         angles="xy",width=0.003,headwidth=4,headlength=5,
                                         alpha=0.85)
                    axes_p[5].set_title("[F] Vectores sobre Frame B", **tkw)
                    fig_p.suptitle("Análisis de Flujo Óptico", color=CYAN, fontsize=13, y=1.0)
                    plt.tight_layout()
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close(fig_p)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Humo",             f"{smoke.sum()/px*100:.1f}%")
                    c2.metric("Movimiento válido", f"{motion.sum()/px*100:.1f}%")
                    c3.metric("Magnitud media",    f"{mm:.2f} px/frame")

            bar.progress((i + 1) / total_pairs, text=f"Par {i+1}/{total_pairs}")

        bar.empty()
        st.session_state.update({
            "timestamps": timestamps, "displacements": displacements,
            "analysis_done": True, "lstm_result": None,
        })

    timestamps    = st.session_state["timestamps"]
    displacements = st.session_state["displacements"]
    if not timestamps:
        return

    ts_arr = np.array(timestamps)
    ds_arr = np.array(displacements)

    _vid_proc_key = (
        f"{st.session_state['cache_key']}|{signal_method}|{signal_window}|"
        f"{signal_polyord}|{signal_cutoff}|{signal_fourier_terms}|{signal_block_size}|"
        f"{frame_range[0]}|{frame_range[1]}|"
        f"{out_active_sb}|{out_order_sb if out_active_sb else ''}|"
        f"{out_method_sb if out_active_sb else ''}|"
        f"{out_iqr_k_sb if out_active_sb else ''}|"
        f"{out_zscore_sb if out_active_sb else ''}"
    )
    if st.session_state.get("vid_processing_key") != _vid_proc_key:
        st.session_state["lstm_result"] = None
        st.session_state["vid_processing_key"] = _vid_proc_key

    fa, fb = frame_range
    fb = min(fb + 1, len(ds_arr))
    fa = min(fa, fb - 1)

    processed_full = ds_arr.copy()
    if fb > fa:
        seg = ds_arr[fa:fb]
        seg_proc = apply_signal_processing(seg, method=signal_method,
            window=signal_window, polyorder=signal_polyord, cutoff=signal_cutoff,
            fourier_terms=signal_fourier_terms, block_size=signal_block_size)
        processed_full[fa:fb] = seg_proc

    ts_seg       = ts_arr[fa:fb]
    seg_for_lstm = processed_full[fa:fb].copy()

    if out_active_sb:
        raw_seg_base = ds_arr[fa:fb].copy()
        if out_order_sb == "Antes del suavizado":
            raw_clean = apply_outlier_filter(raw_seg_base, method=out_method_sb,
                iqr_k=out_iqr_k_sb, zscore_thr=out_zscore_sb,
                clip_min=out_cmin_sb, clip_max=out_cmax_sb, replace=out_replace_sb)
            seg_for_lstm = apply_signal_processing(raw_clean, method=signal_method,
                window=signal_window, polyorder=signal_polyord, cutoff=signal_cutoff,
                fourier_terms=signal_fourier_terms, block_size=signal_block_size)
        else:
            seg_for_lstm = apply_outlier_filter(seg_for_lstm, method=out_method_sb,
                iqr_k=out_iqr_k_sb, zscore_thr=out_zscore_sb,
                clip_min=out_cmin_sb, clip_max=out_cmax_sb, replace=out_replace_sb)

    _ts_iv_raw, _iv_raw   = compute_inv_vel(ds_arr[fa:fb], ts_seg)
    ts_iv_lstm, inv_lstm  = compute_inv_vel(seg_for_lstm, ts_seg)

    st.divider()
    st.subheader("Velocidad óptica y velocidad inversa")
    st.caption(f"Filtro activo: **{signal_method}**  |  Rango pares: {fa} → {fb-1}")

    fig_vel = build_velocity_figure(ts_arr, ds_arr, processed_full, (fa, fb), signal_method)
    st.plotly_chart(fig_vel, use_container_width=True, key="vid_fig_vel")

    fig_inv_init = build_inverse_velocity_figure(_ts_iv_raw, _iv_raw, inv_lstm)
    st.plotly_chart(fig_inv_init, use_container_width=True, key="vid_fig_inv_init")

    # STL
    if stl_show_disp or stl_show_inv:
        st.divider()
        st.subheader("Descomposición STL")

    if stl_show_disp:
        decomp_disp = stl_decompose(seg_for_lstm, ts_seg, period=stl_period)
        if decomp_disp:
            fig_stl_d = build_decomposition_figure(decomp_disp, "Desplazamiento (px/frame)",
                                                    "px/frame", ts_seg)
            if fig_stl_d:
                st.plotly_chart(fig_stl_d, use_container_width=True, key="vid_stl_disp")
            st.caption(f"STL Desplazamiento — Período: **{decomp_disp['period']}** muestras")
        else:
            st.warning("Serie demasiado corta para STL (desplazamiento).")

    if stl_show_inv:
        decomp_inv_stl = stl_decompose(inv_lstm, ts_iv_lstm, period=stl_period)
        if decomp_inv_stl:
            fig_stl_i = build_decomposition_figure(decomp_inv_stl, "Velocidad Inversa (1/v)",
                                                    "1/v", ts_iv_lstm)
            if fig_stl_i:
                st.plotly_chart(fig_stl_i, use_container_width=True, key="vid_stl_inv")
            st.caption(f"STL Vel. Inversa — Período: **{decomp_inv_stl['period']}** muestras")
        else:
            st.warning("Serie demasiado corta para STL (velocidad inversa).")

    # Resumen global
    st.divider()
    st.subheader("Resumen global")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames procesados",  len(frames))
    c2.metric("Pares analizados",   len(frames) - 1)
    c3.metric("Duración video",     f"{dur:.1f} s")
    c4.metric("Intervalo medio",    f"{dur / max(len(frames)-1, 1):.2f} s")
    if len(displacements):
        im = int(np.argmax(displacements))
        c5, c6 = st.columns(2)
        c5.metric("Desp. máximo", f"{max(displacements):.2f} px",
                  delta=f"t={timestamps[im]:.1f}s")
        c6.metric("Desp. medio",  f"{np.mean(displacements):.2f} px")

    # ── FukuzonoLSTM para Video/Imágenes ─────────────────────────────────────
    st.divider()
    st.subheader("Predicción FukuzonoLSTM — Velocidad Inversa")

    fig_lstm_in = go.Figure()
    fig_lstm_in.add_trace(go.Scatter(x=_ts_iv_raw, y=_iv_raw, mode="lines",
        name="1/v Raw", line=dict(color="#888", width=1, dash="dot"), opacity=0.5))
    fig_lstm_in.add_trace(go.Scatter(x=ts_iv_lstm, y=inv_lstm, mode="lines",
        name="1/v Procesada (→ entra al LSTM)",
        line=dict(color=PINK, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,157,0.07)"))
    fig_lstm_in.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Velocidad Inversa — entrada exacta al FukuzonoLSTM",
                   font=dict(color=CYAN, size=13)),
        xaxis_title="Tiempo (s)", yaxis_title="1/(px/frame)", height=280)
    st.plotly_chart(fig_lstm_in, use_container_width=True, key="vid_lstm_input")

    with st.expander("📋 Tabla de trazabilidad"):
        import pandas as pd
        _n_tr = min(len(_ts_iv_raw), len(_iv_raw), len(inv_lstm))
        _df_traz = pd.DataFrame({
            "t (s)":            np.round(_ts_iv_raw[:_n_tr], 4),
            "Desp. Raw (px)":   np.round(ds_arr[fa:fb][1:_n_tr+1], 6),
            "Desp. Proc. (px)": np.round(seg_for_lstm[1:_n_tr+1], 6),
            "1/v Raw":          np.round(_iv_raw[:_n_tr], 6),
            "1/v Procesada":    np.round(inv_lstm[:_n_tr], 6)})
        st.dataframe(_df_traz, use_container_width=True, hide_index=True)

    st.divider()
    lstm_btn = st.button("Entrenar FukuzonoLSTM y proyectar", type="primary")

    if lstm_btn:
        n_min = lstm_seq_len + 4
        if len(inv_lstm) < n_min:
            st.error(
                f"Serie demasiado corta ({len(inv_lstm)} puntos). "
                f"Necesitas al menos {n_min} puntos.")
        else:
            prog_bar = st.progress(0, text="Entrenando FukuzonoLSTM...")

            # Construir vel, disp y t_days desde la serie de velocidad inversa
            vel_vid   = 1.0 / (inv_lstm + 1e-8)
            disp_vid  = seg_for_lstm[1:]   # alinear con inv_lstm (len-1)
            t_days_vid = ts_iv_lstm / 86400.0

            _n = min(len(inv_lstm), len(disp_vid), len(t_days_vid))
            inv_v_ = inv_lstm[:_n]
            disp_  = disp_vid[:_n]
            vel_   = vel_vid[:_n]
            t_d_   = t_days_vid[:_n]

            try:
                pred_tr, fut_inv, fut_t_off, metr, hloss = train_fukuzono(
                    inv_v_, disp_, vel_, t_d_,
                    seq_len=lstm_seq_len,
                    hidden_dim=lstm_hidden, n_layers=lstm_layers,
                    dropout=lstm_dropout, bidirectional=lstm_bidir,
                    lr=lstm_lr, epochs=lstm_epochs, patience=lstm_patience,
                    reg_epochs=reg_epochs, reg_lr=reg_lr,
                    step_hours=lstm_step_h, max_steps=lstm_max_steps,
                    stop_threshold=lstm_stop_thr,
                    progress_cb=lambda f: prog_bar.progress(f, text=f"Entrenando... {f*100:.0f}%"))

                prog_bar.progress(1.0, text="¡Completado!")
                st.session_state["lstm_result"] = (
                    pred_tr, fut_inv, fut_t_off, metr, hloss,
                    ts_iv_lstm[:_n], inv_v_)
            except Exception as e:
                prog_bar.empty()
                st.error(f"Error en FukuzonoLSTM: {e}")
                st.session_state["lstm_result"] = None

    lr_res = st.session_state["lstm_result"]
    if lr_res:
        pred_tr, fut_inv, fut_t_off, metr, hloss, ts_lstm, iv_lstm = lr_res

        fig_main = build_fukuzono_figure(
            ts_lstm, iv_lstm, pred_tr,
            fut_t_off * 86400.0, fut_inv, metr, hloss, lstm_stop_thr)
        st.plotly_chart(fig_main, use_container_width=True, key="vid_fuku_main")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE",  f"{metr['MAE']:.5f}")
        mc2.metric("RMSE", f"{metr['RMSE']:.5f}")
        mc3.metric("MAPE", f"{metr['MAPE']:.2f}%")
        mc4.metric("R²",   f"{metr['R2']:.4f}")

        if len(fut_inv) > 0:
            st.info(
                f"**Falla proyectada:** offset {fut_t_off[-1]*24:.1f} horas desde el último dato  |  "
                f"Pasos autorregresivos: {len(fut_inv)}  |  "
                f"1/v final: {fut_inv[-1]:.5f}")

        with st.expander("Detalles del modelo y pronóstico"):
            import pandas as pd
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"""
| Parámetro | Valor |
|-----------|-------|
| Seq length | {metr['seq_len']} |
| Hidden dim | {metr['hidden_dim']} |
| Capas LSTM | {metr['n_layers']} |
| Bidireccional | {metr['bidirectional']} |
| Parámetros totales | {metr['n_params']:,} |
| Epochs ejecutados | {metr['epochs_run']} |
| Mejor val loss | {metr['best_val_loss']:.6f} |
""")
            with cb:
                _dt = pd.DataFrame({
                    "Paso": list(range(1, len(fut_inv) + 1)),
                    "Offset (h)": np.round(fut_t_off * 24, 1),
                    "1/v proyectado": np.round(fut_inv, 5),
                    "v estimada (px/s)": np.round(1.0 / (fut_inv + 1e-8), 5),
                })
                st.dataframe(_dt, use_container_width=True, hide_index=True)

    # ── Comparación personalizada de frames ───────────────────────────────────
    st.divider()
    st.subheader("🔄 Comparación personalizada de frames")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_av = len(frames)
    col_fa2, col_fb2 = st.columns(2)
    with col_fa2:
        raw_a = st.number_input("Frame inicial (A)", 1, n_av, 1, 1, key="cmp_a")
    with col_fb2:
        raw_b = st.number_input("Frame final (B)", 1, n_av,
                                min(n_av, max(2, n_av // 3)), 1, key="cmp_b")

    idx_a = int(raw_a) - 1
    idx_b = int(raw_b) - 1

    if idx_a == idx_b:
        st.warning("Selecciona dos frames distintos.")
        return

    t_a, img_a = frames[idx_a]
    t_b, img_b = frames[idx_b]
    img_a_roi = apply_roi(img_a); img_b_roi = apply_roi(img_b)
    pa, pb = st.columns(2)
    pa.image(cv2.cvtColor(img_a_roi, cv2.COLOR_BGR2RGB),
             caption=f"Frame {idx_a+1}  t={t_a:.2f}s  (ROI)", use_container_width=True)
    pb.image(cv2.cvtColor(img_b_roi, cv2.COLOR_BGR2RGB),
             caption=f"Frame {idx_b+1}  t={t_b:.2f}s  (ROI)", use_container_width=True)

    if st.button("Calcular comparación", type="secondary"):
        with st.spinner("Calculando flujos..."):
            ref = min(idx_a, idx_b); tgt = max(idx_a, idx_b)
            t_ref, img_ref = frames[ref]; t_tgt, img_tgt = frames[tgt]
            img_ref = apply_roi(img_ref); img_tgt = apply_roi(img_tgt)
            g_ref = to_gray(img_ref); g_tgt = to_gray(img_tgt)
            sm_ref = detect_smoke_mask(g_ref, dark_thresh=dark_thr, texture_thresh=tex_thr)
            sm_tgt = detect_smoke_mask(g_tgt, dark_thresh=dark_thr, texture_thresh=tex_thr)
            mid = max(ref + 1, (ref + tgt) // 2)
            if mid >= tgt: mid = tgt
            t_mid, img_mid = frames[mid]
            img_mid = apply_roi(img_mid)
            g_mid  = to_gray(img_mid)
            sm_mid = detect_smoke_mask(g_mid, dark_thresh=dark_thr, texture_thresh=tex_thr)
            mot_a, _ = detect_motion_opencv(g_ref, g_mid, sm_ref | sm_mid, diff_thr,
                                             min_area=flow_min_area)
            fl_a, vl_a = compute_optical_flow(g_ref, g_mid, mot_a, sm_ref | sm_mid,
                                               algo=flow_algo, params=flow_params)
            mot_b, _ = detect_motion_opencv(g_mid, g_tgt, sm_mid | sm_tgt, diff_thr,
                                             min_area=flow_min_area)
            fl_b, vl_b = compute_optical_flow(g_mid, g_tgt, mot_b, sm_mid | sm_tgt,
                                               algo=flow_algo, params=flow_params)

            fva = fl_a.copy(); fva[np.isnan(fva)] = 0
            fvb = fl_b.copy(); fvb[np.isnan(fvb)] = 0
            ma_ = np.hypot(fva[...,0], fva[...,1])
            mb_ = np.hypot(fvb[...,0], fvb[...,1])
            vu  = vl_a | vl_b
            ma_mean = float(np.nanmean(ma_[vu])) if vu.any() else 0.0
            mb_mean = float(np.nanmean(mb_[vu])) if vu.any() else 0.0

            fig_dist = go.Figure()
            if vu.any():
                fig_dist.add_trace(go.Histogram(x=ma_[vu].ravel(),
                    name=f"Magnitud A (Fr.{ref+1})",
                    marker_color=CYAN, opacity=0.7, nbinsx=40))
                fig_dist.add_trace(go.Histogram(x=mb_[vu].ravel(),
                    name=f"Magnitud B (Fr.{tgt+1})",
                    marker_color=PINK, opacity=0.7, nbinsx=40))
            fig_dist.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                title=dict(text=f"Distribución de magnitudes — Fr.{ref+1} vs Fr.{tgt+1}",
                           font=dict(color=CYAN)),
                xaxis_title="Magnitud (px/frame)", yaxis_title="Frecuencia", height=320)
            st.plotly_chart(fig_dist, use_container_width=True, key="cmp_hist")

            TEXT_ = "#e6edf3"
            fig_mat, axes_mat = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d1117")
            axes_mat = axes_mat.flatten()
            def rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tkw = dict(color=TEXT_, fontsize=10, fontweight="bold", pad=5)

            def qax(ax, img_rgb, flow, valid, color, title):
                ax.imshow(img_rgb)
                h_, w_ = flow.shape[:2]; yc_,xc_ = np.mgrid[0:h_:flow_step,0:w_:flow_step]
                fx_,fy_ = flow[yc_,xc_,0], flow[yc_,xc_,1]
                vm_ = ~np.isnan(fx_) & ~np.isnan(fy_)
                vm_ &= np.hypot(np.where(np.isnan(fx_),0,fx_),
                                np.where(np.isnan(fy_),0,fy_))>0.3
                if vm_.any():
                    ax.quiver(xc_[vm_],yc_[vm_],fx_[vm_],fy_[vm_],color=color,
                              scale=None,scale_units="xy",angles="xy",
                              width=0.003,headwidth=4,headlength=5,alpha=0.9)
                ax.set_title(title, **tkw); ax.set_xticks([]); ax.set_yticks([])

            qax(axes_mat[0], rgb(img_ref), fl_a, vl_a, "#00e5ff", f"[A] Fr.{ref+1} vectores")
            qax(axes_mat[1], rgb(img_tgt), fl_b, vl_b, PINK,      f"[B] Fr.{tgt+1} vectores")

            vmax_ = max(np.nanmax(ma_) if vu.any() else 1, np.nanmax(mb_) if vu.any() else 1, 1)
            sym_  = max(np.nanmax(np.abs(mb_ - ma_)) if vu.any() else 0.01, 0.01)
            for ax_, dat_, ttl_ in [
                (axes_mat[2], np.where(vu,ma_,np.nan), f"[C] Magnitud Fr.{ref+1}"),
                (axes_mat[3], np.where(vu,mb_,np.nan), f"[D] Magnitud Fr.{tgt+1}"),
            ]:
                im_ = ax_.imshow(dat_, cmap="plasma", vmin=0, vmax=vmax_)
                ax_.set_title(ttl_, **tkw); ax_.set_xticks([]); ax_.set_yticks([])
                plt.colorbar(im_, ax=ax_, fraction=0.046, pad=0.04)

            im4_ = axes_mat[4].imshow(np.where(vu,mb_-ma_,np.nan),
                                       cmap="RdBu_r", vmin=-sym_, vmax=sym_)
            axes_mat[4].set_title("[E] Delta magnitud (B-A)", **tkw)
            axes_mat[4].set_xticks([]); axes_mat[4].set_yticks([])
            plt.colorbar(im4_, ax=axes_mat[4], fraction=0.046, pad=0.04)

            ang_a_ = np.degrees(np.arctan2(fva[...,1], fva[...,0])) % 360
            ang_b_ = np.degrees(np.arctan2(fvb[...,1], fvb[...,0])) % 360
            dang_  = np.where(vu, ((ang_b_-ang_a_+180)%360)-180, np.nan)
            im5_ = axes_mat[5].imshow(dang_, cmap="twilight_shifted", vmin=-180, vmax=180)
            axes_mat[5].set_title("[F] Diferencia angular (grados)", **tkw)
            axes_mat[5].set_xticks([]); axes_mat[5].set_yticks([])
            plt.colorbar(im5_, ax=axes_mat[5], fraction=0.046, pad=0.04)

            fig_mat.suptitle(f"Comparación Fr.{ref+1} vs Fr.{tgt+1}",
                              color=CYAN, fontsize=13, y=1.0)
            plt.tight_layout()
            st.pyplot(fig_mat, use_container_width=True)
            plt.close(fig_mat)

            dpct = ((mb_mean - ma_mean) / max(ma_mean, 1e-9)) * 100
            xc1, xc2, xc3, xc4 = st.columns(4)
            xc1.metric(f"Mag. media Fr.{ref+1}", f"{ma_mean:.2f} px/frame")
            xc2.metric(f"Mag. media Fr.{tgt+1}", f"{mb_mean:.2f} px/frame",
                        delta=f"{dpct:+.1f}%")
            xc3.metric("Intervalo temporal", f"{abs(t_tgt - t_ref):.2f} s")
            xc4.metric("Frames diferencia",  str(tgt - ref))


if __name__ == "__main__":
    main()
