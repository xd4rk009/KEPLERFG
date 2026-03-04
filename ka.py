"""
ANÁLISIS ÓPTICO DE VÍDEO TÉRMICO — Streamlit v4 (optimizado)
Mejoras de rendimiento:
  - @st.cache_data en funciones pesadas de procesamiento
  - Eliminación de st.rerun() innecesarios
  - Keys únicas y estables en todos los st.plotly_chart
  - session_state inicializado de forma segura (sin sobrescribir)
  - Lógica de invalidación de caché por clave de parámetros (sin reruns)
  - Evitar recálculos repetidos de señal en cada interacción
"""

import warnings, tempfile, os
import numpy as np
import cv2
import streamlit as st
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# -- remove_small_objects replacement (avoids scikit-image dependency) --
def remove_small_objects(ar, min_size=64):
    """Pure-numpy drop-in for skimage.morphology.remove_small_objects."""
    ar = np.asarray(ar, dtype=bool)
    out = ar.copy()
    num_labels, labels = cv2.connectedComponents(ar.astype(np.uint8), connectivity=8)
    for lbl in range(1, num_labels):
        if np.sum(labels == lbl) < min_size:
            out[labels == lbl] = False
    return out

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

# ════════════════════════════════════════════════════════════════════════════
#  FUNCIONES CORE — flujo óptico avanzado
# ════════════════════════════════════════════════════════════════════════════

def to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── Detección de máscara ────────────────────────────────────────────────────

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


# ─── ALGORITMOS DE FLUJO ────────────────────────────────────────────────────

def _build_pyramid(img, levels, scale=0.5):
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


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


def _flow_raft_lite(g1, g2,
                    iters=12, corr_radius=4, corr_levels=4,
                    update_iters=6, alpha_smooth=0.5,
                    feature_channels=32, downsample_factor=4):
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
#  PROCESAMIENTO DE SEÑAL
# ════════════════════════════════════════════════════════════════════════════

def apply_outlier_filter(signal: np.ndarray, method: str = "IQR",
                          iqr_k: float = 1.5, zscore_thr: float = 3.0,
                          clip_min: float = None, clip_max: float = None,
                          replace: str = "interpolate") -> np.ndarray:
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


def apply_signal_processing(signal: np.ndarray, method: str,
                              window: int = 5, polyorder: int = 2,
                              cutoff: float = 0.1, fourier_terms: int = 10,
                              block_size: int = 5) -> np.ndarray:
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


def stl_decompose(signal: np.ndarray, timestamps: np.ndarray,
                   period: int = None) -> dict:
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


def build_decomposition_figure(decomp: dict, title: str,
                                 y_label: str = "Valor",
                                 timestamps: np.ndarray = None) -> go.Figure:
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
    fig.update_xaxes(title_text="Tiempo (s)", row=4, col=1,
                      gridcolor=BORDER, color="#8b949e")
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"Descomposición STL — {title}",
                   font=dict(color=CYAN, size=13)), height=580)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#cdd9e5", size=11)
    return fig


def render_outlier_and_decomp_ui(series_key: str, ts_arr: np.ndarray,
                                   raw_signal: np.ndarray,
                                   processed_signal: np.ndarray,
                                   signal_label: str = "Señal",
                                   decomp_period: int = None):
    st.markdown(f"##### Outliers y Descomposición — {signal_label}")
    col_out, col_decomp = st.columns([1, 1])
    with col_out:
        st.markdown("**Filtro de Outliers**")
        apply_out = st.checkbox("Activar filtro outliers",
                                 key=f"out_active_{series_key}", value=False)
        out_order = st.radio("Aplicar filtro",
                              ["Antes del tratamiento", "Después del tratamiento"],
                              horizontal=True, key=f"out_order_{series_key}")
        out_method = st.selectbox("Método",
                                   ["IQR", "Z-score", "Rango manual"],
                                   key=f"out_method_{series_key}")
        out_replace = st.selectbox("Reemplazar outlier con",
                                    ["interpolate", "mediana", "NaN→0"],
                                    key=f"out_replace_{series_key}")
        iqr_k = zscore_thr = 1.5
        clip_min = clip_max = None
        if out_method == "IQR":
            iqr_k = st.slider("Factor IQR (k)", 0.5, 5.0, 1.5, 0.1,
                               key=f"iqr_k_{series_key}")
        elif out_method == "Z-score":
            zscore_thr = st.slider("Umbral Z-score", 1.0, 6.0, 3.0, 0.1,
                                    key=f"zscore_{series_key}")
        elif out_method == "Rango manual":
            v_min = float(raw_signal.min()); v_max = float(raw_signal.max())
            clip_min = st.number_input("Valor mínimo permitido", value=v_min,
                                        key=f"clip_min_{series_key}")
            clip_max = st.number_input("Valor máximo permitido", value=v_max,
                                        key=f"clip_max_{series_key}")
    with col_decomp:
        st.markdown("**Descomposición STL**")
        do_decomp = st.checkbox("Mostrar descomposición",
                                 key=f"decomp_active_{series_key}", value=False)
        decomp_on = st.radio("Descomponer la señal",
                              ["Procesada (post-tratamiento)", "Raw (original)"],
                              horizontal=True, key=f"decomp_on_{series_key}")
        period_auto = st.checkbox("Período automático",
                                   key=f"period_auto_{series_key}", value=True)
        if not period_auto:
            decomp_period = st.slider("Período estacional (muestras)",
                                       2, max(3, len(raw_signal)//2),
                                       max(2, decomp_period or len(raw_signal)//8),
                                       key=f"decomp_period_{series_key}")
        else:
            decomp_period = None

    if apply_out and out_order == "Antes del tratamiento":
        final_signal = apply_outlier_filter(raw_signal, method=out_method,
            iqr_k=iqr_k, zscore_thr=zscore_thr,
            clip_min=clip_min, clip_max=clip_max, replace=out_replace)
        out_label = "Outliers quitados (antes del tratamiento)"
    else:
        final_signal = processed_signal.copy()
        out_label = ""

    if apply_out and out_order == "Después del tratamiento":
        final_signal = apply_outlier_filter(final_signal, method=out_method,
            iqr_k=iqr_k, zscore_thr=zscore_thr,
            clip_min=clip_min, clip_max=clip_max, replace=out_replace)
        out_label = "Outliers quitados (después del tratamiento)"

    if apply_out:
        fig_out = go.Figure()
        fig_out.add_trace(go.Scatter(x=ts_arr, y=raw_signal, mode="lines",
            name="Raw", line=dict(color=CYAN, width=1, dash="dot"), opacity=0.5))
        fig_out.add_trace(go.Scatter(x=ts_arr, y=processed_signal, mode="lines",
            name="Tratada", line=dict(color=ORANGE, width=1.5, dash="dash")))
        fig_out.add_trace(go.Scatter(x=ts_arr, y=final_signal, mode="lines",
            name=f"Final ({out_label})", line=dict(color=GREEN, width=2.2)))
        diff_mask = np.abs(final_signal - raw_signal) > 1e-12
        if diff_mask.any():
            fig_out.add_trace(go.Scatter(x=ts_arr[diff_mask], y=raw_signal[diff_mask],
                mode="markers", name="Outliers detectados",
                marker=dict(color=PINK, size=7, symbol="x")))
        fig_out.update_layout(**PLOTLY_LAYOUT,
            title=dict(text=f"Filtro Outliers — {signal_label}", font=dict(color=CYAN)),
            xaxis_title="Tiempo (s)", yaxis_title=signal_label, height=300)
        st.plotly_chart(fig_out, use_container_width=True,
                        key=f"out_fig_{series_key}")

    if do_decomp:
        sig_to_decomp = (raw_signal if decomp_on == "Raw (original)" else final_signal)
        decomp = stl_decompose(sig_to_decomp, ts_arr, period=decomp_period)
        if decomp:
            fig_dec = build_decomposition_figure(decomp, title=f"{signal_label}",
                                                  timestamps=ts_arr)
            if fig_dec:
                st.plotly_chart(fig_dec, use_container_width=True,
                                key=f"decomp_fig_{series_key}")
            period_used = decomp["period"]
            st.caption(
                f"Período detectado/usado: **{period_used}** muestras  |  "
                f"Varianza residual: {decomp['residual'].var():.6f}  |  "
                f"Varianza tendencia: {decomp['trend'].var():.6f}"
            )
        else:
            st.warning("Serie demasiado corta para descomposición.")

    return final_signal


# ════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS PLOTLY
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
        name="1/v Procesada (→ BiLSTM)", line=dict(color=PINK, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,157,0.07)"))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Velocidad Inversa (1/v) — proxy de lentitud / estasis",
                   font=dict(color=CYAN, size=14)),
        xaxis_title="Tiempo (s)", yaxis_title="1 / (px/frame)", height=380)
    return fig


def compute_inv_vel(disp: np.ndarray, timestamps: np.ndarray):
    d      = np.abs(np.diff(disp.astype(np.float64)))
    iv     = 1.0 / (d + 1e-9)
    finite = iv[np.isfinite(iv)]
    p99    = float(np.percentile(finite, 99)) if len(finite) > 1 else float(iv.max())
    iv     = np.clip(iv, 0.0, p99)
    ts_iv  = np.array(timestamps[1:], dtype=np.float64)
    return ts_iv, iv


def inv_vel_table(ts_iv, inv_vel_raw, inv_vel_proc, label=""):
    import pandas as pd
    n = min(len(ts_iv), len(inv_vel_raw), len(inv_vel_proc))
    return pd.DataFrame({
        "t (s)":           np.round(ts_iv[:n], 4),
        "1/v Raw":         np.round(inv_vel_raw[:n], 6),
        "1/v Procesada":   np.round(inv_vel_proc[:n], 6),
    })


def build_lstm_figure(ts, inv_vel, pred_train, future_vals, metrics, history_loss, horizon,
                      history_val=None):
    dt     = float(np.mean(np.diff(ts))) if len(ts) > 1 else 1.0
    ts_fut = ts[-1] + np.arange(1, horizon + 1) * dt
    rmse   = metrics["RMSE"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=["Serie real + pronóstico BiLSTM", "Curva de pérdida"])
    fig.add_trace(go.Scatter(x=ts, y=inv_vel, mode="lines", name="Real",
        line=dict(color=PINK, width=2)), row=1, col=1)
    vm = ~np.isnan(pred_train)
    if vm.any():
        fig.add_trace(go.Scatter(x=ts[vm], y=pred_train[vm], mode="lines",
            name="Reconstrucción BiLSTM",
            line=dict(color=YELLOW, width=1.8, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([ts_fut, ts_fut[::-1]]),
        y=np.concatenate([future_vals + rmse, (future_vals - rmse)[::-1]]),
        fill="toself", fillcolor="rgba(63,185,80,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±1 RMSE ({rmse:.4f})", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts_fut, y=future_vals, mode="lines+markers",
        name=f"Pronóstico (+{horizon} pasos)",
        line=dict(color=GREEN, width=2.2), marker=dict(size=6)), row=1, col=1)
    fig.add_vline(x=float(ts[-1]), line_dash="dot", line_color=BORDER, row=1, col=1)
    fig.add_trace(go.Scatter(y=history_loss, mode="lines", name="Loss entrenamiento",
        line=dict(color=CYAN, width=1.8)), row=1, col=2)
    if history_val:
        fig.add_trace(go.Scatter(y=history_val, mode="lines", name="Loss validación",
            line=dict(color=PINK, width=1.8)), row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Predicción BiLSTM Bidireccional — Velocidad Inversa",
                   font=dict(color=CYAN, size=14)), height=420)
    fig.update_yaxes(title_text="1/(px/frame)", row=1, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
    fig.update_yaxes(title_text="MSE (log)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#cdd9e5", size=12)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_features(series: np.ndarray, lookback: int) -> np.ndarray:
    n  = len(series)
    mn = series.min()
    mx = series.max()
    rng = mx - mn if mx != mn else 1.0
    s   = (series - mn) / rng

    eps = 1e-9
    velocity     = 1.0 / (s + eps)
    d_inv_vel    = np.concatenate([[0.0], np.diff(s)])
    acceleration = np.concatenate([[0.0], np.diff(velocity)])
    jerk         = np.concatenate([[0.0], np.diff(acceleration)])

    log_inv_vel   = np.log(np.abs(s)       + eps)
    log_velocity  = np.log(np.abs(velocity) + eps)
    log_accel_abs = np.log(np.abs(acceleration) + eps)

    def _ewm(x, alpha):
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out

    ewm_fast = _ewm(s, 0.3)
    ewm_slow = _ewm(s, 0.05)
    ewm_ratio = ewm_fast / (ewm_slow + eps)

    def _roll_stat(x, w):
        pad = np.pad(x, (w - 1, 0), mode='edge')
        wins = np.lib.stride_tricks.sliding_window_view(pad, w)
        return wins.mean(axis=1), wins.std(axis=1)

    rmean5, rstd5 = _roll_stat(s, 5)
    rstd5_norm    = rstd5 / (rmean5 + eps)
    residual_tr   = s - rmean5
    slope_angle = np.arctan(d_inv_vel / (1.0 / n + eps))

    feats = np.stack([
        s, velocity, d_inv_vel, acceleration, jerk,
        log_inv_vel, log_velocity, log_accel_abs,
        ewm_fast, ewm_slow, ewm_ratio,
        rmean5, rstd5, rstd5_norm, residual_tr, slope_angle,
    ], axis=1)

    feats = np.clip(feats, -1e6, 1e6)

    M      = n - lookback
    x_lin  = np.arange(lookback, dtype=np.float64)
    x_mean = x_lin.mean()
    x_var  = ((x_lin - x_mean) ** 2).sum()

    s_wins = np.lib.stride_tricks.sliding_window_view(
        s.astype(np.float64), lookback)[:M]

    y_mean = s_wins.mean(axis=1, keepdims=True)
    slopes = ((s_wins - y_mean) * (x_lin - x_mean)).sum(axis=1) / x_var

    Vander = np.stack([x_lin**2, x_lin, np.ones(lookback)], axis=1)
    pinv_V = np.linalg.pinv(Vander)
    curvs  = (pinv_V @ s_wins.T)[0]

    row_idx  = np.arange(M)[:, None] + np.arange(lookback)[None, :]
    feat_wins = feats[row_idx]

    slope_feat = np.repeat(slopes[:, None], lookback, axis=1).astype(np.float32)
    curv_feat  = np.repeat(curvs[:, None],  lookback, axis=1).astype(np.float32)
    extra = np.stack([slope_feat, curv_feat], axis=2)

    X = np.concatenate([feat_wins, extra], axis=2).astype(np.float32)
    y = s[lookback:].astype(np.float32)

    return X, y, mn, rng


N_FEATURES = 18

# ════════════════════════════════════════════════════════════════════════════
#  MODELO HÍBRIDO
# ════════════════════════════════════════════════════════════════════════════

def fit_trend(series: np.ndarray, trend_type: str = "auto"):
    from scipy.optimize import curve_fit

    n = len(series)
    x = np.arange(n, dtype=np.float64)
    y = series.astype(np.float64)
    results = {}

    def exp_func(x, a, b, c):
        return a * np.exp(np.clip(b * x, -30, 30)) + c

    try:
        p0_exp = [y[0] - y[-1], -3.0 / n, y[-1]]
        popt_e, _ = curve_fit(exp_func, x, y, p0=p0_exp, maxfev=8000,
                               bounds=([-np.inf, -10/n, -np.inf],
                                       [np.inf,   0,     np.inf]))
        fitted_e = exp_func(x, *popt_e)
        ss_res = np.sum((y - fitted_e)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2_e = 1 - ss_res / (ss_tot + 1e-12)
        results["exponential"] = dict(r2=r2_e, params=popt_e, fitted=fitted_e,
            fn=lambda xf, p=popt_e: exp_func(xf, *p))
    except Exception:
        pass

    def power_func(x, a, b, c):
        return a * np.power(np.clip(x + 1, 1e-6, None), b) + c

    try:
        p0_pw = [y[0], -0.5, y[-1]]
        popt_pw, _ = curve_fit(power_func, x, y, p0=p0_pw, maxfev=8000)
        fitted_pw = power_func(x, *popt_pw)
        ss_res = np.sum((y - fitted_pw)**2)
        r2_pw = 1 - ss_res / (np.sum((y - y.mean())**2) + 1e-12)
        results["power"] = dict(r2=r2_pw, params=popt_pw, fitted=fitted_pw,
            fn=lambda xf, p=popt_pw: power_func(xf, *p))
    except Exception:
        pass

    def logistic_func(x, a, b, c, d):
        return a / (1.0 + np.exp(np.clip(b * (x - c), -30, 30))) + d

    try:
        p0_lg = [y[0] - y[-1], 6.0 / n, n * 0.8, y[-1]]
        popt_lg, _ = curve_fit(logistic_func, x, y, p0=p0_lg, maxfev=10000,
                                bounds=([0, 0, 0, -np.inf],
                                        [np.inf, np.inf, n*2, np.inf]))
        fitted_lg = logistic_func(x, *popt_lg)
        ss_res = np.sum((y - fitted_lg)**2)
        r2_lg = 1 - ss_res / (np.sum((y - y.mean())**2) + 1e-12)
        results["logistic"] = dict(r2=r2_lg, params=popt_lg, fitted=fitted_lg,
            fn=lambda xf, p=popt_lg: logistic_func(xf, *p))
    except Exception:
        pass

    coeffs = np.polyfit(x, y, 3)
    fitted_poly = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted_poly)**2)
    r2_poly = 1 - ss_res / (np.sum((y - y.mean())**2) + 1e-12)
    results["polynomial"] = dict(r2=r2_poly, params=coeffs, fitted=fitted_poly,
        fn=lambda xf, c=coeffs: np.polyval(c, xf))

    if trend_type == "auto":
        best_key = max(results, key=lambda k: results[k]["r2"])
    elif trend_type in results:
        best_key = trend_type
    else:
        best_key = "polynomial"

    best = results[best_key]
    residual = y - best["fitted"]

    return dict(trend_vals=best["fitted"], residual=residual,
                forecast_fn=best["fn"], trend_type=best_key,
                r2=best["r2"], all_r2={k: v["r2"] for k, v in results.items()},
                x_end=float(n - 1))


def train_hybrid(series: np.ndarray, lookback: int = 15, horizon: int = 5,
                 hidden_dim: int = 48, n_layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = True, lr: float = 1e-3, epochs: int = 150,
                 batch_size: int = 1, weight_decay: float = 1e-4, patience: int = 20,
                 scheduler_step: int = 30, scheduler_gamma: float = 0.5,
                 trend_type: str = "auto"):
    n = len(series)
    trend_info = fit_trend(series, trend_type=trend_type)
    trend_vals = trend_info["trend_vals"]
    residual   = trend_info["residual"]

    res_pred_full, res_future, res_metrics, history_loss, history_val = train_bilstm(
        residual, lookback=lookback, horizon=horizon,
        hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=dropout, bidirectional=bidirectional,
        lr=lr, epochs=epochs, batch_size=batch_size,
        weight_decay=weight_decay, patience=patience,
        scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma)

    pred_full = res_pred_full.copy()
    valid_mask = ~np.isnan(pred_full)
    pred_full[valid_mask] = pred_full[valid_mask] + trend_vals[valid_mask]

    x_future = trend_info["x_end"] + np.arange(1, horizon + 1)
    trend_future = trend_info["forecast_fn"](x_future)
    future_vals = res_future + trend_future

    real_seg  = series[valid_mask]
    pred_seg  = pred_full[valid_mask]
    mae  = float(np.mean(np.abs(real_seg - pred_seg)))
    rmse = float(np.sqrt(np.mean((real_seg - pred_seg)**2)))
    mape = float(np.mean(np.abs((real_seg - pred_seg) / (np.abs(real_seg) + 1e-8))) * 100)
    ss_r = np.sum((real_seg - pred_seg)**2)
    ss_t = np.sum((real_seg - real_seg.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-12))

    metrics = dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2,
                   n_params=res_metrics["n_params"], epochs_run=res_metrics["epochs_run"],
                   best_val_loss=res_metrics["best_val_loss"],
                   lookback=lookback, horizon=horizon, hidden_dim=hidden_dim,
                   n_layers=n_layers, bidirectional=bidirectional, dropout=dropout, lr=lr,
                   trend_type=trend_info["trend_type"], trend_r2=trend_info["r2"])

    return pred_full, future_vals, metrics, history_loss, history_val, trend_info


def build_hybrid_figure(ts, series, pred_full, future_vals, trend_info,
                        metrics, history_loss, horizon, history_val=None):
    dt     = float(np.mean(np.diff(ts))) if len(ts) > 1 else 1.0
    ts_fut = ts[-1] + np.arange(1, horizon + 1) * dt
    rmse   = metrics["RMSE"]
    trend  = trend_info["trend_vals"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Serie real + pronóstico híbrido", "Curva de pérdida (residuo)",
                        "Tendencia paramétrica ajustada", "Residuo real vs. predicho (BiLSTM)"],
        vertical_spacing=0.14, horizontal_spacing=0.10)

    fig.add_trace(go.Scatter(x=ts, y=series, mode="lines", name="Real",
        line=dict(color=PINK, width=2)), row=1, col=1)
    vm = ~np.isnan(pred_full)
    if vm.any():
        fig.add_trace(go.Scatter(x=ts[vm], y=pred_full[vm], mode="lines",
            name="Reconstrucción híbrida",
            line=dict(color=YELLOW, width=1.8, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([ts_fut, ts_fut[::-1]]),
        y=np.concatenate([future_vals + rmse, (future_vals - rmse)[::-1]]),
        fill="toself", fillcolor="rgba(63,185,80,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±1 RMSE ({rmse:.4f})", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts_fut, y=future_vals, mode="lines+markers",
        name=f"Pronóstico híbrido (+{horizon} pasos)",
        line=dict(color=GREEN, width=2.4), marker=dict(size=7, symbol="diamond")), row=1, col=1)
    fig.add_vline(x=float(ts[-1]), line_dash="dot", line_color=BORDER, row=1, col=1)

    fig.add_trace(go.Scatter(y=history_loss, mode="lines", name="Loss entrenamiento",
        line=dict(color=CYAN, width=1.8), showlegend=True), row=1, col=2)
    if history_val:
        fig.add_trace(go.Scatter(y=history_val, mode="lines", name="Loss validación",
            line=dict(color=PINK, width=1.8), showlegend=True), row=1, col=2)

    fig.add_trace(go.Scatter(x=ts, y=series, mode="lines", name="Serie real",
        line=dict(color=PINK, width=1.5, dash="dot"), opacity=0.5,
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=trend, mode="lines",
        name=f"Tendencia ({trend_info['trend_type']}, R²={trend_info['r2']:.3f})",
        line=dict(color=ORANGE, width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ts_fut, y=trend_info["forecast_fn"](trend_info["x_end"] + np.arange(1, horizon + 1)),
        mode="lines", name="Tendencia extrapolada",
        line=dict(color=ORANGE, width=1.8, dash="dash"), showlegend=False), row=2, col=1)

    residual = trend_info["residual"]
    fig.add_trace(go.Scatter(x=ts, y=residual, mode="lines", name="Residuo real",
        line=dict(color="#c678dd", width=1.5)), row=2, col=2)
    if vm.any():
        res_pred = pred_full[vm] - trend[vm]
        fig.add_trace(go.Scatter(x=ts[vm], y=res_pred, mode="lines",
            name="Residuo predicho (BiLSTM)",
            line=dict(color=YELLOW, width=1.5, dash="dash")), row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color=BORDER, row=2, col=2)

    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"Modelo Híbrido — Tendencia {trend_info['trend_type']} + BiLSTM residuo",
                   font=dict(color=CYAN, size=14)), height=620)
    for r, c, yt, xt in [(1,1,"1/(px/frame)","Tiempo (s)"), (1,2,"MSE (log)","Epoch"),
                          (2,1,"1/(px/frame)","Tiempo (s)"), (2,2,"Residuo","Tiempo (s)")]:
        fig.update_yaxes(title_text=yt, row=r, col=c, gridcolor=BORDER)
        fig.update_xaxes(title_text=xt, row=r, col=c, gridcolor=BORDER)
    fig.update_yaxes(type="log", row=1, col=2)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#cdd9e5", size=11)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  MODELO BiLSTM — NumPy puro con BPTT real + Adam
# ════════════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))),
                    np.exp(np.clip(x, -30, 30)) / (1.0 + np.exp(np.clip(x, -30, 30))))

def _tanh(x):    return np.tanh(np.clip(x, -15, 15))
def _dsigmoid(s): return s * (1.0 - s)
def _dtanh(t):    return 1.0 - t ** 2

def _xavier(n_in, n_out, rng):
    lim = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-lim, lim, (n_in, n_out)).astype(np.float64)

def _huber_loss_and_grad(pred, target, delta=0.5):
    r = pred - target
    abs_r = np.abs(r)
    loss  = np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))
    grad  = np.where(abs_r <= delta, r, delta * np.sign(r))
    return float(loss.mean()), grad / len(pred)


class _AdamVar:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.lr  = lr; self.b1 = b1; self.b2 = b2
        self.eps = eps; self.wd = wd
        self.m   = np.zeros(shape, np.float64)
        self.v   = np.zeros(shape, np.float64)
        self.t   = 0

    def step(self, w, g):
        self.t += 1
        g = g + self.wd * w
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return w - self.lr * mh / (np.sqrt(vh) + self.eps)


class _LSTMCell:
    def __init__(self, input_dim, hidden, rng, lr, wd):
        self.H = hidden
        self.W  = _xavier(input_dim + hidden, 4 * hidden, rng)
        self.b  = np.zeros(4 * hidden, np.float64)
        self.aW = _AdamVar(self.W.shape, lr=lr, wd=wd)
        self.ab = _AdamVar(self.b.shape, lr=lr, wd=wd)

    def forward(self, x_seq):
        T = len(x_seq); H = self.H
        h = np.zeros(H, np.float64)
        c = np.zeros(H, np.float64)
        hs, cs, gates_pre, gates_post, xhs = [], [], [], [], []
        for t in range(T):
            xh  = np.concatenate([x_seq[t], h])
            g   = xh @ self.W + self.b
            i_  = _sigmoid(g[:H]);      f_  = _sigmoid(g[H:2*H])
            o_  = _sigmoid(g[2*H:3*H]); g_  = _tanh(g[3*H:])
            c_n = f_ * c + i_ * g_
            h_n = o_ * _tanh(c_n)
            gates_pre.append(g);     gates_post.append((i_, f_, o_, g_))
            cs.append(c.copy());     hs.append(h_n)
            xhs.append(xh)
            h = h_n; c = c_n
        return np.array(hs), (xhs, gates_pre, gates_post, cs, np.array(hs))

    def backward(self, dh_seq, cache):
        xhs, gates_pre, gates_post, cs_list, hs = cache
        T = len(dh_seq); H = self.H
        dW = np.zeros_like(self.W); db = np.zeros_like(self.b)
        dh_next = np.zeros(H, np.float64)
        dc_next = np.zeros(H, np.float64)
        dX = np.zeros((T, xhs[0].shape[0] - H), np.float64)
        for t in reversed(range(T)):
            dh = dh_seq[t] + dh_next
            i_, f_, o_, g_ = gates_post[t]
            c_prev = cs_list[t]
            c_cur  = f_ * c_prev + i_ * g_
            tc     = _tanh(c_cur)
            do  = dh * tc
            dc  = dh * o_ * _dtanh(tc) + dc_next
            df  = dc * c_prev
            di  = dc * g_
            dg  = dc * i_
            dg_pre = np.concatenate([
                di * _dsigmoid(i_), df * _dsigmoid(f_),
                do * _dsigmoid(o_), dg * _dtanh(g_)])
            dxh = dg_pre @ self.W.T
            dW += np.outer(xhs[t], dg_pre)
            db += dg_pre
            dX[t]    = dxh[:xhs[t].shape[0] - H]
            dh_next  = dxh[xhs[t].shape[0] - H:]
            dc_next  = dc * f_
        for arr in [dW, db]:
            np.clip(arr, -1.0, 1.0, out=arr)
        self.W = self.aW.step(self.W, dW)
        self.b = self.ab.step(self.b, db)
        return dX


class _DenseLayer:
    def __init__(self, n_in, n_out, rng, lr, wd, activation="linear"):
        self.W    = _xavier(n_in, n_out, rng)
        self.b    = np.zeros(n_out, np.float64)
        self.act  = activation
        self.aW   = _AdamVar(self.W.shape, lr=lr, wd=wd)
        self.ab   = _AdamVar(self.b.shape, lr=lr, wd=wd)
        self._last_x = None; self._last_z = None

    def forward(self, x):
        self._last_x = x.copy()
        z = x @ self.W + self.b
        self._last_z = z.copy()
        if self.act == "relu":   return np.maximum(0, z)
        if self.act == "tanh":   return _tanh(z)
        return z

    def backward(self, dout):
        if self.act == "relu":  dout = dout * (self._last_z > 0)
        elif self.act == "tanh": dout = dout * _dtanh(_tanh(self._last_z))
        dW = np.outer(self._last_x, dout)
        np.clip(dW, -1.0, 1.0, out=dW)
        np.clip(dout, -1.0, 1.0, out=dout)
        self.W = self.aW.step(self.W, dW)
        self.b = self.ab.step(self.b, dout)
        return dout @ self.W.T


def train_bilstm(series: np.ndarray, lookback: int = 15, horizon: int = 5,
                 hidden_dim: int = 48, n_layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = True, lr: float = 1e-3, epochs: int = 150,
                 batch_size: int = 1, weight_decay: float = 1e-4, patience: int = 20,
                 scheduler_step: int = 30, scheduler_gamma: float = 0.5):
    rng_np = np.random.RandomState(42)
    X, y_all, mn_s, rng_s = build_features(series, lookback)
    X = X.astype(np.float64); y_all = y_all.astype(np.float64)
    N = len(X)
    if N < 6:
        raise ValueError(f"Serie demasiado corta ({len(series)} puntos). "
                         f"Necesitas al menos {lookback + 6} puntos.")
    split  = max(2, int(N * 0.8))
    Xtr, ytr = X[:split], y_all[:split]
    Xte, yte = X[split:], y_all[split:]
    inp_dim  = N_FEATURES
    n_dirs   = 2 if bidirectional else 1
    lstm_out = hidden_dim * n_dirs
    lstm_fwd = _LSTMCell(inp_dim, hidden_dim, rng_np, lr, weight_decay)
    lstm_bwd = _LSTMCell(inp_dim, hidden_dim, rng_np, lr, weight_decay) if bidirectional else None
    proj1    = _DenseLayer(lstm_out, max(32, lstm_out // 2), rng_np, lr, weight_decay, "relu")
    proj2    = _DenseLayer(max(32, lstm_out // 2), 1, rng_np, lr, weight_decay, "linear")
    n_params = (lstm_fwd.W.size + lstm_fwd.b.size) * n_dirs + \
               proj1.W.size + proj1.b.size + proj2.W.size + proj2.b.size

    def _forward(x_seq):
        h_f, cache_f = lstm_fwd.forward(x_seq)
        if bidirectional:
            h_b, cache_b = lstm_bwd.forward(x_seq[::-1])
            h_b = h_b[::-1]
            context = np.concatenate([h_f[-1], h_b[-1]])
        else:
            context = h_f[-1]; cache_b = None
        h1 = proj1.forward(context)
        out = proj2.forward(h1)
        return float(out[0]), (cache_f, cache_b, context, h1)

    def _backward(dy, caches):
        cache_f, cache_b, context, h1 = caches
        dout = np.array([dy], np.float64)
        dh1  = proj2.backward(dout)
        dc   = proj1.backward(dh1)
        if bidirectional:
            dc_f = dc[:hidden_dim]; dc_b = dc[hidden_dim:]
            dh_f = np.zeros((len(cache_f[0]), hidden_dim), np.float64)
            dh_b = np.zeros((len(cache_b[0]), hidden_dim), np.float64)
            dh_f[-1] = dc_f; dh_b[-1] = dc_b
            lstm_fwd.backward(dh_f, cache_f)
            lstm_bwd.backward(dh_b[::-1], cache_b)
        else:
            dh_f = np.zeros((len(cache_f[0]), hidden_dim), np.float64)
            dh_f[-1] = dc
            lstm_fwd.backward(dh_f, cache_f)

    history_loss = []; history_val = []
    best_val = np.inf; best_W = None; wait_es = 0; current_lr = lr

    for ep in range(epochs):
        if ep > 0 and ep % scheduler_step == 0:
            current_lr *= scheduler_gamma
            for obj in [lstm_fwd, lstm_bwd, proj1, proj2]:
                if obj is None: continue
                for av in (obj.aW, obj.ab) if hasattr(obj, 'aW') else []:
                    av.lr = current_lr
        idx_shuf = rng_np.permutation(len(Xtr))
        ep_loss  = 0.0
        _bs = max(1, min(batch_size, len(Xtr)))
        for b_start in range(0, len(Xtr), _bs):
            b_idx = idx_shuf[b_start: b_start + _bs]
            for i in b_idx:
                pred, caches = _forward(Xtr[i])
                loss, dy = _huber_loss_and_grad(np.array([pred]), np.array([ytr[i]]))
                _backward(float(dy[0]), caches)
                ep_loss += loss
        history_loss.append(ep_loss / len(Xtr))
        if len(Xte) > 0:
            val_preds = np.array([_forward(Xte[j])[0] for j in range(len(Xte))])
            val_loss, _ = _huber_loss_and_grad(val_preds, yte)
        else:
            val_loss = history_loss[-1]
        history_val.append(float(val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_W = {
                'fW': lstm_fwd.W.copy(), 'fb': lstm_fwd.b.copy(),
                'bW': lstm_bwd.W.copy() if bidirectional else None,
                'bb': lstm_bwd.b.copy() if bidirectional else None,
                'p1W': proj1.W.copy(), 'p1b': proj1.b.copy(),
                'p2W': proj2.W.copy(), 'p2b': proj2.b.copy(),
            }
            wait_es = 0
        else:
            wait_es += 1
            if wait_es >= patience:
                break

    if best_W:
        lstm_fwd.W = best_W['fW']; lstm_fwd.b = best_W['fb']
        if bidirectional: lstm_bwd.W = best_W['bW']; lstm_bwd.b = best_W['bb']
        proj1.W = best_W['p1W'];  proj1.b = best_W['p1b']
        proj2.W = best_W['p2W'];  proj2.b = best_W['p2b']

    denorm = lambda v: float(v) * rng_s + mn_s

    tr_pred = np.array([_forward(Xtr[i])[0] for i in range(len(Xtr))])
    te_pred = np.array([_forward(Xte[i])[0] for i in range(len(Xte))]) if len(Xte) else np.array([])

    pred_full = np.full(len(series), np.nan)
    for i, pv in enumerate(tr_pred):
        pred_full[i + lookback] = denorm(pv)
    for i, pv in enumerate(te_pred):
        pred_full[split + i + lookback] = denorm(pv)

    s_full = ((series - mn_s) / rng_s).astype(np.float64)

    def _win_features(buf_norm):
        arr  = np.array(buf_norm, np.float64)
        n_   = len(arr)
        eps_ = 1e-9
        velocity_   = 1.0 / (arr + eps_)
        d_inv_vel_  = np.concatenate([[0.0], np.diff(arr)])
        accel_      = np.concatenate([[0.0], np.diff(velocity_)])
        jerk_       = np.concatenate([[0.0], np.diff(accel_)])
        log_iv_     = np.log(np.abs(arr)        + eps_)
        log_vel_    = np.log(np.abs(velocity_)  + eps_)
        log_acc_    = np.log(np.abs(accel_)     + eps_)
        def _ewm_(x, a):
            o = np.empty_like(x); o[0] = x[0]
            for k in range(1, len(x)): o[k] = a*x[k] + (1-a)*o[k-1]
            return o
        ewm_f_ = _ewm_(arr, 0.3)
        ewm_s_ = _ewm_(arr, 0.05)
        ewm_r_ = ewm_f_ / (ewm_s_ + eps_)
        w5 = min(5, n_)
        _p5 = np.pad(arr, (w5-1, 0), mode='edge')
        _wins5 = np.lib.stride_tricks.sliding_window_view(_p5, w5)
        rm5_  = _wins5.mean(axis=1)
        rs5_  = _wins5.std(axis=1)
        rs5n_ = rs5_ / (rm5_ + eps_)
        res_  = arr - rm5_
        sa_   = np.arctan(d_inv_vel_ / (1.0/n_ + eps_))
        x_lin_ = np.arange(n_, dtype=np.float64)
        slope_iv_ = float(np.polyfit(x_lin_, arr, 1)[0])
        curv_iv_  = float(np.polyfit(x_lin_, arr, 2)[0])
        base = np.stack([arr, velocity_, d_inv_vel_, accel_, jerk_,
                         log_iv_, log_vel_, log_acc_,
                         ewm_f_, ewm_s_, ewm_r_,
                         rm5_, rs5_, rs5n_, res_, sa_], axis=1)
        extra = np.column_stack([np.full(n_, slope_iv_), np.full(n_, curv_iv_)])
        result = np.concatenate([base, extra], axis=1)
        return np.clip(result, -1e6, 1e6).astype(np.float64)

    win_buf = list(s_full[-lookback:])
    future_raw = []
    for _ in range(horizon):
        feat_win = _win_features(win_buf[-lookback:])
        nxt, _   = _forward(feat_win)
        future_raw.append(nxt)
        win_buf.append(nxt)

    future_vals = np.array([denorm(v) for v in future_raw])

    if len(te_pred) > 0:
        real_d = np.array([denorm(v) for v in yte])
        pred_d = np.array([denorm(v) for v in te_pred])
    else:
        real_d = np.array([denorm(v) for v in ytr])
        pred_d = np.array([denorm(v) for v in tr_pred])

    mae  = float(np.mean(np.abs(real_d - pred_d)))
    rmse = float(np.sqrt(np.mean((real_d - pred_d)**2)))
    mape = float(np.mean(np.abs((real_d - pred_d) / (np.abs(real_d)+1e-8))) * 100)
    ss_r = np.sum((real_d - pred_d)**2)
    ss_t = np.sum((real_d - real_d.mean())**2)
    r2   = float(1 - ss_r / (ss_t + 1e-12))

    metrics = dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2,
                   n_params=n_params, epochs_run=len(history_loss),
                   best_val_loss=float(best_val),
                   lookback=lookback, horizon=horizon, hidden_dim=hidden_dim,
                   n_layers=n_layers, bidirectional=bidirectional, dropout=dropout, lr=lr)

    return pred_full, future_vals, metrics, history_loss, history_val


# ════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def walk_forward_validation(series: np.ndarray, lookback: int = 15, horizon: int = 5,
                             hidden_dim: int = 48, n_layers: int = 1, dropout: float = 0.0,
                             bidirectional: bool = True, lr: float = 1e-3, epochs: int = 80,
                             batch_size: int = 1, weight_decay: float = 1e-4, patience: int = 15,
                             scheduler_step: int = 20, scheduler_gamma: float = 0.5,
                             n_splits: int = 5, min_train_frac: float = 0.5,
                             progress_cb=None):
    n = len(series)
    min_train = max(lookback + 6, int(n * min_train_frac))
    step = max(1, (n - min_train) // n_splits)
    cuts = []
    for k in range(n_splits):
        train_end = min_train + k * step
        test_end  = min(train_end + step, n)
        if test_end > train_end and train_end >= lookback + 6:
            cuts.append((train_end, test_end))

    if not cuts:
        raise ValueError(
            f"Serie demasiado corta para {n_splits} folds con min_train={min_train}. "
            f"Reduce n_splits o lookback.")

    folds = []
    for fold_idx, (tr_end, te_end) in enumerate(cuts):
        tr_series = series[:tr_end]
        if len(tr_series) < lookback + 6:
            folds.append(dict(fold=fold_idx + 1, error="train too short"))
            if progress_cb:
                progress_cb((fold_idx + 1) / len(cuts))
            continue
        try:
            pred_ext, _, _, _, _ = train_bilstm(
                series[:te_end], lookback=lookback, horizon=1,
                hidden_dim=hidden_dim, n_layers=n_layers,
                dropout=dropout, bidirectional=bidirectional,
                lr=lr, epochs=epochs, batch_size=batch_size,
                weight_decay=weight_decay, patience=patience,
                scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma)
            te_indices  = np.arange(tr_end, te_end)
            y_real      = series[te_indices]
            y_pred_raw  = pred_ext[te_indices]
            valid       = ~np.isnan(y_pred_raw)
            if valid.sum() < 2:
                folds.append(dict(fold=fold_idx + 1, error="not enough valid preds"))
                if progress_cb:
                    progress_cb((fold_idx + 1) / len(cuts))
                continue
            y_real = y_real[valid]
            y_pred = y_pred_raw[valid]
            mae  = float(np.mean(np.abs(y_real - y_pred)))
            rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
            mape = float(np.mean(np.abs((y_real - y_pred) / (np.abs(y_real) + 1e-8))) * 100)
            ss_r = float(np.sum((y_real - y_pred) ** 2))
            ss_t = float(np.sum((y_real - y_real.mean()) ** 2))
            r2   = float(1 - ss_r / (ss_t + 1e-12))
            folds.append(dict(fold=fold_idx + 1, train_size=tr_end,
                               test_size=int(valid.sum()),
                               mae=mae, rmse=rmse, mape=mape, r2=r2,
                               y_real=y_real, y_pred=y_pred))
        except Exception as e:
            folds.append(dict(fold=fold_idx + 1, error=str(e)))
        if progress_cb:
            progress_cb((fold_idx + 1) / (len(cuts) + 1))

    last_forecast = None
    try:
        pred_full_last, future_vals_last, metr_last, hloss_last, hval_last = train_bilstm(
            series, lookback=lookback, horizon=horizon,
            hidden_dim=hidden_dim, n_layers=n_layers,
            dropout=dropout, bidirectional=bidirectional,
            lr=lr, epochs=epochs, batch_size=batch_size,
            weight_decay=weight_decay, patience=patience,
            scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma)
        last_forecast = dict(future_vals=future_vals_last, pred_full=pred_full_last,
                             metrics=metr_last, history_loss=hloss_last, history_val=hval_last)
    except Exception as e:
        last_forecast = dict(error=str(e))

    if progress_cb:
        progress_cb(1.0)

    valid_folds = [f for f in folds if "error" not in f]
    if not valid_folds:
        raise ValueError("Ningún fold completó el entrenamiento correctamente.")

    agg = {
        "MAE_mean":  float(np.mean([f["mae"]  for f in valid_folds])),
        "MAE_std":   float(np.std( [f["mae"]  for f in valid_folds])),
        "RMSE_mean": float(np.mean([f["rmse"] for f in valid_folds])),
        "RMSE_std":  float(np.std( [f["rmse"] for f in valid_folds])),
        "MAPE_mean": float(np.mean([f["mape"] for f in valid_folds])),
        "MAPE_std":  float(np.std( [f["mape"] for f in valid_folds])),
        "R2_mean":   float(np.mean([f["r2"]   for f in valid_folds])),
        "R2_std":    float(np.std( [f["r2"]   for f in valid_folds])),
        "n_folds":   len(valid_folds),
    }

    return dict(folds=folds, agg=agg, last_forecast=last_forecast)


def build_wfv_figure(wfv_result: dict, series: np.ndarray,
                     title: str = "") -> go.Figure:
    valid_folds = [f for f in wfv_result["folds"] if "error" not in f]
    n = len(series)
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=["Predicciones por fold vs. Real", "Métricas por fold"],
        vertical_spacing=0.15, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=np.arange(n), y=series, mode="lines", name="Serie real",
        line=dict(color=PINK, width=2)), row=1, col=1)
    fold_colors = [GREEN, CYAN, YELLOW, ORANGE, "#c678dd", "#56b6c2", "#e06c75", "#ffd700"]
    for fi, fold in enumerate(valid_folds):
        tr_end = fold["train_size"]
        n_test = fold["test_size"]
        x_pred = np.arange(tr_end, tr_end + n_test)
        fig.add_trace(go.Scatter(x=x_pred, y=fold["y_pred"],
            mode="lines+markers", name=f"Fold {fold['fold']} pred.",
            line=dict(color=fold_colors[fi % len(fold_colors)], width=1.6, dash="dash"),
            marker=dict(size=4)), row=1, col=1)
        fig.add_vrect(x0=0, x1=tr_end,
            fillcolor=fold_colors[fi % len(fold_colors)],
            opacity=0.04, layer="below", line_width=0, row=1, col=1)
    fold_nums  = [f["fold"]  for f in valid_folds]
    fold_rmses = [f["rmse"]  for f in valid_folds]
    fold_maes  = [f["mae"]   for f in valid_folds]
    fig.add_trace(go.Bar(x=[f"Fold {n}" for n in fold_nums], y=fold_rmses,
        name="RMSE", marker_color=ORANGE, opacity=0.85), row=2, col=1)
    fig.add_trace(go.Bar(x=[f"Fold {n}" for n in fold_nums], y=fold_maes,
        name="MAE", marker_color=CYAN, opacity=0.85), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text=f"Walk-Forward Validation — {title}",
                   font=dict(color=CYAN, size=14)),
        height=600, barmode="group")
    fig.update_xaxes(title_text="Índice de muestra", row=1, col=1, gridcolor=BORDER)
    fig.update_yaxes(title_text="Valor", row=1, col=1, gridcolor=BORDER)
    fig.update_xaxes(title_text="Fold", row=2, col=1, gridcolor=BORDER)
    fig.update_yaxes(title_text="Error", row=2, col=1, gridcolor=BORDER)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color="#cdd9e5", size=11)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  CARGA DE FRAMES
# ════════════════════════════════════════════════════════════════════════════

def load_frames_from_images(uploaded_files, assumed_fps: float = 1.0):
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
        t = i / assumed_fps
        frames.append((t, img))
    if failed:
        st.warning(f"No se pudieron leer {len(failed)} imagen(es): {', '.join(failed[:5])}")
    if len(frames) < 2:
        raise ValueError("Menos de 2 imágenes válidas cargadas.")
    dur = frames[-1][0]
    fps = assumed_fps
    return frames, dur, fps


def extract_frames_from_video(video_bytes: bytes, n_frames: int):
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


# ════════════════════════════════════════════════════════════════════════════
#  CARGA EXCEL
# ════════════════════════════════════════════════════════════════════════════

def parse_excel_series(uploaded_file) -> dict:
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
#  MAIN — optimizado (sin st.rerun innecesarios, keys únicas, caché eficiente)
# ════════════════════════════════════════════════════════════════════════════

def _init_session_state():
    """Inicializa session_state solo si las claves no existen aún."""
    defaults = dict(
        cache_key=None, frames=None, duration_sec=None, fps=None,
        timestamps=None, displacements=None, analysis_done=False,
        lstm_result=None,
        excel_series=None, excel_cache_key=None, excel_lstm_results=None,
        excel_wfv_results={}, vid_wfv_result=None,
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
        assumed_fps = st.slider("FPS asumido (modo imágenes)", 0.1, 60.0, 1.0, 0.1,
                                help="Intervalo temporal entre imágenes. FPS=1 → 1 segundo.")
        diff_thr    = st.slider("Umbral movimiento",  2.0, 60.0, 12.0, 0.5)
        dark_thr    = st.slider("Oscuridad humo",     20, 150, 80, 2)
        tex_thr     = st.slider("Tratamiento Material en Suspensión", 0.5, 20.0, 3.5, 0.25)
        flow_step   = st.slider("Paso vectores (quiver)", 6, 40, 18, 2)
        show_pairs  = st.checkbox("Mostrar panel por cada par", value=False)

        st.divider()
        st.header("Algoritmo de Flujo Optico")
        flow_algo = st.selectbox("Algoritmo", [
            "RAFT-lite", "DIS", "LK-Pyramid", "Farneback"], index=0)
        st.caption({
            "RAFT-lite":  "Recurrent All-Pairs Field Transforms (NumPy). Volumen de correlacion 4D.",
            "DIS":        "Dense Inverse Search (OpenCV). Muy rapido, preciso.",
            "LK-Pyramid": "Lucas-Kanade piramidal sparse→denso via IDW.",
            "Farneback":  "Flujo polinomial denso clasico.",
        }[flow_algo])

        st.markdown("**Parametros comunes**")
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
        else:  # Farneback
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
        st.markdown("**Método de suavizado / filtrado**")
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
        st.markdown("**Filtro de Outliers — Desplazamiento**")
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
                out_cmin_sb = st.number_input("Valor mínimo permitido", value=0.0, key="cmin_sb")
                out_cmax_sb = st.number_input("Valor máximo permitido", value=999.0, key="cmax_sb")

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
        st.header("BiLSTM — Hiperparámetros")
        lstm_lookback = st.slider("Lookback (pasos historia)",  3, 60, 15, 1)
        lstm_horizon  = st.slider("Horizon (pasos a predecir)", 1, 30, 5,  1)
        lstm_arch_str = st.text_input("Arquitectura de capas (neuronas por capa)",
            value="128, 64",
            help="Ej: 128,64 → 2 capas. hidden_dim=máximo, n_layers=cantidad.")
        try:
            _arch_vals = [max(1, int(x.strip())) for x in lstm_arch_str.split(",") if x.strip()]
            if not _arch_vals: _arch_vals = [128]
        except ValueError:
            _arch_vals = [128]
            st.sidebar.warning("Formato inválido — usando 128 neuronas, 1 capa.")
        lstm_hidden = max(_arch_vals)
        lstm_layers = len(_arch_vals)
        st.sidebar.caption(
            f"→ {lstm_layers} capa{'s' if lstm_layers > 1 else ''} · "
            f"hidden_dim={lstm_hidden} · "
            f"neuronas: {' → '.join(str(v) for v in _arch_vals)}")
        lstm_dropout = st.slider("Dropout",              0.0, 0.7, 0.30, 0.05)
        lstm_bidir   = st.checkbox("Bidireccional", value=True)
        lstm_lr      = st.select_slider("Learning rate",
                        options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2],
                        value=1e-3, format_func=lambda x: f"{x:.0e}")
        lstm_epochs  = st.slider("Epochs máximos",       20, 300, 80, 10)
        lstm_batch   = st.slider("Batch size",           8, 128, 16, 8)
        lstm_wd      = st.select_slider("Weight decay",
                        options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2],
                        value=1e-4, format_func=lambda x: f"{x:.0e}")
        lstm_patience  = st.slider("Early stopping patience", 5, 50, 10, 5)
        lstm_sch_step  = st.slider("Scheduler step (epochs)", 10, 100, 30, 5)
        lstm_sch_gamma = st.slider("Scheduler gamma",         0.1, 0.99, 0.5, 0.05)

        st.divider()
        st.header("Modelo Híbrido")
        use_hybrid   = st.checkbox("Usar modelo híbrido", value=True)
        hybrid_trend = st.selectbox("Tipo de tendencia",
                        ["auto", "exponential", "logistic", "power", "polynomial"], index=0)

        st.divider()
        st.header("Walk-Forward Validation")
        use_wfv = st.checkbox("Activar Walk-Forward Validation", value=True)

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

        # Cargar archivos solo si cambiaron
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

        # Invalidar LSTM si cambiaron parámetros de procesamiento (sin rerun)
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

            # Outlier antes
            if out_active_sb and out_order_sb == "Antes del suavizado":
                seg_step1 = apply_outlier_filter(raw_seg, method=out_method_sb,
                    iqr_k=out_iqr_k_sb, zscore_thr=out_zscore_sb,
                    clip_min=out_cmin_sb, clip_max=out_cmax_sb, replace=out_replace_sb)
            else:
                seg_step1 = raw_seg.copy()

            # Suavizado
            seg_step2 = apply_signal_processing(seg_step1, method=signal_method,
                window=signal_window, polyorder=signal_polyord, cutoff=signal_cutoff,
                fourier_terms=signal_fourier_terms, block_size=signal_block_size)

            # Outlier después
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
                st.plotly_chart(fig_t, use_container_width=True,
                                key=f"excel_sig_proc_{idx}")

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

        st.subheader("Velocidad Inversa — todas las series (señal final)")
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

        st.divider()
        st.subheader("Predicción BiLSTM — Series Temporales Excel")

        st.markdown("##### Datos que entrarán al modelo BiLSTM")
        fig_lstm_input = go.Figure()
        COLORS_PREV = [PINK, "#c678dd", GREEN, ORANGE, CYAN, "#e5c07b"]
        for idx_p, (ts_iv_i, inv_proc_i, inv_raw_i, _, _, _, lbl_i) in enumerate(inv_seg_only_list):
            c_p = COLORS_PREV[idx_p % len(COLORS_PREV)]
            _cap = float(np.percentile(inv_proc_i[inv_proc_i > 0], 99)) if (inv_proc_i > 0).any() else 1.0
            ir_disp = np.where(inv_raw_i > _cap * 3, np.nan, inv_raw_i)
            fig_lstm_input.add_trace(go.Scatter(x=ts_iv_i, y=ir_disp, mode="lines",
                name=f"{lbl_i} raw", line=dict(color=c_p, width=1, dash="dot"), opacity=0.4))
            fig_lstm_input.add_trace(go.Scatter(x=ts_iv_i, y=inv_proc_i, mode="lines",
                name=f"{lbl_i} procesada (→ BiLSTM)",
                line=dict(color=c_p, width=2.5),
                fill="tozeroy", fillcolor="rgba(255,107,157,0.06)"))
        fig_lstm_input.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Velocidad Inversa — entrada exacta al BiLSTM",
                       font=dict(color=CYAN, size=13)),
            xaxis_title="Tiempo (s)", yaxis_title="1/|Δdesplazamiento|", height=300)
        st.plotly_chart(fig_lstm_input, use_container_width=True, key="excel_lstm_input")

        for idx_t, (ts_iv_i, inv_proc_i, inv_raw_i, disp_proc_i, disp_raw_i, ts_full_i, lbl_i) in enumerate(inv_seg_only_list):
            with st.expander(f"📋 Tabla de trazabilidad — {lbl_i}"):
                _n_t = min(len(ts_iv_i), len(inv_raw_i), len(inv_proc_i))
                _df_t = pd.DataFrame({
                    "t (s)":               np.round(ts_iv_i[:_n_t], 4),
                    "Disp. Raw (mm)":      np.round(disp_raw_i[1:_n_t+1], 6),
                    "Disp. Procesado (mm)":np.round(disp_proc_i[1:_n_t+1], 6),
                    "1/v Raw":             np.round(inv_raw_i[:_n_t], 6),
                    "1/v Procesada":       np.round(inv_proc_i[:_n_t], 6)})
                st.dataframe(_df_t, use_container_width=True, hide_index=True)

        # WFV config
        wfv_n_splits = 3; wfv_min_frac = 0.5; wfv_epochs = 40; wfv_patience = 8
        if use_wfv:
            with st.expander("⚙️ Parámetros Walk-Forward Validation", expanded=False):
                wfv_col1, wfv_col2 = st.columns(2)
                with wfv_col1:
                    wfv_n_splits = st.slider("Número de folds WFV", 2, 10, 5, 1, key="wfv_n_splits")
                    wfv_min_frac = st.slider("Fracción mínima de entrenamiento",
                                              0.3, 0.8, 0.5, 0.05, key="wfv_min_frac")
                with wfv_col2:
                    wfv_epochs   = st.slider("Epochs por fold WFV", 20, 200, 60, 10, key="wfv_epochs")
                    wfv_patience = st.slider("Patience por fold WFV", 5, 30, 10, 1, key="wfv_patience")

        lstm_excel_btn = st.button("Entrenar BiLSTM en todas las series",
                                    type="primary", key="lstm_excel_btn")

        if lstm_excel_btn:
            results = {}
            for idx, (ts_iv_i, inv_proc_i, inv_raw_i, _, _, _, lbl_raw) in enumerate(inv_seg_only_list):
                serie_lstm = inv_proc_i
                n_min = lstm_lookback + 6
                lbl = series_list[idx]["name"]
                if len(serie_lstm) < n_min:
                    st.warning(f"{lbl}: serie demasiado corta ({len(serie_lstm)} pts). Saltando.")
                    continue
                st.markdown(f"##### Entrenando: `{lbl}`")
                prog = st.progress(0, text=f"{lbl} — entrenando...")
                trend_info_saved = None
                try:
                    if use_hybrid:
                        pred_tr, fut, metr, hloss, hval, trend_info_saved = train_hybrid(
                            serie_lstm, lookback=lstm_lookback, horizon=lstm_horizon,
                            hidden_dim=lstm_hidden, n_layers=lstm_layers,
                            dropout=lstm_dropout, bidirectional=lstm_bidir,
                            lr=lstm_lr, epochs=lstm_epochs, batch_size=lstm_batch,
                            weight_decay=lstm_wd, patience=lstm_patience,
                            scheduler_step=lstm_sch_step, scheduler_gamma=lstm_sch_gamma,
                            trend_type=hybrid_trend)
                    else:
                        pred_tr, fut, metr, hloss, hval = train_bilstm(
                            serie_lstm, lookback=lstm_lookback, horizon=lstm_horizon,
                            hidden_dim=lstm_hidden, n_layers=lstm_layers,
                            dropout=lstm_dropout, bidirectional=lstm_bidir,
                            lr=lstm_lr, epochs=lstm_epochs, batch_size=lstm_batch,
                            weight_decay=lstm_wd, patience=lstm_patience,
                            scheduler_step=lstm_sch_step, scheduler_gamma=lstm_sch_gamma)
                    prog.progress(0.4, text=f"{lbl} — modelo OK, iniciando WFV...")
                except Exception as e:
                    prog.empty()
                    st.error(f"Error entrenando {lbl}: {e}")
                    continue

                wfv_res = None
                if use_wfv and len(serie_lstm) >= n_min * 2:
                    try:
                        def _wfv_cb(frac, _p=prog, _l=lbl, _n=wfv_n_splits):
                            _p.progress(0.4 + frac * 0.6,
                                        text=f"{_l} — WFV fold {int(frac*_n)}/{_n}")
                        wfv_res = walk_forward_validation(
                            serie_lstm, lookback=lstm_lookback, horizon=lstm_horizon,
                            hidden_dim=lstm_hidden, n_layers=lstm_layers,
                            dropout=lstm_dropout, bidirectional=lstm_bidir,
                            lr=lstm_lr, epochs=wfv_epochs, batch_size=lstm_batch,
                            weight_decay=lstm_wd, patience=wfv_patience,
                            scheduler_step=lstm_sch_step, scheduler_gamma=lstm_sch_gamma,
                            n_splits=wfv_n_splits, min_train_frac=wfv_min_frac,
                            progress_cb=_wfv_cb)
                    except Exception as e:
                        st.warning(f"WFV no pudo completarse para {lbl}: {e}")

                prog.progress(1.0, text=f"{lbl} — completado ✓")
                results[lbl] = (pred_tr, fut, metr, hloss, hval,
                                ts_iv_i, serie_lstm, wfv_res, trend_info_saved)

            st.session_state["excel_lstm_results"] = results if results else None

        # Mostrar resultados
        xlr = st.session_state["excel_lstm_results"]
        if xlr:
            for lbl_res_idx, (lbl, entry) in enumerate(xlr.items()):
                if len(entry) == 9:
                    pred_tr, fut, metr, hloss, hval, ts_s, iv_s, wfv_res, trend_info_s = entry
                else:
                    pred_tr, fut, metr, hloss, hval = entry[0], entry[1], entry[2], entry[3], entry[4]
                    ts_s, iv_s = entry[5], entry[6]
                    wfv_res = entry[7] if len(entry) > 7 else None
                    trend_info_s = entry[8] if len(entry) > 8 else None

                st.markdown(f"#### {lbl}")
                lf = wfv_res.get("last_forecast") if wfv_res else None
                lf_ok = lf and "error" not in lf
                fut_show     = lf["future_vals"]   if lf_ok else fut
                pred_show    = lf["pred_full"]     if lf_ok else pred_tr
                metr_show    = lf["metrics"]       if lf_ok else metr
                hloss_show   = lf["history_loss"]  if lf_ok else hloss
                hval_show    = lf.get("history_val") if lf_ok else hval

                # Key única por serie y por resultado
                _fig_key = f"excel_main_{lbl_res_idx}_{lbl[:20]}"

                if trend_info_s is not None:
                    fig_main = build_hybrid_figure(ts_s, iv_s, pred_show, fut_show,
                        trend_info_s, metr_show, hloss_show, lstm_horizon,
                        history_val=hval_show)
                    all_r2 = trend_info_s.get("all_r2", {})
                    r2_str = "  |  ".join(f"**{k}** R²={v:.3f}" for k, v in
                                          sorted(all_r2.items(), key=lambda x: -x[1]))
                    st.caption(f"Tendencias evaluadas → {r2_str}")
                else:
                    fig_main = build_lstm_figure(ts_s, iv_s, pred_show, fut_show,
                        metr_show, hloss_show, lstm_horizon, history_val=hval_show)

                st.plotly_chart(fig_main, use_container_width=True, key=_fig_key)

                if lf_ok:
                    st.info("**Pronóstico activo: WFV (modelo calibrado con toda la serie)**")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("MAE",  f"{metr_show['MAE']:.5f}")
                mc2.metric("RMSE", f"{metr_show['RMSE']:.5f}")
                mc3.metric("MAPE", f"{metr_show['MAPE']:.2f}%")
                mc4.metric("R²",   f"{metr_show['R2']:.4f}")

                if trend_info_s is not None:
                    st.caption(f"Tendencia: **{trend_info_s['trend_type']}** "
                               f"(R²={trend_info_s['r2']:.4f}) · BiLSTM sobre residuo")

                if lf_ok:
                    with st.expander("Métricas modelo inicial (referencia)"):
                        rm1, rm2, rm3, rm4 = st.columns(4)
                        rm1.metric("MAE",  f"{metr['MAE']:.5f}")
                        rm2.metric("RMSE", f"{metr['RMSE']:.5f}")
                        rm3.metric("MAPE", f"{metr['MAPE']:.2f}%")
                        rm4.metric("R²",   f"{metr['R2']:.4f}")

                if wfv_res:
                    agg = wfv_res["agg"]
                    st.markdown("**Validación Cruzada Walk-Forward**")
                    st.caption(f"{agg['n_folds']} folds completados")
                    wc1, wc2, wc3, wc4 = st.columns(4)
                    wc1.metric("MAE WFV",  f"{agg['MAE_mean']:.5f}", delta=f"±{agg['MAE_std']:.5f}")
                    wc2.metric("RMSE WFV", f"{agg['RMSE_mean']:.5f}", delta=f"±{agg['RMSE_std']:.5f}")
                    wc3.metric("MAPE WFV", f"{agg['MAPE_mean']:.2f}%", delta=f"±{agg['MAPE_std']:.2f}%")
                    wc4.metric("R² WFV",   f"{agg['R2_mean']:.4f}", delta=f"±{agg['R2_std']:.4f}")
                    fig_wfv = build_wfv_figure(wfv_res, iv_s, title=lbl)
                    st.plotly_chart(fig_wfv, use_container_width=True,
                                    key=f"excel_wfv_{lbl_res_idx}_{lbl[:20]}")
                    with st.expander(f"Tabla detallada por fold — {lbl}"):
                        fold_rows = []
                        for f in wfv_res["folds"]:
                            if "error" in f:
                                fold_rows.append({"Fold": f["fold"],
                                    "Estado": f"Error: {f['error']}",
                                    "MAE":"—","RMSE":"—","MAPE":"—","R²":"—",
                                    "Train pts":"—","Test pts":"—"})
                            else:
                                fold_rows.append({"Fold": f["fold"], "Estado": "✓",
                                    "MAE": f"{f['mae']:.5f}", "RMSE": f"{f['rmse']:.5f}",
                                    "MAPE": f"{f['mape']:.2f}%", "R²": f"{f['r2']:.4f}",
                                    "Train pts": f["train_size"], "Test pts": f["test_size"]})
                        st.dataframe(pd.DataFrame(fold_rows),
                                     use_container_width=True, hide_index=True)
                else:
                    st.caption("Serie demasiado corta para Walk-Forward Validation.")

                with st.expander(f"Pronóstico detallado — {lbl}"):
                    dt_ = float(np.mean(np.diff(ts_s))) if len(ts_s) > 1 else 1.0
                    t_fut_ = ts_s[-1] + np.arange(1, lstm_horizon + 1) * dt_
                    fut_table = fut_show if lf_ok else fut
                    st.dataframe(pd.DataFrame({
                        "Paso": list(range(1, lstm_horizon + 1)),
                        "t (s)": [f"{t:.1f}" for t in t_fut_],
                        "1/|Δdisp| pred.": [f"{v:.6f}" for v in fut_table],
                        "|Δdisp| estimado (mm)": [f"{1/(v+1e-9):.8f}" for v in fut_table]}),
                        use_container_width=True)
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
        roi_x = st.slider("Origen X (columna izquierda)",
            min_value=0, max_value=_W - 2,
            value=st.session_state[roi_key][0], step=1, key="roi_x")
        roi_y = st.slider("Origen Y (fila superior)",
            min_value=0, max_value=_H - 2,
            value=st.session_state[roi_key][1], step=1, key="roi_y")
        roi_w = st.slider("Ancho del ROI",
            min_value=2, max_value=_W - roi_x,
            value=min(st.session_state[roi_key][2], _W - roi_x), step=1, key="roi_w")
        roi_h = st.slider("Alto del ROI",
            min_value=2, max_value=_H - roi_y,
            value=min(st.session_state[roi_key][3], _H - roi_y), step=1, key="roi_h")

        # ── IMPORTANTE: resetear ROI usando session_state en lugar de st.rerun ──
        if st.button("Resetear ROI (imagen completa)"):
            st.session_state[roi_key] = (0, 0, _W, _H)
            # Actualizamos directamente los sliders via session_state (sin rerun)
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
| Área total | {_W * _H:,} px² |
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
                    BG_ = "#0d1117"; BG2_ = "#161b22"; TEXT_ = "#e6edf3"
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

    # Invalidar LSTM si cambiaron parámetros (sin rerun)
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

    # Procesamiento de señal sobre rango
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

    _a_seg   = ds_arr[fa:fb]
    _b_seg   = processed_full[fa:fb]
    _ts_seg_full = ts_arr[fa:fb]

    _ts_iv_raw,  _iv_raw  = compute_inv_vel(_a_seg, _ts_seg_full)
    _ts_iv_proc, _iv_proc = compute_inv_vel(_b_seg, _ts_seg_full)

    st.divider()
    st.subheader("Velocidad óptica y velocidad inversa")
    st.caption(f"Filtro activo: **{signal_method}**  |  Rango pares: {fa} → {fb-1}")

    fig_vel = build_velocity_figure(ts_arr, ds_arr, processed_full,
                                     (fa, fb), signal_method)
    st.plotly_chart(fig_vel, use_container_width=True, key="vid_fig_vel")

    fig_inv_init = build_inverse_velocity_figure(_ts_iv_raw, _iv_raw, _iv_proc)
    st.plotly_chart(fig_inv_init, use_container_width=True, key="vid_fig_inv_init")

    # Outliers sobre segmento
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

        diff_out_mask = np.abs(seg_for_lstm - processed_full[fa:fb]) > 1e-12
        if diff_out_mask.any():
            st.divider()
            st.subheader("Desplazamiento — outliers removidos")
            fig_out_vid = go.Figure()
            fig_out_vid.add_trace(go.Scatter(x=ts_seg, y=ds_arr[fa:fb], name="Raw",
                line=dict(color=CYAN, width=1, dash="dot"), opacity=0.4))
            fig_out_vid.add_trace(go.Scatter(x=ts_seg, y=processed_full[fa:fb],
                name="Suavizado", line=dict(color=ORANGE, width=1.5, dash="dash")))
            fig_out_vid.add_trace(go.Scatter(x=ts_seg, y=seg_for_lstm,
                name="Sin outliers (→ LSTM)", line=dict(color=GREEN, width=2.2)))
            fig_out_vid.add_trace(go.Scatter(x=ts_seg[diff_out_mask],
                y=ds_arr[fa:fb][diff_out_mask], mode="markers",
                name="Outliers detectados", marker=dict(color=PINK, size=7, symbol="x")))
            fig_out_vid.update_layout(**PLOTLY_LAYOUT,
                title=dict(text="Desplazamiento — outliers detectados y removidos",
                           font=dict(color=CYAN)),
                xaxis_title="Tiempo (s)", yaxis_title="px/frame", height=300)
            st.plotly_chart(fig_out_vid, use_container_width=True, key="vid_fig_out")

    ts_seg = ts_arr[fa:fb]
    ts_iv_lstm, inv_lstm_seg = compute_inv_vel(seg_for_lstm, ts_seg)

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
        decomp_inv_stl = stl_decompose(inv_lstm_seg, ts_iv_lstm, period=stl_period)
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

    # BiLSTM
    st.divider()
    st.subheader("Predicción BiLSTM — Velocidad Inversa")
    st.markdown("##### Datos que entrarán al modelo BiLSTM")

    _ts_iv_raw_prev, _iv_raw_prev = compute_inv_vel(ds_arr[fa:fb], ts_seg)
    fig_lstm_in = go.Figure()
    fig_lstm_in.add_trace(go.Scatter(x=_ts_iv_raw_prev, y=_iv_raw_prev, mode="lines",
        name="1/v Raw (sin procesar)", line=dict(color="#888", width=1, dash="dot"), opacity=0.5))
    fig_lstm_in.add_trace(go.Scatter(x=ts_iv_lstm, y=inv_lstm_seg, mode="lines",
        name="1/v Procesada (→ entra al BiLSTM)",
        line=dict(color=PINK, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,157,0.07)"))
    fig_lstm_in.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Velocidad Inversa — entrada exacta al BiLSTM",
                   font=dict(color=CYAN, size=13)),
        xaxis_title="Tiempo (s)", yaxis_title="1/(px/frame)", height=300)
    st.plotly_chart(fig_lstm_in, use_container_width=True, key="vid_lstm_input")

    with st.expander("📋 Tabla de trazabilidad"):
        import pandas as pd
        _n_tr = min(len(_ts_iv_raw_prev), len(_iv_raw_prev), len(inv_lstm_seg))
        _df_traz = pd.DataFrame({
            "t (s)":            np.round(_ts_iv_raw_prev[:_n_tr], 4),
            "Desp. Raw (px)":   np.round(ds_arr[fa:fb][1:_n_tr+1], 6),
            "Desp. Proc. (px)": np.round(seg_for_lstm[1:_n_tr+1], 6),
            "1/v Raw":          np.round(_iv_raw_prev[:_n_tr], 6),
            "1/v Procesada":    np.round(inv_lstm_seg[:_n_tr], 6)})
        st.dataframe(_df_traz, use_container_width=True, hide_index=True)

    inv_series_for_lstm = inv_lstm_seg

    # WFV config
    _wfv_splits = 3; _wfv_minfrac = 0.5; _wfv_ep = 40; _wfv_pat = 8
    if use_wfv:
        with st.expander("⚙️ Parámetros Walk-Forward Validation", expanded=False):
            _wfv_c1, _wfv_c2 = st.columns(2)
            with _wfv_c1:
                _wfv_splits  = st.slider("Número de folds WFV", 2, 8, 4, 1, key="vid_wfv_splits")
                _wfv_minfrac = st.slider("Fracción mín. entrenamiento",
                                          0.3, 0.8, 0.5, 0.05, key="vid_wfv_minfrac")
            with _wfv_c2:
                _wfv_ep   = st.slider("Epochs por fold WFV", 20, 150, 50, 10, key="vid_wfv_ep")
                _wfv_pat  = st.slider("Patience por fold WFV", 5, 25, 10, 1, key="vid_wfv_pat")

    st.divider()
    lstm_btn = st.button("Entrenar BiLSTM y pronosticar", type="primary")

    if lstm_btn:
        n_min = lstm_lookback + 6
        if len(inv_series_for_lstm) < n_min:
            st.error(
                f"Serie demasiado corta ({len(inv_series_for_lstm)} puntos). "
                f"Necesitas al menos {n_min} puntos.")
        else:
            prog_bar = st.progress(0, text="Entrenando...")
            wfv_res_vid = None; trend_info_vid = None
            try:
                if use_hybrid:
                    pred_tr, fut, metr, hloss, hval, trend_info_vid = train_hybrid(
                        inv_series_for_lstm, lookback=lstm_lookback, horizon=lstm_horizon,
                        hidden_dim=lstm_hidden, n_layers=lstm_layers, dropout=lstm_dropout,
                        bidirectional=lstm_bidir, lr=lstm_lr, epochs=lstm_epochs,
                        batch_size=lstm_batch, weight_decay=lstm_wd, patience=lstm_patience,
                        scheduler_step=lstm_sch_step, scheduler_gamma=lstm_sch_gamma,
                        trend_type=hybrid_trend)
                else:
                    pred_tr, fut, metr, hloss, hval = train_bilstm(
                        inv_series_for_lstm, lookback=lstm_lookback, horizon=lstm_horizon,
                        hidden_dim=lstm_hidden, n_layers=lstm_layers, dropout=lstm_dropout,
                        bidirectional=lstm_bidir, lr=lstm_lr, epochs=lstm_epochs,
                        batch_size=lstm_batch, weight_decay=lstm_wd, patience=lstm_patience,
                        scheduler_step=lstm_sch_step, scheduler_gamma=lstm_sch_gamma)
                prog_bar.progress(0.4, text="Modelo final OK — iniciando WFV...")

                if use_wfv and len(inv_series_for_lstm) >= n_min * 2:
                    try:
                        def _vid_wfv_cb(frac, _pb=prog_bar, _n=_wfv_splits):
                            _pb.progress(0.4 + frac * 0.6,
                                         text=f"Walk-Forward Validation — fold {int(frac*_n)}/{_n}")
                        wfv_res_vid = walk_forward_validation(
                            inv_series_for_lstm, lookback=lstm_lookback,
                            hidden_dim=lstm_hidden, horizon=lstm_horizon,
                            n_layers=lstm_layers, dropout=lstm_dropout,
                            bidirectional=lstm_bidir, lr=lstm_lr, epochs=_wfv_ep,
                            batch_size=lstm_batch, weight_decay=lstm_wd, patience=_wfv_pat,
                            scheduler_step=lstm_sch_step, scheduler_gamma=lstm_sch_gamma,
                            n_splits=_wfv_splits, min_train_frac=_wfv_minfrac,
                            progress_cb=_vid_wfv_cb)
                    except Exception as e:
                        st.warning(f"WFV no pudo completarse: {e}")

                prog_bar.progress(1.0, text="¡Entrenamiento completado!")
                st.session_state["lstm_result"] = (
                    pred_tr, fut, metr, hloss, hval,
                    ts_iv_lstm, inv_lstm_seg, lstm_horizon,
                    wfv_res_vid, trend_info_vid)
            except Exception as e:
                prog_bar.empty()
                st.error(f"Error en BiLSTM: {e}")
                st.session_state["lstm_result"] = None

    lr_res = st.session_state["lstm_result"]
    if lr_res:
        if len(lr_res) == 10:
            pred_tr, fut, metr, hloss, hval, ts_lstm, iv_lstm, _hor, _wfv_vid, _trend_vid = lr_res
        else:
            pred_tr, fut, metr, hloss = lr_res[0], lr_res[1], lr_res[2], lr_res[3]
            hval = lr_res[4] if len(lr_res) > 4 else None
            ts_lstm = lr_res[5] if len(lr_res) > 5 else ts_iv_lstm
            iv_lstm = lr_res[6] if len(lr_res) > 6 else inv_lstm_seg
            _hor = lr_res[7] if len(lr_res) > 7 else lstm_horizon
            _wfv_vid = lr_res[8] if len(lr_res) > 8 else None
            _trend_vid = lr_res[9] if len(lr_res) > 9 else None

        _lf = _wfv_vid.get("last_forecast") if _wfv_vid else None
        _lf_ok = _lf and "error" not in _lf
        _fut_show   = _lf["future_vals"]   if _lf_ok else fut
        _pred_show  = _lf["pred_full"]     if _lf_ok else pred_tr
        _metr_show  = _lf["metrics"]       if _lf_ok else metr
        _hloss_show = _lf["history_loss"]  if _lf_ok else hloss
        _hval_show  = _lf.get("history_val") if _lf_ok else hval

        if _trend_vid is not None:
            fig_main = build_hybrid_figure(ts_lstm, iv_lstm, _pred_show, _fut_show,
                _trend_vid, _metr_show, _hloss_show, _hor, history_val=_hval_show)
            all_r2 = _trend_vid.get("all_r2", {})
            r2_str = "  |  ".join(f"**{k}** R²={v:.3f}" for k, v in
                                   sorted(all_r2.items(), key=lambda x: -x[1]))
            st.caption(f"Tendencias evaluadas → {r2_str}")
        else:
            fig_main = build_lstm_figure(ts_lstm, iv_lstm, _pred_show, _fut_show,
                _metr_show, _hloss_show, _hor, history_val=_hval_show)

        st.plotly_chart(fig_main, use_container_width=True, key="vid_lstm_main")

        if _lf_ok:
            st.info("**Pronóstico activo: WFV calibrado (toda la serie)**")

        st.markdown("**Métricas modelo activo**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE",  f"{_metr_show['MAE']:.5f}")
        mc2.metric("RMSE", f"{_metr_show['RMSE']:.5f}")
        mc3.metric("MAPE", f"{_metr_show['MAPE']:.2f}%")
        mc4.metric("R²",   f"{_metr_show['R2']:.4f}")

        if _trend_vid is not None:
            st.caption(f"Tendencia: **{_trend_vid['trend_type']}** "
                       f"(R²={_trend_vid['r2']:.4f}) · BiLSTM sobre residuo")

        if _wfv_vid:
            _agg = _wfv_vid["agg"]
            st.markdown("**Validación Cruzada Walk-Forward**")
            st.caption(f"{_agg['n_folds']} folds completados")
            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("MAE WFV",  f"{_agg['MAE_mean']:.5f}", delta=f"±{_agg['MAE_std']:.5f}")
            wc2.metric("RMSE WFV", f"{_agg['RMSE_mean']:.5f}", delta=f"±{_agg['RMSE_std']:.5f}")
            wc3.metric("MAPE WFV", f"{_agg['MAPE_mean']:.2f}%", delta=f"±{_agg['MAPE_std']:.2f}%")
            wc4.metric("R² WFV",   f"{_agg['R2_mean']:.4f}", delta=f"±{_agg['R2_std']:.4f}")
            fig_wfv_v = build_wfv_figure(_wfv_vid, iv_lstm, title="Video/Imágenes")
            st.plotly_chart(fig_wfv_v, use_container_width=True, key="vid_wfv_fig")
            with st.expander("Tabla detallada por fold"):
                import pandas as _pd_v
                _rows = []
                for _f in _wfv_vid["folds"]:
                    if "error" in _f:
                        _rows.append({"Fold": _f["fold"],
                            "Estado": f"Error: {_f['error']}",
                            "MAE":"—","RMSE":"—","MAPE":"—","R²":"—",
                            "Train pts":"—","Test pts":"—"})
                    else:
                        _rows.append({"Fold": _f["fold"], "Estado": "✓",
                            "MAE": f"{_f['mae']:.5f}", "RMSE": f"{_f['rmse']:.5f}",
                            "MAPE": f"{_f['mape']:.2f}%", "R²": f"{_f['r2']:.4f}",
                            "Train pts": _f["train_size"], "Test pts": _f["test_size"]})
                st.dataframe(_pd_v.DataFrame(_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Serie demasiado corta para Walk-Forward Validation.")

        with st.expander("Detalles del modelo y pronóstico"):
            import pandas as pd
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown(f"""
| Parámetro | Valor |
|-----------|-------|
| Lookback  | {metr['lookback']} |
| Horizon   | {metr['horizon']} |
| Hidden dim | {metr['hidden_dim']} |
| Capas LSTM | {metr['n_layers']} |
| Bidireccional | {metr['bidirectional']} |
| Dropout | {metr['dropout']} |
| LR | {metr['lr']:.0e} |
| Parámetros totales | {metr['n_params']:,} |
| Epochs ejecutados | {metr['epochs_run']} |
| Mejor val loss | {metr['best_val_loss']:.6f} |
""")
            with c_b:
                dt = float(np.mean(np.diff(ts_lstm))) if len(ts_lstm) > 1 else 1.0
                t_fut = ts_lstm[-1] + np.arange(1, _hor + 1) * dt
                _fut_table = _lf["future_vals"] if _lf_ok else fut
                st.caption("Pronóstico WFV" if _lf_ok else "Pronóstico modelo final")
                df_fut = pd.DataFrame({
                    "Paso": list(range(1, _hor + 1)),
                    "t (s)": [f"{t:.2f}" for t in t_fut],
                    "1/v predicho": [f"{v:.5f}" for v in _fut_table],
                    "v estimada px/frame": [f"{1/(v+1e-9):.3f}" for v in _fut_table]})
                st.dataframe(df_fut, use_container_width=True)

    # Comparación personalizada de frames
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
