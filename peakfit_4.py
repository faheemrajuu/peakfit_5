# peakfit_4.py (You can not only input peak height and FWHM in this code but also your denominator area is not the area of the fitted curve it's coming from the total area of the raw curve.) 

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List


# -----------------------------
# PLOT STYLE (GLOBAL)
# -----------------------------
AXIS_TITLE_FONT = dict(size=34, family="Arial Black", color="black")
AXIS_TICK_FONT  = dict(size=28, family="Arial Black", color="black")

# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class FitConfig:
    file: str
    save_output: bool = True

    wavelength: float = 0.7863
    qmin: float = 0.08
    qmax: float = 0.27
    peak_positions: Tuple[float, ...] = (0.136, 0.165)
    # Optional: lock peak heights (A) per peak
    lock_A: Optional[Dict[float, float]] = None  # {q0_key: A_fixed}
    # Optional: lock / constrain FWHM (w) per peak
    lock_w: Optional[Dict[float, float]] = None   # {q0_key: w_fixed}

    # How tight the "almost lock" is (curve_fit needs lb < ub)
    lock_A_rel_tol: float = 5e-4   # 0.05%
    lock_w_rel_tol: float = 5e-3   # 0.5% (w is small, keep slightly looser)
    
    def __post_init__(self):
        # Remove duplicates (and keep sorted for consistent ordering)
        self.peak_positions = tuple(sorted(set(self.peak_positions)))
        
    # BG window selection
    use_auto_bg: bool = True
    peak_fraction: float = 0.05  # only used if use_auto_bg=True
    bg_windows: Optional[Dict[float, Tuple[float, float]]] = None  # used if use_auto_bg=False

    # Plot styling
    peak_colors: Optional[Dict[float, str]] = None

    # Peak init / bounds
    q0_half_window: float = 0.00005
    w_init: float = 0.012
    eta_init: float = 0.4
    w_min: float = 0.003
    w_max: float = 0.1


# -----------------------------
# IO helpers
# -----------------------------
def make_output_dirs(file: str) -> Tuple[Path, Path, Path]:
    data_path = Path(file).resolve()
    tables_folder = data_path.parent / "tables"
    plots_folder = data_path.parent / "plots"
    tables_folder.mkdir(exist_ok=True)
    plots_folder.mkdir(exist_ok=True)
    return data_path, tables_folder, plots_folder


# -----------------------------
# Model
# -----------------------------
def pseudo_voigt(q: np.ndarray, A: float, q0: float, w: float, eta: float) -> np.ndarray:
    sigma = w / (2 * np.sqrt(2 * np.log(2)))
    gamma = w / 2
    G = np.exp(-(q - q0) ** 2 / (2 * sigma ** 2))
    L = 1 / (1 + ((q - q0) / gamma) ** 2)
    return A * (eta * L + (1 - eta) * G)


def get_peak_params(p: np.ndarray, i: int) -> Tuple[float, float, float, float]:
    return p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]


def model(q: np.ndarray, peak_positions: Tuple[float, ...], *p: float) -> np.ndarray:
    I = np.zeros_like(q)
    for i in range(len(peak_positions)):
        A, q0, w, eta = get_peak_params(np.asarray(p), i)
        I += pseudo_voigt(q, A, q0, w, eta)
    return I


# -----------------------------
# Data
# -----------------------------
def load_and_preprocess(cfg: FitConfig):
    df = pd.read_csv(cfg.file, sep=r"\s+")
    theta = np.deg2rad(df["2theta"].values / 2)
    q = (4 * np.pi / cfg.wavelength) * np.sin(theta)

    I_raw = df["I_raw"].values
    I_bg = df["I_AmpBg"].values
    I_sub = I_raw - I_bg

    mask = (q >= cfg.qmin) & (q <= cfg.qmax)
    q_fit, I_raw_fit, I_bg_fit, I_sub_fit = q[mask], I_raw[mask], I_bg[mask], I_sub[mask]

    idx = np.argsort(q_fit)
    q_fit, I_raw_fit, I_bg_fit, I_sub_fit = q_fit[idx], I_raw_fit[idx], I_bg_fit[idx], I_sub_fit[idx]

    sigma = df["y_er"].values[mask][idx] if "y_er" in df.columns else None
# ---- sanitize error bars ----
    if sigma is not None:
        pos = sigma > 0
        if np.any(pos):
            sigma_min = np.percentile(sigma[pos], 5)
            sigma = np.clip(sigma, sigma_min, None)
        else:
            sigma = None  # fallback: all invalid
    
    return df, q_fit, I_raw_fit, I_bg_fit, I_sub_fit, sigma

# -----------------------------
# Initial Guess
# -----------------------------

def build_initial_guess_and_bounds(q_fit: np.ndarray, I_sub_fit: np.ndarray, cfg: FitConfig):
    p0, lb, ub = [], [], []

    lock_A = cfg.lock_A or {}
    lock_w = cfg.lock_w or {}

    for q0_key in cfg.peak_positions:
        # ----- initial guesses -----
        A0 = float(np.interp(q0_key, q_fit, I_sub_fit))
        q0_init = q0_key
        w0 = cfg.w_init
        eta0 = cfg.eta_init

        # ----- default bounds -----
        A_lb, A_ub = 0.0, np.inf
        q_lb, q_ub = q0_key - cfg.q0_half_window, q0_key + cfg.q0_half_window
        w_lb, w_ub = cfg.w_min, cfg.w_max
        e_lb, e_ub = 0.0, 1.0

        # ----- lock A if requested -----
        if q0_key in lock_A:
            A_fixed = float(lock_A[q0_key])
            A0 = A_fixed

            rel = float(cfg.lock_A_rel_tol)
            abs_floor = 1e-6
            A_eps = max(abs_floor, rel * abs(A_fixed))

            A_lb = A_fixed - A_eps
            A_ub = A_fixed + A_eps

            if not np.isfinite(A_lb) or not np.isfinite(A_ub) or not (A_lb < A_ub):
                raise ValueError(f"Bad lock_A for q0={q0_key}: A_fixed={A_fixed}, lb={A_lb}, ub={A_ub}")

        # ----- lock w (FWHM) if requested -----
        if q0_key in lock_w:
            w_fixed = float(lock_w[q0_key])
            w0 = w_fixed

            rel = float(cfg.lock_w_rel_tol)
            abs_floor = 1e-6  # w is small, keep absolute floor small too
            w_eps = max(abs_floor, rel * abs(w_fixed))

            w_lb = w_fixed - w_eps
            w_ub = w_fixed + w_eps

            # optional: ensure within global limits
            w_lb = max(w_lb, cfg.w_min)
            w_ub = min(w_ub, cfg.w_max)

            # If clipping collapses the interval, widen slightly
            if not (w_lb < w_ub):
                w_lb = max(cfg.w_min, w_fixed - 2 * abs_floor)
                w_ub = min(cfg.w_max, w_fixed + 2 * abs_floor)

            if not np.isfinite(w_lb) or not np.isfinite(w_ub) or not (w_lb < w_ub):
                raise ValueError(f"Bad lock_w for q0={q0_key}: w_fixed={w_fixed}, lb={w_lb}, ub={w_ub}")

        p0 += [A0, q0_init, w0, eta0]
        lb += [A_lb, q_lb, w_lb, e_lb]
        ub += [A_ub, q_ub, w_ub, e_ub]

    return p0, lb, ub


# -----------------------------
# Fit
# -----------------------------

def fit_peaks(q_fit: np.ndarray, I_sub_fit: np.ndarray, sigma: Optional[np.ndarray], cfg: FitConfig):
    p0, lb, ub = build_initial_guess_and_bounds(q_fit, I_sub_fit, cfg)

    popt, pcov = curve_fit(
        lambda q, *p: model(q, cfg.peak_positions, *p),
        q_fit, I_sub_fit,
        p0=p0, bounds=(lb, ub),
        sigma=sigma,
        absolute_sigma=(sigma is not None),
        maxfev=50000
    )
    return popt, pcov


# -----------------------------
# Curves + BG
# -----------------------------

def evaluate_curves(q_fit: np.ndarray,
                    I_raw_fit: np.ndarray,
                    I_bg_fit: np.ndarray,
                    popt: np.ndarray,
                    cfg: FitConfig,
                    n_dense: int = 2000):

    q_dense = np.linspace(cfg.qmin, cfg.qmax, n_dense)

    pv_curves = []
    for i in range(len(cfg.peak_positions)):
        A, q0, w, eta = get_peak_params(popt, i)
        pv_curves.append(pseudo_voigt(q_dense, A, q0, w, eta))

    I_fit_total = np.sum(pv_curves, axis=0)

    bg_interp  = interp1d(q_fit, I_bg_fit,  bounds_error=False, fill_value="extrapolate")
    raw_interp = interp1d(q_fit, I_raw_fit, bounds_error=False, fill_value="extrapolate")

    I_bg_dense  = bg_interp(q_dense)
    I_raw_dense = raw_interp(q_dense)

    return q_dense, pv_curves, I_fit_total, bg_interp, I_bg_dense, I_raw_dense


# -----------------------------
# Windows
# -----------------------------
def build_windows(q_dense: np.ndarray, pv_curves: List[np.ndarray], cfg: FitConfig) -> Dict[float, Tuple[float, float]]:
    if cfg.use_auto_bg:
        windows = {}
        for i, q0_key in enumerate(cfg.peak_positions):
            curve = pv_curves[i]
            thresh = cfg.peak_fraction * float(np.max(curve))
            idxs = np.where(curve >= thresh)[0]
            if idxs.size < 2:
                # fallback: small window around nominal q0
                q0 = q0_key
                windows[q0_key] = (max(cfg.qmin, q0 - cfg.q0_half_window),
                                  min(cfg.qmax, q0 + cfg.q0_half_window))
            else:
                windows[q0_key] = (float(q_dense[idxs[0]]), float(q_dense[idxs[-1]]))
        return windows

    # manual mode
    if not cfg.bg_windows:
        raise ValueError("use_auto_bg=False but cfg.bg_windows is None/empty.")
    return dict(cfg.bg_windows)

# -----------------------------
# Metrics
# -----------------------------
def compute_global_totals(
    q_dense: np.ndarray,
    pv_curves: List[np.ndarray],
    I_bg_dense: np.ndarray,
    I_raw_dense: np.ndarray,
    windows: Dict[float, Tuple[float, float]],
    denom_mode: str = "full",   # "full" or "union"
):
    if denom_mode not in {"full", "union"}:
        raise ValueError("denom_mode must be 'full' or 'union'")

    if denom_mode == "union":
        win = np.zeros_like(q_dense, dtype=bool)
        for (q_lo, q_hi) in windows.values():
            win |= (q_dense >= q_lo) & (q_dense <= q_hi)
    else:
        win = np.ones_like(q_dense, dtype=bool)

    q_u = q_dense[win]

    I_cry_u = np.sum([c[win] for c in pv_curves], axis=0)
    I_bg_u  = I_bg_dense[win]
    I_raw_u = I_raw_dense[win]

    Q_cry_total = np.trapz(I_cry_u * q_u**2, q_u)
    Q_bg_total  = np.trapz(I_bg_u  * q_u**2, q_u)
    Q_raw_total = np.trapz(I_raw_u * q_u**2, q_u)

    return win, Q_cry_total, Q_bg_total, Q_raw_total

# -----------------------------
# Metrics
# -----------------------------
def q2_weighted_integrals_for_peak(
    i: int,
    q0_key: float,
    q_dense: np.ndarray,
    pv_curves: List[np.ndarray],
    I_bg_dense: np.ndarray,
    windows: Dict[float, Tuple[float, float]],
):
    q_lo, q_hi = windows[q0_key]
    win = (q_dense >= q_lo) & (q_dense <= q_hi)
    q_win = q_dense[win]

    I_cry = pv_curves[i][win]
    I_bg = I_bg_dense[win]

    Q_cry = np.trapz(I_cry * q_win ** 2, q_win)
    Q_bg = np.trapz(I_bg * q_win ** 2, q_win)
    Q_tot = Q_cry + Q_bg
    cryst_pct = 100 * Q_cry / Q_tot if Q_tot > 0 else np.nan

    return q_lo, q_hi, Q_cry, Q_bg, Q_tot, cryst_pct

# -----------------------------
# Table Building
# -----------------------------

def build_peak_table(
    popt: np.ndarray,
    q_dense: np.ndarray,
    pv_curves: List[np.ndarray],
    I_bg_dense: np.ndarray,
    I_raw_dense: np.ndarray,
    windows: Dict[float, Tuple[float, float]],
    cfg: FitConfig,
) -> pd.DataFrame:

    K = 0.94
    rows = []

    _, Q_cry_total, Q_bg_total, Q_raw_total = compute_global_totals(
    q_dense, pv_curves, I_bg_dense, I_raw_dense, windows, denom_mode="full")
    
    for i, q0_key in enumerate(cfg.peak_positions):
        A, q0_fit, w, eta = get_peak_params(popt, i)

        theta_rad = np.arcsin(q0_fit * cfg.wavelength / (4 * np.pi))
        two_theta_deg = 2 * np.degrees(theta_rad)
        two_theta_rad = 2 * theta_rad
        d_spacing = 2 * np.pi / q0_fit

        dtheta_dq = cfg.wavelength / (4 * np.pi * np.cos(theta_rad))
        fwhm_2theta_deg = w * np.degrees(2 * dtheta_dq)
        fwhm_2theta_rad = np.radians(fwhm_2theta_deg)

        beta = fwhm_2theta_rad
        L = K * cfg.wavelength / (beta * np.cos(theta_rad)) if beta > 0 else np.nan

        q_lo, q_hi, Q_cry, Q_bg_local, Q_tot_local, cryst_pct_local = q2_weighted_integrals_for_peak(
            i, q0_key, q_dense, pv_curves, I_bg_dense, windows
        )

        # ✅ denominator is RAW union integral
        pct_peak_over_raw = 100 * Q_cry / Q_raw_total if Q_raw_total > 0 else np.nan

        rows.append({
            "q0_key (Å⁻¹)": q0_key,
            "q0_fit (Å⁻¹)": q0_fit,
            "2θ (deg)": two_theta_deg,
            "2θ (rad)": two_theta_rad,
            "d-spacing (Å)": d_spacing,
            "FWHM Γ (deg)": fwhm_2theta_deg,
            "FWHM Γ (rad)": fwhm_2theta_rad,
            "Crystal Size L (Å)": L,

            "Q_crystal_peak (a.u.)": Q_cry,
            "% Peak / RAW (global q²)": pct_peak_over_raw,

            "Q_raw_all (a.u.)": Q_raw_total,
            "Q_bg_all (a.u.)": Q_bg_total,
            "Q_crystal_all (a.u.)": Q_cry_total,

            "BG q_lo (Å⁻¹)": q_lo,
            "BG q_hi (Å⁻¹)": q_hi,
            "Q_bg_peakwindow (a.u.)": Q_bg_local,
            "% Crystal (local q²)": cryst_pct_local,
        })

    df = pd.DataFrame(rows)

    ROUND_MAP = {
        "q0_key (Å⁻¹)": 4,
        "q0_fit (Å⁻¹)": 4,
        "2θ (deg)": 4,
        "2θ (rad)": 4,
        "d-spacing (Å)": 4,
        "FWHM Γ (deg)": 4,
        "FWHM Γ (rad)": 4,
        "Crystal Size L (Å)": 1,
        "Q_crystal_peak (a.u.)": 4,
        "% Peak / RAW (global q²)": 2,
        "Q_raw_all (a.u.)": 4,
        "Q_bg_all (a.u.)": 4,
        "Q_crystal_all (a.u.)": 4,
        "BG q_lo (Å⁻¹)": 4,
        "BG q_hi (Å⁻¹)": 4,
        "Q_bg_peakwindow (a.u.)": 4,
        "% Crystal (local q²)": 2,
    }

    for col, ndigits in ROUND_MAP.items():
        if col in df.columns:
            df[col] = df[col].round(ndigits)

    return df

# -----------------------------
# Plot
# -----------------------------
def make_figure(res: Dict[str, Any], cfg: FitConfig) -> go.Figure:
    q_fit = res["q_fit"]
    I_raw_fit = res["I_raw_fit"]
    I_bg_fit = res["I_bg_fit"]
    I_sub_fit = res["I_sub_fit"]

    q_dense = res["q_dense"]
    pv_curves = res["pv_curves"]
    I_fit_total = res["I_fit_total"]
    windows = res["windows"]

    y_max = float(max(I_raw_fit.max(), I_bg_fit.max()))
    bg_interp = interp1d(q_fit, I_bg_fit, bounds_error=False, fill_value="extrapolate")

    fig = go.Figure()
    sample_name = Path(cfg.file).stem.replace("_AmpBGkrtd", "")

    fig.add_trace(go.Scatter(x=q_fit, y=I_raw_fit, name=sample_name, line=dict(color="black", width=4)))
    fig.add_trace(go.Scatter(x=q_fit, y=I_bg_fit, name="I_bg", line=dict(color="gray", width=4, dash="dash")))
    fig.add_trace(go.Scatter(x=q_fit, y=I_sub_fit, name="I_sub", line=dict(color="red", width=4)))

    fig.add_trace(go.Scatter(
        x=np.concatenate([q_fit, q_fit[::-1]]),
        y=np.concatenate([I_bg_fit, np.zeros_like(I_bg_fit)]),
        fill="toself",
        fillcolor="rgba(255, 230, 230, 0.8)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False
    ))

    peak_colors = cfg.peak_colors or {}
    for curve, q0_key in zip(pv_curves, cfg.peak_positions):
        fill = peak_colors.get(q0_key, "rgba(180,180,180,0.35)")
        line_col = fill.replace("0.35", "1.0").replace("0.45", "1.0")
        fig.add_trace(go.Scatter(x=q_dense, y=curve, mode="lines",
                                 line=dict(width=3, color=line_col), showlegend=False))
        fig.add_trace(go.Scatter(
            x=np.concatenate([q_dense, q_dense[::-1]]),
            y=np.concatenate([curve, np.zeros_like(curve)]),
            fill="toself",
            fillcolor=fill,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(x=q_dense, y=I_fit_total, name="Total Fit",
                             line=dict(color="royalblue", width=4, dash="dot")))

    for q0_key, (q_lo, q_hi) in windows.items():
        col = peak_colors.get(q0_key, "rgba(180,180,180,0.8)").replace("0.35", "1.0").replace("0.45", "1.0")
        y_lo = max(0.0, float(bg_interp(q_lo)))
        y_hi = max(0.0, float(bg_interp(q_hi)))

    fig.update_layout(
        width=900, height=600,
        margin=dict(l=80, r=40, t=20, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.75, y=0.97, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0),
        legend_font=dict(size=28, family="Arial Black", color="black"))
    
    fig.update_xaxes(title=dict(text = "q (Å⁻¹)",font=AXIS_TITLE_FONT), tickfont=AXIS_TICK_FONT, showline=True, mirror=True, 
                                   linecolor="black", linewidth=3, dtick=0.03)
    fig.update_yaxes(title=dict(text="Intensity (a.u.)", font=AXIS_TITLE_FONT), tickfont=AXIS_TICK_FONT, showline=True,  
                     mirror=True, linecolor="black", linewidth=3, range=[0, y_max])

    return fig
# -----------------------------
# Main runner
# -----------------------------
def run(cfg: FitConfig) -> Dict[str, Any]:
    data_path, tables_folder, plots_folder = make_output_dirs(cfg.file)

    df, q_fit, I_raw_fit, I_bg_fit, I_sub_fit, sigma = load_and_preprocess(cfg)
    popt, pcov = fit_peaks(q_fit, I_sub_fit, sigma, cfg)
    q_dense, pv_curves, I_fit_total, bg_interp, I_bg_dense, I_raw_dense = evaluate_curves(q_fit, I_raw_fit,
                                                                                          I_bg_fit, popt, cfg)
    windows = build_windows(q_dense, pv_curves, cfg)
    table = build_peak_table(popt, q_dense, pv_curves, I_bg_dense, I_raw_dense, windows, cfg)
    if cfg.save_output:
        out_path = tables_folder / data_path.with_suffix(".csv").name
        table.to_csv(out_path, index=False)
        #print(f"\n✅ Table saved to:\n{out_path}")

    return {
        "df": df,
        "q_fit": q_fit, "I_raw_fit": I_raw_fit, "I_bg_fit": I_bg_fit, "I_sub_fit": I_sub_fit,
        "sigma": sigma,
        "popt": popt, "pcov": pcov,
        "q_dense": q_dense, "pv_curves": pv_curves, "I_fit_total": I_fit_total,
        "I_bg_dense": I_bg_dense,
        "windows": windows,
        "table": table,
        "data_path": data_path,
        "I_raw_dense": I_raw_dense,
    }

