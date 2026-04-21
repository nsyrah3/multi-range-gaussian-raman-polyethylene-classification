#!/usr/bin/env python3
"""
Program 3: Gaussian fitting on Program 2 windowed spectra.

Input (default):
- outputs/program_2_cut_smooth_windows/windows_smooth/<class>/<stem>__wX.csv
- outputs/program_2_cut_smooth_windows/windows_raw/<class>/<stem>__wX.csv (optional, for plot overlay)

Output (default root: outputs/program_3_gaussian_fit):
- gaussian_fit_summary.csv
- fit_curves/<class>/<stem>__wX_fit.csv
- plots/<class>/<stem>__program3_gaussian.png
  (simple view: smoothed + fit total)

Model per window:
- y = local baseline (poly order 1/2) + sum of 2 Gaussians
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks


CLASSES = ("HDPE", "LDPE")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs" / "program_2_cut_smooth_windows"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "program_3_gaussian_fit"
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
MIN_POINTS = 8
W3_RESCUE_ROUGH_THRESH = 0.08
W3_RESCUE_SG_WINDOW = 11
W3_RESCUE_SG_POLY = 3

# Optional adaptive forcing for w3 so two humps appear more clearly.
W3_FORCE_DOUBLE = True
W3_FORCE_R2_DROP_MAX = 0.03
W3_FORCE_CFG = {
    # Tighter fallback to split the 2 overlapped CH-stretch humps.
    "peak_bounds": [(2848.0, 2868.0), (2882.0, 2900.0)],
    "sigma_bounds_by_peak": [(8.0, 16.0), (8.0, 22.0)],
    "baseline_order": 3,
}

WINDOW_CFG = {
    "w1": {
        "range": (995.0, 1215.0),
        "peak_bounds": [(1047.0, 1082.0), (1117.0, 1146.0)],
        "sigma_bounds": (4.0, 40.0),
        # w1 often keeps broad curvature from smoothing; cubic baseline stabilizes fit.
        "baseline_order": 3,
    },
    "w2": {
        "range": (1215.0, 1596.0),
        # Tighten w2 decomposition so 2 peaks stay separated (no broad single-peak collapse).
        "peak_bounds": [(1288.0, 1312.0), (1434.0, 1462.0)],
        "sigma_bounds_by_peak": [(5.0, 22.0), (8.0, 40.0)],
        # w2 often has mild curvature; quadratic baseline gives much stabler 2-peak fit.
        "baseline_order": 2,
        "amp_min_frac": 0.04,
    },
    "w3": {
        "range": (2624.0, 3125.0),
        # Use tighter centers and bounded widths so the 2-hump profile does not collapse.
        "peak_bounds": [(2846.0, 2868.0), (2880.0, 2902.0)],
        "sigma_bounds_by_peak": [(8.0, 18.0), (8.0, 28.0)],
        "baseline_order": 3,
    },
}


def _to_float_tuple_pair(value) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected [lo, hi] style value, got: {value!r}")
    return float(value[0]), float(value[1])


def _normalize_window_cfg_entry(entry: dict) -> dict:
    out = dict(entry)
    if "range" in out:
        out["range"] = _to_float_tuple_pair(out["range"])
    if "peak_bounds" in out:
        out["peak_bounds"] = [_to_float_tuple_pair(v) for v in out["peak_bounds"]]
    if "sigma_bounds" in out:
        out["sigma_bounds"] = _to_float_tuple_pair(out["sigma_bounds"])
    if "sigma_bounds_by_peak" in out:
        out["sigma_bounds_by_peak"] = [_to_float_tuple_pair(v) for v in out["sigma_bounds_by_peak"]]
    if "baseline_order" in out:
        out["baseline_order"] = int(out["baseline_order"])
    if "amp_min_frac" in out:
        out["amp_min_frac"] = float(out["amp_min_frac"])
    return out


def apply_fit_config(config_path: Path | None) -> None:
    global WINDOW_CFG, W3_FORCE_DOUBLE, W3_FORCE_CFG
    if config_path is None:
        return
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    window_cfg = cfg.get("window_cfg")
    if window_cfg:
        updated = {k: dict(v) for k, v in WINDOW_CFG.items()}
        for key, value in window_cfg.items():
            if key not in updated:
                raise ValueError(f"Unknown window key in config: {key}")
            updated[key].update(_normalize_window_cfg_entry(value))
        WINDOW_CFG = updated

    if "w3_force_double" in cfg:
        W3_FORCE_DOUBLE = bool(cfg["w3_force_double"])
    if "w3_force_cfg" in cfg:
        W3_FORCE_CFG = _normalize_window_cfg_entry(cfg["w3_force_cfg"])


def read_window_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["x", "y"])
    df = pd.read_csv(path, header=None, names=["x", "y"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna().sort_values("x").reset_index(drop=True)


def gauss(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def safe_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = len(y)
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n if n % 2 == 1 else max(1, n - 1)
    if win < poly + 2 or win < 3 or n < 3:
        return y
    return savgol_filter(y, window_length=win, polyorder=poly)


def rough_ratio(y: np.ndarray) -> float:
    if len(y) < 4:
        return 0.0
    return float(np.std(np.diff(y)) / (np.ptp(y) + 1e-12))


def count_prominent_peaks(y: np.ndarray) -> int:
    if len(y) < 5:
        return 0
    prom = max(1e-6, 0.02 * float(np.max(y) - np.min(y)))
    idx, _ = find_peaks(y, prominence=prom, distance=max(3, len(y) // 14))
    return int(len(idx))


def fit_weights(x: np.ndarray, win_key: str) -> np.ndarray:
    """
    Residual weights for robust, region-prioritized fitting.
    For w3, emphasize CH-stretch hump regions and de-emphasize far tails.
    """
    w = np.ones_like(x, dtype=float)
    if win_key != "w3":
        return w

    # Tails carry less structural info for the 2-hump decomposition.
    tail = (x < 2805.0) | (x > 2970.0)
    w[tail] *= 0.55
    # Main peak envelope
    w[(x >= 2838.0) & (x <= 2922.0)] *= 1.30
    # Second hump focus
    w[(x >= 2882.0) & (x <= 2922.0)] *= 1.20
    return w


def local_rmse_norm(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray, lo: float, hi: float) -> float:
    m = (x >= float(lo)) & (x <= float(hi))
    if int(np.sum(m)) < 3:
        return np.inf
    yy = y_true[m]
    err = float(np.sqrt(np.mean((yy - y_pred[m]) ** 2)))
    return err / (float(np.ptp(yy)) + 1e-12)


def w3_shape_score(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Lower is better.
    Prioritize the two hump regions (especially the second hump) over far tails.
    """
    e_main = local_rmse_norm(y_true, y_pred, x, 2838.0, 2922.0)
    e_h1 = local_rmse_norm(y_true, y_pred, x, 2844.0, 2872.0)
    e_h2 = local_rmse_norm(y_true, y_pred, x, 2882.0, 2914.0)
    e_valley = local_rmse_norm(y_true, y_pred, x, 2870.0, 2884.0)
    e_glob = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / (float(np.ptp(y_true)) + 1e-12)
    return 0.60 * e_main + 1.15 * e_h1 + 1.55 * e_h2 + 0.80 * e_valley + 0.30 * e_glob


def baseline_eval(x: np.ndarray, coeffs: np.ndarray, x0: float) -> np.ndarray:
    dx = x - x0
    yb = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        yb += c * (dx ** i)
    return yb


def model_eval(x: np.ndarray, x0: float, base_coeffs: np.ndarray, g_params: np.ndarray) -> np.ndarray:
    y = baseline_eval(x, base_coeffs, x0)
    n = len(g_params) // 3
    for i in range(n):
        amp = float(g_params[3 * i])
        mu = float(g_params[3 * i + 1])
        sig = float(g_params[3 * i + 2])
        y += gauss(x, amp, mu, sig)
    return y


def baseline_seed(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    q20 = np.percentile(y, 20.0)
    mask = y <= q20
    if int(np.sum(mask)) < 5:
        idx = np.r_[np.arange(min(5, len(y))), np.arange(max(0, len(y) - 5), len(y))]
        mask = np.zeros(len(y), dtype=bool)
        mask[idx] = True
    A = np.vstack([x[mask], np.ones_like(x[mask])]).T
    m, b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
    return float(b), float(m)


def build_init_and_bounds(x: np.ndarray, y: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    x0 = float(np.mean(x))
    order = max(1, int(cfg["baseline_order"]))
    n_base = order + 1
    n_peaks = len(cfg["peak_bounds"])
    sigma_by_peak = cfg.get("sigma_bounds_by_peak")
    if sigma_by_peak is None:
        sig_lo, sig_hi = cfg["sigma_bounds"]
        sigma_by_peak = [(sig_lo, sig_hi)] * n_peaks

    b0, b1 = baseline_seed(x, y)
    base_guess = [b0, b1]
    if order >= 2:
        base_guess += [0.0] * (order - 1)

    if len(x) > 2:
        grad = np.diff(y) / np.diff(x)
        b_abs = 5.0 * float(np.median(np.abs(grad)))
        if not np.isfinite(b_abs) or b_abs < 1e-6:
            b_abs = 1.0
    else:
        b_abs = 1.0
    span = float(max(x[-1] - x[0], 1.0))
    c_abs = 6.0 * float(np.ptp(y)) / (span * span)
    if not np.isfinite(c_abs) or c_abs < 1e-8:
        c_abs = 1e-8

    lb_base = [-np.inf, -b_abs]
    ub_base = [np.inf, b_abs]
    for deg in range(2, order + 1):
        # Higher polynomial orders must be tightly bounded to avoid runaway baseline.
        bound = c_abs / (span ** (deg - 2))
        bound = max(float(bound), 1e-12)
        lb_base.append(-bound)
        ub_base.append(bound)

    g_guess: list[float] = []
    lb_g: list[float] = []
    ub_g: list[float] = []

    y_median = float(np.median(y))
    amp_floor = max(1e-6, float(cfg.get("amp_min_frac", 0.0)) * float(np.ptp(y)))
    for i, (mu_lo, mu_hi) in enumerate(cfg["peak_bounds"]):
        sig_lo_i, sig_hi_i = sigma_by_peak[i]
        m = (x >= float(mu_lo)) & (x <= float(mu_hi))
        if np.any(m):
            local_x = x[m]
            local_y = y[m]
            j = int(np.argmax(local_y))
            mu0 = float(local_x[j])
            amp0 = float(max(local_y[j] - y_median, amp_floor, 1e-3))
        else:
            mu0 = 0.5 * float(mu_lo + mu_hi)
            amp0 = float(max(np.max(y) - y_median, amp_floor, 1e-3))
        sig0 = float(np.clip((float(mu_hi) - float(mu_lo)) / 6.0, sig_lo_i, sig_hi_i))
        g_guess += [amp0, mu0, sig0]
        lb_g += [amp_floor, float(mu_lo), float(sig_lo_i)]
        ub_g += [np.inf, float(mu_hi), float(sig_hi_i)]

    p0 = np.asarray(base_guess + g_guess, dtype=float)
    lb = np.asarray(lb_base + lb_g, dtype=float)
    ub = np.asarray(ub_base + ub_g, dtype=float)
    return p0, lb, ub, x0, n_base


def fit_window_gaussian(x: np.ndarray, y: np.ndarray, win_key: str) -> dict | None:
    cfg = WINDOW_CFG[win_key]
    p0, lb, ub, x0, n_base = build_init_and_bounds(x, y, cfg)
    n_peaks = len(cfg["peak_bounds"])
    w_fit = fit_weights(x, win_key)

    def resid(p: np.ndarray) -> np.ndarray:
        y_hat = model_eval(x, x0, p[:n_base], p[n_base:])
        return w_fit * (y_hat - y)

    try:
        res = least_squares(
            resid,
            x0=p0,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=max(1.0, float(np.std(y))),
            max_nfev=60000,
        )
    except Exception:
        return None

    if not res.success:
        return None

    p = res.x
    base_coeffs = p[:n_base]
    g_params = p[n_base:]
    y_baseline = baseline_eval(x, base_coeffs, x0)
    y_hat = y_baseline.copy()
    peaks: list[tuple[float, float, float, float]] = []
    comps: list[np.ndarray] = []

    for i in range(n_peaks):
        amp = float(g_params[3 * i])
        mu = float(g_params[3 * i + 1])
        sigma = float(abs(g_params[3 * i + 2]))
        gi = gauss(x, amp, mu, sigma)
        comps.append(gi)
        y_hat += gi
        area = float(amp * sigma * np.sqrt(2.0 * np.pi))
        peaks.append((amp, mu, sigma, area))

    # Keep deterministic ordering by mu.
    order = np.argsort([pk[1] for pk in peaks])
    peaks = [peaks[i] for i in order]
    comps = [comps[i] for i in order]

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / max(len(y), 1)))
    rmse_norm = rmse / (float(np.ptp(y)) + 1e-12)

    area_ratio = np.nan
    amp_ratio = np.nan
    if len(peaks) >= 2:
        area_ratio = peaks[1][3] / (peaks[0][3] + 1e-12)
        amp_ratio = peaks[1][0] / (peaks[0][0] + 1e-12)

    a = float(base_coeffs[0]) if len(base_coeffs) > 0 else np.nan
    b = float(base_coeffs[1]) if len(base_coeffs) > 1 else np.nan
    c = float(base_coeffs[2]) if len(base_coeffs) > 2 else np.nan
    d = float(base_coeffs[3]) if len(base_coeffs) > 3 else np.nan

    return {
        "params": peaks,
        "comps": comps,
        "baseline": y_baseline,
        "y_hat": y_hat,
        "r2": float(r2),
        "rmse": float(rmse),
        "rmse_norm": float(rmse_norm),
        "n": int(len(y)),
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "area_ratio": float(area_ratio) if np.isfinite(area_ratio) else np.nan,
        "amp_ratio": float(amp_ratio) if np.isfinite(amp_ratio) else np.nan,
    }


def fit_window_with_override(x: np.ndarray, y: np.ndarray, win_key: str, override_cfg: dict) -> dict | None:
    save_cfg = WINDOW_CFG[win_key].copy()
    try:
        WINDOW_CFG[win_key].update(override_cfg)
        return fit_window_gaussian(x, y, win_key)
    finally:
        WINDOW_CFG[win_key] = save_cfg


def ensure_dirs(out_root: Path) -> None:
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "fit_curves").mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (out_root / "plots" / cls).mkdir(parents=True, exist_ok=True)
        (out_root / "fit_curves" / cls).mkdir(parents=True, exist_ok=True)


def stems_from_class(smooth_dir: Path) -> list[str]:
    stems = set()
    for f in smooth_dir.glob("*__w*.csv"):
        parts = f.stem.rsplit("__", 1)
        if len(parts) == 2:
            stems.add(parts[0])
    return sorted(stems)


def process_sample(stem: str, cls: str, smooth_root: Path, raw_root: Path, out_root: Path) -> list[dict]:
    rows: list[dict] = []
    payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None] = {}

    for win_key in ("w1", "w2", "w3"):
        sm_path = smooth_root / cls / f"{stem}__{win_key}.csv"
        raw_path = raw_root / cls / f"{stem}__{win_key}.csv"
        df_sm = read_window_csv(sm_path)
        df_raw = read_window_csv(raw_path)
        n_peaks = len(WINDOW_CFG[win_key]["peak_bounds"])

        if df_sm.empty or len(df_sm) < MIN_POINTS:
            for k in range(1, n_peaks + 1):
                rows.append(
                    {
                        "class": cls,
                        "file": f"{stem}.csv",
                        "window": win_key,
                        "n_peaks": n_peaks,
                        "peak_idx": k,
                        "amp": np.nan,
                        "mu": np.nan,
                        "sigma": np.nan,
                        "fwhm": np.nan,
                        "area": np.nan,
                        "a": np.nan,
                        "b": np.nan,
                        "c": np.nan,
                        "d": np.nan,
                        "r2": np.nan,
                        "rmse": np.nan,
                        "rmse_norm": np.nan,
                        "n_points": int(len(df_sm)),
                        "area_ratio": np.nan,
                        "amp_ratio": np.nan,
                        "mode": "no-data",
                    }
                )
            payload[win_key] = None
            continue

        x = df_sm["x"].to_numpy(dtype=float)
        y = df_sm["y"].to_numpy(dtype=float)
        y_fit = y.copy()
        fit_mode = "ok"
        if win_key == "w3":
            rr = rough_ratio(y)
            if rr > W3_RESCUE_ROUGH_THRESH:
                y_fit = safe_savgol(y, W3_RESCUE_SG_WINDOW, W3_RESCUE_SG_POLY)
                fit_mode = f"w3-rescue-sg{W3_RESCUE_SG_WINDOW}"

        fit = fit_window_gaussian(x, y_fit, win_key)
        if fit is not None and win_key == "w3" and W3_FORCE_DOUBLE:
            fit_forced = fit_window_with_override(x, y_fit, win_key, W3_FORCE_CFG)
            if fit_forced is not None:
                r2_drop = float(fit["r2"] - fit_forced["r2"])
                base_score = w3_shape_score(x, y_fit, fit["y_hat"])
                forced_score = w3_shape_score(x, y_fit, fit_forced["y_hat"])
                base_pk = count_prominent_peaks(fit["y_hat"])
                forced_pk = count_prominent_peaks(fit_forced["y_hat"])
                better_shape = forced_score < (0.99 * base_score)
                better_peaks = (forced_pk >= base_pk) and (base_pk < 2)
                if r2_drop <= W3_FORCE_R2_DROP_MAX and (better_shape or better_peaks):
                    fit = fit_forced
                    fit_mode = f"{fit_mode}+w3-double"
        if fit is None:
            for k in range(1, n_peaks + 1):
                rows.append(
                    {
                        "class": cls,
                        "file": f"{stem}.csv",
                        "window": win_key,
                        "n_peaks": n_peaks,
                        "peak_idx": k,
                        "amp": np.nan,
                        "mu": np.nan,
                        "sigma": np.nan,
                        "fwhm": np.nan,
                        "area": np.nan,
                        "a": np.nan,
                        "b": np.nan,
                        "c": np.nan,
                        "d": np.nan,
                        "r2": np.nan,
                        "rmse": np.nan,
                        "rmse_norm": np.nan,
                        "n_points": int(len(x)),
                        "area_ratio": np.nan,
                        "amp_ratio": np.nan,
                        "mode": "fit-failed",
                    }
                )
            payload[win_key] = None
            continue

        for k, (amp, mu, sigma, area) in enumerate(fit["params"], start=1):
            rows.append(
                {
                    "class": cls,
                    "file": f"{stem}.csv",
                    "window": win_key,
                    "n_peaks": n_peaks,
                    "peak_idx": k,
                    "amp": float(amp),
                    "mu": float(mu),
                    "sigma": float(sigma),
                    "fwhm": float(FWHM_FACTOR * sigma),
                    "area": float(area),
                    "a": fit["a"],
                    "b": fit["b"],
                    "c": fit["c"],
                    "d": fit["d"],
                    "r2": fit["r2"],
                    "rmse": fit["rmse"],
                    "rmse_norm": fit["rmse_norm"],
                    "n_points": fit["n"],
                    "area_ratio": fit["area_ratio"],
                    "amp_ratio": fit["amp_ratio"],
                    "mode": fit_mode,
                }
            )

        y_raw = np.full_like(x, np.nan, dtype=float)
        if not df_raw.empty and len(df_raw) == len(df_sm):
            xr = df_raw["x"].to_numpy(dtype=float)
            yr = df_raw["y"].to_numpy(dtype=float)
            if np.allclose(xr, x):
                y_raw = yr

        curve_df = pd.DataFrame(
            {
                "x": x,
                "y_raw": y_raw,
                "y_smooth": y,
                "y_fit_input": y_fit,
                "y_fit": fit["y_hat"],
                "baseline": fit["baseline"],
                "g1_plus_baseline": fit["baseline"] + fit["comps"][0],
                "g2_plus_baseline": fit["baseline"] + fit["comps"][1],
                "resid": y - fit["y_hat"],
            }
        )
        curve_path = out_root / "fit_curves" / cls / f"{stem}__{win_key}_fit.csv"
        curve_df.to_csv(curve_path, index=False)
        payload[win_key] = (x, y_fit, y_raw, fit)

    out_plot = out_root / "plots" / cls / f"{stem}__program3_gaussian.png"
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.6))
    for i, win_key in enumerate(("w1", "w2", "w3")):
        ax = axes[i]
        lo, hi = WINDOW_CFG[win_key]["range"]
        item = payload.get(win_key)
        if item is None:
            ax.text(0.5, 0.5, f"{win_key}: no fit", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{win_key}: {lo:.0f}-{hi:.0f} cm$^{{-1}}$")
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity")
            continue

        x, y_sm, y_raw, fit = item
        ax.plot(x, y_sm, color="k", lw=1.35, label="smoothed window")
        ax.plot(x, fit["y_hat"], color="#d62728", lw=1.6, ls="--", label="fit total")
        ax.set_title(f"{win_key}: {lo:.0f}-{hi:.0f} cm$^{{-1}}$ | r2={fit['r2']:.3f} | rmse={fit['rmse_norm']:.3f}")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)

    fig.suptitle(f"{cls} | {stem} | Program 3 Gaussian fit", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_plot, dpi=190)
    plt.close(fig)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Program 3: Gaussian fit on Program 2 windows.")
    ap.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Program 2 output root."
    )
    ap.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Program 3 output root."
    )
    ap.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON config overriding window fitting settings for exploratory runs.",
    )
    args = ap.parse_args()

    apply_fit_config(args.config_json)
    smooth_root = args.input_root / "windows_smooth"
    raw_root = args.input_root / "windows_raw"
    ensure_dirs(args.output_root)

    rows: list[dict] = []
    for cls in CLASSES:
        class_dir = smooth_root / cls
        if not class_dir.exists():
            print(f"[{cls}] missing input directory: {class_dir}")
            continue

        stems = stems_from_class(class_dir)
        if not stems:
            print(f"[{cls}] no window CSV found in {class_dir}")
            continue

        for stem in stems:
            try:
                recs = process_sample(stem, cls, smooth_root, raw_root, args.output_root)
                rows.extend(recs)
                print(f"[{cls}] {stem}: OK")
            except Exception as exc:
                print(f"[{cls}] {stem}: ERROR ({exc})")

    out_summary = args.output_root / "gaussian_fit_summary.csv"
    pd.DataFrame(rows).to_csv(out_summary, index=False)
    print(f"Saved summary: {out_summary}")
    print(f"Fit curves root: {args.output_root / 'fit_curves'}")
    print(f"Plots root: {args.output_root / 'plots'}")


if __name__ == "__main__":
    main()
