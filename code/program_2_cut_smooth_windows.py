#!/usr/bin/env python3
"""
Program 2: cut Raman spectra into 3 windows and apply smoothing only.

Pipeline:
1) Load CSV spectra.
2) Cut into 3 windows:
   - w1: 995-1215
   - w2: 1215-1596
   - w3: 2624-3125
3) No baseline correction.
4) Apply adaptive Savitzky-Golay smoothing per window with peak-preservation guard.
5) Save raw window CSV, smoothed window CSV, and one 3-panel plot per file.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


CLASSES = ("HDPE", "LDPE")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs" / "program_1_baseline_fullspectrum" / "corrected"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "program_2_cut_smooth_windows"
WINDOW_RANGES = {
    "w1": (995.0, 1215.0),
    "w2": (1215.0, 1596.0),
    "w3": (2624.0, 3125.0),
}

# Smoothing only (adaptive, from stronger to weaker candidates)
WINDOW_SG_CANDIDATES = {
    "w2": [7, 5, 3],
    "w3": [9, 7, 5, 3],
}
SG_POLY = 3

# Guardrail: keep main peak height and overall shape.
MIN_PEAK_RATIO = 0.94
MIN_SHAPE_CORR = 0.95

# Special handling for w1:
# keep 2 main peaks near-raw while smoothing harder elsewhere.
W1_HARD_WINDOWS = [15, 13, 11, 9, 7]
W1_POLY = 3
W1_MAIN_PEAK_RANGES_CM = (
    (1047.0, 1082.0),
    (1117.0, 1146.0),
)
W1_PEAK_BLEND_MARGIN_CM = 8.0
# Extra cleanup for w1 second peak to suppress double-spiky apex.
W1_PEAK2_RANGE_CM = (1117.0, 1146.0)
W1_PEAK2_LOCAL_SG_WINDOW = 5
W1_PEAK2_LOCAL_SG_POLY = 2
W1_PEAK2_LOCAL_BLEND = 0.70
# In peak zones, keep mostly raw but allow slight smoothing.
# y_mix = alpha*raw + (1-alpha)*smooth ; alpha close to 1 => very mild smoothing.
W1_PEAK_RAW_WEIGHT = 1.00

# New smoothing strategy for w1 peak zones:
# use a softer SG profile (not raw injection) so apex is rounded, then
# lightly re-lock peak height to avoid excessive flattening.
W1_PEAK_SOFT_SG_WINDOW = 7
W1_PEAK_SOFT_SG_POLY = 3
W1_PEAK_HEIGHT_LOCK_MIN = 0.94
W1_PEAK_HEIGHT_LOCK_MAX = 1.08


def _to_tuple_range(value) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected [lo, hi] style range, got: {value!r}")
    return float(value[0]), float(value[1])


def apply_window_config(config_path: Path | None) -> None:
    global WINDOW_RANGES
    if config_path is None:
        return
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ranges = cfg.get("window_ranges")
    if not ranges:
        return
    updated = dict(WINDOW_RANGES)
    for key, value in ranges.items():
        if key not in updated:
            raise ValueError(f"Unknown window key in config: {key}")
        updated[key] = _to_tuple_range(value)
    WINDOW_RANGES = updated


def read_raman_csv(path: Path) -> pd.DataFrame:
    """Load CSV with 2 columns, tolerant to comma/semicolon separators."""
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None, sep=";")

    if df.shape[1] < 2:
        df = pd.read_csv(path, header=None, sep=";")
    if df.shape[1] > 2:
        df = df.iloc[:, :2].copy()

    df.columns = ["x", "y"]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna().sort_values("x").reset_index(drop=True)


def safe_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay with safety checks for short windows."""
    n = len(y)
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n if n % 2 == 1 else max(1, n - 1)
    if win < poly + 2 or win < 3 or n < 3:
        return y
    return savgol_filter(y, window_length=win, polyorder=poly)


def centered_corr(y_ref: np.ndarray, y_cmp: np.ndarray) -> float:
    a = y_ref - float(np.mean(y_ref))
    b = y_cmp - float(np.mean(y_cmp))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0:
        return np.nan
    return float(np.dot(a, b) / den)


def smooth_preserve_peak(y_raw: np.ndarray, win_key: str) -> tuple[np.ndarray, int, int, float, float, str]:
    """
    Adaptive SG smoothing:
    - Start from stronger candidate window.
    - If peak attenuation too strong or shape drift too large, reduce smoothing.
    - Fallback to raw signal when no candidate passes guardrail.
    """
    candidates = WINDOW_SG_CANDIDATES.get(win_key, [5, 3])
    peak_raw = float(np.max(y_raw)) + 1e-12

    for w in candidates:
        poly = min(SG_POLY, max(1, w - 2))
        y_sm = safe_savgol(y_raw, w, poly)
        peak_ratio = float(np.max(y_sm) / peak_raw)
        corr = centered_corr(y_raw, y_sm)
        if peak_ratio >= MIN_PEAK_RATIO and (np.isnan(corr) or corr >= MIN_SHAPE_CORR):
            return y_sm, int(w), int(poly), peak_ratio, corr, "sg-adaptive"

    # If all candidates flatten/distort too much, keep raw to preserve peak shape.
    return y_raw.copy(), 0, 0, 1.0, 1.0, "fallback-raw"


def build_w1_keep_mask(x: np.ndarray) -> np.ndarray:
    """
    Build blend mask for w1:
    - alpha = raw weight in fixed main peak ranges
    - alpha decays linearly to 0 in blend margin outside each range
    """
    alpha = np.zeros_like(x, dtype=float)
    blend_margin = max(float(W1_PEAK_BLEND_MARGIN_CM), 0.5)

    for lo, hi in W1_MAIN_PEAK_RANGES_CM:
        lo_f = float(min(lo, hi))
        hi_f = float(max(lo, hi))
        local = np.zeros_like(x, dtype=float)
        keep = (x >= lo_f) & (x <= hi_f)
        left_blend = (x >= (lo_f - blend_margin)) & (x < lo_f)
        right_blend = (x > hi_f) & (x <= (hi_f + blend_margin))

        local[keep] = W1_PEAK_RAW_WEIGHT
        local[left_blend] = W1_PEAK_RAW_WEIGHT * (x[left_blend] - (lo_f - blend_margin)) / blend_margin
        local[right_blend] = W1_PEAK_RAW_WEIGHT * ((hi_f + blend_margin) - x[right_blend]) / blend_margin
        alpha = np.maximum(alpha, local)

    return np.clip(alpha, 0.0, 1.0)


def w1_main_peak_ratio(x: np.ndarray, y_raw: np.ndarray, y_sm: np.ndarray) -> float:
    """Peak ratio measured only in fixed main peak ranges."""
    ratios = []
    for lo, hi in W1_MAIN_PEAK_RANGES_CM:
        lo_f = float(min(lo, hi))
        hi_f = float(max(lo, hi))
        m = (x >= lo_f) & (x <= hi_f)
        if not np.any(m):
            continue
        raw_pk = float(np.max(y_raw[m])) + 1e-12
        sm_pk = float(np.max(y_sm[m]))
        ratios.append(sm_pk / raw_pk)
    if not ratios:
        return 1.0
    return float(min(ratios))


def cleanup_w1_second_peak(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Suppress residual double-spiky shape inside the 2nd main peak (1117-1146)
    using local SG smoothing blended with current signal.
    """
    lo_f = float(min(W1_PEAK2_RANGE_CM))
    hi_f = float(max(W1_PEAK2_RANGE_CM))
    m = (x >= lo_f) & (x <= hi_f)
    n = int(np.count_nonzero(m))
    if n < 5:
        return y

    local = y[m]
    w = int(W1_PEAK2_LOCAL_SG_WINDOW)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return y

    p = min(int(W1_PEAK2_LOCAL_SG_POLY), max(1, w - 2))
    local_sm = safe_savgol(local, w, p)
    b = float(np.clip(W1_PEAK2_LOCAL_BLEND, 0.0, 1.0))

    out = y.copy()
    out[m] = (1.0 - b) * local + b * local_sm
    return out


def smooth_w1_two_peak_preserve(x: np.ndarray, y_raw: np.ndarray) -> tuple[np.ndarray, int, int, float, float, str]:
    """
    For w1:
    - smooth strongly outside peaks,
    - preserve 2 main peaks close to raw.
    """
    if len(y_raw) < 5:
        return y_raw.copy(), 0, 0, 1.0, 1.0, "fallback-raw"

    # Force hard smoothing outside peaks, while protecting the two dominant peaks.
    # This avoids w1 fallback-heavy behavior on noisy LDPE.
    w = int(W1_HARD_WINDOWS[0])
    poly = min(W1_POLY, max(1, w - 2))
    y_hard = safe_savgol(y_raw, w, poly)
    # Soft profile for peak zones to remove sharp corners at apex.
    w_soft = int(W1_PEAK_SOFT_SG_WINDOW)
    p_soft = min(int(W1_PEAK_SOFT_SG_POLY), max(1, w_soft - 2))
    y_peak = safe_savgol(y_raw, w_soft, p_soft)

    # Keep peak heights close to raw while preserving smooth shape.
    lock_lo = float(min(W1_PEAK_HEIGHT_LOCK_MIN, W1_PEAK_HEIGHT_LOCK_MAX))
    lock_hi = float(max(W1_PEAK_HEIGHT_LOCK_MIN, W1_PEAK_HEIGHT_LOCK_MAX))
    for lo, hi in W1_MAIN_PEAK_RANGES_CM:
        m = (x >= float(min(lo, hi))) & (x <= float(max(lo, hi)))
        if not np.any(m):
            continue
        pk_raw = float(np.max(y_raw[m])) + 1e-12
        pk_sm = float(np.max(y_peak[m])) + 1e-12
        scale = float(np.clip(pk_raw / pk_sm, lock_lo, lock_hi))
        y_peak[m] *= scale

    alpha = build_w1_keep_mask(x)
    # Blend hard-smoothed background with soft-smoothed peak profile.
    y_mix = alpha * y_peak + (1.0 - alpha) * y_hard
    y_mix = cleanup_w1_second_peak(x, y_mix)

    peak_ratio = w1_main_peak_ratio(x, y_raw, y_mix)
    corr = centered_corr(y_raw, y_mix)
    return y_mix, int(w), int(poly), peak_ratio, corr, "w1-two-peak-preserve"


def ensure_dirs(out_root: Path) -> None:
    (out_root / "windows_raw").mkdir(parents=True, exist_ok=True)
    (out_root / "windows_smooth").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (out_root / "windows_raw" / cls).mkdir(parents=True, exist_ok=True)
        (out_root / "windows_smooth" / cls).mkdir(parents=True, exist_ok=True)
        (out_root / "plots" / cls).mkdir(parents=True, exist_ok=True)


def process_file(csv_path: Path, cls: str, out_root: Path) -> dict:
    df = read_raman_csv(csv_path)
    stem = csv_path.stem

    rec: dict[str, object] = {
        "class": cls,
        "file": csv_path.name,
    }
    plot_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}

    for key, (lo, hi) in WINDOW_RANGES.items():
        m = (df["x"] >= lo) & (df["x"] <= hi)
        sub = df.loc[m].copy()

        raw_out = out_root / "windows_raw" / cls / f"{stem}__{key}.csv"
        sm_out = out_root / "windows_smooth" / cls / f"{stem}__{key}.csv"

        if sub.empty or len(sub) < 3:
            pd.DataFrame({"x": [], "y": []}).to_csv(raw_out, index=False, header=False)
            pd.DataFrame({"x": [], "y": []}).to_csv(sm_out, index=False, header=False)
            rec[f"{key}_n"] = 0
            rec[f"{key}_rough_ratio"] = np.nan
            rec[f"{key}_sg_window"] = np.nan
            rec[f"{key}_sg_poly"] = np.nan
            rec[f"{key}_peak_ratio"] = np.nan
            rec[f"{key}_shape_corr"] = np.nan
            rec[f"{key}_mode"] = "no-data"
            plot_payload[key] = None
            continue

        x = sub["x"].to_numpy(dtype=float)
        y_raw = sub["y"].to_numpy(dtype=float)
        if key == "w1":
            y_sm, sg_w_used, sg_poly_used, peak_ratio, shape_corr, mode = smooth_w1_two_peak_preserve(x, y_raw)
        else:
            y_sm, sg_w_used, sg_poly_used, peak_ratio, shape_corr, mode = smooth_preserve_peak(y_raw, key)

        pd.DataFrame({"x": x, "y": y_raw}).to_csv(raw_out, index=False, header=False)
        pd.DataFrame({"x": x, "y": y_sm}).to_csv(sm_out, index=False, header=False)

        rough_raw = float(np.std(np.diff(y_raw))) if len(y_raw) > 2 else np.nan
        rough_sm = float(np.std(np.diff(y_sm))) if len(y_sm) > 2 else np.nan
        rough_ratio = rough_sm / (rough_raw + 1e-12) if np.isfinite(rough_raw) else np.nan

        rec[f"{key}_n"] = int(len(x))
        rec[f"{key}_rough_ratio"] = float(rough_ratio)
        rec[f"{key}_sg_window"] = int(sg_w_used)
        rec[f"{key}_sg_poly"] = int(sg_poly_used)
        rec[f"{key}_peak_ratio"] = float(peak_ratio)
        rec[f"{key}_shape_corr"] = float(shape_corr) if np.isfinite(shape_corr) else np.nan
        rec[f"{key}_mode"] = mode
        plot_payload[key] = (x, y_raw, y_sm)

    out_plot = out_root / "plots" / cls / f"{stem}__program2_windows.png"
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2))
    for idx, key in enumerate(("w1", "w2", "w3")):
        ax = axes[idx]
        lo, hi = WINDOW_RANGES[key]
        payload = plot_payload.get(key)
        if payload is None:
            ax.text(0.5, 0.5, f"{key}: no data in {lo:.0f}-{hi:.0f}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{key}: {lo:.0f}-{hi:.0f} cm$^{{-1}}$")
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity")
            continue
        x, y_raw, y_sm = payload
        sg_w_used = int(rec[f"{key}_sg_window"])
        sg_poly_used = int(rec[f"{key}_sg_poly"])
        peak_ratio = float(rec[f"{key}_peak_ratio"])
        mode = str(rec[f"{key}_mode"])
        if sg_w_used > 0:
            if mode == "w1-two-peak-preserve":
                lbl = f"smoothed (2-peak preserve, SG w={sg_w_used}, p={sg_poly_used})"
            else:
                lbl = f"smoothed (SG w={sg_w_used}, p={sg_poly_used})"
            # Draw raw first, then smoothed on top so dashed line remains visible at peaks.
            ax.plot(x, y_raw, color="0.15", lw=1.05, label="raw window", zorder=2)
            ax.plot(
                x,
                y_sm,
                color="#1f77b4",
                lw=1.55,
                ls=(0, (4, 2)),
                alpha=0.95,
                label=lbl,
                zorder=3,
            )
        else:
            # Fallback means smoothed == raw; avoid overlaying identical lines.
            ax.plot(x, y_raw, "k-", lw=1.2, label="raw window (fallback: no smoothing)", zorder=3)
        rr = rec[f"{key}_rough_ratio"]
        ax.set_title(
            f"{key}: {lo:.0f}-{hi:.0f} cm$^{{-1}}$ | n={len(x)} | rough={rr:.3f} | peak={peak_ratio:.3f}"
        )
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)

    fig.suptitle(f"{cls} | {stem} | Program 2 (no baseline, smoothing only)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)

    rec["out_plot"] = str(out_plot)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Program 2: cut into 3 windows and smooth only (no baseline)."
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing class folders (HDPE, LDPE).",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output folder for Program 2.",
    )
    ap.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON config overriding window ranges for exploratory runs.",
    )
    args = ap.parse_args()

    apply_window_config(args.config_json)
    ensure_dirs(args.output_root)
    rows: list[dict] = []

    for cls in CLASSES:
        in_dir = args.input_root / cls
        files = sorted(in_dir.glob("*.csv"))
        if not files:
            print(f"[{cls}] no CSV found in {in_dir}")
            continue

        for f in files:
            try:
                rec = process_file(f, cls, args.output_root)
                rows.append(rec)
                print(f"[{cls}] {f.name}: OK")
            except Exception as exc:
                print(f"[{cls}] {f.name}: ERROR ({exc})")

    summary_path = args.output_root / "program_2_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(f"Raw windows root: {args.output_root / 'windows_raw'}")
    print(f"Smoothed windows root: {args.output_root / 'windows_smooth'}")
    print(f"Plots root: {args.output_root / 'plots'}")


if __name__ == "__main__":
    main()
