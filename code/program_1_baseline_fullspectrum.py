#!/usr/bin/env python3
"""
Program 1: baseline correction on full-spectrum Raman (500-3200 cm^-1).

Output:
- Corrected CSV (x, y_corrected) for each input CSV.
- One plot per CSV with 2 curves:
  1) before baseline correction
  2) after baseline correction
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


CLASSES = ("HDPE", "LDPE")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "program_1_baseline_fullspectrum"
CUT_LO = 500.0
CUT_HI = 3200.0

# AsLS baseline parameters (fixed for Program 1)
ASLS_LAM = 1e6
ASLS_P = 0.01
ASLS_NITER = 15

# Presmooth only for baseline estimation stability
DO_BASELINE_PRESMOOTH = True
BASELINE_SG_WINDOW = 11
BASELINE_SG_POLY = 2

# Optional smoothing after baseline (kept off by default)
DO_FINAL_SMOOTH = False
FINAL_SG_WINDOW = 9
FINAL_SG_POLY = 2

SILENT_REGION = (2000.0, 2500.0)

# Cosmic removal at fixed artefact positions (no despike)
DO_COSMIC_PATCH = True
PATCH_TARGETS = (1764.38, 987.69)
PATCH_K = 5


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


def cut_range(df: pd.DataFrame, lo: float = CUT_LO, hi: float = CUT_HI) -> pd.DataFrame:
    m = (df["x"] >= lo) & (df["x"] <= hi)
    return df.loc[m].reset_index(drop=True)


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


def patch_targets(x: np.ndarray, y: np.ndarray, targets=PATCH_TARGETS, k: int = PATCH_K) -> tuple[np.ndarray, int]:
    """Patch fixed cosmic artefacts using local linear interpolation."""
    y_out = y.copy()
    n = len(y_out)
    if n < (2 * k + 2):
        return y_out, 0

    n_patched = 0
    for t in targets:
        idx = int(np.argmin(np.abs(x - t)))
        if idx >= k and (idx + k) < n:
            left = slice(idx - k, idx)
            right = slice(idx + 1, idx + 1 + k)
            x_l, y_l = float(np.mean(x[left])), float(np.mean(y_out[left]))
            x_r, y_r = float(np.mean(x[right])), float(np.mean(y_out[right]))
            x0 = float(x[idx])
            old_val = float(y_out[idx])
            if x_r != x_l:
                y_out[idx] = y_l + (y_r - y_l) * (x0 - x_l) / (x_r - x_l)
            else:
                y_out[idx] = 0.5 * (y_l + y_r)
            if abs(y_out[idx] - old_val) > 1e-12:
                n_patched += 1
    return y_out, n_patched


def asls_baseline(y: np.ndarray, lam: float = ASLS_LAM, p: float = ASLS_P, niter: int = ASLS_NITER) -> np.ndarray:
    """Sparse AsLS baseline."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return np.zeros_like(y)

    d2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), format="csc")
    w = np.ones(n, dtype=float)
    for _ in range(niter):
        w_mat = sparse.diags(w, 0, shape=(n, n), format="csc")
        z_mat = w_mat + lam * (d2.T @ d2)
        z = spsolve(z_mat, w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return np.asarray(z, dtype=float)


def offset_correct_silent(x: np.ndarray, y: np.ndarray, a: float = SILENT_REGION[0], b: float = SILENT_REGION[1]) -> np.ndarray:
    """Center corrected signal using median in silent region."""
    m = (x >= a) & (x <= b)
    if np.sum(m) < 5:
        return y
    return y - float(np.median(y[m]))


def ensure_dirs(out_root: Path) -> None:
    (out_root / "corrected").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (out_root / "corrected" / cls).mkdir(parents=True, exist_ok=True)
        (out_root / "plots" / cls).mkdir(parents=True, exist_ok=True)


def process_file(csv_path: Path, cls: str, out_root: Path) -> dict:
    df = cut_range(read_raman_csv(csv_path))
    if df.empty or len(df) < 8:
        raise ValueError("Not enough points after 500-3200 cut")

    x = df["x"].to_numpy(dtype=float)
    y_before = df["y"].to_numpy(dtype=float)
    y_work = y_before.copy()

    n_cosmic_patched = 0
    if DO_COSMIC_PATCH:
        y_work, n_cosmic_patched = patch_targets(x, y_work, PATCH_TARGETS, PATCH_K)

    y_for_baseline = y_work
    if DO_BASELINE_PRESMOOTH:
        y_for_baseline = safe_savgol(y_work, BASELINE_SG_WINDOW, BASELINE_SG_POLY)

    baseline = asls_baseline(y_for_baseline, lam=ASLS_LAM, p=ASLS_P, niter=ASLS_NITER)
    y_after = y_work - baseline
    y_after = offset_correct_silent(x, y_after)

    if DO_FINAL_SMOOTH:
        y_after = safe_savgol(y_after, FINAL_SG_WINDOW, FINAL_SG_POLY)

    out_csv = out_root / "corrected" / cls / f"{csv_path.stem}_corrected.csv"
    pd.DataFrame({"x": x, "y": y_after}).to_csv(out_csv, index=False, header=False)

    out_plot = out_root / "plots" / cls / f"{csv_path.stem}_before_after.png"
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(x, y_before, color="0.35", lw=1.2, label="before baseline")
    plt.plot(x, y_after, color="#1f77b4", lw=1.2, label="after cosmic+baseline")
    plt.title(f"{cls} | {csv_path.stem} | 500-3200 cm^-1 | patched={n_cosmic_patched}")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=180)
    plt.close()

    return {
        "class": cls,
        "file": csv_path.name,
        "n_points": len(x),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "cosmic_patched_points": int(n_cosmic_patched),
        "out_csv": str(out_csv),
        "out_plot": str(out_plot),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Program 1: full-spectrum baseline correction + before/after plot per CSV.")
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
        help="Output folder for corrected CSV and plots.",
    )
    args = ap.parse_args()

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

    summary_path = args.output_root / "program_1_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(f"Corrected CSV root: {args.output_root / 'corrected'}")
    print(f"Plot root: {args.output_root / 'plots'}")


if __name__ == "__main__":
    main()
