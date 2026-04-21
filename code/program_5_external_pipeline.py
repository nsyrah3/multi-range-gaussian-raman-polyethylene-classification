#!/usr/bin/env python3
"""
Program 5: end-to-end pipeline (Program 1 -> Program 2 -> Program 3) for external TXT data.

Goal:
- Read external SpectraSuite TXT files.
- Convert to CSV (x, y) grouped by class (HDPE/LDPE inferred from filename).
- Run Program 1 baseline correction on full spectrum.
- Run Program 2 cut + smoothing windows.
- Run Program 3 Gaussian fitting.
- Export final Gaussian results for external dataset.

Default input:
- external_data

Default output root:
- outputs/program_5_external_pipeline
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import subprocess
import sys
import shutil

import pandas as pd


BEGIN_MARKER = ">>>>>Begin Processed Spectral Data<<<<<"
END_MARKER = ">>>>>End Processed Spectral Data<<<<<"
VALID_CLASSES = ("HDPE", "LDPE")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_EXTERNAL_DIR = REPO_ROOT / "external_data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "program_5_external_pipeline"


def infer_class_from_name(name: str) -> str | None:
    up = name.upper()
    if "HDPE" in up:
        return "HDPE"
    if "LDPE" in up:
        return "LDPE"
    # Common material hints in local naming
    if any(k in up for k in ("WARP", "ZIPLOCK", "KRESEK")):
        return "LDPE"
    if any(k in up for k in ("BOTOL", "TUTUP")):
        return "HDPE"
    return None


def parse_spectrasuite_txt(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    begin = None
    end = None
    for i, line in enumerate(lines):
        if BEGIN_MARKER in line:
            begin = i + 1
        if END_MARKER in line:
            end = i
            break

    if begin is None:
        raise ValueError("Begin marker not found")
    if end is None:
        end = len(lines)
    if begin >= end:
        raise ValueError("No spectral rows found")

    rows: list[tuple[float, float]] = []
    pat = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[\t; ]+\s*([+-]?\d+(?:[.,]\d+)?)\s*$")
    for raw in lines[begin:end]:
        s = raw.strip()
        if not s:
            continue
        m = pat.match(s)
        if not m:
            continue
        x_s, y_s = m.group(1), m.group(2)
        x = float(x_s.replace(",", "."))
        y = float(y_s.replace(",", "."))
        rows.append((x, y))

    if len(rows) < 10:
        raise ValueError(f"Too few parsed points: {len(rows)}")

    df = pd.DataFrame(rows, columns=["x", "y"]).dropna().sort_values("x").reset_index(drop=True)
    return df


def ensure_external_input_dirs(root: Path) -> None:
    for cls in VALID_CLASSES:
        (root / cls).mkdir(parents=True, exist_ok=True)


def convert_external_txt_to_csv(input_dir: Path, out_root: Path, default_class: str | None) -> tuple[pd.DataFrame, Path]:
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt found in {input_dir}")

    csv_root = out_root / "external_input_csv"
    ensure_external_input_dirs(csv_root)

    rows: list[dict] = []
    for fp in txt_files:
        cls = infer_class_from_name(fp.name)
        if cls is None:
            if default_class in VALID_CLASSES:
                cls = default_class
                status_prefix = "ok-default-class"
            else:
                rows.append(
                    {
                        "file_txt": fp.name,
                        "class": "UNKNOWN",
                        "status": "skipped-no-class-token",
                        "n_points": 0,
                        "out_csv": "",
                        "error": "Filename does not contain HDPE/LDPE and no default class provided",
                    }
                )
                continue
        else:
            status_prefix = "ok"

        try:
            df = parse_spectrasuite_txt(fp)
            out_csv = csv_root / cls / f"{fp.stem}.csv"
            df.to_csv(out_csv, index=False, header=False)
            rows.append(
                {
                    "file_txt": fp.name,
                    "class": cls,
                    "status": status_prefix,
                    "n_points": int(len(df)),
                    "out_csv": str(out_csv),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file_txt": fp.name,
                    "class": cls,
                    "status": "error-parse",
                    "n_points": 0,
                    "out_csv": "",
                    "error": str(exc),
                }
            )

    conv_df = pd.DataFrame(rows)
    conv_csv = out_root / "program_5_external_conversion_summary.csv"
    conv_df.to_csv(conv_csv, index=False)
    return conv_df, csv_root


def copy_external_csv_by_class(input_dir: Path, out_root: Path) -> tuple[pd.DataFrame, Path]:
    """
    External input mode for already-classified CSVs:
    <input_dir>/HDPE/*.csv and <input_dir>/LDPE/*.csv
    """
    csv_root = out_root / "external_input_csv"
    ensure_external_input_dirs(csv_root)

    rows: list[dict] = []
    n_total = 0
    for cls in VALID_CLASSES:
        in_cls = input_dir / cls
        files = sorted(in_cls.glob("*.csv")) if in_cls.exists() else []
        for fp in files:
            n_total += 1
            dst = csv_root / cls / fp.name
            stem = fp.stem
            i = 1
            while dst.exists():
                dst = csv_root / cls / f"{stem}_{i}.csv"
                i += 1
            shutil.copy2(fp, dst)
            rows.append(
                {
                    "file_csv": fp.name,
                    "class": cls,
                    "status": "ok-csv-input",
                    "n_points": 0,
                    "out_csv": str(dst),
                    "error": "",
                }
            )

    if n_total == 0:
        raise FileNotFoundError(f"No class CSV found in {input_dir}/HDPE or {input_dir}/LDPE")

    conv_df = pd.DataFrame(rows)
    conv_csv = out_root / "program_5_external_conversion_summary.csv"
    conv_df.to_csv(conv_csv, index=False)
    return conv_df, csv_root


def run_step(cmd: list[str], step_name: str) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        msg = (
            f"[{step_name}] failed with code {res.returncode}\n"
            f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
        )
        raise RuntimeError(msg)
    return res.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Program 5: External TXT -> Program1/2/3 Gaussian outputs.")
    ap.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR, help="Folder with external TXT files.")
    ap.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for Program 5."
    )
    ap.add_argument(
        "--default-class",
        type=str,
        default=None,
        help="Fallback class for files whose names do not contain class hints.",
    )
    args = ap.parse_args()
    if args.default_class is not None:
        args.default_class = str(args.default_class).upper()
        if args.default_class not in VALID_CLASSES:
            raise ValueError("--default-class must be HDPE or LDPE")

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Step 0: external input -> class CSV root
    txt_files = sorted(args.external_dir.glob("*.txt"))
    has_class_csv_dirs = any((args.external_dir / cls).exists() for cls in VALID_CLASSES)
    if txt_files:
        conv_df, external_csv_root = convert_external_txt_to_csv(args.external_dir, args.output_root, args.default_class)
        input_mode = "txt"
    elif has_class_csv_dirs:
        conv_df, external_csv_root = copy_external_csv_by_class(args.external_dir, args.output_root)
        input_mode = "csv-by-class"
    else:
        raise RuntimeError(
            f"Unsupported external input layout in {args.external_dir}. "
            "Provide *.txt in root, or HDPE/LDPE subfolders with *.csv."
        )

    n_ok = int(conv_df["status"].astype(str).str.startswith("ok").sum()) if not conv_df.empty else 0
    if n_ok == 0:
        raise RuntimeError("No valid external file converted to CSV, pipeline stopped.")

    # Step 1: Program 1
    out_p1 = args.output_root / "Program1"
    cmd1 = [
        sys.executable,
        str(SCRIPT_DIR / "program_1_baseline_fullspectrum.py"),
        "--input-root",
        str(external_csv_root),
        "--output-root",
        str(out_p1),
    ]
    log1 = run_step(cmd1, "Program1")
    (args.output_root / "program1_run.log").write_text(log1, encoding="utf-8")

    # Step 2: Program 2 (input = corrected output from Program 1)
    out_p2 = args.output_root / "Program2"
    p2_input = out_p1 / "corrected"
    cmd2 = [
        sys.executable,
        str(SCRIPT_DIR / "program_2_cut_smooth_windows.py"),
        "--input-root",
        str(p2_input),
        "--output-root",
        str(out_p2),
    ]
    log2 = run_step(cmd2, "Program2")
    (args.output_root / "program2_run.log").write_text(log2, encoding="utf-8")

    # Step 3: Program 3 (input = Program 2 output root)
    out_p3 = args.output_root / "Program3"
    cmd3 = [
        sys.executable,
        str(SCRIPT_DIR / "program_3_fit_gaussian_windows.py"),
        "--input-root",
        str(out_p2),
        "--output-root",
        str(out_p3),
    ]
    log3 = run_step(cmd3, "Program3")
    (args.output_root / "program3_run.log").write_text(log3, encoding="utf-8")

    # Final external Gaussian exports
    summary_src = out_p3 / "gaussian_fit_summary.csv"
    if not summary_src.exists():
        raise FileNotFoundError(f"Missing expected Program3 summary: {summary_src}")

    gdf = pd.read_csv(summary_src)
    gdf.to_csv(args.output_root / "external_gaussian_fit_summary.csv", index=False)

    ml_cols = ["class", "file", "window", "peak_idx", "amp", "mu", "sigma", "fwhm", "area"]
    g_ml = gdf[ml_cols].dropna(subset=["amp", "mu", "sigma", "fwhm", "area"]).copy()
    g_ml = g_ml.sort_values(["class", "file", "window", "peak_idx"]).reset_index(drop=True)
    g_ml.to_csv(args.output_root / "external_gaussian_features_ml_labeled.csv", index=False)

    report = {
        "external_dir": str(args.external_dir),
        "output_root": str(args.output_root),
        "input_mode": input_mode,
        "n_input_total": int(len(conv_df)),
        "n_input_ok": n_ok,
        "n_input_skipped_or_error": int(len(conv_df) - n_ok),
        "program1_output": str(out_p1),
        "program2_output": str(out_p2),
        "program3_output": str(out_p3),
        "final_summary_csv": str(args.output_root / "external_gaussian_fit_summary.csv"),
        "final_features_csv": str(args.output_root / "external_gaussian_features_ml_labeled.csv"),
    }
    pd.DataFrame([report]).to_csv(args.output_root / "program_5_report.csv", index=False)

    print(f"Input mode: {input_mode}")
    print(f"Converted input: {n_ok}/{len(conv_df)}")
    print(f"Program1 output: {out_p1}")
    print(f"Program2 output: {out_p2}")
    print(f"Program3 output: {out_p3}")
    print(f"Saved: {args.output_root / 'external_gaussian_fit_summary.csv'}")
    print(f"Saved: {args.output_root / 'external_gaussian_features_ml_labeled.csv'}")
    print(f"Saved: {args.output_root / 'program_5_report.csv'}")


if __name__ == "__main__":
    main()
