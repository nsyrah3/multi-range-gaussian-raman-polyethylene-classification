#!/usr/bin/env python3
"""
Program 6: Predict external Gaussian data using Random Forest model from Program 4.

Default inputs:
- Model: outputs/program_4_random_forest/rf_model.joblib
- Training feature matrix template: outputs/program_4_random_forest/rf_feature_matrix.csv
- External long features: outputs/program_5_external_pipeline/external_gaussian_features_ml_labeled.csv

Default outputs:
- outputs/program_6_external_prediction/external_feature_matrix_aligned.csv
- outputs/program_6_external_prediction/external_predictions.csv
- outputs/program_6_external_prediction/external_confusion_matrix.png
- outputs/program_6_external_prediction/external_prediction_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix


FEATURE_FIELDS = ("amp", "mu", "sigma", "fwhm", "area")
WINDOW_ORDER = {"w1": 1, "w2": 2, "w3": 3}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "outputs" / "program_4_random_forest" / "rf_model.joblib"
DEFAULT_TRAIN_FEATURE_MATRIX = REPO_ROOT / "outputs" / "program_4_random_forest" / "rf_feature_matrix.csv"
DEFAULT_EXTERNAL_CSV = REPO_ROOT / "outputs" / "program_5_external_pipeline" / "external_gaussian_features_ml_labeled.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "program_6_external_prediction"


def sort_feature_name(name: str) -> tuple[int, int, int]:
    parts = name.split("_")
    if len(parts) < 3:
        return (99, 99, 99)
    w = parts[0]
    p = parts[1]
    f = parts[2]
    w_idx = WINDOW_ORDER.get(w, 99)
    try:
        p_idx = int(p.replace("p", ""))
    except Exception:
        p_idx = 99
    try:
        f_idx = FEATURE_FIELDS.index(f)
    except ValueError:
        f_idx = 99
    return (w_idx, p_idx, f_idx)


def load_external_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"class", "file", "window", "peak_idx", "amp", "mu", "sigma", "fwhm", "area"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out = df.copy()
    out["class"] = out["class"].astype(str)
    out["file"] = out["file"].astype(str)
    out["window"] = out["window"].astype(str).str.lower()
    out["peak_idx"] = pd.to_numeric(out["peak_idx"], errors="coerce")
    for col in FEATURE_FIELDS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["class", "file", "window", "peak_idx", *FEATURE_FIELDS]).copy()
    out["peak_idx"] = out["peak_idx"].astype(int)
    out["sample_id"] = out["file"].str.replace(".csv", "", regex=False)
    return out


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for field in FEATURE_FIELDS:
        pv = (
            df.pivot_table(
                index=["sample_id", "class"],
                columns=["window", "peak_idx"],
                values=field,
                aggfunc="first",
            )
            .sort_index(axis=1)
        )
        pv.columns = [f"{w}_p{int(pk)}_{field}" for (w, pk) in pv.columns]
        parts.append(pv)
    wide = pd.concat(parts, axis=1)
    wide = wide.reindex(sorted(wide.columns, key=sort_feature_name), axis=1)
    return wide


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Program 6 External Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Program 6: Predict external Gaussian data using Program 4 model.")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained RF model from Program 4.",
    )
    ap.add_argument(
        "--train-feature-matrix",
        type=Path,
        default=DEFAULT_TRAIN_FEATURE_MATRIX,
        help="Feature matrix CSV from Program 4 to align feature columns.",
    )
    ap.add_argument(
        "--external-csv",
        type=Path,
        default=DEFAULT_EXTERNAL_CSV,
        help="External Gaussian labeled CSV from Program 5.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output folder for Program 6.",
    )
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Load model
    clf = joblib.load(args.model_path)

    # Load training template and expected feature order
    train_df = pd.read_csv(args.train_feature_matrix)
    if "sample_id" not in train_df.columns or "class" not in train_df.columns:
        raise ValueError("Training feature matrix must contain 'sample_id' and 'class' columns.")
    expected_features = [c for c in train_df.columns if c not in ("sample_id", "class")]
    train_medians = {c: float(train_df[c].median()) for c in expected_features}

    # Build external feature matrix (wide)
    ext_long = load_external_long(args.external_csv)
    ext_wide = long_to_wide(ext_long)

    # Align to expected features
    ext_aligned = ext_wide.copy()
    missing_cols = [c for c in expected_features if c not in ext_aligned.columns]
    extra_cols = [c for c in ext_aligned.columns if c not in expected_features]
    for c in missing_cols:
        ext_aligned[c] = train_medians.get(c, 0.0)
    ext_aligned = ext_aligned.reindex(columns=expected_features)
    for c in expected_features:
        if ext_aligned[c].isna().any():
            ext_aligned[c] = ext_aligned[c].fillna(train_medians.get(c, 0.0))

    # Save aligned feature matrix
    out_feat = ext_aligned.copy().reset_index()
    out_feat.to_csv(args.output_root / "external_feature_matrix_aligned.csv", index=False)

    # Predict
    X_ext = ext_aligned.to_numpy(dtype=float)
    y_true = ext_aligned.index.get_level_values("class").to_numpy()
    sample_id = ext_aligned.index.get_level_values("sample_id").to_numpy()

    y_pred = clf.predict(X_ext)
    y_prob = clf.predict_proba(X_ext)

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "is_correct": (y_true == y_pred).astype(int),
        }
    )
    for i, cls in enumerate(clf.classes_):
        pred_df[f"proba_{cls}"] = y_prob[:, i]
    pred_df = pred_df.sort_values("sample_id").reset_index(drop=True)
    pred_df.to_csv(args.output_root / "external_predictions.csv", index=False)

    labels = sorted(np.unique(np.r_[y_true, y_pred]).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    plot_confusion(cm, labels, args.output_root / "external_confusion_matrix.png")

    report_json = {
        "model_path": str(args.model_path),
        "train_feature_matrix": str(args.train_feature_matrix),
        "external_csv": str(args.external_csv),
        "n_external_rows_long": int(len(ext_long)),
        "n_external_samples": int(len(ext_aligned)),
        "n_expected_features": int(len(expected_features)),
        "n_missing_feature_columns_filled": int(len(missing_cols)),
        "missing_feature_columns": missing_cols,
        "n_extra_feature_columns_ignored": int(len(extra_cols)),
        "extra_feature_columns": extra_cols,
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "confusion_matrix_labels": labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
    }
    (args.output_root / "external_prediction_report.json").write_text(
        json.dumps(report_json, indent=2),
        encoding="utf-8",
    )

    print(f"Model: {args.model_path}")
    print(f"External input: {args.external_csv}")
    print(f"Saved: {args.output_root / 'external_feature_matrix_aligned.csv'}")
    print(f"Saved: {args.output_root / 'external_predictions.csv'}")
    print(f"Saved: {args.output_root / 'external_confusion_matrix.png'}")
    print(f"Saved: {args.output_root / 'external_prediction_report.json'}")
    print(f"External accuracy={acc:.3f}, balanced_accuracy={bacc:.3f}")
    print(f"Missing feature columns filled from train median: {len(missing_cols)}")
    print(f"Extra feature columns ignored: {len(extra_cols)}")


if __name__ == "__main__":
    main()
