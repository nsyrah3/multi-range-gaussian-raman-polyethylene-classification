#!/usr/bin/env python3
"""
Program 4: Random Forest classification from labeled Gaussian features.

Input (default):
- outputs/program_3_gaussian_fit/gaussian_fit_summary.csv
  columns expected:
    class,file,window,peak_idx,amp,mu,sigma,fwhm,area

Output (default root: outputs/program_4_random_forest):
- rf_feature_matrix.csv                 (1 row per sample, wide features)
- rf_train_test_predictions.csv
- rf_feature_importance.csv
- rf_model.joblib
- rf_report.json
- rf_confusion_matrix.png
- rf_feature_importance_top20.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split


FEATURE_FIELDS = ("amp", "mu", "sigma", "fwhm", "area")
WINDOW_ORDER = {"w1": 1, "w2": 2, "w3": 3}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "outputs" / "program_3_gaussian_fit" / "gaussian_fit_summary.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "program_4_random_forest"


def load_labeled_gaussian(path: Path) -> pd.DataFrame:
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


def sort_feature_name(name: str) -> tuple[int, int, int]:
    # Expected format: wX_pY_field
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


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
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

    feat_df = pd.concat(parts, axis=1)
    feat_df = feat_df.reindex(sorted(feat_df.columns, key=sort_feature_name), axis=1)

    na_before = feat_df.isna().sum().astype(int)
    na_info = {k: int(v) for k, v in na_before[na_before > 0].to_dict().items()}
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            med = float(feat_df[c].median()) if feat_df[c].notna().any() else 0.0
            feat_df[c] = feat_df[c].fillna(med)

    return feat_df, na_info


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Random Forest Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_importance_top(imp_df: pd.DataFrame, out_path: Path, top_k: int = 20) -> None:
    top = imp_df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.barh(top["feature"], top["importance"], color="#1f77b4", alpha=0.9)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {min(top_k, len(imp_df))} Feature Importance")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Program 4: Random Forest from Gaussian labeled CSV.")
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input labeled Gaussian CSV.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output folder for Program 4.",
    )
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=10)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    long_df = load_labeled_gaussian(args.input_csv)
    feat_df, na_info = build_feature_matrix(long_df)
    feat_out = args.output_root / "rf_feature_matrix.csv"
    feat_df.reset_index().to_csv(feat_out, index=False)

    X = feat_df.to_numpy(dtype=float)
    y = feat_df.index.get_level_values("class").to_numpy()
    sample_id = feat_df.index.get_level_values("sample_id").to_numpy()
    feature_names = feat_df.columns.tolist()

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(
        idx, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    clf.fit(X[tr_idx], y[tr_idx])
    y_pred = clf.predict(X[te_idx])
    y_prob = clf.predict_proba(X[te_idx])

    labels = sorted(np.unique(y).tolist())
    acc = float(accuracy_score(y[te_idx], y_pred))
    bacc = float(balanced_accuracy_score(y[te_idx], y_pred))
    cm = confusion_matrix(y[te_idx], y_pred, labels=labels)
    report = classification_report(y[te_idx], y_pred, labels=labels, zero_division=0, output_dict=True)

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=args.random_state
    )
    cv_scores = cross_validate(
        clf,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        n_jobs=1,
        return_train_score=False,
    )

    cv_summary = {
        "accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "accuracy_std": float(np.std(cv_scores["test_accuracy"])),
        "balanced_accuracy_mean": float(np.mean(cv_scores["test_balanced_accuracy"])),
        "balanced_accuracy_std": float(np.std(cv_scores["test_balanced_accuracy"])),
        "f1_macro_mean": float(np.mean(cv_scores["test_f1_macro"])),
        "f1_macro_std": float(np.std(cv_scores["test_f1_macro"])),
    }

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_id[te_idx],
            "y_true": y[te_idx],
            "y_pred": y_pred,
            "is_correct": (y[te_idx] == y_pred).astype(int),
        }
    )
    for i, cls in enumerate(clf.classes_):
        pred_df[f"proba_{cls}"] = y_prob[:, i]
    pred_df = pred_df.sort_values("sample_id").reset_index(drop=True)
    pred_df.to_csv(args.output_root / "rf_train_test_predictions.csv", index=False)

    imp_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp_df.to_csv(args.output_root / "rf_feature_importance.csv", index=False)

    model_path = args.output_root / "rf_model.joblib"
    joblib.dump(clf, model_path)

    plot_confusion(cm, labels, args.output_root / "rf_confusion_matrix.png")
    plot_importance_top(imp_df, args.output_root / "rf_feature_importance_top20.png", top_k=20)

    report_json = {
        "input_csv": str(args.input_csv),
        "n_rows_long": int(len(long_df)),
        "n_samples": int(len(feat_df)),
        "n_features": int(len(feature_names)),
        "classes": labels,
        "missing_feature_cells_before_fill": na_info,
        "params": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "cv_splits": args.cv_splits,
            "cv_repeats": args.cv_repeats,
        },
        "holdout": {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "confusion_matrix_labels": labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "cv_repeated_stratified_kfold": cv_summary,
    }
    (args.output_root / "rf_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    print(f"Input: {args.input_csv}")
    print(f"Saved: {feat_out}")
    print(f"Saved: {args.output_root / 'rf_train_test_predictions.csv'}")
    print(f"Saved: {args.output_root / 'rf_feature_importance.csv'}")
    print(f"Saved: {model_path}")
    print(f"Saved: {args.output_root / 'rf_report.json'}")
    print(f"Saved: {args.output_root / 'rf_confusion_matrix.png'}")
    print(f"Saved: {args.output_root / 'rf_feature_importance_top20.png'}")
    print(f"Holdout accuracy={acc:.3f}, balanced_accuracy={bacc:.3f}")
    print(
        "CV accuracy="
        f"{cv_summary['accuracy_mean']:.3f}±{cv_summary['accuracy_std']:.3f}, "
        "CV bacc="
        f"{cv_summary['balanced_accuracy_mean']:.3f}±{cv_summary['balanced_accuracy_std']:.3f}"
    )


if __name__ == "__main__":
    main()
