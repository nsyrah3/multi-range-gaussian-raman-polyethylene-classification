#!/usr/bin/env python3
"""Run ablation experiments for the Gaussian-feature representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split


WINDOWS = ("w1", "w2", "w3")
FIELDS = ("amp", "mu", "sigma", "fwhm", "area")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_MATRIX = REPO_ROOT / "outputs" / "program_4_random_forest" / "rf_feature_matrix.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ablation_study"


def load_feature_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    required = {"sample_id", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    feature_cols = [c for c in df.columns if c not in {"sample_id", "class"}]
    return df.copy(), feature_cols


def select_columns(feature_cols: list[str], windows: tuple[str, ...] | None, fields: tuple[str, ...] | None) -> list[str]:
    cols = feature_cols
    if windows is not None:
        cols = [c for c in cols if any(c.startswith(f"{w}_") for w in windows)]
    if fields is not None:
        cols = [c for c in cols if any(c.endswith(f"_{field}") for field in fields)]
    return cols


def build_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )


def evaluate_configuration(
    *,
    name: str,
    feature_subset_label: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    random_state: int,
) -> dict:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["class"].astype(str).to_numpy()

    model = build_rf(random_state)
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])

    hold_acc = float(accuracy_score(y[test_idx], y_pred))
    hold_bacc = float(balanced_accuracy_score(y[test_idx], y_pred))
    hold_f1 = float(f1_score(y[test_idx], y_pred, average="macro"))
    hold_cm = confusion_matrix(y[test_idx], y_pred, labels=["HDPE", "LDPE"])

    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )
    cv_scores = cross_validate(
        build_rf(random_state),
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

    return {
        "configuration": name,
        "number_of_ranges": len({c.split("_")[0] for c in feature_cols}),
        "feature_subset": feature_subset_label,
        "n_features": len(feature_cols),
        "holdout_accuracy": hold_acc,
        "holdout_balanced_accuracy": hold_bacc,
        "holdout_macro_f1": hold_f1,
        "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
        "cv_balanced_accuracy_mean": float(np.mean(cv_scores["test_balanced_accuracy"])),
        "cv_balanced_accuracy_std": float(np.std(cv_scores["test_balanced_accuracy"])),
        "cv_macro_f1_mean": float(np.mean(cv_scores["test_f1_macro"])),
        "cv_macro_f1_std": float(np.std(cv_scores["test_f1_macro"])),
        "holdout_confusion_matrix": hold_cm.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Gaussian-feature ablation study.")
    ap.add_argument(
        "--feature-matrix",
        type=Path,
        default=DEFAULT_FEATURE_MATRIX,
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=10)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    df, all_feature_cols = load_feature_matrix(args.feature_matrix)
    y = df["class"].astype(str).to_numpy()
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    configs = [
        ("Full proposed model", ("w1", "w2", "w3"), ("amp", "mu", "sigma", "fwhm", "area")),
        ("w1 only", ("w1",), ("amp", "mu", "sigma", "fwhm", "area")),
        ("w2 only", ("w2",), ("amp", "mu", "sigma", "fwhm", "area")),
        ("w3 only", ("w3",), ("amp", "mu", "sigma", "fwhm", "area")),
        ("w1 + w2", ("w1", "w2"), ("amp", "mu", "sigma", "fwhm", "area")),
        ("w1 + w3", ("w1", "w3"), ("amp", "mu", "sigma", "fwhm", "area")),
        ("w2 + w3", ("w2", "w3"), ("amp", "mu", "sigma", "fwhm", "area")),
        ("Position-only features", ("w1", "w2", "w3"), ("mu",)),
        ("Width-only features", ("w1", "w2", "w3"), ("sigma", "fwhm")),
        ("Intensity-only features", ("w1", "w2", "w3"), ("amp", "area")),
    ]

    rows: list[dict] = []
    for name, windows, fields in configs:
        feature_cols = select_columns(all_feature_cols, windows, fields)
        if not feature_cols:
            raise ValueError(f"No columns selected for {name}")
        subset_label = ", ".join(fields)
        rows.append(
            evaluate_configuration(
                name=name,
                feature_subset_label=subset_label,
                df=df,
                feature_cols=feature_cols,
                train_idx=train_idx,
                test_idx=test_idx,
                cv_splits=args.cv_splits,
                cv_repeats=args.cv_repeats,
                random_state=args.random_state,
            )
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_root / "ablation_metrics.csv", index=False)

    report = {
        "feature_matrix": str(args.feature_matrix),
        "output_root": str(args.output_root),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "experiments": rows,
    }
    (args.output_root / "ablation_metrics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {args.output_root / 'ablation_metrics.csv'}")
    print(f"Saved: {args.output_root / 'ablation_metrics_report.json'}")


if __name__ == "__main__":
    main()
