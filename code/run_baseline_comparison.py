#!/usr/bin/env python3
"""Run baseline comparison experiments for manuscript tables.

Produces:
- summary CSV/JSON for multiple baseline experiments
- same-split hold-out metrics and repeated CV metrics

Experiments:
- Proposed Gaussian 3-range + Random Forest
- Raw full-spectrum + Random Forest
- Single-range Gaussian + Random Forest (w1/w2/w3; best one flagged)
- Gaussian 3-range + Logistic Regression
- Gaussian 3-range + SVM
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSES = ("HDPE", "LDPE")
WINDOWS = ("w1", "w2", "w3")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAUSSIAN_FEATURE_MATRIX = REPO_ROOT / "outputs" / "program_4_random_forest" / "rf_feature_matrix.csv"
DEFAULT_CORRECTED_ROOT = REPO_ROOT / "outputs" / "program_1_baseline_fullspectrum" / "corrected"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baseline_comparison"


@dataclass
class DatasetBundle:
    keys: list[tuple[str, str]]
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def read_corrected_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None, names=["x", "y"])
    if df.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns")
    return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def load_full_spectrum_dataset(corrected_root: Path) -> DatasetBundle:
    keys: list[tuple[str, str]] = []
    feats: list[np.ndarray] = []
    labels: list[str] = []
    shift_ref: np.ndarray | None = None

    for cls in CLASSES:
        class_dir = corrected_root / cls
        files = sorted(class_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No corrected spectra found in {class_dir}")
        for path in files:
            shift, intensity = read_corrected_spectrum(path)
            if shift_ref is None:
                shift_ref = shift
            else:
                if len(shift) != len(shift_ref) or np.max(np.abs(shift - shift_ref)) > 1e-6:
                    raise ValueError(f"Inconsistent Raman shift grid: {path}")
            keys.append((cls, path.stem))
            feats.append(intensity)
            labels.append(cls)

    X = np.vstack(feats)
    y = np.array(labels, dtype=object)
    feature_names = [f"shift_{v:.2f}" for v in shift_ref] if shift_ref is not None else []
    return DatasetBundle(keys=keys, X=X, y=y, feature_names=feature_names)


def load_gaussian_dataset(feature_matrix_csv: Path, window: str | None = None) -> DatasetBundle:
    df = pd.read_csv(feature_matrix_csv)
    required = {"sample_id", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {feature_matrix_csv}: {sorted(missing)}")

    feature_cols = [c for c in df.columns if c not in {"sample_id", "class"}]
    if window is not None:
        prefix = f"{window}_"
        feature_cols = [c for c in feature_cols if c.startswith(prefix)]
        if not feature_cols:
            raise ValueError(f"No features found for window {window}")

    sdf = df[["sample_id", "class", *feature_cols]].copy()
    sdf["sample_id"] = sdf["sample_id"].astype(str)
    sdf["class"] = sdf["class"].astype(str)

    keys = list(zip(sdf["class"], sdf["sample_id"], strict=True))
    X = sdf[feature_cols].to_numpy(dtype=float)
    y = sdf["class"].to_numpy(dtype=object)
    return DatasetBundle(keys=keys, X=X, y=y, feature_names=feature_cols)


def align_to_reference(reference_keys: list[tuple[str, str]], candidate: DatasetBundle, name: str) -> DatasetBundle:
    pos = {key: i for i, key in enumerate(candidate.keys)}
    missing = [key for key in reference_keys if key not in pos]
    extra = [key for key in candidate.keys if key not in set(reference_keys)]
    if missing or extra:
        raise ValueError(f"Sample key mismatch for {name}: missing={len(missing)} extra={len(extra)}")
    order = [pos[key] for key in reference_keys]
    return DatasetBundle(
        keys=[candidate.keys[i] for i in order],
        X=candidate.X[order],
        y=candidate.y[order],
        feature_names=candidate.feature_names,
    )


def build_model(model_key: str):
    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        )
    if model_key == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    if model_key == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model key: {model_key}")


def evaluate_experiment(
    *,
    name: str,
    feature_type: str,
    classifier: str,
    dataset: DatasetBundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cv_idx: np.ndarray | None,
    model_key: str,
    cv_splits: int,
    cv_repeats: int,
    random_state: int,
) -> dict:
    model = build_model(model_key)
    model.fit(dataset.X[train_idx], dataset.y[train_idx])
    y_pred = model.predict(dataset.X[test_idx])

    hold_acc = float(accuracy_score(dataset.y[test_idx], y_pred))
    hold_bacc = float(balanced_accuracy_score(dataset.y[test_idx], y_pred))
    hold_f1 = float(f1_score(dataset.y[test_idx], y_pred, average="macro"))
    hold_report = classification_report(dataset.y[test_idx], y_pred, zero_division=0, output_dict=True)
    hold_cm = confusion_matrix(dataset.y[test_idx], y_pred, labels=list(CLASSES))

    if cv_idx is None:
        X_cv = dataset.X
        y_cv = dataset.y
    else:
        X_cv = dataset.X[cv_idx]
        y_cv = dataset.y[cv_idx]

    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )
    cv_scores = cross_validate(
        build_model(model_key),
        X_cv,
        y_cv,
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
        "experiment": name,
        "feature_type": feature_type,
        "classifier": classifier,
        "n_features": int(dataset.X.shape[1]),
        "n_samples": int(dataset.X.shape[0]),
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "n_cv_samples": int(len(y_cv)),
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
        "holdout_macro_precision": float(hold_report["macro avg"]["precision"]),
        "holdout_macro_recall": float(hold_report["macro avg"]["recall"]),
        # individual fold scores — used for Nadeau-Bengio corrected t-test
        "_fold_accuracy": cv_scores["test_accuracy"].tolist(),
        "_fold_balanced_accuracy": cv_scores["test_balanced_accuracy"].tolist(),
        "_fold_f1_macro": cv_scores["test_f1_macro"].tolist(),
    }


def nadeau_bengio_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_samples: int,
    cv_splits: int,
) -> tuple[float, float]:
    """Nadeau-Bengio (2003) corrected paired t-test for repeated CV comparisons.

    Corrects for the positive correlation between CV folds by scaling the
    variance estimate with (1/k + n_test/n_train), where k is the total number
    of folds and n_test/n_train is the test-to-train ratio per fold.

    Returns (t_statistic, p_value), two-tailed.
    """
    diff = scores_a - scores_b
    k = len(diff)
    n_test = n_samples // cv_splits
    n_train = n_samples - n_test
    correction = 1.0 / k + n_test / n_train
    var_diff = float(np.var(diff, ddof=1))
    if var_diff == 0.0:
        return (0.0, 1.0)
    t_stat = float(np.mean(diff)) / np.sqrt(correction * var_diff)
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=k - 1))
    return (t_stat, p_value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline comparison experiments for manuscript tables.")
    ap.add_argument(
        "--gaussian-feature-matrix",
        type=Path,
        default=DEFAULT_GAUSSIAN_FEATURE_MATRIX,
    )
    ap.add_argument(
        "--corrected-root",
        type=Path,
        default=DEFAULT_CORRECTED_ROOT,
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
    ap.add_argument(
        "--cv-on-train-only",
        action="store_true",
        help="If set, repeated CV is run only on the development/train partition rather than on the full dataset.",
    )
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    gaussian_all = load_gaussian_dataset(args.gaussian_feature_matrix)
    full_spectrum = align_to_reference(
        gaussian_all.keys,
        load_full_spectrum_dataset(args.corrected_root),
        "raw full-spectrum baseline",
    )

    idx = np.arange(len(gaussian_all.y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=gaussian_all.y,
        random_state=args.random_state,
    )
    cv_idx = train_idx if args.cv_on_train_only else None

    experiments: list[dict] = []
    experiments.append(
        evaluate_experiment(
            name="Proposed method",
            feature_type="Gaussian features from three Raman ranges",
            classifier="Random Forest",
            dataset=gaussian_all,
            train_idx=train_idx,
            test_idx=test_idx,
            cv_idx=cv_idx,
            model_key="rf",
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
    )
    experiments.append(
        evaluate_experiment(
            name="Raw full-spectrum baseline",
            feature_type="Full Raman intensity after Program 1 preprocessing",
            classifier="Random Forest",
            dataset=full_spectrum,
            train_idx=train_idx,
            test_idx=test_idx,
            cv_idx=cv_idx,
            model_key="rf",
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
    )

    single_range_results: list[dict] = []
    for window in WINDOWS:
        ds = align_to_reference(
            gaussian_all.keys,
            load_gaussian_dataset(args.gaussian_feature_matrix, window=window),
            f"single-range {window}",
        )
        single_range_results.append(
            evaluate_experiment(
                name=f"Single-range Gaussian baseline ({window})",
                feature_type=f"Gaussian features from {window} only",
                classifier="Random Forest",
                dataset=ds,
                train_idx=train_idx,
                test_idx=test_idx,
                cv_idx=cv_idx,
                model_key="rf",
                cv_splits=args.cv_splits,
                cv_repeats=args.cv_repeats,
                random_state=args.random_state,
            )
        )
    experiments.extend(single_range_results)

    experiments.append(
        evaluate_experiment(
            name="Conventional ML baseline (Logistic Regression)",
            feature_type="Gaussian features from three Raman ranges",
            classifier="Logistic Regression",
            dataset=gaussian_all,
            train_idx=train_idx,
            test_idx=test_idx,
            cv_idx=cv_idx,
            model_key="logreg",
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
    )
    experiments.append(
        evaluate_experiment(
            name="Conventional ML baseline (SVM)",
            feature_type="Gaussian features from three Raman ranges",
            classifier="SVM",
            dataset=gaussian_all,
            train_idx=train_idx,
            test_idx=test_idx,
            cv_idx=cv_idx,
            model_key="svm",
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            random_state=args.random_state,
        )
    )

    # --- save per-fold scores ---
    fold_rows = []
    for exp in experiments:
        n_folds = len(exp["_fold_accuracy"])
        for i, (acc, bacc, f1) in enumerate(
            zip(exp["_fold_accuracy"], exp["_fold_balanced_accuracy"], exp["_fold_f1_macro"])
        ):
            fold_rows.append({
                "experiment": exp["experiment"],
                "fold": i,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "f1_macro": f1,
            })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(args.output_root / "baseline_fold_scores.csv", index=False)

    # strip fold scores before saving summary CSVs/JSON
    for exp in experiments:
        exp.pop("_fold_accuracy", None)
        exp.pop("_fold_balanced_accuracy", None)
        exp.pop("_fold_f1_macro", None)

    all_df = pd.DataFrame(experiments)
    all_df.to_csv(args.output_root / "baseline_metrics_all.csv", index=False)

    best_single = max(
        single_range_results,
        key=lambda r: (r["cv_balanced_accuracy_mean"], r["holdout_balanced_accuracy"], r["cv_macro_f1_mean"]),
    )
    curated = []
    for name in (
        "Proposed method",
        "Raw full-spectrum baseline",
        best_single["experiment"],
        "Conventional ML baseline (Logistic Regression)",
        "Conventional ML baseline (SVM)",
    ):
        curated.append(all_df.loc[all_df["experiment"] == name].iloc[0].to_dict())

    curated_df = pd.DataFrame(curated)
    curated_df["note"] = ""
    curated_df.loc[curated_df["experiment"] == best_single["experiment"], "note"] = "Best-performing single-range Gaussian baseline"
    curated_df.to_csv(args.output_root / "table_5_curated_baselines.csv", index=False)

    # --- Nadeau-Bengio corrected t-test comparisons ---
    fold_lookup = {
        row["experiment"]: np.array(
            fold_df.loc[fold_df["experiment"] == row["experiment"], "accuracy"]
        )
        for row in experiments
    }

    reference_name = "Proposed method"
    comparison_targets = [
        "Raw full-spectrum baseline",
        "Conventional ML baseline (Logistic Regression)",
        "Conventional ML baseline (SVM)",
    ]
    stat_rows = []
    for target in comparison_targets:
        if target not in fold_lookup or reference_name not in fold_lookup:
            continue
        t, p = nadeau_bengio_ttest(
            fold_lookup[reference_name],
            fold_lookup[target],
            n_samples=int(len(train_idx) if args.cv_on_train_only else len(gaussian_all.y)),
            cv_splits=args.cv_splits,
        )
        stat_rows.append({
            "comparison": f"{reference_name} vs {target}",
            "mean_diff": float(np.mean(fold_lookup[reference_name] - fold_lookup[target])),
            "t_statistic": round(t, 4),
            "p_value": round(p, 4),
            "significant_at_0.05": p < 0.05,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(args.output_root / "baseline_statistical_comparison.csv", index=False)

    report = {
        "gaussian_feature_matrix": str(args.gaussian_feature_matrix),
        "corrected_root": str(args.corrected_root),
        "test_size": args.test_size,
        "cv_on_train_only": args.cv_on_train_only,
        "random_state": args.random_state,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "best_single_range_baseline": best_single["experiment"],
        "n_samples": int(len(gaussian_all.y)),
        "experiments": experiments,
    }
    (args.output_root / "baseline_metrics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {args.output_root / 'baseline_metrics_all.csv'}")
    print(f"Saved: {args.output_root / 'table_5_curated_baselines.csv'}")
    print(f"Saved: {args.output_root / 'baseline_metrics_report.json'}")
    print(f"Saved: {args.output_root / 'baseline_fold_scores.csv'}")
    print(f"Saved: {args.output_root / 'baseline_statistical_comparison.csv'}")
    print("\n--- Nadeau-Bengio corrected t-test results ---")
    print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
