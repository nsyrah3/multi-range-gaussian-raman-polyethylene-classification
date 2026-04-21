# Multi-Range Gaussian Raman Features for Binary Polyethylene Classification

This repository contains the curated Raman data subset and Python scripts used for preprocessing, Gaussian peak fitting, feature construction, classification, baseline comparison, and ablation analysis for binary HDPE-LDPE classification.

## Repository structure

```text
Repo/
|- README.md
|- requirements.txt
|- .gitignore
|- code/
|  |- program_1_baseline_fullspectrum.py
|  |- program_2_cut_smooth_windows.py
|  |- program_3_fit_gaussian_windows.py
|  |- program_4_random_forest.py
|  |- program_5_external_pipeline.py
|  |- program_6_predict_external.py
|  |- run_ablation_study.py
|  `- run_baseline_comparison.py
|- data/
|  |- HDPE/
|  `- LDPE/
|- external_data/
`- outputs/
```

## Dataset included

- `data/HDPE`: 350 curated handheld Raman spectra
- `data/LDPE`: 350 curated handheld Raman spectra

The `data/` directory contains the fixed 700-spectrum subset used in the main manuscript analyses.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Main pipeline

Run the scripts in this order:

```bash
python code/program_1_baseline_fullspectrum.py
python code/program_2_cut_smooth_windows.py
python code/program_3_fit_gaussian_windows.py
python code/program_4_random_forest.py
```

By default, these scripts read from `data/` and write outputs to `outputs/`.

## Additional analyses

Baseline comparison:

```bash
python code/run_baseline_comparison.py
```

Ablation study:

```bash
python code/run_ablation_study.py
```

External data pipeline:

```bash
python code/program_5_external_pipeline.py
python code/program_6_predict_external.py
```

Place external TXT files inside `external_data/` before running the external pipeline.

## Notes

- The scripts use relative repository paths by default, so the repository can be moved without changing hard-coded locations.
- Generated outputs are written to `outputs/` and are excluded from version control by `.gitignore`.
