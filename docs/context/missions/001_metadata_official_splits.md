# Mission 001 — PTB-XL metadata and official splits

## Objective

Build a small, testable component that validates `ptbxl_database.csv`, assigns
official splits, detects patient-level leakage and produces reproducible counts.
No ECG signals are loaded in this mission.

## Source contract

The official PTB-XL metadata uses:

- `ecg_id` for the record identifier;
- `patient_id` for the patient identifier;
- `strat_fold` for the recommended ten-fold assignment.

## Required behavior

- Reject missing or null critical fields.
- Reject duplicate `ecg_id` values.
- Accept only integer folds 1–10.
- Map folds 1–8 to `train`, 9 to `validation` and 10 to `test`.
- Detect patient overlap for every pair of splits.
- Fail immediately if overlap exists while reporting the affected split pair.
- Return metadata with a `split` column.
- Build a dictionary containing total, fold, split and overlap counts.
- Never discard invalid rows silently.

## Proposed public API

```python
load_metadata(path) -> pandas.DataFrame
prepare_metadata(metadata) -> pandas.DataFrame
find_patient_overlaps(metadata) -> dict
build_metadata_summary(metadata) -> dict
```

Small column and fold checks may remain private helpers.

## Planned files

```text
src/ptbxl/data/__init__.py
src/ptbxl/data/metadata.py
tests/data/test_metadata.py
```

## Required synthetic tests

- [x] Folds 1 and 8 map to `train`, 9 to `validation`, 10 to `test`.
- [x] Folds 0 and 11 are rejected.
- [x] Null and non-interpretable folds are rejected.
- [x] Missing required columns are reported clearly.
- [x] Null critical identifiers are rejected.
- [x] Duplicate `ecg_id` values are rejected.
- [x] Patient overlap is detected with the affected split pair and identifiers.
- [x] Preparing metadata fails immediately on leakage.
- [x] Total, fold and split summary counts are exact.

## Implementation stages

- [x] Repository and official column inspection.
- [x] Contract and policy decisions approved.
- [x] Synthetic contract tests reviewed.
- [x] `pandas` added as a production dependency.
- [x] Minimum implementation completed.
- [x] Local quality checks passed.
- [x] Complete diff reviewed and approved.
- [x] Pull request validated.

## Out of scope

- Downloading the complete dataset.
- WFDB or ECG signal loading.
- Diagnostic labels and `scp_codes`.
- Preprocessing, models, training or evaluation.
- Notebooks, MLflow, APIs and Docker.

## Completion criteria

The API and synthetic tests cover every required behavior, Pytest and Ruff pass,
and the code does not load signals or use fold 10 for any decision.

## Closure

- Issue `#3`: closed.
- Pull request `#4`: squash-merged.
- Merge commit: `f9577e5`.
- GitHub Actions on `main`: passed.
