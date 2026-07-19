# Mission 004 — Define the five-superclass modeling cohort

## Objective

Define and audit the initial modeling population for multilabel classification
of `NORM`, `MI`, `STTC`, `CD` and `HYP`, without loading ECG signals.

- Issue: `#10` — `[DATA] Define and audit the five-superclass modeling cohort`.
- Branch: `data/10-define-five-superclass-cohort`.
- Source labels: the ignored deterministic output of Mission 003.

## Methodological decision

A record belongs to this task if and only if it has at least one target label:

```text
sum(NORM, MI, STTC, CD, HYP) >= 1
```

This is not a learned threshold or frequency rule. It is the logical definition
of membership in a task whose only possible targets are those five labels.
Keeping an all-zero row in the modeling cohort would create an implicit
"none-of-the-above" target that the task does not define.

The 411 all-zero records remain unchanged in the master labels table. They are
excluded only from the derived cohort and receive the explicit reason
`no_target_superclass`.

## Required behavior

- Require every identity, fold, split and target column.
- Reject null or duplicate ECG IDs, missing/invalid splits and non-binary labels.
- Preserve identity, folds, splits and label values byte-for-value conceptually.
- Include single-label and multilabel records.
- Partition every input ECG exactly once into included or excluded.
- Expose exclusions through a separate reusable function.
- Reject a cohort that does not exactly implement the inclusion rule.
- Reject any patient overlap between inherited splits after filtering.
- Never recalculate labels or splits.

## Outputs

Ignored, deterministic local tables:

```text
data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv
data/processed/ptbxl_v1.0.3_cohort_exclusions.csv
```

Versioned deterministic evidence:

```text
reports/cohort/ptbxl_v1.0.3_five_superclass_cohort_summary.json
```

The report records input/included/excluded records and patients, explicit
reasons, split sizes, prevalence, cardinality, active-label counts, observed
combinations, source-label hash and patient overlap.

## Real-data evidence

- Input: 21,799 records and 18,869 patients.
- Included: 21,388 records and 18,617 patients.
- Excluded: 411 records involving 329 patients.
- Excluded-only patients: 252; another 77 have both excluded and eligible ECGs.
- Train: 17,084 records and 14,823 patients.
- Validation: 2,146 records and 1,917 patients.
- Test: 2,158 records and 1,877 patients.
- Cross-split patient overlap after filtering: none.
- Every included row has one to four active targets; no five-target row occurs.

Split statistics are descriptive audit evidence. They do not change eligibility,
taxonomy, balance, sampling or any later model decision.

## Reproducible command

```bash
uv run --locked python scripts/build_modeling_cohort.py \
  --labels-path data/processed/ptbxl_v1.0.3_superclass_labels.csv \
  --cohort-output-path data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv \
  --exclusions-output-path data/processed/ptbxl_v1.0.3_cohort_exclusions.csv \
  --report-output-path reports/cohort/ptbxl_v1.0.3_five_superclass_cohort_summary.json
```

## Acceptance checklist

- [x] Explicit task-membership rule documented.
- [x] Master label table remains intact.
- [x] Every row classified exactly once.
- [x] Exclusion IDs and reasons are locally auditable.
- [x] Identity, fold, split and target values preserved.
- [x] Invalid inputs fail clearly.
- [x] Real counts match 21,799 / 21,388 / 411.
- [x] Split statistics and patient separation verified.
- [x] Processed tables remain ignored.
- [x] No signal file is loaded.
- [x] Repeated generation produces identical bytes.
- [x] Full Pytest and Ruff suite passes.
- [x] Pull request and post-merge main CI pass.

## Leakage constraints

- Eligibility does not use class frequencies or model performance.
- Rare label combinations remain included and are only reported.
- Official splits are inherited and never rebalanced.
- Validation and test statistics remain descriptive only.
- No future preprocessing, sampling, weighting or threshold rule is fitted here.

## Out of scope

WFDB, signal paths, signal processing, sampling frequency, PyTorch, weights,
oversampling, thresholds, metrics, models and notebooks.
