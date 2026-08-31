# Mission 006 — Build the framework-independent sample contract

## Objective

Connect one validated modeling-cohort row to its exact official 100 Hz signal,
ordered five-target vector and inherited official split without introducing
PyTorch or preprocessing.

- Issue: `#16` — `[DATA] Build the framework-independent PTB-XL sample
  contract`.
- Branch: `data/16-build-sample-contract`.
- Pull request: `#17` — `Data: compose framework-independent ECG samples`.
- Dataset: PTB-XL v1.0.3 low-resolution signals at 100 Hz.

## Why this boundary exists

The repository separately proves labels, cohort membership and signal integrity,
but downstream code still needs one authoritative composition boundary. This
mission makes that composition explicit and testable before any framework can
transpose, batch or transform samples.

## Minimal API contract

- `build_sample_index(cohort, metadata)` validates the complete cohort and
  transiently associates each `ecg_id` with its unique official `filename_lr`.
- `load_sample(row, dataset_root)` lazily loads and validates exactly one WFDB
  signal and returns an `ECGSample`.
- `ECGSample` exposes identity, patient, fold, split, source basename, signal,
  target vector, sampling frequency and lead names.
- Signal shape remains `(1000, 12)` and target order remains
  `NORM, MI, STTC, CD, HYP`.
- The sample index preserves cohort order and is never persisted as a new
  versioned data product.

## Required behavior

- Require every cohort identity, fold, split and target column.
- Revalidate binary targets, unique identities, official fold-to-split mapping,
  cross-fold consistency and patient separation before association.
- Require a complete one-to-one `ecg_id` association with unique
  `filename_lr` values.
- Preserve cohort values and row order exactly while adding only the transient
  official signal basename.
- Load one sample on demand rather than retaining the full dataset in memory.
- Reuse the existing safe WFDB path resolution and signal validation boundary.
- Return a NumPy target vector with shape `(5,)`, binary values and stable
  `float32` dtype for the later ML framework boundary.
- Fail clearly on incomplete, ambiguous or changed identity/target/split data.

## Acceptance checklist

- [x] Framework-independent sample index and sample value object implemented.
- [x] Exact signal-target identity association covered by synthetic tests.
- [x] Official target order, shapes, values and metadata covered by tests.
- [x] Cohort order and source values remain unchanged.
- [x] Invalid labels and fold/split mismatches fail before signal loading.
- [x] Missing and ambiguous signal associations fail clearly.
- [x] Real train, validation and test boundary smoke checks pass without fitting
      or calculating statistics.
- [x] Full Pytest and Ruff checks pass: 73 tests.
- [x] Source distribution and wheel build successfully with `samples.py`
      included in the wheel.
- [x] Complete diff reviewed; no dependency, raw-data or artifact changes.
- [x] GitHub issue created with the scoped contract.
- [ ] Pull request `#17` and CI pass.

## Real-data structural evidence

- Sample-index rows: 21,388; cohort order preserved exactly.
- Transient columns: identity, official fold/split, five targets and
  `filename_lr` only.
- One train, one validation and one test sample loaded successfully.
- Each observed signal had shape `(1000, 12)`, 100 Hz and the established
  official lead order.
- Each target vector had shape `(5,)`, `float32` dtype and the fixed target
  order.
- No distribution or amplitude statistic was calculated and this check changed
  no preprocessing or modeling decision.

## Leakage constraints

- Validate the full cohort before any split-level consumer can select rows.
- Inherit labels and official splits; never infer or recalculate targets.
- Calculate no amplitude, prevalence or normalization statistic.
- Perform no selection based on validation or test contents.
- Keep fold 10 available only as structurally valid data; no model or
  preprocessing decision enters this mission.

## Out of scope

Normalization, filtering, resampling, augmentation, channel-first conversion,
PyTorch, DataLoader behavior, sampling strategies, models, loss functions,
metrics, thresholding, training and checkpointing.
