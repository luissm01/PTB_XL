# Mission 007 — Fit train-only global signal standardization

## Objective

Implement and fit the first minimal signal preprocessing boundary: one global
mean and population standard deviation learned exclusively from official train
folds 1–8, frozen for every later split and reproducible inference.

- Issue: `#19` — `[PREPROCESSING] Fit reproducible train-only global
  standardization`.
- Branch: `preprocessing/19-train-only-global-standardization`.
- Pull request: `#20` — `Preprocessing: fit train-only global standardizer`.
- Dataset: PTB-XL v1.0.3, five-superclass cohort, 100 Hz signals.

## Evidence behind the decision

PTB-XL already distributes technically processed, fixed-duration 100 Hz signals.
The public benchmarking code associated with the PTB-XL deep-learning paper
fits one `StandardScaler` on the flattened training signals, saves it and
applies it unchanged to train, validation and test.

This mission adopts the same statistical rule but not its memory-heavy or
library-specific implementation: moments are accumulated sequentially in
`float64`, the artifact is deterministic JSON and no scikit-learn dependency is
added.

## Methodological decision

- Fit one shared scalar mean and population standard deviation across every
  sample and lead value in train.
- Apply `(signal - mean) / standard_deviation` unchanged to all later data.
- Return `float32` transformed arrays with canonical shape `(1000, 12)`.
- Preserve relative amplitudes across records and leads through one shared
  affine transform.
- Do not normalize independently per record or per lead.
- Do not filter, detrend, denoise, resample or augment signals.

Per-record standardization was rejected because it removes between-record
amplitude information. Per-lead standardization was rejected for the first
baseline because it changes relative scale between leads. No normalization was
considered but a frozen global scaler gives an explicit, reproducible numeric
input range and matches the established benchmark policy.

## Minimal API contract

- `fit_global_standardizer(samples)` consumes a stream of `ECGSample` values,
  requires every sample to be `train` and returns frozen statistics.
- `GlobalStandardizer.transform(signal, lead_names)` validates and transforms
  one signal without changing shape.
- `save_global_standardizer(...)` writes a deterministic, versioned JSON
  artifact containing statistics, counts, lead order and source hashes.
- `load_global_standardizer(path)` validates and reconstructs the transformer.
- `scripts/fit_global_standardizer.py` builds the full validated sample index,
  selects train only and fits the artifact lazily.

## Required behavior

- Reject an empty fit stream and any validation/test sample.
- Validate every signal with the established 100 Hz signal contract.
- Require one consistent lead order across fit and transform.
- Accumulate count, mean and second central moment sequentially in `float64`.
- Use population standard deviation (`ddof=0`), matching `StandardScaler`.
- Reject non-finite values, invalid shapes and zero/non-finite variance.
- Store the exact number of training records and scalar values observed.
- Preserve deterministic ordering and JSON bytes for identical inputs.
- Never persist transformed ECG arrays in this mission.

## Versioned output

```text
reports/preprocessing/ptbxl_v1.0.3_train_global_standardizer.json
```

The artifact is both audit evidence and the small frozen transformer used by
later training and inference. It contains no patient-level or waveform values.

## Acceptance checklist

- [x] Streaming fit and frozen transform implemented independently of PyTorch.
- [x] Fit refuses any non-train sample before producing an artifact.
- [x] Population mean and standard deviation match direct NumPy calculations.
- [x] Shape, lead order, finite values and positive variance are enforced.
- [x] Transform preserves shape, returns `float32` and uses frozen parameters.
- [x] JSON save/load and byte determinism covered by synthetic tests.
- [x] Reproducible CLI fits exactly the expected 17,084 training ECGs.
- [x] Two complete real-data fits produce identical artifact bytes.
- [x] Versioned artifact records source hashes, counts and learned parameters.
- [x] Full Pytest, Ruff and package-build checks pass: 86 tests.
- [x] Pull request `#20`, GitGuardian and post-merge CI pass.

## Real-data evidence

- Fitted split: train only (official folds 1–8).
- Training records observed: 17,084.
- Scalar signal values observed: 205,008,000.
- Lead order: `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`.
- Global mean: `-0.0008252533901116082`.
- Population standard deviation: `0.23222258117564917`.
- Artifact SHA-256 after both complete independent runs:
  `f791aeb9795c669a54a391d979f69806ccdca19f05128a9fde8f408ec36090bc`.
- No validation or test waveform was loaded by the fit iterator.
- The artifact reloads successfully through the public API.

## Leakage constraints

- Build and validate the complete sample index before selecting train rows.
- The fit API fails if any row has split `validation` or `test`.
- Validation and test amplitudes are never summarized or inspected here.
- The frozen train artifact is the only transformer allowed downstream.
- No decision may be changed after observing transformed validation/test data in
  this mission.

## Out of scope

Filtering, baseline-wander removal, denoising, resampling, augmentation,
per-record or per-lead scaling, PyTorch, DataLoader behavior, sampling,
class weights, models, losses, metrics, thresholds and checkpointing.

## Closure

- Issue `#19`: closed.
- Pull request `#20`: squash-merged.
- Merge commit: `12b945a`.
- Python quality and GitGuardian on the PR: passed.
- Post-merge Quality workflow on `main`: passed.
- Local and remote implementation branches: removed.
