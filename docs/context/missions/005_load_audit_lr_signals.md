# Mission 005 — Load and audit PTB-XL 100 Hz signals

## Objective

Build a safe boundary from a validated cohort row to its official PTB-XL
v1.0.3 `filename_lr` WFDB signal, without introducing preprocessing or model
behavior.

- Issue: `#13` — `[DATA] Load and audit PTB-XL 100 Hz signals`.
- Branch: `data/13-load-audit-lr-signals`.
- Dataset: PTB-XL v1.0.3 low-resolution signals at 100 Hz.

## Why this boundary exists

Labels and patient-safe splits are already proven, but the system cannot train
until each cohort identity is shown to resolve to the correct readable signal.
Starting at 100 Hz validates that association and the complete data path at the
lowest approved storage and compute cost. It does not claim that 100 Hz is
clinically or experimentally superior to 500 Hz.

## Minimal API contract

- `load_signal_for_row(row, dataset_root)` resolves only the row's
  `filename_lr`, reads the WFDB basename and returns a signal record.
- `validate_signal(signal_record, expected_lead_names=None)` validates numeric
  `(samples, leads)` data, 100 Hz, 1,000 samples, 12 leads, finite values and an
  optional previously observed lead order.
- The signal record exposes `signal: numpy.ndarray`,
  `sampling_frequency: float` and `lead_names: tuple[str, ...]`.

The auditor joins cohort identities to official metadata by `ecg_id` with a
one-to-one contract. The cohort itself remains unchanged and does not gain a
signal-path column.

## Required behavior

- Require unique, non-null `ecg_id` and `filename_lr` metadata values.
- Require every cohort ECG to resolve to exactly one metadata row.
- Reject absolute paths, parent traversal and paths outside the dataset root.
- Pass the WFDB basename unchanged; do not invent a file extension.
- Distinguish missing records from unreadable or structurally invalid records.
- Validate shape `(1000, 12)`, frequency 100 Hz, numeric finite values and
  consistent header lead order.
- Preserve the lead order found in real WFDB headers rather than imposing an
  assumed order before inspection.
- Process one ECG at a time and retain only aggregate audit evidence.

## Outputs

- `src/ptbxl/data/signals.py`.
- `scripts/audit_lr_signals.py`.
- Synthetic WFDB tests under `tests/data/`.
- Versioned deterministic report:
  `reports/signals/ptbxl_v1.0.3_lr_signal_audit.json`.
- Ignored real WFDB files under `data/raw/ptb-xl/1.0.3/`.

## Acceptance checklist

- [x] `numpy` and `wfdb` are added through uv as justified direct runtime
  dependencies.
- [x] A real PTB-XL v1.0.3 low-resolution signal loads successfully.
- [x] Signal values, shape, frequency and lead names have explicit contracts.
- [x] Missing, corrupt, malformed and non-finite signals fail clearly.
- [x] Path resolution proves the row's exact `filename_lr` is used.
- [x] Synthetic loader, validation, association and streaming audit tests pass.
- [x] All 21,388 cohort ECGs are inspected without retaining them in memory.
- [x] The deterministic real-data audit report is versioned.
- [x] Labels, folds and splits remain unchanged.
- [x] Raw signals remain ignored by Git.
- [x] Full Pytest, Ruff, GitHub Actions and GitGuardian checks pass.
- [x] No normalization, filtering, resampling, PyTorch or model logic is added.

## Real-data evidence

- Official `records100/` files: 43,598 files / 21,799 WFDB records.
- Every downloaded file passes the official v1.0.3 SHA-256 manifest.
- Cohort audit: 21,388 expected and loaded; zero missing or invalid.
- Observed shape: `(1000, 12)` for every cohort record.
- Observed frequency: 100 Hz for every cohort record.
- Observed lead order: `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5,
  V6` for every cohort record.
- Non-finite values: zero.
- Repeated audit report SHA-256:
  `8c5c5deed8d77eba5381ccb6e93e295b8d2c96a73d75014678f8c1218fc5040f`.
- Source metadata, labels and cohort hashes remain unchanged.

## Leakage constraints

- Calculate no amplitude mean, deviation, percentile or other learned statistic.
- Use validation and test signals only for descriptive integrity checks fixed by
  this contract.
- Do not select, reorder or remove leads based on downstream performance.
- Never replace a missing or invalid signal with zeros or another record.
- Do not exclude records using amplitude or clinical-quality thresholds.

## Out of scope

Normalization, band-pass filtering, resampling, 500 Hz signals, clinical
quality rules, visualization, PyTorch datasets, batching, augmentation, models,
training, metrics and threshold selection.
