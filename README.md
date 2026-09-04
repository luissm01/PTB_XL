# PTB-XL ML System

Production-oriented machine learning project for multilabel classification of
12-lead ECG signals using PTB-XL.

## Status

Official PTB-XL v1.0.3 metadata, labels, cohort definition and 100 Hz signal
integrity loading are implemented. A framework-independent sample boundary now
composes identities, targets, official splits and signals. Train-only global
standardization is implemented and frozen in a reproducible artifact. A thin
PyTorch Dataset/DataLoader boundary produces channel-first batches without
duplicating data logic. A small 1D-CNN baseline maps those batches to five raw
logits. Reproducible single-epoch train and loss-evaluation functions are also
implemented, together with validation-selected multi-epoch fitting, safe
checkpoints and split-safe multilabel ranking metrics. The first configured
baseline trained on folds 1–8, selected its checkpoint and per-class F1
thresholds using fold 9, and completed its one-time final evaluation on fold
10. That final fold is now closed to further model or threshold decisions. A
reproducible CLI can now apply the exact frozen bundle to a compatible standalone
WFDB ECG without dataset labels or split metadata.

## Project documentation

- [Living project guide](docs/PROJECT_GUIDE.md): an accessible explanation of
  ECG foundations, data, leakage, preprocessing, modeling, evaluation,
  architecture, reproducibility, current evidence and interview questions.
- [Current status](docs/context/STATUS.md): the active handoff and next action.
- [Stable decisions](docs/context/DECISIONS.md): constraints that remain valid
  across missions.

## Requirements

- Linux or WSL2
- `uv`
- Python 3.11 (managed automatically by `uv`)

## Setup

```bash
uv sync
```

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Validate official metadata

Download only `ptbxl_database.csv` from the official PTB-XL v1.0.3 release and
place it at `data/raw/ptbxl_database.csv`. Raw data is intentionally ignored by
Git. Then run:

```bash
uv run --locked python scripts/validate_metadata.py \
  --metadata-path data/raw/ptbxl_database.csv \
  --manifest-path data/ptbxl_metadata_manifest.json \
  --output-path reports/metadata/ptbxl_v1.0.3_summary.json
```

The command verifies the file's official SHA-256 before reading it, validates
the official folds and patient isolation, and regenerates the versioned
deterministic report.

## Build diagnostic superclass labels

After obtaining the official PTB-XL v1.0.3 `scp_statements.csv`, place it at
`data/raw/scp_statements.csv` and run:

```bash
uv run --locked python scripts/build_superclass_labels.py \
  --metadata-path data/raw/ptbxl_database.csv \
  --metadata-manifest-path data/ptbxl_metadata_manifest.json \
  --statements-path data/raw/scp_statements.csv \
  --statements-manifest-path data/ptbxl_scp_statements_manifest.json \
  --labels-output-path data/processed/ptbxl_v1.0.3_superclass_labels.csv \
  --report-output-path reports/labels/ptbxl_v1.0.3_superclass_summary.json
```

This derives the five official diagnostic superclasses without loading ECG
signals. The processed table remains local; its deterministic summary is
versioned for review.

## Build the initial modeling cohort

After regenerating the local superclass-label table, run:

```bash
uv run --locked python scripts/build_modeling_cohort.py \
  --labels-path data/processed/ptbxl_v1.0.3_superclass_labels.csv \
  --cohort-output-path data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv \
  --exclusions-output-path data/processed/ptbxl_v1.0.3_cohort_exclusions.csv \
  --report-output-path reports/cohort/ptbxl_v1.0.3_five_superclass_cohort_summary.json
```

This keeps only records with at least one target for the initial task while
preserving and auditing every excluded record in regenerable local outputs.

## Audit low-resolution ECG signals

Obtain only the official PTB-XL v1.0.3 `records100/` directory and place it
under `data/raw/ptb-xl/1.0.3/`. Verify those files from that directory against
the official manifest already stored at `data/raw/SHA256SUMS.txt`:

```bash
(cd data/raw/ptb-xl/1.0.3 && \
  sha256sum --check ../../SHA256SUMS.txt --ignore-missing --quiet)
```

The waveform files remain ignored by Git. Then run:

```bash
uv run --locked python scripts/audit_lr_signals.py \
  --cohort-path data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv \
  --metadata-path data/raw/ptbxl_database.csv \
  --dataset-root data/raw/ptb-xl/1.0.3 \
  --output-path reports/signals/ptbxl_v1.0.3_lr_signal_audit.json
```

The command resolves each cohort `ecg_id` through the official `filename_lr`,
loads one WFDB record at a time and audits shape, sampling frequency, lead order
and finite numeric values. It performs no normalization, filtering or
resampling.

## Compose a framework-independent sample

The sample boundary validates the complete cohort before associating official
signal basenames, then loads individual signals lazily:

```python
import pandas as pd

from ptbxl.data import build_sample_index, load_sample

cohort = pd.read_csv(
    "data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv"
)
metadata = pd.read_csv(
    "data/raw/ptbxl_database.csv",
    usecols=["ecg_id", "filename_lr"],
)
sample_index = build_sample_index(cohort, metadata)
sample = load_sample(sample_index.iloc[0].to_dict(), "data/raw/ptb-xl/1.0.3")
```

`sample.signal` keeps the canonical NumPy shape `(1000, 12)`.
`sample.targets` has shape `(5,)`, `float32` dtype and the fixed order `NORM`,
`MI`, `STTC`, `CD`, `HYP`. The transient index is not a new persisted data
product, and this boundary performs no preprocessing or PyTorch conversion.

## Fit train-only global standardization

The first preprocessing step fits one global mean and population standard
deviation on official train folds 1–8 only. It follows the statistical rule in
the [PTB-XL benchmarking repository](https://github.com/helme/ecg_ptbxl_benchmarking)
but accumulates moments sequentially and stores deterministic JSON instead of
using a full in-memory tensor, scikit-learn or pickle:

```bash
uv run --locked python scripts/fit_global_standardizer.py \
  --cohort-path data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv \
  --metadata-path data/raw/ptbxl_database.csv \
  --dataset-root data/raw/ptb-xl/1.0.3 \
  --signal-manifest-path data/raw/SHA256SUMS.txt \
  --output-path \
    reports/preprocessing/ptbxl_v1.0.3_train_global_standardizer.json
```

The versioned artifact records 17,084 train ECGs and 205,008,000 values. It is
loaded unchanged for every later split and inference; validation and test never
participate in fitting. This baseline adds no filtering, per-record scaling,
denoising, resampling or augmentation.

## Build channel-first PyTorch batches

Build the complete validated sample index first, then select one split and wrap
it with the same frozen standardizer used by every later stage:

```python
import pandas as pd

from ptbxl.data import PTBXLDataset, build_dataloader, build_sample_index
from ptbxl.preprocessing import load_global_standardizer

cohort = pd.read_csv(
    "data/processed/ptbxl_v1.0.3_five_superclass_cohort.csv"
)
metadata = pd.read_csv(
    "data/raw/ptbxl_database.csv",
    usecols=["ecg_id", "filename_lr"],
)
sample_index = build_sample_index(cohort, metadata)
train_index = sample_index.loc[sample_index["split"] == "train"]
standardizer = load_global_standardizer(
    "reports/preprocessing/ptbxl_v1.0.3_train_global_standardizer.json"
)

dataset = PTBXLDataset(
    train_index,
    "data/raw/ptb-xl/1.0.3",
    standardizer,
    split="train",
)
loader = build_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    seed=2026,
)
batch = next(iter(loader))
```

`batch["signal"]` has shape `(B, 12, 1000)` and `float32` dtype;
`batch["targets"]` has shape `(B, 5)`. Dataset construction is lazy, mixed
splits are rejected, and enabling shuffle requires an explicit seed.

## Run the small 1D-CNN baseline

The first model consumes the batch contract directly:

```python
from ptbxl.models import SmallECGCNN

model = SmallECGCNN()
logits = model(batch["signal"])
```

`logits` has shape `(B, 5)`. The default model has 38,597 trainable parameters
and deliberately contains no sigmoid or softmax: training passes the raw logits
directly to `BCEWithLogitsLoss`.

## Run one training epoch

```python
import torch

from ptbxl.training import resolve_device, seed_random_generators, train_one_epoch

seed_random_generators(2026)
device = resolve_device("auto")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
result = train_one_epoch(
    model,
    loader,
    optimizer,
    torch.nn.BCEWithLogitsLoss(),
    device,
)
```

The result contains sample-weighted mean loss, sample count and batch count.
This API is covered by synthetic tests and used by the configured real baseline.

## Fit and save the best validation checkpoint

`ptbxl.training.fit` runs a fixed number of epochs and saves the first minimum
validation loss. It requires loaders backed by datasets declaring exactly
`train` and `validation`; a test loader is rejected. The checkpoint records
model/optimizer state, complete loss history and dataset, cohort, preprocessing,
seed and Git provenance. It is written atomically and loaded with PyTorch's
weights-only mode. Checkpoints remain ignored by Git.

## Evaluate validation ranking metrics

```python
from ptbxl.evaluation import evaluate_validation

evaluation = evaluate_validation(model, validation_loader, device)
print(evaluation.metrics.macro_auroc)
print(evaluation.metrics.macro_auprc)
```

This boundary accepts only a Dataset explicitly declaring `validation`, keeps
the ordered `ecg_id` values and reports AUROC/AUPRC per class, macro and micro.
AUPRC is non-interpolated average precision. Undefined classes, malformed
predictions and duplicate identities fail explicitly. The configured real
baseline uses this same boundary and records its results below.

## Run the configured baseline experiment

From a clean Git worktree with the previously documented local PTB-XL files:

```bash
uv run --locked python scripts/run_baseline_experiment.py
```

The versioned configuration is
[`configs/baseline_small_cnn_100hz.toml`](configs/baseline_small_cnn_100hz.toml).
The command constructs train and validation loaders only, enforces deterministic
PyTorch algorithms, saves the best checkpoint under ignored `artifacts/` and
writes the attributed validation report under `reports/experiments/`. It refuses
to run from uncommitted code. The test fold is not opened by this command.

The fixed ten-epoch run selected epoch 9 by minimum validation loss. On the
2,146 ECGs of fold 9 it obtained macro AUROC `0.915310`, macro AUPRC `0.785994`,
micro AUROC `0.926058` and micro AUPRC `0.832366`. The complete configuration,
loss history, per-class metrics and provenance are recorded in the
[`baseline_small_cnn_100hz` report](reports/experiments/baseline_small_cnn_100hz.json).
These are internal validation results, not final-test results or evidence of
clinical usefulness.

## Select and freeze validation thresholds

With the baseline checkpoint available locally, run from the clean repository
root:

```bash
uv run --locked python scripts/select_validation_thresholds.py
```

The separate
[`baseline_small_cnn_100hz_thresholds.toml`](configs/baseline_small_cnn_100hz_thresholds.toml)
restores the checkpoint without retraining, reconstructs only the validation
Dataset and maximizes F1 independently for each class. Equal optima choose the
highest threshold and decisions use `probability >= threshold`.

The frozen thresholds are NORM `0.327765`, MI `0.511551`, STTC `0.380263`, CD
`0.387861` and HYP `0.145285`. Validation macro F1 is `0.737768` and micro F1 is
`0.767029`; precision, sensitivity, specificity, confusion counts and complete
provenance are stored in the
[`threshold artifact`](reports/evaluation/baseline_small_cnn_100hz_thresholds.json).
These operating metrics reuse the data that selected the thresholds and are
therefore optimistic.

## Sealed final-test result

The frozen pipeline was evaluated exactly once on the 2,158 ECGs of fold 10
from clean commit `056cdc4`. It obtained macro AUROC `0.908895`, macro AUPRC
`0.785850`, micro AUROC `0.922858`, micro AUPRC `0.827715`, macro F1 `0.725377`
and micro F1 `0.756842`. The complete per-class results and provenance are in
the [`final-test report`](reports/evaluation/baseline_small_cnn_100hz_final_test.json).

The command in `scripts/evaluate_final_test.py` is retained for auditability,
but its outputs already exist and its one-time guard forbids another execution.
These results may support descriptive error analysis only; they must not be
used to retune or replace the frozen pipeline.

## Predict a compatible ECG

With the ignored baseline checkpoint available locally, pass a WFDB basename
without `.hea` or `.dat` and choose a new JSON output path:

```bash
uv run --locked python scripts/predict_ecg.py \
  path/to/record_basename \
  --record-id example-001 \
  --output-path reports/inference/example-001.json
```

The record must contain exactly 1,000 samples at 100 Hz and the 12 canonical
leads in order. The command verifies the exact configuration, report,
checkpoint, train-fitted standardizer and validation-selected thresholds before
loading the ECG. It then applies frozen preprocessing, one model forward pass,
sigmoid and the five frozen decision thresholds.

The non-overwriting output records input fingerprint, five ordered scores and
decisions, all artifact hashes and runtime versions. It needs neither PTB-XL
metadata nor labels. A verified train-record smoke output is available in the
[`inference example`](reports/inference/baseline_small_cnn_100hz_train_example.json).
Scores are experimental, are not clinically calibrated probabilities and do
not constitute diagnoses.

This project is experimental and is not intended for clinical use.
