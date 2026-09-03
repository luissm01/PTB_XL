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
checkpoints and split-safe multilabel ranking metrics. Real model training and
performance results are still pending.

## Project documentation

- [Living project guide](docs/PROJECT_GUIDE.md): an accessible explanation of
  ECG foundations, data, leakage, preprocessing, planned modeling, evaluation,
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
and deliberately contains no sigmoid or softmax: future training will pass the
raw logits directly to `BCEWithLogitsLoss`. This is an architecture contract,
not a trained model or a performance result.

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
This API is tested, but no real PTB-XL training result exists yet.

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
predictions and duplicate identities fail explicitly. This API is tested only
with synthetic data so far; the repository contains no claimed model score.

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

This project is experimental and is not intended for clinical use.
