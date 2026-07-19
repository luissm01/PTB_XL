# PTB-XL ML System

Production-oriented machine learning project for multilabel classification of
12-lead ECG signals using PTB-XL.

## Status

Repository bootstrap and official PTB-XL v1.0.3 metadata validation are
implemented. ECG signal loading and models have not been implemented yet.

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

This project is experimental and is not intended for clinical use.
