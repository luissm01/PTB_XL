# PTB-XL ML System

Production-oriented machine learning project for multilabel classification of
12-lead ECG signals using PTB-XL.

## Status

Initial project setup. Data pipelines and models have not been implemented yet.

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

This project is experimental and is not intended for clinical use.
