# Project status

Last updated: 2026-09-02

## Current mission

Mission 010 — implement a small, configurable 1D-CNN baseline contract from
channel-first ECG batches to five raw logits.

## Current step

Issue `#28` and branch `models/28-small-1d-cnn` are open. The configurable
38,597-parameter model, 22 focused test cases and documentation updates are
implemented. The full 118-test suite, Ruff, format and package build pass; PR
review is next.

## Next actions

1. Review the complete diff and open the implementation PR.
2. Complete GitHub checks and squash-merge.
3. Record Mission 010 closure before training work.

## Last completed mission

Mission 009 — add the thin PyTorch Dataset/DataLoader boundary:

- Issue `#25` closed.
- Pull request `#26` squash-merged as `9e6ab7e`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 96 tests, Ruff lint, Ruff format and package build.

## Mission 009 evidence

- `PTBXLDataset` delegates to `load_sample` and the frozen
  `GlobalStandardizer`, opening no signal during construction.
- Each Dataset accepts exactly one explicit split and produces contiguous
  `float32` signals `(12, 1000)` plus targets `(5,)` and provenance.
- DataLoader batches `(B, 12, 1000)` / `(B, 5)` and requires a seed for shuffle.
- A real smoke check passed for two records per split with one unchanged
  train-fitted standardizer.
- No model, loss, sampling policy, metric or final-test selection was added.

## Stable repository foundation

- Reproducible Python 3.11 environment managed by uv.
- PTB-XL v1.0.3 identity, metadata, folds and patient isolation verified.
- Official five-superclass labels reproducibly constructed and audited.
- Initial modeling cohort explicitly defined without test-driven selection.
- Official 100 Hz signals safely resolved, loaded and audited for the full
  cohort.
- Framework-independent samples compose identity, signal, targets and official
  split without duplicating source logic.
- A deterministic global standardizer is fitted only on train, versioned with
  source provenance and reusable unchanged by later splits and inference.
- A living project guide makes the verified system, future design and interview
  reasoning accessible without overstating unfinished work.
- A thin PyTorch boundary reuses those contracts and produces deterministic,
  channel-first batches without duplicating dataset semantics.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
