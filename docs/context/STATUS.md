# Project status

Last updated: 2026-09-02

## Current mission

No active implementation mission. Mission 009, adding the thin PyTorch
Dataset/DataLoader boundary, is complete.

## Current step

The implementation is merged and synchronized. PR checks and the post-merge
Quality workflow passed; the implementation branch was removed.

## Next actions

1. Define a small 1D-CNN that accepts `(B, 12, 1000)` and returns five logits.
2. Test parameterized input validation, forward/backward behavior and
   compatibility with `BCEWithLogitsLoss` using synthetic tensors.
3. Keep training, metrics, checkpointing and final-test access out of that model
   contract mission.

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
