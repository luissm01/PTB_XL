# Project status

Last updated: 2026-09-02

## Current mission

No active implementation mission. Mission 010, implementing the small 1D-CNN
baseline contract, is complete.

## Current step

The implementation is merged and synchronized. PR checks and the post-merge
Quality workflow passed; the implementation branch was removed.

## Next actions

1. Define reproducible seed/device utilities and a minimal `train_one_epoch`
   boundary using `BCEWithLogitsLoss`.
2. Prove with synthetic data that training produces finite loss and updates
   parameters while evaluation does not update them.
3. Add checkpointing only after the epoch-level train/evaluate contract is
   stable and tested.

## Last completed mission

Mission 010 — implement the small 1D-CNN baseline contract:

- Issue `#28` closed.
- Pull request `#29` squash-merged as `a228b42`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 118 tests, Ruff lint, Ruff format and package build.

## Mission 010 evidence

- `SmallECGCNN` maps `float32 (B, 12, 1000)` to five raw logits.
- The frozen config records channels `32, 64, 128`, kernels `7, 5, 3` and
  dropout `0.2`; the default model has 38,597 trainable parameters.
- Synthetic BCE loss, forward and gradients are finite for every trainable
  parameter; eval output is deterministic.
- The model contains no sigmoid or softmax and opens no real waveform.
- No optimizer, training, metric, threshold or final-test behavior was added.

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
- A small configurable 1D-CNN converts valid batches to five raw logits with a
  tested multilabel loss/gradient contract.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
