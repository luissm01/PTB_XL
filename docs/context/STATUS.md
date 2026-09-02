# Project status

Last updated: 2026-09-02

## Current mission

Mission 012 — add fixed multi-epoch fitting and safe checkpoints selected only
by validation loss.

## Current step

Issue `#34` and branch `training/34-validation-fit-checkpoints` are open. Fit,
atomic weights-only checkpointing and nine synthetic tests are implemented and
passing. Documentation and all local checks are complete; PR review and remote
checks are next.

## Next actions

1. Review and merge the Mission 012 implementation PR after remote checks.
2. Record closure evidence and synchronize `main`.
3. Pause before defining multilabel validation metrics, as requested by the
   owner.

## Last completed mission

Mission 011 — add the reproducible epoch train/evaluate engine:

- Issue `#31` closed.
- Pull request `#32` squash-merged as `79a64f8`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 127 tests, Ruff lint, Ruff format and package build.

## Mission 011 evidence

- One seed resets Python, NumPy, PyTorch and CUDA random streams.
- Train and loss evaluation have separate gradient/mode semantics.
- Epoch loss is weighted by sample count and rejects invalid/non-finite input.
- Synthetic training updates parameters; evaluation changes none.
- No real training, checkpoint selection, metric or final-test behavior exists.

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
- A reproducible epoch engine separates optimization from loss-only evaluation.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
