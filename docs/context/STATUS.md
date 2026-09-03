# Project status

Last updated: 2026-09-02

## Current mission

Mission 013 — add split-safe threshold-independent multilabel evaluation for
the five diagnostic superclasses.

## Current step

Issue `#37` and branch `evaluation/37-multilabel-validation` are active. The
validation-only prediction and ranking-metric boundary is implemented and
documented. All 152 tests, Ruff and package build pass; diff review and the
single implementation PR are next.

## Next actions

1. Review the complete Mission 013 diff and publish its single PR.
2. Merge only after PR quality and security checks pass.
3. Next, build the reproducible experiment command and local tracking contract.

## Operating mode for upcoming work

Use the batched workflow in D023: broader cohesive missions, one issue/branch/PR
including closure documentation, targeted checks during development and one
complete local validation before review. Do not trade away leakage prevention,
reproducibility or final-test isolation for speed.

## Last completed mission

Mission 012 — add validation-selected fit and safe checkpoints:

- Issue `#34` closed.
- Pull request `#35` squash-merged as `1bf5443`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 136 tests, Ruff lint, Ruff format and package build.

## Mission 012 evidence

- Fit accepts only explicitly declared train and validation Dataset roles.
- The first minimum validation loss selects the restorable model and optimizer.
- Checkpoints contain complete loss history and source/model/run provenance.
- Atomic save and weights-only load behavior are covered synthetically.
- No real training, metric, threshold or final-test behavior exists.

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
- Fixed multi-epoch fit selects only by validation and safely restores a
  provenance-bound checkpoint.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
