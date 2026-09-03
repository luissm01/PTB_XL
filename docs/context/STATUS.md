# Project status

Last updated: 2026-09-02

## Current mission

No active implementation mission. Mission 013 delivered the split-safe
threshold-independent multilabel evaluation boundary.

## Current step

Issue `#37` and pull request `#38` carry the implementation, tests and closure
record in one review unit. The local gate passed with 152 tests, Ruff and package
build; GitHub records the authoritative CI and merge state.

## Next actions

1. Build the reproducible experiment command and local tracking contract.
2. Run the first real baseline using train and validation only.
3. Keep threshold selection and final-test access outside that next mission.

## Operating mode for upcoming work

Use the batched workflow in D023: broader cohesive missions, one issue/branch/PR
including closure documentation, targeted checks during development and one
complete local validation before review. Do not trade away leakage prevention,
reproducibility or final-test isolation for speed.

## Last completed mission

Mission 013 — add split-safe multilabel validation evaluation:

- Issue `#37` is closed through pull request `#38`.
- The single PR contains implementation, tests, decisions and closure context.
- Full local suite passed: 152 tests, Ruff lint, Ruff format and package build.
- GitHub is the authoritative source for PR checks and squash-merge identity.

## Mission 013 evidence

- Prediction collection accepts only an explicitly declared validation Dataset.
- Ordered unique ECG identities, binary targets and sigmoid probabilities are
  preserved in immutable result contracts.
- AUROC and non-interpolated average-precision AUPRC are reported per class,
  macro and micro using scikit-learn 1.9.0.
- Undefined classes and malformed inputs fail instead of yielding silent scores.
- No real model score, threshold choice or final-test access exists.

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
- Split-safe evaluation produces strict per-class, macro and micro AUROC/AUPRC
  results without choosing thresholds.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
