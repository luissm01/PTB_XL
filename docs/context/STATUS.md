# Project status

Last updated: 2026-09-03

## Current mission

Mission 016 — execute the fully frozen baseline on fold 10 exactly once.

## Current step

Issue `#43` and branch `evaluation/43-final-test` are active. The test-only
runner, exact-hash seal and one-time output guards pass 187 tests plus lint,
format and build. No fold-10 waveform has yet been opened by a model.

## Next actions

1. Commit the verified implementation so the event has a clean Git identity.
2. Revalidate the five declared hashes and absence of both outputs.
3. Execute fold 10 once, inspect without rerunning, and record the result.

## Operating mode for upcoming work

Use the batched workflow in D023: broader cohesive missions, one issue/branch/PR
including closure documentation, targeted checks during development and one
complete local validation before review. Do not trade away leakage prevention,
reproducibility or final-test isolation for speed.

## Mission 015 evidence

- Threshold selection and evaluation accept the existing validation prediction
  contract and never retrain the model.
- NORM/MI/STTC/CD/HYP thresholds are `0.327765`, `0.511551`, `0.380263`,
  `0.387861`, `0.145285`.
- The artifact records per-class confusion counts and precision, sensitivity,
  specificity and F1 plus macro/micro summaries.
- Its strict loader verifies schema, internal metric consistency and checkpoint
  SHA-256 binding.
- No fold-10 Dataset, signal, prediction or metric was created.

## Previous completed mission

Mission 014 — execute and record the first configured real baseline.

## Mission 014 evidence

- The strict TOML config and runner bind dataset, model, optimizer, seed,
  runtime, Git commit and output identities.
- Deterministic execution includes PyTorch deterministic algorithms and seeded
  DataLoader workers.
- The best checkpoint is epoch 9 with validation loss `0.2949302586`.
- Fold-9 macro AUROC/AUPRC are `0.9153098458` / `0.7859935473`; micro values are
  `0.9260575328` / `0.8323655924`.
- The checkpoint reload and its reported SHA-256/provenance were verified.
- No threshold, test Dataset, test waveform or test metric was created.

## Earlier completed mission

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
