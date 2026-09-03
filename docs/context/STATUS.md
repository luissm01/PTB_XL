# Project status

Last updated: 2026-09-03

## Current mission

Mission 015 — select and freeze per-class operating thresholds using validation
only.

## Current step

Issue `#41` and branch `evaluation/41-validation-thresholds` are active. The
policy is fixed in D026: maximize F1 independently per class on fold 9, choose
the highest threshold on ties, and bind the frozen artifact to its complete
provenance. Fold 10 remains sealed.

## Next actions

1. Implement and test threshold selection, point metrics and artifact loading.
2. Add a clean-tree command that restores the existing checkpoint and evaluates
   validation only.
3. Generate the real frozen artifact, document it and close one PR.

## Operating mode for upcoming work

Use the batched workflow in D023: broader cohesive missions, one issue/branch/PR
including closure documentation, targeted checks during development and one
complete local validation before review. Do not trade away leakage prevention,
reproducibility or final-test isolation for speed.

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

## Previous completed mission

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
