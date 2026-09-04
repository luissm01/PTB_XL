# Project status

Last updated: 2026-09-04

## Current mission

Mission 017 — close reproducible single-record ECG inference.

## Current step

Issue `#45` and branch `inference/45-reproducible-inference` are active. Commit
`be0d297` implements the complete frozen-bundle pipeline and passed 198 tests,
Ruff, format and build. A real CPU smoke on train `ecg_id=1` succeeded and its
strict report was reloaded. Fold 10 remained closed.

## Next actions

1. Commit the smoke report and final documentation.
2. Open the mission's single PR and use its checks as merge gate.
3. Merge, close issue `#45`, prune the branch and assess remaining portfolio
   work without reopening fold 10.

## Mission 017 evidence

- Standalone WFDB inference needs no metadata, cohort, labels or split.
- Exact hashes bind config, baseline report, checkpoint, standardizer and
  thresholds before signal inference.
- CPU smoke record `ptbxl-train-ecg-1` predicted only `NORM` positive.
- Signal fingerprint: `9c0e90bb6b6b7f1d929aa60957a80a11c14be76d10cb5c9a7b9237be4b2c5467`.
- Report SHA-256: `59e546296a10b4edc46459d2c5f83a5562a10b9a4a8d07fb7bf7a990fc5b4919`.
- No train/validation/test target or fold was used by the inference API.

## Mission 016 evidence

- Final ranking macro/micro AUROC: `0.9088946765` / `0.9228580874`.
- Final ranking macro/micro AUPRC: `0.7858496654` / `0.8277150848`.
- Frozen-threshold macro/micro F1: `0.7253766434` / `0.7568415548`.
- Prediction fingerprint: `578c8289e0c863b18603c71e2321c7e355e13686c0eb19547b51fe718c09a5d2`.
- Prediction artifact SHA-256: `07840774d81768cd7c97ea0a1ebe2b8498dda9f6f99361aaebb7a44e4415245f`.
- Aggregate report SHA-256: `ac5fa3510b8f5776068f279a878da5fb5c26b459381599f37bcb9e99b9d63cfb`.
- No training, fitting, threshold selection or repeated test inference occurred.

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
