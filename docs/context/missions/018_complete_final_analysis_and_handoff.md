# Mission 018 — Complete post-hoc analysis and final handoff

## Objective

Complete the portfolio layer with reproducible error analysis, proportional
model attribution, useful figures and a requirement-by-requirement final
handoff.

- Issue: `#47` — `[PORTFOLIO] Add post-hoc error analysis, saliency, and final handoff`.
- Branch: `analysis/47-final-error-interpretability`.

## Contract

- Validate the ignored test-prediction artifact against its exact hash and the
  immutable final-test report before analysis.
- Recompute only descriptive decisions from the saved probabilities and frozen
  thresholds; reconcile every per-class error count with the final report.
- Report exact-match and Hamming error plus every observed true-label
  combination and an explicitly supported problematic subset.
- Select one HYP false negative deterministically and verify its CPU score
  against the saved prediction before attribution.
- Explain that record with absolute input-gradient-times-input attribution,
  per-lead fractions and peak temporal windows.
- Render training history, ROC/PR, operating errors and a 12-lead attribution
  figure from real frozen artifacts.
- Write a strict, non-overwriting JSON report and expose one thin CLI.
- Finish README, project guide, development log and final status only after a
  complete repository audit passes.

## Methodological constraints

- This is post-hoc description, never model development.
- Do not train, fit, calibrate, select a checkpoint or alter thresholds.
- Do not repeat full fold-10 inference or evaluation.
- The one selected ECG forward/backward pass exists only to verify and explain
  its already saved prediction.
- Test observations cannot motivate changes to the frozen baseline.
- Attribution reflects local model sensitivity, not validated clinical
  reasoning, causality or diagnostic localization.

## Acceptance checklist

- [x] Pure error-analysis and attribution functions have synthetic tests.
- [x] Source/hash drift and final-report disagreement fail before analysis.
- [x] The real post-hoc report strictly round-trips and its counts reconcile.
- [x] Four useful real-data figures are versioned at reasonable size.
- [x] No heavy data, checkpoint or row predictions are intended for tracking;
  the final audit verifies this together with secrets.
- [x] README, guide, decisions, status and log provide the final handoff.
- [x] The local gate passed: 216 tests, Ruff lint, format and package build.
- [x] Pull request `#48` is the authoritative remote merge gate; GitHub records
  its result and the branch may merge only when it passes.
- [x] The repository passes the final Definition of Done audit.

## Real post-hoc evidence

- Attributed clean commit: `7856d2e9e023ad37b2130d6a875e484b41eb2b09`.
- Prediction artifact: 2.158 test rows with the immutable fingerprint
  `578c8289e0c863b18603c71e2321c7e355e13686c0eb19547b51fe718c09a5d2`.
- Exact matches: 1.245 (`0.5769230769`).
- Label errors: 1.395/10.790; Hamming loss `0.1292863763`.
- Selected explanation: HYP false negative `ecg_id=1556`, real `MI+HYP`,
  predicted `NORM`; maximum CPU/GPU score difference `0.0001226366` and no
  decision difference.
- Largest attribution shares: V5 `0.160897`, V1 `0.116767`, V4 `0.114902`.
- Report SHA-256:
  `fd7f17f8168b6f5057f5fd80e670563c51990d57afafde6cbdc15f198c55a086`.
- Four PNG files total approximately 1.42 MB; every hash is recorded in the
  strict report.
