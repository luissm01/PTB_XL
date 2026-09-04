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

- [ ] Pure error-analysis and attribution functions have synthetic tests.
- [ ] Source/hash drift and final-report disagreement fail before analysis.
- [ ] The real post-hoc report strictly round-trips and its counts reconcile.
- [ ] Four useful real-data figures are versioned at reasonable size.
- [ ] No heavy data, checkpoint, row predictions or secrets are tracked.
- [ ] README, guide, decisions, status and log provide the final handoff.
- [ ] Full tests, Ruff, format, package build and PR checks pass.
- [ ] The repository passes the final Definition of Done audit.

