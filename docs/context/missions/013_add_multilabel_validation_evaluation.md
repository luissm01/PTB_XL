# Mission 013 — Add split-safe multilabel validation evaluation

## Objective

Build the complete threshold-independent evaluation boundary for the five
PTB-XL diagnostic superclasses, from ordered validation batches to structured
AUROC/AUPRC results.

- Issue: `#37` — `[EVALUATION] Add split-safe multilabel evaluation`.
- Branch: `evaluation/37-multilabel-validation`.

## Contract

- Prediction collection accepts only a loader whose Dataset declares
  `validation`.
- Collection runs the model in evaluation and inference modes, applies sigmoid
  once and preserves ordered unique `ecg_id` values and binary targets.
- Predictions use the fixed target order `NORM`, `MI`, `STTC`, `CD`, `HYP`.
- AUROC and AUPRC are returned per class plus unweighted macro and flattened
  micro averages.
- AUPRC means scikit-learn's non-interpolated average precision, not trapezoidal
  integration of the precision-recall curve.
- Every class must contain at least one positive and one negative target;
  undefined validation metrics fail explicitly instead of becoming `NaN`.
- Result objects are frozen and prediction arrays are defensive read-only
  copies.

## Leakage constraints

- Train and test loaders are rejected before model execution.
- No thresholds or threshold-dependent metrics are selected.
- Checkpoint selection remains based on validation loss as established in D022.
- Tests use synthetic data only; no real PTB-XL prediction is produced.

## Acceptance checklist

- [x] Validation collection preserves sample identity and order.
- [x] Model evaluation creates no gradients and returns finite probabilities.
- [x] Perfect predictions return 1.0 for every ranking metric.
- [x] Tied scores have explicitly tested AUROC and AUPRC behavior.
- [x] Per-class, macro and micro results use the fixed label order.
- [x] Wrong splits, duplicate IDs, malformed shapes/values and degenerate
      classes fail clearly.
- [x] Dependency, README, guide, decisions, status and log are updated.
- [x] Targeted tests and the one final local quality gate pass.
- [ ] One implementation PR passes CI and closes issue `#37`.

## Out of scope

Threshold fitting, F1, sensitivity, specificity, precision at an operating
point, real training/evaluation, experiment comparison, final-test access,
plots, calibration and inference.

## Local validation evidence

- Evaluation tests: 16 passed.
- Complete suite: 152 passed.
- Ruff lint and format checks: passed.
- Source distribution and wheel build: passed.
- `git diff --check`: passed.
