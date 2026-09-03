# Mission 015 — Select and freeze validation thresholds

## Objective

Turn the restored baseline probabilities into a fully attributed operating
point selected exclusively on validation, ready for later final-test evaluation
and reproducible inference.

- Issue: `#41` — `[EVALUATION] Select and freeze validation thresholds`.
- Branch: `evaluation/41-validation-thresholds`.

## Contract

- Select one threshold per target superclass by maximizing its validation F1.
- Evaluate decisions with the rule `probability >= threshold`.
- Resolve equal best F1 values by choosing the highest threshold.
- Report TP, TN, FP, FN, precision, sensitivity/recall, specificity and F1 per
  class, plus unweighted macro and flattened micro summaries.
- Persist a deterministic JSON artifact with thresholds, validation metrics and
  hashes for the checkpoint, baseline configuration/report, preprocessing and
  canonical validation predictions.
- Load the frozen thresholds through a strict schema and optionally require the
  expected checkpoint hash.
- Run from a separate clean-tree CLI without retraining or modifying the frozen
  baseline experiment.

## Leakage constraints

- Prediction collection must accept validation only.
- No fold-10 Dataset, signal, prediction, score or summary may be produced or
  used for any choice; full-cohort structural validation remains unchanged.
- Thresholds may not influence the already selected checkpoint or baseline
  configuration.
- No architecture, preprocessing, loss, sampling or hyperparameter comparison.
- Final-test evaluation remains a separate, explicitly documented mission.

## Acceptance checklist

- [x] Threshold and operating-metric contracts reject malformed input.
- [x] Per-class F1 selection and highest-threshold tie behavior are tested.
- [x] Confusion counts and per-class/macro/micro metrics match known examples.
- [x] The frozen artifact is deterministic, strict and bound to its checkpoint.
- [x] A synthetic end-to-end selection uses validation without retraining.
- [x] The real artifact processes exactly 2,146 fold-9 ECGs.
- [x] No test signal or result is accessed.
- [x] README, guide, decisions, status and log are updated.
- [ ] One final local quality gate and one PR gate pass.

## Real-selection evidence

- Attributed implementation commit: `d69e0571b8e9d67b1bcfb03322375fc7f40df95c`.
- Validation records: 2,146 from fold 9.
- Thresholds: NORM `0.3277647495`, MI `0.5115507841`, STTC `0.3802629411`,
  CD `0.3878610730`, HYP `0.1452849060`.
- Validation macro precision/sensitivity/specificity/F1:
  `0.7163893040` / `0.7693432304` / `0.8950875475` / `0.7377683835`.
- Validation micro precision/sensitivity/specificity/F1:
  `0.7381347494` / `0.7982770998` / `0.9006797583` / `0.7670287981`.
- Artifact SHA-256:
  `f613843e9805f6087964f9ef0a963691bc5f61954ad68d1471efb067bf83b1ce`.
- Validation prediction fingerprint:
  `755f083db3a064a3bdf97ac33a478802c38878e4c622dc0b6bf075de7f3319a9`.
- Strict reload succeeded against checkpoint SHA-256
  `26d058b75dd486e0d81b75fea3cce44fabf6f7c625c57da79640caee6b5c9a11`.
- The operating metrics reuse threshold-selection data and are explicitly
  labeled optimistic; they are not final-test estimates.

## Out of scope

Final-test evaluation, model comparison, retraining, calibration of probability
values, uncertainty intervals, plots, interpretability, error analysis and
inference CLI.

## Local quality evidence

- 176 tests passed.
- Ruff lint passed.
- Ruff format check passed for all 47 Python files.
- Source distribution and wheel built successfully.
