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
- Fold 10 must not be loaded, scored, summarized or used for any choice.
- Thresholds may not influence the already selected checkpoint or baseline
  configuration.
- No architecture, preprocessing, loss, sampling or hyperparameter comparison.
- Final-test evaluation remains a separate, explicitly documented mission.

## Acceptance checklist

- [ ] Threshold and operating-metric contracts reject malformed input.
- [ ] Per-class F1 selection and highest-threshold tie behavior are tested.
- [ ] Confusion counts and per-class/macro/micro metrics match known examples.
- [ ] The frozen artifact is deterministic, strict and bound to its checkpoint.
- [ ] A synthetic end-to-end selection uses validation without retraining.
- [ ] The real artifact processes exactly 2,146 fold-9 ECGs.
- [ ] No test signal or result is accessed.
- [ ] README, guide, decisions, status and log are updated.
- [ ] One final local quality gate and one PR gate pass.

## Out of scope

Final-test evaluation, model comparison, retraining, calibration of probability
values, uncertainty intervals, plots, interpretability, error analysis and
inference CLI.
