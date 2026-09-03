# Mission 016 — Execute the sealed final test once

## Objective

Evaluate the completely frozen baseline once on official fold 10 and preserve
an immutable, attributable final result without turning test into another
development set.

- Issue: `#43` — `[EVALUATION] Execute the sealed final test once`.
- Branch: `evaluation/43-final-test`.

## Frozen inputs before implementation

- Baseline config SHA-256:
  `50b5b68f8497baf27101a92d550588f8817b6c47a4832267c8033d8ccea2d714`.
- Baseline report SHA-256:
  `1c84a1e807c430d0d54eab89f9a47d6e930993091ceefc062b4b8580ad4f91b6`.
- Checkpoint SHA-256:
  `26d058b75dd486e0d81b75fea3cce44fabf6f7c625c57da79640caee6b5c9a11`.
- Train-only standardizer SHA-256:
  `f791aeb9795c669a54a391d979f69806ccdca19f05128a9fde8f408ec36090bc`.
- Threshold artifact SHA-256:
  `f613843e9805f6087964f9ef0a963691bc5f61954ad68d1471efb067bf83b1ce`.
- Frozen NORM/MI/STTC/CD/HYP thresholds: `0.3277647495`, `0.5115507841`,
  `0.3802629411`, `0.3878610730`, `0.1452849060`.

## Contract

- A separate strict TOML and CLI declare the one-time final-test event.
- Before opening any signal, validate configuration, report, checkpoint,
  standardizer and threshold hashes and cross-artifact provenance.
- Refuse execution if the final report already exists; never overwrite it.
- Construct exactly one Dataset with `split="test"` and 2,158 fold-10 ECGs.
- Restore the epoch-9 checkpoint without optimizer steps, fitting, calibration
  or selection.
- Compute per-class, macro and micro AUROC/AUPRC and apply the already frozen
  thresholds to confusion counts, precision, sensitivity, specificity and F1.
- Save an ignored local prediction artifact for reproducible post-hoc error
  analysis and bind its SHA-256/fingerprint in the versioned aggregate report.
- Treat the result as final for this frozen pipeline; no change may be selected
  from it.

## Acceptance checklist

- [x] Test prediction collection rejects train and validation before inference.
- [x] Frozen-input mismatches and pre-existing output fail before test signals.
- [x] A synthetic end-to-end test evaluation runs once and refuses repetition.
- [x] Ranking and operating metrics reuse the existing tested definitions.
- [x] The local prediction artifact round-trips without pickle.
- [ ] The real event processes exactly 2,158 fold-10 ECGs once.
- [ ] The report records complete input/output hashes and final metrics.
- [x] Documentation labels all post-test work descriptive and forbids tuning.
- [ ] One final local quality gate and one PR gate pass.

## Out of scope

Retraining, model comparison, threshold changes, preprocessing changes,
post-test selection, interpretability, error-analysis presentation and
inference CLI.
