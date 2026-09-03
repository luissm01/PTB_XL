# Mission 016 — Execute the sealed final test once

## Objective

Evaluate the completely frozen baseline once on official fold 10 and preserve
an immutable, attributable final result without turning test into another
development set.

- Issue: `#43` — `[EVALUATION] Execute the sealed final test once`.
- Branch: `evaluation/43-final-test`.
- Attributed execution commit: `056cdc4c27029f53cda5c3a6090fea864a9e1cf5`.

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
- [x] The real event processes exactly 2,158 fold-10 ECGs once.
- [x] The report records complete input/output hashes and final metrics.
- [x] Documentation labels all post-test work descriptive and forbids tuning.
- [ ] One final local quality gate and one PR gate pass.

## Out of scope

Retraining, model comparison, threshold changes, preprocessing changes,
post-test selection, interpretability, error-analysis presentation and
inference CLI.

## Final-test evidence

- Records: 2,158 from official fold 10.
- Macro AUROC/AUPRC: `0.9088946765` / `0.7858496654`.
- Micro AUROC/AUPRC: `0.9228580874` / `0.8277150848`.
- Frozen-threshold macro precision/sensitivity/specificity/F1:
  `0.7173560409` / `0.7462298550` / `0.8967218047` / `0.7253766434`.
- Frozen-threshold micro precision/sensitivity/specificity/F1:
  `0.7371816638` / `0.7775787966` / `0.9032258065` / `0.7568415548`.
- Prediction fingerprint:
  `578c8289e0c863b18603c71e2321c7e355e13686c0eb19547b51fe718c09a5d2`.
- Local prediction artifact SHA-256:
  `07840774d81768cd7c97ea0a1ebe2b8498dda9f6f99361aaebb7a44e4415245f`.
- Versioned report SHA-256:
  `ac5fa3510b8f5776068f279a878da5fb5c26b459381599f37bcb9e99b9d63cfb`.
- The saved artifact was reloaded read-only with split `test`, 2,158 unique IDs,
  arrays shaped `(2158, 5)` and the same canonical fingerprint.

## Local quality evidence

- 187 tests passed before the real event.
- Ruff lint and format check passed.
- Source distribution and wheel built successfully.
- The real final-test command ran once and only once.
