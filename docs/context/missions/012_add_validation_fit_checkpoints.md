# Mission 012 — Add validation-selected fit and safe checkpoints

## Objective

Orchestrate a fixed number of train/validation epochs, select the first minimum
validation loss and persist enough state and provenance to restore that exact
training point safely.

- Issue: `#34` — `[TRAINING] Add validation-selected fit and checkpoints`.
- Branch: `training/34-validation-fit-checkpoints`.

## Contract

- `FitConfig` requires a positive fixed epoch count.
- `CheckpointProvenance` requires dataset version, cohort, preprocessing
  SHA-256, model name, seed and Git commit.
- `fit` requires loaders whose datasets declare `train` and `validation`; any
  other role, especially `test`, is rejected before training.
- Each epoch calls the existing train and loss-evaluation boundaries.
- Strict `<` selection keeps the first minimum validation loss on ties.
- The best model and optimizer are restored before return.
- The final checkpoint holds best state plus the complete immutable history.
- Saving uses a same-directory temporary file and atomic `os.replace`.
- Loading uses `torch.load(..., weights_only=True)`, validates schema and
  provenance, then restores state on the requested device.

## Leakage constraints

- Selection uses validation loss only.
- The test split is rejected as a selection loader.
- Tests use synthetic data only.
- No real PTB-XL training, metrics, thresholds or final-test evaluation.

## Acceptance checklist

- [x] Wrong train/validation roles and test selection are rejected.
- [x] Full finite histories and sample/batch counts are returned.
- [x] First minimum validation loss wins deterministically.
- [x] Best model and optimizer state are restored.
- [x] Checkpoint save is atomic and load is weights-only.
- [x] Schema and expected provenance are validated before state restoration.
- [x] Synthetic checkpoint round-trip reproduces logits exactly.
- [x] README, guide, decisions, status and autonomous log are updated.
- [ ] Full local checks pass; GitHub checks are pending.

## Out of scope

Real PTB-XL training, AUROC/AUPRC, thresholds, early stopping, scheduler,
experiment comparison, final-test evaluation and inference.

## Local validation evidence

- 136 tests passed.
- Ruff lint and format checks passed.
- Source distribution and wheel built successfully.
