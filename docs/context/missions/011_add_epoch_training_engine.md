# Mission 011 — Add a reproducible epoch train/evaluate engine

## Objective

Implement the smallest reproducible training boundary around the established
DataLoader and model contracts, without adding multi-epoch orchestration,
checkpoint selection or metrics.

- Issue: `#31` — `[TRAINING] Add a reproducible epoch train/evaluate engine`.
- Branch: `training/31-epoch-engine`.

## Contract

- `seed_random_generators(seed)` resets Python, NumPy, PyTorch and CUDA streams.
- `resolve_device` handles only `auto`, `cpu` and available `cuda` explicitly.
- `train_one_epoch` sets train mode, clears gradients, runs forward/loss/backward
  and takes one optimizer step per batch.
- `evaluate_loss` sets eval mode and uses inference mode without parameter
  updates.
- Both return immutable `EpochResult` with sample-weighted mean loss and counts.
- Mapping batches must contain tensor `signal` and `targets` with equal batch
  sizes; logits and targets must have equal shapes and scalar loss must be finite.

## Leakage constraints

- Synthetic data only; no PTB-XL split is opened.
- No checkpoint or model selection.
- No metrics, thresholds, class weights, sampling or final-test behavior.

## Acceptance checklist

- [x] One validated seed resets all supported random streams.
- [x] CPU/auto/unavailable-CUDA behavior is tested.
- [x] Training updates parameters and returns finite aggregate loss.
- [x] Evaluation updates no parameter and uses sample-weighted loss.
- [x] Empty, malformed and non-finite cases fail clearly.
- [x] Train and eval modes are explicit.
- [x] README, guide, decisions, status and autonomous log are updated.
- [ ] Full tests, Ruff, format, build and GitHub checks pass. All local checks
  pass with 127 tests; GitHub checks are pending the pull request.

## Out of scope

Real training, multi-epoch `fit`, scheduler, checkpointing, early stopping,
metrics, thresholds, experiment tracking, final-test evaluation and inference.

## Local validation evidence

- Pytest: 127 passed, including nine new training-engine tests.
- Ruff lint and format check: passed.
- Source distribution and wheel build: passed.
- `git diff --check`: passed.
