# Project decisions

This file records decisions that should remain stable across tasks. Detailed
task progress belongs in `STATUS.md`; implementation-specific acceptance
criteria belong in the active mission file.

## D001 — Reproducible Python environment

- Use Python 3.11 managed by uv.
- Keep runtime and development dependencies in `pyproject.toml`.
- Commit `uv.lock` and never edit it manually.

Why: local development, clean clones and CI must resolve the same environment.

## D002 — PTB-XL split policy

- Folds 1–8: training.
- Fold 9: validation.
- Fold 10: final test.
- Never use fold 10 for model, preprocessing, calibration or threshold choices.

Why: this follows the official PTB-XL recommendation and preserves an unbiased
final evaluation.

## D003 — Minimum metadata identity

- Require `ecg_id`, `patient_id` and `strat_fold`.
- Reject duplicate `ecg_id` values.

Why: records, patients and their official split assignment must all be
identifiable before downstream processing.

## D004 — Patient leakage policy

- Detect overlap with a separately testable function.
- Fail immediately when validated metadata contains a patient in multiple
  splits.

Why: inspection remains possible, but unsafe metadata cannot silently continue
through the ML pipeline.

## D005 — Metadata output and errors

- Use a nested dictionary for the first metadata summary.
- Use `KeyError` for missing columns.
- Use `ValueError` for nulls, invalid folds, duplicate records and leakage.

Why: these standard structures are simple to test and serialize without adding
custom abstractions prematurely.

## D006 — Durable project context

- Keep the current handoff in `docs/context/STATUS.md`.
- Keep stable choices in this file.
- Keep one scoped contract per mission under `docs/context/missions/`.
- Update context after material decisions or completed workflow stages.

Why: a new developer or Codex session should be able to resume from repository
files without reconstructing decisions from chat history.
