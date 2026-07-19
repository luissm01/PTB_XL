# Project status

Last updated: 2026-07-19

## Current mission

Validate PTB-XL metadata and official splits without loading ECG signals.

- Issue: `#3` — `[DATA] Validate PTB-XL metadata and official splits`
- Branch: `data/3-metadata-official-splits`
- Mission contract: `docs/context/missions/001_metadata_official_splits.md`

## Why this is next

Patient-level leakage would invalidate every later experiment. The split boundary
must therefore be validated before signal loading, label construction or model
training.

## Current step

Publish the approved mission branch and validate its pull request.

## Agreed contract

- Required columns: `ecg_id`, `patient_id`, `strat_fold`.
- `ecg_id` must be unique.
- Critical values cannot be null.
- Valid folds are integers from 1 through 10.
- Folds 1–8 map to `train`, fold 9 to `validation`, fold 10 to `test`.
- Patient overlap across splits is detected and causes an immediate error.
- A separate overlap function preserves inspectability.
- The summary is a serializable nested dictionary.
- Missing columns use `KeyError`; invalid values and leakage use `ValueError`.

## Next actions

1. Commit and publish the mission branch.
2. Open a pull request with `Closes #3`.
3. Verify GitHub Actions before deciding whether to merge.

## Completed foundation

- Reproducible Python 3.11 environment managed by uv.
- Installable `src/` package with a locked environment.
- Pytest and Ruff configured.
- Clean-clone reconstruction verified.
- GitHub Actions quality workflow green on `main`.
- Issue/branch/PR workflow established.

## Current repository health

- `main` was clean and synchronized before branch creation.
- Test suite: 17 passed.
- Ruff lint and format checks: passed.
- No PTB-XL data or signal files are present in the repository.
