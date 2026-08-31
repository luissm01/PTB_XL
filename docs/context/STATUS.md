# Project status

Last updated: 2026-08-31

## Current mission

No active implementation mission. Mission 006, composing the
framework-independent ECG sample contract, is complete.

## Current step

The implementation is squash-merged and synchronized on `main`; its PR checks
and post-merge Quality workflow passed. Close the durable mission context before
opening train-governed preprocessing.

## Next actions

1. Define a minimal preprocessing policy with an explicit train-only fit
   boundary for every learned statistic.
2. Prefer the smallest reversible baseline; do not introduce filters,
   augmentation or framework behavior without experimental justification.
3. Create the next issue, mission contract and implementation branch.

## Last completed mission

Mission 006 — build the framework-independent sample contract:

- Issue `#16` closed.
- Pull request `#17` squash-merged as `d494f54`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 73 tests, Ruff lint and Ruff format.
- The source distribution and wheel built successfully.

## Mission 006 evidence

- The transient sample index contains all 21,388 cohort rows in unchanged order.
- `ecg_id` associates one-to-one with the unique official `filename_lr`.
- Every sample retains patient, official fold/split and the fixed five-target
  order `NORM`, `MI`, `STTC`, `CD`, `HYP`.
- Signals load lazily in canonical NumPy shape `(1000, 12)` and targets use
  shape `(5,)` with `float32` dtype.
- Structural smoke checks loaded one train, validation and test sample at 100 Hz
  with the established 12-lead order.
- Full-cohort validation occurs before a consumer can select a split.
- No filtering, normalization, learned statistic, PyTorch or model decision
  entered the mission.

## Stable repository foundation

- Reproducible Python 3.11 environment managed by uv.
- PTB-XL v1.0.3 identity, metadata, folds and patient isolation verified.
- Official five-superclass labels reproducibly constructed and audited.
- Initial modeling cohort explicitly defined without test-driven selection.
- Official 100 Hz signals safely resolved, loaded and audited for the full
  cohort.
- Framework-independent samples compose identity, signal, targets and official
  split without duplicating source logic.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
