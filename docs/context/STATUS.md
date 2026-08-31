# Project status

Last updated: 2026-08-31

## Current mission

Mission 007 — fit train-only global signal standardization.

- Issue: `#19`.
- Branch: `preprocessing/19-train-only-global-standardization`.
- Contract: `docs/context/missions/007_fit_train_only_global_standardizer.md`.

## Current step

Implementation, synthetic tests, two real train-only fits and local quality
review pass. Review and publish the scoped diff for independent CI.

## Next actions

1. Confirm only aggregate train statistics and intended source files are tracked.
2. Commit, push and open the pull request closing issue `#19`.
3. Merge only after Python quality and GitGuardian pass, then close context.

## Mission 007 evidence

- Two complete fits read exactly 17,084 train ECGs / 205,008,000 values each.
- Validation and test waveforms were not opened by the fit iterator.
- Global mean: `-0.0008252533901116082`.
- Population standard deviation: `0.23222258117564917`.
- Repeated artifact SHA-256:
  `f791aeb9795c669a54a391d979f69806ccdca19f05128a9fde8f408ec36090bc`.
- Artifact identity, configuration, lead order, counts and source hashes are
  versioned as deterministic JSON.
- Full local suite passes: 86 tests, Ruff lint, Ruff format and package build.
- No dependency, filter, resampling, augmentation, PyTorch or model behavior was
  added.

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
