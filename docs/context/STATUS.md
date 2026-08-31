# Project status

Last updated: 2026-08-31

## Current mission

No active implementation mission. Mission 007, fitting reproducible train-only
global signal standardization, is complete.

## Current step

Development is intentionally paused at the owner's request. The repository is
clean and synchronized after the implementation merge; PR checks and the
post-merge Quality workflow passed.

## Next actions

1. When development resumes, read this status, Mission 007 and D017 before
   opening a new issue.
2. Create `docs/PROJECT_GUIDE.md` as the next focused portfolio/documentation
   mission so the completed data and preprocessing foundations are teachable.
3. Then introduce a thin PyTorch Dataset/DataLoader adapter that reuses
   `load_sample` and the frozen global standardizer without duplicating logic.

## Last completed mission

Mission 007 — fit train-only global signal standardization:

- Issue `#19` closed.
- Pull request `#20` squash-merged as `12b945a`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 86 tests, Ruff lint, Ruff format and package build.

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

## Stable repository foundation

- Reproducible Python 3.11 environment managed by uv.
- PTB-XL v1.0.3 identity, metadata, folds and patient isolation verified.
- Official five-superclass labels reproducibly constructed and audited.
- Initial modeling cohort explicitly defined without test-driven selection.
- Official 100 Hz signals safely resolved, loaded and audited for the full
  cohort.
- Framework-independent samples compose identity, signal, targets and official
  split without duplicating source logic.
- A deterministic global standardizer is fitted only on train, versioned with
  source provenance and reusable unchanged by later splits and inference.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
