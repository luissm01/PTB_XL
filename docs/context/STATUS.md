# Project status

Last updated: 2026-08-31

## Current mission

Mission 008 — build `docs/PROJECT_GUIDE.md` as the living, accessible technical
guide for the implemented foundations and planned modeling path.

## Current step

Issue `#22` and branch `docs/22-build-project-guide` are open. The ten-part guide
and repository context updates are drafted and locally verified; pull-request
review and GitHub checks are next.

## Next actions

1. Open the documentation pull request and complete GitHub checks and review.
2. Squash-merge, verify post-merge Quality and record Mission 008 closure.
3. After owner review, introduce a thin PyTorch Dataset/DataLoader adapter that reuses
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
