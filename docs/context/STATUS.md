# Project status

Last updated: 2026-07-19

## Current mission

Build the five official PTB-XL diagnostic superclass labels without loading ECG
signals.

- Issue: `#7` — `[DATA] Build the five PTB-XL diagnostic superclass labels`.
- Branch: `data/7-build-diagnostic-superclass-labels`.
- Mission contract: `docs/context/missions/003_build_diagnostic_superclass_labels.md`.

## Why this is next

Splits and dataset identity are already validated. The next methodological risk
is silently constructing incorrect targets from `scp_codes`, non-diagnostic
statements or an ad-hoc mapping. This mission fixes and audits target semantics
before defining a modeling cohort or loading signals.

## Current step

Implementation, real-data reproducibility and the full local quality suite pass.
Review the complete diff and publish the branch for CI.

## Next actions

1. Confirm raw and processed inputs remain ignored and review the branch diff.
2. Commit, push and open a pull request that closes issue `#7`.
3. Validate CI, mark ready, squash-merge and verify green `main`.

## Last completed mission

Mission 002 — verify official PTB-XL v1.0.3 metadata:

- Issue `#5` closed.
- Pull request `#6` squash-merged as `849ad56`.
- Real metadata verified: 21,799 records and 18,869 patients.
- No cross-split overlap or cross-fold patient conflicts.
- The post-merge Quality workflow passed on `main`.
- Local and remote mission branches were removed.

## Current evidence

- Official `scp_statements.csv` SHA-256 matches the v1.0.3 checksum:
  `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f`.
- Safe parsing and label contracts: 10 focused synthetic tests passed.
- Real label construction produced exactly 21,799 rows.
- 21,388 records have a target, 411 have none and remain preserved, and 5,144
  are multilabel.
- Totals: NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649.
- No code in real metadata is absent from the official statement catalogue.
- The 672 KB derived CSV remains ignored under `data/processed/`.
- No waveform file or WFDB dependency is used.
- Repeated generation preserved table SHA-256 `347f6657dc53e8792e7b1bc05c8dcc8f9dd66ce88eb49b61d61f374302a09515` and report SHA-256 `ecc0334aaa494aa9119b8597c53b357b5ae9ef8e21cb46408501d36da53b4724`.
- Full locked quality suite passed: 33 tests, Ruff lint and Ruff format.
