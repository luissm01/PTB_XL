# Project status

Last updated: 2026-07-19

## Current mission

Define and audit the initial five-superclass modeling cohort without loading ECG
signals.

- Issue: `#10` — `[DATA] Define and audit the five-superclass modeling cohort`.
- Branch: `data/10-define-five-superclass-cohort`.
- Mission contract: `docs/context/missions/004_define_five_superclass_cohort.md`.

## Why this is next

Mission 003 constructed the five targets but intentionally retained 411 all-zero
records. This mission separates task eligibility from label construction so
those records do not silently become an undefined "none-of-the-above" class.

## Current step

Implementation, repeated real generation and the full local quality suite pass.
Review versioned scope and publish the branch for CI.

## Next actions

1. Confirm processed tables remain ignored and review versioned scope.
2. Publish a PR closing issue `#10`, validate checks and squash-merge.
3. Record post-merge state and request the next mentor mission.

## Last completed mission

Mission 003 — build diagnostic superclass labels:

- Issue `#7` closed.
- Pull request `#8` squash-merged as `83d2c0d`.
- Context closure pull request `#9` squash-merged as `a06ec6a`.
- Post-merge Quality workflow passed on `main`.

## Current evidence

- Focused synthetic cohort contract: 11 tests passed.
- Real partition: 21,799 input = 21,388 included + 411 excluded.
- Included patients: 18,617; excluded-only patients: 252.
- Train: 17,084 records / 14,823 patients.
- Validation: 2,146 records / 1,917 patients.
- Test: 2,158 records / 1,877 patients.
- Patient overlap across filtered splits is empty.
- Source master-label SHA-256 remains
  `347f6657dc53e8792e7b1bc05c8dcc8f9dd66ce88eb49b61d61f374302a09515`.
- Cohort and exclusion tables remain ignored under `data/processed/`.
- No waveform file or WFDB dependency is used.
- Repeated generation preserved cohort SHA-256 `f1ac3c713d43222eebba16124f54b1354e7634f6704032286e861196f6967058`, exclusions SHA-256 `ca37675ce05d39695c4a5e7f4ac288cea66384801b5918a75b2466286c2425fb`, and report SHA-256 `357f211d9610652ed1b9f9ddaeae8d4f75d5e013e5d4cd50a9e15a3d4200c123`.
- Full locked quality suite passed: 44 tests, Ruff lint and Ruff format.
