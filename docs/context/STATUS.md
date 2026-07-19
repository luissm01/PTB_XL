# Project status

Last updated: 2026-07-19

## Current mission

No active implementation mission. Mission 004, defining and auditing the initial
five-superclass modeling cohort, is complete.

## Current step

The repository is clean and synchronized. Present the cohort boundary and its
evidence to the mentor before opening the first signal-related mission.

## Next actions

1. Ask the mentor for the next objective, output and acceptance criteria.
2. Decide whether the next boundary is signal-file identity/availability or a
   minimal signal loader; do not assume sampling frequency or preprocessing.
3. Identify leakage and train-only fitting constraints before implementation.
4. Create a new issue, mission contract and branch only after approval.

## Last completed mission

Mission 004 — define the five-superclass modeling cohort:

- Issue `#10` closed.
- Pull request `#11` squash-merged as `b78d7ea`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.

## Mission 004 evidence

- Explicit inclusion rule: at least one of the five target superclasses.
- Exact partition: 21,799 input = 21,388 included + 411 excluded.
- All-zero records remain unchanged in the master label table.
- Included patients: 18,617; excluded-only patients: 252.
- Train: 17,084 records / 14,823 patients.
- Validation: 2,146 records / 1,917 patients.
- Test: 2,158 records / 1,877 patients.
- Zero patient overlap after cohort filtering.
- No label, identity, fold or split value is recalculated or changed.
- Cohort, exclusions and report reproduce identical bytes.
- Processed tables remain ignored; the deterministic report is versioned.
- Full local suite passed: 44 tests, Ruff lint and Ruff format.
- No signal file, class balancing, threshold or model decision entered the
  mission.

## Stable repository foundation

- Reproducible Python 3.11 environment managed by uv.
- PTB-XL v1.0.3 identity, metadata, folds and patient isolation verified.
- Official five-superclass labels reproducibly constructed and audited.
- Initial modeling cohort explicitly defined without test-driven selection.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
