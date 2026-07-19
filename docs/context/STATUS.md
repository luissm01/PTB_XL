# Project status

Last updated: 2026-07-19

## Current mission

No active implementation mission. Mission 003, building the five official
PTB-XL diagnostic superclass labels without loading ECG signals, is complete.

## Current step

The repository is on a clean, synchronized `main`. Ask the mentor and technical
director to define the next small, testable boundary before opening another
issue or implementation branch.

## Next actions

1. Present the Mission 003 evidence and decisions to the mentor.
2. Obtain the next objective, concrete output and acceptance criteria.
3. Identify its leakage risks before implementation.
4. Create the next issue, mission contract and branch only after approval.

## Last completed mission

Mission 003 — build diagnostic superclass labels:

- Issue `#7` closed.
- Pull request `#8` squash-merged as `83d2c0d`.
- Python quality and GitGuardian passed on the PR.
- The post-merge Quality workflow passed on `main`.
- Local and remote mission branches were removed.

## Mission 003 evidence

- Official `scp_statements.csv` v1.0.3 checksum verified.
- Exact row preservation: 21,799 metadata rows and 21,799 label rows.
- 21,388 records have at least one target label.
- 411 zero-target records remain present and explicitly quantified.
- 5,144 records are multilabel.
- Label totals: NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649.
- No real metadata code is absent from the official statement catalogue.
- Mapping derives only from official diagnostic rows and target superclasses.
- Existing validated splits are inherited without duplicate split logic.
- Repeated generation produced identical table and report bytes.
- The derived 672 KB CSV remains ignored; the manifest, code and deterministic
  report are versioned.
- Full locked local suite passed: 33 tests, Ruff lint and Ruff format.
- No waveform data, WFDB, cohort filtering, thresholds or model choices entered
  the mission.

## Stable repository foundation

- Python 3.11 environment and dependencies are locked with uv.
- The package uses a tested `src/` layout.
- GitHub Actions and GitGuardian protect pull requests and `main`.
- PTB-XL v1.0.3 metadata identity, folds and patient isolation are verified.
- Five-superclass label construction is reproducible and auditable.
- Durable decisions live in `docs/context/DECISIONS.md`; completed mission
  contracts remain under `docs/context/missions/`.
