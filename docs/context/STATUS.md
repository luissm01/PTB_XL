# Project status

Last updated: 2026-07-19

## Current mission

Verify the identity, integrity and official folds of the real PTB-XL v1.0.3
metadata without downloading ECG signals.

- Issue: `#5` — `[DATA] Verify official PTB-XL v1.0.3 metadata`
- Branch: `data/5-verify-ptbxl-v1.0.3-metadata`
- Mission contract: `docs/context/missions/002_verify_real_metadata.md`

## Why this is next

Mission 001 proved the metadata rules with synthetic examples. This mission
must demonstrate that the official file itself satisfies those rules and leave
versioned, reproducible evidence of the result.

## Current step

Mission 002 implementation and real-data validation are complete locally.
Review the full branch diff and publish it for CI and peer review.

## Important source fact

The official PhysioNet v1.0.3 page describes 21,799 records and 18,869 patients.
Earlier planning material mentioned 21,837 and 18,885. The real CSV and its
official checksum are the source of truth; no count will be hard-coded before
inspection.

## Next actions

1. Review the complete diff and confirm that raw data is still ignored.
2. Commit and push `data/5-verify-ptbxl-v1.0.3-metadata`.
3. Open a pull request linked to issue `#5` and validate GitHub Actions.

## Last completed mission

Mission 001 — PTB-XL metadata and official splits:

- Issue `#3` closed.
- Pull request `#4` squash-merged as `f9577e5`.
- 17 synthetic tests passed.
- GitHub Actions and GitGuardian passed on the PR and `main`.
- Local and remote mission branches were removed after merge.

## Current repository health

- `main` was synchronized and clean before branch creation.
- Final locked quality checks passed: 23 tests, Ruff lint and Ruff format.
- The official CSV checksum is
  `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`.
- Real v1.0.3 validation passed with 21,799 records and 18,869 patients.
- Folds 1–8 contain 17,418 training records, fold 9 contains 2,183 validation
  records and fold 10 contains 2,198 test records.
- Cross-split patient overlap and cross-fold patient conflicts are empty.
- Two consecutive report generations produced the same report SHA-256:
  `bd275e273505c735fc4abb3e9720b9abfc76bc6626e30115f5e4f121daa5c696`.
- The validation script exposes the agreed `argparse` interface.
- The local PTB-XL CSV and checksum list remain ignored under `data/raw/`.
- Default CI remains synthetic and does not download the dataset.
