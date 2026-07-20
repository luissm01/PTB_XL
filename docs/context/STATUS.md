# Project status

Last updated: 2026-07-19

## Current mission

No active implementation mission. Mission 005, loading and auditing the official
PTB-XL 100 Hz signals, is complete.

## Current step

The repository is clean and synchronized. Present the proven signal boundary to
the mentor before opening a preprocessing or framework-integration mission.

## Next actions

1. Ask the mentor for the next objective, output and acceptance criteria.
2. Decide whether the next boundary is train-only preprocessing or a minimal
   framework-independent dataset adapter; do not assume either.
3. Identify any learned statistics and ensure they are fitted on train only.
4. Create a new issue, mission contract and branch only after approval.

## Last completed mission

Mission 005 — load and audit PTB-XL 100 Hz signals:

- Issue `#13` closed.
- Pull request `#14` squash-merged as `635dfeb`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 62 tests, Ruff lint and Ruff format.

## Mission 005 evidence

- Official `records100/`: 43,598 files; every SHA-256 verified.
- Real cohort audit: 21,388 loaded, zero missing and zero invalid.
- Every signal has shape `(1000, 12)`, frequency 100 Hz and finite numeric
  values.
- All headers use `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`.
- Repeated audit report SHA-256:
  `8c5c5deed8d77eba5381ccb6e93e295b8d2c96a73d75014678f8c1218fc5040f`.
- Metadata, label and cohort hashes remain unchanged.
- Raw signals remain ignored; only deterministic aggregate evidence is
  versioned.
- No normalization, filtering, resampling, PyTorch or model decision entered
  the mission.

## Stable repository foundation

- Reproducible Python 3.11 environment managed by uv.
- PTB-XL v1.0.3 identity, metadata, folds and patient isolation verified.
- Official five-superclass labels reproducibly constructed and audited.
- Initial modeling cohort explicitly defined without test-driven selection.
- Official 100 Hz signals safely resolved, loaded and audited for the full
  cohort.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
