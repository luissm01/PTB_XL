# Project status

Last updated: 2026-08-31

## Current mission

No active implementation mission. Mission 008, building the living project
guide, is complete.

## Current step

Development is intentionally paused after the agreed documentation milestone.
The repository is clean and synchronized after the implementation merge; PR
checks and the post-merge Quality workflow passed.

## Next actions

1. When development resumes, read this status, Mission 008 and D018.
2. Introduce a thin PyTorch Dataset/DataLoader adapter that reuses `load_sample`
   and the frozen global standardizer without duplicating logic.
3. Then implement a small, tested 1D-CNN baseline without using final test for
   model or threshold selection.

## Last completed mission

Mission 008 — build the living project guide:

- Issue `#22` closed.
- Pull request `#23` squash-merged as `395c74b`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.
- Full local suite passed: 86 tests, Ruff lint, Ruff format and package build.

## Mission 008 evidence

- `docs/PROJECT_GUIDE.md` contains all ten agreed parts and 16 explained
  interview questions.
- Implemented behavior, planned work and pending model results are explicitly
  separated.
- Project figures trace to the five versioned evidence reports and important
  general claims link primary or official sources.
- Every local Markdown target exists and all nine external destinations
  responded during validation.
- No dependency, runtime behavior, model result or final-test selection was
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
- A living project guide makes the verified system, future design and interview
  reasoning accessible without overstating unfinished work.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
