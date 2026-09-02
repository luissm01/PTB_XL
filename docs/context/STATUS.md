# Project status

Last updated: 2026-09-02

## Current mission

Mission 009 — add a thin PyTorch Dataset/DataLoader boundary over the validated
sample contract and frozen train-only standardizer.

## Current step

Issue `#25` and branch `data/25-thin-pytorch-dataset` are open. The adapter,
ten synthetic tests and documentation updates are implemented; targeted checks
and a two-record-per-split real smoke check pass. The full 96-test suite, Ruff,
format and package build also pass; PR review is next.

## Next actions

1. Review the complete diff and open the implementation PR.
2. Complete GitHub checks and squash-merge.
3. Record Mission 009 closure before starting the CNN mission.

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
