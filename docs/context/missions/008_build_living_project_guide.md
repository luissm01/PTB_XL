# Mission 008 — Build the living project guide

## Objective

Create `docs/PROJECT_GUIDE.md` as the project's accessible, technically precise
study guide. It must explain the problem, the evidence produced so far and the
planned modeling path without presenting unimplemented work as complete.

- Issue: `#22` — `[DOCS] Build the living PTB-XL project guide`.
- Branch: `docs/22-build-project-guide`.
- Pull request: `#23` — `Docs: add living PTB-XL project guide`.
- Dataset: PTB-XL v1.0.3, five-superclass cohort, 100 Hz signals.

## Required structure

The guide must contain ten navigable parts:

1. Problem and ECG foundations.
2. PTB-XL data, labels, cohort and official splits.
3. Leakage risks and the project's safeguards.
4. Signal representation and train-only preprocessing.
5. Deep-learning concepts and planned model interface.
6. Evaluation concepts and planned threshold policy.
7. Current software architecture and data flow.
8. Reproducibility and MLOps practices.
9. Verified results, future result slots and limitations.
10. Interview questions with explanatory answers.

## Evidence policy

- Repository-specific numbers must come from versioned manifests or reports.
- General dataset and ECG claims must link to primary or official sources.
- Every implementation claim must be traceable to current code or completed
  mission contracts.
- Model, training, metric and threshold content must be marked as planned.
- No final-test performance or model result may appear before it exists through
  the approved evaluation workflow.

## Maintenance policy

- Treat the guide as a living document, not a one-time snapshot.
- Update only affected sections after each material mission.
- Preserve the distinction between verified facts, current decisions and future
  plans.
- Keep operational handoff state in `STATUS.md` and detailed contracts in the
  mission files instead of duplicating them in the guide.

## Acceptance checklist

- [x] All ten agreed parts are complete and linked from a contents section.
- [x] A non-specialist can follow the end-to-end story and key terminology.
- [x] Technical explanations are detailed enough for interview preparation.
- [x] Counts, split assignments, shapes, lead order and scaler values match the
  versioned evidence.
- [x] Patient, preprocessing, threshold and final-test leakage are explained.
- [x] Existing modules and scripts are accurately mapped.
- [x] Implemented and planned stages are visually unambiguous.
- [x] README links the guide and accurately describes preprocessing status.
- [x] Durable context and the autonomous development log are updated.
- [x] Pytest, Ruff, format, package build and GitHub checks pass.

## Out of scope

PyTorch dependencies, Dataset/DataLoader adapters, model code, training,
threshold fitting, final-test evaluation and new experimental results.

## Validation evidence

- All local Markdown targets referenced from the guide and README exist.
- Nine unique external reference destinations respond successfully.
- Every agreed part and all 16 interview questions are present.
- Project figures were checked against the five versioned evidence reports.
- `git diff --check`: passed.
- Pytest: 86 passed.
- Ruff lint and format check: passed.
- Source distribution and wheel build: passed.

## Closure

- Issue `#22`: closed by the merged implementation.
- Pull request `#23`: squash-merged.
- Merge commit: `395c74b`.
- Python quality and GitGuardian on the PR: passed.
- Post-merge Quality workflow on `main`: passed.
- Local and remote implementation branches: removed.
