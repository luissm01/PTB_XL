# Project purpose

Production-oriented ECG classification system using PTB-XL.

# Development principles

- Prefer simple, testable implementations.
- Prevent patient-level data leakage.
- Use the official PTB-XL folds.
- Keep notebooks for exploration only.
- Keep reusable logic under `src/`.
- Do not add dependencies without justification.
- Do not modify generated lock files manually.
- Important data transformations require tests.
- Do not use the final test fold for model selection.
- Avoid unnecessary abstractions.

# Workflow

At the start of a task:

1. Read `docs/context/STATUS.md`.
2. Read the active mission referenced there.
3. Consult `docs/context/DECISIONS.md` for stable constraints.

Before changing code:

1. Inspect relevant files.
2. Explain the proposed change.
3. Identify risks.
4. Implement the smallest useful change.
5. Run the relevant checks.
6. Show and explain the diff.

After a material decision or completed stage, update the relevant context file
without duplicating the same information across documents.

## Delivery cadence

- Prefer cohesive end-to-end missions that combine closely related components;
  do not create a separate mission for every small function or document update.
- Use one issue, one branch and one pull request per mission. Include closure
  evidence and context updates in the implementation PR; do not open a second
  closure-only PR.
- Run targeted tests while developing. Once the implementation is stable, run
  the full test suite, lint, format check and build once before the PR.
- Treat passing PR checks as the merge gate. Do not repeatedly wait for or rerun
  equivalent post-merge checks unless the merged tree changed or GitHub reports
  a failure.
- Update `STATUS.md` during handoff, `DECISIONS.md` only for stable choices, and
  the project guide/log at meaningful milestones rather than after every small
  component.
- Keep methodological controls strict even when delivery is batched: official
  folds, patient isolation, train-only fitting and final-test isolation are not
  shortcuts.

# Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
