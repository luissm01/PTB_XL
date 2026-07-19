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

Before changing code:

1. Inspect relevant files.
2. Explain the proposed change.
3. Identify risks.
4. Implement the smallest useful change.
5. Run the relevant checks.
6. Show and explain the diff.

# Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`
