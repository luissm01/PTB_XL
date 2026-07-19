# PTB-XL setup progress

Last updated: 2026-07-19

## Current objective

Prepare a minimal, reproducible Python project using Git, GitHub, Python 3.11, uv, Pytest and Ruff.

## Decisions

- Project path: `~/projects/ptbxl-ml-system`
- Development environment: Ubuntu 26.04 LTS on WSL2
- Python target: 3.11, managed by uv
- Main Git branch: `main`
- GitHub repository: `luissm01/PTB_XL`
- Git remote: `origin` → `https://github.com/luissm01/PTB_XL.git`
- GitHub visibility: public
- GitHub CLI will be used
- Initial ML dependencies are intentionally deferred

## Completed

- [x] Environment inspected
- [x] Project moved into the WSL Linux filesystem
- [x] Git identity configured with a GitHub private email
- [x] Local Git repository initialized on `main`
- [x] uv 0.11.29 installed with Astral's official installer
- [x] Python package initialized with the import module `ptbxl`
- [x] Virtual environment created with CPython 3.11.15
- [x] Pytest 9.1.1 and Ruff 0.15.22 configured as development dependencies
- [x] Minimum import test added and passing
- [x] `.gitignore` created for local environments, data and artifacts
- [x] Minimal README created
- [x] `AGENTS.md` created
- [x] Initial commit created and pushed to `origin/main`
- [x] GitHub repository created and connected as `origin`
- [x] Clean-clone reproducibility verified

## Current step

Initial environment and repository setup completed.

## Next steps

1. Select the first implementation task from the project roadmap.
2. Create a focused GitHub issue for the task.
3. Create a short-lived branch from `main`.

## Latest verification

- `uv sync`: passed
- Python: 3.11.15 from `.venv`
- Package import: `src/ptbxl/__init__.py`
- Pytest: 1 passed
- Ruff lint: passed
- Ruff format check: passed
- Clean clone: `uv sync --locked`, Pytest and Ruff passed

## Working rules

- Keep changes small and inspect relevant diffs.
- Do not edit `uv.lock` manually.
- Do not add ML dependencies during initial setup.
- Preserve the official PTB-XL folds and prevent patient-level leakage.
