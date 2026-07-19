# PTB-XL setup progress

Last updated: 2026-07-19

## Current objective

Add a minimal GitHub Actions quality workflow for the existing Python checks.

## Active task

- Issue: `#1` — `[CI] Add minimal Python quality workflow`
- Branch: `ci/1-minimal-quality-workflow`
- Workflow: `.github/workflows/quality.yml`

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

Finish local validation, publish the CI branch and confirm the workflow on GitHub.

## Next steps

1. Review the complete issue `#1` diff and run all quality commands.
2. Commit and push the workflow branch.
3. Open a pull request with `Closes #1`.
4. Confirm that the first GitHub Actions run passes before merging.

## Latest verification

- `uv sync`: passed
- Python: 3.11.15 from `.venv`
- Package import: `src/ptbxl/__init__.py`
- Pytest: 1 passed
- Ruff lint: passed
- Ruff format check: passed
- Clean clone: `uv sync --locked`, Pytest and Ruff passed
- Workflow YAML: syntax and style validation passed

## Working rules

- Keep changes small and inspect relevant diffs.
- Do not edit `uv.lock` manually.
- Do not add ML dependencies during initial setup.
- Preserve the official PTB-XL folds and prevent patient-level leakage.
