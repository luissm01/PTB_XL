# Project decisions

This file records decisions that should remain stable across tasks. Detailed
task progress belongs in `STATUS.md`; implementation-specific acceptance
criteria belong in the active mission file.

## D001 — Reproducible Python environment

- Use Python 3.11 managed by uv.
- Keep runtime and development dependencies in `pyproject.toml`.
- Commit `uv.lock` and never edit it manually.

Why: local development, clean clones and CI must resolve the same environment.

## D002 — PTB-XL split policy

- Folds 1–8: training.
- Fold 9: validation.
- Fold 10: final test.
- Never use fold 10 for model, preprocessing, calibration or threshold choices.

Why: this follows the official PTB-XL recommendation and preserves an unbiased
final evaluation.

## D003 — Minimum metadata identity

- Require `ecg_id`, `patient_id` and `strat_fold`.
- Reject duplicate `ecg_id` values.

Why: records, patients and their official split assignment must all be
identifiable before downstream processing.

## D004 — Patient leakage policy

- Detect overlap with a separately testable function.
- Fail immediately when validated metadata contains a patient in multiple
  splits.

Why: inspection remains possible, but unsafe metadata cannot silently continue
through the ML pipeline.

## D005 — Metadata output and errors

- Use a nested dictionary for the first metadata summary.
- Use `KeyError` for missing columns.
- Use `ValueError` for nulls, invalid folds, duplicate records and leakage.

Why: these standard structures are simple to test and serialize without adding
custom abstractions prematurely.

## D006 — Durable project context

- Keep the current handoff in `docs/context/STATUS.md`.
- Keep stable choices in this file.
- Keep one scoped contract per mission under `docs/context/missions/`.
- Update context after material decisions or completed workflow stages.

Why: a new developer or Codex session should be able to resume from repository
files without reconstructing decisions from chat history.

## D007 — Fixed PTB-XL metadata source

- Use PTB-XL v1.0.3 from the versioned PhysioNet distribution.
- Record the exact source URL and official SHA-256 for `ptbxl_database.csv`.
- Do not use a moving `latest` URL as dataset identity.

Why: results must remain attributable to the same dataset release even when a
newer version becomes available.

## D008 — Dataset evidence without DVC

- Keep the real CSV under ignored `data/raw/`.
- Version `data/ptbxl_metadata_manifest.json` with identity and checksum.
- Version `reports/metadata/ptbxl_v1.0.3_summary.json` with deterministic facts.
- Exclude timestamps, absolute paths and machine-specific details.

Why: one small source file can be identified and verified without introducing a
data-versioning system or committing the source metadata.

## D009 — Fold consistency is distinct from split leakage

- Cross-split patient overlap protects the current train/validation/test
  evaluation.
- A patient assigned to more than one `strat_fold` is a separate error, even
  when both folds map to `train`.

Why: the second guarantee preserves the official fold semantics and prevents
future leakage when individual folds are reused for cross-validation.

## D010 — Real-data validation remains local

- The default CI suite uses only synthetic data.
- GitHub Actions does not download PTB-XL in this mission.
- A deterministic JSON report records the locally verified real-data result.

Why: CI stays fast and independent of external data availability while the
reviewable report preserves evidence of the real validation.

## D011 — Official diagnostic superclass taxonomy

- Target labels are `NORM`, `MI`, `STTC`, `CD` and `HYP` in that order.
- Derive code-to-superclass assignments from the versioned official
  `scp_statements.csv`, never from a manually maintained code list.
- Only rows marked diagnostic and assigned to a target superclass participate.

Why: target semantics must be attributable to the dataset taxonomy rather than
to observations made from validation or test frequencies.

## D012 — SCP likelihood and unknown-code policy

- Code presence activates its mapped target regardless of likelihood.
- Validate likelihoods as finite numeric values from 0 to 100 but introduce no
  threshold because zero may mean unknown certainty in PTB-XL.
- Reject codes absent from the official statement catalogue.
- Count known codes outside the target mapping in the evidence report.

Why: this avoids an unjustified label threshold and prevents unresolved codes
from disappearing silently.

## D013 — Labels do not define the modeling cohort

- Preserve every validated metadata row, including records with no target.
- Inherit the existing split column without recalculating it.
- Keep the derived label table ignored and version only its deterministic
  construction code, source identity and summary evidence.

Why: label construction should expose the source population before a separate,
train-governed cohort decision and should not duplicate split logic.

## D014 — Initial five-superclass modeling cohort

- Include an ECG exactly when at least one of the five target labels equals one.
- Keep all-zero target records unchanged in the master labels table and exclude
  them from the initial classification cohort as `no_target_superclass`.
- Do not reinterpret all-zero records as normal or as an implicit sixth class.
- Preserve labels and official splits without frequency-based filtering or
  rebalancing.

Why: cohort membership follows the defined prediction task, not statistics from
validation/test or later model behavior, while exclusions remain reversible and
auditable.
