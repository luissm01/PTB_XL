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

## D015 — First signal boundary uses official 100 Hz records

- Start with PTB-XL `filename_lr` records at 100 Hz; comparison with 500 Hz is a
  later explicit experiment.
- Keep the canonical loader shape as `(samples, leads)`, matching WFDB output;
  channel-first conversion belongs in a later framework adapter.
- Resolve signal paths from official metadata by exact `ecg_id` association and
  never copy a signal path into the established cohort table.
- Validate 1,000 samples, 12 leads, 100 Hz, finite numeric values and consistent
  header lead order without normalization, filtering or resampling.
- Audit every cohort record sequentially and version only deterministic
  aggregate integrity evidence.

Why: this proves the signal-label identity boundary and the complete low-cost
data path before introducing learned transformations or framework-specific
representations, while keeping raw signals outside Git.

## D016 — Compose samples before introducing an ML framework

- Build a transient sample index by joining the validated cohort to official
  `filename_lr` values one-to-one on `ecg_id`; do not add paths to the persisted
  cohort table.
- Revalidate the complete cohort's binary targets, task membership, patient
  isolation and official fold-to-split mapping before association.
- Load signals lazily and keep their canonical NumPy shape `(1000, 12)` at this
  boundary.
- Expose targets as a `float32` NumPy vector in the fixed order `NORM`, `MI`,
  `STTC`, `CD`, `HYP`; channel-first conversion remains a later framework
  responsibility.
- Apply no filtering, normalization, augmentation or learned transformation.

Why: one framework-independent composition boundary prevents PyTorch adapters
from duplicating identity, label, split and path logic, while keeping the raw
data contract directly testable.

## D017 — First preprocessing is train-only global standardization

- Fit one shared mean and population standard deviation (`ddof=0`) across every
  signal value from folds 1–8 of the five-superclass cohort.
- Validate the complete sample index before selecting train, and make the fit
  API reject any validation or test sample.
- Accumulate moments sequentially in `float64`; do not materialize the complete
  signal tensor or add scikit-learn.
- Apply the frozen affine transform unchanged to train, validation, test and
  inference, preserving `(1000, 12)` and returning `float32`.
- Persist the small learned transformer as deterministic, versioned JSON with
  dataset identity, configuration, lead order, counts and source hashes.
- Add no filtering, per-record/per-lead scaling, resampling, denoising or
  augmentation to the first baseline.

Why: the public PTB-XL benchmarking implementation uses a single scaler fitted
on flattened training signals. A shared scalar transform retains relative
amplitudes across records and leads, while a streaming implementation preserves
that simple rule without its high-memory intermediate or pickle dependency.

## D018 — The project guide is living, evidence-led documentation

- Keep `docs/PROJECT_GUIDE.md` as the accessible conceptual and technical
  explanation of the complete system.
- Clearly distinguish implemented and verified behavior from planned work and
  pending results.
- Ground project-specific figures in versioned evidence and important general
  claims in primary or official sources.
- Update only affected guide sections after material missions; operational
  handoff remains in `STATUS.md` and detailed contracts remain in mission files.
- Never fill future result sections with estimates or use final-test evidence to
  rewrite earlier model-selection decisions.

Why: the owner needs one teachable, interview-ready narrative that evolves with
the repository without becoming a second status log or overstating unfinished
work.

## D019 — PyTorch remains a thin, split-explicit framework boundary

- Add PyTorch only now that identity, signals, samples and frozen preprocessing
  have framework-independent contracts.
- Require each `PTBXLDataset` to contain exactly one explicitly declared split.
- Delegate sample loading to `load_sample` and transformation to the supplied
  train-fitted `GlobalStandardizer`; never fit inside the Dataset.
- Convert `(1000, 12)` NumPy signals to contiguous `float32` tensors
  `(12, 1000)` and preserve `float32` targets `(5,)` plus provenance.
- Use default PyTorch collation and require an explicit non-negative seed for
  shuffled DataLoaders.
- Keep the standard PyTorch package because it supports CPU and the available
  local NVIDIA GPU; accept its larger Linux dependency footprint instead of
  introducing separate accelerator environments now.

Why: the framework layer should adapt representation and batching, not become a
second implementation of dataset semantics or preprocessing. Explicit splits
and seeded shuffle make accidental mixing and hidden randomness harder.

## D020 — The first model is a small configurable 1D-CNN

- Use three `Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D` blocks with default
  channels `32, 64, 128` and odd kernels `7, 5, 3`.
- Summarize time with adaptive global average pooling, then configurable dropout
  and one linear layer.
- Require `float32` inputs `(B, 12, 1000)` and emit exactly five raw logits with
  no sigmoid or softmax inside the model.
- Keep architecture fields in a frozen validated dataclass and test the default
  trainable parameter count of 38,597.
- Prove compatibility with `BCEWithLogitsLoss` and finite gradients using only
  synthetic tensors.
- Treat the roughly 0.3-second local receptive field before global pooling as a
  documented baseline limitation, not a result-driven optimum.

Why: a small conventional network is fast to test and easy to explain. It
establishes the model/loss boundary without mixing in training, evaluation or
test-driven architecture selection.

## D021 — Train and evaluation are separate sample-weighted epoch boundaries

- Seed Python, NumPy, PyTorch and CUDA streams from one validated integer.
- Resolve CPU/CUDA explicitly and fail rather than silently falling back from a
  requested unavailable CUDA device.
- Let `train_one_epoch` own gradients and optimizer steps; let `evaluate_loss`
  use eval/inference mode and never update parameters.
- Aggregate batch losses weighted by sample count so a short final batch does
  not receive the same weight as a full batch.
- Reject malformed batches, shape mismatches, empty loaders and non-finite loss.

Why: this establishes reproducible, testable optimization semantics before
multi-epoch state, checkpoint selection or metrics make failures harder to
isolate.

## D022 — Checkpoints are selected only by validation loss and loaded safely

- Require explicit train and validation Dataset roles in `fit`; reject test as
  a selection loader in code.
- Run a fixed positive number of epochs and keep the first strict minimum
  validation loss, avoiding ambiguous replacement on ties.
- Persist best model/optimizer state, full train/validation loss history and
  dataset/cohort/preprocessing/model/seed/Git provenance.
- Write through a same-directory temporary file and atomic replacement.
- Load with PyTorch `weights_only=True`, validate schema/provenance first and
  restore the selected model and optimizer before returning from fit.

Why: model selection and recovery must be reproducible without allowing fold 10
into the development loop or accepting arbitrary pickle objects from a
checkpoint.

## D023 — Delivery is batched without weakening methodological gates

- Group tightly related components into cohesive end-to-end missions instead of
  creating a mission for every small implementation boundary.
- Use one issue, branch and pull request per mission, including documentation and
  closure evidence in that same pull request.
- Use targeted checks during implementation, then run the full suite, Ruff and
  package build once when the mission is stable and ready for review.
- Use passing pull-request CI as the merge gate; avoid duplicate closure PRs and
  repeated waits for equivalent post-merge validation.
- Update operational context continuously, but update the long-form guide and
  autonomous log only at meaningful milestones.
- Preserve all leakage, reproducibility, provenance and final-test protections.

Why: the first twelve missions proved the engineering foundations but incurred
too much repeated GitHub and documentation ceremony. Batching related work
reduces delivery overhead while retaining the controls that affect scientific
validity.

## D024 — Validation ranking metrics use strict, tested definitions

- Collect predictions only from a Dataset explicitly declaring `validation` and
  preserve unique ECG identities in loader order.
- Convert the five logits to sigmoid probabilities once and compute metrics with
  scikit-learn rather than maintaining custom ranking algorithms.
- Report AUROC and AUPRC per class, unweighted macro and flattened micro.
- Define AUPRC as non-interpolated average precision; do not use trapezoidal
  precision-recall integration under the same name.
- Reject a validation result if any class lacks a positive or a negative target,
  because AUROC would be undefined and macro summaries would be misleading.
- Keep threshold-dependent metrics and threshold selection in a later,
  validation-governed stage.

Why: a strict metric boundary makes comparisons unambiguous, exposes invalid
evaluation cohorts early and reuses a mature statistical implementation while
keeping the final test fold outside development.

## D025 — The first real experiment is one predeclared neutral baseline

- Use the unchanged default `SmallECGCNN` on official 100 Hz signals with the
  frozen train-fitted global standardizer.
- Train for 10 fixed epochs with seed 2026, batch size 128, Adam at learning rate
  0.001, no weight decay and unweighted `BCEWithLogitsLoss`.
- Apply standard seeded shuffling only to train; add no class weights,
  oversampling, augmentation or scheduler before observing the neutral baseline.
- Require deterministic PyTorch algorithms and seeded DataLoader workers,
  accepting their performance cost and documenting that releases/hardware can
  still change exact floating-point results.
- Select the first minimum validation loss, then report validation AUROC/AUPRC
  from that restored checkpoint. Do not use those results to retroactively
  change this baseline configuration.
- Track the experiment in one versioned TOML and one deterministic JSON report;
  do not add MLflow before multiple runs make the simpler format insufficient.
- Construct train and validation Datasets only. Fold 10 remains sealed.

Why: one fully attributed baseline proves the complete training path and creates
an honest reference point before any imbalance treatment, tuning or threshold
selection.

## D026 — Operating thresholds maximize per-class validation F1

- Select one threshold independently for each target using fold 9 only.
- Use `probability >= threshold` as the decision rule.
- Maximize per-class F1 because the task is multilabel and the first operating
  point should balance precision and sensitivity without inventing clinical
  costs that the project does not possess.
- If several observed cutoffs produce the same maximum F1, choose the highest
  threshold for a deterministic, more conservative tie break.
- Freeze thresholds with confusion counts, precision, sensitivity/recall,
  specificity and F1 per class plus macro and micro summaries.
- Bind the artifact to the checkpoint, experiment configuration/report,
  preprocessing and a canonical validation-prediction fingerprint.
- Do not alter the baseline or inspect fold 10 after seeing validation operating
  metrics.

Why: ranking metrics do not define deployable binary decisions. A transparent,
validation-only rule supplies the missing operating point while preserving the
independence of the final test. Per-class thresholds accommodate different
prevalences and score distributions without claiming a clinically optimized
utility function.

## D027 — Fold 10 is a one-time immutable final-evaluation event

- Open fold 10 only after the baseline config, train-fitted standardizer,
  epoch-9 checkpoint and validation-selected thresholds are frozen by exact
  hashes.
- Use a separate command that refuses mismatched artifacts and refuses to
  overwrite an existing final report.
- Build a test-only Dataset and perform inference without training, checkpoint
  selection, calibration or threshold selection.
- Report the same ranking definitions plus metrics at the frozen operating
  point, per class and with macro/micro summaries.
- Save row-level test predictions only as an ignored local artifact; version a
  deterministic aggregate report with its hash and canonical fingerprint.
- Once observed, treat results and later error/interpretability analysis as
  descriptive. Do not modify or replace the frozen pipeline because of test.

Why: test estimates how the already selected system generalizes. Repeated use
or result-driven changes would make it another validation set and invalidate
the meaning of the final estimate.

## D028 — Inference accepts a compatible WFDB record and a frozen bundle

- Accept one WFDB record basename independently of PTB-XL metadata, cohort,
  labels or split membership.
- Require the technical training contract: 100 Hz, 1,000 samples, 12 finite
  numeric leads in the standardizer's canonical order.
- Load the baseline configuration/report, train-fitted standardizer, selected
  checkpoint and validation-selected thresholds only when their exact declared
  SHA-256 identities and cross-artifact provenance agree.
- Apply the frozen standardizer, one model forward pass, sigmoid and the fixed
  per-class `probability >= threshold` rule without fitting or selection.
- Emit deterministic JSON containing ordered probabilities, decisions, input
  fingerprint, artifact hashes, runtime and limitations; refuse overwrite.
- Keep the interface as a local CLI. Do not add a web API, container or new
  dependency until a concrete deployment need exists.

Why: a small file-to-prediction boundary demonstrates that the trained system
can actually be reused while preventing silent drift in signal format or model
artifacts. Independence from labels and dataset tables makes it genuine
inference rather than another evaluation path.
