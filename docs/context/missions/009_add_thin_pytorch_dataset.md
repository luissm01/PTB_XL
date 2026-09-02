# Mission 009 — Add a thin PyTorch Dataset/DataLoader boundary

## Objective

Adapt the validated framework-independent sample contract and frozen train-only
standardizer into PyTorch tensors without moving data, split or preprocessing
logic into the framework layer.

- Issue: `#25` — `[DATA] Add a thin PyTorch dataset adapter`.
- Branch: `data/25-thin-pytorch-dataset`.
- Pull request: `#26` — `Data: add thin PyTorch dataset adapter`.
- Dataset: PTB-XL v1.0.3, five-superclass cohort, 100 Hz signals.

## Dependency decision

Add the current stable PyTorch release through `uv add` and commit the generated
lockfile. PyTorch is justified now because the next boundary and every later
model/training stage require its tensor, Dataset and DataLoader APIs.

The standard Linux package includes CUDA support. That larger dependency is
accepted because the available local NVIDIA RTX 3060 Ti is usable from PyTorch,
while the same package remains CPU-compatible. No separate accelerator-specific
environment is introduced in this first framework mission.

## Minimal API contract

- `PTBXLDataset(sample_index, dataset_root, standardizer, split=...)` accepts an
  already validated, non-empty index containing exactly one declared split.
- Construction copies only row metadata and never opens a waveform.
- `__getitem__` delegates waveform and target composition to `load_sample`.
- The supplied `GlobalStandardizer` transforms every split unchanged and is
  never fitted or mutated by the Dataset.
- The signal becomes a contiguous `torch.float32` tensor `(12, 1000)`.
- Targets become a `torch.float32` tensor `(5,)` in the established fixed order.
- Each item preserves `ecg_id`, `patient_id`, `strat_fold`, `split` and
  `filename_lr` for traceability.
- `build_dataloader(...)` provides standard batching and requires a seed when
  shuffle is enabled.

## Required behavior

- Reject missing sample-index columns, an empty index, an invalid declared split
  and rows from a different or mixed split.
- Preserve lazy loading and clear index errors.
- Use no custom collation when PyTorch's default collation satisfies the
  contract.
- Produce batched signals `(B, 12, 1000)` and targets `(B, 5)`.
- Make two shuffled loaders with the same seed produce the same ECG order.
- Keep `num_workers=0` as the portable default while allowing callers to change
  it explicitly.

## Leakage constraints

- Build and validate the complete framework-independent sample index before a
  caller selects one split for this Dataset.
- Never fit preprocessing inside Dataset or DataLoader code.
- Never combine train, validation and test rows in one Dataset.
- Add no sampling weights, class weights, augmentation or distribution-driven
  choice.
- Derive no design decision from final-test contents or frequencies.

## Acceptance checklist

- [x] PyTorch is declared and locked without manual lockfile edits.
- [x] Dataset construction is lazy and one-split-only.
- [x] Item tensors have exact shapes, dtypes, values and contiguous layout.
- [x] Sample provenance is preserved.
- [x] The existing loader and frozen standardizer are reused directly.
- [x] DataLoader batches correctly and seeded shuffle is reproducible.
- [x] Failure paths have synthetic tests.
- [x] README and living guide explain the implemented boundary.
- [x] Stable decisions, status and autonomous log are updated.
- [x] Tests, Ruff, format, build and GitHub checks pass.

## Out of scope

Models, losses, optimizer, device transfer, training loops, class balancing,
augmentation, metrics, thresholds, checkpoints and final-test evaluation.

## Real-data smoke evidence

- Complete sample index built before selecting any split: 21,388 rows.
- Rows loaded per split: two train, two validation and two test.
- Batch signal shape for every split: `(2, 12, 1000)`.
- Batch target shape for every split: `(2, 5)`.
- Signal dtype/layout for every split: `torch.float32`, contiguous.
- One unchanged standardizer artifact fitted on train was used for all splits.
- No model output, label prevalence or performance metric was computed.

## Local validation evidence

- Pytest: 96 passed, including 10 new adapter tests.
- Ruff lint and format check: passed.
- Source distribution and wheel build: passed.
- The built wheel contains `ptbxl/data/pytorch.py` and declares PyTorch.
- `git diff --check`: passed.

## Closure

- Issue `#25`: closed by the merged implementation.
- Pull request `#26`: squash-merged.
- Merge commit: `9e6ab7e`.
- Python quality and GitGuardian on the PR: passed.
- Post-merge Quality workflow on `main`: passed.
- Local and remote implementation branches: removed.
