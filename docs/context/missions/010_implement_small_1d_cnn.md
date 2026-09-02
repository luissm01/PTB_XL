# Mission 010 — Implement a small 1D-CNN baseline contract

## Objective

Add the first explainable model boundary: a small configurable 1D-CNN that
accepts the established channel-first ECG batch and emits five independent raw
logits for the multilabel task.

- Issue: `#28` — `[MODEL] Implement a small 1D CNN baseline contract`.
- Branch: `models/28-small-1d-cnn`.
- Pull request: `#29` — `Model: add small 1D CNN baseline`.
- Input contract: `torch.float32` `(B, 12, 1000)`.
- Output contract: raw logits `(B, 5)`.

## Architecture decision

Use three sequential feature blocks:

```text
Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D
```

with channels `12 -> 32 -> 64 -> 128`, odd kernels `7, 5, 3`, stride-one
same-length convolutions and factor-two temporal pooling. Adaptive global average
pooling summarizes the remaining temporal positions, configurable dropout
regularizes the representation and one linear layer emits five logits.

This is deliberately small and conventional. It is sufficient to prove tensor,
loss and gradient contracts before training infrastructure exists. Architecture
comparisons belong to later validation-governed experiments.

## Minimal API contract

- `ECGCNNConfig` is frozen and validates feature channels, odd kernels and
  dropout probability.
- `SmallECGCNN(config)` exposes the immutable configuration used to construct
  the layers.
- Forward rejects non-tensors, non-`float32`, empty batches and any shape other
  than `(B, 12, 1000)`.
- Forward returns exactly five finite-compatible logits per input record.
- The module contains no sigmoid or softmax; `BCEWithLogitsLoss` consumes logits
  directly and sigmoid belongs to probability-producing evaluation/inference.

## Required tests

- Default configuration and frozen behavior.
- Invalid channel/kernel/dropout configurations.
- Exact output shape for multiple batch sizes.
- Clear rejection of wrong rank, channels, samples, dtype and empty batches.
- Absence of sigmoid/softmax and exact default trainable parameter count.
- Finite `BCEWithLogitsLoss` and finite gradients for every trainable parameter.
- Deterministic repeated forward in evaluation mode.

## Leakage constraints

- Use synthetic tensors only; no PTB-XL waveform is needed.
- Choose the simple baseline from engineering constraints, not validation/test
  performance.
- Add no class weights, sampling, augmentation, thresholds or metric policy.
- Do not create training, checkpoint-selection or final-test code.

## Acceptance checklist

- [x] Frozen configuration validates every architecture field.
- [x] Model maps `(B, 12, 1000)` to five raw logits.
- [x] Input errors fail before entering convolution layers.
- [x] Default architecture and parameter count are tested.
- [x] Forward/backward and BCE compatibility are tested.
- [x] Eval output is deterministic for fixed input and state.
- [x] README and living guide explain the implemented model boundary.
- [x] Stable decisions, status and autonomous log are updated.
- [x] Tests, Ruff, format, build and GitHub checks pass.

## Out of scope

Real-data forward passes, optimizer, training loop, seeds policy, device
selection, metrics, thresholds, checkpoints, experiment tracking and final-test
evaluation.

## Local validation evidence

- Pytest: 118 passed, including 22 new model cases.
- Ruff lint and format check: passed.
- Source distribution and wheel build: passed.
- The built wheel contains `ptbxl/models/__init__.py` and `cnn.py`.
- `git diff --check`: passed.
- No PTB-XL waveform, validation prediction or test prediction was opened.

## Closure

- Issue `#28`: closed by the merged implementation.
- Pull request `#29`: squash-merged.
- Merge commit: `a228b42`.
- Python quality and GitGuardian on the PR: passed.
- Post-merge Quality workflow on `main`: passed.
- Local and remote implementation branches: removed.
