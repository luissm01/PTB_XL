# Mission 014 — Run the reproducible real baseline

## Objective

Connect the verified PTB-XL data, preprocessing, CNN, training, checkpoint and
validation-evaluation boundaries into one configured experiment and record the
first real baseline result.

- Issue: `#39` — `[EXPERIMENT] Run the reproducible real baseline`.
- Branch: `experiment/39-real-baseline`.
- Pull request: `#40` — `Experiment: run reproducible PTB-XL baseline`.

## Contract

- One versioned TOML file defines data paths/identities, expected split counts,
  CNN parameters, seed, device, optimizer settings, epochs and outputs.
- The command refuses a dirty or unidentified Git tree before training.
- Data assembly validates the complete cohort association, then constructs
  loaders only for train and validation.
- Python, NumPy, PyTorch, CUDA algorithms and DataLoader workers are configured
  reproducibly; limitations across hardware/releases remain explicit.
- The unchanged `SmallECGCNN` trains with Adam and unweighted
  `BCEWithLogitsLoss`; no imbalance correction is introduced.
- The first minimum validation loss selects the checkpoint, and the restored
  model produces validation AUROC/AUPRC.
- A deterministic JSON report records configuration, source/config/checkpoint
  hashes, runtime versions, train-only prevalence, full loss history and
  validation ranking metrics.

## Leakage constraints

- No test Dataset, loader, waveform, prediction or metric is created.
- Train prevalence is the only distribution used to describe training choices.
- No class weights, resampling, architecture search or hyperparameter sweep.
- No thresholds or threshold-dependent metrics.
- The test fold remains sealed for a later explicitly authorized final event.

## Acceptance checklist

- [x] Configuration and expected split sizes are validated.
- [x] Git provenance and deterministic execution are enforced.
- [x] A synthetic end-to-end run writes a loadable checkpoint and valid report.
- [x] Report schema contains all required experiment identities and metrics.
- [x] The real run processes exactly 17,084 train and 2,146 validation ECGs.
- [x] The real checkpoint is selected only by validation loss.
- [x] No test signal is opened and no test result is recorded.
- [x] README, guide, decisions, status and log are updated.
- [ ] One final local quality gate and one PR gate pass.

## Real-run evidence

- Attributed implementation commit: `3b35e5f780ac9f7d37b43aaa3b2cb75bf2f2718a`.
- Configuration SHA-256:
  `50b5b68f8497baf27101a92d550588f8817b6c47a4832267c8033d8ccea2d714`.
- Best epoch: 9 of 10, with validation loss `0.2949302586182633`.
- Validation macro AUROC/AUPRC: `0.9153098458` / `0.7859935473`.
- Validation micro AUROC/AUPRC: `0.9260575328` / `0.8323655924`.
- Checkpoint SHA-256:
  `26d058b75dd486e0d81b75fea3cce44fabf6f7c625c57da79640caee6b5c9a11`.
- The ignored checkpoint was reloaded on CPU and its epoch, history, loss,
  provenance and hash matched the report.
- The report lists only train and validation. The runner contains no test
  Dataset, loader, waveform access or metric.

## Out of scope

Hyperparameter comparison, class weighting/sampling, threshold fitting,
final-test evaluation, plots, interpretability, error analysis, deployment and
inference.
