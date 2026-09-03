# Mission 014 — Run the reproducible real baseline

## Objective

Connect the verified PTB-XL data, preprocessing, CNN, training, checkpoint and
validation-evaluation boundaries into one configured experiment and record the
first real baseline result.

- Issue: `#39` — `[EXPERIMENT] Run the reproducible real baseline`.
- Branch: `experiment/39-real-baseline`.

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

- [ ] Configuration and expected split sizes are validated.
- [ ] Git provenance and deterministic execution are enforced.
- [ ] A synthetic end-to-end run writes a loadable checkpoint and valid report.
- [ ] Report schema contains all required experiment identities and metrics.
- [ ] The real run processes exactly 17,084 train and 2,146 validation ECGs.
- [ ] The real checkpoint is selected only by validation loss.
- [ ] No test signal is opened and no test result is recorded.
- [ ] README, guide, decisions, status and log are updated.
- [ ] One final local quality gate and one PR gate pass.

## Out of scope

Hyperparameter comparison, class weighting/sampling, threshold fitting,
final-test evaluation, plots, interpretability, error analysis, deployment and
inference.
