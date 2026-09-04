# Mission 017 — Add reproducible single-record inference

## Objective

Turn one compatible WFDB ECG into the five frozen baseline probabilities and
binary decisions through a simple, attributable command.

- Issue: `#45` — `[INFERENCE] Add reproducible single-record ECG inference`.
- Branch: `inference/45-reproducible-inference`.

## Contract

- Load a standalone WFDB record basename without metadata, targets, cohort or
  split information.
- Require 100 Hz, 1,000 samples, 12 finite numeric leads in canonical order.
- Verify exact hashes and cross-provenance for baseline config/report,
  checkpoint, train-only standardizer and validation-selected thresholds.
- Restore the selected CNN checkpoint without optimizer steps or fitting.
- Apply frozen standardization, one forward pass, sigmoid and the five fixed
  thresholds in canonical `NORM`, `MI`, `STTC`, `CD`, `HYP` order.
- Write one deterministic, non-overwriting JSON report with input fingerprint,
  probabilities, decisions, artifact identities and runtime.
- Provide a strict report loader and a thin CLI driven by versioned TOML.

## Methodological constraints

- Never access fold 10 signals or row-level test predictions.
- Never train, fit, calibrate, select or alter a threshold.
- A prediction is not a diagnosis and probabilities are not claimed to be
  clinically calibrated.
- The smoke inference uses a train record only and makes no model decision.
- Add no service, API, Docker image or dependency.

## Acceptance checklist

- [x] Standalone WFDB loading rejects invalid shape, frequency, leads and data.
- [x] Exact artifact mismatches fail before model or signal inference.
- [x] One record produces five finite ordered probabilities and decisions.
- [x] Decisions exactly implement the frozen `>=` rule.
- [x] The output report is strict, deterministic and refuses overwrite.
- [x] Synthetic CPU end-to-end inference and failure paths are tested.
- [ ] A real train-record smoke inference succeeds from a clean commit.
- [ ] README, guide, decisions, status and log are updated.
- [ ] One final local quality gate and one PR gate pass.

## Out of scope

Batch serving, REST APIs, containers, authentication, monitoring, clinical
calibration, DICOM support, variable-length ECGs, resampling, model changes,
test re-evaluation, interpretability and error analysis.
