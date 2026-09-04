# Mission 017 — Add reproducible single-record inference

## Objective

Turn one compatible WFDB ECG into the five frozen baseline probabilities and
binary decisions through a simple, attributable command.

- Issue: `#45` — `[INFERENCE] Add reproducible single-record ECG inference`.
- Branch: `inference/45-reproducible-inference`.
- Pull request: `#46` — the authoritative merge gate and issue closure.
- Attributed smoke commit: `be0d297e5adb260611ecb718ebfc60417c1e3da5`.

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
- [x] A real train-record smoke inference succeeds from a clean commit.
- [x] README, guide, decisions, status and log are updated.
- [x] The final local quality gate passed; the mission PR is the merge gate.

## Out of scope

Batch serving, REST APIs, containers, authentication, monitoring, clinical
calibration, DICOM support, variable-length ECGs, resampling, model changes,
test re-evaluation, interpretability and error analysis.

## Real smoke evidence

- Input: official 100 Hz train `ecg_id=1`, used without targets.
- Commit: `be0d297e5adb260611ecb718ebfc60417c1e3da5`.
- Ordered scores: NORM `0.9147696495`, MI `0.0254258681`, STTC
  `0.0660511628`, CD `0.0183266401`, HYP `0.0012135925`.
- Frozen decisions: NORM positive; MI, STTC, CD and HYP negative.
- Signal fingerprint:
  `9c0e90bb6b6b7f1d929aa60957a80a11c14be76d10cb5c9a7b9237be4b2c5467`.
- Report SHA-256:
  `59e546296a10b4edc46459d2c5f83a5562a10b9a4a8d07fb7bf7a990fc5b4919`.
- Strict reload succeeded against checkpoint SHA-256
  `26d058b75dd486e0d81b75fea3cce44fabf6f7c625c57da79640caee6b5c9a11`.

## Local quality evidence

- 198 tests passed.
- Ruff lint and format check passed.
- Source distribution and wheel built successfully.
