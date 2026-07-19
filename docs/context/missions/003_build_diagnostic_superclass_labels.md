# Mission 003 — Build diagnostic superclass labels

## Objective

Build an auditable, reproducible one-row-per-ECG multilabel table for the five
official PTB-XL diagnostic superclasses without loading waveform files.

- Issue: `#7` — `[DATA] Build the five PTB-XL diagnostic superclass labels`.
- Dataset: PTB-XL v1.0.3.
- Targets, in stable order: `NORM`, `MI`, `STTC`, `CD`, `HYP`.

## Why this boundary exists

Validated splits do not guarantee valid targets. This mission proves that label
semantics come from the official statement catalogue, that every input ECG is
accounted for and that no cohort or model decision is mixed into label creation.

## Official inputs

- Ignored `data/raw/ptbxl_database.csv`.
- Ignored `data/raw/scp_statements.csv`.
- Versioned source manifests containing exact v1.0.3 URLs and SHA-256 values.
- PTB-XL describes `scp_codes` as statement-to-likelihood dictionaries. A value
  from 0 to 100 expresses certainty, while zero can mean unknown certainty.

The target decision therefore uses presence of an official diagnostic code. No
likelihood threshold is introduced.

## Contract

- Parse `scp_codes` with `ast.literal_eval`, never `eval`.
- Treat a null `scp_codes` value as an empty dictionary and preserve its row.
- Reject malformed dictionaries, invalid likelihood values and unknown codes.
- Build the mapping from rows marked `diagnostic == 1` whose
  `diagnostic_class` is one of the five targets.
- Known non-target codes activate no target and are counted in the report.
- Inherit `ecg_id`, `patient_id`, `strat_fold` and `split` from validated
  metadata; label code does not calculate splits.
- Preserve exactly one output row for every input ECG.
- Keep zero-target rows with five zeros and report them explicitly.
- Keep labels binary and columns deterministically ordered.

## Outputs

- Ignored derived table:
  `data/processed/ptbxl_v1.0.3_superclass_labels.csv`.
- Versioned deterministic report:
  `reports/labels/ptbxl_v1.0.3_superclass_summary.json`.
- Versioned statement identity:
  `data/ptbxl_scp_statements_manifest.json`.

The derived table is 672 KB and PTB-XL uses CC BY 4.0. Despite its modest size
and permissive redistribution terms, it remains ignored under the conservative
project data policy. It can be recreated from identified official inputs.

## API responsibilities

- `load_scp_statements`: CSV loading and catalogue shape validation.
- `parse_scp_codes`: safe parsing and likelihood-shape validation.
- `build_diagnostic_superclass_mapping`: official target taxonomy derivation.
- `build_superclass_labels`: identity-preserving binary transformation.
- `count_excluded_codes`: explicit audit of known non-target annotations.
- `build_label_summary`: deterministic prevalence and combination evidence.

These responsibilities remain separate because each is an independent trust
boundary with distinct failure modes and direct synthetic tests.

## Real-data evidence

- Input/output rows: 21,799.
- With any target: 21,388.
- Without a target: 411, preserved with all-zero targets.
- Multilabel records: 5,144.
- Label totals: NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649.
- Unknown codes: none.
- Label cardinality: 1.273682279003624.

Test-set counts are descriptive evidence only and were not used to change the
mapping, targets, thresholds or population.

## Reproducible command

```bash
uv run --locked python scripts/build_superclass_labels.py \
  --metadata-path data/raw/ptbxl_database.csv \
  --metadata-manifest-path data/ptbxl_metadata_manifest.json \
  --statements-path data/raw/scp_statements.csv \
  --statements-manifest-path data/ptbxl_scp_statements_manifest.json \
  --labels-output-path data/processed/ptbxl_v1.0.3_superclass_labels.csv \
  --report-output-path reports/labels/ptbxl_v1.0.3_superclass_summary.json
```

## Acceptance checklist

- [x] Official statement file version and checksum fixed.
- [x] Safe parser and explicit null contract.
- [x] Mapping uses only official target diagnostic rows.
- [x] Unknown codes fail explicitly; known excluded codes are reported.
- [x] Binary, stable target columns.
- [x] Identity, row count, folds and splits preserved.
- [x] Zero-target and multilabel records reported.
- [x] Per-label and per-split counts reported.
- [x] No ECG signals loaded.
- [x] Synthetic label tests pass.
- [x] Repeated real generation produces identical table and report bytes.
- [x] Full Pytest and Ruff checks pass.
- [ ] Pull request and main CI pass.

## Leakage constraints

- The taxonomy comes only from the official statement catalogue.
- Test frequencies are not used for taxonomy or cohort decisions.
- No global frequency-based filtering occurs.
- Existing validated splits are inherited, not recomputed.
- No zero-target ECG is removed before a separate cohort mission.

## Out of scope

Signals, WFDB, sampling frequency, cohort exclusion, weights, resampling,
thresholds, models, metrics, DVC, MLflow and exploratory notebooks.
