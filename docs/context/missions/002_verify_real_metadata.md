# Mission 002 — Verify official PTB-XL v1.0.3 metadata

## Objective

Verify the identity and integrity of the official PTB-XL v1.0.3 metadata, run
the existing validation boundary against the real CSV and persist deterministic
evidence without loading ECG signals.

## Official source

- Dataset: PTB-XL v1.0.3.
- DOI: `10.13026/kfzx-aw45`.
- Release page: `https://physionet.org/content/ptb-xl/1.0.3/`.
- Metadata: `https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv`.
- Checksums: `https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt`.

Only the metadata and checksum information are needed. The 3 GB waveform
distribution remains out of scope.

## Required outputs

```text
data/ptbxl_metadata_manifest.json
reports/metadata/ptbxl_v1.0.3_summary.json
```

The CSV stays ignored under:

```text
data/raw/ptbxl_database.csv
```

## Evidence contract

The manifest records stable identity facts:

```json
{
  "dataset": "PTB-XL",
  "version": "1.0.3",
  "file": "ptbxl_database.csv",
  "sha256": "<official checksum>",
  "source": "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
}
```

The report contains:

- the dataset identity from the manifest;
- total record and patient counts;
- record counts for every fold;
- record and unique-patient counts for every split;
- cross-split patient overlaps;
- cross-fold patient assignments;
- no timestamps, absolute paths or machine-specific values.

JSON output must use stable ordering and formatting so identical inputs produce
identical bytes.

## Required behavior

- Verify the local CSV SHA-256 against the official checksum before inspection.
- Reuse the mission 001 validation rules.
- Reject a patient assigned to multiple `strat_fold` values with a message that
  is distinct from cross-split leakage.
- Assign every row to exactly one official split.
- Generate the report only after every validation passes.
- Never rewrite or commit the source CSV.

## Proposed design

- Extend `ptbxl.data.metadata` with an inspectable cross-fold conflict function
  and an immediate validation error.
- Add a small reusable reporting/provenance module for checksums and
  deterministic report construction.
- Add `scripts/validate_metadata.py` as a thin `argparse` entry point over the
  reusable functions.
- Do not add Click, Typer, YAML configuration or DVC.

The script is justified because it provides one documented, repeatable command;
business rules remain in `src/` so they can be tested directly.

## Test contract

### Synthetic tests

- [x] A patient in fold 1 and fold 2 is reported as a cross-fold conflict.
- [x] Preparing such metadata fails with a fold-specific error.
- [x] SHA-256 calculation is stable for known bytes.
- [x] A checksum mismatch fails before report generation.
- [x] The report contains the agreed dataset, totals, folds, splits and overlap
      structure.
- [x] Writing the same report twice produces identical bytes.

### Synthetic implementation status

- [x] Cross-fold validation implemented separately from cross-split leakage.
- [x] Checksum and deterministic reporting functions implemented.
- [x] Minimal `argparse` script implemented.
- [x] Pytest: 23 passed.
- [x] Ruff lint and format checks passed.

### Real metadata validation

- [x] The local CSV matches the official SHA-256.
- [x] All existing metadata validations pass.
- [x] Every patient belongs to exactly one fold.
- [x] Cross-split overlap is empty.
- [x] The actual counts are persisted and checked against the official v1.0.3
      description only after reading the real CSV.
- [x] Running the command twice leaves the report unchanged.

### Observed v1.0.3 result

- Records: 21,799.
- Unique patients: 18,869.
- Train (folds 1–8): 17,418 records and 15,023 patients.
- Validation (fold 9): 2,183 records and 1,942 patients.
- Test (fold 10): 2,198 records and 1,904 patients.
- Cross-split overlaps: none.
- Cross-fold patient conflicts: none.
- Deterministic report SHA-256:
  `bd275e273505c735fc4abb3e9720b9abfc76bc6626e30115f5e4f121daa5c696`.

## Reproducible command target

```bash
uv run python scripts/validate_metadata.py \
  --metadata-path data/raw/ptbxl_database.csv \
  --manifest-path data/ptbxl_metadata_manifest.json \
  --output-path reports/metadata/ptbxl_v1.0.3_summary.json
```

## Completion criteria

- PTB-XL version and source are fixed to v1.0.3.
- The official checksum is verified.
- The CSV is not versioned.
- Real metadata passes required-column, identifier, fold and split validation.
- No patient is shared across splits or assigned to multiple folds.
- The deterministic manifest and report are versioned.
- The command succeeds twice without changing the report.
- README documents the proven path.
- Synthetic tests, Ruff and GitHub Actions remain green.

## Out of scope

- ECG waveform files and WFDB.
- `scp_codes`, `scp_statements.csv` and diagnostic superclasses.
- EDA, plots and notebooks.
- DVC, models, MLflow, APIs and Docker.

## Decision to explain

Cross-split validation protects the current evaluation protocol. Cross-fold
validation preserves the official partition semantics for future reuse of
individual training folds.
