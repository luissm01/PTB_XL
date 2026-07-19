from pathlib import Path

import pandas as pd
import pytest

from ptbxl.data.reporting import (
    build_metadata_report,
    compute_sha256,
    verify_sha256,
    write_json_report,
)


MANIFEST = {
    "dataset": "PTB-XL",
    "version": "1.0.3",
    "file": "ptbxl_database.csv",
    "sha256": "a" * 64,
    "source": "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv",
}


def test_computes_sha256_for_known_content(tmp_path: Path) -> None:
    source = tmp_path / "known.bin"
    source.write_bytes(b"abc")

    assert compute_sha256(source) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_rejects_sha256_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "metadata.csv"
    source.write_bytes(b"unexpected content")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(source, "0" * 64)


def test_builds_metadata_report() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "patient_id": [101, 102, 103, 104],
            "strat_fold": [1, 8, 9, 10],
        }
    )

    assert build_metadata_report(metadata, MANIFEST) == {
        "dataset": {
            "name": "PTB-XL",
            "version": "1.0.3",
            "metadata_file": "ptbxl_database.csv",
            "sha256": "a" * 64,
            "source": MANIFEST["source"],
        },
        "totals": {"records": 4, "patients": 4},
        "folds": {"1": 1, "8": 1, "9": 1, "10": 1},
        "splits": {
            "train": {"records": 2, "patients": 2},
            "validation": {"records": 1, "patients": 1},
            "test": {"records": 1, "patients": 1},
        },
        "patient_overlap": {
            "train_validation": [],
            "train_test": [],
            "validation_test": [],
        },
        "patient_fold_conflicts": {},
    }


def test_writes_json_report_deterministically(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "summary.json"
    report = {"z": 1, "a": 2}

    write_json_report(report, output)
    first_write = output.read_bytes()
    write_json_report(report, output)

    assert output.read_bytes() == first_write
    assert first_write == b'{\n  "a": 2,\n  "z": 1\n}\n'
