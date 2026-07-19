"""Integrity checks and deterministic reports for PTB-XL metadata."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ptbxl.data.metadata import (
    build_metadata_summary,
    find_patient_fold_conflicts,
)


MANIFEST_FIELDS = ("dataset", "version", "file", "sha256", "source")


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Verify a file against an expected SHA-256 digest and return its digest."""
    actual = compute_sha256(path)
    if actual != expected.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def build_metadata_report(
    metadata: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic evidence report from validated metadata."""
    missing_fields = [field for field in MANIFEST_FIELDS if field not in manifest]
    if missing_fields:
        raise KeyError(f"Missing metadata manifest fields: {missing_fields}")

    summary = build_metadata_summary(metadata)
    overlap = {
        pair.replace("/", "_"): patients
        for pair, patients in summary["patient_overlap"].items()
    }
    fold_conflicts = {
        str(patient_id): folds
        for patient_id, folds in find_patient_fold_conflicts(metadata).items()
    }

    return {
        "dataset": {
            "name": manifest["dataset"],
            "version": manifest["version"],
            "metadata_file": manifest["file"],
            "sha256": manifest["sha256"],
            "source": manifest["source"],
        },
        "totals": {
            "records": summary["records"],
            "patients": summary["patients"],
        },
        "folds": {str(fold): count for fold, count in summary["folds"].items()},
        "splits": summary["splits"],
        "patient_overlap": overlap,
        "patient_fold_conflicts": fold_conflicts,
    }


def write_json_report(report: Mapping[str, Any], path: str | Path) -> None:
    """Write JSON with stable ordering and formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output_path.write_text(content, encoding="utf-8")
