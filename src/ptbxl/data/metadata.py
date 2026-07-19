"""Validation and split assignment for PTB-XL metadata."""

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("ecg_id", "patient_id", "strat_fold")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PAIRS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load PTB-XL metadata from a CSV file."""
    return pd.read_csv(path)


def prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Validate metadata, assign official splits and reject patient leakage."""
    prepared = _validate_and_assign_splits(metadata)
    overlaps = _find_overlaps_in_prepared_metadata(prepared)
    detected = {pair: patients for pair, patients in overlaps.items() if patients}

    if detected:
        details = ", ".join(f"{pair}={patients}" for pair, patients in detected.items())
        raise ValueError(f"Patient leakage detected across splits: {details}")

    return prepared


def find_patient_overlaps(metadata: pd.DataFrame) -> dict[str, list[Any]]:
    """Return patient identifiers shared by each pair of official splits."""
    prepared = _validate_and_assign_splits(metadata)
    return _find_overlaps_in_prepared_metadata(prepared)


def build_metadata_summary(metadata: pd.DataFrame) -> dict[str, Any]:
    """Build reproducible record, patient, fold and split counts."""
    prepared = prepare_metadata(metadata)

    fold_counts = {
        int(fold): int(count)
        for fold, count in prepared["strat_fold"].value_counts().sort_index().items()
    }
    split_counts = {}
    for split in SPLIT_NAMES:
        split_metadata = prepared.loc[prepared["split"] == split]
        split_counts[split] = {
            "records": len(split_metadata),
            "patients": int(split_metadata["patient_id"].nunique()),
        }

    return {
        "records": len(prepared),
        "patients": int(prepared["patient_id"].nunique()),
        "folds": fold_counts,
        "splits": split_counts,
        "patient_overlap": _find_overlaps_in_prepared_metadata(prepared),
    }


def _validate_and_assign_splits(metadata: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in metadata.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required metadata columns: {missing_columns}")

    prepared = metadata.copy()

    for column in REQUIRED_COLUMNS:
        if prepared[column].isna().any():
            raise ValueError(f"Column {column!r} contains null values")

    duplicate_ecg_ids = prepared.loc[
        prepared["ecg_id"].duplicated(keep=False), "ecg_id"
    ].unique()
    if len(duplicate_ecg_ids) > 0:
        raise ValueError(
            f"Column 'ecg_id' contains duplicate values: {duplicate_ecg_ids.tolist()}"
        )

    numeric_folds = pd.to_numeric(prepared["strat_fold"], errors="coerce")
    invalid_folds = (
        numeric_folds.isna()
        | numeric_folds.mod(1).ne(0)
        | ~numeric_folds.between(1, 10)
    )
    if invalid_folds.any():
        values = prepared.loc[invalid_folds, "strat_fold"].tolist()
        raise ValueError(
            "Column 'strat_fold' must contain integers from 1 to 10; "
            f"invalid values: {values}"
        )

    prepared["strat_fold"] = numeric_folds.astype("int64")
    prepared["split"] = "train"
    prepared.loc[prepared["strat_fold"] == 9, "split"] = "validation"
    prepared.loc[prepared["strat_fold"] == 10, "split"] = "test"
    return prepared


def _find_overlaps_in_prepared_metadata(
    metadata: pd.DataFrame,
) -> dict[str, list[Any]]:
    patients_by_split = {
        split: set(metadata.loc[metadata["split"] == split, "patient_id"])
        for split in SPLIT_NAMES
    }

    overlaps = {}
    for left_split, right_split in SPLIT_PAIRS:
        pair = f"{left_split}/{right_split}"
        shared_patients = sorted(
            patients_by_split[left_split] & patients_by_split[right_split]
        )
        overlaps[pair] = [_to_python_scalar(value) for value in shared_patients]
    return overlaps


def _to_python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
