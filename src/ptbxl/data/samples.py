"""Compose validated PTB-XL identities, targets, splits and ECG signals."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ptbxl.data.cohort import ALLOWED_SPLITS, COHORT_COLUMNS
from ptbxl.data.labels import IDENTITY_COLUMNS, TARGET_SUPERCLASSES
from ptbxl.data.metadata import prepare_metadata
from ptbxl.data.signals import (
    associate_signal_paths,
    load_signal_for_row,
    validate_signal,
)


SAMPLE_INDEX_COLUMNS = (*COHORT_COLUMNS, "filename_lr")


@dataclass(frozen=True)
class ECGSample:
    """One framework-independent ECG sample with its provenance and targets."""

    target_names: ClassVar[tuple[str, ...]] = TARGET_SUPERCLASSES

    ecg_id: int
    patient_id: Any
    strat_fold: int
    split: str
    filename_lr: str
    signal: np.ndarray
    targets: np.ndarray
    sampling_frequency: float
    lead_names: tuple[str, ...]


def build_sample_index(
    cohort: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and transiently associate cohort rows with signal basenames."""
    missing = [column for column in COHORT_COLUMNS if column not in cohort]
    if missing:
        raise KeyError(f"Missing sample cohort columns: {missing}")

    source = cohort.loc[:, COHORT_COLUMNS].copy()
    _validate_targets(source)
    _validate_inherited_splits(source)
    signal_paths = associate_signal_paths(source, metadata)

    sample_index = source.copy()
    sample_index["filename_lr"] = signal_paths["filename_lr"].to_numpy(copy=True)
    return sample_index.loc[:, SAMPLE_INDEX_COLUMNS]


def load_sample(
    row: Mapping[str, Any],
    dataset_root: str | Path,
) -> ECGSample:
    """Load one indexed sample without preprocessing or framework conversion."""
    missing = [column for column in SAMPLE_INDEX_COLUMNS if column not in row]
    if missing:
        raise KeyError(f"Missing sample fields: {missing}")

    targets = _target_vector(row)
    strat_fold = _coerce_fold(row["strat_fold"])
    split = row["split"]
    expected_split = _split_for_fold(strat_fold)
    if split != expected_split:
        raise ValueError(
            f"Inherited split {split!r} does not match strat_fold {strat_fold} "
            f"({expected_split!r})"
        )

    signal_record = validate_signal(load_signal_for_row(row, dataset_root))
    return ECGSample(
        ecg_id=int(row["ecg_id"]),
        patient_id=_to_python_scalar(row["patient_id"]),
        strat_fold=strat_fold,
        split=str(split),
        filename_lr=str(row["filename_lr"]),
        signal=signal_record.signal,
        targets=targets,
        sampling_frequency=signal_record.sampling_frequency,
        lead_names=signal_record.lead_names,
    )


def _validate_targets(cohort: pd.DataFrame) -> None:
    if not cohort.loc[:, TARGET_SUPERCLASSES].isin([0, 1]).all().all():
        raise ValueError("Sample target labels must contain only 0 or 1")
    if cohort.loc[:, TARGET_SUPERCLASSES].sum(axis=1).lt(1).any():
        raise ValueError("Every sample cohort row must contain at least one target")


def _validate_inherited_splits(cohort: pd.DataFrame) -> None:
    if cohort["split"].isna().any():
        raise ValueError("Column 'split' contains null values")
    invalid = sorted(set(cohort["split"]) - set(ALLOWED_SPLITS))
    if invalid:
        raise ValueError(f"Invalid inherited split values: {invalid}")

    prepared = prepare_metadata(cohort.loc[:, IDENTITY_COLUMNS[:-1]])
    mismatch = cohort["split"].ne(prepared["split"])
    if mismatch.any():
        details = cohort.loc[mismatch, ["ecg_id", "strat_fold", "split"]].to_dict(
            "records"
        )
        raise ValueError(
            f"Inherited splits do not match official strat_fold assignments: {details}"
        )


def _target_vector(row: Mapping[str, Any]) -> np.ndarray:
    values = [row[label] for label in TARGET_SUPERCLASSES]
    if any(value not in (0, 1) for value in values):
        raise ValueError("Sample target labels must contain only 0 or 1")
    targets = np.asarray(values, dtype=np.float32)
    if not targets.any():
        raise ValueError("Every sample cohort row must contain at least one target")
    return targets


def _coerce_fold(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"strat_fold must be an integer from 1 to 10, got {value!r}")
    try:
        fold = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"strat_fold must be an integer from 1 to 10, got {value!r}"
        ) from exc
    if fold != value or not 1 <= fold <= 10:
        raise ValueError(f"strat_fold must be an integer from 1 to 10, got {value!r}")
    return fold


def _split_for_fold(fold: int) -> str:
    if fold <= 8:
        return "train"
    return "validation" if fold == 9 else "test"


def _to_python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
