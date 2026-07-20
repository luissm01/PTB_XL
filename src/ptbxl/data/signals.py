"""Load and audit PTB-XL low-resolution WFDB signals."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb


EXPECTED_SAMPLING_FREQUENCY_HZ = 100.0
EXPECTED_SAMPLES = 1_000
EXPECTED_LEADS = 12


class SignalLoadError(ValueError):
    """Raised when an existing WFDB record cannot be read."""


@dataclass(frozen=True)
class ECGSignal:
    """A loaded ECG and the WFDB header facts needed to validate it."""

    signal: np.ndarray
    sampling_frequency: float
    lead_names: tuple[str, ...]


def load_signal_for_row(
    row: Mapping[str, Any],
    dataset_root: str | Path,
) -> ECGSignal:
    """Load the exact low-resolution WFDB basename associated with one ECG row."""
    missing = [field for field in ("ecg_id", "filename_lr") if field not in row]
    if missing:
        raise KeyError(f"Missing signal row fields: {missing}")

    ecg_id = _coerce_ecg_id(row["ecg_id"])
    record_path = _resolve_record_path(row["filename_lr"], dataset_root, ecg_id)
    header_path = Path(f"{record_path}.hea")
    if not header_path.is_file():
        raise FileNotFoundError(
            f"WFDB header not found for ecg_id {ecg_id}: {header_path}"
        )

    try:
        signal, fields = wfdb.rdsamp(str(record_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"WFDB record file missing for ecg_id {ecg_id}: {record_path}"
        ) from exc
    except Exception as exc:
        raise SignalLoadError(
            f"Could not read WFDB record for ecg_id {ecg_id}: {record_path}"
        ) from exc

    try:
        sampling_frequency = float(fields["fs"])
        lead_names = tuple(str(name) for name in fields["sig_name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SignalLoadError(
            f"WFDB header fields are invalid for ecg_id {ecg_id}: {record_path}"
        ) from exc

    return ECGSignal(
        signal=np.asarray(signal),
        sampling_frequency=sampling_frequency,
        lead_names=lead_names,
    )


def validate_signal(
    record: ECGSignal,
    expected_lead_names: Sequence[str] | None = None,
) -> ECGSignal:
    """Validate the fixed technical contract for one 100 Hz PTB-XL ECG."""
    signal = record.signal
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal values must be a NumPy array")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("Signal values must be numeric")
    if signal.shape != (EXPECTED_SAMPLES, EXPECTED_LEADS):
        raise ValueError(
            "Signal shape must be "
            f"({EXPECTED_SAMPLES}, {EXPECTED_LEADS}), got {signal.shape}"
        )
    if record.sampling_frequency != EXPECTED_SAMPLING_FREQUENCY_HZ:
        raise ValueError(
            "Sampling frequency must be "
            f"{EXPECTED_SAMPLING_FREQUENCY_HZ:g} Hz, got "
            f"{record.sampling_frequency:g} Hz"
        )
    if len(record.lead_names) != EXPECTED_LEADS:
        raise ValueError(
            f"Signal must expose {EXPECTED_LEADS} lead names, "
            f"got {len(record.lead_names)}"
        )
    if len(set(record.lead_names)) != EXPECTED_LEADS:
        raise ValueError("Signal lead names must be unique")
    non_finite = int(signal.size - np.count_nonzero(np.isfinite(signal)))
    if non_finite:
        raise ValueError(f"Signal contains {non_finite} non-finite values")

    if expected_lead_names is not None:
        expected = tuple(expected_lead_names)
        if record.lead_names != expected:
            raise ValueError(
                f"Unexpected lead order: expected {expected}, got {record.lead_names}"
            )
    return record


def audit_lr_signals(
    cohort: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Inspect each cohort signal sequentially and return aggregate evidence."""
    signal_rows = _associate_signal_paths(cohort, metadata)
    shapes: Counter[str] = Counter()
    frequencies: Counter[str] = Counter()
    lead_orders: Counter[str] = Counter()
    loaded = 0
    missing = 0
    invalid = 0
    non_finite_values = 0
    expected_lead_names: tuple[str, ...] | None = None

    for row in signal_rows.itertuples(index=False):
        signal_row = {"ecg_id": row.ecg_id, "filename_lr": row.filename_lr}
        try:
            record = load_signal_for_row(signal_row, dataset_root)
        except FileNotFoundError:
            missing += 1
            continue
        except (SignalLoadError, TypeError, ValueError):
            invalid += 1
            continue

        shapes[_shape_key(record.signal.shape)] += 1
        frequencies[_number_key(record.sampling_frequency)] += 1
        lead_orders[",".join(record.lead_names)] += 1
        if isinstance(record.signal, np.ndarray) and np.issubdtype(
            record.signal.dtype, np.number
        ):
            non_finite_values += int(
                record.signal.size - np.count_nonzero(np.isfinite(record.signal))
            )

        try:
            validate_signal(record, expected_lead_names)
        except (TypeError, ValueError):
            invalid += 1
            continue

        if expected_lead_names is None:
            expected_lead_names = record.lead_names
        loaded += 1

    return {
        "dataset": {
            "name": "PTB-XL",
            "version": "1.0.3",
            "signal_source": "filename_lr",
            "expected_sampling_frequency_hz": 100,
        },
        "records": {
            "expected": len(signal_rows),
            "loaded": loaded,
            "missing": missing,
            "invalid": invalid,
        },
        "shapes": dict(sorted(shapes.items())),
        "sampling_frequencies": dict(sorted(frequencies.items())),
        "lead_orders": dict(sorted(lead_orders.items())),
        "non_finite_values": non_finite_values,
    }


def _associate_signal_paths(
    cohort: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    for name, table, required in (
        ("cohort", cohort, ("ecg_id",)),
        ("metadata", metadata, ("ecg_id", "filename_lr")),
    ):
        missing = [column for column in required if column not in table]
        if missing:
            raise KeyError(f"Missing {name} signal columns: {missing}")
        if table.loc[:, required].isna().any().any():
            raise ValueError(f"{name.capitalize()} signal columns contain null values")
        if table["ecg_id"].duplicated().any():
            raise ValueError(f"{name.capitalize()} contains duplicate ecg_id values")

    if metadata["filename_lr"].duplicated().any():
        raise ValueError("Metadata contains duplicate filename_lr values")

    associated = cohort.loc[:, ["ecg_id"]].merge(
        metadata.loc[:, ["ecg_id", "filename_lr"]],
        on="ecg_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if associated["filename_lr"].isna().any():
        missing_ids = associated.loc[
            associated["filename_lr"].isna(), "ecg_id"
        ].tolist()
        raise ValueError(f"Cohort ecg_id values missing from metadata: {missing_ids}")
    return associated


def _coerce_ecg_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"ecg_id must be an integer, got {value!r}")
    try:
        ecg_id = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"ecg_id must be an integer, got {value!r}") from exc
    if ecg_id != value or ecg_id < 1:
        raise ValueError(f"ecg_id must be a positive integer, got {value!r}")
    return ecg_id


def _resolve_record_path(
    filename_lr: Any,
    dataset_root: str | Path,
    ecg_id: int,
) -> Path:
    if not isinstance(filename_lr, str) or not filename_lr.strip():
        raise ValueError("filename_lr must be a non-empty relative WFDB basename")
    relative_path = Path(filename_lr)
    if relative_path.is_absolute() or relative_path.suffix:
        raise ValueError(
            "filename_lr must be a relative WFDB basename without an extension"
        )
    if relative_path.name != f"{ecg_id:05d}_lr":
        raise ValueError(f"filename_lr does not match ecg_id {ecg_id}: {filename_lr}")

    root = Path(dataset_root).resolve()
    record_path = (root / relative_path).resolve()
    if not record_path.is_relative_to(root):
        raise ValueError("filename_lr resolves outside dataset_root")
    return record_path


def _shape_key(shape: tuple[int, ...]) -> str:
    return "x".join(str(dimension) for dimension in shape)


def _number_key(value: float) -> str:
    return f"{value:g}"
