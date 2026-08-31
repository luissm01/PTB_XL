"""Train-only global standardization for PTB-XL signals."""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptbxl.data.reporting import write_json_report
from ptbxl.data.samples import ECGSample
from ptbxl.data.signals import (
    EXPECTED_LEADS,
    EXPECTED_SAMPLES,
    EXPECTED_SAMPLING_FREQUENCY_HZ,
    ECGSignal,
    validate_signal,
)


SCHEMA_VERSION = 1
METHOD = "global_standardization"
FITTED_SPLIT = "train"
DATASET_IDENTITY = {
    "name": "PTB-XL",
    "version": "1.0.3",
    "signal_source": "filename_lr",
    "sampling_frequency_hz": 100,
}
CONFIGURATION = {
    "scope": "all_train_signal_values",
    "variance_ddof": 0,
    "output_dtype": "float32",
}
SOURCE_FIELDS = (
    "cohort_sha256",
    "metadata_sha256",
    "signal_manifest_sha256",
)


@dataclass(frozen=True)
class GlobalStandardizer:
    """Frozen global mean and population standard deviation fitted on train."""

    mean: float
    standard_deviation: float
    record_count: int
    value_count: int
    lead_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.mean, bool) or not isinstance(
            self.mean, (int, float, np.integer, np.floating)
        ):
            raise ValueError("Standardizer mean must be finite")
        if not math.isfinite(float(self.mean)):
            raise ValueError("Standardizer mean must be finite")
        if isinstance(self.standard_deviation, bool) or not isinstance(
            self.standard_deviation, (int, float, np.integer, np.floating)
        ):
            raise ValueError("Standardizer standard deviation must be positive")
        if (
            not math.isfinite(float(self.standard_deviation))
            or self.standard_deviation <= 0
        ):
            raise ValueError("Standardizer standard deviation must be positive")
        if not _is_positive_integer(self.record_count) or not _is_positive_integer(
            self.value_count
        ):
            raise ValueError("Standardizer counts must be positive")
        expected_values = self.record_count * EXPECTED_SAMPLES * EXPECTED_LEADS
        if self.value_count != expected_values:
            raise ValueError(
                f"Standardizer value count must equal {expected_values} for "
                f"{self.record_count} records"
            )
        if (
            len(self.lead_names) != EXPECTED_LEADS
            or len(set(self.lead_names)) != EXPECTED_LEADS
            or any(not isinstance(name, str) or not name for name in self.lead_names)
        ):
            raise ValueError("Standardizer must contain 12 unique lead names")

    def transform(
        self,
        signal: np.ndarray,
        lead_names: Sequence[str],
    ) -> np.ndarray:
        """Apply frozen parameters while preserving the canonical signal shape."""
        record = ECGSignal(
            signal=np.asarray(signal),
            sampling_frequency=EXPECTED_SAMPLING_FREQUENCY_HZ,
            lead_names=tuple(lead_names),
        )
        validate_signal(record, self.lead_names)
        transformed = (
            record.signal.astype(np.float64, copy=False) - self.mean
        ) / self.standard_deviation
        output = transformed.astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Standardized signal contains non-finite values")
        return output


def fit_global_standardizer(samples: Iterable[ECGSample]) -> GlobalStandardizer:
    """Fit global moments sequentially from train samples only."""
    record_count = 0
    value_count = 0
    mean = 0.0
    second_central_moment = 0.0
    expected_lead_names: tuple[str, ...] | None = None

    for sample in samples:
        if sample.split != FITTED_SPLIT:
            raise ValueError(
                "Global standardizer fit accepts train samples only; "
                f"got split {sample.split!r} for ecg_id {sample.ecg_id}"
            )

        record = ECGSignal(
            signal=sample.signal,
            sampling_frequency=sample.sampling_frequency,
            lead_names=sample.lead_names,
        )
        validate_signal(record, expected_lead_names)
        if expected_lead_names is None:
            expected_lead_names = record.lead_names

        values = record.signal.astype(np.float64, copy=False).reshape(-1)
        batch_count = values.size
        batch_mean = float(values.mean(dtype=np.float64))
        deviations = values - batch_mean
        batch_moment = float(np.sum(deviations * deviations, dtype=np.float64))

        if value_count == 0:
            mean = batch_mean
            second_central_moment = batch_moment
            value_count = batch_count
        else:
            combined_count = value_count + batch_count
            delta = batch_mean - mean
            second_central_moment += batch_moment + (
                delta * delta * value_count * batch_count / combined_count
            )
            mean += delta * batch_count / combined_count
            value_count = combined_count
        record_count += 1

    if record_count == 0 or expected_lead_names is None:
        raise ValueError("Cannot fit global standardizer from an empty sample stream")

    variance = second_central_moment / value_count
    standard_deviation = math.sqrt(variance)
    return GlobalStandardizer(
        mean=float(mean),
        standard_deviation=float(standard_deviation),
        record_count=record_count,
        value_count=value_count,
        lead_names=expected_lead_names,
    )


def save_global_standardizer(
    standardizer: GlobalStandardizer,
    path: str | Path,
    sources: Mapping[str, str],
) -> None:
    """Write a deterministic versioned artifact with source provenance."""
    missing = [field for field in SOURCE_FIELDS if field not in sources]
    if missing:
        raise KeyError(f"Missing standardizer source fields: {missing}")
    normalized_sources = {
        field: _validate_sha256(field, sources[field]) for field in SOURCE_FIELDS
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_IDENTITY,
        "method": METHOD,
        "fitted_split": FITTED_SPLIT,
        "configuration": CONFIGURATION,
        "lead_names": list(standardizer.lead_names),
        "statistics": {
            "mean": float(standardizer.mean),
            "standard_deviation": float(standardizer.standard_deviation),
            "records": int(standardizer.record_count),
            "values": int(standardizer.value_count),
        },
        "sources": normalized_sources,
    }
    write_json_report(artifact, path)


def load_global_standardizer(path: str | Path) -> GlobalStandardizer:
    """Load and validate a versioned global-standardization artifact."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("Global standardizer artifact must contain a JSON object")
    _require_artifact_value(artifact, "schema_version", SCHEMA_VERSION)
    _require_artifact_value(artifact, "dataset", DATASET_IDENTITY)
    _require_artifact_value(artifact, "method", METHOD)
    _require_artifact_value(artifact, "fitted_split", FITTED_SPLIT)
    _require_artifact_value(artifact, "configuration", CONFIGURATION)

    try:
        serialized_lead_names = artifact["lead_names"]
        statistics = artifact["statistics"]
        sources = artifact["sources"]
        if not isinstance(serialized_lead_names, list) or not isinstance(
            statistics, dict
        ):
            raise ValueError("Artifact leads and statistics must be JSON containers")
        lead_names = tuple(serialized_lead_names)
        standardizer = GlobalStandardizer(
            mean=_artifact_float(statistics, "mean"),
            standard_deviation=_artifact_float(
                statistics,
                "standard_deviation",
            ),
            record_count=_artifact_positive_integer(statistics, "records"),
            value_count=_artifact_positive_integer(statistics, "values"),
            lead_names=lead_names,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Global standardizer artifact fields are invalid") from exc

    if not isinstance(sources, dict):
        raise ValueError("Global standardizer sources must contain a JSON object")
    missing = [field for field in SOURCE_FIELDS if field not in sources]
    if missing:
        raise ValueError(f"Global standardizer sources are missing fields: {missing}")
    for field in SOURCE_FIELDS:
        _validate_sha256(field, sources[field])
    return standardizer


def _require_artifact_value(
    artifact: Mapping[str, Any],
    field: str,
    expected: Any,
) -> None:
    if artifact.get(field) != expected:
        raise ValueError(
            f"Global standardizer {field} must be {expected!r}, "
            f"got {artifact.get(field)!r}"
        )


def _validate_sha256(field: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Standardizer source {field} must be a SHA-256 digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"Standardizer source {field} must be a SHA-256 digest")
    return normalized


def _artifact_float(values: Mapping[str, Any], field: str) -> float:
    value = values[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Standardizer statistic {field} must be numeric")
    return float(value)


def _artifact_positive_integer(values: Mapping[str, Any], field: str) -> int:
    value = values[field]
    if not _is_positive_integer(value):
        raise ValueError(f"Standardizer statistic {field} must be a positive integer")
    return int(value)


def _is_positive_integer(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and value > 0
    )
