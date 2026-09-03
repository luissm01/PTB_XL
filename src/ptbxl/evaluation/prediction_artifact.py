"""Safe local persistence for row-level prediction sets."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptbxl.evaluation.multilabel import PredictionSet


PREDICTION_ARTIFACT_SCHEMA_VERSION = 1
ALLOWED_PREDICTION_SPLITS = frozenset({"validation", "test"})


@dataclass(frozen=True)
class LoadedPredictionArtifact:
    """A validated split identity and its prediction payload."""

    split: str
    predictions: PredictionSet


def save_prediction_artifact(
    path: str | Path,
    predictions: PredictionSet,
    *,
    split: str,
) -> None:
    """Atomically save numeric arrays in NPZ without pickle objects."""
    destination = _validated_path(path)
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    _validate_split(split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(PREDICTION_ARTIFACT_SCHEMA_VERSION),
                split=np.asarray(split),
                ecg_ids=np.asarray(predictions.ecg_ids, dtype=np.int64),
                targets=predictions.targets,
                probabilities=predictions.probabilities,
                batches=np.asarray(predictions.batches),
            )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_prediction_artifact(path: str | Path) -> LoadedPredictionArtifact:
    """Load and validate a numeric NPZ prediction artifact without pickle."""
    source = _validated_path(path)
    try:
        with np.load(source, allow_pickle=False) as raw:
            expected = {
                "schema_version",
                "split",
                "ecg_ids",
                "targets",
                "probabilities",
                "batches",
            }
            actual = set(raw.files)
            if missing := sorted(expected - actual):
                raise ValueError(f"Prediction artifact is missing arrays: {missing}")
            if unexpected := sorted(actual - expected):
                raise ValueError(
                    f"Prediction artifact has unexpected arrays: {unexpected}"
                )
            schema_version = _scalar_value(raw["schema_version"], "schema_version")
            split = _scalar_value(raw["split"], "split")
            batches = _scalar_value(raw["batches"], "batches")
            ecg_ids = np.array(raw["ecg_ids"], copy=True)
            targets = np.array(raw["targets"], copy=True)
            probabilities = np.array(raw["probabilities"], copy=True)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "Prediction artifact"
        ):
            raise
        raise ValueError("Could not load prediction artifact safely") from error
    if schema_version != PREDICTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Prediction artifact schema_version is invalid")
    if not isinstance(split, str):
        raise TypeError("Prediction artifact split must be a string")
    _validate_split(split)
    if ecg_ids.ndim != 1:
        raise ValueError("Prediction artifact ecg_ids must be one-dimensional")
    predictions = PredictionSet(
        ecg_ids=tuple(ecg_ids.tolist()),
        targets=targets,
        probabilities=probabilities,
        batches=batches,
    )
    return LoadedPredictionArtifact(split=split, predictions=predictions)


def _validated_path(path: str | Path) -> Path:
    value = Path(path)
    if value.suffix != ".npz":
        raise ValueError("Prediction artifact path must end in .npz")
    return value


def _validate_split(split: str) -> None:
    if split not in ALLOWED_PREDICTION_SPLITS:
        raise ValueError(
            f"Prediction artifact split must be one of "
            f"{sorted(ALLOWED_PREDICTION_SPLITS)}"
        )


def _scalar_value(value: np.ndarray, name: str) -> Any:
    if value.ndim != 0:
        raise ValueError(f"Prediction artifact {name} must be scalar")
    return value.item()
