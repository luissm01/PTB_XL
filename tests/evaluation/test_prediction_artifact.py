from pathlib import Path

import numpy as np
import pytest

from ptbxl.evaluation import (
    PredictionSet,
    load_prediction_artifact,
    save_prediction_artifact,
)


def _predictions() -> PredictionSet:
    targets = np.asarray(
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        dtype=np.int8,
    )
    return PredictionSet(
        ecg_ids=(11, 22),
        targets=targets,
        probabilities=np.asarray(targets * 0.8 + 0.1, dtype=np.float64),
        batches=1,
    )


def test_prediction_artifact_round_trips_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"

    save_prediction_artifact(path, _predictions(), split="test")
    loaded = load_prediction_artifact(path)

    assert loaded.split == "test"
    assert loaded.predictions.ecg_ids == (11, 22)
    assert loaded.predictions.batches == 1
    np.testing.assert_array_equal(loaded.predictions.targets, _predictions().targets)
    np.testing.assert_array_equal(
        loaded.predictions.probabilities,
        _predictions().probabilities,
    )
    with np.load(path, allow_pickle=False) as raw:
        assert all(raw[name].dtype != object for name in raw.files)


def test_prediction_artifact_rejects_invalid_path_split_and_arrays(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="end in .npz"):
        save_prediction_artifact(
            tmp_path / "predictions.json", _predictions(), split="test"
        )
    with pytest.raises(ValueError, match="one of"):
        save_prediction_artifact(
            tmp_path / "predictions.npz", _predictions(), split="train"
        )

    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, schema_version=np.asarray(1))
    with pytest.raises(ValueError, match="missing arrays"):
        load_prediction_artifact(malformed)


def test_prediction_artifact_refuses_pickled_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(
        path,
        schema_version=np.asarray(1),
        split=np.asarray("test"),
        ecg_ids=np.asarray([object()], dtype=object),
        targets=np.zeros((1, 5), dtype=np.int8),
        probabilities=np.zeros((1, 5), dtype=np.float64),
        batches=np.asarray(1),
    )

    with pytest.raises(ValueError, match="safely"):
        load_prediction_artifact(path)
