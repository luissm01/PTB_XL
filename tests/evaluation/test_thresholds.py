import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.evaluation import (
    DECISION_RULE,
    THRESHOLD_METHOD,
    TIE_BREAK_RULE,
    PredictionSet,
    ThresholdSet,
    compute_operating_metrics,
    fingerprint_predictions,
    load_frozen_thresholds,
    select_validation_thresholds,
)


def _predictions(
    binary_targets: list[int],
    probabilities: list[float],
) -> PredictionSet:
    targets = np.repeat(np.asarray(binary_targets, dtype=np.int8)[:, None], 5, axis=1)
    scores = np.repeat(np.asarray(probabilities, dtype=np.float64)[:, None], 5, axis=1)
    return PredictionSet(
        ecg_ids=tuple(range(1, len(targets) + 1)),
        targets=targets,
        probabilities=scores,
        batches=1,
    )


def _artifact() -> dict[str, object]:
    source = {"path": "source.file", "sha256": "a" * 64}
    thresholds = ThresholdSet(TARGET_SUPERCLASSES, (0.5,) * 5)
    metrics = compute_operating_metrics(
        _predictions([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]), thresholds
    )
    return {
        "schema_version": 1,
        "selection": {
            "name": "synthetic_per_class_f1",
            "git_commit": "b" * 40,
            "method": THRESHOLD_METHOD,
            "decision_rule": DECISION_RULE,
            "tie_break": TIE_BREAK_RULE,
            "thresholds": {label: 0.5 for label in TARGET_SUPERCLASSES},
        },
        "dataset": {
            "name": "PTB-XL",
            "version": "1.0.3",
            "cohort": "synthetic",
            "split": "validation",
            "samples": 4,
            "targets": list(TARGET_SUPERCLASSES),
        },
        "sources": {
            "selection_config": source,
            "experiment_config": source,
            "experiment_report": source,
            "checkpoint": source,
            "standardizer": source,
            "validation_predictions_sha256": "c" * 64,
        },
        "runtime": {
            "python": "3.11",
            "numpy": "test",
            "scikit_learn": "test",
            "torch": "test",
            "cuda": None,
            "hardware": {},
            "deterministic_algorithms": True,
        },
        "validation": {
            "samples": 4,
            "per_class": [asdict(item) for item in metrics.per_class],
            "macro": asdict(metrics.macro),
            "micro": asdict(metrics.micro),
        },
        "limitations": ["Synthetic test artifact."],
    }


def test_selects_unique_per_class_f1_maximum() -> None:
    predictions = _predictions([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])

    result = select_validation_thresholds(predictions)

    assert result.thresholds.values == (0.8,) * 5
    assert result.thresholds.by_label() == {label: 0.8 for label in TARGET_SUPERCLASSES}
    assert result.metrics.macro.f1 == 1.0
    assert result.metrics.micro.f1 == 1.0
    assert result.validation_predictions_sha256 == fingerprint_predictions(predictions)


def test_equal_best_f1_chooses_highest_threshold() -> None:
    predictions = _predictions([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6])

    result = select_validation_thresholds(predictions)

    assert result.thresholds.values == (0.9,) * 5
    for item in result.metrics.per_class:
        assert item.true_positives == 1
        assert item.true_negatives == 2
        assert item.false_positives == 0
        assert item.false_negatives == 1
        assert item.precision == 1.0
        assert item.sensitivity == 0.5
        assert item.specificity == 1.0
        assert item.f1 == pytest.approx(2 / 3)
    assert result.metrics.macro.f1 == pytest.approx(2 / 3)
    assert result.metrics.micro.f1 == pytest.approx(2 / 3)


def test_operating_metrics_define_zero_precision_without_predictions() -> None:
    predictions = _predictions([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])
    thresholds = ThresholdSet(TARGET_SUPERCLASSES, (1.0,) * 5)

    metrics = compute_operating_metrics(predictions, thresholds)

    assert metrics.samples == 4
    assert all(item.predicted_positives == 0 for item in metrics.per_class)
    assert metrics.macro.precision == 0.0
    assert metrics.macro.sensitivity == 0.0
    assert metrics.macro.specificity == 1.0
    assert metrics.macro.f1 == 0.0
    assert metrics.micro.precision == 0.0


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ((0.5,) * 4, ValueError),
        ((0.5, 0.5, 0.5, 0.5, float("nan")), ValueError),
        ((0.5, 0.5, 0.5, 0.5, 1.1), ValueError),
        ((0.5, 0.5, 0.5, 0.5, True), TypeError),
    ],
)
def test_rejects_invalid_threshold_values(
    values: tuple[float, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ThresholdSet(TARGET_SUPERCLASSES, values)


def test_rejects_wrong_labels_method_and_degenerate_targets() -> None:
    with pytest.raises(ValueError, match="fixed order"):
        ThresholdSet(tuple(reversed(TARGET_SUPERCLASSES)), (0.5,) * 5)
    with pytest.raises(ValueError, match="method"):
        ThresholdSet(TARGET_SUPERCLASSES, (0.5,) * 5, method="unknown")
    degenerate = _predictions([1, 1, 1, 1], [0.9, 0.8, 0.7, 0.6])
    with pytest.raises(ValueError, match="positive and negative"):
        select_validation_thresholds(degenerate)


def test_prediction_fingerprint_is_order_and_value_sensitive() -> None:
    original = _predictions([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])
    repeated = _predictions([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])
    changed = _predictions([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.2])

    assert fingerprint_predictions(original) == fingerprint_predictions(repeated)
    assert fingerprint_predictions(original) != fingerprint_predictions(changed)


def test_loads_artifact_and_requires_checkpoint_binding(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    frozen = load_frozen_thresholds(path, expected_checkpoint_sha256="a" * 64)

    assert frozen.name == "synthetic_per_class_f1"
    assert frozen.thresholds.values == (0.5,) * 5
    assert frozen.validation_samples == 4
    assert frozen.checkpoint_sha256 == "a" * 64
    with pytest.raises(ValueError, match="does not match"):
        load_frozen_thresholds(path, expected_checkpoint_sha256="d" * 64)


def test_artifact_loader_rejects_unknown_fields_and_nonvalidation(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    artifact["unknown"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        load_frozen_thresholds(path)

    artifact = _artifact()
    artifact["dataset"]["split"] = "test"  # type: ignore[index]
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="selected on validation"):
        load_frozen_thresholds(path)


def test_artifact_loader_rejects_metrics_inconsistent_with_counts(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    artifact["validation"]["per_class"][0]["f1"] = 0.25  # type: ignore[index]
    path = tmp_path / "inconsistent.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent with counts"):
        load_frozen_thresholds(path)
