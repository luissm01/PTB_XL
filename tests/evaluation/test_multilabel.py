from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.evaluation import (
    PredictionSet,
    collect_validation_predictions,
    compute_ranking_metrics,
    evaluate_validation,
)


TARGETS = torch.tensor(
    [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ],
    dtype=torch.float32,
)


class SyntheticEvaluationDataset(Dataset[dict[str, Any]]):
    def __init__(self, split: str = "validation", targets: torch.Tensor = TARGETS):
        self.split = split
        self.targets = targets
        self.signals = targets * 8.0 - 4.0

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "signal": self.signals[index],
            "targets": self.targets[index],
            "ecg_id": 101 + index,
        }


def _identity_model() -> nn.Linear:
    model = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(5))
    return model


def _prediction_set(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> PredictionSet:
    return PredictionSet(
        ecg_ids=tuple(range(1, len(targets) + 1)),
        targets=targets,
        probabilities=probabilities,
        batches=1,
    )


def test_collects_ordered_validation_predictions_without_gradients() -> None:
    model = _identity_model()
    loader = DataLoader(SyntheticEvaluationDataset(), batch_size=2, shuffle=False)

    predictions = collect_validation_predictions(model, loader, torch.device("cpu"))

    assert predictions.ecg_ids == (101, 102, 103, 104, 105, 106)
    assert predictions.batches == 3
    np.testing.assert_array_equal(predictions.targets, TARGETS.numpy())
    np.testing.assert_allclose(
        predictions.probabilities,
        torch.sigmoid(TARGETS * 8.0 - 4.0).numpy(),
        rtol=0,
        atol=0,
    )
    assert not predictions.targets.flags.writeable
    assert not predictions.probabilities.flags.writeable
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_complete_validation_evaluation_is_perfect_and_ordered() -> None:
    result = evaluate_validation(
        _identity_model(),
        DataLoader(SyntheticEvaluationDataset(), batch_size=4, shuffle=False),
        torch.device("cpu"),
    )

    assert result.metrics.samples == len(TARGETS)
    assert tuple(item.label for item in result.metrics.per_class) == TARGET_SUPERCLASSES
    assert result.metrics.macro_auroc == 1.0
    assert result.metrics.macro_auprc == 1.0
    assert result.metrics.micro_auroc == 1.0
    assert result.metrics.micro_auprc == 1.0
    assert all(item.auroc == item.auprc == 1.0 for item in result.metrics.per_class)


def test_tied_scores_have_chance_auroc_and_prevalence_auprc() -> None:
    targets = TARGETS.numpy().astype(np.int8)
    probabilities = np.full(targets.shape, 0.5)

    metrics = compute_ranking_metrics(_prediction_set(targets, probabilities))

    prevalences = np.array([item.prevalence for item in metrics.per_class])
    assert all(item.auroc == 0.5 for item in metrics.per_class)
    np.testing.assert_allclose(
        [item.auprc for item in metrics.per_class], prevalences, rtol=0, atol=0
    )
    assert metrics.macro_auroc == 0.5
    assert metrics.macro_auprc == pytest.approx(float(prevalences.mean()))
    assert metrics.micro_auroc == 0.5
    assert metrics.micro_auprc == pytest.approx(float(targets.mean()))


def test_auprc_uses_non_interpolated_average_precision() -> None:
    binary_targets = np.array([0, 0, 1, 1], dtype=np.int8)
    binary_scores = np.array([0.1, 0.4, 0.35, 0.8])
    targets = np.repeat(binary_targets[:, None], 5, axis=1)
    probabilities = np.repeat(binary_scores[:, None], 5, axis=1)

    metrics = compute_ranking_metrics(_prediction_set(targets, probabilities))

    assert metrics.macro_auroc == 0.75
    assert metrics.macro_auprc == pytest.approx(5 / 6)
    assert all(item.auprc == pytest.approx(5 / 6) for item in metrics.per_class)
    assert metrics.macro_auroc == pytest.approx(
        np.mean([item.auroc for item in metrics.per_class])
    )
    assert metrics.macro_auprc == pytest.approx(
        np.mean([item.auprc for item in metrics.per_class])
    )


@pytest.mark.parametrize("split", ["train", "test"])
def test_rejects_non_validation_loader_before_model_execution(split: str) -> None:
    class FailIfCalled(nn.Module):
        def forward(self, signal: torch.Tensor) -> torch.Tensor:
            raise AssertionError("model must not run")

    loader = DataLoader(SyntheticEvaluationDataset(split), batch_size=2)

    with pytest.raises(ValueError, match="requires a validation Dataset"):
        collect_validation_predictions(FailIfCalled(), loader, torch.device("cpu"))


def test_rejects_empty_or_malformed_prediction_batches() -> None:
    empty_loader = DataLoader(
        SyntheticEvaluationDataset(targets=torch.empty((0, 5))), batch_size=2
    )
    with pytest.raises(ValueError, match="empty loader"):
        collect_validation_predictions(
            _identity_model(), empty_loader, torch.device("cpu")
        )

    class DeclaredLoader(list[dict[str, Any]]):
        dataset = SimpleNamespace(split="validation")

    missing_id = DeclaredLoader([{"signal": torch.ones(1, 5), "targets": TARGETS[:1]}])
    with pytest.raises(KeyError, match="missing fields"):
        collect_validation_predictions(
            _identity_model(), missing_id, torch.device("cpu")
        )


def test_rejects_duplicate_ids_nonbinary_targets_and_nonfinite_logits() -> None:
    class DeclaredLoader(list[dict[str, Any]]):
        dataset = SimpleNamespace(split="validation")

    duplicate_ids = DeclaredLoader(
        [
            {
                "signal": torch.ones(2, 5),
                "targets": TARGETS[:2],
                "ecg_id": [7, 7],
            }
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        collect_validation_predictions(
            _identity_model(), duplicate_ids, torch.device("cpu")
        )

    nonbinary = TARGETS[:2].clone()
    nonbinary[0, 0] = 0.5
    nonbinary_loader = DeclaredLoader(
        [{"signal": torch.ones(2, 5), "targets": nonbinary, "ecg_id": [1, 2]}]
    )
    with pytest.raises(ValueError, match="binary"):
        collect_validation_predictions(
            _identity_model(), nonbinary_loader, torch.device("cpu")
        )

    class NonFiniteModel(nn.Module):
        def forward(self, signal: torch.Tensor) -> torch.Tensor:
            return torch.full((len(signal), 5), torch.nan)

    with pytest.raises(ValueError, match="logits must be finite"):
        collect_validation_predictions(
            NonFiniteModel(),
            DeclaredLoader(
                [
                    {
                        "signal": torch.ones(2, 5),
                        "targets": TARGETS[:2],
                        "ecg_id": [1, 2],
                    }
                ]
            ),
            torch.device("cpu"),
        )

    class WrongShapeModel(nn.Module):
        def forward(self, signal: torch.Tensor) -> torch.Tensor:
            return torch.ones((len(signal), 4))

    with pytest.raises(ValueError, match="equal shapes"):
        collect_validation_predictions(
            WrongShapeModel(),
            DeclaredLoader(
                [
                    {
                        "signal": torch.ones(2, 5),
                        "targets": TARGETS[:2],
                        "ecg_id": [1, 2],
                    }
                ]
            ),
            torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "targets, probabilities, ecg_ids, error",
    [
        (
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            (1, 2),
            "target columns",
        ),
        (
            np.zeros((2, 5)),
            np.zeros((3, 5)),
            (1, 2),
            "equal shapes",
        ),
        (
            np.full((2, 5), 0.5),
            np.zeros((2, 5)),
            (1, 2),
            "binary",
        ),
        (
            np.zeros((2, 5)),
            np.full((2, 5), np.inf),
            (1, 2),
            "finite values",
        ),
        (
            np.zeros((2, 5)),
            np.full((2, 5), 1.1),
            (1, 2),
            "finite values",
        ),
        (
            np.zeros((2, 5)),
            np.zeros((2, 5)),
            (1, 1),
            "unique",
        ),
    ],
)
def test_prediction_set_rejects_invalid_values(
    targets: np.ndarray,
    probabilities: np.ndarray,
    ecg_ids: tuple[int, ...],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        PredictionSet(ecg_ids, targets, probabilities, batches=1)


def test_prediction_set_owns_defensive_copies() -> None:
    targets = TARGETS.numpy().copy()
    probabilities = np.full(targets.shape, 0.5)
    predictions = _prediction_set(targets, probabilities)

    targets[0, 0] = 0
    probabilities[0, 0] = 0.9

    assert predictions.targets[0, 0] == 1
    assert predictions.probabilities[0, 0] == 0.5


def test_rejects_undefined_per_class_ranking_metrics() -> None:
    targets = TARGETS.numpy().astype(np.int8)
    targets[:, 4] = 0
    predictions = _prediction_set(targets, np.full(targets.shape, 0.5))

    with pytest.raises(ValueError, match="positive and negative targets for HYP"):
        compute_ranking_metrics(predictions)
