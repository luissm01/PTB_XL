import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from ptbxl.training import (
    evaluate_loss,
    resolve_device,
    seed_random_generators,
    train_one_epoch,
)


def _batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "signal": torch.tensor([[1.0, -1.0], [0.5, 0.25]]),
            "targets": torch.tensor([[1.0], [0.0]]),
        },
        {
            "signal": torch.tensor([[-0.5, 0.75]]),
            "targets": torch.tensor([[1.0]]),
        },
    ]


def test_seed_resets_python_numpy_and_torch() -> None:
    seed_random_generators(2026)
    first = (random.random(), np.random.random(), torch.rand(1))
    seed_random_generators(2026)
    second = (random.random(), np.random.random(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


@pytest.mark.parametrize("seed", [True, -1, 2**32])
def test_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        seed_random_generators(seed)  # type: ignore[arg-type]


def test_resolves_cpu_and_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


def test_train_updates_parameters_and_returns_finite_weighted_loss() -> None:
    seed_random_generators(2026)
    model = nn.Linear(2, 1)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_epoch(
        model,
        _batches(),
        optimizer,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
    )

    assert result.samples == 3
    assert result.batches == 2
    assert np.isfinite(result.loss)
    assert model.training
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_evaluate_is_sample_weighted_and_does_not_update_parameters() -> None:
    model = nn.Linear(2, 1)
    before = copy.deepcopy(model.state_dict())
    loss_function = nn.BCEWithLogitsLoss()
    batches = _batches()
    with torch.no_grad():
        all_signal = torch.cat([batch["signal"] for batch in batches])
        all_targets = torch.cat([batch["targets"] for batch in batches])
        expected = float(loss_function(model(all_signal), all_targets).item())

    result = evaluate_loss(model, batches, loss_function, torch.device("cpu"))

    assert result.loss == pytest.approx(expected)
    assert result.samples == 3
    assert result.batches == 2
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_rejects_empty_and_invalid_batches() -> None:
    model = nn.Linear(2, 1)
    loss = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="empty loader"):
        evaluate_loss(model, [], loss, device)
    with pytest.raises(KeyError, match="missing fields"):
        evaluate_loss(model, [{"signal": torch.ones(1, 2)}], loss, device)
    with pytest.raises(ValueError, match="counts must match"):
        evaluate_loss(
            model,
            [{"signal": torch.ones(2, 2), "targets": torch.ones(1, 1)}],
            loss,
            device,
        )


def test_rejects_non_finite_or_non_scalar_loss() -> None:
    model = nn.Linear(2, 1)
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="finite"):
        evaluate_loss(
            model,
            _batches(),
            lambda outputs, targets: outputs.sum() * torch.nan,
            device,
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one scalar"):
        evaluate_loss(model, _batches(), nn.BCEWithLogitsLoss(reduction="none"), device)
