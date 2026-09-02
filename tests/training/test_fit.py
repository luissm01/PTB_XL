import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ptbxl.training import (
    CheckpointProvenance,
    EpochResult,
    FitConfig,
    fit,
    load_training_checkpoint,
)


class SplitDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, split: str, count: int = 6) -> None:
        self.split = split
        self.signal = torch.linspace(-1.0, 1.0, count * 2).reshape(count, 2)
        self.targets = (self.signal.sum(dim=1, keepdim=True) > 0).to(torch.float32)

    def __len__(self) -> int:
        return len(self.signal)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"signal": self.signal[index], "targets": self.targets[index]}


def _loader(split: str) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(SplitDataset(split), batch_size=2, shuffle=False)


def _provenance(seed: int = 2026) -> CheckpointProvenance:
    return CheckpointProvenance(
        dataset_version="PTB-XL-1.0.3",
        cohort_name="five-superclass",
        preprocessing_sha256="a" * 64,
        model_name="linear-test-model",
        seed=seed,
        git_commit="abcdef0",
    )


def test_complete_fit_saves_full_history_and_round_trips_logits(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2026)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.1)
    checkpoint = tmp_path / "best.pt"

    result = fit(
        model,
        _loader("train"),
        _loader("validation"),
        optimizer,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        FitConfig(epochs=3),
        checkpoint,
        _provenance(),
    )
    with torch.no_grad():
        expected_logits = model(torch.tensor([[0.25, -0.5]]))

    assert checkpoint.is_file()
    assert len(result.history) == 3
    assert result.best_epoch in (1, 2, 3)
    assert result.best_validation_loss == min(
        item.validation.loss for item in result.history
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    loaded = load_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        device=torch.device("cpu"),
        expected_provenance=_provenance(),
    )
    with torch.no_grad():
        restored_logits = model(torch.tensor([[0.25, -0.5]]))

    assert loaded.epoch == result.best_epoch
    assert loaded.history == result.history
    torch.testing.assert_close(restored_logits, expected_logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "train_split, validation_split",
    [("validation", "validation"), ("train", "train"), ("train", "test")],
)
def test_fit_rejects_wrong_split_roles(
    tmp_path: Path,
    train_split: str,
    validation_split: str,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="requires"):
        fit(
            model,
            _loader(train_split),
            _loader(validation_split),
            optimizer,
            nn.BCEWithLogitsLoss(),
            torch.device("cpu"),
            FitConfig(epochs=1),
            tmp_path / "unused.pt",
            _provenance(),
        )


def test_first_minimum_validation_loss_wins_and_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_module = importlib.import_module("ptbxl.training.fit")
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    validation_losses = iter([0.5, 0.5, 0.7])

    def fake_train(*args: object, **kwargs: object) -> EpochResult:
        del args, kwargs
        with torch.no_grad():
            model.weight.add_(1)
        return EpochResult(1.0, 6, 3)

    def fake_evaluate(*args: object, **kwargs: object) -> EpochResult:
        del args, kwargs
        return EpochResult(next(validation_losses), 6, 3)

    monkeypatch.setattr(fit_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(fit_module, "evaluate_loss", fake_evaluate)

    result = fit_module.fit(
        model,
        _loader("train"),
        _loader("validation"),
        optimizer,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        FitConfig(epochs=3),
        tmp_path / "best.pt",
        _provenance(),
    )

    assert result.best_epoch == 1
    assert result.best_validation_loss == 0.5
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))


@pytest.mark.parametrize("epochs", [True, 0, -1])
def test_rejects_invalid_fit_config(epochs: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FitConfig(epochs=epochs)  # type: ignore[arg-type]


def test_rejects_invalid_or_mismatched_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "invalid.pt"
    torch.save({"schema_version": 2}, path)

    with pytest.raises(ValueError, match="schema_version"):
        load_training_checkpoint(
            path,
            model,
            optimizer,
            device=torch.device("cpu"),
        )

    valid_path = tmp_path / "valid.pt"
    fit(
        model,
        _loader("train"),
        _loader("validation"),
        optimizer,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        FitConfig(epochs=1),
        valid_path,
        _provenance(),
    )
    with pytest.raises(ValueError, match="provenance"):
        load_training_checkpoint(
            valid_path,
            model,
            optimizer,
            device=torch.device("cpu"),
            expected_provenance=_provenance(seed=7),
        )
