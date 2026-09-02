"""Validation-selected multi-epoch fitting and safe checkpoint persistence."""

import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ptbxl.training.engine import EpochResult, evaluate_loss, train_one_epoch


CHECKPOINT_SCHEMA_VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")


@dataclass(frozen=True)
class FitConfig:
    """Immutable orchestration settings."""

    epochs: int

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True)
class CheckpointProvenance:
    """Minimum identities required to attribute a training checkpoint."""

    dataset_version: str
    cohort_name: str
    preprocessing_sha256: str
    model_name: str
    seed: int
    git_commit: str

    def __post_init__(self) -> None:
        for field in ("dataset_version", "cohort_name", "model_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if not isinstance(
            self.preprocessing_sha256, str
        ) or not SHA256_PATTERN.fullmatch(self.preprocessing_sha256):
            raise ValueError("preprocessing_sha256 must be a lowercase SHA-256")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.git_commit, str) or not GIT_COMMIT_PATTERN.fullmatch(
            self.git_commit
        ):
            raise ValueError("git_commit must be a lowercase hexadecimal commit")


@dataclass(frozen=True)
class FitEpochResult:
    """Train and validation aggregates for one epoch."""

    epoch: int
    train: EpochResult
    validation: EpochResult


@dataclass(frozen=True)
class FitResult:
    """Complete history and validation-selected checkpoint identity."""

    history: tuple[FitEpochResult, ...]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated checkpoint metadata restored into model and optimizer."""

    epoch: int
    validation_loss: float
    history: tuple[FitEpochResult, ...]
    provenance: CheckpointProvenance


def fit(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    config: FitConfig,
    checkpoint_path: str | Path,
    provenance: CheckpointProvenance,
) -> FitResult:
    """Fit fixed epochs and restore the first minimum-validation-loss state."""
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig")
    if not isinstance(provenance, CheckpointProvenance):
        raise TypeError("provenance must be CheckpointProvenance")
    _require_loader_split(train_loader, "train")
    _require_loader_split(validation_loader, "validation")
    path = Path(checkpoint_path)
    history: list[FitEpochResult] = []
    best_epoch = 0
    best_validation_loss = math.inf

    for epoch in range(1, config.epochs + 1):
        train_result = train_one_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        validation_result = evaluate_loss(
            model, validation_loader, loss_function, device
        )
        epoch_result = FitEpochResult(epoch, train_result, validation_result)
        history.append(epoch_result)
        if validation_result.loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_result.loss
            save_training_checkpoint(
                path,
                model,
                optimizer,
                epoch=best_epoch,
                validation_loss=best_validation_loss,
                history=tuple(history),
                provenance=provenance,
            )

    load_training_checkpoint(
        path,
        model,
        optimizer,
        device=device,
        expected_provenance=provenance,
    )
    save_training_checkpoint(
        path,
        model,
        optimizer,
        epoch=best_epoch,
        validation_loss=best_validation_loss,
        history=tuple(history),
        provenance=provenance,
    )
    return FitResult(tuple(history), best_epoch, best_validation_loss, path)


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    epoch: int,
    validation_loss: float,
    history: tuple[FitEpochResult, ...],
    provenance: CheckpointProvenance,
) -> None:
    """Atomically save plain weights-only-compatible checkpoint values."""
    _validate_checkpoint_summary(epoch, validation_loss, history)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "history": [_serialize_epoch(item) for item in history],
        "provenance": asdict(provenance),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    device: torch.device,
    expected_provenance: CheckpointProvenance | None = None,
) -> LoadedCheckpoint:
    """Validate and restore a checkpoint using PyTorch weights-only loading."""
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as error:
        raise ValueError("Checkpoint could not be loaded safely") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Checkpoint schema_version is invalid")
    required = {
        "epoch",
        "validation_loss",
        "history",
        "provenance",
        "model_state_dict",
        "optimizer_state_dict",
    }
    if missing := sorted(required - payload.keys()):
        raise ValueError(f"Checkpoint is missing fields: {missing}")
    provenance = _deserialize_provenance(payload["provenance"])
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("Checkpoint provenance does not match expected provenance")
    history = _deserialize_history(payload["history"])
    epoch = payload["epoch"]
    validation_loss = payload["validation_loss"]
    _validate_checkpoint_summary(epoch, validation_loss, history)
    try:
        model.to(device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except Exception as error:
        raise ValueError(
            "Checkpoint state is incompatible with model or optimizer"
        ) from error
    return LoadedCheckpoint(epoch, float(validation_loss), history, provenance)


def _require_loader_split(loader: Any, expected: str) -> None:
    actual = getattr(getattr(loader, "dataset", None), "split", None)
    if actual != expected:
        raise ValueError(
            f"Fit requires a {expected!r} Dataset; loader declares {actual!r}"
        )


def _serialize_epoch(item: FitEpochResult) -> dict[str, Any]:
    return {
        "epoch": item.epoch,
        "train": asdict(item.train),
        "validation": asdict(item.validation),
    }


def _deserialize_history(value: Any) -> tuple[FitEpochResult, ...]:
    if not isinstance(value, list):
        raise ValueError("Checkpoint history must be a list")
    try:
        history = tuple(
            FitEpochResult(
                epoch=item["epoch"],
                train=EpochResult(**item["train"]),
                validation=EpochResult(**item["validation"]),
            )
            for item in value
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint history is invalid") from error
    return history


def _deserialize_provenance(value: Any) -> CheckpointProvenance:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint provenance must be a mapping")
    try:
        return CheckpointProvenance(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint provenance is invalid") from error


def _validate_checkpoint_summary(
    epoch: Any,
    validation_loss: Any,
    history: tuple[FitEpochResult, ...],
) -> None:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Checkpoint epoch must be a positive integer")
    if not isinstance(validation_loss, (int, float)) or not math.isfinite(
        float(validation_loss)
    ):
        raise ValueError("Checkpoint validation loss must be finite")
    if not history or epoch > len(history):
        raise ValueError("Checkpoint history must contain the selected epoch")
    if float(validation_loss) != history[epoch - 1].validation.loss:
        raise ValueError("Checkpoint validation loss does not match selected epoch")
    for expected_epoch, item in enumerate(history, start=1):
        if item.epoch != expected_epoch:
            raise ValueError("Checkpoint history epochs must be consecutive")
        for result in (item.train, item.validation):
            if (
                isinstance(result.loss, bool)
                or not isinstance(result.loss, (int, float))
                or not math.isfinite(result.loss)
                or isinstance(result.samples, bool)
                or not isinstance(result.samples, int)
                or result.samples < 1
                or isinstance(result.batches, bool)
                or not isinstance(result.batches, int)
                or result.batches < 1
            ):
                raise ValueError("Checkpoint history contains invalid epoch results")
