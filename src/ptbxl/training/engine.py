"""Minimal epoch-level train and loss-evaluation functions."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class EpochResult:
    """Sample-weighted aggregate from one complete loader pass."""

    loss: float
    samples: int
    batches: int


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Train for one loader pass and return sample-weighted mean loss."""
    model.to(device)
    model.train()
    total_loss = 0.0
    sample_count = 0
    batch_count = 0

    for batch in batches:
        signal, targets = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signal)
        loss = _validated_loss(loss_function, logits, targets)
        loss.backward()
        optimizer.step()
        count = signal.shape[0]
        total_loss += float(loss.detach().item()) * count
        sample_count += count
        batch_count += 1

    return _epoch_result(total_loss, sample_count, batch_count)


def evaluate_loss(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    loss_function: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Evaluate loss for one loader pass without creating gradients."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    sample_count = 0
    batch_count = 0

    with torch.inference_mode():
        for batch in batches:
            signal, targets = _prepare_batch(batch, device)
            logits = model(signal)
            loss = _validated_loss(loss_function, logits, targets)
            count = signal.shape[0]
            total_loss += float(loss.item()) * count
            sample_count += count
            batch_count += 1

    return _epoch_result(total_loss, sample_count, batch_count)


def _prepare_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("Each batch must be a mapping")
    missing = [field for field in ("signal", "targets") if field not in batch]
    if missing:
        raise KeyError(f"Batch is missing fields: {missing}")
    signal = batch["signal"]
    targets = batch["targets"]
    if not isinstance(signal, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Batch signal and targets must be torch.Tensor values")
    if signal.ndim < 1 or targets.ndim < 1 or signal.shape[0] == 0:
        raise ValueError("Batch tensors must contain at least one sample")
    if signal.shape[0] != targets.shape[0]:
        raise ValueError("Batch signal and target counts must match")
    return signal.to(device), targets.to(device)


def _validated_loss(
    loss_function: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(
            f"Logit and target shapes must match; got {logits.shape} and {targets.shape}"
        )
    loss = loss_function(logits, targets)
    if loss.ndim != 0:
        raise ValueError("Loss function must return one scalar")
    if not torch.isfinite(loss):
        raise ValueError("Loss must be finite")
    return loss


def _epoch_result(total_loss: float, samples: int, batches: int) -> EpochResult:
    if samples == 0 or batches == 0:
        raise ValueError("Cannot compute an epoch result from an empty loader")
    mean_loss = total_loss / samples
    if not math.isfinite(mean_loss):
        raise ValueError("Epoch loss must be finite")
    return EpochResult(loss=mean_loss, samples=samples, batches=batches)
