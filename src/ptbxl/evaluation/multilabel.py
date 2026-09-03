"""Validation-only prediction collection and multilabel ranking metrics."""

import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from ptbxl.data.labels import TARGET_SUPERCLASSES


@dataclass(frozen=True)
class PredictionSet:
    """Ordered identities, targets and defensive probability copies."""

    ecg_ids: tuple[int, ...]
    targets: np.ndarray
    probabilities: np.ndarray
    batches: int

    def __post_init__(self) -> None:
        if not isinstance(self.targets, np.ndarray) or not isinstance(
            self.probabilities, np.ndarray
        ):
            raise TypeError("targets and probabilities must be NumPy arrays")
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape (samples, 5)")
        if self.probabilities.shape != self.targets.shape:
            raise ValueError("targets and probabilities must have equal shapes")
        if self.targets.shape[0] == 0:
            raise ValueError("Predictions must contain at least one sample")
        if self.targets.shape[1] != len(TARGET_SUPERCLASSES):
            raise ValueError(
                f"Predictions must contain {len(TARGET_SUPERCLASSES)} target columns"
            )
        if not _is_real_array(self.targets):
            raise TypeError("targets must contain real numeric or boolean values")
        if (
            not np.isfinite(self.targets).all()
            or not np.isin(self.targets, (0, 1)).all()
        ):
            raise ValueError("targets must contain only finite binary values")
        if not _is_real_array(self.probabilities, allow_boolean=False):
            raise TypeError("probabilities must contain real numeric values")
        if (
            not np.isfinite(self.probabilities).all()
            or not ((self.probabilities >= 0.0) & (self.probabilities <= 1.0)).all()
        ):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if (
            isinstance(self.batches, bool)
            or not isinstance(self.batches, int)
            or not 1 <= self.batches <= self.targets.shape[0]
        ):
            raise ValueError("batches must be a positive count no larger than samples")

        ecg_ids = _validated_ecg_ids(self.ecg_ids, self.targets.shape[0])
        targets = np.array(self.targets, dtype=np.int8, copy=True)
        probabilities = np.array(self.probabilities, dtype=np.float64, copy=True)
        targets.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "ecg_ids", ecg_ids)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True)
class LabelRankingMetrics:
    """Threshold-independent metrics for one diagnostic superclass."""

    label: str
    positives: int
    negatives: int
    prevalence: float
    auroc: float
    auprc: float


@dataclass(frozen=True)
class MultilabelRankingMetrics:
    """Per-class, macro and micro multilabel ranking metrics."""

    samples: int
    per_class: tuple[LabelRankingMetrics, ...]
    macro_auroc: float
    macro_auprc: float
    micro_auroc: float
    micro_auprc: float


@dataclass(frozen=True)
class ValidationEvaluation:
    """Complete threshold-independent result from one validation pass."""

    predictions: PredictionSet
    metrics: MultilabelRankingMetrics


@dataclass(frozen=True)
class FinalTestEvaluation:
    """Complete threshold-independent result from one final-test pass."""

    predictions: PredictionSet
    metrics: MultilabelRankingMetrics


def collect_validation_predictions(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> PredictionSet:
    """Collect ordered probabilities from an explicitly validation-only loader."""
    return _collect_split_predictions(
        model, batches, device, required_split="validation"
    )


def collect_final_test_predictions(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> PredictionSet:
    """Collect ordered probabilities from an explicitly test-only loader."""
    return _collect_split_predictions(model, batches, device, required_split="test")


def _collect_split_predictions(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    required_split: str,
) -> PredictionSet:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    actual_split = getattr(getattr(batches, "dataset", None), "split", None)
    if actual_split != required_split:
        raise ValueError(
            f"Prediction collection requires a {required_split} Dataset; "
            f"loader declares {actual_split!r}"
        )

    model.to(device)
    model.eval()
    all_ecg_ids: list[int] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    batch_count = 0

    with torch.inference_mode():
        for batch in batches:
            signal, targets, ecg_ids = _prepare_prediction_batch(batch)
            logits = model(signal.to(device))
            if not isinstance(logits, torch.Tensor):
                raise TypeError("model must return a torch.Tensor")
            if logits.shape != targets.shape:
                raise ValueError(
                    "logits and targets must have equal shapes; "
                    f"got {tuple(logits.shape)} and {tuple(targets.shape)}"
                )
            if not torch.isfinite(logits).all():
                raise ValueError("model logits must be finite")

            all_ecg_ids.extend(ecg_ids)
            all_targets.append(targets.detach().cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            batch_count += 1

    if batch_count == 0:
        raise ValueError("Cannot collect predictions from an empty loader")
    return PredictionSet(
        ecg_ids=tuple(all_ecg_ids),
        targets=np.concatenate(all_targets, axis=0),
        probabilities=np.concatenate(all_probabilities, axis=0),
        batches=batch_count,
    )


def compute_ranking_metrics(
    predictions: PredictionSet,
) -> MultilabelRankingMetrics:
    """Compute strict AUROC and average-precision AUPRC summaries."""
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    targets = predictions.targets
    probabilities = predictions.probabilities
    for column, label in enumerate(TARGET_SUPERCLASSES):
        if np.unique(targets[:, column]).size != 2:
            raise ValueError(
                f"Ranking metrics require positive and negative targets for {label}"
            )

    per_class_auroc = np.asarray(
        roc_auc_score(targets, probabilities, average=None), dtype=np.float64
    )
    per_class_auprc = np.asarray(
        average_precision_score(targets, probabilities, average=None),
        dtype=np.float64,
    )
    macro_auroc = float(per_class_auroc.mean())
    macro_auprc = float(per_class_auprc.mean())
    micro_auroc = float(roc_auc_score(targets, probabilities, average="micro"))
    micro_auprc = float(
        average_precision_score(targets, probabilities, average="micro")
    )
    metric_values = (
        *per_class_auroc,
        *per_class_auprc,
        macro_auroc,
        macro_auprc,
        micro_auroc,
        micro_auprc,
    )
    if not all(math.isfinite(float(value)) for value in metric_values):
        raise ValueError("Ranking metric computation returned a non-finite value")

    samples = targets.shape[0]
    per_class = tuple(
        LabelRankingMetrics(
            label=label,
            positives=int(targets[:, column].sum()),
            negatives=int(samples - targets[:, column].sum()),
            prevalence=float(targets[:, column].mean()),
            auroc=float(per_class_auroc[column]),
            auprc=float(per_class_auprc[column]),
        )
        for column, label in enumerate(TARGET_SUPERCLASSES)
    )
    return MultilabelRankingMetrics(
        samples=samples,
        per_class=per_class,
        macro_auroc=macro_auroc,
        macro_auprc=macro_auprc,
        micro_auroc=micro_auroc,
        micro_auprc=micro_auprc,
    )


def evaluate_validation(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> ValidationEvaluation:
    """Collect and score one explicitly validation-only model pass."""
    predictions = collect_validation_predictions(model, batches, device)
    return ValidationEvaluation(predictions, compute_ranking_metrics(predictions))


def evaluate_final_test(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> FinalTestEvaluation:
    """Collect and score one explicitly test-only model pass."""
    predictions = collect_final_test_predictions(model, batches, device)
    return FinalTestEvaluation(predictions, compute_ranking_metrics(predictions))


def _prepare_prediction_batch(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if not isinstance(batch, Mapping):
        raise TypeError("Each prediction batch must be a mapping")
    missing = [field for field in ("signal", "targets", "ecg_id") if field not in batch]
    if missing:
        raise KeyError(f"Prediction batch is missing fields: {missing}")
    signal = batch["signal"]
    targets = batch["targets"]
    if not isinstance(signal, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Prediction signal and targets must be torch.Tensor values")
    if signal.ndim < 1 or signal.shape[0] == 0:
        raise ValueError("Prediction signal must contain at least one sample")
    if targets.ndim != 2 or targets.shape[1] != len(TARGET_SUPERCLASSES):
        raise ValueError(
            f"Prediction targets must have shape (samples, {len(TARGET_SUPERCLASSES)})"
        )
    if signal.shape[0] != targets.shape[0]:
        raise ValueError("Prediction signal and target counts must match")
    ecg_ids = _validated_ecg_ids(batch["ecg_id"], signal.shape[0])
    return signal, targets, ecg_ids


def _validated_ecg_ids(value: Any, expected_count: int) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError("ecg_id values must form a one-dimensional sequence")
        raw_values: Sequence[Any] = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("ecg_id values must form a one-dimensional sequence")
        raw_values = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = value
    else:
        raise TypeError("ecg_id values must form a sequence of positive integers")
    if len(raw_values) != expected_count:
        raise ValueError("ecg_id count must match the prediction sample count")

    ecg_ids: list[int] = []
    for value in raw_values:
        if isinstance(value, bool):
            raise TypeError("ecg_id values must be positive integers")
        try:
            ecg_id = operator.index(value)
        except TypeError as error:
            raise TypeError("ecg_id values must be positive integers") from error
        if ecg_id <= 0:
            raise ValueError("ecg_id values must be positive integers")
        ecg_ids.append(ecg_id)
    if len(set(ecg_ids)) != len(ecg_ids):
        raise ValueError("ecg_id values must be unique")
    return tuple(ecg_ids)


def _is_real_array(value: np.ndarray, *, allow_boolean: bool = True) -> bool:
    return (
        (allow_boolean and np.issubdtype(value.dtype, np.bool_))
        or np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    )
