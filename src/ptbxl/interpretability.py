"""Small gradient-based attribution boundary for ECG model explanations."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ptbxl.data.labels import TARGET_SUPERCLASSES
from ptbxl.models.cnn import INPUT_CHANNELS, INPUT_SAMPLES


@dataclass(frozen=True)
class InputAttribution:
    """Absolute input-gradient-times-input relevance for one output logit."""

    label: str
    logit: float
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.label not in TARGET_SUPERCLASSES:
            raise ValueError(f"label must be one of {TARGET_SUPERCLASSES}")
        if not math.isfinite(self.logit):
            raise ValueError("logit must be finite")
        if not isinstance(self.values, np.ndarray):
            raise TypeError("attribution values must be a NumPy array")
        if self.values.shape != (INPUT_SAMPLES, INPUT_CHANNELS):
            raise ValueError("attribution values must have shape (1000, 12)")
        if not np.issubdtype(self.values.dtype, np.number):
            raise TypeError("attribution values must be numeric")
        if not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("attribution values must be finite and non-negative")
        if not np.any(self.values > 0):
            raise ValueError("attribution values cannot all be zero")
        values = np.array(self.values, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class LeadAttributionSummary:
    """Attribution share and strongest fixed-width interval for one lead."""

    lead: str
    attribution_fraction: float
    peak_start_seconds: float
    peak_end_seconds: float
    peak_fraction_within_lead: float


def compute_input_gradient_attribution(
    model: nn.Module,
    standardized_signal: np.ndarray,
    *,
    class_index: int,
    device: torch.device,
) -> InputAttribution:
    """Compute local ``abs(input * d(logit)/d(input))`` attribution."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if isinstance(class_index, bool) or not isinstance(class_index, int):
        raise TypeError("class_index must be an integer")
    if not 0 <= class_index < len(TARGET_SUPERCLASSES):
        raise ValueError("class_index is outside the target range")
    signal = _validated_signal(standardized_signal)
    batch = (
        torch.from_numpy(np.ascontiguousarray(signal.T))
        .unsqueeze(0)
        .to(device)
        .requires_grad_(True)
    )
    was_training = model.training
    model.to(device)
    model.eval()
    try:
        logits = model(batch)
        if not isinstance(logits, torch.Tensor) or logits.shape != (
            1,
            len(TARGET_SUPERCLASSES),
        ):
            raise ValueError("model output must have shape (1, 5)")
        if not torch.isfinite(logits).all():
            raise ValueError("model logits must be finite")
        gradient = torch.autograd.grad(logits[0, class_index], batch)[0]
        relevance = torch.abs(gradient * batch)
    finally:
        model.train(was_training)
    values = relevance.detach().cpu().numpy()[0].T
    return InputAttribution(
        label=TARGET_SUPERCLASSES[class_index],
        logit=float(logits[0, class_index].detach().cpu()),
        values=values,
    )


def summarize_attribution_by_lead(
    attribution: InputAttribution,
    lead_names: tuple[str, ...],
    *,
    sampling_frequency_hz: float,
    window_samples: int,
) -> tuple[LeadAttributionSummary, ...]:
    """Summarize global lead shares and each lead's strongest time window."""
    if not isinstance(attribution, InputAttribution):
        raise TypeError("attribution must be an InputAttribution")
    if (
        not isinstance(lead_names, tuple)
        or len(lead_names) != attribution.values.shape[1]
        or len(set(lead_names)) != len(lead_names)
        or any(not isinstance(name, str) or not name for name in lead_names)
    ):
        raise ValueError("lead_names must uniquely identify every attribution column")
    if isinstance(sampling_frequency_hz, bool) or not isinstance(
        sampling_frequency_hz, (int, float)
    ):
        raise TypeError("sampling_frequency_hz must be numeric")
    frequency = float(sampling_frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if isinstance(window_samples, bool) or not isinstance(window_samples, int):
        raise TypeError("window_samples must be an integer")
    if not 1 <= window_samples <= attribution.values.shape[0]:
        raise ValueError("window_samples must fit within the signal")

    values = attribution.values
    global_total = float(values.sum())
    summaries = []
    for column, lead in enumerate(lead_names):
        lead_values = values[:, column]
        lead_total = float(lead_values.sum())
        windows = np.convolve(lead_values, np.ones(window_samples), mode="valid")
        start = int(np.argmax(windows))
        peak_total = float(windows[start])
        summaries.append(
            LeadAttributionSummary(
                lead=lead,
                attribution_fraction=lead_total / global_total,
                peak_start_seconds=start / frequency,
                peak_end_seconds=(start + window_samples) / frequency,
                peak_fraction_within_lead=(
                    peak_total / lead_total if lead_total > 0 else 0.0
                ),
            )
        )
    return tuple(summaries)


def _validated_signal(signal: object) -> np.ndarray:
    if not isinstance(signal, np.ndarray):
        raise TypeError("standardized_signal must be a NumPy array")
    if signal.shape != (INPUT_SAMPLES, INPUT_CHANNELS):
        raise ValueError("standardized_signal must have shape (1000, 12)")
    if not np.issubdtype(signal.dtype, np.number) or np.issubdtype(
        signal.dtype, np.complexfloating
    ):
        raise TypeError("standardized_signal must contain real numeric values")
    if not np.isfinite(signal).all():
        raise ValueError("standardized_signal must contain finite values")
    return np.asarray(signal, dtype=np.float32)
