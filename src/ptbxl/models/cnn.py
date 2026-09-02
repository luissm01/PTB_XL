"""Small explainable 1D-CNN baseline for PTB-XL ECG batches."""

import math
from dataclasses import dataclass

import torch
from torch import nn


INPUT_CHANNELS = 12
INPUT_SAMPLES = 1_000
OUTPUT_LOGITS = 5


@dataclass(frozen=True)
class ECGCNNConfig:
    """Immutable architecture settings for the small ECG CNN."""

    feature_channels: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (7, 5, 3)
    dropout_probability: float = 0.2

    def __post_init__(self) -> None:
        _validate_positive_integer_tuple("feature_channels", self.feature_channels)
        _validate_positive_integer_tuple("kernel_sizes", self.kernel_sizes)
        if len(self.feature_channels) != len(self.kernel_sizes):
            raise ValueError(
                "feature_channels and kernel_sizes must have the same length"
            )
        if any(kernel_size % 2 == 0 for kernel_size in self.kernel_sizes):
            raise ValueError("CNN kernel sizes must be odd for same-length padding")
        if 2 ** len(self.feature_channels) > INPUT_SAMPLES:
            raise ValueError("Too many pooling blocks for 1,000 input samples")
        if isinstance(self.dropout_probability, bool) or not isinstance(
            self.dropout_probability, (int, float)
        ):
            raise TypeError("dropout_probability must be a finite number")
        if not math.isfinite(float(self.dropout_probability)) or not (
            0.0 <= self.dropout_probability < 1.0
        ):
            raise ValueError("dropout_probability must be in [0, 1)")


class SmallECGCNN(nn.Module):
    """Map a channel-first 12-lead ECG batch to five raw logits."""

    def __init__(self, config: ECGCNNConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ECGCNNConfig()
        if not isinstance(config, ECGCNNConfig):
            raise TypeError("config must be an ECGCNNConfig")
        self.config = config

        layers: list[nn.Module] = []
        input_channels = INPUT_CHANNELS
        for output_channels, kernel_size in zip(
            config.feature_channels,
            config.kernel_sizes,
            strict=True,
        ):
            layers.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            input_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.dropout_probability)
        self.classifier = nn.Linear(input_channels, OUTPUT_LOGITS)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Return raw multilabel logits for one validated ECG batch."""
        _validate_input(signal)
        features = self.features(signal)
        pooled = self.global_pool(features).squeeze(-1)
        return self.classifier(self.dropout(pooled))


def _validate_positive_integer_tuple(name: str, values: object) -> None:
    if not isinstance(values, tuple) or not values:
        raise TypeError(f"{name} must be a non-empty tuple of positive integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must contain only positive integers")


def _validate_input(signal: object) -> None:
    if not isinstance(signal, torch.Tensor):
        raise TypeError("CNN input must be a torch.Tensor")
    if signal.ndim != 3:
        raise ValueError(
            f"CNN input must have shape (batch, 12, 1000); got {tuple(signal.shape)}"
        )
    if signal.shape[0] == 0:
        raise ValueError("CNN input batch cannot be empty")
    if signal.shape[1] != INPUT_CHANNELS:
        raise ValueError(
            f"CNN input must contain {INPUT_CHANNELS} channels; got {signal.shape[1]}"
        )
    if signal.shape[2] != INPUT_SAMPLES:
        raise ValueError(
            f"CNN input must contain {INPUT_SAMPLES} samples; got {signal.shape[2]}"
        )
    if signal.dtype != torch.float32:
        raise TypeError(f"CNN input must use torch.float32; got {signal.dtype}")
