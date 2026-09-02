from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from torch import nn

from ptbxl.models import ECGCNNConfig, SmallECGCNN


def test_default_config_is_explicit_and_frozen() -> None:
    config = ECGCNNConfig()

    assert config.feature_channels == (32, 64, 128)
    assert config.kernel_sizes == (7, 5, 3)
    assert config.dropout_probability == 0.2
    with pytest.raises(FrozenInstanceError):
        config.dropout_probability = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"feature_channels": []}, TypeError),
        ({"feature_channels": ()}, TypeError),
        ({"feature_channels": (32, 0)}, ValueError),
        ({"feature_channels": (32, True)}, ValueError),
        ({"kernel_sizes": (7, 4, 3)}, ValueError),
        ({"kernel_sizes": (7, 5)}, ValueError),
        ({"dropout_probability": True}, TypeError),
        ({"dropout_probability": float("nan")}, ValueError),
        ({"dropout_probability": 1.0}, ValueError),
        ({"feature_channels": (1,) * 10, "kernel_sizes": (3,) * 10}, ValueError),
    ],
)
def test_rejects_invalid_config(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ECGCNNConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_returns_five_raw_logits(batch_size: int) -> None:
    model = SmallECGCNN()
    signal = torch.randn(batch_size, 12, 1_000, dtype=torch.float32)

    logits = model(signal)

    assert logits.shape == (batch_size, 5)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert logits.requires_grad


@pytest.mark.parametrize(
    "invalid_input, error, message",
    [
        (np.zeros((2, 12, 1_000), dtype=np.float32), TypeError, "torch.Tensor"),
        (torch.empty(0, 12, 1_000), ValueError, "cannot be empty"),
        (torch.randn(12, 1_000), ValueError, "must have shape"),
        (torch.randn(2, 11, 1_000), ValueError, "12 channels"),
        (torch.randn(2, 12, 999), ValueError, "1000 samples"),
        (torch.randn(2, 12, 1_000, dtype=torch.float64), TypeError, "float32"),
    ],
)
def test_rejects_invalid_input(
    invalid_input: object,
    error: type[Exception],
    message: str,
) -> None:
    model = SmallECGCNN()

    with pytest.raises(error, match=message):
        model(invalid_input)  # type: ignore[arg-type]


def test_default_model_has_stable_size_and_no_output_activation() -> None:
    model = SmallECGCNN()

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert parameter_count == 38_597
    assert not any(
        isinstance(module, (nn.Sigmoid, nn.Softmax)) for module in model.modules()
    )


def test_bce_loss_and_all_gradients_are_finite() -> None:
    torch.manual_seed(2026)
    model = SmallECGCNN()
    signal = torch.randn(3, 12, 1_000)
    targets = torch.randint(0, 2, (3, 5), dtype=torch.int64).to(torch.float32)

    logits = model(signal)
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )


def test_eval_forward_is_deterministic() -> None:
    torch.manual_seed(2026)
    model = SmallECGCNN(ECGCNNConfig(dropout_probability=0.5))
    model.eval()
    signal = torch.randn(2, 12, 1_000)

    with torch.no_grad():
        first = model(signal)
        second = model(signal)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
