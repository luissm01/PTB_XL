import numpy as np
import pytest
import torch
from torch import nn

from ptbxl.interpretability import (
    compute_input_gradient_attribution,
    summarize_attribution_by_lead,
)


class MeanLeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = torch.zeros((5, 12), dtype=torch.float32)
        weights[2, 4] = 2.0
        self.weights = nn.Parameter(weights)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal.mean(dim=2) @ self.weights.T


def test_gradient_input_attribution_is_local_and_does_not_mutate_parameters() -> None:
    model = MeanLeadModel()
    model.train()
    signal = np.ones((1_000, 12), dtype=np.float32)

    result = compute_input_gradient_attribution(
        model, signal, class_index=2, device=torch.device("cpu")
    )

    assert result.label == "STTC"
    assert result.logit == pytest.approx(2.0)
    assert np.all(result.values[:, 4] > 0)
    assert np.count_nonzero(result.values[:, [*range(4), *range(5, 12)]]) == 0
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not result.values.flags.writeable


def test_attribution_summary_reports_lead_share_and_earliest_peak() -> None:
    result = compute_input_gradient_attribution(
        MeanLeadModel(),
        np.ones((1_000, 12), dtype=np.float32),
        class_index=2,
        device=torch.device("cpu"),
    )
    leads = tuple(f"L{index}" for index in range(12))

    summaries = summarize_attribution_by_lead(
        result,
        leads,
        sampling_frequency_hz=100.0,
        window_samples=50,
    )

    assert summaries[4].attribution_fraction == pytest.approx(1.0)
    assert summaries[4].peak_start_seconds == 0.0
    assert summaries[4].peak_end_seconds == 0.5
    assert summaries[4].peak_fraction_within_lead == pytest.approx(0.05)
    assert sum(item.attribution_fraction for item in summaries) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("signal", "class_index", "error"),
    [
        (np.ones((12, 1_000), dtype=np.float32), 0, ValueError),
        (np.ones((1_000, 12), dtype=np.float32), 5, ValueError),
        (np.ones((1_000, 12), dtype=np.float32), True, TypeError),
    ],
)
def test_attribution_rejects_invalid_input(
    signal: np.ndarray, class_index: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        compute_input_gradient_attribution(
            MeanLeadModel(),
            signal,
            class_index=class_index,  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )
