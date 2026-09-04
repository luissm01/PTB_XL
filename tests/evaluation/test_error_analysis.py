import numpy as np
import pytest

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.evaluation import (
    PredictionSet,
    ThresholdSet,
    analyze_prediction_errors,
    select_prediction_example,
)


def _predictions() -> PredictionSet:
    return PredictionSet(
        ecg_ids=(10, 20, 15, 30),
        targets=np.asarray(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
        probabilities=np.asarray(
            [
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.1, 0.1, 0.1],
            ]
        ),
        batches=2,
    )


def _thresholds() -> ThresholdSet:
    return ThresholdSet(TARGET_SUPERCLASSES, (0.5,) * 5)


def test_error_analysis_counts_and_ranks_supported_combinations() -> None:
    analysis = analyze_prediction_errors(
        _predictions(),
        _thresholds(),
        minimum_combination_support=2,
        maximum_problematic_combinations=1,
    )

    assert analysis.samples == 4
    assert analysis.exact_matches == 1
    assert analysis.exact_match_rate == 0.25
    assert analysis.label_errors == 3
    assert analysis.hamming_loss == 3 / 20
    assert analysis.per_class[0].false_positives == 1
    assert analysis.per_class[0].false_negatives == 2
    assert analysis.per_class[1].true_positives == 1
    assert analysis.combinations[0].labels == ("NORM",)
    assert analysis.combinations[0].support == 3
    assert analysis.combinations[0].exact_match_rate == pytest.approx(1 / 3)
    assert analysis.problematic_combinations == (analysis.combinations[0],)


def test_example_selection_uses_confidence_then_smallest_ecg_id() -> None:
    selected = select_prediction_example(
        _predictions(), _thresholds(), label="NORM", kind="false_negative"
    )

    assert selected.ecg_id == 15
    assert selected.row_index == 2
    assert selected.probability == 0.2
    assert selected.targets == (1, 0, 0, 0, 0)
    assert selected.decisions == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("minimum_support", "maximum_combinations", "error"),
    [(0, 1, ValueError), (1.0, 1, TypeError), (1, True, TypeError)],
)
def test_error_analysis_rejects_invalid_limits(
    minimum_support: object, maximum_combinations: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        analyze_prediction_errors(
            _predictions(),
            _thresholds(),
            minimum_combination_support=minimum_support,  # type: ignore[arg-type]
            maximum_problematic_combinations=maximum_combinations,  # type: ignore[arg-type]
        )


def test_example_selection_rejects_missing_outcome() -> None:
    with pytest.raises(ValueError, match="No false_positive example exists for HYP"):
        select_prediction_example(
            _predictions(), _thresholds(), label="HYP", kind="false_positive"
        )
