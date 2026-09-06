"""Post-hoc descriptive error analysis for frozen multilabel predictions."""

from dataclasses import dataclass

import numpy as np

from ptbxl.data.labels import TARGET_SUPERCLASSES
from ptbxl.evaluation.multilabel import PredictionSet
from ptbxl.evaluation.thresholds import ThresholdSet


ERROR_KINDS = frozenset(
    {"false_positive", "false_negative", "true_positive", "true_negative"}
)


@dataclass(frozen=True)
class LabelErrorAnalysis:
    """Confusion counts for one class at its frozen threshold."""

    label: str
    threshold: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class TargetCombinationAnalysis:
    """Exact-match behavior for one observed true-label combination."""

    labels: tuple[str, ...]
    support: int
    exact_matches: int
    exact_match_rate: float
    label_errors: int


@dataclass(frozen=True)
class ErrorAnalysis:
    """Descriptive errors for one immutable prediction set."""

    samples: int
    exact_matches: int
    exact_match_rate: float
    label_errors: int
    hamming_loss: float
    per_class: tuple[LabelErrorAnalysis, ...]
    combinations: tuple[TargetCombinationAnalysis, ...]
    problematic_combinations: tuple[TargetCombinationAnalysis, ...]
    minimum_combination_support: int


@dataclass(frozen=True)
class SelectedPredictionExample:
    """One deterministically selected row for post-hoc explanation."""

    ecg_id: int
    row_index: int
    label: str
    kind: str
    probability: float
    threshold: float
    targets: tuple[int, ...]
    decisions: tuple[int, ...]


def analyze_prediction_errors(
    predictions: PredictionSet,
    thresholds: ThresholdSet,
    *,
    minimum_combination_support: int,
    maximum_problematic_combinations: int = 5,
) -> ErrorAnalysis:
    """Describe errors without fitting or changing the frozen operating point."""
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    if not isinstance(thresholds, ThresholdSet):
        raise TypeError("thresholds must be a ThresholdSet")
    _positive_integer("minimum_combination_support", minimum_combination_support)
    _positive_integer(
        "maximum_problematic_combinations", maximum_problematic_combinations
    )

    targets = predictions.targets.astype(bool, copy=False)
    decisions = predictions.probabilities >= np.asarray(thresholds.values)[None, :]
    errors = targets != decisions
    samples = targets.shape[0]
    exact_rows = ~errors.any(axis=1)

    per_class = tuple(
        LabelErrorAnalysis(
            label=label,
            threshold=thresholds.values[column],
            true_positives=int(np.sum(targets[:, column] & decisions[:, column])),
            true_negatives=int(np.sum(~targets[:, column] & ~decisions[:, column])),
            false_positives=int(np.sum(~targets[:, column] & decisions[:, column])),
            false_negatives=int(np.sum(targets[:, column] & ~decisions[:, column])),
        )
        for column, label in enumerate(TARGET_SUPERCLASSES)
    )
    combinations = _combination_analysis(targets, exact_rows, errors)
    eligible = [
        item for item in combinations if item.support >= minimum_combination_support
    ]
    problematic = tuple(
        sorted(
            eligible,
            key=lambda item: (
                item.exact_match_rate,
                -item.label_errors,
                -item.support,
                item.labels,
            ),
        )[:maximum_problematic_combinations]
    )
    exact_matches = int(exact_rows.sum())
    label_errors = int(errors.sum())
    return ErrorAnalysis(
        samples=samples,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / samples,
        label_errors=label_errors,
        hamming_loss=label_errors / errors.size,
        per_class=per_class,
        combinations=combinations,
        problematic_combinations=problematic,
        minimum_combination_support=minimum_combination_support,
    )


def select_prediction_example(
    predictions: PredictionSet,
    thresholds: ThresholdSet,
    *,
    label: str,
    kind: str,
) -> SelectedPredictionExample:
    """Select the most confident requested outcome, breaking ties by ECG ID."""
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    if not isinstance(thresholds, ThresholdSet):
        raise TypeError("thresholds must be a ThresholdSet")
    if label not in TARGET_SUPERCLASSES:
        raise ValueError(f"label must be one of {TARGET_SUPERCLASSES}")
    if kind not in ERROR_KINDS:
        raise ValueError(f"kind must be one of {sorted(ERROR_KINDS)}")

    column = TARGET_SUPERCLASSES.index(label)
    targets = predictions.targets.astype(bool, copy=False)
    decisions = predictions.probabilities >= np.asarray(thresholds.values)[None, :]
    class_targets = targets[:, column]
    class_decisions = decisions[:, column]
    masks = {
        "false_positive": ~class_targets & class_decisions,
        "false_negative": class_targets & ~class_decisions,
        "true_positive": class_targets & class_decisions,
        "true_negative": ~class_targets & ~class_decisions,
    }
    rows = np.flatnonzero(masks[kind])
    if rows.size == 0:
        raise ValueError(f"No {kind} example exists for {label}")
    probabilities = predictions.probabilities[rows, column]
    ecg_ids = np.asarray(predictions.ecg_ids, dtype=np.int64)[rows]
    if kind in {"false_positive", "true_positive"}:
        order = np.lexsort((ecg_ids, -probabilities))
    else:
        order = np.lexsort((ecg_ids, probabilities))
    row = int(rows[int(order[0])])
    return SelectedPredictionExample(
        ecg_id=predictions.ecg_ids[row],
        row_index=row,
        label=label,
        kind=kind,
        probability=float(predictions.probabilities[row, column]),
        threshold=thresholds.values[column],
        targets=tuple(int(value) for value in targets[row]),
        decisions=tuple(int(value) for value in decisions[row]),
    )


def _combination_analysis(
    targets: np.ndarray,
    exact_rows: np.ndarray,
    errors: np.ndarray,
) -> tuple[TargetCombinationAnalysis, ...]:
    unique_rows, inverse, counts = np.unique(
        targets.astype(np.int8), axis=0, return_inverse=True, return_counts=True
    )
    combinations = []
    for combination_index, target_row in enumerate(unique_rows):
        mask = inverse == combination_index
        support = int(counts[combination_index])
        exact_matches = int(exact_rows[mask].sum())
        combinations.append(
            TargetCombinationAnalysis(
                labels=tuple(
                    label
                    for label, active in zip(
                        TARGET_SUPERCLASSES, target_row, strict=True
                    )
                    if active
                ),
                support=support,
                exact_matches=exact_matches,
                exact_match_rate=exact_matches / support,
                label_errors=int(errors[mask].sum()),
            )
        )
    return tuple(sorted(combinations, key=lambda item: (-item.support, item.labels)))


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
