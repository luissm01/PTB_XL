"""Validation-only threshold selection and threshold-dependent metrics."""

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ptbxl.data.labels import TARGET_SUPERCLASSES
from ptbxl.evaluation.multilabel import PredictionSet


THRESHOLD_ARTIFACT_SCHEMA_VERSION = 1
THRESHOLD_METHOD = "per_class_max_f1"
DECISION_RULE = "probability_greater_than_or_equal_to_threshold"
TIE_BREAK_RULE = "highest_threshold"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")


@dataclass(frozen=True)
class ThresholdSet:
    """Five ordered operating thresholds and their fixed selection semantics."""

    labels: tuple[str, ...]
    values: tuple[float, ...]
    method: str = THRESHOLD_METHOD
    decision_rule: str = DECISION_RULE
    tie_break: str = TIE_BREAK_RULE

    def __post_init__(self) -> None:
        if self.labels != TARGET_SUPERCLASSES:
            raise ValueError(
                f"Threshold labels must equal the fixed order {TARGET_SUPERCLASSES}"
            )
        if len(self.values) != len(TARGET_SUPERCLASSES):
            raise ValueError(
                f"Thresholds must contain {len(TARGET_SUPERCLASSES)} values"
            )
        validated: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Threshold values must be real numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("Threshold values must be finite values in [0, 1]")
            validated.append(numeric)
        if self.method != THRESHOLD_METHOD:
            raise ValueError(f"Threshold method must be {THRESHOLD_METHOD!r}")
        if self.decision_rule != DECISION_RULE:
            raise ValueError(f"Decision rule must be {DECISION_RULE!r}")
        if self.tie_break != TIE_BREAK_RULE:
            raise ValueError(f"Tie-break rule must be {TIE_BREAK_RULE!r}")
        object.__setattr__(self, "values", tuple(validated))

    def by_label(self) -> dict[str, float]:
        """Return thresholds keyed in canonical target order."""
        return dict(zip(self.labels, self.values, strict=True))


@dataclass(frozen=True)
class LabelOperatingMetrics:
    """Confusion counts and operating metrics for one target."""

    label: str
    threshold: float
    positives: int
    negatives: int
    predicted_positives: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    sensitivity: float
    specificity: float
    f1: float


@dataclass(frozen=True)
class OperatingMetricSummary:
    """Precision, sensitivity, specificity and F1 at one aggregation level."""

    precision: float
    sensitivity: float
    specificity: float
    f1: float


@dataclass(frozen=True)
class MultilabelOperatingMetrics:
    """Per-class, macro and flattened micro operating-point metrics."""

    samples: int
    per_class: tuple[LabelOperatingMetrics, ...]
    macro: OperatingMetricSummary
    micro: OperatingMetricSummary


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-selected thresholds, metrics and prediction fingerprint."""

    thresholds: ThresholdSet
    metrics: MultilabelOperatingMetrics
    validation_predictions_sha256: str

    def __post_init__(self) -> None:
        if not SHA256_PATTERN.fullmatch(self.validation_predictions_sha256):
            raise ValueError(
                "validation_predictions_sha256 must be a lowercase SHA-256"
            )


@dataclass(frozen=True)
class FrozenThresholds:
    """Loadable subset of a provenance-bound threshold artifact."""

    name: str
    thresholds: ThresholdSet
    dataset_name: str
    dataset_version: str
    cohort_name: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    experiment_config_sha256: str
    experiment_report_sha256: str
    validation_predictions_sha256: str
    validation_samples: int


def select_validation_thresholds(predictions: PredictionSet) -> ThresholdSelection:
    """Maximize F1 independently per class on validation predictions."""
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    _require_two_target_states(predictions.targets)
    values = tuple(
        _select_highest_best_f1_threshold(
            predictions.targets[:, column], predictions.probabilities[:, column]
        )
        for column in range(len(TARGET_SUPERCLASSES))
    )
    thresholds = ThresholdSet(TARGET_SUPERCLASSES, values)
    return ThresholdSelection(
        thresholds=thresholds,
        metrics=compute_operating_metrics(predictions, thresholds),
        validation_predictions_sha256=fingerprint_predictions(predictions),
    )


def compute_operating_metrics(
    predictions: PredictionSet,
    thresholds: ThresholdSet,
) -> MultilabelOperatingMetrics:
    """Compute strict multilabel metrics at frozen per-class thresholds."""
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    if not isinstance(thresholds, ThresholdSet):
        raise TypeError("thresholds must be a ThresholdSet")
    targets = predictions.targets.astype(bool, copy=False)
    _require_two_target_states(targets)
    decisions = predictions.probabilities >= np.asarray(thresholds.values)[None, :]
    true_positives = np.sum(targets & decisions, axis=0, dtype=np.int64)
    true_negatives = np.sum(~targets & ~decisions, axis=0, dtype=np.int64)
    false_positives = np.sum(~targets & decisions, axis=0, dtype=np.int64)
    false_negatives = np.sum(targets & ~decisions, axis=0, dtype=np.int64)

    per_class = tuple(
        _label_metrics(
            label=label,
            threshold=thresholds.values[column],
            true_positives=int(true_positives[column]),
            true_negatives=int(true_negatives[column]),
            false_positives=int(false_positives[column]),
            false_negatives=int(false_negatives[column]),
        )
        for column, label in enumerate(TARGET_SUPERCLASSES)
    )
    macro = OperatingMetricSummary(
        precision=float(np.mean([item.precision for item in per_class])),
        sensitivity=float(np.mean([item.sensitivity for item in per_class])),
        specificity=float(np.mean([item.specificity for item in per_class])),
        f1=float(np.mean([item.f1 for item in per_class])),
    )
    micro = _summary_from_counts(
        true_positives=int(true_positives.sum()),
        true_negatives=int(true_negatives.sum()),
        false_positives=int(false_positives.sum()),
        false_negatives=int(false_negatives.sum()),
    )
    return MultilabelOperatingMetrics(
        samples=predictions.targets.shape[0],
        per_class=per_class,
        macro=macro,
        micro=micro,
    )


def fingerprint_predictions(
    predictions: PredictionSet,
    *,
    split: str = "validation",
) -> str:
    """Hash ordered IDs, binary targets and probabilities for one split."""
    if not isinstance(predictions, PredictionSet):
        raise TypeError("predictions must be a PredictionSet")
    if split not in {"validation", "test"}:
        raise ValueError("Prediction fingerprint split must be validation or test")
    digest = hashlib.sha256()
    digest.update(f"ptbxl-{split}-predictions-v1\0".encode())
    digest.update(struct.pack("<QQ", *predictions.targets.shape))
    digest.update(np.asarray(predictions.ecg_ids, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(predictions.targets, dtype="i1").tobytes(order="C"))
    digest.update(np.asarray(predictions.probabilities, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def load_frozen_thresholds(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> FrozenThresholds:
    """Load a strict threshold artifact and optionally bind it to a checkpoint."""
    import json

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Could not load threshold artifact") from error
    if not isinstance(raw, dict):
        raise TypeError("Threshold artifact must be a JSON object")
    _require_exact_fields(
        "threshold artifact",
        raw,
        {
            "schema_version",
            "selection",
            "dataset",
            "sources",
            "runtime",
            "validation",
            "limitations",
        },
    )
    if raw["schema_version"] != THRESHOLD_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Threshold schema_version must be {THRESHOLD_ARTIFACT_SCHEMA_VERSION}"
        )
    selection = _require_mapping(
        raw,
        "selection",
        {"name", "git_commit", "method", "decision_rule", "tie_break", "thresholds"},
    )
    dataset = _require_mapping(
        raw,
        "dataset",
        {"name", "version", "cohort", "split", "samples", "targets"},
    )
    sources = _require_mapping(
        raw,
        "sources",
        {
            "selection_config",
            "experiment_config",
            "experiment_report",
            "checkpoint",
            "standardizer",
            "validation_predictions_sha256",
        },
    )
    thresholds_raw = selection["thresholds"]
    if not isinstance(thresholds_raw, dict):
        raise TypeError("selection.thresholds must be an object")
    _require_exact_fields(
        "selection.thresholds", thresholds_raw, set(TARGET_SUPERCLASSES)
    )
    if dataset["targets"] != list(TARGET_SUPERCLASSES):
        raise ValueError("Threshold artifact targets are not in canonical order")
    if dataset["split"] != "validation":
        raise ValueError("Threshold artifact must be selected on validation")
    for field in ("name", "version", "cohort"):
        if not isinstance(dataset[field], str) or not dataset[field].strip():
            raise ValueError(f"Threshold artifact dataset.{field} must be non-empty")
    samples = dataset["samples"]
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Threshold artifact samples must be a positive integer")
    if not isinstance(selection["name"], str) or not selection["name"].strip():
        raise ValueError("Threshold selection name must be non-empty")
    if not isinstance(selection["git_commit"], str) or not GIT_COMMIT_PATTERN.fullmatch(
        selection["git_commit"]
    ):
        raise ValueError("Threshold selection git_commit is invalid")
    thresholds = ThresholdSet(
        labels=TARGET_SUPERCLASSES,
        values=tuple(thresholds_raw[label] for label in TARGET_SUPERCLASSES),
        method=selection["method"],
        decision_rule=selection["decision_rule"],
        tie_break=selection["tie_break"],
    )
    _source_sha256(sources, "selection_config")
    experiment_config_sha256 = _source_sha256(sources, "experiment_config")
    experiment_report_sha256 = _source_sha256(sources, "experiment_report")
    checkpoint_sha256 = _source_sha256(sources, "checkpoint")
    preprocessing_sha256 = _source_sha256(sources, "standardizer")
    predictions_sha256 = sources["validation_predictions_sha256"]
    if not isinstance(predictions_sha256, str) or not SHA256_PATTERN.fullmatch(
        predictions_sha256
    ):
        raise ValueError("validation_predictions_sha256 must be a lowercase SHA-256")
    if expected_checkpoint_sha256 is not None:
        if not isinstance(
            expected_checkpoint_sha256, str
        ) or not SHA256_PATTERN.fullmatch(expected_checkpoint_sha256):
            raise ValueError("expected_checkpoint_sha256 must be a lowercase SHA-256")
        if checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("Threshold artifact checkpoint SHA-256 does not match")
    runtime = raw["runtime"]
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be an object")
    _require_exact_fields(
        "runtime",
        runtime,
        {
            "python",
            "numpy",
            "scikit_learn",
            "torch",
            "cuda",
            "hardware",
            "deterministic_algorithms",
        },
    )
    _validate_validation_metrics(raw["validation"], thresholds, samples)
    if not isinstance(raw["limitations"], list) or not all(
        isinstance(item, str) and item.strip() for item in raw["limitations"]
    ):
        raise ValueError("limitations must be a list of non-empty strings")
    return FrozenThresholds(
        name=selection["name"],
        thresholds=thresholds,
        dataset_name=dataset["name"],
        dataset_version=dataset["version"],
        cohort_name=dataset["cohort"],
        checkpoint_sha256=checkpoint_sha256,
        preprocessing_sha256=preprocessing_sha256,
        experiment_config_sha256=experiment_config_sha256,
        experiment_report_sha256=experiment_report_sha256,
        validation_predictions_sha256=predictions_sha256,
        validation_samples=samples,
    )


def _select_highest_best_f1_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    best_threshold = 0.0
    best_numerator = -1
    best_denominator = 1
    for threshold in np.unique(probabilities):
        decisions = probabilities >= threshold
        true_positives = int(np.sum((targets == 1) & decisions))
        false_positives = int(np.sum((targets == 0) & decisions))
        false_negatives = int(np.sum((targets == 1) & ~decisions))
        numerator = 2 * true_positives
        denominator = numerator + false_positives + false_negatives
        comparison = numerator * best_denominator - best_numerator * denominator
        if comparison > 0 or (comparison == 0 and float(threshold) > best_threshold):
            best_threshold = float(threshold)
            best_numerator = numerator
            best_denominator = denominator
    return best_threshold


def _label_metrics(
    *,
    label: str,
    threshold: float,
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
) -> LabelOperatingMetrics:
    summary = _summary_from_counts(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    return LabelOperatingMetrics(
        label=label,
        threshold=threshold,
        positives=true_positives + false_negatives,
        negatives=true_negatives + false_positives,
        predicted_positives=true_positives + false_positives,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=summary.precision,
        sensitivity=summary.sensitivity,
        specificity=summary.specificity,
        f1=summary.f1,
    )


def _summary_from_counts(
    *,
    true_positives: int,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
) -> OperatingMetricSummary:
    return OperatingMetricSummary(
        precision=_safe_ratio(true_positives, true_positives + false_positives),
        sensitivity=_safe_ratio(true_positives, true_positives + false_negatives),
        specificity=_safe_ratio(true_negatives, true_negatives + false_positives),
        f1=_safe_ratio(
            2 * true_positives,
            2 * true_positives + false_positives + false_negatives,
        ),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _require_two_target_states(targets: np.ndarray) -> None:
    for column, label in enumerate(TARGET_SUPERCLASSES):
        if np.unique(targets[:, column]).size != 2:
            raise ValueError(
                f"Operating metrics require positive and negative targets for {label}"
            )


def _require_mapping(
    parent: dict[str, Any],
    field: str,
    expected_fields: set[str],
) -> dict[str, Any]:
    value = parent[field]
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    _require_exact_fields(field, value, expected_fields)
    return value


def _require_exact_fields(
    name: str, values: dict[str, Any], expected: set[str]
) -> None:
    actual = set(values)
    if missing := sorted(expected - actual):
        raise KeyError(f"{name} is missing fields: {missing}")
    if unexpected := sorted(actual - expected):
        raise ValueError(f"{name} has unexpected fields: {unexpected}")


def _source_sha256(sources: dict[str, Any], name: str) -> str:
    source = sources[name]
    if not isinstance(source, dict):
        raise TypeError(f"sources.{name} must be an object")
    _require_exact_fields(f"sources.{name}", source, {"path", "sha256"})
    sha256 = source["sha256"]
    if not isinstance(source["path"], str) or not source["path"].strip():
        raise ValueError(f"sources.{name}.path must be non-empty")
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise ValueError(f"sources.{name}.sha256 must be a lowercase SHA-256")
    return sha256


def _validate_validation_metrics(
    raw: Any,
    thresholds: ThresholdSet,
    samples: int,
) -> None:
    if not isinstance(raw, dict):
        raise TypeError("validation must be an object")
    _require_exact_fields("validation", raw, {"samples", "per_class", "macro", "micro"})
    if raw["samples"] != samples:
        raise ValueError("Validation sample count does not match dataset")
    per_class = raw["per_class"]
    if not isinstance(per_class, list) or len(per_class) != len(TARGET_SUPERCLASSES):
        raise ValueError("validation.per_class must contain five class results")
    parsed: list[dict[str, Any]] = []
    metric_fields = {"precision", "sensitivity", "specificity", "f1"}
    count_fields = {
        "positives",
        "negatives",
        "predicted_positives",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    }
    expected_fields = {"label", "threshold", *count_fields, *metric_fields}
    for column, (item, label) in enumerate(
        zip(per_class, TARGET_SUPERCLASSES, strict=True)
    ):
        if not isinstance(item, dict):
            raise TypeError("Each validation.per_class item must be an object")
        _require_exact_fields(f"validation.per_class[{column}]", item, expected_fields)
        if item["label"] != label:
            raise ValueError("Validation class results are not in canonical order")
        if item["threshold"] != thresholds.values[column]:
            raise ValueError("Validation metric threshold does not match selection")
        for field in count_fields:
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Validation count {field} must be non-negative")
        if item["positives"] + item["negatives"] != samples:
            raise ValueError("Validation class support does not match sample count")
        if item["true_positives"] + item["false_negatives"] != item["positives"]:
            raise ValueError("Validation positive confusion counts are inconsistent")
        if item["true_negatives"] + item["false_positives"] != item["negatives"]:
            raise ValueError("Validation negative confusion counts are inconsistent")
        if (
            item["true_positives"] + item["false_positives"]
            != item["predicted_positives"]
        ):
            raise ValueError("Validation predicted-positive count is inconsistent")
        expected_summary = _summary_from_counts(
            true_positives=item["true_positives"],
            true_negatives=item["true_negatives"],
            false_positives=item["false_positives"],
            false_negatives=item["false_negatives"],
        )
        _require_summary_matches(
            {field: item[field] for field in metric_fields},
            expected_summary,
            metric_fields,
        )
        parsed.append(item)

    macro = OperatingMetricSummary(
        **{
            field: float(np.mean([item[field] for item in parsed]))
            for field in metric_fields
        }
    )
    _require_summary_matches(raw["macro"], macro, metric_fields)
    micro = _summary_from_counts(
        true_positives=sum(item["true_positives"] for item in parsed),
        true_negatives=sum(item["true_negatives"] for item in parsed),
        false_positives=sum(item["false_positives"] for item in parsed),
        false_negatives=sum(item["false_negatives"] for item in parsed),
    )
    _require_summary_matches(raw["micro"], micro, metric_fields)


def _require_summary_matches(
    raw: Any,
    expected: OperatingMetricSummary,
    metric_fields: set[str],
) -> None:
    if not isinstance(raw, dict):
        raise TypeError("Operating metric summary must be an object")
    _require_exact_fields("operating metric summary", raw, metric_fields)
    for field in metric_fields:
        value = raw[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"Operating metric {field} must be finite in [0, 1]")
        if not math.isclose(
            float(value), getattr(expected, field), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"Operating metric {field} is inconsistent with counts")
