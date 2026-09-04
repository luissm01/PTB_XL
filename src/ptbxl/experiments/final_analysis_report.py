"""Strict schema validation for the versioned final-analysis report."""

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.evaluation import DECISION_RULE
from ptbxl.visualization import FINAL_FIGURE_FILENAMES


FINAL_ANALYSIS_REPORT_SCHEMA_VERSION = 1
FINAL_ANALYSIS_MODE = "posthoc_descriptive_analysis"
ATTRIBUTION_METHOD = "absolute_input_times_input_gradient"
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def load_final_analysis_report(
    path: str | Path,
    *,
    verify_figures: bool = False,
) -> dict[str, Any]:
    """Load and structurally validate one final post-hoc analysis report."""
    if not isinstance(verify_figures, bool):
        raise TypeError("verify_figures must be boolean")
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Could not load final-analysis report") from error
    if not isinstance(raw, dict):
        raise TypeError("Final-analysis report must be a JSON object")
    _require_exact_fields(
        "final-analysis report",
        raw,
        {
            "schema_version",
            "analysis",
            "dataset",
            "sources",
            "error_analysis",
            "attribution",
            "figures",
            "runtime",
            "limitations",
        },
    )
    if raw["schema_version"] != FINAL_ANALYSIS_REPORT_SCHEMA_VERSION:
        raise ValueError("Final-analysis report schema_version is invalid")
    _validate_analysis(raw)
    dataset = _validate_dataset(raw)
    _validate_sources(raw)
    _validate_error_analysis(raw, samples=dataset["samples"])
    _validate_attribution(raw)
    _validate_figures(raw, verify_figures=verify_figures)
    _validate_runtime(raw)
    if not isinstance(raw["limitations"], list) or not all(
        isinstance(item, str) and item.strip() for item in raw["limitations"]
    ):
        raise ValueError("Final-analysis limitations must be non-empty strings")
    return raw


def _validate_analysis(raw: Mapping[str, Any]) -> None:
    analysis = _require_mapping(
        raw,
        "analysis",
        {
            "name",
            "mode",
            "git_commit",
            "posthoc_only",
            "model_changes_forbidden",
            "full_test_inference_repeated",
        },
    )
    if (
        analysis["mode"] != FINAL_ANALYSIS_MODE
        or analysis["posthoc_only"] is not True
        or analysis["model_changes_forbidden"] is not True
        or analysis["full_test_inference_repeated"] is not False
    ):
        raise ValueError("Final-analysis methodological declaration is invalid")
    if not isinstance(analysis["git_commit"], str) or not GIT_COMMIT_PATTERN.fullmatch(
        analysis["git_commit"]
    ):
        raise ValueError("Final-analysis git_commit is invalid")


def _validate_dataset(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset = _require_mapping(
        raw, "dataset", {"name", "version", "split", "fold", "samples", "targets"}
    )
    if (
        dataset["name"] != "PTB-XL"
        or dataset["split"] != "test"
        or dataset["fold"] != 10
        or dataset["targets"] != list(TARGET_SUPERCLASSES)
        or not isinstance(dataset["samples"], int)
        or dataset["samples"] < 1
    ):
        raise ValueError("Final-analysis dataset declaration is invalid")
    return dataset


def _validate_sources(raw: Mapping[str, Any]) -> None:
    sources = raw["sources"]
    expected = {
        "analysis_config",
        "inference_config",
        "final_test_report",
        "prediction_artifact",
        "metadata",
        "experiment_config",
        "experiment_report",
        "checkpoint",
        "standardizer",
        "threshold_artifact",
    }
    if not isinstance(sources, Mapping):
        raise TypeError("Final-analysis sources must be an object")
    _require_exact_fields("sources", sources, expected)
    for name in expected:
        source = _require_mapping(sources, name, {"path", "sha256"})
        if not isinstance(source["path"], str) or not source["path"]:
            raise ValueError(f"Final-analysis source path is invalid: {name}")
        _validate_sha256(source["sha256"], f"sources.{name}.sha256")


def _validate_error_analysis(raw: Mapping[str, Any], *, samples: int) -> None:
    analysis = _require_mapping(
        raw,
        "error_analysis",
        {
            "decision_rule",
            "samples",
            "exact_matches",
            "exact_match_rate",
            "label_errors",
            "hamming_loss",
            "per_class",
            "combinations",
            "problematic_combinations",
            "minimum_combination_support",
        },
    )
    if analysis["samples"] != samples:
        raise ValueError("Final-analysis sample counts disagree")
    if analysis["decision_rule"] != DECISION_RULE:
        raise ValueError("Final-analysis decision rule is invalid")
    exact_matches = _bounded_integer(
        analysis["exact_matches"], "exact_matches", maximum=samples
    )
    label_errors = _bounded_integer(
        analysis["label_errors"], "label_errors", maximum=samples * 5
    )
    _require_rate(
        analysis["exact_match_rate"], exact_matches / samples, "exact_match_rate"
    )
    _require_rate(
        analysis["hamming_loss"], label_errors / (samples * 5), "hamming_loss"
    )
    minimum_support = _bounded_integer(
        analysis["minimum_combination_support"],
        "minimum_combination_support",
        minimum=1,
    )
    per_class = analysis["per_class"]
    if (
        not isinstance(per_class, list)
        or not all(isinstance(item, Mapping) for item in per_class)
        or [item.get("label") for item in per_class] != list(TARGET_SUPERCLASSES)
    ):
        raise ValueError("Final-analysis per-class errors are invalid")
    counted_label_errors = sum(
        _validate_per_class_error(item, samples=samples) for item in per_class
    )
    if counted_label_errors != label_errors:
        raise ValueError("Final-analysis label error total is inconsistent")
    all_combinations: list[Mapping[str, Any]] | None = None
    for field in ("combinations", "problematic_combinations"):
        combinations = analysis[field]
        if not isinstance(combinations, list) or not combinations:
            raise ValueError(f"Final-analysis {field} must be a non-empty list")
        for combination in combinations:
            _validate_combination(combination)
        if field == "combinations":
            all_combinations = combinations
            if sum(item["support"] for item in combinations) != samples:
                raise ValueError("Final-analysis combination supports are inconsistent")
            if sum(item["exact_matches"] for item in combinations) != exact_matches:
                raise ValueError("Final-analysis combination matches are inconsistent")
        elif any(item["support"] < minimum_support for item in combinations):
            raise ValueError("Problematic combination is below minimum support")
    assert all_combinations is not None
    identities = {tuple(item["labels"]) for item in all_combinations}
    if len(identities) != len(all_combinations):
        raise ValueError("Final-analysis combinations must be unique")
    if any(
        tuple(item["labels"]) not in identities
        for item in analysis["problematic_combinations"]
    ):
        raise ValueError("Problematic combination was not observed")


def _validate_per_class_error(item: Mapping[str, Any], *, samples: int) -> int:
    _require_exact_fields(
        "per-class error",
        item,
        {
            "label",
            "threshold",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        },
    )
    counts = [
        item[name]
        for name in (
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        )
    ]
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("Final-analysis confusion counts are invalid")
    if sum(counts) != samples:
        raise ValueError("Final-analysis confusion counts do not sum to samples")
    _require_rate(item["threshold"], float(item["threshold"]), "threshold")
    return item["false_positives"] + item["false_negatives"]


def _validate_attribution(raw: Mapping[str, Any]) -> None:
    attribution = _require_mapping(
        raw,
        "attribution",
        {
            "method",
            "label",
            "selection",
            "record",
            "logit",
            "window_samples",
            "per_lead",
            "top_leads",
            "interpretation",
        },
    )
    if attribution["method"] != ATTRIBUTION_METHOD:
        raise ValueError("Final-analysis attribution method is invalid")
    if attribution["label"] not in TARGET_SUPERCLASSES:
        raise ValueError("Final-analysis attribution label is invalid")
    selection = attribution["selection"]
    if not isinstance(selection, Mapping):
        raise TypeError("Final-analysis attribution selection must be an object")
    _require_exact_fields("selection", selection, {"kind", "rule"})
    if (
        selection["kind"]
        not in {
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative",
        }
        or selection["rule"] != "most_confident_then_smallest_ecg_id"
    ):
        raise ValueError("Final-analysis attribution selection is invalid")
    _validate_attribution_record(attribution["record"])
    if not isinstance(attribution["logit"], (int, float)) or not math.isfinite(
        attribution["logit"]
    ):
        raise ValueError("Final-analysis attribution logit is invalid")
    _bounded_integer(attribution["window_samples"], "window_samples", minimum=1)
    per_lead = attribution["per_lead"]
    if (
        not isinstance(per_lead, list)
        or len(per_lead) != 12
        or not all(isinstance(item, Mapping) for item in per_lead)
    ):
        raise ValueError("Final-analysis attribution must describe 12 leads")
    fractions = [item.get("attribution_fraction") for item in per_lead]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in fractions
    ):
        raise ValueError("Final-analysis attribution fractions are invalid")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Final-analysis attribution fractions must sum to one")
    leads = []
    for item in per_lead:
        _require_exact_fields(
            "lead attribution",
            item,
            {
                "lead",
                "attribution_fraction",
                "peak_start_seconds",
                "peak_end_seconds",
                "peak_fraction_within_lead",
            },
        )
        leads.append(item["lead"])
        _require_rate(
            item["attribution_fraction"],
            float(item["attribution_fraction"]),
            "attribution_fraction",
        )
        _require_rate(
            item["peak_fraction_within_lead"],
            float(item["peak_fraction_within_lead"]),
            "peak_fraction_within_lead",
        )
        start = item["peak_start_seconds"]
        end = item["peak_end_seconds"]
        if not all(isinstance(value, (int, float)) for value in (start, end)) or not (
            0 <= start < end <= 10
        ):
            raise ValueError("Final-analysis attribution time window is invalid")
    if len(set(leads)) != 12:
        raise ValueError("Final-analysis attribution lead names must be unique")
    top_leads = attribution["top_leads"]
    if (
        not isinstance(top_leads, list)
        or len(top_leads) != 3
        or len(set(top_leads)) != 3
        or any(lead not in leads for lead in top_leads)
    ):
        raise ValueError("Final-analysis top leads are invalid")
    interpretation = attribution["interpretation"]
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ValueError("Final-analysis attribution interpretation is invalid")


def _validate_figures(raw: Mapping[str, Any], *, verify_figures: bool) -> None:
    figures = raw["figures"]
    if (
        not isinstance(figures, list)
        or not all(isinstance(item, Mapping) for item in figures)
        or [Path(item.get("path", "")).name for item in figures]
        != list(FINAL_FIGURE_FILENAMES)
    ):
        raise ValueError("Final-analysis figure set is invalid")
    for item in figures:
        _require_exact_fields("figure", item, {"name", "path", "sha256"})
        if item["name"] != Path(item["path"]).stem:
            raise ValueError("Final-analysis figure name and path disagree")
        _validate_sha256(item["sha256"], "figure.sha256")
        if verify_figures:
            figure_path = Path(item["path"])
            if compute_sha256(figure_path) != item["sha256"]:
                raise ValueError(f"Final-analysis figure hash mismatch: {figure_path}")


def _validate_runtime(raw: Mapping[str, Any]) -> None:
    runtime = raw["runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("Final-analysis runtime must be an object")
    _require_exact_fields(
        "runtime",
        runtime,
        {
            "python",
            "numpy",
            "pandas",
            "scikit_learn",
            "matplotlib",
            "torch",
            "device",
            "prediction_artifact_fingerprint",
        },
    )
    _validate_sha256(
        runtime["prediction_artifact_fingerprint"],
        "runtime.prediction_artifact_fingerprint",
    )


def _validate_combination(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("Final-analysis combinations must be objects")
    _require_exact_fields(
        "combination",
        value,
        {"labels", "support", "exact_matches", "exact_match_rate", "label_errors"},
    )
    labels = value["labels"]
    if not isinstance(labels, list) or any(
        label not in TARGET_SUPERCLASSES for label in labels
    ):
        raise ValueError("Final-analysis combination labels are invalid")
    if labels != [label for label in TARGET_SUPERCLASSES if label in labels]:
        raise ValueError("Final-analysis combination labels are not canonical")
    support = value["support"]
    exact_matches = value["exact_matches"]
    label_errors = value["label_errors"]
    if any(
        not isinstance(item, int) or item < 0
        for item in (support, exact_matches, label_errors)
    ):
        raise ValueError("Final-analysis combination counts are invalid")
    if support < 1 or exact_matches > support:
        raise ValueError("Final-analysis combination support is invalid")
    rate = value["exact_match_rate"]
    if not isinstance(rate, (int, float)) or not math.isclose(
        float(rate), exact_matches / support, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Final-analysis combination rate is inconsistent")


def _validate_attribution_record(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("Final-analysis attribution record must be an object")
    _require_exact_fields(
        "attribution record",
        value,
        {
            "ecg_id",
            "signal_sha256",
            "targets",
            "decisions",
            "saved_probabilities",
            "verified_cpu_probabilities",
            "maximum_probability_difference",
        },
    )
    _bounded_integer(value["ecg_id"], "ecg_id", minimum=1)
    _validate_sha256(value["signal_sha256"], "attribution.signal_sha256")
    for field in ("targets", "decisions"):
        mapping = value[field]
        if not isinstance(mapping, Mapping) or set(mapping) != set(TARGET_SUPERCLASSES):
            raise ValueError(f"Final-analysis attribution {field} are invalid")
        if any(item not in {0, 1} for item in mapping.values()):
            raise ValueError(f"Final-analysis attribution {field} must be binary")
    probabilities = []
    for field in ("saved_probabilities", "verified_cpu_probabilities"):
        mapping = value[field]
        if not isinstance(mapping, Mapping) or set(mapping) != set(TARGET_SUPERCLASSES):
            raise ValueError(f"Final-analysis attribution {field} are invalid")
        for probability in mapping.values():
            _require_rate(probability, float(probability), field)
        probabilities.append([float(mapping[label]) for label in TARGET_SUPERCLASSES])
    maximum_difference = max(
        abs(saved - verified) for saved, verified in zip(*probabilities, strict=True)
    )
    recorded_difference = value["maximum_probability_difference"]
    if not isinstance(recorded_difference, (int, float)) or not math.isclose(
        recorded_difference,
        maximum_difference,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Final-analysis probability difference is inconsistent")


def _bounded_integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Final-analysis {name} is invalid")
    if maximum is not None and value > maximum:
        raise ValueError(f"Final-analysis {name} is invalid")
    return value


def _require_rate(value: Any, expected: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"Final-analysis {name} is invalid")


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_mapping(
    values: Mapping[str, Any], name: str, expected_fields: set[str]
) -> Mapping[str, Any]:
    value = values[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _require_exact_fields(name, value, expected_fields)
    return value


def _require_exact_fields(
    name: str, values: Mapping[str, Any], expected_fields: set[str]
) -> None:
    actual = set(values)
    if missing := sorted(expected_fields - actual):
        raise KeyError(f"{name} is missing fields: {missing}")
    if unexpected := sorted(actual - expected_fields):
        raise ValueError(f"{name} has unexpected fields: {unexpected}")
