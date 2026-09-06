"""Reproducible post-hoc analysis of the immutable PTB-XL final predictions."""

import json
import importlib.metadata
import math
import platform
import re
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from ptbxl.data import TARGET_SUPERCLASSES, load_wfdb_record, validate_signal
from ptbxl.data.reporting import compute_sha256, write_json_report
from ptbxl.evaluation import (
    ErrorAnalysis,
    PredictionSet,
    SelectedPredictionExample,
    analyze_prediction_errors,
    fingerprint_predictions,
    load_prediction_artifact,
    select_prediction_example,
)
from ptbxl.inference import (
    FrozenInferenceBundle,
    InferenceConfig,
    InferenceResult,
    load_frozen_inference_bundle,
    load_inference_config,
    predict_ecg_record,
)
from ptbxl.interpretability import (
    InputAttribution,
    LeadAttributionSummary,
    compute_input_gradient_attribution,
    summarize_attribution_by_lead,
)
from ptbxl.experiments.final_analysis_report import (
    ATTRIBUTION_METHOD,
    FINAL_ANALYSIS_MODE,
    FINAL_ANALYSIS_REPORT_SCHEMA_VERSION,
    load_final_analysis_report,
)
from ptbxl.visualization import FINAL_FIGURE_FILENAMES, render_final_figures


FINAL_ANALYSIS_CONFIG_SCHEMA_VERSION = 1
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONFIG_TABLE_FIELDS = {
    "analysis": {
        "name",
        "mode",
        "minimum_combination_support",
        "maximum_problematic_combinations",
        "attribution_label",
        "attribution_kind",
        "attribution_window_samples",
        "prediction_tolerance",
    },
    "inputs": {
        "inference_config_path",
        "inference_config_sha256",
        "final_test_report_path",
        "final_test_report_sha256",
        "prediction_artifact_path",
        "prediction_artifact_sha256",
        "metadata_path",
        "metadata_sha256",
    },
    "outputs": {"report_path", "figure_directory"},
}


@dataclass(frozen=True)
class FinalAnalysisConfig:
    """Immutable sources and descriptive choices for the final analysis."""

    name: str
    mode: str
    minimum_combination_support: int
    maximum_problematic_combinations: int
    attribution_label: str
    attribution_kind: str
    attribution_window_samples: int
    prediction_tolerance: float
    inference_config_path: Path
    inference_config_sha256: str
    final_test_report_path: Path
    final_test_report_sha256: str
    prediction_artifact_path: Path
    prediction_artifact_sha256: str
    metadata_path: Path
    metadata_sha256: str
    report_path: Path
    figure_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", self.name
        ):
            raise ValueError(
                "Analysis name must use lowercase letters, numbers, _ or -"
            )
        if self.mode != FINAL_ANALYSIS_MODE:
            raise ValueError(f"Analysis mode must be {FINAL_ANALYSIS_MODE!r}")
        _positive_integer(
            "minimum_combination_support", self.minimum_combination_support
        )
        _positive_integer(
            "maximum_problematic_combinations",
            self.maximum_problematic_combinations,
        )
        _positive_integer("attribution_window_samples", self.attribution_window_samples)
        if self.attribution_window_samples > 1_000:
            raise ValueError("attribution_window_samples cannot exceed 1,000")
        if self.attribution_label != "HYP":
            raise ValueError("attribution_label must be 'HYP' for this final analysis")
        if self.attribution_kind != "false_negative":
            raise ValueError(
                "attribution_kind must be 'false_negative' for this final analysis"
            )
        if isinstance(self.prediction_tolerance, bool) or not isinstance(
            self.prediction_tolerance, (int, float)
        ):
            raise TypeError("prediction_tolerance must be numeric")
        tolerance = float(self.prediction_tolerance)
        if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-3:
            raise ValueError("prediction_tolerance must be in (0, 1e-3]")
        object.__setattr__(self, "prediction_tolerance", tolerance)

        suffixes = {
            "inference_config_path": ".toml",
            "final_test_report_path": ".json",
            "prediction_artifact_path": ".npz",
            "metadata_path": ".csv",
            "report_path": ".json",
        }
        for field, suffix in suffixes.items():
            value = getattr(self, field)
            if not isinstance(value, Path):
                raise TypeError(f"{field} must be a pathlib.Path")
            if value.suffix != suffix:
                raise ValueError(f"{field} must end in {suffix}")
        if not isinstance(self.figure_directory, Path):
            raise TypeError("figure_directory must be a pathlib.Path")
        if self.figure_directory == Path(".") or self.figure_directory.suffix:
            raise ValueError("figure_directory must be a dedicated suffix-free path")
        if self.report_path == self.final_test_report_path:
            raise ValueError("Analysis report must differ from final-test report")
        for field in (
            "inference_config_sha256",
            "final_test_report_sha256",
            "prediction_artifact_sha256",
            "metadata_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256")


def load_final_analysis_config(path: str | Path) -> FinalAnalysisConfig:
    """Load one strict final-analysis TOML configuration."""
    config_path = Path(path)
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    _require_exact_fields(
        "configuration", raw, {"schema_version", *CONFIG_TABLE_FIELDS}
    )
    if raw["schema_version"] != FINAL_ANALYSIS_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Final-analysis config schema_version must be "
            f"{FINAL_ANALYSIS_CONFIG_SCHEMA_VERSION}"
        )
    tables = {
        name: _require_mapping(raw, name, fields)
        for name, fields in CONFIG_TABLE_FIELDS.items()
    }
    analysis = tables["analysis"]
    inputs = tables["inputs"]
    outputs = tables["outputs"]
    return FinalAnalysisConfig(
        name=analysis["name"],
        mode=analysis["mode"],
        minimum_combination_support=analysis["minimum_combination_support"],
        maximum_problematic_combinations=analysis["maximum_problematic_combinations"],
        attribution_label=analysis["attribution_label"],
        attribution_kind=analysis["attribution_kind"],
        attribution_window_samples=analysis["attribution_window_samples"],
        prediction_tolerance=analysis["prediction_tolerance"],
        inference_config_path=_config_path(
            inputs["inference_config_path"], "inference_config_path"
        ),
        inference_config_sha256=inputs["inference_config_sha256"],
        final_test_report_path=_config_path(
            inputs["final_test_report_path"], "final_test_report_path"
        ),
        final_test_report_sha256=inputs["final_test_report_sha256"],
        prediction_artifact_path=_config_path(
            inputs["prediction_artifact_path"], "prediction_artifact_path"
        ),
        prediction_artifact_sha256=inputs["prediction_artifact_sha256"],
        metadata_path=_config_path(inputs["metadata_path"], "metadata_path"),
        metadata_sha256=inputs["metadata_sha256"],
        report_path=_config_path(outputs["report_path"], "report_path"),
        figure_directory=_config_path(outputs["figure_directory"], "figure_directory"),
    )


def run_final_analysis(
    config: FinalAnalysisConfig,
    config_path: str | Path,
    git_commit: str,
) -> dict[str, Any]:
    """Analyze saved predictions and explain one record without model changes."""
    if not isinstance(config, FinalAnalysisConfig):
        raise TypeError("config must be a FinalAnalysisConfig")
    if not isinstance(git_commit, str) or not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise ValueError("git_commit must be a lowercase hexadecimal commit")
    attributed_config_path = Path(config_path)
    if load_final_analysis_config(attributed_config_path) != config:
        raise ValueError("config does not match the attributed config_path")
    _require_outputs_absent(config)
    _verify_declared_hashes(config)

    final_report = _load_final_test_report(config.final_test_report_path)
    saved = load_prediction_artifact(config.prediction_artifact_path)
    if saved.split != "test":
        raise ValueError("Final analysis requires a test prediction artifact")
    prediction_fingerprint = fingerprint_predictions(saved.predictions, split="test")
    _validate_prediction_binding(
        config, final_report, saved.predictions, prediction_fingerprint
    )

    inference_config = load_inference_config(config.inference_config_path)
    bundle = load_frozen_inference_bundle(inference_config)
    _validate_bundle_binding(config, final_report, bundle)
    errors = analyze_prediction_errors(
        saved.predictions,
        bundle.thresholds.thresholds,
        minimum_combination_support=config.minimum_combination_support,
        maximum_problematic_combinations=config.maximum_problematic_combinations,
    )
    _reconcile_final_counts(errors, final_report)

    selected = select_prediction_example(
        saved.predictions,
        bundle.thresholds.thresholds,
        label=config.attribution_label,
        kind=config.attribution_kind,
    )
    record_path = _resolve_record_path(config, bundle, selected.ecg_id)
    verified_prediction = predict_ecg_record(
        bundle, record_path, record_id=f"ptbxl-test-ecg-{selected.ecg_id}"
    )
    verified_probabilities = np.asarray(
        [item.probability for item in verified_prediction.predictions]
    )
    verified_decisions = tuple(
        int(item.predicted) for item in verified_prediction.predictions
    )
    saved_probabilities = saved.predictions.probabilities[selected.row_index]
    if verified_decisions != selected.decisions:
        raise ValueError("Selected-record CPU decisions do not match sealed decisions")
    if not np.allclose(
        verified_probabilities,
        saved_probabilities,
        rtol=0.0,
        atol=config.prediction_tolerance,
    ):
        difference = float(np.max(np.abs(verified_probabilities - saved_probabilities)))
        raise ValueError(
            "Selected-record CPU prediction does not match sealed prediction; "
            f"maximum difference {difference}"
        )

    record = load_wfdb_record(record_path)
    validate_signal(record, bundle.standardizer.lead_names)
    standardized = bundle.standardizer.transform(record.signal, record.lead_names)
    attribution = compute_input_gradient_attribution(
        bundle.model,
        standardized,
        class_index=TARGET_SUPERCLASSES.index(config.attribution_label),
        device=bundle.device,
    )
    lead_summaries = summarize_attribution_by_lead(
        attribution,
        record.lead_names,
        sampling_frequency_hz=record.sampling_frequency,
        window_samples=config.attribution_window_samples,
    )
    figure_hashes = render_final_figures(
        config.figure_directory,
        experiment_report=bundle.baseline.report,
        predictions=saved.predictions,
        errors=errors,
        selected=selected,
        raw_signal=record.signal,
        attribution=attribution,
        lead_names=bundle.standardizer.lead_names,
    )
    try:
        report = _build_report(
            config,
            config_path=attributed_config_path,
            git_commit=git_commit,
            final_report=final_report,
            inference_config=inference_config,
            bundle=bundle,
            prediction_fingerprint=prediction_fingerprint,
            errors=errors,
            selected=selected,
            verified_prediction=verified_prediction,
            saved_probabilities=saved_probabilities,
            verified_probabilities=verified_probabilities,
            attribution=attribution,
            lead_summaries=lead_summaries,
            figure_hashes=figure_hashes,
        )
        write_json_report(report, config.report_path)
        loaded = load_final_analysis_report(config.report_path, verify_figures=True)
        if loaded != report:
            raise ValueError("Saved final-analysis report failed strict round-trip")
    except Exception:
        config.report_path.unlink(missing_ok=True)
        shutil.rmtree(config.figure_directory, ignore_errors=True)
        raise
    return report


def _verify_declared_hashes(config: FinalAnalysisConfig) -> None:
    expected = {
        "inference config": (
            config.inference_config_path,
            config.inference_config_sha256,
        ),
        "final-test report": (
            config.final_test_report_path,
            config.final_test_report_sha256,
        ),
        "prediction artifact": (
            config.prediction_artifact_path,
            config.prediction_artifact_sha256,
        ),
        "metadata": (config.metadata_path, config.metadata_sha256),
    }
    mismatched = [
        name
        for name, (path, digest) in expected.items()
        if compute_sha256(path) != digest
    ]
    if mismatched:
        raise ValueError(f"Final-analysis source SHA-256 mismatch: {mismatched}")


def _load_final_test_report(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Could not load final-test report for analysis") from error
    if not isinstance(raw, dict):
        raise TypeError("Final-test report must be a JSON object")
    required = {
        "schema_version",
        "event",
        "dataset",
        "sources",
        "outputs",
        "runtime",
        "ranking",
        "operating_point",
        "limitations",
    }
    _require_exact_fields("final-test report", raw, required)
    try:
        valid = (
            raw["schema_version"] == 1
            and raw["event"]["mode"] == "one_time_final_test"
            and raw["event"]["repeat_evaluation_forbidden"] is True
            and raw["event"]["test_driven_model_changes_forbidden"] is True
            and raw["dataset"]["split"] == "test"
            and raw["dataset"]["fold"] == 10
            and raw["dataset"]["targets"] == list(TARGET_SUPERCLASSES)
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Final-test report provenance is incomplete") from error
    if not valid:
        raise ValueError("Final-test report is not the immutable fold-10 event")
    return raw


def _validate_prediction_binding(
    config: FinalAnalysisConfig,
    final_report: Mapping[str, Any],
    predictions: PredictionSet,
    fingerprint: str,
) -> None:
    try:
        dataset = final_report["dataset"]
        artifact = final_report["outputs"]["prediction_artifact"]
    except (KeyError, TypeError) as error:
        raise ValueError("Final-test prediction provenance is missing") from error
    expected = {
        "artifact path": (
            artifact.get("path"),
            config.prediction_artifact_path.as_posix(),
        ),
        "artifact SHA-256": (artifact.get("sha256"), config.prediction_artifact_sha256),
        "prediction fingerprint": (artifact.get("fingerprint"), fingerprint),
        "sample count": (dataset.get("samples"), predictions.targets.shape[0]),
    }
    mismatched = [name for name, values in expected.items() if values[0] != values[1]]
    if mismatched:
        raise ValueError(f"Final prediction binding mismatch: {mismatched}")


def _validate_bundle_binding(
    config: FinalAnalysisConfig,
    final_report: Mapping[str, Any],
    bundle: FrozenInferenceBundle,
) -> None:
    if bundle.baseline.config.metadata_path != config.metadata_path:
        raise ValueError("Analysis metadata path does not match frozen baseline")
    sources = final_report["sources"]
    actual = {
        "experiment_config": bundle.baseline.hashes.experiment_config,
        "experiment_report": bundle.baseline.hashes.experiment_report,
        "checkpoint": bundle.baseline.hashes.checkpoint,
        "standardizer": bundle.baseline.hashes.standardizer,
        "threshold_artifact": bundle.threshold_artifact_sha256,
    }
    mismatched = [
        name
        for name, digest in actual.items()
        if sources.get(name, {}).get("sha256") != digest
    ]
    if mismatched:
        raise ValueError(f"Final report and inference bundle disagree: {mismatched}")


def _reconcile_final_counts(
    errors: ErrorAnalysis, final_report: Mapping[str, Any]
) -> None:
    recorded = final_report["operating_point"]["per_class"]
    if not isinstance(recorded, list) or len(recorded) != len(errors.per_class):
        raise ValueError("Final report per-class operating metrics are invalid")
    count_fields = (
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    )
    mismatched = []
    for actual, expected in zip(errors.per_class, recorded, strict=True):
        if expected.get("label") != actual.label or any(
            expected.get(field) != getattr(actual, field) for field in count_fields
        ):
            mismatched.append(actual.label)
    if mismatched:
        raise ValueError(
            f"Saved prediction counts disagree with final report: {mismatched}"
        )


def _resolve_record_path(
    config: FinalAnalysisConfig,
    bundle: FrozenInferenceBundle,
    ecg_id: int,
) -> Path:
    metadata = pd.read_csv(config.metadata_path, usecols=["ecg_id", "filename_lr"])
    rows = metadata.loc[metadata["ecg_id"] == ecg_id, "filename_lr"]
    if len(rows) != 1 or not isinstance(rows.iloc[0], str) or not rows.iloc[0]:
        raise ValueError("Selected ECG does not resolve uniquely in frozen metadata")
    return bundle.baseline.config.dataset_root / rows.iloc[0]


def _build_report(
    config: FinalAnalysisConfig,
    *,
    config_path: Path,
    git_commit: str,
    final_report: Mapping[str, Any],
    inference_config: InferenceConfig,
    bundle: FrozenInferenceBundle,
    prediction_fingerprint: str,
    errors: ErrorAnalysis,
    selected: SelectedPredictionExample,
    verified_prediction: InferenceResult,
    saved_probabilities: np.ndarray,
    verified_probabilities: np.ndarray,
    attribution: InputAttribution,
    lead_summaries: tuple[LeadAttributionSummary, ...],
    figure_hashes: Mapping[str, str],
) -> dict[str, Any]:
    baseline = bundle.baseline
    top_leads = sorted(
        lead_summaries,
        key=lambda item: (-item.attribution_fraction, item.lead),
    )[:3]
    return {
        "schema_version": FINAL_ANALYSIS_REPORT_SCHEMA_VERSION,
        "analysis": {
            "name": config.name,
            "mode": config.mode,
            "git_commit": git_commit,
            "posthoc_only": True,
            "model_changes_forbidden": True,
            "full_test_inference_repeated": False,
        },
        "dataset": {
            "name": "PTB-XL",
            "version": final_report["dataset"]["version"],
            "split": "test",
            "fold": 10,
            "samples": errors.samples,
            "targets": list(TARGET_SUPERCLASSES),
        },
        "sources": {
            "analysis_config": _source(config_path),
            "inference_config": _source(config.inference_config_path),
            "final_test_report": _source(config.final_test_report_path),
            "prediction_artifact": _source(config.prediction_artifact_path),
            "metadata": _source(config.metadata_path),
            "experiment_config": {
                "path": inference_config.experiment_config_path.as_posix(),
                "sha256": baseline.hashes.experiment_config,
            },
            "experiment_report": {
                "path": inference_config.experiment_report_path.as_posix(),
                "sha256": baseline.hashes.experiment_report,
            },
            "checkpoint": {
                "path": baseline.config.checkpoint_path.as_posix(),
                "sha256": baseline.hashes.checkpoint,
            },
            "standardizer": {
                "path": baseline.config.standardizer_path.as_posix(),
                "sha256": baseline.hashes.standardizer,
            },
            "threshold_artifact": {
                "path": inference_config.threshold_artifact_path.as_posix(),
                "sha256": bundle.threshold_artifact_sha256,
            },
        },
        "error_analysis": {
            "decision_rule": bundle.thresholds.thresholds.decision_rule,
            "samples": errors.samples,
            "exact_matches": errors.exact_matches,
            "exact_match_rate": errors.exact_match_rate,
            "label_errors": errors.label_errors,
            "hamming_loss": errors.hamming_loss,
            "per_class": [asdict(item) for item in errors.per_class],
            "combinations": [_combination_dict(item) for item in errors.combinations],
            "problematic_combinations": [
                _combination_dict(item) for item in errors.problematic_combinations
            ],
            "minimum_combination_support": errors.minimum_combination_support,
        },
        "attribution": {
            "method": ATTRIBUTION_METHOD,
            "label": attribution.label,
            "selection": {
                "kind": selected.kind,
                "rule": "most_confident_then_smallest_ecg_id",
            },
            "record": {
                "ecg_id": selected.ecg_id,
                "signal_sha256": verified_prediction.signal_sha256,
                "targets": dict(
                    zip(TARGET_SUPERCLASSES, selected.targets, strict=True)
                ),
                "decisions": dict(
                    zip(TARGET_SUPERCLASSES, selected.decisions, strict=True)
                ),
                "saved_probabilities": dict(
                    zip(
                        TARGET_SUPERCLASSES,
                        map(float, saved_probabilities),
                        strict=True,
                    )
                ),
                "verified_cpu_probabilities": dict(
                    zip(
                        TARGET_SUPERCLASSES,
                        map(float, verified_probabilities),
                        strict=True,
                    )
                ),
                "maximum_probability_difference": float(
                    np.max(np.abs(saved_probabilities - verified_probabilities))
                ),
            },
            "logit": attribution.logit,
            "window_samples": config.attribution_window_samples,
            "per_lead": [asdict(item) for item in lead_summaries],
            "top_leads": [item.lead for item in top_leads],
            "interpretation": (
                "Larger values identify input regions to which this output logit "
                "was locally sensitive; they do not establish clinical relevance, "
                "causality or diagnostic localization."
            ),
        },
        "figures": [
            {
                "name": Path(filename).stem,
                "path": (config.figure_directory / filename).as_posix(),
                "sha256": figure_hashes[filename],
            }
            for filename in FINAL_FIGURE_FILENAMES
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "torch": torch.__version__,
            "device": bundle.device.type,
            "prediction_artifact_fingerprint": prediction_fingerprint,
        },
        "limitations": [
            "This is post-hoc analysis of one internal PTB-XL test fold.",
            "Test observations were not used to modify the frozen model or thresholds.",
            "No repeated-seed uncertainty or external validation is available.",
            "Input-gradient attribution can be noisy and is not a clinical explanation.",
            "The selected error is illustrative and not representative of all failures.",
        ],
    }


def _source(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": compute_sha256(path)}


def _combination_dict(item: Any) -> dict[str, Any]:
    return {
        "labels": list(item.labels),
        "support": item.support,
        "exact_matches": item.exact_matches,
        "exact_match_rate": item.exact_match_rate,
        "label_errors": item.label_errors,
    }


def _require_outputs_absent(config: FinalAnalysisConfig) -> None:
    existing = [
        path.as_posix()
        for path in (config.report_path, config.figure_directory)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Final analysis will not overwrite outputs: {existing}")


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


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


def _config_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty path string")
    return Path(value)
