"""Configured selection of a frozen validation operating point."""

import math
import platform
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from ptbxl.data import (
    PTBXLDataset,
    TARGET_SUPERCLASSES,
    build_dataloader,
    build_sample_index,
)
from ptbxl.data.reporting import compute_sha256, write_json_report
from ptbxl.evaluation import (
    THRESHOLD_METHOD,
    ThresholdSelection,
    evaluate_validation,
    select_validation_thresholds,
)
from ptbxl.experiments.baseline import ExperimentConfig
from ptbxl.experiments.frozen import (
    load_frozen_baseline,
    validate_loaded_checkpoint,
)
from ptbxl.models import SmallECGCNN
from ptbxl.preprocessing import load_global_standardizer
from ptbxl.training import (
    configure_deterministic_execution,
    load_training_checkpoint,
    resolve_device,
    seed_random_generators,
)


THRESHOLD_CONFIG_SCHEMA_VERSION = 1
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
CONFIG_TABLE_FIELDS = {
    "selection": {"name", "method", "device", "batch_size", "num_workers"},
    "inputs": {"experiment_config_path", "experiment_report_path", "checkpoint_path"},
    "output": {"artifact_path"},
}


@dataclass(frozen=True)
class ThresholdExperimentConfig:
    """Immutable configuration for validation threshold selection."""

    name: str
    method: str
    device: str
    batch_size: int
    num_workers: int
    experiment_config_path: Path
    experiment_report_path: Path
    checkpoint_path: Path
    artifact_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", self.name
        ):
            raise ValueError(
                "Threshold selection name must use lowercase letters, numbers, _ or -"
            )
        if self.method != THRESHOLD_METHOD:
            raise ValueError(f"Threshold method must be {THRESHOLD_METHOD!r}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Threshold device must be 'auto', 'cpu', or 'cuda'")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("Threshold batch_size must be a positive integer")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("Threshold num_workers must be a non-negative integer")
        for field in (
            "experiment_config_path",
            "experiment_report_path",
            "checkpoint_path",
            "artifact_path",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a pathlib.Path")
        if self.experiment_config_path.suffix != ".toml":
            raise ValueError("experiment_config_path must end in .toml")
        if self.experiment_report_path.suffix != ".json":
            raise ValueError("experiment_report_path must end in .json")
        if self.checkpoint_path.suffix != ".pt":
            raise ValueError("checkpoint_path must end in .pt")
        if self.artifact_path.suffix != ".json":
            raise ValueError("artifact_path must end in .json")
        input_paths = {
            self.experiment_config_path,
            self.experiment_report_path,
            self.checkpoint_path,
        }
        if self.artifact_path in input_paths:
            raise ValueError("artifact_path must not overwrite an input")


def load_threshold_experiment_config(
    path: str | Path,
) -> ThresholdExperimentConfig:
    """Load one strict versioned threshold-selection TOML."""
    config_path = Path(path)
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    expected_top_level = {"schema_version", *CONFIG_TABLE_FIELDS}
    _require_exact_fields("configuration", raw, expected_top_level)
    if raw["schema_version"] != THRESHOLD_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Threshold config schema_version must be {THRESHOLD_CONFIG_SCHEMA_VERSION}"
        )
    tables = {
        name: _require_table(raw, name, fields)
        for name, fields in CONFIG_TABLE_FIELDS.items()
    }
    selection = tables["selection"]
    inputs = tables["inputs"]
    output = tables["output"]
    return ThresholdExperimentConfig(
        name=selection["name"],
        method=selection["method"],
        device=selection["device"],
        batch_size=selection["batch_size"],
        num_workers=selection["num_workers"],
        experiment_config_path=_config_path(
            inputs["experiment_config_path"], "experiment_config_path"
        ),
        experiment_report_path=_config_path(
            inputs["experiment_report_path"], "experiment_report_path"
        ),
        checkpoint_path=_config_path(inputs["checkpoint_path"], "checkpoint_path"),
        artifact_path=_config_path(output["artifact_path"], "artifact_path"),
    )


def run_validation_threshold_selection(
    config: ThresholdExperimentConfig,
    config_path: str | Path,
    git_commit: str,
) -> dict[str, Any]:
    """Restore the frozen baseline and select thresholds on validation only."""
    if not isinstance(config, ThresholdExperimentConfig):
        raise TypeError("config must be a ThresholdExperimentConfig")
    if not isinstance(git_commit, str) or not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise ValueError("git_commit must be a lowercase hexadecimal commit")
    config_path = Path(config_path)
    if load_threshold_experiment_config(config_path) != config:
        raise ValueError("config does not match the attributed config_path")

    frozen = load_frozen_baseline(
        config.experiment_config_path,
        config.experiment_report_path,
        config.checkpoint_path,
    )
    baseline_config = frozen.config
    baseline_report = frozen.report
    input_hashes = {
        "selection_config": compute_sha256(config_path),
        "experiment_config": frozen.hashes.experiment_config,
        "experiment_report": frozen.hashes.experiment_report,
        "checkpoint": frozen.hashes.checkpoint,
        "standardizer": frozen.hashes.standardizer,
    }

    cohort = pd.read_csv(baseline_config.cohort_path)
    metadata = pd.read_csv(
        baseline_config.metadata_path,
        usecols=["ecg_id", "filename_lr"],
    )
    sample_index = build_sample_index(cohort, metadata)
    validation_index = sample_index.loc[
        sample_index["split"] == "validation"
    ].reset_index(drop=True)
    if len(validation_index) != baseline_config.expected_validation_records:
        raise ValueError(
            "Expected "
            f"{baseline_config.expected_validation_records} validation records, "
            f"got {len(validation_index)}"
        )

    configure_deterministic_execution()
    seed_random_generators(baseline_config.seed)
    device = resolve_device(config.device)
    standardizer = load_global_standardizer(baseline_config.standardizer_path)
    validation_dataset = PTBXLDataset(
        validation_index,
        baseline_config.dataset_root,
        standardizer,
        split="validation",
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=baseline_config.seed,
        num_workers=config.num_workers,
    )

    model = SmallECGCNN(baseline_config.model_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(baseline_config.learning_rate),
        weight_decay=float(baseline_config.weight_decay),
    )
    loaded = load_training_checkpoint(
        config.checkpoint_path,
        model,
        optimizer,
        device=device,
        expected_provenance=frozen.provenance,
    )
    validate_loaded_checkpoint(loaded, frozen)
    evaluation = evaluate_validation(model, validation_loader, device)
    _validate_reproduced_ranking(evaluation.metrics, baseline_report)
    selection = select_validation_thresholds(evaluation.predictions)
    artifact = _build_threshold_artifact(
        config,
        config_path=config_path,
        git_commit=git_commit,
        baseline_config=baseline_config,
        input_hashes=input_hashes,
        selection=selection,
        device=device,
    )
    write_json_report(artifact, config.artifact_path)
    return artifact


def _build_threshold_artifact(
    config: ThresholdExperimentConfig,
    *,
    config_path: Path,
    git_commit: str,
    baseline_config: ExperimentConfig,
    input_hashes: Mapping[str, str],
    selection: ThresholdSelection,
    device: torch.device,
) -> dict[str, Any]:
    metrics = selection.metrics
    return {
        "schema_version": 1,
        "selection": {
            "name": config.name,
            "git_commit": git_commit,
            "method": selection.thresholds.method,
            "decision_rule": selection.thresholds.decision_rule,
            "tie_break": selection.thresholds.tie_break,
            "thresholds": selection.thresholds.by_label(),
        },
        "dataset": {
            "name": "PTB-XL",
            "version": baseline_config.dataset_version,
            "cohort": baseline_config.cohort_name,
            "split": "validation",
            "samples": metrics.samples,
            "targets": list(TARGET_SUPERCLASSES),
        },
        "sources": {
            "selection_config": {
                "path": config_path.as_posix(),
                "sha256": input_hashes["selection_config"],
            },
            "experiment_config": {
                "path": config.experiment_config_path.as_posix(),
                "sha256": input_hashes["experiment_config"],
            },
            "experiment_report": {
                "path": config.experiment_report_path.as_posix(),
                "sha256": input_hashes["experiment_report"],
            },
            "checkpoint": {
                "path": config.checkpoint_path.as_posix(),
                "sha256": input_hashes["checkpoint"],
            },
            "standardizer": {
                "path": baseline_config.standardizer_path.as_posix(),
                "sha256": input_hashes["standardizer"],
            },
            "validation_predictions_sha256": (selection.validation_predictions_sha256),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hardware": {
                "requested_device": config.device,
                "resolved_device": device.type,
                **(
                    {"accelerator_name": torch.cuda.get_device_name(device)}
                    if device.type == "cuda"
                    else {}
                ),
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "validation": {
            "samples": metrics.samples,
            "per_class": [asdict(item) for item in metrics.per_class],
            "macro": asdict(metrics.macro),
            "micro": asdict(metrics.micro),
        },
        "limitations": [
            "Thresholds maximize per-class F1 on internal validation only.",
            "Validation operating metrics reuse the threshold-selection data and are optimistic.",
            "No clinical error costs or operating requirements were available.",
            "No final-test sample or metric was accessed.",
            "This operating point is not evidence of clinical utility.",
        ],
    }


def _validate_reproduced_ranking(metrics: Any, report: dict[str, Any]) -> None:
    try:
        expected = report["validation"]
        comparisons = {
            "samples": (metrics.samples, expected["samples"]),
            "macro_auroc": (metrics.macro_auroc, expected["macro_auroc"]),
            "macro_auprc": (metrics.macro_auprc, expected["macro_auprc"]),
            "micro_auroc": (metrics.micro_auroc, expected["micro_auroc"]),
            "micro_auprc": (metrics.micro_auprc, expected["micro_auprc"]),
        }
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Baseline validation metrics are invalid") from error
    mismatched = []
    for name, (actual, wanted) in comparisons.items():
        if name == "samples":
            equal = actual == wanted
        else:
            equal = isinstance(wanted, (int, float)) and math.isclose(
                actual, float(wanted), rel_tol=0.0, abs_tol=1e-12
            )
        if not equal:
            mismatched.append(name)
    if mismatched:
        raise ValueError(f"Reproduced validation metrics do not match: {mismatched}")


def _require_table(
    raw: Mapping[str, Any], name: str, expected_fields: set[str]
) -> Mapping[str, Any]:
    value = raw[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration table {name!r} must be a mapping")
    _require_exact_fields(name, value, expected_fields)
    return value


def _require_exact_fields(
    name: str, values: Mapping[str, Any], expected: set[str]
) -> None:
    actual = set(values)
    if missing := sorted(expected - actual):
        raise KeyError(f"{name} is missing fields: {missing}")
    if unexpected := sorted(actual - expected):
        raise ValueError(f"{name} has unexpected fields: {unexpected}")


def _config_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty path string")
    return Path(value)
