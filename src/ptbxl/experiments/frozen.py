"""Validation of the frozen baseline bundle shared by later stages."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.evaluation.thresholds import FrozenThresholds
from ptbxl.experiments.baseline import ExperimentConfig, load_experiment_config
from ptbxl.training import CheckpointProvenance, LoadedCheckpoint


@dataclass(frozen=True)
class FrozenBaselineHashes:
    """Exact file identities for the frozen baseline bundle."""

    experiment_config: str
    experiment_report: str
    checkpoint: str
    cohort: str | None
    metadata: str | None
    standardizer: str


@dataclass(frozen=True)
class FrozenBaseline:
    """Validated baseline configuration, report, provenance and file hashes."""

    config: ExperimentConfig
    report: dict[str, Any]
    provenance: CheckpointProvenance
    hashes: FrozenBaselineHashes


def load_frozen_baseline(
    experiment_config_path: str | Path,
    experiment_report_path: str | Path,
    checkpoint_path: str | Path,
    *,
    verify_dataset_sources: bool = True,
) -> FrozenBaseline:
    """Cross-check a frozen baseline, optionally including training tables."""
    if not isinstance(verify_dataset_sources, bool):
        raise TypeError("verify_dataset_sources must be boolean")
    config_path = Path(experiment_config_path)
    report_path = Path(experiment_report_path)
    requested_checkpoint = Path(checkpoint_path)
    config = load_experiment_config(config_path)
    if requested_checkpoint != config.checkpoint_path:
        raise ValueError("Checkpoint path does not match baseline config")
    report = _load_json_object(report_path, "baseline experiment report")
    hashes = FrozenBaselineHashes(
        experiment_config=compute_sha256(config_path),
        experiment_report=compute_sha256(report_path),
        checkpoint=compute_sha256(requested_checkpoint),
        cohort=compute_sha256(config.cohort_path) if verify_dataset_sources else None,
        metadata=(
            compute_sha256(config.metadata_path) if verify_dataset_sources else None
        ),
        standardizer=compute_sha256(config.standardizer_path),
    )
    try:
        report_config = report["configuration"]
        report_checkpoint = report["artifacts"]["checkpoint"]
        report_sources = report["sources"]
        report_dataset = report["dataset"]
        report_run = report["run"]
        report_training = report["training"]
    except (KeyError, TypeError) as error:
        raise ValueError("Baseline report is missing required provenance") from error
    objects = (
        report_config,
        report_checkpoint,
        report_sources,
        report_dataset,
        report_run,
        report_training,
    )
    if not all(isinstance(value, dict) for value in objects):
        raise ValueError("Baseline report provenance must use JSON objects")
    try:
        expected_hashes = {
            "baseline config": (report_config["sha256"], hashes.experiment_config),
            "checkpoint": (report_checkpoint["sha256"], hashes.checkpoint),
            "standardizer": (
                report_sources["standardizer"]["sha256"],
                hashes.standardizer,
            ),
        }
        expected_paths = {
            "baseline config": (report_config["path"], config_path.as_posix()),
            "checkpoint": (report_checkpoint["path"], requested_checkpoint.as_posix()),
            "cohort": (report_sources["cohort"]["path"], config.cohort_path.as_posix()),
            "metadata": (
                report_sources["metadata"]["path"],
                config.metadata_path.as_posix(),
            ),
            "standardizer": (
                report_sources["standardizer"]["path"],
                config.standardizer_path.as_posix(),
            ),
        }
    except (KeyError, TypeError) as error:
        raise ValueError("Baseline report source provenance is invalid") from error
    if verify_dataset_sources:
        expected_hashes.update(
            {
                "cohort": (report_sources["cohort"]["sha256"], hashes.cohort),
                "metadata": (report_sources["metadata"]["sha256"], hashes.metadata),
            }
        )
    mismatched_hashes = [
        name
        for name, (recorded, actual) in expected_hashes.items()
        if recorded != actual
    ]
    if mismatched_hashes:
        raise ValueError(f"Baseline source SHA-256 mismatch: {mismatched_hashes}")
    mismatched_paths = [
        name
        for name, (recorded, actual) in expected_paths.items()
        if recorded != actual
    ]
    if mismatched_paths:
        raise ValueError(f"Baseline source path mismatch: {mismatched_paths}")
    expected_dataset = {
        "name": "PTB-XL",
        "version": config.dataset_version,
        "cohort": config.cohort_name,
        "targets": list(TARGET_SUPERCLASSES),
        "splits_used": {
            "train": config.expected_train_records,
            "validation": config.expected_validation_records,
        },
    }
    for field, expected in expected_dataset.items():
        if report_dataset.get(field) != expected:
            if field == "splits_used":
                raise ValueError("Baseline report split counts do not match")
            raise ValueError(f"Baseline report dataset {field} does not match")
    provenance = _checkpoint_provenance(
        report_checkpoint,
        report_run,
        config,
        hashes.standardizer,
    )
    return FrozenBaseline(config, report, provenance, hashes)


def validate_loaded_checkpoint(
    loaded: LoadedCheckpoint,
    frozen: FrozenBaseline,
) -> None:
    """Require restored checkpoint selection to equal the attributed report."""
    try:
        expected_epoch = frozen.report["training"]["best_epoch"]
        expected_loss = frozen.report["training"]["best_validation_loss"]
    except (KeyError, TypeError) as error:
        raise ValueError("Baseline report checkpoint selection is invalid") from error
    if loaded.provenance != frozen.provenance:
        raise ValueError("Loaded checkpoint provenance does not match frozen baseline")
    if loaded.epoch != expected_epoch or loaded.validation_loss != expected_loss:
        raise ValueError("Loaded checkpoint does not match baseline selection")


def validate_frozen_baseline_hashes(
    frozen: FrozenBaseline,
    *,
    experiment_config_sha256: str,
    experiment_report_sha256: str,
    checkpoint_sha256: str,
    standardizer_sha256: str,
) -> None:
    """Require a frozen baseline to equal four externally declared hashes."""
    if not isinstance(frozen, FrozenBaseline):
        raise TypeError("frozen must be a FrozenBaseline")
    expected = {
        "experiment config": (
            frozen.hashes.experiment_config,
            experiment_config_sha256,
        ),
        "experiment report": (
            frozen.hashes.experiment_report,
            experiment_report_sha256,
        ),
        "checkpoint": (frozen.hashes.checkpoint, checkpoint_sha256),
        "standardizer": (frozen.hashes.standardizer, standardizer_sha256),
    }
    mismatched = [name for name, values in expected.items() if values[0] != values[1]]
    if mismatched:
        raise ValueError(f"Frozen baseline SHA-256 mismatch: {mismatched}")


def validate_frozen_threshold_binding(
    frozen: FrozenBaseline,
    thresholds: FrozenThresholds,
) -> None:
    """Require thresholds to originate from the same frozen baseline bundle."""
    if not isinstance(frozen, FrozenBaseline):
        raise TypeError("frozen must be a FrozenBaseline")
    if not isinstance(thresholds, FrozenThresholds):
        raise TypeError("thresholds must be FrozenThresholds")
    expected = {
        "dataset name": (thresholds.dataset_name, "PTB-XL"),
        "dataset version": (thresholds.dataset_version, frozen.config.dataset_version),
        "cohort": (thresholds.cohort_name, frozen.config.cohort_name),
        "experiment config": (
            thresholds.experiment_config_sha256,
            frozen.hashes.experiment_config,
        ),
        "experiment report": (
            thresholds.experiment_report_sha256,
            frozen.hashes.experiment_report,
        ),
        "standardizer": (
            thresholds.preprocessing_sha256,
            frozen.hashes.standardizer,
        ),
    }
    mismatched = [name for name, values in expected.items() if values[0] != values[1]]
    if mismatched:
        raise ValueError(f"Threshold artifact binding mismatch: {mismatched}")


def _checkpoint_provenance(
    checkpoint: dict[str, Any],
    run: dict[str, Any],
    config: ExperimentConfig,
    standardizer_sha256: str,
) -> CheckpointProvenance:
    try:
        raw = checkpoint["provenance"]
        if not isinstance(raw, dict):
            raise TypeError
        provenance = CheckpointProvenance(**raw)
        report_git_commit = run["git_commit"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Baseline checkpoint provenance is invalid") from error
    expected = CheckpointProvenance(
        dataset_version=f"PTB-XL-{config.dataset_version}",
        cohort_name=config.cohort_name,
        preprocessing_sha256=standardizer_sha256,
        model_name="small_ecg_cnn",
        seed=config.seed,
        git_commit=provenance.git_commit,
    )
    if provenance != expected or report_git_commit != provenance.git_commit:
        raise ValueError("Baseline checkpoint provenance does not match configuration")
    return provenance


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {name}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value
