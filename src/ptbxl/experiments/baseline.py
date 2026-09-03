"""Configured, traceable execution of the first real PTB-XL baseline."""

import json
import math
import platform
import re
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from ptbxl.data import (
    TARGET_SUPERCLASSES,
    PTBXLDataset,
    build_dataloader,
    build_sample_index,
)
from ptbxl.data.reporting import compute_sha256, write_json_report
from ptbxl.evaluation import ValidationEvaluation, evaluate_validation
from ptbxl.models import ECGCNNConfig, SmallECGCNN
from ptbxl.preprocessing import load_global_standardizer
from ptbxl.training import (
    CheckpointProvenance,
    FitConfig,
    FitEpochResult,
    FitResult,
    configure_deterministic_execution,
    fit,
    resolve_device,
    seed_random_generators,
)


EXPERIMENT_SCHEMA_VERSION = 1
DATASET_VERSION = "1.0.3"
MODEL_NAME = "small_ecg_cnn"
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
CONFIG_TABLE_FIELDS = {
    "experiment": {"name", "seed", "device"},
    "data": {
        "dataset_version",
        "cohort_name",
        "cohort_path",
        "metadata_path",
        "dataset_root",
        "standardizer_path",
        "expected_train_records",
        "expected_validation_records",
    },
    "model": {"feature_channels", "kernel_sizes", "dropout_probability"},
    "training": {
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
    },
    "outputs": {"checkpoint_path", "report_path"},
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete immutable configuration for one baseline experiment."""

    name: str
    seed: int
    device: str
    dataset_version: str
    cohort_name: str
    cohort_path: Path
    metadata_path: Path
    dataset_root: Path
    standardizer_path: Path
    expected_train_records: int
    expected_validation_records: int
    feature_channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    dropout_probability: float
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    checkpoint_path: Path
    report_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", self.name
        ):
            raise ValueError(
                "Experiment name must use lowercase letters, numbers, _ or -"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise ValueError("Experiment seed must be an integer from 0 to 2^32 - 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Experiment device must be 'auto', 'cpu', or 'cuda'")
        if self.dataset_version != DATASET_VERSION:
            raise ValueError(f"Dataset version must be {DATASET_VERSION}")
        if not isinstance(self.cohort_name, str) or not self.cohort_name.strip():
            raise ValueError("cohort_name must be a non-empty string")
        for field in (
            "cohort_path",
            "metadata_path",
            "dataset_root",
            "standardizer_path",
            "checkpoint_path",
            "report_path",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a pathlib.Path")
        for field in (
            "expected_train_records",
            "expected_validation_records",
            "epochs",
            "batch_size",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        _require_finite_number("learning_rate", self.learning_rate, positive=True)
        _require_finite_number("weight_decay", self.weight_decay, non_negative=True)
        ECGCNNConfig(
            feature_channels=self.feature_channels,
            kernel_sizes=self.kernel_sizes,
            dropout_probability=self.dropout_probability,
        )
        FitConfig(self.epochs)
        if self.checkpoint_path.suffix != ".pt":
            raise ValueError("checkpoint_path must end in .pt")
        if self.report_path.suffix != ".json":
            raise ValueError("report_path must end in .json")
        if self.checkpoint_path == self.report_path:
            raise ValueError("checkpoint_path and report_path must differ")

    @property
    def model_config(self) -> ECGCNNConfig:
        """Return the validated CNN configuration."""
        return ECGCNNConfig(
            feature_channels=self.feature_channels,
            kernel_sizes=self.kernel_sizes,
            dropout_probability=self.dropout_probability,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one strict versioned TOML baseline configuration."""
    config_path = Path(path)
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    expected_top_level = {"schema_version", *CONFIG_TABLE_FIELDS}
    _require_exact_fields("configuration", raw, expected_top_level)
    if raw["schema_version"] != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Experiment schema_version must be {EXPERIMENT_SCHEMA_VERSION}"
        )
    tables = {
        name: _require_table(raw, name, fields)
        for name, fields in CONFIG_TABLE_FIELDS.items()
    }
    experiment = tables["experiment"]
    data = tables["data"]
    model = tables["model"]
    training = tables["training"]
    outputs = tables["outputs"]
    return ExperimentConfig(
        name=experiment["name"],
        seed=experiment["seed"],
        device=experiment["device"],
        dataset_version=data["dataset_version"],
        cohort_name=data["cohort_name"],
        cohort_path=_config_path(data["cohort_path"], "cohort_path"),
        metadata_path=_config_path(data["metadata_path"], "metadata_path"),
        dataset_root=_config_path(data["dataset_root"], "dataset_root"),
        standardizer_path=_config_path(data["standardizer_path"], "standardizer_path"),
        expected_train_records=data["expected_train_records"],
        expected_validation_records=data["expected_validation_records"],
        feature_channels=_integer_tuple(model["feature_channels"], "feature_channels"),
        kernel_sizes=_integer_tuple(model["kernel_sizes"], "kernel_sizes"),
        dropout_probability=model["dropout_probability"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        checkpoint_path=_config_path(outputs["checkpoint_path"], "checkpoint_path"),
        report_path=_config_path(outputs["report_path"], "report_path"),
    )


def get_clean_git_commit(repository_root: str | Path) -> str:
    """Return HEAD only when the repository worktree is clean."""
    root = Path(repository_root)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not resolve Git provenance") from error
    if status.strip():
        raise RuntimeError("Experiment requires a clean Git worktree")
    if not GIT_COMMIT_PATTERN.fullmatch(commit):
        raise RuntimeError("Git HEAD is not a valid commit identity")
    return commit


def run_baseline_experiment(
    config: ExperimentConfig,
    config_path: str | Path,
    git_commit: str,
    *,
    on_epoch_end: Callable[[FitEpochResult], None] | None = None,
) -> dict[str, Any]:
    """Train, evaluate and persist one fully attributed baseline experiment."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(git_commit, str) or not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise ValueError("git_commit must be a lowercase hexadecimal commit")
    config_path = Path(config_path)
    if load_experiment_config(config_path) != config:
        raise ValueError("config does not match the attributed config_path")
    config_sha256 = compute_sha256(config_path)
    cohort_sha256 = compute_sha256(config.cohort_path)
    metadata_sha256 = compute_sha256(config.metadata_path)
    standardizer_sha256 = compute_sha256(config.standardizer_path)
    standardizer = load_global_standardizer(config.standardizer_path)
    _validate_standardizer_sources(
        config.standardizer_path,
        cohort_sha256=cohort_sha256,
        metadata_sha256=metadata_sha256,
    )

    cohort = pd.read_csv(config.cohort_path)
    metadata = pd.read_csv(
        config.metadata_path,
        usecols=["ecg_id", "filename_lr"],
    )
    sample_index = build_sample_index(cohort, metadata)
    train_index = sample_index.loc[sample_index["split"] == "train"].reset_index(
        drop=True
    )
    validation_index = sample_index.loc[
        sample_index["split"] == "validation"
    ].reset_index(drop=True)
    _require_split_count("train", train_index, config.expected_train_records)
    _require_split_count(
        "validation", validation_index, config.expected_validation_records
    )

    configure_deterministic_execution()
    seed_random_generators(config.seed)
    device = resolve_device(config.device)
    train_dataset = PTBXLDataset(
        train_index,
        config.dataset_root,
        standardizer,
        split="train",
    )
    validation_dataset = PTBXLDataset(
        validation_index,
        config.dataset_root,
        standardizer,
        split="validation",
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    model = SmallECGCNN(config.model_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    provenance = CheckpointProvenance(
        dataset_version=f"PTB-XL-{config.dataset_version}",
        cohort_name=config.cohort_name,
        preprocessing_sha256=standardizer_sha256,
        model_name=MODEL_NAME,
        seed=config.seed,
        git_commit=git_commit,
    )
    fit_result = fit(
        model,
        train_loader,
        validation_loader,
        optimizer,
        nn.BCEWithLogitsLoss(),
        device,
        FitConfig(config.epochs),
        config.checkpoint_path,
        provenance,
        on_epoch_end=on_epoch_end,
    )
    evaluation = evaluate_validation(model, validation_loader, device)
    checkpoint_sha256 = compute_sha256(config.checkpoint_path)
    report = _build_experiment_report(
        config,
        config_path=config_path,
        config_sha256=config_sha256,
        git_commit=git_commit,
        cohort_sha256=cohort_sha256,
        metadata_sha256=metadata_sha256,
        standardizer_sha256=standardizer_sha256,
        checkpoint_sha256=checkpoint_sha256,
        train_index=train_index,
        fit_result=fit_result,
        evaluation=evaluation,
        device=device,
        provenance=provenance,
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    write_json_report(report, config.report_path)
    return report


def _build_experiment_report(
    config: ExperimentConfig,
    *,
    config_path: Path,
    config_sha256: str,
    git_commit: str,
    cohort_sha256: str,
    metadata_sha256: str,
    standardizer_sha256: str,
    checkpoint_sha256: str,
    train_index: pd.DataFrame,
    fit_result: FitResult,
    evaluation: ValidationEvaluation,
    device: torch.device,
    provenance: CheckpointProvenance,
    trainable_parameters: int,
) -> dict[str, Any]:
    metrics = evaluation.metrics
    train_prevalence = {
        label: {
            "count": int(train_index[label].sum()),
            "fraction": float(train_index[label].mean()),
        }
        for label in TARGET_SUPERCLASSES
    }
    hardware: dict[str, Any] = {
        "requested_device": config.device,
        "resolved_device": device.type,
    }
    if device.type == "cuda":
        hardware["accelerator_name"] = torch.cuda.get_device_name(device)

    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run": {
            "name": config.name,
            "identity": f"{config.name}__{git_commit[:7]}__seed{config.seed}",
            "git_commit": git_commit,
        },
        "dataset": {
            "name": "PTB-XL",
            "version": config.dataset_version,
            "cohort": config.cohort_name,
            "signal_source": "filename_lr",
            "sampling_frequency_hz": 100,
            "targets": list(TARGET_SUPERCLASSES),
            "splits_used": {"train": len(train_index), "validation": metrics.samples},
        },
        "configuration": {
            "path": config_path.as_posix(),
            "sha256": config_sha256,
            "seed": config.seed,
            "model": {
                "feature_channels": list(config.feature_channels),
                "kernel_sizes": list(config.kernel_sizes),
                "dropout_probability": float(config.dropout_probability),
                "trainable_parameters": trainable_parameters,
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": float(config.learning_rate),
                "weight_decay": float(config.weight_decay),
                "loss": "BCEWithLogitsLoss",
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "checkpoint_selection": "first_minimum_validation_loss",
                "class_weighting": "none",
                "sampling": "standard_seeded_shuffle",
            },
        },
        "sources": {
            "cohort": {
                "path": config.cohort_path.as_posix(),
                "sha256": cohort_sha256,
            },
            "metadata": {
                "path": config.metadata_path.as_posix(),
                "sha256": metadata_sha256,
            },
            "standardizer": {
                "path": config.standardizer_path.as_posix(),
                "sha256": standardizer_sha256,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hardware": hardware,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "train_prevalence": train_prevalence,
        "training": {
            "best_epoch": fit_result.best_epoch,
            "best_validation_loss": fit_result.best_validation_loss,
            "history": [
                {
                    "epoch": item.epoch,
                    "train": asdict(item.train),
                    "validation": asdict(item.validation),
                }
                for item in fit_result.history
            ],
        },
        "validation": {
            "samples": metrics.samples,
            "macro_auroc": metrics.macro_auroc,
            "macro_auprc": metrics.macro_auprc,
            "micro_auroc": metrics.micro_auroc,
            "micro_auprc": metrics.micro_auprc,
            "per_class": [asdict(item) for item in metrics.per_class],
        },
        "artifacts": {
            "checkpoint": {
                "path": config.checkpoint_path.as_posix(),
                "sha256": checkpoint_sha256,
                "provenance": asdict(provenance),
            }
        },
        "limitations": [
            "No threshold-dependent metric or clinical operating point is defined.",
            "No final-test sample or metric was accessed.",
            "Exact numerics are not guaranteed across PyTorch releases or hardware.",
            "This internal validation result is not evidence of clinical utility.",
        ],
    }


def _validate_standardizer_sources(
    path: Path,
    *,
    cohort_sha256: str,
    metadata_sha256: str,
) -> None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        sources = artifact["sources"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Could not validate standardizer source provenance") from error
    expected = {
        "cohort_sha256": cohort_sha256,
        "metadata_sha256": metadata_sha256,
    }
    mismatched = [
        field for field, value in expected.items() if sources.get(field) != value
    ]
    if mismatched:
        raise ValueError(f"Standardizer source provenance mismatch: {mismatched}")


def _require_split_count(split: str, rows: pd.DataFrame, expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {split} records, got {len(rows)}")


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


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a TOML array")
    return tuple(value)


def _require_finite_number(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if positive and numeric <= 0:
        raise ValueError(f"{name} must be positive")
    if non_negative and numeric < 0:
        raise ValueError(f"{name} must be non-negative")
