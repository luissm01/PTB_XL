"""One-time evaluation of the fully frozen baseline on official fold 10."""

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
    FrozenThresholds,
    MultilabelOperatingMetrics,
    MultilabelRankingMetrics,
    compute_operating_metrics,
    evaluate_final_test,
    fingerprint_predictions,
    load_frozen_thresholds,
    load_prediction_artifact,
    save_prediction_artifact,
)
from ptbxl.experiments.frozen import (
    FrozenBaseline,
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


FINAL_TEST_CONFIG_SCHEMA_VERSION = 1
FINAL_TEST_REPORT_SCHEMA_VERSION = 1
FINAL_TEST_MODE = "one_time_final_test"
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
CONFIG_TABLE_FIELDS = {
    "evaluation": {
        "name",
        "mode",
        "device",
        "batch_size",
        "num_workers",
        "expected_test_records",
    },
    "inputs": {
        "experiment_config_path",
        "experiment_config_sha256",
        "experiment_report_path",
        "experiment_report_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "standardizer_sha256",
        "threshold_artifact_path",
        "threshold_artifact_sha256",
    },
    "outputs": {"report_path", "prediction_artifact_path"},
}


@dataclass(frozen=True)
class FinalTestConfig:
    """Immutable declaration of the one-time final-test event."""

    name: str
    mode: str
    device: str
    batch_size: int
    num_workers: int
    expected_test_records: int
    experiment_config_path: Path
    experiment_config_sha256: str
    experiment_report_path: Path
    experiment_report_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    standardizer_sha256: str
    threshold_artifact_path: Path
    threshold_artifact_sha256: str
    report_path: Path
    prediction_artifact_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", self.name
        ):
            raise ValueError(
                "Final-test name must use lowercase letters, numbers, _ or -"
            )
        if self.mode != FINAL_TEST_MODE:
            raise ValueError(f"Final-test mode must be {FINAL_TEST_MODE!r}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Final-test device must be 'auto', 'cpu', or 'cuda'")
        for field in ("batch_size", "expected_test_records"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        for field in (
            "experiment_config_path",
            "experiment_report_path",
            "checkpoint_path",
            "threshold_artifact_path",
            "report_path",
            "prediction_artifact_path",
        ):
            if not isinstance(getattr(self, field), Path):
                raise TypeError(f"{field} must be a pathlib.Path")
        for field in (
            "experiment_config_sha256",
            "experiment_report_sha256",
            "checkpoint_sha256",
            "standardizer_sha256",
            "threshold_artifact_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        expected_suffixes = {
            "experiment_config_path": ".toml",
            "experiment_report_path": ".json",
            "checkpoint_path": ".pt",
            "threshold_artifact_path": ".json",
            "report_path": ".json",
            "prediction_artifact_path": ".npz",
        }
        for field, suffix in expected_suffixes.items():
            if getattr(self, field).suffix != suffix:
                raise ValueError(f"{field} must end in {suffix}")
        inputs = {
            self.experiment_config_path,
            self.experiment_report_path,
            self.checkpoint_path,
            self.threshold_artifact_path,
        }
        outputs = {self.report_path, self.prediction_artifact_path}
        if len(outputs) != 2 or inputs & outputs:
            raise ValueError("Final-test outputs must be distinct from every input")


def load_final_test_config(path: str | Path) -> FinalTestConfig:
    """Load one strict versioned final-test configuration."""
    config_path = Path(path)
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    expected_top_level = {"schema_version", *CONFIG_TABLE_FIELDS}
    _require_exact_fields("configuration", raw, expected_top_level)
    if raw["schema_version"] != FINAL_TEST_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Final-test config schema_version must be {FINAL_TEST_CONFIG_SCHEMA_VERSION}"
        )
    tables = {
        name: _require_table(raw, name, fields)
        for name, fields in CONFIG_TABLE_FIELDS.items()
    }
    evaluation = tables["evaluation"]
    inputs = tables["inputs"]
    outputs = tables["outputs"]
    return FinalTestConfig(
        name=evaluation["name"],
        mode=evaluation["mode"],
        device=evaluation["device"],
        batch_size=evaluation["batch_size"],
        num_workers=evaluation["num_workers"],
        expected_test_records=evaluation["expected_test_records"],
        experiment_config_path=_config_path(
            inputs["experiment_config_path"], "experiment_config_path"
        ),
        experiment_config_sha256=inputs["experiment_config_sha256"],
        experiment_report_path=_config_path(
            inputs["experiment_report_path"], "experiment_report_path"
        ),
        experiment_report_sha256=inputs["experiment_report_sha256"],
        checkpoint_path=_config_path(inputs["checkpoint_path"], "checkpoint_path"),
        checkpoint_sha256=inputs["checkpoint_sha256"],
        standardizer_sha256=inputs["standardizer_sha256"],
        threshold_artifact_path=_config_path(
            inputs["threshold_artifact_path"], "threshold_artifact_path"
        ),
        threshold_artifact_sha256=inputs["threshold_artifact_sha256"],
        report_path=_config_path(outputs["report_path"], "report_path"),
        prediction_artifact_path=_config_path(
            outputs["prediction_artifact_path"], "prediction_artifact_path"
        ),
    )


def run_final_test_evaluation(
    config: FinalTestConfig,
    config_path: str | Path,
    git_commit: str,
) -> dict[str, Any]:
    """Evaluate fold 10 once with frozen inputs and persist immutable outputs."""
    if not isinstance(config, FinalTestConfig):
        raise TypeError("config must be a FinalTestConfig")
    if not isinstance(git_commit, str) or not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise ValueError("git_commit must be a lowercase hexadecimal commit")
    config_path = Path(config_path)
    if load_final_test_config(config_path) != config:
        raise ValueError("config does not match the attributed config_path")
    _require_outputs_absent(config)
    frozen = load_frozen_baseline(
        config.experiment_config_path,
        config.experiment_report_path,
        config.checkpoint_path,
    )
    _validate_declared_baseline_hashes(config, frozen)
    threshold_artifact_sha256 = compute_sha256(config.threshold_artifact_path)
    if threshold_artifact_sha256 != config.threshold_artifact_sha256:
        raise ValueError("Threshold artifact SHA-256 does not match final-test config")
    frozen_thresholds = load_frozen_thresholds(
        config.threshold_artifact_path,
        expected_checkpoint_sha256=frozen.hashes.checkpoint,
    )
    _validate_threshold_binding(frozen, frozen_thresholds)

    baseline_config = frozen.config
    cohort = pd.read_csv(baseline_config.cohort_path)
    metadata = pd.read_csv(
        baseline_config.metadata_path,
        usecols=["ecg_id", "filename_lr"],
    )
    sample_index = build_sample_index(cohort, metadata)
    test_index = sample_index.loc[sample_index["split"] == "test"].reset_index(
        drop=True
    )
    if len(test_index) != config.expected_test_records:
        raise ValueError(
            f"Expected {config.expected_test_records} test records, "
            f"got {len(test_index)}"
        )

    configure_deterministic_execution()
    seed_random_generators(baseline_config.seed)
    device = resolve_device(config.device)
    standardizer = load_global_standardizer(baseline_config.standardizer_path)
    test_dataset = PTBXLDataset(
        test_index,
        baseline_config.dataset_root,
        standardizer,
        split="test",
    )
    test_loader = build_dataloader(
        test_dataset,
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
    evaluation = evaluate_final_test(model, test_loader, device)
    operating_metrics = compute_operating_metrics(
        evaluation.predictions,
        frozen_thresholds.thresholds,
    )
    prediction_fingerprint = fingerprint_predictions(
        evaluation.predictions,
        split="test",
    )
    save_prediction_artifact(
        config.prediction_artifact_path,
        evaluation.predictions,
        split="test",
    )
    saved_predictions = load_prediction_artifact(config.prediction_artifact_path)
    saved_fingerprint = fingerprint_predictions(
        saved_predictions.predictions,
        split=saved_predictions.split,
    )
    if saved_predictions.split != "test" or saved_fingerprint != prediction_fingerprint:
        raise ValueError("Saved final-test predictions failed round-trip validation")
    prediction_artifact_sha256 = compute_sha256(config.prediction_artifact_path)
    report = _build_final_test_report(
        config,
        config_path=config_path,
        git_commit=git_commit,
        frozen=frozen,
        frozen_thresholds=frozen_thresholds,
        threshold_artifact_sha256=threshold_artifact_sha256,
        prediction_artifact_sha256=prediction_artifact_sha256,
        prediction_fingerprint=prediction_fingerprint,
        ranking_metrics=evaluation.metrics,
        operating_metrics=operating_metrics,
        device=device,
    )
    write_json_report(report, config.report_path)
    return report


def _build_final_test_report(
    config: FinalTestConfig,
    *,
    config_path: Path,
    git_commit: str,
    frozen: FrozenBaseline,
    frozen_thresholds: FrozenThresholds,
    threshold_artifact_sha256: str,
    prediction_artifact_sha256: str,
    prediction_fingerprint: str,
    ranking_metrics: MultilabelRankingMetrics,
    operating_metrics: MultilabelOperatingMetrics,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_TEST_REPORT_SCHEMA_VERSION,
        "event": {
            "name": config.name,
            "mode": config.mode,
            "git_commit": git_commit,
            "repeat_evaluation_forbidden": True,
            "test_driven_model_changes_forbidden": True,
        },
        "dataset": {
            "name": "PTB-XL",
            "version": frozen.config.dataset_version,
            "cohort": frozen.config.cohort_name,
            "split": "test",
            "fold": 10,
            "samples": ranking_metrics.samples,
            "targets": list(TARGET_SUPERCLASSES),
        },
        "sources": {
            "final_test_config": {
                "path": config_path.as_posix(),
                "sha256": compute_sha256(config_path),
            },
            "experiment_config": {
                "path": config.experiment_config_path.as_posix(),
                "sha256": frozen.hashes.experiment_config,
            },
            "experiment_report": {
                "path": config.experiment_report_path.as_posix(),
                "sha256": frozen.hashes.experiment_report,
            },
            "checkpoint": {
                "path": config.checkpoint_path.as_posix(),
                "sha256": frozen.hashes.checkpoint,
            },
            "standardizer": {
                "path": frozen.config.standardizer_path.as_posix(),
                "sha256": frozen.hashes.standardizer,
            },
            "threshold_artifact": {
                "path": config.threshold_artifact_path.as_posix(),
                "sha256": threshold_artifact_sha256,
            },
        },
        "outputs": {
            "prediction_artifact": {
                "path": config.prediction_artifact_path.as_posix(),
                "sha256": prediction_artifact_sha256,
                "fingerprint": prediction_fingerprint,
                "format": "npz_numeric_arrays_no_pickle",
                "version_control": "ignored",
            }
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
        "ranking": {
            "samples": ranking_metrics.samples,
            "per_class": [asdict(item) for item in ranking_metrics.per_class],
            "macro_auroc": ranking_metrics.macro_auroc,
            "macro_auprc": ranking_metrics.macro_auprc,
            "micro_auroc": ranking_metrics.micro_auroc,
            "micro_auprc": ranking_metrics.micro_auprc,
        },
        "operating_point": {
            "method": frozen_thresholds.thresholds.method,
            "decision_rule": frozen_thresholds.thresholds.decision_rule,
            "tie_break": frozen_thresholds.thresholds.tie_break,
            "thresholds": {
                item.label: item.threshold for item in operating_metrics.per_class
            },
            "samples": operating_metrics.samples,
            "per_class": [asdict(item) for item in operating_metrics.per_class],
            "macro": asdict(operating_metrics.macro),
            "micro": asdict(operating_metrics.micro),
        },
        "limitations": [
            "This is one internal PTB-XL test fold, not external validation.",
            "No uncertainty interval or repeated-seed estimate is available.",
            "The frozen operating point has no clinically defined error costs.",
            "This final result is not evidence of clinical utility.",
        ],
    }


def _validate_threshold_binding(
    frozen: FrozenBaseline,
    thresholds: FrozenThresholds,
) -> None:
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


def _validate_declared_baseline_hashes(
    config: FinalTestConfig,
    frozen: FrozenBaseline,
) -> None:
    expected = {
        "experiment config": (
            frozen.hashes.experiment_config,
            config.experiment_config_sha256,
        ),
        "experiment report": (
            frozen.hashes.experiment_report,
            config.experiment_report_sha256,
        ),
        "checkpoint": (frozen.hashes.checkpoint, config.checkpoint_sha256),
        "standardizer": (frozen.hashes.standardizer, config.standardizer_sha256),
    }
    mismatched = [name for name, values in expected.items() if values[0] != values[1]]
    if mismatched:
        raise ValueError(f"Frozen baseline SHA-256 mismatch: {mismatched}")


def _require_outputs_absent(config: FinalTestConfig) -> None:
    existing = [
        path.as_posix()
        for path in (config.report_path, config.prediction_artifact_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Final-test evaluation is one-time and will not overwrite outputs: "
            f"{existing}"
        )


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
