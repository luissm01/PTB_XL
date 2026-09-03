import json
from pathlib import Path

import pytest

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.evaluation import load_frozen_thresholds
from ptbxl.experiments import (
    ExperimentConfig,
    ThresholdExperimentConfig,
    load_threshold_experiment_config,
    run_baseline_experiment,
    run_validation_threshold_selection,
)


@pytest.fixture
def synthetic_threshold_experiment(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> tuple[ThresholdExperimentConfig, Path]:
    baseline_config, baseline_config_path = synthetic_experiment
    run_baseline_experiment(
        baseline_config,
        baseline_config_path,
        "a" * 40,
    )
    threshold_config_path = baseline_config_path.parent / "thresholds.toml"
    threshold_config_path.write_text(
        f'''schema_version = 1

[selection]
name = "synthetic_per_class_f1"
method = "per_class_max_f1"
device = "cpu"
batch_size = 2
num_workers = 0

[inputs]
experiment_config_path = "{baseline_config_path.as_posix()}"
experiment_report_path = "{baseline_config.report_path.as_posix()}"
checkpoint_path = "{baseline_config.checkpoint_path.as_posix()}"

[output]
artifact_path = "{(baseline_config_path.parent / "thresholds.json").as_posix()}"
''',
        encoding="utf-8",
    )
    return load_threshold_experiment_config(
        threshold_config_path
    ), threshold_config_path


def test_loads_strict_threshold_experiment_configuration(
    synthetic_threshold_experiment: tuple[ThresholdExperimentConfig, Path],
) -> None:
    config, _ = synthetic_threshold_experiment

    assert config.name == "synthetic_per_class_f1"
    assert config.method == "per_class_max_f1"
    assert config.device == "cpu"
    assert config.batch_size == 2
    assert config.num_workers == 0


def test_rejects_unknown_threshold_configuration_field(
    synthetic_threshold_experiment: tuple[ThresholdExperimentConfig, Path],
) -> None:
    _, config_path = synthetic_threshold_experiment
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nunknown = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        load_threshold_experiment_config(config_path)


def test_selects_synthetic_thresholds_without_modifying_checkpoint(
    synthetic_threshold_experiment: tuple[ThresholdExperimentConfig, Path],
) -> None:
    config, config_path = synthetic_threshold_experiment
    checkpoint_sha256 = compute_sha256(config.checkpoint_path)
    report_sha256 = compute_sha256(config.experiment_report_path)

    artifact = run_validation_threshold_selection(config, config_path, "b" * 40)

    assert json.loads(config.artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["dataset"]["split"] == "validation"
    assert artifact["dataset"]["samples"] == 4
    assert "test" not in artifact["dataset"]
    assert list(artifact["selection"]["thresholds"]) == list(TARGET_SUPERCLASSES)
    assert len(artifact["validation"]["per_class"]) == 5
    assert set(artifact["validation"]["macro"]) == {
        "precision",
        "sensitivity",
        "specificity",
        "f1",
    }
    assert compute_sha256(config.checkpoint_path) == checkpoint_sha256
    assert compute_sha256(config.experiment_report_path) == report_sha256
    assert artifact["sources"]["selection_config"]["sha256"] == compute_sha256(
        config_path
    )
    assert artifact["sources"]["checkpoint"]["sha256"] == checkpoint_sha256
    frozen = load_frozen_thresholds(
        config.artifact_path,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    assert frozen.validation_samples == 4


def test_rejects_checkpoint_not_bound_to_baseline_report(
    synthetic_threshold_experiment: tuple[ThresholdExperimentConfig, Path],
) -> None:
    config, config_path = synthetic_threshold_experiment
    config.checkpoint_path.write_bytes(b"tampered checkpoint")

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        run_validation_threshold_selection(config, config_path, "b" * 40)
    assert not config.artifact_path.exists()


def test_rejects_baseline_report_with_unexpected_split(
    synthetic_threshold_experiment: tuple[ThresholdExperimentConfig, Path],
) -> None:
    config, config_path = synthetic_threshold_experiment
    report = json.loads(config.experiment_report_path.read_text(encoding="utf-8"))
    report["dataset"]["splits_used"]["test"] = 1
    config.experiment_report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="split counts"):
        run_validation_threshold_selection(config, config_path, "b" * 40)
    assert not config.artifact_path.exists()
