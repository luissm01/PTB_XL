import json
from pathlib import Path

import pytest

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.evaluation import fingerprint_predictions, load_prediction_artifact
from ptbxl.experiments import (
    ExperimentConfig,
    FinalTestConfig,
    ThresholdExperimentConfig,
    load_final_test_config,
    load_threshold_experiment_config,
    run_baseline_experiment,
    run_final_test_evaluation,
    run_validation_threshold_selection,
)


@pytest.fixture
def synthetic_final_test(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> tuple[FinalTestConfig, Path]:
    baseline_config, baseline_config_path = synthetic_experiment
    run_baseline_experiment(baseline_config, baseline_config_path, "a" * 40)

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
    threshold_config: ThresholdExperimentConfig = load_threshold_experiment_config(
        threshold_config_path
    )
    run_validation_threshold_selection(
        threshold_config,
        threshold_config_path,
        "b" * 40,
    )

    final_config_path = baseline_config_path.parent / "final_test.toml"
    final_config_path.write_text(
        f'''schema_version = 1

[evaluation]
name = "synthetic_final_test"
mode = "one_time_final_test"
device = "cpu"
batch_size = 2
num_workers = 0
expected_test_records = 4

[inputs]
experiment_config_path = "{baseline_config_path.as_posix()}"
experiment_config_sha256 = "{compute_sha256(baseline_config_path)}"
experiment_report_path = "{baseline_config.report_path.as_posix()}"
experiment_report_sha256 = "{compute_sha256(baseline_config.report_path)}"
checkpoint_path = "{baseline_config.checkpoint_path.as_posix()}"
checkpoint_sha256 = "{compute_sha256(baseline_config.checkpoint_path)}"
standardizer_sha256 = "{compute_sha256(baseline_config.standardizer_path)}"
threshold_artifact_path = "{threshold_config.artifact_path.as_posix()}"
threshold_artifact_sha256 = "{compute_sha256(threshold_config.artifact_path)}"

[outputs]
report_path = "{(baseline_config_path.parent / "final_report.json").as_posix()}"
prediction_artifact_path = "{(baseline_config_path.parent / "final_predictions.npz").as_posix()}"
''',
        encoding="utf-8",
    )
    return load_final_test_config(final_config_path), final_config_path


def test_loads_strict_final_test_configuration(
    synthetic_final_test: tuple[FinalTestConfig, Path],
) -> None:
    config, _ = synthetic_final_test

    assert config.mode == "one_time_final_test"
    assert config.expected_test_records == 4
    assert config.report_path != config.prediction_artifact_path


def test_rejects_unknown_field_and_output_collision(
    synthetic_final_test: tuple[FinalTestConfig, Path],
) -> None:
    config, config_path = synthetic_final_test
    original = config_path.read_text(encoding="utf-8")
    config_path.write_text(original + "\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        load_final_test_config(config_path)

    config_path.write_text(
        original.replace(
            config.report_path.as_posix(), config.experiment_report_path.as_posix()
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distinct from every input"):
        load_final_test_config(config_path)


def test_executes_synthetic_final_test_once_with_frozen_inputs(
    synthetic_final_test: tuple[FinalTestConfig, Path],
) -> None:
    config, config_path = synthetic_final_test
    immutable_hashes = {
        path: compute_sha256(path)
        for path in (
            config.experiment_config_path,
            config.experiment_report_path,
            config.checkpoint_path,
            config.threshold_artifact_path,
        )
    }

    report = run_final_test_evaluation(config, config_path, "c" * 40)

    assert json.loads(config.report_path.read_text(encoding="utf-8")) == report
    assert report["dataset"]["split"] == "test"
    assert report["dataset"]["fold"] == 10
    assert report["dataset"]["samples"] == 4
    assert "train" not in report["dataset"]
    assert "validation" not in report["dataset"]
    assert len(report["ranking"]["per_class"]) == len(TARGET_SUPERCLASSES)
    assert len(report["operating_point"]["per_class"]) == len(TARGET_SUPERCLASSES)
    assert report["operating_point"]["method"] == "per_class_max_f1"
    assert (
        report["operating_point"]["decision_rule"]
        == "probability_greater_than_or_equal_to_threshold"
    )
    loaded = load_prediction_artifact(config.prediction_artifact_path)
    assert loaded.split == "test"
    assert loaded.predictions.targets.shape == (4, 5)
    assert report["outputs"]["prediction_artifact"]["fingerprint"] == (
        fingerprint_predictions(loaded.predictions, split="test")
    )
    assert report["outputs"]["prediction_artifact"]["sha256"] == compute_sha256(
        config.prediction_artifact_path
    )
    assert immutable_hashes == {path: compute_sha256(path) for path in immutable_hashes}

    with pytest.raises(FileExistsError, match="one-time"):
        run_final_test_evaluation(config, config_path, "c" * 40)


def test_existing_output_fails_before_loading_frozen_inputs(
    synthetic_final_test: tuple[FinalTestConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path = synthetic_final_test
    config.report_path.write_text("already observed", encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("frozen inputs must not load")

    monkeypatch.setattr(
        "ptbxl.experiments.final_test.load_frozen_baseline", fail_if_called
    )
    with pytest.raises(FileExistsError, match="one-time"):
        run_final_test_evaluation(config, config_path, "c" * 40)


def test_threshold_binding_mismatch_fails_before_signal_metadata(
    synthetic_final_test: tuple[FinalTestConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path = synthetic_final_test
    artifact = json.loads(config.threshold_artifact_path.read_text(encoding="utf-8"))
    artifact["dataset"]["cohort"] = "wrong_cohort"
    config.threshold_artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("metadata must not load")

    monkeypatch.setattr("ptbxl.experiments.final_test.pd.read_csv", fail_if_called)
    with pytest.raises(ValueError, match="Threshold artifact SHA-256"):
        run_final_test_evaluation(config, config_path, "c" * 40)
    assert not config.report_path.exists()
    assert not config.prediction_artifact_path.exists()
