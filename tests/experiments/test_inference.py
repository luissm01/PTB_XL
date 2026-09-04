import json
from pathlib import Path

import numpy as np
import pytest

from ptbxl.data import TARGET_SUPERCLASSES, ECGSignal, load_wfdb_record
from ptbxl.data.reporting import compute_sha256
from ptbxl.experiments import (
    ExperimentConfig,
    load_experiment_config,
    load_threshold_experiment_config,
    run_baseline_experiment,
    run_validation_threshold_selection,
)
from ptbxl.inference import (
    InferenceConfig,
    fingerprint_ecg_signal,
    load_frozen_inference_bundle,
    load_inference_config,
    load_inference_report,
    predict_ecg_record,
    run_single_record_inference,
)


@pytest.fixture
def synthetic_inference(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> tuple[InferenceConfig, Path, Path]:
    baseline_config, baseline_config_path = synthetic_experiment
    run_baseline_experiment(baseline_config, baseline_config_path, "a" * 40)

    threshold_config_path = baseline_config_path.parent / "thresholds.toml"
    threshold_artifact_path = baseline_config_path.parent / "thresholds.json"
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
artifact_path = "{threshold_artifact_path.as_posix()}"
''',
        encoding="utf-8",
    )
    threshold_config = load_threshold_experiment_config(threshold_config_path)
    run_validation_threshold_selection(
        threshold_config,
        threshold_config_path,
        "b" * 40,
    )

    inference_config_path = baseline_config_path.parent / "inference.toml"
    inference_config_path.write_text(
        f'''schema_version = 1

[inference]
name = "synthetic_baseline"
mode = "single_record"
device = "cpu"

[inputs]
experiment_config_path = "{baseline_config_path.as_posix()}"
experiment_config_sha256 = "{compute_sha256(baseline_config_path)}"
experiment_report_path = "{baseline_config.report_path.as_posix()}"
experiment_report_sha256 = "{compute_sha256(baseline_config.report_path)}"
checkpoint_path = "{baseline_config.checkpoint_path.as_posix()}"
checkpoint_sha256 = "{compute_sha256(baseline_config.checkpoint_path)}"
standardizer_sha256 = "{compute_sha256(baseline_config.standardizer_path)}"
threshold_artifact_path = "{threshold_artifact_path.as_posix()}"
threshold_artifact_sha256 = "{compute_sha256(threshold_artifact_path)}"
''',
        encoding="utf-8",
    )
    record_path = baseline_config.dataset_root / "records100/00000/00001_lr"
    return (
        load_inference_config(inference_config_path),
        inference_config_path,
        record_path,
    )


def test_loads_strict_inference_config(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
) -> None:
    config, config_path, _ = synthetic_inference

    assert config.name == "synthetic_baseline"
    assert config.mode == "single_record"
    assert config.device == "cpu"
    assert config.experiment_config_sha256 == compute_sha256(
        config.experiment_config_path
    )

    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nunknown = true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected fields"):
        load_inference_config(config_path)


def test_predicts_one_record_with_frozen_order_and_decision_rule(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
) -> None:
    config, _, record_path = synthetic_inference
    bundle = load_frozen_inference_bundle(config)

    result = predict_ecg_record(bundle, record_path, record_id="train-example")

    assert result.record_id == "train-example"
    assert result.record_path == record_path
    assert result.samples == 1_000
    assert result.sampling_frequency_hz == 100.0
    assert result.signal_sha256 == fingerprint_ecg_signal(load_wfdb_record(record_path))
    assert tuple(item.label for item in result.predictions) == TARGET_SUPERCLASSES
    assert all(0.0 <= item.probability <= 1.0 for item in result.predictions)
    assert all(
        item.predicted == (item.probability >= item.threshold)
        for item in result.predictions
    )
    assert all(parameter.grad is None for parameter in bundle.model.parameters())
    assert not bundle.model.training


def test_bundle_loading_does_not_require_training_tables(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
) -> None:
    config, _, _ = synthetic_inference
    baseline_config = load_experiment_config(config.experiment_config_path)
    baseline_config.cohort_path.unlink()
    baseline_config.metadata_path.unlink()

    bundle = load_frozen_inference_bundle(config)

    assert bundle.baseline.hashes.cohort is None
    assert bundle.baseline.hashes.metadata is None
    assert bundle.baseline.hashes.checkpoint == config.checkpoint_sha256


def test_run_writes_deterministic_strict_report_and_refuses_overwrite(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
) -> None:
    config, config_path, record_path = synthetic_inference
    first_path = config_path.parent / "prediction_1.json"
    second_path = config_path.parent / "prediction_2.json"

    first = run_single_record_inference(
        config,
        config_path,
        record_path,
        first_path,
        "c" * 40,
        record_id="synthetic-train-1",
    )
    second = run_single_record_inference(
        config,
        config_path,
        record_path,
        second_path,
        "c" * 40,
        record_id="synthetic-train-1",
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert (
        load_inference_report(
            first_path,
            expected_checkpoint_sha256=config.checkpoint_sha256,
        )
        == first
    )
    assert "targets" not in first["input"]
    assert "split" not in first["input"]
    assert list(item["label"] for item in first["predictions"]) == list(
        TARGET_SUPERCLASSES
    )
    with pytest.raises(FileExistsError, match="already exists"):
        run_single_record_inference(
            config,
            config_path,
            record_path,
            first_path,
            "c" * 40,
        )


def test_artifact_mismatch_fails_before_loading_signal(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, record_path = synthetic_inference
    artifact = json.loads(config.threshold_artifact_path.read_text(encoding="utf-8"))
    artifact["selection"]["thresholds"]["NORM"] = 0.123
    config.threshold_artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("signal must not load")

    monkeypatch.setattr("ptbxl.inference.load_wfdb_record", fail_if_called)
    with pytest.raises(ValueError, match="Threshold artifact SHA-256"):
        run_single_record_inference(
            config,
            config_path,
            record_path,
            config_path.parent / "prediction.json",
            "c" * 40,
        )


def test_existing_output_fails_before_loading_bundle(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, record_path = synthetic_inference
    output_path = config_path.parent / "prediction.json"
    output_path.write_text("existing", encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("bundle must not load")

    monkeypatch.setattr("ptbxl.inference.load_frozen_inference_bundle", fail_if_called)
    with pytest.raises(FileExistsError, match="already exists"):
        run_single_record_inference(
            config,
            config_path,
            record_path,
            output_path,
            "c" * 40,
        )


def test_report_loader_rejects_tampered_decision_and_checkpoint(
    synthetic_inference: tuple[InferenceConfig, Path, Path],
) -> None:
    config, config_path, record_path = synthetic_inference
    output_path = config_path.parent / "prediction.json"
    report = run_single_record_inference(
        config,
        config_path,
        record_path,
        output_path,
        "c" * 40,
    )
    report["predictions"][0]["predicted"] = not report["predictions"][0]["predicted"]
    output_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_inference_report(output_path)

    report["predictions"][0]["predicted"] = not report["predictions"][0]["predicted"]
    output_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_inference_report(
            output_path,
            expected_checkpoint_sha256="d" * 64,
        )


def test_signal_fingerprint_is_value_and_header_sensitive() -> None:
    signal = np.zeros((1_000, 12), dtype=np.float64)
    leads = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
    original = ECGSignal(signal, 100.0, leads)
    changed_signal = signal.copy()
    changed_signal[0, 0] = 1.0
    changed = ECGSignal(changed_signal, 100.0, leads)

    assert fingerprint_ecg_signal(original) == fingerprint_ecg_signal(original)
    assert fingerprint_ecg_signal(original) != fingerprint_ecg_signal(changed)
