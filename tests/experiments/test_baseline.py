import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.experiments import (
    ExperimentConfig,
    get_clean_git_commit,
    load_experiment_config,
    run_baseline_experiment,
)
from ptbxl.preprocessing import GlobalStandardizer, save_global_standardizer


LEAD_NAMES = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
TRAIN_TARGETS = (
    (1, 0, 1, 0, 1),
    (0, 1, 0, 1, 0),
    (1, 1, 0, 0, 0),
    (0, 0, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
)
VALIDATION_TARGETS = (
    (1, 0, 1, 0, 1),
    (0, 1, 0, 1, 0),
    (1, 1, 0, 0, 1),
    (0, 0, 1, 1, 0),
)


@pytest.fixture
def synthetic_experiment(tmp_path: Path) -> tuple[ExperimentConfig, Path]:
    rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    all_targets = (*TRAIN_TARGETS, *VALIDATION_TARGETS)
    for position, targets in enumerate(all_targets, start=1):
        split = "train" if position <= len(TRAIN_TARGETS) else "validation"
        fold = 1 if split == "train" else 9
        rows.append(
            {
                "ecg_id": position,
                "patient_id": 100 + position,
                "strat_fold": fold,
                "split": split,
                **dict(zip(TARGET_SUPERCLASSES, targets, strict=True)),
            }
        )
        relative = Path("records100") / "00000" / f"{position:05d}_lr"
        metadata_rows.append({"ecg_id": position, "filename_lr": relative.as_posix()})
        record_dir = tmp_path / relative.parent
        record_dir.mkdir(parents=True, exist_ok=True)
        signal = np.zeros((1_000, 12), dtype=np.float64)
        signal += np.linspace(-0.01, 0.01, 1_000)[:, None]
        for column, target in enumerate(targets):
            signal[:, column] += 1.0 if target else -1.0
        signal[:, 5:] += position / 100
        wfdb.wrsamp(
            record_name=relative.name,
            fs=100,
            units=["mV"] * 12,
            sig_name=list(LEAD_NAMES),
            p_signal=signal,
            write_dir=str(record_dir),
        )

    cohort_path = tmp_path / "cohort.csv"
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(cohort_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    standardizer_path = tmp_path / "standardizer.json"
    save_global_standardizer(
        GlobalStandardizer(
            mean=0.0,
            standard_deviation=1.0,
            record_count=len(TRAIN_TARGETS),
            value_count=len(TRAIN_TARGETS) * 12_000,
            lead_names=LEAD_NAMES,
        ),
        standardizer_path,
        {
            "cohort_sha256": compute_sha256(cohort_path),
            "metadata_sha256": compute_sha256(metadata_path),
            "signal_manifest_sha256": "c" * 64,
        },
    )
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        f'''schema_version = 1

[experiment]
name = "synthetic_baseline"
seed = 2026
device = "cpu"

[data]
dataset_version = "1.0.3"
cohort_name = "synthetic_five_superclass"
cohort_path = "{cohort_path.as_posix()}"
metadata_path = "{metadata_path.as_posix()}"
dataset_root = "{tmp_path.as_posix()}"
standardizer_path = "{standardizer_path.as_posix()}"
expected_train_records = {len(TRAIN_TARGETS)}
expected_validation_records = {len(VALIDATION_TARGETS)}

[model]
feature_channels = [4]
kernel_sizes = [3]
dropout_probability = 0.0

[training]
epochs = 1
batch_size = 2
num_workers = 0
learning_rate = 0.001
weight_decay = 0.0

[outputs]
checkpoint_path = "{(tmp_path / "checkpoint.pt").as_posix()}"
report_path = "{(tmp_path / "report.json").as_posix()}"
''',
        encoding="utf-8",
    )
    return load_experiment_config(config_path), config_path


def test_loads_strict_versioned_configuration(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> None:
    config, _ = synthetic_experiment

    assert config.name == "synthetic_baseline"
    assert config.dataset_version == "1.0.3"
    assert config.feature_channels == (4,)
    assert config.kernel_sizes == (3,)
    assert config.epochs == 1
    assert config.expected_train_records == 6
    assert config.expected_validation_records == 4


def test_rejects_unknown_configuration_field(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> None:
    _, config_path = synthetic_experiment
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nunknown = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        load_experiment_config(config_path)


def test_runs_synthetic_experiment_and_writes_attributed_outputs(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> None:
    config, config_path = synthetic_experiment
    epochs = []

    report = run_baseline_experiment(
        config,
        config_path,
        "a" * 40,
        on_epoch_end=epochs.append,
    )

    assert len(epochs) == 1
    assert config.checkpoint_path.is_file()
    assert config.report_path.is_file()
    assert json.loads(config.report_path.read_text(encoding="utf-8")) == report
    assert report["dataset"]["splits_used"] == {"train": 6, "validation": 4}
    assert "test" not in report["dataset"]["splits_used"]
    assert report["training"]["best_epoch"] == 1
    assert len(report["training"]["history"]) == 1
    assert report["training"]["history"][0]["train"]["samples"] == 6
    assert report["training"]["history"][0]["validation"]["samples"] == 4
    assert report["validation"]["samples"] == 4
    assert len(report["validation"]["per_class"]) == 5
    assert report["configuration"]["model"]["trainable_parameters"] > 0
    assert report["configuration"]["training"]["class_weighting"] == "none"
    assert report["artifacts"]["checkpoint"]["sha256"] == compute_sha256(
        config.checkpoint_path
    )
    checkpoint = torch.load(config.checkpoint_path, weights_only=True)
    assert checkpoint["provenance"]["git_commit"] == "a" * 40


def test_rejects_unexpected_real_split_count_before_loading_signals(
    synthetic_experiment: tuple[ExperimentConfig, Path],
) -> None:
    config, config_path = synthetic_experiment
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "expected_train_records = 6", "expected_train_records = 7"
        ),
        encoding="utf-8",
    )
    invalid_config = load_experiment_config(config_path)

    with pytest.raises(ValueError, match="Expected 7 train records, got 6"):
        run_baseline_experiment(
            invalid_config,
            config_path,
            "a" * 40,
        )
    assert not config.checkpoint_path.exists()


def test_resolves_only_clean_committed_git_state(tmp_path: Path) -> None:
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.com"),
        ("git", "config", "user.name", "Test User"),
    )
    for command in commands:
        subprocess.run(command, cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "Initial"), cwd=tmp_path, check=True)

    commit = get_clean_git_commit(tmp_path)

    assert len(commit) == 40
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        get_clean_git_commit(tmp_path)
