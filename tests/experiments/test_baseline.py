import json
import subprocess
from pathlib import Path

import pytest
import torch

from ptbxl.data.reporting import compute_sha256
from ptbxl.experiments import (
    ExperimentConfig,
    get_clean_git_commit,
    load_experiment_config,
    run_baseline_experiment,
)


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
