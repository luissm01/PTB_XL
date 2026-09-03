from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.experiments import ExperimentConfig, load_experiment_config
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
TEST_TARGETS = (
    (1, 0, 1, 0, 1),
    (0, 1, 0, 1, 0),
    (1, 1, 0, 0, 1),
    (0, 0, 1, 1, 0),
)


@pytest.fixture
def synthetic_experiment(tmp_path: Path) -> tuple[ExperimentConfig, Path]:
    rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    all_targets = (*TRAIN_TARGETS, *VALIDATION_TARGETS, *TEST_TARGETS)
    for position, targets in enumerate(all_targets, start=1):
        if position <= len(TRAIN_TARGETS):
            split, fold = "train", 1
        elif position <= len(TRAIN_TARGETS) + len(VALIDATION_TARGETS):
            split, fold = "validation", 9
        else:
            split, fold = "test", 10
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
