from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

import ptbxl.data.samples as samples_module
from ptbxl.data.labels import TARGET_SUPERCLASSES
from ptbxl.data.samples import build_sample_index, load_sample


LEAD_NAMES = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [2, 1, 3],
            "patient_id": [20, 10, 30],
            "strat_fold": [9, 1, 10],
            "split": ["validation", "train", "test"],
            "NORM": [0, 1, 0],
            "MI": [1, 0, 0],
            "STTC": [1, 0, 0],
            "CD": [0, 0, 1],
            "HYP": [0, 0, 0],
        }
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "filename_lr": [
                "records100/00000/00001_lr",
                "records100/00000/00002_lr",
                "records100/00000/00003_lr",
            ],
        }
    )


def _write_record(root: Path, ecg_id: int) -> str:
    relative = Path("records100") / "00000" / f"{ecg_id:05d}_lr"
    record_dir = root / relative.parent
    record_dir.mkdir(parents=True, exist_ok=True)
    base = np.linspace(-1.0, 1.0, 1_000)
    values = np.column_stack([base + index / 100 for index in range(12)])
    wfdb.wrsamp(
        record_name=relative.name,
        fs=100,
        units=["mV"] * 12,
        sig_name=list(LEAD_NAMES),
        p_signal=values,
        write_dir=str(record_dir),
    )
    return relative.as_posix()


def test_builds_index_in_cohort_order_without_changing_source_values() -> None:
    cohort = _cohort()
    original = cohort.copy(deep=True)

    index = build_sample_index(cohort, _metadata())

    assert index["ecg_id"].tolist() == [2, 1, 3]
    assert index["filename_lr"].tolist() == [
        "records100/00000/00002_lr",
        "records100/00000/00001_lr",
        "records100/00000/00003_lr",
    ]
    pd.testing.assert_frame_equal(
        index.drop(columns="filename_lr"),
        original,
    )
    pd.testing.assert_frame_equal(cohort, original)


def test_loads_exact_signal_targets_and_inherited_split(tmp_path: Path) -> None:
    filename_lr = _write_record(tmp_path, 1)
    metadata = _metadata()
    metadata.loc[metadata["ecg_id"] == 1, "filename_lr"] = filename_lr
    index = build_sample_index(_cohort(), metadata)
    row = index.loc[index["ecg_id"] == 1].iloc[0].to_dict()

    sample = load_sample(row, tmp_path)

    assert sample.ecg_id == 1
    assert sample.patient_id == 10
    assert sample.strat_fold == 1
    assert sample.split == "train"
    assert sample.filename_lr == filename_lr
    assert sample.signal.shape == (1_000, 12)
    assert sample.targets.shape == (len(TARGET_SUPERCLASSES),)
    assert sample.targets.dtype == np.float32
    np.testing.assert_array_equal(sample.targets, [1, 0, 0, 0, 0])
    assert sample.target_names == TARGET_SUPERCLASSES
    assert sample.sampling_frequency == 100.0
    assert sample.lead_names == LEAD_NAMES


def test_building_index_does_not_load_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("signal loading must remain lazy")

    monkeypatch.setattr(samples_module, "load_signal_for_row", fail_if_called)

    assert len(build_sample_index(_cohort(), _metadata())) == 3


def test_rejects_non_binary_target_before_association() -> None:
    cohort = _cohort()
    cohort.loc[0, "MI"] = 2

    with pytest.raises(ValueError, match="only 0 or 1"):
        build_sample_index(cohort, _metadata())


def test_rejects_row_outside_five_superclass_cohort() -> None:
    cohort = _cohort()
    cohort.loc[0, list(TARGET_SUPERCLASSES)] = 0

    with pytest.raises(ValueError, match="at least one target"):
        build_sample_index(cohort, _metadata())


def test_rejects_fold_split_mismatch_before_association() -> None:
    cohort = _cohort()
    cohort.loc[0, "split"] = "train"

    with pytest.raises(ValueError, match="official strat_fold"):
        build_sample_index(cohort, _metadata())


def test_rejects_patient_leakage_before_association() -> None:
    cohort = _cohort()
    cohort.loc[0, "patient_id"] = cohort.loc[1, "patient_id"]

    with pytest.raises(ValueError, match="Patient leakage"):
        build_sample_index(cohort, _metadata())


def test_rejects_missing_signal_association() -> None:
    with pytest.raises(ValueError, match="missing from metadata"):
        build_sample_index(_cohort(), _metadata().iloc[:2])


def test_rejects_ambiguous_signal_association() -> None:
    metadata = _metadata()
    metadata.loc[1, "filename_lr"] = metadata.loc[0, "filename_lr"]

    with pytest.raises(ValueError, match="duplicate filename_lr"):
        build_sample_index(_cohort(), metadata)


def test_load_revalidates_row_split_before_reading_signal(tmp_path: Path) -> None:
    row = build_sample_index(_cohort(), _metadata()).iloc[0].to_dict()
    row["split"] = "train"

    with pytest.raises(ValueError, match="does not match strat_fold"):
        load_sample(row, tmp_path)


def test_load_rejects_missing_target_field(tmp_path: Path) -> None:
    row = build_sample_index(_cohort(), _metadata()).iloc[0].to_dict()
    del row["HYP"]

    with pytest.raises(KeyError, match="Missing sample fields"):
        load_sample(row, tmp_path)
