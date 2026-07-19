from pathlib import Path

import pandas as pd
import pytest

from ptbxl.data.metadata import (
    build_metadata_summary,
    find_patient_fold_conflicts,
    find_patient_overlaps,
    load_metadata,
    prepare_metadata,
)


REQUIRED_COLUMNS = ("ecg_id", "patient_id", "strat_fold")


def make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "patient_id": [101, 102, 103, 104],
            "strat_fold": [1, 8, 9, 10],
        }
    )


def test_load_metadata_reads_csv(tmp_path: Path) -> None:
    expected = make_metadata()
    csv_path = tmp_path / "ptbxl_database.csv"
    expected.to_csv(csv_path, index=False)

    result = load_metadata(csv_path)

    pd.testing.assert_frame_equal(result, expected)


def test_assigns_official_splits() -> None:
    result = prepare_metadata(make_metadata())

    assert result["split"].tolist() == [
        "train",
        "train",
        "validation",
        "test",
    ]


@pytest.mark.parametrize("invalid_fold", [0, 11, float("nan"), "invalid"])
def test_rejects_invalid_folds(invalid_fold: object) -> None:
    metadata = make_metadata()
    metadata["strat_fold"] = [invalid_fold, 8, 9, 10]

    with pytest.raises(ValueError, match="strat_fold"):
        prepare_metadata(metadata)


@pytest.mark.parametrize("missing_column", REQUIRED_COLUMNS)
def test_rejects_missing_required_columns(missing_column: str) -> None:
    metadata = make_metadata().drop(columns=missing_column)

    with pytest.raises(KeyError, match=missing_column):
        prepare_metadata(metadata)


@pytest.mark.parametrize("null_column", REQUIRED_COLUMNS)
def test_rejects_null_critical_values(null_column: str) -> None:
    metadata = make_metadata()
    metadata.loc[0, null_column] = None

    with pytest.raises(ValueError, match=null_column):
        prepare_metadata(metadata)


def test_rejects_duplicate_ecg_ids() -> None:
    metadata = make_metadata()
    metadata.loc[1, "ecg_id"] = metadata.loc[0, "ecg_id"]

    with pytest.raises(ValueError, match="ecg_id"):
        prepare_metadata(metadata)


def test_finds_patient_overlap_between_splits() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [42, 42, 99],
            "strat_fold": [1, 9, 10],
        }
    )

    assert find_patient_overlaps(metadata) == {
        "train/validation": [42],
        "train/test": [],
        "validation/test": [],
    }


def test_prepare_metadata_fails_on_patient_leakage() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [42, 42],
            "strat_fold": [1, 9],
        }
    )

    with pytest.raises(ValueError, match="train/validation"):
        prepare_metadata(metadata)


def test_finds_patient_assigned_to_multiple_folds() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [42, 42, 99],
            "strat_fold": [1, 2, 8],
        }
    )

    assert find_patient_fold_conflicts(metadata) == {42: [1, 2]}


def test_prepare_metadata_fails_when_patient_has_multiple_folds() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [42, 42],
            "strat_fold": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="multiple strat_fold"):
        prepare_metadata(metadata)


def test_builds_metadata_summary() -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4, 5, 6],
            "patient_id": [101, 101, 102, 103, 104, 104],
            "strat_fold": [1, 1, 8, 9, 10, 10],
        }
    )
    prepared = prepare_metadata(metadata)

    assert build_metadata_summary(prepared) == {
        "records": 6,
        "patients": 4,
        "folds": {1: 2, 8: 1, 9: 1, 10: 2},
        "splits": {
            "train": {"records": 3, "patients": 2},
            "validation": {"records": 1, "patients": 1},
            "test": {"records": 2, "patients": 1},
        },
        "patient_overlap": {
            "train/validation": [],
            "train/test": [],
            "validation/test": [],
        },
    }
