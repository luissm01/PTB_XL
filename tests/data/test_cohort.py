from pathlib import Path

import pandas as pd
import pytest

from ptbxl.data.cohort import (
    COHORT_COLUMNS,
    build_cohort_summary,
    build_modeling_cohort,
    find_cohort_exclusions,
)
from ptbxl.data.reporting import write_json_report


def labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "patient_id": [10, 20, 30, 40],
            "strat_fold": [1, 2, 9, 10],
            "split": ["train", "train", "validation", "test"],
            "NORM": [1, 0, 0, 0],
            "MI": [0, 1, 0, 0],
            "STTC": [0, 1, 0, 0],
            "CD": [0, 0, 0, 1],
            "HYP": [0, 0, 0, 0],
        }
    )


def test_includes_single_and_multilabel_records_and_excludes_zero_target() -> None:
    cohort = build_modeling_cohort(labels())
    exclusions = find_cohort_exclusions(labels())

    assert cohort["ecg_id"].tolist() == [1, 2, 4]
    assert exclusions.to_dict("records") == [
        {
            "ecg_id": 3,
            "patient_id": 30,
            "split": "validation",
            "reason": "no_target_superclass",
        }
    ]


def test_partition_preserves_values_and_accounts_for_every_input() -> None:
    source = labels()
    cohort = build_modeling_cohort(source)
    exclusions = find_cohort_exclusions(source)

    pd.testing.assert_frame_equal(
        cohort,
        source.loc[source["ecg_id"].isin([1, 2, 4]), COHORT_COLUMNS].reset_index(
            drop=True
        ),
    )
    assert set(cohort["ecg_id"]).isdisjoint(exclusions["ecg_id"])
    assert set(cohort["ecg_id"]) | set(exclusions["ecg_id"]) == set(source["ecg_id"])
    assert len(cohort) + len(exclusions) == len(source)


@pytest.mark.parametrize("missing", ["NORM", "split"])
def test_rejects_missing_required_column(missing: str) -> None:
    with pytest.raises(KeyError, match="Missing cohort input"):
        build_modeling_cohort(labels().drop(columns=missing))


def test_rejects_non_binary_target() -> None:
    source = labels()
    source.loc[0, "NORM"] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        build_modeling_cohort(source)


def test_rejects_duplicate_ecg_id() -> None:
    source = labels()
    source.loc[1, "ecg_id"] = 1
    with pytest.raises(ValueError, match="duplicate ecg_id"):
        build_modeling_cohort(source)


def test_rejects_invalid_split() -> None:
    source = labels()
    source.loc[0, "split"] = "development"
    with pytest.raises(ValueError, match="Invalid inherited split"):
        build_modeling_cohort(source)


def test_builds_exact_cohort_summary() -> None:
    source = labels()
    cohort = build_modeling_cohort(source)
    summary = build_cohort_summary(source, cohort)

    assert summary["records"] == {"input": 4, "included": 3, "excluded": 1}
    assert summary["splits"] == {
        "train": {"records": 2, "patients": 2},
        "validation": {"records": 0, "patients": 0},
        "test": {"records": 1, "patients": 1},
    }
    assert summary["label_prevalence"]["train"]["MI"] == {
        "count": 1,
        "fraction": 0.5,
    }
    assert summary["label_cardinality"] == {
        "train": 1.5,
        "validation": 0.0,
        "test": 1.0,
    }
    assert summary["label_count_distribution"]["train"] == {
        "1": 1,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
    }
    assert summary["label_combinations"]["train"] == {"MI+STTC": 1, "NORM": 1}
    assert summary["patient_overlap"] == {
        "train_validation": [],
        "train_test": [],
        "validation_test": [],
    }


def test_summary_rejects_changed_cohort_values() -> None:
    source = labels()
    cohort = build_modeling_cohort(source)
    cohort.loc[0, "patient_id"] = 999
    with pytest.raises(ValueError, match="changed identity"):
        build_cohort_summary(source, cohort)


def test_summary_rejects_patient_overlap_after_filtering() -> None:
    source = labels()
    source.loc[3, "patient_id"] = 10
    cohort = build_modeling_cohort(source)
    with pytest.raises(ValueError, match="Patient overlap"):
        build_cohort_summary(source, cohort)


def test_cohort_report_writes_identical_bytes(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    report = build_cohort_summary(labels(), build_modeling_cohort(labels()))
    write_json_report(report, output)
    first = output.read_bytes()
    write_json_report(report, output)
    assert output.read_bytes() == first
