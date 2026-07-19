from pathlib import Path

import pandas as pd
import pytest

from ptbxl.data.labels import (
    TARGET_SUPERCLASSES,
    build_diagnostic_superclass_mapping,
    build_label_summary,
    build_superclass_labels,
    count_excluded_codes,
    load_scp_statements,
    parse_scp_codes,
)
from ptbxl.data.reporting import write_json_report


def statements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 0.0, 1.0],
            "diagnostic_class": ["MI", "STTC", None, "OTHER"],
        },
        index=["IMI", "NDT", "SR", "OUT"],
    )


def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [10, 20, 30],
            "strat_fold": [1, 9, 10],
            "split": ["train", "validation", "test"],
            "scp_codes": [
                "{'IMI': 100.0, 'NDT': 0.0}",
                "{'SR': 0.0}",
                "{'OUT': 50.0}",
            ],
        }
    )


def test_parse_scp_codes_handles_one_and_multiple_codes() -> None:
    assert parse_scp_codes("{'IMI': 100.0}") == {"IMI": 100.0}
    assert parse_scp_codes("{'IMI': 100, 'NDT': 0}") == {
        "IMI": 100.0,
        "NDT": 0.0,
    }


def test_parse_scp_codes_treats_null_as_empty() -> None:
    assert parse_scp_codes(None) == {}
    assert parse_scp_codes(float("nan")) == {}


def test_parse_scp_codes_rejects_malformed_or_unsafe_values() -> None:
    with pytest.raises(ValueError, match="Malformed scp_codes"):
        parse_scp_codes("__import__('os').system('bad')")
    with pytest.raises(ValueError, match="dictionary"):
        parse_scp_codes("['IMI']")


def test_load_scp_statements_uses_code_index(tmp_path: Path) -> None:
    source = tmp_path / "statements.csv"
    source.write_text(
        ",diagnostic,diagnostic_class\nIMI,1.0,MI\n",
        encoding="utf-8",
    )
    loaded = load_scp_statements(source)
    assert list(loaded.index) == ["IMI"]


def test_mapping_uses_only_official_target_diagnostic_codes() -> None:
    mapping = build_diagnostic_superclass_mapping(statements())
    assert mapping == {"IMI": "MI", "NDT": "STTC"}


def test_builds_multilabel_binary_table_and_preserves_identity() -> None:
    source = metadata()
    mapping = build_diagnostic_superclass_mapping(statements())
    labels = build_superclass_labels(source, mapping, set(statements().index))

    assert list(labels.columns) == [
        "ecg_id",
        "patient_id",
        "strat_fold",
        "split",
        *TARGET_SUPERCLASSES,
    ]
    pd.testing.assert_frame_equal(
        labels.loc[:, ["ecg_id", "patient_id", "strat_fold", "split"]],
        source.loc[:, ["ecg_id", "patient_id", "strat_fold", "split"]],
    )
    assert labels.loc[0, ["MI", "STTC"]].tolist() == [1, 1]
    assert labels.loc[1, list(TARGET_SUPERCLASSES)].sum() == 0
    assert labels.loc[:, TARGET_SUPERCLASSES].isin([0, 1]).all().all()


def test_unknown_code_fails_explicitly() -> None:
    source = metadata().iloc[[0]].copy()
    source["scp_codes"] = "{'UNKNOWN': 100}"
    with pytest.raises(ValueError, match="absent from scp_statements"):
        build_superclass_labels(source, {}, set(statements().index))


def test_counts_known_non_target_codes() -> None:
    assert count_excluded_codes(
        metadata(),
        build_diagnostic_superclass_mapping(statements()),
        set(statements().index),
    ) == {"OUT": 1, "SR": 1}


def test_label_summary_counts_splits_multilabel_and_unlabelled() -> None:
    labels = build_superclass_labels(
        metadata(),
        build_diagnostic_superclass_mapping(statements()),
        set(statements().index),
    )
    summary = build_label_summary(labels, {"SR": 1, "OUT": 1})

    assert summary["records"] == {
        "total": 3,
        "with_any_target_label": 1,
        "without_target_label": 2,
        "multilabel_records": 1,
    }
    assert summary["labels"]["MI"] == {
        "total": 1,
        "train": 1,
        "validation": 0,
        "test": 0,
    }
    assert summary["label_cardinality"] == pytest.approx(2 / 3)
    assert summary["label_combinations"] == {"MI+STTC": 1, "NONE": 2}


def test_label_report_writes_identical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    report = {"labels": {"MI": 1}, "records": 3}
    write_json_report(report, path)
    first = path.read_bytes()
    write_json_report(report, path)
    assert path.read_bytes() == first
