from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

import ptbxl.data.signals as signals_module
from ptbxl.data.reporting import write_json_report
from ptbxl.data.signals import (
    ECGSignal,
    SignalLoadError,
    audit_lr_signals,
    load_signal_for_row,
    validate_signal,
)


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


def _signal(
    samples: int = 1_000,
    leads: int = 12,
    sampling_frequency: float = 100.0,
    lead_names: tuple[str, ...] = LEAD_NAMES,
) -> ECGSignal:
    values = np.arange(samples * leads, dtype=np.float64).reshape(samples, leads)
    return ECGSignal(values, sampling_frequency, lead_names)


def _write_record(
    root: Path,
    ecg_id: int,
    *,
    samples: int = 1_000,
    sampling_frequency: float = 100.0,
    lead_names: tuple[str, ...] = LEAD_NAMES,
) -> str:
    relative = Path("records100") / "00000" / f"{ecg_id:05d}_lr"
    record_dir = root / relative.parent
    record_dir.mkdir(parents=True, exist_ok=True)
    base = np.linspace(-1.0, 1.0, samples)
    values = np.column_stack(
        [base + lead_index / 100 for lead_index in range(len(lead_names))]
    )
    wfdb.wrsamp(
        record_name=relative.name,
        fs=sampling_frequency,
        units=["mV"] * len(lead_names),
        sig_name=list(lead_names),
        p_signal=values,
        write_dir=str(record_dir),
    )
    return relative.as_posix()


def test_loads_valid_wfdb_signal_and_preserves_header_facts(tmp_path: Path) -> None:
    filename_lr = _write_record(tmp_path, 1)

    record = load_signal_for_row(
        {"ecg_id": 1, "filename_lr": filename_lr},
        tmp_path,
    )

    assert record.signal.shape == (1_000, 12)
    assert np.issubdtype(record.signal.dtype, np.number)
    assert record.sampling_frequency == 100.0
    assert record.lead_names == LEAD_NAMES
    assert validate_signal(record) is record


def test_passes_exact_filename_lr_basename_to_wfdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = Path("records100/00000/00001_lr")
    header = tmp_path / f"{relative}.hea"
    header.parent.mkdir(parents=True)
    header.touch()
    observed: list[str] = []

    def fake_rdsamp(record_name: str) -> tuple[np.ndarray, dict[str, object]]:
        observed.append(record_name)
        return _signal().signal, {"fs": 100.0, "sig_name": list(LEAD_NAMES)}

    monkeypatch.setattr(wfdb, "rdsamp", fake_rdsamp)

    load_signal_for_row(
        {"ecg_id": 1, "filename_lr": relative.as_posix()},
        tmp_path,
    )

    assert observed == [str((tmp_path / relative).resolve())]


def test_missing_wfdb_record_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ecg_id 1"):
        load_signal_for_row(
            {"ecg_id": 1, "filename_lr": "records100/00000/00001_lr"},
            tmp_path,
        )


def test_corrupt_wfdb_record_fails_clearly(tmp_path: Path) -> None:
    header = tmp_path / "records100/00000/00001_lr.hea"
    header.parent.mkdir(parents=True)
    header.write_text("not a WFDB header\n", encoding="utf-8")

    with pytest.raises(SignalLoadError, match="ecg_id 1"):
        load_signal_for_row(
            {"ecg_id": 1, "filename_lr": "records100/00000/00001_lr"},
            tmp_path,
        )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_signal(leads=11, lead_names=LEAD_NAMES[:11]), "Signal shape"),
        (_signal(samples=999), "Signal shape"),
        (_signal(sampling_frequency=500.0), "Sampling frequency"),
        (
            ECGSignal(_signal().signal, 100.0, LEAD_NAMES[:11]),
            "12 lead names",
        ),
    ],
)
def test_rejects_invalid_signal_contract(record: ECGSignal, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_signal(record)


@pytest.mark.parametrize("non_finite", [np.nan, np.inf])
def test_rejects_non_finite_signal_values(non_finite: float) -> None:
    record = _signal()
    record.signal[0, 0] = non_finite

    with pytest.raises(ValueError, match="non-finite"):
        validate_signal(record)


def test_rejects_non_numeric_signal_values() -> None:
    record = ECGSignal(
        np.full((1_000, 12), "not numeric", dtype=object),
        100.0,
        LEAD_NAMES,
    )

    with pytest.raises(TypeError, match="numeric"):
        validate_signal(record)


def test_rejects_unexpected_lead_order() -> None:
    record = _signal(lead_names=tuple(reversed(LEAD_NAMES)))

    with pytest.raises(ValueError, match="Unexpected lead order"):
        validate_signal(record, LEAD_NAMES)


@pytest.mark.parametrize(
    "filename_lr",
    [
        "records100/00000/00002_lr",
        "../../00001_lr",
        "records100/00000/00001_lr.hea",
    ],
)
def test_rejects_wrong_or_unsafe_signal_path(
    filename_lr: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        load_signal_for_row(
            {"ecg_id": 1, "filename_lr": filename_lr},
            tmp_path,
        )


def test_audits_loaded_missing_and_invalid_records_deterministically(
    tmp_path: Path,
) -> None:
    valid_path = _write_record(tmp_path, 1)
    invalid_path = _write_record(tmp_path, 3, lead_names=LEAD_NAMES[:11])
    cohort = pd.DataFrame({"ecg_id": [1, 2, 3]})
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "filename_lr": [
                valid_path,
                "records100/00000/00002_lr",
                invalid_path,
            ],
        }
    )

    report = audit_lr_signals(cohort, metadata, tmp_path)

    assert report["records"] == {
        "expected": 3,
        "loaded": 1,
        "missing": 1,
        "invalid": 1,
    }
    assert report["shapes"] == {"1000x11": 1, "1000x12": 1}
    assert report["sampling_frequencies"] == {"100": 2}
    assert report["lead_orders"] == {
        ",".join(LEAD_NAMES): 1,
        ",".join(LEAD_NAMES[:11]): 1,
    }
    assert report["non_finite_values"] == 0

    output = tmp_path / "audit.json"
    write_json_report(report, output)
    first = output.read_bytes()
    write_json_report(report, output)
    assert output.read_bytes() == first


def test_audit_rejects_missing_cohort_identity_mapping(tmp_path: Path) -> None:
    cohort = pd.DataFrame({"ecg_id": [1, 2]})
    metadata = pd.DataFrame(
        {"ecg_id": [1], "filename_lr": ["records100/00000/00001_lr"]}
    )

    with pytest.raises(ValueError, match="missing from metadata"):
        audit_lr_signals(cohort, metadata, tmp_path)


def test_audit_counts_non_finite_values_as_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = pd.DataFrame({"ecg_id": [1, 2]})
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "filename_lr": [
                "records100/00000/00001_lr",
                "records100/00000/00002_lr",
            ],
        }
    )

    def fake_loader(row: dict[str, object], dataset_root: Path) -> ECGSignal:
        del dataset_root
        record = _signal()
        if row["ecg_id"] == 2:
            record.signal[0, 0] = np.nan
        return record

    monkeypatch.setattr(signals_module, "load_signal_for_row", fake_loader)

    report = audit_lr_signals(cohort, metadata, tmp_path)

    assert report["records"] == {
        "expected": 2,
        "loaded": 1,
        "missing": 0,
        "invalid": 1,
    }
    assert report["non_finite_values"] == 1
