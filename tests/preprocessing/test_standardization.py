import json
from pathlib import Path

import numpy as np
import pytest

from ptbxl.data.samples import ECGSample
from ptbxl.preprocessing import (
    GlobalStandardizer,
    fit_global_standardizer,
    load_global_standardizer,
    save_global_standardizer,
)


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
SOURCES = {
    "cohort_sha256": "a" * 64,
    "metadata_sha256": "b" * 64,
    "signal_manifest_sha256": "c" * 64,
}


def _sample(
    value: float = 1.0,
    *,
    ecg_id: int = 1,
    split: str = "train",
    lead_names: tuple[str, ...] = LEAD_NAMES,
) -> ECGSample:
    fold = {"train": 1, "validation": 9, "test": 10}[split]
    return ECGSample(
        ecg_id=ecg_id,
        patient_id=ecg_id * 10,
        strat_fold=fold,
        split=split,
        filename_lr=f"records100/00000/{ecg_id:05d}_lr",
        signal=np.full((1_000, 12), value, dtype=np.float64),
        targets=np.array([1, 0, 0, 0, 0], dtype=np.float32),
        sampling_frequency=100.0,
        lead_names=lead_names,
    )


def test_streaming_fit_matches_direct_population_statistics() -> None:
    samples = [_sample(1.0, ecg_id=1), _sample(3.0, ecg_id=2)]
    samples[0].signal[:, :] = np.linspace(-2.0, 2.0, 12_000).reshape(1_000, 12)
    direct = np.concatenate([sample.signal.reshape(-1) for sample in samples])

    standardizer = fit_global_standardizer(iter(samples))

    assert standardizer.mean == pytest.approx(float(direct.mean()))
    assert standardizer.standard_deviation == pytest.approx(float(direct.std(ddof=0)))
    assert standardizer.record_count == 2
    assert standardizer.value_count == 24_000
    assert standardizer.lead_names == LEAD_NAMES


@pytest.mark.parametrize("split", ["validation", "test"])
def test_fit_rejects_non_train_sample(split: str) -> None:
    with pytest.raises(ValueError, match="train samples only"):
        fit_global_standardizer([_sample(split=split)])


def test_fit_rejects_empty_stream() -> None:
    with pytest.raises(ValueError, match="empty sample stream"):
        fit_global_standardizer([])


def test_fit_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="standard deviation must be positive"):
        fit_global_standardizer([_sample(1.0)])


def test_fit_rejects_non_finite_signal() -> None:
    sample = _sample()
    sample.signal[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        fit_global_standardizer([sample])


def test_fit_rejects_inconsistent_lead_order() -> None:
    with pytest.raises(ValueError, match="Unexpected lead order"):
        fit_global_standardizer(
            [
                _sample(1.0, ecg_id=1),
                _sample(3.0, ecg_id=2, lead_names=tuple(reversed(LEAD_NAMES))),
            ]
        )


def test_transform_preserves_shape_and_returns_float32() -> None:
    standardizer = GlobalStandardizer(
        mean=2.0,
        standard_deviation=2.0,
        record_count=2,
        value_count=24_000,
        lead_names=LEAD_NAMES,
    )
    signal = np.full((1_000, 12), 4.0, dtype=np.float64)

    transformed = standardizer.transform(signal, LEAD_NAMES)

    assert transformed.shape == signal.shape
    assert transformed.dtype == np.float32
    np.testing.assert_array_equal(transformed, np.ones_like(transformed))
    np.testing.assert_array_equal(signal, np.full_like(signal, 4.0))


def test_transform_rejects_unexpected_lead_order() -> None:
    standardizer = fit_global_standardizer(
        [_sample(1.0, ecg_id=1), _sample(3.0, ecg_id=2)]
    )

    with pytest.raises(ValueError, match="Unexpected lead order"):
        standardizer.transform(_sample().signal, tuple(reversed(LEAD_NAMES)))


def test_artifact_round_trip_and_bytes_are_deterministic(tmp_path: Path) -> None:
    standardizer = fit_global_standardizer(
        [_sample(1.0, ecg_id=1), _sample(3.0, ecg_id=2)]
    )
    output = tmp_path / "standardizer.json"

    save_global_standardizer(standardizer, output, SOURCES)
    first = output.read_bytes()
    save_global_standardizer(standardizer, output, SOURCES)

    assert output.read_bytes() == first
    assert load_global_standardizer(output) == standardizer
    artifact = json.loads(first)
    assert artifact["schema_version"] == 1
    assert artifact["dataset"] == {
        "name": "PTB-XL",
        "version": "1.0.3",
        "signal_source": "filename_lr",
        "sampling_frequency_hz": 100,
    }
    assert artifact["method"] == "global_standardization"
    assert artifact["fitted_split"] == "train"
    assert artifact["configuration"] == {
        "scope": "all_train_signal_values",
        "variance_ddof": 0,
        "output_dtype": "float32",
    }
    assert artifact["sources"] == SOURCES


def test_save_rejects_missing_or_invalid_provenance(tmp_path: Path) -> None:
    standardizer = fit_global_standardizer(
        [_sample(1.0, ecg_id=1), _sample(3.0, ecg_id=2)]
    )

    with pytest.raises(KeyError, match="source fields"):
        save_global_standardizer(standardizer, tmp_path / "missing.json", {})
    with pytest.raises(ValueError, match="SHA-256"):
        save_global_standardizer(
            standardizer,
            tmp_path / "invalid.json",
            {**SOURCES, "cohort_sha256": "invalid"},
        )


def test_load_rejects_wrong_schema_or_invalid_statistics(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {
                    "name": "PTB-XL",
                    "version": "1.0.3",
                    "signal_source": "filename_lr",
                    "sampling_frequency_hz": 100,
                },
                "method": "global_standardization",
                "fitted_split": "train",
                "configuration": {
                    "scope": "all_train_signal_values",
                    "variance_ddof": 0,
                    "output_dtype": "float32",
                },
                "lead_names": list(LEAD_NAMES),
                "statistics": {
                    "mean": 0.0,
                    "standard_deviation": 0.0,
                    "records": 1,
                    "values": 12_000,
                },
                "sources": SOURCES,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_global_standardizer(path)


def test_load_rejects_fractional_counts(tmp_path: Path) -> None:
    standardizer = fit_global_standardizer(
        [_sample(1.0, ecg_id=1), _sample(3.0, ecg_id=2)]
    )
    path = tmp_path / "fractional.json"
    save_global_standardizer(standardizer, path, SOURCES)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["statistics"]["records"] = 2.5
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact fields are invalid"):
        load_global_standardizer(path)
