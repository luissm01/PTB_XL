from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

import ptbxl.data.pytorch as pytorch_module
from ptbxl.data import build_sample_index
from ptbxl.data.pytorch import PTBXLDataset, build_dataloader
from ptbxl.data.samples import load_sample
from ptbxl.preprocessing import GlobalStandardizer


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


def _cohort(count: int, split: str = "train") -> pd.DataFrame:
    fold = {"train": 1, "validation": 9, "test": 10}[split]
    return pd.DataFrame(
        {
            "ecg_id": range(1, count + 1),
            "patient_id": range(101, 101 + count),
            "strat_fold": [fold] * count,
            "split": [split] * count,
            "NORM": [1] * count,
            "MI": [0] * count,
            "STTC": [0] * count,
            "CD": [0] * count,
            "HYP": [0] * count,
        }
    )


def _metadata(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": range(1, count + 1),
            "filename_lr": [
                f"records100/00000/{ecg_id:05d}_lr" for ecg_id in range(1, count + 1)
            ],
        }
    )


def _write_record(root: Path, ecg_id: int) -> None:
    relative = Path("records100") / "00000" / f"{ecg_id:05d}_lr"
    record_dir = root / relative.parent
    record_dir.mkdir(parents=True, exist_ok=True)
    base = np.linspace(-1.0, 1.0, 1_000) + ecg_id
    values = np.column_stack([base + lead / 100 for lead in range(12)])
    wfdb.wrsamp(
        record_name=relative.name,
        fs=100,
        units=["mV"] * 12,
        sig_name=list(LEAD_NAMES),
        p_signal=values,
        write_dir=str(record_dir),
    )


def _sample_index(count: int, split: str = "train") -> pd.DataFrame:
    return build_sample_index(_cohort(count, split), _metadata(count))


def _standardizer() -> GlobalStandardizer:
    return GlobalStandardizer(
        mean=1.0,
        standard_deviation=2.0,
        record_count=1,
        value_count=12_000,
        lead_names=LEAD_NAMES,
    )


def _dataset(
    tmp_path: Path,
    *,
    count: int = 1,
    split: str = "train",
) -> PTBXLDataset:
    for ecg_id in range(1, count + 1):
        _write_record(tmp_path, ecg_id)
    return PTBXLDataset(
        _sample_index(count, split),
        tmp_path,
        _standardizer(),
        split=split,
    )


def test_dataset_construction_is_lazy_and_copies_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _sample_index(2)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Dataset construction must not load a signal")

    monkeypatch.setattr(pytorch_module, "load_sample", fail_if_called)
    dataset = PTBXLDataset(index, tmp_path, _standardizer(), split="train")
    index.loc[:, "ecg_id"] = 999

    assert len(dataset) == 2
    assert dataset._rows[0]["ecg_id"] == 1


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_item_reuses_sample_and_frozen_standardizer(
    tmp_path: Path,
    split: str,
) -> None:
    _write_record(tmp_path, 1)
    index = _sample_index(1, split)
    standardizer = _standardizer()
    expected_sample = load_sample(index.iloc[0].to_dict(), tmp_path)
    expected_signal = standardizer.transform(
        expected_sample.signal,
        expected_sample.lead_names,
    )
    original_standardizer = _standardizer()
    dataset = PTBXLDataset(
        index,
        tmp_path,
        standardizer,
        split=split,
    )

    item = dataset[0]

    assert item["signal"].shape == (12, 1_000)
    assert item["signal"].dtype == torch.float32
    assert item["signal"].is_contiguous()
    torch.testing.assert_close(
        item["signal"],
        torch.from_numpy(np.ascontiguousarray(expected_signal.T)),
    )
    assert item["targets"].shape == (5,)
    assert item["targets"].dtype == torch.float32
    torch.testing.assert_close(
        item["targets"],
        torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
    )
    assert item["ecg_id"] == 1
    assert item["patient_id"] == 101
    assert item["strat_fold"] == {"train": 1, "validation": 9, "test": 10}[split]
    assert item["split"] == split
    assert item["filename_lr"] == "records100/00000/00001_lr"
    assert standardizer == original_standardizer


def test_rejects_invalid_sample_index_contract(tmp_path: Path) -> None:
    index = _sample_index(2)

    with pytest.raises(KeyError, match="sample-index columns"):
        PTBXLDataset(
            index.drop(columns="filename_lr"),
            tmp_path,
            _standardizer(),
            split="train",
        )
    with pytest.raises(ValueError, match="at least one sample"):
        PTBXLDataset(index.iloc[:0], tmp_path, _standardizer(), split="train")
    with pytest.raises(ValueError, match="Invalid Dataset split"):
        PTBXLDataset(index, tmp_path, _standardizer(), split="development")


def test_rejects_mixed_or_wrong_split(tmp_path: Path) -> None:
    index = _sample_index(2)
    index.loc[1, ["strat_fold", "split"]] = [9, "validation"]

    with pytest.raises(ValueError, match="contains"):
        PTBXLDataset(index, tmp_path, _standardizer(), split="train")
    with pytest.raises(ValueError, match="contains"):
        PTBXLDataset(
            _sample_index(1, "validation"),
            tmp_path,
            _standardizer(),
            split="train",
        )


def test_rejects_invalid_standardizer_and_index(tmp_path: Path) -> None:
    index = _sample_index(1)

    with pytest.raises(TypeError, match="GlobalStandardizer"):
        PTBXLDataset(index, tmp_path, object(), split="train")  # type: ignore[arg-type]

    dataset = PTBXLDataset(index, tmp_path, _standardizer(), split="train")
    with pytest.raises(IndexError, match="out of range"):
        dataset[1]
    with pytest.raises(TypeError, match="integer"):
        dataset["0"]  # type: ignore[index]


def test_dataloader_batches_default_collation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=3)

    batch = next(
        iter(
            build_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
            )
        )
    )

    assert batch["signal"].shape == (2, 12, 1_000)
    assert batch["signal"].dtype == torch.float32
    assert batch["signal"].is_contiguous()
    assert batch["targets"].shape == (2, 5)
    assert batch["targets"].dtype == torch.float32
    assert batch["ecg_id"].tolist() == [1, 2]
    assert batch["split"] == ["train", "train"]


def test_seeded_shuffle_is_reproducible(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=6)
    first = build_dataloader(dataset, batch_size=2, shuffle=True, seed=2026)
    second = build_dataloader(dataset, batch_size=2, shuffle=True, seed=2026)

    first_order = [ecg_id for batch in first for ecg_id in batch["ecg_id"].tolist()]
    second_order = [ecg_id for batch in second for ecg_id in batch["ecg_id"].tolist()]

    assert first_order == second_order
    assert sorted(first_order) == list(range(1, 7))


def test_shuffle_requires_valid_seed(tmp_path: Path) -> None:
    dataset = PTBXLDataset(
        _sample_index(1),
        tmp_path,
        _standardizer(),
        split="train",
    )

    with pytest.raises(ValueError, match="seed is required"):
        build_dataloader(dataset, batch_size=1, shuffle=True)
    with pytest.raises(ValueError, match="non-negative"):
        build_dataloader(dataset, batch_size=1, shuffle=True, seed=-1)
