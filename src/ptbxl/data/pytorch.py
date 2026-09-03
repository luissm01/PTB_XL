"""Thin PyTorch adapters for validated PTB-XL samples."""

import operator
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ptbxl.data.cohort import ALLOWED_SPLITS
from ptbxl.data.samples import SAMPLE_INDEX_COLUMNS, load_sample
from ptbxl.preprocessing import GlobalStandardizer


class PTBXLDataset(Dataset[dict[str, Any]]):
    """Lazily adapt one validated split to channel-first PyTorch tensors."""

    def __init__(
        self,
        sample_index: pd.DataFrame,
        dataset_root: str | Path,
        standardizer: GlobalStandardizer,
        *,
        split: str,
    ) -> None:
        missing = [
            column for column in SAMPLE_INDEX_COLUMNS if column not in sample_index
        ]
        if missing:
            raise KeyError(f"Missing PyTorch sample-index columns: {missing}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid Dataset split: {split!r}")
        if sample_index.empty:
            raise ValueError("PyTorch Dataset requires at least one sample")
        if sample_index["split"].isna().any():
            raise ValueError("PyTorch Dataset split values cannot be null")

        row_splits = set(sample_index["split"])
        if row_splits != {split}:
            raise ValueError(
                f"PyTorch Dataset declared split {split!r} but contains "
                f"{sorted(row_splits)!r}"
            )
        if not isinstance(standardizer, GlobalStandardizer):
            raise TypeError("standardizer must be a fitted GlobalStandardizer")

        self._rows: tuple[dict[str, Any], ...] = tuple(
            sample_index.loc[:, SAMPLE_INDEX_COLUMNS].to_dict(orient="records")
        )
        self._dataset_root = Path(dataset_root)
        self._standardizer = standardizer
        self.split = split

    def __len__(self) -> int:
        """Return the number of samples in the declared split."""
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load, standardize and adapt one sample without changing provenance."""
        try:
            position = operator.index(index)
        except TypeError as error:
            raise TypeError("Dataset index must be an integer") from error
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        sample = load_sample(self._rows[position], self._dataset_root)
        standardized = self._standardizer.transform(
            sample.signal,
            sample.lead_names,
        )
        channel_first = np.ascontiguousarray(standardized.T)
        targets = np.ascontiguousarray(sample.targets)

        return {
            "signal": torch.from_numpy(channel_first),
            "targets": torch.from_numpy(targets),
            "ecg_id": sample.ecg_id,
            "patient_id": sample.patient_id,
            "strat_fold": sample.strat_fold,
            "split": sample.split,
            "filename_lr": sample.filename_lr,
        }


def build_dataloader(
    dataset: PTBXLDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    num_workers: int = 0,
) -> DataLoader[Mapping[str, Any]]:
    """Build a standard DataLoader with explicit reproducible shuffling."""
    if not isinstance(dataset, PTBXLDataset):
        raise TypeError("dataset must be a PTBXLDataset")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean")
    if shuffle and seed is None:
        raise ValueError("A seed is required when DataLoader shuffle is enabled")
    if isinstance(seed, bool) or (seed is not None and not isinstance(seed, int)):
        raise TypeError("DataLoader seed must be an integer or None")
    if seed is not None and seed < 0:
        raise ValueError("DataLoader seed must be non-negative")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
