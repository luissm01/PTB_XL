"""Fit deterministic global signal standardization on PTB-XL train only."""

import argparse
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ptbxl.data.reporting import compute_sha256
from ptbxl.data.samples import ECGSample, build_sample_index, load_sample
from ptbxl.preprocessing import fit_global_standardizer, save_global_standardizer


EXPECTED_TRAIN_RECORDS = 17_084


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-path", required=True, type=Path)
    parser.add_argument(
        "--metadata-path",
        default=Path("data/raw/ptbxl_database.csv"),
        type=Path,
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--signal-manifest-path",
        default=Path("data/raw/SHA256SUMS.txt"),
        type=Path,
    )
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument(
        "--expected-train-records",
        default=EXPECTED_TRAIN_RECORDS,
        type=int,
    )
    return parser.parse_args()


def iter_samples(
    sample_index: pd.DataFrame,
    dataset_root: Path,
) -> Iterator[ECGSample]:
    """Load indexed samples sequentially without retaining waveforms."""
    for row in sample_index.itertuples(index=False):
        yield load_sample(row._asdict(), dataset_root)


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(args.cohort_path)
    metadata = pd.read_csv(
        args.metadata_path,
        usecols=["ecg_id", "filename_lr"],
    )
    sample_index = build_sample_index(cohort, metadata)
    train_index = sample_index.loc[sample_index["split"] == "train"].reset_index(
        drop=True
    )
    if len(train_index) != args.expected_train_records:
        raise ValueError(
            f"Expected {args.expected_train_records} train records, "
            f"got {len(train_index)}"
        )

    standardizer = fit_global_standardizer(iter_samples(train_index, args.dataset_root))
    sources = {
        "cohort_sha256": compute_sha256(args.cohort_path),
        "metadata_sha256": compute_sha256(args.metadata_path),
        "signal_manifest_sha256": compute_sha256(args.signal_manifest_path),
    }
    save_global_standardizer(standardizer, args.output_path, sources)
    print(
        f"Fitted {standardizer.record_count} train records and "
        f"{standardizer.value_count} values; artifact: {args.output_path}"
    )


if __name__ == "__main__":
    main()
