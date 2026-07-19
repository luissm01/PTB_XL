"""Validate real PTB-XL metadata and write a deterministic evidence report."""

import argparse
import json
from pathlib import Path
from typing import Any

from ptbxl.data.metadata import load_metadata
from ptbxl.data.reporting import (
    build_metadata_report,
    verify_sha256,
    write_json_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Metadata manifest must contain a JSON object")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest_path)
    verify_sha256(args.metadata_path, manifest["sha256"])
    metadata = load_metadata(args.metadata_path)
    report = build_metadata_report(metadata, manifest)
    write_json_report(report, args.output_path)
    print(f"Validated {len(metadata)} records; report: {args.output_path}")


if __name__ == "__main__":
    main()
