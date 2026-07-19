"""Build PTB-XL diagnostic superclass labels and deterministic evidence."""

import argparse
import json
from pathlib import Path
from typing import Any

from ptbxl.data.labels import (
    build_diagnostic_superclass_mapping,
    build_label_summary,
    build_superclass_labels,
    count_excluded_codes,
    load_scp_statements,
)
from ptbxl.data.metadata import load_metadata, prepare_metadata
from ptbxl.data.reporting import verify_sha256, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--metadata-manifest-path", required=True, type=Path)
    parser.add_argument("--statements-path", required=True, type=Path)
    parser.add_argument("--statements-manifest-path", required=True, type=Path)
    parser.add_argument("--labels-output-path", required=True, type=Path)
    parser.add_argument("--report-output-path", required=True, type=Path)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    metadata_manifest = load_manifest(args.metadata_manifest_path)
    statements_manifest = load_manifest(args.statements_manifest_path)
    verify_sha256(args.metadata_path, metadata_manifest["sha256"])
    verify_sha256(args.statements_path, statements_manifest["sha256"])

    metadata = prepare_metadata(load_metadata(args.metadata_path))
    statements = load_scp_statements(args.statements_path)
    mapping = build_diagnostic_superclass_mapping(statements)
    known_codes = set(statements.index)
    labels = build_superclass_labels(metadata, mapping, known_codes)
    excluded = count_excluded_codes(metadata, mapping, known_codes)
    summary = build_label_summary(labels, excluded)
    report = {
        "dataset": {"name": "PTB-XL", "version": metadata_manifest["version"]},
        "sources": {
            "metadata_sha256": metadata_manifest["sha256"],
            "scp_statements_sha256": statements_manifest["sha256"],
        },
        **summary,
    }

    args.labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.labels_output_path, index=False, lineterminator="\n")
    write_json_report(report, args.report_output_path)
    print(
        f"Built {len(labels)} label rows; report: {args.report_output_path}; "
        f"derived table: {args.labels_output_path}"
    )


if __name__ == "__main__":
    main()
