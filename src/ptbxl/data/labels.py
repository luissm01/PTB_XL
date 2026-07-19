"""Build auditable PTB-XL diagnostic superclass labels."""

import ast
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


TARGET_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
IDENTITY_COLUMNS = ("ecg_id", "patient_id", "strat_fold", "split")
STATEMENT_COLUMNS = ("diagnostic", "diagnostic_class")


def load_scp_statements(path: str | Path) -> pd.DataFrame:
    """Load and validate the official SCP statement catalogue."""
    statements = pd.read_csv(path, index_col=0)
    statements.index = statements.index.astype(str)
    missing = [column for column in STATEMENT_COLUMNS if column not in statements]
    if missing:
        raise KeyError(f"Missing SCP statement columns: {missing}")
    if statements.index.has_duplicates:
        duplicates = statements.index[statements.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate SCP statement codes: {duplicates}")
    return statements


def parse_scp_codes(value: Any) -> dict[str, float]:
    """Safely parse a serialized SCP code-to-likelihood dictionary."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Malformed scp_codes value: {value!r}") from error
    elif isinstance(value, dict):
        parsed = value
    else:
        raise ValueError(
            f"scp_codes must be a dictionary or string, got {type(value).__name__}"
        )

    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must contain a dictionary")

    result: dict[str, float] = {}
    for code, likelihood in parsed.items():
        if not isinstance(code, str) or not code:
            raise ValueError(f"Invalid SCP code: {code!r}")
        if isinstance(likelihood, bool) or not isinstance(likelihood, (int, float)):
            raise ValueError(f"Likelihood for {code!r} must be numeric")
        numeric_likelihood = float(likelihood)
        if not math.isfinite(numeric_likelihood) or not 0 <= numeric_likelihood <= 100:
            raise ValueError(f"Likelihood for {code!r} must be between 0 and 100")
        result[code] = numeric_likelihood
    return result


def build_diagnostic_superclass_mapping(statements: pd.DataFrame) -> dict[str, str]:
    """Map official diagnostic codes to the five target superclasses."""
    missing = [column for column in STATEMENT_COLUMNS if column not in statements]
    if missing:
        raise KeyError(f"Missing SCP statement columns: {missing}")

    mapping: dict[str, str] = {}
    for code, row in statements.iterrows():
        if row["diagnostic"] == 1 and row["diagnostic_class"] in TARGET_SUPERCLASSES:
            mapping[str(code)] = str(row["diagnostic_class"])
    return mapping


def build_superclass_labels(
    metadata: pd.DataFrame,
    mapping: Mapping[str, str],
    known_codes: set[str],
) -> pd.DataFrame:
    """Build binary labels while preserving validated identities and splits."""
    required = (*IDENTITY_COLUMNS, "scp_codes")
    missing = [column for column in required if column not in metadata]
    if missing:
        raise KeyError(f"Missing label input columns: {missing}")
    if metadata["ecg_id"].duplicated().any():
        raise ValueError("Label input contains duplicate ecg_id values")
    invalid_splits = sorted(set(metadata["split"]) - {"train", "validation", "test"})
    if invalid_splits:
        raise ValueError(f"Invalid inherited split values: {invalid_splits}")

    parsed_rows = [parse_scp_codes(value) for value in metadata["scp_codes"]]
    unknown_codes = sorted(
        {code for codes in parsed_rows for code in codes if code not in known_codes}
    )
    if unknown_codes:
        raise ValueError(f"SCP codes absent from scp_statements.csv: {unknown_codes}")

    output = metadata.loc[:, IDENTITY_COLUMNS].copy().reset_index(drop=True)
    for superclass in TARGET_SUPERCLASSES:
        output[superclass] = [
            int(any(mapping.get(code) == superclass for code in codes))
            for codes in parsed_rows
        ]
    return output


def count_excluded_codes(
    metadata: pd.DataFrame,
    mapping: Mapping[str, str],
    known_codes: set[str],
) -> dict[str, int]:
    """Count known codes that do not contribute to target labels."""
    counts: Counter[str] = Counter()
    for value in metadata["scp_codes"]:
        for code in parse_scp_codes(value):
            if code not in known_codes:
                raise ValueError(f"SCP code absent from scp_statements.csv: {code}")
            if code not in mapping:
                counts[code] += 1
    return dict(sorted(counts.items()))


def build_label_summary(
    labels: pd.DataFrame,
    excluded_code_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Summarize target prevalence and label cardinality deterministically."""
    missing = [
        column
        for column in (*IDENTITY_COLUMNS, *TARGET_SUPERCLASSES)
        if column not in labels
    ]
    if missing:
        raise KeyError(f"Missing label summary columns: {missing}")

    target_values = labels.loc[:, TARGET_SUPERCLASSES]
    if not target_values.isin([0, 1]).all().all():
        raise ValueError("Target labels must be binary")
    label_counts = target_values.sum(axis=1)

    per_label: dict[str, dict[str, int]] = {}
    for label in TARGET_SUPERCLASSES:
        per_label[label] = {"total": int(labels[label].sum())}
        for split in ("train", "validation", "test"):
            per_label[label][split] = int(
                labels.loc[labels["split"] == split, label].sum()
            )

    combinations: Counter[str] = Counter()
    for row in target_values.itertuples(index=False, name=None):
        active = [label for label, present in zip(TARGET_SUPERCLASSES, row) if present]
        combinations["+".join(active) if active else "NONE"] += 1

    return {
        "records": {
            "total": len(labels),
            "with_any_target_label": int(label_counts.gt(0).sum()),
            "without_target_label": int(label_counts.eq(0).sum()),
            "multilabel_records": int(label_counts.gt(1).sum()),
        },
        "labels": per_label,
        "label_cardinality": float(label_counts.mean()) if len(labels) else 0.0,
        "label_combinations": dict(sorted(combinations.items())),
        "excluded_known_codes": dict(sorted(excluded_code_counts.items())),
    }
