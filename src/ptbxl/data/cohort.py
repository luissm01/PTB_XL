"""Define and audit the initial five-superclass modeling cohort."""

from collections import Counter
from typing import Any

import pandas as pd

from ptbxl.data.labels import IDENTITY_COLUMNS, TARGET_SUPERCLASSES


ALLOWED_SPLITS = ("train", "validation", "test")
EXCLUSION_REASON = "no_target_superclass"
COHORT_COLUMNS = (*IDENTITY_COLUMNS, *TARGET_SUPERCLASSES)
EXCLUSION_COLUMNS = ("ecg_id", "patient_id", "split", "reason")


def build_modeling_cohort(labels: pd.DataFrame) -> pd.DataFrame:
    """Include records with at least one target superclass."""
    validated = _validate_label_table(labels)
    has_target = validated.loc[:, TARGET_SUPERCLASSES].sum(axis=1).ge(1)
    return validated.loc[has_target, COHORT_COLUMNS].reset_index(drop=True)


def find_cohort_exclusions(labels: pd.DataFrame) -> pd.DataFrame:
    """Return excluded records and their explicit task-level reason."""
    validated = _validate_label_table(labels)
    has_no_target = validated.loc[:, TARGET_SUPERCLASSES].sum(axis=1).eq(0)
    exclusions = validated.loc[has_no_target, ["ecg_id", "patient_id", "split"]].copy()
    exclusions["reason"] = EXCLUSION_REASON
    return exclusions.loc[:, EXCLUSION_COLUMNS].reset_index(drop=True)


def build_cohort_summary(
    full_labels: pd.DataFrame,
    cohort: pd.DataFrame,
) -> dict[str, Any]:
    """Build deterministic evidence for the cohort boundary."""
    full = _validate_label_table(full_labels)
    included = _validate_label_table(cohort)
    expected = build_modeling_cohort(full)

    if set(included["ecg_id"]) != set(expected["ecg_id"]):
        raise ValueError("Cohort ecg_id values do not match the inclusion rule")

    full_by_id = full.set_index("ecg_id").loc[:, COHORT_COLUMNS[1:]].sort_index()
    included_by_id = (
        included.set_index("ecg_id").loc[:, COHORT_COLUMNS[1:]].sort_index()
    )
    expected_values = full_by_id.loc[included_by_id.index]
    if not included_by_id.equals(expected_values):
        raise ValueError("Cohort changed identity, split, fold, or target values")

    exclusions = find_cohort_exclusions(full)
    overlap = _find_patient_overlap(included)
    detected_overlap = {pair: values for pair, values in overlap.items() if values}
    if detected_overlap:
        raise ValueError(
            f"Patient overlap detected after cohort filtering: {detected_overlap}"
        )

    included_patients = set(included["patient_id"])
    excluded_patients = set(exclusions["patient_id"])
    split_summary: dict[str, dict[str, int]] = {}
    prevalence: dict[str, dict[str, dict[str, float | int]]] = {}
    cardinality: dict[str, float] = {}
    combinations: dict[str, dict[str, int]] = {}
    label_count_distribution: dict[str, dict[str, int]] = {}

    for split in ALLOWED_SPLITS:
        split_labels = included.loc[included["split"] == split]
        split_summary[split] = {
            "records": len(split_labels),
            "patients": int(split_labels["patient_id"].nunique()),
        }
        prevalence[split] = {}
        for label in TARGET_SUPERCLASSES:
            count = int(split_labels[label].sum())
            prevalence[split][label] = {
                "count": count,
                "fraction": count / len(split_labels) if len(split_labels) else 0.0,
            }
        row_cardinality = split_labels.loc[:, TARGET_SUPERCLASSES].sum(axis=1)
        cardinality[split] = float(row_cardinality.mean()) if len(split_labels) else 0.0
        combinations[split] = _count_label_combinations(split_labels)
        label_count_distribution[split] = _count_active_labels(split_labels)

    return {
        "dataset": {"name": "PTB-XL", "version": "1.0.3"},
        "task": {
            "type": "multilabel_classification",
            "targets": list(TARGET_SUPERCLASSES),
            "inclusion_rule": "at_least_one_target_superclass",
        },
        "records": {
            "input": len(full),
            "included": len(included),
            "excluded": len(exclusions),
        },
        "patients": {
            "input": int(full["patient_id"].nunique()),
            "included": len(included_patients),
            "excluded": len(excluded_patients),
            "excluded_only": len(excluded_patients - included_patients),
        },
        "exclusion_reasons": {EXCLUSION_REASON: len(exclusions)},
        "splits": split_summary,
        "label_prevalence": prevalence,
        "label_cardinality": cardinality,
        "label_count_distribution": label_count_distribution,
        "label_combinations": combinations,
        "patient_overlap": overlap,
    }


def _validate_label_table(labels: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in COHORT_COLUMNS if column not in labels]
    if missing:
        raise KeyError(f"Missing cohort input columns: {missing}")
    validated = labels.loc[:, COHORT_COLUMNS].copy()
    if validated["ecg_id"].isna().any():
        raise ValueError("Column 'ecg_id' contains null values")
    if validated["ecg_id"].duplicated().any():
        raise ValueError("Cohort input contains duplicate ecg_id values")
    if validated["split"].isna().any():
        raise ValueError("Column 'split' contains null values")
    invalid_splits = sorted(set(validated["split"]) - set(ALLOWED_SPLITS))
    if invalid_splits:
        raise ValueError(f"Invalid inherited split values: {invalid_splits}")
    if not validated.loc[:, TARGET_SUPERCLASSES].isin([0, 1]).all().all():
        raise ValueError("Target labels must contain only 0 or 1")
    return validated


def _find_patient_overlap(labels: pd.DataFrame) -> dict[str, list[Any]]:
    patients = {
        split: set(labels.loc[labels["split"] == split, "patient_id"])
        for split in ALLOWED_SPLITS
    }
    return {
        "train_validation": sorted(patients["train"] & patients["validation"]),
        "train_test": sorted(patients["train"] & patients["test"]),
        "validation_test": sorted(patients["validation"] & patients["test"]),
    }


def _count_label_combinations(labels: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in labels.loc[:, TARGET_SUPERCLASSES].itertuples(index=False, name=None):
        active = [label for label, value in zip(TARGET_SUPERCLASSES, row) if value]
        counts["+".join(active)] += 1
    return dict(sorted(counts.items()))


def _count_active_labels(labels: pd.DataFrame) -> dict[str, int]:
    counts = labels.loc[:, TARGET_SUPERCLASSES].sum(axis=1).value_counts().sort_index()
    return {
        str(label_count): int(counts.get(label_count, 0)) for label_count in range(1, 6)
    }
