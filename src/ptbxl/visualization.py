"""Static, non-interactive figures for the final PTB-XL portfolio report."""

import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ptbxl.data import TARGET_SUPERCLASSES
from ptbxl.data.reporting import compute_sha256
from ptbxl.evaluation import ErrorAnalysis, PredictionSet, SelectedPredictionExample
from ptbxl.interpretability import InputAttribution


FINAL_FIGURE_FILENAMES = (
    "training_history.png",
    "final_roc_pr_curves.png",
    "final_operating_errors.png",
    "hyp_false_negative_saliency.png",
)


def render_final_figures(
    destination: str | Path,
    *,
    experiment_report: Mapping[str, Any],
    predictions: PredictionSet,
    errors: ErrorAnalysis,
    selected: SelectedPredictionExample,
    raw_signal: np.ndarray,
    attribution: InputAttribution,
    lead_names: tuple[str, ...],
) -> dict[str, str]:
    """Atomically render the four fixed final figures and return their hashes."""
    output_directory = Path(destination)
    if output_directory.exists():
        raise FileExistsError(f"Figure directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=output_directory.parent,
            prefix=f".{output_directory.name}.",
        )
    )
    try:
        _plot_training_history(experiment_report, temporary / FINAL_FIGURE_FILENAMES[0])
        _plot_roc_pr(predictions, temporary / FINAL_FIGURE_FILENAMES[1])
        _plot_operating_errors(errors, temporary / FINAL_FIGURE_FILENAMES[2])
        _plot_saliency(
            raw_signal,
            attribution,
            selected,
            lead_names,
            temporary / FINAL_FIGURE_FILENAMES[3],
        )
        hashes = {
            name: compute_sha256(temporary / name) for name in FINAL_FIGURE_FILENAMES
        }
        os.replace(temporary, output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return hashes


def _plot_training_history(report: Mapping[str, Any], path: Path) -> None:
    from matplotlib.figure import Figure

    history = report["training"]["history"]
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train"]["loss"] for item in history]
    validation_loss = [item["validation"]["loss"] for item in history]
    best_epoch = report["training"]["best_epoch"]
    figure = Figure(figsize=(8, 4.8), layout="constrained")
    axis = figure.subplots()
    axis.plot(epochs, train_loss, marker="o", label="Train BCE loss")
    axis.plot(epochs, validation_loss, marker="o", label="Validation BCE loss")
    axis.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        alpha=0.65,
        label=f"Selected epoch {best_epoch}",
    )
    axis.set(
        title="Baseline training history",
        xlabel="Epoch",
        ylabel="Sample-weighted loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def _plot_roc_pr(predictions: PredictionSet, path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 5.2), layout="constrained")
    roc_axis, pr_axis = figure.subplots(1, 2)
    for column, label in enumerate(TARGET_SUPERCLASSES):
        targets = predictions.targets[:, column]
        scores = predictions.probabilities[:, column]
        false_positive_rate, true_positive_rate, _ = roc_curve(targets, scores)
        precision, recall, _ = precision_recall_curve(targets, scores)
        auroc = roc_auc_score(targets, scores)
        average_precision = average_precision_score(targets, scores)
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} ({auroc:.3f})",
        )
        pr_axis.plot(recall, precision, label=f"{label} ({average_precision:.3f})")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    roc_axis.set(
        title="Final test ROC curves",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    pr_axis.set(
        title="Final test precision-recall curves",
        xlabel="Recall",
        ylabel="Precision",
    )
    for axis in (roc_axis, pr_axis):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend(title="Class (area)", fontsize=8)
    _save_figure(figure, path)


def _plot_operating_errors(errors: ErrorAnalysis, path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 5.2), layout="constrained")
    count_axis, combination_axis = figure.subplots(1, 2)
    x = np.arange(len(TARGET_SUPERCLASSES))
    false_positives = [item.false_positives for item in errors.per_class]
    false_negatives = [item.false_negatives for item in errors.per_class]
    width = 0.38
    count_axis.bar(x - width / 2, false_positives, width, label="False positives")
    count_axis.bar(x + width / 2, false_negatives, width, label="False negatives")
    count_axis.set(
        title="Errors at frozen thresholds",
        ylabel="ECGs",
        xticks=x,
        xticklabels=TARGET_SUPERCLASSES,
    )
    count_axis.grid(axis="y", alpha=0.2)
    count_axis.legend()

    combinations = list(reversed(errors.problematic_combinations))
    names = ["+".join(item.labels) or "none" for item in combinations]
    rates = [item.exact_match_rate for item in combinations]
    bars = combination_axis.barh(names, rates)
    combination_axis.set(
        title=(
            "Lowest exact-match combinations\n"
            f"(support ≥ {errors.minimum_combination_support})"
        ),
        xlabel="Exact-match rate",
        xlim=(0, 1),
    )
    combination_axis.grid(axis="x", alpha=0.2)
    for bar, item in zip(bars, combinations, strict=True):
        combination_axis.text(
            min(bar.get_width() + 0.02, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"n={item.support}",
            va="center",
            fontsize=8,
        )
    _save_figure(figure, path)


def _plot_saliency(
    raw_signal: np.ndarray,
    attribution: InputAttribution,
    selected: SelectedPredictionExample,
    lead_names: tuple[str, ...],
    path: Path,
) -> None:
    from matplotlib.figure import Figure

    time = np.arange(raw_signal.shape[0]) / 100.0
    figure = Figure(figsize=(13, 14), layout="constrained")
    axes = figure.subplots(12, 1, sharex=True)
    for column, (axis, lead) in enumerate(zip(axes, lead_names, strict=True)):
        values = raw_signal[:, column]
        relevance = attribution.values[:, column]
        maximum = float(relevance.max())
        normalized = relevance / maximum if maximum > 0 else relevance
        lower = float(values.min())
        upper = float(values.max())
        if math.isclose(lower, upper):
            lower -= 1.0
            upper += 1.0
        axis.imshow(
            normalized[np.newaxis, :],
            extent=(0, 10, lower, upper),
            origin="lower",
            aspect="auto",
            cmap="Reds",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )
        axis.plot(time, values, color="black", linewidth=0.65)
        axis.set_ylabel(lead, rotation=0, labelpad=18, va="center")
        axis.grid(alpha=0.12)
    axes[-1].set_xlabel("Time (seconds)")
    target_labels = [
        label
        for label, active in zip(TARGET_SUPERCLASSES, selected.targets, strict=True)
        if active
    ]
    predicted_labels = [
        label
        for label, active in zip(TARGET_SUPERCLASSES, selected.decisions, strict=True)
        if active
    ]
    figure.suptitle(
        f"ECG {selected.ecg_id}: {selected.label} "
        f"{selected.kind.replace('_', ' ')}\n"
        f"targets={'+'.join(target_labels)} | "
        f"predictions={'+'.join(predicted_labels) or 'none'} | "
        "red = stronger local model attribution",
        fontsize=12,
    )
    _save_figure(figure, path)


def _save_figure(figure: Any, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "ptbxl-ml-system"},
    )
