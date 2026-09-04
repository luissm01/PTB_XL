"""Reproducible single-record inference with the frozen PTB-XL baseline."""

import hashlib
import json
import math
import platform
import re
import struct
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ptbxl.data import (
    TARGET_SUPERCLASSES,
    ECGSignal,
    load_wfdb_record,
    validate_signal,
)
from ptbxl.data.reporting import compute_sha256, write_json_report
from ptbxl.evaluation import FrozenThresholds, load_frozen_thresholds
from ptbxl.experiments.frozen import (
    FrozenBaseline,
    load_frozen_baseline,
    validate_frozen_baseline_hashes,
    validate_frozen_threshold_binding,
    validate_loaded_checkpoint,
)
from ptbxl.models import SmallECGCNN
from ptbxl.preprocessing import GlobalStandardizer, load_global_standardizer
from ptbxl.training import (
    configure_deterministic_execution,
    load_model_checkpoint,
    resolve_device,
    seed_random_generators,
)


INFERENCE_CONFIG_SCHEMA_VERSION = 1
INFERENCE_REPORT_SCHEMA_VERSION = 1
INFERENCE_MODE = "single_record"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
CONFIG_TABLE_FIELDS = {
    "inference": {"name", "mode", "device"},
    "inputs": {
        "experiment_config_path",
        "experiment_config_sha256",
        "experiment_report_path",
        "experiment_report_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "standardizer_sha256",
        "threshold_artifact_path",
        "threshold_artifact_sha256",
    },
}
SOURCE_NAMES = (
    "inference_config",
    "experiment_config",
    "experiment_report",
    "checkpoint",
    "standardizer",
    "threshold_artifact",
)


@dataclass(frozen=True)
class InferenceConfig:
    """Exact frozen artifacts and runtime choice for single-record inference."""

    name: str
    mode: str
    device: str
    experiment_config_path: Path
    experiment_config_sha256: str
    experiment_report_path: Path
    experiment_report_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    standardizer_sha256: str
    threshold_artifact_path: Path
    threshold_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", self.name
        ):
            raise ValueError(
                "Inference name must use lowercase letters, numbers, _ or -"
            )
        if self.mode != INFERENCE_MODE:
            raise ValueError(f"Inference mode must be {INFERENCE_MODE!r}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Inference device must be 'auto', 'cpu', or 'cuda'")
        suffixes = {
            "experiment_config_path": ".toml",
            "experiment_report_path": ".json",
            "checkpoint_path": ".pt",
            "threshold_artifact_path": ".json",
        }
        for field, suffix in suffixes.items():
            value = getattr(self, field)
            if not isinstance(value, Path):
                raise TypeError(f"{field} must be a pathlib.Path")
            if value.suffix != suffix:
                raise ValueError(f"{field} must end in {suffix}")
        for field in (
            "experiment_config_sha256",
            "experiment_report_sha256",
            "checkpoint_sha256",
            "standardizer_sha256",
            "threshold_artifact_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True)
class LabelPrediction:
    """One frozen operating decision and its score."""

    label: str
    probability: float
    threshold: float
    predicted: bool


@dataclass(frozen=True)
class InferenceResult:
    """Prediction result for one validated standalone ECG."""

    record_id: str
    record_path: Path
    signal_sha256: str
    sampling_frequency_hz: float
    samples: int
    lead_names: tuple[str, ...]
    predictions: tuple[LabelPrediction, ...]


@dataclass
class FrozenInferenceBundle:
    """Loaded model and immutable preprocessing/decision artifacts."""

    baseline: FrozenBaseline
    thresholds: FrozenThresholds
    standardizer: GlobalStandardizer
    model: SmallECGCNN
    device: torch.device
    threshold_artifact_sha256: str


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load one strict versioned inference configuration."""
    config_path = Path(path)
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    _require_exact_fields(
        "configuration",
        raw,
        {"schema_version", *CONFIG_TABLE_FIELDS},
    )
    if raw["schema_version"] != INFERENCE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Inference config schema_version must be {INFERENCE_CONFIG_SCHEMA_VERSION}"
        )
    tables = {
        name: _require_mapping(raw, name, fields)
        for name, fields in CONFIG_TABLE_FIELDS.items()
    }
    inference = tables["inference"]
    inputs = tables["inputs"]
    return InferenceConfig(
        name=inference["name"],
        mode=inference["mode"],
        device=inference["device"],
        experiment_config_path=_config_path(
            inputs["experiment_config_path"], "experiment_config_path"
        ),
        experiment_config_sha256=inputs["experiment_config_sha256"],
        experiment_report_path=_config_path(
            inputs["experiment_report_path"], "experiment_report_path"
        ),
        experiment_report_sha256=inputs["experiment_report_sha256"],
        checkpoint_path=_config_path(inputs["checkpoint_path"], "checkpoint_path"),
        checkpoint_sha256=inputs["checkpoint_sha256"],
        standardizer_sha256=inputs["standardizer_sha256"],
        threshold_artifact_path=_config_path(
            inputs["threshold_artifact_path"], "threshold_artifact_path"
        ),
        threshold_artifact_sha256=inputs["threshold_artifact_sha256"],
    )


def load_frozen_inference_bundle(config: InferenceConfig) -> FrozenInferenceBundle:
    """Validate and restore the exact frozen model bundle without signal access."""
    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")
    baseline = load_frozen_baseline(
        config.experiment_config_path,
        config.experiment_report_path,
        config.checkpoint_path,
        verify_dataset_sources=False,
    )
    validate_frozen_baseline_hashes(
        baseline,
        experiment_config_sha256=config.experiment_config_sha256,
        experiment_report_sha256=config.experiment_report_sha256,
        checkpoint_sha256=config.checkpoint_sha256,
        standardizer_sha256=config.standardizer_sha256,
    )
    threshold_artifact_sha256 = compute_sha256(config.threshold_artifact_path)
    if threshold_artifact_sha256 != config.threshold_artifact_sha256:
        raise ValueError("Threshold artifact SHA-256 does not match inference config")
    thresholds = load_frozen_thresholds(
        config.threshold_artifact_path,
        expected_checkpoint_sha256=baseline.hashes.checkpoint,
    )
    validate_frozen_threshold_binding(baseline, thresholds)

    configure_deterministic_execution()
    seed_random_generators(baseline.config.seed)
    device = resolve_device(config.device)
    standardizer = load_global_standardizer(baseline.config.standardizer_path)
    model = SmallECGCNN(baseline.config.model_config)
    loaded = load_model_checkpoint(
        config.checkpoint_path,
        model,
        device=device,
        expected_provenance=baseline.provenance,
    )
    validate_loaded_checkpoint(loaded, baseline)
    model.eval()
    return FrozenInferenceBundle(
        baseline=baseline,
        thresholds=thresholds,
        standardizer=standardizer,
        model=model,
        device=device,
        threshold_artifact_sha256=threshold_artifact_sha256,
    )


def predict_ecg_record(
    bundle: FrozenInferenceBundle,
    record_path: str | Path,
    *,
    record_id: str | None = None,
) -> InferenceResult:
    """Load and predict one standalone compatible WFDB record."""
    if not isinstance(bundle, FrozenInferenceBundle):
        raise TypeError("bundle must be a FrozenInferenceBundle")
    path = _record_path(record_path)
    identifier = path.name if record_id is None else record_id
    _validate_record_id(identifier)
    record = load_wfdb_record(path)
    validate_signal(record, bundle.standardizer.lead_names)
    signal_sha256 = fingerprint_ecg_signal(record)
    standardized = bundle.standardizer.transform(record.signal, record.lead_names)
    channel_first = np.ascontiguousarray(standardized.T)
    batch = torch.from_numpy(channel_first).unsqueeze(0).to(bundle.device)
    with torch.inference_mode():
        logits = bundle.model(batch)
        if logits.shape != (1, len(TARGET_SUPERCLASSES)):
            raise ValueError("Inference model output must have shape (1, 5)")
        if not torch.isfinite(logits).all():
            raise ValueError("Inference model logits must be finite")
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]
    if not np.isfinite(probabilities).all():
        raise ValueError("Inference probabilities must be finite")
    thresholds = bundle.thresholds.thresholds.values
    predictions = tuple(
        LabelPrediction(
            label=label,
            probability=float(probabilities[column]),
            threshold=thresholds[column],
            predicted=bool(probabilities[column] >= thresholds[column]),
        )
        for column, label in enumerate(TARGET_SUPERCLASSES)
    )
    return InferenceResult(
        record_id=identifier,
        record_path=path,
        signal_sha256=signal_sha256,
        sampling_frequency_hz=record.sampling_frequency,
        samples=record.signal.shape[0],
        lead_names=record.lead_names,
        predictions=predictions,
    )


def run_single_record_inference(
    config: InferenceConfig,
    config_path: str | Path,
    record_path: str | Path,
    output_path: str | Path,
    git_commit: str,
    *,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Run one attributable inference and write a non-overwriting JSON report."""
    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")
    if not isinstance(git_commit, str) or not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise ValueError("git_commit must be a lowercase hexadecimal commit")
    attributed_config_path = Path(config_path)
    if load_inference_config(attributed_config_path) != config:
        raise ValueError("config does not match the attributed config_path")
    destination = Path(output_path)
    if destination.suffix != ".json":
        raise ValueError("Inference output path must end in .json")
    if destination.exists():
        raise FileExistsError(f"Inference output already exists: {destination}")

    bundle = load_frozen_inference_bundle(config)
    result = predict_ecg_record(bundle, record_path, record_id=record_id)
    report = _build_inference_report(
        config,
        config_path=attributed_config_path,
        git_commit=git_commit,
        bundle=bundle,
        result=result,
    )
    write_json_report(report, destination)
    loaded = load_inference_report(
        destination,
        expected_checkpoint_sha256=config.checkpoint_sha256,
    )
    if loaded != report:
        raise ValueError("Saved inference report failed round-trip validation")
    return report


def fingerprint_ecg_signal(record: ECGSignal) -> str:
    """Hash calibrated values and technical header facts canonically."""
    if not isinstance(record, ECGSignal):
        raise TypeError("record must be an ECGSignal")
    validate_signal(record)
    digest = hashlib.sha256()
    digest.update(b"ptbxl-compatible-ecg-signal-v1\0")
    digest.update(
        struct.pack(
            "<QQd",
            record.signal.shape[0],
            record.signal.shape[1],
            record.sampling_frequency,
        )
    )
    for lead_name in record.lead_names:
        encoded = lead_name.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    digest.update(np.asarray(record.signal, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def load_inference_report(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate one deterministic inference report."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Could not load inference report") from error
    if not isinstance(raw, dict):
        raise TypeError("Inference report must be a JSON object")
    _require_exact_fields(
        "inference report",
        raw,
        {
            "schema_version",
            "inference",
            "input",
            "sources",
            "runtime",
            "predictions",
            "limitations",
        },
    )
    if raw["schema_version"] != INFERENCE_REPORT_SCHEMA_VERSION:
        raise ValueError("Inference report schema_version is invalid")
    inference = _require_mapping(
        raw,
        "inference",
        {"name", "mode", "git_commit"},
    )
    if inference["mode"] != INFERENCE_MODE:
        raise ValueError("Inference report mode is invalid")
    if not isinstance(inference["name"], str) or not inference["name"].strip():
        raise ValueError("Inference report name must be non-empty")
    if not isinstance(inference["git_commit"], str) or not GIT_COMMIT_PATTERN.fullmatch(
        inference["git_commit"]
    ):
        raise ValueError("Inference report git_commit is invalid")
    input_record = _require_mapping(
        raw,
        "input",
        {
            "record_id",
            "record_path",
            "signal_sha256",
            "sampling_frequency_hz",
            "samples",
            "lead_names",
        },
    )
    _validate_record_id(input_record["record_id"])
    if (
        not isinstance(input_record["record_path"], str)
        or not input_record["record_path"]
    ):
        raise ValueError("Inference report record_path must be non-empty")
    _validate_sha256(input_record["signal_sha256"], "input.signal_sha256")
    if input_record["sampling_frequency_hz"] != 100.0:
        raise ValueError("Inference report sampling frequency is invalid")
    if input_record["samples"] != 1_000:
        raise ValueError("Inference report sample count is invalid")
    lead_names = input_record["lead_names"]
    if (
        not isinstance(lead_names, list)
        or len(lead_names) != 12
        or len(set(lead_names)) != 12
        or any(not isinstance(name, str) or not name for name in lead_names)
    ):
        raise ValueError("Inference report lead names are invalid")
    sources = _require_mapping(raw, "sources", set(SOURCE_NAMES))
    for source_name in SOURCE_NAMES:
        source = _require_mapping(
            sources,
            source_name,
            {"path", "sha256"},
        )
        if not isinstance(source["path"], str) or not source["path"]:
            raise ValueError(f"Inference report {source_name} path is invalid")
        _validate_sha256(source["sha256"], f"sources.{source_name}.sha256")
    if expected_checkpoint_sha256 is not None:
        _validate_sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
        if sources["checkpoint"]["sha256"] != expected_checkpoint_sha256:
            raise ValueError("Inference report checkpoint SHA-256 does not match")
    runtime = _require_mapping(
        raw,
        "runtime",
        {
            "python",
            "numpy",
            "torch",
            "cuda",
            "hardware",
            "deterministic_algorithms",
        },
    )
    if not isinstance(runtime["hardware"], dict):
        raise TypeError("Inference report runtime.hardware must be an object")
    if not isinstance(runtime["deterministic_algorithms"], bool):
        raise TypeError("Inference report deterministic_algorithms must be boolean")
    predictions = raw["predictions"]
    if not isinstance(predictions, list) or len(predictions) != len(
        TARGET_SUPERCLASSES
    ):
        raise ValueError("Inference report must contain five predictions")
    for expected_label, prediction in zip(
        TARGET_SUPERCLASSES, predictions, strict=True
    ):
        if not isinstance(prediction, Mapping):
            raise TypeError("Inference report predictions must be objects")
        _require_exact_fields(
            "prediction",
            prediction,
            {"label", "probability", "threshold", "predicted"},
        )
        if prediction["label"] != expected_label:
            raise ValueError("Inference report labels are not in canonical order")
        probability = _unit_float(prediction["probability"], "probability")
        threshold = _unit_float(prediction["threshold"], "threshold")
        if not isinstance(prediction["predicted"], bool):
            raise TypeError("Inference report predicted values must be boolean")
        if prediction["predicted"] != (probability >= threshold):
            raise ValueError("Inference report decision is inconsistent with threshold")
    if not isinstance(raw["limitations"], list) or not all(
        isinstance(item, str) and item.strip() for item in raw["limitations"]
    ):
        raise ValueError("Inference report limitations must be non-empty strings")
    return raw


def _build_inference_report(
    config: InferenceConfig,
    *,
    config_path: Path,
    git_commit: str,
    bundle: FrozenInferenceBundle,
    result: InferenceResult,
) -> dict[str, Any]:
    baseline = bundle.baseline
    return {
        "schema_version": INFERENCE_REPORT_SCHEMA_VERSION,
        "inference": {
            "name": config.name,
            "mode": config.mode,
            "git_commit": git_commit,
        },
        "input": {
            "record_id": result.record_id,
            "record_path": result.record_path.as_posix(),
            "signal_sha256": result.signal_sha256,
            "sampling_frequency_hz": result.sampling_frequency_hz,
            "samples": result.samples,
            "lead_names": list(result.lead_names),
        },
        "sources": {
            "inference_config": {
                "path": config_path.as_posix(),
                "sha256": compute_sha256(config_path),
            },
            "experiment_config": {
                "path": config.experiment_config_path.as_posix(),
                "sha256": baseline.hashes.experiment_config,
            },
            "experiment_report": {
                "path": config.experiment_report_path.as_posix(),
                "sha256": baseline.hashes.experiment_report,
            },
            "checkpoint": {
                "path": config.checkpoint_path.as_posix(),
                "sha256": baseline.hashes.checkpoint,
            },
            "standardizer": {
                "path": baseline.config.standardizer_path.as_posix(),
                "sha256": baseline.hashes.standardizer,
            },
            "threshold_artifact": {
                "path": config.threshold_artifact_path.as_posix(),
                "sha256": bundle.threshold_artifact_sha256,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hardware": {
                "requested_device": config.device,
                "resolved_device": bundle.device.type,
                **(
                    {"accelerator_name": torch.cuda.get_device_name(bundle.device)}
                    if bundle.device.type == "cuda"
                    else {}
                ),
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "predictions": [
            {
                "label": item.label,
                "probability": item.probability,
                "threshold": item.threshold,
                "predicted": item.predicted,
            }
            for item in result.predictions
        ],
        "limitations": [
            "This experimental output is not a clinical diagnosis.",
            "Sigmoid scores have not been calibrated as clinical probabilities.",
            "Compatibility checks do not establish acquisition-domain equivalence.",
            "The model has only internal PTB-XL evaluation, not external validation.",
        ],
    }


def _record_path(value: str | Path) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ValueError("record_path must be non-empty")
    try:
        path = Path(value)
    except TypeError as error:
        raise TypeError("record_path must be a string or pathlib.Path") from error
    if path.suffix:
        raise ValueError("record_path must be a WFDB basename without an extension")
    return path


def _validate_record_id(value: Any) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("record_id must be a non-empty printable string")


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Inference report {name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"Inference report {name} must be finite and in [0, 1]")
    return numeric


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_mapping(
    values: Mapping[str, Any],
    name: str,
    expected_fields: set[str],
) -> Mapping[str, Any]:
    value = values[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _require_exact_fields(name, value, expected_fields)
    return value


def _require_exact_fields(
    name: str,
    values: Mapping[str, Any],
    expected_fields: set[str],
) -> None:
    actual = set(values)
    if missing := sorted(expected_fields - actual):
        raise KeyError(f"{name} is missing fields: {missing}")
    if unexpected := sorted(actual - expected_fields):
        raise ValueError(f"{name} has unexpected fields: {unexpected}")


def _config_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty path string")
    return Path(value)
