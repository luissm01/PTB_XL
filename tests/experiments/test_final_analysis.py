from pathlib import Path

import pytest

from ptbxl.experiments import load_final_analysis_config, run_final_analysis


SHA = "a" * 64


def _write_config(
    path: Path,
    *,
    report_path: Path,
    figure_directory: Path,
    extra: str = "",
) -> None:
    path.write_text(
        f"""schema_version = 1

[analysis]
name = "test_final_analysis"
mode = "posthoc_descriptive_analysis"
minimum_combination_support = 2
maximum_problematic_combinations = 3
attribution_label = "HYP"
attribution_kind = "false_negative"
attribution_window_samples = 50
prediction_tolerance = 0.00001
{extra}
[inputs]
inference_config_path = "inference.toml"
inference_config_sha256 = "{SHA}"
final_test_report_path = "final.json"
final_test_report_sha256 = "{SHA}"
prediction_artifact_path = "predictions.npz"
prediction_artifact_sha256 = "{SHA}"
metadata_path = "metadata.csv"
metadata_sha256 = "{SHA}"

[outputs]
report_path = "{report_path.as_posix()}"
figure_directory = "{figure_directory.as_posix()}"
""",
        encoding="utf-8",
    )


def test_load_final_analysis_config_is_strict(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.toml"
    report_path = tmp_path / "analysis.json"
    figure_directory = tmp_path / "figures"
    _write_config(
        config_path,
        report_path=report_path,
        figure_directory=figure_directory,
    )

    config = load_final_analysis_config(config_path)

    assert config.name == "test_final_analysis"
    assert config.attribution_label == "HYP"
    assert config.minimum_combination_support == 2
    assert config.report_path == report_path
    assert config.figure_directory == figure_directory


def test_load_final_analysis_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.toml"
    _write_config(
        config_path,
        report_path=tmp_path / "analysis.json",
        figure_directory=tmp_path / "figures",
        extra="unexpected = true\n",
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        load_final_analysis_config(config_path)


def test_run_refuses_existing_output_before_reading_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.toml"
    report_path = tmp_path / "analysis.json"
    figure_directory = tmp_path / "figures"
    _write_config(
        config_path,
        report_path=report_path,
        figure_directory=figure_directory,
    )
    report_path.write_text("reserved", encoding="utf-8")
    config = load_final_analysis_config(config_path)

    with pytest.raises(FileExistsError, match="will not overwrite"):
        run_final_analysis(config, config_path, "abcdef0")


def test_run_rejects_hash_drift_before_loading_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("analysis.toml")
    _write_config(
        config_path,
        report_path=tmp_path / "analysis.json",
        figure_directory=tmp_path / "figures",
    )
    for name in ("inference.toml", "final.json", "predictions.npz", "metadata.csv"):
        Path(name).write_text("not the declared content", encoding="utf-8")
    config = load_final_analysis_config(config_path)

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        run_final_analysis(config, config_path, "abcdef0")
