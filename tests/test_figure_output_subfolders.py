from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.main_pipeline import sleep_dendrite_spine_pipeline as pipeline


def _write_dummy_svg(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<svg><!-- {label} --></svg>")
    return path


def _fake_plotter(label: str):
    def _plotter(*args, **kwargs):
        if len(args) >= 2:
            fig_dir = Path(args[1])
        else:
            fig_dir = Path(kwargs["fig_dir"])
        output_name = str(kwargs.get("output_name", f"{label}.svg"))
        return str(_write_dummy_svg(fig_dir / output_name, label))

    return _plotter


@pytest.mark.parametrize(
    ("family", "results", "expected_tokens"),
    [
        (
            "correlation",
            {},
            [("correlation_summary",)],
        ),
        (
            "direct_trial_type_comparison",
            {
                "direct_trial_type_comparison": {
                    "video_state_rows": [{"state": "quiet_awake_movies", "mean_response": 1.0}],
                    "state_pair_rows": [
                        {
                            "state_a": "quiet_awake_movies",
                            "state_b": "quiet_awake_blank",
                            "shuffle_p": 0.01,
                        }
                    ],
                }
            },
            [("direct_trial_type_comparison",)],
        ),
        (
            "matrix_similarity",
            {
                "matrix_similarity": [
                    {"compartment": "basal", "animal_id": "animal_1", "exp_id": "day_1", "global_dendrite_id": "d1"},
                    {"compartment": "apical", "animal_id": "animal_1", "exp_id": "day_2", "global_dendrite_id": "d2"},
                ]
            },
            [("matrix_similarity",), ("basal",), ("apical",)],
        ),
        (
            "spine_coactivity",
            {
                "spine_coactivity": {
                    "table_rows": [
                        {
                            "compartment": "basal",
                            "animal_id": "animal_1",
                            "day_id": "day_1",
                            "exp_id": "day_1",
                            "global_dendrite_id": "d1",
                            "global_pair_id": "p1",
                            "state": "quiet_awake_movies",
                            "status": "ok",
                            "coactive": True,
                        },
                        {
                            "compartment": "apical",
                            "animal_id": "animal_1",
                            "day_id": "day_2",
                            "exp_id": "day_2",
                            "global_dendrite_id": "d2",
                            "global_pair_id": "p2",
                            "state": "quiet_awake_blank",
                            "status": "ok",
                            "coactive": True,
                        },
                    ],
                    "pair_state_rows": [],
                }
            },
            [("spine_coactivity",), ("basal",), ("apical",)],
        ),
        (
            "mixed_model",
            {
                "mixed_model": {
                    "designs": {"response": {}},
                    "summary_rows": {"response": [1]},
                    "contrast_rows": [{"response": "response"}],
                },
                "mixed_model_selected_state": {
                    "designs": {"response": {}},
                    "summary_rows": {"response": [1]},
                    "contrast_rows": [{"response": "response"}],
                },
            },
            [("mixed_model",), ("all_state",), ("selected_state",)],
        ),
    ],
)
def test_render_analysis_family_figures_use_nested_family_folders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    family: str,
    results: dict,
    expected_tokens: list[tuple[str, ...]],
) -> None:
    cache = {"animals": {}, "experiments": {}, "config": {}, "alerts": []}
    figure_root = tmp_path / "figures"

    fake = _fake_plotter(family)
    for name in [
        "plot_correlation_summary",
        "plot_matrix_similarity_distribution",
        "plot_matrix_similarity_heatmap",
        "plot_spine_coactivity_distribution_figure",
        "plot_spine_coactivity_tendency_figure",
        "plot_spine_coactivity_pair_state_heatmap_figure",
        "plot_spine_coactivity_pair_state_summary_figure",
        "plot_spine_coactivity_basal_apical_distribution_figure",
        "plot_direct_trial_type_distribution_figure",
        "plot_direct_trial_type_state_comparison_figure",
        "plot_mixed_model_forest_figure",
        "plot_mixed_model_predicted_means_figure",
        "plot_mixed_model_contrasts_checkpoint",
        "plot_demo_validation_figure",
    ]:
        monkeypatch.setattr(pipeline, name, fake)

    saved = pipeline.render_analysis_family_figures(tmp_path / "analysis", results, cache, family, figure_root=figure_root)
    assert saved
    saved_paths = [Path(path) for path in saved]
    for token_set in expected_tokens:
        assert any(all(token in path.parts for token in token_set) for path in saved_paths)

    if family == "mixed_model":
        assert any("mixed_model" in path.parts for path in saved_paths)
        assert any("all_state" in path.parts for path in saved_paths)
        assert any("selected_state" in path.parts for path in saved_paths)
    elif family in {"matrix_similarity", "spine_coactivity"}:
        assert any("basal" in path.parts for path in saved_paths)
        assert any("apical" in path.parts for path in saved_paths)


def test_render_analysis_family_figures_skips_empty_inputs_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _none_plotter(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "plot_correlation_summary", _none_plotter)
    monkeypatch.setattr(pipeline, "plot_spine_coactivity_basal_apical_distribution_figure", _none_plotter)
    saved = pipeline.render_analysis_family_figures(
        tmp_path / "analysis",
        {},
        {"animals": {}, "experiments": {}, "config": {}, "alerts": []},
        "correlation",
        figure_root=tmp_path / "figures",
    )
    assert saved == []


def test_day_figure_helpers_use_family_subfolders(tmp_path: Path) -> None:
    matrix_path = pipeline.build_matrix_similarity_day_figure_path(tmp_path, "animal_1", "2024-06-01", "basal", "d1")
    coactivity_path = pipeline.build_spine_coactivity_day_figure_path(tmp_path, "animal_1", "2024-06-01", "basal", "d1")

    assert matrix_path.parts[-5:-1] == ("matrix_similarity", "animal_1", "basal", "2024-06-01")
    assert coactivity_path.parts[-5:-1] == ("spine_coactivity", "animal_1", "basal", "2024-06-01")


def test_write_analysis_outputs_plots_only_skips_nonfigure_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    figure_root = tmp_path / "figures"
    output_dir = tmp_path / "results"
    cache = {"animals": {}, "experiments": {}, "config": {}, "alerts": []}
    results = {"state_comparisons": [{"state": "quiet_awake_movies"}]}

    def _write_dummy(path: Path, label: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label)
        return str(path)

    monkeypatch.setattr(pipeline, "generate_analysis_figures", lambda *args, **kwargs: [_write_dummy(figure_root / "analysis" / "figure.svg", "analysis")])
    monkeypatch.setattr(pipeline, "generate_review_figures", lambda *args, **kwargs: [_write_dummy(figure_root / "review" / "figure.svg", "review")])
    monkeypatch.setattr(pipeline, "generate_checkpoint_gallery", lambda *args, **kwargs: {"manifest_path": str(_write_dummy(output_dir / "checkpoint" / "manifest.json", "checkpoint")), "entries": [], "files": [str(_write_dummy(output_dir / "checkpoint" / "figure.svg", "checkpoint"))]})
    monkeypatch.setattr(pipeline, "generate_event_detection_example_gallery", lambda *args, **kwargs: [str(_write_dummy(figure_root / "event_examples" / "figure.svg", "event"))])
    monkeypatch.setattr(pipeline, "write_text_report", lambda *args, **kwargs: pytest.fail("report should not be written in plots_only mode"))
    monkeypatch.setattr(pipeline, "write_csv_rows", lambda *args, **kwargs: pytest.fail("csv should not be written in plots_only mode"))
    monkeypatch.setattr(pipeline, "save_npz_cache", lambda *args, **kwargs: pytest.fail("cache should not be saved in plots_only mode"))
    monkeypatch.setattr(pipeline, "save_analysis_tables_cache", lambda *args, **kwargs: pytest.fail("analysis-tables cache should not be saved in plots_only mode"))
    monkeypatch.setattr(pipeline, "save_analysis_results_cache", lambda *args, **kwargs: pytest.fail("analysis-results cache should not be saved in plots_only mode"))

    written = pipeline.write_analysis_outputs(output_dir, results, cache, figure_root=figure_root, plots_only=True)

    assert written
    assert (figure_root / "analysis" / "figure.svg").exists()
    assert (figure_root / "review" / "figure.svg").exists()
    assert (output_dir / "checkpoint" / "figure.svg").exists()
    assert (figure_root / "event_examples" / "figure.svg").exists()
    assert not (output_dir / "analysis_results.json").exists()
    assert not (output_dir / "state_comparisons.csv").exists()
    assert not (output_dir / "analysis_report.txt").exists()
