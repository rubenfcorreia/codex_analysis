from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.dendrites_pipeline import dendrites_pipeline as pipeline
from analysis.shared.plots import poster_ready


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
    day_id = "animal_1|2024-06-01|basal"
    matrix_path = pipeline.build_matrix_similarity_day_figure_path(tmp_path, "animal_1", day_id, "basal", "d1")
    coactivity_path = pipeline.build_spine_coactivity_day_figure_path(tmp_path, "animal_1", day_id, "basal", "d1")

    assert matrix_path.parts[-5:-1] == ("matrix_similarity", "animal_1", "basal", "2024-06-01")
    assert coactivity_path.parts[-6:-1] == ("spine_coactivity", "pair_state_heatmap", "animal_1", "basal", "2024-06-01")


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
    assert not (output_dir / "analysis_results.json").exists()
    assert not (output_dir / "state_comparisons.csv").exists()
    assert not (output_dir / "analysis_report.txt").exists()


def test_mixed_model_row_selector_prefers_mean_activity() -> None:
    branch = {
        "summary_rows": {
            "mean_dendrite_activity": [{"term": "state[nrem]", "estimate": 2.0}],
            "mean_activity": [{"term": "state[nrem]", "estimate": 1.0}],
            "event_frequency_per_min": [{"term": "state[nrem]", "estimate": 3.0}],
        }
    }

    rows = poster_ready._select_mixed_model_rows(branch)

    assert [row["estimate"] for row in rows] == [1.0]


class _FakeAxes:
    def __init__(self) -> None:
        self.boxplot_positions = None
        self.boxplot_data = None
        self.xticks = None
        self.xticklabels = None
        self.yticks = None
        self.yticklabels = None
        self.xlim = (0.0, 0.0)
        self.ylim = (0.0, 0.0)
        self.transAxes = object()

    def boxplot(self, data, positions, **kwargs):
        self.boxplot_positions = list(positions)
        self.boxplot_data = [list(series) for series in data]
        return {"boxes": [], "whiskers": [], "caps": [], "medians": []}

    def scatter(self, *args, **kwargs):
        return None

    def set_xlim(self, left, right):
        self.xlim = (float(left), float(right))

    def get_xlim(self):
        return self.xlim

    def set_ylim(self, bottom, top):
        self.ylim = (float(bottom), float(top))

    def get_ylim(self):
        return self.ylim

    def set_xticks(self, ticks):
        self.xticks = [float(tick) for tick in ticks]

    def set_yticks(self, ticks):
        self.yticks = [float(tick) for tick in ticks]

    def set_xticklabels(self, labels, **kwargs):
        self.xticklabels = [str(label) for label in labels]

    def set_yticklabels(self, labels, **kwargs):
        self.yticklabels = [str(label) for label in labels]

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def text(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None


def test_boxplot_keeps_empty_state_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAxes()
    state_order = [
        "quiet_awake_blank",
        "nrem_blank",
        "rem_blank",
        "quiet_awake_movies",
    ]

    monkeypatch.setattr(poster_ready, "_style_axes", lambda ax: None)

    poster_ready._boxplot(
        fake,
        {"quiet_awake_blank": [1.0, 2.0], "rem_blank": [3.0]},
        state_order,
        title="Mixed model",
        ylabel="Mean response",
        significance_flags=[False, False, False, False],
        sample_sizes={"quiet_awake_blank": 2, "nrem_blank": 0, "rem_blank": 1, "quiet_awake_movies": 0},
    )

    assert fake.boxplot_positions == [1.0, 3.0]
    assert fake.xticks == [1.0, 2.0, 3.0, 4.0]
    assert fake.xticklabels == [poster_ready._state_display_label(state) for state in state_order]
    assert fake.xlim[0] == 0.5
    assert fake.xlim[1] >= 4.5


def test_state_mixed_model_poster_figure_keeps_full_state_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[tuple[tuple[str, ...], str | None]] = []

    def _capture_boxplot(ax, state_values, state_order, **kwargs):
        captured.append((tuple(state_order), str(kwargs.get("cohort_label"))))

    monkeypatch.setattr(poster_ready, "_boxplot", _capture_boxplot)
    monkeypatch.setattr(poster_ready, "_forest_panel", lambda *args, **kwargs: None)

    state_order = [
        "quiet_awake_blank",
        "nrem_blank",
        "rem_blank",
        "quiet_awake_movies",
        "nrem_movies",
        "rem_movies",
        "quiet_awake",
        "nrem",
        "rem",
    ]
    state_values = {"nrem": [1.0, 2.0], "rem": [3.0]}
    mixed_rows = {
        "responsive": {
            "summary_rows": {
                "mean_activity": [
                    {"term": "state[quiet_awake_blank]", "estimate": 0.1, "p_value": 0.01},
                    {"term": "state[nrem]", "estimate": 0.2, "p_value": 0.2},
                ]
            }
        },
        "nonresponsive": {
            "summary_rows": {
                "mean_activity": [
                    {"term": "state[quiet_awake_blank]", "estimate": 0.3, "p_value": 0.01},
                    {"term": "state[nrem]", "estimate": 0.4, "p_value": 0.2},
                ]
            }
        },
    }

    output = poster_ready.write_state_mixed_model_poster_figure(
        output_dir=tmp_path,
        entity_label="soma",
        responsive_state_values=state_values,
        nonresponsive_state_values=state_values,
        mixed_model_rows=mixed_rows,
        state_order=state_order,
        preferred_response_keys=("mean_activity", "mean"),
        mixed_model_contrast_p_source="classical",
    )

    assert output is not None
    assert [order for order, _ in captured] == [tuple(state_order), tuple(state_order)]
    assert {cohort for _, cohort in captured} == {"responsive", "nonresponsive"}
