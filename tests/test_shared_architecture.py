from __future__ import annotations

import ast
from pathlib import Path

from analysis.shared.analysis_families import require_split_groups, scope_split_rows
from analysis.shared.analysis_families.common_helpers import build_state_masks_movie, classify_movie_name, movie_trial_type_suffix
from analysis.shared.figure_paths import (
    comparison_leaf_figure_root,
    shared_figure_root,
    standalone_figure_root,
)
from analysis.shared.result_manifest import write_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_shared_figure_roots_are_canonical() -> None:
    assert comparison_leaf_figure_root({"branch_first_output_root": "/tmp/preset"}, branch="activity_split", basis="nrem") == Path("/tmp/preset/activity_split/nrem")
    assert shared_figure_root("/tmp/pipeline") == Path("/tmp/pipeline/general/figures")
    assert standalone_figure_root("/tmp/run") == Path("/tmp/run/figures")


def test_split_rows_are_scoped_and_require_two_groups() -> None:
    rows = [
        {"state": "nrem", "split_group": "more_active", "analysis_branch_name": "activity_split", "analysis_basis_name": "nrem"},
        {"state": "nrem", "split_group": "less_active", "analysis_branch_name": "activity_split", "analysis_basis_name": "nrem"},
        {"state": "rem", "split_group": "wrong_state", "analysis_branch_name": "activity_split", "analysis_basis_name": "rem"},
    ]
    scoped = scope_split_rows(rows, branch="activity_split", basis="nrem", selected_states=["nrem"])
    assert {row["split_group"] for row in scoped} == {"more_active", "less_active"}
    assert require_split_groups(scoped, branch="activity_split", basis="nrem") == ("more_active", "less_active")


def test_movie_masks_accept_time_duration_and_f1_type() -> None:
    import numpy as np

    masks, metadata, _ = build_state_masks_movie(
        np.arange(0.0, 40.0, 1.0),
        [{"time": "10", "duration": "5", "F1_type": "movie"}],
        ["time", "duration", "F1_type"],
        None,
        None,
        {"state_10hz_t": np.array([0.0, 40.0]), "state_10hz": np.array([1.0, 1.0])},
        0.0,
    )
    assert metadata[0]["state_label"] == "nrem_movies"
    assert masks["nrem_movies"].sum() == 6


def test_split_rows_normalize_roi_membership_group() -> None:
    rows = [
        {"state": "nrem", "group": "more_active", "branch_name": "activity_split", "basis_name": "nrem"},
        {"state": "nrem", "group": "less_active", "branch_name": "activity_split", "basis_name": "nrem"},
    ]
    scoped = scope_split_rows(rows, branch="activity_split", basis="nrem", selected_states=["nrem"])
    assert {row["split_group"] for row in scoped} == {"more_active", "less_active"}
    assert require_split_groups(scoped, branch="activity_split", basis="nrem") == ("more_active", "less_active")


def test_split_rows_reject_missing_group() -> None:
    try:
        require_split_groups([{"split_group": "only"}], branch="frequency_split", basis="nrem")
    except ValueError as exc:
        assert "at least 2 split groups" in str(exc)
    else:
        raise AssertionError("one split group must fail")


def test_manifest_write_is_atomic_and_rejects_missing_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "leaf"
    artifact = root / "state" / "figure.svg"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("<svg />")
    path = write_manifest(root, {"output_root": str(root), "output_artifacts": ["state/figure.svg"]})
    assert path.exists()
    try:
        write_manifest(root, {"output_root": str(root), "output_artifacts": ["missing.svg"]})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing artifact must prevent manifest publication")


def test_soma_plot_module_is_compatibility_only() -> None:
    tree = ast.parse((ROOT / "analysis/soma_bouton_pipeline/plots.py").read_text())
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savefig" for node in ast.walk(tree))


def test_soma_analysis_family_modules_are_compatibility_only() -> None:
    for name in ("core.py", "state.py", "correlation.py", "lag.py", "transitions.py"):
        tree = ast.parse((ROOT / "analysis/soma_bouton_pipeline/analysis_families" / name).read_text())
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body if getattr(node, "name", "") != "__all__")


def test_active_pipeline_modules_do_not_save_figures_directly() -> None:
    for directory in (ROOT / "analysis/dendrites_pipeline", ROOT / "analysis/soma_bouton_pipeline"):
        for path in directory.rglob("*.py"):
            if "/shared/" in str(path):
                continue
            assert "savefig(" not in path.read_text(), path


def test_shared_layer_does_not_import_pipeline_modules() -> None:
    for path in (ROOT / "analysis/shared").rglob("*.py"):
        text = path.read_text()
        assert "analysis.dendrites_pipeline" not in text, path
        assert "analysis.soma_bouton_pipeline" not in text, path


def test_analysis_family_dispatch_preserves_scope() -> None:
    from analysis.shared.analysis_families import AnalysisScope, run_analysis_family
    scope = AnalysisScope("dendrites", "preset", "activity_split", "nrem", ("nrem",), "activity")
    result = run_analysis_family("calcium_events", [], scope, {})
    assert result.family == "calcium_events"
    assert result.scope == scope
    assert result.metadata["contract_version"]


def test_movie_name_categories_and_suffixes() -> None:
    assert classify_movie_name(r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00000") == "blank"
    assert classify_movie_name(r"D:\bonsai_resources\all_movie_clips_bv_sets\007\01007") == "grating"
    assert classify_movie_name(r"D:\bonsai_resources\all_movie_clips_bv_sets\007\02001") == "zebra"
    assert classify_movie_name(r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00303") == "movies"
    assert movie_trial_type_suffix("grating") == "gratings"
    assert movie_trial_type_suffix("zebra") == "zebras"


def test_movie_masks_assign_wheel_based_quiet_and_active_labels() -> None:
    import numpy as np

    rows = [
        {"time": 0, "duration": 4, "F1_type": "movie", "F1_name": r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00000"},
        {"time": 5, "duration": 4, "F1_type": "movie", "F1_name": r"D:\bonsai_resources\all_movie_clips_bv_sets\007\02001"},
    ]
    masks, metadata, _ = build_state_masks_movie(
        np.arange(0.0, 10.0),
        rows,
        list(rows[0]),
        np.arange(0.0, 10.0),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 2.0, 0.1]),
        None,
        1.0,
    )
    assert metadata[0]["state_label"] == "quiet_awake_blank"
    assert metadata[1]["state_label"] == "active_awake_zebras"
    assert masks["quiet_awake_blank"].sum() == 5
    assert masks["active_awake_zebras"].sum() == 5
