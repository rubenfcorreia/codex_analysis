from pathlib import Path

from analysis.shared.union_rows import (
    deduplicate_rows,
    filter_rows_by_states,
    filter_table_rows_by_states,
    load_union_rows_cache,
    save_union_rows_cache,
    union_rows_meta,
    union_state_labels,
)


def test_union_state_labels_are_sorted_and_deduplicated():
    assert union_state_labels([
        {"movie": ["quiet", "active"]},
        {"movie": ["quiet", "rem"], "sleep": ["nrem"]},
    ]) == {"movie": ["active", "quiet", "rem"], "sleep": ["nrem"]}


def test_filter_preserves_invariant_rows_and_filters_modes():
    rows = [
        {"id": "q", "state": "quiet", "mode": "movie"},
        {"id": "r", "state": "rem", "mode": "sleep"},
        {"id": "invariant", "value": 1},
        {"id": "unknown", "state": "quiet", "mode": "other"},
    ]
    filtered = filter_rows_by_states(rows, {"movie": ["quiet"], "sleep": ["nrem"]})
    assert [row["id"] for row in filtered] == ["q", "invariant", "unknown"]


def test_table_filter_deduplicates_without_changing_rows():
    rows = [{"id": 1, "state": "quiet"}, {"id": 1, "state": "quiet"}, {"id": 2, "value": 3}]
    assert deduplicate_rows(rows) == [{"id": 1, "state": "quiet"}, {"id": 2, "value": 3}]
    tables = filter_table_rows_by_states({"activity": {"table_rows": rows}}, {"movie": ["quiet"]})
    assert tables["activity"]["table_rows"] == [{"id": 1, "state": "quiet"}, {"id": 2, "value": 3}]


def test_union_cache_roundtrip_and_metadata_invalidation(tmp_path: Path):
    path = tmp_path / "union.npz"
    meta = union_rows_meta(
        source_signature="source-a",
        union_states_by_mode={"movie": ["quiet"]},
        parameters={
            "channel": 1,
            "event_detection_method": "amplitude",
            "visual_response_metric": "calcium_events",
        },
    )
    save_union_rows_cache(path, {"dendrites_analysis_table_rows": {"activity": [{"id": 1}]}}, meta=meta)
    payload, status, _ = load_union_rows_cache(path, expected_meta=meta)
    assert status == "reused"
    assert payload["dendrites_analysis_table_rows"]["activity"] == [{"id": 1}]
    changed = dict(meta)
    changed["parameters"] = dict(meta["parameters"], channel=2)
    _, status, _ = load_union_rows_cache(path, expected_meta=changed)
    assert status == "invalid"


def test_missing_union_cache_status(tmp_path: Path):
    _, status, _ = load_union_rows_cache(
        tmp_path / "missing.npz",
        expected_meta=union_rows_meta(
            source_signature="s",
            union_states_by_mode={},
            parameters={},
        ),
    )
    assert status == "missing"


def test_soma_preset_orchestration_shares_union_cache(monkeypatch, tmp_path):
    import analysis.soma_bouton_pipeline.soma_bouton_pipeline as pipeline

    captured = []
    monkeypatch.setattr(pipeline, "resolve_analysis_state_selections", lambda config, mode: list(config.get("state_comparison_states", [])))
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config: captured.append(dict(config)) or {"ok": True})
    pipeline.run_comparison_preset_runs({
        "result_root": str(tmp_path / "results"),
        "cache_root": str(tmp_path / "cache"),
        "movie_expids": ["movie-1"],
        "comparison_presets": {
            "first": {"state_comparison_states": ["quiet"]},
            "second": {"state_comparison_states": ["active"]},
        },
    })
    analysis_runs = [entry for entry in captured if not entry.get("plots_only")]
    assert len(analysis_runs) == 2
    assert analysis_runs[0]["union_rows_cache_builder"] is True
    assert analysis_runs[1]["union_rows_cache_builder"] is False
    assert analysis_runs[0]["shared_union_rows_cache_path"] == analysis_runs[1]["shared_union_rows_cache_path"]
    assert analysis_runs[0]["union_state_labels_by_mode"]["movie"] == ["active", "quiet"]
    assert analysis_runs[0]["generate_shared_general_outputs"] is True
    assert analysis_runs[1]["generate_shared_general_outputs"] is False


def test_dendrites_preset_orchestration_shares_union_cache(monkeypatch, tmp_path):
    import json
    import subprocess
    import analysis.dendrites_pipeline.dendrites_pipeline as pipeline

    captured = []
    def fake_run(command, check):
        config_path = Path(command[-1])
        captured.append(json.loads(config_path.read_text()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline.run_comparison_preset_subprocesses({
        "output_dir": str(tmp_path / "results"),
        "cache_path": str(tmp_path / "cache.npz"),
        "movie_expids": ["movie-1"],
        "comparison_presets": {
            "first": {"state_comparison_states": ["quiet_awake"]},
            "second": {"state_comparison_states": ["active_awake"]},
        },
    })
    analysis_runs = [entry for entry in captured if not entry.get("plots_only")]
    assert len(analysis_runs) == 2
    assert analysis_runs[0]["union_rows_cache_builder"] is True
    assert analysis_runs[1]["union_rows_cache_builder"] is False
    assert analysis_runs[0]["shared_union_rows_cache_path"] == analysis_runs[1]["shared_union_rows_cache_path"]
    assert analysis_runs[0]["union_state_labels_by_mode"]["state_comparison"] == ["active_awake", "quiet_awake"]
    assert analysis_runs[0]["figure_output_dir"].endswith("/first/pooled/all/figures")
    assert analysis_runs[1]["figure_output_dir"].endswith("/second/pooled/all/figures")
    assert analysis_runs[0]["generate_shared_general_outputs"] is True
    assert analysis_runs[1]["generate_shared_general_outputs"] is False


def test_cache_entrypoint_preset_orchestration_uses_branch_first_figures(monkeypatch, tmp_path):
    import analysis.soma_bouton_pipeline_cache as pipeline

    captured = []
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config: captured.append(dict(config)) or {"ok": True})
    pipeline.run_comparison_preset_runs({
        "result_root": str(tmp_path / "results"),
        "cache_root": str(tmp_path / "cache"),
        "comparison_presets": {"first": {}, "second": {}},
    })

    assert [entry["branch_first_figures"] for entry in captured] == [True, True]
    assert [Path(entry["result_root"]).name for entry in captured] == ["first", "second"]


def test_comparison_leaf_root_is_pooled_all(tmp_path):
    from analysis.shared.branch_tree import comparison_leaf_root

    assert comparison_leaf_root(tmp_path / "preset") == tmp_path / "preset" / "pooled" / "all"
