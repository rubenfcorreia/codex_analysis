from pathlib import Path

from analysis.cache_builders.common import load_json_manifest, save_json_manifest, stage_manifest, load_indexed_records, save_indexed_records
from analysis.shared.cache_config import normalize_staged_config


def test_nested_config_preserves_legacy_keys_and_stage_flags():
    config = normalize_staged_config(
        {
            "inputs": {"movie_expids": ["movie-1"]},
            "paths": {"cache_root": "results/cache"},
            "cache": {"rebuild_stages": ["classification"], "workers": 3},
            "classification": {"visual_response_metric": "mean"},
        },
        pipeline="dendrites",
    )
    assert config["movie_expids"] == ["movie-1"]
    assert config["visual_response_metric"] == "mean"
    assert config["classification_cache_rebuild"] is True
    assert config["cache_workers"] == 3
    assert config["cache_paths"]["source"] == "results/cache/source"
    assert config["cache_paths"]["for_analysis"] == "results/cache/for_analysis"


def test_manifest_round_trip_is_atomic_and_versioned(tmp_path: Path):
    path = tmp_path / "cache" / "manifest.json"
    payload = stage_manifest(
        pipeline="soma_bouton",
        stage="source",
        config_signature="config",
        active_expids=["b", "a", "a"],
    )
    save_json_manifest(path, payload)
    loaded = load_json_manifest(path)
    assert loaded is not None
    assert loaded["schema_version"] == 3
    assert loaded["active_expids"] == ["a", "b"]



def test_analysis_family_spec_declares_scope_and_dependencies():
    from analysis.shared.analysis_families import family_spec, register_family_spec

    spec = register_family_spec(
        "test_family",
        scope="experiment_day",
        input_tables=("activity_rows",),
        config_keys=("event_detection_method",),
        cache_stage="results",
    )
    assert spec == family_spec("test_family")
    assert spec.input_tables == ("activity_rows",)
    assert spec.config_keys == ("event_detection_method",)


def test_pipeline_configs_load_external_family_and_plot_registries():
    import json
    for path, pipeline, expected in (
        (Path("analysis/dendrites_pipeline/pipeline_config.json"), "dendrites", "spine_coactivity"),
        (Path("analysis/soma_bouton_pipeline/pipeline_config.json"), "soma_bouton", "visual_response"),
    ):
        config = normalize_staged_config(json.loads(path.read_text()), pipeline=pipeline, base_dir=path.parent)
        assert expected in config["analysis_families"]
        assert config["visual_response_metric"] == "mean"
        assert config["analysis_families_file"].endswith("analysis_families.json")
        assert config["plot_config_file"].endswith("plot_config.json")


def test_indexed_records_reuse_by_exp_id_and_day(tmp_path: Path):
    from analysis.cache_builders.common import save_indexed_records

    records = {
        "e1|day1": [{"expid": "e1", "day_id": "day1", "value": 1}],
        "e2|day2": [{"expid": "e2", "day_id": "day2", "value": 2}],
    }
    first = save_indexed_records(
        tmp_path / "processed",
        pipeline="dendrites",
        stage="processed",
        records_by_scope=records,
        config_signature="config",
        upstream_signatures={"e1|day1": "s1", "e2|day2": "s2"},
    )
    assert first["records"]["built_count"] == 2
    second = save_indexed_records(
        tmp_path / "processed",
        pipeline="dendrites",
        stage="processed",
        records_by_scope=records,
        config_signature="config",
        upstream_signatures={"e1|day1": "s1", "e2|day2": "s2"},
    )
    assert second["records"]["reused_count"] == 2
    changed = dict(records)
    changed["e2|day2"] = [{"expid": "e2", "day_id": "day2", "value": 20}]
    third = save_indexed_records(
        tmp_path / "processed",
        pipeline="dendrites",
        stage="processed",
        records_by_scope=changed,
        config_signature="config",
        upstream_signatures={"e1|day1": "s1", "e2|day2": "s2"},
    )
    assert third["records"]["reused_count"] == 1
    assert third["records"]["built_count"] == 1



def test_indexed_records_reject_payload_signature_mismatch(tmp_path: Path):
    from analysis.shared.cache_utils import load_npz_cache, save_npz_cache

    root = tmp_path / "processed"
    save_indexed_records(
        root,
        pipeline="dendrites",
        stage="processed",
        records_by_scope={"e1|d1": [{"expid": "e1", "day_id": "d1", "value": 1}]},
        config_signature="config",
        upstream_signatures={"e1|d1": "source"},
    )
    path = next((root / "records").glob("*.npz"))
    payload = load_npz_cache(path)
    payload["signature"] = "tampered"
    save_npz_cache(path, payload)
    assert load_indexed_records(root, "e1|d1") is None


def test_family_cache_reuse_skips_family_execution(tmp_path: Path):
    from analysis.dendrites_pipeline.analysis_families.core import _reuse_family_cache
    from analysis.shared.cache_utils import save_family_results_cache

    cache_path = tmp_path / "analysis_run.npz"
    meta = {"source_signature": "source-1", "analysis_families": ["state"], "analysis_family_settings": {"state": {"enabled": True}}}
    save_family_results_cache(cache_path, "state", {"state_rows": [{"expid": "e1", "day_id": "d1"}]}, base_meta=meta)
    results = {}
    assert _reuse_family_cache(results, "state", cache_path, meta)
    assert results["state_rows"] == [{"expid": "e1", "day_id": "d1"}]
    assert results["cache_summary"]["state_family_cache_reused"] is True


def test_processed_cache_warm_run_processes_only_new_expid(tmp_path: Path, monkeypatch):
    import json
    from analysis.cache_builders import build_processed_cache as module
    from analysis.cache_builders.build_processed_cache import build_processed_cache
    from analysis.shared.cache_utils import save_npz_cache

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"pipeline": {"name": "dendrites"}, "result_root": str(tmp_path / "results")}))
    source_path = tmp_path / "source.npz"
    output_path = tmp_path / "processed" / "processed.npz"
    calls = []

    def fake_records(source, config=None):
        ids = sorted(source.get("experiments", {}))
        calls.append(ids)
        return [{"roi_key": f"roi-{expid}", "expid": expid, "day_id": expid, "state_activity": {"wake": 1.0}, "state_frequency": {"wake": 2.0}, "source_signature": source["experiments"][expid]["source_signature"]} for expid in ids]

    monkeypatch.setattr(module, "_dendrite_records", fake_records)
    first_source = {"experiments": {"e1": {"source_signature": "s1"}, "e2": {"source_signature": "s2"}}, "animals": {}}
    save_npz_cache(source_path, first_source)
    build_processed_cache(config_path, "dendrites", source_path, output_path)
    second_source = {"experiments": {**first_source["experiments"], "e3": {"source_signature": "s3"}}, "animals": {}}
    save_npz_cache(source_path, second_source)
    build_processed_cache(config_path, "dendrites", source_path, output_path)
    assert calls == [["e1", "e2"], ["e3"]]


def test_family_scope_reuse_report_lists_reused_and_recomputed_scopes():
    from analysis.shared.cache_utils import family_scope_reuse_report

    report = family_scope_reuse_report("correlation", {"e1|d1": "a", "e2|d2": "b"}, {"e1|d1": []})
    assert report["family"] == "correlation"
    assert report["requested_scope_count"] == 2
    assert report["reused_scope_count"] == 1
    assert report["recomputed_scope_count"] == 1
    assert report["recomputed_scopes"] == ["e2|d2"]


def test_family_scope_cache_reuses_only_matching_scope_signatures(tmp_path: Path):
    from analysis.shared.cache_utils import load_reusable_family_scope_rows, save_family_results_cache

    cache_path = tmp_path / "analysis_run.npz"
    save_family_results_cache(
        cache_path,
        "visual_response",
        {"rows": [
            {"expid": "e1", "day_id": "d1", "value": 1},
            {"expid": "e2", "day_id": "d2", "value": 2},
        ]},
        base_meta={"scope_signatures": {"e1|d1": "sig-1", "e2|d2": "sig-2"}},
    )
    reusable = load_reusable_family_scope_rows(
        cache_path, "visual_response", {"e1|d1": "sig-1", "e2|d2": "changed"}
    )
    assert list(reusable) == ["e1|d1"]
    assert reusable["e1|d1"][0]["value"] == 1


def test_family_scope_cache_rejects_changed_family_configuration(tmp_path: Path):
    from analysis.shared.cache_utils import load_reusable_family_scope_rows, save_family_results_cache

    cache_path = tmp_path / "analysis_run.npz"
    meta = {
        "source_signature": "source-1",
        "analysis_families": ["visual_response"],
        "analysis_family_settings": {"visual_response": {"metric": "peak"}},
        "scope_signatures": {"e1|d1": "sig-1"},
    }
    save_family_results_cache(
        cache_path,
        "visual_response",
        {"rows": [{"expid": "e1", "day_id": "d1", "value": 1}]},
        base_meta=meta,
    )
    changed_meta = {**meta, "analysis_family_settings": {"visual_response": {"metric": "auc"}}}
    reusable = load_reusable_family_scope_rows(
        cache_path,
        "visual_response",
        {"e1|d1": "sig-1"},
        expected_family_meta=changed_meta,
    )
    assert reusable == {}


def test_family_registry_validates_scope_and_aggregation(tmp_path: Path):
    import json
    family_path = tmp_path / "families.json"
    family_path.write_text(json.dumps({"families": {"state": {"enabled": True, "scope": "invalid"}}}))
    config = {"analysis": {"families_file": str(family_path)}, "paths": {"cache_root": str(tmp_path / "cache")}}
    import pytest
    with pytest.raises(ValueError, match="Unsupported cache scope"):
        normalize_staged_config(config, pipeline="dendrites", base_dir=tmp_path)


def test_family_cache_indexes_nested_table_rows(tmp_path: Path):
    from analysis.shared.cache_utils import load_reusable_family_scope_rows, save_family_results_cache

    cache_path = tmp_path / "analysis_run.npz"
    save_family_results_cache(
        cache_path,
        "spine_coactivity",
        {"spine_coactivity": {"table_rows": [{"expid": "e1", "day_id": "d1", "value": 3}]}},
        base_meta={"scope_signatures": {"e1|d1": "sig-1"}},
    )
    reusable = load_reusable_family_scope_rows(cache_path, "spine_coactivity", {"e1|d1": "sig-1"})
    assert reusable["e1|d1"][0]["value"] == 3


def test_relative_external_registries_fallback_to_canonical_pipeline_files(tmp_path: Path):
    import json
    copied_config = tmp_path / "pipeline_config.json"
    copied_config.write_text(json.dumps({
        "pipeline": {"name": "dendrites"},
        "analysis": {"families_file": "analysis_families.json"},
        "plots": {"config_file": "plot_config.json"},
    }))
    config = normalize_staged_config(json.loads(copied_config.read_text()), pipeline="dendrites", base_dir=tmp_path)
    assert config["analysis_families_file"].endswith("analysis/dendrites_pipeline/analysis_families.json")
    assert config["plot_config_file"].endswith("analysis/dendrites_pipeline/plot_config.json")
