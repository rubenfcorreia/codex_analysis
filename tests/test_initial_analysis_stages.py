from pathlib import Path

import json

from analysis.cache_builders.build_classification_cache import build_classification_cache
from analysis.cache_builders.build_initial_analysis import build_initial_analysis
from analysis.cache_builders.initial_analysis_plots import write_initial_analysis_figures
from analysis.shared.cache_utils import load_npz_cache, save_npz_cache
from analysis.shared.classification_summary import apply_summary


def test_initial_analysis_preserves_preferred_state_ties_and_general_figures(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"pipeline": {"name": "dendrites"}, "result_root": str(tmp_path / "results"), "initial_analysis": {"state_tie_relative_tolerance": 0.02}}))
    processed = tmp_path / "processed.npz"
    save_npz_cache(processed, {"roi_records": [{"roi_key": "d1", "expid": "e1", "state_activity": {"wake": 2.0, "nrem": 1.99}, "state_frequency": {"wake": 4.0, "nrem": 2.0}, "visual_responsive": True}]})
    initial = tmp_path / "initial.npz"
    build_initial_analysis(config, "dendrites", processed, initial)
    record = load_npz_cache(initial)["records"][0]
    assert record["preferred_states"] == ["nrem", "wake"]
    assert record["preferred_state_class"] == "tie"
    assert (tmp_path / "results" / "figures" / "general" / "initial_analysis" / "preferred_state_counts.png").exists()
    assert (tmp_path / "results" / "figures" / "general" / "initial_analysis" / "state_frequency_heatmap.png").exists()


def test_classification_thresholds_are_reused(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"pipeline": {"name": "dendrites"}}))
    initial = tmp_path / "initial.npz"
    save_npz_cache(initial, {"records": [{"roi_key": "a", "expid": "e1", "activity_mean": 1.0, "frequency_mean": 1.0}, {"roi_key": "b", "expid": "e2", "activity_mean": 3.0, "frequency_mean": 3.0}]})
    output = tmp_path / "classification.npz"
    build_classification_cache(config, "dendrites", initial, output)
    first = load_npz_cache(output)
    save_npz_cache(initial, {"records": [{"roi_key": "a", "expid": "e1", "activity_mean": 100.0, "frequency_mean": 100.0}, {"roi_key": "b", "expid": "e2", "activity_mean": 3.0, "frequency_mean": 3.0}]})
    build_classification_cache(config, "dendrites", initial, output)
    assert load_npz_cache(output)["thresholds"] == first["thresholds"]


def test_family_summary_uses_staged_classification():
    summary = {"rows": [{"global_dendrite_id": "d1", "compartment": "basal", "responsive": False}]}
    updated = apply_summary(summary, [{"roi_key": "d1", "visual_class": "visual"}])
    assert updated["rows"][0]["cohort"] == "responsive"
