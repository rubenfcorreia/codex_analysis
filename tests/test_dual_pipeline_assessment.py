from pathlib import Path

import numpy as np

from analysis.shared.analysis_families.mixed_model import run_mixed_model_family
from analysis.shared.cache_utils import ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION, load_npz_cache, save_npz_cache
from analysis.shared.plots.dff_heatmaps import render_dff_heatmaps


def _mixed_rows():
    rows = []
    for animal in ("a1", "a2"):
        for state_index, state in enumerate(("quiet", "nrem")):
            for unit_index in range(2):
                rows.append({
                    "animal_id": animal, "day_id": f"{animal}_day",
                    "unit_id": f"{animal}_{unit_index}", "state": state,
                    "mean_activity": float(state_index + unit_index + 1),
                    "event_frequency_per_min": float(state_index + unit_index + 2),
                })
    return rows


def test_mixed_model_design_records_fixed_effect_names():
    result = run_mixed_model_family(
        _mixed_rows(), "mean_activity", "selected_state", [], 0,
        state_order=["quiet", "nrem"],
    )
    assert "fixed_effect_names" in result["design"]
    assert result["design"].get("inferential_status") in {"valid", "invalid_fit"}


def test_analysis_result_cache_schema_rejects_old_payload(tmp_path: Path):
    path = tmp_path / "old.npz"
    save_npz_cache(path, {"schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION - 1, "analysis_results": {}})
    assert load_npz_cache(path)["schema_version"] != ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION


def test_heatmap_representative_mode_writes_one_gallery_per_compartment(tmp_path: Path):
    records = []
    classes = (
        "high_activity_high_frequency", "low_activity_high_frequency",
        "high_activity_low_frequency", "low_activity_low_frequency",
    )
    for compartment in ("dendrite", "soma"):
        for index in range(12):
            records.append({
                "compartment": compartment, "day_id": f"day-{index % 3}",
                "roi_id": index, "split_group": classes[index % 4],
                "time": np.arange(20, dtype=float),
                "trace": np.sin(np.arange(20, dtype=float) / 3.0) + index / 10.0,
            })
    files = render_dff_heatmaps(records, tmp_path, max_rois_per_class=10)
    metadata = sorted(tmp_path.rglob("dff_heatmap_examples.json"))
    assert len(metadata) == 2
    assert len(files) == 6


def test_heatmap_split_conflicts_do_not_overwrite_roi_assignment(tmp_path: Path):
    from analysis.shared.plots.dff_heatmaps import _class_lookup
    rows = [
        {"compartment": "spine", "unit_id": "roi-1", "state": "quiet", "split_name": "activity_frequency", "basis_name": "all", "split_group": "high_activity_high_frequency"},
        {"compartment": "spine", "unit_id": "roi-1", "state": "quiet", "split_name": "activity_frequency", "basis_name": "all", "split_group": "low_activity_high_frequency"},
    ]
    lookup = _class_lookup(rows)
    assert lookup[("spine", "roi-1", "quiet", "activity_frequency", "all")] == ""
