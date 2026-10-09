from pathlib import Path

from analysis.compartment_common import write_csv_rows
from analysis.shared.analysis_families.mixed_model import run_mixed_model_family
from analysis.shared.serialization_diagnostics import SerializationCollector, set_active_collector


def _rows(states=("quiet", "run"), animals=("a1", "a2")):
    rows = []
    for animal in animals:
        for state_index, state in enumerate(states):
            for unit_index in range(2):
                rows.append({
                    "animal_id": animal,
                    "day_id": f"{animal}_day",
                    "unit_id": f"{animal}_unit_{unit_index}",
                    "state": state,
                    "mean_activity": float(state_index + unit_index + (0.1 if animal == "a2" else 0.0)),
                    "event_frequency_per_min": float(state_index + unit_index + 1),
                })
    return rows


def test_mixed_model_preflight_skips_single_group():
    result = run_mixed_model_family(
        _rows(animals=("a1",)),
        response="mean_activity",
        scope="selected_state",
        contrast_specs=[{"kind": "state_pair", "state_a": "quiet", "state_b": "run"}],
        shuffle_n=0,
        state_order=["quiet", "run"],
    )
    preflight = result["design"]["preflight"]
    assert preflight["status"] == "insufficient_groups"
    assert result["fit"]["fit_method"] == "preflight_skipped"


def test_mixed_model_preflight_preserves_valid_fit_metadata():
    result = run_mixed_model_family(
        _rows(),
        response="mean_activity",
        scope="selected_state",
        contrast_specs=[{"kind": "state_pair", "state_a": "quiet", "state_b": "run"}],
        shuffle_n=0,
        state_order=["quiet", "run"],
    )
    assert result["design"]["preflight"]["status"] in {"eligible", "near_singular_warning"}
    assert "preflight_status" in result["fit"]


def test_csv_serialization_is_atomic_and_reported(tmp_path: Path):
    collector = SerializationCollector()
    set_active_collector(collector)
    try:
        output = tmp_path / "csv" / "table.csv"
        write_csv_rows(output, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], ["a", "b"])
    finally:
        set_active_collector(None)
    assert output.read_text().splitlines() == ["a,b", "1,x", "2,y"]
    assert len(collector.records) == 1
    assert collector.records[0]["format"] == "csv"
    assert collector.records[0]["rows"] == 2
    assert collector.records[0]["columns"] == 2
    assert collector.records[0]["output_bytes"] == output.stat().st_size
