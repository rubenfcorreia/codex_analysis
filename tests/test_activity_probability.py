from analysis.shared.analysis_families.activity_probability import run_family


def test_activity_probability_bounds_and_roi_pairing():
    rows = [
        {"day_id": "d1", "mode": "sleep", "state": "nrem", "compartment": "soma", "unit_id": "r1", "n": 4, "event_active_frame_count": 2},
        {"day_id": "d1", "mode": "sleep", "state": "rem", "compartment": "soma", "unit_id": "r1", "n": 4, "event_active_frame_count": 1},
        {"day_id": "d1", "mode": "sleep", "state": "nrem", "compartment": "soma", "unit_id": "r2", "n": 4, "event_active_frame_count": 0},
        {"day_id": "d1", "mode": "sleep", "state": "rem", "compartment": "soma", "unit_id": "r2", "n": 4, "event_active_frame_count": 4},
    ]
    result = run_family(rows, selected_states=["nrem", "rem"], shuffle_n=0)
    assert [row["event_probability"] for row in result["rows"]] == [0.5, 0.25, 0.0, 1.0]
    soma_comparison = next(row for row in result["comparison_rows"] if row["compartment"] == "soma")
    assert soma_comparison["paired"] is True
    assert soma_comparison["n_subjects"] == 2


def test_activity_probability_returns_nan_without_valid_frames():
    result = run_family(
        [{"day_id": "d1", "state": "nrem", "compartment": "bouton", "unit_id": "r1", "n": 0, "event_active_frame_count": 0}],
        selected_states=["nrem"],
    )
    assert result["rows"][0]["event_probability"] != result["rows"][0]["event_probability"]
