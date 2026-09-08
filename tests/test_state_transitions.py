import numpy as np

from analysis.shared.state_transitions import add_window_metadata, transition_events


def test_transitions_stay_within_one_experiment():
    time = np.arange(240.0)
    masks = {"quiet": time < 120.0, "nrem": time >= 120.0}
    events = transition_events(time, masks, ["quiet", "nrem"], scope="sleep_states", window_s=60.0)
    assert len(events) == 1
    assert events[0]["state_before"] == "quiet"
    assert events[0]["state_after"] == "nrem"
    assert add_window_metadata(events[0], "strict")["pre_duration_s"] == 60.0


def test_partial_windows_are_separate_from_strict_windows():
    time = np.arange(100.0)
    masks = {"quiet": time < 50.0, "nrem": time >= 50.0}
    events = transition_events(time, masks, ["quiet", "nrem"], scope="sleep_states", window_s=60.0)
    assert len(events) == 1
    assert add_window_metadata(events[0], "strict") is None
    partial = add_window_metadata(events[0], "max_available")
    assert partial is not None
    assert partial["pre_duration_s"] < 60.0


def test_unknown_gap_does_not_create_transition():
    time = np.arange(180.0)
    masks = {
        "quiet": time < 60.0,
        "nrem": time >= 120.0,
    }
    assert transition_events(time, masks, ["quiet", "nrem"], scope="sleep_states", window_s=60.0) == []


def test_state_precedence_is_deterministic():
    time = np.arange(180.0)
    masks = {"first": time < 120.0, "second": time >= 60.0}
    events = transition_events(time, masks, ["first", "second"], scope="all_states", window_s=30.0)
    assert len(events) == 1
    assert events[0]["transition_time"] == 120.0
