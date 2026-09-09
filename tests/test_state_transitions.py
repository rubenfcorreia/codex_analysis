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


def _plot_rows():
    rows = []
    for index, (pre, post) in enumerate(((1.0, 2.0), (1.2, 2.1), (0.9, 1.8))):
        rows.append({
            "scope": "all_states",
            "window_mode": "strict",
            "state_before": "wake",
            "state_after": "nrem",
            "metric": "mean_activity",
            "compartment": "soma",
            "entity_id": f"soma-{index}",
            "pre_value": pre,
            "post_value": post,
        })
    return rows


def test_transition_boxplot_has_no_paired_connecting_lines(tmp_path, monkeypatch):
    import matplotlib.axes
    from analysis.shared.state_transitions import plot_transition_summaries

    calls = []
    original_plot = matplotlib.axes.Axes.plot
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", lambda self, *args, **kwargs: calls.append((args, kwargs)) or original_plot(self, *args, **kwargs))
    paths = plot_transition_summaries(_plot_rows(), tmp_path, pipeline_name="test")
    assert len(paths) == 1
    assert not any(list(args[0]) == [1.0, 2.0] and len(args[1]) == 2 for args, _ in calls)
    assert "state_transition" in paths[0]


def test_transition_trace_is_zero_aligned_and_direction_specific(tmp_path, monkeypatch):
    import matplotlib.axes
    from analysis.shared.state_transitions import plot_transition_summaries

    trace_segments = []
    for before, after, offset in (("wake", "nrem", 0.0), ("nrem", "wake", 1.0)):
        trace_segments.append({
            "scope": "all_states",
            "window_mode": "strict",
            "state_before": before,
            "state_after": after,
            "metric": "mean_activity",
            "compartment": "soma",
            "window_s": 2.0,
            "relative_time_s": np.arange(-2.0, 2.0, 0.5),
            "values": np.arange(-2.0, 2.0, 0.5) + offset,
        })
    vlines = []
    labels = []
    original_axvline = matplotlib.axes.Axes.axvline
    original_plot = matplotlib.axes.Axes.plot
    monkeypatch.setattr(matplotlib.axes.Axes, "axvline", lambda self, x=0, *args, **kwargs: vlines.append(float(x)) or original_axvline(self, x, *args, **kwargs))
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", lambda self, *args, **kwargs: labels.append(kwargs.get("label")) or original_plot(self, *args, **kwargs))
    paths = plot_transition_summaries(_plot_rows(), tmp_path, pipeline_name="test", trace_segments=trace_segments)
    assert any(path.endswith("_trace.svg") for path in paths)
    assert 0.0 in vlines
    assert any("wake → nrem" in str(label) for label in labels)
    assert any("nrem → wake" in str(label) for label in labels)
