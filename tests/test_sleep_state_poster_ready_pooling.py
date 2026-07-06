from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.sleep_state_across_days import sleep_state_across_days as sleepmod


def _summary(exp_id: str, category: str, sleep_state_path: str, *, active_wake: float, nrem: float) -> sleepmod.SessionSummary:
    total_time_s = float(active_wake + nrem)
    return sleepmod.SessionSummary(
        scope='animal',
        animal_id='mouse_1',
        date='2026-01-01',
        category=category,
        exp_ids=[exp_id],
        sleep_state_paths=[sleep_state_path],
        epoch_count=int(total_time_s / 60.0),
        epoch_duration_s=60.0,
        total_time_s=total_time_s,
        unknown_epoch_count=0,
        unknown_epoch_fraction=0.0,
        state_time_s={
            'active_wake': float(active_wake),
            'quiet_wake': 0.0,
            'nrem': float(nrem),
            'rem': 0.0,
        },
        state_fraction={
            'active_wake': float(active_wake / total_time_s) if total_time_s else float('nan'),
            'quiet_wake': 0.0,
            'nrem': float(nrem / total_time_s) if total_time_s else float('nan'),
            'rem': 0.0,
        },
        bout_count={state: 1 if state in {'active_wake', 'nrem'} else 0 for state in sleepmod.DEFAULT_STATE_ORDER},
        bout_total_time_s={
            'active_wake': float(active_wake),
            'quiet_wake': 0.0,
            'nrem': float(nrem),
            'rem': 0.0,
        },
        bout_mean_duration_s={
            'active_wake': 60.0 if active_wake else float('nan'),
            'quiet_wake': float('nan'),
            'nrem': 60.0 if nrem else float('nan'),
            'rem': float('nan'),
        },
    )


def test_same_day_sleep_expids_pool_and_sleep_start_uses_sleep_session_start(monkeypatch) -> None:
    arrays = {
        'awake': (np.zeros(90, dtype=int), np.arange(90, dtype=float) * 60.0),
        'sleep_a': (np.ones(60, dtype=int) * 2, np.arange(60, dtype=float) * 60.0),
        'sleep_b': (np.ones(60, dtype=int) * 2, np.arange(60, dtype=float) * 60.0),
    }

    def fake_load_sleep_state_arrays(path):
        key = Path(path).stem
        return arrays[key]

    monkeypatch.setattr(sleepmod, 'load_sleep_state_arrays', fake_load_sleep_state_arrays)

    exp_summaries = [
        _summary('awake_1', 'awake', 'awake.npy', active_wake=300.0, nrem=0.0),
        _summary('sleep_1', 'sleep', 'sleep_a.npy', active_wake=0.0, nrem=300.0),
        _summary('sleep_2', 'sleep', 'sleep_b.npy', active_wake=0.0, nrem=300.0),
    ]

    day_summaries = sleepmod.aggregate_day_summaries(exp_summaries)
    assert len(day_summaries) == 2

    sleep_day = next(summary for summary in day_summaries if summary.category == 'sleep')
    assert sleep_day.exp_ids == ['sleep_1', 'sleep_2']
    assert sleep_day.state_time_s['nrem'] == 600.0

    time_min, profile, sleep_start_min = sleepmod.build_day_timeline_profile(exp_summaries)
    assert time_min.size == 42
    assert np.isclose(float(time_min[0]), 2.5)
    assert np.isclose(float(time_min[-1]), 207.5)
    assert sleep_start_min is not None
    assert np.isclose(float(sleep_start_min), 90.0)
    assert np.isclose(float(sleepmod.estimate_average_sleep_start_min(exp_summaries)), 90.0)
    assert np.isfinite(np.asarray(profile['nrem'], dtype=float)).any()

    combined_time_s, combined_profile = sleepmod.average_combined_day_probability_summaries(exp_summaries)
    assert combined_time_s.size == 42
    assert np.isclose(float(combined_time_s[0]), 150.0)
    assert np.isclose(float(combined_time_s[-1]), 12450.0)
    assert np.isfinite(np.asarray(combined_profile['active_wake'], dtype=float)).any()
    assert np.isfinite(np.asarray(combined_profile['nrem'], dtype=float)).any()


def test_within_day_sleep_state_fractions_uses_combined_movie_sleep_trace(tmp_path: Path, monkeypatch) -> None:
    arrays = {
        'awake': (np.zeros(90, dtype=int), np.arange(90, dtype=float) * 60.0),
        'sleep_a': (np.ones(60, dtype=int) * 2, np.arange(60, dtype=float) * 60.0),
        'sleep_b': (np.ones(60, dtype=int) * 2, np.arange(60, dtype=float) * 60.0),
    }

    def fake_load_sleep_state_arrays(path):
        key = Path(path).stem
        return arrays[key]

    monkeypatch.setattr(sleepmod, 'load_sleep_state_arrays', fake_load_sleep_state_arrays)

    captured: dict[str, object] = {}
    original_subplots = sleepmod.plt.subplots

    def wrapped_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured['fig'] = fig
        captured['axes'] = axes
        return fig, axes

    monkeypatch.setattr(sleepmod.plt, 'subplots', wrapped_subplots)

    exp_summaries = [
        _summary('awake_1', 'awake', 'awake.npy', active_wake=5400.0, nrem=0.0),
        _summary('sleep_1', 'sleep', 'sleep_a.npy', active_wake=0.0, nrem=3600.0),
        _summary('sleep_2', 'sleep', 'sleep_b.npy', active_wake=0.0, nrem=3600.0),
    ]

    output_paths = sleepmod.plot_within_day_sleep_state_fractions(exp_summaries, tmp_path)
    assert len(output_paths) == 2
    assert all(Path(path).exists() for path in output_paths)

    fig = captured['fig']
    assert fig is not None
    try:
        axes = [ax for ax in fig.axes if ax.lines]
        assert len(axes) == 4
        for ax in axes:
            trace_x = np.asarray(ax.lines[0].get_xdata(), dtype=float)
            assert trace_x.size == 42
            assert np.isclose(float(trace_x[0]), 2.5)
            assert np.isclose(float(trace_x[-1]), 207.5)
            assert len(ax.lines) >= 2
            marker_x = np.asarray(ax.lines[1].get_xdata(), dtype=float)
            assert np.allclose(marker_x, np.array([90.0, 90.0], dtype=float))
    finally:
        if sleepmod.plt is not None:
            sleepmod.plt.close(fig)
