from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.shared.plots import poster_ready


def _plot_data(offset: float) -> dict[str, np.ndarray]:
    cut_time = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    return {
        'cut_time': cut_time,
        'visual_mean_trace': np.array([offset + 0.0, offset + 1.0, offset + 2.0, offset + 1.5], dtype=float),
        'blank_mean_trace': np.array([offset - 2.0, offset - 1.0, offset - 1.5, offset - 0.5], dtype=float),
        'visual_values': np.array([offset + 0.5, offset + 2.5], dtype=float),
        'blank_values': np.array([offset - 2.5, offset - 1.0], dtype=float),
        'event_onset': 0.0,
        'event_duration': 1.0,
    }


def test_poster_ready_modes_visual_response_uses_shared_y_limits(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {'compartment': 'soma', 'global_soma_id': 'responsive_soma', 'responsive': True, 'delta': 3.0},
        {'compartment': 'soma', 'global_soma_id': 'nonresponsive_soma', 'responsive': False, 'delta': 0.2},
    ]

    def fake_load_visual_response_plot_data(row, *, locomotion_threshold=None):
        return _plot_data(0.0 if bool(row.get('responsive', False)) else 20.0)

    captured: dict[str, object] = {}
    original_write_figure = poster_ready._write_figure

    def wrapped_write_figure(fig, output_path):
        captured['fig'] = fig
        captured['output_path'] = Path(output_path)
        return original_write_figure(fig, output_path)

    monkeypatch.setattr(poster_ready, '_load_visual_response_plot_data', fake_load_visual_response_plot_data)
    monkeypatch.setattr(poster_ready, '_write_figure', wrapped_write_figure)

    output_path = poster_ready.write_visual_response_poster_figure(
        output_dir=tmp_path,
        entity_label='soma',
        visual_response_rows=rows,
        kind='soma',
    )

    assert output_path is not None
    assert Path(output_path).exists()
    fig = captured['fig']
    assert fig is not None

    response_axes = [ax for ax in fig.axes if ax.lines]
    assert len(response_axes) == 6

    expected_limits = poster_ready._visual_response_poster_shared_limits(_plot_data(0.0), _plot_data(20.0))
    assert expected_limits is not None
    for ax in response_axes:
        assert np.allclose(ax.get_ylim(), expected_limits)

    if poster_ready.plt is not None:
        poster_ready.plt.close(fig)
