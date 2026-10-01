"""Compatibility imports for the shared soma/bouton plotters.

Plot implementations live in :mod:`analysis.shared.plots.state`.  This
module remains as a stable import path for older pipeline callers.
"""

from analysis.shared.plots.state import (
    canonical_state_label,
    ordered_state_labels,
    plot_lag_heatmap,
    plot_state_activity,
    plot_state_correlation,
    plot_state_event_frequency,
    pretty_state_label,
    state_display_color,
)

__all__ = [
    "canonical_state_label", "ordered_state_labels", "plot_lag_heatmap",
    "plot_state_activity", "plot_state_correlation",
    "plot_state_event_frequency", "pretty_state_label", "state_display_color",
]
