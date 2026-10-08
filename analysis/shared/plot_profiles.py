"""Shared plotting-profile normalization for both analysis pipelines."""

from __future__ import annotations

from typing import Any, MutableMapping

PLOT_PROFILES = {"full", "analysis_only", "poster_only"}


def apply_plot_profile(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Apply an explicit profile while preserving legacy full-mode defaults."""
    profile = str(config.get("plot_profile") or "full").strip().lower()
    if profile not in PLOT_PROFILES:
        raise ValueError(f"Unknown plot_profile={profile!r}; choose one of {sorted(PLOT_PROFILES)}")
    config["plot_profile"] = profile
    if profile == "analysis_only":
        config.update(
            poster_ready_only=True,
            generate_poster_ready_figures=False,
            generate_visual_response_entity_figures=False,
            generate_dff_heatmaps=False,
            branch_first_figures=False,
        )
    elif profile == "poster_only":
        config.update(
            poster_ready_only=True,
            generate_poster_ready_figures=True,
            generate_visual_response_entity_figures=False,
            generate_dff_heatmaps=False,
            branch_first_figures=False,
        )
    return config
