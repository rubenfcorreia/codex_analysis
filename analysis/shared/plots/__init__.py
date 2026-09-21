from importlib import import_module

from .registry import plot_analysis_family, plot_family_names, register_plot_family

_EXPORTS = {
    "build_coincidence_example_figure_path": (".coincidence", "build_coincidence_example_figure_path"),
    "coincidence_example_figure_dir": (".coincidence", "coincidence_example_figure_dir"),
    "draw_boxplot_series": (".boxplots", "draw_boxplot_series"),
    "plot_boxplot_series": (".boxplots", "plot_boxplot_series"),
    "plot_grouped_boxplot_series": (".boxplots", "plot_grouped_boxplot_series"),
    "save_figure": (".figure_io", "save_figure"),
    "plot_lag_heatmap": (".state", "plot_lag_heatmap"),
    "plot_state_activity": (".state", "plot_state_activity"),
    "plot_state_correlation": (".state", "plot_state_correlation"),
    "plot_state_event_frequency": (".state", "plot_state_event_frequency"),
    "plot_coincidence_event_example_figure": (".coincidence", "plot_coincidence_event_example_figure"),
    "plot_mixed_model_contrasts_checkpoint": (".mixed_model", "plot_mixed_model_contrasts_checkpoint"),
    "plot_mixed_model_forest_figure": (".mixed_model", "plot_mixed_model_forest_figure"),
    "plot_mixed_model_predicted_means_figure": (".mixed_model", "plot_mixed_model_predicted_means_figure"),
    "plot_roi_split_bundle_figure": (".roi_split", "plot_roi_split_bundle_figure"),
    "plot_visual_response_boxplot_figure": (".visual_response", "plot_visual_response_boxplot_figure"),
    "plot_visual_response_entity_figure": (".visual_response", "plot_visual_response_entity_figure"),
    "render_visual_response_entity_figures": (".visual_response", "render_visual_response_entity_figures"),
    "roi_split_figure_output_dir": (".roi_split", "roi_split_figure_output_dir"),
    "visual_response_figure_output_dir": (".visual_response", "visual_response_figure_output_dir"),
    "write_blank_movie_state_boxplot_figure": (".poster_ready", "write_blank_movie_state_boxplot_figure"),
    "write_state_mixed_model_poster_figure": (".poster_ready", "write_state_mixed_model_poster_figure"),
    "write_visual_response_poster_figure": (".poster_ready", "write_visual_response_poster_figure"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(set(_EXPORTS) | {"plot_analysis_family", "plot_family_names", "register_plot_family"})
