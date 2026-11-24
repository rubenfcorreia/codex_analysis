"""Configuration normalization for the staged cache workflow.

The pipelines historically use a flat JSON configuration.  This module accepts
the new nested layout while emitting the legacy flat keys consumed by the
existing implementations.  Keeping that translation in one place lets the
cache stages and the legacy pipeline entry points share identical semantics.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _put_if_present(target: Dict[str, Any], source: Mapping[str, Any], key: str, *aliases: str) -> None:
    for name in (key, *aliases):
        if name in source and source[name] is not None:
            target[key] = deepcopy(source[name])
            return


def normalize_staged_config(raw: Mapping[str, Any], *, pipeline: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return a backwards-compatible flat config with staged sections attached."""
    config = deepcopy(dict(raw))
    config.setdefault("pipeline", {})
    config["pipeline"] = dict(config["pipeline"] or {})
    config["pipeline"].setdefault("name", pipeline)

    # Project identity lives in the staged pipeline section while legacy entry points expect top-level keys.
    _put_if_present(config, config["pipeline"], "user_id", "user")
    _put_if_present(config, config["pipeline"], "repo_base")

    inputs = dict(config.get("inputs") or {})
    lists_file = inputs.get("expids_file") or config.get("experiment_lists_file")
    if lists_file and base_dir is not None:
        lists_path = Path(str(lists_file))
        if not lists_path.is_absolute():
            lists_path = base_dir / lists_path
        if lists_path.exists():
            import json
            with lists_path.open() as handle:
                external_inputs = json.load(handle)
            if not isinstance(external_inputs, Mapping):
                raise ValueError(f"Experiment list file must contain an object: {lists_path}")
            inputs = _deep_merge(dict(external_inputs), inputs)
            config["experiment_lists_file"] = str(lists_path)
    paths = config.get("paths") or {}
    source = config.get("source") or {}
    processing = config.get("processing") or {}
    classification = config.get("classification") or {}
    initial_analysis = config.get("initial_analysis") or {}
    shuffle = config.get("shuffle") or {}
    analysis = config.get("analysis") or {}
    plots = config.get("plots") or {}
    cache = config.get("cache") or {}

    # Keep the long, pipeline-specific family registry outside the main config.
    family_settings = {}
    families_file = analysis.get("families_file")
    if families_file and base_dir is not None:
        families_path = Path(str(families_file))
        if not families_path.is_absolute():
            families_path = base_dir / families_path
        if not families_path.exists() and not Path(str(families_file)).is_absolute():
            canonical_path = Path(__file__).resolve().parents[1] / f"{pipeline}_pipeline" / Path(str(families_file)).name
            if canonical_path.exists():
                families_path = canonical_path
        if not families_path.exists():
            raise FileNotFoundError(f"Analysis-family config does not exist: {families_path}")
        import json
        with families_path.open() as handle:
            family_config = json.load(handle)
        if isinstance(family_config, Mapping):
            family_settings = dict(family_config.get("families") or {})
        elif isinstance(family_config, list):
            family_settings = {str(name): {"enabled": True} for name in family_config}
        else:
            raise ValueError(f"Analysis-family config must contain an object or list: {families_path}")
        for family_name, settings in family_settings.items():
            if not isinstance(settings, Mapping):
                continue
            scope = str(settings.get("scope", "expid_day"))
            aggregate = str(settings.get("aggregate", "pooled"))
            if scope not in {"expid_day", "pooled"}:
                raise ValueError(f"Unsupported cache scope for analysis family {family_name!r}: {scope!r}")
            if aggregate not in {"pooled", "per_scope"}:
                raise ValueError(f"Unsupported cache aggregation for analysis family {family_name!r}: {aggregate!r}")
            settings = dict(settings)
            settings["scope"] = scope
            settings["aggregate"] = aggregate
            family_settings[family_name] = settings
        enabled_families = [
            str(name) for name, settings in family_settings.items()
            if not isinstance(settings, Mapping) or bool(settings.get("enabled", True))
        ]
        analysis = _deep_merge(dict(analysis), {"families": enabled_families})
        config["analysis_families_file"] = str(families_path)

    for key in ("movie_expids", "sleep_expids", "basal_expids", "apical_expids"):
        _put_if_present(config, inputs, key)
    _put_if_present(config, paths, "result_root", "output_root")
    _put_if_present(config, paths, "output_dir", "output_root")
    _put_if_present(config, paths, "figure_output_dir", "figure_root")
    _put_if_present(config, paths, "analysis_output_dir", "analysis_root")
    _put_if_present(config, source, "repo_base")
    _put_if_present(config, source, "channel")
    _put_if_present(config, source, "soma_channel")
    _put_if_present(config, source, "bouton_channel")

    for key in ("event_detection_method", "high_pass_hz", "locomotion_threshold", "state_mode", "movie_trial_types"):
        _put_if_present(config, processing, key)
    _put_if_present(config, classification, "classification_cache_path")
    for key in (
        "visual_response_metric",
        "visual_response_trial_types",
        "visual_response_blank_trial_type",
        "visual_response_cohort",
    ):
        _put_if_present(config, classification, key)
    _put_if_present(config, shuffle, "shuffle_n")
    for key in (
        "correlation_method",
        "correlation_inference",
        "correlation_shuffle_n",
        "correlation_shuffle_seed",
        "correlation_min_shift_frames",
        "pairwise_correlation_inference",
        "pairwise_correlation_shuffle_n",
        "pairwise_correlation_batch_size",
        "pairwise_correlation_shift_block_size",
        "pairwise_correlation_null_model",
    ):
        _put_if_present(config, shuffle, key)
    _put_if_present(config, analysis, "analysis_families", "families")
    if family_settings:
        config["analysis_family_settings"] = deepcopy(family_settings)

    plot_config_file = plots.get("config_file")
    if plot_config_file and base_dir is not None:
        plot_path = Path(str(plot_config_file))
        if not plot_path.is_absolute():
            plot_path = base_dir / plot_path
        if not plot_path.exists() and not Path(str(plot_config_file)).is_absolute():
            canonical_path = Path(__file__).resolve().parents[1] / f"{pipeline}_pipeline" / Path(str(plot_config_file)).name
            if canonical_path.exists():
                plot_path = canonical_path
        if not plot_path.exists():
            raise FileNotFoundError(f"Plot config does not exist: {plot_path}")
        import json
        with plot_path.open() as handle:
            external_plots = json.load(handle)
        if not isinstance(external_plots, Mapping):
            raise ValueError(f"Plot config must contain an object: {plot_path}")
        plots = _deep_merge(dict(external_plots), plots)
        config["plot_config_file"] = str(plot_path)

    _put_if_present(config, plots, "plots_only")
    _put_if_present(config, plots, "generate_visual_response_entity_figures")
    _put_if_present(config, plots, "visual_response_entity_max_trace_points")
    _put_if_present(config, plots, "visual_response_entity_max_figures")
    _put_if_present(config, plots, "poster_ready_only", "poster_ready")
    _put_if_present(config, plots, "plots_only_include_supporting_figures", "supporting_figures")

    if "cache_paths" in config:
        config["cache_paths"] = dict(config["cache_paths"] or {})
    else:
        root = config.get("cache_root") or paths.get("cache_root")
        if not root:
            root = Path(config.get("output_dir") or config.get("result_root", "results")) / "cache"
        if root:
            config["cache_paths"] = {
                stage: f"{root}/{stage}" for stage in
                ("source", "processed", "initial_analysis", "classification", "for_analysis", "shuffle", "results")
            }

    rebuild_stages = list(cache.get("rebuild_stages") or [])
    stage_flags = {
        "source": "source_cache_rebuild",
        "processed": "processed_cache_rebuild",
        "initial_analysis": "initial_analysis_cache_rebuild",
        "classification": "classification_cache_rebuild",
        "for_analysis": "analysis_tables_rebuild",
        "shuffle": "shared_shuffle_cache_rebuild",
        "results": "analysis_results_rebuild",
    }
    for stage, flag in stage_flags.items():
        if stage in rebuild_stages:
            config[flag] = True
    if cache.get("rebuild_expids") is not None:
        config["rebuild_expids"] = list(cache.get("rebuild_expids") or [])
    if cache.get("validate_source_signatures") is not None:
        config["source_cache_validate"] = bool(cache["validate_source_signatures"])
    if cache.get("workers") is not None:
        config["cache_workers"] = int(cache["workers"])

    # Keep nested settings available to new cache builders while preserving the
    # flat shape required by the established pipeline implementations.
    config["staged_config"] = {
        "pipeline": deepcopy(config["pipeline"]),
        "inputs": deepcopy(inputs),
        "paths": deepcopy(paths),
        "source": deepcopy(source),
        "processing": deepcopy(processing),
        "classification": deepcopy(classification),
        "initial_analysis": deepcopy(initial_analysis),
        "shuffle": deepcopy(shuffle),
        "analysis": deepcopy(analysis),
        "cache": deepcopy(cache),
        "plots": deepcopy(plots),
    }
    return config


__all__ = ["normalize_staged_config"]
