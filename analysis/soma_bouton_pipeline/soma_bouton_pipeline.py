from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.compartment_common import (
    ensure_dir,
    filter_comparison_presets,
    normalize_comparison_presets,
    resolve_repo_root,
    safe_filename_component,
    write_csv_rows,
    write_json_file,
)
from analysis.shared.analysis_families.core import ExperimentContext, build_experiment_context, experiment_summary_row
from analysis.shared.analysis_families.mixed_model import run_family as run_mixed_model_family
from analysis.shared.analysis_families.state import activity_rows_for_context, state_comparison_rows, state_summary_rows
from analysis.shared.shared_calcium_response import (
    DEFAULT_VISUAL_RESPONSE_COHORT,
    VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
    VISUAL_RESPONSE_COHORTS,
    VISUAL_RESPONSE_VISUAL_TRIAL_TYPES,
    load_visual_response_cut_data,
    set_active_event_detection_method,
    set_active_visual_response_metric,
    summarize_visual_response_entity_rows,
    visual_response_trial_rows,
)
from analysis.soma_bouton_pipeline.analysis_families.correlation import bouton_soma_correlation_rows, correlation_summary_rows
from analysis.soma_bouton_pipeline.analysis_families.lag import lag_scan_rows, lag_summary_rows
from analysis.soma_bouton_pipeline.plots import plot_lag_heatmap, plot_state_activity, plot_state_correlation
from analysis.compartment_common import resolve_analysis_state_selections
from analysis.main_pipeline.sleep_dendrite_spine_pipeline import (
    ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
    analysis_cache_meta_hash,
    analysis_results_cache_payload,
    load_analysis_results_cache,
    save_analysis_results_cache,
)


logger = logging.getLogger(__name__)


def _stage(label: str, detail: str | None = None) -> None:
    if detail:
        logger.info("[soma] %s: %s", label, detail)
    else:
        logger.info("[soma] %s", label)


DEFAULT_CONFIG = {
    "analysis_name": "soma_bouton_pipeline",
    "result_root": "results/soma_bouton_pipeline",
    "cache_root": "results/soma_bouton_pipeline/cache",
    "movie_expids": [],
    "sleep_expids": [],
    "soma_channel": 1,
    "bouton_channel": 0,
    "lag_window_s": 2.0,
    "lag_step_s": 0.1,
    "shuffle_n": 200,
    "rebuild": False,
    "cache_path": None,
    "analysis_results_cache_path": None,
    "analysis_tables_cache_path": None,
    "analysis_run_cache_path": None,
    "analysis_results_rebuild": False,
    "analysis_tables_rebuild": False,
    "source_cache_rebuild": False,
    "shared_shuffle_cache_rebuild": False,
    "plots_only": False,
    "comparison_presets": None,
    "comparison_preset_name": None,
    "comparison_preset_names": None,
    "event_detection_method": "amplitude",
    "visual_response_metric": "calcium_events",
    "visual_response_cohort": DEFAULT_VISUAL_RESPONSE_COHORT,
    "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
    "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
    "state_comparison_states": [
        "quiet_awake_blank",
        "nrem_blank",
        "rem_blank",
        "quiet_awake_movies",
        "nrem_movies",
        "rem_movies",
        "quiet_awake",
        "nrem",
        "rem",
    ],
    "basal_apical_states": [
        "quiet_awake_blank",
        "nrem_blank",
        "rem_blank",
        "quiet_awake_movies",
        "nrem_movies",
        "rem_movies",
        "quiet_awake",
        "nrem",
        "rem",
    ],
}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    with path.open() as fh:
        cfg = json.load(fh)
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def _preset_selection_names(value: Any) -> List[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
        return names or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names or None
    text = str(value).strip()
    return [text] if text else None


def run_comparison_preset_runs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    preset_names = _preset_selection_names(config.get("comparison_preset_names"))
    presets = normalize_comparison_presets(config.get("comparison_presets"))
    presets = filter_comparison_presets(presets, preset_names)
    if not presets:
        return []

    base_result_root = Path(config.get("result_root") or DEFAULT_CONFIG["result_root"])
    _stage("comparison presets", f"running {len(presets)} preset(s)")
    manifests: List[Dict[str, Any]] = []
    for preset_name, overrides in presets:
        preset_config = copy.deepcopy(dict(config))
        preset_config.pop("comparison_presets", None)
        preset_config.pop("comparison_preset_names", None)
        preset_config.pop("comparison_preset_name", None)
        preset_config.update(overrides)
        preset_config["comparison_preset_name"] = preset_name
        preset_result_root = base_result_root / safe_filename_component(preset_name)
        preset_config["result_root"] = str(preset_result_root)
        _stage("comparison preset", f"{preset_name} -> {preset_result_root}")
        manifests.append(run_pipeline(preset_config))
    return manifests


def build_day_groups(expids_by_mode: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, List[str]]]:
    from ..compartment_common import grouped_experiments_by_day

    grouped: Dict[str, Dict[str, List[str]]] = {}
    for mode, expids in expids_by_mode.items():
        mode_groups = grouped_experiments_by_day(expids)
        for day_id, members in mode_groups.items():
            grouped.setdefault(day_id, {})[mode] = list(members)
    return grouped


def _visual_response_entity_rows(
    ctx: ExperimentContext,
    *,
    compartment: str,
    channel: int,
    response_metric: str,
    event_detection_method: str,
    locomotion_threshold: float | None,
) -> List[Dict[str, Any]]:
    if ctx.mode != "movie":
        return []
    trial_rows = ctx.state_bundle.get("rows", []) if isinstance(ctx.state_bundle, Mapping) else []
    cut_data = load_visual_response_cut_data(
        ctx.exp_root,
        channel,
        trial_rows,
        locomotion_threshold=locomotion_threshold,
    )
    if not cut_data:
        return []
    cut_neural = cut_data["cut_neural"]
    cut_time = cut_data["cut_time"]
    trial_meta = cut_data.get("trial_meta", []) if isinstance(cut_data, dict) else []
    if not isinstance(trial_meta, list) or cut_neural.size == 0 or cut_time.size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    for roi_index in range(cut_neural.shape[0]):
        trace = np.asarray(cut_neural[roi_index], dtype=float)
        trial_rows = visual_response_trial_rows(
            trace,
            cut_time,
            trial_meta,
            response_metric=response_metric,
            event_detection_method=event_detection_method,
        )
        summary = summarize_visual_response_entity_rows(trial_rows)
        row = {
            "expid": ctx.expid,
            "mode": ctx.mode,
            "animal_id": ctx.animal_id,
            "date": ctx.date,
            "day_id": ctx.day_id,
            "compartment": compartment,
            "roi_index": int(roi_index),
            "response_metric": summary.get("response_metric", response_metric),
            "event_detection_method": event_detection_method,
            "source_label": cut_data.get("source_label"),
            "source_path": cut_data.get("source_path"),
            "available": bool(summary.get("available", False)),
            "comparison": summary.get("comparison", "visual_response_movie_vs_blank"),
            "statistic": float(summary.get("statistic", float("nan"))),
            "raw_pvalue": float(summary.get("raw_pvalue", float("nan"))),
            "adjusted_pvalue": float(summary.get("adjusted_pvalue", float("nan"))),
            "n_visual_values": int(summary.get("n_visual_values", 0)),
            "n_blank_values": int(summary.get("n_blank_values", 0)),
            "mean_visual": float(summary.get("mean_visual", float("nan"))),
            "mean_blank": float(summary.get("mean_blank", float("nan"))),
            "delta": float(summary.get("delta", float("nan"))),
            "paired_stimulus_values": list(summary.get("paired_stimulus_values", [])) if isinstance(summary.get("paired_stimulus_values"), list) else [],
            "blank_reference_values": list(summary.get("blank_reference_values", [])) if isinstance(summary.get("blank_reference_values"), list) else [],
            "visual_trial_labels": list(dict.fromkeys(summary.get("visual_trial_labels", []))) if isinstance(summary.get("visual_trial_labels"), list) else [],
            "blank_trial_labels": list(dict.fromkeys(summary.get("blank_trial_labels", []))) if isinstance(summary.get("blank_trial_labels"), list) else [],
            "significant": bool(summary.get("significant", False)),
            "star": str(summary.get("star", "")),
            "responsive": bool(summary.get("responsive", False)),
            "cohort": str(summary.get("cohort", "nonresponsive")),
            "cohort_requested": str(DEFAULT_VISUAL_RESPONSE_COHORT),
        }
        rows.append(row)
    return rows


def _visual_response_day_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    grouped: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("day_id") or ""),
            str(row.get("mode") or ""),
            str(row.get("compartment") or ""),
            str(row.get("cohort") or "nonresponsive"),
        )
        grouped.setdefault(key, []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for (day_id, mode, compartment, cohort), members in grouped.items():
        visual = np.asarray([float(row.get("mean_visual", float("nan"))) for row in members], dtype=float)
        blank = np.asarray([float(row.get("mean_blank", float("nan"))) for row in members], dtype=float)
        delta = np.asarray([float(row.get("delta", float("nan"))) for row in members], dtype=float)
        responsive = sum(bool(row.get("responsive", False)) for row in members)
        summary_rows.append(
            {
                "day_id": day_id,
                "mode": mode,
                "compartment": compartment,
                "cohort": cohort,
                "n_rois": int(len(members)),
                "n_responsive": int(responsive),
                "responsive_fraction": float(responsive / len(members)) if members else float("nan"),
                "mean_visual": float(np.nanmean(visual)) if visual.size else float("nan"),
                "mean_blank": float(np.nanmean(blank)) if blank.size else float("nan"),
                "mean_delta": float(np.nanmean(delta)) if delta.size else float("nan"),
            }
        )
    return summary_rows




def _soma_analysis_results_meta(config: Mapping[str, Any], selected_states_by_mode: Mapping[str, Sequence[str]], state_modes: Sequence[str], visual_response_cohort: str, event_detection_method: str, visual_response_metric: str) -> Dict[str, Any]:
    return {
        "analysis_name": str(config.get("analysis_name") or "soma_bouton_pipeline"),
        "comparison_preset_name": str(config.get("comparison_preset_name") or "default"),
        "state_modes": list(state_modes),
        "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
        "shuffle_n": int(config.get("shuffle_n", 200)),
        "event_detection_method": str(event_detection_method),
        "visual_response_metric": str(visual_response_metric),
        "visual_response_cohort": str(visual_response_cohort),
        "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "state_comparison_states": list(config.get("state_comparison_states") or []),
        "basal_apical_states": list(config.get("basal_apical_states") or []),
        "movie_expids": list(config.get("movie_expids") or []),
        "sleep_expids": list(config.get("sleep_expids") or []),
        "soma_channel": int(config.get("soma_channel", 1)),
        "bouton_channel": int(config.get("bouton_channel", 0)),
    }


def _analysis_results_cache_path(config: Mapping[str, Any], result_root: Path) -> Path:
    if config.get("analysis_results_cache_path"):
        return Path(config["analysis_results_cache_path"])
    cache_root = Path(config.get("cache_root") or (result_root / "cache"))
    return cache_root / "analysis_results_cache.npz"


def _analysis_tables_cache_path(config: Mapping[str, Any], result_root: Path) -> Path:
    if config.get("analysis_tables_cache_path"):
        return Path(config["analysis_tables_cache_path"])
    cache_root = Path(config.get("cache_root") or (result_root / "cache"))
    return cache_root / "analysis_tables_cache.npz"


def _source_cache_path(config: Mapping[str, Any], result_root: Path) -> Path:
    if config.get("cache_path"):
        return Path(config["cache_path"])
    cache_root = Path(config.get("cache_root") or (result_root / "cache"))
    return cache_root / "source_cache.npz"


def run_pipeline(config: Mapping[str, Any]) -> Dict[str, Any]:
    repo_root = resolve_repo_root(Path(__file__))
    result_root = Path(config["result_root"])
    if not result_root.is_absolute():
        result_root = repo_root / result_root
    preset_name = str(config.get("comparison_preset_name") or "default")
    _stage("run preset", f"{preset_name} -> {result_root}")
    ensure_dir(result_root)
    ensure_dir(result_root / "csv")
    ensure_dir(result_root / "figures")
    ensure_dir(result_root / "cache")
    ensure_dir(result_root / "summary")

    expids_by_mode = {
        "movie": list(config.get("movie_expids", [])),
        "sleep": list(config.get("sleep_expids", [])),
    }
    state_modes = [mode for mode in ("movie", "sleep") if expids_by_mode.get(mode)]
    if config.get("state_mode") in {"movie", "sleep"}:
        state_modes = [str(config["state_mode"])]
    _stage("input selection", f"modes={','.join(state_modes) if state_modes else 'none'}")

    event_detection_method = set_active_event_detection_method(config.get("event_detection_method"))
    visual_response_metric = set_active_visual_response_metric(config.get("visual_response_metric"))
    visual_response_cohort = str(config.get("visual_response_cohort") or DEFAULT_VISUAL_RESPONSE_COHORT)
    _stage(
        "config knobs",
        f"event_detection_method={event_detection_method}, visual_response_metric={visual_response_metric}, visual_response_cohort={visual_response_cohort}",
    )

    shuffle_n = int(config.get("shuffle_n", 200))
    analysis_results_cache_file = _analysis_results_cache_path(config, result_root)
    analysis_tables_cache_file = _analysis_tables_cache_path(config, result_root)
    source_cache_file = _source_cache_path(config, result_root)
    experiment_rows: List[Dict[str, Any]] = []
    activity_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    lag_rows: List[Dict[str, Any]] = []
    visual_response_rows: List[Dict[str, Any]] = []
    selected_states_by_mode: Dict[str, List[str]] = {mode: list(resolve_analysis_state_selections(config, mode)) for mode in state_modes}
    analysis_results_meta = _soma_analysis_results_meta(
        config,
        selected_states_by_mode,
        state_modes,
        visual_response_cohort,
        event_detection_method,
        visual_response_metric,
    )
    if not bool(config.get("rebuild")):
        cached_results, cached_status = load_analysis_results_cache(
            analysis_results_cache_file,
            expected_meta=analysis_results_meta,
            rebuild=bool(config.get("analysis_results_rebuild")),
        )
        if cached_status == "ok" and isinstance(cached_results, dict):
            manifest = dict(cached_results.get("analysis_results", {}))
            manifest.setdefault("cache_summary", {})
            manifest["cache_summary"].update(
                {
                    "analysis_results_cache_path": str(analysis_results_cache_file),
                    "analysis_results_cache_reused": True,
                    "analysis_tables_cache_path": str(analysis_tables_cache_file),
                    "source_cache_path": str(source_cache_file),
                }
            )
            manifest.setdefault("run_parameters", {})
            manifest["run_parameters"].update(
                {
                    "analysis_results_cache_path": str(analysis_results_cache_file),
                    "analysis_tables_cache_path": str(analysis_tables_cache_file),
                    "cache_path": str(source_cache_file),
                    "analysis_results_cache_reused": True,
                }
            )
            _stage("cache load", f"reused analysis-results cache at {analysis_results_cache_file}")
            return manifest

    for mode in state_modes:
        selected_states = list(selected_states_by_mode.get(mode, []))
        _stage("state selection", f"{mode}: {', '.join(selected_states) if selected_states else 'none'}")
        for expid in expids_by_mode.get(mode, []):
            ctx = build_experiment_context(
                expid=expid,
                mode=mode,
                soma_channel=int(config["soma_channel"]),
                bouton_channel=int(config["bouton_channel"]),
                repo_root=repo_root,
            )
            experiment_rows.append(experiment_summary_row(ctx))
            if config.get("plots_only"):
                continue
            activity_rows.extend(activity_rows_for_context(ctx, selected_states))
            correlation_rows.extend(bouton_soma_correlation_rows(ctx, selected_states))
            lag_rows.extend(
                lag_scan_rows(
                    ctx,
                    selected_states,
                    lag_window_s=float(config.get("lag_window_s", 2.0)),
                    lag_step_s=float(config.get("lag_step_s", 0.1)),
                )
            )
            if mode == "movie":
                visual_response_rows.extend(
                    _visual_response_entity_rows(
                        ctx,
                        compartment="soma",
                        channel=int(config["soma_channel"]),
                        response_metric=visual_response_metric,
                        event_detection_method=event_detection_method,
                        locomotion_threshold=float(config.get("locomotion_threshold", 0.0)) if config.get("locomotion_threshold") is not None else None,
                    )
                )
                visual_response_rows.extend(
                    _visual_response_entity_rows(
                        ctx,
                        compartment="bouton",
                        channel=int(config["bouton_channel"]),
                        response_metric=visual_response_metric,
                        event_detection_method=event_detection_method,
                        locomotion_threshold=float(config.get("locomotion_threshold", 0.0)) if config.get("locomotion_threshold") is not None else None,
                    )
                )
        _stage(
            "mode complete",
            f"{mode}: experiments={len(expids_by_mode.get(mode, []))}, activity_rows={len(activity_rows)}, correlation_rows={len(correlation_rows)}, lag_rows={len(lag_rows)}, visual_response_rows={len(visual_response_rows)}",
        )

    day_groups = build_day_groups(expids_by_mode)

    _stage("summaries", "building day-level comparison tables")
    activity_summary_rows = state_summary_rows(activity_rows)
    state_comparison_summary_rows = state_comparison_rows(
        [row for row in activity_summary_rows if str(row.get("mode") or "") == "movie"],
        selected_states_by_mode.get("movie", []),
        shuffle_n,
    )
    sleep_state_comparison_summary_rows = state_comparison_rows(
        [row for row in activity_summary_rows if str(row.get("mode") or "") == "sleep"],
        selected_states_by_mode.get("sleep", []),
        shuffle_n,
    )
    mixed_model_results = run_mixed_model_family(
        activity_summary_rows,
        state_comparison_states=selected_states_by_mode.get("movie", []),
        basal_apical_states=selected_states_by_mode.get("movie", []),
        shuffle_n=shuffle_n,
        mixed_model_contrast_p_source=str(config.get("mixed_model_contrast_p_source") or "classical"),
    )
    correlation_summary = correlation_summary_rows(correlation_rows)
    lag_summary = lag_summary_rows(lag_rows)
    visual_response_day_rows = _visual_response_day_rows(visual_response_rows)
    mixed_model_contrast_count = len(mixed_model_results.get("all_state", {}).get("contrast_rows", [])) if isinstance(mixed_model_results, dict) else 0
    _stage(
        "summary counts",
        f"activity={len(activity_summary_rows)}, movie_comparisons={len(state_comparison_summary_rows)}, sleep_comparisons={len(sleep_state_comparison_summary_rows)}, correlation={len(correlation_summary)}, lag={len(lag_summary)}, visual_response={len(visual_response_rows)}, mixed_model={mixed_model_contrast_count}",
    )

    _stage("writing csv", "experiments")
    write_csv_rows(result_root / "csv" / "experiments.csv", experiment_rows, list(experiment_rows[0].keys()) if experiment_rows else ["expid"])
    if activity_rows:
        _stage("writing csv", "state_activity_by_experiment")
        write_csv_rows(result_root / "csv" / "state_activity_by_experiment.csv", activity_rows, list(activity_rows[0].keys()))
    if correlation_rows:
        _stage("writing csv", "bouton_soma_correlation_by_roi")
        write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_roi.csv", correlation_rows, list(correlation_rows[0].keys()))
    if lag_rows:
        _stage("writing csv", "bouton_soma_lag_scan_by_roi")
        write_csv_rows(result_root / "csv" / "bouton_soma_lag_scan_by_roi.csv", lag_rows, list(lag_rows[0].keys()))
    if activity_summary_rows:
        _stage("writing csv", "state_activity_by_day")
        write_csv_rows(result_root / "csv" / "state_activity_by_day.csv", activity_summary_rows, list(activity_summary_rows[0].keys()))
    if state_comparison_summary_rows:
        _stage("writing csv", "state_comparisons_movie")
        write_csv_rows(result_root / "csv" / "state_comparisons_movie.csv", state_comparison_summary_rows, list(state_comparison_summary_rows[0].keys()))
    if sleep_state_comparison_summary_rows:
        _stage("writing csv", "state_comparisons_sleep")
        write_csv_rows(result_root / "csv" / "state_comparisons_sleep.csv", sleep_state_comparison_summary_rows, list(sleep_state_comparison_summary_rows[0].keys()))
    if correlation_summary:
        _stage("writing csv", "bouton_soma_correlation_by_day")
        write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_day.csv", correlation_summary, list(correlation_summary[0].keys()))
    if lag_summary:
        _stage("writing csv", "bouton_soma_lag_summary_by_day")
        write_csv_rows(result_root / "csv" / "bouton_soma_lag_summary_by_day.csv", lag_summary, list(lag_summary[0].keys()))
    if visual_response_rows:
        _stage("writing csv", "visual_response_by_roi")
        write_csv_rows(result_root / "csv" / "visual_response_by_roi.csv", visual_response_rows, list(visual_response_rows[0].keys()))
    if visual_response_day_rows:
        _stage("writing csv", "visual_response_by_day")
        write_csv_rows(result_root / "csv" / "visual_response_by_day.csv", visual_response_day_rows, list(visual_response_day_rows[0].keys()))

    _stage("plotting", "state activity")
    plot_state_activity(activity_summary_rows or activity_rows, result_root)
    _stage("plotting", "correlation")
    plot_state_correlation(correlation_summary, result_root)
    _stage("plotting", "lag heatmap")
    plot_lag_heatmap(lag_rows, result_root)

    visual_response_counts = {
        "all": int(len(visual_response_rows)),
        "responsive": int(sum(bool(row.get("responsive", False)) for row in visual_response_rows)),
        "nonresponsive": int(sum(not bool(row.get("responsive", False)) for row in visual_response_rows)),
    }
    manifest = {
        "config": dict(config),
        "comparison_preset_name": preset_name,
        "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
        "state_modes": state_modes,
        "day_groups": day_groups,
        "counts": {
            "experiments": len(experiment_rows),
            "activity_rows": len(activity_rows),
            "state_comparison_rows_movie": len(state_comparison_summary_rows),
            "state_comparison_rows_sleep": len(sleep_state_comparison_summary_rows),
            "correlation_rows": len(correlation_rows),
            "lag_rows": len(lag_rows),
            "visual_response_rows": len(visual_response_rows),
            "visual_response_responsive_rows": visual_response_counts["responsive"],
            "visual_response_nonresponsive_rows": visual_response_counts["nonresponsive"],
        },
        "event_detection_method": str(config.get("event_detection_method") or "amplitude"),
        "visual_response_metric": str(config.get("visual_response_metric") or "calcium_events"),
        "visual_response_cohort": visual_response_cohort,
        "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "visual_response_cohort_ids": {
            "soma": {
                "responsive": [str(row.get("roi_index")) for row in visual_response_rows if str(row.get("compartment")) == "soma" and bool(row.get("responsive", False))],
                "nonresponsive": [str(row.get("roi_index")) for row in visual_response_rows if str(row.get("compartment")) == "soma" and not bool(row.get("responsive", False))],
            },
            "bouton": {
                "responsive": [str(row.get("roi_index")) for row in visual_response_rows if str(row.get("compartment")) == "bouton" and bool(row.get("responsive", False))],
                "nonresponsive": [str(row.get("roi_index")) for row in visual_response_rows if str(row.get("compartment")) == "bouton" and not bool(row.get("responsive", False))],
            },
        },
        "mixed_model": mixed_model_results,
        "cache_summary": {
            "analysis_results_cache_path": str(analysis_results_cache_file),
            "analysis_results_cache_reused": False,
            "analysis_tables_cache_path": str(analysis_tables_cache_file),
            "source_cache_path": str(source_cache_file),
        },
        "output_root": str(result_root),
    }
    write_json_file(result_root / "summary" / "manifest.json", manifest)
    save_analysis_results_cache(
        analysis_results_cache_file,
        {
            "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
            "meta": analysis_results_meta,
            "meta_hash": analysis_cache_meta_hash(analysis_results_meta),
            "analysis_results": analysis_results_cache_payload(manifest),
        },
    )
    _stage("completed", f"{preset_name} with {len(experiment_rows)} experiments")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Compare soma (ch2) and bouton (ch1) activity across movie and sleep states.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("soma_bouton_pipeline_config.json"))
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding outputs even if caches exist.")
    parser.add_argument("--plots-only", action="store_true", help="Skip metric recomputation and regenerate plots from written CSVs only.")
    parser.add_argument(
        "--comparison-presets",
        nargs="*",
        help="Optional subset of preset names to run from a comparison_presets config block.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.rebuild:
        config["rebuild"] = True
    if args.plots_only:
        config["plots_only"] = True
    if args.comparison_presets:
        config["comparison_preset_names"] = list(args.comparison_presets)
    if config.get("comparison_presets"):
        manifests = run_comparison_preset_runs(config)
        if manifests:
            print(json.dumps({"comparison_presets": manifests}, indent=2, sort_keys=True))
            return 0
    manifest = run_pipeline(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
