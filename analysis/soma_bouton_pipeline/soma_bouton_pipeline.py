from __future__ import annotations

import argparse
import copy
import datetime as dt
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
    normalize_comparison_presets,
    read_csv_rows,
    resolve_repo_root,
    write_csv_rows,
    write_json_file,
)
from analysis.shared.comparison_preset_flow import (
    POSTER_REQUIRED_COMPARISON_PRESETS,
    build_comparison_preset_batch_plan,
)
from analysis.shared.state_utils import canonical_state_label, resolve_analysis_state_selections, resolve_repo_path, safe_filename_component
from analysis.shared.analysis_families.core import ExperimentContext, build_experiment_context, experiment_summary_row, make_global_bouton_id, make_global_soma_id
from analysis.shared.analysis_families.pairwise import pairwise_correlation_summary_rows
from analysis.shared.analysis_families.mixed_model import run_family as run_mixed_model_family
from analysis.shared.analysis_families.state import activity_rows_for_context, state_comparison_rows, state_summary_rows
from analysis.shared.analysis_families.visual_response import run_family as run_visual_response_family, visual_response_day_rows as shared_visual_response_day_rows
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
from analysis.soma_bouton_pipeline.analysis_families.correlation import bouton_pairwise_correlation_rows, bouton_soma_correlation_rows, correlation_summary_rows, soma_pairwise_correlation_rows
from analysis.soma_bouton_pipeline.analysis_families.lag import lag_scan_rows, lag_summary_rows
from analysis.soma_bouton_pipeline.plots import plot_lag_heatmap, plot_state_activity, plot_state_correlation, plot_state_event_frequency
from analysis.shared.plots.mixed_model import (
    plot_mixed_model_contrasts_checkpoint,
    plot_mixed_model_forest_figure,
    plot_mixed_model_predicted_means_figure,
)
from analysis.shared.plots.poster_ready import (
    _assign_visual_response_cohorts,
    _poster_mixed_model_significant_states,
    _select_mixed_model_rows,
    _significant_state_labels_from_comparison_rows,
    assign_pairwise_visual_response_cohorts,
    split_rows_by_cohort,
    write_blank_movie_and_correlation_poster_figure,
    write_blank_movie_state_boxplot_figure,
    write_correlation_poster_figure,
    write_state_mixed_model_poster_figure,
    write_visual_response_poster_figure,
)
from analysis.shared.plots.visual_response import plot_visual_response_boxplot_figure, render_visual_response_entity_figures
from analysis.dendrites_pipeline.dendrites_pipeline import (
    ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
    ANALYSIS_TABLE_CACHE_SCHEMA_VERSION,
    analysis_cache_meta_hash,
    analysis_results_cache_payload,
    load_analysis_results_cache,
    save_analysis_results_cache,
    save_analysis_tables_cache,
)
from analysis.shared.cache_utils import family_results_cache_path, load_family_results_cache, save_family_results_cache


logger = logging.getLogger(__name__)


def _stage(label: str, detail: str | None = None) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if detail:
        logger.info("%s %s: %s", stamp, label, detail)
    else:
        logger.info("%s %s", stamp, label)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


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
    "poster_ready_only": False,
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
    "compartment_states": [
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
    plan = build_comparison_preset_batch_plan(
        presets,
        selected_names=preset_names,
        poster_ready_only=bool(config.get("poster_ready_only")),
        poster_required_names=POSTER_REQUIRED_COMPARISON_PRESETS,
    )
    if not plan.presets:
        return []

    repo_root = resolve_repo_root(Path(__file__))
    base_result_root = resolve_repo_path(config.get("result_root") or DEFAULT_CONFIG["result_root"], repo_root)
    base_cache_root = resolve_repo_path(config.get("cache_root") or DEFAULT_CONFIG["cache_root"], repo_root)
    shared_source_cache_path = resolve_repo_path(config.get("cache_path") or (base_cache_root / "source_cache.npz"), repo_root)
    _stage("comparison presets", f"running {len(plan.presets)} preset(s)")
    manifests: List[Dict[str, Any]] = []
    preset_configs: Dict[str, Dict[str, Any]] = {}
    for preset_index, (preset_name, overrides) in enumerate(plan.presets):
        safe_name = safe_filename_component(preset_name)
        preset_config = copy.deepcopy(dict(config))
        preset_config.pop("comparison_presets", None)
        preset_config.pop("comparison_preset_names", None)
        preset_config.pop("comparison_preset_name", None)
        preset_config.update(overrides)
        preset_config["comparison_preset_name"] = preset_name
        preset_result_root = base_result_root / safe_name
        preset_config["result_root"] = str(preset_result_root)
        preset_cache_root = base_cache_root / safe_name
        preset_run_cache_path = preset_cache_root / "analysis_run_cache.npz"
        preset_results_cache_path = preset_cache_root / "analysis_results_cache.npz"
        preset_tables_cache_path = preset_cache_root / "analysis_tables_cache.npz"
        preset_config["cache_path"] = str(shared_source_cache_path)
        preset_config["analysis_run_cache_path"] = str(preset_run_cache_path)
        preset_config["analysis_tables_cache_path"] = str(preset_tables_cache_path)
        preset_config["analysis_results_cache_path"] = str(preset_results_cache_path)
        preset_rebuild = bool(config.get("rebuild")) if preset_index == 0 else False
        preset_config["rebuild"] = preset_rebuild
        preset_config["source_cache_rebuild"] = preset_rebuild
        preset_config["analysis_tables_rebuild"] = preset_rebuild
        preset_config["analysis_results_rebuild"] = True
        preset_config["shared_shuffle_cache_rebuild"] = preset_rebuild
        preset_config["generate_poster_ready_figures"] = False
        _stage("comparison preset", f"{preset_name} -> {preset_result_root}")
        manifests.append(run_pipeline(preset_config))
        preset_configs[preset_name] = preset_config

    if plan.reference_preset_name in preset_configs:
        final_config = copy.deepcopy(preset_configs[plan.reference_preset_name])
        final_config["plots_only"] = True
        final_config["poster_ready_only"] = True
        final_config["generate_poster_ready_figures"] = True
        final_config["plots_only_include_supporting_figures"] = False
        _stage("comparison poster readback", f"{plan.reference_preset_name} -> {base_result_root}")
        manifests.append(run_pipeline(final_config))
    return manifests


def build_day_groups(expids_by_mode: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, List[str]]]:
    from analysis.compartment_common import grouped_experiments_by_day

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
    roi_ids = list(ctx.soma.roi_ids()) if compartment == "soma" else list(ctx.bouton.roi_ids())
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
        roi_id = roi_ids[roi_index] if roi_index < len(roi_ids) else roi_index
        global_soma_id = make_global_soma_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=channel, roi_id=roi_id)
        global_bouton_id = make_global_bouton_id(animal_id=ctx.animal_id, day_id=ctx.day_id, channel=channel, roi_id=roi_id)
        unit_id = global_soma_id if compartment == "soma" else global_bouton_id if compartment == "bouton" else ""
        row = {
            "expid": ctx.expid,
            "mode": ctx.mode,
            "animal_id": ctx.animal_id,
            "date": ctx.date,
            "day_id": ctx.day_id,
            "channel": int(channel),
            "compartment": compartment,
            "roi_index": int(roi_index),
            "roi_id": roi_id,
            "unit_id": unit_id,
            "roi_key": unit_id,
            "soma_id": roi_id if compartment == "soma" else None,
            "bouton_id": roi_id if compartment == "bouton" else None,
            "global_soma_id": global_soma_id if compartment == "soma" else None,
            "global_bouton_id": global_bouton_id if compartment == "bouton" else None,
            "response_metric": summary.get(response_metric, response_metric),
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


def _row_roi_lookup_key(row: Mapping[str, Any]) -> tuple[str, str]:
    compartment = str(row.get("compartment") or "").strip().lower()
    unit_id = row.get("global_soma_id") if compartment == "soma" else row.get("global_bouton_id") if compartment == "bouton" else None
    return compartment, str(unit_id).strip() if unit_id is not None else ""


def _coerce_visual_response_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.bool_)):
        try:
            return bool(int(value))
        except Exception:
            return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def _canonical_visual_response_cohort_map(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str], str]:
    cohort_by_key: Dict[tuple[str, str], str] = {}
    for row in rows:
        key = _row_roi_lookup_key(row)
        if _coerce_visual_response_bool(row.get("responsive", False)) or str(row.get("cohort") or "").strip().lower() == "responsive":
            cohort_by_key[key] = "responsive"
        elif key not in cohort_by_key:
            cohort_by_key[key] = "nonresponsive"
    return cohort_by_key


def _assign_visual_response_cohorts(rows: Sequence[Mapping[str, Any]], visual_response_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lookup = _canonical_visual_response_cohort_map(visual_response_rows)
    assigned: List[Dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        row_copy["cohort"] = lookup.get(_row_roi_lookup_key(row_copy), "nonresponsive")
        assigned.append(row_copy)
    return assigned






def _reload_plot_rows_from_csv(
    result_root: Path,
    *,
    plots_only: bool,
    activity_rows: List[Dict[str, Any]],
    correlation_rows: List[Dict[str, Any]],
    soma_pairwise_rows: List[Dict[str, Any]],
    bouton_pairwise_rows: List[Dict[str, Any]],
    lag_rows: List[Dict[str, Any]],
    visual_response_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not plots_only:
        return activity_rows, correlation_rows, soma_pairwise_rows, bouton_pairwise_rows, lag_rows, visual_response_rows
    preferred_csv_root = result_root / "all_requested_comparisons" / "csv"
    csv_root = preferred_csv_root if preferred_csv_root.exists() else result_root / "csv"
    if not activity_rows:
        activity_csv = csv_root / "state_activity_by_experiment.csv"
        if not activity_csv.exists():
            raise SystemExit(f"Missing poster-ready activity CSV: {activity_csv}")
        activity_rows = read_csv_rows(activity_csv)
    if not correlation_rows:
        correlation_csv = csv_root / "bouton_soma_correlation_by_roi.csv"
        if not correlation_csv.exists():
            raise SystemExit(f"Missing poster-ready correlation CSV: {correlation_csv}")
        correlation_rows = read_csv_rows(correlation_csv)
    if not soma_pairwise_rows:
        soma_pairwise_csv = csv_root / "soma_pairwise_correlation_by_roi.csv"
        if soma_pairwise_csv.exists():
            soma_pairwise_rows = read_csv_rows(soma_pairwise_csv)
    if not bouton_pairwise_rows:
        bouton_pairwise_csv = csv_root / "bouton_pairwise_correlation_by_roi.csv"
        if bouton_pairwise_csv.exists():
            bouton_pairwise_rows = read_csv_rows(bouton_pairwise_csv)
    if not lag_rows:
        lag_csv = csv_root / "bouton_soma_lag_scan_by_roi.csv"
        if not lag_csv.exists():
            raise SystemExit(f"Missing poster-ready lag CSV: {lag_csv}")
        lag_rows = read_csv_rows(lag_csv)
    if not visual_response_rows:
        visual_csv = csv_root / "visual_response_by_roi.csv"
        if not visual_csv.exists():
            raise SystemExit(f"Missing poster-ready visual-response CSV: {visual_csv}")
        visual_response_rows = read_csv_rows(visual_csv)
    return activity_rows, correlation_rows, soma_pairwise_rows, bouton_pairwise_rows, lag_rows, visual_response_rows


def _load_pairwise_family_rows_from_csv(result_root: Path) -> Dict[str, List[Dict[str, Any]]] | None:
    csv_root = result_root / "csv"
    correlation_csv = csv_root / "bouton_soma_correlation_by_roi.csv"
    soma_pairwise_csv = csv_root / "soma_pairwise_correlation_by_roi.csv"
    bouton_pairwise_csv = csv_root / "bouton_pairwise_correlation_by_roi.csv"
    if not (correlation_csv.exists() and soma_pairwise_csv.exists() and bouton_pairwise_csv.exists()):
        return None
    correlation_rows = read_csv_rows(correlation_csv)
    soma_pairwise_rows = read_csv_rows(soma_pairwise_csv)
    bouton_pairwise_rows = read_csv_rows(bouton_pairwise_csv)
    return {
        "correlation_rows": correlation_rows,
        "soma_pairwise_rows": soma_pairwise_rows,
        "bouton_pairwise_rows": bouton_pairwise_rows,
        "correlation_summary_rows": correlation_summary_rows(correlation_rows),
        "soma_pairwise_summary_rows": pairwise_correlation_summary_rows(soma_pairwise_rows),
        "bouton_pairwise_summary_rows": pairwise_correlation_summary_rows(bouton_pairwise_rows),
    }


def _normalize_scoped_unit_ids(rows: Sequence[Mapping[str, Any]], *, soma_channel: int, bouton_channel: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        compartment = str(row_copy.get("compartment") or "").strip().lower()
        channel_value = row_copy.get("channel")
        if channel_value in {None, ""}:
            if compartment == "soma":
                channel_value = soma_channel
            elif compartment == "bouton":
                channel_value = bouton_channel
            if channel_value not in {None, ""}:
                row_copy["channel"] = int(channel_value)
        roi_id = row_copy.get("roi_id")
        if compartment == "soma":
            global_id = str(row_copy.get("global_soma_id") or "").strip()
            if not global_id:
                global_id = make_global_soma_id(animal_id=row_copy.get("animal_id"), day_id=row_copy.get("day_id"), channel=row_copy.get("channel", soma_channel), roi_id=roi_id)
            row_copy["global_soma_id"] = global_id
            row_copy["global_bouton_id"] = row_copy.get("global_bouton_id") if str(row_copy.get("global_bouton_id") or "").strip() else None
            if not str(row_copy.get("unit_id") or "").strip():
                row_copy["unit_id"] = global_id
            if not str(row_copy.get("roi_key") or "").strip():
                row_copy["roi_key"] = row_copy["unit_id"]
        elif compartment == "bouton":
            global_id = str(row_copy.get("global_bouton_id") or "").strip()
            if not global_id:
                global_id = make_global_bouton_id(animal_id=row_copy.get("animal_id"), day_id=row_copy.get("day_id"), channel=row_copy.get("channel", bouton_channel), roi_id=roi_id)
            row_copy["global_bouton_id"] = global_id
            row_copy["global_soma_id"] = row_copy.get("global_soma_id") if str(row_copy.get("global_soma_id") or "").strip() else None
            if not str(row_copy.get("unit_id") or "").strip():
                row_copy["unit_id"] = global_id
            if not str(row_copy.get("roi_key") or "").strip():
                row_copy["roi_key"] = row_copy["unit_id"]
        normalized.append(row_copy)
    return normalized


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
        "compartment_states": list(config.get("compartment_states") or config.get("basal_apical_states") or []),
        "movie_expids": list(config.get("movie_expids") or []),
        "sleep_expids": list(config.get("sleep_expids") or []),
        "soma_channel": int(config.get("soma_channel", 1)),
        "bouton_channel": int(config.get("bouton_channel", 0)),
        "pairwise_correlation_schema_version": 2,
    }


def _source_cache_path(config: Mapping[str, Any], repo_root: Path, result_root: Path) -> Path:
    if config.get("cache_path"):
        return resolve_repo_path(config["cache_path"], repo_root)
    cache_root = resolve_repo_path(config.get("cache_root") or (result_root / "cache"), repo_root)
    return cache_root / "source_cache.npz"


def _analysis_run_cache_path(config: Mapping[str, Any], repo_root: Path, result_root: Path) -> Path:
    if config.get("analysis_run_cache_path"):
        return resolve_repo_path(config["analysis_run_cache_path"], repo_root)
    cache_root = resolve_repo_path(config.get("cache_root") or (result_root / "cache"), repo_root)
    preset_name = safe_filename_component(str(config.get("comparison_preset_name") or "default"))
    return cache_root / f"{preset_name}_analysis_run_cache.npz"


def _analysis_tables_cache_path(config: Mapping[str, Any], repo_root: Path, result_root: Path) -> Path:
    if config.get("analysis_tables_cache_path"):
        return resolve_repo_path(config["analysis_tables_cache_path"], repo_root)
    analysis_run_cache_file = _analysis_run_cache_path(config, repo_root, result_root)
    return analysis_run_cache_file.with_name(f"{analysis_run_cache_file.stem}_analysis_tables_cache.npz")


def _analysis_results_cache_path(config: Mapping[str, Any], repo_root: Path, result_root: Path) -> Path:
    if config.get("analysis_results_cache_path"):
        return resolve_repo_path(config["analysis_results_cache_path"], repo_root)
    analysis_run_cache_file = _analysis_run_cache_path(config, repo_root, result_root)
    return analysis_run_cache_file.with_name(f"{analysis_run_cache_file.stem}_analysis_results_cache.npz")


def run_pipeline(config: Mapping[str, Any]) -> Dict[str, Any]:
    repo_root = resolve_repo_root(Path(__file__))
    result_root = resolve_repo_path(config["result_root"], repo_root)
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
    analysis_run_cache_file = _analysis_run_cache_path(config, repo_root, result_root)
    analysis_results_cache_file = _analysis_results_cache_path(config, repo_root, result_root)
    analysis_tables_cache_file = _analysis_tables_cache_path(config, repo_root, result_root)
    source_cache_file = _source_cache_path(config, repo_root, result_root)
    experiment_rows: List[Dict[str, Any]] = []
    activity_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    soma_pairwise_rows: List[Dict[str, Any]] = []
    bouton_pairwise_rows: List[Dict[str, Any]] = []
    lag_rows: List[Dict[str, Any]] = []
    visual_response_rows: List[Dict[str, Any]] = []
    selected_states_by_mode: Dict[str, List[str]] = {mode: list(resolve_analysis_state_selections(config, mode)) for mode in state_modes}
    selected_states_by_mode_payload = {mode: list(states) for mode, states in selected_states_by_mode.items()}
    day_groups = build_day_groups(expids_by_mode)
    analysis_results_meta = _soma_analysis_results_meta(
        config,
        selected_states_by_mode,
        state_modes,
        visual_response_cohort,
        event_detection_method,
        visual_response_metric,
    )
    pairwise_family_cache_file = family_results_cache_path(analysis_results_cache_file, "pairwise_correlation")
    pairwise_family_meta = {
        "analysis_name": str(config.get("analysis_name") or "soma_bouton_pipeline"),
        "comparison_preset_name": str(config.get("comparison_preset_name") or "default"),
        "state_modes": list(state_modes),
        "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
        "soma_channel": int(config.get("soma_channel", 1)),
        "bouton_channel": int(config.get("bouton_channel", 0)),
        "pairwise_correlation_schema_version": 2,
        "family_result_stage": "pairwise_correlation",
    }
    pairwise_family_cache: Dict[str, Any] | None = None
    pairwise_family_rows: Dict[str, List[Dict[str, Any]]] | None = None
    if not bool(config.get("rebuild")):
        pairwise_family_cache, pairwise_family_status = load_family_results_cache(
            pairwise_family_cache_file,
            expected_meta=pairwise_family_meta,
            rebuild=bool(config.get("analysis_results_rebuild")),
        )
        if pairwise_family_status == "ok" and isinstance(pairwise_family_cache, dict):
            pairwise_family_rows = dict(pairwise_family_cache.get("analysis_results", {}))
            correlation_rows = list(pairwise_family_rows.get("correlation_rows", []))
            soma_pairwise_rows = list(pairwise_family_rows.get("soma_pairwise_rows", []))
            bouton_pairwise_rows = list(pairwise_family_rows.get("bouton_pairwise_rows", []))
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
                    "analysis_run_cache_path": str(analysis_run_cache_file),
                    "analysis_results_cache_path": str(analysis_results_cache_file),
                    "analysis_results_cache_reused": True,
                    "analysis_tables_cache_path": str(analysis_tables_cache_file),
                    "analysis_tables_cache_reused": True,
                    "source_cache_path": str(source_cache_file),
                }
            )
            manifest.setdefault("run_parameters", {})
            manifest["run_parameters"].update(
                {
                    "analysis_run_cache_path": str(analysis_run_cache_file),
                    "analysis_results_cache_path": str(analysis_results_cache_file),
                    "analysis_tables_cache_path": str(analysis_tables_cache_file),
                    "cache_path": str(source_cache_file),
                    "analysis_results_cache_reused": True,
                }
            )
            _stage("cache load", f"reused analysis-results cache at {analysis_results_cache_file}")
            if bool(config.get("poster_ready_only")):
                config = dict(config)
                config["plots_only"] = True
            if not bool(config.get("plots_only")) and pairwise_family_rows is None:
                pairwise_family_rows = _load_pairwise_family_rows_from_csv(result_root)
                if pairwise_family_rows is not None:
                    correlation_rows = list(pairwise_family_rows.get("correlation_rows", []))
                    soma_pairwise_rows = list(pairwise_family_rows.get("soma_pairwise_rows", []))
                    bouton_pairwise_rows = list(pairwise_family_rows.get("bouton_pairwise_rows", []))
                    save_family_results_cache(
                        pairwise_family_cache_file,
                        "pairwise_correlation",
                        pairwise_family_rows,
                        base_meta=pairwise_family_meta,
                    )
            if not bool(config.get("plots_only")) and pairwise_family_rows is not None:
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
            if pairwise_family_rows is None:
                correlation_rows.extend(bouton_soma_correlation_rows(ctx, selected_states))
                soma_pairwise_rows.extend(soma_pairwise_correlation_rows(ctx, selected_states))
                bouton_pairwise_rows.extend(bouton_pairwise_correlation_rows(ctx, selected_states))
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
            f"{mode}: experiments={len(expids_by_mode.get(mode, []))}, activity_rows={len(activity_rows)}, correlation_rows={len(correlation_rows)}, soma_pairwise_rows={len(soma_pairwise_rows)}, bouton_pairwise_rows={len(bouton_pairwise_rows)}, lag_rows={len(lag_rows)}, visual_response_rows={len(visual_response_rows)}",
        )

    activity_rows, correlation_rows, soma_pairwise_rows, bouton_pairwise_rows, lag_rows, visual_response_rows = _reload_plot_rows_from_csv(
        result_root,
        plots_only=bool(config.get("plots_only")),
        activity_rows=activity_rows,
        correlation_rows=correlation_rows,
        soma_pairwise_rows=soma_pairwise_rows,
        bouton_pairwise_rows=bouton_pairwise_rows,
        lag_rows=lag_rows,
        visual_response_rows=visual_response_rows,
    )
    activity_rows = _normalize_scoped_unit_ids(activity_rows, soma_channel=int(config["soma_channel"]), bouton_channel=int(config["bouton_channel"]))
    correlation_rows = _normalize_scoped_unit_ids(correlation_rows, soma_channel=int(config["soma_channel"]), bouton_channel=int(config["bouton_channel"]))
    soma_pairwise_rows = _normalize_scoped_unit_ids(soma_pairwise_rows, soma_channel=int(config["soma_channel"]), bouton_channel=int(config["bouton_channel"]))
    bouton_pairwise_rows = _normalize_scoped_unit_ids(bouton_pairwise_rows, soma_channel=int(config["soma_channel"]), bouton_channel=int(config["bouton_channel"]))
    lag_rows = _normalize_scoped_unit_ids(lag_rows, soma_channel=int(config["soma_channel"]), bouton_channel=int(config["bouton_channel"]))
    visual_response_rows = _normalize_scoped_unit_ids(visual_response_rows, soma_channel=int(config["soma_channel"]), bouton_channel=int(config["bouton_channel"]))

    visual_response_day_rows = shared_visual_response_day_rows(visual_response_rows)
    visual_response_family = run_visual_response_family(
        visual_response_rows,
        cohort=visual_response_cohort,
        response_metric=visual_response_metric,
    )

    activity_summary_rows = state_summary_rows(activity_rows)
    state_comparison_summary_rows = state_comparison_rows(
        [row for row in activity_rows if str(row.get("mode") or "") == "movie"],
        selected_states_by_mode.get("movie", []),
        shuffle_n,
    )
    sleep_state_comparison_summary_rows = state_comparison_rows(
        [row for row in activity_rows if str(row.get("mode") or "") == "sleep"],
        selected_states_by_mode.get("sleep", []),
        shuffle_n,
    )
    state_event_comparison_summary_rows = state_comparison_rows(
        [row for row in activity_rows if str(row.get("mode") or "") == "movie"],
        selected_states_by_mode.get("movie", []),
        shuffle_n,
        metric_col="event_frequency_per_min",
    )
    sleep_state_event_comparison_summary_rows = state_comparison_rows(
        [row for row in activity_rows if str(row.get("mode") or "") == "sleep"],
        selected_states_by_mode.get("sleep", []),
        shuffle_n,
        metric_col="event_frequency_per_min",
    )
    correlation_summary = correlation_summary_rows(correlation_rows)
    soma_pairwise_summary = pairwise_correlation_summary_rows(soma_pairwise_rows)
    bouton_pairwise_summary = pairwise_correlation_summary_rows(bouton_pairwise_rows)
    lag_summary = lag_summary_rows(lag_rows)
    if pairwise_family_rows is None:
        save_family_results_cache(
            pairwise_family_cache_file,
            "pairwise_correlation",
            {
                "correlation_rows": correlation_rows,
                "soma_pairwise_rows": soma_pairwise_rows,
                "bouton_pairwise_rows": bouton_pairwise_rows,
                "correlation_summary_rows": correlation_summary,
                "soma_pairwise_summary_rows": soma_pairwise_summary,
                "bouton_pairwise_summary_rows": bouton_pairwise_summary,
            },
            base_meta=pairwise_family_meta,
        )

    activity_rows_with_cohort = _assign_visual_response_cohorts(activity_rows, visual_response_rows)
    correlation_rows_with_cohort = _assign_visual_response_cohorts(correlation_rows, visual_response_rows)
    lag_rows_with_cohort = _assign_visual_response_cohorts(lag_rows, visual_response_rows)
    soma_pairwise_rows_with_cohort = assign_pairwise_visual_response_cohorts(soma_pairwise_rows, visual_response_rows)
    bouton_pairwise_rows_with_cohort = assign_pairwise_visual_response_cohorts(bouton_pairwise_rows, visual_response_rows)
    analysis_state_order: List[str] = []
    seen_analysis_states: set[str] = set()
    for mode in state_modes:
        for state in selected_states_by_mode.get(mode, []):
            state_key = str(state)
            if state_key and state_key not in seen_analysis_states:
                analysis_state_order.append(state_key)
                seen_analysis_states.add(state_key)

    cohort_activity_rows = split_rows_by_cohort(activity_rows_with_cohort)
    cohort_correlation_rows = split_rows_by_cohort(correlation_rows_with_cohort)
    cohort_soma_pairwise_rows = split_rows_by_cohort(soma_pairwise_rows_with_cohort)
    cohort_bouton_pairwise_rows = split_rows_by_cohort(bouton_pairwise_rows_with_cohort)
    cohort_lag_rows = split_rows_by_cohort(lag_rows_with_cohort)
    cohort_state_comparison_rows: Dict[str, List[Dict[str, Any]]] = {}
    cohort_state_event_comparison_rows: Dict[str, List[Dict[str, Any]]] = {}
    cohort_correlation_summary: Dict[str, List[Dict[str, Any]]] = {}
    cohort_soma_pairwise_summary: Dict[str, List[Dict[str, Any]]] = {}
    cohort_bouton_pairwise_summary: Dict[str, List[Dict[str, Any]]] = {}
    mixed_model_results: Dict[str, Any] = {}
    for cohort_name in ("all", "responsive", "nonresponsive"):
        cohort_rows = cohort_activity_rows.get(cohort_name, [])
        if not cohort_rows:
            continue
        cohort_state_comparison_rows[cohort_name] = state_comparison_rows(
            cohort_rows,
            analysis_state_order,
            shuffle_n,
        )
        cohort_state_event_comparison_rows[cohort_name] = state_comparison_rows(
            cohort_rows,
            analysis_state_order,
            shuffle_n,
            metric_col="event_frequency_per_min",
        )
        cohort_correlation_summary[cohort_name] = correlation_summary_rows(cohort_correlation_rows.get(cohort_name, []))
        cohort_soma_pairwise_summary[cohort_name] = correlation_summary_rows(cohort_soma_pairwise_rows.get(cohort_name, []))
        cohort_bouton_pairwise_summary[cohort_name] = correlation_summary_rows(cohort_bouton_pairwise_rows.get(cohort_name, []))
        cohort_state_order = list(analysis_state_order)
        mixed_model_results[cohort_name] = run_mixed_model_family(
            cohort_rows,
            state_comparison_states=cohort_state_order,
            basal_apical_states=cohort_state_order,
            shuffle_n=shuffle_n,
            mixed_model_contrast_p_source=str(config.get("mixed_model_contrast_p_source") or "classical"),
        )
    mixed_model_contrast_count = sum(
        len(branch.get("contrast_rows", []))
        for cohort_results in (mixed_model_results or {}).values()
        if isinstance(cohort_results, dict)
        for compartment_results in cohort_results.values()
        if isinstance(compartment_results, dict)
        for branch in compartment_results.values()
        if isinstance(branch, dict)
    ) if isinstance(mixed_model_results, dict) else 0
    _stage(
        "summary counts",
        f"activity={len(activity_summary_rows)}, movie_comparisons={len(state_comparison_summary_rows)}, sleep_comparisons={len(sleep_state_comparison_summary_rows)}, movie_event_comparisons={len(state_event_comparison_summary_rows)}, sleep_event_comparisons={len(sleep_state_event_comparison_summary_rows)}, correlation={len(correlation_summary)}, soma_pairwise={len(soma_pairwise_summary)}, bouton_pairwise={len(bouton_pairwise_summary)}, lag={len(lag_summary)}, visual_response={len(visual_response_rows)}, mixed_model={mixed_model_contrast_count}",
    )

    poster_ready_only = bool(config.get("poster_ready_only"))

    _stage("writing csv", "experiments")
    write_csv_rows(result_root / "csv" / "experiments.csv", experiment_rows, list(experiment_rows[0].keys()) if experiment_rows else ["expid"])
    if activity_rows:
        _stage("writing csv", "state_activity_by_experiment")
        write_csv_rows(result_root / "csv" / "state_activity_by_experiment.csv", activity_rows, ["expid", "mode", "animal_id", "date", "day_id", "channel", "state", "state_display", "state_color", "compartment", "roi_index", "roi_id", "unit_id", "roi_key", "soma_id", "bouton_id", "global_soma_id", "global_bouton_id", "n", "mean", "median", "std", "min", "max", "event_count", "event_frequency_per_min"])
    if correlation_rows:
        _stage("writing csv", "bouton_soma_correlation_by_roi")
        write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_roi.csv", correlation_rows, list(correlation_rows[0].keys()))
    if soma_pairwise_rows:
        _stage("writing csv", "soma_pairwise_correlation_by_roi")
        write_csv_rows(result_root / "csv" / "soma_pairwise_correlation_by_roi.csv", soma_pairwise_rows, list(soma_pairwise_rows[0].keys()))
    if bouton_pairwise_rows:
        _stage("writing csv", "bouton_pairwise_correlation_by_roi")
        write_csv_rows(result_root / "csv" / "bouton_pairwise_correlation_by_roi.csv", bouton_pairwise_rows, list(bouton_pairwise_rows[0].keys()))
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
    if state_event_comparison_summary_rows:
        _stage("writing csv", "state_event_comparisons_movie")
        write_csv_rows(result_root / "csv" / "state_event_comparisons_movie.csv", state_event_comparison_summary_rows, list(state_event_comparison_summary_rows[0].keys()))
    if sleep_state_event_comparison_summary_rows:
        _stage("writing csv", "state_event_comparisons_sleep")
        write_csv_rows(result_root / "csv" / "state_event_comparisons_sleep.csv", sleep_state_event_comparison_summary_rows, list(sleep_state_event_comparison_summary_rows[0].keys()))
    if correlation_summary:
        _stage("writing csv", "bouton_soma_correlation_by_day")
        write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_day.csv", correlation_summary, list(correlation_summary[0].keys()))
        for cohort_name in ("all", "responsive", "nonresponsive"):
            cohort_rows = cohort_correlation_summary.get(cohort_name, [])
            if not cohort_rows:
                continue
            cohort_csv_dir = ensure_dir(result_root / "csv" / "cohort" / cohort_name)
            write_csv_rows(cohort_csv_dir / "bouton_soma_correlation_by_day.csv", cohort_rows, list(cohort_rows[0].keys()))
    if soma_pairwise_summary:
        _stage("writing csv", "soma_pairwise_correlation_by_day")
        write_csv_rows(result_root / "csv" / "soma_pairwise_correlation_by_day.csv", soma_pairwise_summary, list(soma_pairwise_summary[0].keys()))
        for cohort_name in ("all", "responsive", "nonresponsive"):
            cohort_rows = cohort_soma_pairwise_summary.get(cohort_name, [])
            if not cohort_rows:
                continue
            cohort_csv_dir = ensure_dir(result_root / "csv" / "cohort" / cohort_name)
            write_csv_rows(cohort_csv_dir / "soma_pairwise_correlation_by_day.csv", cohort_rows, list(cohort_rows[0].keys()))
    if bouton_pairwise_summary:
        _stage("writing csv", "bouton_pairwise_correlation_by_day")
        write_csv_rows(result_root / "csv" / "bouton_pairwise_correlation_by_day.csv", bouton_pairwise_summary, list(bouton_pairwise_summary[0].keys()))
        for cohort_name in ("all", "responsive", "nonresponsive"):
            cohort_rows = cohort_bouton_pairwise_summary.get(cohort_name, [])
            if not cohort_rows:
                continue
            cohort_csv_dir = ensure_dir(result_root / "csv" / "cohort" / cohort_name)
            write_csv_rows(cohort_csv_dir / "bouton_pairwise_correlation_by_day.csv", cohort_rows, list(cohort_rows[0].keys()))
    if lag_summary:
        _stage("writing csv", "bouton_soma_lag_summary_by_day")
        write_csv_rows(result_root / "csv" / "bouton_soma_lag_summary_by_day.csv", lag_summary, list(lag_summary[0].keys()))
        for cohort_name in ("all", "responsive", "nonresponsive"):
            cohort_rows = cohort_lag_rows.get(cohort_name, [])
            if not cohort_rows:
                continue
            cohort_csv_dir = ensure_dir(result_root / "csv" / "cohort" / cohort_name)
            write_csv_rows(cohort_csv_dir / "bouton_soma_lag_scan_by_roi.csv", cohort_rows, list(cohort_rows[0].keys()))
            cohort_lag_summary_rows = lag_summary_rows(cohort_rows)
            if cohort_lag_summary_rows:
                write_csv_rows(cohort_csv_dir / "bouton_soma_lag_summary_by_day.csv", cohort_lag_summary_rows, list(cohort_lag_summary_rows[0].keys()))
    if visual_response_rows:
        _stage("writing csv", "visual_response_by_roi")
        write_csv_rows(result_root / "csv" / "visual_response_by_roi.csv", visual_response_rows, ["expid", "mode", "animal_id", "date", "day_id", "channel", "compartment", "roi_index", "roi_id", "unit_id", "roi_key", "soma_id", "bouton_id", "global_soma_id", "global_bouton_id", "response_metric", "event_detection_method", "source_label", "source_path", "available", "comparison", "statistic", "raw_pvalue", "adjusted_pvalue", "n_visual_values", "n_blank_values", "mean_visual", "mean_blank", "delta", "paired_stimulus_values", "blank_reference_values", "visual_trial_labels", "blank_trial_labels", "significant", "star", "responsive", "cohort", "cohort_requested"])
    if visual_response_day_rows:
        _stage("writing csv", "visual_response_by_day")
        write_csv_rows(result_root / "csv" / "visual_response_by_day.csv", visual_response_day_rows, list(visual_response_day_rows[0].keys()))

    if not poster_ready_only:
        _stage("plotting", "state activity")
        for cohort_name in ("all", "responsive", "nonresponsive"):
            cohort_rows = cohort_activity_rows.get(cohort_name, [])
            if not cohort_rows:
                continue
            plot_state_activity(
                cohort_rows,
                result_root,
                comparison_rows=cohort_state_comparison_rows.get(cohort_name, []),
                cohort_label=cohort_name,
            )
            _stage("plotting", f"state event frequency - {cohort_name}")
            plot_state_event_frequency(
                cohort_rows,
                result_root,
                comparison_rows=cohort_state_event_comparison_rows.get(cohort_name, []),
                cohort_label=cohort_name,
            )
            _stage("plotting", f"axon-soma correlation - {cohort_name}")
            plot_state_correlation(
                cohort_correlation_summary.get(cohort_name, []),
                result_root,
                comparison_rows=cohort_state_comparison_rows.get(cohort_name, []),
                cohort_label=cohort_name,
                title="Axon-soma correlation",
                output_stem="axon_soma_state_summary_boxplots_correlation",
            )
            _stage("plotting", f"soma-soma correlation - {cohort_name}")
            plot_state_correlation(
                cohort_soma_pairwise_summary.get(cohort_name, []),
                result_root,
                cohort_label=cohort_name,
                title="Soma-soma correlation",
                output_stem="soma_pairwise_state_summary_boxplots_correlation",
            )
            _stage("plotting", f"axon-axon correlation - {cohort_name}")
            plot_state_correlation(
                cohort_bouton_pairwise_summary.get(cohort_name, []),
                result_root,
                cohort_label=cohort_name,
                title="Axon-axon correlation",
                output_stem="axon_axon_state_summary_boxplots_correlation",
            )
        for cohort_name in ("all", "responsive", "nonresponsive"):
            cohort_rows = cohort_lag_rows.get(cohort_name, [])
            if not cohort_rows:
                continue
            _stage("plotting", f"lag heatmap - {cohort_name}")
            plot_lag_heatmap(cohort_rows, result_root, cohort_label=cohort_name)
        if visual_response_rows:
            visual_response_fig_dir = ensure_dir(result_root / "figures" / "visual_response")
            for compartment in ("soma", "bouton"):
                compartment_rows = [row for row in visual_response_rows if str(row.get("compartment") or "") == compartment]
                if not compartment_rows:
                    continue
                compartment_dir = ensure_dir(visual_response_fig_dir / compartment)
                for cohort in ("all", "responsive", "nonresponsive"):
                    cohort_rows = compartment_rows if cohort == "all" else [row for row in compartment_rows if str(row.get("cohort") or "nonresponsive") == cohort]
                    if not cohort_rows:
                        continue
                    cohort_dir = ensure_dir(compartment_dir / cohort)
                    plot_visual_response_boxplot_figure(
                        {"rows": cohort_rows},
                        cohort_dir,
                        output_name="visual_response_movie_vs_blank.svg",
                        title=f"{compartment.capitalize()} visual response - {cohort.capitalize()}",
                        cohort_label=cohort,
                        kind=compartment,
                    )
                    render_visual_response_entity_figures(
                        cohort_rows,
                        cohort_dir / "entities",
                        cohort_label=cohort,
                        kind=compartment,
                    )
        if not poster_ready_only and mixed_model_results:
            mixed_model_fig_dir = ensure_dir(result_root / "figures" / "mixed_model")
            for cohort_name, cohort_results in mixed_model_results.items():
                if not isinstance(cohort_results, dict) or not cohort_results:
                    continue
                cohort_dir = ensure_dir(mixed_model_fig_dir / cohort_name)
                for compartment_key, compartment_results in cohort_results.items():
                    if not isinstance(compartment_results, dict) or not compartment_results:
                        continue
                    compartment_dir = ensure_dir(cohort_dir / compartment_key)
                    for scope_key in ("all_state", "selected_state"):
                        branch = compartment_results.get(scope_key, {})
                        scope_label = scope_key.replace("_", " ")
                        if not isinstance(branch, dict) or not branch:
                            continue
                        mixed_model_payload = {
                            "analysis_state_selection": {
                                "state_comparison_states": list(analysis_state_order),
                                "compartment_states": list(analysis_state_order),
                            },
                            "analysis_compartment": compartment_key,
                            "mixed_model": branch,
                        }
                        plot_mixed_model_forest_figure(
                            mixed_model_payload,
                            compartment_dir,
                            output_name=f"mixed_model_{cohort_name}_{compartment_key}_{scope_key}_forest.svg",
                            title=f"Mixed-model fixed effects - {cohort_name} - {compartment_key} - {scope_label}",
                            model_key="mixed_model",
                        )
                        plot_mixed_model_predicted_means_figure(
                            mixed_model_payload,
                            compartment_dir,
                            output_name=f"mixed_model_{cohort_name}_{compartment_key}_{scope_key}_predicted_means.svg",
                            title=f"Mixed-model predicted means - {cohort_name} - {compartment_key} - {scope_label}",
                            model_key="mixed_model",
                        )
                        plot_mixed_model_contrasts_checkpoint(
                            mixed_model_payload,
                            compartment_dir,
                            scope=scope_key,
                            output_name=f"mixed_model_{cohort_name}_{compartment_key}_{scope_key}_contrasts.svg",
                            title=f"Mixed-model contrasts - {cohort_name} - {compartment_key} - {scope_label}",
                            model_key="mixed_model",
                        )

    poster_ready_figures: List[str] = []
    poster_ready_errors: List[str] = []
    if bool(config.get("generate_poster_ready_figures", True)):
        poster_output_dir = ensure_dir(REPO_ROOT / "results" / "poster_ready")

        def _state_values_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
            grouped: Dict[str, Dict[str, List[float]]] = {}
            for row in rows:
                state = canonical_state_label(row.get("state") or row.get("state_label") or row.get("state_display") or "")
                if not state:
                    continue
                value = row.get("mean")
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if not np.isfinite(value_f):
                    continue
                compartment = str(row.get("compartment") or "").strip().lower()
                if compartment == "soma":
                    unit_id = str(row.get("global_soma_id") or "").strip()
                elif compartment == "bouton":
                    unit_id = str(row.get("global_bouton_id") or "").strip()
                else:
                    unit_id = str(row.get("global_soma_id") or row.get("global_bouton_id") or "").strip()
                if not unit_id:
                    continue
                grouped.setdefault(state, {}).setdefault(unit_id, []).append(value_f)
            return {state: [float(np.nanmean(values)) for values in unit_map.values() if values] for state, unit_map in grouped.items()}

        def _state_sample_sizes_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
            grouped: Dict[str, set[str]] = {}
            for row in rows:
                state = canonical_state_label(row.get("state") or row.get("state_label") or row.get("state_display") or "")
                if not state:
                    continue
                compartment = str(row.get("compartment") or "").strip().lower()
                if compartment == "soma":
                    unit_id = str(row.get("global_soma_id") or "").strip()
                elif compartment == "bouton":
                    unit_id = str(row.get("global_bouton_id") or "").strip()
                else:
                    unit_id = str(row.get("global_soma_id") or row.get("global_bouton_id") or "").strip()
                if not unit_id:
                    continue
                grouped.setdefault(state, set()).add(unit_id)
            return {state: len(roi_ids) for state, roi_ids in grouped.items()}

        def _state_row_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            unit_keys = []
            state_keys = []
            cohort_keys = []
            raw_count = 0
            non_nan_count = 0
            for row in rows:
                raw_count += 1
                state = canonical_state_label(row.get("state") or row.get("state_label") or row.get("state_display") or "")
                if state:
                    state_keys.append(state)
                cohort = str(row.get("cohort") or "").strip().lower()
                if cohort:
                    cohort_keys.append(cohort)
                compartment = str(row.get("compartment") or "").strip().lower()
                if compartment == "soma":
                    unit_id = str(row.get("global_soma_id") or "").strip()
                elif compartment == "bouton":
                    unit_id = str(row.get("global_bouton_id") or "").strip()
                else:
                    unit_id = str(row.get("global_soma_id") or row.get("global_bouton_id") or "").strip()
                if unit_id:
                    unit_keys.append(unit_id)
                try:
                    value = float(row.get("mean"))
                except Exception:
                    continue
                if np.isfinite(value):
                    non_nan_count += 1
            return {
                "raw_count": raw_count,
                "non_nan_count": non_nan_count,
                "unique_roi_count": len(set(unit_keys)),
                "states": list(dict.fromkeys(state_keys)),
                "cohorts": list(dict.fromkeys(cohort_keys)),
            }
        poster_state_order = ["quiet_awake_blank", "quiet_awake_movies", "quiet_awake", "nrem", "rem"]
        blank_state_order = ["quiet_awake_blank", "nrem_blank", "rem_blank"]
        movie_state_order = ["quiet_awake_movies", "nrem_movies", "rem_movies"]
        mixed_model_contrast_p_source = str(config.get("mixed_model_contrast_p_source") or "classical")
        preset_result_root = Path(config.get("result_root") or DEFAULT_CONFIG["result_root"])
        if not preset_result_root.is_absolute():
            preset_result_root = REPO_ROOT / preset_result_root
        if preset_result_root.name != "all_requested_comparisons" and not (preset_result_root / "blank_state_comparisons").exists() and (preset_result_root.parent / "blank_state_comparisons").exists():
            preset_result_root = preset_result_root.parent

        def _load_preset_csv_rows(preset_name: str, csv_name: str) -> List[Dict[str, Any]]:
            candidate_paths = []
            for root in (preset_result_root, preset_result_root.parent):
                candidate_paths.append(root / preset_name / "csv" / csv_name)
                if root.name == "all_requested_comparisons":
                    candidate_paths.append(root / "csv" / csv_name)
            for csv_path in candidate_paths:
                if csv_path.exists():
                    return read_csv_rows(csv_path)
            logger.warning("Missing preset CSV %s; tried %s", csv_name, ", ".join(str(path) for path in candidate_paths))
            return []

        def _load_all_requested_comparison_rows(csv_name: str) -> List[Dict[str, Any]]:
            candidate_paths = []
            for root in (preset_result_root, preset_result_root.parent):
                if root.name == "all_requested_comparisons":
                    candidate_paths.append(root / "csv" / csv_name)
                candidate_paths.append(root / "all_requested_comparisons" / "csv" / csv_name)
            for csv_path in candidate_paths:
                if csv_path.exists():
                    return read_csv_rows(csv_path)
            logger.warning("Missing all_requested_comparisons CSV %s; tried %s", csv_name, ", ".join(str(path) for path in candidate_paths))
            return []

        blank_preset_activity_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("blank_state_comparisons", "state_activity_by_experiment.csv"), visual_response_rows)
        movie_preset_activity_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("movies_state_comparisons", "state_activity_by_experiment.csv"), visual_response_rows)
        blank_preset_comparison_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("blank_state_comparisons", "state_comparisons_movie.csv"), visual_response_rows)
        movie_preset_comparison_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("movies_state_comparisons", "state_comparisons_movie.csv"), visual_response_rows)

        for compartment in ("soma", "bouton"):
            compartment_root = ensure_dir(poster_output_dir / compartment)
            visual_dir = ensure_dir(compartment_root / "visual_response")
            mixed_dir = ensure_dir(compartment_root / "mixed_model")
            blank_dir = ensure_dir(compartment_root / "blank_movie_states")
            compartment_visual_rows = [row for row in visual_response_rows if str(row.get("compartment") or "") == compartment]
            if not compartment_visual_rows:
                continue
            visual_path = write_visual_response_poster_figure(
                output_dir=visual_dir,
                entity_label=compartment,
                visual_response_rows=compartment_visual_rows,
                output_stem=f"{compartment}_visual_response_poster_ready",
            )
            if visual_path:
                poster_ready_figures.append(str(visual_path))
            responsive_rows = [row for row in cohort_activity_rows.get("responsive", []) if str(row.get("compartment") or "") == compartment]
            nonresponsive_rows = [row for row in cohort_activity_rows.get("nonresponsive", []) if str(row.get("compartment") or "") == compartment]
            responsive_values = _state_values_from_rows(responsive_rows)
            nonresponsive_values = _state_values_from_rows(nonresponsive_rows)
            responsive_sample_sizes = _state_sample_sizes_from_rows(responsive_rows)
            nonresponsive_sample_sizes = _state_sample_sizes_from_rows(nonresponsive_rows)
            blank_responsive_values = {state: values for state, values in responsive_values.items() if state in blank_state_order}
            blank_nonresponsive_values = {state: values for state, values in nonresponsive_values.items() if state in blank_state_order}
            movie_responsive_values = {state: values for state, values in responsive_values.items() if state in movie_state_order}
            movie_nonresponsive_values = {state: values for state, values in nonresponsive_values.items() if state in movie_state_order}
            blank_responsive_sample_sizes = {state: count for state, count in responsive_sample_sizes.items() if state in blank_state_order}
            blank_nonresponsive_sample_sizes = {state: count for state, count in nonresponsive_sample_sizes.items() if state in blank_state_order}
            movie_responsive_sample_sizes = {state: count for state, count in responsive_sample_sizes.items() if state in movie_state_order}
            movie_nonresponsive_sample_sizes = {state: count for state, count in nonresponsive_sample_sizes.items() if state in movie_state_order}
            if responsive_values or nonresponsive_values:
                print(f"[poster] {compartment} responsive blank/movie n summary = {_state_row_summary([row for row in responsive_rows if canonical_state_label(row.get('state') or row.get('state_label') or row.get('state_display') or '') in blank_state_order + movie_state_order])}", file=sys.stderr)
                print(f"[poster] {compartment} nonresponsive blank/movie n summary = {_state_row_summary([row for row in nonresponsive_rows if canonical_state_label(row.get('state') or row.get('state_label') or row.get('state_display') or '') in blank_state_order + movie_state_order])}", file=sys.stderr)
                mixed_path = write_state_mixed_model_poster_figure(
                    output_dir=mixed_dir,
                    entity_label=compartment,
                    responsive_state_values=responsive_values,
                    nonresponsive_state_values=nonresponsive_values,
                    mixed_model_rows={
                        "responsive": mixed_model_results.get("responsive", {}).get(compartment, {}).get("all_state", {})
                        or mixed_model_results.get("responsive", {}).get(compartment, {}).get("selected_state", {}),
                        "nonresponsive": mixed_model_results.get("nonresponsive", {}).get(compartment, {}).get("all_state", {})
                        or mixed_model_results.get("nonresponsive", {}).get(compartment, {}).get("selected_state", {}),
                    },
                    state_order=poster_state_order,
                    output_stem=f"{compartment}_state_mixed_model_poster_ready",
                    title="Quiet blank vs sleep states",
                    preferred_response_keys=(("mean_dendrite_activity", "mean") if compartment == "soma" else ("mean_spine_activity_per_dendrite", "mean", "mean_dendrite_activity")),
                    mixed_model_contrast_p_source=mixed_model_contrast_p_source,
                )
                if mixed_path:
                    poster_ready_figures.append(str(mixed_path))
            responsive_significant_states = set()
            nonresponsive_significant_states = set()
            for cohort_label, source, target in (
                ("responsive", mixed_model_results.get("responsive", {}).get(compartment, {}).get("all_state", {})
                or mixed_model_results.get("responsive", {}).get(compartment, {}).get("selected_state", {}), responsive_significant_states),
                ("nonresponsive", mixed_model_results.get("nonresponsive", {}).get(compartment, {}).get("all_state", {})
                or mixed_model_results.get("nonresponsive", {}).get(compartment, {}).get("selected_state", {}), nonresponsive_significant_states),
            ):
                selected_rows = _select_mixed_model_rows(source, preferred_response_keys=("mean_dendrite_activity", "mean") if compartment == "soma" else ("mean_spine_activity_per_dendrite", "mean", "mean_dendrite_activity"))
                target.update(_poster_mixed_model_significant_states(selected_rows))
            responsive_comparison_rows = [
                row for row in cohort_state_comparison_rows.get("responsive", [])
                if str(row.get("compartment") or "all").strip().lower() in {"all", compartment}
            ]
            nonresponsive_comparison_rows = [
                row for row in cohort_state_comparison_rows.get("nonresponsive", [])
                if str(row.get("compartment") or "all").strip().lower() in {"all", compartment}
            ]
            blank_comparison_rows = [
                row for row in blank_preset_comparison_rows
                if str(row.get("compartment") or "all").strip().lower() in {"all", compartment}
            ]
            movie_comparison_rows = [
                row for row in movie_preset_comparison_rows
                if str(row.get("compartment") or "all").strip().lower() in {"all", compartment}
            ]
            responsive_significant_states.update(_significant_state_labels_from_comparison_rows(responsive_comparison_rows))
            nonresponsive_significant_states.update(_significant_state_labels_from_comparison_rows(nonresponsive_comparison_rows))
            responsive_significant_states.update(_significant_state_labels_from_comparison_rows(blank_comparison_rows))
            responsive_significant_states.update(_significant_state_labels_from_comparison_rows(movie_comparison_rows))
            nonresponsive_significant_states.update(_significant_state_labels_from_comparison_rows(blank_comparison_rows))
            nonresponsive_significant_states.update(_significant_state_labels_from_comparison_rows(movie_comparison_rows))
            blank_path = write_blank_movie_state_boxplot_figure(
                output_dir=blank_dir,
                entity_label=compartment,
                responsive_blank_values=blank_responsive_values,
                responsive_movie_values=movie_responsive_values,
                nonresponsive_blank_values=blank_nonresponsive_values,
                nonresponsive_movie_values=movie_nonresponsive_values,
                blank_state_order=blank_state_order,
                movie_state_order=movie_state_order,
                responsive_blank_sample_sizes=blank_responsive_sample_sizes,
                responsive_movie_sample_sizes=movie_responsive_sample_sizes,
                nonresponsive_blank_sample_sizes=blank_nonresponsive_sample_sizes,
                nonresponsive_movie_sample_sizes=movie_nonresponsive_sample_sizes,
                responsive_significant_states=sorted(responsive_significant_states),
                nonresponsive_significant_states=sorted(nonresponsive_significant_states),
                responsive_comparison_rows=blank_comparison_rows + movie_comparison_rows,
                nonresponsive_comparison_rows=blank_comparison_rows + movie_comparison_rows,
                output_stem=f"{compartment}_blank_movie_states_poster_ready",
                title="Blank vs movie states",
            )
            if blank_path:
                poster_ready_figures.append(str(blank_path))
            blank_expected_states = [state for state in blank_state_order if "rem" not in state]
            movie_expected_states = [state for state in movie_state_order if "rem" not in state]
            blank_missing_states = [state for state in blank_expected_states if state not in blank_responsive_values and state not in blank_nonresponsive_values]
            movie_missing_states = [state for state in movie_expected_states if state not in movie_responsive_values and state not in movie_nonresponsive_values]
            if blank_missing_states or movie_missing_states:
                poster_ready_errors.append(
                    f"[ERROR] {compartment} blank/movie poster missing states: blank_missing={blank_missing_states or '[]'}; movie_missing={movie_missing_states or '[]'}"
                )

            correlation_dir = ensure_dir(compartment_root / "correlation")
            all_correlation_path: str | None = None
            for cohort_name in ("all", "responsive", "nonresponsive"):
                if cohort_name == "all":
                    cohort_rows = list(_load_all_requested_comparison_rows("cohort/all/bouton_soma_correlation_by_day.csv"))
                    cohort_soma_pairwise_rows = list(_load_all_requested_comparison_rows("cohort/all/soma_pairwise_correlation_by_day.csv"))
                    cohort_bouton_pairwise_rows = list(_load_all_requested_comparison_rows("cohort/all/bouton_pairwise_correlation_by_day.csv"))
                else:
                    cohort_rows = list(cohort_correlation_rows.get(cohort_name, []))
                    cohort_soma_pairwise_rows = list(cohort_soma_pairwise_summary.get(cohort_name, []))
                    cohort_bouton_pairwise_rows = list(cohort_bouton_pairwise_summary.get(cohort_name, []))
                if cohort_rows:
                    cohort_summary_rows = cohort_rows if cohort_name == "all" else correlation_summary_rows(cohort_rows)
                    if cohort_summary_rows:
                        comparison_rows = [
                            row for row in cohort_state_comparison_rows.get(cohort_name, [])
                            if str(row.get("compartment") or "all").strip().lower() in {"all", compartment}
                        ]
                        correlation_path = write_correlation_poster_figure(
                            output_dir=ensure_dir(correlation_dir / cohort_name),
                            entity_label=compartment,
                            correlation_rows=cohort_summary_rows,
                            comparison_rows=comparison_rows,
                            output_stem=f"{compartment}_state_summary_boxplots_correlation_poster_ready",
                            title="Correlation summaries",
                            figure_label="Axon-soma",
                        )
                        if correlation_path:
                            poster_ready_figures.append(str(correlation_path))
                            if cohort_name == "all":
                                all_correlation_path = str(correlation_path)
                if cohort_soma_pairwise_rows:
                    soma_pairwise_path = write_correlation_poster_figure(
                        output_dir=ensure_dir(correlation_dir / cohort_name / "soma_pairwise"),
                        entity_label="soma",
                        correlation_rows=cohort_soma_pairwise_rows,
                        output_stem="soma_pairwise_state_summary_boxplots_correlation_poster_ready",
                        title="Correlation summaries",
                        figure_label="Soma-soma",
                    )
                    if soma_pairwise_path:
                        poster_ready_figures.append(str(soma_pairwise_path))
                if cohort_bouton_pairwise_rows:
                    bouton_pairwise_path = write_correlation_poster_figure(
                        output_dir=ensure_dir(correlation_dir / cohort_name / "bouton_pairwise"),
                        entity_label="bouton",
                        correlation_rows=cohort_bouton_pairwise_rows,
                        output_stem="axon_axon_state_summary_boxplots_correlation_poster_ready",
                        title="Correlation summaries",
                        figure_label="Axon-axon",
                    )
                    if bouton_pairwise_path:
                        poster_ready_figures.append(str(bouton_pairwise_path))
            if compartment == "bouton":
                combined_path = write_blank_movie_and_correlation_poster_figure(
                    output_dir=blank_dir,
                    entity_label=compartment,
                    responsive_blank_values=blank_responsive_values,
                    responsive_movie_values=movie_responsive_values,
                    nonresponsive_blank_values=blank_nonresponsive_values,
                    nonresponsive_movie_values=movie_nonresponsive_values,
                    blank_state_order=blank_state_order,
                    movie_state_order=movie_state_order,
                    responsive_blank_sample_sizes=blank_responsive_sample_sizes,
                    responsive_movie_sample_sizes=movie_responsive_sample_sizes,
                    nonresponsive_blank_sample_sizes=blank_nonresponsive_sample_sizes,
                    nonresponsive_movie_sample_sizes=movie_nonresponsive_sample_sizes,
                    responsive_significant_states=sorted(responsive_significant_states),
                    nonresponsive_significant_states=sorted(nonresponsive_significant_states),
                    responsive_comparison_rows=blank_comparison_rows + movie_comparison_rows,
                    nonresponsive_comparison_rows=blank_comparison_rows + movie_comparison_rows,
                    responsive_compartment_significant_states=sorted(responsive_significant_states),
                    nonresponsive_compartment_significant_states=sorted(nonresponsive_significant_states),
                    correlation_rows=_load_all_requested_comparison_rows("cohort/all/bouton_soma_correlation_by_day.csv"),
                    correlation_comparison_rows=[
                        row for row in cohort_state_comparison_rows.get("all", [])
                        if str(row.get("compartment") or "all").strip().lower() in {"all", compartment}
                    ],
                    output_stem=f"{compartment}_blank_movie_states_with_correlation_poster_ready",
                    title="Blank vs movie states",
                    correlation_title="Axon-soma correlation summaries",
                )
                if combined_path:
                    poster_ready_figures.append(str(combined_path))

    canonical_visual_response_lookup = _canonical_visual_response_cohort_map(visual_response_rows)
    missing_visual_response_global_ids = [
        row for row in visual_response_rows
        if str(row.get("compartment") or "") == "soma" and not str(row.get("global_soma_id") or "").strip()
        or str(row.get("compartment") or "") == "bouton" and not str(row.get("global_bouton_id") or "").strip()
    ]
    if missing_visual_response_global_ids:
        sample_rows = []
        for row in missing_visual_response_global_ids[:20]:
            sample_rows.append(
                "|".join(
                    [
                        f"compartment={row.get('compartment')}",
                        f"expid={row.get('expid')}",
                        f"day_id={row.get('day_id')}",
                        f"channel={row.get('channel')}",
                        f"roi_id={row.get('roi_id')}",
                        f"roi_index={row.get('roi_index')}",
                        f"unit_id={row.get('unit_id')}",
                    ]
                )
            )
        print(
            f"[ERROR] {len(missing_visual_response_global_ids)} visual-response rows are missing global_soma_id/global_bouton_id; samples: "
            + "; ".join(sample_rows),
            file=sys.stderr,
        )

    def _missing_global_id_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        missing: List[Mapping[str, Any]] = []
        for row in rows:
            compartment = str(row.get("compartment") or "").strip().lower()
            if compartment == "soma":
                if not str(row.get("global_soma_id") or "").strip():
                    missing.append(row)
            elif compartment == "bouton":
                if not str(row.get("global_bouton_id") or "").strip():
                    missing.append(row)
        return missing

    missing_activity_global_ids = _missing_global_id_rows(cohort_activity_rows.get("responsive", []) + cohort_activity_rows.get("nonresponsive", []))
    missing_preset_global_ids = _missing_global_id_rows(blank_preset_activity_rows + movie_preset_activity_rows)
    if missing_activity_global_ids or missing_preset_global_ids:
        for label, rows_with_missing in (("activity", missing_activity_global_ids), ("preset", missing_preset_global_ids)):
            if not rows_with_missing:
                continue
            sample_rows = []
            for row in rows_with_missing[:20]:
                sample_rows.append(
                    "|".join(
                        [
                            f'compartment={row.get("compartment")}',
                            f'expid={row.get("expid")}',
                            f'day_id={row.get("day_id")}',
                            f'channel={row.get("channel")}',
                            f'roi_id={row.get("roi_id")}',
                            f'roi_index={row.get("roi_index")}',
                            f'unit_id={row.get("unit_id")}',
                        ]
                    )
                )
            print(
                f"[ERROR] {len(rows_with_missing)} {label} rows are missing compartment-matched global IDs; samples: "
                + "; ".join(sample_rows),
                file=sys.stderr,
            )

    def _visual_response_unique_count(rows: Sequence[Mapping[str, Any]], *, compartment: str | None = None, responsive: bool | None = None) -> int:
        unique_keys: set[tuple[str, str]] = set()
        for row in rows:
            if compartment is not None and str(row.get("compartment") or "") != compartment:
                continue
            row_responsive = canonical_visual_response_lookup.get(_row_roi_lookup_key(row), "nonresponsive") == "responsive"
            if responsive is True and not row_responsive:
                continue
            if responsive is False and row_responsive:
                continue
            unique_keys.add(_row_roi_lookup_key(row))
        return int(len(unique_keys))

    visual_response_counts = {
        "all": _visual_response_unique_count(visual_response_rows),
        "responsive": _visual_response_unique_count(visual_response_rows, responsive=True),
        "nonresponsive": _visual_response_unique_count(visual_response_rows, responsive=False),
    }
    manifest = {
        "config": dict(config),
        "comparison_preset_name": preset_name,
        "selected_states_by_mode": selected_states_by_mode_payload,
        "state_modes": state_modes,
        "day_groups": day_groups,
        "counts": {
            "experiments": len(experiment_rows),
            "activity_rows": len(activity_rows),
            "state_comparison_rows_movie": len(state_comparison_summary_rows),
            "state_comparison_rows_sleep": len(sleep_state_comparison_summary_rows),
            "correlation_rows": len(correlation_rows),
            "soma_pairwise_correlation_rows": len(soma_pairwise_rows),
            "bouton_pairwise_correlation_rows": len(bouton_pairwise_rows),
            "lag_rows": len(lag_rows),
            "visual_response_rows": len(visual_response_rows),
            "visual_response_responsive_rows": visual_response_counts["responsive"],
            "visual_response_nonresponsive_rows": visual_response_counts["nonresponsive"],
            "cohort_activity_rows": {
                cohort_name: len(rows) for cohort_name, rows in cohort_activity_rows.items()
            },
            "cohort_state_comparison_rows": {
                cohort_name: len(rows) for cohort_name, rows in cohort_state_comparison_rows.items()
            },
            "cohort_state_event_comparison_rows": {
                cohort_name: len(rows) for cohort_name, rows in cohort_state_event_comparison_rows.items()
            },
            "cohort_correlation_summary_rows": {
                cohort_name: len(rows) for cohort_name, rows in cohort_correlation_summary.items()
            },
            "soma_pairwise_summary_rows": len(soma_pairwise_summary),
            "bouton_pairwise_summary_rows": len(bouton_pairwise_summary),
            "cohort_lag_rows": {
                cohort_name: len(rows) for cohort_name, rows in cohort_lag_rows.items()
            },
            "cohort_mixed_model_contrasts": {
                cohort_name: sum(
                    len(branch.get("contrast_rows", []))
                    for compartment_results in cohort_results.values()
                    if isinstance(compartment_results, dict)
                    for branch in compartment_results.values()
                    if isinstance(branch, dict)
                )
                for cohort_name, cohort_results in mixed_model_results.items()
                if isinstance(cohort_results, dict)
            },
        },
        "event_detection_method": str(config.get("event_detection_method") or "amplitude"),
        "visual_response_metric": str(config.get("visual_response_metric") or "calcium_events"),
        "visual_response_cohort": visual_response_cohort,
        "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "visual_response_family": visual_response_family,
        "visual_response_cohort_ids": {
            "soma": {
                "responsive": list(dict.fromkeys(key[1] for key, cohort in canonical_visual_response_lookup.items() if key[0] == "soma" and cohort == "responsive")),
                "nonresponsive": list(dict.fromkeys(key[1] for key, cohort in canonical_visual_response_lookup.items() if key[0] == "soma" and cohort != "responsive")),
            },
            "bouton": {
                "responsive": list(dict.fromkeys(key[1] for key, cohort in canonical_visual_response_lookup.items() if key[0] == "bouton" and cohort == "responsive")),
                "nonresponsive": list(dict.fromkeys(key[1] for key, cohort in canonical_visual_response_lookup.items() if key[0] == "bouton" and cohort != "responsive")),
            },
        },
        "mixed_model": mixed_model_results,
        "poster_ready_figures": list(poster_ready_figures),
        "cache_summary": {
            "analysis_run_cache_path": str(analysis_run_cache_file),
            "analysis_results_cache_path": str(analysis_results_cache_file),
            "analysis_results_cache_reused": False,
            "analysis_tables_cache_path": str(analysis_tables_cache_file),
            "analysis_tables_cache_reused": False,
            "source_cache_path": str(source_cache_file),
        },
        "output_root": str(result_root),
    }
    manifest_json = _json_safe(manifest)
    analysis_tables_payload = {
        "schema_version": ANALYSIS_TABLE_CACHE_SCHEMA_VERSION,
        "meta": analysis_results_meta,
        "meta_hash": analysis_cache_meta_hash(analysis_results_meta),
        "analysis_tables": {
            "experiment_rows": experiment_rows,
            "activity_rows": activity_rows,
            "correlation_rows": correlation_rows,
            "lag_rows": lag_rows,
            "visual_response_rows": visual_response_rows,
            "activity_summary_rows": activity_summary_rows,
            "state_comparison_summary_rows": state_comparison_summary_rows,
            "sleep_state_comparison_summary_rows": sleep_state_comparison_summary_rows,
            "correlation_summary": correlation_summary,
            "lag_summary": lag_summary,
            "visual_response_day_rows": visual_response_day_rows,
            "visual_response_counts": visual_response_counts,
            "selected_states_by_mode": selected_states_by_mode,
            "state_modes": state_modes,
            "day_groups": day_groups,
        },
    }
    save_analysis_tables_cache(analysis_tables_cache_file, analysis_tables_payload)
    write_json_file(result_root / "summary" / "manifest.json", manifest_json)
    save_analysis_results_cache(
        analysis_results_cache_file,
        {
            "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
            "meta": analysis_results_meta,
            "meta_hash": analysis_cache_meta_hash(analysis_results_meta),
            "analysis_results": analysis_results_cache_payload(manifest_json),
        },
    )
    if poster_ready_errors:
        for error in poster_ready_errors:
            print(error, file=sys.stderr)
    if poster_ready_figures:
        _stage("poster-ready outputs", f"{len(poster_ready_figures)} figures written")
        for figure_path in poster_ready_figures:
            print(f"  - {figure_path}")
    _stage("completed", f"{preset_name} with {len(experiment_rows)} experiments")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Compare soma (ch2) and bouton (ch1) activity across movie and sleep states.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("soma_bouton_pipeline_config.json"))
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding outputs even if caches exist.")
    parser.add_argument("--plots-only", action="store_true", help="Skip metric recomputation and regenerate plots from written CSVs only.")
    parser.add_argument(
        "--poster-ready-only",
        action="store_true",
        help="Compute only the stats needed for poster-ready figures and skip the regular plot pass.",
    )
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
    if args.poster_ready_only:
        config["poster_ready_only"] = True
    if args.comparison_presets:
        config["comparison_preset_names"] = list(args.comparison_presets)
    if config.get("comparison_presets"):
        run_comparison_preset_runs(config)
        return 0
    run_pipeline(config)
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
