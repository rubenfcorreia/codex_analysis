from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.compartment_common import (  # noqa: E402
    derive_animal_id,
    derive_date,
    ensure_dir,
    experiment_root_from_expid,
    filter_comparison_presets,
    grouped_experiments_by_day,
    normalize_comparison_presets,
    read_csv_rows,
    resolve_analysis_state_selections,
    resolve_repo_root,
    safe_filename_component,
    stable_hash,
    write_csv_rows,
    write_json_file,
)
from analysis.soma_bouton_pipeline.analysis_families.core import (  # noqa: E402
    ExperimentContext,
    build_experiment_context,
    experiment_summary_row,
)
from analysis.soma_bouton_pipeline.analysis_families.correlation import (  # noqa: E402
    bouton_pairwise_correlation_rows,
    bouton_soma_correlation_rows,
    correlation_summary_rows,
    soma_pairwise_correlation_rows,
)
from analysis.soma_bouton_pipeline.analysis_families.lag import lag_scan_rows, lag_summary_rows  # noqa: E402
from analysis.soma_bouton_pipeline.analysis_families.state import activity_rows_for_context, state_summary_rows  # noqa: E402
from analysis.soma_bouton_pipeline.plots import plot_lag_heatmap, plot_state_activity, plot_state_correlation  # noqa: E402
from analysis.shared.plots.poster_ready import assign_pairwise_visual_response_cohorts, split_rows_by_cohort  # noqa: E402


CACHE_SCHEMA_VERSION = 2
logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "analysis_name": "soma_bouton_pipeline_cache",
    "result_root": "results/soma_bouton_pipeline",
    "cache_root": "results/soma_bouton_pipeline/cache",
    "movie_expids": [],
    "sleep_expids": [],
    "soma_channel": 1,
    "bouton_channel": 0,
    "lag_window_s": 2.0,
    "lag_step_s": 0.1,
    "rebuild": True,
    "plots_only": False,
    "comparison_presets": None,
    "comparison_preset_name": None,
    "comparison_preset_names": None,
    "state_mode": "both",
    "movie_states": ["running", "still", "all"],
    "sleep_states": ["nrem", "rem", "wake", "all"],
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
    base_cache_root = Path(config.get("cache_root") or (base_result_root / "cache"))
    logger.info("Running %d soma/bouton preset(s) with shared cache root %s", len(presets), base_cache_root)
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
        preset_config["cache_root"] = str(base_cache_root)
        logger.info("Preset %s -> result_root=%s", preset_name, preset_result_root)
        manifests.append(run_pipeline(preset_config))
    return manifests


def dump_gzip_pickle(path: Path | str, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with gzip.open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_gzip_pickle(path: Path | str) -> Any:
    with gzip.open(Path(path), "rb") as fh:
        return pickle.load(fh)


def _mode_expids(config: Mapping[str, Any]) -> Dict[str, List[str]]:
    return {
        "movie": list(config.get("movie_expids", [])),
        "sleep": list(config.get("sleep_expids", [])),
    }


def _state_modes(config: Mapping[str, Any], expids_by_mode: Mapping[str, Sequence[str]]) -> List[str]:
    state_modes = [mode for mode in ("movie", "sleep") if expids_by_mode.get(mode)]
    if config.get("state_mode") in {"movie", "sleep"}:
        state_modes = [str(config["state_mode"])]
    return state_modes


def _movie_state_bundle_path(exp_root: Path) -> Path:
    candidates = sorted(exp_root.glob("*_all_trials.csv"))
    if not candidates:
        raise FileNotFoundError(f"No movie trial bundle found in {exp_root}")
    return candidates[0]


def _sleep_state_bundle_path(exp_root: Path) -> Path:
    path = exp_root / "sleep_score" / "sleep_state.pickle"
    if not path.exists():
        raise FileNotFoundError(f"No sleep state bundle found at {path}")
    return path


def _state_bundle_path(exp_root: Path, mode: str) -> Path:
    if mode == "movie":
        return _movie_state_bundle_path(exp_root)
    if mode == "sleep":
        return _sleep_state_bundle_path(exp_root)
    raise ValueError(f"Unknown mode {mode!r}")


def _source_paths(repo_root: Path, expid: str, mode: str, soma_channel: int, bouton_channel: int) -> Dict[str, Path]:
    animal_id = derive_animal_id(expid)
    exp_root = experiment_root_from_expid(repo_root, expid, animal_id=animal_id)
    return {
        "exp_root": exp_root,
        "state_bundle": _state_bundle_path(exp_root, mode),
        "soma": exp_root / "recordings" / f"s2p_ch{soma_channel}.pickle",
        "bouton": exp_root / "recordings" / f"s2p_ch{bouton_channel}.pickle",
    }


def _source_signature(paths: Mapping[str, Path]) -> Dict[str, Dict[str, int | str | None]]:
    signature: Dict[str, Dict[str, int | str | None]] = {}
    for label, path in paths.items():
        if path.exists():
            stat = path.stat()
            signature[label] = {
                "path": str(path),
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
        else:
            signature[label] = {
                "path": str(path),
                "mtime_ns": None,
                "size": None,
            }
    return signature


def _experiment_cache_key(config: Mapping[str, Any], expid: str, mode: str) -> str:
    return stable_hash(
        {
            "analysis_name": config.get("analysis_name"),
            "expid": expid,
            "mode": mode,
            "soma_channel": int(config.get("soma_channel", 1)),
            "bouton_channel": int(config.get("bouton_channel", 0)),
            "lag_window_s": float(config.get("lag_window_s", 2.0)),
            "lag_step_s": float(config.get("lag_step_s", 0.1)),
        }
    )


def _pipeline_cache_key(
    config: Mapping[str, Any],
    expids_by_mode: Mapping[str, Sequence[str]],
    state_modes: Sequence[str],
) -> str:
    return stable_hash(
        {
            "analysis_name": config.get("analysis_name"),
            "soma_channel": int(config.get("soma_channel", 1)),
            "bouton_channel": int(config.get("bouton_channel", 0)),
            "lag_window_s": float(config.get("lag_window_s", 2.0)),
            "lag_step_s": float(config.get("lag_step_s", 0.1)),
            "state_modes": list(state_modes),
            "expids_by_mode": {mode: list(expids) for mode, expids in expids_by_mode.items()},
        }
    )


def _experiment_cache_path(cache_root: Path, mode: str, expid: str, cache_key: str) -> Path:
    return ensure_dir(cache_root / "experiments" / mode) / f"{safe_filename_component(expid)}__{cache_key}.pkl.gz"


def _summary_cache_path(cache_root: Path, cache_key: str) -> Path:
    return ensure_dir(cache_root / "summary") / f"{cache_key}.pkl.gz"


def _experiment_cache_valid(payload: Mapping[str, Any], cache_key: str, source_signature: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema_version") == CACHE_SCHEMA_VERSION
        and payload.get("cache_key") == cache_key
        and payload.get("source_signature") == source_signature
        and isinstance(payload.get("summary"), dict)
    )


def _summary_cache_valid(payload: Mapping[str, Any], cache_key: str, source_signatures: Sequence[Mapping[str, Any]]) -> bool:
    return (
        payload.get("schema_version") == CACHE_SCHEMA_VERSION
        and payload.get("cache_key") == cache_key
        and payload.get("source_signatures") == list(source_signatures)
        and isinstance(payload.get("summary"), dict)
    )


def _build_experiment_payload(
    ctx: ExperimentContext,
    selected_states: Sequence[str],
    config: Mapping[str, Any],
    cache_key: str,
    source_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    activity_rows = activity_rows_for_context(ctx, selected_states)
    correlation_rows = bouton_soma_correlation_rows(ctx, selected_states)
    soma_pairwise_rows = soma_pairwise_correlation_rows(ctx, selected_states)
    bouton_pairwise_rows = bouton_pairwise_correlation_rows(ctx, selected_states)
    lag_rows = lag_scan_rows(
        ctx,
        selected_states,
        lag_window_s=float(config.get("lag_window_s", 2.0)),
        lag_step_s=float(config.get("lag_step_s", 0.1)),
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "source_signature": dict(source_signature),
        "experiment_row": experiment_summary_row(ctx),
        "summary": {
            "activity_rows": len(activity_rows),
            "correlation_rows": len(correlation_rows),
            "soma_pairwise_correlation_rows": len(soma_pairwise_rows),
            "bouton_pairwise_correlation_rows": len(bouton_pairwise_rows),
            "lag_rows": len(lag_rows),
        },
        "raw_data": {
            "state_bundle_path": str(ctx.state_bundle_path),
            "state_bundle": ctx.state_bundle,
            "soma": {"path": str(ctx.soma.path), "data": ctx.soma.data},
            "bouton": {"path": str(ctx.bouton.path), "data": ctx.bouton.data},
        },
        "activity_rows": activity_rows,
        "correlation_rows": correlation_rows,
        "soma_pairwise_rows": soma_pairwise_rows,
        "bouton_pairwise_rows": bouton_pairwise_rows,
        "lag_rows": lag_rows,
    }


def _load_or_build_experiment_payload(
    repo_root: Path,
    expid: str,
    mode: str,
    config: Mapping[str, Any],
    cache_root: Path,
    selected_states: Sequence[str],
    rebuild: bool,
) -> Dict[str, Any]:
    paths = _source_paths(repo_root, expid, mode, int(config["soma_channel"]), int(config["bouton_channel"]))
    current_signature = _source_signature(paths)
    exp_cache_key = _experiment_cache_key(config, expid, mode)
    cache_path = _experiment_cache_path(cache_root, mode, expid, exp_cache_key)
    if not rebuild and cache_path.exists():
        try:
            payload = load_gzip_pickle(cache_path)
            if _experiment_cache_valid(payload, exp_cache_key, current_signature):
                logger.info("Reused experiment cache for %s/%s from %s", expid, mode, cache_path)
                return payload
        except Exception:
            pass

    ctx = build_experiment_context(
        expid=expid,
        mode=mode,
        soma_channel=int(config["soma_channel"]),
        bouton_channel=int(config["bouton_channel"]),
        repo_root=repo_root,
    )
    payload = _build_experiment_payload(ctx, selected_states, config, exp_cache_key, current_signature)
    dump_gzip_pickle(cache_path, payload)
    logger.info("Built experiment cache for %s/%s at %s", expid, mode, cache_path)
    return payload


def _rows_from_csv_bundle(result_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    csv_root = result_root / "csv"

    def _load(name: str) -> List[Dict[str, Any]]:
        path = csv_root / name
        return [dict(row) for row in read_csv_rows(path)] if path.exists() else []

    return {
        "experiments": _load("experiments.csv"),
        "activity": _load("state_activity_by_experiment.csv"),
        "correlation": _load("bouton_soma_correlation_by_roi.csv"),
        "soma_pairwise": _load("soma_pairwise_correlation_by_roi.csv"),
        "bouton_pairwise": _load("bouton_pairwise_correlation_by_roi.csv"),
        "lag": _load("bouton_soma_lag_scan_by_roi.csv"),
        "activity_summary": _load("state_activity_by_day.csv"),
        "correlation_summary": _load("bouton_soma_correlation_by_day.csv"),
        "soma_pairwise_summary": _load("soma_pairwise_correlation_by_day.csv"),
        "bouton_pairwise_summary": _load("bouton_pairwise_correlation_by_day.csv"),
        "lag_summary": _load("bouton_soma_lag_summary_by_day.csv"),
        "visual_response": _load("visual_response_by_roi.csv"),
    }


def _load_summary_payload_from_csv(result_root: Path) -> Dict[str, Any]:
    rows = _rows_from_csv_bundle(result_root)
    activity_summary_rows = rows["activity_summary"] or state_summary_rows(rows["activity"])
    correlation_summary = rows["correlation_summary"] or correlation_summary_rows(rows["correlation"])
    soma_pairwise_summary = rows.get("soma_pairwise_summary") or correlation_summary_rows(rows["soma_pairwise"])
    bouton_pairwise_summary = rows.get("bouton_pairwise_summary") or correlation_summary_rows(rows["bouton_pairwise"])
    lag_summary = rows["lag_summary"] or lag_summary_rows(rows["lag"])
    return {
        "rows": {
            "experiments": rows["experiments"],
            "activity": rows["activity"],
            "correlation": rows["correlation"],
            "soma_pairwise": rows["soma_pairwise"],
            "bouton_pairwise": rows["bouton_pairwise"],
            "lag": rows["lag"],
        },
        "summary": {
            "activity_summary_rows": activity_summary_rows,
            "correlation_summary_rows": correlation_summary,
            "soma_pairwise_summary_rows": soma_pairwise_summary,
            "bouton_pairwise_summary_rows": bouton_pairwise_summary,
            "lag_summary_rows": lag_summary,
        },
        "counts": {
            "experiments": len(rows["experiments"]),
            "activity_rows": len(rows["activity"]),
            "correlation_rows": len(rows["correlation"]),
            "soma_pairwise_correlation_rows": len(rows["soma_pairwise"]),
            "bouton_pairwise_correlation_rows": len(rows["bouton_pairwise"]),
            "lag_rows": len(rows["lag"]),
        },
        "source_signatures": [],
        "day_groups": {},
    }


def _plot_pairwise_correlation_figures(
    result_root: Path,
    cohort_correlation_rows: Mapping[str, List[Dict[str, Any]]],
    cohort_soma_pairwise_rows: Mapping[str, List[Dict[str, Any]]],
    cohort_bouton_pairwise_rows: Mapping[str, List[Dict[str, Any]]],
) -> None:
    for cohort_name in ("all", "responsive", "nonresponsive"):
        if cohort_correlation_rows.get(cohort_name):
            plot_state_correlation(
                cohort_correlation_rows[cohort_name],
                result_root,
                cohort_label=cohort_name,
                title="Axon-soma correlation",
                output_stem="axon_soma_state_summary_boxplots_correlation",
            )
        if cohort_soma_pairwise_rows.get(cohort_name):
            plot_state_correlation(
                cohort_soma_pairwise_rows[cohort_name],
                result_root,
                cohort_label=cohort_name,
                title="Soma-soma correlation",
                output_stem="soma_pairwise_state_summary_boxplots_correlation",
            )
        if cohort_bouton_pairwise_rows.get(cohort_name):
            plot_state_correlation(
                cohort_bouton_pairwise_rows[cohort_name],
                result_root,
                cohort_label=cohort_name,
                title="Axon-axon correlation",
                output_stem="axon_axon_state_summary_boxplots_correlation",
            )


def _pairwise_rows_by_cohort(rows: Sequence[Mapping[str, Any]], visual_response_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return split_rows_by_cohort(assign_pairwise_visual_response_cohorts(rows, visual_response_rows))


def run_pipeline(config: Mapping[str, Any]) -> Dict[str, Any]:
    repo_root = resolve_repo_root(Path(__file__))
    result_root = Path(config["result_root"])
    if not result_root.is_absolute():
        result_root = repo_root / result_root
    cache_root = Path(config.get("cache_root", result_root / "cache"))
    if not cache_root.is_absolute():
        cache_root = repo_root / cache_root

    preset_name = str(config.get("comparison_preset_name") or "default")
    logger.info("Running preset %s -> result_root=%s cache_root=%s", preset_name, result_root, cache_root)

    ensure_dir(result_root)
    ensure_dir(cache_root)
    ensure_dir(result_root / "csv")
    ensure_dir(result_root / "figures")
    ensure_dir(result_root / "summary")

    expids_by_mode = _mode_expids(config)
    state_modes = _state_modes(config, expids_by_mode)
    selected_states_by_mode: Dict[str, List[str]] = {}
    for mode in state_modes:
        selected_states_by_mode[mode] = list(resolve_analysis_state_selections(config, mode))
    pipeline_key = _pipeline_cache_key(config, expids_by_mode, state_modes)
    summary_cache = _summary_cache_path(cache_root, pipeline_key)

    current_signatures: List[Dict[str, Any]] = []
    for mode in state_modes:
        for expid in expids_by_mode.get(mode, []):
            paths = _source_paths(repo_root, expid, mode, int(config["soma_channel"]), int(config["bouton_channel"]))
            current_signatures.append(_source_signature(paths))

    summary_cache_payload: Dict[str, Any] | None = None
    if summary_cache.exists() and not bool(config.get("rebuild", True)):
        try:
            payload = load_gzip_pickle(summary_cache)
            if _summary_cache_valid(payload, pipeline_key, current_signatures):
                summary_cache_payload = payload
                logger.info("Reused summary cache for preset %s from %s", preset_name, summary_cache)
        except Exception:
            pass

    if summary_cache_payload is not None:
        summary = summary_cache_payload["summary"]
        rows = summary_cache_payload["rows"]
        visual_response_rows = list(rows.get("visual_response", []))
        cohort_correlation_rows = split_rows_by_cohort(assign_pairwise_visual_response_cohorts(rows["correlation"], visual_response_rows)) if visual_response_rows else {"all": list(rows["correlation"]), "responsive": [], "nonresponsive": []}
        cohort_soma_pairwise_rows = _pairwise_rows_by_cohort(rows["soma_pairwise"], visual_response_rows) if visual_response_rows else {"all": list(rows["soma_pairwise"]), "responsive": [], "nonresponsive": []}
        cohort_bouton_pairwise_rows = _pairwise_rows_by_cohort(rows["bouton_pairwise"], visual_response_rows) if visual_response_rows else {"all": list(rows["bouton_pairwise"]), "responsive": [], "nonresponsive": []}
        write_csv_rows(result_root / "csv" / "experiments.csv", rows["experiments"], list(rows["experiments"][0].keys()) if rows["experiments"] else ["expid"])
        if rows["activity"]:
            write_csv_rows(result_root / "csv" / "state_activity_by_experiment.csv", rows["activity"], list(rows["activity"][0].keys()))
        if rows["correlation"]:
            write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_roi.csv", rows["correlation"], list(rows["correlation"][0].keys()))
        if rows["soma_pairwise"]:
            write_csv_rows(result_root / "csv" / "soma_pairwise_correlation_by_roi.csv", rows["soma_pairwise"], list(rows["soma_pairwise"][0].keys()))
        if rows["bouton_pairwise"]:
            write_csv_rows(result_root / "csv" / "bouton_pairwise_correlation_by_roi.csv", rows["bouton_pairwise"], list(rows["bouton_pairwise"][0].keys()))
        if rows["lag"]:
            write_csv_rows(result_root / "csv" / "bouton_soma_lag_scan_by_roi.csv", rows["lag"], list(rows["lag"][0].keys()))
        if summary["activity_summary_rows"]:
            write_csv_rows(result_root / "csv" / "state_activity_by_day.csv", summary["activity_summary_rows"], list(summary["activity_summary_rows"][0].keys()))
        if summary["correlation_summary_rows"]:
            write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_day.csv", summary["correlation_summary_rows"], list(summary["correlation_summary_rows"][0].keys()))
        if summary["soma_pairwise_summary_rows"]:
            write_csv_rows(result_root / "csv" / "soma_pairwise_correlation_by_day.csv", summary["soma_pairwise_summary_rows"], list(summary["soma_pairwise_summary_rows"][0].keys()))
        if summary["bouton_pairwise_summary_rows"]:
            write_csv_rows(result_root / "csv" / "bouton_pairwise_correlation_by_day.csv", summary["bouton_pairwise_summary_rows"], list(summary["bouton_pairwise_summary_rows"][0].keys()))
        if summary["lag_summary_rows"]:
            write_csv_rows(result_root / "csv" / "bouton_soma_lag_summary_by_day.csv", summary["lag_summary_rows"], list(summary["lag_summary_rows"][0].keys()))
        plot_state_activity(summary["activity_summary_rows"], result_root)
        _plot_pairwise_correlation_figures(result_root, cohort_correlation_rows, cohort_soma_pairwise_rows, cohort_bouton_pairwise_rows)
        plot_lag_heatmap(rows["lag"], result_root)
        manifest = {
            "config": dict(config),
            "comparison_preset_name": preset_name,
            "cache_key": pipeline_key,
            "state_modes": state_modes,
            "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
            "day_groups": summary_cache_payload.get("day_groups", {}),
            "counts": summary_cache_payload.get("counts", {}),
            "output_root": str(result_root),
            "cache_root": str(cache_root),
            "loaded_from": "summary_cache",
        }
        write_json_file(result_root / "summary" / "manifest.json", manifest)
        return manifest

    if config.get("plots_only"):
        payload = _load_summary_payload_from_csv(result_root)
        summary = payload["summary"]
        rows = payload["rows"]
        visual_response_rows = list(rows.get("visual_response", []))
        cohort_correlation_rows = split_rows_by_cohort(assign_pairwise_visual_response_cohorts(rows["correlation"], visual_response_rows)) if visual_response_rows else {"all": list(rows["correlation"]), "responsive": [], "nonresponsive": []}
        cohort_soma_pairwise_rows = _pairwise_rows_by_cohort(rows["soma_pairwise"], visual_response_rows) if visual_response_rows else {"all": list(rows["soma_pairwise"]), "responsive": [], "nonresponsive": []}
        cohort_bouton_pairwise_rows = _pairwise_rows_by_cohort(rows["bouton_pairwise"], visual_response_rows) if visual_response_rows else {"all": list(rows["bouton_pairwise"]), "responsive": [], "nonresponsive": []}
        plot_state_activity(summary["activity_summary_rows"], result_root)
        _plot_pairwise_correlation_figures(result_root, cohort_correlation_rows, cohort_soma_pairwise_rows, cohort_bouton_pairwise_rows)
        plot_lag_heatmap(rows["lag"], result_root)
        day_groups = grouped_experiments_by_day([row["expid"] for row in rows["experiments"]]) if rows["experiments"] else {}
        manifest = {
            "config": dict(config),
            "comparison_preset_name": preset_name,
            "cache_key": pipeline_key,
            "state_modes": state_modes,
            "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
            "day_groups": day_groups,
            "counts": payload["counts"],
            "output_root": str(result_root),
            "cache_root": str(cache_root),
            "loaded_from": "csv_cache",
        }
        write_json_file(result_root / "summary" / "manifest.json", manifest)
        return manifest

    experiment_payloads: List[Dict[str, Any]] = []
    experiment_rows: List[Dict[str, Any]] = []
    activity_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    soma_pairwise_rows: List[Dict[str, Any]] = []
    bouton_pairwise_rows: List[Dict[str, Any]] = []
    lag_rows: List[Dict[str, Any]] = []

    for mode in state_modes:
        selected_states = selected_states_by_mode[mode]
        for expid in expids_by_mode.get(mode, []):
            payload = _load_or_build_experiment_payload(
                repo_root=repo_root,
                expid=expid,
                mode=mode,
                config=config,
                cache_root=cache_root,
                selected_states=selected_states,
                rebuild=bool(config.get("rebuild", True)),
            )
            experiment_payloads.append(payload)
            experiment_rows.append(payload["experiment_row"])
            activity_rows.extend(payload["activity_rows"])
            correlation_rows.extend(payload["correlation_rows"])
            soma_pairwise_rows.extend(payload.get("soma_pairwise_rows", []))
            bouton_pairwise_rows.extend(payload.get("bouton_pairwise_rows", []))
            lag_rows.extend(payload["lag_rows"])

    activity_summary_rows = state_summary_rows(activity_rows)
    correlation_summary = correlation_summary_rows(correlation_rows)
    lag_summary = lag_summary_rows(lag_rows)
    day_groups = grouped_experiments_by_day([row["expid"] for row in experiment_rows])

    if not config.get("plots_only"):
        write_csv_rows(result_root / "csv" / "experiments.csv", experiment_rows, list(experiment_rows[0].keys()) if experiment_rows else ["expid"])
        if activity_rows:
            write_csv_rows(result_root / "csv" / "state_activity_by_experiment.csv", activity_rows, list(activity_rows[0].keys()))
        if correlation_rows:
            write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_roi.csv", correlation_rows, list(correlation_rows[0].keys()))
        if soma_pairwise_rows:
            write_csv_rows(result_root / "csv" / "soma_pairwise_correlation_by_roi.csv", soma_pairwise_rows, list(soma_pairwise_rows[0].keys()))
        if bouton_pairwise_rows:
            write_csv_rows(result_root / "csv" / "bouton_pairwise_correlation_by_roi.csv", bouton_pairwise_rows, list(bouton_pairwise_rows[0].keys()))
        if lag_rows:
            write_csv_rows(result_root / "csv" / "bouton_soma_lag_scan_by_roi.csv", lag_rows, list(lag_rows[0].keys()))
        if activity_summary_rows:
            write_csv_rows(result_root / "csv" / "state_activity_by_day.csv", activity_summary_rows, list(activity_summary_rows[0].keys()))
        if correlation_summary:
            write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_day.csv", correlation_summary, list(correlation_summary[0].keys()))
        if soma_pairwise_summary:
            write_csv_rows(result_root / "csv" / "soma_pairwise_correlation_by_day.csv", soma_pairwise_summary, list(soma_pairwise_summary[0].keys()))
        if bouton_pairwise_summary:
            write_csv_rows(result_root / "csv" / "bouton_pairwise_correlation_by_day.csv", bouton_pairwise_summary, list(bouton_pairwise_summary[0].keys()))
        if lag_summary:
            write_csv_rows(result_root / "csv" / "bouton_soma_lag_summary_by_day.csv", lag_summary, list(lag_summary[0].keys()))

    plot_state_activity(activity_summary_rows, result_root)
    _plot_pairwise_correlation_figures(
        result_root,
        {"all": list(correlation_rows), "responsive": [], "nonresponsive": []},
        {"all": list(soma_pairwise_rows), "responsive": [], "nonresponsive": []},
        {"all": list(bouton_pairwise_rows), "responsive": [], "nonresponsive": []},
    )
    plot_lag_heatmap(lag_rows, result_root)

    summary_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": pipeline_key,
        "counts": {
            "experiments": len(experiment_rows),
            "activity_rows": len(activity_rows),
            "correlation_rows": len(correlation_rows),
            "soma_pairwise_correlation_rows": len(soma_pairwise_rows),
            "bouton_pairwise_correlation_rows": len(bouton_pairwise_rows),
            "lag_rows": len(lag_rows),
        },
        "rows": {
            "experiments": experiment_rows,
            "activity": activity_rows,
            "correlation": correlation_rows,
            "soma_pairwise": soma_pairwise_rows,
            "bouton_pairwise": bouton_pairwise_rows,
            "lag": lag_rows,
        },
        "summary": {
            "activity_summary_rows": activity_summary_rows,
            "correlation_summary_rows": correlation_summary,
            "soma_pairwise_summary_rows": soma_pairwise_summary,
            "bouton_pairwise_summary_rows": bouton_pairwise_summary,
            "lag_summary_rows": lag_summary,
        },
        "day_groups": day_groups,
        "source_signatures": current_signatures,
    }
    dump_gzip_pickle(summary_cache, summary_payload)
    logger.info("Built summary cache for preset %s at %s", preset_name, summary_cache)

    cache_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "comparison_preset_name": preset_name,
        "cache_key": pipeline_key,
        "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
        "summary_cache": str(summary_cache),
        "cache_root": str(cache_root),
        "experiment_cache_count": len(experiment_payloads),
    }
    write_json_file(result_root / "summary" / "cache_manifest.json", cache_manifest)

    manifest = {
        "config": dict(config),
        "comparison_preset_name": preset_name,
        "cache_key": pipeline_key,
        "state_modes": state_modes,
        "selected_states_by_mode": {mode: list(states) for mode, states in selected_states_by_mode.items()},
        "day_groups": day_groups,
        "counts": summary_payload["counts"],
        "output_root": str(result_root),
        "cache_root": str(cache_root),
        "loaded_from": "rebuild" if bool(config.get("rebuild", True)) else "cache_or_rebuild",
    }
    write_json_file(result_root / "summary" / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Cache-enabled soma/bouton analysis pipeline.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("soma_bouton_pipeline") / "soma_bouton_pipeline_config.json")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding experiment and summary caches.")
    parser.add_argument("--plots-only", action="store_true", help="Load cached summaries or CSVs and only regenerate figures.")
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
