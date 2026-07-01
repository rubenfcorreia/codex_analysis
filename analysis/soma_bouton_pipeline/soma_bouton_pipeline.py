from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
from analysis.soma_bouton_pipeline.analysis_families.core import ExperimentContext, build_experiment_context, experiment_summary_row
from analysis.soma_bouton_pipeline.analysis_families.correlation import bouton_soma_correlation_rows, correlation_summary_rows
from analysis.soma_bouton_pipeline.analysis_families.lag import lag_scan_rows, lag_summary_rows
from analysis.soma_bouton_pipeline.analysis_families.state import activity_rows_for_context, state_comparison_rows, state_summary_rows
from analysis.soma_bouton_pipeline.plots import plot_lag_heatmap, plot_state_activity, plot_state_correlation
from analysis.compartment_common import resolve_analysis_state_selections


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
    "rebuild": True,
    "plots_only": False,
    "comparison_presets": None,
    "comparison_preset_name": None,
    "comparison_preset_names": None,
    "event_detection_method": "derivative",
    "visual_response_metric": "mean",
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
    "event_detection_method": "derivative",
    "visual_response_metric": "mean",
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
    _stage("config knobs", f"event_detection_method={config.get('event_detection_method', 'derivative')}, visual_response_metric={config.get('visual_response_metric', 'mean')}")

    shuffle_n = int(config.get("shuffle_n", 200))
    experiment_rows: List[Dict[str, Any]] = []
    activity_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    lag_rows: List[Dict[str, Any]] = []
    selected_states_by_mode: Dict[str, List[str]] = {}

    for mode in state_modes:
        selected_states = list(resolve_analysis_state_selections(config, mode))
        selected_states_by_mode[mode] = selected_states
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
        _stage("mode complete", f"{mode}: experiments={len(expids_by_mode.get(mode, []))}, activity_rows={len(activity_rows)}, correlation_rows={len(correlation_rows)}, lag_rows={len(lag_rows)}")

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
    correlation_summary = correlation_summary_rows(correlation_rows)
    lag_summary = lag_summary_rows(lag_rows)
    _stage("summary counts", f"activity={len(activity_summary_rows)}, movie_comparisons={len(state_comparison_summary_rows)}, sleep_comparisons={len(sleep_state_comparison_summary_rows)}, correlation={len(correlation_summary)}, lag={len(lag_summary)}")

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

    _stage("plotting", "state activity")
    plot_state_activity(activity_summary_rows or activity_rows, result_root)
    _stage("plotting", "correlation")
    plot_state_correlation(correlation_summary, result_root)
    _stage("plotting", "lag heatmap")
    plot_lag_heatmap(lag_rows, result_root)

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
        },
        "event_detection_method": str(config.get("event_detection_method") or "derivative"),
        "visual_response_metric": str(config.get("visual_response_metric") or "mean"),
        "output_root": str(result_root),
    }
    write_json_file(result_root / "summary" / "manifest.json", manifest)
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
