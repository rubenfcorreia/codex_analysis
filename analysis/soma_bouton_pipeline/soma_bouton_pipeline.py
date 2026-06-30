from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from analysis.compartment_common import ensure_dir, resolve_repo_root, write_csv_rows, write_json_file
from analysis.soma_bouton_pipeline.analysis_families.core import ExperimentContext, build_experiment_context, experiment_summary_row
from analysis.soma_bouton_pipeline.analysis_families.correlation import bouton_soma_correlation_rows, correlation_summary_rows
from analysis.soma_bouton_pipeline.analysis_families.lag import lag_scan_rows, lag_summary_rows
from analysis.soma_bouton_pipeline.analysis_families.state import activity_rows_for_context, state_summary_rows
from analysis.soma_bouton_pipeline.plots import plot_lag_heatmap, plot_state_activity, plot_state_correlation
from ..compartment_common import resolve_analysis_state_selections


DEFAULT_CONFIG = {
    "analysis_name": "soma_bouton_pipeline",
    "result_root": "results/soma_bouton_pipeline",
    "movie_expids": [],
    "sleep_expids": [],
    "soma_channel": 1,
    "bouton_channel": 0,
    "lag_window_s": 2.0,
    "lag_step_s": 0.1,
    "rebuild": True,
    "plots_only": False,
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

    experiment_rows: List[Dict[str, Any]] = []
    activity_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    lag_rows: List[Dict[str, Any]] = []
    selected_states_cache: Dict[str, Sequence[str]] = {}

    for mode in state_modes:
        selected_states_cache[mode] = resolve_analysis_state_selections(config, mode)
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
            activity_rows.extend(activity_rows_for_context(ctx, selected_states_cache[mode]))
            correlation_rows.extend(bouton_soma_correlation_rows(ctx, selected_states_cache[mode]))
            lag_rows.extend(
                lag_scan_rows(
                    ctx,
                    selected_states_cache[mode],
                    lag_window_s=float(config.get("lag_window_s", 2.0)),
                    lag_step_s=float(config.get("lag_step_s", 0.1)),
                )
            )

    day_groups = build_day_groups(expids_by_mode)

    activity_summary_rows = state_summary_rows(activity_rows)
    correlation_summary = correlation_summary_rows(correlation_rows)
    lag_summary = lag_summary_rows(lag_rows)

    write_csv_rows(result_root / "csv" / "experiments.csv", experiment_rows, list(experiment_rows[0].keys()) if experiment_rows else ["expid"])
    if activity_rows:
        write_csv_rows(result_root / "csv" / "state_activity_by_experiment.csv", activity_rows, list(activity_rows[0].keys()))
    if correlation_rows:
        write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_roi.csv", correlation_rows, list(correlation_rows[0].keys()))
    if lag_rows:
        write_csv_rows(result_root / "csv" / "bouton_soma_lag_scan_by_roi.csv", lag_rows, list(lag_rows[0].keys()))
    if activity_summary_rows:
        write_csv_rows(result_root / "csv" / "state_activity_by_day.csv", activity_summary_rows, list(activity_summary_rows[0].keys()))
    if correlation_summary:
        write_csv_rows(result_root / "csv" / "bouton_soma_correlation_by_day.csv", correlation_summary, list(correlation_summary[0].keys()))
    if lag_summary:
        write_csv_rows(result_root / "csv" / "bouton_soma_lag_summary_by_day.csv", lag_summary, list(lag_summary[0].keys()))

    plot_state_activity(activity_summary_rows or activity_rows, result_root)
    plot_state_correlation(correlation_summary, result_root)
    plot_lag_heatmap(lag_rows, result_root)

    manifest = {
        "config": dict(config),
        "state_modes": state_modes,
        "day_groups": day_groups,
        "counts": {
            "experiments": len(experiment_rows),
            "activity_rows": len(activity_rows),
            "correlation_rows": len(correlation_rows),
            "lag_rows": len(lag_rows),
        },
        "output_root": str(result_root),
    }
    write_json_file(result_root / "summary" / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare soma (ch2) and bouton (ch1) activity across movie and sleep states.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("soma_bouton_pipeline_config.json"))
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding outputs even if caches exist.")
    parser.add_argument("--plots-only", action="store_true", help="Skip metric recomputation and regenerate plots from written CSVs only.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.rebuild:
        config["rebuild"] = True
    if args.plots_only:
        config["plots_only"] = True
    manifest = run_pipeline(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
