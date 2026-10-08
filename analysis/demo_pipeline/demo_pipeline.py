#!/usr/bin/env python3
"""Build, preview, validate, and run the reproducible synthetic demos."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pickle
import shutil
import subprocess
import sys
import platform
import importlib.metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .preview import generate_preview
from .recipe_schema import DEFAULT_STATES, load_recipe, recipe_hash, resolved_recipe
from .validation import validate_demo

ROOT = Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_trials(path: Path, t: np.ndarray) -> None:
    rows = []
    for index, start in enumerate(np.linspace(5.0, float(t[-1]) - 5.0, 16)):
        rows.append({
            "time": f"{float(start):.3f}",
            "duration": "4.000",
            "F1_type": "movie",
            "F1_name": f"demo_state_{index % len(DEFAULT_STATES)}",
            "F1_onset": "0",
            "F1_duration": "4.000",
            "F1_speed": "1",
            "F1_loop": "0",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _state_codes(t: np.ndarray, states: Sequence[str]) -> np.ndarray:
    codes = np.zeros(t.size, dtype=int)
    edges = np.linspace(0.0, float(t[-1]) + (t[1] - t[0]), len(states) + 1)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        codes[(t >= start) & (t < end)] = index
    return codes


def _sleep_bundle(t: np.ndarray, states: Sequence[str], codes: np.ndarray) -> Dict[str, Any]:
    labels = {index: state for index, state in enumerate(states)}
    return {
        "state_10hz_t": t,
        "state_10hz": codes,
        "state_epoch_t": t[::10],
        "state_epoch": codes[::10],
        "epoch_t": t[::10],
        "state_labels": labels,
        "locomotion_threshold": 0.35,
        "wheel_10hz_t": t,
        "wheel_10hz": np.zeros_like(t),
    }


def _pulse_trace(t: np.ndarray, codes: np.ndarray, n_events: Sequence[int], amplitude: float, width: float, seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trace = np.zeros_like(t, dtype=float)
    for code, count in enumerate(n_events):
        available = np.flatnonzero(codes == code)
        if available.size == 0 or count <= 0:
            continue
        picks = np.linspace(available[0], available[-1], int(count) + 2, dtype=int)[1:-1]
        for pick in picks:
            trace += float(amplitude) * np.exp(-0.5 * ((t - t[pick]) / float(width)) ** 2)
    if noise:
        trace += float(noise) * rng.normal(size=t.size)
    return trace


def _conversion_library() -> Dict[int, Dict[str, Any]]:
    conversion: Dict[int, Dict[str, Any]] = {}
    index = 0
    for dendrite in (1, 2):
        conversion[index] = {"roi-type": [0, dendrite, 0, 0], "conversion index": index, "plane": 0, "conversion": [0, index]}
        index += 1
        for spine in (1, 2, 3):
            conversion[index] = {"roi-type": [2, dendrite, dendrite, spine], "conversion index": index, "plane": 0, "conversion": [0, index]}
            index += 1
    return conversion


def _write_experiment(repo_root: Path, expid: str, animal: str, trace_by_channel: Mapping[int, np.ndarray], t: np.ndarray, states: Sequence[str], codes: np.ndarray, *, dendrite: bool = False, seed: int = 0) -> None:
    exp_root = repo_root / animal / expid
    recordings = exp_root / "recordings"
    sleep_score = exp_root / "sleep_score"
    recordings.mkdir(parents=True, exist_ok=True)
    sleep_score.mkdir(parents=True, exist_ok=True)
    for channel, matrix in trace_by_channel.items():
        _write_pickle(recordings / f"s2p_ch{channel}.pickle", {"t": t, "dF": np.asarray(matrix, dtype=float), "OriginalSuite2pCellIDs": np.arange(matrix.shape[0], dtype=int)})
    _write_pickle(recordings / "wheel.pickle", {"t": t, "speed": np.zeros_like(t)})
    _write_pickle(recordings / "dlcEyeLeft_resampled.pickle", {"t": t, "pupil_diameter": np.ones_like(t), "speed": np.zeros_like(t)})
    _write_pickle(sleep_score / "sleep_state.pickle", _sleep_bundle(t, states, codes))
    _write_trials(exp_root / f"{expid}_all_trials.csv", t)
    if dendrite:
        spinesgui = exp_root / "suite2p" / "SpinesGUI"
        spinesgui.mkdir(parents=True, exist_ok=True)
        np.save(spinesgui / "ROIs_normal_mode_conversion.npy", _conversion_library(), allow_pickle=True)
        plane = exp_root / "suite2p" / "plane0"
        plane.mkdir(parents=True, exist_ok=True)
        np.save(plane / "ops.npy", {"Ly": 128, "Lx": 128, "meanImg": np.zeros((128, 128), dtype=np.float32)}, allow_pickle=True)
        np.save(plane / "stat.npy", np.asarray([], dtype=object), allow_pickle=True)


def _build_dendrites(recipe: Dict[str, Any], repo_root: Path, truth: Dict[str, Any]) -> List[str]:
    expids: List[str] = []
    t = np.arange(0.0, float(recipe["duration_s"]), float(recipe["dt_s"]))
    codes = _state_codes(t, recipe["states"])
    style = recipe["trace_style"]
    base_events = int(style["base_events_per_state"])
    for compartment in ("basal", "apical"):
        factors = recipe["dendrites"][compartment]
        for replicate in range(int(recipe["replicates"])):
            animal = "DEMO"
            expid = f"2026-01-{replicate + 1:02d}_01_DEMO_{compartment.upper()}_{replicate + 1:02d}"
            amplitude = float(style["event_amplitude"]) * float(factors["activity_scale"])
            count = max(1, int(round(base_events * float(factors["frequency_scale"]))))
            parent = _pulse_trace(t, codes, [count] * len(recipe["states"]), amplitude, float(style["event_width_s"]), int(recipe["seed"]) + replicate, float(style["noise"])) + float(style["baseline"])
            traces = []
            for roi in range(8):
                if roi in (0, 1):
                    trace = parent
                else:
                    trace = 0.8 * parent if roi % 2 else parent
                traces.append(trace)
            _write_experiment(repo_root, expid, animal, {0: np.asarray(traces)}, t, recipe["states"], codes, dendrite=True, seed=replicate)
            expids.append(expid)
            truth.setdefault("dendrites", []).append({"expid": expid, "compartment": compartment, "activity_scale": float(factors["activity_scale"]), "frequency_scale": float(factors["frequency_scale"]), "states": list(recipe["states"])})
    return expids


def _build_soma_boutons(recipe: Dict[str, Any], repo_root: Path, truth: Dict[str, Any]) -> List[str]:
    expids: List[str] = []
    t = np.arange(0.0, float(recipe["duration_s"]), float(recipe["dt_s"]))
    codes = _state_codes(t, recipe["states"])
    style = recipe["trace_style"]
    base_events = int(style["base_events_per_state"])
    for replicate in range(int(recipe["replicates"])):
        animal = f"{replicate + 1:02d}"
        expid = f"2026-02-{replicate + 1:02d}_01_DEMO_SOMA_{replicate + 1:02d}"
        soma_traces: List[np.ndarray] = []
        for index, name in enumerate(("soma_1", "soma_2")):
            factors = recipe["somas"][name]
            count = max(1, int(round(base_events * float(factors["frequency_scale"]))))
            trace = _pulse_trace(t, codes, [count] * len(recipe["states"]), float(style["event_amplitude"]) * float(factors["activity_scale"]), float(style["event_width_s"]), int(recipe["seed"]) + 100 + replicate + index, float(style["noise"])) + float(style["baseline"])
            soma_traces.append(trace)
        bouton_traces: List[np.ndarray] = []
        for index, name in enumerate(("bouton_1", "bouton_2", "bouton_3")):
            target = recipe["boutons"][name].get("coupled_to")
            if target == "soma_1":
                trace = soma_traces[0].copy()
            elif target == "soma_2":
                trace = soma_traces[1].copy()
            else:
                trace = _pulse_trace(t, codes, [base_events] * len(recipe["states"]), float(style["event_amplitude"]), float(style["event_width_s"]), int(recipe["seed"]) + 200 + replicate + index, float(style["noise"])) + float(style["baseline"])
            bouton_traces.append(trace)
        _write_experiment(repo_root, expid, animal, {0: np.asarray(bouton_traces), 1: np.asarray(soma_traces)}, t, recipe["states"], codes)
        expids.append(expid)
        truth.setdefault("soma_boutons", []).append({"expid": expid, "somas": recipe["somas"], "boutons": recipe["boutons"]})
    return expids


def _package_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _git_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=False, capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def build_demo(recipe: Dict[str, Any], output_dir: Path, *, original_recipe: Dict[str, Any] | None = None) -> Dict[str, Any]:
    original_recipe = dict(original_recipe or recipe)
    recipe = resolved_recipe(recipe)
    output_dir = Path(output_dir).resolve()
    repo_root = output_dir
    if output_dir.exists():
        for child in (repo_root, output_dir / "expected_preview", output_dir / "observed_validation"):
            if child.exists():
                shutil.rmtree(child)
    repo_root.mkdir(parents=True, exist_ok=True)
    truth: Dict[str, Any] = {"schema_version": 1, "recipe_hash": recipe_hash(recipe), "states": list(recipe["states"]), "dendrites": [], "soma_boutons": []}
    dendrite_expids = _build_dendrites(recipe, repo_root, truth) if recipe["analysis"].get("run_dendrites", True) else []
    soma_expids = _build_soma_boutons(recipe, repo_root, truth) if recipe["analysis"].get("run_soma_bouton", True) else []
    truth["topology"] = recipe["boutons"]
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    dendrite_config = {
        "analysis_name": "dendrites_pipeline", "repo_base": str(output_dir), "user_id": "demo", "sleep_expids": dendrite_expids,
        "basal_expids": [x for x in dendrite_expids if "BASAL" in x], "apical_expids": [x for x in dendrite_expids if "APICAL" in x],
        "state_comparison_states": list(recipe["states"]), "basal_apical_states": list(recipe["states"]), "channel": 0,
        "shuffle_n": int(recipe.get("shuffle_n", 2)), "correlation_shuffle_n": int(recipe.get("shuffle_n", 2)), "rebuild": True,
        "analysis_families": ["state", "basal_apical", "correlation", "mixed_model"], "output_dir": str(output_dir / "dendrites_results"),
        "figure_output_dir": str(output_dir / "dendrites_results" / "figures"), "demo": False, "plot_profile": "full" if bool(recipe["analysis"].get("generate_figures", True)) else "analysis_only", "cpu_thread_limit": int(recipe["runtime"].get("cpu_thread_limit", 1)),
    }
    soma_config = {
        "analysis_name": "soma_bouton_pipeline", "repo_root": str(output_dir), "repo_base": str(output_dir), "user_id": "demo",
        "sleep_expids": soma_expids, "soma_channel": 1, "bouton_channel": 0, "state_comparison_states": list(recipe["states"]),
        "compartment_states": list(recipe["states"]), "shuffle_n": int(recipe.get("shuffle_n", 2)), "correlation_shuffle_n": int(recipe.get("shuffle_n", 2)),
        "correlation_inference": "circular_shift", "rebuild": True, "result_root": str(output_dir / "soma_bouton_results"),
        "analysis_output_dir": str(output_dir / "soma_bouton_results" / "analysis"), "cache_root": str(output_dir / "soma_bouton_results" / "cache"), "plot_profile": "full" if bool(recipe["analysis"].get("generate_figures", True)) else "analysis_only", "cpu_thread_limit": int(recipe["runtime"].get("cpu_thread_limit", 1)),
    }
    _write_json(output_dir / "demo_config_original.json", original_recipe)
    _write_json(output_dir / "demo_config_resolved.json", recipe)
    _write_json(output_dir / "demo_truth.json", truth)
    _write_json(config_dir / "dendrites_config.json", dendrite_config)
    _write_json(config_dir / "soma_bouton_config.json", soma_config)
    metadata = {
        "recipe_hash": recipe["recipe_hash"],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib") if _package_available(name)},
        "git_revision": _git_revision(),
        "methodology_version": "demo-platform-v1",
        "generated_experiment_ids": {"dendrites": dendrite_expids, "soma_bouton": soma_expids},
    }
    _write_json(output_dir / "demo_run_metadata.json", metadata)
    _write_json(output_dir / "demo_manifest.json", {"recipe_hash": recipe["recipe_hash"], "repo_root": str(repo_root), "dendrite_config": dendrite_config, "soma_bouton_config": soma_config, "metadata": metadata, "generated_at": dt.datetime.now().isoformat(timespec="seconds")})
    if recipe["analysis"].get("generate_figures", True):
        generate_preview(recipe, truth, output_dir / "expected_preview")
    return truth


def _output_size_bytes(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in Path(output_dir).rglob("*") if path.is_file())


def _load_pipeline_manifests(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifests: Dict[str, Dict[str, Any]] = {}
    for name, relative in (
        ("dendrites", "dendrites_results/analysis/summary/manifest.json"),
        ("soma_bouton", "soma_bouton_results/analysis/summary/manifest.json"),
    ):
        path = output_dir / relative
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                manifests[name] = payload
    return manifests


def _cache_diagnostics(cache_summary: Any) -> List[Dict[str, Any]]:
    """Normalize the two pipeline cache-summary shapes for benchmark reports."""
    if not isinstance(cache_summary, dict):
        return []

    def size(path: Path | None) -> int:
        if path is None or not path.exists():
            chunk_path = path.with_suffix(".chunks") if path is not None else None
            path = chunk_path if chunk_path and chunk_path.exists() else None
        if path is None:
            return 0
        if path.is_file():
            return int(path.stat().st_size)
        return sum(int(child.stat().st_size) for child in path.rglob("*") if child.is_file())

    records: List[Dict[str, Any]] = []
    existing_records = cache_summary.get("cache_records")
    if isinstance(existing_records, list):
        for value in existing_records:
            if not isinstance(value, dict):
                continue
            record = dict(value)
            path = Path(str(record["path"])) if record.get("path") else None
            record["exists"] = bool(path and (path.exists() or path.with_suffix(".chunks").exists()))
            actual = path if path and path.exists() else (path.with_suffix(".chunks") if path else None)
            if actual and actual.exists():
                record["file_size_bytes"] = sum(int(child.stat().st_size) for child in actual.rglob("*") if child.is_file()) if actual.is_dir() else int(actual.stat().st_size)
            records.append(record)
        return records
    # Dendrites use one mapping per cache; soma/bouton also exposes a flat
    # mapping with *_cache_path, *_cache_status, and *_cache_key fields.
    named_values = {name: value for name, value in cache_summary.items() if isinstance(value, dict)}
    if named_values:
        for name, value in named_values.items():
            path_value = value.get("path") or value.get("cache_path")
            path = Path(str(path_value)) if path_value else None
            record = {
                "name": str(name), "path": str(path) if path else None,
                "scope": value.get("scope"), "key": value.get("key") or value.get("cache_key"),
                "status": value.get("status") or value.get("cache_status"),
                "exists": bool(path and (path.exists() or path.with_suffix(".chunks").exists())),
                "file_size_bytes": size(path),
            }
            for key in ("row_counts", "row_count", "load_elapsed_s", "build_elapsed_s", "invalidation_reason"):
                if key in value:
                    record[key] = value[key]
            records.append(record)
    else:
        grouped: Dict[str, Dict[str, Any]] = {}
        for key, value in cache_summary.items():
            key = str(key)
            if not key.endswith("_cache_path"):
                continue
            base = key[:-len("_cache_path")]
            path = Path(str(value)) if value else None
            grouped[base] = {
                "name": base, "path": str(path) if path else None,
                "key": cache_summary.get(base + "_cache_key"),
                "status": cache_summary.get(base + "_cache_status"),
                "exists": bool(path and (path.exists() or path.with_suffix(".chunks").exists())),
                "file_size_bytes": size(path),
            }
        records.extend(grouped.values())
    return records


def _benchmark_snapshot(output_dir: Path, manifests: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pipelines: Dict[str, Any] = {}
    for name, manifest in manifests.items():
        runtime = manifest.get("runtime_diagnostics", {})
        start = runtime.get("start", {}) if isinstance(runtime, dict) else {}
        end = runtime.get("end", {}) if isinstance(runtime, dict) else {}
        stages = manifest.get("stage_timings", [])
        if not isinstance(stages, list):
            stages = []
        stage_seconds = {
            str(stage.get("name")): float(stage.get("elapsed_s", 0.0) or 0.0)
            for stage in stages
            if isinstance(stage, dict) and stage.get("name")
        }
        cache_seconds = sum(elapsed for stage_name, elapsed in stage_seconds.items() if "cache" in stage_name.lower())
        plotting_seconds = sum(
            elapsed for stage_name, elapsed in stage_seconds.items()
            if any(token in stage_name.lower() for token in ("plot", "figure", "poster"))
        )
        analysis_seconds = sum(
            elapsed for stage_name, elapsed in stage_seconds.items()
            if any(token in stage_name.lower() for token in ("analysis", "family", "comparison", "model"))
        )
        pipelines[name] = {
            "pipeline_elapsed_s": manifest.get("pipeline_elapsed_s"),
            "stage_timings": stages,
            "stage_totals_s": {"analysis": analysis_seconds, "plotting": plotting_seconds, "cache": cache_seconds},
            "peak_rss_bytes": max(int(start.get("peak_rss_bytes", 0) or 0), int(end.get("peak_rss_bytes", 0) or 0)),
            "open_figures_before": start.get("open_figures"),
            "open_figures_after": end.get("open_figures"),
            "figure_leak_count": runtime.get("figure_leak_count") if isinstance(runtime, dict) else None,
            "file_count": end.get("file_count"),
            "output_bytes": end.get("output_bytes"),
            "row_counts": manifest.get("analysis_cache_summary", {}),
            "cache_records": _cache_diagnostics(manifest.get("cache_summary")),
        }
    return {
        "pipelines": pipelines,
        "peak_rss_bytes": max((item["peak_rss_bytes"] for item in pipelines.values()), default=0),
        "figure_leak_count": sum(int(item["figure_leak_count"] or 0) for item in pipelines.values()),
        "file_count": sum(int(item["file_count"] or 0) for item in pipelines.values()),
        "output_bytes": _output_size_bytes(output_dir),
    }


def run_pipeline_configs(output_dir: Path, *, only: str | None = None) -> int:
    configs = []
    if only in (None, "dendrites"):
        configs.append([sys.executable, str(ROOT / "analysis/dendrites_pipeline/dendrites_pipeline.py"), "--config", str(output_dir / "configs/dendrites_config.json")])
    if only in (None, "soma_bouton"):
        configs.append([sys.executable, str(ROOT / "analysis/soma_bouton_pipeline/soma_bouton_pipeline.py"), "--config", str(output_dir / "configs/soma_bouton_config.json")])
    runtime = json.loads((output_dir / "demo_config_resolved.json").read_text()).get("runtime", {})
    timeout_s = float(runtime.get("timeout_s", 3600))
    max_output_bytes = int(runtime.get("max_output_bytes", 2_000_000_000))
    metadata_path = output_dir / "demo_run_metadata.json"
    run_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    run_metadata.setdefault("pipeline_runs", [])
    for command in configs:
        started = dt.datetime.now().isoformat()
        try:
            completed = subprocess.run(command, cwd=str(ROOT), check=False, timeout=timeout_s)
            status = "completed" if completed.returncode == 0 else "failed"
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            status, exit_code = "timeout", 124
        run_metadata["pipeline_runs"].append({"command": command, "started_at": started, "ended_at": dt.datetime.now().isoformat(), "status": status, "exit_code": exit_code})
        _write_json(metadata_path, run_metadata)
        if exit_code:
            return exit_code
        if _output_size_bytes(output_dir) > max_output_bytes:
            run_metadata["pipeline_runs"][-1]["status"] = "output_limit_exceeded"
            _write_json(metadata_path, run_metadata)
            return 125
    result = validate_demo(output_dir)
    return 0 if result["passed"] else 1


def benchmark_demo(recipe: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    import time

    output_dir = Path(output_dir)
    build_demo(recipe, output_dir)
    started = time.perf_counter()
    cold_exit = run_pipeline_configs(output_dir)
    cold_elapsed = time.perf_counter() - started
    cold_manifests = _load_pipeline_manifests(output_dir)
    cold_snapshot = _benchmark_snapshot(output_dir, cold_manifests)
    if cold_exit:
        report = {
            "passed": False,
            "cold_exit": cold_exit,
            "cold_elapsed_s": cold_elapsed,
            "cold": cold_snapshot,
        }
        _write_json(output_dir / "benchmark_report.json", report)
        return report
    for name in ("dendrites", "soma_bouton"):
        config_path = output_dir / "configs" / f"{name}_config.json"
        config = json.loads(config_path.read_text())
        config.update({"rebuild": False, "source_cache_rebuild": False, "analysis_tables_rebuild": False, "analysis_results_rebuild": False, "shared_shuffle_cache_rebuild": False})
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    warm_exit = run_pipeline_configs(output_dir)
    warm_elapsed = time.perf_counter() - started
    warm_manifests = _load_pipeline_manifests(output_dir)
    warm_snapshot = _benchmark_snapshot(output_dir, warm_manifests)
    report = {
        "passed": warm_exit == 0,
        "cold_exit": cold_exit,
        "warm_exit": warm_exit,
        "cold_elapsed_s": cold_elapsed,
        "warm_elapsed_s": warm_elapsed,
        "speedup": (cold_elapsed / warm_elapsed) if warm_elapsed > 0 else None,
        "cold": cold_snapshot,
        "warm": warm_snapshot,
        "manifests": warm_manifests,
        "output_bytes": _output_size_bytes(output_dir),
    }
    _write_json(output_dir / "benchmark_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "preview", "validate", "run", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--output-dir", type=Path, required=True)
    sub.choices["run"].add_argument("--only", choices=("dendrites", "soma_bouton"))
    args = parser.parse_args(argv)
    original_recipe = {}
    if args.config:
        original_recipe = json.loads(args.config.read_text(encoding="utf-8"))
    recipe = load_recipe(args.config)
    if args.command in ("build", "preview", "run"):
        build_demo(recipe, args.output_dir, original_recipe=original_recipe or recipe)
    if args.command == "preview":
        return 0
    if args.command == "validate":
        result = validate_demo(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "run":
        return run_pipeline_configs(args.output_dir, only=args.only)
    if args.command == "benchmark":
        report = benchmark_demo(recipe, args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("passed") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
