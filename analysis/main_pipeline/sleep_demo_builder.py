#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "main_pipeline" / "demo"
DEFAULT_REPO_SUBDIR = "demo_repository"
DEFAULT_STIMULUS_SOURCE_ROOT = Path("/data/Remote_Repository/bv_resources/all_movie_clips_bv_sets")
DEFAULT_ANALYSIS_FAMILIES = ["state", "basal_apical", "correlation", "matrix", "mixed_model"]
DEFAULT_CHANNEL = 0
DEFAULT_HIGH_PASS_HZ = 0.02
DEFAULT_SHUFFLES = 200
DEFAULT_LOCOMOTION_THRESHOLD = 0.35

BLANK_MOVIE_PATH = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00000"
GRATING_PREFIX = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\01"
ZEBRA_PREFIX = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\02"

NORMAL_CONVERSION_FILENAME = "ROIs_normal_mode_conversion.npy"
DEND_AXON_CONVERSION_FILENAME = "ROIs_dendrite_axon_mode_conversion.npy"


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return float(value)
        except Exception:
            return None
    return None


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    return None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(jsonable(payload), sort_keys=True, default=str).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def write_pickle(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_list_argument(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        return [item for item in text.split() if item.strip()]
    if isinstance(values, (list, tuple, set)):
        flattened: List[str] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and "," in value:
                flattened.extend([item.strip() for item in value.split(",") if item.strip()])
            else:
                text = str(value).strip()
                if text:
                    flattened.append(text)
        return flattened
    text = str(values).strip()
    return [text] if text else []


def interval_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return (t >= start) & (t < end)


def gaussian(t: np.ndarray, center: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((np.asarray(t, dtype=float) - float(center)) / sigma) ** 2)


def normalize_name(text: str) -> str:
    return str(text).strip().lower().replace("-", "_")


def canonical_analysis_families(values: Any) -> List[str]:
    selected = parse_list_argument(values)
    if not selected:
        return list(DEFAULT_ANALYSIS_FAMILIES)
    aliases = {
        "state_comparisons": "state",
        "basal_apical_comparisons": "basal_apical",
        "correlations": "correlation",
        "matrix_similarity": "matrix",
        "mixed_model_analysis": "mixed_model",
        "mixed-model": "mixed_model",
    }
    resolved: List[str] = []
    seen = set()
    for item in selected:
        normalized = aliases.get(normalize_name(item), normalize_name(item))
        if normalized == "all":
            return list(DEFAULT_ANALYSIS_FAMILIES)
        if normalized not in DEFAULT_ANALYSIS_FAMILIES:
            raise SystemExit(
                f"Unknown analysis family: {item}. "
                f"Allowed values are: all, {', '.join(DEFAULT_ANALYSIS_FAMILIES)}"
            )
        if normalized not in seen:
            resolved.append(normalized)
            seen.add(normalized)
    return resolved or list(DEFAULT_ANALYSIS_FAMILIES)


def discover_clip_names(source_root: Path, limit: int = 12) -> List[str]:
    if not source_root.exists():
        return []
    paths = [path for path in source_root.glob("*/*") if path.is_dir()]
    return [str(path) for path in sorted(paths)[:limit]]


def default_feature_sets() -> List[Dict[str, Any]]:
    return [
        {"F1_x": -20, "F1_y": 0, "F1_angle": 45, "F1_width": 152, "F1_height": 85.514, "F1_speed": 1, "F1_loop": 0},
        {"F1_x": -15, "F1_y": 5, "F1_angle": 45, "F1_width": 152, "F1_height": 85.514, "F1_speed": 1, "F1_loop": 0},
        {"F1_x": -25, "F1_y": -5, "F1_angle": 45, "F1_width": 152, "F1_height": 85.514, "F1_speed": 1, "F1_loop": 0},
        {"F1_x": 20, "F1_y": 0, "F1_angle": 0, "F1_width": 152, "F1_height": 85.514, "F1_speed": 1, "F1_loop": 0},
    ]


def default_state_trial_specs() -> List[Dict[str, Any]]:
    return [
        {
            "category": "blank",
            "name": BLANK_MOVIE_PATH,
            "repeats": 4,
            "active_repeats": [1, 3],
        },
        {
            "category": "grating",
            "name": GRATING_PREFIX + r"\00001",
            "repeats": 4,
            "active_pattern": [False, True, False, True],
        },
        {
            "category": "zebra",
            "name": ZEBRA_PREFIX + r"\02001",
            "repeats": 4,
            "active_repeats": [2],
        },
        {
            "category": "movies",
            "name": r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03001",
            "repeats": 4,
            "active_repeats": [1, 2],
        },
    ]


def default_stimulus_trial_specs(source_root: Path) -> List[Dict[str, Any]]:
    clip_names = discover_clip_names(source_root, limit=8)
    if not clip_names:
        clip_names = [
            r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03001",
            r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03002",
            r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03003",
        ]
    feature_sets = default_feature_sets()
    specs: List[Dict[str, Any]] = []
    for index, clip_name in enumerate(clip_names):
        feature_set = feature_sets[index % len(feature_sets)]
        specs.append(
            {
                "category": "movies",
                "name": clip_name,
                "repeats": 2,
                "feature_sets": [feature_set],
                "active_repeats": [0, 1],
            }
        )
    return specs


def default_recipe() -> Dict[str, Any]:
    return {
        "repo_subdir": DEFAULT_REPO_SUBDIR,
        "user_id": "demo_user",
        "channel": DEFAULT_CHANNEL,
        "locomotion_threshold": DEFAULT_LOCOMOTION_THRESHOLD,
        "analysis_families": list(DEFAULT_ANALYSIS_FAMILIES),
        "stimulus_source_root": str(DEFAULT_STIMULUS_SOURCE_ROOT),
        "experiments": [
            {
                "exp_id": "2025-01-01_01_DEMOGEN",
                "animal_id": "DEMOGEN",
                "kind": "movie",
                "compartment": "basal",
                "mode": "normal",
                "seed": 101,
                "t_end": 180.0,
                "dt": 0.05,
                "trial_start": 4.0,
                "trial_duration": 6.0,
                "trial_gap": 0.75,
                "n_rois": 8,
                "layout": {
                    "n_cells": 1,
                    "dendrites_per_cell": 1,
                    "spines_per_dendrite": 3,
                    "background_rois": 2,
                },
                "trial_specs": default_state_trial_specs(),
                "responses": [
                    {
                        "roi_ref": "cell1.dend1",
                        "response": {
                            "kind": "state",
                            "state_weights": {
                                "quiet_blank": 0.00,
                                "active_blank": 0.05,
                                "quiet_grating": 0.10,
                                "active_grating": 0.18,
                                "quiet_zebra": 0.22,
                                "active_zebra": 0.28,
                                "quiet_movies": 0.30,
                                "active_movies": 0.38,
                            },
                            "baseline": 0.05,
                            "noise_scale": 0.02,
                        },
                    },
                    {
                        "roi_ref": "cell1.dend1.spine1",
                        "response": {
                            "kind": "inherit",
                            "parent_ref": "cell1.dend1",
                            "alpha": 0.8,
                            "specific": {
                                "kind": "state",
                                "state_weights": {
                                    "quiet_movies": 0.06,
                                    "active_movies": 0.10,
                                },
                                "baseline": 0.00,
                                "noise_scale": 0.02,
                            },
                        },
                    },
                ],
            },
            {
                "exp_id": "2025-01-02_01_DEMOGEN",
                "animal_id": "DEMOGEN",
                "kind": "movie",
                "compartment": "apical",
                "mode": "normal",
                "seed": 202,
                "t_end": 180.0,
                "dt": 0.05,
                "trial_start": 5.0,
                "trial_duration": 5.0,
                "trial_gap": 0.5,
                "n_rois": 8,
                "layout": {
                    "n_cells": 1,
                    "dendrites_per_cell": 1,
                    "spines_per_dendrite": 3,
                    "background_rois": 2,
                },
                "trial_specs": default_stimulus_trial_specs(DEFAULT_STIMULUS_SOURCE_ROOT),
                "responses": [
                    {
                        "roi_ref": "cell1.dend1",
                        "response": {
                            "kind": "stimulus",
                            "selector": {"F1_x": -20, "F1_y": 0, "F1_angle": 45},
                            "bandwidth": {"F1_x": 10, "F1_y": 10, "F1_angle": 12},
                            "amplitude": 0.90,
                            "baseline": 0.08,
                            "kernel_width": 0.65,
                            "latency": 0.15,
                            "noise_scale": 0.015,
                        },
                    },
                    {
                        "roi_ref": "cell1.dend1.spine1",
                        "response": {
                            "kind": "inherit",
                            "parent_ref": "cell1.dend1",
                            "alpha": 0.75,
                            "specific": {
                                "kind": "stimulus",
                                "selector": {"F1_x": -20, "F1_y": 0, "F1_angle": 45},
                                "bandwidth": {"F1_x": 8, "F1_y": 8, "F1_angle": 10},
                                "amplitude": 0.35,
                                "baseline": 0.02,
                                "kernel_width": 0.60,
                                "latency": 0.10,
                                "noise_scale": 0.015,
                            },
                        },
                    },
                    {
                        "roi_ref": "cell1.dend1.spine2",
                        "response": {
                            "kind": "stimulus",
                            "selector": {"F1_x": 20, "F1_y": 0, "F1_angle": 0},
                            "bandwidth": {"F1_x": 6, "F1_y": 6, "F1_angle": 8},
                            "amplitude": 0.05,
                            "baseline": 0.02,
                            "kernel_width": 0.50,
                            "latency": 0.10,
                            "noise_scale": 0.010,
                        },
                    },
                ],
            },
        ],
    }


def merge_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    merged = default_recipe()
    merged.update({k: v for k, v in recipe.items() if k != "experiments"})
    if "analysis_families" in recipe:
        merged["analysis_families"] = canonical_analysis_families(recipe.get("analysis_families"))
    experiments = recipe.get("experiments")
    if experiments is not None:
        if not isinstance(experiments, list) or not experiments:
            raise SystemExit("Recipe must provide a non-empty experiments list")
        merged["experiments"] = experiments
    if "stimulus_source_root" in merged:
        merged["stimulus_source_root"] = str(merged["stimulus_source_root"])
    return merged


def build_trial_rows(
    experiment: Dict[str, Any],
    t: np.ndarray,
    source_root: Path,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
    trial_specs = experiment.get("trial_specs")
    if trial_specs is None:
        response_kinds = {
            normalize_name((item.get("response") or {}).get("kind", ""))
            for item in experiment.get("responses", [])
            if isinstance(item, dict)
        }
        if "stimulus" in response_kinds:
            trial_specs = default_stimulus_trial_specs(source_root)
        elif experiment.get("kind", "movie") == "sleep":
            trial_specs = default_stimulus_trial_specs(source_root)
        elif "state" in response_kinds:
            trial_specs = default_state_trial_specs()
        else:
            trial_specs = default_stimulus_trial_specs(source_root)
    if not isinstance(trial_specs, list) or not trial_specs:
        raise SystemExit(f"Experiment {experiment.get('exp_id')} must define a non-empty trial_specs list")
    trial_rows: List[Dict[str, Any]] = []
    trial_state_labels: List[str] = []
    trial_meta: List[Dict[str, Any]] = []
    trial_start = as_float(experiment.get("trial_start")) or 5.0
    trial_duration = as_float(experiment.get("trial_duration")) or 5.0
    trial_gap = as_float(experiment.get("trial_gap")) or 0.0
    movie_feature_fields = ["F1_x", "F1_y", "F1_angle", "F1_width", "F1_height", "F1_speed", "F1_loop"]
    for trial_spec in trial_specs:
        category = str(trial_spec.get("category", "movies"))
        name = str(trial_spec.get("name") or trial_spec.get("clip") or trial_spec.get("path") or "")
        if not name:
            raise SystemExit(f"Trial spec in {experiment.get('exp_id')} is missing a stimulus name")
        repeats = int(trial_spec.get("repeats", 1))
        active_repeats = {int(v) for v in trial_spec.get("active_repeats", [])}
        active_pattern = trial_spec.get("active_pattern")
        repeat_duration = as_float(trial_spec.get("duration")) or trial_duration
        repeat_gap = as_float(trial_spec.get("gap")) or trial_gap
        feature_sets = trial_spec.get("feature_sets")
        features = trial_spec.get("features") or {}
        extra_features = trial_spec.get("extra_features", [])
        for repeat in range(repeats):
            if isinstance(active_pattern, list) and repeat < len(active_pattern):
                active = bool(active_pattern[repeat])
            elif active_repeats:
                active = repeat in active_repeats
            else:
                active = True if category == "movies" else (repeat % 2 == 1)
            row: Dict[str, Any] = {
                "time": f"{trial_start:.3f}",
                "duration": f"{repeat_duration:.3f}",
                "F1_type": "movie",
                "F1_name": name,
                "F1_onset": "0",
                "F1_duration": f"{repeat_duration:.3f}",
                "F1_speed": "1",
                "F1_loop": "0",
                "state_label": f"{'active' if active else 'quiet'}_{category}",
                "active": "1" if active else "0",
            }
            if isinstance(feature_sets, list) and feature_sets:
                feature_spec = feature_sets[repeat % len(feature_sets)]
                if isinstance(feature_spec, dict):
                    features = {**features, **feature_spec}
            for field in movie_feature_fields:
                if field in features:
                    row[field] = f"{as_float(features[field]) if as_float(features[field]) is not None else features[field]}"
            for extra_index, extra_feature in enumerate(extra_features, start=2):
                prefix = str(extra_feature.get("prefix", f"F{extra_index}"))
                row[f"{prefix}_type"] = str(extra_feature.get("type", "movie"))
                row[f"{prefix}_name"] = str(extra_feature.get("name", name))
                row[f"{prefix}_onset"] = str(extra_feature.get("onset", "0"))
                row[f"{prefix}_duration"] = str(extra_feature.get("duration", repeat_duration))
                row[f"{prefix}_speed"] = str(extra_feature.get("speed", "1"))
                row[f"{prefix}_loop"] = str(extra_feature.get("loop", "0"))
            if active:
                row["_demo_active"] = "1"
            trial_rows.append(row)
            trial_state_labels.append(str(row["state_label"]))
            trial_meta.append(
                {
                    "trial_index": len(trial_rows) - 1,
                    "category": category,
                    "state_label": trial_state_labels[-1],
                    "start": trial_start,
                    "end": trial_start + repeat_duration,
                    "duration": repeat_duration,
                    "active": active,
                    "row": row,
                }
            )
            trial_start += repeat_duration + repeat_gap
    fieldnames = sorted({key for row in trial_rows for key in row.keys() if not key.startswith("_")}, key=lambda x: (0 if x == "time" else 1, x))
    cleaned_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in trial_rows]
    return cleaned_rows, trial_state_labels, trial_meta, fieldnames


def build_sleep_state_bundle(t: np.ndarray, experiment: Dict[str, Any], wheel_speed: np.ndarray) -> Dict[str, Any]:
    sleep_segments = experiment.get("sleep_state_segments")
    if not sleep_segments:
        sleep_segments = [
            {"start": 0.0, "end": float(t[-1]) * 0.25, "code": 1},
            {"start": float(t[-1]) * 0.25, "end": float(t[-1]) * 0.50, "code": 0},
            {"start": float(t[-1]) * 0.50, "end": float(t[-1]) * 0.75, "code": 2},
            {"start": float(t[-1]) * 0.75, "end": float(t[-1]) + (t[1] - t[0] if t.size > 1 else 0.1), "code": 3},
        ]
    state_codes = np.zeros_like(t, dtype=int)
    if isinstance(sleep_segments, list) and sleep_segments and isinstance(sleep_segments[0], dict):
        for segment in sleep_segments:
            start = as_float(segment.get("start"))
            end = as_float(segment.get("end"))
            code = as_int(segment.get("code"))
            if start is None or end is None or code is None:
                continue
            state_codes[(t >= start) & (t < end)] = code
    else:
        for segment in sleep_segments:
            if len(segment) < 3:
                continue
            start, end, code = segment[:3]
            state_codes[(t >= float(start)) & (t < float(end))] = int(code)
    state_labels = {0: "active_awake", 1: "quiet_awake", 2: "nrem", 3: "rem"}
    return {
        "state_10hz_t": t,
        "state_10hz": state_codes,
        "state_epoch_t": t[::10] if t.size >= 10 else t,
        "state_epoch": state_codes[::10] if t.size >= 10 else state_codes,
        "epoch_t": t[::10] if t.size >= 10 else t,
        "state_labels": state_labels,
        "locomotion_threshold": as_float(experiment.get("sleep_locomotion_threshold")) or DEFAULT_LOCOMOTION_THRESHOLD,
        "emg_rms_10hz": np.abs(np.sin(np.linspace(0, 4 * np.pi, t.size))) + 0.05,
        "emg_rms_10hz_t": t,
        "wheel_10hz": wheel_speed,
        "wheel_10hz_t": t,
    }


def response_weight(row: Dict[str, Any], selector: Dict[str, Any], bandwidth: Dict[str, Any]) -> float:
    if not selector:
        return 1.0
    weight = 1.0
    for field, target in selector.items():
        value = as_float(row.get(field))
        if value is None:
            continue
        sigma = as_float(bandwidth.get(field)) if bandwidth else None
        if sigma is None or sigma <= 0:
            sigma = 10.0
        weight *= math.exp(-0.5 * ((value - float(target)) / sigma) ** 2)
    return float(weight)


def build_drive_trace(
    t: np.ndarray,
    response: Dict[str, Any],
    trial_rows: Sequence[Dict[str, Any]],
    trial_state_labels: Sequence[str],
    trial_meta: Sequence[Dict[str, Any]],
    t_step: float,
    *,
    experiment_kind: str = "movie",
    sleep_bundle: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    kind = normalize_name(response.get("kind", "constant"))
    if kind == "state":
        weights = response.get("state_weights") or response.get("state_effects") or {}
        baseline = as_float(response.get("baseline")) or 0.0
        trace = np.full_like(t, baseline, dtype=float)
        if normalize_name(experiment_kind) == "sleep" and sleep_bundle is not None:
            sleep_t = np.asarray(sleep_bundle.get("state_10hz_t"), dtype=float)
            sleep_codes = np.asarray(sleep_bundle.get("state_10hz"), dtype=int)
            state_labels = sleep_bundle.get("state_labels") or {0: "active_awake", 1: "quiet_awake", 2: "nrem", 3: "rem"}
            if sleep_t.size and sleep_codes.size:
                for code, label in state_labels.items():
                    amp = as_float(weights.get(label))
                    if amp is None:
                        continue
                    trace[np.asarray(sleep_codes == int(code), dtype=bool)] += amp
            return trace
        for trial_index, meta in enumerate(trial_meta):
            state_label = str(trial_state_labels[trial_index]) if trial_index < len(trial_state_labels) else str(meta.get("state_label", ""))
            amp = as_float(weights.get(state_label))
            if amp is None:
                continue
            trace[interval_mask(t, float(meta["start"]), float(meta["end"]))] += amp
        return trace
    if kind == "stimulus":
        selector = response.get("selector") or {}
        bandwidth = response.get("bandwidth") or {}
        amplitude = as_float(response.get("amplitude")) or 1.0
        baseline = as_float(response.get("baseline")) or 0.0
        latency = as_float(response.get("latency")) or 0.0
        kernel_width = as_float(response.get("kernel_width"))
        if kernel_width is None or kernel_width <= 0:
            kernel_width = max(as_float(response.get("kernel_sigma")) or 0.0, t_step * 3.0)
        trace = np.full_like(t, baseline, dtype=float)
        for row, meta in zip(trial_rows, trial_meta):
            weight = response_weight(row, selector, bandwidth)
            if weight <= 0.0:
                continue
            center = float(meta["start"]) + float(meta["duration"]) / 2.0 + latency
            trace += amplitude * weight * gaussian(t, center, kernel_width)
        return trace
    if kind == "constant":
        amplitude = as_float(response.get("amplitude")) or 0.0
        return np.full_like(t, amplitude, dtype=float)
    if kind == "noise":
        return np.zeros_like(t, dtype=float)
    raise SystemExit(f"Unknown response kind: {response.get('kind')}")


def assign_layout_nodes(experiment: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, Any]]]:
    layout = experiment.get("layout") or {}
    mode = normalize_name(experiment.get("mode", layout.get("mode", "soma_only")))
    n_cells = int(layout.get("n_cells", 1))
    dendrites_per_cell = int(layout.get("dendrites_per_cell", 1))
    spines_per_dendrite = int(layout.get("spines_per_dendrite", 3))
    children_label = "spine" if mode == "normal" else "axon"
    nodes: List[Dict[str, Any]] = []
    ref_to_index: Dict[str, int] = {}
    ref_to_node: Dict[str, Dict[str, Any]] = {}

    def add_node(node: Dict[str, Any]) -> None:
        ref = str(node["ref"])
        node = dict(node)
        node["index"] = len(nodes)
        nodes.append(node)
        ref_to_index[ref] = node["index"]
        ref_to_node[ref] = node

    for cell_index in range(1, n_cells + 1):
        soma_ref = f"cell{cell_index}.soma1"
        add_node(
            {
                "ref": soma_ref,
                "role": "soma",
                "mode": mode,
                "cell_index": cell_index,
                "dendrite_index": None,
                "child_index": None,
                "parent_ref": None,
                "compartment": str(experiment.get("compartment", "soma")),
                "analysis_index": None,
            }
        )
        if mode == "soma_only":
            continue
        for dend_index in range(1, dendrites_per_cell + 1):
            dend_ref = f"cell{cell_index}.dend{dend_index}"
            add_node(
                {
                    "ref": dend_ref,
                    "role": "dendrite",
                    "mode": mode,
                    "cell_index": cell_index,
                    "dendrite_index": dend_index,
                    "child_index": None,
                    "parent_ref": soma_ref,
                    "compartment": str(experiment.get("compartment", "basal")),
                    "analysis_index": None,
                }
            )
            for child_index in range(1, spines_per_dendrite + 1):
                child_ref = f"{dend_ref}.{children_label}{child_index}"
                add_node(
                    {
                        "ref": child_ref,
                        "role": children_label,
                        "mode": mode,
                        "cell_index": cell_index,
                        "dendrite_index": dend_index,
                        "child_index": child_index,
                        "parent_ref": dend_ref,
                        "compartment": str(experiment.get("compartment", "basal")),
                        "analysis_index": None,
                    }
                )
    background_rois = int(layout.get("background_rois", 0))
    for background_index in range(1, background_rois + 1):
        ref = f"background{background_index}"
        add_node(
            {
                "ref": ref,
                "role": "background",
                "mode": mode,
                "cell_index": None,
                "dendrite_index": None,
                "child_index": None,
                "parent_ref": None,
                "compartment": "other",
                "analysis_index": None,
            }
        )
    return nodes, ref_to_index, ref_to_node


def assign_conversion_indices(nodes: List[Dict[str, Any]], mode: str) -> None:
    analysis_counter = 0
    for node in nodes:
        role = str(node.get("role", ""))
        if mode == "soma_only":
            node["analysis_index"] = None
            continue
        if role in {"dendrite", "spine", "axon", "bouton"}:
            node["analysis_index"] = analysis_counter
            analysis_counter += 1
        else:
            node["analysis_index"] = None


def build_conversion_library(nodes: Sequence[Dict[str, Any]], mode: str) -> Dict[int, Dict[str, Any]]:
    conversion: Dict[int, Dict[str, Any]] = {}
    for node in nodes:
        if node.get("analysis_index") is None:
            continue
        analysis_index = int(node["analysis_index"])
        role = str(node.get("role"))
        if mode == "normal":
            if role == "dendrite":
                roi_type = [1, int(node["cell_index"]), int(node["dendrite_index"]), 0]
            elif role == "spine":
                roi_type = [2, int(node["cell_index"]), int(node["dendrite_index"]), int(node["child_index"])]
            else:
                continue
        elif mode == "dendrite_axon":
            if role == "dendrite":
                roi_type = [0, int(node["dendrite_index"]), 0, 0, 0]
            elif role in {"axon", "bouton"}:
                roi_type = [1, int(node["dendrite_index"]), int(node["child_index"]), 0, 0]
            else:
                continue
        else:
            continue
        conversion[analysis_index] = {
            "roi-type": roi_type,
            "conversion index": analysis_index,
            "plane": 0,
            "conversion": [0, analysis_index],
        }
    return conversion


def default_response_spec(node: Dict[str, Any]) -> Dict[str, Any]:
    role = str(node.get("role", ""))
    if role == "soma":
        return {"kind": "noise", "amplitude": 0.0}
    if role == "dendrite":
        return {
            "kind": "state",
            "state_weights": {"quiet_movies": 0.1, "active_movies": 0.2},
            "baseline": 0.05,
            "noise_scale": 0.02,
        }
    if role in {"spine", "axon", "bouton"}:
        return {
            "kind": "inherit",
            "parent_ref": node.get("parent_ref"),
            "alpha": 0.75,
            "specific": {"kind": "noise", "amplitude": 0.0},
        }
    return {"kind": "noise", "amplitude": 0.0}


def ellipse_points(cx: float, cy: float, rx: float, ry: float, n_points: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, int(max(8, n_points)), endpoint=False)
    x = cx + rx * np.cos(angles)
    y = cy + ry * np.sin(angles)
    return x, y


def demo_roi_geometry(node: Dict[str, Any], cell_stride: Tuple[int, int], image_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    role = normalize_name(node.get("role", ""))
    cell_index = max(1, int(node.get("cell_index") or 1))
    dendrite_index = max(1, int(node.get("dendrite_index") or 1))
    child_index = max(1, int(node.get("child_index") or 1))
    stride_x, stride_y = cell_stride
    width, height = image_shape[1], image_shape[0]
    base_x = 62 + stride_x * (cell_index - 1)
    base_y = 78 + stride_y * (cell_index - 1)
    base_x = float(min(max(base_x, 24), max(24, width - 24)))
    base_y = float(min(max(base_y, 24), max(24, height - 24)))

    if role == "soma":
        return base_x, base_y, 13.5, 11.0
    if role == "dendrite":
        return base_x + 42 + 14 * (dendrite_index - 1), base_y, 20.0, 5.5
    if role == "spine":
        cx = base_x + 58 + 14 * (dendrite_index - 1) + 10 * child_index
        cy = base_y + (8 if child_index % 2 else -8)
        return float(min(cx, width - 18)), float(min(max(cy, 18), height - 18)), 5.0, 3.8
    if role == "axon":
        return base_x + 50 + 16 * (dendrite_index - 1), base_y + (6 if child_index % 2 else -6), 18.0, 4.0
    if role == "bouton":
        return base_x + 56 + 12 * (dendrite_index - 1) + 8 * child_index, base_y + (6 if child_index % 2 else -6), 6.0, 6.0
    return base_x, base_y, 6.0, 6.0


def write_demo_suite2p_artifacts(exp_root: Path, nodes: Sequence[Dict[str, Any]], seed: int) -> None:
    suite2p_plane_dir = ensure_dir(exp_root / "suite2p" / "plane0")
    analysis_nodes = [node for node in nodes if node.get("analysis_index") is not None]
    if not analysis_nodes:
        analysis_nodes = [node for node in nodes if str(node.get("role")) == "soma"] or list(nodes)
    analysis_nodes = sorted(
        analysis_nodes,
        key=lambda node: (
            int(node.get("analysis_index") or 0),
            str(node.get("role") or ""),
            int(node.get("cell_index") or 0),
            int(node.get("dendrite_index") or 0),
            int(node.get("child_index") or 0),
        ),
    )
    cell_count = max(1, max(int(node.get("cell_index") or 1) for node in nodes))
    stride_x = 120 if cell_count <= 2 else 100
    stride_y = 100 if cell_count <= 2 else 90
    canvas_width = int(max(240, 180 + stride_x * max(0, cell_count - 1)))
    canvas_height = int(max(220, 180 + stride_y * max(0, cell_count - 1)))
    image_shape = (canvas_height, canvas_width)
    xx, yy = np.meshgrid(np.arange(canvas_width, dtype=float), np.arange(canvas_height, dtype=float))
    mean_img = np.full(image_shape, 0.04, dtype=float)
    rng = np.random.default_rng(int(seed) if seed is not None else 0)
    for node in nodes:
        cx, cy, rx, ry = demo_roi_geometry(node, (stride_x, stride_y), image_shape)
        role = normalize_name(node.get("role", ""))
        amplitude = {
            "soma": 1.1,
            "dendrite": 0.95,
            "spine": 0.82,
            "axon": 0.88,
            "bouton": 0.76,
            "background": 0.24,
        }.get(role, 0.5)
        sx = max(rx * 1.5, 3.0)
        sy = max(ry * 1.5, 3.0)
        mean_img += amplitude * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    mean_img += 0.01 * rng.normal(size=image_shape)
    mean_img = mean_img - float(np.nanmin(mean_img))
    if np.nanmax(mean_img) > 0:
        mean_img = mean_img / float(np.nanmax(mean_img))
    mean_img_e = np.clip(mean_img + 0.015 * rng.normal(size=image_shape), 0.0, 1.0)

    ops = {
        "Ly": int(image_shape[0]),
        "Lx": int(image_shape[1]),
        "nplanes": 1,
        "meanImg": mean_img.astype(np.float32),
        "meanImgE": mean_img_e.astype(np.float32),
    }
    np.save(suite2p_plane_dir / "ops.npy", ops, allow_pickle=True)

    stat_entries: List[Dict[str, Any]] = []
    for node in analysis_nodes:
        cx, cy, rx, ry = demo_roi_geometry(node, (stride_x, stride_y), image_shape)
        role = normalize_name(node.get("role", ""))
        if role == "dendrite":
            contour_rx, contour_ry = 19.0, 5.6
        elif role == "spine":
            contour_rx, contour_ry = 5.5, 3.7
        elif role == "axon":
            contour_rx, contour_ry = 17.0, 4.2
        elif role == "bouton":
            contour_rx, contour_ry = 6.0, 5.8
        else:
            contour_rx, contour_ry = max(rx, 6.0), max(ry, 5.0)
        xpix, ypix = ellipse_points(cx, cy, contour_rx, contour_ry, n_points=48)
        xpix = np.clip(np.rint(xpix), 0, image_shape[1] - 1).astype(np.int32)
        ypix = np.clip(np.rint(ypix), 0, image_shape[0] - 1).astype(np.int32)
        stat_entries.append(
            {
                "xpix": xpix,
                "ypix": ypix,
                "med": np.asarray([float(cy), float(cx)], dtype=np.float32),
            }
        )
    np.save(suite2p_plane_dir / "stat.npy", np.asarray(stat_entries, dtype=object), allow_pickle=True)


def compute_trial_mean(trace: np.ndarray, t: np.ndarray, start: float, end: float) -> float:
    mask = interval_mask(t, start, end)
    if not np.any(mask):
        return float("nan")
    values = np.asarray(trace[mask], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def stimulus_match_score(row: Dict[str, Any], selector: Dict[str, Any], bandwidth: Dict[str, Any]) -> float:
    if not selector:
        return 1.0
    score = 1.0
    for field, target in selector.items():
        value = as_float(row.get(field))
        if value is None:
            continue
        sigma = as_float(bandwidth.get(field)) if bandwidth else None
        if sigma is None or sigma <= 0:
            sigma = 10.0
        score *= math.exp(-0.5 * ((value - float(target)) / sigma) ** 2)
    return float(score)


def build_experiment(
    output_repo: Path,
    source_root: Path,
    experiment: Dict[str, Any],
    top_level_recipe: Dict[str, Any],
) -> Dict[str, Any]:
    exp_id = str(experiment["exp_id"])
    animal_id = str(experiment.get("animal_id", top_level_recipe.get("user_id", "demo")))
    mode = normalize_name(experiment.get("mode", "soma_only"))
    kind = normalize_name(experiment.get("kind", "movie"))
    exp_root = ensure_dir(output_repo / animal_id / exp_id)
    recordings_dir = ensure_dir(exp_root / "recordings")
    cut_dir = ensure_dir(exp_root / "cut")
    sleep_score_dir = ensure_dir(exp_root / "sleep_score")
    spinesgui_dir = ensure_dir(exp_root / "suite2p" / "SpinesGUI")
    t_start = as_float(experiment.get("t_start")) or 0.0
    t_end = as_float(experiment.get("t_end")) or 120.0
    dt = as_float(experiment.get("dt")) or 0.05
    t = np.arange(t_start, t_end, dt, dtype=float)
    t_step = float(dt)
    n_rois = int(experiment.get("n_rois", 8))
    base_noise = as_float(experiment.get("base_noise_scale")) or 0.03
    rng = np.random.default_rng(int(experiment.get("seed") or int(stable_hash(exp_id)[:8], 16)))

    trial_rows, trial_state_labels, trial_meta, trial_fieldnames = build_trial_rows(experiment, t, source_root)
    nodes, ref_to_index, ref_to_node = assign_layout_nodes(experiment)
    assign_conversion_indices(nodes, mode)
    n_rois = max(n_rois, len(nodes))
    traces = np.zeros((n_rois, t.size), dtype=float)

    # Baseline background traces.
    for roi_index in range(n_rois):
        traces[roi_index] = base_noise * rng.normal(size=t.size)

    # Build a wheel/pupil trace so the loader can produce state masks and QC plots.
    wheel_noise_scale = as_float(experiment.get("wheel_noise_scale")) or 0.06
    wheel_motion_scale = as_float(experiment.get("wheel_motion_scale")) or 1.0
    pupil_scale = as_float(experiment.get("pupil_scale")) or 0.6
    pupil_noise_scale = as_float(experiment.get("pupil_noise_scale")) or 0.04
    wheel_speed = wheel_noise_scale * rng.normal(size=t.size)
    wheel_active_segments = experiment.get(
        "wheel_active_segments",
        [
            (6, 12),
            (24, 32),
            (48, 56),
            (78, 86),
            (116, 124),
            (150, 160),
        ],
    )
    for start, end in wheel_active_segments:
        mask = interval_mask(t, float(start), float(end))
        if np.any(mask):
            wheel_speed[mask] += wheel_motion_scale + 0.1 * rng.normal(size=int(mask.sum()))
    wheel_bundle = {"t": t, "speed": wheel_speed}
    pupil = 1.4 + pupil_scale * np.tanh(wheel_speed) + pupil_noise_scale * rng.normal(size=t.size)
    write_pickle(recordings_dir / "wheel.pickle", wheel_bundle)
    write_pickle(recordings_dir / "dlcEyeLeft_resampled.pickle", {"t": t, "pupil_diameter": pupil, "speed": wheel_speed})
    write_pickle(recordings_dir / "dlcEyeRight_resampled.pickle", {"t": t, "pupil_diameter": pupil + 0.01 * rng.normal(size=t.size), "speed": wheel_speed})

    sleep_bundle: Optional[Dict[str, Any]] = None
    if kind == "sleep" or experiment.get("make_sleep_state"):
        sleep_bundle = build_sleep_state_bundle(t, experiment, wheel_speed)

    # Build drives in topological order so inherit responses can reference parents.
    response_specs: Dict[str, Dict[str, Any]] = {}
    for item in experiment.get("responses", []):
        if not isinstance(item, dict):
            continue
        roi_ref = str(item.get("roi_ref") or item.get("ref") or "")
        if not roi_ref:
            continue
        response_specs[roi_ref] = dict(item.get("response") or {})

    trace_map: Dict[str, np.ndarray] = {}
    pending = list(nodes)
    guard = 0
    while pending:
        guard += 1
        if guard > 1000:
            raise SystemExit(f"Could not resolve ROI response dependencies for experiment {exp_id}")
        node = pending.pop(0)
        ref = str(node["ref"])
        response = response_specs.get(ref, default_response_spec(node))
        kind_name = normalize_name(response.get("kind", "noise"))
        if kind_name == "inherit":
            parent_ref = str(response.get("parent_ref") or node.get("parent_ref") or "")
            if not parent_ref or parent_ref not in trace_map:
                pending.append(node)
                continue
            alpha = as_float(response.get("alpha")) or 0.75
            specific = response.get("specific") or {"kind": "noise", "amplitude": 0.0}
            parent_trace = trace_map[parent_ref]
            drive = build_drive_trace(
                t,
                specific,
                trial_rows,
                trial_state_labels,
                trial_meta,
                t_step,
                experiment_kind=kind,
                sleep_bundle=sleep_bundle,
            )
            baseline = as_float(response.get("baseline")) or 0.0
            noise_scale = as_float(response.get("noise_scale")) or 0.02
            trace = alpha * parent_trace + drive + baseline + noise_scale * rng.normal(size=t.size)
        else:
            drive = build_drive_trace(
                t,
                response,
                trial_rows,
                trial_state_labels,
                trial_meta,
                t_step,
                experiment_kind=kind,
                sleep_bundle=sleep_bundle,
            )
            baseline = as_float(response.get("baseline")) or 0.0
            noise_scale = as_float(response.get("noise_scale")) or 0.02
            trace = drive + baseline + noise_scale * rng.normal(size=t.size)
        trace_map[ref] = np.asarray(trace, dtype=float)
        index = int(node["index"])
        if index < traces.shape[0]:
            traces[index] = trace_map[ref]

    # Add a couple of low-amplitude nuisance traces so the ROI count is not entirely explained by the targets.
    for node in nodes:
        index = int(node["index"])
        if node.get("role") in {"background", "soma"} and index < traces.shape[0]:
            traces[index] = traces[index] + 0.01 * rng.normal(size=t.size)

    write_pickle(
        recordings_dir / "s2p_ch0.pickle",
        {
            "t": t,
            "dF": traces,
            "OriginalSuite2pCellIDs": np.arange(traces.shape[0], dtype=int),
        },
    )

    # Trial-aligned cut file.
    cut_points = int(experiment.get("cut_points", 50))
    cut_t = np.linspace(0.0, as_float(experiment.get("trial_duration")) or 5.0, cut_points, dtype=float)
    cut_neural = np.zeros((traces.shape[0], len(trial_rows), cut_points), dtype=float)
    cut_wheel = np.zeros((len(trial_rows), cut_points), dtype=float)
    for trial_index, row in enumerate(trial_rows):
        trial_start = as_float(row.get("time")) or 0.0
        trial_duration = as_float(row.get("duration")) or 5.0
        absolute_time = trial_start + cut_t
        cut_wheel[trial_index] = np.interp(absolute_time, t, wheel_speed)
        for roi_index in range(traces.shape[0]):
            cut_neural[roi_index, trial_index] = np.interp(absolute_time, t, traces[roi_index])
    write_pickle(
        cut_dir / "s2p_ch0_dF_cut.pickle",
        {
            "t": cut_t,
            "dF": cut_neural,
            "trial_index": np.arange(len(trial_rows), dtype=int),
            "trial_rows": trial_rows,
        },
    )
    write_pickle(
        cut_dir / "wheel.pickle",
        {
            "t": cut_t,
            "speed": cut_wheel,
            "trial_index": np.arange(len(trial_rows), dtype=int),
        },
    )

    # Optional sleep state bundle.
    if sleep_bundle is not None:
        if not bool(experiment.get("omit_sleep_state", False)):
            write_pickle(sleep_score_dir / "sleep_state.pickle", sleep_bundle)

    # Conversion library for dendrite/spine or dendrite/axon layouts.
    if mode in {"normal", "dendrite_axon"}:
        conversion = build_conversion_library(nodes, mode)
        if conversion:
            np.save(
                spinesgui_dir / (NORMAL_CONVERSION_FILENAME if mode == "normal" else DEND_AXON_CONVERSION_FILENAME),
                conversion,
                allow_pickle=True,
            )

    # Synthetic Suite2p anatomy for the shared day figure.
    write_demo_suite2p_artifacts(exp_root, nodes, int(experiment.get("seed") or 0))

    # Persist the trial table.
    write_csv_rows(exp_root / f"{exp_id}_all_trials.csv", trial_rows, trial_fieldnames)

    # Record planted truth for validation and downstream manifesting.
    target_rows: List[Dict[str, Any]] = []
    for roi_ref, response in response_specs.items():
        node = ref_to_node.get(roi_ref)
        if node is None:
            continue
        target_rows.append(
            {
                "roi_ref": roi_ref,
                "role": node.get("role"),
                "index": node.get("index"),
                "mode": normalize_name(response.get("kind", "noise")),
                "response": jsonable(response),
                "parent_ref": response.get("parent_ref") or node.get("parent_ref"),
            }
        )

    # Derive global dendrite/spine IDs when the conversion file exists.
    expected_ids: List[Dict[str, Any]] = []
    date = str(exp_id).split("_", 1)[0]
    if mode in {"normal", "dendrite_axon"}:
        for node in nodes:
            if node.get("analysis_index") is None:
                continue
            if str(node.get("role")) not in {"dendrite", "spine", "axon", "bouton"}:
                continue
            if str(node.get("role")) == "dendrite":
                expected_ids.append(
                    {
                        "roi_ref": node["ref"],
                        "global_dendrite_id": f"{animal_id}|{date}|cell{node.get('cell_index', 1)}|d{int(node.get('dendrite_index', 1))}",
                    }
                )
            else:
                expected_ids.append(
                    {
                        "roi_ref": node["ref"],
                        "global_spine_id": f"{animal_id}|{date}|cell{node.get('cell_index', 1)}|d{int(node.get('dendrite_index', 1))}|s{int(node.get('child_index', 1))}",
                    }
                )

    return {
        "exp_id": exp_id,
        "animal_id": animal_id,
        "kind": kind,
        "mode": mode,
        "compartment": str(experiment.get("compartment", "basal")),
        "repo_root": str(exp_root),
        "source_root": str(source_root),
        "trial_rows": trial_rows,
        "trial_state_labels": trial_state_labels,
        "trial_meta": trial_meta,
        "trial_fieldnames": trial_fieldnames,
        "nodes": nodes,
        "roi_refs": {node["ref"]: node["index"] for node in nodes},
        "expected_ids": expected_ids,
        "targets": target_rows,
        "sleep_state": jsonable(sleep_bundle) if sleep_bundle is not None else None,
        "analysis_families": list(top_level_recipe.get("analysis_families", DEFAULT_ANALYSIS_FAMILIES)),
    }


def build_repository(recipe: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    recipe = merge_recipe(recipe)
    repo_subdir = str(recipe.get("repo_subdir", DEFAULT_REPO_SUBDIR))
    repo_base = ensure_dir(output_dir / repo_subdir)
    experiments = recipe.get("experiments") or []
    if not experiments:
        raise SystemExit("Recipe must include at least one experiment")
    source_root = Path(recipe.get("stimulus_source_root") or DEFAULT_STIMULUS_SOURCE_ROOT)

    if repo_base.exists():
        shutil.rmtree(repo_base)
    ensure_dir(repo_base)

    exp_summaries: List[Dict[str, Any]] = []
    movie_expids: List[str] = []
    sleep_expids: List[str] = []
    basal_expids: List[str] = []
    apical_expids: List[str] = []
    for experiment in experiments:
        exp_summary = build_experiment(repo_base, source_root, experiment, recipe)
        exp_summaries.append(exp_summary)
        exp_id = str(exp_summary["exp_id"])
        kind = str(exp_summary["kind"])
        compartment = str(exp_summary["compartment"])
        if kind == "sleep":
            sleep_expids.append(exp_id)
        else:
            movie_expids.append(exp_id)
        if compartment == "basal":
            basal_expids.append(exp_id)
        if compartment == "apical":
            apical_expids.append(exp_id)

    if "movie_expids" in recipe:
        movie_expids = parse_list_argument(recipe.get("movie_expids"))
    if "sleep_expids" in recipe:
        sleep_expids = parse_list_argument(recipe.get("sleep_expids"))
    if "basal_expids" in recipe:
        basal_expids = parse_list_argument(recipe.get("basal_expids"))
    if "apical_expids" in recipe:
        apical_expids = parse_list_argument(recipe.get("apical_expids"))

    manifest = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "recipe_hash": stable_hash(recipe),
        "output_dir": str(output_dir),
        "repo_subdir": repo_subdir,
        "repo_base": str(repo_base),
        "user_id": str(recipe.get("user_id", "demo_user")),
        "channel": int(recipe.get("channel", DEFAULT_CHANNEL)),
        "locomotion_threshold": as_float(recipe.get("locomotion_threshold")) or DEFAULT_LOCOMOTION_THRESHOLD,
        "shuffle_n": int(recipe.get("shuffle_n", DEFAULT_SHUFFLES)),
        "high_pass_hz": as_float(recipe.get("high_pass_hz")) or DEFAULT_HIGH_PASS_HZ,
        "analysis_families": canonical_analysis_families(recipe.get("analysis_families")),
        "movie_expids": movie_expids,
        "sleep_expids": sleep_expids,
        "basal_expids": basal_expids,
        "apical_expids": apical_expids,
        "stimulus_source_root": str(source_root),
        "experiments": exp_summaries,
    }

    write_json(output_dir / repo_subdir / "demo_recipe_applied.json", recipe)
    write_json(output_dir / repo_subdir / "demo_manifest.json", manifest)
    write_json(output_dir / repo_subdir / "demo_truth.json", {"experiments": exp_summaries, "recipe": recipe})
    return manifest


def build_analysis_config(recipe: Dict[str, Any], manifest: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    repo_base = Path(manifest["repo_base"])
    return {
        "user_id": str(manifest["user_id"]),
        "repo_base": str(repo_base),
        "movie_expids": list(manifest["movie_expids"]),
        "sleep_expids": list(manifest["sleep_expids"]),
        "basal_expids": list(manifest["basal_expids"]),
        "apical_expids": list(manifest["apical_expids"]),
        "channel": int(manifest["channel"]),
        "locomotion_threshold": float(manifest["locomotion_threshold"]),
        "shuffle_n": int(manifest["shuffle_n"]),
        "high_pass_hz": float(manifest["high_pass_hz"]),
        "rebuild": True,
        "output_dir": str(output_dir),
        "figure_output_dir": str(output_dir / "figures" / "demo"),
        "cache_path": str(output_dir / "sleep_dendrite_spine_cache.npz"),
        "analysis_families": canonical_analysis_families(recipe.get("analysis_families")),
        "stimulus_source_root": str(manifest["stimulus_source_root"]),
        "stimulus_cache_root": str(recipe.get("stimulus_cache_root", "/home/rubencorreia/data/zebra_movies")),
        "gabor_save_path": str(recipe.get("gabor_save_path", "/home/rubencorreia/data/zebra_movies/gabors_library.npy")),
        "build_full_gabor_library": bool(recipe.get("build_full_gabor_library", False)),
    }


def validate_targets(manifest: Dict[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    repo_base = Path(manifest["repo_base"])
    rows: List[Dict[str, Any]] = []
    for experiment in manifest.get("experiments", []):
        exp_id = str(experiment.get("exp_id"))
        animal_id = str(experiment.get("animal_id"))
        exp_root = repo_base / animal_id / exp_id
        trial_csv = exp_root / f"{exp_id}_all_trials.csv"
        if not trial_csv.exists():
            continue

        with trial_csv.open("r", encoding="utf-8", newline="") as handle:
            trial_rows = list(csv.DictReader(handle))

        calcium_bundle = read_pickle(exp_root / "recordings" / "s2p_ch0.pickle")
        t = np.asarray(calcium_bundle["t"], dtype=float)
        dff = np.asarray(calcium_bundle["dF"], dtype=float)

        sleep_bundle = None
        sleep_state_path = exp_root / "sleep_score" / "sleep_state.pickle"
        if sleep_state_path.exists():
            try:
                sleep_bundle = read_pickle(sleep_state_path)
            except Exception:
                sleep_bundle = None

        ref_to_index = {str(k): int(v) for k, v in (experiment.get("roi_refs") or {}).items()}
        response_specs = {
            str(target["roi_ref"]): target["response"] for target in experiment.get("targets", []) if target.get("roi_ref")
        }

        for roi_ref, response in response_specs.items():
            index = ref_to_index.get(roi_ref)
            if index is None or index >= dff.shape[0]:
                continue
            trace = dff[index]
            kind = normalize_name(response.get("kind", "noise"))

            if kind == "state":
                weights = response.get("state_weights") or response.get("state_effects") or {}
                if experiment.get("kind") == "sleep" and sleep_bundle is not None:
                    state_labels = sleep_bundle.get("state_labels") or {0: "active_awake", 1: "quiet_awake", 2: "nrem", 3: "rem"}
                    state_codes = np.asarray(sleep_bundle.get("state_10hz"), dtype=int)
                    matched_mask = np.zeros_like(state_codes, dtype=bool)
                    control_mask = np.zeros_like(state_codes, dtype=bool)
                    for code, label in state_labels.items():
                        mask = state_codes == int(code)
                        if str(label) in weights:
                            matched_mask |= mask
                        else:
                            control_mask |= mask
                    if not np.any(matched_mask):
                        matched_mask = np.ones_like(state_codes, dtype=bool)
                    if not np.any(control_mask):
                        control_mask = ~matched_mask
                    matched_values = np.asarray(trace[matched_mask], dtype=float)
                    control_values = np.asarray(trace[control_mask], dtype=float)
                    matched_values = matched_values[np.isfinite(matched_values)]
                    control_values = control_values[np.isfinite(control_values)]
                    matched_mean = float(np.nanmean(matched_values)) if matched_values.size else float("nan")
                    control_mean = float(np.nanmean(control_values)) if control_values.size else float("nan")
                    matched_trials = int(np.count_nonzero(matched_mask))
                    control_trials = int(np.count_nonzero(control_mask))
                else:
                    matched = [
                        i
                        for i, row in enumerate(trial_rows)
                        if str(row.get("state_label", "")) in weights or str(row.get("category", "")) in weights
                    ]
                    control = [i for i in range(len(trial_rows)) if i not in matched]
                    if not matched:
                        matched = list(range(len(trial_rows)))
                    if not control:
                        control = matched
                    matched_means = [
                        compute_trial_mean(
                            trace,
                            t,
                            as_float(trial_rows[i]["time"]) or 0.0,
                            (as_float(trial_rows[i]["time"]) or 0.0) + (as_float(trial_rows[i]["duration"]) or 0.0),
                        )
                        for i in matched
                    ]
                    control_means = [
                        compute_trial_mean(
                            trace,
                            t,
                            as_float(trial_rows[i]["time"]) or 0.0,
                            (as_float(trial_rows[i]["time"]) or 0.0) + (as_float(trial_rows[i]["duration"]) or 0.0),
                        )
                        for i in control
                    ]
                    matched_mean = float(np.nanmean(matched_means)) if matched_means else float("nan")
                    control_mean = float(np.nanmean(control_means)) if control_means else float("nan")
                    matched_trials = len(matched)
                    control_trials = len(control)
                expected = float(max([as_float(v) or 0.0 for v in weights.values()] or [0.0]))
                recovered = bool(np.isfinite(matched_mean) and np.isfinite(control_mean) and matched_mean > control_mean)
                rows.append(
                    {
                        "exp_id": exp_id,
                        "roi_ref": roi_ref,
                        "response_kind": "state",
                        "state_weights": json.dumps(jsonable(weights), sort_keys=True),
                        "matched_trials": matched_trials,
                        "control_trials": control_trials,
                        "matched_mean": matched_mean,
                        "control_mean": control_mean,
                        "effect_size": matched_mean - control_mean,
                        "expected_peak": expected,
                        "recovered": recovered,
                    }
                )
                continue

            selector = response.get("selector") or {}
            bandwidth = response.get("bandwidth") or {}
            scores = [stimulus_match_score(row, selector, bandwidth) for row in trial_rows]
            max_score = max(scores) if scores else 0.0
            matched = [i for i, score in enumerate(scores) if score >= max_score * 0.8 and score > 0]
            control = [i for i, score in enumerate(scores) if score <= max_score * 0.2]
            if not matched:
                matched = [int(np.argmax(scores))] if scores else [0]
            if not control:
                control = [i for i in range(len(trial_rows)) if i not in matched]
            if not control:
                control = matched
            matched_means = [
                compute_trial_mean(
                    trace,
                    t,
                    as_float(trial_rows[i]["time"]) or 0.0,
                    (as_float(trial_rows[i]["time"]) or 0.0) + (as_float(trial_rows[i]["duration"]) or 0.0),
                )
                for i in matched
            ]
            control_means = [
                compute_trial_mean(
                    trace,
                    t,
                    as_float(trial_rows[i]["time"]) or 0.0,
                    (as_float(trial_rows[i]["time"]) or 0.0) + (as_float(trial_rows[i]["duration"]) or 0.0),
                )
                for i in control
            ]
            matched_mean = float(np.nanmean(matched_means)) if matched_means else float("nan")
            control_mean = float(np.nanmean(control_means)) if control_means else float("nan")
            expected = float(as_float(response.get("amplitude")) or 0.0)
            recovered = bool(np.isfinite(matched_mean) and np.isfinite(control_mean) and matched_mean > control_mean)
            rows.append(
                {
                    "exp_id": exp_id,
                    "roi_ref": roi_ref,
                    "response_kind": "stimulus",
                    "selector": json.dumps(jsonable(selector), sort_keys=True),
                    "matched_trials": len(matched),
                    "control_trials": len(control),
                    "matched_mean": matched_mean,
                    "control_mean": control_mean,
                    "effect_size": matched_mean - control_mean,
                    "expected_peak": expected,
                    "recovered": recovered,
                }
            )

    rows.sort(key=lambda row: (str(row.get("exp_id", "")), str(row.get("roi_ref", ""))))
    write_csv_rows(output_dir / "demo_target_validation.csv", rows, sorted({k for row in rows for k in row.keys()}))
    return rows


def run_pipeline_validation(recipe: Dict[str, Any], manifest: Dict[str, Any], output_dir: Path) -> int:
    analysis_config = build_analysis_config(recipe, manifest, output_dir)
    config_path = output_dir / "demo_analysis_config.json"
    write_json(config_path, analysis_config)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "sleep_dendrite_spine_pipeline.py"),
        "--config",
        str(config_path),
    ]
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def build_command(args: argparse.Namespace) -> int:
    recipe = default_recipe() if args.recipe is None else merge_recipe(load_json(args.recipe))
    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    ensure_dir(output_dir)
    manifest = build_repository(recipe, output_dir)
    write_json(output_dir / "demo_build_manifest.json", manifest)
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True))
    return 0


def validate_command(args: argparse.Namespace) -> int:
    recipe = default_recipe() if args.recipe is None else merge_recipe(load_json(args.recipe))
    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    ensure_dir(output_dir)
    manifest = build_repository(recipe, output_dir)
    write_json(output_dir / "demo_build_manifest.json", manifest)
    analysis_families = list(manifest.get("analysis_families", DEFAULT_ANALYSIS_FAMILIES))
    rc = run_pipeline_validation(recipe, manifest, output_dir)
    validation_rows = validate_targets(manifest, output_dir)
    validation_summary = {
        "analysis_return_code": rc,
        "n_validation_rows": len(validation_rows),
        "n_recovered": int(sum(1 for row in validation_rows if row.get("recovered"))),
        "analysis_families": analysis_families,
        "manifest": manifest,
    }
    write_json(output_dir / "demo_validation_summary.json", validation_summary)
    print(json.dumps(jsonable(validation_summary), indent=2, sort_keys=True))
    return rc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone synthetic demo builder for sleep/codex_analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a synthetic demo repository from a recipe")
    build_parser.add_argument("--recipe", type=Path, help="JSON recipe describing the synthetic experiment")
    build_parser.add_argument("--output-dir", type=Path, help="Directory that will hold the synthetic repo and manifests")

    validate_parser = subparsers.add_parser("validate", help="Build the synthetic repo and run the analysis pipeline")
    validate_parser.add_argument("--recipe", type=Path, help="JSON recipe describing the synthetic experiment")
    validate_parser.add_argument("--output-dir", type=Path, help="Directory that will hold the synthetic repo and outputs")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return build_command(args)
    if args.command == "validate":
        return validate_command(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
