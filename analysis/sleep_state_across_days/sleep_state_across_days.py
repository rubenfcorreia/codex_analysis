#!/usr/bin/env python3
from __future__ import annotations

import io
import argparse
import csv
import datetime as dt
import json
import math
import pickle
import re
import shutil
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as mpatheffects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import colors as mcolors
except Exception:  # pragma: no cover - matplotlib is required for real plotting
    plt = None
    mdates = None
    Line2D = None
    Patch = None
    mcolors = None
    mpatheffects = None

try:
    import cv2
except Exception:  # pragma: no cover - optional video decoding support
    cv2 = None

try:
    from scipy import signal as scipy_signal
except Exception:  # pragma: no cover - optional spectrogram support
    scipy_signal = None

from poster_plotting import (
    POSTER_DPI,
    POSTER_FONT_SIZE,
    POSTER_LABEL_SIZE,
    POSTER_LEGEND_SIZE,
    POSTER_NOTE_SIZE,
    POSTER_SUPTITLE_SIZE,
    POSTER_TITLE_SIZE,
    POSTER_WIDE_FIGSIZE,
    configure_poster_matplotlib,
    save_figure,
    set_sparse_numeric_ticks,
)

if plt is not None:
    configure_poster_matplotlib()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "sleep_state_across_days_config.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "sleep_state_across_days"
DEFAULT_POSTER_READY_DIR = ROOT_DIR / "results" / "poster_ready"
DEFAULT_REVIEW_FIGURES_DIRNAME = "review_figures"
DEFAULT_REVIEW_FIGURES_DIR = ROOT_DIR / DEFAULT_REVIEW_FIGURES_DIRNAME
DEFAULT_FIGURE_DIRNAME = "figures"
DEFAULT_STACKED_FIGURE_DIRNAME = "stacked_area"
DEFAULT_PROBABILITY_FIGURE_DIRNAME = "probability_time"
DEFAULT_PROBABILITY_PERCENT_FIGURE_DIRNAME = "probability_time_percent"
DEFAULT_REM_FIGURE_DIRNAME = "rem_summary"
DEFAULT_REM_FRACTION_FIGURE_DIRNAME = "fraction_time"
DEFAULT_COMPOSITION_FIGURE_DIRNAME = "composition_summary"
DEFAULT_POSTER_READY_FIGURE_STEM = "sleep_state_poster_composite"
DEFAULT_STATE_ORDER = ["active_wake", "quiet_wake", "nrem", "rem"]
DEFAULT_STATE_MONTAGE_EXP_FIGURE_DIRNAME = "per_exp"
DEFAULT_REM_LATENCY_METRICS = [
    ("latency_since_start_s", "First REM latency from recording start"),
    ("latency_since_active_wake_s", "First REM latency from active wake"),
]
DEFAULT_REM_PROBABILITY_METRICS = [
    ("latency_since_start_s", "REM probability from recording start"),
    ("latency_since_active_wake_s", "REM probability from active wake"),
]
DEFAULT_STATE_DISPLAY = {
    "active_wake": "Active wake",
    "quiet_wake": "Quiet wake",
    "nrem": "NREM",
    "rem": "REM",
}
DEFAULT_STACKED_STATE_ORDER = list(DEFAULT_STATE_ORDER) + ["unclassified"]
DEFAULT_STACKED_STATE_DISPLAY = {
    **DEFAULT_STATE_DISPLAY,
    "unclassified": "Unclassified",
}
DEFAULT_STACKED_STATE_COLORS = {
    "active_wake": "#4C78A8",
    "quiet_wake": "#F58518",
    "nrem": "#54A24B",
    "rem": "#E45756",
    "unclassified": "#BDBDBD",
}
DEFAULT_CATEGORY_ORDER = ["movie", "sleep"]
DEFAULT_CATEGORY_DISPLAY = {
    "movie": "Movie expIDs",
    "sleep": "Sleep expIDs",
}
DEFAULT_CATEGORY_COLORS = {
    "movie": "#4C78A8",
    "sleep": "#F58518",
}
DEFAULT_METRIC_SPECS = [
    ("state_fraction", "Fraction of time", "Fraction"),
    ("bout_count", "Bout count", "Bouts"),
    ("bout_mean_duration_s", "Mean bout duration (s)", "Bout duration (s)"),
]
DEFAULT_STACKED_UNCLASSIFIED_TOL = 1e-6
DEFAULT_PROBABILITY_BIN_S = 300.0
DEFAULT_PROBABILITY_CMAP = "viridis"
DEFAULT_PROBABILITY_TIME_LABEL = "Elapsed time (min)"
DEFAULT_STATE_MONTAGE_FIGURE_DIRNAME = "state_montage"
DEFAULT_REVIEW_STATE_MONTAGE_FIGURE_DIRNAME = "state_montage"
DEFAULT_REVIEW_STATE_MONTAGE_EXAMPLE_EXP_ID = "2026-05-19_03_ESRC028"
DEFAULT_POSTER_READY_STATE_MONTAGE_EXAMPLE_EXP_ID = "2026-05-04_03_ESRC028"


@dataclass
class SessionSummary:
    scope: str
    animal_id: str
    date: str
    category: str
    exp_ids: List[str]
    sleep_state_paths: List[str]
    epoch_count: int
    epoch_duration_s: float
    total_time_s: float
    unknown_epoch_count: int
    unknown_epoch_fraction: float
    state_time_s: Dict[str, float]
    state_fraction: Dict[str, float]
    bout_count: Dict[str, int]
    bout_total_time_s: Dict[str, float]
    bout_mean_duration_s: Dict[str, float]
    probability_time_s: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    state_probability_profile: Dict[str, np.ndarray] = field(default_factory=dict)
    animal_day_index: Optional[int] = None


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def parse_list_argument(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [item for item in text.split() if item.strip()]


def derive_animal_id(exp_id: str) -> str:
    parts = str(exp_id).split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive animalID from expID: {exp_id}")
    return parts[2]


def derive_date(exp_id: str) -> str:
    return str(exp_id).split("_", 1)[0]


def resolve_repo_root(repo_base: Path, animal_id: str, exp_id: str) -> Path:
    return repo_base / animal_id / exp_id


def safe_filename_component(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown"


def report_relative_path(path: Any, output_dir: Path) -> str:
    if path is None:
        return "n/a"
    text = str(path)
    try:
        path_obj = Path(text)
    except Exception:
        return text
    if not path_obj.is_absolute():
        return text.replace("\\", "/")
    try:
        return str(path_obj.relative_to(output_dir)).replace("\\", "/")
    except Exception:
        return text.replace("\\", "/")


def project_relative_path(path: Any) -> str:
    if path is None:
        return "n/a"
    text = str(path)
    try:
        path_obj = Path(text)
    except Exception:
        return text
    if not path_obj.is_absolute():
        return text.replace("\\", "/")
    try:
        return str(path_obj.relative_to(ROOT_DIR)).replace("\\", "/")
    except Exception:
        return text.replace("\\", "/")


def format_report_number(value: Any, precision: int = 3) -> str:
    number = as_float(value)
    if number is None or not np.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f}"


def format_report_percent(value: Any, precision: int = 1) -> str:
    number = as_float(value)
    if number is None or not np.isfinite(number):
        return "n/a"
    return f"{100.0 * number:.{precision}f}%"


def format_report_list(values: Any, max_items: int = 10) -> str:
    if values is None:
        return "none"
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return text if text else "none"
    try:
        items = [str(item) for item in values if str(item).strip()]
    except TypeError:
        text = str(values).strip()
        return text if text else "none"
    if not items:
        return "none"
    if len(items) > max_items:
        shown = ", ".join(items[:max_items])
        return f"{shown}, ... (+{len(items) - max_items} more)"
    return ", ".join(items)


def write_text_report(path: Path, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


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


def format_display_state(state: str) -> str:
    return DEFAULT_STATE_DISPLAY.get(state, state.replace("_", " ").title())


def format_display_category(category: str) -> str:
    return DEFAULT_CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())


def mean_of_finite(values: Sequence[Any]) -> float:
    arr = np.asarray([as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def finite_mean_and_sem(values: Sequence[Any]) -> Tuple[float, float]:
    arr = np.asarray([as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    if arr.size < 2:
        return mean, float("nan")
    sem = float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))
    return mean, sem


def columnwise_finite_mean(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    valid = np.isfinite(arr)
    counts = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, arr, 0.0), axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )


def columnwise_finite_mean_and_sem(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    valid = np.isfinite(arr)
    counts = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, arr, 0.0), axis=0)
    mean = np.divide(
        sums,
        counts,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    centered = np.where(valid, arr - mean, 0.0)
    sq_sum = np.sum(centered ** 2, axis=0)
    variance = np.divide(
        sq_sum,
        counts - 1,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 1,
    )
    sem = np.divide(
        np.sqrt(variance),
        np.sqrt(counts),
        out=np.zeros(arr.shape[1], dtype=float),
        where=counts > 1,
    )
    return mean, sem


def elapsed_experimental_time_positions(
    state_epoch_t: np.ndarray,
    epoch_count: int,
    epoch_duration_s: float,
) -> np.ndarray:
    if epoch_count <= 0:
        return np.asarray([], dtype=float)
    t = np.asarray(state_epoch_t, dtype=float).ravel()
    if t.size != epoch_count:
        t = t[:epoch_count]
    if t.size != epoch_count:
        step = float(epoch_duration_s) if np.isfinite(epoch_duration_s) and epoch_duration_s > 0 else 10.0
        return np.arange(epoch_count, dtype=float) * step
    if epoch_count == 1:
        return np.zeros(1, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        step = float(epoch_duration_s) if np.isfinite(epoch_duration_s) and epoch_duration_s > 0 else 10.0
        return np.arange(epoch_count, dtype=float) * step
    start_t = float(t[0])
    rel = np.asarray(t, dtype=float) - start_t
    if not np.all(np.isfinite(rel)):
        step = float(epoch_duration_s) if np.isfinite(epoch_duration_s) and epoch_duration_s > 0 else 10.0
        return np.arange(epoch_count, dtype=float) * step
    rel = np.clip(rel, 0.0, None)
    if rel.size and not np.all(np.diff(rel) >= 0):
        step = float(epoch_duration_s) if np.isfinite(epoch_duration_s) and epoch_duration_s > 0 else 10.0
        return np.arange(epoch_count, dtype=float) * step
    return rel


def pad_probability_series(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if length <= 0:
        return np.asarray([], dtype=float)
    if arr.size >= length:
        return arr[:length]
    out = np.full(length, np.nan, dtype=float)
    out[: arr.size] = arr
    return out


def build_probability_profile(
    state_epoch: np.ndarray,
    state_epoch_t: np.ndarray,
    epoch_duration_s: float,
    bin_size_s: float = DEFAULT_PROBABILITY_BIN_S,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    state_epoch = np.asarray(state_epoch, dtype=int).ravel()
    state_epoch_t = np.asarray(state_epoch_t, dtype=float).ravel()
    state_epoch, state_epoch_t = align_length(state_epoch, state_epoch_t)
    epoch_count = int(state_epoch.size)
    if epoch_count == 0:
        empty_profile = {state: np.asarray([], dtype=float) for state in DEFAULT_STACKED_STATE_ORDER}
        return np.asarray([], dtype=float), empty_profile

    bin_size_s = float(bin_size_s) if np.isfinite(bin_size_s) and bin_size_s > 0 else DEFAULT_PROBABILITY_BIN_S
    positions = elapsed_experimental_time_positions(state_epoch_t, epoch_count, epoch_duration_s)
    total_time_s = float(epoch_count * float(epoch_duration_s)) if np.isfinite(epoch_duration_s) and epoch_duration_s > 0 else float(np.nanmax(positions) if positions.size else 0.0)
    if not np.isfinite(total_time_s) or total_time_s <= 0:
        total_time_s = bin_size_s
    n_bins = max(1, int(math.ceil(total_time_s / bin_size_s)))
    centers = (np.arange(n_bins, dtype=float) + 0.5) * bin_size_s
    bin_index = np.floor(np.divide(positions, bin_size_s, out=np.zeros_like(positions, dtype=float), where=np.isfinite(positions))).astype(int)
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    total_counts = np.bincount(bin_index, minlength=n_bins).astype(float)
    profile: Dict[str, np.ndarray] = {}
    state_map = {
        0: "active_wake",
        1: "quiet_wake",
        2: "nrem",
        3: "rem",
    }
    for code, state in state_map.items():
        mask = state_epoch == code
        counts = np.bincount(bin_index[mask], minlength=n_bins).astype(float)
        profile[state] = np.divide(
            counts,
            total_counts,
            out=np.full(n_bins, np.nan, dtype=float),
            where=total_counts > 0,
        )
    unknown_mask = ~np.isin(state_epoch, list(state_map))
    counts = np.bincount(bin_index[unknown_mask], minlength=n_bins).astype(float)
    profile["unclassified"] = np.divide(
        counts,
        total_counts,
        out=np.full(n_bins, np.nan, dtype=float),
        where=total_counts > 0,
    )
    return centers, profile


def average_probability_summaries(
    summaries: Sequence[SessionSummary],
    include_unclassified: Optional[bool] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    valid = [summary for summary in summaries if summary.probability_time_s.size and summary.state_probability_profile]
    if not valid:
        return np.asarray([], dtype=float), {}
    max_len = max(int(summary.probability_time_s.size) for summary in valid)
    time_s = (np.arange(max_len, dtype=float) + 0.5) * DEFAULT_PROBABILITY_BIN_S
    if include_unclassified is None:
        include_unclassified = any(
            probability_series_has_unclassified(summary.state_probability_profile)
            for summary in valid
        )
    state_order = list(DEFAULT_STATE_ORDER)
    if include_unclassified:
        state_order.append("unclassified")
    profile: Dict[str, np.ndarray] = {}
    for state in state_order:
        matrix = np.vstack(
            [
                pad_probability_series(
                    summary.state_probability_profile.get(state, np.full(summary.probability_time_s.size, np.nan)),
                    max_len,
                )
                for summary in valid
            ]
        )
        profile[state] = columnwise_finite_mean(matrix)
    return time_s, profile


def probability_series_has_unclassified(profile: Mapping[str, np.ndarray]) -> bool:
    values = profile.get("unclassified")
    if values is None:
        return False
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(np.nanmax(arr) > DEFAULT_STACKED_UNCLASSIFIED_TOL)


def resample_probability_profile_to_percent(
    time_s: np.ndarray,
    values: np.ndarray,
    target_percent: np.ndarray,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    target_percent = np.asarray(target_percent, dtype=float).ravel()
    if time_s.size == 0 or values.size == 0 or target_percent.size == 0:
        return np.asarray([], dtype=float)
    if time_s.size != values.size:
        time_s, values = align_length(time_s, values)
    if time_s.size == 0 or values.size == 0:
        return np.full(target_percent.size, np.nan, dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(values)
    if not np.any(finite):
        return np.full(target_percent.size, np.nan, dtype=float)
    x_src = np.asarray(time_s[finite], dtype=float)
    y_src = np.asarray(values[finite], dtype=float)
    if x_src.size == 1:
        return np.full(target_percent.size, float(y_src[0]), dtype=float)
    x_min = float(np.nanmin(x_src))
    x_max = float(np.nanmax(x_src))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.full(target_percent.size, float(np.nanmean(y_src)), dtype=float)
    percent_src = 100.0 * (x_src - x_min) / max(x_max - x_min, 1e-12)
    order = np.argsort(percent_src)
    percent_src = percent_src[order]
    y_src = y_src[order]
    unique_percent, unique_indices = np.unique(percent_src, return_index=True)
    y_src = y_src[unique_indices]
    if unique_percent.size == 1:
        return np.full(target_percent.size, float(y_src[0]), dtype=float)
    return np.interp(target_percent, unique_percent, y_src)


def resample_probability_matrix_to_percent(
    summaries: Sequence[SessionSummary],
    state: str,
    target_percent: np.ndarray,
) -> np.ndarray:
    target_percent = np.asarray(target_percent, dtype=float).ravel()
    if target_percent.size == 0:
        return np.asarray([], dtype=float).reshape(0, 0)
    rows: List[np.ndarray] = []
    for summary in summaries:
        time_s = np.asarray(summary.probability_time_s, dtype=float).ravel()
        values = np.asarray(
            summary.state_probability_profile.get(state, np.full(time_s.size, np.nan)),
            dtype=float,
        ).ravel()
        if time_s.size == 0 or values.size == 0:
            continue
        if time_s.size != values.size:
            time_s, values = align_length(time_s, values)
        if time_s.size == 0 or values.size == 0:
            continue
        rows.append(resample_probability_profile_to_percent(time_s, values, target_percent))
    if not rows:
        return np.asarray([], dtype=float).reshape(0, target_percent.size)
    return np.vstack(rows)


def stacked_composition(summary: SessionSummary) -> Dict[str, float]:
    composition: Dict[str, float] = {}
    canonical_values = []
    for state in DEFAULT_STATE_ORDER:
        value = as_float(summary.state_fraction.get(state))
        if value is None or not np.isfinite(value):
            value = float("nan")
        composition[state] = float(value)
        if np.isfinite(value):
            canonical_values.append(float(value))
    residual = as_float(summary.unknown_epoch_fraction)
    if residual is None or not np.isfinite(residual):
        if len(canonical_values) == len(DEFAULT_STATE_ORDER):
            residual = max(0.0, 1.0 - float(sum(canonical_values)))
        else:
            residual = float("nan")
    residual = float(residual) if residual is not None and np.isfinite(residual) else float("nan")
    if np.isfinite(residual):
        residual = max(0.0, residual)
    composition["unclassified"] = residual
    return composition


def needs_unclassified_band(summaries: Sequence[SessionSummary]) -> bool:
    for summary in summaries:
        residual = as_float(summary.unknown_epoch_fraction)
        if residual is not None and np.isfinite(residual) and residual > DEFAULT_STACKED_UNCLASSIFIED_TOL:
            return True
    return False


def build_stacked_plot_handles(include_unclassified: bool) -> List[Any]:
    if Patch is None:
        return []
    labels = list(DEFAULT_STATE_ORDER)
    if include_unclassified:
        labels.append("unclassified")
    return [
        Patch(
            facecolor=DEFAULT_STACKED_STATE_COLORS[label],
            edgecolor="white",
            linewidth=0.6,
            label=DEFAULT_STACKED_STATE_DISPLAY[label],
        )
        for label in labels
    ]


def build_stacked_panel_series(
    summaries: Sequence[SessionSummary],
    x_mode: str,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], bool]:
    ordered = [summary for summary in summaries if summary.category in DEFAULT_CATEGORY_ORDER]
    if x_mode == 'date':
        ordered = sorted(
            ordered,
            key=lambda summary: (
                str(summary.date),
                str(summary.category),
                str(summary.exp_ids[0]) if summary.exp_ids else '',
            ),
        )
        include_unclassified = needs_unclassified_band(ordered)
        state_order = list(DEFAULT_STATE_ORDER)
        if include_unclassified:
            state_order.append('unclassified')
        x_values = np.asarray([mdates.date2num(dt.date.fromisoformat(str(summary.date))) for summary in ordered], dtype=float)
        compositions = [stacked_composition(summary) for summary in ordered]
        series_map = {state: np.asarray([composition[state] for composition in compositions], dtype=float) for state in state_order}
        return x_values, state_order, series_map, include_unclassified
    if x_mode == 'day_index':
        ordered = [summary for summary in ordered if summary.animal_day_index is not None]
        ordered = sorted(
            ordered,
            key=lambda summary: (
                int(summary.animal_day_index or 0),
                str(summary.animal_id),
                str(summary.date),
                str(summary.exp_ids[0]) if summary.exp_ids else '',
                str(summary.category),
            ),
        )
        include_unclassified = needs_unclassified_band(ordered)
        state_order = list(DEFAULT_STATE_ORDER)
        if include_unclassified:
            state_order.append('unclassified')
        grouped: Dict[int, List[SessionSummary]] = defaultdict(list)
        for summary in ordered:
            grouped[int(summary.animal_day_index)].append(summary)
        x_indices = sorted(grouped)
        x_values = np.asarray(x_indices, dtype=float)
        series_map: Dict[str, np.ndarray] = {}
        for state in state_order:
            values = []
            for day_index in x_indices:
                day_values = [stacked_composition(summary)[state] for summary in grouped[day_index]]
                values.append(mean_of_finite(day_values))
            series_map[state] = np.asarray(values, dtype=float)
        return x_values, state_order, series_map, include_unclassified
    raise ValueError(f'Unsupported stacked series mode: {x_mode}')


def stacked_panel_missing_state_messages(series_map: Mapping[str, np.ndarray], state_order: Sequence[str]) -> List[str]:
    messages: List[str] = []
    for state in state_order:
        if state == 'unclassified':
            continue
        series = np.asarray(series_map.get(state, np.asarray([], dtype=float)), dtype=float)
        finite = series[np.isfinite(series)]
        if finite.size == 0 or np.all(np.isclose(finite, 0.0, atol=1e-9)):
            messages.append(f'No {format_display_state(state)} epochs in source data')
    return messages


def render_stacked_area_panels_on_axes(
    axes: Any,
    category_summaries: Mapping[str, Sequence[SessionSummary]],
    *,
    x_mode: str,
    x_label: str,
) -> None:
    if plt is None:
        raise RuntimeError('matplotlib is required to generate figures')
    axes_arr = np.asarray(axes)
    for row_idx, category in enumerate(DEFAULT_CATEGORY_ORDER):
        ax = axes_arr[row_idx, 0] if axes_arr.ndim == 2 else axes_arr[row_idx]
        summaries = list(category_summaries.get(category, []))
        if summaries:
            x_values, state_order, series_map, _ = build_stacked_panel_series(summaries, x_mode)
            if x_values.size == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=max(8, POSTER_FONT_SIZE - 7), color='#666666')
            else:
                colors = [DEFAULT_STACKED_STATE_COLORS[state] for state in state_order]
                ax.stackplot(x_values, *[series_map[state] for state in state_order], colors=colors, alpha=0.95, linewidth=0.35, edgecolor='white')
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.22)
                set_sparse_numeric_ticks(ax, axis='y', nbins=5)
                if x_mode == 'date':
                    ax.xaxis_date()
                    format_date_axis(ax)
                    if x_values.size == 1:
                        ax.set_xlim(float(x_values[0]) - 0.5, float(x_values[0]) + 0.5)
                    else:
                        ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)))
                else:
                    if x_values.size == 1:
                        ax.set_xlim(float(x_values[0]) - 0.5, float(x_values[0]) + 0.5)
                    else:
                        ax.set_xlim(0.5, float(np.nanmax(x_values)) + 0.5)
                    set_sparse_numeric_ticks(ax, axis='x', nbins=min(6, int(np.nanmax(x_values)) + 1), integer=True)
                missing_state_messages = stacked_panel_missing_state_messages(series_map, state_order)
                if missing_state_messages:
                    ax.text(0.985, 0.985, '\n'.join(missing_state_messages), transform=ax.transAxes, ha='right', va='top', fontsize=max(12, POSTER_NOTE_SIZE - 1), color='#555555', bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#d0d0d0', linewidth=0.8, alpha=0.88), zorder=5)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=max(8, POSTER_FONT_SIZE - 7), color='#666666')
        ax.set_ylabel(f'Fraction of time\n{format_display_category(category)}', fontsize=max(13, POSTER_LABEL_SIZE - 2), labelpad=6)
        if x_mode == 'date' and summaries:
            if row_idx == 0:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.tick_params(axis='x', labelsize=POSTER_FONT_SIZE)
        else:
            if row_idx == 0:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.tick_params(axis='x', labelsize=POSTER_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=POSTER_FONT_SIZE)
    axes[1, 0].set_xlabel(x_label, fontsize=POSTER_LABEL_SIZE, labelpad=8)


def normalize_sleep_bundle(bundle: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(bundle, dict):
        raise TypeError(f"Unexpected sleep_state bundle type in {path}: {type(bundle).__name__}")
    if "state_epoch" not in bundle or "state_epoch_t" not in bundle:
        raise KeyError(f"sleep_state bundle missing state_epoch/state_epoch_t in {path}")
    return bundle


def load_sleep_state_bundle(exp_root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[str]]:
    sleep_state_path = exp_root / "sleep_score" / "sleep_state.pickle"
    if not sleep_state_path.exists():
        return None, None, f"missing {sleep_state_path.name}"
    try:
        bundle = normalize_sleep_bundle(load_pickle(sleep_state_path), sleep_state_path)
    except Exception as exc:
        return None, sleep_state_path, f"failed to load sleep_state.pickle: {exc}"
    return bundle, sleep_state_path, None


def align_length(*arrays: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
    valid = [np.asarray(arr) for arr in arrays if arr is not None]
    if not valid:
        return tuple(np.asarray(arr) if arr is not None else np.asarray([]) for arr in arrays)
    min_len = min(arr.shape[0] for arr in valid if arr.ndim > 0)
    aligned: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            aligned.append(np.asarray([]))
            continue
        arr = np.asarray(arr)
        if arr.ndim == 0:
            aligned.append(arr)
            continue
        aligned.append(arr[:min_len])
    return tuple(aligned)


def contiguous_runs(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=int).ravel()
    if arr.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), np.asarray([], dtype=int)
    change_points = np.flatnonzero(np.diff(arr) != 0) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, arr.size]
    lengths = ends - starts
    states = arr[starts]
    return starts, lengths, states


def epoch_duration_seconds(state_epoch_t: np.ndarray) -> float:
    t = np.asarray(state_epoch_t, dtype=float).ravel()
    if t.size < 2:
        return 10.0
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 10.0
    return float(np.nanmedian(diffs))


def summarize_state_bundle(
    exp_id: str,
    animal_id: str,
    date: str,
    category: str,
    bundle: Mapping[str, Any],
    sleep_state_path: Path,
) -> SessionSummary:
    state_epoch = np.asarray(bundle["state_epoch"], dtype=int).ravel()
    state_epoch_t = np.asarray(bundle["state_epoch_t"], dtype=float).ravel()
    state_epoch, state_epoch_t = align_length(state_epoch, state_epoch_t)
    epoch_duration_s = epoch_duration_seconds(state_epoch_t)
    epoch_count = int(state_epoch.size)
    total_time_s = float(epoch_count * epoch_duration_s)
    probability_time_s, state_probability_profile = build_probability_profile(
        state_epoch,
        state_epoch_t,
        epoch_duration_s,
    )

    canonical_codes = {0, 1, 2, 3}
    unknown_mask = ~np.isin(state_epoch, list(canonical_codes)) if state_epoch.size else np.asarray([], dtype=bool)
    unknown_epoch_count = int(np.count_nonzero(unknown_mask))
    unknown_epoch_time_s = float(unknown_epoch_count * epoch_duration_s)
    denominator_time_s = total_time_s if total_time_s > 0 else 0.0
    unknown_epoch_fraction = float(unknown_epoch_time_s / denominator_time_s) if denominator_time_s > 0 else float("nan")

    state_time_s: Dict[str, float] = {}
    state_fraction: Dict[str, float] = {}
    bout_count: Dict[str, int] = {}
    bout_total_time_s: Dict[str, float] = {}
    bout_mean_duration_s: Dict[str, float] = {}

    _, lengths, states = contiguous_runs(state_epoch)
    for state_code, state_key in [(0, "active_wake"), (1, "quiet_wake"), (2, "nrem"), (3, "rem")]:
        count = int(np.count_nonzero(state_epoch == state_code))
        state_time = float(count * epoch_duration_s)
        state_time_s[state_key] = state_time
        state_fraction[state_key] = float(state_time / denominator_time_s) if denominator_time_s > 0 else float("nan")
        state_runs = lengths[states == state_code] if lengths.size else np.asarray([], dtype=int)
        bout_count[state_key] = int(state_runs.size)
        bout_total = float(np.sum(state_runs) * epoch_duration_s) if state_runs.size else 0.0
        bout_total_time_s[state_key] = bout_total
        bout_mean_duration_s[state_key] = float(bout_total / state_runs.size) if state_runs.size else float("nan")

    return SessionSummary(
        scope="exp",
        animal_id=animal_id,
        date=date,
        category=category,
        exp_ids=[exp_id],
        sleep_state_paths=[str(sleep_state_path)],
        epoch_count=epoch_count,
        epoch_duration_s=epoch_duration_s,
        total_time_s=total_time_s,
        unknown_epoch_count=unknown_epoch_count,
        unknown_epoch_fraction=unknown_epoch_fraction,
        state_time_s=state_time_s,
        state_fraction=state_fraction,
        bout_count=bout_count,
        bout_total_time_s=bout_total_time_s,
        bout_mean_duration_s=bout_mean_duration_s,
        probability_time_s=probability_time_s,
        state_probability_profile=state_probability_profile,
    )


def aggregate_summaries(
    scope: str,
    animal_id: str,
    date: str,
    category: str,
    exp_summaries: Sequence[SessionSummary],
    day_index: Optional[int] = None,
) -> SessionSummary:
    if not exp_summaries:
        raise ValueError("Cannot aggregate an empty summary list")
    exp_ids = [exp.exp_ids[0] for exp in exp_summaries]
    sleep_state_paths = [path for exp in exp_summaries for path in exp.sleep_state_paths]
    total_time_s = float(sum(exp.total_time_s for exp in exp_summaries))
    epoch_count = int(sum(exp.epoch_count for exp in exp_summaries))
    epoch_duration_s = float(total_time_s / epoch_count) if epoch_count > 0 else float("nan")
    unknown_epoch_count = int(sum(exp.unknown_epoch_count for exp in exp_summaries))
    unknown_epoch_fraction = float(
        unknown_epoch_count / epoch_count if epoch_count > 0 else float("nan")
    )
    state_time_s: Dict[str, float] = {}
    state_fraction: Dict[str, float] = {}
    bout_count: Dict[str, int] = {}
    bout_total_time_s: Dict[str, float] = {}
    bout_mean_duration_s: Dict[str, float] = {}
    probability_time_s = np.asarray([], dtype=float)
    state_probability_profile: Dict[str, np.ndarray] = {}
    for state in DEFAULT_STATE_ORDER:
        total_state_time = float(sum(exp.state_time_s.get(state, 0.0) for exp in exp_summaries))
        total_bout_count = int(sum(exp.bout_count.get(state, 0) for exp in exp_summaries))
        total_bout_time = float(sum(exp.bout_total_time_s.get(state, 0.0) for exp in exp_summaries))
        state_time_s[state] = total_state_time
        state_fraction[state] = float(total_state_time / total_time_s) if total_time_s > 0 else float("nan")
        bout_count[state] = total_bout_count
        bout_total_time_s[state] = total_bout_time
        bout_mean_duration_s[state] = (
            float(total_bout_time / total_bout_count) if total_bout_count > 0 else float("nan")
        )
    if str(category) == "sleep" and str(scope) == "day" and len(exp_summaries) > 1:
        pooled_state_epoch, pooled_state_epoch_t = concatenate_sleep_state_arrays(exp_summaries)
        if pooled_state_epoch.size and pooled_state_epoch_t.size:
            pooled_epoch_duration_s = epoch_duration_seconds(pooled_state_epoch_t)
            probability_time_s, state_probability_profile = build_probability_profile(
                pooled_state_epoch,
                pooled_state_epoch_t,
                pooled_epoch_duration_s,
            )
        else:
            probability_time_s, state_probability_profile = average_probability_summaries(exp_summaries)
    else:
        probability_time_s, state_probability_profile = average_probability_summaries(exp_summaries)
    return SessionSummary(
        scope=scope,
        animal_id=animal_id,
        date=date,
        category=category,
        exp_ids=sorted(exp_ids),
        sleep_state_paths=sorted(sleep_state_paths),
        epoch_count=epoch_count,
        epoch_duration_s=epoch_duration_s,
        total_time_s=total_time_s,
        unknown_epoch_count=unknown_epoch_count,
        unknown_epoch_fraction=unknown_epoch_fraction,
        state_time_s=state_time_s,
        state_fraction=state_fraction,
        bout_count=bout_count,
        bout_total_time_s=bout_total_time_s,
        bout_mean_duration_s=bout_mean_duration_s,
        probability_time_s=probability_time_s,
        state_probability_profile=state_probability_profile,
        animal_day_index=day_index,
    )


def summary_to_rows(summary: SessionSummary) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for state_index, state in enumerate(DEFAULT_STATE_ORDER):
        rows.append(
            {
                "scope": summary.scope,
                "animal_id": summary.animal_id,
                "date": summary.date,
                "animal_day_index": summary.animal_day_index if summary.animal_day_index is not None else "",
                "category": summary.category,
                "category_display": format_display_category(summary.category),
                "exp_ids": ";".join(summary.exp_ids),
                "n_expids": len(summary.exp_ids),
                "sleep_state_paths": ";".join(summary.sleep_state_paths),
                "state": state,
                "state_display": format_display_state(state),
                "state_order": state_index,
                "epoch_count": summary.epoch_count,
                "epoch_duration_s": summary.epoch_duration_s,
                "total_time_s": summary.total_time_s,
                "unknown_epoch_count": summary.unknown_epoch_count,
                "unknown_epoch_fraction": summary.unknown_epoch_fraction,
                "state_time_s": summary.state_time_s.get(state, float("nan")),
                "state_fraction": summary.state_fraction.get(state, float("nan")),
                "bout_count": summary.bout_count.get(state, 0),
                "bout_total_time_s": summary.bout_total_time_s.get(state, float("nan")),
                "bout_mean_duration_s": summary.bout_mean_duration_s.get(state, float("nan")),
            }
        )
    return rows


def build_requested_expids(config: Mapping[str, Any]) -> Dict[str, str]:
    movie_expids = parse_list_argument(config.get("movie_expids"))
    sleep_expids = parse_list_argument(config.get("sleep_expids"))
    overlap = sorted(set(movie_expids) & set(sleep_expids))
    if overlap:
        raise SystemExit(f"movie_expids and sleep_expids overlap; please separate them: {', '.join(overlap)}")
    category_map: Dict[str, str] = {}
    for exp_id in movie_expids:
        category_map[exp_id] = "movie"
    for exp_id in sleep_expids:
        category_map[exp_id] = "sleep"
    return dict(sorted(category_map.items(), key=lambda item: (derive_date(item[0]), item[0])))


def collect_exp_summaries(
    requested_expids: Mapping[str, str],
    repo_base: Path,
) -> Tuple[List[SessionSummary], List[Dict[str, Any]]]:
    exp_summaries: List[SessionSummary] = []
    skipped: List[Dict[str, Any]] = []
    for exp_id, category in requested_expids.items():
        animal_id = derive_animal_id(exp_id)
        date = derive_date(exp_id)
        exp_root = resolve_repo_root(repo_base, animal_id, exp_id)
        if not exp_root.exists():
            skipped.append(
                {
                    "exp_id": exp_id,
                    "animal_id": animal_id,
                    "date": date,
                    "category": category,
                    "repo_root": str(exp_root),
                    "sleep_state_path": str(exp_root / "sleep_score" / "sleep_state.pickle"),
                    "reason": "missing experiment root",
                }
            )
            continue
        bundle, sleep_state_path, error = load_sleep_state_bundle(exp_root)
        if bundle is None or sleep_state_path is None:
            skipped.append(
                {
                    "exp_id": exp_id,
                    "animal_id": animal_id,
                    "date": date,
                    "category": category,
                    "repo_root": str(exp_root),
                    "sleep_state_path": str(sleep_state_path) if sleep_state_path is not None else str(
                        exp_root / "sleep_score" / "sleep_state.pickle"
                    ),
                    "reason": error or "missing sleep_state.pickle",
                }
            )
            continue
        try:
            summary = summarize_state_bundle(exp_id, animal_id, date, category, bundle, sleep_state_path)
        except Exception as exc:
            skipped.append(
                {
                    "exp_id": exp_id,
                    "animal_id": animal_id,
                    "date": date,
                    "category": category,
                    "repo_root": str(exp_root),
                    "sleep_state_path": str(sleep_state_path),
                    "reason": f"failed to summarize sleep_state.pickle: {exc}",
                }
            )
            continue
        exp_summaries.append(summary)
    return exp_summaries, skipped


def aggregate_day_summaries(exp_summaries: Sequence[SessionSummary]) -> List[SessionSummary]:
    grouped: Dict[Tuple[str, str, str], List[SessionSummary]] = defaultdict(list)
    for summary in exp_summaries:
        grouped[(summary.animal_id, summary.date, summary.category)].append(summary)
    day_summaries: List[SessionSummary] = []
    for (animal_id, date, category), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        day_summaries.append(
            aggregate_summaries("day", animal_id, date, category, sorted(items, key=lambda s: s.exp_ids[0]))
        )
    return day_summaries


def assign_day_indices(day_summaries: Sequence[SessionSummary]) -> None:
    by_animal: Dict[str, List[str]] = defaultdict(list)
    for summary in day_summaries:
        by_animal[summary.animal_id].append(summary.date)
    date_index: Dict[str, Dict[str, int]] = {}
    for animal_id, dates in by_animal.items():
        unique_dates = sorted(set(dates))
        date_index[animal_id] = {date: idx + 1 for idx, date in enumerate(unique_dates)}
    for summary in day_summaries:
        summary.animal_day_index = date_index[summary.animal_id][summary.date]


def summary_rows_by_metric(rows: Sequence[Mapping[str, Any]], metric_name: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if row.get(metric_name) is not None]


def rows_to_metric_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def group_rows_by_animal(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("animal_id"))].append(dict(row))
    for key in grouped:
        grouped[key].sort(key=lambda row: (str(row.get("date")), str(row.get("category")), str(row.get("state"))))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def group_summaries_by_animal(summaries: Sequence[SessionSummary]) -> Dict[str, List[SessionSummary]]:
    grouped: Dict[str, List[SessionSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[str(summary.animal_id)].append(summary)
    for key in grouped:
        grouped[key].sort(
            key=lambda summary: (
                str(summary.date),
                str(summary.category),
                str(summary.exp_ids[0]) if summary.exp_ids else "",
            )
        )
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def load_sleep_state_arrays(sleep_state_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    bundle = normalize_sleep_bundle(load_pickle(sleep_state_path), sleep_state_path)
    state_epoch = np.asarray(bundle["state_epoch"], dtype=int).ravel()
    state_epoch_t = np.asarray(bundle["state_epoch_t"], dtype=float).ravel()
    return align_length(state_epoch, state_epoch_t)


def concatenate_sleep_state_arrays(exp_summaries: Sequence[SessionSummary]) -> Tuple[np.ndarray, np.ndarray]:
    ordered = sorted(
        [summary for summary in exp_summaries if summary.sleep_state_paths],
        key=lambda summary: (
            str(summary.date),
            str(summary.exp_ids[0]) if summary.exp_ids else "",
            str(summary.sleep_state_paths[0]),
        ),
    )
    state_segments: List[np.ndarray] = []
    time_segments: List[np.ndarray] = []
    offset_s = 0.0
    previous_duration_s = 10.0
    for summary in ordered:
        sleep_state_path = Path(summary.sleep_state_paths[0])
        state_epoch, state_epoch_t = load_sleep_state_arrays(sleep_state_path)
        if state_epoch.size == 0:
            continue
        finite_t = np.asarray(state_epoch_t, dtype=float)
        finite_t = finite_t[np.isfinite(finite_t)]
        start_t = float(finite_t[0]) if finite_t.size else 0.0
        relative_t = np.asarray(state_epoch_t, dtype=float) - start_t
        if state_segments:
            relative_t = relative_t + offset_s
        state_segments.append(np.asarray(state_epoch, dtype=int))
        time_segments.append(relative_t)
        duration_s = epoch_duration_seconds(state_epoch_t)
        if not np.isfinite(duration_s) or duration_s <= 0:
            duration_s = previous_duration_s
        previous_duration_s = duration_s
        offset_s = float(relative_t[-1]) + previous_duration_s
    if not state_segments:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(state_segments), np.concatenate(time_segments)


@dataclass
class MontageSelection:
    animal_id: str
    date: str
    exp_id: str
    state: str
    state_code: int
    sleep_state_path: Path
    exp_root: Path
    bundle: Mapping[str, Any]
    bout_start_s: float
    bout_end_s: float
    center_time_s: float
    window_start_s: float
    window_end_s: float
    bout_duration_s: float


def probability_profiles_by_category(day_summaries: Sequence[SessionSummary]) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    profiles: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
    for category in DEFAULT_CATEGORY_ORDER:
        category_summaries = [
            summary
            for summary in day_summaries
            if str(summary.category) == category and summary.probability_time_s.size and summary.state_probability_profile
        ]
        if not category_summaries:
            continue
        animal_summaries = aggregate_probability_summaries_by_animal(category_summaries)
        valid = [
            summary
            for summary in animal_summaries
            if str(summary.category) == category and summary.probability_time_s.size and summary.state_probability_profile
        ]
        if not valid:
            continue
        time_s, profile = average_probability_summaries(valid)
        if time_s.size and profile:
            profiles[category] = (np.asarray(time_s, dtype=float) / 60.0, profile)
    return profiles


def time_window_indices(time_s: np.ndarray, start_s: float, end_s: float, min_points: int = 2) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float).ravel()
    if time_s.size == 0:
        return np.asarray([], dtype=int)
    mask = np.isfinite(time_s) & (time_s >= start_s) & (time_s <= end_s)
    indices = np.flatnonzero(mask)
    if indices.size >= min_points:
        return indices
    center = 0.5 * (float(start_s) + float(end_s))
    order = np.argsort(np.abs(time_s - center))
    keep = min(max(int(min_points), 2), time_s.size)
    return np.sort(order[:keep])


def find_eye_video_path(exp_root: Path) -> Optional[Path]:
    search_order = [
        '*eye*left*.avi',
        '*left*eye*.avi',
        '*eye*right*.avi',
        '*right*eye*.avi',
    ]
    for pattern in search_order:
        matches = sorted(exp_root.glob(pattern))
        if matches:
            return matches[0]
        matches = sorted(exp_root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def load_eye_frame_times(exp_root: Path) -> Optional[np.ndarray]:
    candidates = [
        exp_root / 'recordings' / 'eye_frame_times.npy',
        exp_root / 'eye_frame_times.npy',
        exp_root / 'recordings' / 'eye' / 'eye_frame_times.npy',
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return np.asarray(np.load(candidate), dtype=float).ravel()
            except Exception:
                continue
    return None


def load_eye_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    if not video_path.exists():
        return None
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(frame_index))))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        frame_rgb = frame[:, :, ::-1]
        return frame_rgb
    finally:
        capture.release()


def pad_image_to_aspect(image: np.ndarray, target_aspect: float = 16.0 / 9.0, fill_value: int = 255) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2:
        return arr
    height, width = arr.shape[:2]
    if height <= 0 or width <= 0:
        return arr
    current_aspect = float(width) / float(height)
    if np.isclose(current_aspect, target_aspect, rtol=1e-3, atol=1e-3):
        return arr
    if current_aspect > target_aspect:
        target_height = int(round(float(width) / float(target_aspect)))
        pad_total = max(0, target_height - height)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_width = ((pad_top, pad_bottom), (0, 0)) + ((0, 0),) * max(0, arr.ndim - 2)
    else:
        target_width = int(round(float(height) * float(target_aspect)))
        pad_total = max(0, target_width - width)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_width = ((0, 0), (pad_left, pad_right)) + ((0, 0),) * max(0, arr.ndim - 2)
    return np.pad(arr, pad_width, mode='constant', constant_values=fill_value)


def _interval_gap_seconds(start_s: float, end_s: float, other_starts_s: np.ndarray, other_ends_s: np.ndarray) -> float:
    other_starts_s = np.asarray(other_starts_s, dtype=float).ravel()
    other_ends_s = np.asarray(other_ends_s, dtype=float).ravel()
    if other_starts_s.size == 0 or other_ends_s.size == 0:
        return float('inf')
    gaps: List[float] = []
    for other_start_s, other_end_s in zip(other_starts_s, other_ends_s):
        if not (np.isfinite(other_start_s) and np.isfinite(other_end_s)):
            continue
        if end_s < other_start_s:
            gaps.append(float(other_start_s - end_s))
        elif other_end_s < start_s:
            gaps.append(float(start_s - other_end_s))
        else:
            return 0.0
    if not gaps:
        return float('inf')
    return float(np.min(gaps))


def _active_wake_locomotion_rise_time(bundle: Mapping[str, Any], bout_start_s: float, bout_end_s: float) -> Optional[float]:
    wheel_t = np.asarray(bundle.get('wheel_10hz_t', []), dtype=float).ravel()
    wheel_values = np.asarray(bundle.get('wheel_10hz', []), dtype=float).ravel()
    wheel_t, wheel_values = align_length(wheel_t, wheel_values)
    if wheel_t.size < 5 or wheel_values.size < 5:
        return None
    mask = np.isfinite(wheel_t) & np.isfinite(wheel_values) & (wheel_t >= bout_start_s) & (wheel_t <= bout_end_s)
    if np.count_nonzero(mask) < 5:
        return None
    local_t = wheel_t[mask]
    local_values = wheel_values[mask]
    finite_mask = np.isfinite(local_values)
    if np.count_nonzero(finite_mask) < 5:
        return None
    if not np.all(finite_mask):
        local_values = np.interp(local_t, local_t[finite_mask], local_values[finite_mask])
    diffs = np.diff(local_t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        dt = 0.1
    else:
        dt = float(np.nanmedian(diffs))
    smooth_n = int(max(5, round(30.0 / max(dt, 1e-3))))
    smooth_n = min(smooth_n, int(local_values.size))
    if smooth_n % 2 == 0:
        smooth_n = max(5, smooth_n - 1)
    if smooth_n < 5:
        return None
    kernel = np.ones(smooth_n, dtype=float) / float(smooth_n)
    smoothed = np.convolve(local_values, kernel, mode='same')
    threshold = float(bundle.get('locomotion_threshold', float('nan')))
    if np.isfinite(threshold):
        above = smoothed >= threshold
        crossings = np.where(~above[:-1] & above[1:])[0]
        if crossings.size:
            rise_idx = int(crossings[0] + 1)
            return float(local_t[rise_idx])
    slope = np.gradient(smoothed, local_t)
    if slope.size and np.any(np.isfinite(slope)):
        best_idx = int(np.nanargmax(slope))
        if np.isfinite(slope[best_idx]) and slope[best_idx] > 0:
            return float(local_t[best_idx])
    return None


def select_representative_state_window(
    summary: SessionSummary,
    state_code: int,
    *,
    window_s: float = 600.0,
    active_wake_rem_buffer_s: float = 600.0,
) -> Optional[MontageSelection]:
    if str(summary.category) != 'sleep' or not summary.sleep_state_paths:
        return None
    sleep_state_path = Path(summary.sleep_state_paths[0])
    try:
        bundle = normalize_sleep_bundle(load_pickle(sleep_state_path), sleep_state_path)
    except Exception:
        return None

    state_epoch = np.asarray(bundle.get('state_epoch', []), dtype=int).ravel()
    state_epoch_t = np.asarray(bundle.get('state_epoch_t', []), dtype=float).ravel()
    state_epoch, state_epoch_t = align_length(state_epoch, state_epoch_t)
    if state_epoch.size == 0 or state_epoch_t.size == 0:
        return None

    epoch_duration_s = epoch_duration_seconds(state_epoch_t)
    starts, lengths, states = contiguous_runs(state_epoch)
    candidate_mask = states == int(state_code)
    if not np.any(candidate_mask):
        return None

    candidate_starts = starts[candidate_mask]
    candidate_lengths = lengths[candidate_mask]

    if int(state_code) == 0:
        rem_mask = states == 3
        rem_starts = starts[rem_mask]
        rem_ends = rem_starts + lengths[rem_mask]
        rem_bout_starts_s = np.asarray(state_epoch_t[rem_starts] - 0.5 * epoch_duration_s, dtype=float)
        rem_bout_ends_s = np.asarray(state_epoch_t[rem_ends - 1] + 0.5 * epoch_duration_s, dtype=float)
        qualifying_indices: List[int] = []
        for idx, run_start in enumerate(candidate_starts):
            run_length = int(candidate_lengths[idx])
            run_end = int(run_start + run_length)
            bout_start_s = float(state_epoch_t[run_start] - 0.5 * epoch_duration_s)
            bout_end_s = float(state_epoch_t[run_end - 1] + 0.5 * epoch_duration_s)
            gap_s = _interval_gap_seconds(bout_start_s, bout_end_s, rem_bout_starts_s, rem_bout_ends_s)
            if gap_s >= float(active_wake_rem_buffer_s):
                qualifying_indices.append(idx)
        if qualifying_indices:
            candidate_lengths = candidate_lengths[qualifying_indices]
            candidate_starts = candidate_starts[qualifying_indices]

    best_idx = int(np.argmax(candidate_lengths))
    run_start = int(candidate_starts[best_idx])
    run_length = int(candidate_lengths[best_idx])
    run_end = int(run_start + run_length)
    bout_start_s = float(state_epoch_t[run_start] - 0.5 * epoch_duration_s)
    bout_end_s = float(state_epoch_t[run_end - 1] + 0.5 * epoch_duration_s)
    if int(state_code) == 0:
        center_time_s = _active_wake_locomotion_rise_time(bundle, bout_start_s, bout_end_s)
        if center_time_s is None:
            center_time_s = float(0.5 * (bout_start_s + bout_end_s))
    else:
        center_time_s = float(0.5 * (bout_start_s + bout_end_s))
    data_start_s = float(state_epoch_t[0] - 0.5 * epoch_duration_s)
    data_end_s = float(state_epoch_t[-1] + 0.5 * epoch_duration_s)
    half_window = 0.5 * float(window_s)
    window_start_s = max(data_start_s, center_time_s - half_window)
    window_end_s = min(data_end_s, center_time_s + half_window)
    bout_duration_s = max(0.0, bout_end_s - bout_start_s)
    exp_id = str(summary.exp_ids[0]) if summary.exp_ids else sleep_state_path.parent.parent.name
    return MontageSelection(
        animal_id=str(summary.animal_id),
        date=str(summary.date),
        exp_id=exp_id,
        state=DEFAULT_STATE_ORDER[int(state_code)],
        state_code=int(state_code),
        sleep_state_path=sleep_state_path,
        exp_root=sleep_state_path.parent.parent,
        bundle=bundle,
        bout_start_s=bout_start_s,
        bout_end_s=bout_end_s,
        center_time_s=center_time_s,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        bout_duration_s=bout_duration_s,
    )



def extract_eye_frame_for_selection(selection: MontageSelection) -> Tuple[Optional[np.ndarray], str]:
    exp_root = selection.exp_root
    video_path = find_eye_video_path(exp_root)
    frame_times = load_eye_frame_times(exp_root)
    if video_path is None:
        return None, 'Eye frame unavailable (no eye AVI found)'
    if frame_times is None or frame_times.size == 0:
        return None, 'Eye frame unavailable (missing eye_frame_times.npy)'
    if cv2 is None:
        return None, f'Eye frame unavailable (no video decoder; {video_path.name})'
    frame_index = int(np.argmin(np.abs(frame_times - float(selection.center_time_s))))
    frame = load_eye_frame(video_path, frame_index)
    if frame is None:
        return None, f'Eye frame unavailable ({video_path.name})'
    return frame, f'{video_path.name} | frame {frame_index}'



def infer_eye_side_from_path(path: Path) -> Optional[str]:
    name = path.name.lower()
    if 'left' in name:
        return 'left'
    if 'right' in name:
        return 'right'
    return None



def load_resampled_eye_pupil_series(exp_root: Path, eye_side: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    eye_side = str(eye_side).lower()
    if eye_side not in {'left', 'right'}:
        return None, None, None
    candidates = [
        exp_root / 'recordings' / f'dlcEye{eye_side.capitalize()}_resampled.pickle',
        exp_root / 'recordings' / f'dlcEye{eye_side.capitalize()}.pickle',
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            bundle = load_pickle(candidate)
        except Exception:
            continue
        if not isinstance(bundle, Mapping):
            continue
        pupil_t = np.asarray(bundle.get('t', []), dtype=float).ravel()
        pupil_radius = np.asarray(bundle.get('radius', []), dtype=float).ravel()
        pupil_t, pupil_radius = align_length(pupil_t, pupil_radius)
        if pupil_t.size and pupil_radius.size:
            return pupil_t, 2.0 * pupil_radius, candidate.name
    return None, None, None



def extract_pupil_series_for_selection(selection: MontageSelection) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    exp_root = selection.exp_root
    video_path = find_eye_video_path(exp_root)
    if video_path is None:
        return None, None, 'Pupil unavailable (no eye AVI found)'
    eye_side = infer_eye_side_from_path(video_path)
    if eye_side is None:
        return None, None, f'Pupil unavailable ({video_path.name})'
    pupil_t, pupil_diameter, source_name = load_resampled_eye_pupil_series(exp_root, eye_side)
    if pupil_t is None or pupil_diameter is None or source_name is None:
        return None, None, f'Pupil unavailable ({eye_side} eye)'
    return pupil_t, pupil_diameter, f'{source_name} | {eye_side} eye'



def draw_example_marker(ax: Any, x_value: float, *, color: str = '#222222') -> None:
    line = ax.axvline(x_value, color=color, linestyle='--', linewidth=1.0, alpha=0.9, zorder=7)
    if mpatheffects is not None:
        line.set_path_effects([
            mpatheffects.Stroke(linewidth=2.4, foreground='white'),
            mpatheffects.Normal(),
        ])


def draw_column_example_marker(fig: Any, eye_ax: Any, hyp_ax: Any, x_value: float, *, color: str = '#222222') -> None:
    ref_xlim = hyp_ax.get_xlim()
    x_min = float(np.nanmin(ref_xlim)) if np.isfinite(np.nanmin(ref_xlim)) else 0.0
    x_max = float(np.nanmax(ref_xlim)) if np.isfinite(np.nanmax(ref_xlim)) else x_min + 1.0
    x_span = max(x_max - x_min, 1e-6)
    eye_pos = eye_ax.get_position()
    x_fig = float(eye_pos.x0 + ((float(x_value) - x_min) / x_span) * eye_pos.width)
    y_bottom = float(hyp_ax.get_position().y0)
    y_top = float(eye_pos.y0)
    line = Line2D([x_fig, x_fig], [y_bottom, y_top], transform=fig.transFigure, color=color, linestyle='--', linewidth=1.0, alpha=0.95, zorder=50, clip_on=False)
    if mpatheffects is not None:
        line.set_path_effects([
            mpatheffects.Stroke(linewidth=2.6, foreground='white'),
            mpatheffects.Normal(),
        ])
    fig.add_artist(line)


def compute_emg_spectrogram(segment: np.ndarray, segment_t: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if scipy_signal is None:
        return None
    segment = np.asarray(segment, dtype=float).ravel()
    segment_t = np.asarray(segment_t, dtype=float).ravel()
    if segment.size < 8 or segment_t.size < 2:
        return None
    diffs = np.diff(segment_t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        fs = 10.0
    else:
        fs = float(1.0 / np.nanmedian(diffs))
    nperseg = int(min(128, max(16, segment.size)))
    if nperseg < 16:
        return None
    noverlap = int(min(nperseg - 1, max(0, round(nperseg * 0.75))))
    freqs, times, power = scipy_signal.spectrogram(
        segment,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd',
    )
    return np.asarray(freqs, dtype=float), np.asarray(times, dtype=float), np.asarray(power, dtype=float)


def render_state_montage_panel(
    ax: Any,
    message: str,
    *,
    title: Optional[str] = None,
    show_title: bool = True,
) -> None:
    ax.axis('off')
    if title is not None and show_title:
        ax.set_title(title, fontsize=max(12, POSTER_TITLE_SIZE - 10), pad=2)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=max(10, POSTER_NOTE_SIZE - 2),
        color='#666666',
        wrap=True,
    )


def lighten_color(color: str, amount: float = 0.65) -> str:
    if mcolors is None:
        return color
    try:
        rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    except Exception:
        return color
    amount = float(np.clip(amount, 0.0, 1.0))
    lightened = rgb + (1.0 - rgb) * amount
    return mcolors.to_hex(lightened, keep_alpha=False)


def plot_state_montage_per_exp(
    summary: SessionSummary,
    output_dir: Path,
) -> List[Path]:
    if plt is None:
        raise RuntimeError('matplotlib is required to generate figures')
    if str(summary.category) != 'sleep':
        raise ValueError('State montage can only be generated for sleep expIDs')
    if not summary.sleep_state_paths:
        raise ValueError(f'No sleep state bundle available for expID {summary.exp_ids[0] if summary.exp_ids else "unknown"}')

    animal_id = str(summary.animal_id)
    exp_id = str(summary.exp_ids[0]) if summary.exp_ids else Path(summary.sleep_state_paths[0]).stem
    selections = {
        state: select_representative_state_window(summary, state_code)
        for state_code, state in enumerate(DEFAULT_STATE_ORDER)
    }

    fig = plt.figure(figsize=(19.2, 19.0))
    gs = fig.add_gridspec(
        6,
        len(DEFAULT_STATE_ORDER),
        wspace=0.18,
        hspace=0.14,
        height_ratios=[1.88, 0.74, 1.26, 0.88, 0.94, 1.0],
    )
    axes = np.empty((6, len(DEFAULT_STATE_ORDER)), dtype=object)
    pupil_limits: List[Tuple[float, float]] = []
    emg_limits: List[Tuple[float, float]] = []
    wheel_limits: List[Tuple[float, float]] = []

    for col_idx, state in enumerate(DEFAULT_STATE_ORDER):
        axes[0, col_idx] = fig.add_subplot(gs[0, col_idx])
        axes[1, col_idx] = fig.add_subplot(gs[1, col_idx])
        axes[2, col_idx] = fig.add_subplot(gs[2, col_idx], sharex=axes[1, col_idx])
        axes[3, col_idx] = fig.add_subplot(gs[3, col_idx], sharex=axes[1, col_idx])
        axes[4, col_idx] = fig.add_subplot(gs[4, col_idx], sharex=axes[1, col_idx])
        axes[5, col_idx] = fig.add_subplot(gs[5, col_idx], sharex=axes[1, col_idx])

    left_labels = [
        ('Eye frame', None),
        ('Pupil size', None),
        ('Spectrogram', None),
        ('EMG RMS', 'tab:blue'),
        ('Wheel', 'tab:orange'),
        ('Hypnogram', None),
    ]
    for row_idx, (row_label, row_color) in enumerate(left_labels):
        label_kwargs = dict(fontsize=max(12, POSTER_LABEL_SIZE - 5), labelpad=14)
        if row_color:
            label_kwargs['color'] = row_color
        axes[row_idx, 0].set_ylabel(row_label, **label_kwargs)
        if row_color:
            axes[row_idx, 0].spines['left'].set_color(row_color)

    column_marker_positions: List[Tuple[int, float]] = []
    for col_idx, state in enumerate(DEFAULT_STATE_ORDER):
        selection = selections.get(state)
        eye_ax, pupil_ax, spec_ax, emg_ax, wheel_ax, hyp_ax = axes[:, col_idx]
        eye_ax.set_title(format_display_state(state), fontsize=max(13, POSTER_TITLE_SIZE - 11), pad=10)
        if selection is None:
            render_state_montage_panel(eye_ax, 'No eye frame available')
            render_state_montage_panel(pupil_ax, 'No pupil trace available')
            render_state_montage_panel(spec_ax, 'No spectrogram available')
            render_state_montage_panel(emg_ax, 'No EMG trace available')
            render_state_montage_panel(wheel_ax, 'No locomotion trace available')
            render_state_montage_panel(hyp_ax, f'No representative {format_display_state(state)} state found')
            continue

        bundle = selection.bundle
        window_start_s = float(selection.window_start_s)
        window_end_s = float(selection.window_end_s)
        window_start_min = window_start_s / 60.0
        window_end_min = window_end_s / 60.0
        center_min = float(selection.center_time_s / 60.0)
        video_path = find_eye_video_path(selection.exp_root)
        pupil_side = infer_eye_side_from_path(video_path) if video_path is not None else None
        pupil_color = '#7C3AED' if pupil_side == 'left' else '#0F766E' if pupil_side == 'right' else '#6B7280'

        frame, frame_note = extract_eye_frame_for_selection(selection)
        if frame is not None:
            eye_ax.imshow(
                frame,
                aspect='equal',
                interpolation='nearest',
                origin='upper',
            )
            eye_ax.set_box_aspect(float(frame.shape[0]) / max(float(frame.shape[1]), 1.0))
            eye_ax.set_anchor('C')
            eye_ax.set_xticks([])
            eye_ax.set_yticks([])
            eye_ax.tick_params(axis='both', left=False, bottom=False, labelleft=False, labelbottom=False)
            eye_ax.set_facecolor('white')
            for side in ('left', 'right', 'top', 'bottom'):
                eye_ax.spines[side].set_visible(True)
                eye_ax.spines[side].set_color('#222222')
                eye_ax.spines[side].set_linewidth(1.4 if side == 'bottom' else 1.1)
        else:
            render_state_montage_panel(eye_ax, frame_note)

        pupil_t, pupil_diameter, pupil_note = extract_pupil_series_for_selection(selection)
        if pupil_t is not None and pupil_diameter is not None:
            pupil_idx = time_window_indices(pupil_t, window_start_s, window_end_s)
            if pupil_idx.size:
                pupil_x = pupil_t[pupil_idx] / 60.0
                pupil_y = np.asarray(pupil_diameter[pupil_idx], dtype=float)
                finite_pupil = pupil_y[np.isfinite(pupil_y)]
                if finite_pupil.size:
                    pupil_cap = float(np.percentile(finite_pupil, 80.0))
                    pupil_y = np.minimum(pupil_y, pupil_cap)
                    pupil_low = float(np.nanmin(finite_pupil))
                    pupil_high = float(np.nanmax(np.minimum(finite_pupil, pupil_cap)))
                    pupil_span = max(pupil_high - pupil_low, 1e-6)
                    pupil_ylim = (pupil_low - 0.12 * pupil_span, pupil_high + 0.12 * pupil_span)
                    pupil_ax.set_ylim(*pupil_ylim)
                    pupil_limits.append(pupil_ylim)
                pupil_ax.plot(pupil_x, pupil_y, color=pupil_color, linewidth=1.0)
        else:
            render_state_montage_panel(pupil_ax, pupil_note)
        if col_idx == 0:
            pupil_ax.tick_params(axis='y', labelcolor=pupil_color)
            pupil_ax.spines['left'].set_color(pupil_color)
        pupil_ax.spines['top'].set_visible(False)
        pupil_ax.spines['right'].set_visible(False)
        pupil_ax.tick_params(axis='x', labelbottom=False)
        pupil_ax.grid(axis='y', alpha=0.12)

        spectrogram = np.asarray(bundle.get('eeg_spectrogram', []), dtype=float)
        frequencies = np.asarray(bundle.get('eeg_spectrogram_freqs', []), dtype=float)
        spectrogram_t = np.asarray(bundle.get('eeg_spectrogram_t', []), dtype=float)
        spectrogram_mask = time_window_indices(spectrogram_t, window_start_s, window_end_s, min_points=2)
        if spectrogram.ndim == 2 and frequencies.size and spectrogram_t.size and spectrogram_mask.size:
            freq_mask = np.isfinite(frequencies) & (frequencies >= 0.0) & (frequencies <= 20.0)
            if not np.any(freq_mask):
                render_state_montage_panel(spec_ax, 'No spectrogram available')
            else:
                spec_band = np.asarray(spectrogram[freq_mask][:, spectrogram_mask], dtype=float)
                finite_vals = spec_band[np.isfinite(spec_band)]
                if finite_vals.size:
                    vmin = float(np.percentile(finite_vals, 5.0))
                    vmax = float(np.percentile(finite_vals, 95.0))
                else:
                    vmin, vmax = -6.0, 0.0
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin = float(np.nanmin(spec_band)) if np.isfinite(np.nanmin(spec_band)) else -6.0
                    vmax = vmin + 1.0
                extent = [
                    float(spectrogram_t[spectrogram_mask][0] / 60.0),
                    float(spectrogram_t[spectrogram_mask][-1] / 60.0),
                    float(frequencies[freq_mask][0]),
                    float(frequencies[freq_mask][-1]),
                ]
                spec_ax.imshow(
                    spec_band,
                    aspect='auto',
                    origin='lower',
                    extent=extent,
                    interpolation='nearest',
                    vmin=vmin,
                    vmax=vmax,
                )
                delta_low, delta_high = bundle.get('delta_band', (1.0, 4.0))
                theta_low, theta_high = bundle.get('theta_band', (5.0, 10.0))
                for y, color in [
                    (delta_low, 'white'),
                    (delta_high, 'white'),
                    (theta_low, 'cyan'),
                    (theta_high, 'cyan'),
                ]:
                    if np.isfinite(y):
                        spec_ax.axhline(float(y), color=color, linestyle=':', linewidth=1.0, alpha=0.9)
                spec_ax.set_ylim(0.0, 20.0)
                spec_ax.set_xlim(extent[0], extent[1])
                set_sparse_numeric_ticks(spec_ax, axis='x', nbins=4)
                set_sparse_numeric_ticks(spec_ax, axis='y', nbins=5)
                spec_ax.tick_params(axis='x', labelbottom=False)
                spec_ax.tick_params(axis='x', labelsize=max(5, POSTER_FONT_SIZE - 12))
                spec_ax.grid(axis='x', alpha=0.12)
        else:
            render_state_montage_panel(spec_ax, 'No spectrogram available')

        emg_t = np.asarray(bundle.get('emg_rms_10hz_t', bundle.get('emg_10hz_t', [])), dtype=float).ravel()
        emg_values = np.asarray(bundle.get('emg_rms_10hz', bundle.get('emg_10hz', [])), dtype=float).ravel()
        emg_t, emg_values = align_length(emg_t, emg_values)
        emg_idx = time_window_indices(emg_t, window_start_s, window_end_s)
        emg_threshold = float(bundle.get('emg_rms_threshold', bundle.get('emg_threshold', float('nan'))))
        if emg_values.size and emg_t.size and emg_idx.size:
            emg_x = emg_t[emg_idx] / 60.0
            emg_y = emg_values[emg_idx]
            emg_ax.plot(emg_x, emg_y, color='#334155', linewidth=1.0)
            finite_emg = np.asarray(emg_y, dtype=float)
            finite_emg = finite_emg[np.isfinite(finite_emg)]
            if finite_emg.size:
                emg_low = float(np.nanmin(finite_emg))
                emg_high = float(np.nanmax(finite_emg))
                if np.isfinite(emg_threshold):
                    emg_low = min(emg_low, emg_threshold)
                    emg_high = max(emg_high, emg_threshold)
                emg_span = max(emg_high - emg_low, 1e-6)
                emg_pad = 0.12 * emg_span
                emg_ylim = (emg_low - emg_pad, emg_high + emg_pad)
                emg_ax.set_ylim(*emg_ylim)
                emg_limits.append(emg_ylim)
            if np.isfinite(emg_threshold):
                emg_ax.axhline(emg_threshold, color='#334155', linestyle='--', linewidth=0.9, alpha=0.85)
        else:
            render_state_montage_panel(emg_ax, 'No EMG trace available')
        if col_idx == 0:
            emg_ax.tick_params(axis='y', labelcolor='#334155')
            emg_ax.spines['left'].set_color('#334155')
        emg_ax.spines['top'].set_visible(False)
        emg_ax.spines['right'].set_visible(False)
        emg_ax.tick_params(axis='x', labelbottom=False)
        emg_ax.grid(axis='y', alpha=0.12)

        wheel_t = np.asarray(bundle.get('wheel_10hz_t', []), dtype=float).ravel()
        wheel_values = np.asarray(bundle.get('wheel_10hz', []), dtype=float).ravel()
        wheel_t, wheel_values = align_length(wheel_t, wheel_values)
        wheel_idx = time_window_indices(wheel_t, window_start_s, window_end_s)
        wheel_threshold = float(bundle.get('locomotion_threshold', float('nan')))
        if wheel_values.size and wheel_t.size and wheel_idx.size:
            wheel_vals = wheel_values[wheel_idx]
            wheel_x = wheel_t[wheel_idx] / 60.0
            wheel_ax.plot(wheel_x, wheel_vals, color='#8C6D31', linewidth=1.0)
            finite_wheel = np.asarray(wheel_vals, dtype=float)
            finite_wheel = finite_wheel[np.isfinite(finite_wheel)]
            if finite_wheel.size:
                wheel_low = float(np.nanmin(finite_wheel))
                wheel_high = float(np.nanmax(finite_wheel))
            else:
                wheel_low, wheel_high = -0.5, 0.5
            wheel_low -= 0.1
            wheel_high += 0.1
            if np.isfinite(wheel_threshold):
                wheel_low = min(wheel_low, -abs(wheel_threshold) - 0.1)
                wheel_high = max(wheel_high, abs(wheel_threshold) + 0.1)
            wheel_ylim = (wheel_low, wheel_high)
            wheel_ax.set_ylim(*wheel_ylim)
            wheel_limits.append(wheel_ylim)
            if np.isfinite(wheel_threshold):
                wheel_ax.axhline(wheel_threshold, color='#8C6D31', linestyle='--', linewidth=0.9, alpha=0.85)
                wheel_ax.axhline(-wheel_threshold, color='#8C6D31', linestyle='--', linewidth=0.9, alpha=0.85)
        else:
            render_state_montage_panel(wheel_ax, 'No locomotion trace available')
        if col_idx == 0:
            wheel_ax.tick_params(axis='y', labelcolor='#8C6D31')
            wheel_ax.spines['left'].set_color('#8C6D31')
        wheel_ax.spines['top'].set_visible(False)
        wheel_ax.spines['right'].set_visible(False)
        wheel_ax.tick_params(axis='x', labelbottom=False)
        wheel_ax.axhline(0, color='0.7', linestyle='--', linewidth=0.8)
        wheel_ax.grid(axis='y', alpha=0.12)

        state_time = np.asarray(bundle.get('state_10hz_t', []), dtype=float).ravel()
        state_values = np.asarray(bundle.get('state_10hz', []), dtype=int).ravel()
        state_time, state_values = align_length(state_time, state_values)
        idx = time_window_indices(state_time, window_start_s, window_end_s)
        if idx.size:
            t_min = state_time[idx] / 60.0
            hyp_values = state_values[idx]
            hyp_ax.step(t_min, hyp_values, where='post', color='0.75', linewidth=0.8, alpha=0.8, zorder=1)
            hyp_point_colors = [
                DEFAULT_STACKED_STATE_COLORS.get(DEFAULT_STATE_ORDER[int(code)], '#444444')
                if int(code) in range(len(DEFAULT_STATE_ORDER))
                else '#444444'
                for code in hyp_values
            ]
            hyp_ax.scatter(
                t_min,
                hyp_values,
                c=hyp_point_colors,
                s=10,
                linewidths=0,
                alpha=0.95,
                zorder=2,
            )
        else:
            render_state_montage_panel(hyp_ax, 'No state data in window')
        hyp_ax.set_ylim(-0.5, 3.5)
        hyp_ax.set_yticks(list(range(len(DEFAULT_STATE_ORDER))))
        hyp_ax.set_yticklabels([format_display_state(s) for s in DEFAULT_STATE_ORDER], fontsize=max(8, POSTER_FONT_SIZE - 7))
        hyp_ax.set_xlabel('')
        hyp_ax.grid(axis='y', alpha=0.18)

        pupil_ax.tick_params(axis='both', labelsize=max(8, POSTER_FONT_SIZE - 7))
        for ax in (pupil_ax, spec_ax, emg_ax, wheel_ax, hyp_ax):
            ax.set_xlim(window_start_min, window_end_min)
        if col_idx > 0:
            for ax in [pupil_ax, spec_ax, emg_ax, wheel_ax, hyp_ax]:
                ax.set_ylabel('')
                ax.yaxis.label.set_visible(False)
                ax.tick_params(axis='y', left=False, labelleft=False)
                ax.spines['left'].set_visible(False)
        else:
            eye_ax.tick_params(axis='y', left=False, labelleft=False)
            eye_ax.spines['left'].set_visible(True)
        column_marker_positions.append((col_idx, center_min))

    def _apply_shared_limits(axis_row: int, limits: List[Tuple[float, float]], *, pad: float = 0.0) -> None:
        finite_limits = [
            (float(lo), float(hi))
            for lo, hi in limits
            if np.isfinite(lo) and np.isfinite(hi)
        ]
        if not finite_limits:
            return
        lo = min(item[0] for item in finite_limits)
        hi = max(item[1] for item in finite_limits)
        if lo == hi:
            hi = lo + 1.0
        lo -= pad
        hi += pad
        for col_idx in range(len(DEFAULT_STATE_ORDER)):
            axes[axis_row, col_idx].set_ylim(lo, hi)

    _apply_shared_limits(1, pupil_limits)
    _apply_shared_limits(3, emg_limits)
    _apply_shared_limits(4, wheel_limits)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.975])
    for col_idx, center_min in column_marker_positions:
        draw_column_example_marker(fig, axes[0, col_idx], axes[-1, col_idx], center_min)
    figure_dir = ensure_dir(
        output_dir
        / DEFAULT_FIGURE_DIRNAME
        / DEFAULT_STATE_MONTAGE_FIGURE_DIRNAME
        / DEFAULT_STATE_MONTAGE_EXP_FIGURE_DIRNAME
        / safe_filename_component(animal_id)
        / safe_filename_component(exp_id)
    )
    output_stem = f'{safe_filename_component(animal_id)}_{safe_filename_component(exp_id)}_state_montage'
    output_paths = save_figure(fig, figure_dir / output_stem)
    return output_paths


def build_day_timeline_profile(day_exp_summaries: Sequence[SessionSummary]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[float]]:
    ordered = sorted(
        [summary for summary in day_exp_summaries if summary.sleep_state_paths],
        key=lambda summary: (
            str(summary.date),
            str(summary.exp_ids[0]) if summary.exp_ids else '',
            str(summary.sleep_state_paths[0]),
        ),
    )
    if not ordered:
        return np.asarray([], dtype=float), {state: np.asarray([], dtype=float) for state in DEFAULT_STATE_ORDER}, None
    state_segments: List[np.ndarray] = []
    time_segments: List[np.ndarray] = []
    offset_s = 0.0
    previous_duration_s = 10.0
    sleep_start_s: Optional[float] = None
    for summary in ordered:
        sleep_state_path = Path(summary.sleep_state_paths[0])
        state_epoch, state_epoch_t = load_sleep_state_arrays(sleep_state_path)
        if state_epoch.size == 0:
            continue
        finite_t = np.asarray(state_epoch_t, dtype=float)
        finite_t = finite_t[np.isfinite(finite_t)]
        start_t = float(finite_t[0]) if finite_t.size else 0.0
        relative_t = np.asarray(state_epoch_t, dtype=float) - start_t
        if sleep_start_s is None and str(summary.category) == "sleep":
            sleep_start_s = float(offset_s if state_segments else 0.0)
        if state_segments:
            relative_t = relative_t + offset_s
        state_segments.append(np.asarray(state_epoch, dtype=int))
        time_segments.append(relative_t)
        duration_s = epoch_duration_seconds(state_epoch_t)
        if not np.isfinite(duration_s) or duration_s <= 0:
            duration_s = previous_duration_s
        previous_duration_s = duration_s
        offset_s = float(relative_t[-1]) + previous_duration_s
    if not state_segments:
        return np.asarray([], dtype=float), {state: np.asarray([], dtype=float) for state in DEFAULT_STATE_ORDER}, None
    pooled_state_epoch = np.concatenate(state_segments)
    pooled_state_epoch_t = np.concatenate(time_segments)
    pooled_epoch_duration_s = epoch_duration_seconds(pooled_state_epoch_t)
    probability_time_s, state_probability_profile = build_probability_profile(
        pooled_state_epoch,
        pooled_state_epoch_t,
        pooled_epoch_duration_s,
    )
    return probability_time_s / 60.0, state_probability_profile, (sleep_start_s / 60.0 if sleep_start_s is not None else None)


def select_poster_ready_day_summaries(exp_summaries: Sequence[SessionSummary], example_exp_id: str) -> List[SessionSummary]:
    target: Optional[SessionSummary] = None
    for summary in exp_summaries:
        if str(example_exp_id) in {str(exp_id) for exp_id in summary.exp_ids}:
            target = summary
            break
    if target is None:
        return []
    return [summary for summary in exp_summaries if str(summary.animal_id) == str(target.animal_id) and str(summary.date) == str(target.date)]


def resolve_poster_ready_montage_svg(state_montage_artifacts: Sequence[str], output_dir: Path, example_exp_id: str) -> Optional[Path]:
    candidate_paths: List[Path] = []
    for artifact in state_montage_artifacts:
        artifact_path = Path(str(artifact))
        if not artifact_path.is_absolute():
            artifact_path = output_dir / artifact_path
        if example_exp_id not in artifact_path.name and example_exp_id not in str(artifact_path):
            continue
        candidate_paths.append(artifact_path)
    for path_candidate in candidate_paths:
        if path_candidate.suffix.lower() == '.svg' and path_candidate.exists():
            return path_candidate
    for path_candidate in candidate_paths:
        if path_candidate.suffix.lower() == '.png':
            svg_path = path_candidate.with_suffix('.svg')
            if svg_path.exists():
                return svg_path
    return None


def render_svg_to_image(svg_path: Path) -> Optional[np.ndarray]:
    try:
        import cairosvg
    except Exception:  # pragma: no cover - optional SVG rasterization support
        return None

    try:
        png_bytes = cairosvg.svg2png(url=str(svg_path), dpi=POSTER_DPI)
        return plt.imread(io.BytesIO(png_bytes), format='png')
    except Exception:
        return None

def inline_svg_panel_into_matplotlib_svg(svg_path: Path, panel_svg_path: Path) -> bool:
    """Replace Matplotlib's rasterized image node with the original SVG panel."""
    svg_path = Path(svg_path)
    panel_svg_path = Path(panel_svg_path)

    if not svg_path.exists() or not panel_svg_path.exists():
        return False

    try:
        svg_tree = ET.parse(str(svg_path))
        svg_root = svg_tree.getroot()
        panel_root = ET.parse(str(panel_svg_path)).getroot()
    except Exception:
        return False

    image_parent = None
    image_index = None
    image_el = None

    for parent in svg_root.iter():
        children = list(parent)
        for index, child in enumerate(children):
            if str(child.tag).endswith("image"):
                image_parent = parent
                image_index = index
                image_el = child
                break
        if image_el is not None:
            break

    if image_parent is None or image_index is None or image_el is None:
        return False

    replacement = deepcopy(panel_root)

    # Match the position/size of the original Matplotlib image slot.
    image_x = image_el.attrib.get("x")
    image_y = image_el.attrib.get("y")
    image_width = image_el.attrib.get("width")
    image_height = image_el.attrib.get("height")

    if image_x is not None:
        replacement.attrib["x"] = image_x
    if image_y is not None:
        replacement.attrib["y"] = image_y
    if image_width is not None:
        replacement.attrib["width"] = image_width
    if image_height is not None:
        replacement.attrib["height"] = image_height

    image_parent.remove(image_el)
    image_parent.insert(image_index, replacement)
    svg_tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)
    return True

def plot_sleep_state_poster_ready_composite(
    state_composition_rows: Sequence[Mapping[str, Any]],
    rem_day_presence_rows: Sequence[Mapping[str, Any]],
    exp_summaries: Sequence[SessionSummary],
    day_summaries: Sequence[SessionSummary],
    state_montage_artifacts: Sequence[str],
    output_dir: Path,
) -> List[Path]:
    if plt is None:
        raise RuntimeError('matplotlib is required to generate figures')

    figure_dir = ensure_dir(DEFAULT_POSTER_READY_DIR)
    fig = plt.figure(figsize=(15.8, 24.0))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[2.0, 2.15, 1.0, 1.0],
        height_ratios=[9.6, 2.4],
        wspace=0.36,
        hspace=0.15,
    )
    ax_montage = fig.add_subplot(gs[0, :])
    ax_stack = fig.add_subplot(gs[1, 0])
    frac_gs = gs[1, 1].subgridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.24, wspace=0.16)
    ax_frac00 = fig.add_subplot(frac_gs[0, 0])
    ax_frac01 = fig.add_subplot(frac_gs[0, 1], sharey=ax_frac00)
    ax_frac10 = fig.add_subplot(frac_gs[1, 0], sharex=ax_frac00)
    ax_frac11 = fig.add_subplot(frac_gs[1, 1], sharex=ax_frac01, sharey=ax_frac10)
    ax_frac = np.asarray([[ax_frac00, ax_frac01], [ax_frac10, ax_frac11]], dtype=object)
    pies_gs = gs[1, 2:].subgridspec(2, 1, hspace=0.28)
    ax_comp = fig.add_subplot(pies_gs[0, 0])
    ax_rem = fig.add_subplot(pies_gs[1, 0])
    montage_svg = resolve_poster_ready_montage_svg(
        state_montage_artifacts,
        output_dir,
        DEFAULT_POSTER_READY_STATE_MONTAGE_EXAMPLE_EXP_ID,
    )
    if montage_svg is None:
        montage_svg = resolve_poster_ready_montage_svg(
            state_montage_artifacts,
            output_dir,
            DEFAULT_REVIEW_STATE_MONTAGE_EXAMPLE_EXP_ID,
        )
    montage_img = render_svg_to_image(montage_svg) if montage_svg is not None and montage_svg.exists() else None
    if montage_img is not None:
        ax_montage.imshow(montage_img, aspect='equal', interpolation='nearest')
        ax_montage.set_anchor('C')
        ax_montage.axis('off')
    else:
        ax_montage.text(
            0.5,
            0.5,
            'State montage example unavailable',
            transform=ax_montage.transAxes,
            ha='center',
            va='center',
            fontsize=POSTER_NOTE_SIZE,
            color='#666666',
        )
        ax_montage.axis('off')

    x_values, state_order_stack, series_map, _ = build_stacked_panel_series(day_summaries, 'day_index')
    if x_values.size == 0:
        ax_stack.text(0.5, 0.5, 'No data', transform=ax_stack.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
    else:
        colors = [DEFAULT_STACKED_STATE_COLORS[state] for state in state_order_stack]
        ax_stack.stackplot(x_values, *[series_map[state] for state in state_order_stack], colors=colors, alpha=0.95, linewidth=0.35, edgecolor='white')
        ax_stack.set_ylim(0.0, 1.05)
        ax_stack.grid(axis='y', alpha=0.22)
        set_sparse_numeric_ticks(ax_stack, axis='y', nbins=5)
        if x_values.size == 1:
            ax_stack.set_xlim(float(x_values[0]) - 0.5, float(x_values[0]) + 0.5)
        else:
            ax_stack.set_xlim(0.5, float(np.nanmax(x_values)) + 0.5)
        set_sparse_numeric_ticks(ax_stack, axis='x', nbins=min(6, int(np.nanmax(x_values)) + 1), integer=True)
        missing_state_messages = stacked_panel_missing_state_messages(series_map, state_order_stack)
        if missing_state_messages:
            ax_stack.text(0.985, 0.985, '\n'.join(missing_state_messages), transform=ax_stack.transAxes, ha='right', va='top', fontsize=max(12, POSTER_NOTE_SIZE - 1), color='#555555', bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#d0d0d0', linewidth=0.8, alpha=0.88), zorder=5)
    ax_stack.set_ylabel('Fraction of time', fontsize=max(12, POSTER_LABEL_SIZE - 5), labelpad=12)
    ax_stack.set_title('Between-days\nstacked comparison', fontsize=max(13, POSTER_TITLE_SIZE - 11), pad=8, loc='left')
    ax_stack.set_xlabel('Within-animal day\nindex (movie + sleep)', fontsize=max(12, POSTER_LABEL_SIZE - 5), labelpad=8)

    day_timeline = select_poster_ready_day_summaries(
        exp_summaries,
        DEFAULT_POSTER_READY_STATE_MONTAGE_EXAMPLE_EXP_ID,
    )
    time_min, profile, sleep_start_min = build_day_timeline_profile(day_timeline)
    state_order = list(DEFAULT_STATE_ORDER)
    x_max = max(float(np.nanmax(time_min)) + 0.5 * (DEFAULT_PROBABILITY_BIN_S / 60.0), DEFAULT_PROBABILITY_BIN_S / 60.0) if time_min.size else 1.0
    fraction_axes = ax_frac.ravel().tolist()
    for idx, state in enumerate(state_order):
        ax = fraction_axes[idx]
        values = np.asarray(profile.get(state, []), dtype=float)
        if time_min.size and values.size and np.isfinite(values).any():
            band = 0.06
            band_color = lighten_color(DEFAULT_STACKED_STATE_COLORS[state], amount=0.68)
            ax.fill_between(
                time_min,
                np.clip(values - band, 0.0, 1.05),
                np.clip(values + band, 0.0, 1.05),
                color=band_color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                time_min,
                values,
                color=DEFAULT_STACKED_STATE_COLORS[state],
                linewidth=2.0,
                alpha=0.95,
                zorder=2,
            )
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=max(8, POSTER_FONT_SIZE - 7), color='#666666')
        if sleep_start_min is not None and np.isfinite(sleep_start_min):
            if sleep_start_min > 0.0:
                ax.axvspan(0.0, sleep_start_min, color='0.94', alpha=0.55, zorder=0)
            ax.axvline(sleep_start_min, color='#222222', linestyle='--', linewidth=1.15, alpha=0.98, zorder=7)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis='y', alpha=0.22)
        set_sparse_numeric_ticks(ax, axis='y', nbins=5)
        if idx < 2:
            ax.set_xticks([])
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
        else:
            set_sparse_numeric_ticks(ax, axis='x', nbins=3)
            ax.tick_params(axis='x', labelsize=max(8, POSTER_FONT_SIZE - 7), pad=0)
        ax.tick_params(axis='y', labelsize=max(8, POSTER_FONT_SIZE - 7))
        if idx in (0, 2):
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.tick_params(axis='y', labelleft=False)
            ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
    for ax in fraction_axes:
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.05)
    fraction_axes[0].set_ylabel('Fraction', fontsize=max(12, POSTER_LABEL_SIZE - 5), labelpad=4)
    frac_block_x0 = min(ax.get_position().x0 for ax in fraction_axes)
    frac_block_x1 = max(ax.get_position().x1 for ax in fraction_axes)
    frac_block_y0 = min(ax.get_position().y0 for ax in fraction_axes)
    frac_block_y1 = max(ax.get_position().y1 for ax in fraction_axes)
    fig.text(
        0.5 * (float(frac_block_x0) + float(frac_block_x1)),
        float(frac_block_y0) - 0.070,
        'Time\n(min)',
        ha='center',
        va='top',
        fontsize=max(11, POSTER_LABEL_SIZE - 7),
        linespacing=0.8,
    )
    fig.text(
        float(frac_block_x1),
        float(frac_block_y1) + 0.006,
        'sleep session starts',
        ha='right',
        va='bottom',
        fontsize=max(8, POSTER_FONT_SIZE - 7),
        color='#666666',
    )
    fig.text(
        0.5 * (float(frac_block_x0) + float(frac_block_x1)),
        float(frac_block_y1) + 0.011,
        'Sleep-state fractions',
        ha='center',
        va='bottom',
        fontsize=max(13, POSTER_TITLE_SIZE - 11),
    )

    composition_rows = [
        dict(row)
        for row in state_composition_rows
        if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))
    ]
    if composition_rows:
        values = [float(as_float(row.get('fraction'))) for row in composition_rows]
        labels = [str(row.get('state_display', row.get('state', 'unknown'))) for row in composition_rows]
        colors = [DEFAULT_STACKED_STATE_COLORS.get(str(row.get('state')), '#777777') for row in composition_rows]
        ax_comp.set_anchor('C')
        draw_compact_pie_panel(
            ax_comp,
            values,
            labels,
            colors,
            title='Sleep-state %',
            force_pct_labels=['REM'],
            legend_ncol=4,
            radius=0.84,
            legend_inside=False,
            show_legend=False,
        )
        ax_comp.set_title('Sleep-state %', fontsize=max(13, POSTER_TITLE_SIZE - 11), pad=4)
    else:
        ax_comp.text(0.5, 0.5, 'No data', transform=ax_comp.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        ax_comp.set_title('Sleep-state %', fontsize=max(14, POSTER_TITLE_SIZE - 8), pad=4)

    rem_rows = [
        dict(row)
        for row in rem_day_presence_rows
        if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))
    ]
    if rem_rows:
        values = [float(as_float(row.get('fraction'))) for row in rem_rows]
        labels = [str(row.get('state_display', row.get('state', 'unknown'))) for row in rem_rows]
        colors = ['#6A3D9A', '#A6761D'][: len(values)]
        ax_rem.set_anchor('C')
        draw_compact_pie_panel(
            ax_rem,
            values,
            labels,
            colors,
            title='Experimental days with REM',
            force_pct_labels=['Experimental days with REM', 'Experimental days without REM'],
            legend_ncol=1,
            radius=0.82,
            legend_inside=True,
        )
        ax_rem.set_title('Experimental days with REM', fontsize=max(13, POSTER_TITLE_SIZE - 11), pad=4)
        legend = ax_rem.get_legend()
        if legend is not None:
            legend.remove()
    else:
        ax_rem.text(0.5, 0.5, 'No data', transform=ax_rem.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        ax_rem.set_title('Experimental days with REM', fontsize=max(14, POSTER_TITLE_SIZE - 8), pad=4)

    state_handles = build_stacked_plot_handles(False)
    if state_handles:
        fig.legend(
            handles=state_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.004),
            ncol=4,
            frameon=False,
            fontsize=max(8, POSTER_FONT_SIZE - 7),
            handlelength=1.4,
            columnspacing=1.2,
        )

    ax_stack.spines['top'].set_visible(False)
    ax_stack.spines['right'].set_visible(False)
    ax_stack.tick_params(axis='x', labelsize=max(8, POSTER_FONT_SIZE - 7))
    ax_stack.tick_params(axis='y', labelsize=max(8, POSTER_FONT_SIZE - 7))
    ax_stack.set_xlim(left=0.0)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.988, bottom=0.045, wspace=0.28, hspace=0.18)

    output_svg = figure_dir / f'{DEFAULT_POSTER_READY_FIGURE_STEM}.svg'
    fig.savefig(output_svg, format='svg', dpi=72)

    if montage_svg is not None and montage_svg.exists():
        if not inline_svg_panel_into_matplotlib_svg(output_svg, montage_svg):
            warnings.warn(
                "Could not inline the state montage SVG into the poster composite; the montage may remain rasterized.",
                RuntimeWarning,
                stacklevel=2,
            )

    plt.close(fig)
    return [output_svg]


def format_date_axis(ax: Any) -> None:
    if mdates is None:
        return
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _metric_series_by_state(rows: Sequence[Mapping[str, Any]], metric_name: str) -> Dict[str, List[Tuple[float, float]]]:
    series: Dict[str, List[Tuple[float, float]]] = {state: [] for state in DEFAULT_STATE_ORDER}
    for row in rows:
        state = str(row.get('state'))
        if state not in series:
            continue
        metric_value = as_float(row.get(metric_name))
        if metric_value is None or not np.isfinite(metric_value):
            continue
        date_value = str(row.get('date', ''))
        try:
            x_value = mdates.date2num(dt.date.fromisoformat(date_value)) if mdates is not None else float(row.get('animal_day_index', 0) or 0)
        except Exception:
            x_value = float(row.get('animal_day_index', 0) or 0)
        series[state].append((float(x_value), float(metric_value)))
    for state in series:
        series[state].sort(key=lambda item: item[0])
    return series


def _metric_axis_label(metric_name: str, metric_label: str) -> str:
    if metric_name == 'state_fraction':
        return metric_label
    if metric_name == 'bout_count':
        return metric_label
    if metric_name == 'bout_mean_duration_s':
        return metric_label
    return metric_label


def _plot_metric_panels(
    category_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    metric_name: str,
    metric_label: str,
    title: str,
    output_dir: Path,
    figure_subdir: Path,
    output_stem: str,
) -> List[Path]:
    if plt is None or mdates is None:
        raise RuntimeError('matplotlib is required to generate figures')
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.4), squeeze=False, sharex=False)
    for row_idx, category in enumerate(DEFAULT_CATEGORY_ORDER):
        ax = axes[row_idx, 0]
        rows = list(category_rows.get(category, []))
        if not rows:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', color='#666666', fontsize=POSTER_NOTE_SIZE)
            continue
        series = _metric_series_by_state(rows, metric_name)
        for state in DEFAULT_STATE_ORDER:
            points = series.get(state, [])
            if not points:
                continue
            xs = np.asarray([p[0] for p in points], dtype=float)
            ys = np.asarray([p[1] for p in points], dtype=float)
            ax.plot(xs, ys, marker='o', markersize=3.5, color=DEFAULT_STACKED_STATE_COLORS[state], label=format_display_state(state), alpha=0.95)
        if mdates is not None:
            ax.xaxis_date()
            format_date_axis(ax)
        ax.grid(axis='y', alpha=0.2)
        ax.set_ylabel(f'{metric_label}\n{format_display_category(category)}', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
        ax.tick_params(axis='both', labelsize=max(9, POSTER_FONT_SIZE - 7))
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)
    axes[1, 0].set_xlabel('Calendar date', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    axes[0, 0].set_title(title, fontsize=max(16, POSTER_TITLE_SIZE - 8), pad=8, loc='left')
    handles = [Line2D([0], [0], color=DEFAULT_STACKED_STATE_COLORS[state], label=format_display_state(state)) for state in DEFAULT_STATE_ORDER]
    axes[0, 0].legend(handles=handles, loc='upper right', frameon=False, ncol=2, fontsize=max(10, POSTER_NOTE_SIZE - 2))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.96])
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / 'per_animal' / output_stem.split('_', 1)[0])
    output_path = figure_dir / f'{output_stem}.svg'
    return save_figure(fig, output_path, dpi=POSTER_DPI)


def plot_per_animal_metric(animal_id: str, animal_rows: Sequence[Mapping[str, Any]], metric_name: str, metric_label: str, output_dir: Path) -> List[Path]:
    rows = [dict(row) for row in animal_rows if str(row.get('animal_id')) == str(animal_id)]
    return _plot_metric_panels(
        {category: [row for row in rows if str(row.get('category')) == category] for category in DEFAULT_CATEGORY_ORDER},
        metric_name,
        metric_label,
        f'{animal_id} {metric_label.lower()} across days',
        output_dir,
        Path('per_animal') / safe_filename_component(animal_id),
        f'{safe_filename_component(animal_id)}_{safe_filename_component(metric_name)}',
    )


def plot_combined_metric(day_rows: Sequence[Mapping[str, Any]], metric_name: str, metric_label: str, output_dir: Path) -> List[Path]:
    rows = [dict(row) for row in day_rows]
    return _plot_metric_panels(
        {category: [row for row in rows if str(row.get('category')) == category] for category in DEFAULT_CATEGORY_ORDER},
        metric_name,
        metric_label,
        f'Combined {metric_label.lower()} across days',
        output_dir,
        Path('combined'),
        f'combined_{safe_filename_component(metric_name)}',
    )


def _plot_stacked_area_panel(
    category_summaries: Mapping[str, Sequence[SessionSummary]],
    *,
    x_mode: str,
    x_label: str,
    title: str,
    note: str,
    output_dir: Path,
    figure_subdir: Path,
    output_stem: str,
) -> List[Path]:
    if plt is None or mdates is None:
        raise RuntimeError('matplotlib is required to generate figures')
    fig_width = max(16.0, POSTER_WIDE_FIGSIZE[0] + 5.0)
    fig_height = max(9.4, POSTER_WIDE_FIGSIZE[1] + 2.6)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
    include_unclassified = any(needs_unclassified_band(summaries) for summaries in category_summaries.values())
    legend_handles = build_stacked_plot_handles(include_unclassified)
    for row_idx, category in enumerate(DEFAULT_CATEGORY_ORDER):
        ax = axes[row_idx, 0]
        summaries = list(category_summaries.get(category, []))
        if summaries:
            x_values, state_order, series_map, _ = build_stacked_panel_series(summaries, x_mode)
            if x_values.size == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=max(8, POSTER_FONT_SIZE - 7), color='#666666')
            else:
                colors = [DEFAULT_STACKED_STATE_COLORS[state] for state in state_order]
                ax.stackplot(x_values, *[series_map[state] for state in state_order], colors=colors, alpha=0.95, linewidth=0.35, edgecolor='white')
                ax.set_ylim(0.0, 1.05)
                ax.grid(axis='y', alpha=0.22)
                set_sparse_numeric_ticks(ax, axis='y', nbins=5)
                if x_mode == 'date':
                    ax.xaxis_date()
                    format_date_axis(ax)
                    if x_values.size == 1:
                        ax.set_xlim(float(x_values[0]) - 0.5, float(x_values[0]) + 0.5)
                    else:
                        ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)))
                else:
                    if x_values.size == 1:
                        ax.set_xlim(float(x_values[0]) - 0.5, float(x_values[0]) + 0.5)
                    else:
                        ax.set_xlim(0.5, float(np.nanmax(x_values)) + 0.5)
                    set_sparse_numeric_ticks(ax, axis='x', nbins=min(6, int(np.nanmax(x_values)) + 1), integer=True)
                missing_state_messages = stacked_panel_missing_state_messages(series_map, state_order)
                if missing_state_messages:
                    ax.text(0.985, 0.985, '\n'.join(missing_state_messages), transform=ax.transAxes, ha='right', va='top', fontsize=max(12, POSTER_NOTE_SIZE - 1), color='#555555', bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#d0d0d0', linewidth=0.8, alpha=0.88), zorder=5)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=max(8, POSTER_FONT_SIZE - 7), color='#666666')
        ax.set_ylabel(f'Fraction of time\n{format_display_category(category)}', fontsize=max(13, POSTER_LABEL_SIZE - 2), labelpad=6)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.tick_params(axis='x', labelsize=POSTER_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=POSTER_FONT_SIZE)
    axes[1, 0].set_xlabel(x_label, fontsize=POSTER_LABEL_SIZE, labelpad=8)
    fig.suptitle(title, fontsize=max(22, POSTER_SUPTITLE_SIZE - 2), y=0.985)
    fig.text(0.5, 0.93, note, ha='center', va='top', fontsize=max(15, POSTER_NOTE_SIZE + 2), color='#444444')
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.968), ncol=min(len(legend_handles), 5), frameon=False, fontsize=max(13, POSTER_NOTE_SIZE + 1))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.88])
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_STACKED_FIGURE_DIRNAME / figure_subdir)
    output_path = figure_dir / f'{output_stem}.svg'
    return save_figure(fig, output_path, dpi=POSTER_DPI)


def plot_per_animal_stacked_area(animal_id: str, day_summaries: Sequence[SessionSummary], output_dir: Path) -> List[Path]:
    animal_summaries = [summary for summary in day_summaries if str(summary.animal_id) == animal_id]
    return _plot_stacked_area_panel({category: [summary for summary in animal_summaries if str(summary.category) == category] for category in DEFAULT_CATEGORY_ORDER}, x_mode='date', x_label='Calendar date', title=f'{animal_id} sleep-state composition across days', note='Stacked areas show the fraction of time in each state; gray is Unclassified when present.', output_dir=output_dir, figure_subdir=Path('per_animal') / safe_filename_component(animal_id), output_stem=f'{safe_filename_component(animal_id)}_stacked_area')


def plot_combined_stacked_area(day_summaries: Sequence[SessionSummary], output_dir: Path) -> List[Path]:
    return _plot_stacked_area_panel({category: [summary for summary in day_summaries if str(summary.category) == category] for category in DEFAULT_CATEGORY_ORDER}, x_mode='day_index', x_label='Within-animal day index', title='Sleep-state composition across days - all animals', note='Stacked areas show the mean fraction across animals at each within-animal day index; gray is Unclassified when present.', output_dir=output_dir, figure_subdir=Path('combined'), output_stem='combined_stacked_area')


def _probability_heatmap_axes(ax: Any, summaries: Sequence[SessionSummary], *, percent: bool = False) -> None:
    if not summaries:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        return
    time_s, profile = average_probability_summaries(summaries)
    if time_s.size == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        return
    x = (time_s / np.nanmax(time_s) * 100.0) if percent and np.nanmax(time_s) > 0 else time_s / 60.0
    matrix = np.vstack([np.asarray(profile.get(state, np.zeros_like(x)), dtype=float) for state in DEFAULT_STATE_ORDER])
    ax.imshow(matrix, aspect='auto', origin='lower', interpolation='nearest', cmap=DEFAULT_PROBABILITY_CMAP, extent=[float(np.nanmin(x)), float(np.nanmax(x)), -0.5, len(DEFAULT_STATE_ORDER) - 0.5])
    ax.set_yticks(range(len(DEFAULT_STATE_ORDER)))
    ax.set_yticklabels([format_display_state(state) for state in DEFAULT_STATE_ORDER])
    ax.grid(axis='x', alpha=0.1)


def _probability_line_axes(ax: Any, summaries: Sequence[SessionSummary], *, percent: bool = False) -> None:
    if not summaries:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        return
    time_s, profile = average_probability_summaries(summaries)
    if time_s.size == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        return
    x = (time_s / np.nanmax(time_s) * 100.0) if percent and np.nanmax(time_s) > 0 else time_s / 60.0
    for state in DEFAULT_STATE_ORDER:
        values = np.asarray(profile.get(state, []), dtype=float)
        ax.plot(x, values, color=DEFAULT_STACKED_STATE_COLORS[state], label=format_display_state(state))
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=max(10, POSTER_NOTE_SIZE - 2))


def _plot_probability_figure(summaries: Sequence[SessionSummary], *, output_dir: Path, figure_subdir: Path, output_stem: str, title_prefix: str, percent: bool = False) -> List[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.6), squeeze=False)
    _probability_line_axes(axes[0, 0], summaries, percent=percent)
    _probability_heatmap_axes(axes[1, 0], summaries, percent=percent)
    axes[0, 0].set_title(f'{title_prefix} probability line', fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    axes[1, 0].set_title(f'{title_prefix} probability heatmap', fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    axes[1, 0].set_xlabel('Elapsed time (%)' if percent else DEFAULT_PROBABILITY_TIME_LABEL, fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.98])
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_PROBABILITY_FIGURE_DIRNAME / figure_subdir)
    line_path = figure_dir / f'{output_stem}_line.svg'
    heatmap_path = figure_dir / f'{output_stem}_heatmap.svg'
    return save_figure(fig, line_path, dpi=POSTER_DPI)


def plot_probability_per_animal(animal_id: str, day_summaries: Sequence[SessionSummary], output_dir: Path) -> List[Path]:
    summaries = [summary for summary in day_summaries if str(summary.animal_id) == animal_id]
    return _plot_probability_figure(summaries, output_dir=output_dir, figure_subdir=Path('per_animal') / safe_filename_component(animal_id), output_stem=f'{safe_filename_component(animal_id)}_probability', title_prefix=animal_id, percent=False)


def plot_probability_combined(day_summaries: Sequence[SessionSummary], output_dir: Path) -> List[Path]:
    return _plot_probability_figure(day_summaries, output_dir=output_dir, figure_subdir=Path('combined'), output_stem='combined_probability', title_prefix='Combined', percent=False)


def plot_probability_per_animal_percent(animal_id: str, day_summaries: Sequence[SessionSummary], output_dir: Path) -> List[Path]:
    summaries = [summary for summary in day_summaries if str(summary.animal_id) == animal_id]
    return _plot_probability_figure(summaries, output_dir=output_dir, figure_subdir=Path('per_animal') / safe_filename_component(animal_id), output_stem=f'{safe_filename_component(animal_id)}_probability_percent', title_prefix=animal_id, percent=True)


def plot_probability_combined_percent(day_summaries: Sequence[SessionSummary], output_dir: Path) -> List[Path]:
    return _plot_probability_figure(day_summaries, output_dir=output_dir, figure_subdir=Path('combined'), output_stem='combined_probability_percent', title_prefix='Combined', percent=True)


def build_sleep_state_composition(day_summaries: Sequence[SessionSummary]) -> List[Dict[str, Any]]:
    totals = {state: 0.0 for state in DEFAULT_STATE_ORDER}
    for summary in day_summaries:
        if str(summary.category) != 'sleep':
            continue
        for state in DEFAULT_STATE_ORDER:
            totals[state] += float(summary.state_time_s.get(state, 0.0))
    total_time = float(sum(totals.values()))
    rows: List[Dict[str, Any]] = []
    for state in DEFAULT_STATE_ORDER:
        state_time = float(totals[state])
        fraction = state_time / total_time if total_time > 0 else float('nan')
        rows.append({
            'state': state,
            'state_display': format_display_state(state),
            'total_time_s': state_time,
            'fraction': fraction,
            'percent': fraction * 100.0 if np.isfinite(fraction) else float('nan'),
        })
    return rows


def draw_compact_pie_panel(
    ax: Any,
    values: Sequence[float],
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    title: str,
    force_pct_labels: Optional[Sequence[str]] = None,
    legend_ncol: int = 2,
    radius: float = 0.84,
    legend_inside: bool = False,
    show_legend: bool = True,
) -> None:
    force_pct_labels = set(force_pct_labels or [])
    ax.clear()
    ax.set_aspect('equal')
    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f'{pct:.1f}%',
        pctdistance=1.08,
        radius=radius,
        wedgeprops=dict(linewidth=1.2, edgecolor='white'),
        textprops=dict(fontsize=max(10, POSTER_NOTE_SIZE - 2), color='#222222'),
    )
    ax.set_title(title, fontsize=max(14, POSTER_TITLE_SIZE - 8), pad=4)
    if show_legend:
        handles = [Patch(facecolor=color, edgecolor='none', label=label) for color, label in zip(colors, labels)]
        if legend_inside:
            ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=legend_ncol, fontsize=max(10, POSTER_NOTE_SIZE - 3), borderaxespad=0.0, handletextpad=0.4, columnspacing=0.8)
        else:
            ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=legend_ncol, fontsize=max(10, POSTER_NOTE_SIZE - 3), borderaxespad=0.0, handletextpad=0.4, columnspacing=0.8)


def plot_sleep_state_composition_pie(state_composition_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_COMPOSITION_FIGURE_DIRNAME / 'overall')
    output_path = figure_dir / 'overall_sleep_state_composition.svg'
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    values = [float(as_float(row.get('fraction'))) for row in state_composition_rows if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))]
    labels = [str(row.get('state_display', row.get('state', 'unknown'))) for row in state_composition_rows if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))]
    colors = [DEFAULT_STACKED_STATE_COLORS.get(str(row.get('state')), '#777777') for row in state_composition_rows if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))]
    draw_compact_pie_panel(ax, values, labels, colors, title='Sleep-state composition', force_pct_labels=['REM'], legend_ncol=2, radius=0.84)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    return save_figure(fig, output_path, dpi=POSTER_DPI)


def build_rem_day_presence_composition(day_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, Any], List[Mapping[str, Any]]] = defaultdict(list)
    for row in day_rows:
        if str(row.get('scope')) != 'day':
            continue
        key = (str(row.get('animal_id')), str(row.get('date')), str(row.get('category')), row.get('animal_day_index'))
        grouped[key].append(row)
    with_rem = 0
    without_rem = 0
    for rows in grouped.values():
        has_rem = any(str(row.get('state')) == 'rem' and float(row.get('state_fraction', 0.0)) > 0.0 for row in rows)
        if has_rem:
            with_rem += 1
        else:
            without_rem += 1
    total = float(with_rem + without_rem)
    rows_out = [
        {'state': 'experimental_days_with_rem', 'state_display': 'Experimental days with REM', 'fraction': with_rem / total if total else float('nan'), 'percent': (with_rem / total * 100.0) if total else float('nan')},
        {'state': 'experimental_days_without_rem', 'state_display': 'Experimental days without REM', 'fraction': without_rem / total if total else float('nan'), 'percent': (without_rem / total * 100.0) if total else float('nan')},
    ]
    return rows_out


def plot_rem_day_presence_pie(rem_day_presence_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_REM_FIGURE_DIRNAME / 'overall')
    output_path = figure_dir / 'overall_rem_day_presence.svg'
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    values = [float(as_float(row.get('fraction'))) for row in rem_day_presence_rows if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))]
    labels = [str(row.get('state_display', row.get('state', 'unknown'))) for row in rem_day_presence_rows if as_float(row.get('fraction')) is not None and np.isfinite(as_float(row.get('fraction')))]
    colors = ['#6A3D9A', '#A6761D'][: len(values)]
    draw_compact_pie_panel(ax, values, labels, colors, title='Experimental days with REM', force_pct_labels=labels, legend_ncol=1, radius=0.82)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    return save_figure(fig, output_path, dpi=POSTER_DPI)


def _rem_latencies_for_summaries(summaries: Sequence[SessionSummary]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        if str(summary.category) != 'sleep' or not summary.sleep_state_paths:
            continue
        epoch_state, epoch_t = concatenate_sleep_state_arrays([summary])
        epoch_state = np.asarray(epoch_state, dtype=int).ravel()
        epoch_t = np.asarray(epoch_t, dtype=float).ravel()
        if epoch_state.size == 0 or epoch_t.size == 0:
            continue
        rem_idx = np.where(epoch_state == 3)[0]
        wake_idx = np.where(epoch_state == 0)[0]
        first_rem_t = float(epoch_t[rem_idx[0]]) if rem_idx.size else float('nan')
        first_wake_t = float(epoch_t[wake_idx[0]]) if wake_idx.size else float('nan')
        latency_start = first_rem_t - float(epoch_t[0]) if np.isfinite(first_rem_t) else float('nan')
        latency_wake = first_rem_t - first_wake_t if np.isfinite(first_rem_t) and np.isfinite(first_wake_t) else float('nan')
        rows.append({
            'animal_id': str(summary.animal_id),
            'date': str(summary.date),
            'exp_id': ';'.join(summary.exp_ids),
            'animal_day_index': summary.animal_day_index if summary.animal_day_index is not None else '',
            'has_active_wake': bool(wake_idx.size),
            'has_rem': bool(rem_idx.size),
            'first_active_wake_time_s': first_wake_t,
            'first_rem_time_s': first_rem_t,
            'latency_since_start_s': latency_start,
            'latency_since_active_wake_s': latency_wake,
            'notes': '',
        })
    return rows


def _cdf_rows(values_by_scope: Mapping[Tuple[str, str], Sequence[float]], metric_name: str, *, bin_s: float = 300.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (scope, animal_id), values in values_by_scope.items():
        finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
        if finite.size == 0:
            continue
        max_t = float(np.nanmax(finite))
        time_bins = np.arange(0.0, max_t + bin_s, bin_s, dtype=float)
        n = float(finite.size)
        for t in time_bins:
            p = float(np.mean(finite <= t))
            sem = float(np.sqrt(max(p * (1.0 - p), 0.0) / n)) if n > 0 else float('nan')
            rows.append({
                'scope': scope,
                'animal_id': animal_id,
                'metric': metric_name,
                'time_bin_s': float(t),
                'cumulative_probability': p,
                'cumulative_probability_sem': sem,
                'fraction': p,
                'fraction_sem': sem,
                'n_days': int(n),
                'n_animals': 1 if scope.startswith('animal:') else len({aid for (_, aid) in values_by_scope if _ == scope}),
                'n_days_with_finite_metric': int(n),
            })
    return rows


def build_sleep_rem_analysis(exp_summaries: Sequence[SessionSummary], day_summaries: Sequence[SessionSummary]) -> Dict[str, Any]:
    sleep_exp_summaries = [summary for summary in exp_summaries if str(summary.category) == 'sleep']
    sleep_day_summaries = [summary for summary in day_summaries if str(summary.category) == 'sleep']
    exp_rows = _rem_latencies_for_summaries(sleep_exp_summaries)
    day_rows = _rem_latencies_for_summaries(sleep_day_summaries)

    day_index_rows: List[Dict[str, Any]] = []
    grouped_day_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in day_rows:
        day_index = row.get('animal_day_index')
        if day_index in {'', None}:
            continue
        grouped_day_index[int(day_index)].append(row)
    for day_index in sorted(grouped_day_index):
        day_group = grouped_day_index[day_index]
        start_values = [as_float(row.get('latency_since_start_s')) for row in day_group]
        awake_values = [as_float(row.get('latency_since_active_wake_s')) for row in day_group]
        finite_start = np.asarray([value for value in start_values if value is not None and np.isfinite(value)], dtype=float)
        finite_awake = np.asarray([value for value in awake_values if value is not None and np.isfinite(value)], dtype=float)
        day_index_rows.append({
            'animal_day_index': day_index,
            'n_days': len(day_group),
            'n_animals': len({str(row.get('animal_id')) for row in day_group}),
            'mean_latency_since_start_s': float(np.nanmean(finite_start)) if finite_start.size else float('nan'),
            'sem_latency_since_start_s': float(np.nanstd(finite_start, ddof=1) / np.sqrt(finite_start.size)) if finite_start.size > 1 else float('nan'),
            'mean_latency_since_active_wake_s': float(np.nanmean(finite_awake)) if finite_awake.size else float('nan'),
            'sem_latency_since_active_wake_s': float(np.nanstd(finite_awake, ddof=1) / np.sqrt(finite_awake.size)) if finite_awake.size > 1 else float('nan'),
        })

    start_by_scope: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    awake_by_scope: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in day_rows:
        animal_id = str(row.get('animal_id'))
        start = as_float(row.get('latency_since_start_s'))
        awake = as_float(row.get('latency_since_active_wake_s'))
        start_by_scope[(f'animal:{animal_id}', animal_id)].append(float(start) if start is not None else float('nan'))
        if awake is not None and np.isfinite(awake):
            awake_by_scope[(f'animal:{animal_id}', animal_id)].append(float(awake))
    combined_start = [float(as_float(row.get('latency_since_start_s'))) for row in day_rows if as_float(row.get('latency_since_start_s')) is not None and np.isfinite(as_float(row.get('latency_since_start_s')))]
    combined_awake = [float(as_float(row.get('latency_since_active_wake_s'))) for row in day_rows if as_float(row.get('latency_since_active_wake_s')) is not None and np.isfinite(as_float(row.get('latency_since_active_wake_s')))]
    start_by_scope[('combined', 'combined')] = combined_start
    awake_by_scope[('combined', 'combined')] = combined_awake

    start_curve_rows = _cdf_rows(start_by_scope, DEFAULT_REM_PROBABILITY_METRICS[0][0])
    awake_curve_rows = _cdf_rows(awake_by_scope, DEFAULT_REM_PROBABILITY_METRICS[1][0])
    fraction_curve_rows = [dict(row) for row in start_curve_rows]
    for row in fraction_curve_rows:
        row['fraction'] = row.get('cumulative_probability', float('nan'))
        row['fraction_sem'] = row.get('cumulative_probability_sem', float('nan'))

    return {
        'exp_rows': exp_rows,
        'day_rows': day_rows,
        'day_index_rows': day_index_rows,
        'start_curve_rows': start_curve_rows,
        'awake_curve_rows': awake_curve_rows,
        'fraction_curve_rows': fraction_curve_rows,
    }


def _plot_rem_metric_panels(rows: Sequence[Mapping[str, Any]], output_dir: Path, figure_subdir: Path, output_stem: str, title: str) -> List[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), squeeze=False, sharex=False)
    metrics = [('latency_since_start_s', 'First REM latency from recording start'), ('latency_since_active_wake_s', 'First REM latency from active wake')]
    for row_idx, (metric_name, metric_title) in enumerate(metrics):
        ax = axes[row_idx, 0]
        metric_rows = [row for row in rows if np.isfinite(as_float(row.get(metric_name)) or np.nan)]
        if not metric_rows:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=max(8, POSTER_FONT_SIZE - 7), color='#666666')
            continue
        by_animal: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in metric_rows:
            day_index = row.get('animal_day_index')
            if day_index in {'', None}:
                continue
            value = as_float(row.get(metric_name))
            if value is None or not np.isfinite(value):
                continue
            by_animal[str(row.get('animal_id'))].append((float(day_index), float(value)))
        for animal_id, points in sorted(by_animal.items()):
            if not points:
                continue
            points = sorted(points, key=lambda item: item[0])
            xs = np.asarray([p[0] for p in points], dtype=float)
            ys = np.asarray([p[1] for p in points], dtype=float)
            ax.plot(xs, ys, marker='o', linewidth=1.2, label=animal_id)
        ax.set_ylabel(metric_title, fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
        ax.grid(axis='y', alpha=0.22)
        ax.legend(frameon=False, fontsize=max(9, POSTER_NOTE_SIZE - 3), ncol=2)
        if row_idx == 0:
            ax.tick_params(axis='x', labelbottom=False)
    axes[1, 0].set_xlabel('Within-animal day index', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    axes[0, 0].set_title(title, fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_REM_FIGURE_DIRNAME / figure_subdir)
    return save_figure(fig, figure_dir / f'{output_stem}.svg', dpi=POSTER_DPI)


def plot_rem_latency_per_animal(animal_id: str, day_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    rows = [dict(row) for row in day_rows if str(row.get('animal_id')) == str(animal_id)]
    return _plot_rem_metric_panels(rows, output_dir, Path('per_animal') / safe_filename_component(animal_id), f'{safe_filename_component(animal_id)}_rem_latency', f'{animal_id} REM latency')


def plot_rem_latency_combined(day_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    rows = [dict(row) for row in day_rows]
    return _plot_rem_metric_panels(rows, output_dir, Path('combined'), 'combined_rem_latency', 'Combined REM latency')


def _plot_rem_probability_axes(ax: Any, rows: Sequence[Mapping[str, Any]], metric: str, title: str) -> None:
    metric_rows = [row for row in rows if str(row.get('metric')) == metric]
    if not metric_rows:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        return
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in metric_rows:
        scope = str(row.get('scope'))
        t = as_float(row.get('time_bin_s'))
        p = as_float(row.get('cumulative_probability'))
        if t is None or p is None or not np.isfinite(t) or not np.isfinite(p):
            continue
        grouped[scope].append((float(t), float(p)))
    for scope, points in sorted(grouped.items()):
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        xs = np.asarray([p[0] for p in points], dtype=float) / 60.0
        ys = np.asarray([p[1] for p in points], dtype=float)
        ax.plot(xs, ys, linewidth=1.4, label=scope)
    ax.set_ylabel(title, fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False, fontsize=max(9, POSTER_NOTE_SIZE - 3), ncol=2)


def plot_rem_probability_per_animal(animal_id: str, rem_analysis: Mapping[str, Any], output_dir: Path) -> List[Path]:
    rows = [row for row in rem_analysis.get('start_curve_rows', []) + rem_analysis.get('awake_curve_rows', []) if str(row.get('animal_id')) in {str(animal_id), 'combined'}]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    _plot_rem_probability_axes(ax, rows, DEFAULT_REM_PROBABILITY_METRICS[0][0], 'REM probability')
    ax.set_title(f'{animal_id} REM probability', fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    ax.set_xlabel('Elapsed time (min)', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_REM_FIGURE_DIRNAME / 'per_animal' / safe_filename_component(animal_id))
    return save_figure(fig, figure_dir / f'{safe_filename_component(animal_id)}_rem_probability.svg', dpi=POSTER_DPI)


def plot_rem_probability_combined(rem_analysis: Mapping[str, Any], output_dir: Path) -> List[Path]:
    rows = [row for row in rem_analysis.get('start_curve_rows', []) + rem_analysis.get('awake_curve_rows', []) if str(row.get('scope')) == 'combined']
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    _plot_rem_probability_axes(ax, rows, DEFAULT_REM_PROBABILITY_METRICS[0][0], 'REM probability')
    ax.set_title('Combined REM probability', fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    ax.set_xlabel('Elapsed time (min)', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_REM_FIGURE_DIRNAME / 'combined')
    return save_figure(fig, figure_dir / 'combined_rem_probability.svg', dpi=POSTER_DPI)


def _plot_rem_fraction_axes(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_rows = [row for row in rows if np.isfinite(as_float(row.get('fraction')) or np.nan)]
    if not metric_rows:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=POSTER_NOTE_SIZE, color='#666666')
        return
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in metric_rows:
        scope = str(row.get('scope'))
        t = as_float(row.get('time_bin_s'))
        p = as_float(row.get('fraction'))
        if t is None or p is None or not np.isfinite(t) or not np.isfinite(p):
            continue
        grouped[scope].append((float(t), float(p)))
    for scope, points in sorted(grouped.items()):
        points = sorted(points, key=lambda item: item[0])
        xs = np.asarray([p[0] for p in points], dtype=float) / 60.0
        ys = np.asarray([p[1] for p in points], dtype=float)
        ax.plot(xs, ys, linewidth=1.5, label=scope)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False, fontsize=max(9, POSTER_NOTE_SIZE - 3), ncol=2)


def plot_rem_fraction_per_animal(animal_id: str, fraction_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    rows = [row for row in fraction_rows if str(row.get('animal_id')) in {str(animal_id), 'combined'}]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    _plot_rem_fraction_axes(ax, rows)
    ax.set_title(f'{animal_id} REM fraction', fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    ax.set_xlabel('Elapsed time (min)', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    ax.set_ylabel('Fraction', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_REM_FIGURE_DIRNAME / 'fraction_time' / 'per_animal' / safe_filename_component(animal_id))
    return save_figure(fig, figure_dir / f'{safe_filename_component(animal_id)}_rem_fraction.svg', dpi=POSTER_DPI)


def plot_rem_fraction_combined(fraction_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[Path]:
    rows = [row for row in fraction_rows if str(row.get('scope')) == 'combined']
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    _plot_rem_fraction_axes(ax, rows)
    ax.set_title('Combined REM fraction', fontsize=max(15, POSTER_TITLE_SIZE - 8), pad=6, loc='left')
    ax.set_xlabel('Elapsed time (min)', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    ax.set_ylabel('Fraction', fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=6)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout*')
        fig.tight_layout()
    figure_dir = ensure_dir(output_dir / DEFAULT_FIGURE_DIRNAME / DEFAULT_REM_FIGURE_DIRNAME / 'fraction_time' / 'combined')
    return save_figure(fig, figure_dir / 'combined_rem_fraction.svg', dpi=POSTER_DPI)


def write_sleep_state_report(
    report_path: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
    rem_analysis: Mapping[str, Any],
) -> None:
    lines: List[str] = []

    def append_section(title: str) -> None:
        if lines:
            lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def append_kv(label: str, value: Any) -> None:
        lines.append(f"- {label}: {value}")

    def format_cell(value: Any, column: str) -> str:
        if value is None or value == "":
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            if not np.isfinite(value):
                return "n/a"
            if column.endswith("_s") or "time" in column or "latency" in column:
                return f"{float(value):.2f}"
            if "probability" in column or "fraction" in column:
                return f"{float(value):.3f}"
            return f"{float(value):.3f}"
        text = str(value).replace("\n", " ").replace("|", "\\|")
        if column in {"exp_ids", "sleep_state_paths"} and len(text) > 120:
            return text[:117] + "..."
        return text

    def render_table(title: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        append_section(title)
        if not rows:
            lines.append("- none")
            return
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.append(header)
        lines.append(separator)
        for row in rows:
            cells = [format_cell(row.get(column), column) for column in columns]
            lines.append("| " + " | ".join(cells) + " |")

    generated_at = str(manifest.get("generated_at", "n/a"))
    config_path = report_relative_path(manifest.get("config_path"), output_dir)
    manifest_path = report_relative_path(manifest.get("manifest_path"), output_dir)
    report_rel = report_relative_path(report_path, output_dir)

    lines.append("# Sleep-state across days report")
    lines.append(f"Generated at: {generated_at}")
    lines.append(f"Report file: {report_rel}")
    lines.append(f"Config file: {config_path}")
    lines.append(f"Manifest file: {manifest_path}")

    append_section("Overview")
    append_kv("user_id", manifest.get("user_id", "n/a"))
    append_kv("repo_base", manifest.get("repo_base", "n/a"))
    append_kv("output_dir", manifest.get("output_dir", "n/a"))
    append_kv("requested expIDs", manifest.get("n_requested_expids", 0))
    append_kv("processed expIDs", manifest.get("n_processed_expids", 0))
    append_kv("skipped expIDs", manifest.get("n_skipped_expids", 0))
    append_kv("animals", manifest.get("n_animals", 0))
    append_kv("day groups", manifest.get("n_day_groups", 0))
    append_kv("movie expIDs", len(parse_list_argument(manifest.get("movie_expids"))))
    append_kv("sleep expIDs", len(parse_list_argument(manifest.get("sleep_expids"))))

    append_section("Inputs")
    append_kv("movie expIDs", format_report_list(manifest.get("movie_expids")))
    append_kv("sleep expIDs", format_report_list(manifest.get("sleep_expids")))
    append_kv("processed expIDs", format_report_list(manifest.get("processed_expids")))
    if manifest.get("skipped_expids"):
        append_kv("skipped expIDs", format_report_list([item.get("exp_id") for item in manifest.get("skipped_expids", [])]))
    else:
        append_kv("skipped expIDs", "none")

    append_section("Primary outputs")
    append_kv("exp-level summary CSV", report_relative_path(manifest.get("exp_level_summary_path"), output_dir))
    append_kv("day-level summary CSV", report_relative_path(manifest.get("day_level_summary_path"), output_dir))
    append_kv("sleep REM per-exp CSV", report_relative_path(manifest.get("rem_exp_transition_summary_path"), output_dir))
    append_kv("sleep REM per-day CSV", report_relative_path(manifest.get("rem_day_transition_summary_path"), output_dir))
    append_kv("REM probability curve CSV", report_relative_path(manifest.get("rem_probability_curve_path"), output_dir))
    append_kv("fraction-vs-time figures", format_report_list(manifest.get("probability_artifacts")))
    append_kv("elapsed-time percent fraction figures", format_report_list(manifest.get("probability_percent_artifacts")))
    append_kv("REM fraction curve CSV", report_relative_path(manifest.get("rem_fraction_curve_path"), output_dir))
    append_kv("poster-ready composite", format_report_list(manifest.get("poster_ready_artifacts")))
    append_kv("state montage figures", format_report_list(manifest.get("state_montage_artifacts")))
    append_kv("report", report_rel)

    append_section("Metric definitions")
    lines.append("- Overall sleep-state fraction for state $s$:")
    lines.append(r"  $$f_s = \\frac{T_s}{\\sum_{k \\in \\{\\mathrm{active\\ wake}, \\mathrm{quiet\\ wake}, \\mathrm{NREM}, \\mathrm{REM}, \\mathrm{unclassified}\\}} T_k}$$")
    lines.append("  - $T_s$ is the total time spent in state $s$ after pooling all processed recordings.")
    lines.append("  - The denominator is the total pooled time across the canonical states plus any unclassified time.")
    lines.append("- REM experimental-day presence fraction:")
    lines.append(r"  $$p_{\mathrm{REM\ day}} = \frac{N_{\mathrm{days\ with\ REM}}}{N_{\mathrm{experimental\ days}}}$$")
    lines.append("  - The numerator counts all experimental day summaries that contain at least one REM epoch.")
    lines.append("  - The denominator counts all experimental day summaries that were analyzed.")
    lines.append("- First-REM latency from recording start:")
    lines.append(r"  $$L_{\\mathrm{start}} = t_{\\mathrm{first\\ REM}} - t_{\\mathrm{recording\\ start}}$$")
    lines.append("  - This is the elapsed time from the start of the sleep recording to the first REM epoch.")
    lines.append("- First-REM latency from first active wake:")
    lines.append(r"  $$L_{\\mathrm{awake}} = t_{\\mathrm{first\\ REM}} - t_{\\mathrm{first\\ active\\ wake}}$$")
    lines.append("  - This is measured only when the day contains active wake and REM occurs after that first active-wake epoch.")
    lines.append("- Cumulative REM probability by elapsed time $t$:")
    lines.append(r"  $$F_{\\mathrm{REM}}(t) = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbf{1}(L_i \\le t)$$")
    lines.append("  - This is the empirical cumulative distribution of REM latencies across valid days.")
    lines.append("  - $N$ is the number of valid days contributing to the curve.")
    lines.append("- REM fraction in elapsed-time bin $b$:")
    lines.append(r"  $$q_{\\mathrm{REM}}(b) = \\frac{n_{\\mathrm{REM}, b}}{n_b}$$")
    lines.append(r"  - $n_{\mathrm{REM}, b}$ is the number of REM epochs in elapsed-time bin $b$ after pooling same-day recordings and binning each session in 5-minute steps.")
    lines.append("  - $n_b$ is the total epoch count in that 5-minute bin.")
    append_section("REM analysis notes")
    lines.append("- only sleep-category expIDs are used for the REM transition summary tables")
    lines.append("- same-day sleep recordings are concatenated in exp_id order before REM latency is measured")
    lines.append("- same-day sleep recordings are pooled in exp_id order before sleep-state fraction is calculated")
    lines.append("- fraction figures include both elapsed-time minute and elapsed-time percent views")
    lines.append("- days without active wake are excluded from the active-wake-aligned probability curve")
    lines.append("- days without REM stay in the recording-start probability curve as no-event days")
    lines.append("- the REM fraction curve shows the mean REM occupancy in 5-minute elapsed-time bins, pooled within day and then across animals")
    lines.append("- the poster-ready composite uses a top montage example, representative day-timeline views, and summary pies")

    figure_artifacts = list(manifest.get("figure_artifacts", []))
    stacked_area_artifacts = list(manifest.get("stacked_area_artifacts", []))
    probability_artifacts = list(manifest.get("probability_artifacts", []))
    state_montage_artifacts = list(manifest.get("state_montage_artifacts", []))
    review_state_montage_artifacts = list(manifest.get("review_state_montage_artifacts", []))
    rem_artifacts = list(manifest.get("rem_artifacts", []))
    rem_latency_artifacts = list(manifest.get("rem_latency_artifacts", []))
    rem_probability_artifacts = list(manifest.get("rem_probability_artifacts", []))
    rem_fraction_artifacts = list(manifest.get("rem_fraction_artifacts", []))
    rem_day_presence_artifacts = list(manifest.get("rem_day_presence_artifacts", []))
    composition_artifacts = list(manifest.get("composition_artifacts", []))
    poster_ready_artifacts = list(manifest.get("poster_ready_artifacts", []))
    render_table(
        "Overall sleep-state composition",
        manifest.get("state_composition_rows", []),
        ["state_display", "total_time_s", "fraction", "percent"],
    )
    append_section("Figures")
    if figure_artifacts:
        lines.append("- standard summary figures")
        for artifact in figure_artifacts:
            lines.append(f"  - {artifact}")
    if stacked_area_artifacts:
        lines.append("- stacked-area figures")
        for artifact in stacked_area_artifacts:
            lines.append(f"  - {artifact}")
    if probability_artifacts:
        lines.append("- probability-vs-time figures")
        for artifact in probability_artifacts:
            lines.append(f"  - {artifact}")
    if rem_latency_artifacts:
        lines.append("- REM latency figures")
        for artifact in rem_latency_artifacts:
            lines.append(f"  - {artifact}")
    if rem_probability_artifacts:
        lines.append("- REM probability figures")
        for artifact in rem_probability_artifacts:
            lines.append(f"  - {artifact}")
    if rem_fraction_artifacts:
        lines.append("- REM fraction figures")
        for artifact in rem_fraction_artifacts:
            lines.append(f"  - {artifact}")
    if rem_day_presence_artifacts:
        lines.append("- REM day-presence figures")
        for artifact in rem_day_presence_artifacts:
            lines.append(f"  - {artifact}")
    if composition_artifacts:
        lines.append("- sleep-state composition figures")
        for artifact in composition_artifacts:
            lines.append(f"  - {artifact}")
    if state_montage_artifacts:
        lines.append("- state montage figures")
        for artifact in state_montage_artifacts:
            lines.append(f"  - {artifact}")
    if review_state_montage_artifacts:
        lines.append("- review montage example")
        for artifact in review_state_montage_artifacts:
            lines.append(f"  - {artifact}")
    if poster_ready_artifacts:
        lines.append("- poster-ready composite")
        for artifact in poster_ready_artifacts:
            lines.append(f"  - {artifact}")
    if not (figure_artifacts or stacked_area_artifacts or probability_artifacts or rem_latency_artifacts or rem_probability_artifacts or rem_fraction_artifacts or rem_day_presence_artifacts or composition_artifacts or state_montage_artifacts or review_state_montage_artifacts or poster_ready_artifacts):
        lines.append("- none")

    render_table(
        "Sleep REM per-exp transitions",
        rem_analysis.get("exp_rows", []),
        [
            "animal_id",
            "date",
            "exp_id",
            "has_active_wake",
            "has_rem",
            "first_active_wake_time_s",
            "first_rem_time_s",
            "latency_since_start_s",
            "latency_since_active_wake_s",
            "notes",
        ],
    )

    render_table(
        "Sleep REM day-pooled transitions",
        rem_analysis.get("day_rows", []),
        [
            "animal_id",
            "date",
            "animal_day_index",
            "n_expids",
            "exp_ids",
            "has_active_wake",
            "has_rem",
            "first_active_wake_time_s",
            "first_rem_time_s",
            "latency_since_start_s",
            "latency_since_active_wake_s",
            "notes",
        ],
    )

    render_table(
        "Day-index summary across animals",
        rem_analysis.get("day_index_rows", []),
        [
            "animal_day_index",
            "n_days",
            "n_animals",
            "mean_latency_since_start_s",
            "sem_latency_since_start_s",
            "mean_latency_since_active_wake_s",
            "sem_latency_since_active_wake_s",
        ],
    )

    start_rows = [row for row in rem_analysis.get("start_curve_rows", []) if str(row.get("scope", "")).startswith("animal:")]
    combined_start_rows = [row for row in rem_analysis.get("start_curve_rows", []) if str(row.get("scope")) == "combined"]
    awake_rows = [row for row in rem_analysis.get("awake_curve_rows", []) if str(row.get("scope", "")).startswith("animal:")]
    combined_awake_rows = [row for row in rem_analysis.get("awake_curve_rows", []) if str(row.get("scope")) == "combined"]

    render_table(
        "REM cumulative probability since recording start",
        start_rows + combined_start_rows,
        [
            "scope",
            "animal_id",
            "metric",
            "time_bin_s",
            "cumulative_probability",
            "cumulative_probability_sem",
            "n_days",
            "n_animals",
            "n_days_with_finite_metric",
        ],
    )

    render_table(
        "REM cumulative probability since first active wake",
        awake_rows + combined_awake_rows,
        [
            "scope",
            "animal_id",
            "metric",
            "time_bin_s",
            "cumulative_probability",
            "cumulative_probability_sem",
            "n_days",
            "n_animals",
            "n_days_with_finite_metric",
        ],
    )

    render_table(
        "REM fraction over elapsed time",
        rem_analysis.get("fraction_curve_rows", []),
        [
            "scope",
            "animal_id",
            "metric",
            "time_bin_s",
            "fraction",
            "fraction_sem",
            "n_days",
            "n_animals",
            "n_days_with_finite_metric",
        ],
    )

    render_table(
        "Sleep REM day presence summary",
        manifest.get("rem_day_presence_rows", []),
        [
            "state_display",
            "day_count",
            "fraction",
            "percent",
        ],
    )

    append_section("Missing data notes")
    skipped = manifest.get("skipped_expids", [])
    if skipped:
        for row in skipped:
            append_kv(
                str(row.get("exp_id", "unknown")),
                f"{row.get('reason', 'skipped')} | {row.get('category', 'n/a')} | {row.get('animal_id', 'n/a')} | {row.get('date', 'n/a')}",
            )
    else:
        lines.append("- none")

    write_text_report(report_path, lines)


def build_config_payload(config: Mapping[str, Any]) -> Dict[str, Any]:
    user_id = config.get("user_id")
    repo_base = config.get("repo_base")
    if repo_base is None:
        if not user_id:
            raise SystemExit("Config must define either repo_base or user_id.")
        repo_base = f"/home/{user_id}/data/Repository"
    return {
        "user_id": str(user_id) if user_id is not None else None,
        "repo_base": str(repo_base),
        "movie_expids": parse_list_argument(config.get("movie_expids")),
        "sleep_expids": parse_list_argument(config.get("sleep_expids")),
        "output_dir": str(config.get("output_dir") or DEFAULT_OUTPUT_DIR),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sleep state across days for each animal and across animals.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON config with repo path and expID lists.")
    parser.add_argument("--user-id", type=str, default=None, help="Override the user_id from the config.")
    parser.add_argument("--repo-base", type=Path, default=None, help="Override the repository base path.")
    parser.add_argument("--movie-expids", type=str, default=None, help="Override movie expIDs as a comma-separated list.")
    parser.add_argument("--sleep-expids", type=str, default=None, help="Override sleep expIDs as a comma-separated list.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the analysis output directory.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if plt is None:
        raise RuntimeError("matplotlib is required to run this analysis")

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_json_config(args.config) if args.config.exists() else {}
    config_payload = build_config_payload(config)
    if args.user_id is not None:
        config_payload["user_id"] = args.user_id
    if args.repo_base is not None:
        config_payload["repo_base"] = str(args.repo_base)
    if args.movie_expids is not None:
        config_payload["movie_expids"] = parse_list_argument(args.movie_expids)
    if args.sleep_expids is not None:
        config_payload["sleep_expids"] = parse_list_argument(args.sleep_expids)
    if args.output_dir is not None:
        config_payload["output_dir"] = str(args.output_dir)

    repo_base = Path(config_payload["repo_base"])
    output_dir = Path(config_payload["output_dir"])
    ensure_dir(output_dir)
    if not repo_base.exists():
        raise SystemExit(f"Repository base does not exist: {repo_base}")

    requested_expids = build_requested_expids(config_payload)
    if not requested_expids:
        raise SystemExit("No expIDs were configured.")

    exp_summaries, skipped = collect_exp_summaries(requested_expids, repo_base)
    if not exp_summaries:
        raise SystemExit("No sleep_state.pickle bundles were loaded successfully.")

    day_summaries = aggregate_day_summaries(exp_summaries)
    assign_day_indices(day_summaries)

    exp_rows = rows_to_metric_rows(
        [row for summary in exp_summaries for row in summary_to_rows(summary)]
    )
    day_rows = rows_to_metric_rows(
        [row for summary in day_summaries for row in summary_to_rows(summary)]
    )

    exp_table_path = output_dir / "exp_level_summary.csv"
    day_table_path = output_dir / "day_level_summary.csv"
    write_csv_rows(exp_table_path, exp_rows, fieldnames=sorted({key for row in exp_rows for key in row.keys()}))
    write_csv_rows(day_table_path, day_rows, fieldnames=sorted({key for row in day_rows for key in row.keys()}))

    rem_analysis = build_sleep_rem_analysis(exp_summaries, day_summaries)
    rem_exp_table_path = output_dir / "sleep_rem_exp_transition_summary.csv"
    rem_day_table_path = output_dir / "sleep_rem_day_transition_summary.csv"
    rem_probability_table_path = output_dir / "sleep_rem_probability_curve.csv"
    rem_fraction_table_path = output_dir / "sleep_rem_fraction_curve.csv"
    write_csv_rows(
        rem_exp_table_path,
        rem_analysis["exp_rows"],
        fieldnames=sorted({key for row in rem_analysis["exp_rows"] for key in row.keys()}),
    )
    write_csv_rows(
        rem_day_table_path,
        rem_analysis["day_rows"],
        fieldnames=sorted({key for row in rem_analysis["day_rows"] for key in row.keys()}),
    )
    write_csv_rows(
        rem_probability_table_path,
        rem_analysis["start_curve_rows"] + rem_analysis["awake_curve_rows"],
        fieldnames=sorted({key for row in rem_analysis["start_curve_rows"] + rem_analysis["awake_curve_rows"] for key in row.keys()}),
    )
    write_csv_rows(
        rem_fraction_table_path,
        rem_analysis["fraction_curve_rows"],
        fieldnames=sorted({key for row in rem_analysis["fraction_curve_rows"] for key in row.keys()}),
    )

    figure_artifacts: List[str] = []
    stacked_area_artifacts: List[str] = []
    probability_artifacts: List[str] = []
    probability_percent_artifacts: List[str] = []
    rem_latency_artifacts: List[str] = []
    rem_probability_artifacts: List[str] = []
    rem_fraction_artifacts: List[str] = []
    rem_day_presence_artifacts: List[str] = []
    composition_artifacts: List[str] = []
    state_montage_artifacts: List[str] = []
    poster_ready_artifacts: List[str] = []
    sleep_exp_summaries = [summary for summary in exp_summaries if str(summary.category) == "sleep"]
    per_animal_rows = group_rows_by_animal(day_rows)
    for animal_id, animal_rows in per_animal_rows.items():
        for metric_name, metric_label, _ in DEFAULT_METRIC_SPECS:
            saved = plot_per_animal_metric(animal_id, animal_rows, metric_name, metric_label, output_dir)
            figure_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    for metric_name, metric_label, _ in DEFAULT_METRIC_SPECS:
        saved = plot_combined_metric(day_rows, metric_name, metric_label, output_dir)
        figure_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    per_animal_summaries = group_summaries_by_animal(day_summaries)
    for animal_id, animal_summaries in per_animal_summaries.items():
        saved = plot_per_animal_stacked_area(animal_id, animal_summaries, output_dir)
        stacked_area_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    saved = plot_combined_stacked_area(day_summaries, output_dir)
    stacked_area_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    for animal_id in per_animal_summaries:
        saved = plot_probability_per_animal(animal_id, day_summaries, output_dir)
        probability_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    saved = plot_probability_combined(day_summaries, output_dir)
    probability_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    for animal_id in per_animal_summaries:
        saved = plot_probability_per_animal_percent(animal_id, day_summaries, output_dir)
        probability_percent_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    saved = plot_probability_combined_percent(day_summaries, output_dir)
    probability_percent_artifacts.extend(report_relative_path(path, output_dir) for path in saved)

    for animal_id in per_animal_rows:
        saved = plot_rem_latency_per_animal(animal_id, day_rows, output_dir)
        rem_latency_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    saved = plot_rem_latency_combined(day_rows, output_dir)
    rem_latency_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    for animal_id in per_animal_summaries:
        saved = plot_rem_probability_per_animal(animal_id, rem_analysis, output_dir)
        rem_probability_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    saved = plot_rem_probability_combined(rem_analysis, output_dir)
    rem_probability_artifacts.extend(report_relative_path(path, output_dir) for path in saved)

    for animal_id in sorted({str(summary.animal_id) for summary in day_summaries if str(summary.category) == 'sleep'}):
        saved = plot_rem_fraction_per_animal(animal_id, rem_analysis.get('fraction_curve_rows', []), output_dir)
        rem_fraction_artifacts.extend(report_relative_path(path, output_dir) for path in saved)
    saved = plot_rem_fraction_combined(rem_analysis.get('fraction_curve_rows', []), output_dir)
    rem_fraction_artifacts.extend(report_relative_path(path, output_dir) for path in saved)

    state_composition_rows = build_sleep_state_composition(day_summaries)
    saved = plot_sleep_state_composition_pie(state_composition_rows, output_dir)
    composition_artifacts.extend(report_relative_path(path, output_dir) for path in saved)

    rem_day_presence_rows = build_rem_day_presence_composition(day_rows)
    saved = plot_rem_day_presence_pie(rem_day_presence_rows, output_dir)
    rem_day_presence_artifacts.extend(report_relative_path(path, output_dir) for path in saved)

    for summary in sorted(sleep_exp_summaries, key=lambda s: (str(s.animal_id), str(s.date), str(s.exp_ids[0]) if s.exp_ids else "", str(s.sleep_state_paths[0]) if s.sleep_state_paths else "")):
        saved = plot_state_montage_per_exp(summary, output_dir)
        state_montage_artifacts.extend(report_relative_path(path, output_dir) for path in saved)

    saved = plot_sleep_state_poster_ready_composite(state_composition_rows, rem_day_presence_rows, exp_summaries, day_summaries, state_montage_artifacts, output_dir)
    poster_ready_artifacts.extend(project_relative_path(path) for path in saved)

    review_state_montage_artifacts: List[str] = []
    if state_montage_artifacts:
        review_dir = ensure_dir(DEFAULT_REVIEW_FIGURES_DIR / DEFAULT_REVIEW_STATE_MONTAGE_FIGURE_DIRNAME)
        selected_state_montage = None
        for artifact in state_montage_artifacts:
            if DEFAULT_REVIEW_STATE_MONTAGE_EXAMPLE_EXP_ID in artifact:
                selected_state_montage = Path(output_dir / artifact)
                break
        if selected_state_montage is None:
            selected_state_montage = Path(output_dir / state_montage_artifacts[0])
        if selected_state_montage.exists():
            selected_state_svg = selected_state_montage.with_suffix('.svg')
            if selected_state_svg.exists():
                review_svg_copy = review_dir / selected_state_svg.name
                shutil.copy2(selected_state_svg, review_svg_copy)
                review_state_montage_artifacts.append(str(review_svg_copy.relative_to(ROOT_DIR)))
    rem_artifacts = sorted(set(rem_latency_artifacts + rem_probability_artifacts + rem_fraction_artifacts + rem_day_presence_artifacts))
    probability_all_artifacts = sorted(set(probability_artifacts + probability_percent_artifacts))
    all_figure_artifacts = sorted(set(figure_artifacts + stacked_area_artifacts + probability_all_artifacts + rem_artifacts + composition_artifacts + state_montage_artifacts + poster_ready_artifacts + review_state_montage_artifacts))

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "config_path": str(args.config),
        "user_id": config_payload.get("user_id"),
        "repo_base": str(repo_base),
        "output_dir": str(output_dir),
        "movie_expids": config_payload["movie_expids"],
        "sleep_expids": config_payload["sleep_expids"],
        "requested_expids": requested_expids,
        "processed_expids": [summary.exp_ids[0] for summary in exp_summaries],
        "skipped_expids": skipped,
        "n_requested_expids": len(requested_expids),
        "n_processed_expids": len(exp_summaries),
        "n_skipped_expids": len(skipped),
        "n_animals": len({summary.animal_id for summary in exp_summaries}),
        "n_day_groups": len(day_summaries),
        "state_order": DEFAULT_STATE_ORDER,
        "category_order": DEFAULT_CATEGORY_ORDER,
        "figure_artifacts": figure_artifacts,
        "all_figure_artifacts": all_figure_artifacts,
        "stacked_area_artifacts": sorted(set(stacked_area_artifacts)),
        "probability_artifacts": sorted(set(probability_artifacts)),
        "probability_percent_artifacts": sorted(set(probability_percent_artifacts)),
        "rem_artifacts": rem_artifacts,
        "rem_latency_artifacts": sorted(set(rem_latency_artifacts)),
        "rem_probability_artifacts": sorted(set(rem_probability_artifacts)),
        "rem_fraction_artifacts": sorted(set(rem_fraction_artifacts)),
        "rem_day_presence_artifacts": sorted(set(rem_day_presence_artifacts)),
        "composition_artifacts": sorted(set(composition_artifacts)),
        "state_montage_artifacts": sorted(set(state_montage_artifacts)),
        "review_state_montage_artifacts": sorted(set(review_state_montage_artifacts)),
        "poster_ready_artifacts": sorted(set(poster_ready_artifacts)),
        "state_composition_rows": state_composition_rows,
        "rem_day_presence_rows": rem_day_presence_rows,
        "rem_fraction_rows": rem_analysis.get("fraction_curve_rows", []),
        "rem_table_artifacts": [
            report_relative_path(rem_exp_table_path, output_dir),
            report_relative_path(rem_day_table_path, output_dir),
            report_relative_path(rem_probability_table_path, output_dir),
            report_relative_path(rem_fraction_table_path, output_dir),
        ],
        "table_artifacts": [
            report_relative_path(exp_table_path, output_dir),
            report_relative_path(day_table_path, output_dir),
            report_relative_path(rem_exp_table_path, output_dir),
            report_relative_path(rem_day_table_path, output_dir),
            report_relative_path(rem_probability_table_path, output_dir),
            report_relative_path(rem_fraction_table_path, output_dir),
        ],
        "exp_level_summary_path": report_relative_path(exp_table_path, output_dir),
        "day_level_summary_path": report_relative_path(day_table_path, output_dir),
        "rem_exp_transition_summary_path": report_relative_path(rem_exp_table_path, output_dir),
        "rem_day_transition_summary_path": report_relative_path(rem_day_table_path, output_dir),
        "rem_probability_curve_path": report_relative_path(rem_probability_table_path, output_dir),
        "rem_fraction_curve_path": report_relative_path(rem_fraction_table_path, output_dir),
        "source_sleep_state_paths": {
            summary.exp_ids[0]: summary.sleep_state_paths[0] if summary.sleep_state_paths else None
            for summary in exp_summaries
        },
    }

    manifest_path = output_dir / "sleep_state_across_days_manifest.json"
    report_path = output_dir / "sleep_state_across_days_report.md"
    manifest["manifest_path"] = report_relative_path(manifest_path, output_dir)
    manifest["report_path"] = report_relative_path(report_path, output_dir)
    manifest["report_artifacts"] = [report_relative_path(report_path, output_dir)]
    manifest["poster_ready_composite_path"] = poster_ready_artifacts[0] if poster_ready_artifacts else None
    write_sleep_state_report(report_path, output_dir, manifest, rem_analysis)
    write_json(manifest_path, manifest)

    print(json.dumps(
        {
            "n_processed_expids": manifest["n_processed_expids"],
            "n_skipped_expids": manifest["n_skipped_expids"],
            "n_animals": manifest["n_animals"],
            "n_day_groups": manifest["n_day_groups"],
            "output_dir": str(output_dir),
            "manifest_path": report_relative_path(manifest_path, output_dir),
            "report_path": report_relative_path(report_path, output_dir),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
