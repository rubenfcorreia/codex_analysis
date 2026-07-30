#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
ANALYSIS_DIR = ROOT_DIR / 'analysis'
DENDRITES_PIPELINE_DIR = ANALYSIS_DIR / 'dendrites_pipeline'
for extra_path in (ROOT_DIR, ANALYSIS_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

if __name__ == "__main__":
    sys.modules.setdefault("analysis.deprecated.visual_response.movie_visual_response", sys.modules[__name__])

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import colors as mcolors
except Exception:  # pragma: no cover - matplotlib is required for real figure generation
    plt = None

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

from analysis.dendrites_pipeline.dendrites_pipeline import (
    as_float,
    as_int,
    classify_movie_name,
    derive_animal_id,
    derive_date,
    determine_conversion_mode,
    ensure_dir,
    extract_cut_neural_bundle,
    extract_movie_feature_prefixes,
    find_first_key,
    jsonable,
    load_conversion_library,
    locate_conversion_file,
    movie_feature_blocks,
    normalize_conversion_library,
    parse_list_argument,
    read_csv_rows,
    read_pickle,
    resolve_repo_root,
    safe_filename_component,
)

if plt is not None:
    configure_poster_matplotlib()

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "visual_response"
DEFAULT_POSTER_OUTPUT_DIR = ROOT_DIR / "results" / "poster_ready"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "movie_visual_response_config.json"
DEFAULT_REMOTE_REPO_BASE = Path("/data/Remote_Repository")
DEFAULT_PRE_WINDOW_S = 0.5
DEFAULT_POST_WINDOW_S = 1.5
DEFAULT_DPI = POSTER_DPI
MOVIE_CATEGORIES = ("blank", "zebra", "movies", "gratings")
MOVIE_CATEGORY_ALIASES = {"grating": "gratings"}
MOVIE_CATEGORY_COLORS = {
    "basal": "#4C78A8",
    "apical": "#F58518",
}
SOMA_TRACE_COLOR = "#9B59B6"
SOMA_RETINO_COLOR = "#2F855A"
RETINO_AXIS_SUFFIXES = (
    "angle",
    "azimuth",
    "elevation",
    "orientation",
    "ori",
    "position",
    "pos",
    "x",
    "y",
    "phase",
)

@dataclass
class TrialRecord:
    trial_index: int
    onset: float
    duration: float
    category: Optional[str]
    label: str
    axis_value: Optional[float] = None
    axis_label: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    position_x_label: Optional[str] = None
    position_y_label: Optional[str] = None

@dataclass
class LoadedExperiment:
    exp_id: str
    animal_id: str
    date: str
    role: str
    exp_root: Path
    trial_csv_path: Path
    cut_path: Optional[Path]
    cut_t: np.ndarray
    cut_array: Optional[np.ndarray]
    cut_bundle: Optional[Dict[str, Any]]
    soma_source_path: Optional[Path]
    soma_source_kind: Optional[str]
    soma_trace_t: Optional[np.ndarray]
    soma_trace: Optional[np.ndarray]
    soma_bundle: Optional[Dict[str, Any]]
    conversion_path: Optional[Path]
    conversion_mode: Optional[str]
    conversion_library: Dict[str, Dict[str, Any]]
    conversion_source_exp_id: Optional[str]
    used_same_day_fallback: bool
    trial_rows: List[Dict[str, str]]
    trial_columns: List[str]
    trial_records: List[TrialRecord]
    alerts: List[str]

def compact_list(values: Any) -> List[str]:
    items = parse_list_argument(values)
    deduped: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped

def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")

def load_json_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data

def merge_config(cli: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(file_config)
    for key, value in cli.items():
        if value is None:
            continue
        merged[key] = value
    return merged

def load_trial_rows_and_columns(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    rows = read_csv_rows(path)
    if not rows:
        return [], []
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    return rows, columns

def resolve_trial_csv_path(repo_base: Path, remote_repo_base: Path, animal_id: str, exp_id: str) -> Path:
    processed_path = resolve_repo_root(repo_base, animal_id, exp_id) / f"{exp_id}_all_trials.csv"
    if processed_path.exists():
        return processed_path
    remote_path = resolve_repo_root(remote_repo_base, animal_id, exp_id) / f"{exp_id}_all_trials.csv"
    return remote_path if remote_path.exists() else processed_path

def load_mat_bundle(path: Path) -> Dict[str, Any]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    bundle = data.get("s2pData", data)
    if hasattr(bundle, "_fieldnames"):
        bundle = {field: getattr(bundle, field) for field in bundle._fieldnames}
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a MATLAB struct or dictionary in {path}")
    return bundle

def load_continuous_trace_bundle(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if path.suffix.lower() == ".pickle":
        bundle = read_pickle(path)
        if not isinstance(bundle, dict):
            raise TypeError(f"Expected a dictionary in {path}")
    elif path.suffix.lower() == ".mat":
        bundle = load_mat_bundle(path)
    else:
        raise ValueError(f"Unsupported continuous trace bundle format: {path.suffix}")
    t = np.asarray(find_first_key(bundle, ["t"]), dtype=float)
    trace = find_first_key(bundle, ["dF", "alldF", "dff", "trace", "calcium"])
    if trace is None:
        raise KeyError(f"No trace array found in {path}")
    trace_array = np.asarray(trace, dtype=float)
    if trace_array.ndim == 2 and trace_array.shape[0] == t.size and trace_array.shape[1] != t.size:
        trace_array = trace_array.T
    if trace_array.ndim == 2 and trace_array.shape[-1] != t.size and trace_array.shape[0] != t.size:
        raise ValueError(f"Trace array in {path} does not align with time vector {t.shape}")
    return t, trace_array, bundle

def normalize_continuous_traces(trace: np.ndarray, t: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    t = np.asarray(t, dtype=float)
    if trace.ndim == 1:
        return trace[np.newaxis, :]
    if trace.ndim != 2:
        raise ValueError(f"Expected 1D or 2D continuous traces, got {trace.shape}")
    if trace.shape[0] == t.size and trace.shape[1] != t.size:
        trace = trace.T
    if trace.shape[-1] != t.size:
        raise ValueError(f"Trace shape {trace.shape} does not align with time vector {t.shape}")
    return trace

def extract_aligned_continuous_trace(
    trace: np.ndarray,
    t: np.ndarray,
    onset: float,
    pre_window_s: float,
    post_window_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=float)
    t = np.asarray(t, dtype=float)
    if trace.shape != t.shape:
        if trace.size == t.size:
            trace = trace.reshape(t.shape)
        else:
            raise ValueError("Trace and time vector must have matching shapes for interpolation")
    if t.size < 2:
        relative_t = np.linspace(-pre_window_s, post_window_s, 2)
        aligned = np.interp(onset + relative_t, t, trace, left=np.nan, right=np.nan)
        return relative_t, aligned
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        dt = (pre_window_s + post_window_s) / max(int(t.size), 1)
    n_samples = max(2, int(round((pre_window_s + post_window_s) / dt)) + 1)
    relative_t = np.linspace(-pre_window_s, post_window_s, n_samples)
    aligned = np.interp(onset + relative_t, t, trace, left=np.nan, right=np.nan)
    return relative_t, aligned

def resolve_cut_bundle_path(
    exp_root: Path,
    channel: int,
    include_plain_cut: bool = False,
) -> Tuple[Optional[Path], Optional[str]]:
    candidates = [
        (exp_root / "cut_intertrials" / f"s2p_ch{channel}_dF_cut.pickle", "cut_intertrials"),
        (exp_root / "cut_with_intertrials" / f"s2p_ch{channel}_dF_cut.pickle", "cut_with_intertrials"),
    ]
    if include_plain_cut:
        candidates.append((exp_root / "cut" / f"s2p_ch{channel}_dF_cut.pickle", "cut"))
    for path, label in candidates:
        if path.exists():
            return path, label
    return None, None

def lighten_color(color: str, mix_with_white: float = 0.65) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - mix_with_white) + np.ones(3, dtype=float) * mix_with_white
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))

def normalize_group_ids(values: Any) -> List[str]:
    return compact_list(values)

def normalize_soma_group_map(raw_map: Any) -> List[Dict[str, Any]]:
    if raw_map is None:
        return []
    if isinstance(raw_map, list):
        items = raw_map
    elif isinstance(raw_map, dict):
        items = []
        for name, group in raw_map.items():
            if not isinstance(group, dict):
                raise TypeError(f"soma_group_map entry for {name!r} must be an object")
            item = dict(group)
            item.setdefault("name", name)
            items.append(item)
    else:
        raise TypeError("soma_group_map must be either an object or a list of objects")

    normalized: List[Dict[str, Any]] = []
    for index, group in enumerate(items, start=1):
        if not isinstance(group, dict):
            raise TypeError(f"soma_group_map entry #{index} must be an object")
        name = str(group.get("name") or group.get("label") or f"group_{index}")
        soma_expids = normalize_group_ids(group.get("soma_expids"))
        basal_expids = normalize_group_ids(group.get("basal_expids"))
        apical_expids = normalize_group_ids(group.get("apical_expids"))
        if not soma_expids:
            raise ValueError(f"soma group {name!r} does not define any soma_expids")
        if not basal_expids and not apical_expids:
            raise ValueError(f"soma group {name!r} must reference at least one basal or apical expID")
        normalized.append(
            {
                "name": name,
                "soma_expids": soma_expids,
                "basal_expids": basal_expids,
                "apical_expids": apical_expids,
                "label": str(group.get("label") or name),
            }
        )
    return normalized

def normalize_movie_category(category: Optional[str]) -> Optional[str]:
    if category is None:
        return None
    return MOVIE_CATEGORY_ALIASES.get(str(category), str(category))

def trial_label_from_movie_row(row: Dict[str, str], columns: Sequence[str]) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
    blocks = movie_feature_blocks(row, columns)
    if not blocks:
        return None, [], None
    if len(blocks) > 1:
        return None, blocks, "multiple movie features detected"
    category = normalize_movie_category(classify_movie_name(blocks[0]["name"]))
    return category, blocks, None

def infer_retino_axis(row: Dict[str, str], columns: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    prefixes = extract_movie_feature_prefixes(columns)
    for prefix in prefixes:
        for suffix in RETINO_AXIS_SUFFIXES:
            value = as_float(row.get(f"{prefix}_{suffix}"))
            if value is not None:
                return value, f"{prefix}_{suffix}"
    for field in ("stim", "stimulus", "position", "angle", "orientation", "phase", "x", "y"):
        value = as_float(row.get(field))
        if value is not None:
            return value, field
    return None, None

def infer_retino_position(row: Dict[str, str], columns: Sequence[str]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    prefixes = extract_movie_feature_prefixes(columns)
    for prefix in prefixes:
        x_value = as_float(row.get(f"{prefix}_x"))
        y_value = as_float(row.get(f"{prefix}_y"))
        if x_value is not None and y_value is not None:
            return x_value, y_value, f"{prefix}_x", f"{prefix}_y"
    x_value = as_float(row.get("position_x"))
    y_value = as_float(row.get("position_y"))
    if x_value is not None and y_value is not None:
        return x_value, y_value, "position_x", "position_y"
    x_value = as_float(row.get("x"))
    y_value = as_float(row.get("y"))
    if x_value is not None and y_value is not None:
        return x_value, y_value, "x", "y"
    return None, None, None, None

def build_movie_trial_records(rows: List[Dict[str, str]], columns: Sequence[str]) -> Tuple[List[TrialRecord], List[str]]:
    records: List[TrialRecord] = []
    alerts: List[str] = []
    for trial_index, row in enumerate(rows):
        onset = as_float(row.get("time"))
        duration = as_float(row.get("duration"))
        if onset is None or duration is None:
            alerts.append(f"Skipping trial {trial_index}: missing onset or duration")
            continue
        category, blocks, warning = trial_label_from_movie_row(row, columns)
        if warning is not None:
            alerts.append(f"Skipping trial {trial_index}: {warning}")
            continue
        if category not in MOVIE_CATEGORIES:
            alerts.append(f"Skipping trial {trial_index}: unsupported movie category {category!r}")
            continue
        label = str(blocks[0]["name"])
        records.append(
            TrialRecord(
                trial_index=trial_index,
                onset=float(onset),
                duration=float(duration),
                category=category,
                label=label,
            )
        )
    return records, alerts

def build_soma_trial_records(rows: List[Dict[str, str]], columns: Sequence[str]) -> Tuple[List[TrialRecord], List[str]]:
    records: List[TrialRecord] = []
    alerts: List[str] = []
    for trial_index, row in enumerate(rows):
        onset = as_float(row.get("time"))
        duration = as_float(row.get("duration"))
        if onset is None or duration is None:
            alerts.append(f"Skipping soma trial {trial_index}: missing onset or duration")
            continue
        position_x, position_y, position_x_label, position_y_label = infer_retino_position(row, columns)
        axis_value, axis_label = infer_retino_axis(row, columns)
        if position_x is not None and position_y is not None:
            axis_value = position_x
            axis_label = position_x_label or axis_label
        label = str(row.get("stim") or row.get("stimulus") or row.get("name") or f"trial_{trial_index}")
        records.append(
            TrialRecord(
                trial_index=trial_index,
                onset=float(onset),
                duration=float(duration),
                category=None,
                label=label,
                axis_value=axis_value,
                axis_label=axis_label,
                position_x=position_x,
                position_y=position_y,
                position_x_label=position_x_label,
                position_y_label=position_y_label,
            )
        )
    return records, alerts

def baseline_correct_trace(trace: np.ndarray, t: np.ndarray, pre_window_s: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    t = np.asarray(t, dtype=float)
    if trace.shape != t.shape:
        trace = trace[: t.shape[0]]
    return trace

def trace_mean_and_sem(traces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    valid = [np.asarray(trace, dtype=float) for trace in traces if trace is not None]
    if not valid:
        return np.asarray([]), np.asarray([])
    stacked = np.vstack(valid)
    mean = np.nanmean(stacked, axis=0)
    count = np.sum(np.isfinite(stacked), axis=0)
    std = np.nanstd(stacked, axis=0)
    sem = np.divide(std, np.sqrt(np.maximum(count, 1)), out=np.full_like(std, np.nan), where=count > 0)
    return mean, sem

def trace_mean_and_std(traces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    valid = [np.asarray(trace, dtype=float) for trace in traces if trace is not None]
    if not valid:
        return np.asarray([]), np.asarray([])
    stacked = np.vstack(valid)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean, std

def window_mean(trace: np.ndarray, t: np.ndarray, start_s: Optional[float] = None, end_s: Optional[float] = None) -> float:
    trace = np.asarray(trace, dtype=float)
    t = np.asarray(t, dtype=float)
    if trace.shape != t.shape:
        if trace.size == t.size:
            trace = trace.reshape(t.shape)
        else:
            raise ValueError("Trace and time vector must have matching shapes for window averaging")
    mask = np.isfinite(trace) & np.isfinite(t)
    if start_s is not None:
        mask &= t >= float(start_s)
    if end_s is not None:
        mask &= t < float(end_s)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(trace[mask]))

def response_amplitude(trace: np.ndarray, t: np.ndarray, response_end_s: Optional[float]) -> float:
    return window_mean(trace, t, 0.0, response_end_s)

def trial_activity_means(trace: np.ndarray, t: np.ndarray, duration_s: Optional[float]) -> Tuple[float, float]:
    baseline = window_mean(trace, t, None, 0.0)
    stimulus = response_amplitude(trace, t, duration_s)
    return baseline, stimulus

def resample_trace_to_axis(source_t: np.ndarray, source_trace: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=float)
    source_trace = np.asarray(source_trace, dtype=float)
    target_t = np.asarray(target_t, dtype=float)
    if source_trace.size == 0:
        return source_trace
    if source_t.size != source_trace.size or source_t.size == 0:
        source_t = np.linspace(0.0, float(source_trace.size - 1), int(source_trace.size))
    if target_t.size == 0:
        return source_trace
    if source_t.size == target_t.size and np.allclose(source_t, target_t):
        return source_trace
    return np.interp(target_t, source_t, source_trace, left=np.nan, right=np.nan)

def select_dendrite_entries(conversion_library: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = []
    for entry in conversion_library.values():
        if entry.get("conversion_index") is None:
            continue
        if bool(entry.get("is_dendrite")):
            entries.append(entry)
    return entries

def select_soma_entries(conversion_library: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = []
    for entry in conversion_library.values():
        if entry.get("conversion_index") is None:
            continue
        if entry.get("cell_id") is not None and not bool(entry.get("is_dendrite")):
            entries.append(entry)
    return entries

def format_significance_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n.s."
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."

def paired_ttest_summary(baseline_values: Sequence[float], stimulus_values: Sequence[float]) -> Dict[str, Any]:
    baseline = np.asarray(baseline_values, dtype=float)
    stimulus = np.asarray(stimulus_values, dtype=float)
    mask = np.isfinite(baseline) & np.isfinite(stimulus)
    baseline = baseline[mask]
    stimulus = stimulus[mask]
    if baseline.size < 2 or stimulus.size < 2:
        return {
            "available": False,
            "comparison": "paired_pre_vs_stimulus",
            "statistic": float("nan"),
            "raw_pvalue": float("nan"),
            "adjusted_pvalue": float("nan"),
            "n_pairs": int(min(baseline.size, stimulus.size)),
            "significant": False,
            "star": "n.s.",
        }
    result = stats.ttest_rel(stimulus, baseline, nan_policy="omit")
    raw_pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else float("nan")
    return {
        "available": True,
        "comparison": "paired_pre_vs_stimulus",
        "statistic": float(result.statistic),
        "raw_pvalue": raw_pvalue,
        "adjusted_pvalue": raw_pvalue,
        "n_pairs": int(baseline.size),
        "significant": False,
        "star": format_significance_stars(raw_pvalue),
    }

def welch_ttest_summary(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {
            "available": False,
            "comparison": "stimulus_vs_blank",
            "statistic": float("nan"),
            "raw_pvalue": float("nan"),
            "adjusted_pvalue": float("nan"),
            "n_a": int(a.size),
            "n_b": int(b.size),
            "significant": False,
            "star": "n.s.",
        }
    result = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    raw_pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else float("nan")
    return {
        "available": True,
        "comparison": "stimulus_vs_blank",
        "statistic": float(result.statistic),
        "raw_pvalue": raw_pvalue,
        "adjusted_pvalue": raw_pvalue,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "significant": False,
        "star": format_significance_stars(raw_pvalue),
    }

def apply_bonferroni_correction(test_records: List[Dict[str, Any]]) -> int:
    valid_records = [record for record in test_records if record.get("available") and np.isfinite(as_float(record.get("raw_pvalue")))]
    valid_ids = {id(record) for record in valid_records}
    n_tests = int(len(valid_records))
    if n_tests == 0:
        for record in test_records:
            record["adjusted_pvalue"] = float("nan")
            record["significant"] = False
            record["star"] = "n.s."
        return 0
    for record in valid_records:
        raw_pvalue = float(record.get("raw_pvalue"))
        adjusted_pvalue = min(raw_pvalue * n_tests, 1.0)
        record["adjusted_pvalue"] = float(adjusted_pvalue)
        record["significant"] = bool(np.isfinite(adjusted_pvalue) and adjusted_pvalue < 0.05)
        record["star"] = format_significance_stars(adjusted_pvalue)
    for record in test_records:
        if id(record) not in valid_ids:
            record["adjusted_pvalue"] = float("nan")
            record["significant"] = False
            record["star"] = "n.s."
    return n_tests

def add_significance_bracket(ax: Any, x1: float, x2: float, y: float, text: str, color: str) -> None:
    if not text or text == "n.s.":
        return
    ax.plot([x1, x2], [y, y], color=color, linewidth=1.8, solid_capstyle="round", clip_on=False)
    y_limits = ax.get_ylim()
    y_span = float(abs(y_limits[1] - y_limits[0]) if len(y_limits) == 2 else 1.0)
    ax.text(
        float((x1 + x2) / 2.0),
        float(y + 0.03 * max(y_span, 1e-6)),
        text,
        ha="center",
        va="bottom",
        fontsize=POSTER_NOTE_SIZE,
        color=color,
        clip_on=False,
    )

def get_category_trial_values(category_summary: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.asarray(category_summary.get("paired_baseline_values", []), dtype=float)
    stimulus = np.asarray(category_summary.get("paired_stimulus_values", []), dtype=float)
    blank = np.asarray(category_summary.get("blank_reference_values", []), dtype=float)
    return baseline, stimulus, blank

def compute_movie_compartment_statistics(compartment_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    for category in MOVIE_CATEGORIES:
        category_summary = compartment_summary.get(category)
        if not category_summary:
            continue
        baseline_values = np.asarray(category_summary.get("paired_baseline_values", []), dtype=float)
        stimulus_values = np.asarray(category_summary.get("paired_stimulus_values", []), dtype=float)
        paired_test = paired_ttest_summary(baseline_values, stimulus_values)
        paired_test.update({"category": category, "compartment": category_summary.get("compartment")})
        category_summary["stats"] = {
            "paired_pre_vs_stimulus": paired_test,
            "stimulus_vs_blank": None,
        }
        tests.append(paired_test)

    blank_reference_values = np.asarray(compartment_summary.get("blank", {}).get("paired_stimulus_values", []), dtype=float)
    for category in MOVIE_CATEGORIES:
        if category == "blank":
            continue
        category_summary = compartment_summary.get(category)
        if not category_summary:
            continue
        stimulus_values = np.asarray(category_summary.get("paired_stimulus_values", []), dtype=float)
        blank_test = welch_ttest_summary(stimulus_values, blank_reference_values)
        blank_test.update({"category": category, "compartment": category_summary.get("compartment")})
        category_summary["stats"]["stimulus_vs_blank"] = blank_test
        tests.append(blank_test)

    n_tests = apply_bonferroni_correction(tests)
    for category in MOVIE_CATEGORIES:
        category_summary = compartment_summary.get(category)
        if not category_summary:
            continue
        paired = category_summary["stats"]["paired_pre_vs_stimulus"]
        paired["comparison_name"] = "paired_pre_vs_stimulus"
        paired["n_tests_corrected"] = n_tests
        blank = category_summary["stats"].get("stimulus_vs_blank")
        if blank is not None:
            blank["comparison_name"] = "stimulus_vs_blank"
            blank["n_tests_corrected"] = n_tests
    compartment_summary["stat_tests"] = tests
    compartment_summary["n_tests_corrected"] = n_tests
    return compartment_summary

def summarize_movie_session(
    exp: LoadedExperiment,
    compartment: str,
    pre_window_s: float,
) -> Dict[str, Any]:
    if exp.cut_array is None:
        return {
            "exp_id": exp.exp_id,
            "available": False,
            "alerts": ["No intertrial cut bundle loaded"],
            "categories": {},
        }

    roi_entries = select_dendrite_entries(exp.conversion_library)
    if not roi_entries:
        return {
            "exp_id": exp.exp_id,
            "available": False,
            "alerts": ["No dendrite ROIs found in conversion library"],
            "categories": {},
        }

    trial_ids_by_category: Dict[str, List[int]] = {category: [] for category in MOVIE_CATEGORIES}
    for record in exp.trial_records:
        if record.category in trial_ids_by_category:
            trial_ids_by_category[record.category].append(record.trial_index)

    categories: Dict[str, Any] = {}
    for category in MOVIE_CATEGORIES:
        trial_ids = [trial_id for trial_id in trial_ids_by_category[category] if trial_id < exp.cut_array.shape[1]]
        if not trial_ids:
            continue
        roi_means: List[np.ndarray] = []
        roi_response_values: List[float] = []
        roi_traces: List[Dict[str, Any]] = []
        paired_baseline_values: List[float] = []
        paired_stimulus_values: List[float] = []
        for roi in roi_entries:
            roi_idx = as_int(roi.get("conversion_index"))
            if roi_idx is None or roi_idx < 0 or roi_idx >= exp.cut_array.shape[0]:
                continue
            traces = []
            trial_baseline_values: List[float] = []
            trial_stimulus_values: List[float] = []
            for trial_id in trial_ids:
                trial_trace = np.asarray(exp.cut_array[roi_idx, trial_id], dtype=float)
                traces.append(trial_trace)
                trial_duration = as_float(exp.trial_rows[trial_id].get("duration")) if trial_id < len(exp.trial_rows) else None
                baseline_value, stimulus_value = trial_activity_means(trial_trace, exp.cut_t, trial_duration)
                paired_baseline_values.append(baseline_value)
                paired_stimulus_values.append(stimulus_value)
                trial_baseline_values.append(baseline_value)
                trial_stimulus_values.append(stimulus_value)
            if not traces:
                continue
            roi_mean, roi_std = trace_mean_and_std(traces)
            if roi_mean.size == 0:
                continue
            roi_means.append(roi_mean)
            roi_response_value = float(np.nanmean(trial_stimulus_values)) if trial_stimulus_values else float("nan")
            roi_response_values.append(roi_response_value)
            roi_traces.append(
                {
                    "general_roi_id": roi.get("general_roi_id"),
                    "conversion_index": roi_idx,
                    "t": np.asarray(exp.cut_t, dtype=float),
                    "trial_t": np.asarray(exp.cut_t, dtype=float),
                    "trace": roi_mean,
                    "std_trace": roi_std,
                    "response_amplitude": roi_response_value,
                    "paired_baseline_values": trial_baseline_values,
                    "paired_stimulus_values": trial_stimulus_values,
                    "trial_traces": traces,
                }
            )
        if not roi_means:
            continue
        mean_trace, sem_trace = trace_mean_and_sem(roi_means)
        _, std_trace = trace_mean_and_std(roi_means)
        stimulus_end_values = [as_float(exp.trial_rows[trial_id].get("duration")) for trial_id in trial_ids if trial_id < len(exp.trial_rows)]
        stimulus_end_values = [value for value in stimulus_end_values if value is not None]
        categories[category] = {
            "t": np.asarray(exp.cut_t, dtype=float),
            "mean_trace": mean_trace,
            "sem_trace": sem_trace,
            "std_trace": std_trace,
            "roi_traces": roi_traces,
            "paired_baseline_values": paired_baseline_values,
            "paired_stimulus_values": paired_stimulus_values,
            "blank_reference_values": [],
            "stimulus_end_s": float(np.nanmedian(stimulus_end_values)) if stimulus_end_values else float("nan"),
            "response_mean": float(np.nanmean(roi_response_values)) if roi_response_values else float("nan"),
            "n_rois": int(len(roi_means)),
            "n_trials": int(len(trial_ids)),
        }

    return {
        "exp_id": exp.exp_id,
        "animal_id": exp.animal_id,
        "date": exp.date,
        "role": exp.role,
        "exp_root": exp.exp_root,
        "trial_csv_path": exp.trial_csv_path,
        "available": bool(categories),
        "alerts": list(exp.alerts),
        "roi_count": int(sum(int(summary.get("n_rois") or 0) for summary in categories.values())),
        "trial_count": int(sum(int(summary.get("n_trials") or 0) for summary in categories.values())),
        "categories": categories,
        "used_same_day_fallback": bool(exp.used_same_day_fallback),
        "conversion_source_exp_id": exp.conversion_source_exp_id,
    }

def summarize_soma_session(
    exp: LoadedExperiment,
    pre_window_s: float,
    post_window_s: float,
) -> Dict[str, Any]:
    if exp.cut_array is None and exp.soma_trace is None:
        return {
            "exp_id": exp.exp_id,
            "available": False,
            "alerts": ["No soma trace bundle loaded"],
            "response_points": [],
            "paired_baseline_values": [],
            "paired_stimulus_values": [],
            "mean_trace": np.asarray([]),
            "std_trace": np.asarray([]),
        }

    if exp.cut_array is not None:
        soma_entries = select_soma_entries(exp.conversion_library)
        if not soma_entries:
            soma_entries = [
                {"conversion_index": index, "general_roi_id": f"cut_roi_{index}", "cell_id": None, "is_dendrite": False}
                for index in range(int(exp.cut_array.shape[0]))
            ]
        if not soma_entries:
            return {
                "exp_id": exp.exp_id,
                "available": False,
                "alerts": ["No soma ROIs found in cut bundle"],
                "response_points": [],
                "paired_baseline_values": [],
                "paired_stimulus_values": [],
                "mean_trace": np.asarray([]),
                "std_trace": np.asarray([]),
            }

        roi_mean_traces: List[np.ndarray] = []
        all_trial_traces: List[np.ndarray] = []
        trial_amplitudes_by_roi: List[List[float]] = []
        paired_baseline_values: List[float] = []
        paired_stimulus_values: List[float] = []
        for roi in soma_entries:
            roi_idx = as_int(roi.get("conversion_index"))
            if roi_idx is None or roi_idx < 0 or roi_idx >= exp.cut_array.shape[0]:
                continue
            trial_traces: List[np.ndarray] = []
            trial_amplitudes: List[float] = []
            for record in exp.trial_records:
                if record.trial_index >= exp.cut_array.shape[1]:
                    continue
                trace = np.asarray(exp.cut_array[roi_idx, record.trial_index], dtype=float)
                trial_traces.append(trace)
                all_trial_traces.append(trace)
                trial_duration = as_float(record.duration)
                baseline_value, stimulus_value = trial_activity_means(trace, exp.cut_t, trial_duration)
                paired_baseline_values.append(baseline_value)
                paired_stimulus_values.append(stimulus_value)
                trial_amplitudes.append(stimulus_value)
            if not trial_traces:
                continue
            roi_mean, _ = trace_mean_and_sem(trial_traces)
            if roi_mean.size == 0:
                continue
            roi_mean_traces.append(roi_mean)
            trial_amplitudes_by_roi.append(trial_amplitudes)

        if not roi_mean_traces:
            return {
                "exp_id": exp.exp_id,
                "available": False,
                "alerts": ["No usable soma traces were found"],
                "response_points": [],
                "paired_baseline_values": paired_baseline_values,
                "paired_stimulus_values": paired_stimulus_values,
                "mean_trace": np.asarray([]),
                "std_trace": np.asarray([]),
            }

        mean_trace, sem_trace = trace_mean_and_sem(all_trial_traces if all_trial_traces else roi_mean_traces)
        _, std_trace = trace_mean_and_std(all_trial_traces if all_trial_traces else roi_mean_traces)
        response_points: List[Dict[str, Any]] = []
        if trial_amplitudes_by_roi:
            amplitude_matrix = np.asarray(trial_amplitudes_by_roi, dtype=float)
            mean_amplitudes = np.nanmean(amplitude_matrix, axis=0)
            for index, record in enumerate(exp.trial_records):
                if index >= mean_amplitudes.size:
                    break
                response_points.append(
                    {
                        "trial_index": int(record.trial_index),
                        "axis_value": record.axis_value,
                        "axis_label": record.axis_label,
                        "position_x": record.position_x,
                        "position_y": record.position_y,
                        "position_x_label": record.position_x_label,
                        "position_y_label": record.position_y_label,
                        "response_amplitude": float(mean_amplitudes[index]),
                        "label": record.label,
                    }
                )

        return {
            "exp_id": exp.exp_id,
            "animal_id": exp.animal_id,
            "date": exp.date,
            "available": True,
            "alerts": list(exp.alerts),
            "roi_count": int(len(roi_mean_traces)),
            "trial_count": int(len(exp.trial_records)),
            "stimulus_end_s": float(np.nanmedian([record.duration for record in exp.trial_records])) if exp.trial_records else float("nan"),
            "t": np.asarray(exp.cut_t, dtype=float),
            "mean_trace": mean_trace,
            "sem_trace": sem_trace,
            "std_trace": std_trace,
            "response_points": response_points,
            "paired_baseline_values": paired_baseline_values,
            "paired_stimulus_values": paired_stimulus_values,
            "used_same_day_fallback": bool(exp.used_same_day_fallback),
            "conversion_source_exp_id": exp.conversion_source_exp_id,
            "source_kind": exp.soma_source_kind,
            "source_path": str(exp.soma_source_path) if exp.soma_source_path is not None else None,
        }

    trace_rows = normalize_continuous_traces(exp.soma_trace, exp.soma_trace_t)
    soma_entries = select_soma_entries(exp.conversion_library)
    if soma_entries and trace_rows.shape[0] > 1:
        selected_rows: List[np.ndarray] = []
        for roi in soma_entries:
            roi_idx = as_int(roi.get("conversion_index"))
            if roi_idx is None or roi_idx < 0 or roi_idx >= trace_rows.shape[0]:
                continue
            selected_rows.append(np.asarray(trace_rows[roi_idx], dtype=float))
        if selected_rows:
            trace_rows = np.vstack(selected_rows)

    roi_mean_traces: List[np.ndarray] = []
    trial_amplitudes_by_roi: List[List[float]] = []
    paired_baseline_values: List[float] = []
    paired_stimulus_values: List[float] = []
    aligned_t_ref: Optional[np.ndarray] = None
    for trace in trace_rows:
        trial_traces: List[np.ndarray] = []
        trial_amplitudes: List[float] = []
        for record in exp.trial_records:
            aligned_t, aligned_trace = extract_aligned_continuous_trace(trace, exp.soma_trace_t, record.onset, pre_window_s, post_window_s)
            if aligned_t_ref is None:
                aligned_t_ref = np.asarray(aligned_t, dtype=float)
            trial_traces.append(aligned_trace)
            baseline_value = window_mean(aligned_trace, aligned_t, None, 0.0)
            stimulus_value = response_amplitude(aligned_trace, aligned_t, record.duration)
            paired_baseline_values.append(baseline_value)
            paired_stimulus_values.append(stimulus_value)
            trial_amplitudes.append(stimulus_value)
        if not trial_traces:
            continue
        roi_mean, _ = trace_mean_and_sem(trial_traces)
        if roi_mean.size == 0:
            continue
        roi_mean_traces.append(roi_mean)
        trial_amplitudes_by_roi.append(trial_amplitudes)

    if not roi_mean_traces:
        return {
            "exp_id": exp.exp_id,
            "animal_id": exp.animal_id,
            "date": exp.date,
            "available": False,
            "alerts": ["No usable soma traces were found"],
            "response_points": [],
            "paired_baseline_values": paired_baseline_values,
            "paired_stimulus_values": paired_stimulus_values,
            "mean_trace": np.asarray([]),
            "std_trace": np.asarray([]),
        }

    mean_trace, sem_trace = trace_mean_and_sem(roi_mean_traces)
    _, std_trace = trace_mean_and_std(roi_mean_traces)
    response_points: List[Dict[str, Any]] = []
    if trial_amplitudes_by_roi:
        amplitude_matrix = np.asarray(trial_amplitudes_by_roi, dtype=float)
        mean_amplitudes = np.nanmean(amplitude_matrix, axis=0)
        for index, record in enumerate(exp.trial_records):
            if index >= mean_amplitudes.size:
                break
            response_points.append(
                {
                    "trial_index": int(record.trial_index),
                    "axis_value": record.axis_value,
                    "axis_label": record.axis_label,
                    "position_x": record.position_x,
                    "position_y": record.position_y,
                    "position_x_label": record.position_x_label,
                    "position_y_label": record.position_y_label,
                    "response_amplitude": float(mean_amplitudes[index]),
                    "label": record.label,
                }
            )

    return {
        "exp_id": exp.exp_id,
        "animal_id": exp.animal_id,
        "date": exp.date,
        "available": True,
        "alerts": list(exp.alerts),
        "roi_count": int(len(roi_mean_traces)),
        "trial_count": int(len(exp.trial_records)),
        "t": np.asarray(aligned_t_ref, dtype=float) if aligned_t_ref is not None else np.asarray([]),
        "mean_trace": mean_trace,
        "sem_trace": sem_trace,
        "std_trace": std_trace,
        "response_points": response_points,
        "paired_baseline_values": paired_baseline_values,
        "paired_stimulus_values": paired_stimulus_values,
        "used_same_day_fallback": bool(exp.used_same_day_fallback),
        "conversion_source_exp_id": exp.conversion_source_exp_id,
        "source_kind": exp.soma_source_kind,
        "source_path": str(exp.soma_source_path) if exp.soma_source_path is not None else None,
    }

def pool_movie_session_summaries(session_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_sessions = [session for session in session_summaries if session.get("available")]
    if not valid_sessions:
        return {
            "available": False,
            "alerts": ["No usable movie sessions were found"],
            "categories": {},
        }

    pooled: Dict[str, Dict[str, Any]] = {}
    for category in MOVIE_CATEGORIES:
        roi_trace_groups: Dict[str, Dict[str, Any]] = {}
        paired_baseline_values: List[float] = []
        paired_stimulus_values: List[float] = []
        blank_reference_values: List[float] = []
        n_trials = 0
        n_sessions = 0
        response_means: List[float] = []
        stimulus_end_values: List[float] = []
        for session in valid_sessions:
            category_summary = session.get("categories", {}).get(category)
            if not category_summary:
                continue
            mean_trace = np.asarray(category_summary.get("mean_trace", []), dtype=float)
            if mean_trace.size == 0:
                continue
            n_sessions += 1
            response_mean = as_float(category_summary.get("response_mean"))
            if response_mean is not None and np.isfinite(response_mean):
                response_means.append(float(response_mean))
            stimulus_end = as_float(category_summary.get("stimulus_end_s"))
            if stimulus_end is not None and np.isfinite(stimulus_end):
                stimulus_end_values.append(float(stimulus_end))
            paired_baseline_values.extend([float(value) for value in category_summary.get("paired_baseline_values", [])])
            paired_stimulus_values.extend([float(value) for value in category_summary.get("paired_stimulus_values", [])])
            if category != "blank":
                blank_reference_values.extend([float(value) for value in category_summary.get("blank_reference_values", [])])
            for trace_record in category_summary.get("roi_traces", []):
                trace = np.asarray(trace_record.get("trace", []), dtype=float)
                if trace.size == 0:
                    continue
                group_key = trace_record.get("general_roi_id")
                if group_key in (None, ""):
                    group_key = trace_record.get("conversion_index")
                group_key = str(group_key)
                group = roi_trace_groups.setdefault(
                    group_key,
                    {
                        "t": [],
                        "traces": [],
                        "response_amplitudes": [],
                        "general_roi_id": trace_record.get("general_roi_id"),
                        "conversion_index": trace_record.get("conversion_index"),
                    },
                )
                group["t"].append(np.asarray(trace_record.get("t", []), dtype=float))
                group["traces"].append(trace)
                amplitude = as_float(trace_record.get("response_amplitude"))
                if amplitude is not None and np.isfinite(amplitude):
                    group["response_amplitudes"].append(float(amplitude))
            n_trials += int(category_summary.get("n_trials") or 0)

        if not roi_trace_groups:
            continue

        all_times = [trace_t for group in roi_trace_groups.values() for trace_t in group["t"] if np.asarray(trace_t, dtype=float).size]
        ref_t = np.asarray(all_times[0], dtype=float) if all_times else np.asarray([])
        if ref_t.size == 0:
            max_len = max(int(max(np.asarray(trace, dtype=float).size for trace in group["traces"])) for group in roi_trace_groups.values())
            ref_t = np.arange(max_len, dtype=float)

        roi_traces: List[Dict[str, Any]] = []
        for group in roi_trace_groups.values():
            aligned_traces: List[np.ndarray] = []
            for source_t, source_trace in zip(group["t"], group["traces"]):
                source_t = np.asarray(source_t, dtype=float)
                source_trace = np.asarray(source_trace, dtype=float)
                if source_trace.size == 0:
                    continue
                if source_t.size != source_trace.size or source_t.size == 0:
                    source_t = np.linspace(0.0, float(source_trace.size - 1), int(source_trace.size))
                aligned_traces.append(resample_trace_to_axis(source_t, source_trace, ref_t))
            if not aligned_traces:
                continue
            roi_trace, roi_std = trace_mean_and_std(aligned_traces)
            if roi_trace.size == 0:
                continue
            response_values = np.asarray(group["response_amplitudes"], dtype=float)
            roi_traces.append(
                {
                    "t": ref_t,
                    "trace": roi_trace,
                    "std_trace": roi_std,
                    "conversion_index": group["conversion_index"],
                    "general_roi_id": group["general_roi_id"],
                    "response_amplitude": float(np.nanmean(response_values)) if response_values.size else float("nan"),
                }
            )

        mean_trace, std_trace = trace_mean_and_std([trace_record["trace"] for trace_record in roi_traces])
        category_pooled = {
            "t": ref_t,
            "mean_trace": mean_trace,
            "std_trace": std_trace,
            "roi_traces": roi_traces,
            "paired_baseline_values": paired_baseline_values,
            "paired_stimulus_values": paired_stimulus_values,
            "blank_reference_values": blank_reference_values if category != "blank" else [],
            "stimulus_end_s": float(np.nanmedian(stimulus_end_values)) if stimulus_end_values else float("nan"),
            "response_mean": float(np.nanmean(response_means)) if response_means else float("nan"),
            "n_sessions": int(n_sessions),
            "n_trials": int(n_trials),
            "n_rois": int(len(roi_traces)),
        }
        pooled[category] = category_pooled

    if "blank" in pooled:
        blank_reference_values = list(pooled["blank"].get("paired_stimulus_values", []))
        for category, summary in pooled.items():
            if category != "blank":
                summary["blank_reference_values"] = list(blank_reference_values)

    compute_movie_compartment_statistics(pooled)
    pooled_summary: Dict[str, Any] = dict(pooled)
    pooled_summary.update(
        {
            "available": bool(pooled),
            "alerts": [alert for session in session_summaries for alert in session.get("alerts", [])],
            "n_sessions": int(len(valid_sessions)),
            "n_rois": int(sum(int(summary.get("n_rois") or 0) for summary in pooled.values() if isinstance(summary, dict))),
            "n_trials": int(sum(int(summary.get("n_trials") or 0) for summary in pooled.values() if isinstance(summary, dict))),
        }
    )
    return pooled_summary

def pool_soma_session_summaries(session_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_sessions = [session for session in session_summaries if session.get("available")]
    if not valid_sessions:
        return {
            "available": False,
            "alerts": ["No usable soma sessions were found"],
            "response_points": [],
            "paired_baseline_values": [],
            "paired_stimulus_values": [],
            "mean_trace": np.asarray([]),
            "std_trace": np.asarray([]),
            "t": np.asarray([]),
            "stimulus_end_s": float("nan"),
        }
    mean_traces = [np.asarray(session.get("mean_trace", []), dtype=float) for session in valid_sessions if np.asarray(session.get("mean_trace", [])).size]
    if not mean_traces:
        return {
            "available": False,
            "alerts": ["No usable soma traces were found"],
            "response_points": [],
            "paired_baseline_values": [],
            "paired_stimulus_values": [],
            "mean_trace": np.asarray([]),
            "std_trace": np.asarray([]),
            "t": np.asarray([]),
            "stimulus_end_s": float("nan"),
        }
    mean_trace, sem_trace = trace_mean_and_sem(mean_traces)
    _, std_trace = trace_mean_and_std(mean_traces)
    response_points: List[Dict[str, Any]] = []
    paired_baseline_values: List[float] = []
    paired_stimulus_values: List[float] = []
    for session in valid_sessions:
        response_points.extend(session.get("response_points", []))
        paired_baseline_values.extend([float(value) for value in session.get("paired_baseline_values", [])])
        paired_stimulus_values.extend([float(value) for value in session.get("paired_stimulus_values", [])])
    stats = paired_ttest_summary(np.asarray(paired_baseline_values, dtype=float), np.asarray(paired_stimulus_values, dtype=float))
    stats["comparison_name"] = "paired_pre_vs_stimulus"
    stats["n_tests_corrected"] = apply_bonferroni_correction([stats])
    pooled_t = next((np.asarray(session.get("t", []), dtype=float) for session in valid_sessions if np.asarray(session.get("t", [])).size), np.asarray([]))
    return {
        "available": True,
        "alerts": [alert for session in session_summaries for alert in session.get("alerts", [])],
        "t": pooled_t,
        "mean_trace": mean_trace,
        "sem_trace": sem_trace,
        "response_points": response_points,
        "paired_baseline_values": paired_baseline_values,
        "paired_stimulus_values": paired_stimulus_values,
        "stats": {"paired_pre_vs_stimulus": stats},
        "n_sessions": int(len(valid_sessions)),
        "n_rois": int(sum(int(session.get("roi_count") or 0) for session in valid_sessions)),
        "n_trials": int(sum(int(session.get("trial_count") or 0) for session in valid_sessions)),
        "stimulus_end_s": float(np.nanmedian([as_float(session.get("stimulus_end_s")) for session in valid_sessions if as_float(session.get("stimulus_end_s")) is not None])) if any(as_float(session.get("stimulus_end_s")) is not None for session in valid_sessions) else float("nan"),
    }

def build_retinotopy_response_map(response_points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_points: List[Dict[str, Any]] = []
    for point in response_points:
        x_value = as_float(point.get("position_x"))
        y_value = as_float(point.get("position_y"))
        amplitude = as_float(point.get("response_amplitude"))
        if x_value is None or y_value is None or amplitude is None or not np.isfinite(amplitude):
            continue
        valid_points.append(
            {
                "position_x": float(x_value),
                "position_y": float(y_value),
                "response_amplitude": float(amplitude),
                "position_x_label": point.get("position_x_label"),
                "position_y_label": point.get("position_y_label"),
            }
        )
    if not valid_points:
        return {
            "available": False,
            "response_matrix": np.asarray([]),
            "response_counts": np.asarray([]),
            "x_values": np.asarray([]),
            "y_values": np.asarray([]),
            "x_label": None,
            "y_label": None,
            "valid_points": [],
        }
    x_values = np.asarray(sorted({point["position_x"] for point in valid_points}), dtype=float)
    y_values = np.asarray(sorted({point["position_y"] for point in valid_points}), dtype=float)
    x_index = {float(value): index for index, value in enumerate(x_values)}
    y_index = {float(value): index for index, value in enumerate(y_values)}
    sum_matrix = np.zeros((int(y_values.size), int(x_values.size)), dtype=float)
    count_matrix = np.zeros_like(sum_matrix, dtype=int)
    for point in valid_points:
        yi = y_index[float(point["position_y"])]
        xi = x_index[float(point["position_x"])]
        sum_matrix[yi, xi] += float(point["response_amplitude"])
        count_matrix[yi, xi] += 1
    response_matrix = np.divide(
        sum_matrix,
        count_matrix,
        out=np.full_like(sum_matrix, np.nan, dtype=float),
        where=count_matrix > 0,
    )
    x_label = next((str(point.get("position_x_label")) for point in valid_points if point.get("position_x_label")), None) or "stimulus x position"
    y_label = next((str(point.get("position_y_label")) for point in valid_points if point.get("position_y_label")), None) or "stimulus y position"
    return {
        "available": True,
        "response_matrix": response_matrix,
        "response_counts": count_matrix,
        "x_values": x_values,
        "y_values": y_values,
        "x_label": x_label,
        "y_label": y_label,
        "valid_points": valid_points,
    }

def load_experiment_bundle(
    repo_base: Path,
    exp_id: str,
    role: str,
    channel: int,
    remote_repo_base: Path = DEFAULT_REMOTE_REPO_BASE,
) -> LoadedExperiment:
    animal_id = derive_animal_id(exp_id)
    exp_root = resolve_repo_root(repo_base, animal_id, exp_id)
    remote_exp_root = resolve_repo_root(remote_repo_base, animal_id, exp_id)
    trial_csv_path = resolve_trial_csv_path(repo_base, remote_repo_base, animal_id, exp_id)
    trial_rows, trial_columns = load_trial_rows_and_columns(trial_csv_path) if trial_csv_path.exists() else ([], [])
    if role == "movie":
        trial_records, trial_alerts = build_movie_trial_records(trial_rows, trial_columns)
    else:
        trial_records, trial_alerts = build_soma_trial_records(trial_rows, trial_columns)

    cut_path, cut_kind = resolve_cut_bundle_path(exp_root, channel, include_plain_cut=(role == "soma"))
    cut_t = np.asarray([], dtype=float)
    cut_array = None
    cut_bundle = None
    cut_alerts: List[str] = []
    if cut_path is not None:
        try:
            cut_t, cut_array, cut_bundle = extract_cut_neural_bundle(cut_path)
        except Exception as exc:  # pragma: no cover - exercised only on malformed data
            cut_label = cut_kind or "intertrial"
            cut_alerts.append(f"Could not load {cut_label} cut bundle {cut_path}: {exc}")
            cut_path = None
            cut_t = np.asarray([], dtype=float)
            cut_array = None
            cut_bundle = None
    elif role != "soma":
        cut_alerts.append("No intertrial cut bundle found")

    soma_source_path: Optional[Path] = None
    soma_source_kind: Optional[str] = None
    soma_trace_t: Optional[np.ndarray] = None
    soma_trace: Optional[np.ndarray] = None
    soma_bundle: Optional[Dict[str, Any]] = None
    soma_alerts: List[str] = []
    if role == "soma":
        if cut_path is not None and cut_array is not None:
            soma_source_path = cut_path
            soma_source_kind = "cut"
            soma_trace_t = cut_t
            soma_trace = cut_array
            soma_bundle = cut_bundle
        else:
            processed_candidates = [
                exp_root / "recordings" / f"s2p_ch{channel}.pickle",
                exp_root / "recordings" / "s2p_ch0.pickle",
            ]
            for candidate in processed_candidates:
                if not candidate.exists():
                    continue
                try:
                    soma_source_path = candidate
                    soma_source_kind = "recordings_s2p"
                    soma_trace_t, soma_trace, soma_bundle = load_continuous_trace_bundle(candidate)
                    break
                except Exception as exc:  # pragma: no cover - exercised only on malformed data
                    soma_alerts.append(f"Could not load processed soma bundle {candidate}: {exc}")
                    soma_source_path = None
                    soma_source_kind = None
                    soma_trace_t = None
                    soma_trace = None
                    soma_bundle = None
            if soma_trace is None:
                rapid_ret_candidates = [
                    remote_exp_root / "rapret" / "s2pData.mat",
                ]
                for candidate in rapid_ret_candidates:
                    if not candidate.exists():
                        continue
                    try:
                        soma_source_path = candidate
                        soma_source_kind = "rapid_ret"
                        soma_trace_t, soma_trace, soma_bundle = load_continuous_trace_bundle(candidate)
                        break
                    except Exception as exc:  # pragma: no cover - exercised only on malformed data
                        soma_alerts.append(f"Could not load rapid-ret soma bundle {candidate}: {exc}")
                        soma_source_path = None
                        soma_source_kind = None
                        soma_trace_t = None
                        soma_trace = None
                        soma_bundle = None

    conversion_path, conversion_source_exp_id, used_same_day_fallback = locate_conversion_file(repo_base, animal_id, exp_id)
    conversion_mode: Optional[str] = None
    conversion_library: Dict[str, Dict[str, Any]] = {}
    conversion_alerts: List[str] = []
    if conversion_path is not None:
        conversion_mode = determine_conversion_mode(conversion_path)
        conversion_library = normalize_conversion_library(load_conversion_library(conversion_path), conversion_mode)
    else:
        conversion_alerts.append("No SpinesGUI conversion library found")

    alerts = list(trial_alerts) + cut_alerts + soma_alerts + conversion_alerts
    return LoadedExperiment(
        exp_id=exp_id,
        animal_id=animal_id,
        date=derive_date(exp_id),
        role=role,
        exp_root=exp_root,
        trial_csv_path=trial_csv_path,
        cut_path=cut_path,
        cut_t=cut_t,
        cut_array=cut_array,
        cut_bundle=cut_bundle,
        soma_source_path=soma_source_path,
        soma_source_kind=soma_source_kind,
        soma_trace_t=soma_trace_t,
        soma_trace=soma_trace,
        soma_bundle=soma_bundle,
        conversion_path=conversion_path,
        conversion_mode=conversion_mode,
        conversion_library=conversion_library,
        conversion_source_exp_id=conversion_source_exp_id,
        used_same_day_fallback=used_same_day_fallback,
        trial_rows=trial_rows,
        trial_columns=trial_columns,
        trial_records=trial_records,
        alerts=alerts,
    )

def session_summary_panel(ax: Any, t: np.ndarray, summary: Dict[str, Any], label: str, color: str) -> None:
    mean_trace = np.asarray(summary.get("mean_trace", []), dtype=float)
    sem_trace = np.asarray(summary.get("sem_trace", []), dtype=float)
    if mean_trace.size == 0:
        ax.text(0.5, 0.5, f"No data for {label}", transform=ax.transAxes, ha="center", va="center", fontsize=POSTER_NOTE_SIZE)
        ax.set_axis_off()
        return
    ax.plot(t, mean_trace, color=color, linewidth=2.0, label=label)
    if sem_trace.size == mean_trace.size:
        ax.fill_between(t, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.18, linewidth=0)
    ax.axvline(0, color="#666666", linewidth=1.0, linestyle="--")
    set_sparse_numeric_ticks(ax, axis="both", nbins=5)
    ax.set_xlabel("Time from stimulus onset (s)", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Raw dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.tick_params(labelsize=POSTER_FONT_SIZE)

def collect_movie_dendrite_onset_panels(
    session_summaries: Sequence[Dict[str, Any]],
    category_name: str,
) -> List[Dict[str, Any]]:
    panels: List[Dict[str, Any]] = []
    for session_summary in session_summaries:
        if not session_summary.get("available"):
            continue
        exp_id = str(session_summary.get("exp_id") or "session")
        category_summary = session_summary.get("categories", {}).get(category_name) or {}
        category_t = np.asarray(category_summary.get("t", []), dtype=float)
        for roi_trace in category_summary.get("roi_traces", []):
            trial_traces = [np.asarray(trace, dtype=float) for trace in roi_trace.get("trial_traces", []) if np.asarray(trace, dtype=float).size]
            if not trial_traces:
                continue
            trial_t = np.asarray(roi_trace.get("trial_t", []), dtype=float)
            ref_t = trial_t if trial_t.size else category_t
            if ref_t.size == 0:
                ref_t = np.arange(int(trial_traces[0].size), dtype=float)
            aligned_trials: List[np.ndarray] = []
            for trace in trial_traces:
                source_t = trial_t
                if source_t.size != trace.size or source_t.size == 0:
                    source_t = np.arange(int(trace.size), dtype=float)
                aligned_trials.append(resample_trace_to_axis(source_t, trace, ref_t))
            mean_trace, _ = trace_mean_and_sem(aligned_trials)
            if mean_trace.size == 0:
                continue
            roi_label = roi_trace.get("general_roi_id")
            if roi_label in (None, ""):
                roi_label = roi_trace.get("conversion_index")
            panels.append(
                {
                    "session_exp_id": exp_id,
                    "roi_label": str(roi_label),
                    "t": np.asarray(ref_t, dtype=float),
                    "trial_traces": aligned_trials,
                    "mean_trace": mean_trace,
                }
            )
    return panels

def plot_movie_compartment_onset_figure(
    output_path: Path,
    group_name: str,
    compartment_name: str,
    category_name: str,
    category_summary: Dict[str, Any],
    session_summaries: Sequence[Dict[str, Any]],
    group_meta: Dict[str, Any],
) -> List[Path]:
    if plt is None:
        return []

    def _plot_dendrite_panel(ax: Any, panel: Dict[str, Any], trial_color: str) -> None:
        t = np.asarray(panel.get("t", []), dtype=float)
        trial_traces = [np.asarray(trace, dtype=float) for trace in panel.get("trial_traces", []) if np.asarray(trace, dtype=float).size]
        mean_trace = np.asarray(panel.get("mean_trace", []), dtype=float)
        if t.size == 0 and trial_traces:
            t = np.arange(int(trial_traces[0].size), dtype=float)
        for trace in trial_traces:
            trace_t = t
            if trace_t.size != trace.size or trace_t.size == 0:
                trace_t = np.arange(int(trace.size), dtype=float)
            ax.plot(trace_t, trace, color=trial_color, linewidth=1.0, alpha=0.55, zorder=1)
        if mean_trace.size:
            if t.size != mean_trace.size or t.size == 0:
                t = np.arange(int(mean_trace.size), dtype=float)
            ax.plot(t, mean_trace, color="#000000", linewidth=2.4, zorder=3)
        ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
        ax.text(
            0.03,
            0.93,
            f"{panel.get('session_exp_id')} {panel.get('roi_label')}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=POSTER_NOTE_SIZE - 3,
            color="#444444",
        )
        set_sparse_numeric_ticks(ax, axis="both", nbins=4)
        ax.tick_params(labelsize=POSTER_FONT_SIZE - 1)
        ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
        ax.set_xlabel("Time from stimulus onset (s)", fontsize=POSTER_LABEL_SIZE)

    def _panel_output_path(base_dir: Path, base_stem: str, compartment: str, category: str, panel: Dict[str, Any]) -> Path:
        session_part = safe_filename_component(str(panel.get("session_exp_id") or "session"))
        roi_part = safe_filename_component(str(panel.get("roi_label") or "roi"))
        return base_dir / f"{base_stem}_{compartment}_{category}_movie_onset_{session_part}_{roi_part}.svg"

    compartment_color = MOVIE_CATEGORY_COLORS[compartment_name]
    trial_color = lighten_color(compartment_color, 0.58)
    dendrite_panels = collect_movie_dendrite_onset_panels(session_summaries, category_name)
    mean_trace = np.asarray(category_summary.get("mean_trace", []), dtype=float)

    if not dendrite_panels and mean_trace.size == 0:
        fig, onset_ax = plt.subplots(1, 1, figsize=(6.9, 4.5))
        onset_ax.text(0.5, 0.5, f"No data for {category_name}", transform=onset_ax.transAxes, ha="center", va="center", fontsize=POSTER_NOTE_SIZE)
        onset_ax.set_axis_off()
        fig.suptitle(
            f"{compartment_name.capitalize()} {category_name.capitalize()} onset responses",
            fontsize=POSTER_SUPTITLE_SIZE - 2,
            y=0.985,
        )
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.88)
        return save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",))

    outputs: List[Path] = []
    base_dir = output_path.parent
    base_stem = output_path.stem if output_path.suffix else output_path.name

    if dendrite_panels:
        n_panels = len(dendrite_panels)
        ncols = 2 if n_panels > 1 else 1
        nrows = int(np.ceil(n_panels / ncols))
        fig_width = 7.4 if ncols == 2 else 6.9
        fig_height = max(2.6 * nrows, 3.2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.ravel()
        for ax, panel in zip(axes_flat, dendrite_panels):
            _plot_dendrite_panel(ax, panel, trial_color)
        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()
        handles = [
            Line2D([0], [0], color="#000000", linewidth=2.4, label="Mean"),
            Line2D([0], [0], color=trial_color, linewidth=1.0, alpha=0.55, label="Trials"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=POSTER_LEGEND_SIZE, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(
            f"{compartment_name.capitalize()} {category_name.capitalize()} onset responses",
            fontsize=POSTER_SUPTITLE_SIZE - 2,
            y=0.985,
        )
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.88)
        outputs.extend(save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",)))

        for panel in dendrite_panels:
            per_panel_output = _panel_output_path(base_dir, base_stem, compartment_name, category_name, panel)
            fig_panel, ax_panel = plt.subplots(1, 1, figsize=(6.9, 4.5))
            _plot_dendrite_panel(ax_panel, panel, trial_color)
            ax_panel.set_title(
                f"{panel.get('session_exp_id')} {panel.get('roi_label')}",
                fontsize=POSTER_TITLE_SIZE - 1,
            )
            fig_panel.suptitle(
                f"{compartment_name.capitalize()} {category_name.capitalize()} dendrite onset response",
                fontsize=POSTER_SUPTITLE_SIZE - 2,
                y=0.985,
            )
            fig_panel.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=POSTER_LEGEND_SIZE, bbox_to_anchor=(0.5, 1.01))
            fig_panel.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.88)
            outputs.extend(save_figure(fig_panel, per_panel_output, dpi=DEFAULT_DPI, extra_formats=("svg",)))
    else:
        fig, onset_ax = plt.subplots(1, 1, figsize=(6.9, 4.5))
        source_t = np.asarray(category_summary.get("t", []), dtype=float)
        if source_t.size != mean_trace.size or source_t.size == 0:
            source_t = np.arange(mean_trace.size, dtype=float)
        onset_ax.plot(source_t, mean_trace, color="#000000", linewidth=2.4, zorder=3)
        onset_ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
        onset_ax.text(0.02, 0.93, f"n={int(category_summary.get('n_rois', 0) or 0)}", transform=onset_ax.transAxes, ha="left", va="top", fontsize=POSTER_NOTE_SIZE - 2, color="#444444")
        set_sparse_numeric_ticks(onset_ax, axis="both", nbins=5)
        onset_ax.tick_params(labelsize=POSTER_FONT_SIZE)
        onset_ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
        onset_ax.set_xlabel("Time from stimulus onset (s)", fontsize=POSTER_LABEL_SIZE)
        handles = [Line2D([0], [0], color="#000000", linewidth=2.4, label="Mean")]
        fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=POSTER_LEGEND_SIZE, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(
            f"{compartment_name.capitalize()} {category_name.capitalize()} onset responses",
            fontsize=POSTER_SUPTITLE_SIZE - 2,
            y=0.985,
        )
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.88)
        outputs.extend(save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",)))

    return outputs

def plot_movie_compartment_boxplot_figure(


    output_path: Path,
    group_name: str,
    compartment_name: str,
    category_name: str,
    category_summary: Dict[str, Any],
    group_meta: Dict[str, Any],
) -> List[Path]:
    if plt is None:
        return []

    compartment_color = MOVIE_CATEGORY_COLORS[compartment_name]
    fig, box_ax = plt.subplots(1, 1, figsize=(4.2, 4.5))
    baseline_values, stimulus_values, blank_values = get_category_trial_values(category_summary)
    box_records: List[Tuple[np.ndarray, float, str, str, str]] = []
    comparison_groups: List[Tuple[str, float, float]] = []
    if category_name == "blank":
        series = [
            (baseline_values, 1.0, "pre", lighten_color(compartment_color, 0.65), compartment_color),
            (stimulus_values, 2.0, "during", lighten_color(compartment_color, 0.35), compartment_color),
        ]
        comparison_groups.append(("paired", 1.0, 2.0))
    else:
        series = [
            (baseline_values, 1.0, "pre", lighten_color(compartment_color, 0.65), compartment_color),
            (stimulus_values, 2.0, "during", lighten_color(compartment_color, 0.20), compartment_color),
            (blank_values, 3.0, "blank", "#D5D5D5", "#777777"),
        ]
        comparison_groups.append(("paired", 1.0, 2.0))
        comparison_groups.append(("blank", 2.0, 3.0))
    for values, xpos, label, facecolor, edgecolor in series:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        box_records.append((values, xpos, label, facecolor, edgecolor))

    if box_records:
        data = [record[0] for record in box_records]
        positions = [record[1] for record in box_records]
        labels = [record[2] for record in box_records]
        bp = box_ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, (_, _, _, facecolor, edgecolor) in zip(bp["boxes"], box_records):
            patch.set_facecolor(facecolor)
            patch.set_edgecolor(edgecolor)
            patch.set_linewidth(1.4)
            patch.set_alpha(0.95)
        for whisker in bp["whiskers"]:
            whisker.set_color("#555555")
            whisker.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("#555555")
            cap.set_linewidth(1.0)
        for median in bp["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.5)

        y_values = np.concatenate([record[0] for record in box_records if record[0].size]) if box_records else np.asarray([])
        y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
        y_min = float(np.nanmin(y_values)) if y_values.size else 0.0
        y_span = max(y_max - y_min, 1e-6)
        bracket_y = y_max + 0.08 * y_span
        bracket_step = 0.10 * y_span
        upper_limit = bracket_y + max(len(comparison_groups) - 1, 0) * bracket_step + 0.18 * y_span
        lower_limit = y_min - 0.12 * y_span
        box_ax.set_ylim(lower_limit, upper_limit)
        for comparison_index, (comparison_kind, x1, x2) in enumerate(comparison_groups):
            stats_block = category_summary.get("stats", {})
            comparison = stats_block.get("paired_pre_vs_stimulus") if comparison_kind == "paired" else stats_block.get("stimulus_vs_blank")
            if not comparison or not comparison.get("significant"):
                continue
            add_significance_bracket(
                box_ax,
                x1,
                x2,
                bracket_y + comparison_index * bracket_step,
                comparison.get("star", "*"),
                compartment_color if comparison_kind == "paired" else "#777777",
            )

        box_ax.set_xlim(0.5, 3.5)
        box_ax.set_xticks(positions)
        box_ax.set_xticklabels(labels, fontsize=POSTER_FONT_SIZE - 2, rotation=0)
        box_ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
        box_ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
        if category_name == "blank":
            box_ax.text(0.98, 0.98, "Blank comparison not shown", transform=box_ax.transAxes, ha="right", va="top", fontsize=POSTER_NOTE_SIZE - 3, color="#666666")
        box_ax.set_axisbelow(True)
        box_ax.yaxis.grid(True, alpha=0.18)
    else:
        box_ax.set_axis_off()

    fig.suptitle(
        f"{group_name} {compartment_name} {category_name} trial distributions",
        fontsize=POSTER_SUPTITLE_SIZE - 2,
        y=0.985,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.88)
    return save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",))

def plot_movie_group_figure(
    output_path: Path,
    group_name: str,
    basal_summary: Dict[str, Any],
    apical_summary: Dict[str, Any],
    basal_session_summaries: Sequence[Dict[str, Any]],
    apical_session_summaries: Sequence[Dict[str, Any]],
    group_meta: Dict[str, Any],
) -> List[Path]:
    if plt is None:
        return []
    output_path = Path(output_path)
    base_dir = output_path.parent
    base_stem = output_path.stem if output_path.suffix else output_path.name
    outputs: List[Path] = []
    for compartment_name, compartment_summary in (("basal", basal_summary), ("apical", apical_summary)):
        session_summaries = basal_session_summaries if compartment_name == "basal" else apical_session_summaries
        for category in MOVIE_CATEGORIES:
            category_summary = compartment_summary.get(category) or {}
            if not category_summary or np.asarray(category_summary.get("mean_trace", []), dtype=float).size == 0:
                continue
            onset_output = base_dir / f"{base_stem}_{compartment_name}_{category}_movie_onset.svg"
            box_output = base_dir / f"{base_stem}_{compartment_name}_{category}_movie_boxplots.svg"
            outputs.extend(plot_movie_compartment_onset_figure(onset_output, group_name, compartment_name, category, category_summary, session_summaries, group_meta))
            outputs.extend(plot_movie_compartment_boxplot_figure(box_output, group_name, compartment_name, category, category_summary, group_meta))
    return outputs

def summarize_movie_dendrite_significance_counts(session_summaries_by_compartment: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    def _roi_key(roi_trace: Dict[str, Any]) -> str:
        key = roi_trace.get("general_roi_id")
        if key in (None, ""):
            key = roi_trace.get("conversion_index")
        return str(key)

    def _blank_roi_lookup(session_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        blank_summary = session_summary.get("categories", {}).get("blank") or {}
        return {_roi_key(roi_trace): roi_trace for roi_trace in blank_summary.get("roi_traces", [])}

    def _classify_dendrite(movie_roi: Dict[str, Any], blank_roi: Optional[Dict[str, Any]]) -> Tuple[bool, bool]:
        paired = paired_ttest_summary(movie_roi.get("paired_baseline_values", []), movie_roi.get("paired_stimulus_values", []))
        blank = None
        if blank_roi is not None:
            blank = welch_ttest_summary(movie_roi.get("paired_stimulus_values", []), blank_roi.get("paired_stimulus_values", []))
        apply_bonferroni_correction([record for record in (paired, blank) if record is not None])
        return bool(paired.get("significant")), bool(blank.get("significant")) if blank is not None else False

    counts: Dict[str, Dict[str, int]] = {
        compartment: {
            "n_dendrites": 0,
            "none": 0,
            "intertrial_only": 0,
            "blank_only": 0,
            "both": 0,
        }
        for compartment in session_summaries_by_compartment
    }

    for compartment_name, session_summaries in session_summaries_by_compartment.items():
        valid_sessions = [session for session in session_summaries if session.get("available")]
        test_records: List[Dict[str, Any]] = []
        dendrite_records: List[Dict[str, Any]] = []
        for session_summary in valid_sessions:
            movies_summary = session_summary.get("categories", {}).get("movies") or {}
            movie_rois = movies_summary.get("roi_traces", [])
            if not movie_rois:
                continue
            blank_lookup = _blank_roi_lookup(session_summary)
            for movie_roi in movie_rois:
                blank_roi = blank_lookup.get(_roi_key(movie_roi))
                paired = paired_ttest_summary(movie_roi.get("paired_baseline_values", []), movie_roi.get("paired_stimulus_values", []))
                blank = None
                if blank_roi is not None:
                    blank = welch_ttest_summary(movie_roi.get("paired_stimulus_values", []), blank_roi.get("paired_stimulus_values", []))
                paired.update({"comparison_name": "paired_pre_vs_stimulus", "compartment": compartment_name})
                test_records.append(paired)
                if blank is not None:
                    blank.update({"comparison_name": "stimulus_vs_blank", "compartment": compartment_name})
                    test_records.append(blank)
                dendrite_records.append({"paired": paired, "blank": blank})

        apply_bonferroni_correction(test_records)
        for record in dendrite_records:
            paired = record["paired"]
            blank = record["blank"]
            paired_sig = bool(paired.get("significant"))
            blank_sig = bool(blank.get("significant")) if blank is not None else False
            counts[compartment_name]["n_dendrites"] += 1
            if paired_sig and blank_sig:
                counts[compartment_name]["both"] += 1
            elif paired_sig:
                counts[compartment_name]["intertrial_only"] += 1
            elif blank_sig:
                counts[compartment_name]["blank_only"] += 1
            else:
                counts[compartment_name]["none"] += 1
    return counts


def plot_movie_significance_counts_figure(
    output_path: Path,
    group_name: str,
    session_summaries_by_compartment: Dict[str, Sequence[Dict[str, Any]]],
    group_meta: Dict[str, Any],
) -> List[Path]:
    if plt is None:
        return []
    counts = summarize_movie_dendrite_significance_counts(session_summaries_by_compartment)
    categories = ["none", "intertrial_only", "blank_only", "both"]
    labels = ["No significant", "Intertrial only", "Blank only", "Both"]
    if not any(counts.get(compartment, {}).get("n_dendrites", 0) for compartment in counts):
        return []

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 6.2))
    y_positions = np.arange(len(categories), dtype=float)
    height = 0.34
    offsets = {"basal": -height / 2.0, "apical": height / 2.0}
    colors = {"basal": MOVIE_CATEGORY_COLORS.get("basal", "#4C78A8"), "apical": MOVIE_CATEGORY_COLORS.get("apical", "#F58518")}

    for compartment_name in ("basal", "apical"):
        values = np.asarray([counts.get(compartment_name, {}).get(category, 0) for category in categories], dtype=float)
        ax.barh(
            y_positions + offsets[compartment_name],
            values,
            height=height,
            color=lighten_color(colors[compartment_name], 0.12),
            edgecolor=colors[compartment_name],
            linewidth=1.4,
            label=f"{compartment_name.capitalize()} dendrites",
        )
        for y_position, value in zip(y_positions + offsets[compartment_name], values):
            if value <= 0:
                continue
            ax.text(
                float(value + max(0.05, 0.03 * max(values.max(), 1.0))),
                float(y_position),
                f"{int(value)}",
                ha="left",
                va="center",
                fontsize=POSTER_NOTE_SIZE,
                color=colors[compartment_name],
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=POSTER_FONT_SIZE)
    ax.set_xlabel("Dendrites", fontsize=POSTER_LABEL_SIZE)
    ax.set_title("Movies dendrite significance counts", fontsize=POSTER_TITLE_SIZE - 1)
    if group_name:
        ax.text(
            0.5,
            1.02,
            str(group_meta.get("label") or group_name),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=POSTER_NOTE_SIZE,
            color="#666666",
        )
    upper = max(float(counts.get("basal", {}).get("n_dendrites", 0)), float(counts.get("apical", {}).get("n_dendrites", 0)), 1.0)
    ax.set_xlim(0.0, upper * 1.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    ax.tick_params(axis="x", labelsize=POSTER_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
    ax.legend(loc="upper right", frameon=False, fontsize=POSTER_NOTE_SIZE)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.12, top=0.85)
    return save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",))

def plot_soma_onset_figure(
    output_path: Path,
    group_name: str,
    soma_summary: Dict[str, Any],
    pre_window_s: float,
    post_window_s: float,
) -> List[Path]:
    if plt is None:
        return []
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.2))
    mean_trace = np.asarray(soma_summary.get("mean_trace", []), dtype=float)
    sem_trace = np.asarray(soma_summary.get("sem_trace", []), dtype=float)
    std_trace = np.asarray(soma_summary.get("std_trace", []), dtype=float)
    t = np.asarray(soma_summary.get("t", []), dtype=float)
    if mean_trace.size and t.size == mean_trace.size:
        pass
    elif mean_trace.size and t.size > 1:
        t = np.linspace(float(t[0]), float(t[-1]), int(mean_trace.size))
    elif mean_trace.size:
        t = np.arange(int(mean_trace.size), dtype=float)
    else:
        t = np.arange(100, dtype=float)
    stimulus_end_s = as_float(soma_summary.get("stimulus_end_s"))
    if mean_trace.size == 0:
        ax.text(0.5, 0.5, "No soma traces available", transform=ax.transAxes, ha="center", va="center", fontsize=POSTER_NOTE_SIZE)
        ax.set_axis_off()
    else:
        ax.plot(t, mean_trace, color=SOMA_TRACE_COLOR, linewidth=2.5, label="Group mean")
        fill_band = std_trace if std_trace.size == mean_trace.size else sem_trace
        if fill_band.size == mean_trace.size:
            ax.fill_between(t, mean_trace - fill_band, mean_trace + fill_band, color="#D0D0D0", alpha=0.55, linewidth=0, zorder=0)
        ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
        if stimulus_end_s is not None and np.isfinite(stimulus_end_s):
            ax.axvspan(0.0, float(stimulus_end_s), color="#D0D0D0", alpha=0.35, linewidth=0, zorder=0)
            ax.axvline(stimulus_end_s, color="#999999", linestyle=":", linewidth=1.4)
        if t.size:
            ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        ax.set_xlabel("Time from grating onset (s)", fontsize=POSTER_LABEL_SIZE)
        ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
        set_sparse_numeric_ticks(ax, axis="both", nbins=5)
        ax.tick_params(labelsize=POSTER_FONT_SIZE)
    fig.suptitle(
        "Grating onset response",
        fontsize=POSTER_SUPTITLE_SIZE - 2,
        y=0.98,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.85)
    return save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",))

def plot_soma_boxplot_figure(
    output_path: Path,
    group_name: str,
    soma_summary: Dict[str, Any],
    pre_window_s: float,
    post_window_s: float,
) -> List[Path]:
    if plt is None:
        return []
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
    paired_baseline_values = np.asarray(soma_summary.get("paired_baseline_values", []), dtype=float)
    paired_stimulus_values = np.asarray(soma_summary.get("paired_stimulus_values", []), dtype=float)
    if paired_baseline_values.size and paired_stimulus_values.size:
        bp = ax.boxplot([paired_baseline_values, paired_stimulus_values], positions=[1.0, 2.0], widths=0.55, patch_artist=True, showfliers=False)
        box_records = [
            (1.0, lighten_color(SOMA_TRACE_COLOR, 0.70), SOMA_TRACE_COLOR),
            (2.0, lighten_color(SOMA_TRACE_COLOR, 0.35), SOMA_TRACE_COLOR),
        ]
        for patch, (_, facecolor, edgecolor) in zip(bp["boxes"], box_records):
            patch.set_facecolor(facecolor)
            patch.set_edgecolor(edgecolor)
            patch.set_linewidth(1.4)
            patch.set_alpha(0.95)
        for whisker in bp["whiskers"]:
            whisker.set_color("#555555")
            whisker.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("#555555")
            cap.set_linewidth(1.0)
        for median in bp["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.5)
        y_values = np.concatenate([paired_baseline_values, paired_stimulus_values])
        y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
        y_min = float(np.nanmin(y_values)) if y_values.size else 0.0
        y_span = max(y_max - y_min, 1e-6)
        bracket_y = y_max + 0.08 * y_span
        upper_limit = bracket_y + 0.18 * y_span
        lower_limit = y_min - 0.12 * y_span
        ax.set_ylim(lower_limit, upper_limit)
        stats_block = soma_summary.get("stats", {}).get("paired_pre_vs_stimulus") or {}
        if stats_block.get("significant"):
            add_significance_bracket(ax, 1.0, 2.0, bracket_y, stats_block.get("star", "*"), SOMA_TRACE_COLOR)
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1.0, 2.0])
        ax.set_xticklabels(["pre", "during"], fontsize=POSTER_FONT_SIZE - 2, rotation=0)
        ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
        ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=0.18)
    else:
        ax.set_axis_off()
    fig.suptitle(
        "Grating trial distributions",
        fontsize=POSTER_SUPTITLE_SIZE - 2,
        y=0.98,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.85)
    return save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",))

def plot_soma_retinotopy_map_figure(
    output_path: Path,
    group_name: str,
    soma_summary: Dict[str, Any],
    pre_window_s: float,
    post_window_s: float,
) -> List[Path]:
    if plt is None:
        return []
    fig, ax2 = plt.subplots(1, 1, figsize=(4.6, 4.2))
    response_points = soma_summary.get("response_points", [])
    retino_map = build_retinotopy_response_map(response_points)
    if retino_map.get("available"):
        response_matrix = np.asarray(retino_map.get("response_matrix", []), dtype=float)
        x_values = np.asarray(retino_map.get("x_values", []), dtype=float)
        y_values = np.asarray(retino_map.get("y_values", []), dtype=float)
        if response_matrix.size == 0 or x_values.size == 0 or y_values.size == 0:
            retino_map = {"available": False}
        else:
            x_step = float(np.nanmin(np.diff(x_values))) if x_values.size > 1 else 1.0
            y_step = float(np.nanmin(np.diff(y_values))) if y_values.size > 1 else 1.0
            extent = [float(x_values.min() - x_step / 2.0), float(x_values.max() + x_step / 2.0), float(y_values.min() - y_step / 2.0), float(y_values.max() + y_step / 2.0)]
            cmap = mcolors.LinearSegmentedColormap.from_list("soma_retino", ["#F5FBF7", SOMA_RETINO_COLOR])
            im = ax2.imshow(response_matrix, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, extent=extent)
            ax2.set_xticks(x_values)
            ax2.set_yticks(y_values)
            ax2.set_xticklabels([f"{value:g}" for value in x_values], fontsize=POSTER_FONT_SIZE)
            ax2.set_yticklabels([f"{value:g}" for value in y_values], fontsize=POSTER_FONT_SIZE)
            ax2.set_xlabel(str(retino_map.get("x_label") or "stimulus x position"), fontsize=POSTER_LABEL_SIZE)
            ax2.set_ylabel(str(retino_map.get("y_label") or "stimulus y position"), fontsize=POSTER_LABEL_SIZE)
            cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("dF/F", fontsize=POSTER_NOTE_SIZE)
            cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
    if not retino_map.get("available"):
        ax2.text(0.5, 0.5, "No retinotopy map available", transform=ax2.transAxes, ha="center", va="center", fontsize=POSTER_NOTE_SIZE)
        ax2.set_axis_off()
    fig.suptitle(
        "2D retinotopy response map",
        fontsize=POSTER_SUPTITLE_SIZE - 2,
        y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.85)
    return save_figure(fig, output_path, dpi=DEFAULT_DPI, extra_formats=("svg",))

def plot_soma_group_figure(
    output_path: Path,
    group_name: str,
    soma_summary: Dict[str, Any],
    pre_window_s: float,
    post_window_s: float,
) -> List[Path]:
    if plt is None:
        return []
    output_path = Path(output_path)
    base_dir = output_path.parent
    base_stem = output_path.stem if output_path.suffix else output_path.name
    outputs: List[Path] = []
    outputs.extend(plot_soma_onset_figure(base_dir / f"{base_stem}_soma_grating_onset.svg", group_name, soma_summary, pre_window_s, post_window_s))
    outputs.extend(plot_soma_boxplot_figure(base_dir / f"{base_stem}_soma_grating_boxplots.svg", group_name, soma_summary, pre_window_s, post_window_s))
    outputs.extend(plot_soma_retinotopy_map_figure(base_dir / f"{base_stem}_soma_retinotopy_map.svg", group_name, soma_summary, pre_window_s, post_window_s))
    return outputs

def parse_soma_group_map_from_config(
config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return normalize_soma_group_map(config.get("soma_group_map"))

def validate_group_map(
    group_map: Sequence[Dict[str, Any]],
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
    soma_expids: Sequence[str],
) -> List[str]:
    alerts: List[str] = []
    basal_set = set(basal_expids)
    apical_set = set(apical_expids)
    soma_set = set(soma_expids)
    mapped_basal: set[str] = set()
    mapped_apical: set[str] = set()
    mapped_soma: set[str] = set()
    for group in group_map:
        group_basal = [exp_id for exp_id in group.get("basal_expids", [])]
        group_apical = [exp_id for exp_id in group.get("apical_expids", [])]
        group_soma = [exp_id for exp_id in group.get("soma_expids", [])]
        missing_basal = [exp_id for exp_id in group_basal if exp_id not in basal_set]
        missing_apical = [exp_id for exp_id in group_apical if exp_id not in apical_set]
        missing_soma = [exp_id for exp_id in group_soma if exp_id not in soma_set]
        if missing_basal:
            alerts.append(f"Group {group['name']} references basal expIDs that are not in basal_expids: {missing_basal}")
        if missing_apical:
            alerts.append(f"Group {group['name']} references apical expIDs that are not in apical_expids: {missing_apical}")
        if missing_soma:
            alerts.append(f"Group {group['name']} references soma expIDs that are not in soma_expids: {missing_soma}")
        mapped_basal.update(group_basal)
        mapped_apical.update(group_apical)
        mapped_soma.update(group_soma)
    missing_basal_coverage = [exp_id for exp_id in basal_expids if exp_id not in mapped_basal]
    missing_apical_coverage = [exp_id for exp_id in apical_expids if exp_id not in mapped_apical]
    missing_soma_coverage = [exp_id for exp_id in soma_expids if exp_id not in mapped_soma]
    if missing_basal_coverage:
        alerts.append(f"basal_expids not covered by soma_group_map: {missing_basal_coverage}")
    if missing_apical_coverage:
        alerts.append(f"apical_expids not covered by soma_group_map: {missing_apical_coverage}")
    if missing_soma_coverage:
        alerts.append(f"soma_expids not covered by soma_group_map: {missing_soma_coverage}")
    return alerts

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Movie dendrite and rapid-retinotopy soma visual-response workflow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to a JSON config file.")
    parser.add_argument("--user-id", default=None, help="Lab user ID used to resolve /home/<user_id>/data/Repository.")
    parser.add_argument("--repo-base", type=Path, default=None, help="Override the lab repository base.")
    parser.add_argument("--remote-repo-base", type=Path, default=None, help="Override the raw remote repository base used for trial CSV and rapid-ret fallback.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for figures and manifests.")
    parser.add_argument("--channel", type=int, default=None, help="Suite2p channel index used for cut bundles.")
    parser.add_argument("--basal-expids", nargs="*", default=None, help="Basal movie expIDs.")
    parser.add_argument("--apical-expids", nargs="*", default=None, help="Apical movie expIDs.")
    parser.add_argument("--soma-expids", nargs="*", default=None, help="Optional rapid-retinotopy soma expIDs. Leave empty for movie-only runs.")
    parser.add_argument(
        "--soma-group-map",
        default=None,
        help="JSON string or JSON file describing the explicit soma-to-dendrite pairing map.",
    )
    parser.add_argument("--pre-window-s", type=float, default=None, help="Baseline window before onset in seconds.")
    parser.add_argument("--post-window-s", type=float, default=None, help="Display window after onset in seconds.")
    return parser

def parse_group_map_argument(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(text)

def load_experiments(
    repo_base: Path,
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
    soma_expids: Sequence[str],
    channel: int,
    remote_repo_base: Path = DEFAULT_REMOTE_REPO_BASE,
) -> Tuple[Dict[str, LoadedExperiment], List[str]]:
    experiments: Dict[str, LoadedExperiment] = {}
    alerts: List[str] = []
    for exp_id, role in [(exp_id, "movie") for exp_id in basal_expids] + [(exp_id, "movie") for exp_id in apical_expids] + [(exp_id, "soma") for exp_id in soma_expids]:
        try:
            experiments[exp_id] = load_experiment_bundle(repo_base, exp_id, role, channel, remote_repo_base=remote_repo_base)
        except Exception as exc:  # pragma: no cover - exercised only on malformed repositories
            alerts.append(f"Failed to load {exp_id}: {exc}")
    return experiments, alerts

def build_group_output_dir(output_dir: Path, group_name: str) -> Path:
    return ensure_dir(output_dir / "figures" / safe_filename_component(group_name))


def render_poster_ready_figure(output_dir: Path, group_name: str, experiments: Dict[str, LoadedExperiment], group: Dict[str, Any]) -> Path:
    import analysis.deprecated.visual_response.poster_ready_visual_response as pr
    print(pr.__file__)

    poster_output_dir = ensure_dir(output_dir)
    poster_spec = pr.build_poster_spec(group_name, experiments, group)
    poster_path = poster_output_dir / f"{safe_filename_component(group_name)}_poster.svg"
    pr.render_poster_spec(poster_spec, poster_path)
    return poster_path

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    file_config = load_json_config_file(args.config if args.config and args.config.exists() else None)
    cli_config = {
        "user_id": args.user_id,
        "repo_base": str(args.repo_base) if args.repo_base is not None else None,
        "remote_repo_base": str(args.remote_repo_base) if args.remote_repo_base is not None else None,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "channel": args.channel,
        "basal_expids": args.basal_expids,
        "apical_expids": args.apical_expids,
        "soma_expids": args.soma_expids,
        "soma_group_map": parse_group_map_argument(args.soma_group_map),
        "pre_window_s": args.pre_window_s,
        "post_window_s": args.post_window_s,
    }
    config = merge_config(cli_config, file_config)

    user_id = str(config.get("user_id") or "rubencorreia")
    repo_base = Path(config.get("repo_base") or f"/home/{user_id}/data/Repository")
    remote_repo_base = Path(config.get("remote_repo_base") or DEFAULT_REMOTE_REPO_BASE)
    output_dir = Path(config.get("output_dir") or DEFAULT_OUTPUT_DIR)
    poster_output_dir = Path(config.get("poster_output_dir") or DEFAULT_POSTER_OUTPUT_DIR)
    channel = int(config.get("channel") or 0)
    basal_expids = compact_list(config.get("basal_expids"))
    apical_expids = compact_list(config.get("apical_expids"))
    soma_expids = compact_list(config.get("soma_expids"))
    soma_group_map = parse_soma_group_map_from_config(config)
    pre_window_s = float(config.get("pre_window_s") or DEFAULT_PRE_WINDOW_S)
    post_window_s = float(config.get("post_window_s") or DEFAULT_POST_WINDOW_S)

    if not basal_expids:
        raise SystemExit("basal_expids is required")
    if not apical_expids:
        raise SystemExit("apical_expids is required")

    movie_only_mode = not bool(soma_expids)
    if not movie_only_mode and not soma_group_map:
        raise SystemExit("soma_group_map is required when soma_expids are provided")
    if not movie_only_mode:
        validation_alerts = validate_group_map(soma_group_map, basal_expids, apical_expids, soma_expids)
        if validation_alerts:
            raise SystemExit("\n".join(validation_alerts))

    experiments, load_alerts = load_experiments(repo_base, basal_expids, apical_expids, soma_expids, channel, remote_repo_base=remote_repo_base)
    all_alerts = list(load_alerts)

    movie_session_summaries: Dict[str, Dict[str, Any]] = {"basal": {}, "apical": {}}
    soma_session_summaries: Dict[str, Dict[str, Any]] = {}
    for exp_id in basal_expids:
        exp = experiments.get(exp_id)
        if exp is None:
            continue
        if exp.cut_array is None:
            all_alerts.extend(exp.alerts)
            continue
        basal_session_summary = summarize_movie_session(exp, "basal", pre_window_s)
        basal_session_summary["categories"] = compute_movie_compartment_statistics(basal_session_summary.get("categories", {}))
        movie_session_summaries["basal"][exp_id] = basal_session_summary
    for exp_id in apical_expids:
        exp = experiments.get(exp_id)
        if exp is None:
            continue
        if exp.cut_array is None:
            all_alerts.extend(exp.alerts)
            continue
        apical_session_summary = summarize_movie_session(exp, "apical", pre_window_s)
        apical_session_summary["categories"] = compute_movie_compartment_statistics(apical_session_summary.get("categories", {}))
        movie_session_summaries["apical"][exp_id] = apical_session_summary
    if not movie_only_mode:
        for exp_id in soma_expids:
            exp = experiments.get(exp_id)
            if exp is None:
                continue
            if exp.cut_array is None and exp.soma_trace is None:
                all_alerts.extend(exp.alerts)
                continue
            soma_trial_durations = [as_float(record.duration) for record in exp.trial_records if as_float(record.duration) is not None]
            soma_trial_duration_s = float(np.nanmedian(soma_trial_durations)) if soma_trial_durations else float(post_window_s or 1.0)
            soma_pre_window_s = soma_trial_duration_s
            soma_post_window_s = soma_trial_duration_s * 2.0
            soma_session_summaries[exp_id] = summarize_soma_session(exp, soma_pre_window_s, soma_post_window_s)

    group_definitions = soma_group_map if not movie_only_mode else [
        {
            "name": "movie_summary",
            "label": "movie_summary",
            "soma_expids": [],
            "basal_expids": list(basal_expids),
            "apical_expids": list(apical_expids),
        }
    ]

    group_records: List[Dict[str, Any]] = []
    for group in group_definitions:
        group_name = str(group["name"])
        group_output_dir = build_group_output_dir(output_dir, group_name)
        basal_group_sessions = [movie_session_summaries["basal"][exp_id] for exp_id in group["basal_expids"] if exp_id in movie_session_summaries["basal"]]
        apical_group_sessions = [movie_session_summaries["apical"][exp_id] for exp_id in group["apical_expids"] if exp_id in movie_session_summaries["apical"]]
        soma_group_sessions = [soma_session_summaries[exp_id] for exp_id in group.get("soma_expids", []) if exp_id in soma_session_summaries] if not movie_only_mode else []

        basal_group_summary = pool_movie_session_summaries(basal_group_sessions)
        apical_group_summary = pool_movie_session_summaries(apical_group_sessions)
        soma_group_summary = pool_soma_session_summaries(soma_group_sessions) if soma_group_sessions else {"available": False, "alerts": ["No usable soma sessions were found"], "response_points": [], "paired_baseline_values": [], "paired_stimulus_values": [], "mean_trace": np.asarray([]), "std_trace": np.asarray([]), "t": np.asarray([]), "stimulus_end_s": float("nan")}

        movie_figure_paths = plot_movie_group_figure(
            group_output_dir / safe_filename_component(group_name),
            group_name,
            basal_group_summary,
            apical_group_summary,
            basal_group_sessions,
            apical_group_sessions,
            group,
        )
        significance_count_paths = plot_movie_significance_counts_figure(
            group_output_dir / f"{safe_filename_component(group_name)}_movies_significance_counts.svg",
            group_name,
            {"basal": basal_group_sessions, "apical": apical_group_sessions},
            group,
        )
        significance_summary = summarize_movie_dendrite_significance_counts({"basal": basal_group_sessions, "apical": apical_group_sessions})
        if movie_only_mode:
            soma_figure_paths: List[str] = []
            poster_ready_path: Optional[Path] = None
        else:
            soma_figure_paths = plot_soma_group_figure(
                group_output_dir / safe_filename_component(group_name),
                group_name,
                soma_group_summary,
                pre_window_s,
                post_window_s,
            )
            poster_ready_path = render_poster_ready_figure(poster_output_dir, group_name, experiments, group)

        group_records.append(
            {
                "name": group_name,
                "label": group.get("label"),
                "soma_expids": list(group.get("soma_expids", [])),
                "basal_expids": list(group.get("basal_expids", [])),
                "apical_expids": list(group.get("apical_expids", [])),
                "movie_summary": {
                    "basal": basal_group_summary,
                    "apical": apical_group_summary,
                },
                "significance_summary": significance_summary,
                "soma_summary": soma_group_summary if not movie_only_mode else None,
                "sources": {
                    "basal": {
                        exp_id: {
                            "cut_path": str(experiments[exp_id].cut_path) if experiments.get(exp_id) and experiments[exp_id].cut_path is not None else None,
                            "cut_kind": "intertrials" if experiments.get(exp_id) and experiments[exp_id].cut_path is not None else None,
                        }
                        for exp_id in group.get("basal_expids", [])
                        if exp_id in experiments
                    },
                    "apical": {
                        exp_id: {
                            "cut_path": str(experiments[exp_id].cut_path) if experiments.get(exp_id) and experiments[exp_id].cut_path is not None else None,
                            "cut_kind": "intertrials" if experiments.get(exp_id) and experiments[exp_id].cut_path is not None else None,
                        }
                        for exp_id in group.get("apical_expids", [])
                        if exp_id in experiments
                    },
                    "soma": {exp_id: {"source_kind": experiments[exp_id].soma_source_kind, "source_path": str(experiments[exp_id].soma_source_path) if experiments.get(exp_id) and experiments[exp_id].soma_source_path is not None else None} for exp_id in group.get("soma_expids", []) if exp_id in experiments},
                },
                "outputs": {
                    "movie_figures": [str(path) for path in movie_figure_paths],
                    "significance_count_figures": [str(path) for path in significance_count_paths],
                    "soma_figures": [str(path) for path in soma_figure_paths],
                    "poster_ready_figure": str(poster_ready_path) if poster_ready_path is not None else None,
                },
            }
        )

    manifest = {
        "user_id": user_id,
        "repo_base": str(repo_base),
        "remote_repo_base": str(remote_repo_base),
        "output_dir": str(output_dir),
        "poster_output_dir": str(poster_output_dir),
        "channel": channel,
        "pre_window_s": pre_window_s,
        "post_window_s": post_window_s,
        "movie_only_mode": movie_only_mode,
        "basal_expids": basal_expids,
        "apical_expids": apical_expids,
        "soma_expids": soma_expids,
        "soma_group_map": soma_group_map,
        "groups": group_records,
        "alerts": all_alerts,
    }
    write_json_file(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
