#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
import logging
import csv
import hashlib
import json
import math
import datetime
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from analysis.compartment_common import normalize_comparison_presets
from analysis.shared.comparison_preset_flow import POSTER_REQUIRED_COMPARISON_PRESETS, build_comparison_preset_batch_plan, load_comparison_preset_csv_rows
from analysis.shared.branch_tree import ANALYSIS_BASES, ANALYSIS_BRANCHES, branch_leaf_figure_root, branch_leaf_root, iter_branch_basis_leaves, scoped_branch_results
from analysis.shared.result_manifest import AnalysisJobSpec, collect_output_artifacts, write_manifest
from analysis.shared.state_utils import resolve_repo_path
from analysis.shared.roi_split import annotate_rows_with_split_group, build_roi_split_results
from analysis.shared.plots.boxplots import plot_grouped_boxplot_series
from analysis.shared.analysis_families.coincidence import annotate_spine_event_info as shared_annotate_spine_event_info
from analysis.dendrites_pipeline.analysis_families.shared_metrics import (
    DEFAULT_EVENT_DETECTION_METHOD,
    DEFAULT_VISUAL_RESPONSE_METRIC,
    EVENT_DETECTION_METHODS,
    VISUAL_RESPONSE_METRICS,
    get_active_event_detection_method,
    get_active_visual_response_metric,
    normalize_event_detection_method,
    normalize_visual_response_metric,
    set_active_event_detection_method,
    set_active_visual_response_metric,
    visual_response_metric_field,
    visual_response_metric_label,
)
from scipy import signal, stats
try:
    from statsmodels.regression.mixed_linear_model import MixedLM
except Exception:
    MixedLM = None
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception:
    ConvergenceWarning = Warning
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="Using deprecated variance components format",
    module="statsmodels.regression.mixed_linear_model",
)

logger = logging.getLogger(__name__)

STATE_SUMMARY_PAYLOAD_CACHE_KEY = "state_summary_payload_cache"
_STATE_SUMMARY_VALUES_CACHE: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
_STATE_SUMMARY_DENDRITE_VALUES_CACHE: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
_STATE_SUMMARY_Y_LIMITS_CACHE: Dict[str, Dict[str, Tuple[float, float]]] = {}
_STATE_SUMMARY_GALLERY_RESULTS_CACHE: Dict[str, Dict[str, Any]] = {}


def _state_summary_filter_key(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if values is None:
        return None
    return tuple(sorted({str(value) for value in values if str(value)}))


def _state_summary_cache_signature(cache: Dict[str, Any]) -> str:
    analysis_tables = cache.get("analysis_tables", {})
    if not isinstance(analysis_tables, dict):
        analysis_tables = {}
    return stable_hash(
        {
            "config_hash": str(cache.get("config_hash", "")),
            "analysis_unit": str(cache.get("analysis_unit", "")),
            "source_signature": str(cache.get("source_signature", "")),
            "analysis_tables_signature": analysis_cache_meta_hash(analysis_tables),
        }
    )


def _state_summary_cache_store(cache: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cache, dict):
        return {}
    store = cache.get(STATE_SUMMARY_PAYLOAD_CACHE_KEY)
    if not isinstance(store, dict):
        store = {}
        cache[STATE_SUMMARY_PAYLOAD_CACHE_KEY] = store
    return store


def _state_summary_cache_key(
    namespace: str,
    cache: Dict[str, Any],
    *,
    metric_kind: Optional[str] = None,
    state_labels: Sequence[str] = (),
    compartment_filter: Optional[str] = None,
    subject_key: str = "day_id",
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> str:
    return stable_hash(
        {
            "namespace": str(namespace),
            "cache_signature": _state_summary_cache_signature(cache),
            "metric_kind": metric_kind,
            "state_labels": [str(state_label) for state_label in state_labels],
            "compartment_filter": compartment_filter,
            "subject_key": subject_key,
            "dendrite_ids_filter": _state_summary_filter_key(dendrite_ids_filter),
            "spine_ids_filter": _state_summary_filter_key(spine_ids_filter),
        }
    )


def _state_summary_y_limits_cache_key(cache: Dict[str, Any], state_labels: Sequence[str]) -> str:
    return _state_summary_cache_key("y_limits", cache, state_labels=state_labels)


def _state_summary_gallery_cache_key(
    cache: Dict[str, Any],
    state_labels: Sequence[str],
    compartment_filter: Optional[str] = None,
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> str:
    return _state_summary_cache_key(
        "gallery",
        cache,
        state_labels=state_labels,
        compartment_filter=compartment_filter,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    )


def extract_dendrite_token(global_dendrite_id: Any) -> str:
    parts = [part for part in str(global_dendrite_id or "").split("|") if part]
    if len(parts) >= 3:
        return str(parts[2])
    if len(parts) >= 1:
        return str(parts[-1])
    return "unknown"
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if __name__ == "__main__":
    sys.modules.setdefault("dendrites_pipeline", sys.modules[__name__])
    sys.modules.setdefault("analysis.dendrites_pipeline.dendrites_pipeline", sys.modules[__name__])
# The file is intentionally grouped into: shared constants, low-level helpers,
# cache builders, analysis, demo generation, and the CLI entrypoint.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
except Exception:
    plt = None
from poster_plotting import (
    POSTER_DPI,
    POSTER_FONT_SIZE,
    POSTER_LABEL_SIZE,
    POSTER_LEGEND_SIZE,
    POSTER_NOTE_SIZE,
    POSTER_SINGLE_FIGSIZE,
    POSTER_DOUBLE_FIGSIZE,
    POSTER_DENSE_FIGSIZE,
    POSTER_WIDE_FIGSIZE,
    POSTER_SUPTITLE_SIZE,
    POSTER_TITLE_SIZE,
    configure_poster_matplotlib,
    save_figure as save_poster_figure,
    set_sparse_colorbar_ticks,
    set_sparse_numeric_ticks,
)
if plt is not None:
    configure_poster_matplotlib()
BLANK_MOVIE_PATH = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00000"
GRATING_PREFIX = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\01"
ZEBRA_PREFIX = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\02"
NORMAL_CONVERSION_FILENAME = "ROIs_normal_mode_conversion.npy"
DEND_AXON_CONVERSION_FILENAME = "ROIs_dendrite_axon_mode_conversion.npy"
DEFAULT_CHANNEL = 0
DEFAULT_SHUFFLES = 200
DEFAULT_CPU_THREAD_LIMIT = 1
DEFAULT_Locomotion_THRESHOLD_FRACTION = 3.0
DEFAULT_CACHE_NAME = "dendrites_cache.npz"
DEFAULT_ANALYSIS_TABLES_CACHE_NAME = "dendrites_cache_analysis_tables.npz"
DEFAULT_ANALYSIS_RESULTS_CACHE_NAME = "dendrites_cache_analysis_results.npz"
DEFAULT_SHARED_SHUFFLE_CACHE_NAME = "dendrites_cache_shuffle_cache.npz"
CACHE_SCHEMA_VERSION = 3
ANALYSIS_CACHE_SCHEMA_VERSION = 1
ANALYSIS_TABLE_CACHE_SCHEMA_VERSION = 3
ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION = 3
SHARED_SHUFFLE_CACHE_SCHEMA_VERSION = 2
REPORT_SIGNIFICANCE_ALPHA = 0.05
CPU_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_CPU_THREAD_LIMIT_CONTROLLER = None
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "dendrites_pipeline"
DEFAULT_CACHE_DIRNAME = "cache"
DEFAULT_CHECKPOINT_GALLERY_DIRNAME = "checkpoint_examples"
DEFAULT_REVIEW_FIGURES_DIRNAME = "review_figures"
DEFAULT_STATE_SUMMARY_FIGURES_DIRNAME = "state_summary"
DEFAULT_STATE_SUMMARY_FIGURES_SUBDIRNAME = "selected_states"
DEFAULT_VISUAL_RESPONSE_FIGURES_DIRNAME = "visual_response"
STATE_SUMMARY_DENDRITE_METRICS = {"dendrite_mean", "dendrite_event_frequency_per_min"}
STATE_SUMMARY_SPINE_METRICS = {"spine_specific_mean", "spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"}
DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME = "matrix_similarity"
DEFAULT_MIXED_MODEL_FIGURES_DIRNAME = "mixed_model"
DEFAULT_SPINE_COACTIVITY_FIGURES_DIRNAME = "spine_coactivity"
DEFAULT_CORRELATION_FIGURES_DIRNAME = "correlation_summary"
DEFAULT_DIRECT_TRIAL_TYPE_FIGURES_DIRNAME = "direct_trial_type_comparison"
DEFAULT_EVENT_EXAMPLE_FIGURES_DIRNAME = "event_examples"
DEFAULT_SHARED_FIGURES_DIRNAME = "shared_figures"
DEFAULT_REVIEW_FIGURES_DIR = ROOT_DIR / DEFAULT_REVIEW_FIGURES_DIRNAME
# These labels are reused everywhere so the cache, analysis, and plots stay aligned.
MOVIE_TRIAL_TYPES = ["blank", "grating", "zebra", "movies"]
SLEEP_STATE_LABELS = ["active_awake", "quiet_awake", "nrem", "rem"]
SPINE_COACTIVITY_ANCHOR_STATE = "quiet_awake_movies"
SPINE_COACTIVITY_QUIET_ANCHOR_SELECTION_FIELD = "quiet_awake_movies_selected"
DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD = 0.05
VISUAL_RESPONSE_MOVIE_STATE = "quiet_awake_movies"
VISUAL_RESPONSE_BLANK_STATE = "quiet_awake_blank"
DEFAULT_DENDRITE_RESPONSE_COHORT = "all"
DENDRITE_RESPONSE_COHORTS = ("all", "responsive", "nonresponsive")
LEGACY_EVENT_DETECTION_METHOD = "amplitude"

VISUAL_RESPONSE_CLASSIFIER_VERSION = 3
VISUAL_RESPONSE_CLASSIFIER_METHOD = "movie_style_blank_vs_movies"
VISUAL_RESPONSE_COVARIATE_NAME = "none"
VISUAL_RESPONSE_VISUAL_TRIAL_TYPES = ["movies", "gratings", "zebras"]
VISUAL_RESPONSE_BLANK_TRIAL_TYPE = "blank"
VISUAL_RESPONSE_VISUAL_SUFFIXES = tuple(f"_{trial_type}" for trial_type in VISUAL_RESPONSE_VISUAL_TRIAL_TYPES)
VISUAL_RESPONSE_BLANK_SUFFIX = f"_{VISUAL_RESPONSE_BLANK_TRIAL_TYPE}"
MOVIE_TRIAL_TYPE_SUFFIXES = {
    "blank": "blank",
    "grating": "gratings",
    "zebra": "zebras",
    "movies": "movies",
}

def movie_trial_type_suffix(trial_type: str) -> str:
    return MOVIE_TRIAL_TYPE_SUFFIXES.get(str(trial_type), f"{trial_type}s")


def combined_movie_state_label(sleep_label: str, trial_type: str) -> str:
    return f"{sleep_label}_{movie_trial_type_suffix(trial_type)}"

LEGACY_MOVIE_STATE_LABEL_ALIASES = {
    "quiet_awake_blank": "quiet_awake_blank",
    "active_awake_blank": "active_awake_blank",
    "nrem_blank": "nrem_blank",
    "rem_blank": "rem_blank",
    "quiet_awake_blanks": "quiet_awake_blank",
    "active_awake_blanks": "active_awake_blank",
    "nrem_blanks": "nrem_blank",
    "rem_blanks": "rem_blank",
}

STATE_FAMILY_COLORS = {
    "active": "#1f77b4",
    "quiet": "#ff7f0e",
    "nrem": "#2ca02c",
    "rem": "#d62728",
}

def canonical_state_label(state_label: Any) -> str:
    text = str(state_label).strip() if state_label is not None else ""
    return LEGACY_MOVIE_STATE_LABEL_ALIASES.get(text, text)


def state_family_label(state_label: Any) -> str:
    canonical = canonical_state_label(state_label)
    if canonical.startswith("active"):
        return "active"
    if canonical.startswith("quiet"):
        return "quiet"
    if canonical.startswith("nrem"):
        return "nrem"
    if canonical.startswith("rem"):
        return "rem"
    return canonical.split("_", 1)[0] if "_" in canonical else canonical


def state_display_label(state_label: Any) -> str:
    canonical = canonical_state_label(state_label)
    parts = [part for part in canonical.split("_") if part]
    if len(parts) >= 2 and parts[0] in {"active", "quiet"} and parts[1] == "awake":
        suffix = " ".join(parts[2:])
        return f"{parts[0]} {suffix}".strip()
    return canonical.replace("_", " ")


def state_display_color(state_label: Any) -> str:
    return STATE_FAMILY_COLORS.get(state_family_label(state_label), "#444444")

MOVIE_STATE_LABELS = [
    combined_movie_state_label(sleep_label, trial_type)
    for trial_type in MOVIE_TRIAL_TYPES
    for sleep_label in SLEEP_STATE_LABELS
]
MOVIE_TRIAL_TYPE_TO_STATE_LABELS = {
    trial_type: tuple(combined_movie_state_label(sleep_label, trial_type) for sleep_label in SLEEP_STATE_LABELS)
    for trial_type in MOVIE_TRIAL_TYPES
}
STATE_MODE_CHOICES = ["all", "quiet", "active"]
STATE_MODE_SLEEP_LABELS = {
    "all": ["quiet_awake", "active_awake", "nrem", "rem"],
    "quiet": ["quiet_awake", "nrem", "rem"],
    "active": ["active_awake"],
}
# Basal/apical comparisons start from the movie-state set and are expanded below once all labels are defined.
DEFAULT_BASAL_APICAL_STATES = list(MOVIE_STATE_LABELS)
# Quiet-state comparisons focus on the baseline-like states requested in the paper-inspired analysis.
PRIMARY_QUIET_STATES = [
    "quiet_awake",
    "quiet_awake_blank",
    "quiet_awake_movies",
    "quiet_awake_gratings",
    "quiet_awake_zebras",
    "nrem",
    "nrem_blank",
    "nrem_movies",
    "nrem_gratings",
    "nrem_zebras",
    "rem",
    "rem_blank",
    "rem_movies",
    "rem_gratings",
    "rem_zebras",
]
# Every state label the pipeline knows about is tracked for coverage/QC reporting.
ALL_REQUESTED_STATES = [
    *MOVIE_STATE_LABELS,
    *SLEEP_STATE_LABELS,
]
# Basal/apical comparisons default to all requested states, but callers can still override them.
DEFAULT_BASAL_APICAL_STATES = list(ALL_REQUESTED_STATES)
SLEEP_STATE_MAP = {
    0: "active_awake",
    1: "quiet_awake",
    2: "nrem",
    3: "rem",
}
STATE_LABEL_TO_CODE = {label: code for code, label in SLEEP_STATE_MAP.items()}
# User-editable defaults.
# This block is the quickest place to edit the run by hand when you do not
# want to pass a long CLI command. The config file has the same knobs, and
# `state_mode` / `movie_trial_types` now drive the state-comparison list.
# `compare_states` remains as a compatibility shortcut.
USER_EDITABLE_DEFAULTS = {
    "user_id": None,
    "repo_base": None,
    "movie_expids": [],
    "sleep_expids": [],
    "basal_expids": [],
    "apical_expids": [],
    "compare_states": None,
    "state_mode": None,
    "movie_trial_types": None,
    "state_comparison_states": None,
    "basal_apical_states": None,
    "dendrite_response_cohort": DEFAULT_DENDRITE_RESPONSE_COHORT,
    "spine_coactivity_anchor_state": "quiet_awake_movies",
    "spine_coactivity_abs_threshold": DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
    "source_cache_validate": True,
    "fit_spine_coactivity_mixed_model": False,
    "spine_coactivity_only": False,
    "mixed_model_only": False,
    "mixed_model_contrast_p_source": "classical",
    "plots_only": False,
    "poster_ready_only": False,
    "plots_only_include_supporting_figures": False,
    "generate_poster_ready_figures": True,
    "source_cache_rebuild": False,
    "analysis_tables_rebuild": False,
    "analysis_results_rebuild": False,
    "shared_shuffle_cache_rebuild": False,
    "demo": False,
    "channel": DEFAULT_CHANNEL,
    "shuffle_n": DEFAULT_SHUFFLES,
    "cpu_thread_limit": DEFAULT_CPU_THREAD_LIMIT,
    "locomotion_threshold": None,
    "rebuild": False,
    "cache_path": None,
    "analysis_tables_cache_path": None,
    "analysis_run_cache_path": None,
    "analysis_results_cache_path": None,
    "output_dir": str(DEFAULT_RESULTS_DIR),
    "comparison_presets": None,
    "demo_spec": None,
}
def eprint(*args: Any) -> None:
    message = " ".join(str(arg) for arg in args)
    prefix = current_step_prefix()
    if prefix and not message.startswith(prefix) and not message.startswith("[STEP"):
        message = f"{prefix} {message}"
    print(message, file=sys.stderr)
def info(*args: Any) -> None:
    print(*args)
@dataclass
class StepFrame:
    name: str
    index: Optional[int]
    total: Optional[int]
    started_at: float
_STEP_STACK: List[StepFrame] = []
def current_step_prefix() -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not _STEP_STACK:
        return f"[{timestamp}]"
    frame = _STEP_STACK[-1]
    if frame.index is not None and frame.total is not None:
        return f"[{timestamp} {frame.index}/{frame.total}]"
    if frame.index is not None:
        return f"[{timestamp} {frame.index}]"
    return f"[{timestamp}]"
def step_message(message: str) -> None:
    prefix = current_step_prefix()
    print(f"{prefix} {message}", file=sys.stderr, flush=True)
@contextmanager
def step_scope(name: str, index: Optional[int] = None, total: Optional[int] = None):
    frame = StepFrame(name=name, index=index, total=total, started_at=time.perf_counter())
    _STEP_STACK.append(frame)
    step_message(f"START {name}")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - frame.started_at
        step_message(f"FAIL {name} ({elapsed:.1f}s): {exc}")
        raise
    else:
        elapsed = time.perf_counter() - frame.started_at
        step_message(f"DONE {name} ({elapsed:.1f}s)")
    finally:
        _STEP_STACK.pop()
def step_progress(current: int, total: int, label: Optional[str] = None) -> None:
    prefix = current_step_prefix()
    detail = f"{current}/{total}"
    if label:
        detail = f"{detail} {label}"
    if prefix:
        print(f"{prefix} PROGRESS {detail}", file=sys.stderr)
    else:
        print(f"PROGRESS {detail}", file=sys.stderr)
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def figure_family_dir(root: Path, family: str, *parts: str) -> Path:
    return ensure_dir(Path(root).joinpath(family, *parts))


def figure_nested_dir(root: Path, *parts: str) -> Path:
    return ensure_dir(Path(root).joinpath(*parts))


def visual_response_figure_dir(root: Path) -> Path:
    return figure_family_dir(root, DEFAULT_VISUAL_RESPONSE_FIGURES_DIRNAME)

def cleanup_roi_detail_figures(figure_root: Optional[Path]) -> List[str]:
    if figure_root is None:
        return []
    root = Path(figure_root)
    if not root.exists():
        return []
    removed: List[str] = []
    for path in sorted(root.rglob("*_detail.*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".svg", ".svg"}:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(str(path))
    return removed
def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
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
        if text == "":
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
def cacheable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): cacheable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cacheable(v) for v in value]
    if isinstance(value, tuple):
        return [cacheable(v) for v in value]
    if isinstance(value, set):
        return [cacheable(v) for v in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value
def stable_hash(payload: Any) -> str:
    encoded = json.dumps(jsonable(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
def derive_animal_id(exp_id: str) -> str:
    parts = exp_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive animalID from expID: {exp_id}")
    return parts[2]
def derive_date(exp_id: str) -> str:
    return exp_id.split("_", 1)[0]
def resolve_repo_root(repo_base: Path, animal_id: str, exp_id: str) -> Path:
    return repo_base / animal_id / exp_id
def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)
def write_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
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
        
def safe_filename_component(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown"

def observation_compartment(
    cache: Dict[str, Any],
    exp_id: Any,
    observation: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return basal/apical compartment for a dendrite or spine observation."""
    if not isinstance(observation, dict):
        return None

    # Best case: the observation already stores the compartment.
    compartment = observation.get("compartment")
    if compartment is not None and str(compartment).strip():
        return str(compartment)

    # Fallback: look up experiment-level compartment annotations.
    exp_meta = cache.get("experiments", {}).get(str(exp_id), {}) if isinstance(cache, dict) else {}
    if isinstance(exp_meta, dict):
        for key in ("compartment", "compartment_label", "dendrite_compartment"):
            compartment = exp_meta.get(key)
            if compartment is not None and str(compartment).strip():
                return str(compartment)

    # Fallback for older cache layouts: infer from conversion/source metadata.
    for key in ("conversion_compartment", "source_compartment"):
        compartment = observation.get(key)
        if compartment is not None and str(compartment).strip():
            return str(compartment)

    return None

def state_summary_figure_dir(root: Path, state_group: Any = None) -> Path:
    """Return the output directory for state-summary figures.

    state_group is optional for compatibility with analysis_families/core.py,
    which renders separate selected-state / basal-apical state groups.
    """
    base = Path(root) / "state_summary"

    if state_group is None:
        return base

    group = str(state_group or "").strip().lower()
    if not group or group in {"all", "default", "none"}:
        return base

    safe_group = group.replace(" ", "_").replace("-", "_")
    return base / safe_group

def _output_compartments_from_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Return basal/apical compartments that are present in analysis rows."""
    present = sorted(
        {
            str(row.get("compartment"))
            for row in rows
            if isinstance(row, dict)
            and str(row.get("compartment") or "") in {"basal", "apical"}
        }
    )
    return present

def sorted_present_compartments(cache: Dict[str, Any]) -> List[str]:
    """Return basal/apical compartments present in the cache, in stable order.

    Figure rendering uses this to decide which compartment-specific outputs to
    draw.  Keep the order fixed so generated figures/reports are reproducible.
    """
    compartment_order = ("basal", "apical")
    present = set()

    def add_compartment(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip().lower()
        if text in compartment_order:
            present.add(text)

    if not isinstance(cache, dict):
        return []

    # Experiment-level metadata.
    experiments = cache.get("experiments", {})
    if isinstance(experiments, dict):
        for exp_meta in experiments.values():
            if not isinstance(exp_meta, dict):
                continue
            for key in ("compartment", "compartment_label", "dendrite_compartment"):
                add_compartment(exp_meta.get(key))

    # Top-level observation/table rows.
    for value in cache.values():
        if isinstance(value, dict):
            rows = value.values()
        elif isinstance(value, list):
            rows = value
        else:
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            add_compartment(row.get("compartment"))
            add_compartment(row.get("compartment_label"))
            add_compartment(row.get("dendrite_compartment"))
            add_compartment(row.get("conversion_compartment"))
            add_compartment(row.get("source_compartment"))

    return [compartment for compartment in compartment_order if compartment in present]
 
def gallery_compartment_title(compartment: Any) -> str:
    """Return a readable compartment title for figure-gallery sections."""
    text = str(compartment or "").strip().lower()

    if text == "basal":
        return "Basal dendrites"
    if text == "apical":
        return "Apical dendrites"
    if text == "all":
        return "All compartments"
    if text == "soma":
        return "Soma"
    if text == "bouton":
        return "Bouton"

    if not text:
        return "Unknown compartment"

    return text.replace("_", " ").title()

def matrix_similarity_output_compartments(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Return compartments to render for matrix-similarity figures."""
    return _output_compartments_from_rows(rows)


def spine_coactivity_output_compartments(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Return compartments to render for spine-coactivity figures."""
    return _output_compartments_from_rows(rows)

def format_dendrite_display_name(
    animal_id: Any,
    compartment: Any,
    global_dendrite_id: Any,
) -> str:
    """Format a compact dendrite label for figure titles and reports."""
    animal_text = str(animal_id or "unknown")
    compartment_text = str(compartment or "unknown")
    dendrite_text = str(global_dendrite_id or "unknown")

    # Keep the useful terminal part if IDs are pipe-separated.
    dendrite_short = dendrite_text.split("|")[-1]

    return f"{animal_text} {compartment_text} {dendrite_short}"

def format_report_number(value: Any, precision: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude < 1e-4 or magnitude >= 1e4:
        return f"{number:.3e}"
    text = f"{number:.{precision}f}".rstrip("0").rstrip(".")
    return text if text else "0"
def format_report_pvalue(value: Any) -> str:
    return format_report_number(value, precision=4)
def format_report_list(values: Any, max_items: int = 10) -> str:
    if values is None:
        return "none"
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return text if text else "none"
    items = [str(item) for item in values if item is not None and str(item).strip()]
    if not items:
        return "none"
    if len(items) <= max_items:
        return ", ".join(items)
    shown = ", ".join(items[:max_items])
    return f"{shown}, ... (+{len(items) - max_items} more)"
def is_significant_row(row: Dict[str, Any], alpha: float = REPORT_SIGNIFICANCE_ALPHA, p_key: str = "shuffle_p") -> bool:
    try:
        p_value = float(row.get(p_key, float("nan")))
    except Exception:
        return False
    return bool(np.isfinite(p_value) and p_value < alpha)
def selected_matrix_state_labels(results: Dict[str, Any]) -> List[str]:
    selection = results.get("analysis_state_selection", {})
    raw_states = selection.get("state_comparison_states") if isinstance(selection, dict) else None
    if isinstance(raw_states, (list, tuple)):
        labels = [canonical_state_label(state) for state in raw_states if state is not None and str(state).strip()]
        if labels:
            return labels
    return [canonical_state_label(state) for state in PRIMARY_QUIET_STATES]


def selected_direct_trial_state_labels(results: Dict[str, Any]) -> List[str]:
    direct = results.get("direct_trial_type_comparison", {})
    selection = direct.get("selection", {}) if isinstance(direct, dict) else {}
    raw_states = selection.get("state_labels") if isinstance(selection, dict) else None
    if isinstance(raw_states, (list, tuple)):
        labels = [canonical_state_label(state) for state in raw_states if state is not None and str(state).strip()]
        if labels:
            return labels
    return selected_matrix_state_labels(results)


def selected_mixed_model_state_labels(results: Dict[str, Any]) -> List[str]:
    selection = results.get("analysis_state_selection", {})
    raw_states = selection.get("state_comparison_states") if isinstance(selection, dict) else None
    if isinstance(raw_states, (list, tuple)):
        labels = [canonical_state_label(state) for state in raw_states if state is not None and str(state).strip()]
        if labels:
            return labels
    return selected_matrix_state_labels(results)


def selected_matrix_plot_state_labels(results: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [canonical_state_label(state) for state in selected_matrix_state_labels(results)]
    present = [
        state
        for state in preferred
        if any(canonical_state_label(row.get("state_a")) == state or canonical_state_label(row.get("state_b")) == state for row in rows)
    ]
    if present:
        return present
    return preferred


def selected_basal_apical_state_labels(results: Dict[str, Any]) -> List[str]:
    selection = results.get("analysis_state_selection", {})
    if isinstance(selection, dict):
        for key in ("state_comparison_states", "basal_apical_states"):
            raw_states = selection.get(key)
            if isinstance(raw_states, (list, tuple)):
                labels = [canonical_state_label(state) for state in raw_states if state is not None and str(state).strip()]
                if labels:
                    return labels
    return [canonical_state_label(state) for state in DEFAULT_BASAL_APICAL_STATES]
def sort_rows_by_shuffle_p(rows: Sequence[Dict[str, Any]], p_key: str = "shuffle_p") -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, str]:
        try:
            p_value = float(row.get(p_key, float("nan")))
        except Exception:
            p_value = float("nan")
        try:
            effect_size = float(row.get("effect_size", row.get("estimate", float("nan"))))
        except Exception:
            effect_size = float("nan")
        return (
            p_value if np.isfinite(p_value) else float("inf"),
            -abs(effect_size) if np.isfinite(effect_size) else float("inf"),
            str(row.get("comparison", row.get("contrast_name", row.get("analysis", "")))),
        )
    return sorted([dict(row) for row in rows], key=sort_key)
def write_text_report(path: Path, lines: Sequence[str]) -> None:
    with path.open("w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
def save_figure(fig: Any, path: Path, dpi: int = POSTER_DPI, extra_formats: Sequence[str] = ("svg",)) -> None:
    if plt is None:
        return
    save_poster_figure(fig, path, dpi=dpi, extra_formats=extra_formats)




def cleanup_stale_state_coverage_artifacts(output_dir: Path) -> int:
    removed = 0
    for path in Path(output_dir).rglob("state_coverage*.svg"):
        try:
            path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def flatten_state_summary_values(state_summary: Dict[str, List[float]]) -> np.ndarray:
    flattened: List[float] = []
    for values in state_summary.values():
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size:
            arr = arr[np.isfinite(arr)]
            if arr.size:
                flattened.extend(arr.tolist())
    return np.asarray(flattened, dtype=float)


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
            "star": "",
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
        "star": "",
    }


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
            "star": "",
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
        "star": "",
    }


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


def apply_bonferroni_correction(test_records: List[Dict[str, Any]]) -> int:
    valid_records = [record for record in test_records if record.get("available") and np.isfinite(as_float(record.get("raw_pvalue")))]
    valid_ids = {id(record) for record in valid_records}
    n_tests = int(len(valid_records))
    if n_tests == 0:
        for record in test_records:
            record["adjusted_pvalue"] = float("nan")
            record["significant"] = False
            record["star"] = ""
        return 0
    for record in valid_records:
        raw_pvalue = float(record.get("raw_pvalue"))
        adjusted_pvalue = min(raw_pvalue * n_tests, 1.0)
        record["adjusted_pvalue"] = float(adjusted_pvalue)
        record["significant"] = bool(np.isfinite(adjusted_pvalue) and adjusted_pvalue < REPORT_SIGNIFICANCE_ALPHA)
        record["star"] = "*" if record["significant"] else ""
    for record in test_records:
        if id(record) not in valid_ids:
            record["adjusted_pvalue"] = float("nan")
            record["significant"] = False
            record["star"] = ""
    return n_tests


def _visual_response_trial_group(state_label: Any) -> Optional[str]:
    canonical = canonical_state_label(state_label)
    if not canonical:
        return None
    if canonical == VISUAL_RESPONSE_BLANK_TRIAL_TYPE or canonical.endswith(VISUAL_RESPONSE_BLANK_SUFFIX):
        return "blank"
    if canonical in VISUAL_RESPONSE_VISUAL_TRIAL_TYPES or any(canonical.endswith(suffix) for suffix in VISUAL_RESPONSE_VISUAL_SUFFIXES):
        return "visual"
    return None


def _visual_response_trial_labels_from_meta(trial_meta: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    visual_labels: List[str] = []
    blank_labels: List[str] = []
    for meta in trial_meta:
        if not isinstance(meta, dict):
            continue
        label = canonical_state_label(meta.get("state_label"))
        group = _visual_response_trial_group(label)
        if group == "visual" and label not in visual_labels:
            visual_labels.append(label)
        elif group == "blank" and label not in blank_labels:
            blank_labels.append(label)
    return visual_labels, blank_labels


def _resolve_visual_response_source_exp_id(
    source_cache: Optional[Dict[str, Any]],
    exp_id: str,
    observation: Optional[Dict[str, Any]] = None,
) -> str:
    if not isinstance(source_cache, dict):
        return exp_id
    experiments = source_cache.get("experiments", {})
    if exp_id in experiments:
        return exp_id
    if isinstance(observation, dict):
        for candidate in (
            observation.get("representative_exp_id"),
            *(observation.get("source_exp_ids", []) or []),
        ):
            candidate = str(candidate or "")
            if candidate and candidate in experiments:
                return candidate
    return exp_id


def _load_visual_response_cut_data(
    source_cache: Optional[Dict[str, Any]],
    exp_id: str,
    cut_cache: Dict[str, Optional[Dict[str, Any]]],
    observation: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if exp_id in cut_cache:
        return cut_cache[exp_id]
    if not isinstance(source_cache, dict):
        cut_cache[exp_id] = None
        return None
    resolved_exp_id = _resolve_visual_response_source_exp_id(source_cache, exp_id, observation)
    exp_meta = source_cache.get("experiments", {}).get(resolved_exp_id, {})
    source_paths = exp_meta.get("source_paths", {}) if isinstance(exp_meta, dict) else {}
    exp_root = Path(str(source_paths.get("exp_root") or ""))
    cut_dir = Path(str(source_paths.get("cut") or (exp_root / "cut" if exp_root else "")))
    if not cut_dir.exists():
        cut_cache[exp_id] = None
        return None
    channel = as_int(source_cache.get("config", {}).get("channel")) if isinstance(source_cache.get("config", {}), dict) else None
    channel = int(channel) if channel is not None else DEFAULT_CHANNEL
    selected_path = None
    selected_label = None
    for candidate_path, candidate_label in (
        (exp_root / "cut_intertrials" / f"s2p_ch{channel}_dF_cut.pickle", "cut_intertrials"),
        (exp_root / "cut_with_intertrials" / f"s2p_ch{channel}_dF_cut.pickle", "cut_with_intertrials"),
    ):
        if candidate_path.exists():
            selected_path = candidate_path
            selected_label = candidate_label
            break
    if selected_path is None:
        eprint(
            f"[ALERT] {exp_id}: missing cut_intertrials/cut_with_intertrials bundle at "
            f"{exp_root / 'cut_intertrials' / f's2p_ch{channel}_dF_cut.pickle'}"
        )
        cut_cache[exp_id] = None
        return None
    wheel_path = cut_dir / "wheel.pickle"
    if not wheel_path.exists():
        cut_cache[exp_id] = None
        return None
    try:
        cut_time, cut_neural, _ = extract_cut_neural_bundle(selected_path)
    except Exception:
        cut_cache[exp_id] = None
        return None
    try:
        wheel_bundle = read_pickle(wheel_path)
    except Exception:
        wheel_bundle = {}
    wheel_speed = find_first_key(wheel_bundle, ["speed", "wheel", "motion", "velocity"])
    if wheel_speed is None:
        cut_cache[exp_id] = None
        return None
    trial_meta = [dict(meta) for meta in exp_meta.get("trial_meta", []) if isinstance(meta, dict)]
    visual_trial_labels, blank_trial_labels = _visual_response_trial_labels_from_meta(trial_meta)
    payload = {
        "cut_time": np.asarray(cut_time, dtype=float),
        "cut_neural": np.asarray(cut_neural, dtype=float),
        "cut_wheel": np.asarray(wheel_speed, dtype=float),
        "trial_meta": trial_meta,
        "source_label": selected_label,
        "source_path": str(selected_path),
        "source_exp_id": resolved_exp_id,
    }
    cut_cache[exp_id] = payload
    return payload


def _visual_response_entity_observation(
    cache: Dict[str, Any],
    kind: str,
    response_row: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    animal_id = str(response_row.get("animal_id") or "")
    if not animal_id:
        return None, None, None, None
    animals = cache.get("animals", {}) if isinstance(cache, dict) else {}
    animal_entry = animals.get(animal_id, {}) if isinstance(animals, dict) else {}
    if kind == "dendrite":
        entity_id = str(response_row.get("global_dendrite_id") or "")
        dendrite_record = animal_entry.get("dendrites", {}).get(entity_id) if isinstance(animal_entry, dict) else None
        if not isinstance(dendrite_record, dict):
            return None, None, None, None
        observations = dendrite_record.get("observations", {}) if isinstance(dendrite_record, dict) else {}
        if not isinstance(observations, dict) or not observations:
            return None, None, dendrite_record, None
        exp_id, observation = next(iter(sorted(observations.items())))
        return str(exp_id), observation if isinstance(observation, dict) else None, dendrite_record, None
    entity_id = str(response_row.get("global_spine_id") or "")
    dendrites = animal_entry.get("dendrites", {}) if isinstance(animal_entry, dict) else {}
    if not isinstance(dendrites, dict):
        return None, None, None, None
    for dendrite_record in dendrites.values():
        if not isinstance(dendrite_record, dict):
            continue
        spine_record = dendrite_record.get("spines", {}).get(entity_id) if isinstance(dendrite_record.get("spines", {}), dict) else None
        if not isinstance(spine_record, dict):
            continue
        observations = spine_record.get("observations", {}) if isinstance(spine_record, dict) else {}
        if not isinstance(observations, dict) or not observations:
            return None, None, dendrite_record, None
        exp_id, observation = next(iter(sorted(observations.items())))
        parent_observation = dendrite_record.get("observations", {}).get(exp_id) if isinstance(dendrite_record.get("observations", {}), dict) else None
        return str(exp_id), observation if isinstance(observation, dict) else None, dendrite_record, parent_observation if isinstance(parent_observation, dict) else None
    return None, None, None, None


def _visual_response_trial_metric_values(
    trial_trace: np.ndarray,
    cut_time: np.ndarray,
    duration_s: Optional[float],
    *,
    response_metric: Optional[str] = None,
    event_detection_method: Optional[str] = None,
) -> Tuple[float, float]:
    metric = get_active_visual_response_metric(response_metric)
    if metric == "mean":
        return trial_activity_means(trial_trace, cut_time, duration_s)
    event_method = get_active_event_detection_method(event_detection_method)
    trial_trace = np.asarray(trial_trace, dtype=float)
    cut_time = np.asarray(cut_time, dtype=float)
    baseline_mask = np.isfinite(trial_trace) & np.isfinite(cut_time) & (cut_time < 0)
    stimulus_mask = np.isfinite(trial_trace) & np.isfinite(cut_time) & (cut_time >= 0)
    if duration_s is not None and np.isfinite(duration_s):
        stimulus_mask &= cut_time < float(duration_s)
    baseline_info = build_event_info(trial_trace[baseline_mask], cut_time[baseline_mask], method=event_method, include_all_methods=False)
    stimulus_info = build_event_info(trial_trace[stimulus_mask], cut_time[stimulus_mask], method=event_method, include_all_methods=False)
    return (
        float(as_float(baseline_info.get("event_frequency_per_min")) if as_float(baseline_info.get("event_frequency_per_min")) is not None else float("nan")),
        float(as_float(stimulus_info.get("event_frequency_per_min")) if as_float(stimulus_info.get("event_frequency_per_min")) is not None else float("nan")),
    )


def _collect_visual_response_trial_rows(
    source_cache: Optional[Dict[str, Any]],
    exp_id: str,
    observation: Dict[str, Any],
    *,
    kind: str,
    cut_cache: Dict[str, Optional[Dict[str, Any]]],
    parent_dendrite_observation: Optional[Dict[str, Any]] = None,
    response_metric: Optional[str] = None,
    event_detection_method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cut_data = _load_visual_response_cut_data(source_cache, exp_id, cut_cache, observation)
    if not cut_data:
        return []
    roi_index = as_int(observation.get("local_ids", {}).get("conversion_index"))
    if roi_index is None:
        return []
    cut_neural = np.asarray(cut_data.get("cut_neural"), dtype=float)
    cut_time = np.asarray(cut_data.get("cut_time"), dtype=float)
    trial_meta = cut_data.get("trial_meta", []) if isinstance(cut_data, dict) else []
    if roi_index < 0 or roi_index >= cut_neural.shape[0] or cut_time.size == 0:
        return []
    if kind == "spine":
        if parent_dendrite_observation is None:
            return []
        dendrite_index = as_int(parent_dendrite_observation.get("local_ids", {}).get("conversion_index"))
        alpha = as_float(observation.get("alpha"))
        if dendrite_index is None or alpha is None:
            return []
        if dendrite_index < 0 or dendrite_index >= cut_neural.shape[0]:
            return []
        trial_matrix = np.asarray(cut_neural[roi_index] - alpha * cut_neural[dendrite_index], dtype=float)
    else:
        trial_matrix = np.asarray(cut_neural[roi_index], dtype=float)
    response_metric_name = get_active_visual_response_metric(response_metric)
    rows: List[Dict[str, Any]] = []
    for meta in trial_meta:
        if not isinstance(meta, dict):
            continue
        trial_label = canonical_state_label(meta.get("state_label"))
        trial_index = as_int(meta.get("trial_index"))
        if trial_index is None or trial_index < 0 or trial_index >= trial_matrix.shape[0]:
            continue
        group = _visual_response_trial_group(trial_label)
        if group is None:
            continue
        trial_trace = np.asarray(trial_matrix[trial_index], dtype=float)
        trial_duration = as_float(meta.get("duration"))
        baseline, stimulus = _visual_response_trial_metric_values(
            trial_trace,
            cut_time,
            trial_duration,
            response_metric=response_metric_name,
            event_detection_method=event_detection_method,
        )
        if not np.isfinite(stimulus):
            continue
        if group == "visual" and not np.isfinite(baseline):
            baseline = float("nan")
        rows.append(
            {
                "group": group,
                "trial_label": trial_label,
                "response": float(stimulus),
                "baseline": float(baseline),
                "response_metric": response_metric_name,
                "source_path": str(cut_data.get("source_path") or ""),
                "mode": str(observation.get("mode") or observation.get("state_mode") or "movie"),
                "roi_index": int(roi_index),
                "exp_id": str(exp_id),
            }
        )
    return rows


def _movie_style_blank_vs_movies_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    response_metric = get_active_visual_response_metric(rows[0].get("response_metric") if rows else None)
    visual_trial_labels: List[str] = []
    blank_trial_labels: List[str] = []
    visual_values: List[float] = []
    blank_reference_values: List[float] = []
    for row in rows:
        trial_label = canonical_state_label(row.get("trial_label"))
        group = str(row.get("group") or "")
        stimulus_value = as_float(row.get("response"))
        if not np.isfinite(stimulus_value) if stimulus_value is not None else True:
            continue
        if group == "visual":
            if trial_label and trial_label not in visual_trial_labels:
                visual_trial_labels.append(trial_label)
            visual_values.append(float(stimulus_value))
        elif group == "blank":
            if trial_label and trial_label not in blank_trial_labels:
                blank_trial_labels.append(trial_label)
            blank_reference_values.append(float(stimulus_value))
    visual_arr = np.asarray(visual_values, dtype=float)
    blank_arr = np.asarray(blank_reference_values, dtype=float)
    mean_visual = float(np.nanmean(visual_arr)) if visual_arr.size else float("nan")
    mean_blank = float(np.nanmean(blank_arr)) if blank_arr.size else float("nan")
    delta = mean_visual - mean_blank if visual_arr.size and blank_arr.size else float("nan")
    blank = welch_ttest_summary(visual_arr, blank_arr)
    if blank.get("available"):
        apply_bonferroni_correction([blank])
    return {
        "available": bool(visual_arr.size and blank_arr.size),
        "comparison": "visual_response_movie_vs_blank",
        "test_name": "movie_style_blank_vs_movies",
        "covariate": VISUAL_RESPONSE_COVARIATE_NAME,
        "response_metric": response_metric,
        "statistic": float(blank.get("statistic", float("nan"))),
        "raw_pvalue": float(blank.get("raw_pvalue", float("nan"))),
        "adjusted_pvalue": float(blank.get("adjusted_pvalue", float("nan"))),
        "n_visual_values": int(visual_arr.size),
        "n_blank_values": int(blank_arr.size),
        "mean_visual": mean_visual,
        "mean_blank": mean_blank,
        "delta": delta,
        "paired_baseline_values": [],
        "paired_stimulus_values": visual_values,
        "blank_reference_values": blank_reference_values,
        "visual_trial_labels": visual_trial_labels,
        "blank_trial_labels": blank_trial_labels,
        "paired_pre_vs_stimulus": {"available": False},
        "stimulus_vs_blank": blank,
        "significant": bool(blank.get("significant", False)),
        "star": str(blank.get("star", "")),
    }


def _classify_visual_response_rows(
    cache: Dict[str, Any],
    source_cache: Optional[Dict[str, Any]],
    *,
    kind: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    cut_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for animal_id, animal_entry in sorted(cache.get("animals", {}).items()):
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            compartments_seen: List[str] = []
            observation_rows: List[Dict[str, Any]] = []
            if kind == "dendrite":
                for exp_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                    compartment = observation_compartment(cache, exp_id, d_obs)
                    if compartment is not None:
                        compartments_seen.append(str(compartment))
                    observation_rows.extend(
                        _collect_visual_response_trial_rows(
                            source_cache,
                            exp_id,
                            d_obs,
                            kind="dendrite",
                            cut_cache=cut_cache,
                        )
                    )
            else:
                for global_spine_id, spine_record in sorted(dendrite_record.get("spines", {}).items()):
                    spine_rows: List[Dict[str, Any]] = []
                    for exp_id, s_obs in sorted(spine_record.get("observations", {}).items()):
                        compartment = observation_compartment(cache, exp_id, s_obs)
                        if compartment is not None:
                            compartments_seen.append(str(compartment))
                        parent_dendrite_observation = dendrite_record.get("observations", {}).get(exp_id)
                        if parent_dendrite_observation is None:
                            continue
                        spine_rows.extend(
                            _collect_visual_response_trial_rows(
                                source_cache,
                                exp_id,
                                s_obs,
                                kind="spine",
                                cut_cache=cut_cache,
                                parent_dendrite_observation=parent_dendrite_observation,
                            )
                        )
                    summary = _movie_style_blank_vs_movies_summary(spine_rows)
                    if compartments_seen:
                        unique_compartments = list(dict.fromkeys(compartments_seen))
                        compartment = unique_compartments[0] if len(unique_compartments) == 1 else "mixed"
                    else:
                        compartment = None
                    row = {
                        "animal_id": animal_id,
                        "global_spine_id": global_spine_id,
                        "compartment": compartment,
                        "classifier_type": "spine",
                        "comparison": summary.get("comparison", "visual_response_movie_vs_blank_ancova"),
                        "test_name": summary.get("test_name", "ancova_ols"),
                        "covariate": summary.get("covariate", VISUAL_RESPONSE_COVARIATE_NAME),
                        "response_metric": summary.get("response_metric", get_active_visual_response_metric()),
                        "visual_trial_labels": list(dict.fromkeys(summary.get("visual_trial_labels", []))) if isinstance(summary.get("visual_trial_labels"), list) else [],
                        "blank_trial_labels": list(dict.fromkeys(summary.get("blank_trial_labels", []))) if isinstance(summary.get("blank_trial_labels"), list) else [],
                        "available": bool(summary.get("available", False)),
                        "n_visual_values": int(summary.get("n_visual_values", 0)),
                        "n_blank_values": int(summary.get("n_blank_values", 0)),
                        "mean_visual": float(summary.get("mean_visual", float("nan"))),
                        "mean_blank": float(summary.get("mean_blank", float("nan"))),
                        "paired_baseline_values": list(summary.get("paired_baseline_values", [])) if isinstance(summary.get("paired_baseline_values"), list) else [],
                        "paired_stimulus_values": list(summary.get("paired_stimulus_values", [])) if isinstance(summary.get("paired_stimulus_values"), list) else [],
                        "blank_reference_values": list(summary.get("blank_reference_values", [])) if isinstance(summary.get("blank_reference_values"), list) else [],
                        "delta": float(summary.get("delta", float("nan"))),
                        "effect_size": float(summary.get("delta", float("nan"))),
                        "statistic": float(summary.get("statistic", float("nan"))),
                        "raw_pvalue": float(summary.get("raw_pvalue", float("nan"))),
                        "adjusted_pvalue": float(summary.get("adjusted_pvalue", float("nan"))),
                        "significant": bool(summary.get("significant", False)),
                        "responsive": False,
                        "cohort": "nonresponsive",
                    }
                    rows.append(row)
                continue
            if compartments_seen:
                unique_compartments = list(dict.fromkeys(compartments_seen))
                compartment = unique_compartments[0] if len(unique_compartments) == 1 else "mixed"
            else:
                compartment = None
            summary = _movie_style_blank_vs_movies_summary(observation_rows)
            row = {
                "animal_id": animal_id,
                "global_dendrite_id": global_dendrite_id,
                "compartment": compartment,
                "classifier_type": "dendrite",
                "comparison": summary.get("comparison", "visual_response_movie_vs_blank_ancova"),
                "test_name": summary.get("test_name", "ancova_ols"),
                "covariate": summary.get("covariate", VISUAL_RESPONSE_COVARIATE_NAME),
                "response_metric": summary.get("response_metric", get_active_visual_response_metric()),
                "visual_trial_labels": list(dict.fromkeys(summary.get("visual_trial_labels", []))) if isinstance(summary.get("visual_trial_labels"), list) else [],
                "blank_trial_labels": list(dict.fromkeys(summary.get("blank_trial_labels", []))) if isinstance(summary.get("blank_trial_labels"), list) else [],
                "available": bool(summary.get("available", False)),
                "n_visual_values": int(summary.get("n_visual_values", 0)),
                "n_blank_values": int(summary.get("n_blank_values", 0)),
                "mean_visual": float(summary.get("mean_visual", float("nan"))),
                "mean_blank": float(summary.get("mean_blank", float("nan"))),
                "paired_baseline_values": list(summary.get("paired_baseline_values", [])) if isinstance(summary.get("paired_baseline_values"), list) else [],
                "paired_stimulus_values": list(summary.get("paired_stimulus_values", [])) if isinstance(summary.get("paired_stimulus_values"), list) else [],
                "blank_reference_values": list(summary.get("blank_reference_values", [])) if isinstance(summary.get("blank_reference_values"), list) else [],
                "delta": float(summary.get("delta", float("nan"))),
                "effect_size": float(summary.get("delta", float("nan"))),
                "statistic": float(summary.get("statistic", float("nan"))),
                "raw_pvalue": float(summary.get("raw_pvalue", float("nan"))),
                "adjusted_pvalue": float(summary.get("adjusted_pvalue", float("nan"))),
                "significant": bool(summary.get("significant", False)),
                "responsive": False,
                "cohort": "nonresponsive",
            }
            rows.append(row)
    by_compartment: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment in {"basal", "apical"}:
            by_compartment[compartment].append(row)
    for compartment_rows in by_compartment.values():
        n_tests = apply_bonferroni_correction(compartment_rows)
        for row in compartment_rows:
            row["n_tests_corrected"] = int(n_tests)
            row["responsive"] = bool(row.get("significant", False) and np.isfinite(as_float(row.get("delta"))) and float(row.get("delta")) > 0)
            row["cohort"] = "responsive" if row["responsive"] else "nonresponsive"
    for row in rows:
        if "n_tests_corrected" not in row:
            row["n_tests_corrected"] = 0
            row["responsive"] = False
            row["cohort"] = "nonresponsive"
    cohort_ids: Dict[str, Dict[str, List[str]]] = {
        "basal": {"responsive": [], "nonresponsive": []},
        "apical": {"responsive": [], "nonresponsive": []},
    }
    counts: Dict[str, Dict[str, int]] = {
        "basal": {"responsive": 0, "nonresponsive": 0, "available": 0, "tested": 0},
        "apical": {"responsive": 0, "nonresponsive": 0, "available": 0, "tested": 0},
    }
    id_field = "global_spine_id" if kind == "spine" else "global_dendrite_id"
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment not in cohort_ids:
            continue
        counts[compartment]["tested"] += 1
        if row.get("available"):
            counts[compartment]["available"] += 1
        cohort = "responsive" if row.get("responsive") else "nonresponsive"
        cohort_ids[compartment][cohort].append(str(row.get(id_field)))
        counts[compartment][cohort] += 1
    for compartment in cohort_ids:
        for cohort in cohort_ids[compartment]:
            cohort_ids[compartment][cohort] = sorted(dict.fromkeys(cohort_ids[compartment][cohort]))
    return {
        "classifier_type": kind,
        "method": VISUAL_RESPONSE_CLASSIFIER_METHOD,
        "version": VISUAL_RESPONSE_CLASSIFIER_VERSION,
        "covariate": VISUAL_RESPONSE_COVARIATE_NAME,
        "response_metric": get_active_visual_response_metric(),
        "visual_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "rows": rows,
        "cohort_ids": cohort_ids,
        "counts": counts,
    }


def classify_visual_responsive_dendrites(
    cache: Dict[str, Any],
    source_cache: Optional[Dict[str, Any]] = None,
    movie_state: str = VISUAL_RESPONSE_MOVIE_STATE,
    blank_state: str = VISUAL_RESPONSE_BLANK_STATE,
) -> Dict[str, Any]:
    if source_cache is not None:
        return _classify_visual_response_rows(cache, source_cache, kind="dendrite")
    rows: List[Dict[str, Any]] = []
    for animal_id, animal_entry in sorted(cache.get("animals", {}).items()):
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            movie_values: List[float] = []
            blank_values: List[float] = []
            compartments_seen: List[str] = []
            for exp_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                compartment = observation_compartment(cache, exp_id, d_obs)
                if compartment is not None:
                    compartments_seen.append(str(compartment))
                cut_means = d_obs.get("cut_state_means")
                if not isinstance(cut_means, dict):
                    continue
                movie_value = as_float(cut_means.get(movie_state))
                blank_value = as_float(cut_means.get(blank_state))
                if movie_value is None or blank_value is None:
                    continue
                if not np.isfinite(movie_value) or not np.isfinite(blank_value):
                    continue
                movie_values.append(float(movie_value))
                blank_values.append(float(blank_value))
            compartment = None
            if compartments_seen:
                unique_compartments = list(dict.fromkeys(compartments_seen))
                compartment = unique_compartments[0] if len(unique_compartments) == 1 else "mixed"
            test = welch_ttest_summary(movie_values, blank_values)
            effect_size = float(np.nanmean(movie_values) - np.nanmean(blank_values)) if movie_values and blank_values else float("nan")
            rows.append(
                {
                    "animal_id": animal_id,
                    "global_dendrite_id": global_dendrite_id,
                    "compartment": compartment,
                    "classifier_type": "dendrite",
                    "comparison": "visual_response_movie_vs_blank",
                    "test_name": "welch_ttest",
                    "covariate": VISUAL_RESPONSE_COVARIATE_NAME,
                    "response_metric": get_active_visual_response_metric(),
                    "available": bool(test.get("available", False)),
                    "n_visual_values": int(len(movie_values)),
                    "n_blank_values": int(len(blank_values)),
                    "mean_visual": float(np.nanmean(movie_values)) if movie_values else float("nan"),
                    "mean_blank": float(np.nanmean(blank_values)) if blank_values else float("nan"),
                    "delta": effect_size,
                    "effect_size": effect_size,
                    "raw_pvalue": float(test.get("raw_pvalue", float("nan"))),
                    "adjusted_pvalue": float(test.get("adjusted_pvalue", float("nan"))),
                    "significant": bool(test.get("significant", False)),
                    "responsive": False,
                    "cohort": "nonresponsive",
                }
            )
    by_compartment: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment in {"basal", "apical"}:
            by_compartment[compartment].append(row)
    for compartment_rows in by_compartment.values():
        n_tests = apply_bonferroni_correction(compartment_rows)
        for row in compartment_rows:
            row["n_tests_corrected"] = int(n_tests)
            row["responsive"] = bool(row.get("significant", False) and np.isfinite(as_float(row.get("delta"))) and float(row.get("delta")) > 0)
            row["cohort"] = "responsive" if row["responsive"] else "nonresponsive"
    for row in rows:
        if "n_tests_corrected" not in row:
            row["n_tests_corrected"] = 0
            row["responsive"] = False
            row["cohort"] = "nonresponsive"
    cohort_ids: Dict[str, Dict[str, List[str]]] = {
        "basal": {"responsive": [], "nonresponsive": []},
        "apical": {"responsive": [], "nonresponsive": []},
    }
    counts: Dict[str, Dict[str, int]] = {
        "basal": {"responsive": 0, "nonresponsive": 0, "available": 0, "tested": 0},
        "apical": {"responsive": 0, "nonresponsive": 0, "available": 0, "tested": 0},
    }
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment not in cohort_ids:
            continue
        counts[compartment]["tested"] += 1
        if row.get("available"):
            counts[compartment]["available"] += 1
        cohort = "responsive" if row.get("responsive") else "nonresponsive"
        cohort_ids[compartment][cohort].append(str(row.get("global_dendrite_id")))
        counts[compartment][cohort] += 1
    for compartment in cohort_ids:
        for cohort in cohort_ids[compartment]:
            cohort_ids[compartment][cohort] = sorted(dict.fromkeys(cohort_ids[compartment][cohort]))
    return {
        "classifier_type": "dendrite",
        "method": VISUAL_RESPONSE_CLASSIFIER_METHOD,
        "version": VISUAL_RESPONSE_CLASSIFIER_VERSION,
        "covariate": VISUAL_RESPONSE_COVARIATE_NAME,
        "response_metric": get_active_visual_response_metric(),
        "visual_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "rows": rows,
        "cohort_ids": cohort_ids,
        "counts": counts,
    }


def classify_visual_responsive_spines(
    cache: Dict[str, Any],
    source_cache: Optional[Dict[str, Any]] = None,
    movie_state: str = VISUAL_RESPONSE_MOVIE_STATE,
    blank_state: str = VISUAL_RESPONSE_BLANK_STATE,
) -> Dict[str, Any]:
    if source_cache is not None:
        return _classify_visual_response_rows(cache, source_cache, kind="spine")
    rows: List[Dict[str, Any]] = []
    for animal_id, animal_entry in sorted(cache.get("animals", {}).items()):
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            compartments_seen: List[str] = []
            for global_spine_id, spine_record in sorted(dendrite_record.get("spines", {}).items()):
                spine_observation_rows: List[Dict[str, Any]] = []
                for exp_id, s_obs in sorted(spine_record.get("observations", {}).items()):
                    compartment = observation_compartment(cache, exp_id, s_obs)
                    if compartment is not None:
                        compartments_seen.append(str(compartment))
                    cut_means = s_obs.get("cut_state_means")
                    if not isinstance(cut_means, dict):
                        continue
                    movie_value = as_float(cut_means.get(movie_state))
                    blank_value = as_float(cut_means.get(blank_state))
                    if movie_value is None or blank_value is None:
                        continue
                    if not np.isfinite(movie_value) or not np.isfinite(blank_value):
                        continue
                    spine_observation_rows.append(
                        {
                            "group": "visual",
                            "trial_label": movie_state,
                            "response": float(movie_value),
                            "baseline": float(blank_value),
                        }
                    )
                    spine_observation_rows.append(
                        {
                            "group": "blank",
                            "trial_label": blank_state,
                            "response": float(blank_value),
                            "baseline": float(blank_value),
                        }
                    )
                summary = _movie_style_blank_vs_movies_summary(spine_observation_rows)
                if compartments_seen:
                    unique_compartments = list(dict.fromkeys(compartments_seen))
                    compartment = unique_compartments[0] if len(unique_compartments) == 1 else "mixed"
                else:
                    compartment = None
                rows.append(
                    {
                        "animal_id": animal_id,
                        "global_spine_id": global_spine_id,
                        "compartment": compartment,
                        "classifier_type": "spine",
                        "comparison": summary.get("comparison", "visual_response_movie_vs_blank"),
                        "test_name": summary.get("test_name", "movie_style_blank_vs_movies"),
                        "covariate": summary.get("covariate", VISUAL_RESPONSE_COVARIATE_NAME),
                        "response_metric": summary.get("response_metric", get_active_visual_response_metric()),
                        "visual_trial_labels": list(dict.fromkeys(summary.get("visual_trial_labels", []))) if isinstance(summary.get("visual_trial_labels"), list) else [],
                        "blank_trial_labels": list(dict.fromkeys(summary.get("blank_trial_labels", []))) if isinstance(summary.get("blank_trial_labels"), list) else [],
                        "available": bool(summary.get("available", False)),
                        "n_visual_values": int(summary.get("n_visual_values", 0)),
                        "n_blank_values": int(summary.get("n_blank_values", 0)),
                        "mean_visual": float(summary.get("mean_visual", float("nan"))),
                        "mean_blank": float(summary.get("mean_blank", float("nan"))),
                        "delta": float(summary.get("delta", float("nan"))),
                        "effect_size": float(summary.get("delta", float("nan"))),
                        "statistic": float(summary.get("statistic", float("nan"))),
                        "raw_pvalue": float(summary.get("raw_pvalue", float("nan"))),
                        "adjusted_pvalue": float(summary.get("adjusted_pvalue", float("nan"))),
                        "significant": bool(summary.get("significant", False)),
                        "responsive": False,
                        "cohort": "nonresponsive",
                    }
                )
    by_compartment: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment in {"basal", "apical"}:
            by_compartment[compartment].append(row)
    for compartment_rows in by_compartment.values():
        n_tests = apply_bonferroni_correction(compartment_rows)
        for row in compartment_rows:
            row["n_tests_corrected"] = int(n_tests)
            row["responsive"] = bool(row.get("significant", False) and np.isfinite(as_float(row.get("delta"))) and float(row.get("delta")) > 0)
            row["cohort"] = "responsive" if row["responsive"] else "nonresponsive"
    for row in rows:
        if "n_tests_corrected" not in row:
            row["n_tests_corrected"] = 0
            row["responsive"] = False
            row["cohort"] = "nonresponsive"
    cohort_ids: Dict[str, Dict[str, List[str]]] = {
        "basal": {"responsive": [], "nonresponsive": []},
        "apical": {"responsive": [], "nonresponsive": []},
    }
    counts: Dict[str, Dict[str, int]] = {
        "basal": {"responsive": 0, "nonresponsive": 0, "available": 0, "tested": 0},
        "apical": {"responsive": 0, "nonresponsive": 0, "available": 0, "tested": 0},
    }
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment not in cohort_ids:
            continue
        counts[compartment]["tested"] += 1
        if row.get("available"):
            counts[compartment]["available"] += 1
        cohort = "responsive" if row.get("responsive") else "nonresponsive"
        cohort_ids[compartment][cohort].append(str(row.get("global_spine_id")))
        counts[compartment][cohort] += 1
    for compartment in cohort_ids:
        for cohort in cohort_ids[compartment]:
            cohort_ids[compartment][cohort] = sorted(dict.fromkeys(cohort_ids[compartment][cohort]))
    return {
        "classifier_type": "spine",
        "method": VISUAL_RESPONSE_CLASSIFIER_METHOD,
        "version": VISUAL_RESPONSE_CLASSIFIER_VERSION,
        "covariate": VISUAL_RESPONSE_COVARIATE_NAME,
        "response_metric": get_active_visual_response_metric(),
        "visual_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "rows": rows,
        "cohort_ids": cohort_ids,
        "counts": counts,
    }


def visual_response_dendrite_ids(
    response_summary: Dict[str, Any],
    compartment: str,
    cohort: str,
) -> List[str]:
    cohort_ids = response_summary.get("cohort_ids", {}) if isinstance(response_summary, dict) else {}
    compartment_entry = cohort_ids.get(compartment, {}) if isinstance(cohort_ids, dict) else {}
    if isinstance(compartment_entry, dict):
        ids = compartment_entry.get(cohort, [])
        if isinstance(ids, (list, tuple)):
            return [str(dendrite_id) for dendrite_id in ids if str(dendrite_id)]
    return []


def visual_response_spine_ids(
    response_summary: Dict[str, Any],
    compartment: str,
    cohort: str,
) -> List[str]:
    return visual_response_dendrite_ids(response_summary, compartment, cohort)


def padded_value_limits(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if low == high:
        pad = max(0.15, 0.1 * (abs(low) if low != 0 else 1.0))
    else:
        span = high - low
        pad = max(0.06 * span, 0.05 * max(abs(low), abs(high), 1.0))
    return low - pad, high + pad
def state_summary_y_limits(
    cache: Dict[str, Any],
    state_labels: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    cache_key = _state_summary_y_limits_cache_key(cache, state_labels)
    cached = _STATE_SUMMARY_Y_LIMITS_CACHE.get(cache_key)
    if cached is None:
        cached = _state_summary_cache_store(cache).get(cache_key)
        if isinstance(cached, dict):
            _STATE_SUMMARY_Y_LIMITS_CACHE[cache_key] = cached
            return cached
    else:
        return cached
    y_limits: Dict[str, Tuple[float, float]] = {}
    for metric_kind in ["dendrite_mean", "spine_specific_mean", "dendrite_event_frequency_per_min", "spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"]:
        combined_values: List[float] = []
        for compartment in [None, "basal", "apical"]:
            metric_summary = summarize_state_values(cache, metric_kind, state_labels, compartment)
            for state_label in state_labels:
                state_values = flatten_state_summary_values(metric_summary.get(state_label, {}))
                if state_values.size:
                    combined_values.extend(state_values.tolist())
        if combined_values:
            y_limits[metric_kind] = padded_value_limits(combined_values)
    _STATE_SUMMARY_Y_LIMITS_CACHE[cache_key] = y_limits
    _state_summary_cache_store(cache)[cache_key] = y_limits
    return y_limits


def state_summary_y_limits_from_results(
    results: Dict[str, Any],
    state_labels: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    state_summaries = results.get("state_summaries", {})
    if not isinstance(state_summaries, dict):
        return {}
    y_limits: Dict[str, Tuple[float, float]] = {}
    for metric_kind in ["dendrite_mean", "spine_specific_mean", "dendrite_event_frequency_per_min", "spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"]:
        metric_summary = state_summaries.get(metric_kind, {})
        if not isinstance(metric_summary, dict):
            continue
        combined_values: List[float] = []
        for state_label in state_labels:
            state_values = flatten_state_summary_values(metric_summary.get(state_label, {}))
            if state_values.size:
                combined_values.extend(state_values.tolist())
        if combined_values:
            y_limits[metric_kind] = padded_value_limits(combined_values)
    return y_limits


def format_requested_state_label(state_label: str) -> str:
    return state_display_label(state_label)


def _square_heatmap_state_labels(state_labels: Sequence[str]) -> List[str]:
    return [format_requested_state_label(label) for label in state_labels]


def _pad_boxplot_ylim(
    ax: Any,
    value_groups: Sequence[Sequence[float] | np.ndarray],
    y_limit: Optional[Tuple[float, float]] = None,
) -> None:
    finite_values: List[float] = []
    for values in value_groups:
        arr = np.asarray(values, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_values.extend(float(value) for value in finite)
    if not finite_values:
        return
    low = float(np.nanmin(finite_values))
    high = float(np.nanmax(finite_values))
    if not np.isfinite(low) or not np.isfinite(high):
        return
    if low == high:
        pad = max(0.15, 0.1 * (abs(low) if low != 0 else 1.0))
    else:
        span = high - low
        pad = max(0.06 * span, 0.05 * max(abs(low), abs(high), 1.0))
    ax.set_ylim(low - pad, high + pad)
    if y_limit is not None and len(y_limit) == 2:
        try:
            ax.set_ylim(float(y_limit[0]), float(y_limit[1]))
        except Exception:
            pass


def color_state_tick_labels(ax: Any, state_labels: Sequence[str], axis: str = "x") -> None:
    tick_labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for tick_label, state_label in zip(tick_labels, state_labels):
        tick_label.set_color(state_display_color(state_label))
def _configure_square_heatmap_axes(
    ax: Any,
    state_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    *,
    label_fontsize: Optional[float] = None,
    show_axis_labels: bool = True,
) -> None:
    positions = np.arange(len(state_labels))
    formatted_labels = _square_heatmap_state_labels(state_labels)
    tick_fontsize = max(9, int(label_fontsize if label_fontsize is not None else POSTER_FONT_SIZE - 1))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(
        formatted_labels,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(formatted_labels, fontsize=tick_fontsize)
    color_state_tick_labels(ax, state_labels, axis="x")
    color_state_tick_labels(ax, state_labels, axis="y")
    if show_axis_labels:
        ax.set_xlabel(xlabel, fontsize=max(POSTER_LABEL_SIZE - 6, tick_fontsize + 2))
        ax.set_ylabel(ylabel, fontsize=max(POSTER_LABEL_SIZE - 6, tick_fontsize + 2))
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=tick_fontsize, pad=4)
    ax.tick_params(axis="y", labelsize=tick_fontsize, pad=3)
def set_requested_state_ticks(ax: Any, state_labels: Sequence[str], axis: str = "x") -> None:
    positions = np.arange(1, len(state_labels) + 1)
    formatted_labels = [format_requested_state_label(label) for label in state_labels]
    if axis == "y":
        ax.set_ylim(0.5, len(state_labels) + 0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(formatted_labels)
        color_state_tick_labels(ax, state_labels, axis="y")
        ax.tick_params(axis="y", labelsize=14, pad=3)
        return
    ax.set_xlim(0.5, len(state_labels) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        formatted_labels,
        rotation=55,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    color_state_tick_labels(ax, state_labels, axis="x")
    ax.tick_params(axis="x", labelsize=14, pad=3)

def summarize_state_values_by_dendrite(
    cache: Dict[str, Any],
    metric_kind: str,
    state_labels: Sequence[str],
    compartment_filter: Optional[str] = None,
    subject_key: str = "day_id",
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    cache_key = _state_summary_cache_key(
        "dendrite_values",
        cache,
        metric_kind=metric_kind,
        state_labels=state_labels,
        compartment_filter=compartment_filter,
        subject_key=subject_key,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    )
    cached = _STATE_SUMMARY_DENDRITE_VALUES_CACHE.get(cache_key)
    if cached is None:
        cached = _state_summary_cache_store(cache).get(cache_key)
        if isinstance(cached, dict):
            _STATE_SUMMARY_DENDRITE_VALUES_CACHE[cache_key] = cached
            return cached
    else:
        return cached
    by_state = _collect_state_summary_values(
        cache,
        metric_kind,
        state_labels,
        compartment_filter,
        subject_key=subject_key,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
        by_dendrite=True,
    )
    _STATE_SUMMARY_DENDRITE_VALUES_CACHE[cache_key] = by_state
    _state_summary_cache_store(cache)[cache_key] = by_state
    return by_state


def build_state_summary_gallery_results(
    cache: Dict[str, Any],
    state_labels: Sequence[str],
    compartment_filter: Optional[str] = None,
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    cache_key = _state_summary_gallery_cache_key(
        cache,
        state_labels,
        compartment_filter,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    )
    cached = _STATE_SUMMARY_GALLERY_RESULTS_CACHE.get(cache_key)
    if cached is None:
        cached = _state_summary_cache_store(cache).get(cache_key)
        if isinstance(cached, dict):
            _STATE_SUMMARY_GALLERY_RESULTS_CACHE[cache_key] = cached
            return cached
    else:
        return cached
    result = {
        "state_summaries": {
            "dendrite_mean": summarize_state_values(cache, "dendrite_mean", state_labels, compartment_filter, dendrite_ids_filter=dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "spine_specific_mean": summarize_state_values(cache, "spine_specific_mean", state_labels, compartment_filter, dendrite_ids_filter=dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "dendrite_event_frequency_per_min": summarize_state_values(cache, "dendrite_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter=dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "spine_event_frequency_per_min": summarize_state_values(cache, "spine_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter=dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "coincident_event_frequency_per_min": summarize_state_values(cache, "coincident_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter=dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "noncoincident_event_frequency_per_min": summarize_state_values(cache, "noncoincident_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter=dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
        },
        "state_dendrite_summaries": {
            "dendrite_mean": summarize_state_values_by_dendrite(cache, "dendrite_mean", state_labels, compartment_filter, dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "spine_specific_mean": summarize_state_values_by_dendrite(cache, "spine_specific_mean", state_labels, compartment_filter, dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "dendrite_event_frequency_per_min": summarize_state_values_by_dendrite(cache, "dendrite_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "spine_event_frequency_per_min": summarize_state_values_by_dendrite(cache, "spine_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "coincident_event_frequency_per_min": summarize_state_values_by_dendrite(cache, "coincident_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
            "noncoincident_event_frequency_per_min": summarize_state_values_by_dendrite(cache, "noncoincident_event_frequency_per_min", state_labels, compartment_filter, dendrite_ids_filter, spine_ids_filter=spine_ids_filter),
        },
    }
    _STATE_SUMMARY_GALLERY_RESULTS_CACHE[cache_key] = result
    _state_summary_cache_store(cache)[cache_key] = result
    return result


def build_visual_response_dendrite_summary_results(
    cache: Dict[str, Any],
    state_labels: Sequence[str],
    response_summary: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    cohort_results: Dict[str, Dict[str, Any]] = {}
    for cohort in DENDRITE_RESPONSE_COHORTS[1:]:
        cohort_results[cohort] = {}
        for compartment in ("basal", "apical"):
            dendrite_ids = visual_response_dendrite_ids(response_summary, compartment, cohort)
            cohort_results[cohort][compartment] = build_state_summary_gallery_results(
                cache,
                state_labels,
                compartment,
                dendrite_ids,
            )
    return cohort_results


def build_visual_response_spine_summary_results(
    response_summary: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    cohort_results: Dict[str, Dict[str, List[str]]] = {}
    for cohort in DENDRITE_RESPONSE_COHORTS[1:]:
        cohort_results[cohort] = {
            "basal": visual_response_spine_ids(response_summary, "basal", cohort),
            "apical": visual_response_spine_ids(response_summary, "apical", cohort),
        }
    return cohort_results


def build_visual_response_spine_state_summary_results(
    cache: Dict[str, Any],
    state_labels: Sequence[str],
    response_summary: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    cohort_results: Dict[str, Dict[str, Any]] = {}
    for cohort in DENDRITE_RESPONSE_COHORTS[1:]:
        cohort_results[cohort] = {
            "basal": build_state_summary_gallery_results(
                cache,
                state_labels,
                "basal",
                spine_ids_filter=visual_response_spine_ids(response_summary, "basal", cohort),
            ),
            "apical": build_state_summary_gallery_results(
                cache,
                state_labels,
                "apical",
                spine_ids_filter=visual_response_spine_ids(response_summary, "apical", cohort),
            ),
        }
    return cohort_results


def visual_response_figure_output_dir(root: Path, kind: str, cohort_label: str) -> Path:
    return figure_nested_dir(visual_response_figure_dir(root), kind, cohort_label)


def plot_visual_response_boxplot_figure(
    response_summary: Dict[str, Any],
    fig_dir: Path,
    *,
    output_name: str,
    title: str,
    cohort_label: str = "all",
    kind: str = "dendrites",
) -> Optional[str]:
    if plt is None:
        return None
    rows = response_summary.get("rows", []) if isinstance(response_summary, dict) else []
    if not isinstance(rows, list):
        rows = []
    if cohort_label != "all":
        rows = [row for row in rows if isinstance(row, dict) and str(row.get("cohort") or "all") == cohort_label]
    blank_values: List[float] = []
    visual_values: List[float] = []
    paired_rows: List[Tuple[float, float, bool]] = []
    for row in rows:
        blank_value = as_float(row.get("mean_blank"))
        visual_value = as_float(row.get("mean_visual"))
        if blank_value is None or visual_value is None or not np.isfinite(blank_value) or not np.isfinite(visual_value):
            paired_blank_values = [as_float(value) for value in (row.get("blank_reference_values") or row.get("paired_baseline_values") or [])]
            paired_visual_values = [as_float(value) for value in (row.get("paired_stimulus_values") or [])]
            paired_blank_values = [float(value) for value in paired_blank_values if value is not None and np.isfinite(value)]
            paired_visual_values = [float(value) for value in paired_visual_values if value is not None and np.isfinite(value)]
            if paired_blank_values and paired_visual_values:
                blank_value = float(np.nanmean(np.asarray(paired_blank_values, dtype=float)))
                visual_value = float(np.nanmean(np.asarray(paired_visual_values, dtype=float)))
        if blank_value is None or visual_value is None:
            continue
        if not np.isfinite(blank_value) or not np.isfinite(visual_value):
            continue
        blank_values.append(float(blank_value))
        visual_values.append(float(visual_value))
        paired_rows.append((float(blank_value), float(visual_value), bool(row.get("responsive"))))
    if not blank_values or not visual_values:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.7))
    data = [np.asarray(blank_values, dtype=float), np.asarray(visual_values, dtype=float)]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.58, patch_artist=True, showfliers=False)
    box_colors = ["#D9D9D9", "#4C78A8"]
    for patch, color in zip(bp.get("boxes", []), box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.92)
    for whisker in bp.get("whiskers", []):
        whisker.set_color("#555555")
    for cap in bp.get("caps", []):
        cap.set_color("#555555")
    for median in bp.get("medians", []):
        median.set_color("#222222")
        median.set_linewidth(1.5)
    rng = np.random.default_rng(7)
    for blank_value, visual_value, responsive in paired_rows:
        line_color = "#2F855A" if responsive else "#888888"
        ax.plot([1, 2], [blank_value, visual_value], color=line_color, alpha=0.20, linewidth=0.9, zorder=0)
    for xpos, values, color in [(1, data[0], "#6B6B6B"), (2, data[1], "#2F5D8A")]:
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(np.full(values.size, xpos) + jitter, values, s=14, alpha=0.45, color=color, edgecolor="none")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Blank", "Movies"], fontsize=POSTER_FONT_SIZE)
    ax.set_ylabel("Mean activity during cut stimulus", fontsize=POSTER_LABEL_SIZE)
    ax.set_title(title, fontsize=max(17, POSTER_TITLE_SIZE - 5), pad=4)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
    all_values = np.concatenate(data)
    _pad_boxplot_ylim(ax, [all_values])
    annotate_sample_size(ax, 0.02, 0.98, f"n={len(blank_values)} ROI pairs", ha="left", va="top", fontsize=POSTER_NOTE_SIZE - 1, transform=ax.transAxes)
    ttest = stats.ttest_ind(np.asarray(visual_values, dtype=float), np.asarray(blank_values, dtype=float), equal_var=False, nan_policy="omit")
    p_value = float(ttest.pvalue) if np.isfinite(ttest.pvalue) else float("nan")
    if np.isfinite(p_value) and p_value < REPORT_SIGNIFICANCE_ALPHA:
        finite = np.concatenate([blank_values, visual_values])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y = float(np.nanmax(finite)) + max(0.05 * float(np.ptp(finite)), 0.05)
            ax.plot([1.0, 1.0, 2.0, 2.0], [y * 0.98, y, y, y * 0.98], color="#8b0000", linewidth=1.2)
            ax.text(1.5, y, "*", ha="center", va="bottom", fontsize=24, color="#8b0000", fontweight="bold")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.14, top=0.88)
    output_path = Path(fig_dir) / output_name
    save_figure(fig, output_path, dpi=POSTER_DPI, extra_formats=("svg",))
    return str(output_path)


def _visual_response_entity_plot_data(
    source_cache: Optional[Dict[str, Any]],
    kind: str,
    exp_id: str,
    cut_cache: Dict[str, Optional[Dict[str, Any]]],
    observation: Dict[str, Any],
    parent_dendrite_observation: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    exp_id = str(exp_id or "")
    if not exp_id:
        return None
    cut_data = _load_visual_response_cut_data(source_cache, exp_id, cut_cache, observation)
    if not cut_data:
        return None
    roi_index = as_int(observation.get("local_ids", {}).get("conversion_index"))
    if roi_index is None:
        return None
    cut_neural = np.asarray(cut_data.get("cut_neural"), dtype=float)
    cut_time = np.asarray(cut_data.get("cut_time"), dtype=float)
    trial_meta = cut_data.get("trial_meta", []) if isinstance(cut_data, dict) else []
    if roi_index < 0 or roi_index >= cut_neural.shape[0] or cut_time.size == 0:
        return None
    if kind == "spine":
        if parent_dendrite_observation is None:
            return None
        dendrite_index = as_int(parent_dendrite_observation.get("local_ids", {}).get("conversion_index"))
        alpha = as_float(observation.get("alpha"))
        if dendrite_index is None or alpha is None:
            return None
        if dendrite_index < 0 or dendrite_index >= cut_neural.shape[0]:
            return None
        trial_matrix = np.asarray(cut_neural[roi_index] - alpha * cut_neural[dendrite_index], dtype=float)
    else:
        trial_matrix = np.asarray(cut_neural[roi_index], dtype=float)

    visual_traces: List[np.ndarray] = []
    blank_traces: List[np.ndarray] = []
    visual_values: List[float] = []
    blank_values: List[float] = []
    for meta in trial_meta:
        if not isinstance(meta, dict):
            continue
        trial_label = canonical_state_label(meta.get("state_label"))
        group = _visual_response_trial_group(trial_label)
        if group is None:
            continue
        trial_index = as_int(meta.get("trial_index"))
        if trial_index is None or trial_index < 0 or trial_index >= trial_matrix.shape[0]:
            continue
        trial_trace = np.asarray(trial_matrix[trial_index], dtype=float)
        if not np.isfinite(trial_trace).any():
            continue
        trial_duration = as_float(meta.get("duration"))
        _, stimulus = trial_activity_means(trial_trace, cut_time, trial_duration)
        if not np.isfinite(stimulus):
            continue
        if group == "visual":
            visual_traces.append(trial_trace)
            visual_values.append(float(stimulus))
        else:
            blank_traces.append(trial_trace)
            blank_values.append(float(stimulus))

    if not visual_values or not blank_values:
        return None

    def _mean_trace(traces: Sequence[np.ndarray]) -> np.ndarray:
        if not traces:
            return np.asarray([], dtype=float)
        stacked = np.asarray([np.asarray(trace, dtype=float) for trace in traces], dtype=float)
        return np.asarray(np.nanmean(stacked, axis=0), dtype=float)

    return {
        "cut_time": cut_time,
        "visual_traces": [np.asarray(trace, dtype=float) for trace in visual_traces],
        "blank_traces": [np.asarray(trace, dtype=float) for trace in blank_traces],
        "visual_mean_trace": _mean_trace(visual_traces),
        "blank_mean_trace": _mean_trace(blank_traces),
        "visual_values": np.asarray(visual_values, dtype=float),
        "blank_values": np.asarray(blank_values, dtype=float),
        "visual_label": VISUAL_RESPONSE_VISUAL_TRIAL_TYPES[0],
        "blank_label": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
    }


def _plot_visual_response_entity_trace_figure(
    response_row: Dict[str, Any],
    cut_time: np.ndarray,
    traces: Sequence[np.ndarray],
    mean_trace: np.ndarray,
    label: str,
    color: str,
    fig_dir: Path,
    *,
    kind: str,
    cohort_label: str,
    entity_id: str,
    count_label: str,
) -> Optional[str]:
    if plt is None:
        return None
    if cut_time.size == 0 or mean_trace.size == 0 or not traces:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.8))
    for trace in traces:
        ax.plot(cut_time, trace, color=color, linewidth=0.8, alpha=0.24, zorder=1)
    ax.plot(cut_time, mean_trace, color=color, linewidth=2.2, zorder=3)
    ax.set_title(f"{label.capitalize()} traces", fontsize=POSTER_TITLE_SIZE - 4)
    ax.set_xlabel("Time (s)", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(ax, axis="both", nbins=4)
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.02,
        0.98,
        f"trials: {count_label}={len(traces)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE - 3,
        color="#444444",
    )
    cohort_text = f" ({cohort_label})" if cohort_label and cohort_label != "all" else ""
    title_prefix = kind.capitalize()
    fig.suptitle(f"{title_prefix} {response_row.get('animal_id', '')} {entity_id}{cohort_text}", fontsize=POSTER_SUPTITLE_SIZE - 3, y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.84)
    output_path = Path(fig_dir) / f"{safe_filename_component(str(response_row.get('animal_id') or 'animal'))}_{safe_filename_component(entity_id)}_{cohort_label}_{label}_traces.svg"
    save_figure(fig, output_path, dpi=POSTER_DPI, extra_formats=("png",))
    return str(output_path)


def plot_visual_response_entity_figure(
    response_row: Dict[str, Any],
    cache: Dict[str, Any],
    source_cache: Optional[Dict[str, Any]],
    fig_dir: Path,
    *,
    kind: str,
    cohort_label: str,
    cut_cache: Dict[str, Optional[Dict[str, Any]]],
) -> Optional[str]:
    if plt is None:
        return None
    exp_id, observation, parent_dendrite_record, parent_dendrite_observation = _visual_response_entity_observation(cache, kind, response_row)
    if exp_id is None or observation is None:
        return None
    entity_id = str(response_row.get("global_dendrite_id") if kind == "dendrite" else response_row.get("global_spine_id") or "")
    plot_data = _visual_response_entity_plot_data(source_cache, kind, exp_id, cut_cache, observation, parent_dendrite_observation)
    if not plot_data:
        return None
    cut_time = np.asarray(plot_data.get("cut_time", []), dtype=float)
    visual_mean_trace = np.asarray(plot_data.get("visual_mean_trace", []), dtype=float)
    blank_mean_trace = np.asarray(plot_data.get("blank_mean_trace", []), dtype=float)
    visual_values = np.asarray(plot_data.get("visual_values", []), dtype=float)
    blank_values = np.asarray(plot_data.get("blank_values", []), dtype=float)
    visual_traces = [np.asarray(trace, dtype=float) for trace in (plot_data.get("visual_traces", []) or [])]
    blank_traces = [np.asarray(trace, dtype=float) for trace in (plot_data.get("blank_traces", []) or [])]
    if cut_time.size == 0 or visual_mean_trace.size == 0 or blank_mean_trace.size == 0:
        return None

    title_label = f"{response_row.get('animal_id', '')} {entity_id}".strip()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), gridspec_kw={"width_ratios": [1.05, 1.05, 0.95]})
    blank_ax, movie_ax, box_ax = axes

    for trace in blank_traces:
        blank_ax.plot(cut_time, trace, color="#9AA0A6", linewidth=0.7, alpha=0.10, zorder=1)
    blank_ax.plot(cut_time, blank_mean_trace, color="#7F8790", linewidth=2.6, zorder=3)
    blank_ax.set_title("Blank traces", fontsize=POSTER_TITLE_SIZE - 4)
    blank_ax.set_xlabel("Time (s)", fontsize=POSTER_LABEL_SIZE)
    blank_ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
    blank_ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(blank_ax, axis="both", nbins=4)
    blank_ax.grid(axis="y", alpha=0.2)
    blank_ax.text(0.02, 0.98, f"trials: blank={len(blank_traces)}", transform=blank_ax.transAxes, ha="left", va="top", fontsize=POSTER_NOTE_SIZE - 3, color="#444444")

    for trace in visual_traces:
        movie_ax.plot(cut_time, trace, color="#F58518", linewidth=0.7, alpha=0.10, zorder=1)
    movie_ax.plot(cut_time, visual_mean_trace, color="#D97706", linewidth=2.6, zorder=3)
    movie_ax.set_title("Movies traces", fontsize=POSTER_TITLE_SIZE - 4)
    movie_ax.set_xlabel("Time (s)", fontsize=POSTER_LABEL_SIZE)
    movie_ax.set_ylabel("dF/F", fontsize=POSTER_LABEL_SIZE)
    movie_ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(movie_ax, axis="both", nbins=4)
    movie_ax.grid(axis="y", alpha=0.2)
    movie_ax.text(0.02, 0.98, f"trials: movies={len(visual_traces)}", transform=movie_ax.transAxes, ha="left", va="top", fontsize=POSTER_NOTE_SIZE - 3, color="#444444")

    data = [blank_values, visual_values]
    bp = box_ax.boxplot(data, positions=[1.0, 2.0], widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(bp.get("boxes", []), ["#D9D9D9", "#F8C38B"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.95)
    for whisker in bp.get("whiskers", []):
        whisker.set_color("#555555")
        whisker.set_linewidth(1.0)
    for cap in bp.get("caps", []):
        cap.set_color("#555555")
        cap.set_linewidth(1.0)
    for median in bp.get("medians", []):
        median.set_color("#222222")
        median.set_linewidth(1.5)
    rng = np.random.default_rng(7)
    for xpos, values, color in [(1.0, blank_values, "#6B6B6B"), (2.0, visual_values, "#C96E00")]:
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        box_ax.scatter(np.full(values.size, xpos) + jitter, values, s=12, alpha=0.42, color=color, edgecolor="none")
    all_values = np.concatenate([blank_values, visual_values])
    _pad_boxplot_ylim(box_ax, [all_values])
    box_ax.set_xticks([1.0, 2.0])
    box_ax.set_xticklabels(["blank", "movies"], fontsize=POSTER_FONT_SIZE - 1)
    box_ax.set_ylabel("Mean cut-stimulus activity", fontsize=POSTER_LABEL_SIZE)
    box_ax.set_title("Blank vs movies", fontsize=POSTER_TITLE_SIZE - 4)
    box_ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
    box_ax.grid(axis="y", alpha=0.2)
    box_ax.text(0.02, 0.98, f"n={int(blank_values.size)} blank, n={int(visual_values.size)} visual", transform=box_ax.transAxes, ha="left", va="top", fontsize=POSTER_NOTE_SIZE - 2, color="#444444")

    cohort_text = f" ({cohort_label})" if cohort_label and cohort_label != "all" else ""
    fig.suptitle(f"{kind.capitalize()} {title_label}{cohort_text}", fontsize=POSTER_SUPTITLE_SIZE - 3, y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.84, wspace=0.36)
    output_path = Path(fig_dir) / f"{safe_filename_component(str(response_row.get('animal_id') or 'animal'))}_{safe_filename_component(entity_id)}_{cohort_label}_blank_vs_movies.svg"
    save_figure(fig, output_path, dpi=POSTER_DPI, extra_formats=("png",))
    return str(output_path)


def render_cached_visual_response_figures(
    output_dir: Path,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    source_cache: Optional[Dict[str, Any]],
    *,
    figure_root: Optional[Path] = None,
) -> List[str]:
    if plt is None:
        return []
    fig_root = ensure_dir(Path(figure_root) if figure_root is not None else (output_dir / "figures"))
    cut_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    saved: List[str] = []
    visual_response_specs = [
        ("dendrites", "dendrite_visual_response", "dendrite"),
        ("spines", "spine_visual_response", "spine"),
    ]
    with step_scope("visual response figures"):
        for kind, source_key, entity_kind in visual_response_specs:
            response_summary = results.get(source_key, {})
            if not isinstance(response_summary, dict) or not response_summary:
                step_message(f"no cached visual-response data for {kind}")
                continue
            rows = response_summary.get("rows", []) if isinstance(response_summary, dict) else []
            if not isinstance(rows, list) or not rows:
                step_message(f"no cached visual-response rows for {kind}")
                continue
            for cohort in DENDRITE_RESPONSE_COHORTS:
                cohort_rows = rows if cohort == "all" else [row for row in rows if isinstance(row, dict) and str(row.get("cohort") or "all") == cohort]
                cohort_dir = visual_response_figure_output_dir(fig_root, kind, cohort)
                entity_dir = cohort_dir / "entities"
                finite_pairs = 0
                saved_entities = 0
                for row in cohort_rows:
                    blank_value = row.get("mean_blank")
                    visual_value = row.get("mean_visual")
                    if blank_value is None or visual_value is None:
                        continue
                    try:
                        blank_value = float(blank_value)
                        visual_value = float(visual_value)
                    except Exception:
                        continue
                    if np.isfinite(blank_value) and np.isfinite(visual_value):
                        finite_pairs += 1
                    output_path = plot_visual_response_entity_figure(
                        row,
                        cache,
                        source_cache,
                        entity_dir,
                        kind=entity_kind,
                        cohort_label=cohort,
                        cut_cache=cut_cache,
                    )
                    if output_path:
                        saved.append(output_path)
                        saved_entities += 1
                output_path = plot_visual_response_boxplot_figure(
                    response_summary,
                    cohort_dir,
                    output_name="visual_response_blank_vs_movies.svg",
                    title=f"{kind[:-1].capitalize()} visual response - {cohort.capitalize()}",
                    cohort_label=cohort,
                    kind=kind,
                )
                if output_path:
                    saved.append(output_path)
                else:
                    step_message(
                        f"no visual response figure for {kind}/{cohort} (rows={len(cohort_rows)}, finite_pairs={finite_pairs}, entities={saved_entities})"
                    )
    return saved

def gallery_compartment_suffix(compartment: Any) -> str:
    """Return a safe filename suffix for a compartment-specific gallery figure."""
    text = str(compartment or "").strip().lower()
    if not text or text in {"all", "none"}:
        return "all"
    return text.replace(" ", "_").replace("-", "_")


def filter_rows_by_compartment(
    rows: Sequence[Dict[str, Any]],
    compartment: Any,
) -> List[Dict[str, Any]]:
    """Filter analysis rows to one compartment.

    For compartment='all' or empty compartment, return rows unchanged.
    """
    compartment_key = str(compartment or "").strip().lower()
    if not compartment_key or compartment_key == "all":
        return list(rows or [])

    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        row_compartment = str(row.get("compartment") or "").strip().lower()
        if row_compartment == compartment_key:
            filtered.append(row)

    return filtered


def filter_rows_by_matrix_similarity(
    rows: Sequence[Dict[str, Any]],
    compartment: Any = None,
    *,
    compartment_filter: Any = None,
    **_compat_kwargs,
) -> List[Dict[str, Any]]:
    """Filter matrix-similarity rows to the requested output compartment."""
    selected_compartment = compartment_filter if compartment_filter is not None else compartment
    compartment_key = str(selected_compartment or "").strip().lower()
    if not compartment_key or compartment_key == "all":
        return list(rows or [])

    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue

        candidates = [
            row.get("output_compartment"),
            row.get("compartment"),
            row.get("dendrite_compartment"),
            row.get("source_compartment"),
            row.get("target_compartment"),
        ]
        candidate_keys = {str(value or "").strip().lower() for value in candidates}
        if compartment_key in candidate_keys:
            filtered.append(row)

    return filtered


def filter_rows_by_spine_coactivity(
    rows: Sequence[Dict[str, Any]],
    compartment: Any = None,
    *,
    compartment_filter: Any = None,
    **_compat_kwargs,
) -> List[Dict[str, Any]]:
    """Filter spine-coactivity rows to the requested output compartment."""
    selected_compartment = compartment_filter if compartment_filter is not None else compartment
    compartment_key = str(selected_compartment or "").strip().lower()
    if not compartment_key or compartment_key == "all":
        return list(rows or [])

    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue

        candidates = [
            row.get("output_compartment"),
            row.get("compartment"),
            row.get("dendrite_compartment"),
            row.get("spine_compartment"),
            row.get("source_compartment"),
            row.get("target_compartment"),
        ]
        candidate_keys = {str(value or "").strip().lower() for value in candidates}
        if compartment_key in candidate_keys:
            filtered.append(row)

    return filtered


def summarize_loaded_counts(cache: Dict[str, Any], compartment_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Summarize loaded checkpoint-gallery counts by compartment.

    The loading QC checkpoint expects one row per compartment with the basic
    loaded-object counts. We derive those counts from the cached animals /
    dendrites / observations so the plot stays consistent with the rest of the
    analysis cache.
    """

    def _matches_compartment(value: Any, selected: Optional[str]) -> bool:
        selected_key = str(selected or "").strip().lower()
        if not selected_key or selected_key == "all":
            return True
        return str(value or "").strip().lower() == selected_key

    def _count_rows_for_compartment(selected: Optional[str]) -> Optional[Dict[str, Any]]:
        animals = cache.get("animals", {})
        if not isinstance(animals, dict):
            return None

        day_ids = set()
        dendrite_ids = set()
        spine_ids = set()
        observation_count = 0

        for animal_entry in animals.values():
            if not isinstance(animal_entry, dict):
                continue
            dendrites = animal_entry.get("dendrites", {})
            if not isinstance(dendrites, dict):
                continue
            for dendrite_id, dendrite_record in dendrites.items():
                if not isinstance(dendrite_record, dict):
                    continue
                observations = dendrite_record.get("observations", {})
                if not isinstance(observations, dict):
                    continue
                for exp_id, d_obs in observations.items():
                    if not isinstance(d_obs, dict):
                        continue
                    if not _matches_compartment(observation_compartment(cache, exp_id, d_obs), selected):
                        continue
                    observation_count += 1
                    if exp_id is not None:
                        day_ids.add(str(exp_id))
                    if dendrite_id is not None:
                        dendrite_ids.add(str(dendrite_id))
                    spine_iterable = d_obs.get("spine_ids")
                    if isinstance(spine_iterable, (list, tuple, set)):
                        for spine_id in spine_iterable:
                            if spine_id is not None:
                                spine_ids.add(str(spine_id))
                    else:
                        spines = dendrite_record.get("spines", {})
                        if isinstance(spines, dict):
                            for spine_id in spines.keys():
                                if spine_id is not None:
                                    spine_ids.add(str(spine_id))

        if observation_count == 0:
            return None

        label = str(selected or "all").strip().lower() or "all"
        return {
            "compartment": label,
            "n_days": int(len(day_ids)),
            "n_dendrites": int(len(dendrite_ids)),
            "n_spines": int(len(spine_ids)),
            "n_observations": int(observation_count),
        }

    compartments: List[Optional[str]]
    if compartment_filter is None:
        compartments = [None]
        compartments.extend([comp for comp in sorted_present_compartments(cache) if comp not in compartments])
    else:
        compartments = [compartment_filter]

    rows: List[Dict[str, Any]] = []
    for compartment in compartments:
        row = _count_rows_for_compartment(compartment)
        if row is not None:
            rows.append(row)
    return rows

def select_representative_trace_record(
    records: Sequence[Dict[str, Any]],
    *,
    compartment: Any = None,
    compartment_filter: Any = None,
    trace_kind: Any = None,
    figure_kind: Any = None,
    state: Any = None,
    require_spine: bool = False,
    **_compat_kwargs,
) -> Optional[Dict[str, Any]]:
    """Pick a representative trace/example record from loaded gallery records.

    The function prefers records matching the requested metadata and then picks
    the first stable sorted match.
    """
    if not records:
        return None

    selected_compartment = compartment_filter if compartment_filter is not None else compartment
    compartment_key = str(selected_compartment or "").strip().lower()
    trace_kind_key = str(trace_kind or "").strip().lower()
    figure_kind_key = str(figure_kind or "").strip().lower()
    state_key = str(state or "").strip().lower()

    candidates: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        if compartment_key and compartment_key != "all":
            record_compartment = str(record.get("compartment") or "").strip().lower()
            if record_compartment and record_compartment != compartment_key:
                continue

        if trace_kind_key:
            record_trace_kind = str(record.get("trace_kind") or record.get("kind") or "").strip().lower()
            if record_trace_kind and record_trace_kind != trace_kind_key:
                continue

        if figure_kind_key:
            record_figure_kind = str(record.get("figure_kind") or "").strip().lower()
            if record_figure_kind and record_figure_kind != figure_kind_key:
                continue

        if state_key:
            record_state = str(record.get("state") or record.get("state_display") or "").strip().lower()
            if record_state and record_state != state_key:
                continue

        if require_spine:
            has_spine_marker = any(record.get(key) is not None for key in ("global_spine_id", "spine_id", "spine_ids", "spine_compartment"))
            if not has_spine_marker:
                continue

        candidates.append(record)

    if not candidates:
        candidates = [record for record in records if isinstance(record, dict)]
        if require_spine:
            spine_candidates = [
                record
                for record in candidates
                if any(record.get(key) is not None for key in ("global_spine_id", "spine_id", "spine_ids", "spine_compartment"))
            ]
            if spine_candidates:
                candidates = spine_candidates

    if not candidates:
        return None

    def sort_key(record: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            str(record.get("exp_id") or record.get("expid") or ""),
            str(record.get("cell_id") or record.get("roi_id") or record.get("spine_id") or ""),
            str(record.get("state") or record.get("state_display") or ""),
            str(record.get("trace_kind") or record.get("kind") or ""),
        )

    return sorted(candidates, key=sort_key)[0]

def build_filtered_correlation_results(results: Dict[str, Any], compartment_filter: Optional[str] = None) -> Dict[str, Any]:
    return {"correlations": filter_rows_by_compartment(results.get("correlations", []), compartment_filter)}
def build_filtered_matrix_similarity_results(
    results: Dict[str, Any],
    compartment_filter: Optional[str] = None,
    global_dendrite_id_filter: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "matrix_similarity": filter_rows_by_matrix_similarity(
            results.get("matrix_similarity", []),
            compartment_filter=compartment_filter,
            global_dendrite_id_filter=global_dendrite_id_filter,
        )
    }
def build_filtered_spine_coactivity_results(
    results: Dict[str, Any],
    compartment_filter: Optional[str] = None,
    global_dendrite_id_filter: Optional[str] = None,
) -> Dict[str, Any]:
    coactivity = results.get("spine_coactivity", {})
    table_rows = coactivity.get("table_rows", []) if isinstance(coactivity, dict) else []
    pair_state_rows = coactivity.get("pair_state_rows", []) if isinstance(coactivity, dict) else []
    filtered_rows = filter_rows_by_spine_coactivity(
        table_rows,
        compartment_filter=compartment_filter,
        global_dendrite_id_filter=global_dendrite_id_filter,
    )
    filtered_pair_state_rows = filter_rows_by_spine_coactivity(
        pair_state_rows if isinstance(pair_state_rows, list) and pair_state_rows else filtered_rows,
        compartment_filter=compartment_filter,
        global_dendrite_id_filter=global_dendrite_id_filter,
    )
    return {
        "spine_coactivity": {
            **coactivity,
            "table_rows": filtered_rows,
            "pair_state_rows": [row for row in filtered_pair_state_rows if str(row.get("status")) == "ok"],
        }
    }


def _coerce_boolish(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "sig", "significant", "selected", "coactive"}:
        return True
    if text in {"0", "false", "no", "n", "ns", "non-significant", "nonsignificant", "not_selected"}:
        return False
    return None


def spine_coactivity_abs_threshold_from_results(results: Dict[str, Any], default: float = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD) -> float:
    for container in (
        results.get("spine_coactivity", {}).get("selection", {}) if isinstance(results.get("spine_coactivity", {}), dict) else {},
        results.get("analysis_state_selection", {}) if isinstance(results.get("analysis_state_selection", {}), dict) else {},
        results.get("run_parameters", {}) if isinstance(results.get("run_parameters", {}), dict) else {},
        results.get("config", {}) if isinstance(results.get("config", {}), dict) else {},
    ):
        value = as_float(container.get("spine_coactivity_abs_threshold"))
        if value is not None and np.isfinite(value) and float(value) >= 0.0:
            return float(value)
    return float(default)


def spine_coactivity_anchor_selection_text(abs_threshold: float) -> str:
    return f"shuffle_significant and abs(coactivity_r) >= {float(abs_threshold):g}"


def spine_coactivity_anchor_selection_note(results: Dict[str, Any]) -> str:
    return spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold_from_results(results))


def spine_coactivity_quiet_anchor_selected(row: Dict[str, Any], abs_threshold: Optional[float] = None) -> bool:
    threshold = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD if abs_threshold is None else float(abs_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        threshold = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD

    explicit = _coerce_boolish(row.get(SPINE_COACTIVITY_QUIET_ANCHOR_SELECTION_FIELD))
    if explicit is not None:
        return explicit

    shuffle_significant = _coerce_boolish(row.get("shuffle_significant"))
    if shuffle_significant is None:
        shuffle_p = as_float(row.get("shuffle_p"))
        shuffle_significant = bool(shuffle_p is not None and np.isfinite(shuffle_p) and float(shuffle_p) < REPORT_SIGNIFICANCE_ALPHA)

    coactivity_r = as_float(row.get("coactivity_r"))
    return bool(shuffle_significant and coactivity_r is not None and np.isfinite(coactivity_r) and abs(float(coactivity_r)) >= threshold)


def spine_coactivity_pair_state_scope_name(anchor_state_filter: Optional[str], coactive_only: bool) -> str:
    anchor_text = safe_filename_component(anchor_state_filter) if anchor_state_filter is not None else "all_states"
    coactive_text = "coactive" if coactive_only else "all_pairs"
    return f"{anchor_text}_{coactive_text}"
def spine_coactivity_pair_state_output_name(kind: str, anchor_state_filter: Optional[str], compartment: Optional[str], coactive_only: bool = True) -> str:
    scope_name = spine_coactivity_pair_state_scope_name(anchor_state_filter, coactive_only)
    return f"spine_coactivity_{kind}_{scope_name}_{gallery_compartment_suffix(compartment)}.svg"
def spine_coactivity_basal_apical_distribution_output_name(anchor_state_filter: Optional[str], coactive_only: bool = True) -> str:
    scope_name = spine_coactivity_pair_state_scope_name(anchor_state_filter, coactive_only)
    return f"spine_coactivity_basal_vs_apical_distribution_{scope_name}.svg"
def spine_coactivity_anchor_state_compartments(rows: Sequence[Dict[str, Any]]) -> List[Optional[str]]:
    compartments: List[Optional[str]] = [None]
    compartments.extend(spine_coactivity_output_compartments(rows))
    deduped: List[Optional[str]] = []
    seen = set()
    for compartment in compartments:
        key = gallery_compartment_suffix(compartment)
        if key in seen:
            continue
        deduped.append(compartment)
        seen.add(key)
    return deduped
def _make_count_heatmap(count_rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[str]]:
    if not count_rows:
        return np.empty((0, 0), dtype=float), [], []
    row_labels = ["n_days", "n_dendrites", "n_spines", "n_observations"]
    col_labels = [row["compartment"] for row in count_rows]
    matrix = np.asarray([[float(row[label]) for row in count_rows] for label in row_labels], dtype=float)
    return matrix, row_labels, col_labels
def plot_loading_qc_checkpoint(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    fig_dir: Path,
    compartment_filter: Optional[str] = None,
    output_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    if plt is None:
        return None
    count_rows = summarize_loaded_counts(cache, compartment_filter)
    if not count_rows:
        return None
    matrix, row_labels, col_labels = _make_count_heatmap(count_rows)
    if matrix.size == 0:
        return None
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(min(max(7.0, POSTER_DOUBLE_FIGSIZE[0] - 2.6), 8.8), min(max(4.2, POSTER_DOUBLE_FIGSIZE[1] - 1.2), 5.8)),
        squeeze=False,
    )
    ax = ax.ravel()[0]
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([label.replace("_", "\n") for label in row_labels])
    ax.set_title("Loaded day counts", fontsize=POSTER_TITLE_SIZE)
    annotate_sample_size(
        ax,
        0.02,
        0.98,
        f"n days={int(np.nansum(matrix[0]))} | n observations={int(np.nansum(matrix[-1]))}",
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE - 1,
        transform=ax.transAxes,
        bbox=True,
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{int(matrix[row_index, col_index])}", ha="center", va="center", fontsize=POSTER_NOTE_SIZE, color="#1f1f1f")
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    loaded_cbar_ax = cbar.ax
    cbar.set_label("Count", fontsize=POSTER_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
    set_sparse_colorbar_ticks(cbar, nbins=5)
    suffix = gallery_compartment_suffix(compartment_filter)
    output_path = fig_dir / (output_name or f"01_loading_qc_{suffix}.svg")
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)


def plot_spine_regression_qc_checkpoint(
    cache: Dict[str, Any],
    fig_dir: Path,
    compartment_filter: Optional[str] = None,
    output_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    if plt is None:
        return None
    representative = select_representative_trace_record(cache, compartment_filter=compartment_filter, require_spine=True)
    if representative is None:
        return None
    exp_id = representative["exp_id"]
    d_obs = representative["d_obs"]
    s_obs = representative["s_obs"]
    if s_obs is None:
        return None
    time = np.asarray(d_obs.get("time"), dtype=float)
    dend_trace = np.asarray(d_obs.get("trace"), dtype=float)
    spine_trace = np.asarray(s_obs.get("trace"), dtype=float)
    spine_specific = np.asarray(s_obs.get("spine_specific"), dtype=float)
    fit = s_obs.get("fit") or {}
    event_info = s_obs.get("event_info") or {}
    dendrite_event_info = s_obs.get("dendrite_event_info") or d_obs.get("event_info") or {}
    alpha = float(fit.get("alpha", s_obs.get("alpha", float("nan"))))
    intercept = float(fit.get("intercept", float("nan")))
    threshold = float(event_info.get("threshold", float("nan")))
    event_runs = event_info.get("event_runs", [])
    dendrite_event_runs = dendrite_event_info.get("event_runs", [])
    coincident_event_runs = event_info.get("coincident_event_runs", [])
    noncoincident_event_runs = event_info.get("noncoincident_event_runs", [])
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(min(max(9.2, POSTER_WIDE_FIGSIZE[0] - 1.6), 10.0), min(max(5.6, POSTER_WIDE_FIGSIZE[1] - 1.0), 7.2)))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time, dend_trace, color="#4477aa", linewidth=1.2, label="Dendrite dF/F")
    ax.plot(time, spine_trace, color="#dd8452", linewidth=1.0, alpha=0.9, label="Spine dF/F")
    ax.plot(time, alpha * dend_trace + intercept, color="#555555", linestyle="--", linewidth=1.1, label="Robust fit")
    ax.set_title("Dendrite and spine traces", fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("Time (s)", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Dendrite dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.legend(frameon=False, fontsize=POSTER_LEGEND_SIZE)
    ax.grid(alpha=0.2)
    set_sparse_numeric_ticks(ax, axis="both", nbins=5)
    ax = fig.add_subplot(gs[0, 1])
    valid = np.isfinite(dend_trace) & np.isfinite(spine_trace)
    ax.scatter(dend_trace[valid], spine_trace[valid], s=12, alpha=0.25, color="#444444", edgecolor="none")
    if np.any(valid):
        x_min = float(np.nanpercentile(dend_trace[valid], 2))
        x_max = float(np.nanpercentile(dend_trace[valid], 98))
        if x_min == x_max:
            x_min -= 0.1
            x_max += 0.1
        x = np.linspace(x_min, x_max, 100)
        ax.plot(x, intercept + alpha * x, color="#aa3377", linewidth=1.5, label=f"alpha={alpha:.3f}")
    ax.set_xlabel("Dendrite dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Spine dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.set_title("Robust regression", fontsize=POSTER_TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.legend(frameon=False, fontsize=POSTER_LEGEND_SIZE)
    ax.grid(alpha=0.2)
    set_sparse_numeric_ticks(ax, axis="both", nbins=5)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(time, spine_specific, color="#7a5195", linewidth=1.2, label="Spine-specific dF/F", zorder=5)
    if np.isfinite(threshold):
        ax.axhline(threshold, color="#8b0000", linestyle="--", linewidth=1.0, label="3σ threshold", zorder=4)
    ax.axhline(0.0, color="#444444", linewidth=0.8, zorder=3)
    for start, end in dendrite_event_runs:
        if start < 0 or end <= start or start >= time.size:
            continue
        end_index = min(end - 1, time.size - 1)
        ax.axvspan(time[start], time[end_index], color="#4c78a8", alpha=0.08, zorder=1)
    for start, end in noncoincident_event_runs:
        if start < 0 or end <= start or start >= time.size:
            continue
        end_index = min(end - 1, time.size - 1)
        ax.axvspan(time[start], time[end_index], color="#f58518", alpha=0.10, zorder=2)
    for start, end in coincident_event_runs:
        if start < 0 or end <= start or start >= time.size:
            continue
        end_index = min(end - 1, time.size - 1)
        ax.axvspan(time[start], time[end_index], color="#d62728", alpha=0.14, zorder=3)
    ax.set_title("Spine-specific trace and event runs", fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("Time (s)", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Spine-specific dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    legend_handles = [
        Line2D([], [], color="#7a5195", linewidth=1.2, label="Spine-specific dF/F"),
        Line2D([], [], color="#8b0000", linestyle="--", linewidth=1.0, label="3σ threshold"),
    ]
    if dendrite_event_runs:
        legend_handles.append(Patch(facecolor="#4c78a8", edgecolor="none", alpha=0.18, label="Dendrite event"))
    if noncoincident_event_runs:
        legend_handles.append(Patch(facecolor="#f58518", edgecolor="none", alpha=0.18, label="Spine event (noncoincident)"))
    if coincident_event_runs:
        legend_handles.append(Patch(facecolor="#d62728", edgecolor="none", alpha=0.18, label="Spine event (coincident)"))
    ax.legend(handles=legend_handles, frameon=False, fontsize=POSTER_LEGEND_SIZE, loc="upper right")
    freq_lines = []
    def fmt_rate(value: Any) -> str:
        return "n/a" if value is None or not np.isfinite(value) else f"{float(value):.2f}/min"
    dendrite_rate = dendrite_event_info.get("event_frequency_per_min")
    spine_rate = event_info.get("spine_event_frequency_per_min", event_info.get("event_frequency_per_min"))
    coincident_rate = event_info.get("coincident_event_frequency_per_min")
    noncoincident_rate = event_info.get("noncoincident_event_frequency_per_min")
    coincident_fraction = as_float(event_info.get("coincident_event_fraction"))
    freq_lines.append(f"Dendrite events: {fmt_rate(dendrite_rate)}")
    freq_lines.append(f"Spine events: {fmt_rate(spine_rate)}")
    freq_lines.append(f"Coincident: {fmt_rate(coincident_rate)}")
    freq_lines.append(f"Noncoincident: {fmt_rate(noncoincident_rate)}")
    if coincident_fraction is not None and np.isfinite(coincident_fraction):
        freq_lines.append(f"Coincident fraction: {coincident_fraction:.2f}")
    if any(value is not None and np.isfinite(value) for value in [dendrite_rate, spine_rate, coincident_rate, noncoincident_rate, coincident_fraction]):
        ax.text(
            0.01,
            0.98,
            "\n".join(freq_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=POSTER_FONT_SIZE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
        )
    ax.grid(alpha=0.2)
    set_sparse_numeric_ticks(ax, axis="both", nbins=6)
    compartment = representative.get("compartment")
    suffix = gallery_compartment_suffix(compartment_filter)
    output_path = fig_dir / (output_name or f"02_spine_regression_qc_{suffix}.svg")
    save_figure(fig, output_path, extra_formats=())
    plt.close(fig)
    return str(output_path)


def plot_matrix_similarity_distribution(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "matrix_similarity_distribution.svg",
    title: str = "Spine-spine coefficient distributions",
    compartment_filter: Optional[str] = None,
) -> Optional[str]:
    if plt is None:
        return None
    rows = results.get("matrix_similarity", [])
    if compartment_filter is not None:
        rows = filter_rows_by_matrix_similarity(rows, compartment_filter=compartment_filter)
    if not rows:
        return None

    def _render_distribution(plot_rows: List[Dict[str, Any]], *, output_stem_name: str, plot_title: str) -> Optional[str]:
        state_labels = selected_matrix_plot_state_labels(results, plot_rows)
        pair_order = list(combinations(state_labels, 2))
        if not pair_order:
            return None
        compartments = matrix_similarity_output_compartments(plot_rows)
        if compartment_filter is None:
            if len(compartments) != 1:
                return None
            compartment = compartments[0]
        else:
            compartment = compartment_filter
        pair_labels = [f"{format_requested_state_label(state_a)} vs {format_requested_state_label(state_b)}" for state_a, state_b in pair_order]
        class_styles = {
            "positive significant": {"color": "#2a9d8f"},
            "negative significant": {"color": "#e76f51"},
            "non-significant": {"color": "#7f7f7f"},
        }
        try:
            from matplotlib.lines import Line2D
        except Exception:
            Line2D = None
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(min(max(8.4, POSTER_DENSE_FIGSIZE[0] - 0.2), 10.6), min(max(4.6, 0.48 * len(pair_labels) + 2.6), 10.8)),
            squeeze=False,
        )
        ax = ax.ravel()[0]
        palette = plt.get_cmap("Dark2")
        y_positions = np.arange(1, len(pair_order) + 1)
        legend_handles = []
        if Line2D is not None:
            legend_handles = [
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=style["color"], markeredgecolor="none", markersize=6, label=label)
                for label, style in class_styles.items()
            ]
        plotted_values: List[np.ndarray] = []
        plotted_positions: List[int] = []
        plotted_class_data: List[Dict[str, np.ndarray]] = []
        panel_class_counts = {label: 0 for label in class_styles}
        for position, (state_a, state_b) in zip(y_positions, pair_order):
            pair_key = frozenset((state_a, state_b))
            matched_rows = [row for row in plot_rows if frozenset((str(row.get("state_a")), str(row.get("state_b")))) == pair_key]
            coeffs = np.asarray([value for value in (as_float(row.get("matrix_similarity_r")) for row in matched_rows) if value is not None], dtype=float)
            coeffs = coeffs[np.isfinite(coeffs)]
            if coeffs.size == 0:
                continue
            class_data = {label: [] for label in class_styles}
            for row in matched_rows:
                r_value = as_float(row.get("matrix_similarity_r"))
                p_value = as_float(row.get("shuffle_p"))
                if r_value is None or not np.isfinite(r_value):
                    continue
                if p_value is None or not np.isfinite(p_value):
                    label = "non-significant"
                elif p_value < REPORT_SIGNIFICANCE_ALPHA and r_value > 0:
                    label = "positive significant"
                elif p_value < REPORT_SIGNIFICANCE_ALPHA and r_value < 0:
                    label = "negative significant"
                else:
                    label = "non-significant"
                class_data[label].append(float(r_value))
                panel_class_counts[label] += 1
            plotted_values.append(coeffs)
            plotted_positions.append(position)
            plotted_class_data.append({label: np.asarray(values, dtype=float) for label, values in class_data.items()})
        if not plotted_values:
            return None
        bp = ax.boxplot(plotted_values, positions=plotted_positions, widths=0.6, patch_artist=True, showfliers=False, vert=False)
        _set_boxplot_colors(bp, [palette((position - 1) % palette.N) for position in plotted_positions])
        for position, coeffs, class_data in zip(plotted_positions, plotted_values, plotted_class_data):
            for label, style in class_styles.items():
                values = class_data[label]
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                jitter = np.random.default_rng(50 + position).uniform(-0.12, 0.12, size=values.size)
                ax.scatter(
                    values,
                    np.full(values.size, position) + jitter,
                    s=16,
                    alpha=0.65,
                    color=style["color"],
                    edgecolor="none",
                    zorder=3,
                )
            ax.text(0.98, position, f"n={coeffs.size}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=POSTER_NOTE_SIZE, clip_on=False)
        ax.axvline(0.0, color="#333333", linewidth=1)
        ax.set_title(plot_title, fontsize=POSTER_TITLE_SIZE)
        ax.set_xlabel("Pearson r", fontsize=POSTER_LABEL_SIZE)
        ax.set_ylabel("State pair", fontsize=POSTER_LABEL_SIZE)
        ax.set_yticks(y_positions[: len(pair_labels)])
        ax.set_yticklabels(pair_labels)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=max(11, POSTER_FONT_SIZE - 3))
        ax.tick_params(axis="x", labelsize=POSTER_FONT_SIZE)
        ax.grid(axis="x", alpha=0.25)
        set_sparse_numeric_ticks(ax, axis="x", nbins=5)
        ax.text(
            0.02,
            0.98,
            (
                f"+sig {panel_class_counts['positive significant']} | "
                f"-sig {panel_class_counts['negative significant']} | "
                f"ns {panel_class_counts['non-significant']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
        )
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=POSTER_LEGEND_SIZE,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(legend_handles),
                columnspacing=1.0,
                handletextpad=0.5,
            )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.95)
        output_path = fig_dir / output_stem_name
        save_figure(fig, output_path, extra_formats=())
        return str(output_path)

    group_col = "split_group" if any(str(row.get("split_group") or "").strip() for row in rows) else None
    if group_col is not None:
        group_rows: Dict[str, List[Dict[str, Any]]] = {}
        group_display: Dict[str, str] = {}
        group_order: List[str] = []
        unassigned_rows: List[Dict[str, Any]] = []
        for row in rows:
            group_value = str(row.get(group_col) or "").strip()
            if not group_value:
                unassigned_rows.append(dict(row))
                continue
            group_key = group_value.lower().replace(" ", "_").replace("-", "_")
            while "__" in group_key:
                group_key = group_key.replace("__", "_")
            if group_key not in group_rows:
                group_order.append(group_key)
            group_rows.setdefault(group_key, []).append(dict(row))
            group_display.setdefault(group_key, str(row.get("split_group_display") or group_value).strip() or group_key)
        if unassigned_rows:
            group_rows["unassigned"] = unassigned_rows
            group_display.setdefault("unassigned", "Unassigned")
            group_order.append("unassigned")
        if len(group_order) > 1:
            generated: List[str] = []
            base_stem = Path(output_name).stem
            for group_key in group_order:
                subset = group_rows.get(group_key, [])
                if not subset:
                    continue
                rendered = _render_distribution(
                    subset,
                    output_stem_name=f"{base_stem}_{group_key}.svg",
                    plot_title=f"{title} - {group_display.get(group_key, group_key)}",
                )
                if rendered:
                    generated.append(rendered)
            return generated or None
    return _render_distribution(rows, output_stem_name=output_name, plot_title=title)


def mixed_model_branch_render_specs(
results: Dict[str, Any], review: bool = False) -> List[Dict[str, Any]]:
    branch_configs = [
        {
            "key": "mixed_model_selected_state",
            "scope": "selected_state",
            "name": "mixed_model_selected_state",
            "forest_output_name": "mixed_model_selected_state_forest.svg",
            "predicted_output_name": "mixed_model_selected_state_predicted_means.svg",
            "contrast_output_name": "mixed_model_contrasts_selected_state.svg",
            "forest_title": "Mixed-model fixed effects - selected state",
            "predicted_title": "Mixed-model predicted means - selected state",
            "contrast_title": "Mixed-model contrasts - selected state",
        },
        {
            "key": "mixed_model_visual_response_responsive_selected_state",
            "scope": "visual_response_responsive_selected_state",
            "name": "mixed_model_visual_response_responsive_selected_state",
            "forest_output_name": "mixed_model_visual_response_responsive_selected_state_forest.svg",
            "predicted_output_name": "mixed_model_visual_response_responsive_selected_state_predicted_means.svg",
            "contrast_output_name": "mixed_model_visual_response_responsive_selected_state_contrasts.svg",
            "forest_title": "Mixed-model fixed effects - visual response responsive selected state",
            "predicted_title": "Mixed-model predicted means - visual response responsive selected state",
            "contrast_title": "Mixed-model contrasts - visual response responsive selected state",
        },
        {
            "key": "mixed_model_visual_response_nonresponsive_selected_state",
            "scope": "visual_response_nonresponsive_selected_state",
            "name": "mixed_model_visual_response_nonresponsive_selected_state",
            "forest_output_name": "mixed_model_visual_response_nonresponsive_selected_state_forest.svg",
            "predicted_output_name": "mixed_model_visual_response_nonresponsive_selected_state_predicted_means.svg",
            "contrast_output_name": "mixed_model_visual_response_nonresponsive_selected_state_contrasts.svg",
            "forest_title": "Mixed-model fixed effects - visual response nonresponsive selected state",
            "predicted_title": "Mixed-model predicted means - visual response nonresponsive selected state",
            "contrast_title": "Mixed-model contrasts - visual response nonresponsive selected state",
        },
    ]
    review_prefix = "review_" if review else ""
    specs: List[Dict[str, Any]] = []
    for branch in branch_configs:
        branch_data = results.get(branch["key"], {})
        if not isinstance(branch_data, dict):
            continue
        if not (branch_data.get("designs") or branch_data.get("contrast_rows") or any(branch_data.get("summary_rows", {}).values())):
            continue
        specs.extend(
            [
                {
                    "name": f"{branch['name']}_forest",
                    "scope": branch["scope"],
                    "model_key": branch["key"],
                    "output_name": review_prefix + branch["forest_output_name"],
                    "title": ("Review: " if review else "") + branch["forest_title"],
                    "plotter": plot_mixed_model_forest_figure,
                },
                {
                    "name": f"{branch['name']}_predicted_means",
                    "scope": branch["scope"],
                    "model_key": branch["key"],
                    "output_name": review_prefix + branch["predicted_output_name"],
                    "title": ("Review: " if review else "") + branch["predicted_title"],
                    "plotter": plot_mixed_model_predicted_means_figure,
                },
                {
                    "name": f"{branch['name']}_contrasts",
                    "scope": branch["scope"],
                    "accepts_scope": True,
                    "model_key": branch["key"],
                    "output_name": review_prefix + branch["contrast_output_name"],
                    "title": ("Review: " if review else "") + branch["contrast_title"],
                    "plotter": plot_mixed_model_contrasts_checkpoint,
                },
            ]
        )
    return specs
def mixed_model_scope_is_shared(scope: Any) -> bool:
    text = str(scope or "").strip().lower()
    return text == "all_state" or text.endswith("_all_state")


def _mixed_model_response_payload(results: Dict[str, Any], response: str, model_key: str = "mixed_model") -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    mixed_model = results.get(model_key, {})
    if not isinstance(mixed_model, dict):
        return None, []
    designs = mixed_model.get("designs", {})
    summary_rows = mixed_model.get("summary_rows", {})
    if not isinstance(designs, dict) or not isinstance(summary_rows, dict):
        return None, []
    design = designs.get(response)
    if not isinstance(design, dict):
        return None, []
    rows = list(summary_rows.get(response, []))
    return design, rows
def _poster_state_display_label(state_label: Any) -> str:
    canonical = canonical_state_label(state_label)
    parts = [part for part in canonical.split("_") if part]
    if not parts:
        return ""
    if len(parts) == 1 and parts[0] == "nrem":
        return "NREM"
    if len(parts) == 1 and parts[0] == "rem":
        return "REM"
    if len(parts) >= 2 and parts[0] in {"quiet", "active"} and parts[1] == "awake":
        head = f"{parts[0].capitalize()} Awake"
        tail = " ".join(part.capitalize() for part in parts[2:])
        return f"{head} {tail}".strip()
    if parts[0] == "nrem":
        tail = " ".join(part.capitalize() for part in parts[1:])
        return f"NREM {tail}".strip()
    if parts[0] == "rem":
        tail = " ".join(part.capitalize() for part in parts[1:])
        return f"REM {tail}".strip()
    return " ".join(part.capitalize() for part in parts)


def _mixed_model_term_kind(term: str) -> str:
    if term == "Intercept":
        return "intercept"
    if ":" in term:
        return "interaction"
    if term.startswith("state["):
        return "state"
    if term.startswith("compartment["):
        return "compartment"
    if term.startswith("split_group["):
        return "split_group"
    if term.startswith("visual_response_cohort["):
        return "cohort"
    return "covariate"
def _mixed_model_term_component_label(term: str) -> str:
    if term == "Intercept":
        return term
    if "[" in term and term.endswith("]"):
        prefix, value = term.split("[", 1)
        value = value[:-1]
        if prefix.strip() in {"state", "compartment", "split_group", "visual_response_cohort"}:
            return _poster_state_display_label(value.strip())
        return f"{prefix.strip()}: {_poster_state_display_label(value.strip())}"
    return _poster_state_display_label(term)
def _mixed_model_term_value_label(term: str) -> str:
    if term == "Intercept":
        return term
    if "[" in term and term.endswith("]"):
        return term.split("[", 1)[1][:-1].strip()
    return term.replace("state", "").replace("compartment", "").replace(":", "").strip() or term
def _mixed_model_term_interaction_value_label(term: str) -> str:
    value = _mixed_model_term_value_label(term)
    state_aliases = {
        "quiet_awake_blank": "quiet_awake_blank",
        "active_awake_blank": "active_awake_blank",
        "quiet_awake_blanks": "quiet_awake_blank",
        "active_awake_blanks": "active_awake_blank",
        "quiet_awake_gratings": "quiet_awake_gratings",
        "active_awake_gratings": "active_awake_gratings",
        "quiet_awake_zebras": "quiet_awake_zebras",
        "active_awake_zebras": "active_awake_zebras",
        "quiet_awake_movies": "quiet_awake_movies",
        "active_awake_movies": "active_awake_movies",
        "quiet_awake": "quiet_awake",
        "active_awake": "active_awake",
        "nrem_blank": "nrem_blank",
        "nrem_blanks": "nrem_blank",
        "nrem_gratings": "nrem_grating",
        "nrem_zebras": "nrem_zebra",
        "nrem_movies": "nrem_movies",
        "rem_blank": "rem_blank",
        "rem_blanks": "rem_blank",
        "rem_gratings": "rem_grating",
        "rem_zebras": "rem_zebra",
        "rem_movies": "rem_movies",
    }
    return _poster_state_display_label(state_aliases.get(value, value))
def _mixed_model_term_label(term: str) -> str:
    if ":" not in term:
        return _mixed_model_term_component_label(term)
    parts = [_mixed_model_term_interaction_value_label(part) for part in term.split(":")]
    return " × ".join(parts)
def _mixed_model_observed_mean_payload(
    mixed_model: Dict[str, Any],
    response: str,
    design: Dict[str, Any],
) -> Dict[Tuple[Optional[str], str], Dict[str, float]]:
    rows = mixed_model.get("table_rows", [])
    if not isinstance(rows, list):
        return {}
    include_compartment = bool(design.get("include_compartment", False))
    include_split_group = bool(design.get("include_split_group", False))
    payload: Dict[Tuple[Optional[str], str], List[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(response))
        if value is None or not np.isfinite(value):
            continue
        state = canonical_state_label(row.get("state"))
        if include_split_group:
            split_group = str(row.get("split_group") or "").strip() or None
            payload[(split_group, state)].append(float(value))
        elif include_compartment:
            compartment = str(row.get("compartment")) if include_compartment else None
            payload[(compartment, state)].append(float(value))
        else:
            payload[(None, state)].append(float(value))
    summary: Dict[Tuple[Optional[str], str], Dict[str, float]] = {}
    for key, values in payload.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean = float(np.nanmean(arr))
        sem = float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")
        summary[key] = {"mean": mean, "sem": sem, "n": float(arr.size)}
    return summary

def _mixed_model_series_specs(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    fallback_palette = ["#1f77b4", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    if design.get("include_split_group"):
        split_group_reference = design.get("split_group_reference")
        split_group_levels = list(dict.fromkeys([split_group_reference] + [group for group in design.get("split_group_levels", []) if group != split_group_reference])) if split_group_reference is not None else list(design.get("split_group_levels", []))
        split_group_labels = {str(key): str(value) for key, value in (design.get("split_group_labels") or {}).items()}
        split_group_colors = {str(key): str(value) for key, value in (design.get("split_group_colors") or {}).items()}
        series_specs: List[Dict[str, Any]] = []
        for index, split_group in enumerate(split_group_levels):
            if split_group is None:
                continue
            split_group_text = str(split_group)
            if not split_group_text:
                continue
            series_specs.append(
                {
                    "compartment": design.get("compartment_reference") if design.get("include_compartment") else None,
                    "split_group": split_group_text,
                    "label": split_group_labels.get(split_group_text, _poster_state_display_label(split_group_text)),
                    "color": split_group_colors.get(split_group_text, fallback_palette[index % len(fallback_palette)]),
                }
            )
        if series_specs:
            return series_specs
    if design.get("include_compartment"):
        compartment_reference = design.get("compartment_reference")
        series_specs = [
            {
                "compartment": compartment_reference,
                "split_group": None,
                "label": str(compartment_reference or "reference"),
                "color": "#1f77b4",
            }
        ]
        for index, compartment in enumerate(design.get("interaction_compartments", []), start=1):
            compartment_text = str(compartment)
            if compartment_text not in {str(item["compartment"]) for item in series_specs}:
                series_specs.append(
                    {
                        "compartment": compartment_text,
                        "split_group": None,
                        "label": compartment_text,
                        "color": fallback_palette[index % len(fallback_palette)],
                    }
                )
        return series_specs
    return [{"compartment": None, "split_group": None, "label": "predicted mean", "color": "#1f77b4"}]

def _mixed_model_contrast_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    contrast_type = str(row.get("contrast_type"))
    if contrast_type == "state_pair":
        state_a = str(row.get("state_a"))
        state_b = str(row.get("state_b"))
        return (
            0,
            ALL_REQUESTED_STATES.index(state_a) if state_a in ALL_REQUESTED_STATES else len(ALL_REQUESTED_STATES),
            ALL_REQUESTED_STATES.index(state_b) if state_b in ALL_REQUESTED_STATES else len(ALL_REQUESTED_STATES),
            0,
            str(row.get("contrast_name")),
        )
    if contrast_type == "basal_apical":
        state = str(row.get("state"))
        return (
            1,
            ALL_REQUESTED_STATES.index(state) if state in ALL_REQUESTED_STATES else len(ALL_REQUESTED_STATES),
            0,
            0,
            str(row.get("contrast_name")),
        )
    if contrast_type == "split_group_pair":
        state = str(row.get("state"))
        return (
            2,
            ALL_REQUESTED_STATES.index(state) if state in ALL_REQUESTED_STATES else len(ALL_REQUESTED_STATES),
            0,
            0,
            str(row.get("contrast_name")),
        )
    if contrast_type == "visual_response_cohort":
        return (
            3,
            0,
            0,
            0,
            str(row.get("contrast_name")),
        )
    return (4, 0, 0, 0, str(row.get("contrast_name")))

def normalize_mixed_model_contrast_p_source(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else "classical"
    return text if text in {"classical", "shuffle"} else "classical"

def mixed_model_visual_response_state_order(
    state_order: Sequence[str],
    visual_response_cohort: Optional[str] = None,
) -> List[str]:
    states = [canonical_state_label(state) for state in state_order if canonical_state_label(state)]
    if str(visual_response_cohort or "").strip().lower() in {"responsive", "nonresponsive"}:
        states = [state for state in states if state != "quiet_awake_movies"]
    return list(dict.fromkeys(states))

def mixed_model_contrast_p_label(value: Any) -> str:
    return "shuffle p" if normalize_mixed_model_contrast_p_source(value) == "shuffle" else "classical p"


def normalize_non_negative_float(value: Any, default: float, field_name: str) -> float:
    parsed = as_float(value)
    if parsed is None:
        if value is None:
            parsed = float(default)
        else:
            raise SystemExit(f"Invalid {field_name}: {value!r}. Expected a non-negative float.")
    if not np.isfinite(parsed) or parsed < 0:
        raise SystemExit(f"Invalid {field_name}: {value!r}. Expected a non-negative float.")
    return float(parsed)


def normalize_positive_int(value: Any, default: int, field_name: str) -> int:
    parsed = as_int(value)
    if parsed is None:
        if value is None:
            parsed = int(default)
        else:
            raise SystemExit(f"Invalid {field_name}: {value!r}. Expected a positive integer.")
    if parsed <= 0:
        raise SystemExit(f"Invalid {field_name}: {value!r}. Expected a positive integer.")
    return int(parsed)


def apply_cpu_thread_limit(thread_limit: Any) -> int:
    """Limit native math-library threads for this process and subprocesses."""
    global _CPU_THREAD_LIMIT_CONTROLLER
    limit = normalize_positive_int(thread_limit, DEFAULT_CPU_THREAD_LIMIT, "cpu_thread_limit")
    for env_var in CPU_THREAD_LIMIT_ENV_VARS:
        os.environ[env_var] = str(limit)
    if threadpool_limits is not None:
        if _CPU_THREAD_LIMIT_CONTROLLER is not None:
            try:
                _CPU_THREAD_LIMIT_CONTROLLER.__exit__(None, None, None)
            except Exception:
                pass
        controller = threadpool_limits(limits=limit)
        controller.__enter__()
        _CPU_THREAD_LIMIT_CONTROLLER = controller
    return limit


def mixed_model_response_display_label(response: Any) -> str:
    response_text = str(response)
    return {
        "mean_dendrite_activity": "Dendrite mean dF/F",
        "mean_spine_activity_per_dendrite": "Spine-specific mean dF/F",
        "dendrite_event_frequency_per_min": "Dendrite calcium event frequency (per min)",
        "spine_event_frequency_per_min": "Spine calcium event frequency (per min)",
        "coincident_event_frequency_per_min": "Coincident spine event frequency (per min)",
        "noncoincident_event_frequency_per_min": "Noncoincident spine event frequency (per min)",
        "coactivity_r": "Spine coactivity coefficient",
    }.get(response_text, response_text.replace("_", " "))


def _mixed_model_contrast_label(row: Dict[str, Any]) -> str:
    contrast_name = str(row.get("contrast_name", "contrast"))
    p_source = normalize_mixed_model_contrast_p_source(row.get("p_value_source"))
    p_label = mixed_model_contrast_p_label(p_source)
    active_p = row.get("shuffle_p") if p_source == "shuffle" else row.get("shuffle_p", row.get("classical_p"))
    return f"{contrast_name}\n{p_label}={format_report_pvalue(active_p)}"
def _spine_coactivity_state_subject_values(
    rows: Sequence[Dict[str, Any]],
    state_labels: Sequence[str],
    value_key: str,
) -> Dict[str, Dict[str, List[float]]]:
    values_by_state: Dict[str, Dict[str, List[float]]] = {}
    for state in state_labels:
        subject_values: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            if canonical_state_label(row.get("state")) != state:
                continue
            value = as_float(row.get(value_key))
            if value is None or not np.isfinite(value):
                continue
            subject = str(row.get("global_pair_id") or row.get("day_id") or row.get("exp_id") or row.get("animal_id") or "unknown")
            subject_values[subject].append(float(value))
        values_by_state[state] = dict(subject_values)
    return values_by_state



def _spine_coactivity_pair_state_rows(
    results: Dict[str, Any],
    compartment_filter: Optional[str] = None,
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    coactivity = results.get("spine_coactivity", {})
    if not isinstance(coactivity, dict):
        return [], []
    source_rows = coactivity.get("pair_state_rows")
    if not isinstance(source_rows, list) or not source_rows:
        source_rows = coactivity.get("table_rows", [])
    rows = filter_rows_by_spine_coactivity(
        source_rows,
        compartment_filter=compartment_filter,
    )
    rows = [row for row in rows if str(row.get("status")) == "ok" and np.isfinite(as_float(row.get("coactivity_r")))]
    abs_threshold = spine_coactivity_abs_threshold_from_results(results)
    rows = [dict(row) for row in rows]
    for row in rows:
        row[SPINE_COACTIVITY_QUIET_ANCHOR_SELECTION_FIELD] = spine_coactivity_quiet_anchor_selected(row, abs_threshold)
    if anchor_state_filter is not None:
        anchor_state = canonical_state_label(anchor_state_filter)
        anchor_rows = [row for row in rows if canonical_state_label(row.get("state")) == anchor_state]
        if coactive_only:
            anchor_pair_ids = {
                str(row.get("global_pair_id"))
                for row in anchor_rows
                if bool(row.get(SPINE_COACTIVITY_QUIET_ANCHOR_SELECTION_FIELD))
            }
        else:
            anchor_pair_ids = {str(row.get("global_pair_id")) for row in anchor_rows}
        rows = [row for row in rows if str(row.get("global_pair_id")) in anchor_pair_ids]
    state_labels = [state for state in selected_matrix_state_labels(results) if any(canonical_state_label(row.get("state")) == state for row in rows)]
    if state_labels:
        rows = [row for row in rows if canonical_state_label(row.get("state")) in state_labels]
    return rows, state_labels

def _spine_coactivity_pair_state_display_label(row: Dict[str, Any]) -> str:
    _, day_date, _ = split_day_id(row.get("day_id") or row.get("exp_id") or "unknown")
    day_text = day_date or str(row.get("day_id") or row.get("exp_id") or "unknown")
    dendrite_text = extract_dendrite_token(row.get("global_dendrite_id"))
    spine_1 = str(row.get("global_spine_id_1") or "spine1").split("|")[-1]
    spine_2 = str(row.get("global_spine_id_2") or "spine2").split("|")[-1]
    return f"{day_text} | {dendrite_text} | {spine_1}-{spine_2}"


def plot_spine_coactivity_distribution_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "spine_coactivity_distribution.svg",
    title: str = "Spine coactivity coefficient distributions",
    compartment_filter: Optional[str] = None,
    state_filter: Optional[str] = None,
    coactive_only: bool = False,
    value_kind: str = "coactivity_r",
) -> Optional[str]:
    if plt is None:
        return None
    coactivity = results.get("spine_coactivity", {})
    if not isinstance(coactivity, dict):
        return None
    value_key = "coactivity_r" if value_kind != "shuffle_p" else "shuffle_p"
    rows = filter_rows_by_spine_coactivity(
        coactivity.get("table_rows", []),
        compartment_filter=compartment_filter,
        state_filter=state_filter,
        coactive_only=coactive_only,
    )
    rows = [row for row in rows if row.get("status") == "ok" and np.isfinite(as_float(row.get(value_key))) and np.isfinite(as_float(row.get("coactivity_r"))) ]
    if not rows:
        return None
    compartments = spine_coactivity_output_compartments(rows)
    if compartment_filter is None:
        if len(compartments) != 1:
            return None
        compartment = compartments[0]
    else:
        compartment = compartment_filter
    state_labels = selected_matrix_plot_state_labels(results, rows)
    if not state_labels:
        return None
    state_summary_lookup = {
        str(row.get("state")): dict(row)
        for row in coactivity.get("state_summary_rows", [])
        if str(row.get("compartment")) == compartment
    }
    fig_width = min(max(8.2, 0.58 * len(state_labels) + 2.9), 9.8)
    fig_height = min(max(4.5, 0.54 * len(state_labels) + 2.5), 8.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), squeeze=True)
    positions = np.arange(1, len(state_labels) + 1)
    series: List[np.ndarray] = []
    labels: List[str] = []
    for state in state_labels:
        state_rows = [row for row in rows if canonical_state_label(row.get("state")) == state]
        arr = np.asarray([as_float(row.get(value_key)) for row in state_rows if np.isfinite(as_float(row.get(value_key)))], dtype=float)
        if arr.size == 0:
            continue
        labels.append(state)
        series.append(arr)
    if not labels:
        return None
    used_positions = positions[: len(labels)]
    bp = ax.boxplot(series, positions=used_positions, widths=0.6, patch_artist=True, showfliers=False, vert=False)
    _set_boxplot_colors(bp, [state_display_color(state) for state in labels])
    for pos, arr, state in zip(used_positions, series, labels):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        jitter = np.random.default_rng(11 if value_key == "coactivity_r" else 12).uniform(-0.12, 0.12, size=finite.size)
        ax.scatter(finite, np.full(finite.size, pos) + jitter, s=12, alpha=0.45, color=state_display_color(state), edgecolor="none")
        ax.text(0.98, pos, f"n={finite.size}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=POSTER_NOTE_SIZE)
        if value_key == "shuffle_p" and np.nanmin(finite) < REPORT_SIGNIFICANCE_ALPHA:
            ax.scatter(min(1.0, np.nanmax(finite) + 0.04), pos, s=90, marker="*", color="#8b0000", zorder=4)
    if value_key == "coactivity_r":
        ax.axvline(0.0, color="#333333", linewidth=1)
        ax.set_xlabel("Coactivity coefficient", fontsize=POSTER_LABEL_SIZE)
    else:
        ax.axvline(REPORT_SIGNIFICANCE_ALPHA, color="#8b0000", linestyle="--", linewidth=1)
        ax.set_xlabel("Shuffle p", fontsize=POSTER_LABEL_SIZE)
    ax.set_title(title, fontsize=POSTER_TITLE_SIZE)
    ax.set_ylabel("State", fontsize=POSTER_LABEL_SIZE)
    set_requested_state_ticks(ax, labels, axis="y")
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=max(POSTER_FONT_SIZE, 14))
    ax.grid(axis="x", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    if value_key == "coactivity_r":
        active_state = format_requested_state_label(state_filter) if state_filter is not None else None
        active_text = f" | state={active_state}" if active_state is not None else ""
        active_text += f" | selected if {spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold_from_results(results))}" if coactive_only else ""
        ax.text(
            0.02,
            0.98,
            f"mean positive fraction = {format_report_number(next((as_float(row.get('positive_fraction')) for row in coactivity.get('state_summary_rows', []) if str(row.get('compartment')) == compartment), float('nan')))}{active_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
        )
    output_path = fig_dir / output_name
    fig.tight_layout(pad=0.95)
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)


def plot_spine_coactivity_tendency_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "spine_coactivity_heatmap.svg",
    title: str = "Spine coactivity coefficient comparisons across states",
    compartment_filter: Optional[str] = None,
    value_kind: str = "coactivity_r",
) -> Optional[str]:
    if plt is None:
        return None
    coactivity = results.get("spine_coactivity", {})
    if not isinstance(coactivity, dict):
        return None
    rows = filter_rows_by_spine_coactivity(coactivity.get("table_rows", []), compartment_filter=compartment_filter)
    rows = [row for row in rows if row.get("status") == "ok" and np.isfinite(as_float(row.get("coactivity_r")))]
    if not rows:
        return None
    state_labels = selected_matrix_plot_state_labels(results, rows)
    if not state_labels:
        return None
    compartments = spine_coactivity_output_compartments(rows)
    if compartment_filter is None:
        if len(compartments) != 1:
            return None
        compartment = compartments[0]
    else:
        compartment = compartment_filter
    shuffle_n = int(results.get("run_parameters", {}).get("shuffle_n", DEFAULT_SHUFFLES) or DEFAULT_SHUFFLES)
    values_by_state = _spine_coactivity_state_subject_values(rows, state_labels, "coactivity_r")
    from matplotlib.colors import Normalize
    try:
        from matplotlib.patches import Rectangle
    except Exception:
        Rectangle = None
    if value_kind == "shuffle_p":
        cmap = plt.get_cmap("viridis_r")
        norm = Normalize(vmin=0.0, vmax=1.0)
        colorbar_label = "Shuffle p"
        cell_value_label = "p"
    else:
        cmap = plt.get_cmap("coolwarm")
        norm = Normalize(vmin=-1.0, vmax=1.0)
        colorbar_label = "Coactivity coefficient change"
        cell_value_label = "Δ"
    fig = plt.figure(figsize=(min(max(6.2, 0.62 * len(state_labels) + 3.2), 11.0), min(max(5.4, 0.62 * len(state_labels) + 2.8), 9.5)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.08], wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    matrix = np.full((len(state_labels), len(state_labels)), np.nan, dtype=float)
    pair_obs_counts: Dict[Tuple[int, int], int] = {}
    pair_sig_counts: Dict[Tuple[int, int], int] = {}
    for i, state_a in enumerate(state_labels):
        for j, state_b in enumerate(state_labels):
            if state_a == state_b:
                continue
            comparison = paired_comparison(values_by_state, state_a, state_b, "coactivity_r", shuffle_n)
            if value_kind == "shuffle_p":
                value = as_float(comparison.get("shuffle_p"))
            else:
                value = as_float(comparison.get("effect_size"))
            if value is None or not np.isfinite(value):
                continue
            matrix[i, j] = float(value)
            pair_obs_counts[(i, j)] = int(comparison.get("n_subjects", 0))
            pair_sig_counts[(i, j)] = int(is_significant_row(comparison, p_key="shuffle_p"))
    if not np.isfinite(matrix).any():
        return None
    ax.text(
        0.02,
        0.98,
        f"n={len(rows)} | sig={sum(pair_sig_counts.values())}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    ax.set_title(title, fontsize=max(POSTER_TITLE_SIZE - 3, 16), pad=10)
    _configure_square_heatmap_axes(
        ax,
        state_labels,
        "State B",
        "State A",
        label_fontsize=max(8, POSTER_FONT_SIZE - 7),
        show_axis_labels=False,
    )
    ax.set_xticks(np.arange(-0.5, len(state_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(state_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, len(state_labels) - 0.5)
    ax.set_ylim(len(state_labels) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    if Rectangle is not None:
        for i, state_a in enumerate(state_labels):
            for j, state_b in enumerate(state_labels):
                if state_a == state_b:
                    continue
                value = matrix[i, j]
                if not np.isfinite(value):
                    continue
                obs_count = pair_obs_counts.get((i, j), 0)
                sig_count = pair_sig_counts.get((i, j), 0)
                square_size = 0.80 if sig_count == 0 else 0.56
                ax.add_patch(
                    Rectangle(
                        (j - square_size / 2.0, i - square_size / 2.0),
                        square_size,
                        square_size,
                        facecolor=cmap(norm(value)),
                        edgecolor="#1f1f1f",
                        linewidth=0.8,
                        zorder=3,
                    )
                )
                if sig_count > 0:
                    continue
                if square_size >= 0.52:
                    text_color = "white" if value_kind == "shuffle_p" and value <= 0.15 else ("white" if abs(value) >= 0.45 else "#111111")
                    label = format_report_pvalue(value) if value_kind == "shuffle_p" else format_report_number(value, precision=2)
                    ax.text(
                        j,
                        i,
                        f"{cell_value_label}={label}" if value_kind == "shuffle_p" else label,
                        ha="center",
                        va="center",
                        fontsize=max(8, POSTER_FONT_SIZE - 5),
                        color=text_color,
                        zorder=4,
                        clip_on=False,
                    )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(colorbar_label, fontsize=POSTER_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
    set_sparse_colorbar_ticks(cbar, nbins=5)
    output_path = fig_dir / output_name
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)


def plot_spine_coactivity_pair_state_heatmap_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "spine_coactivity_pair_state_heatmap.svg",
    title: str = "Spine coactivity coefficient across selected states",
    compartment_filter: Optional[str] = None,
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Optional[str]:
    if plt is None:
        return None
    rows, state_labels = _spine_coactivity_pair_state_rows(
        results,
        compartment_filter=compartment_filter,
        anchor_state_filter=anchor_state_filter,
        coactive_only=coactive_only,
    )
    if not rows or not state_labels:
        return None
    coactivity = results.get("spine_coactivity", {})
    pair_summary_rows = [row for row in coactivity.get("pair_summary_rows", []) if isinstance(row, dict)] if isinstance(coactivity, dict) else []
    if compartment_filter is not None:
        pair_summary_rows = [row for row in pair_summary_rows if str(row.get("compartment")) == compartment_filter]
    pair_summary_lookup = {str(row.get("global_pair_id")): dict(row) for row in pair_summary_rows}
    pair_rows_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_rows_by_id[str(row.get("global_pair_id"))].append(dict(row))
    pair_ids = list(pair_rows_by_id)
    if not pair_ids:
        return None

    def sort_key(pair_id: str) -> Tuple[int, float, int, float, str]:
        summary = pair_summary_lookup.get(pair_id, {})
        range_value = as_float(summary.get("coactivity_r_range"))
        profile_similarity = as_float(summary.get("profile_similarity_r"))
        return (
            0 if range_value is not None and np.isfinite(range_value) else 1,
            -(range_value if range_value is not None and np.isfinite(range_value) else float("-inf")),
            0 if profile_similarity is not None and np.isfinite(profile_similarity) else 1,
            -(profile_similarity if profile_similarity is not None and np.isfinite(profile_similarity) else float("-inf")),
            pair_id,
        )

    pair_ids = sorted(pair_ids, key=sort_key)
    matrix = np.full((len(pair_ids), len(state_labels)), np.nan, dtype=float)
    pair_labels: List[str] = []
    for row_index, pair_id in enumerate(pair_ids):
        pair_rows = pair_rows_by_id.get(pair_id, [])
        reference_row = pair_rows[0] if pair_rows else pair_summary_lookup.get(pair_id, {})
        pair_labels.append(_spine_coactivity_pair_state_display_label(reference_row))
        for col_index, state in enumerate(state_labels):
            state_value = next((as_float(row.get("coactivity_r")) for row in pair_rows if canonical_state_label(row.get("state")) == state), None)
            if state_value is not None and np.isfinite(state_value):
                matrix[row_index, col_index] = float(state_value)
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return None
    max_abs = float(np.nanmax(np.abs(finite_values))) if finite_values.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    fig_width = min(max(6.8, 0.68 * len(state_labels) + 3.6), 11.8)
    fig_height = min(max(5.0, 0.14 * len(pair_ids) + 3.0), 18.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), squeeze=True)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("State", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Spine pair", fontsize=POSTER_LABEL_SIZE)
    ax.set_xticks(np.arange(len(state_labels)))
    ax.set_xticklabels([format_requested_state_label(state) for state in state_labels], rotation=40, ha="right")
    color_state_tick_labels(ax, state_labels, axis="x")
    y_positions = np.arange(len(pair_labels))
    label_step = max(1, int(math.ceil(len(pair_labels) / 20.0)))
    sparse_y_labels = [label if (idx % label_step == 0 or idx in {0, len(pair_labels) - 1}) else "" for idx, label in enumerate(pair_labels)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sparse_y_labels, fontsize=max(8, POSTER_FONT_SIZE - 5))
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.set_xlim(-0.5, len(state_labels) - 0.5)
    ax.set_ylim(len(pair_labels) - 0.5, -0.5)
    ax.grid(which="major", color="white", linestyle="-", linewidth=0.6, alpha=0.55)
    anchor_text = f" | anchor={format_requested_state_label(anchor_state_filter)}" if anchor_state_filter is not None else ""
    active_text = f" | selected if {spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold_from_results(results))}" if coactive_only else ""
    ax.text(
        0.02,
        0.98,
        f"sorted by coactivity_r_range | n pairs = {len(pair_ids)}{anchor_text}{active_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coactivity coefficient", fontsize=POSTER_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
    set_sparse_colorbar_ticks(cbar, nbins=5)
    fig.tight_layout(pad=0.95)
    output_path = fig_dir / output_name
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)

def _spine_coactivity_basal_apical_distribution_rows(
    results: Dict[str, Any],
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[str]]:
    rows, state_labels = _spine_coactivity_pair_state_rows(
        results,
        compartment_filter=None,
        anchor_state_filter=anchor_state_filter,
        coactive_only=coactive_only,
    )
    compartment_state_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {"basal": defaultdict(list), "apical": defaultdict(list)}
    pair_values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    pair_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    selected_state_set = {canonical_state_label(state) for state in state_labels}
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment not in {"basal", "apical"}:
            continue
        state = canonical_state_label(row.get("state"))
        if state not in selected_state_set:
            continue
        value = as_float(row.get("coactivity_r"))
        if value is None or not np.isfinite(value):
            continue
        pair_id = str(row.get("global_pair_id") or "unknown")
        key = (compartment, state, pair_id)
        pair_values[key].append(float(value))
        pair_meta.setdefault(key, dict(row))
    for compartment in ["basal", "apical"]:
        for state in state_labels:
            pair_ids = sorted(pair_id for comp, state_label, pair_id in pair_values if comp == compartment and state_label == state)
            for pair_id in pair_ids:
                values = pair_values.get((compartment, state, pair_id), [])
                if not values:
                    continue
                meta = pair_meta.get((compartment, state, pair_id), {})
                compartment_state_rows[compartment][state].append(
                    {
                        "compartment": compartment,
                        "state": state,
                        "global_pair_id": pair_id,
                        "animal_id": meta.get("animal_id"),
                        "day_id": meta.get("day_id"),
                        "exp_id": meta.get("exp_id"),
                        "global_dendrite_id": meta.get("global_dendrite_id"),
                        "mean_coactivity_r": float(np.nanmean(values)),
                        "n_rows": int(len(values)),
                    }
                )
    return {compartment: dict(state_rows) for compartment, state_rows in compartment_state_rows.items()}, state_labels

def plot_spine_coactivity_basal_apical_distribution_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "spine_coactivity_basal_vs_apical_distribution.svg",
    title: str = "Quiet awake movies coactive-pair distribution - Basal vs apical",
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Optional[str]:
    if plt is None:
        return None
    compartment_state_rows, state_labels = _spine_coactivity_basal_apical_distribution_rows(
        results,
        anchor_state_filter=anchor_state_filter,
        coactive_only=coactive_only,
    )
    if not state_labels:
        return None
    compartment_order = [comp for comp in ["basal", "apical"] if any(compartment_state_rows.get(comp, {}).get(state) for state in state_labels)]
    if not compartment_order:
        return None
    fig_width = min(max(8.4, 0.75 * len(state_labels) + 4.2), 12.0)
    fig_height = min(max(4.4, 0.58 * len(state_labels) + 2.9), 10.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), squeeze=True)
    state_positions = np.arange(1, len(state_labels) + 1, dtype=float)
    compartment_colors = {"basal": "#1f77b4", "apical": "#d95f02"}
    compartment_offsets = {"basal": -0.16, "apical": 0.16}
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=compartment_colors.get(compartment, "#444444"), edgecolor="#444444", label=compartment.capitalize())
        for compartment in compartment_order
    ]
    for state_index, state in enumerate(state_labels):
        base_pos = state_positions[state_index]
        for compartment in compartment_order:
            pair_rows = compartment_state_rows.get(compartment, {}).get(state, [])
            values = np.asarray(
                [
                    as_float(row.get("mean_coactivity_r"))
                    for row in pair_rows
                    if np.isfinite(as_float(row.get("mean_coactivity_r")))
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            pos = base_pos + compartment_offsets.get(compartment, 0.0)
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=0.26,
                patch_artist=True,
                showfliers=False,
                vert=False,
            )
            _set_boxplot_colors(bp, [compartment_colors.get(compartment, "#444444")])
            jitter = np.random.default_rng(91 if compartment == "basal" else 92).uniform(-0.08, 0.08, size=values.size)
            ax.scatter(
                values,
                np.full(values.size, pos) + jitter,
                s=13,
                alpha=0.45,
                color=compartment_colors.get(compartment, "#444444"),
                edgecolor="none",
                zorder=3,
            )
            ax.text(
                0.98,
                pos,
                f"n pairs={values.size}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=POSTER_NOTE_SIZE,
            )
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("Mean coactivity coefficient per spine pair", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("State", fontsize=POSTER_LABEL_SIZE)
    ax.set_yticks(state_positions)
    ax.set_yticklabels([format_requested_state_label(state) for state in state_labels])
    ax.tick_params(axis="both", labelsize=max(POSTER_FONT_SIZE, 14))
    ax.grid(axis="x", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=POSTER_NOTE_SIZE)
    state_text = format_report_list([format_requested_state_label(state) for state in state_labels], max_items=6)
    anchor_text = format_requested_state_label(anchor_state_filter) if anchor_state_filter is not None else "selected states"
    selection_text = spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold_from_results(results))
    active_text = f" | selected if {selection_text}" if coactive_only else ""
    ax.text(
        0.02,
        0.98,
        f"anchor={anchor_text} | selected states={state_text} | n values are spine pairs{active_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    fig.tight_layout(pad=0.95)
    output_path = fig_dir / output_name
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)

def plot_spine_coactivity_pair_state_summary_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "spine_coactivity_pair_state_summary.svg",
    title: str = "Spine coactivity state-change summary",
    compartment_filter: Optional[str] = None,
    anchor_state_filter: Optional[str] = None,
    coactive_only: bool = False,
) -> Optional[str]:
    if plt is None:
        return None
    rows, state_labels = _spine_coactivity_pair_state_rows(
        results,
        compartment_filter=compartment_filter,
        anchor_state_filter=anchor_state_filter,
        coactive_only=coactive_only,
    )
    if not rows or not state_labels:
        return None
    coactivity = results.get("spine_coactivity", {})
    pair_summary_rows = [row for row in coactivity.get("pair_summary_rows", []) if isinstance(row, dict)] if isinstance(coactivity, dict) else []
    if compartment_filter is not None:
        pair_summary_rows = [row for row in pair_summary_rows if str(row.get("compartment")) == compartment_filter]
    pair_summary_lookup = {str(row.get("global_pair_id")): dict(row) for row in pair_summary_rows}
    def summary_sort_key(pair_id: str) -> Tuple[int, float, int, float, str]:
        range_value = as_float(pair_summary_lookup.get(pair_id, {}).get("coactivity_r_range"))
        profile_similarity = as_float(pair_summary_lookup.get(pair_id, {}).get("profile_similarity_r"))
        return (
            0 if range_value is not None and np.isfinite(range_value) else 1,
            -(range_value if range_value is not None and np.isfinite(range_value) else float("-inf")),
            0 if profile_similarity is not None and np.isfinite(profile_similarity) else 1,
            -(profile_similarity if profile_similarity is not None and np.isfinite(profile_similarity) else float("-inf")),
            pair_id,
        )
    pair_ids = sorted({str(row.get("global_pair_id")) for row in rows}, key=summary_sort_key)
    summary_rows: List[Dict[str, Any]] = []
    for pair_id in pair_ids:
        summary = dict(pair_summary_lookup.get(pair_id, {}))
        if not summary:
            pair_rows = [row for row in rows if str(row.get("global_pair_id")) == pair_id]
            values = np.asarray([as_float(row.get("coactivity_r")) for row in pair_rows if np.isfinite(as_float(row.get("coactivity_r")))], dtype=float)
            summary["coactivity_r_range"] = float(np.nanmax(values) - np.nanmin(values)) if values.size else float("nan")
            summary["mean_coactivity_r"] = float(np.nanmean(values)) if values.size else float("nan")
        summary_rows.append(summary)
    range_values = np.asarray([as_float(row.get("coactivity_r_range")) for row in summary_rows if np.isfinite(as_float(row.get("coactivity_r_range")))], dtype=float)
    similarity_values = np.asarray([as_float(row.get("profile_similarity_r")) for row in summary_rows if np.isfinite(as_float(row.get("profile_similarity_r")))], dtype=float)
    mean_values = np.asarray([as_float(row.get("mean_coactivity_r")) for row in summary_rows if np.isfinite(as_float(row.get("mean_coactivity_r")))], dtype=float)
    if range_values.size == 0 or similarity_values.size == 0:
        return None
    fig_width = min(max(6.3, 0.36 * max(len(summary_rows), 1) + 3.6), 10.2)
    fig_height = 5.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), squeeze=True)
    max_abs = float(np.nanmax(np.abs(mean_values))) if mean_values.size else 0.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    scatter = ax.scatter(
        [as_float(row.get("coactivity_r_range")) for row in summary_rows],
        [as_float(row.get("profile_similarity_r")) for row in summary_rows],
        c=[as_float(row.get("mean_coactivity_r")) for row in summary_rows],
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        s=36,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("Coactivity coefficient range", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Profile similarity r", fontsize=POSTER_LABEL_SIZE)
    ax.grid(alpha=0.2)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    set_sparse_numeric_ticks(ax, axis="y", nbins=5)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    x_min, x_max = padded_value_limits(range_values.tolist() + [0.0])
    y_min, y_max = padded_value_limits(similarity_values.tolist())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    anchor_text = f" | anchor={format_requested_state_label(anchor_state_filter)}" if anchor_state_filter is not None else ""
    active_text = f" | selected if {spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold_from_results(results))}" if coactive_only else ""
    ax.text(
        0.02,
        0.98,
        f"Each dot = one pair | n pairs = {len(summary_rows)} | rows sorted in heatmap by coactivity_r_range{anchor_text}{active_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean coactivity coefficient", fontsize=POSTER_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
    set_sparse_colorbar_ticks(cbar, nbins=5)
    fig.tight_layout(pad=0.95)
    output_path = fig_dir / output_name
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)

def plot_direct_trial_type_distribution_figure(

    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "direct_trial_type_distribution.svg",
    title: str = "Direct trial-type comparison - video means by state",
    model_key: str = "direct_trial_type_comparison",
) -> Optional[str]:
    if plt is None:
        return None
    direct = results.get(model_key, {})
    if not isinstance(direct, dict):
        return None
    rows = [row for row in direct.get("video_state_rows", []) if as_float(row.get("mean_response")) is not None]
    if not rows:
        return None
    state_labels = [state for state in selected_direct_trial_state_labels(results) if any(canonical_state_label(row.get("state")) == state for row in rows)]
    if not state_labels:
        return None
    has_split_groups = any(_direct_trial_type_split_group_value(row) is not None for row in rows)
    state_position_lookup = {state: position for position, state in enumerate(state_labels, start=1)}
    comparison_rows = [
        {
            "x1": float(state_position_lookup[canonical_state_label(row.get("state_a"))]),
            "x2": float(state_position_lookup[canonical_state_label(row.get("state_b"))]),
            "shuffle_p": row.get("shuffle_p"),
        }
        for row in direct.get("state_pair_rows", [])
        if is_significant_row(row)
        and canonical_state_label(row.get("state_a")) in state_position_lookup
        and canonical_state_label(row.get("state_b")) in state_position_lookup
    ]
    if has_split_groups:
        generated = plot_grouped_boxplot_series(
            rows,
            fig_dir,
            state_col="state",
            value_col="mean_response",
            state_order=state_labels,
            stem=Path(output_name).stem,
            title=title,
            ylabel="Mean trial response",
            xlabel="State",
            title_color="#334155",
            edge_color="#334155",
            group_col="split_group",
            state_label_col="state_display" if any("state_display" in row for row in rows) else None,
            state_color_col="state_color" if any("state_color" in row for row in rows) else None,
            group_label_col="split_group_display" if any("split_group_display" in row for row in rows) else None,
            group_color_col="split_group_color" if any("split_group_color" in row for row in rows) else None,
            group_rank_col="split_group_rank" if any("split_group_rank" in row for row in rows) else None,
            comparison_rows=None if len([split_group for split_group in _direct_trial_type_split_group_order(rows) if split_group is not None]) > 1 else comparison_rows,
            horizontal=True,
        )
        return str(generated[-1]) if generated else None
    state_data: List[np.ndarray] = []
    state_positions: List[int] = []
    state_labels_present: List[str] = []
    palette = plt.get_cmap("Dark2")
    for position, state in enumerate(state_labels, start=1):
        values = np.asarray([as_float(row.get("mean_response")) for row in rows if str(row.get("state")) == state], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        state_data.append(values)
        state_positions.append(position)
        state_labels_present.append(state)
    if not state_data:
        return None
    fig_width = min(max(7.6, 0.58 * len(state_labels_present) + 3.3), 10.2)
    fig_height = min(max(4.4, 0.52 * len(state_labels_present) + 2.6), 8.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bp = ax.boxplot(state_data, positions=state_positions, widths=0.62, patch_artist=True, showfliers=False, vert=False)
    _set_boxplot_colors(bp, [state_display_color(state) for state in state_labels_present])
    state_position_lookup = {state: position for state, position in zip(state_labels_present, state_positions)}
    for position, state, values in zip(state_positions, state_labels_present, state_data):
        jitter = np.random.default_rng(410 + position).uniform(-0.14, 0.14, size=values.size)
        ax.scatter(
            values,
            np.full(values.size, position) + jitter,
            s=20,
            alpha=0.70,
            color=state_display_color(state),
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.text(0.98, position, f"n={values.size}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=POSTER_NOTE_SIZE, clip_on=False)
    _draw_boxplot_significance_annotations(ax, comparison_rows, orientation="horizontal")
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title(title, fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("Mean trial response", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("State", fontsize=POSTER_LABEL_SIZE)
    set_requested_state_ticks(ax, state_labels_present, axis="y")
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.grid(axis="x", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="x", nbins=5)
    ax.text(
        0.02,
        0.98,
        "Each dot = one video ID averaged across animals",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=POSTER_NOTE_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
    )
    selection = results.get("analysis_state_selection", {}) if isinstance(results.get("analysis_state_selection", {}), dict) else {}
    state_mode = selection.get("state_mode")
    movie_trial_types = selection.get("movie_trial_types")
    subtitle = None
    if state_mode is not None or movie_trial_types:
        subtitle = f"state_mode={state_mode or 'n/a'} | movie_trial_types={format_report_list(movie_trial_types) if movie_trial_types else 'n/a'}"
        ax.text(
            0.02,
            0.90,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            color="#444444",
        )
    output_path = fig_dir / output_name
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)


def plot_direct_trial_type_state_comparison_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "direct_trial_type_state_comparison.svg",
    title: str = "Direct trial-type comparison - state pair scatter",
    model_key: str = "direct_trial_type_comparison",
) -> Optional[str]:
    if plt is None:
        return None
    direct = results.get(model_key, {})
    if not isinstance(direct, dict):
        return None
    video_state_rows_all = [row for row in direct.get("video_state_rows", []) if as_float(row.get("mean_response")) is not None]
    if not video_state_rows_all:
        return None
    split_group_order = _direct_trial_type_split_group_order(video_state_rows_all)
    has_split_groups = any(split_group is not None for split_group in split_group_order)
    component_dir = ensure_dir(Path(fig_dir) / f"{Path(output_name).stem}_components")
    panel_paths: List[Path] = []

    def _safe_suffix(split_group: str | None) -> str:
        if split_group is None:
            return ""
        text = str(split_group).strip()
        if not text:
            return ""
        return "_" + "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")

    def render_subset(split_group: str | None, rows_subset: Sequence[Mapping[str, Any]]) -> List[Path]:
        if not rows_subset:
            return []
        state_labels = [
            state
            for state in selected_direct_trial_state_labels(results)
            if any(canonical_state_label(row.get("state")) == state for row in rows_subset)
        ]
        if not state_labels:
            return []
        pair_order = [pair for pair in combinations(state_labels, 2)]
        if not pair_order:
            return []
        by_state_video: Dict[str, Dict[str, float]] = defaultdict(dict)
        for row in rows_subset:
            state = str(row.get("state"))
            video_id = str(row.get("video_id"))
            mean_response = as_float(row.get("mean_response"))
            if not state or not video_id or mean_response is None or not np.isfinite(mean_response):
                continue
            by_state_video[state][video_id] = float(mean_response)
        if not by_state_video:
            return []
        all_values: List[float] = []
        for state_a, state_b in pair_order:
            all_values.extend(list(by_state_video.get(state_a, {}).values()))
            all_values.extend(list(by_state_video.get(state_b, {}).values()))
        x_limits = padded_value_limits(all_values + [0.0])
        pair_rows = list(direct.get("state_pair_rows", []))
        if has_split_groups:
            pair_rows = [row for row in pair_rows if _direct_trial_type_split_group_value(row) == split_group]
        pair_lookup = {(str(row.get("state_a")), str(row.get("state_b"))): dict(row) for row in pair_rows}
        split_label = None
        if split_group is not None:
            split_label = str(rows_subset[0].get("split_group_display") or split_group).strip() or str(split_group)
        elif has_split_groups:
            split_label = "Unassigned"
        suffix = _safe_suffix(split_group if split_group is not None else split_label)
        local_paths: List[Path] = []

        def render_pair_panel(state_a: str, state_b: str, pair_row: Dict[str, Any], index: int) -> Optional[Path]:
            common_videos = sorted(set(by_state_video.get(state_a, {})).intersection(by_state_video.get(state_b, {})))
            if len(common_videos) < 1:
                return None
            x = np.asarray([by_state_video[state_a][video_id] for video_id in common_videos], dtype=float)
            y = np.asarray([by_state_video[state_b][video_id] for video_id in common_videos], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size == 0:
                return None
            fig, ax = plt.subplots(figsize=(4.0, 3.25))
            palette = plt.get_cmap("tab10")
            ax.scatter(
                x,
                y,
                s=34,
                alpha=0.82,
                color=palette(0),
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.axline((0.0, 0.0), slope=1.0, color="#666666", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_xlim(x_limits)
            ax.set_ylim(x_limits)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
            ax.grid(alpha=0.2)
            set_sparse_numeric_ticks(ax, axis="x", nbins=4)
            set_sparse_numeric_ticks(ax, axis="y", nbins=4)
            pair_label = f"{format_requested_state_label(state_a)} vs {format_requested_state_label(state_b)}"
            metrics = [f"n={x.size}"]
            effect_size = as_float(pair_row.get("effect_size"))
            shuffle_p = as_float(pair_row.get("shuffle_p"))
            agreement_r = as_float(pair_row.get("agreement_r"))
            if effect_size is not None:
                metrics.append(f"Δ={format_report_number(effect_size)}")
            if shuffle_p is not None:
                metrics.append(f"p={format_report_pvalue(shuffle_p)}")
            if agreement_r is not None:
                metrics.append(f"r={format_report_number(agreement_r)}")
            title_suffix = f" ({split_label})" if split_label else ""
            ax.set_title(f"{pair_label}{title_suffix}", fontsize=max(14, POSTER_TITLE_SIZE - 8), pad=3)
            ax.text(
                0.02,
                0.98,
                " | ".join(metrics),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=POSTER_NOTE_SIZE,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", alpha=0.85),
            )
            panel_path = component_dir / f"{Path(output_name).stem}_{index:02d}_{state_a}_vs_{state_b}{suffix}.svg"
            save_figure(fig, panel_path, extra_formats=())
            return panel_path

        for index, (state_a, state_b) in enumerate(pair_order, start=1):
            panel_path = render_pair_panel(state_a, state_b, pair_lookup.get((state_a, state_b), {}), index)
            if panel_path is not None:
                local_paths.append(panel_path)
        return local_paths

    for split_group in split_group_order:
        rows_subset = [row for row in video_state_rows_all if _direct_trial_type_split_group_value(row) == split_group] if has_split_groups else list(video_state_rows_all)
        panel_paths.extend(render_subset(split_group, rows_subset))
    if not panel_paths:
        return None
    return str(panel_paths[0])


def plot_mixed_model_forest_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "mixed_model_forest.svg",
    title: Optional[str] = None,
    model_key: str = "mixed_model",
) -> Optional[str]:
    if plt is None:
        return None

    mixed_model = results.get(model_key, {})
    if not isinstance(mixed_model, dict):
        return None

    summary_rows = mixed_model.get("summary_rows", {})
    if not isinstance(summary_rows, dict):
        return None

    preferred_responses = [
        "mean_activity",
        "mean_dendrite_activity",
        "mean_spine_activity_per_dendrite",
        "dendrite_event_frequency_per_min",
        "spine_event_frequency_per_min",
        "coincident_event_frequency_per_min",
        "noncoincident_event_frequency_per_min",
        "coactivity_r",
    ]

    response_order = [
        response
        for response in preferred_responses
        if isinstance(summary_rows.get(response), list) and summary_rows.get(response)
    ]
    for response in summary_rows:
        if (
            response not in response_order
            and isinstance(summary_rows.get(response), list)
            and summary_rows.get(response)
        ):
            response_order.append(response)

    if not response_order:
        return None

    payloads: List[Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]] = []

    for response in response_order:
        design, rows = _mixed_model_response_payload(results, response, model_key=model_key)
        if design is None or not rows:
            continue

        term_lookup = {str(row.get("term")): dict(row) for row in rows}
        fixed_effect_names = [str(term) for term in design.get("fixed_effect_names", [])]
        selected_state_set = {
            canonical_state_label(state)
            for state in selected_mixed_model_state_labels(results)
        }

        display_terms: List[str] = []
        for term in fixed_effect_names:
            kind = _mixed_model_term_kind(term)

            if kind == "state":
                if canonical_state_label(_mixed_model_term_value_label(term)) not in selected_state_set:
                    continue

            elif kind == "interaction":
                state_terms = [
                    part for part in str(term).split(":")
                    if part.startswith("state[")
                ]
                if state_terms and not any(
                    canonical_state_label(_mixed_model_term_value_label(part)) in selected_state_set
                    for part in state_terms
                ):
                    continue

            display_terms.append(term)

        if not display_terms:
            continue

        payloads.append((response, design, term_lookup, display_terms))

    if not payloads:
        return None

    component_dir = ensure_dir(Path(fig_dir) / f"{Path(output_name).stem}_components")
    panel_paths: List[Path] = []
    label_fontsize = max(9, POSTER_FONT_SIZE - 4)

    term_colors = {
        "intercept": "#7f7f7f",
        "state": "#1f77b4",
        "compartment": "#2ca02c",
        "split_group": "#9467bd",
        "interaction": "#ff7f0e",
        "cohort": "#17a2b8",
        "covariate": "#9467bd",
    }

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for label, color in [
            ("intercept", term_colors["intercept"]),
            ("state", term_colors["state"]),
            ("compartment", term_colors["compartment"]),
            ("interaction", term_colors["interaction"]),
            ("cohort", term_colors["cohort"]),
            ("split_group", term_colors["split_group"]),
            ("covariate", term_colors["covariate"]),
        ]
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=10,
            label="p < 0.05",
        )
    )

    figure_title = title or "Mixed-model fixed effects"
    figure_subtitle = (
        "Colors: intercept, state, compartment, split group, interaction, covariate; "
        "stars indicate p < 0.05."
    )

    for index, (response, design, term_lookup, fixed_effect_names) in enumerate(payloads, start=1):
        y_positions = np.arange(len(fixed_effect_names))[::-1]
        significance_map: Dict[str, bool] = {}
        panel_bounds: List[float] = []

        fig_width = min(max(6.4, 0.62 * max(len(fixed_effect_names), 1) + 2.4), 8.8)
        fig_height = min(max(0.42 * max(len(fixed_effect_names), 1) + 1.7, 3.8), 7.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.axvline(0.0, color="#333333", linewidth=1)

        for y_pos, term in zip(y_positions, fixed_effect_names):
            row = term_lookup.get(term, {})
            estimate = as_float(row.get("estimate"))
            se = as_float(row.get("se"))
            p_value = as_float(row.get("p_value"))

            significant = bool(
                p_value is not None
                and np.isfinite(p_value)
                and p_value < REPORT_SIGNIFICANCE_ALPHA
            )
            significance_map[term] = significant

            if estimate is None or not np.isfinite(estimate):
                continue

            ci = 1.96 * se if se is not None and np.isfinite(se) else float("nan")

            if np.isfinite(ci):
                panel_bounds.extend([estimate - ci, estimate + ci])
            else:
                panel_bounds.append(estimate)

            color = term_colors.get(_mixed_model_term_kind(term), term_colors["covariate"])

            if np.isfinite(ci):
                ax.errorbar(
                    estimate,
                    y_pos,
                    xerr=ci,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.5,
                    capsize=3,
                    zorder=1,
                )

            ax.scatter(
                estimate,
                y_pos,
                s=54,
                color=color,
                edgecolor="#222222" if significant else "white",
                linewidth=0.9,
                zorder=2,
            )

            if significant:
                ax.scatter(
                    estimate,
                    y_pos,
                    s=120,
                    marker="*",
                    color="#111111",
                    zorder=3,
                )

        panel_bounds.append(0.0)
        ax.set_xlim(padded_value_limits(panel_bounds))

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [_mixed_model_term_label(term) for term in fixed_effect_names],
            fontsize=label_fontsize,
        )

        for tick_label, term in zip(ax.get_yticklabels(), fixed_effect_names):
            if significance_map.get(term, False):
                tick_label.set_fontweight("bold")
                tick_label.set_color("#8b0000")

        ax.tick_params(axis="y", pad=4)
        ax.set_title(
            mixed_model_response_display_label(response),
            fontsize=max(16, POSTER_TITLE_SIZE - 7),
            pad=2,
        )
        ax.set_xlabel("Estimate (95% CI)", fontsize=max(17, POSTER_LABEL_SIZE - 1))
        ax.set_ylabel("Term", fontsize=max(15, POSTER_LABEL_SIZE - 3))
        ax.tick_params(axis="x", labelsize=POSTER_FONT_SIZE)
        ax.grid(axis="x", alpha=0.25)
        set_sparse_numeric_ticks(ax, axis="x", nbins=5)

        annotate_sample_size(
            ax,
            0.99,
            0.02,
            f"n={int(design.get('n_obs', 0) or 0)} ROI-day obs",
            ha="right",
            va="bottom",
            fontsize=POSTER_NOTE_SIZE,
            transform=ax.transAxes,
        )

        if index == 1:
            fig.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=max(16, POSTER_LEGEND_SIZE - 1),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(legend_handles),
                columnspacing=1.0,
                handletextpad=0.5,
            )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.95)

        panel_path = component_dir / f"{Path(output_name).stem}_{index:02d}_{response}.svg"
        save_figure(fig, panel_path, extra_formats=())
        panel_paths.append(panel_path)
        plt.close(fig)

    if not panel_paths:
        return None

    return str(panel_paths[0])

def plot_mixed_model_predicted_means_figure(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "mixed_model_predicted_means.svg",
    title: Optional[str] = None,
    model_key: str = "mixed_model",
) -> Optional[str]:
    if plt is None:
        return None
    mixed_model = results.get(model_key, {})
    if not isinstance(mixed_model, dict):
        return None
    summary_rows = mixed_model.get("summary_rows", {})
    if not isinstance(summary_rows, dict):
        return None
    preferred_responses = ["mean_activity", "mean_dendrite_activity", "mean_spine_activity_per_dendrite", "dendrite_event_frequency_per_min", "spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min", "coactivity_r"]
    response_order = [response for response in preferred_responses if isinstance(summary_rows.get(response), list) and summary_rows.get(response)]
    for response in summary_rows:
        if response not in response_order and isinstance(summary_rows.get(response), list) and summary_rows.get(response):
            response_order.append(response)
    if not response_order:
        return None
    model_equations = mixed_model.get("model_equations", {}) if isinstance(mixed_model.get("model_equations", {}), dict) else {}
    payloads: List[Dict[str, Any]] = []
    all_values: List[float] = []
    equation_lines: List[str] = []
    for response in response_order:
        design, rows = _mixed_model_response_payload(results, response, model_key=model_key)
        if design is None or not rows:
            continue
        fixed_effect_names = [str(term) for term in design.get("fixed_effect_names", [])]
        term_lookup = {str(row.get("term")): dict(row) for row in rows}
        beta = np.asarray([as_float(term_lookup.get(term, {}).get("estimate")) for term in fixed_effect_names], dtype=float)
        if beta.size == 0 or not np.all(np.isfinite(beta)):
            continue
        selected_state_set = {canonical_state_label(state) for state in selected_mixed_model_state_labels(results)}
        state_levels = [canonical_state_label(state) for state in design.get("state_levels", []) if canonical_state_label(state) in selected_state_set]
        if not state_levels:
            continue
        series_specs = _mixed_model_series_specs(design)
        observed_lookup = _mixed_model_observed_mean_payload(mixed_model, response, design)
        series_data: List[Dict[str, Any]] = []
        include_split_group = bool(design.get("include_split_group"))
        for series_spec in series_specs:
            compartment = series_spec.get("compartment")
            split_group = series_spec.get("split_group")
            label = str(series_spec.get("label") or "predicted mean")
            color = str(series_spec.get("color") or "#1f77b4")
            predicted: List[float] = []
            observed: List[float] = []
            observed_sem: List[float] = []
            for state in state_levels:
                predicted_value = float(np.dot(mixed_model_design_row(design, state, compartment, split_group=split_group), beta))
                predicted.append(predicted_value)
                all_values.append(predicted_value)
                lookup_key = (str(split_group) if include_split_group else str(compartment) if compartment is not None else None, state)
                obs = observed_lookup.get(lookup_key)
                if obs is None:
                    observed.append(float("nan"))
                    observed_sem.append(float("nan"))
                else:
                    observed.append(float(obs["mean"]))
                    observed_sem.append(float(obs["sem"]))
                    if np.isfinite(obs["mean"]):
                        all_values.append(float(obs["mean"]))
            series_data.append(
                {
                    "compartment": compartment,
                    "split_group": split_group,
                    "label": label,
                    "color": color,
                    "predicted": predicted,
                    "observed": observed,
                    "observed_sem": observed_sem,
                }
            )
        payloads.append(
            {
                "response": response,
                "state_levels": state_levels,
                "series_data": series_data,
            }
        )
        equation = str(model_equations.get(response, "")).strip()
        if equation:
            equation_lines.append(textwrap.fill(f"{mixed_model_response_display_label(response)}: {equation}", width=96))
    if not payloads:
        return None
    all_values.append(0.0)
    y_limits = padded_value_limits(all_values)
    component_dir = ensure_dir(Path(fig_dir) / f"{Path(output_name).stem}_components")
    panel_paths: List[Path] = []
    figure_title = title or "Mixed-model predicted means"
    figure_subtitle = "Solid filled = model; open dashed = observed."
    for index, payload in enumerate(payloads, start=1):
        response = str(payload["response"])
        state_levels = list(payload["state_levels"])
        series_data = list(payload["series_data"])
        fig_width = min(max(5.6, 0.46 * max(len(state_levels), 1) + 2.1), 7.4)
        fig_height = min(max(1.25 * max(len(series_data), 1) + 0.55, 3.6), 5.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        x_positions = np.arange(len(state_levels))
        for series_index, series in enumerate(series_data):
            compartment = series["compartment"]
            label = str(series["label"])
            color = series.get("color") or plt.get_cmap("tab10")(series_index % 10)
            predicted = np.asarray(series["predicted"], dtype=float)
            observed = np.asarray(series["observed"], dtype=float)
            observed_sem = np.asarray(series["observed_sem"], dtype=float)
            ax.plot(x_positions, predicted, marker="o", linewidth=2.4, markersize=5.8, color=color, label=label, zorder=2)
            if np.isfinite(observed).any():
                yerr = observed_sem.copy()
                yerr[~np.isfinite(yerr)] = 0.0
                ax.errorbar(
                    x_positions,
                    observed,
                    yerr=yerr,
                    fmt="o--",
                    markersize=5.0,
                    mfc="white",
                    mec=color,
                    mew=1.1,
                    color=color,
                    alpha=0.85,
                    linewidth=1.3,
                    capsize=3,
                    zorder=3,
                )
        ax.set_xticks(x_positions)
        ax.set_xticklabels([format_requested_state_label(state) for state in state_levels], rotation=28, ha="right")
        color_state_tick_labels(ax, state_levels, axis="x")
        ax.set_xlabel("State")
        ax.set_ylabel(f"Predicted {mixed_model_response_display_label(response)}", fontsize=POSTER_LABEL_SIZE)
        ax.set_title(mixed_model_response_display_label(response), fontsize=max(15, POSTER_TITLE_SIZE - 7), pad=2)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
        if len(series_data) > 1:
            ax.legend(frameon=False, fontsize=max(16, POSTER_LEGEND_SIZE - 1), loc="upper left")
        ax.set_ylim(y_limits)
        annotate_sample_size(
            ax,
            0.99,
            0.98,
            f"n={int(design.get('n_obs', 0) or 0)} ROI-day obs",
            ha="right",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            transform=ax.transAxes,
            bbox=True,
        )
        fig.tight_layout(pad=0.95)
        panel_path = component_dir / f"{Path(output_name).stem}_{index:02d}_{response}.svg"
        save_figure(fig, panel_path, extra_formats=())
        panel_paths.append(panel_path)
    if not panel_paths:
        return None
    return str(panel_paths[0])
def plot_mixed_model_contrasts_checkpoint(
    results: Dict[str, Any],
    fig_dir: Path,
    scope: str,
    output_name: Optional[str] = None,
    title: Optional[str] = None,
    model_key: str = "mixed_model",
) -> Optional[str]:
    if plt is None:
        return None
    mixed_model = results.get(model_key, {})
    selected_state_set = {canonical_state_label(state) for state in selected_mixed_model_state_labels(results)}
    selected_basal_apical_set = {canonical_state_label(state) for state in selected_basal_apical_state_labels(results)}
    rows = [
        row
        for row in mixed_model.get("contrast_rows", [])
        if str(row.get("scope")) == scope
        and (
            str(row.get("contrast_type")) not in {"state_pair", "basal_apical"}
            or (str(row.get("contrast_type")) == "state_pair" and canonical_state_label(row.get("state_a")) in selected_state_set and canonical_state_label(row.get("state_b")) in selected_state_set)
            or (str(row.get("contrast_type")) == "basal_apical" and canonical_state_label(row.get("state")) in selected_basal_apical_set)
        )
    ]
    if not rows:
        return None
    preferred_responses = ["mean_activity", "mean_dendrite_activity", "mean_spine_activity_per_dendrite", "dendrite_event_frequency_per_min", "spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min", "coactivity_r"]
    response_order = [response for response in preferred_responses if any(row.get("response") == response for row in rows)]
    for response in sorted({str(row.get("response")) for row in rows}):
        if response not in response_order:
            response_order.append(response)
    if not response_order:
        return None
    designs_lookup = mixed_model.get("designs", {}) if isinstance(mixed_model.get("designs", {}), dict) else {}
    state_order_lookup = {state: idx for idx, state in enumerate(selected_mixed_model_state_labels(results))}
    def contrast_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
        contrast_type = str(row.get("contrast_type"))
        if contrast_type == "state_pair":
            state_a = canonical_state_label(row.get("state_a"))
            state_b = canonical_state_label(row.get("state_b"))
            return (
                0,
                state_order_lookup.get(state_a, len(state_order_lookup)),
                state_order_lookup.get(state_b, len(state_order_lookup)),
                0,
                str(row.get("contrast_name")),
            )
        if contrast_type == "basal_apical":
            state = canonical_state_label(row.get("state"))
            return (
                1,
                state_order_lookup.get(state, len(state_order_lookup)),
                0,
                0,
                str(row.get("contrast_name")),
            )
        if contrast_type == "split_group_pair":
            state = canonical_state_label(row.get("state"))
            return (
                2,
                state_order_lookup.get(state, len(state_order_lookup)),
                0,
                0,
                str(row.get("contrast_name")),
            )
        return (3, 0, 0, 0, str(row.get("contrast_name")))
    ordered_rows = sorted(rows, key=contrast_sort_key)
    contrast_labels = [_mixed_model_contrast_label(row) for row in ordered_rows]
    type_colors = {
        "state_pair": "#1f77b4",
        "basal_apical": "#d95f02",
        "split_group_pair": "#9467bd",
        "visual_response_cohort": "#17a2b8",
    }
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=type_colors["state_pair"], markeredgecolor="white", markersize=8, label="state pair"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=type_colors["basal_apical"], markeredgecolor="white", markersize=8, label="basal/apical"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=type_colors["split_group_pair"], markeredgecolor="white", markersize=8, label="split group"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=type_colors["visual_response_cohort"], markeredgecolor="white", markersize=8, label="visual response cohort"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=10, label="p < 0.05"),
    ]
    component_dir = ensure_dir(Path(fig_dir) / f"{Path(output_name or f'07_mixed_model_contrasts_{scope}.svg').stem}_components")
    panel_paths: List[Path] = []
    p_source = normalize_mixed_model_contrast_p_source(mixed_model.get("p_value_source", "classical"))
    p_label = mixed_model_contrast_p_label(p_source)
    figure_title = title or f"Mixed-model contrasts - {scope.replace('_', ' ')}"
    figure_subtitle = f"Left: estimate with 95% CI; right: {p_label}."
    for index, response in enumerate(response_order, start=1):
        subset = [row for row in ordered_rows if str(row.get("response")) == response]
        if not subset:
            continue
        subset_lookup = {str(row.get("contrast_name")): dict(row) for row in subset}
        subset_labels = [_mixed_model_contrast_label(row) for row in subset]
        estimate_bounds: List[float] = []
        p_bounds: List[float] = []
        for row in subset:
            estimate = as_float(row.get("estimate"))
            se = as_float(row.get("se"))
            shuffle_p = as_float(row.get("shuffle_p"))
            if estimate is not None and np.isfinite(estimate):
                if se is not None and np.isfinite(se):
                    ci = 1.96 * se
                    estimate_bounds.extend([estimate - ci, estimate + ci])
                else:
                    estimate_bounds.append(estimate)
            if shuffle_p is not None and np.isfinite(shuffle_p) and shuffle_p > 0:
                p_bounds.append(-np.log10(np.clip(shuffle_p, 1e-300, 1.0)))
        estimate_bounds.append(0.0)
        p_bounds.append(0.0)
        estimate_limits = padded_value_limits(estimate_bounds)
        p_max = max([bound for bound in p_bounds if np.isfinite(bound)], default=0.0)
        p_limit = max(0.1, p_max * 1.1)
        y_positions = np.arange(len(subset))[::-1]
        fig_width = min(max(7.4, POSTER_DENSE_FIGSIZE[0] - 2.2), 9.2)
        fig_height = min(max(0.36 * max(len(subset), 1) + 2.4, 4.8), 7.6)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), squeeze=False, gridspec_kw={"wspace": 0.26})
        ax_est = axes[0, 0]
        ax_sig = axes[0, 1]
        ax_est.axvline(0.0, color="#333333", linewidth=1)
        ax_sig.axvline(-np.log10(0.05), color="#8b0000", linestyle="--", linewidth=1.2)
        for y_pos, row in zip(y_positions, subset):
            row_data = subset_lookup.get(str(row.get("contrast_name")))
            color = type_colors.get(str(row.get("contrast_type")), "#7f7f7f")
            if row_data is not None:
                estimate = as_float(row_data.get("estimate"))
                se = as_float(row_data.get("se"))
                if estimate is not None and np.isfinite(estimate):
                    ci = 1.96 * se if se is not None and np.isfinite(se) else float("nan")
                    if np.isfinite(ci):
                        ax_est.errorbar(estimate, y_pos, xerr=ci, fmt="none", ecolor=color, elinewidth=1.5, capsize=3, zorder=1)
                    ax_est.scatter(estimate, y_pos, s=56, color=color, edgecolor="#222222", linewidth=0.8, zorder=2)
                p_value = as_float(row_data.get("shuffle_p"))
                if p_value is not None and np.isfinite(p_value) and p_value > 0:
                    neglog = -np.log10(np.clip(p_value, 1e-300, 1.0))
                    ax_sig.barh(y_pos, neglog, color=color, alpha=0.88)
        ax_est.set_xlim(estimate_limits)
        ax_est.set_yticks(y_positions)
        ax_est.set_yticklabels(subset_labels)
        for tick_label, row in zip(ax_est.get_yticklabels(), subset):
            row_data = subset_lookup.get(str(row.get("contrast_name")), {})
            active_p = as_float(row_data.get("shuffle_p"))
            if active_p is not None and np.isfinite(active_p) and active_p < REPORT_SIGNIFICANCE_ALPHA:
                tick_label.set_fontweight("bold")
                tick_label.set_color("#8b0000")
        ax_est.set_xlabel("Estimate (95% CI)", fontsize=max(18, POSTER_LABEL_SIZE - 1))
        ax_est.set_ylabel("Contrast", fontsize=max(16, POSTER_LABEL_SIZE - 2))
        ax_est.set_title(mixed_model_response_display_label(response), fontsize=max(15, POSTER_TITLE_SIZE - 7), pad=2)
        ax_est.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
        ax_est.grid(axis="x", alpha=0.25)
        set_sparse_numeric_ticks(ax_est, axis="x", nbins=5)
        ax_sig.set_yticks(y_positions)
        ax_sig.set_yticklabels([])
        ax_sig.tick_params(axis="y", left=False, labelleft=False)
        ax_sig.set_xlabel(r"$-\log_{10}(p)$", fontsize=max(17, POSTER_LABEL_SIZE - 2))
        ax_sig.set_title(f"{p_label}", fontsize=max(14, POSTER_TITLE_SIZE - 8), pad=2)
        design = designs_lookup.get(response, {}) if isinstance(designs_lookup, dict) else {}
        annotate_sample_size(
            ax_est,
            0.02,
            0.98,
            f"n={int(design.get('n_obs', 0) or 0)} ROI-day obs",
            ha="left",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            transform=ax_est.transAxes,
            bbox=True,
        )
        ax_sig.tick_params(axis="x", labelsize=POSTER_FONT_SIZE)
        ax_sig.grid(axis="x", alpha=0.25)
        ax_sig.set_xlim(0.0, p_limit)
        set_sparse_numeric_ticks(ax_sig, axis="x", nbins=5)
        if index == 1:
            fig.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=max(16, POSTER_LEGEND_SIZE - 1),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(legend_handles),
                columnspacing=1.0,
                handletextpad=0.5,
            )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.95)
        panel_path = component_dir / f"{Path(output_name or f'07_mixed_model_contrasts_{scope}.svg').stem}_{index:02d}_{response}.svg"
        save_figure(fig, panel_path, extra_formats=())
        panel_paths.append(panel_path)
    if not panel_paths:
        return None
    return str(panel_paths[0])
def plot_state_summary_figure(
    results,
    output_dir,
    *,
    output_name="state_summary.svg",
    title="State summary distributions",
    state_labels=None,
    y_limits=None,
    comparison_rows=None,
    cohort_label="all",
    state_group=DEFAULT_STATE_SUMMARY_FIGURES_SUBDIRNAME,
    **_compat_kwargs,
):
    """Backward-compatible public wrapper for the old state-summary API."""
    if isinstance(results, dict) and results.get("analysis_branch_name") and results.get("analysis_basis_name"):
        roi_split_results = results.get("roi_split", {})
        split_rows = roi_split_results.get("subject_state_rows", []) if isinstance(roi_split_results, dict) else []
        split_rows = [dict(row) for row in split_rows if isinstance(row, Mapping)]
        if split_rows and isinstance(roi_split_results, dict):
            split_rows = annotate_rows_with_split_group(split_rows, roi_split_results.get("membership_rows", []))
        if split_rows:
            state_order = list(state_labels) if state_labels is not None else list(DEFAULT_BASAL_APICAL_STATES)
            output_paths: List[Path] = []
            metric_specs = {
                "dendrite_mean": "Dendrite mean dF/F",
                "spine_specific_mean": "Spine-specific mean dF/F",
                "dendrite_event_frequency_per_min": "Dendrite calcium event frequency (per min)",
                "spine_event_frequency_per_min": "Spine calcium event frequency (per min)",
                "coincident_event_frequency_per_min": "Coincident spine event frequency (per min)",
                "noncoincident_event_frequency_per_min": "Noncoincident spine event frequency (per min)",
            }
            for metric_name, metric_title in metric_specs.items():
                metric_rows = [
                    dict(row)
                    for row in split_rows
                    if np.isfinite(as_float(row.get(metric_name)))
                    and canonical_state_label(row.get("state")) in {canonical_state_label(state) for state in state_order}
                ]
                if not metric_rows:
                    continue
                metric_comparison_rows = comparison_rows
                if metric_comparison_rows is None:
                    metric_comparison_rows = _state_summary_significant_pair_rows(
                        results.get("state_comparisons", []),
                        metric_name=metric_name,
                        state_order=state_order,
                        comparison_name="state_comparison",
                    )
                metric_output_path = state_summary_metric_output_dir(
                    output_dir,
                    metric_name,
                    cohort_label,
                    state_group,
                ) / f"{Path(output_name).stem}_{metric_name}.svg"
                plotted = plot_grouped_boxplot_series(
                    metric_rows,
                    metric_output_path.parent,
                    state_col="state",
                    value_col=metric_name,
                    state_order=state_order,
                    stem=metric_output_path.stem,
                    title=metric_title,
                    ylabel="Dendrite dF/F",
                    xlabel="State",
                    title_color="#334155",
                    edge_color="#334155",
                    group_col="split_group",
                    state_label_col="state_display",
                    state_color_col="state_color",
                    group_label_col="split_group_display",
                    group_color_col="split_group_color",
                    group_rank_col="split_group_rank",
                    comparison_rows=metric_comparison_rows,
                )
                if plotted:
                    output_paths.append(metric_output_path)
            return str(output_paths[0]) if output_paths else None
    if isinstance(results, dict) and isinstance(results.get("state_summaries"), dict) and not isinstance(results.get("apical_results"), dict):
        state_order = list(state_labels) if state_labels is not None else list(DEFAULT_BASAL_APICAL_STATES)
        output_paths: List[Path] = []
        for metric_name, metric_title in {
            "dendrite_mean": "Dendrite mean dF/F",
            "spine_specific_mean": "Spine-specific mean dF/F",
            "dendrite_event_frequency_per_min": "Dendrite calcium event frequency (per min)",
            "spine_event_frequency_per_min": "Spine calcium event frequency (per min)",
            "coincident_event_frequency_per_min": "Coincident spine event frequency (per min)",
            "noncoincident_event_frequency_per_min": "Noncoincident spine event frequency (per min)",
        }.items():
            summary = results.get("state_summaries", {}).get(metric_name, {})
            metric_comparison_rows = comparison_rows
            if metric_comparison_rows is None:
                metric_comparison_rows = _state_summary_significant_pair_rows(
                    results.get("state_comparisons", []),
                    metric_name=metric_name,
                    state_order=state_order,
                    comparison_name="state_comparison",
                )
            panel_fig = _render_state_summary_single_panel_figure(
                metric_name,
                metric_title,
                summary,
                state_order,
                y_limits.get(metric_name) if y_limits else None,
                comparison_rows=metric_comparison_rows,
            )
            if panel_fig is None:
                continue
            metric_output_path = state_summary_metric_output_dir(
                output_dir,
                metric_name,
                cohort_label,
                state_group,
            ) / f"{Path(output_name).stem}_{metric_name}.svg"
            save_figure(panel_fig, metric_output_path, extra_formats=())
            output_paths.append(metric_output_path)
        return str(output_paths[0]) if output_paths else None
    return plot_state_summary_compartment_comparison_figure(
        results,
        None,
        output_dir,
        output_name=output_name,
        title=title,
        state_labels=state_labels,
        y_limits=y_limits,
        comparison_rows=comparison_rows,
        cohort_label=cohort_label,
        state_group=state_group,
    )
   











# Restored event-run helper dependencies from commit 64f20508a23d

def _event_run_center(run: Tuple[int, int]) -> float:
    return 0.5 * (float(int(run[0])) + float(int(run[1])))


def _event_run_overlaps_window(window_start: int, window_end: int, run: Tuple[int, int]) -> bool:
    return bool(max(int(window_start), int(run[0])) < min(int(window_end), int(run[1])))


# Restored event-window helper dependencies from commit 64f20508a23d

def _window_overlaps_any(window_start: int, window_end: int, runs: Sequence[Tuple[int, int]]) -> bool:
    return any(_event_run_overlaps_window(window_start, window_end, run) for run in runs)


def _select_event_example_windows(
    trace_size: int,
    event_runs: Sequence[Tuple[int, int]],
    *,
    max_examples: int = 10,
    pad_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    trace_size = int(trace_size)
    if trace_size <= 0:
        return []
    if pad_frames is None:
        pad_frames = max(4, min(25, max(6, trace_size // 12)))
    pad_frames = int(max(1, pad_frames))
    selected: List[Dict[str, Any]] = []
    ordered_runs = sorted([(int(start), int(end)) for start, end in event_runs], key=_event_run_center)
    if ordered_runs:
        if len(ordered_runs) > max_examples:
            indices = np.unique(np.round(np.linspace(0, len(ordered_runs) - 1, max_examples)).astype(int))
            ordered_runs = [ordered_runs[index] for index in indices[:max_examples]]
        for run in ordered_runs:
            window_start = max(0, int(run[0]) - pad_frames)
            window_end = min(trace_size, int(run[1]) + pad_frames)
            if window_end <= window_start:
                continue
            selected.append(
                {
                    "kind": "event",
                    "label": "event",
                    "run": run,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )
    context_centers = np.linspace(0, max(trace_size - 1, 0), max_examples * 4 if trace_size > 1 else 1)
    for center_value in context_centers:
        if len(selected) >= max_examples:
            break
        center = int(round(float(center_value)))
        window_start = max(0, center - pad_frames)
        window_end = min(trace_size, center + pad_frames + 1)
        if window_end <= window_start:
            continue
        if _window_overlaps_any(window_start, window_end, ordered_runs):
            continue
        if any(_window_overlaps_any(window_start, window_end, [item["run"]]) for item in selected if item.get("run") is not None):
            continue
        selected.append(
            {
                "kind": "context",
                "label": "context",
                "run": None,
                "window_start": window_start,
                "window_end": window_end,
            }
        )
    if not selected:
        selected.append(
            {
                "kind": "context",
                "label": "context",
                "run": None,
                "window_start": 0,
                "window_end": trace_size,
            }
        )
    selected = sorted(selected, key=lambda item: (int(item["window_start"]), int(item["window_end"])))
    if len(selected) > max_examples:
        selected = selected[:max_examples]
    while len(selected) < max_examples:
        selected.append(dict(selected[-1]))
    return selected


# Restored second-order figure/helper dependencies from commit 64f20508a23d

def state_summary_metric_family(metric_name: str) -> str:
    return "dendrites" if metric_name in STATE_SUMMARY_DENDRITE_METRICS else "spines"


def _build_event_detection_example_figure(
    *,
    time: np.ndarray,
    trace: np.ndarray,
    event_info: Dict[str, Any],
    title: str,
    trace_label: str,
    trace_kind: str,
    dendrite_event_info: Optional[Dict[str, Any]] = None,
    dendrite_trace: Optional[np.ndarray] = None,
    dendrite_time: Optional[np.ndarray] = None,
) -> Optional[Any]:
    if plt is None:
        return None

    time = np.asarray(time, dtype=float).ravel()
    trace = np.asarray(trace, dtype=float).ravel()
    if time.size != trace.size:
        time = np.arange(trace.size, dtype=float)

    valid = np.isfinite(time) & np.isfinite(trace)
    if not np.any(valid):
        return None

    event_info = event_info or {}
    dendrite_event_info = dendrite_event_info or {}

    event_method = str(event_info.get("method") or "amplitude").strip().lower()
    if event_method not in EVENT_DETECTION_METHODS:
        event_method = "amplitude"
    display_trace = np.diff(trace, prepend=np.nan) if event_method == "derivative" else trace
    display_trace_label = f"First derivative of {trace_label}" if event_method == "derivative" else trace_label
    dendrite_display_trace = None
    dendrite_display_label = "Dendrite dF/F"

    threshold = as_float(event_info.get("threshold"))
    event_runs = [(int(start), int(end)) for start, end in (event_info.get("event_runs") or [])]
    dendrite_event_runs = [(int(start), int(end)) for start, end in (dendrite_event_info.get("event_runs") or [])]

    selected_windows = _select_event_example_windows(trace.size, event_runs, max_examples=10)
    if not selected_windows:
        return None

    if trace_kind == "spine":
        if dendrite_trace is None:
            dendrite_trace = np.asarray(
                dendrite_event_info.get("trace") if isinstance(dendrite_event_info, dict) else None,
                dtype=float,
            )
        else:
            dendrite_trace = np.asarray(dendrite_trace, dtype=float).ravel()

        if dendrite_time is None:
            dendrite_time = time
        else:
            dendrite_time = np.asarray(dendrite_time, dtype=float).ravel()

        if dendrite_trace is None or dendrite_trace.size == 0:
            dendrite_trace = None
            dendrite_time = None
        elif dendrite_time.size != dendrite_trace.size:
            dendrite_time = np.arange(dendrite_trace.size, dtype=float)
        if dendrite_trace is not None:
            dendrite_display_trace = np.diff(dendrite_trace, prepend=np.nan) if event_method == "derivative" else dendrite_trace
            dendrite_display_label = "First derivative of Dendrite dF/F" if event_method == "derivative" else "Dendrite dF/F"
    else:
        dendrite_trace = None
        dendrite_time = None

    trace_color = "#4477aa" if trace_kind == "dendrite" else "#7a5195"
    spine_color = "#7a5195"
    dendrite_color = "#4477aa"
    event_color = "#8b0000"
    spine_event_color = "#d62728"
    noncoincident_color = "#f58518"

    compact_rc = {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 16,
    }

    with plt.rc_context(compact_rc):
        fig, axes = plt.subplots(5, 2, figsize=(11.5, 13.0), squeeze=False)
        axes_flat = axes.ravel()

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        for idx, window in enumerate(selected_windows):
            ax = axes_flat[idx]
            row = idx // 2
            col = idx % 2

            window_start = int(window["window_start"])
            window_end = int(window["window_end"])

            window_time = time[window_start:window_end]
            window_trace = display_trace[window_start:window_end]
            window_valid = np.isfinite(window_time) & np.isfinite(window_trace)

            if not np.any(window_valid):
                ax.text(
                    0.5,
                    0.5,
                    "No valid signal",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.set_axis_off()
                continue

            window_runs = [run for run in event_runs if _window_overlaps_any(window_start, window_end, [run])]

            # Use the first event in the window as t = 0.
            # If there is no event, center the context window.
            if window_runs:
                reference_idx = int(window_runs[0][0])
                reference_time = float(time[reference_idx]) if 0 <= reference_idx < time.size else float(window_time[0])
            else:
                reference_time = float(np.nanmedian(window_time[window_valid]))

            window_time_rel = window_time - reference_time

            if trace_kind == "spine":
                ax_spine = ax
                ax_dendrite = ax_spine.twinx()

                ax_spine.plot(
                    window_time_rel[window_valid],
                    window_trace[window_valid],
                    color=spine_color,
                    linewidth=1.1,
                    label=display_trace_label,
                )

                if dendrite_trace is not None and dendrite_time is not None and dendrite_time.size > 0:
                    dend_window_trace = dendrite_display_trace[window_start:window_end]
                    dend_window_time = dendrite_time[window_start:window_end]
                    dend_valid = np.isfinite(dend_window_time) & np.isfinite(dend_window_trace)

                    if np.any(dend_valid):
                        ax_dendrite.plot(
                            dend_window_time[dend_valid] - reference_time,
                            dend_window_trace[dend_valid],
                            color=dendrite_color,
                            linewidth=1.0,
                            alpha=0.9,
                            label=dendrite_display_label,
                        )

                if np.isfinite(threshold):
                    ax_spine.axhline(
                        threshold,
                        color=event_color,
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.9,
                    )

                dend_window_runs = [
                    run for run in dendrite_event_runs
                    if _window_overlaps_any(window_start, window_end, [run])
                ]

                for run in dend_window_runs:
                    start_i, end_i = run
                    if 0 <= start_i < time.size and 0 <= end_i - 1 < time.size:
                        ax_spine.axvspan(
                            time[start_i] - reference_time,
                            time[end_i - 1] - reference_time,
                            color=dendrite_color,
                            alpha=0.08,
                            zorder=1,
                            lw=0,
                        )

                if window_runs:
                    for run in window_runs:
                        coincident = any(int(run[0]) == int(dend_run[0]) for dend_run in dendrite_event_runs)
                        run_color = spine_event_color if coincident else noncoincident_color
                        run_label = "coincident" if coincident else "noncoincident"

                        start_i, end_i = run
                        if 0 <= start_i < time.size and 0 <= end_i - 1 < time.size:
                            x0 = time[start_i] - reference_time
                            x1 = time[end_i - 1] - reference_time
                            ax_spine.axvspan(x0, x1, color=run_color, alpha=0.20, zorder=4, lw=0)

                            y0, y1 = ax_spine.get_ylim()
                            ax_spine.text(
                                0.5 * (x0 + x1),
                                y0 + 0.90 * (y1 - y0),
                                run_label,
                                ha="center",
                                va="top",
                                fontsize=8,
                                color=run_color,
                                bbox={
                                    "boxstyle": "round,pad=0.15",
                                    "facecolor": "white",
                                    "edgecolor": "none",
                                    "alpha": 0.75,
                                },
                            )
                else:
                    ax_spine.text(
                        0.04,
                        0.92,
                        "no detected spine event",
                        transform=ax_spine.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#555555",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.75,
                        },
                    )

                spine_y_values = window_trace[window_valid]
                if np.isfinite(threshold):
                    spine_y_values = np.concatenate([spine_y_values, np.asarray([threshold], dtype=float)])

                spine_y_min = float(np.nanmin(spine_y_values))
                spine_y_max = float(np.nanmax(spine_y_values))
                spine_pad = max(0.12 * (spine_y_max - spine_y_min), 0.05) if spine_y_min != spine_y_max else 0.1
                ax_spine.set_ylim(spine_y_min - spine_pad, spine_y_max + spine_pad)

                if dendrite_trace is not None and dendrite_time is not None and dendrite_time.size > 0:
                    dend_window_trace = dendrite_display_trace[window_start:window_end]
                    dend_valid = np.isfinite(dend_window_trace)
                    if np.any(dend_valid):
                        dend_y_min = float(np.nanmin(dend_window_trace[dend_valid]))
                        dend_y_max = float(np.nanmax(dend_window_trace[dend_valid]))
                        dend_pad = max(0.12 * (dend_y_max - dend_y_min), 0.05) if dend_y_min != dend_y_max else 0.1
                        ax_dendrite.set_ylim(dend_y_min - dend_pad, dend_y_max + dend_pad)

                if col == 0:
                    ax_spine.set_ylabel(display_trace_label, color=spine_color)
                else:
                    ax_spine.set_ylabel("")
                    ax_spine.tick_params(axis="y", labelleft=False)

                ax_spine.tick_params(axis="y", colors=spine_color)
                ax_spine.spines["left"].set_color(spine_color)
                ax_spine.spines["right"].set_visible(False)

                if col == 1:
                    ax_dendrite.set_ylabel(dendrite_display_label, color=dendrite_color)
                else:
                    ax_dendrite.set_ylabel("")
                    ax_dendrite.tick_params(axis="y", labelright=False)

                ax_dendrite.yaxis.set_label_position("right")
                ax_dendrite.yaxis.tick_right()
                ax_dendrite.tick_params(axis="y", colors=dendrite_color)
                ax_dendrite.spines["right"].set_color(dendrite_color)
                ax_dendrite.spines["left"].set_visible(False)
                ax_dendrite.grid(False)
                ax_dendrite.patch.set_alpha(0.0)

                ax_spine.text(
                    0.98,
                    0.04,
                    "spine + dendrite",
                    transform=ax_spine.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

                ax = ax_spine

            else:
                ax.plot(
                    window_time_rel[window_valid],
                    window_trace[window_valid],
                    color=trace_color,
                    linewidth=1.1,
                    label=display_trace_label,
                )

                if np.isfinite(threshold):
                    ax.axhline(threshold, color=event_color, linestyle="--", linewidth=0.8, alpha=0.9)

                if window_runs:
                    for run in window_runs:
                        start_i, end_i = run
                        if 0 <= start_i < time.size and 0 <= end_i - 1 < time.size:
                            ax.axvspan(
                                time[start_i] - reference_time,
                                time[end_i - 1] - reference_time,
                                color=trace_color,
                                alpha=0.20,
                                zorder=4,
                                lw=0,
                            )
                else:
                    ax.text(
                        0.04,
                        0.92,
                        "context window",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#555555",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.75,
                        },
                    )

                y_values = window_trace[window_valid]
                if np.isfinite(threshold):
                    y_values = np.concatenate([y_values, np.asarray([threshold], dtype=float)])

                y_min = float(np.nanmin(y_values))
                y_max = float(np.nanmax(y_values))
                pad = max(0.12 * (y_max - y_min), 0.05) if y_min != y_max else 0.1
                ax.set_ylim(y_min - pad, y_max + pad)

                if col == 0:
                    ax.set_ylabel(display_trace_label)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

                ax.text(
                    0.98,
                    0.04,
                    display_trace_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

            panel_label = f"Example {idx + 1}"
            if trace_kind == "spine":
                if window_runs:
                    any_coincident = any(
                        any(int(run[0]) == int(dend_run[0]) for dend_run in dendrite_event_runs)
                        for run in window_runs
                    )
                    panel_label += " | coincident" if any_coincident else " | noncoincident"
                else:
                    panel_label += " | context"
            else:
                panel_label += " | event" if window_runs else " | context"

            ax.set_title(panel_label, fontsize=10, pad=3)
            ax.axvline(0, color="0.25", linewidth=0.7, alpha=0.45)
            ax.axhline(0, color="0.4", linewidth=0.6, alpha=0.25)
            ax.grid(alpha=0.18)

            ax.set_xlim(float(np.nanmin(window_time_rel[window_valid])), float(np.nanmax(window_time_rel[window_valid])))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)

            if row == 4:
                ax.set_xlabel("Time from event/window center (s)")
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

        for ax in axes_flat[len(selected_windows):]:
            ax.set_axis_off()

        fig.subplots_adjust(
            top=0.955,
            bottom=0.055,
            left=0.075,
            right=0.925,
            hspace=0.34,
            wspace=0.16,
        )

    return fig


# Restored figure/helper dependencies from commit 64f20508a23d

def _boxplot_significance_stars(p_value: Any) -> str:
    p = as_float(p_value)
    if p is None or not np.isfinite(p) or p >= REPORT_SIGNIFICANCE_ALPHA:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _render_state_summary_comparison_panel_figure(
    metric_key: str,
    metric_title: str,
    basal_summary: Dict[str, Dict[str, List[float]]],
    apical_summary: Dict[str, Dict[str, List[float]]],
    state_order: Sequence[str],
    y_limit: Optional[Tuple[float, float]] = None,
    comparison_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Any]:
    if plt is None:
        return None
    fig_width = min(max(6.9, 0.76 * len(state_order) + 2.8), 8.5)
    fig_height = min(max(4.2, POSTER_DOUBLE_FIGSIZE[1] - 0.7), 4.9)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), squeeze=False)
    ax = ax.ravel()[0]
    rng = np.random.default_rng(7)
    all_data: List[np.ndarray] = []
    compartment_specs = [
        ("basal", basal_summary, "#4C72B0", -0.18),
        ("apical", apical_summary, "#DD8452", 0.18),
    ]
    for compartment, summary, color, offset in compartment_specs:
        positions: List[float] = []
        data: List[np.ndarray] = []
        for idx, state in enumerate(state_order, start=1):
            arr = flatten_state_summary_values(summary.get(state, {}))
            if arr.size:
                positions.append(idx + offset)
                data.append(arr)
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False)
        _set_boxplot_colors(bp, [color] * len(data))
        for pos, arr in zip(positions, data):
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            ax.scatter(np.full(arr.size, pos) + jitter, arr, s=14, alpha=0.48, color=color, edgecolor="none")
        all_data.extend(data)
    set_requested_state_ticks(ax, state_order)
    ax.set_ylabel("Dendrite dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.set_title(metric_title, fontsize=max(17, POSTER_TITLE_SIZE - 5), pad=1)
    _pad_boxplot_ylim(ax, all_data, y_limit=y_limit)
    y0, y1 = ax.get_ylim()
    y_range = max(float(y1 - y0), 1e-6)
    for compartment, summary, color, offset in compartment_specs:
        positions: List[float] = []
        data: List[np.ndarray] = []
        for idx, state in enumerate(state_order, start=1):
            arr = flatten_state_summary_values(summary.get(state, {}))
            if arr.size:
                positions.append(idx + offset)
                data.append(arr)
        for pos, arr in zip(positions, data):
            annotate_sample_size(
                ax,
                pos,
                min(float(np.nanmax(arr)) + 0.03 * y_range, float(y1) - 0.01 * y_range),
                f"n={arr.size}",
                fontsize=POSTER_NOTE_SIZE - 1,
                color=color,
            )
    ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    comparison_subset = [
        {
            "x1": float(idx - 0.18),
            "x2": float(idx + 0.18),
            "shuffle_p": row.get("shuffle_p"),
        }
        for row in (comparison_rows or [])
        if str(row.get("comparison")) == "basal_vs_apical"
        and str(row.get("metric")) == metric_key
        and is_significant_row(row)
        and str(row.get("state")) in state_order
        for idx in [state_order.index(str(row.get("state"))) + 1]
    ]
    _draw_boxplot_significance_annotations(ax, comparison_subset)
    legend_handles = [
        Line2D([0], [0], color="#4C72B0", marker="s", linestyle="", markersize=8, label="Basal"),
        Line2D([0], [0], color="#DD8452", marker="s", linestyle="", markersize=8, label="Apical"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=POSTER_LEGEND_SIZE)
    annotate_sample_size(
        ax,
        0.02,
        0.98,
        f"n={len(comparison_subset)} comparisons",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=True,
    )
    fig.tight_layout()
    return fig


SPINE_COACTIVITY_FIGURE_SUBDIRS = {
    "distribution": "distribution",
    "tendency": "pair_state_heatmap",
    "pair_state_heatmap": "pair_state_heatmap",
    "pair_state_summary": "pair_state_summary",
    "basal_apical_distribution": "basal_vs_apical",
    "basal_vs_apical": "basal_vs_apical",
}


def state_summary_metric_output_dir(
    root: Path,
    metric_name: str,
    cohort_label: str = "all",
    state_group: str = DEFAULT_STATE_SUMMARY_FIGURES_SUBDIRNAME,
) -> Path:
    return figure_nested_dir(root, state_summary_metric_family(metric_name), state_group, cohort_label)


def _state_summary_significant_pair_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_name: str,
    state_order: Sequence[str],
    comparison_name: str,
) -> List[Dict[str, Any]]:
    state_set = {canonical_state_label(state) for state in state_order}
    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if str(row.get('comparison')) != comparison_name:
            continue
        if str(row.get('metric')) != metric_name:
            continue
        if not is_significant_row(row):
            continue
        state_a = canonical_state_label(row.get('state_a'))
        state_b = canonical_state_label(row.get('state_b'))
        if state_a not in state_set or state_b not in state_set:
            continue
        filtered.append(dict(row))
    return filtered


def _state_summary_significant_single_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_name: str,
    state_order: Sequence[str],
    comparison_name: str,
) -> List[Dict[str, Any]]:
    state_set = {canonical_state_label(state) for state in state_order}
    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if str(row.get('comparison')) != comparison_name:
            continue
        if str(row.get('metric')) != metric_name:
            continue
        if not is_significant_row(row):
            continue
        state = canonical_state_label(row.get('state'))
        if state not in state_set:
            continue
        filtered.append(dict(row))
    return filtered


def _state_summary_significant_basal_apical_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    state_order: Sequence[str],
    comparison_name: str,
) -> List[Dict[str, Any]]:
    state_set = {canonical_state_label(state) for state in state_order}
    filtered: List[Dict[str, Any]] = []
    for row in rows or []:
        if str(row.get('comparison')) != comparison_name:
            continue
        if not is_significant_row(row):
            continue
        state = canonical_state_label(row.get('state'))
        if state not in state_set:
            continue
        filtered.append(dict(row))
    return filtered


def plot_event_detection_example_figure(
    *,
    output_path: Path,
    time: np.ndarray,
    trace: np.ndarray,
    event_info: Dict[str, Any],
    title: str,
    trace_label: str,
    trace_kind: str,
    figure_kind: Optional[str] = None,
    dendrite_event_info: Optional[Dict[str, Any]] = None,
    dendrite_trace: Optional[np.ndarray] = None,
    dendrite_time: Optional[np.ndarray] = None,
) -> Optional[str]:
    if plt is None:
        return None
    fig = _build_event_detection_example_figure(
        time=time,
        trace=trace,
        event_info=event_info,
        title=title,
        trace_label=trace_label,
        trace_kind=trace_kind,
        dendrite_event_info=dendrite_event_info,
        dendrite_trace=dendrite_trace,
        dendrite_time=dendrite_time,
    )
    if fig is None:
        return None
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    try:
        save_figure(fig, output_path, extra_formats=())
    finally:
        plt.close(fig)
    return str(output_path)


def day_figure_compartment_folder(compartment: Any) -> str:
    text = str(compartment or "").strip().lower()
    if text in {"basal", "apical"}:
        return text
    return "other"


def build_event_example_day_figure_path(
    output_dir: Path,
    animal_id: Any,
    day_id: Any,
    compartment: Any,
    global_dendrite_id: Any,
    global_spine_id: Any = None,
    *,
    kind: str,
) -> Path:
    day_animal_id, day_date, day_compartment = split_day_id(day_id)
    animal_slug = safe_filename_component(animal_id or day_animal_id or "unknown_animal")
    compartment_slug = safe_filename_component(day_figure_compartment_folder(compartment or day_compartment))
    date_slug = safe_filename_component(day_date or "unknown_date")
    dendrite_slug = safe_filename_component(extract_dendrite_token(global_dendrite_id))
    figure_dir = figure_family_dir(output_dir, DEFAULT_EVENT_EXAMPLE_FIGURES_DIRNAME, animal_slug, compartment_slug, date_slug)
    if kind == "spine":
        spine_slug = safe_filename_component(extract_dendrite_token(global_spine_id))
        figure_name = f"{animal_slug}_{compartment_slug}_{date_slug}_{dendrite_slug}_{spine_slug}_spine_event_example.svg"
    else:
        figure_name = f"{animal_slug}_{compartment_slug}_{date_slug}_{dendrite_slug}_dendrite_event_example.svg"
    return figure_dir / figure_name


# Restored figure/helper functions from commit 64f20508a23d

def _set_boxplot_colors(bp: Dict[str, Any], colors: Sequence[str]) -> None:
    for patch, color in zip(bp.get("boxes", []), colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for whisker in bp.get("whiskers", []):
        whisker.set_color("#555555")
    for cap in bp.get("caps", []):
        cap.set_color("#555555")
    for median in bp.get("medians", []):
        median.set_color("#222222")
        median.set_linewidth(1.5)


def annotate_sample_size(
    ax: Any,
    x: float,
    y: float,
    text: str,
    *,
    ha: str = "center",
    va: str = "bottom",
    fontsize: int = POSTER_NOTE_SIZE,
    color: str = "#444444",
    transform: Optional[Any] = None,
    bbox: bool = False,
) -> None:
    text_kwargs = {
        "ha": ha,
        "va": va,
        "fontsize": fontsize,
        "color": color,
        "clip_on": False,
    }
    if transform is not None:
        text_kwargs["transform"] = transform
    if bbox:
        text_kwargs["bbox"] = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", alpha=0.85)
    ax.text(x, y, text, **text_kwargs)


def _draw_boxplot_significance_annotations(
    ax: Any,
    comparisons: Sequence[Dict[str, Any]],
    *,
    label_key: str = "shuffle_p",
    base_color: str = "#444444",
    orientation: str = "vertical",
) -> None:
    items: List[Dict[str, Any]] = []
    for row in comparisons:
        p_value = as_float(row.get(label_key))
        label = str(row.get("label") or _boxplot_significance_stars(p_value))
        if not label:
            continue
        x1 = as_float(row.get("x1"))
        x2 = as_float(row.get("x2"))
        x = as_float(row.get("x"))
        if (x1 is None or x2 is None or not np.isfinite(x1) or not np.isfinite(x2)) and (x is None or not np.isfinite(x)):
            continue
        point_only = False
        if x is not None and np.isfinite(x) and (x1 is None or x2 is None):
            x1 = x2 = float(x)
            point_only = True
        items.append(
            {
                "x1": float(min(x1, x2)),
                "x2": float(max(x1, x2)),
                "label": label,
                "p_value": float(p_value),
                "point_only": point_only or np.isclose(float(x1), float(x2)),
            }
        )
    if not items:
        return

    if orientation == "horizontal":
        x0, x1 = ax.get_xlim()
        if not np.isfinite(x0) or not np.isfinite(x1):
            return
        x_range = x1 - x0
        if not np.isfinite(x_range) or x_range <= 0:
            x_range = 1.0
        bracket_base = x1 + max(0.002 * x_range, 0.0015)
        bracket_step = max(0.003 * x_range, 0.002)
        text_offset = max(0.001 * x_range, 0.0015)
        bracket_height = max(0.002 * x_range, 0.002)
        max_levels = 3
        levels: List[List[Tuple[float, float]]] = []
        placed: List[Tuple[float, float, int, str, bool]] = []
        # Place longer spans first so they settle into the topmost tiers.
        for item in sorted(
            items,
            key=lambda entry: (
                -(entry["x2"] - entry["x1"]),
                entry["p_value"],
                entry["x1"],
                entry["x2"],
            ),
        ):
            level = 0
            while level < len(levels):
                overlap = any(
                    not (item["x2"] < existing[0] - 0.05 or item["x1"] > existing[1] + 0.05)
                    for existing in levels[level]
                )
                if not overlap:
                    break
                level += 1
            if level == len(levels):
                levels.append([])
            levels[level].append((item["x1"], item["x2"]))
            placed.append(
                (
                    item["x1"],
                    item["x2"],
                    min(level, max_levels - 1),
                    item["label"],
                    bool(item.get("point_only", False)),
                )
            )
        render_max_level = min(len(levels), max_levels) - 1
        top_needed = bracket_base + (render_max_level + 1) * bracket_step + bracket_height + text_offset + 0.002 * x_range
        if top_needed > x1:
            ax.set_xlim(x0, top_needed)
        for y1_pos, y2_pos, level, label, point_only in placed:
            x = bracket_base + (render_max_level - level) * bracket_step
            if point_only or np.isclose(y1_pos, y2_pos):
                ax.text(
                    x + bracket_height + text_offset,
                    y1_pos,
                    label,
                    ha="left",
                    va="center",
                    fontsize=POSTER_NOTE_SIZE,
                    color="#222222",
                    clip_on=False,
                    zorder=6,
                )
            else:
                ax.plot([x, x + bracket_height, x + bracket_height, x], [y1_pos, y1_pos, y2_pos, y2_pos], color=base_color, linewidth=1.0, clip_on=False, zorder=5)
                ax.text(
                    x + bracket_height + text_offset,
                    (y1_pos + y2_pos) / 2.0,
                    label,
                    ha="left",
                    va="center",
                    fontsize=POSTER_NOTE_SIZE,
                    color="#222222",
                    clip_on=False,
                    zorder=6,
                )
        return

    y0, y1 = ax.get_ylim()
    if not np.isfinite(y0) or not np.isfinite(y1):
        return
    y_range = y1 - y0
    if not np.isfinite(y_range) or y_range <= 0:
        y_range = 1.0
    bracket_base = y1 + max(0.002 * y_range, 0.0015)
    bracket_step = max(0.003 * y_range, 0.002)
    text_offset = max(0.001 * y_range, 0.0015)
    bracket_height = max(0.002 * y_range, 0.002)
    max_levels = 3

    levels: List[List[Tuple[float, float]]] = []
    placed: List[Tuple[float, float, int, str, bool]] = []
    for item in sorted(items, key=lambda entry: (entry["x2"] - entry["x1"], entry["p_value"], entry["x1"], entry["x2"])):
        level = 0
        while level < len(levels):
            overlap = any(not (item["x2"] < existing[0] - 0.05 or item["x1"] > existing[1] + 0.05) for existing in levels[level])
            if not overlap:
                break
            level += 1
        if level == len(levels):
            levels.append([])
        levels[level].append((item["x1"], item["x2"]))
        placed.append((item["x1"], item["x2"], min(level, max_levels - 1), item["label"], bool(item.get("point_only", False))))

    top_needed = bracket_base + min(len(levels), max_levels) * bracket_step + bracket_height + text_offset + 0.002 * y_range
    if top_needed > y1:
        ax.set_ylim(y0, top_needed)

    render_max_level = min(len(levels), max_levels) - 1
    for x1, x2, level, label, point_only in placed:
        y = bracket_base + (render_max_level - level) * bracket_step
        if point_only or np.isclose(x1, x2):
            ax.text(
                x1,
                y + bracket_height + text_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=POSTER_NOTE_SIZE,
                color="#222222",
                clip_on=False,
                zorder=6,
            )
        else:
            ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color=base_color, linewidth=1.0, clip_on=False, zorder=5)
            ax.text(
                (x1 + x2) / 2.0,
                y + bracket_height + text_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=POSTER_NOTE_SIZE,
                color="#222222",
                clip_on=False,
                zorder=6,
            )


def spine_coactivity_figure_dir(root: Path, figure_kind: str, *parts: Any) -> Path:
    figure_kind = str(figure_kind)

    if figure_kind not in SPINE_COACTIVITY_FIGURE_SUBDIRS:
        raise ValueError(f"Unknown spine coactivity figure kind: {figure_kind}")

    subdir = SPINE_COACTIVITY_FIGURE_SUBDIRS[figure_kind]

    return figure_family_dir(
        root,
        DEFAULT_SPINE_COACTIVITY_FIGURES_DIRNAME,
        subdir,
        *parts,
    )


def _render_state_summary_single_panel_figure(
    metric_key: str,
    metric_title: str,
    summary: Dict[str, Dict[str, List[float]]],
    state_order: Sequence[str],
    y_limit: Optional[Tuple[float, float]] = None,
    comparison_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Any]:
    if plt is None:
        return None
    fig_width = min(max(6.9, 0.76 * len(state_order) + 2.8), 8.5)
    fig_height = min(max(4.2, POSTER_DOUBLE_FIGSIZE[1] - 0.7), 4.9)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), squeeze=False)
    ax = ax.ravel()[0]
    rng = np.random.default_rng(7)
    all_data: List[np.ndarray] = []
    positions: List[float] = []
    data: List[np.ndarray] = []
    labels: List[str] = []
    for idx, state in enumerate(state_order, start=1):
        arr = flatten_state_summary_values(summary.get(state, {}))
        if arr.size:
            positions.append(float(idx))
            data.append(arr)
            labels.append(state)
    if not data:
        plt.close(fig)
        return None
    bp = ax.boxplot(data, positions=positions, widths=0.50, patch_artist=True, showfliers=False)
    _set_boxplot_colors(bp, [state_display_color(state) for state in labels])
    for pos, state, arr in zip(positions, labels, data):
        jitter = rng.uniform(-0.08, 0.08, size=arr.size)
        ax.scatter(np.full(arr.size, pos) + jitter, arr, s=14, alpha=0.48, color=state_display_color(state), edgecolor="none")
    set_requested_state_ticks(ax, [state for state in state_order if state in labels])
    ax.set_ylabel("Dendrite dF/F", fontsize=POSTER_LABEL_SIZE)
    ax.set_title(metric_title, fontsize=max(17, POSTER_TITLE_SIZE - 5), pad=1)
    _pad_boxplot_ylim(ax, all_data + data)
    if y_limit is not None and len(y_limit) == 2:
        try:
            ax.set_ylim(float(y_limit[0]), float(y_limit[1]))
        except Exception:
            pass
    y0, y1 = ax.get_ylim()
    y_range = max(float(y1 - y0), 1e-6)
    for pos, state, arr in zip(positions, labels, data):
        annotate_sample_size(
            ax,
            pos,
            min(float(np.nanmax(arr)) + 0.03 * y_range, float(y1) - 0.01 * y_range),
            f"n={arr.size}",
            fontsize=POSTER_NOTE_SIZE - 1,
            color=state_display_color(state),
        )
    ax.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    comparison_subset = []
    state_position_lookup = {state: float(idx) for idx, state in enumerate(state_order, start=1)}
    for row in (comparison_rows or []):
        if str(row.get("comparison")) != metric_key:
            continue
        if not is_significant_row(row):
            continue
        state_a = canonical_state_label(row.get("state_a"))
        state_b = canonical_state_label(row.get("state_b"))
        if state_a not in state_position_lookup or state_b not in state_position_lookup:
            continue
        comparison_subset.append(
            {
                "x1": float(state_position_lookup[state_a]),
                "x2": float(state_position_lookup[state_b]),
                "shuffle_p": row.get("shuffle_p"),
            }
        )
    _draw_boxplot_significance_annotations(ax, comparison_subset)
    fig.tight_layout()
    return fig


def plot_state_summary_compartment_comparison_figure(
    basal_results: Dict[str, Any],
    apical_results: Dict[str, Any],
    fig_dir: Path,
    output_name: str,
    title: str,
    state_labels: Optional[Sequence[str]] = None,
    y_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    comparison_rows: Optional[Sequence[Dict[str, Any]]] = None,
    cohort_label: str = "all",
    state_group: str = DEFAULT_STATE_SUMMARY_FIGURES_SUBDIRNAME,
) -> Optional[str]:
    if plt is None:
        return None
    state_order = list(state_labels) if state_labels is not None else list(DEFAULT_BASAL_APICAL_STATES)
    state_position_lookup = {state: position for position, state in enumerate(state_order, start=1)}
    metric_titles = {
        "dendrite_mean": "Dendrite mean dF/F",
        "spine_specific_mean": "Spine-specific mean dF/F",
        "dendrite_event_frequency_per_min": "Dendrite calcium event frequency (per min)",
        "spine_event_frequency_per_min": "Spine calcium event frequency (per min)",
        "coincident_event_frequency_per_min": "Coincident spine event frequency (per min)",
        "noncoincident_event_frequency_per_min": "Noncoincident spine event frequency (per min)",
    }
    metric_specs = [(metric_name, metric_title) for metric_name, metric_title in metric_titles.items()]
    output_paths: List[Path] = []
    for metric_name, metric_title in metric_specs:
        basal_summary = basal_results.get("state_summaries", {}).get(metric_name, {})
        apical_summary = apical_results.get("state_summaries", {}).get(metric_name, {})
        panel_comparisons = [
            row
            for row in (comparison_rows or [])
            if str(row.get("comparison")) == "basal_vs_apical"
            and str(row.get("metric")) == metric_name
            and is_significant_row(row)
            and str(row.get("state")) in state_order
        ]
        panel_fig = _render_state_summary_comparison_panel_figure(
            metric_name,
            metric_title,
            basal_summary,
            apical_summary,
            state_order,
            y_limits.get(metric_name) if y_limits else None,
            comparison_rows=panel_comparisons,
        )
        if panel_fig is not None:
            metric_output_path = state_summary_metric_output_dir(
                fig_dir,
                metric_name,
                cohort_label,
                state_group,
            ) / f"{Path(output_name).stem}_{metric_name}.svg"
            save_figure(panel_fig, metric_output_path, extra_formats=())
            output_paths.append(metric_output_path)
    return str(output_paths[0]) if output_paths else None


def plot_basal_apical_summary(results: Dict[str, Any], fig_dir: Path) -> Optional[str]:
    if plt is None:
        return None
    rows = [row for row in results.get("basal_apical_comparisons", []) if row.get("comparison") == "basal_vs_apical"]
    if not rows:
        return None
    state_order = [state for state in selected_basal_apical_state_labels(results) if any(row.get("state") == state for row in rows)]
    metric_order = [
        metric
        for metric in [
            "dendrite_mean",
            "spine_specific_mean",
            "dendrite_event_frequency_per_min",
            "spine_event_frequency_per_min",
            "coincident_event_frequency_per_min",
            "noncoincident_event_frequency_per_min",
        ]
        if any(row.get("metric") == metric for row in rows)
    ]
    if not state_order or not metric_order:
        return None
    effect_lookup: Dict[Tuple[str, str], float] = {}
    p_lookup: Dict[Tuple[str, str], float] = {}
    for row in rows:
        effect_lookup[(str(row.get("metric")), str(row.get("state")))] = float(row.get("effect_size", float("nan")))
        p_lookup[(str(row.get("metric")), str(row.get("state")))] = float(row.get("shuffle_p", float("nan")))
    fig_width = min(max(9.0, 0.70 * len(state_order) + 3.5), 10.8)
    fig_height = min(max(4.4, POSTER_DOUBLE_FIGSIZE[1] - 0.9), 4.9)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), squeeze=False, gridspec_kw={"wspace": 0.28})
    x = np.arange(len(state_order))
    width = 0.35 if len(metric_order) > 1 else 0.6
    palette = plt.get_cmap("Dark2")
    ax = axes[0, 0]
    for idx, metric in enumerate(metric_order):
        offsets = x + (idx - (len(metric_order) - 1) / 2.0) * width
        values = [effect_lookup.get((metric, state), np.nan) for state in state_order]
        ax.bar(offsets, values, width=width, label=metric.replace("_", " "), color=palette(idx), alpha=0.85)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title("Basal vs apical effect sizes", fontsize=POSTER_TITLE_SIZE)
    ax.set_ylabel("Effect size", fontsize=POSTER_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([format_requested_state_label(state) for state in state_order], rotation=0)
    color_state_tick_labels(ax, state_order, axis="x")
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.legend(frameon=False, fontsize=POSTER_LEGEND_SIZE)
    ax.grid(axis="y", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="y", nbins=5)
    ax = axes[0, 1]
    for idx, metric in enumerate(metric_order):
        offsets = x + (idx - (len(metric_order) - 1) / 2.0) * width
        p_vals = [p_lookup.get((metric, state), np.nan) for state in state_order]
        p_vals = np.asarray(p_vals, dtype=float)
        neglog = np.full_like(p_vals, np.nan, dtype=float)
        valid = np.isfinite(p_vals) & (p_vals > 0)
        neglog[valid] = -np.log10(np.clip(p_vals[valid], 1e-300, 1.0))
        ax.bar(offsets, neglog, width=width, label=metric.replace("_", " "), color=palette(idx), alpha=0.85)
    ax.axhline(-np.log10(0.05), color="#8b0000", linestyle="--", linewidth=1, label="p=0.05")
    ax.set_title("Basal vs apical shuffle significance", fontsize=POSTER_TITLE_SIZE)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=POSTER_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([format_requested_state_label(state) for state in state_order], rotation=0)
    color_state_tick_labels(ax, state_order, axis="x")
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.legend(frameon=False, fontsize=POSTER_LEGEND_SIZE)
    ax.grid(axis="y", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="y", nbins=5)
    output_path = fig_dir / "basal_apical_summary.svg"
    save_figure(fig, output_path, extra_formats=())
    plt.close(fig)
    return str(output_path)


def generate_event_detection_example_gallery(cache: Dict[str, Any], fig_dir: Path) -> List[str]:
    if plt is None:
        return []
    event_root = ensure_dir(Path(fig_dir))
    saved: List[str] = []
    jobs: List[Dict[str, Any]] = []
    for animal_id in sorted(cache.get("animals", {})):
        animal_entry = cache.get("animals", {}).get(animal_id, {})
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            for day_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                compartment = observation_compartment(cache, day_id, d_obs)
                jobs.append(
                    {
                        "kind": "dendrite",
                        "animal_id": animal_id,
                        "day_id": d_obs.get("day_id") or d_obs.get("exp_id") or day_id,
                        "compartment": compartment,
                        "global_dendrite_id": global_dendrite_id,
                        "global_spine_id": None,
                        "time": d_obs.get("time"),
                        "trace": d_obs.get("trace"),
                        "event_info": d_obs.get("event_info") or {},
                        "dendrite_event_info": None,
                        "dendrite_trace": None,
                        "dendrite_time": None,
                        "title": f"{format_dendrite_display_name(animal_id, compartment, global_dendrite_id)} - event examples",
                        "trace_label": "Dendrite dF/F",
                    }
                )
                for global_spine_id, spine_record in sorted(dendrite_record.get("spines", {}).items()):
                    s_obs = spine_record.get("observations", {}).get(day_id)
                    if s_obs is None:
                        continue
                    compartment = observation_compartment(cache, day_id, s_obs)
                    spine_specific = s_obs.get("spine_specific")
                    trace = spine_specific if spine_specific is not None else s_obs.get("trace")
                    jobs.append(
                        {
                            "kind": "spine",
                            "animal_id": animal_id,
                            "day_id": s_obs.get("day_id") or s_obs.get("exp_id") or day_id,
                            "compartment": compartment,
                            "global_dendrite_id": global_dendrite_id,
                            "global_spine_id": global_spine_id,
                            "time": s_obs.get("time"),
                            "trace": trace,
                            "event_info": s_obs.get("event_info") or {},
                            "dendrite_event_info": s_obs.get("dendrite_event_info") or d_obs.get("event_info") or {},
                            "dendrite_trace": d_obs.get("trace"),
                            "dendrite_time": d_obs.get("time"),
                            "title": f"{format_dendrite_display_name(animal_id, compartment, global_dendrite_id)} / {safe_filename_component(global_spine_id)} - event examples",
                            "trace_label": "Spine-specific dF/F",
                        }
                    )
    with step_scope("event detection example gallery", total=len(jobs)):
        for idx, job in enumerate(jobs, start=1):
            step_progress(idx, len(jobs), label=f"{job['animal_id']} | {extract_dendrite_token(job['global_dendrite_id'])} | {job['kind']}")
            path = build_event_example_day_figure_path(
                event_root,
                job["animal_id"],
                job["day_id"],
                job["compartment"],
                job["global_dendrite_id"],
                job["global_spine_id"],
                kind=str(job["kind"]),
            )
            saved_path = plot_event_detection_example_figure(
                output_path=path,
                time=np.asarray(job["time"], dtype=float),
                trace=np.asarray(job["trace"], dtype=float),
                event_info=dict(job["event_info"] or {}),
                title=str(job["title"]),
                trace_label=str(job["trace_label"]),
                trace_kind=str(job["kind"]),
                figure_kind="pair_state_heatmap",
                dendrite_event_info=dict(job["dendrite_event_info"] or {}) if job["dendrite_event_info"] is not None else None,
                dendrite_trace=np.asarray(job["dendrite_trace"], dtype=float) if job.get("dendrite_trace") is not None else None,
                dendrite_time=np.asarray(job["dendrite_time"], dtype=float) if job.get("dendrite_time") is not None else None,
            )
            if saved_path:
                saved.append(saved_path)
    return saved


def plot_correlation_summary(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "correlation_summary.svg",
    title: str = "Correlation summaries",
) -> Optional[str]:
    if plt is None:
        return None
    rows = results.get("correlations", [])
    if not rows:
        return None
    analysis_order = [
        "dendrite_wheel",
        "dendrite_pupil",
        "spine_dendrite_raw",
        "spine_dendrite_specific",
    ]
    analysis_order = [analysis for analysis in analysis_order if any(row.get("analysis") == analysis for row in rows)]
    if not analysis_order:
        return None
    label_lookup = {
        "dendrite_wheel": "dendrite activity vs wheel",
        "dendrite_pupil": "dendrite activity vs pupil",
        "spine_dendrite_raw": "spine-specific activity vs dendrite activity (raw)",
        "spine_dendrite_specific": "spine-specific activity vs dendrite activity (specific)",
    }
    has_split_groups = any(str(row.get("split_group") or "").strip() for row in rows)
    r_values: List[np.ndarray] = []
    p_values: List[np.ndarray] = []
    labels: List[str] = []
    for analysis in analysis_order:
        analysis_rows = [row for row in rows if row.get("analysis") == analysis]
        if has_split_groups:
            split_groups: List[str] = []
            split_display: Dict[str, str] = {}
            for row in analysis_rows:
                split_group = str(row.get("split_group") or "").strip()
                if not split_group:
                    continue
                if split_group not in split_groups:
                    split_groups.append(split_group)
                split_display.setdefault(split_group, str(row.get("split_group_display") or split_group).strip())
            if split_groups:
                for split_group in split_groups:
                    subset = [row for row in analysis_rows if str(row.get("split_group") or "").strip() == split_group]
                    r_arr = np.asarray([float(row.get("r", float("nan"))) for row in subset], dtype=float)
                    p_arr = np.asarray([float(row.get("shuffle_p", float("nan"))) for row in subset], dtype=float)
                    r_arr = r_arr[np.isfinite(r_arr)]
                    p_arr = p_arr[np.isfinite(p_arr)]
                    if r_arr.size == 0 and p_arr.size == 0:
                        continue
                    labels.append(f"{label_lookup.get(analysis, analysis.replace('_', ' '))} | {split_display.get(split_group, split_group)}")
                    r_values.append(r_arr if r_arr.size else np.asarray([np.nan], dtype=float))
                    p_values.append(p_arr if p_arr.size else np.asarray([np.nan], dtype=float))
                continue
        r_arr = np.asarray([float(row.get("r", float("nan"))) for row in analysis_rows], dtype=float)
        p_arr = np.asarray([float(row.get("shuffle_p", float("nan"))) for row in analysis_rows], dtype=float)
        r_arr = r_arr[np.isfinite(r_arr)]
        p_arr = p_arr[np.isfinite(p_arr)]
        if r_arr.size == 0 and p_arr.size == 0:
            continue
        labels.append(label_lookup.get(analysis, analysis.replace("_", " ")))
        r_values.append(r_arr if r_arr.size else np.asarray([np.nan], dtype=float))
        p_values.append(p_arr if p_arr.size else np.asarray([np.nan], dtype=float))
    if not labels:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5), squeeze=False, gridspec_kw={"wspace": 0.28})
    x = np.arange(1, len(labels) + 1)
    palette = plt.get_cmap("Set1")
    ax = axes[0, 0]
    bp = ax.boxplot(r_values, positions=x, widths=0.6, patch_artist=True, showfliers=False)
    _set_boxplot_colors(bp, [palette(i % palette.N) for i in range(len(labels))])
    for pos, arr in zip(x, r_values):
        if arr.size == 0:
            continue
        jitter = np.random.default_rng(11).uniform(-0.12, 0.12, size=arr.size)
        ax.scatter(np.full(arr.size, pos) + jitter, arr, s=12, alpha=0.45, color="#444444", edgecolor="none")
        ax.text(pos, np.nanmax(arr), f"n={arr.size}", ha="center", va="bottom", fontsize=POSTER_NOTE_SIZE)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title("Correlation coefficients", fontsize=POSTER_TITLE_SIZE)
    ax.set_ylabel("Pearson r", fontsize=POSTER_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" vs ", "\nvs ") for label in labels], rotation=0)
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="y", nbins=5)
    ax = axes[0, 1]
    bp = ax.boxplot(p_values, positions=x, widths=0.6, patch_artist=True, showfliers=False)
    _set_boxplot_colors(bp, [palette(i % palette.N) for i in range(len(labels))])
    for pos, arr in zip(x, p_values):
        if arr.size == 0:
            continue
        jitter = np.random.default_rng(12).uniform(-0.12, 0.12, size=arr.size)
        ax.scatter(np.full(arr.size, pos) + jitter, arr, s=12, alpha=0.45, color="#444444", edgecolor="none")
        ax.text(pos, np.nanmax(arr), f"n={arr.size}", ha="center", va="bottom", fontsize=POSTER_NOTE_SIZE)
        if np.nanmin(arr) < REPORT_SIGNIFICANCE_ALPHA:
            ax.scatter(pos, min(1.0, np.nanmax(arr) + 0.04), s=90, marker="*", color="#8b0000", zorder=4)
    ax.axhline(0.05, color="#8b0000", linestyle="--", linewidth=1)
    ax.set_title("Shuffle p-values", fontsize=POSTER_TITLE_SIZE)
    ax.set_ylabel("p", fontsize=POSTER_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(" vs ", "\nvs ") for label in labels], rotation=0)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    set_sparse_numeric_ticks(ax, axis="y", nbins=5)
    output_path = fig_dir / output_name
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)

def plot_matrix_similarity_heatmap(
    results: Dict[str, Any],
    fig_dir: Path,
    output_name: str = "matrix_similarity_heatmap.svg",
    title: str = "Matrix spine-spine similarity",
    compartment_filter: Optional[str] = None,
    dendrite_filter: Optional[str] = None,
) -> Optional[str]:
    if plt is None:
        return None
    rows = build_filtered_matrix_similarity_results(
        results,
        compartment_filter=compartment_filter,
        global_dendrite_id_filter=dendrite_filter,
    ).get("matrix_similarity", [])
    if not rows:
        return None

    def _render_heatmap(plot_rows: List[Dict[str, Any]], *, output_stem_name: str, plot_title: str) -> Optional[str]:
        state_labels = selected_matrix_plot_state_labels(results, plot_rows)
        if not state_labels:
            return None
        pair_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        pair_sig_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_obs_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        sig_rows = 0
        for row in plot_rows:
            state_a = canonical_state_label(row.get("state_a"))
            state_b = canonical_state_label(row.get("state_b"))
            if not state_a or not state_b or state_a == state_b:
                continue
            value = float(row.get("matrix_similarity_r", float("nan")))
            if np.isfinite(value):
                pair_values[(state_a, state_b)].append(value)
                pair_values[(state_b, state_a)].append(value)
            p_value = float(row.get("shuffle_p", float("nan")))
            if np.isfinite(p_value):
                pair_obs_counts[(state_a, state_b)] += 1
                pair_obs_counts[(state_b, state_a)] += 1
                if p_value < REPORT_SIGNIFICANCE_ALPHA:
                    sig_rows += 1
                    pair_sig_counts[(state_a, state_b)] += 1
                    pair_sig_counts[(state_b, state_a)] += 1
        matrix = np.full((len(state_labels), len(state_labels)), np.nan, dtype=float)
        for i, state_a in enumerate(state_labels):
            for j, state_b in enumerate(state_labels):
                if state_a == state_b:
                    continue
                values = pair_values.get((state_a, state_b), [])
                if values:
                    matrix[i, j] = float(np.nanmean(values))
        if not np.isfinite(matrix).any():
            return None
        side = min(max(6.2, 0.64 * len(state_labels) + 2.6), 9.6)
        fig = plt.figure(figsize=(side + 0.2, max(5.6, side * 0.88)))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.065], wspace=0.28)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        cmap = plt.get_cmap("coolwarm")
        norm = Normalize(vmin=-1.0, vmax=1.0)
        _configure_square_heatmap_axes(ax, state_labels, "State B", "State A")
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.set_xlabel("State B", fontsize=max(POSTER_LABEL_SIZE - 6, 12))
        ax.set_title(plot_title, fontsize=POSTER_TITLE_SIZE, pad=12)
        ax.set_xticks(np.arange(-0.5, len(state_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(state_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, len(state_labels) - 0.5)
        ax.set_ylim(len(state_labels) - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        try:
            from matplotlib.patches import Rectangle
        except Exception:
            Rectangle = None
        if Rectangle is not None:
            for i, state_a in enumerate(state_labels):
                for j, state_b in enumerate(state_labels):
                    if state_a == state_b:
                        continue
                    value = matrix[i, j]
                    if not np.isfinite(value):
                        continue
                    obs_count = pair_obs_counts.get((state_a, state_b), 0)
                    if obs_count <= 0:
                        continue
                    sig_count = pair_sig_counts.get((state_a, state_b), 0)
                    sig_fraction = (sig_count / obs_count) if obs_count else 0.0
                    square_size = 0.82 if sig_fraction >= 0.5 and sig_count > 0 else 0.48
                    ax.add_patch(
                        Rectangle(
                            (j - square_size / 2.0, i - square_size / 2.0),
                            square_size,
                            square_size,
                            facecolor=cmap(norm(value)),
                            edgecolor="#1f1f1f",
                            linewidth=0.8,
                            zorder=3,
                        )
                    )
        ax.text(
            0.02,
            0.98,
            f"n={len(plot_rows)} | sig={sig_rows}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.85),
        )
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("Pearson r", fontsize=POSTER_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
        set_sparse_colorbar_ticks(cbar, nbins=5)
        output_path = fig_dir / output_stem_name
        save_figure(fig, output_path, extra_formats=())
        return str(output_path)

    group_col = "split_group" if any(str(row.get("split_group") or "").strip() for row in rows) else None
    if group_col is not None:
        group_rows: Dict[str, List[Dict[str, Any]]] = {}
        group_display: Dict[str, str] = {}
        group_order: List[str] = []
        unassigned_rows: List[Dict[str, Any]] = []
        for row in rows:
            group_value = str(row.get(group_col) or "").strip()
            if not group_value:
                unassigned_rows.append(dict(row))
                continue
            group_key = group_value.lower().replace(" ", "_").replace("-", "_")
            while "__" in group_key:
                group_key = group_key.replace("__", "_")
            if group_key not in group_rows:
                group_order.append(group_key)
            group_rows.setdefault(group_key, []).append(dict(row))
            group_display.setdefault(group_key, str(row.get("split_group_display") or group_value).strip() or group_key)
        if unassigned_rows:
            group_rows["unassigned"] = unassigned_rows
            group_display.setdefault("unassigned", "Unassigned")
            group_order.append("unassigned")
        if len(group_order) > 1:
            generated: List[str] = []
            base_stem = Path(output_name).stem
            for group_key in group_order:
                subset = group_rows.get(group_key, [])
                if not subset:
                    continue
                rendered = _render_heatmap(
                    subset,
                    output_stem_name=f"{base_stem}_{group_key}.svg",
                    plot_title=f"{title} - {group_display.get(group_key, group_key)}",
                )
                if rendered:
                    generated.append(rendered)
            return generated or None
    return _render_heatmap(rows, output_stem_name=output_name, plot_title=title)


def plot_demo_validation_figure(
results: Dict[str, Any], fig_dir: Path) -> Optional[str]:
    if plt is None:
        return None
    rows = results.get("demo_validation", [])
    if not rows:
        return None
    expected = np.asarray([float(row.get("expected_alpha", float("nan"))) for row in rows], dtype=float)
    observed = np.asarray([float(row.get("observed_alpha", float("nan"))) for row in rows], dtype=float)
    abs_error = np.asarray([float(row.get("abs_error", float("nan"))) for row in rows], dtype=float)
    mask = np.isfinite(expected) & np.isfinite(observed)
    if mask.sum() == 0:
        return None
    expected = expected[mask]
    observed = observed[mask]
    abs_error = abs_error[mask]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5), squeeze=False, gridspec_kw={"wspace": 0.28})
    ax = axes[0, 0]
    scatter = ax.scatter(expected, observed, c=abs_error, cmap="magma", s=70, edgecolor="#222222", linewidth=0.5)
    lims = [
        float(np.nanmin([expected.min(), observed.min()])),
        float(np.nanmax([expected.max(), observed.max()])),
    ]
    if lims[0] == lims[1]:
        pad = 0.1 if lims[0] == 0 else max(0.05 * abs(lims[0]), 0.05)
        lims[0] -= pad
        lims[1] += pad
    ax.plot(lims, lims, linestyle="--", color="#444444", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Expected alpha", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Observed alpha", fontsize=POSTER_LABEL_SIZE)
    ax.set_title("Demo alpha recovery", fontsize=POSTER_TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(ax, axis="both", nbins=5)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error", fontsize=POSTER_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)
    set_sparse_colorbar_ticks(cbar, nbins=5)
    ax = axes[0, 1]
    ax.hist(abs_error, bins=min(10, max(3, abs_error.size)), color="#4c72b0", edgecolor="white")
    ax.set_title("Demo alpha absolute error", fontsize=POSTER_TITLE_SIZE)
    ax.set_xlabel("Absolute error", fontsize=POSTER_LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=POSTER_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(ax, axis="both", nbins=5)
    ax.grid(axis="y", alpha=0.25)
    annotate_sample_size(ax, 0.02, 0.98, f"n={int(mask.sum())} validated pairs", ha="left", va="top", transform=ax.transAxes, bbox=True)
    output_path = fig_dir / "demo_validation_scatter.svg"
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)


def generate_analysis_figures(
    output_dir: Path,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    figure_root: Optional[Path] = None,
) -> List[str]:
    if plt is None:
        eprint("[ALERT] matplotlib is unavailable; skipping figure generation.")
        return []
    fig_dir = ensure_dir(Path(figure_root) if figure_root is not None else (output_dir / "figures"))
    summary_fig_dir = state_summary_figure_dir(fig_dir)
    saved: List[str] = []
    coactivity_dir = fig_dir
    shuffle_n = int(results.get("run_parameters", {}).get("shuffle_n", DEFAULT_SHUFFLES) or DEFAULT_SHUFFLES)
    summary_metrics = [
        "dendrite_mean",
        "spine_specific_mean",
        "dendrite_event_frequency_per_min",
        "spine_event_frequency_per_min",
        "coincident_event_frequency_per_min",
        "noncoincident_event_frequency_per_min",
    ]
    state_labels = selected_matrix_state_labels(results)
    basal_apical_state_labels = selected_basal_apical_state_labels(results)
    present_compartments = sorted_present_compartments(cache)
    matrix_similarity_results_cache: Dict[tuple, Dict[str, Any]] = {}
    spine_coactivity_results_cache: Dict[tuple, Dict[str, Any]] = {}
    spine_coactivity_anchor_compartments: Optional[List[Optional[str]]] = None

    def cached_matrix_similarity_results(compartment: Optional[str], dendrite_id: Optional[str] = None) -> Dict[str, Any]:
        key = (compartment, dendrite_id)
        if key not in matrix_similarity_results_cache:
            matrix_similarity_results_cache[key] = build_filtered_matrix_similarity_results(
                results,
                compartment,
                dendrite_id,
            )
        return matrix_similarity_results_cache[key]

    def cached_spine_coactivity_results(compartment: Optional[str], dendrite_id: Optional[str] = None) -> Dict[str, Any]:
        key = (compartment, dendrite_id)
        if key not in spine_coactivity_results_cache:
            spine_coactivity_results_cache[key] = build_filtered_spine_coactivity_results(
                results,
                compartment,
                dendrite_id,
            )
        return spine_coactivity_results_cache[key]
    with step_scope("figure prep: state summary y-limits"):
        y_limits = state_summary_y_limits_from_results(results, state_labels)
        if not y_limits:
            y_limits = state_summary_y_limits(cache, state_labels)
    with step_scope("figure prep: state summary comparison y-limits"):
        comparison_y_limits = state_summary_y_limits(cache, basal_apical_state_labels)
    roi_split_results = results.get("roi_split", {})
    with step_scope("figure prep: state summary overview results"):
        overview_results = results
        if isinstance(roi_split_results, dict) and roi_split_results.get("subject_state_rows"):
            overview_results = dict(results)
            overview_results["analysis_branch_name"] = "roi_split"
            overview_results["analysis_basis_name"] = "all"
            overview_results["roi_split"] = roi_split_results
    with step_scope("figure prep: state summary basal results"):
        basal_results = build_state_summary_gallery_results(cache, state_labels, "basal")
    with step_scope("figure prep: state summary apical results"):
        apical_results = build_state_summary_gallery_results(cache, state_labels, "apical")
    state_summary_specs = [
        {
            "kind": "overview",
            "compartment": None,
            "output_name": "state_summary_boxplots.svg",
            "title": "Selected-state summary distributions - All compartments",
            "results": overview_results,
        },
    ]
    for compartment in [comp for comp in ["basal", "apical"] if comp in present_compartments]:
        state_summary_specs.append(
            {
                "kind": "overview",
                "compartment": compartment,
                "output_name": f"state_summary_boxplots_{compartment}.svg",
                "title": f"Selected-state summary distributions - {gallery_compartment_title(compartment)}",
                "results": basal_results if compartment == "basal" else apical_results,
            }
        )
    state_summary_specs.extend(
        [
            {
                "kind": "comparison",
                "output_name": "state_summary_boxplots_basal_vs_apical.svg",
                "title": "Selected-state summary distributions - Basal vs apical",
                "results": (basal_results, apical_results),
            },
        ]
    )
    visual_response_state_summaries = results.get("dendrite_visual_response_state_summaries", {})
    visual_response_summary = results.get("dendrite_visual_response", {})
    if isinstance(visual_response_state_summaries, dict):
        for cohort in [cohort for cohort in DENDRITE_RESPONSE_COHORTS[1:] if cohort in visual_response_state_summaries]:
            cohort_results = visual_response_state_summaries.get(cohort, {})
            if not isinstance(cohort_results, dict):
                continue
            cohort_basal_results = cohort_results.get("basal")
            cohort_apical_results = cohort_results.get("apical")
            if not isinstance(cohort_basal_results, dict) or not isinstance(cohort_apical_results, dict):
                continue
            cohort_title = cohort.capitalize()
            cohort_filters = {
                "basal": visual_response_dendrite_ids(visual_response_summary, "basal", cohort),
                "apical": visual_response_dendrite_ids(visual_response_summary, "apical", cohort),
            }
            for compartment, compartment_results in [("basal", cohort_basal_results), ("apical", cohort_apical_results)]:
                if compartment not in present_compartments:
                    continue
                compartment_filter = cohort_filters.get(compartment, [])
                cohort_metric_rows: List[Dict[str, Any]] = []
                for metric_name in summary_metrics:
                    cohort_metric_rows.extend(
                        pairwise_state_comparisons(
                            cache,
                            metric_name,
                            state_labels,
                            shuffle_n,
                            compartment_filter=compartment,
                            dendrite_ids_filter=compartment_filter,
                        )
                    )
                state_summary_specs.append(
                    {
                        "kind": "overview",
                        "name": f"state_summary_boxplots_{gallery_compartment_suffix(compartment)}_{cohort}",
                        "compartment": compartment,
                        "output_name": f"state_summary_boxplots_{gallery_compartment_suffix(compartment)}_{cohort}.svg",
                        "title": f"Selected-state summary distributions - {gallery_compartment_title(compartment)} ({cohort_title})",
                        "results": compartment_results,
                        "comparison_rows": cohort_metric_rows,
                        "cohort_label": cohort,
                    }
                )
            cohort_comparison_rows: List[Dict[str, Any]] = []
            for metric_name in summary_metrics:
                for state_label in basal_apical_state_labels:
                    cohort_comparison_rows.append(
                        basal_apical_comparison(
                            cache,
                            metric_name,
                            state_label,
                            shuffle_n,
                            dendrite_ids_filter_by_compartment=cohort_filters,
                        )
                    )
            state_summary_specs.append(
                {
                    "kind": "comparison",
                    "name": f"state_summary_boxplots_basal_vs_apical_{cohort}",
                    "output_name": f"state_summary_boxplots_basal_vs_apical_{cohort}.svg",
                    "title": f"Selected-state summary distributions - Basal vs apical ({cohort_title})",
                    "results": (cohort_basal_results, cohort_apical_results),
                    "comparison_rows": cohort_comparison_rows,
                    "cohort_label": cohort,
                }
            )
    for plot_idx, spec in enumerate(state_summary_specs, start=1):
        compartment = spec.get("compartment")
        scope_label = gallery_compartment_suffix(compartment) if spec["kind"] == "overview" else "basal_vs_apical"
        with step_scope(
            f"figure plotter: state_summary_boxplots[{scope_label}]",
            index=plot_idx,
            total=len(state_summary_specs),
        ):
            try:
                if spec["kind"] == "overview":
                    summary_results = spec["results"]
                    output_path = plot_state_summary_figure(
                        summary_results,
                        summary_fig_dir,
                        output_name=spec["output_name"],
                        title=spec["title"],
                        state_labels=state_labels,
                        y_limits=y_limits,
                        comparison_rows=spec.get("comparison_rows"),
                        cohort_label=str(spec.get("cohort_label") or "all"),
                    )
                else:
                    basal_summary, apical_summary = spec["results"]
                    output_path = plot_state_summary_compartment_comparison_figure(
                        basal_summary,
                        apical_summary,
                        summary_fig_dir,
                        output_name=spec["output_name"],
                        title=spec["title"],
                        state_labels=basal_apical_state_labels,
                        y_limits=comparison_y_limits,
                        comparison_rows=spec.get("comparison_rows") if spec.get("comparison_rows") is not None else [
                            row
                            for row in results.get("basal_apical_comparisons", [])
                            if str(row.get("comparison")) == "basal_vs_apical"
                            and is_significant_row(row)
                            and str(row.get("state")) in set(basal_apical_state_labels)
                        ],
                        cohort_label=str(spec.get("cohort_label") or "all"),
                    )
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with state summary plotter ({scope_label}): {exc}")
                continue
            if output_path:
                saved.append(output_path)
            else:
                step_message(f"plotter returned no output: {spec.get('name') or spec.get('output_name') or scope_label}")
    from analysis.shared.plots.roi_split import plot_roi_split_bundle_figure

    roi_split_results = results.get("roi_split", {})
    roi_split_bundles = roi_split_results.get("bundles", []) if isinstance(roi_split_results, dict) else []
    for plot_idx, bundle in enumerate(roi_split_bundles, start=1):
        if not isinstance(bundle, dict) or not bundle:
            continue
        roi_type = str(bundle.get("roi_type") or "roi").strip().lower() or "roi"
        compartment = str(bundle.get("compartment") or "").strip().lower() or "all"
        split_name = str(bundle.get("split_name") or "split").strip().lower() or "split"
        with step_scope(
            f"figure plotter: roi_split[{roi_type}|{compartment}|{split_name}]",
            index=plot_idx,
            total=len(roi_split_bundles),
        ):
            try:
                output_paths = plot_roi_split_bundle_figure(bundle, output_dir)
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with roi_split[{roi_type}|{compartment}|{split_name}]: {exc}")
                continue
            if output_paths:
                saved.extend(str(path) for path in output_paths)
            else:
                step_message(f"plotter returned no output: roi_split[{roi_type}|{compartment}|{split_name}]")

    spine_coactivity_property_split = results.get("spine_coactivity", {}).get("property_split", {})
    property_split_bundles = spine_coactivity_property_split.get("bundles", []) if isinstance(spine_coactivity_property_split, dict) else []
    for plot_idx, bundle in enumerate(property_split_bundles, start=1):
        if not isinstance(bundle, dict) or not bundle:
            continue
        roi_type = str(bundle.get("roi_type") or "roi").strip().lower() or "roi"
        compartment = str(bundle.get("compartment") or "").strip().lower() or "all"
        split_name = str(bundle.get("split_name") or "split").strip().lower() or "split"
        basis_name = str(bundle.get("basis_name") or "all").strip().lower() or "all"
        with step_scope(
            f"figure plotter: spine_coactivity_property_split[{roi_type}|{compartment}|{split_name}|{basis_name}]",
            index=plot_idx,
            total=len(property_split_bundles),
        ):
            try:
                output_paths = plot_roi_split_bundle_figure(bundle, output_dir)
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with spine_coactivity_property_split[{roi_type}|{compartment}|{split_name}|{basis_name}]: {exc}")
                continue
            if output_paths:
                saved.extend(str(path) for path in output_paths)
            else:
                step_message(f"plotter returned no output: spine_coactivity_property_split[{roi_type}|{compartment}|{split_name}|{basis_name}]")
    plotters = [
        plot_basal_apical_summary,
        plot_correlation_summary,
        plot_demo_validation_figure,
    ]
    for plot_idx, plotter in enumerate(plotters, start=1):
        with step_scope(f"figure plotter: {plotter.__name__}", index=plot_idx, total=len(plotters)):
            try:
                target_dir = figure_family_dir(fig_dir, DEFAULT_CORRELATION_FIGURES_DIRNAME) if plotter is plot_correlation_summary else fig_dir
                output_path = plotter(results, target_dir)
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with {plotter.__name__}: {exc}")
                continue
            if output_path:
                saved.append(output_path)
    matrix_review_rows = results.get("matrix_similarity", [])
    matrix_review_compartments = matrix_similarity_output_compartments(matrix_review_rows)
    for comp_idx, compartment in enumerate(matrix_review_compartments, start=1):
        with step_scope(
            f"figure plotter: matrix_similarity_distribution[{gallery_compartment_suffix(compartment)}]",
            index=comp_idx,
            total=len(matrix_review_compartments),
        ):
            try:
                compartment_results = cached_matrix_similarity_results(compartment)
                output_path = plot_matrix_similarity_distribution(
                    compartment_results,
                    figure_family_dir(fig_dir, DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME),
                    output_name=f"review_matrix_similarity_distribution_{gallery_compartment_suffix(compartment)}.svg",
                    title=f"Review: Spine-spine coefficient distributions - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                )
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with matrix_similarity_distribution[{compartment}]: {exc}")
                continue
            if output_path:
                saved.append(output_path)
    run_params = results.get("run_parameters", {}) if isinstance(results.get("run_parameters"), dict) else {}
    comparison_preset_name = str(run_params.get("comparison_preset_name") or "default")
    generate_shared_general_figures = bool(run_params.get("generate_shared_general_figures", True))
    mixed_model_dir = figure_family_dir(fig_dir, DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
    shared_mixed_model_dir = None
    if generate_shared_general_figures:
        base_root = fig_dir
        shared_mixed_model_dir = ensure_dir(base_root / DEFAULT_SHARED_FIGURES_DIRNAME / DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
    mixed_model_plotters = mixed_model_branch_render_specs(results, review=False)
    for plot_idx, spec in enumerate(mixed_model_plotters, start=1):
        scope = str(spec.get("scope") or "all_state")
        is_shared_scope = mixed_model_scope_is_shared(scope)
        if is_shared_scope and shared_mixed_model_dir is None:
            continue
        target_dir = shared_mixed_model_dir if is_shared_scope else mixed_model_dir
        with step_scope(f"figure plotter: {spec['name']}", index=plot_idx, total=len(mixed_model_plotters)):
            try:
                kwargs = {
                    "results": results,
                    "fig_dir": figure_nested_dir(target_dir, scope),
                    "output_name": spec["output_name"],
                    "title": spec["title"],
                    "model_key": str(spec.get("model_key", "mixed_model")),
                }
                if spec.get("accepts_scope") and spec.get("scope") is not None:
                    kwargs["scope"] = scope
                output_path = spec["plotter"](**kwargs)
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with {spec['name']}: {exc}")
                continue
            if output_path:
                saved.append(output_path)
    coactivity_dir = figure_family_dir(fig_dir, DEFAULT_SPINE_COACTIVITY_FIGURES_DIRNAME)
    coactivity_rows = results.get("spine_coactivity", {}).get("table_rows", [])
    coactivity_compartments = spine_coactivity_output_compartments(coactivity_rows)
    for plot_idx, compartment in enumerate(coactivity_compartments, start=1):
        with step_scope(
            f"figure plotter: spine_coactivity[{gallery_compartment_suffix(compartment)}]",
            index=plot_idx,
            total=len(coactivity_compartments),
        ):
            try:
                compartment_results = cached_spine_coactivity_results(compartment)
                distribution_path = plot_spine_coactivity_distribution_figure(
                    compartment_results,
                    spine_coactivity_figure_dir(coactivity_dir, "distribution"),
                    output_name=f"spine_coactivity_distribution_coefficient_{gallery_compartment_suffix(compartment)}.svg",
                    title=f"Review: Spine coactivity coefficients - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    value_kind="coactivity_r",
                )
                if distribution_path:
                    saved.append(distribution_path)
                else:
                    step_message(f"plotter returned no output: {spec.get('name', scope_label)}")
                distribution_pvalue_path = plot_spine_coactivity_distribution_figure(
                    compartment_results,
                    spine_coactivity_figure_dir(coactivity_dir, "distribution"),
                    output_name=f"spine_coactivity_distribution_pvalue_{gallery_compartment_suffix(compartment)}.svg",
                    title=f"Review: Spine coactivity shuffle p-values - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    value_kind="shuffle_p",
                )
                if distribution_pvalue_path:
                    saved.append(distribution_pvalue_path)
                else:
                    step_message(f"plotter returned no output: spine_coactivity_distribution_pvalue[{gallery_compartment_suffix(compartment)}]")
                heatmap_path = plot_spine_coactivity_tendency_figure(
                    compartment_results,
                    spine_coactivity_figure_dir(coactivity_dir, "pair_state_heatmap"),
                    output_name=f"spine_coactivity_heatmap_coefficient_{gallery_compartment_suffix(compartment)}.svg",
                    title=f"Review: Spine coactivity coefficient comparisons - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    value_kind="coactivity_r",
                )
                if heatmap_path:
                    saved.append(heatmap_path)
                else:
                    step_message(f"plotter returned no output: spine_coactivity_heatmap_coefficient[{gallery_compartment_suffix(compartment)}]")
                heatmap_pvalue_path = plot_spine_coactivity_tendency_figure(
                    compartment_results,
                    spine_coactivity_figure_dir(coactivity_dir, "pair_state_heatmap"),
                    output_name=f"spine_coactivity_heatmap_pvalue_{gallery_compartment_suffix(compartment)}.svg",
                    title=f"Review: Spine coactivity shuffle p-values - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    value_kind="shuffle_p",
                )
                if heatmap_pvalue_path:
                    saved.append(heatmap_pvalue_path)
                else:
                    step_message(f"plotter returned no output: spine_coactivity_heatmap_pvalue[{gallery_compartment_suffix(compartment)}]")
            except Exception as exc:
                eprint(f"[ALERT] Failed to create figure with spine coactivity plotter ({gallery_compartment_suffix(compartment)}): {exc}")
                continue
    anchor_rows = results.get("spine_coactivity", {}).get("table_rows", [])
    if spine_coactivity_anchor_compartments is None:
        spine_coactivity_anchor_compartments = spine_coactivity_anchor_state_compartments(anchor_rows)
    for plot_idx, compartment in enumerate(spine_coactivity_anchor_compartments, start=1):
        with step_scope(
            f"figure plotter: spine_coactivity_anchor[{gallery_compartment_suffix(compartment)}]",
            index=plot_idx,
            total=len(spine_coactivity_anchor_compartments),
        ):
            try:
                scope_results = results if compartment is None else cached_spine_coactivity_results(compartment)
                distribution_path = plot_spine_coactivity_distribution_figure(
                    scope_results,
                    spine_coactivity_figure_dir(coactivity_dir, "distribution"),
                    output_name=spine_coactivity_pair_state_output_name("anchor_distribution", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
                    title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive-pair distribution - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                    coactive_only=True,
                )
                if distribution_path:
                    saved.append(distribution_path)
                else:
                    step_message(f"plotter returned no output: spine_coactivity_anchor_distribution[{gallery_compartment_suffix(compartment)}]")
                heatmap_path = plot_spine_coactivity_pair_state_heatmap_figure(
                    scope_results,
                    spine_coactivity_figure_dir(coactivity_dir, "pair_state_heatmap"),
                    output_name=spine_coactivity_pair_state_output_name("pair_state_heatmap", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
                    title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive pairs across states - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                    coactive_only=True,
                )
                if heatmap_path:
                    saved.append(heatmap_path)
                else:
                    step_message(f"plotter returned no output: spine_coactivity_pair_state_heatmap[{gallery_compartment_suffix(compartment)}]")
                summary_path = plot_spine_coactivity_pair_state_summary_figure(
                    scope_results,
                    spine_coactivity_figure_dir(coactivity_dir, "pair_state_summary"),
                    output_name=spine_coactivity_pair_state_output_name("pair_state_summary", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
                    title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive-pair summary - {gallery_compartment_title(compartment)}",
                    compartment_filter=compartment,
                    anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                    coactive_only=True,
                )
                if summary_path:
                    saved.append(summary_path)
                else:
                    step_message(f"plotter returned no output: spine_coactivity_anchor_summary[{gallery_compartment_suffix(compartment)}]")
            except Exception as exc:
                eprint(f"[ALERT] Failed to create quiet-awake-movies coactivity figures ({gallery_compartment_suffix(compartment)}): {exc}")
                continue
    with step_scope("figure family: matrix similarity review"):
        matrix_rows = results.get("matrix_similarity", [])
        compartments = matrix_similarity_output_compartments(matrix_rows)
        for comp_idx, compartment in enumerate(compartments, start=1):
            with step_scope(
                f"matrix heatmaps: {gallery_compartment_title(compartment)}",
                index=comp_idx,
                total=len(compartments),
            ):
                try:
                    output_path = plot_matrix_similarity_heatmap(
                        results,
                        figure_family_dir(fig_dir, DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME),
                        output_name=f"matrix_similarity_heatmap_{compartment}.svg",
                        title=f"Matrix spine-spine similarity\n{gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                    )
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create matrix heatmap for {compartment}: {exc}")
                    continue
                if output_path:
                    saved.append(output_path)
                matrix_results = cached_matrix_similarity_results(compartment)
                dendrite_ids = sorted(
                    {
                        str(row.get("global_dendrite_id"))
                        for row in matrix_results.get("matrix_similarity", [])
                        if row.get("global_dendrite_id") is not None
                    }
                )
                for dend_idx, dendrite_id in enumerate(dendrite_ids, start=1):
                    step_progress(dend_idx, len(dendrite_ids), label=str(dendrite_id))
                    dendrite_results = cached_matrix_similarity_results(compartment, dendrite_id)
                    dendrite_rows = dendrite_results.get("matrix_similarity", [])
                    if not dendrite_rows:
                        continue
                    representative = sorted(
                        dendrite_rows,
                        key=lambda row: (
                            str(row.get("animal_id", "")),
                            str(row.get("day_id", row.get("exp_id", ""))),
                            str(row.get("state_a", "")),
                            str(row.get("state_b", "")),
                        ),
                    )[0]
                    dendrite_output_path = build_matrix_similarity_day_figure_path(
                        fig_dir,
                        representative.get("animal_id"),
                        representative.get("day_id", representative.get("exp_id")),
                        representative.get("compartment", compartment),
                        dendrite_id,
                    )
                    try:
                        dendrite_path = plot_matrix_similarity_heatmap(
                            dendrite_results,
                            dendrite_output_path.parent,
                            output_name=dendrite_output_path.name,
                            title=(
                                "Matrix spine-spine similarity\n"
                                f"{format_dendrite_display_name(representative['animal_id'], representative.get('compartment', compartment), representative['global_dendrite_id'])}"
                            ),
                            compartment_filter=compartment,
                            dendrite_filter=dendrite_id,
                        )
                    except Exception as exc:
                        eprint(f"[ALERT] Failed to create matrix heatmap for {compartment} / {dendrite_id}: {exc}")
                        continue
                    if dendrite_path:
                        saved.append(dendrite_path)
    with step_scope("figure family: spine coactivity review"):
        coactivity_rows = results.get("spine_coactivity", {}).get("table_rows", [])
        coactivity_compartments = spine_coactivity_output_compartments(coactivity_rows)
        for comp_idx, compartment in enumerate(coactivity_compartments, start=1):
            with step_scope(
                f"spine coactivity heatmaps: {gallery_compartment_title(compartment)}",
                index=comp_idx,
                total=len(coactivity_compartments),
            ):
                compartment_rows = filter_rows_by_spine_coactivity(coactivity_rows, compartment_filter=compartment)
                dendrite_ids = sorted(
                    {
                        str(row.get("global_dendrite_id"))
                        for row in compartment_rows
                        if row.get("global_dendrite_id") is not None and str(row.get("status")) == "ok"
                    }
                )
                for dend_idx, dendrite_id in enumerate(dendrite_ids, start=1):
                    step_progress(dend_idx, len(dendrite_ids), label=str(dendrite_id))
                    dendrite_results = cached_spine_coactivity_results(compartment, dendrite_id)
                    dendrite_rows = dendrite_results.get("spine_coactivity", {}).get("table_rows", [])
                    if not dendrite_rows:
                        continue
                    representative = sorted(
                        dendrite_rows,
                        key=lambda row: (
                            str(row.get("animal_id", "")),
                            str(row.get("day_id", row.get("exp_id", ""))),
                            str(row.get("state", "")),
                            str(row.get("global_pair_id", "")),
                        ),
                    )[0]
                    dendrite_output_path = build_spine_coactivity_day_figure_path(
                        fig_dir,
                        representative.get("animal_id"),
                        representative.get("day_id", representative.get("exp_id")),
                        representative.get("compartment", compartment),
                        dendrite_id,
                    )
                    try:
                        dendrite_path = plot_spine_coactivity_tendency_figure(
                            dendrite_results,
                            dendrite_output_path.parent,
                            output_name=dendrite_output_path.name,
                            title=(
                                "Spine coactivity heatmap across states\n"
                                f"{format_dendrite_display_name(representative['animal_id'], representative.get('compartment', compartment), representative['global_dendrite_id'])}"
                            ),
                        )
                    except Exception as exc:
                        eprint(f"[ALERT] Failed to create spine coactivity heatmap for {compartment} / {dendrite_id}: {exc}")
                        continue
                    if dendrite_path:
                        saved.append(dendrite_path)
    with step_scope("cleanup ROI detail figures"):
        removed_detail_files = cleanup_roi_detail_figures(fig_dir)
        if removed_detail_files:
            step_message(f"removed {len(removed_detail_files)} stale ROI detail PNG/SVG files")
    with step_scope("figure family: spine coactivity basal/apical distribution"):
        basal_vs_apical_path = plot_spine_coactivity_basal_apical_distribution_figure(
            results,
            spine_coactivity_figure_dir(coactivity_dir, "basal_apical_distribution"),
            output_name=spine_coactivity_basal_apical_distribution_output_name(SPINE_COACTIVITY_ANCHOR_STATE, True),
            title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive-pair distribution - basal vs apical",
            anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        if basal_vs_apical_path:
            saved.append(basal_vs_apical_path)
        else:
            step_message("plotter returned no output: spine_coactivity_basal_apical_distribution")
    return saved


def generate_review_figures(
    output_dir: Path,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    review_root: Optional[Path] = None,
) -> List[str]:
    if plt is None:
        eprint("[ALERT] matplotlib is unavailable; skipping review figure generation.")
        return []
    review_dir = ensure_dir(Path(review_root) if review_root is not None else DEFAULT_REVIEW_FIGURES_DIR)
    summary_review_dir = state_summary_figure_dir(review_dir)
    saved: List[str] = []
    state_labels = selected_matrix_state_labels(results)
    basal_apical_state_labels = selected_basal_apical_state_labels(results)
    present_compartments = sorted_present_compartments(cache)
    spine_coactivity_results_cache: Dict[Optional[str], Dict[str, Any]] = {}
    spine_coactivity_anchor_compartments: Optional[List[Optional[str]]] = None

    def cached_spine_coactivity_results(compartment: Optional[str]) -> Dict[str, Any]:
        if compartment not in spine_coactivity_results_cache:
            spine_coactivity_results_cache[compartment] = build_filtered_spine_coactivity_results(results, compartment)
        return spine_coactivity_results_cache[compartment]
    with step_scope("review figure prep: state summary y-limits"):
        y_limits = state_summary_y_limits(cache, state_labels)
    with step_scope("review figure prep: state summary comparison y-limits"):
        comparison_y_limits = state_summary_y_limits(cache, basal_apical_state_labels)
    with step_scope("review figure prep: state summary overview results"):
        overview_results = results
    with step_scope("review figure prep: state summary basal results"):
        basal_results = build_state_summary_gallery_results(cache, state_labels, "basal")
    with step_scope("review figure prep: state summary apical results"):
        apical_results = build_state_summary_gallery_results(cache, state_labels, "apical")
    review_specs = [
        {
            "kind": "overview",
            "compartment": None,
            "output_name": "review_state_summary_boxplots.svg",
            "title": "Review: Selected-state summary distributions - All compartments",
            "results": overview_results,
        },
    ]
    for compartment in [comp for comp in ["basal", "apical"] if comp in present_compartments]:
        review_specs.append(
            {
                "kind": "overview",
                "compartment": compartment,
                "output_name": f"review_state_summary_boxplots_{compartment}.svg",
                "title": f"Review: Selected-state summary distributions - {gallery_compartment_title(compartment)}",
                "results": basal_results if compartment == "basal" else apical_results,
            }
        )
    review_specs.append(
        {
            "kind": "comparison",
            "output_name": "review_state_summary_boxplots_basal_vs_apical.svg",
            "title": "Review: Selected-state summary distributions - Basal vs apical",
            "results": (basal_results, apical_results),
        }
    )
    for plot_idx, spec in enumerate(review_specs, start=1):
        compartment = spec.get("compartment")
        scope_label = gallery_compartment_suffix(compartment) if spec["kind"] == "overview" else "basal_vs_apical"
        with step_scope(
            f"review figure plotter: state_summary_boxplots[{scope_label}]",
            index=plot_idx,
            total=len(review_specs),
        ):
            try:
                if spec["kind"] == "overview":
                    summary_results = spec["results"]
                    output_path = plot_state_summary_figure(
                        summary_results,
                        summary_review_dir,
                        output_name=spec["output_name"],
                        title=spec["title"],
                        state_labels=state_labels,
                        y_limits=y_limits,
                    )
                else:
                    basal_summary, apical_summary = spec["results"]
                    output_path = plot_state_summary_compartment_comparison_figure(
                        basal_summary,
                        apical_summary,
                        summary_review_dir,
                        output_name=spec["output_name"],
                        title=spec["title"],
                        state_labels=basal_apical_state_labels,
                        y_limits=comparison_y_limits,
                        comparison_rows=[
                            row
                            for row in results.get("basal_apical_comparisons", [])
                            if str(row.get("comparison")) == "basal_vs_apical"
                            and is_significant_row(row)
                            and str(row.get("state")) in set(basal_apical_state_labels)
                        ],
                        cohort_label=str(spec.get("cohort_label") or "all"),
                    )
            except Exception as exc:
                eprint(f"[ALERT] Failed to create review figure ({scope_label}): {exc}")
                continue
            if not output_path:
                step_message(f"plotter returned no output: {spec.get('name', scope_label)}")
                continue
            output_path_obj = Path(output_path)
            saved.append(str(output_path_obj))
            svg_path = output_path_obj.with_suffix(".svg")
            if svg_path.exists():
                saved.append(str(svg_path))
            component_dir = summary_review_dir / f"{output_path_obj.stem}_components"
            if component_dir.exists():
                for component_path in sorted(component_dir.glob("*.svg")):
                    saved.append(str(component_path))
    with step_scope("review family: direct trial-type figures"):
        direct_review_dir = figure_family_dir(review_dir, DEFAULT_DIRECT_TRIAL_TYPE_FIGURES_DIRNAME)
        direct_review_specs = [
            {
                "name": "direct_trial_type_distribution",
                "output_name": "review_direct_trial_type_distribution.svg",
                "title": "Review: Direct trial-type comparison - video means by state",
                "plotter": plot_direct_trial_type_distribution_figure,
            },
            {
                "name": "direct_trial_type_state_comparison",
                "output_name": "review_direct_trial_type_state_comparison.svg",
                "title": "Review: Direct trial-type comparison - state pair scatter",
                "plotter": plot_direct_trial_type_state_comparison_figure,
            },
        ]
        for plot_idx, spec in enumerate(direct_review_specs, start=1):
            with step_scope(f"review figure plotter: {spec['name']}", index=plot_idx, total=len(direct_review_specs)):
                try:
                    output_path = spec["plotter"](
                        results,
                        direct_review_dir,
                        output_name=spec["output_name"],
                        title=spec["title"],
                    )
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create review figure ({spec['name']}): {exc}")
                    continue
                if not output_path:
                    step_message(f"plotter returned no output: {spec.get('name', scope_label)}")
                    continue
                output_path_obj = Path(output_path)
                saved.append(str(output_path_obj))
                svg_path = output_path_obj.with_suffix(".svg")
                if svg_path.exists():
                    saved.append(str(svg_path))
    run_params = results.get("run_parameters", {}) if isinstance(results.get("run_parameters"), dict) else {}
    comparison_preset_name = str(run_params.get("comparison_preset_name") or "default")
    generate_shared_general_figures = bool(run_params.get("generate_shared_general_figures", True))
    shared_mixed_model_dir = None
    if generate_shared_general_figures:
        base_root = review_dir
        shared_mixed_model_dir = ensure_dir(base_root / DEFAULT_SHARED_FIGURES_DIRNAME / DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
    with step_scope("review family: mixed model figures"):
        mixed_model_review_dir = figure_family_dir(review_dir, DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
        mixed_model_specs = mixed_model_branch_render_specs(results, review=True)
        for plot_idx, spec in enumerate(mixed_model_specs, start=1):
            scope = str(spec.get("scope") or "all_state")
            is_shared_scope = mixed_model_scope_is_shared(scope)
            if is_shared_scope and shared_mixed_model_dir is None:
                continue
            target_dir = shared_mixed_model_dir if is_shared_scope else mixed_model_review_dir
            with step_scope(f"review figure plotter: {spec['name']}", index=plot_idx, total=len(mixed_model_specs)):
                try:
                    kwargs = {
                        "results": results,
                        "fig_dir": figure_nested_dir(target_dir, scope),
                        "output_name": spec["output_name"],
                        "title": spec["title"],
                        "model_key": str(spec.get("model_key", "mixed_model")),
                    }
                    if spec.get("accepts_scope") and spec.get("scope") is not None:
                        kwargs["scope"] = scope
                    output_path = spec["plotter"](**kwargs)
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create review figure ({spec['name']}): {exc}")
                    continue
                if not output_path:
                    step_message(f"plotter returned no output: {spec.get('name', scope_label)}")
                    continue
                output_path_obj = Path(output_path)
                saved.append(str(output_path_obj))
                svg_path = output_path_obj.with_suffix(".svg")
                if svg_path.exists():
                    saved.append(str(svg_path))
    with step_scope("review family: spine coactivity figures"):
        coactivity_review_dir = review_dir
        coactivity_rows = results.get("spine_coactivity", {}).get("table_rows", [])
        coactivity_compartments = spine_coactivity_output_compartments(coactivity_rows)
        for plot_idx, compartment in enumerate(coactivity_compartments, start=1):
            with step_scope(
                f"review figure plotter: spine_coactivity[{gallery_compartment_suffix(compartment)}]",
                index=plot_idx,
                total=len(coactivity_compartments),
            ):
                try:
                    compartment_results = cached_spine_coactivity_results(compartment)
                    distribution_path = plot_spine_coactivity_distribution_figure(
                        compartment_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "distribution"),
                        output_name=f"review_spine_coactivity_distribution_{gallery_compartment_suffix(compartment)}.svg",
                        title=f"Review: Spine coactivity distributions - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                    )
                    if distribution_path:
                        saved.append(distribution_path)
                    heatmap_path = plot_spine_coactivity_tendency_figure(
                        compartment_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "pair_state_heatmap"),
                        output_name=f"review_spine_coactivity_heatmap_{gallery_compartment_suffix(compartment)}.svg",
                        title=f"Review: Derived state-state similarity of coactivity coefficient - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                    )
                    if heatmap_path:
                        saved.append(heatmap_path)
                    pair_heatmap_path = plot_spine_coactivity_pair_state_heatmap_figure(
                        compartment_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "pair_state_heatmap"),
                        output_name=f"review_spine_coactivity_pair_state_heatmap_{gallery_compartment_suffix(compartment)}.svg",
                        title=f"Review: Spine coactivity coefficient across selected states - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                    )
                    if pair_heatmap_path:
                        saved.append(pair_heatmap_path)
                    else:
                        step_message(f"plotter returned no output: spine_coactivity_pair_state_heatmap[{gallery_compartment_suffix(compartment)}]")
                    pair_summary_path = plot_spine_coactivity_pair_state_summary_figure(
                        compartment_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "pair_state_summary"),
                        output_name=f"review_spine_coactivity_pair_state_summary_{gallery_compartment_suffix(compartment)}.svg",
                        title=f"Review: Spine coactivity state-change summary - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                    )
                    if pair_summary_path:
                        saved.append(pair_summary_path)
                    else:
                        step_message(f"plotter returned no output: spine_coactivity_pair_state_summary[{gallery_compartment_suffix(compartment)}]")
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create review figure (spine_coactivity[{compartment}]): {exc}")
        anchor_rows = results.get("spine_coactivity", {}).get("table_rows", [])
        if spine_coactivity_anchor_compartments is None:
            spine_coactivity_anchor_compartments = spine_coactivity_anchor_state_compartments(anchor_rows)
        for plot_idx, compartment in enumerate(spine_coactivity_anchor_compartments, start=1):
            with step_scope(
                f"review figure plotter: spine_coactivity_anchor[{gallery_compartment_suffix(compartment)}]",
                index=plot_idx,
                total=len(spine_coactivity_anchor_compartments),
            ):
                try:
                    scope_results = results if compartment is None else cached_spine_coactivity_results(compartment)
                    distribution_path = plot_spine_coactivity_distribution_figure(
                        scope_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "distribution"),
                        output_name=spine_coactivity_pair_state_output_name("anchor_distribution", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
                        title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive-pair distribution - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                        state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                        coactive_only=True,
                    )
                    if distribution_path:
                        saved.append(distribution_path)
                    heatmap_path = plot_spine_coactivity_pair_state_heatmap_figure(
                        scope_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "pair_state_heatmap"),
                        output_name=spine_coactivity_pair_state_output_name("pair_state_heatmap", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
                        title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive pairs across states - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                        anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                        coactive_only=True,
                    )
                    if heatmap_path:
                        saved.append(heatmap_path)
                    summary_path = plot_spine_coactivity_pair_state_summary_figure(
                        scope_results,
                        spine_coactivity_figure_dir(coactivity_review_dir, "pair_state_summary"),
                        output_name=spine_coactivity_pair_state_output_name("pair_state_summary", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
                        title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive-pair summary - {gallery_compartment_title(compartment)}",
                        compartment_filter=compartment,
                        anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                        coactive_only=True,
                    )
                    if summary_path:
                        saved.append(summary_path)
                    else:
                        step_message(f"plotter returned no output: spine_coactivity_pair_state_summary[{gallery_compartment_suffix(compartment)}]")
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create quiet-awake-movies coactivity review figures ({gallery_compartment_suffix(compartment)}): {exc}")
        basal_vs_apical_path = plot_spine_coactivity_basal_apical_distribution_figure(
            results,
            spine_coactivity_figure_dir(coactivity_review_dir, "basal_apical_distribution"),
            output_name=spine_coactivity_basal_apical_distribution_output_name(SPINE_COACTIVITY_ANCHOR_STATE, True),
            title=f"{format_requested_state_label(SPINE_COACTIVITY_ANCHOR_STATE)} coactive-pair distribution - basal vs apical",
            anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        if basal_vs_apical_path:
            saved.append(basal_vs_apical_path)
    return saved


def ordered_compartment_levels(values: Sequence[Any]) -> List[str]:
    seen = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.append(text)
    priority = {"basal": 0, "apical": 1, "sleep": 2, "movie": 3, "other": 4}
    return sorted(seen, key=lambda item: (priority.get(item, 10), item))


def split_day_id(day_id: Any) -> Tuple[str, str, str]:
    parts = [part for part in str(day_id or "").split("|") if part]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], "other"
    if len(parts) == 1:
        return parts[0], "unknown", "other"
    return "unknown", "unknown", "other"


def build_matrix_similarity_day_figure_path(
    output_dir: Path,
    animal_id: Any,
    day_id: Any,
    compartment: Any,
    global_dendrite_id: Any,
) -> Path:
    day_animal_id, day_date, day_compartment = split_day_id(day_id)
    animal_slug = safe_filename_component(animal_id or day_animal_id or "unknown_animal")
    compartment_slug = safe_filename_component(day_figure_compartment_folder(compartment or day_compartment))
    date_slug = safe_filename_component(day_date or "unknown_date")
    dendrite_slug = safe_filename_component(extract_dendrite_token(global_dendrite_id))
    figure_dir = figure_family_dir(output_dir, DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME, animal_slug, compartment_slug, date_slug)
    figure_name = f"{animal_slug}_{compartment_slug}_{date_slug}_{dendrite_slug}_matrix_similarity_heatmap.svg"
    return figure_dir / figure_name


def build_spine_coactivity_day_figure_path(
    output_dir: Path,
    animal_id: Any,
    day_id: Any,
    compartment: Any,
    global_dendrite_id: Any,
    *,
    figure_kind: str = "pair_state_heatmap",
) -> Path:
    day_animal_id, day_date, day_compartment = split_day_id(day_id)
    animal_slug = safe_filename_component(animal_id or day_animal_id or "unknown_animal")
    compartment_slug = safe_filename_component(day_figure_compartment_folder(compartment or day_compartment))
    date_slug = safe_filename_component(day_date or "unknown_date")
    dendrite_slug = safe_filename_component(extract_dendrite_token(global_dendrite_id))
    figure_dir = spine_coactivity_figure_dir(output_dir, figure_kind, animal_slug, compartment_slug, date_slug)
    figure_name = f"{animal_slug}_{compartment_slug}_{date_slug}_{dendrite_slug}_spine_coactivity_heatmap.svg"
    return figure_dir / figure_name


def render_analysis_family_figures(
    output_dir: Path,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    family: str,
    *,
    figure_root: Optional[Path] = None,
) -> List[str]:
    if plt is None:
        return []
    fig_dir = ensure_dir(Path(figure_root) if figure_root is not None else (output_dir / "figures"))
    saved: List[str] = []
    def record(path: Optional[str] | Sequence[str], plot_name: Optional[str] = None) -> None:
        if path:
            if isinstance(path, (list, tuple, set)):
                for item in path:
                    if item:
                        saved.append(str(item))
            else:
                saved.append(str(path))
            return
        resolved_plot_name = plot_name or "unknown plot"
        try:
            import inspect
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            locals_map = caller.f_locals if caller is not None else {}
            if resolved_plot_name == "unknown plot":
                spec = locals_map.get("spec")
                if isinstance(spec, dict):
                    resolved_plot_name = str(spec.get("name") or spec.get("output_name") or spec.get("title") or resolved_plot_name)
                else:
                    resolved_plot_name = str(locals_map.get("scope_label") or locals_map.get("plot_name") or resolved_plot_name)
        except Exception:
            pass
        step_message(f"plotter returned no output: {resolved_plot_name}")
    coactivity_dir = figure_family_dir(fig_dir, DEFAULT_SPINE_COACTIVITY_FIGURES_DIRNAME)
    matrix_similarity_results_cache: Dict[tuple, Dict[str, Any]] = {}
    spine_coactivity_results_cache: Dict[tuple, Dict[str, Any]] = {}
    spine_coactivity_anchor_compartments: Optional[List[Optional[str]]] = None

    def cached_matrix_similarity_results(compartment: Optional[str], dendrite_id: Optional[str] = None) -> Dict[str, Any]:
        key = (compartment, dendrite_id)
        if key not in matrix_similarity_results_cache:
            matrix_similarity_results_cache[key] = build_filtered_matrix_similarity_results(
                results,
                compartment,
                dendrite_id,
            )
        return matrix_similarity_results_cache[key]

    def cached_spine_coactivity_results(compartment: Optional[str], dendrite_id: Optional[str] = None) -> Dict[str, Any]:
        key = (compartment, dendrite_id)
        if key not in spine_coactivity_results_cache:
            spine_coactivity_results_cache[key] = build_filtered_spine_coactivity_results(
                results,
                compartment,
                dendrite_id,
            )
        return spine_coactivity_results_cache[key]

    if family == "state":
        summary_fig_dir = state_summary_figure_dir(fig_dir)
        state_labels = selected_matrix_state_labels(results)
        basal_apical_state_labels = selected_basal_apical_state_labels(results)
        present_compartments = sorted_present_compartments(cache)
        y_limits = state_summary_y_limits(cache, state_labels)
        comparison_y_limits = state_summary_y_limits(cache, basal_apical_state_labels)
        overview_results = results
        basal_results = build_state_summary_gallery_results(cache, state_labels, "basal")
        apical_results = build_state_summary_gallery_results(cache, state_labels, "apical")
        state_summary_specs = [
            {
                "kind": "overview",
                "name": "state_summary_boxplots_all",
                "compartment": None,
                "output_name": "state_summary_boxplots.svg",
                "title": "Selected-state summary distributions - All compartments",
                "results": overview_results,
            },
        ]
        for compartment in [comp for comp in ["basal", "apical"] if comp in present_compartments]:
            state_summary_specs.append(
                {
                    "kind": "overview",
                    "name": f"state_summary_boxplots_{compartment}",
                    "compartment": compartment,
                    "output_name": f"state_summary_boxplots_{compartment}.svg",
                    "title": f"Selected-state summary distributions - {gallery_compartment_title(compartment)}",
                    "results": basal_results if compartment == "basal" else apical_results,
                }
            )
        state_summary_specs.extend(
            [
                {
                    "kind": "comparison",
                    "name": "state_summary_boxplots_basal_vs_apical",
                    "output_name": "state_summary_boxplots_basal_vs_apical.svg",
                    "title": "Selected-state summary distributions - Basal vs apical",
                    "results": (basal_results, apical_results),
                },
            ]
        )
        for plot_idx, spec in enumerate(state_summary_specs, start=1):
            compartment = spec.get("compartment")
            scope_label = gallery_compartment_suffix(compartment) if spec["kind"] == "overview" else "basal_vs_apical"
            with step_scope(
                f"figure plotter: state_summary_boxplots[{scope_label}]",
                index=plot_idx,
                total=len(state_summary_specs),
            ):
                try:
                    if spec["kind"] == "overview":
                        summary_results = spec["results"]
                        record(
                            plot_state_summary_figure(
                                summary_results,
                                summary_fig_dir,
                                output_name=spec["output_name"],
                                title=spec["title"],
                                state_labels=state_labels,
                                y_limits=y_limits,
                            )
                        )
                    else:
                        basal_summary, apical_summary = spec["results"]
                        record(
                            plot_state_summary_compartment_comparison_figure(
                                basal_summary,
                                apical_summary,
                                summary_fig_dir,
                                output_name=spec["output_name"],
                                title=spec["title"],
                                state_labels=basal_apical_state_labels,
                                y_limits=comparison_y_limits,
                                comparison_rows=[
                                    row
                                    for row in results.get("basal_apical_comparisons", [])
                                    if str(row.get("comparison")) == "basal_vs_apical"
                                    and is_significant_row(row)
                                    and str(row.get("state")) in set(basal_apical_state_labels)
                                ],
                            )
                        )
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create figure with state summary plotter ({scope_label}): {exc}")
    elif family == "basal_apical":
        record(plot_basal_apical_summary(results, fig_dir), plot_name="basal_apical_summary")
    elif family == "direct_trial_type_comparison":
        direct_dir = figure_family_dir(fig_dir, DEFAULT_DIRECT_TRIAL_TYPE_FIGURES_DIRNAME)
        direct_plotters = [
            {
                "name": "direct_trial_type_distribution",
                "output_name": "direct_trial_type_distribution.svg",
                "title": "Direct trial-type comparison - video means by state",
                "plotter": plot_direct_trial_type_distribution_figure,
            },
            {
                "name": "direct_trial_type_state_comparison",
                "output_name": "direct_trial_type_state_comparison.svg",
                "title": "Direct trial-type comparison - state pair scatter",
                "plotter": plot_direct_trial_type_state_comparison_figure,
            },
        ]
        for plot_idx, spec in enumerate(direct_plotters, start=1):
            with step_scope(f"figure plotter: {spec['name']}", index=plot_idx, total=len(direct_plotters)):
                try:
                    output_path = spec["plotter"](
                        results,
                        direct_dir,
                        output_name=spec["output_name"],
                        title=spec["title"],
                    )
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create figure with {spec['name']}: {exc}")
                    continue
                if output_path:
                    record(output_path)
    elif family == "correlation":
        record(
            plot_correlation_summary(
                results,
                figure_family_dir(fig_dir, DEFAULT_CORRELATION_FIGURES_DIRNAME),
                output_name="correlation_summary.svg",
                title="Correlation summaries",
            ),
            plot_name="correlation_summary",
        )
    elif family == "matrix_similarity":
        matrix_rows = results.get("matrix_similarity", [])
        compartments = matrix_similarity_output_compartments(matrix_rows)
        for comp_idx, compartment in enumerate(compartments, start=1):
            with step_scope(
                f"matrix similarity figures: {gallery_compartment_title(compartment)}",
                index=comp_idx,
                total=len(compartments),
            ):
                try:
                    compartment_results = cached_matrix_similarity_results(compartment)
                    record(
                        plot_matrix_similarity_distribution(
                            compartment_results,
                            figure_family_dir(fig_dir, DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME),
                            output_name=f"matrix_similarity_distribution_{compartment}.svg",
                            title=f"Spine-spine coefficient distributions - {gallery_compartment_title(compartment)}",
                            compartment_filter=compartment,
                        ),
                        plot_name=f"matrix_similarity_distribution[{compartment}]",
                    )
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create matrix distribution for {compartment}: {exc}")
                    continue
                matrix_results = cached_matrix_similarity_results(compartment)
                dendrite_ids = sorted(
                    {
                        str(row.get("global_dendrite_id"))
                        for row in matrix_results.get("matrix_similarity", [])
                        if row.get("global_dendrite_id") is not None
                    }
                )
                for dend_idx, dendrite_id in enumerate(dendrite_ids, start=1):
                    step_progress(dend_idx, len(dendrite_ids), label=str(dendrite_id))
                    dendrite_results = cached_matrix_similarity_results(compartment, dendrite_id)
                    dendrite_rows = dendrite_results.get("matrix_similarity", [])
                    if not dendrite_rows:
                        continue
                    representative = sorted(
                        dendrite_rows,
                        key=lambda row: (
                            str(row.get("animal_id", "")),
                            str(row.get("day_id", row.get("exp_id", ""))),
                            str(row.get("state_a", "")),
                            str(row.get("state_b", "")),
                        ),
                    )[0]
                    dendrite_output_path = build_matrix_similarity_day_figure_path(
                        fig_dir,
                        representative.get("animal_id"),
                        representative.get("day_id", representative.get("exp_id")),
                        representative.get("compartment", compartment),
                        dendrite_id,
                    )
                    try:
                        record(
                            plot_matrix_similarity_heatmap(
                                dendrite_results,
                                dendrite_output_path.parent,
                                output_name=dendrite_output_path.name,
                                title=(
                                    "Derived matrix similarity state-state r - "
                                    f"{format_dendrite_display_name(representative['animal_id'], representative.get('compartment', compartment), representative['global_dendrite_id'])}"
                                ),
                                compartment_filter=compartment,
                                dendrite_filter=dendrite_id,
                            ),
                            plot_name=f"matrix_similarity_heatmap[{compartment}/{dendrite_id}]",
                        )
                    except Exception as exc:
                        eprint(f"[ALERT] Failed to create matrix heatmap for {compartment} / {dendrite_id}: {exc}")
    elif family == "mixed_model":
        mixed_model_dir = figure_family_dir(fig_dir, DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
        mixed_model_plotters = mixed_model_branch_render_specs(results, review=False)
        for plot_idx, spec in enumerate(mixed_model_plotters, start=1):
            with step_scope(f"figure plotter: {spec['name']}", index=plot_idx, total=len(mixed_model_plotters)):
                try:
                    kwargs = {
                        "results": results,
                        "fig_dir": figure_nested_dir(mixed_model_dir, str(spec.get("scope") or "all_state")),
                        "output_name": spec["output_name"],
                        "title": spec["title"],
                        "model_key": str(spec.get("model_key", "mixed_model")),
                    }
                    if spec.get("accepts_scope") and spec.get("scope") is not None:
                        kwargs["scope"] = str(spec["scope"])
                    record(spec["plotter"](**kwargs))
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create figure with {spec['name']}: {exc}")
        record(plot_demo_validation_figure(results, mixed_model_dir), plot_name="mixed_model_demo_validation")
    elif family == "spine_coactivity":
        coactivity_dir = figure_family_dir(fig_dir, DEFAULT_SPINE_COACTIVITY_FIGURES_DIRNAME)
        coactivity_rows = results.get("spine_coactivity", {}).get("table_rows", [])
        coactivity_compartments = spine_coactivity_output_compartments(coactivity_rows)
        for plot_idx, compartment in enumerate(coactivity_compartments, start=1):
            with step_scope(
                f"figure plotter: spine_coactivity[{gallery_compartment_suffix(compartment)}]",
                index=plot_idx,
                total=len(coactivity_compartments),
            ):
                try:
                    compartment_results = cached_spine_coactivity_results(compartment)
                    record(
                        plot_spine_coactivity_distribution_figure(
                            compartment_results,
                            spine_coactivity_figure_dir(coactivity_dir, "distribution"),
                            output_name=f"spine_coactivity_distribution_{gallery_compartment_suffix(compartment)}.svg",
                            title=f"Spine coactivity distributions - {gallery_compartment_title(compartment)}",
                            compartment_filter=compartment,
                        ),
                        plot_name=f"spine_coactivity_distribution[{gallery_compartment_suffix(compartment)}]",
                    )
                    record(
                        plot_spine_coactivity_tendency_figure(
                            compartment_results,
                            spine_coactivity_figure_dir(coactivity_dir, "pair_state_heatmap"),
                            output_name=f"spine_coactivity_heatmap_{gallery_compartment_suffix(compartment)}.svg",
                            title=f"Derived state-state similarity of coactivity coefficient - {gallery_compartment_title(compartment)}",
                            compartment_filter=compartment,
                        ),
                        plot_name=f"spine_coactivity_heatmap[{gallery_compartment_suffix(compartment)}]",
                    )
                    record(
                        plot_spine_coactivity_pair_state_heatmap_figure(
                            compartment_results,
                            spine_coactivity_figure_dir(coactivity_dir, "pair_state_heatmap"),
                            output_name=f"spine_coactivity_pair_state_heatmap_{gallery_compartment_suffix(compartment)}.svg",
                            title=f"Spine coactivity coefficient across selected states - {gallery_compartment_title(compartment)}",
                            compartment_filter=compartment,
                        ),
                        plot_name=f"spine_coactivity_pair_state_heatmap[{gallery_compartment_suffix(compartment)}]",
                    )
                    record(
                        plot_spine_coactivity_pair_state_summary_figure(
                            compartment_results,
                            spine_coactivity_figure_dir(coactivity_dir, "pair_state_summary"),
                            output_name=f"spine_coactivity_pair_state_summary_{gallery_compartment_suffix(compartment)}.svg",
                            title=f"Spine coactivity state-change summary - {gallery_compartment_title(compartment)}",
                            compartment_filter=compartment,
                        ),
                        plot_name=f"spine_coactivity_pair_state_summary[{gallery_compartment_suffix(compartment)}]",
                    )
                except Exception as exc:
                    eprint(f"[ALERT] Failed to create figure with spine coactivity plotter ({gallery_compartment_suffix(compartment)}): {exc}")
                    continue
                compartment_rows = filter_rows_by_spine_coactivity(coactivity_rows, compartment_filter=compartment)
                dendrite_ids = sorted(
                    {
                        str(row.get("global_dendrite_id"))
                        for row in compartment_rows
                        if row.get("global_dendrite_id") is not None and str(row.get("status")) == "ok"
                    }
                )
                for dend_idx, dendrite_id in enumerate(dendrite_ids, start=1):
                    step_progress(dend_idx, len(dendrite_ids), label=str(dendrite_id))
                    dendrite_results = build_filtered_spine_coactivity_results(results, compartment, dendrite_id)
                    dendrite_rows = dendrite_results.get("spine_coactivity", {}).get("table_rows", [])
                    if not dendrite_rows:
                        continue
                    representative = sorted(
                        dendrite_rows,
                        key=lambda row: (
                            str(row.get("animal_id", "")),
                            str(row.get("day_id", row.get("exp_id", ""))),
                            str(row.get("state", "")),
                            str(row.get("global_pair_id", "")),
                        ),
                    )[0]
                    dendrite_output_path = build_spine_coactivity_day_figure_path(
                        fig_dir,
                        representative.get("animal_id"),
                        representative.get("day_id", representative.get("exp_id")),
                        representative.get("compartment", compartment),
                        dendrite_id,
                        figure_kind="pair_state_heatmap",
                    )
                    try:
                        record(
                            plot_spine_coactivity_tendency_figure(
                                dendrite_results,
                                dendrite_output_path.parent,
                                output_name=dendrite_output_path.name,
                                title=(
                                    "Derived state-state similarity of coactivity coefficient - "
                                    f"{format_dendrite_display_name(representative['animal_id'], representative.get('compartment', compartment), representative['global_dendrite_id'])}"
                                ),
                                compartment_filter=compartment,
                            ),
                            plot_name=f"spine_coactivity_heatmap[{gallery_compartment_suffix(compartment)}/{dendrite_id}]",
                        )
                    except Exception as exc:
                        eprint(f"[ALERT] Failed to create spine coactivity heatmap for {compartment} / {dendrite_id}: {exc}")
    try:
        record(
            plot_spine_coactivity_basal_apical_distribution_figure(
                results,
                spine_coactivity_figure_dir(coactivity_dir, "basal_apical_distribution"),
                output_name=spine_coactivity_basal_apical_distribution_output_name(SPINE_COACTIVITY_ANCHOR_STATE, True),
                title=f"Quiet awake movies coactive-pair distribution - basal vs apical",
                anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
                coactive_only=True,
            ),
            plot_name="spine_coactivity_basal_apical_distribution",
        )
    except Exception as exc:
        eprint(f"[ALERT] Failed to create basal-vs-apical coactivity figure: {exc}")
    return saved
def generate_checkpoint_gallery(output_dir: Path, cache: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    if plt is None:
        eprint("[ALERT] matplotlib is unavailable; skipping checkpoint gallery generation.")
        return {"manifest_path": None, "entries": [], "files": []}
    gallery_dir = ensure_dir(output_dir / DEFAULT_CHECKPOINT_GALLERY_DIRNAME)
    entries: List[Dict[str, Any]] = []
    files: List[str] = []
    def append_entry(
        checkpoint: str,
        variant: str,
        path: Optional[str],
        *,
        compartment: Optional[str] = None,
        animal_id: Optional[str] = None,
        exp_id: Optional[str] = None,
        global_dendrite_id: Optional[str] = None,
        global_spine_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> None:
        if not path:
            step_message(f"skipped checkpoint gallery {checkpoint}[{variant}]: plotter returned no output")
            return
        path_obj = Path(path)
        try:
            rel_path = str(path_obj.relative_to(output_dir))
        except Exception:
            rel_path = str(path_obj)
        entry = {
            "checkpoint": checkpoint,
            "variant": variant,
            "file": rel_path,
            "title": path_obj.stem.replace("_", " "),
            "compartment": compartment,
            "animal_id": animal_id,
            "exp_id": exp_id,
            "global_dendrite_id": global_dendrite_id,
            "global_spine_id": global_spine_id,
            "scope": scope,
        }
        entries.append(entry)
        files.append(rel_path)
    present_compartments = sorted_present_compartments(cache)
    gallery_compartments = [None] + [comp for comp in ["basal", "apical"] if comp in present_compartments]
    matrix_similarity_results_cache: Dict[tuple, Dict[str, Any]] = {}
    spine_coactivity_results_cache: Dict[tuple, Dict[str, Any]] = {}
    spine_coactivity_anchor_compartments: Optional[List[Optional[str]]] = None

    def cached_matrix_similarity_results(compartment: Optional[str], dendrite_id: Optional[str] = None) -> Dict[str, Any]:
        key = (compartment, dendrite_id)
        if key not in matrix_similarity_results_cache:
            matrix_similarity_results_cache[key] = build_filtered_matrix_similarity_results(
                results,
                compartment,
                dendrite_id,
            )
        return matrix_similarity_results_cache[key]

    def cached_spine_coactivity_results(compartment: Optional[str], dendrite_id: Optional[str] = None) -> Dict[str, Any]:
        key = (compartment, dendrite_id)
        if key not in spine_coactivity_results_cache:
            spine_coactivity_results_cache[key] = build_filtered_spine_coactivity_results(
                results,
                compartment,
                dendrite_id,
            )
        return spine_coactivity_results_cache[key]
    # Loading / QC checkpoint examples.
    for compartment in gallery_compartments:
        path = plot_loading_qc_checkpoint(
            cache,
            results,
            gallery_dir,
            compartment_filter=compartment,
            output_name=f"01_loading_qc_{gallery_compartment_suffix(compartment)}.svg",
            title=f"Loading / QC - {gallery_compartment_title(compartment)}",
        )
        rep = select_representative_trace_record(cache, compartment_filter=compartment, require_spine=False)
        append_entry(
            "loading_qc",
            gallery_compartment_suffix(compartment),
            path,
            compartment=compartment,
            animal_id=None if rep is None else rep["animal_id"],
            exp_id=None if rep is None else rep["exp_id"],
            global_dendrite_id=None if rep is None else rep["global_dendrite_id"],
        )
    # Spine-specific regression QC checkpoint examples.
    for compartment in gallery_compartments:
        path = plot_spine_regression_qc_checkpoint(
            cache,
            gallery_dir,
            compartment_filter=compartment,
            output_name=f"02_spine_regression_qc_{gallery_compartment_suffix(compartment)}.svg",
            title=f"Spine-specific regression QC - {gallery_compartment_title(compartment)}",
        )
        rep = select_representative_trace_record(cache, compartment_filter=compartment, require_spine=True)
        append_entry(
            "regression_qc",
            gallery_compartment_suffix(compartment),
            path,
            compartment=compartment,
            animal_id=None if rep is None else rep["animal_id"],
            exp_id=None if rep is None else rep["exp_id"],
            global_dendrite_id=None if rep is None else rep["global_dendrite_id"],
            global_spine_id=None if rep is None else rep["global_spine_id"],
        )
    # State summary checkpoint examples.
    state_labels = selected_matrix_state_labels(results)
    basal_apical_state_labels = selected_basal_apical_state_labels(results)
    y_limits = state_summary_y_limits(cache, state_labels)
    comparison_y_limits = state_summary_y_limits(cache, basal_apical_state_labels)
    overview_results = results
    basal_results = build_state_summary_gallery_results(cache, state_labels, "basal")
    apical_results = build_state_summary_gallery_results(cache, state_labels, "apical")
    summary_gallery_dir = state_summary_figure_dir(gallery_dir)
    path = plot_state_summary_figure(
        overview_results,
        summary_gallery_dir,
        output_name="03_state_summary_all.svg",
        title="Selected-state summary distributions - All compartments",
        state_labels=state_labels,
        y_limits=y_limits,
    )
    rep = select_representative_trace_record(cache, compartment_filter=None, require_spine=False)
    append_entry(
        "state_summary",
        "all",
        path,
        animal_id=None if rep is None else rep["animal_id"],
        exp_id=None if rep is None else rep["exp_id"],
        global_dendrite_id=None if rep is None else rep["global_dendrite_id"],
    )
    for compartment in gallery_compartments[1:]:
        compartment_results = basal_results if compartment == "basal" else apical_results
        path = plot_state_summary_figure(
            compartment_results,
            summary_gallery_dir,
            output_name=f"03_state_summary_{gallery_compartment_suffix(compartment)}.svg",
            title=f"Selected-state summary distributions - {gallery_compartment_title(compartment)}",
            state_labels=state_labels,
            y_limits=y_limits,
        )
        rep = select_representative_trace_record(cache, compartment_filter=compartment, require_spine=False)
        append_entry(
            "state_summary",
            gallery_compartment_suffix(compartment),
            path,
            compartment=compartment,
            animal_id=None if rep is None else rep["animal_id"],
            exp_id=None if rep is None else rep["exp_id"],
            global_dendrite_id=None if rep is None else rep["global_dendrite_id"],
        )
    path = plot_state_summary_compartment_comparison_figure(
        basal_results,
        apical_results,
        summary_gallery_dir,
        output_name="03_state_summary_basal_vs_apical.svg",
        title="Selected-state summary distributions - Basal vs apical",
        state_labels=basal_apical_state_labels,
        y_limits=comparison_y_limits,
        comparison_rows=_state_summary_significant_basal_apical_rows(
            results.get("basal_apical_comparisons", []),
            state_order=basal_apical_state_labels,
            comparison_name="basal_vs_apical",
        ),
    )
    append_entry(
        "state_summary_comparison",
        "basal_vs_apical",
        path,
    )
    # Basal/apical summary checkpoint.
    path = plot_basal_apical_summary(results, gallery_dir)
    append_entry("basal_apical_summary", "combined", path)
    # Correlation checkpoint examples.
    for compartment in gallery_compartments:
        corr_results = build_filtered_correlation_results(results, compartment)
        path = plot_correlation_summary(
            corr_results,
            figure_family_dir(gallery_dir, DEFAULT_CORRELATION_FIGURES_DIRNAME),
            output_name=f"05_correlation_summary_{gallery_compartment_suffix(compartment)}.svg",
            title=f"Correlation summaries - {gallery_compartment_title(compartment)}",
        )
        rep = select_representative_trace_record(cache, compartment_filter=compartment, require_spine=True)
        append_entry(
            "correlation_summary",
            gallery_compartment_suffix(compartment),
            path,
            compartment=compartment,
            animal_id=None if rep is None else rep["animal_id"],
            exp_id=None if rep is None else rep["exp_id"],
            global_dendrite_id=None if rep is None else rep["global_dendrite_id"],
            global_spine_id=None if rep is None else rep["global_spine_id"],
        )
    # Spine-spine matrix similarity checkpoint examples.
    matrix_rows = results.get("matrix_similarity", [])
    for compartment in matrix_similarity_output_compartments(matrix_rows):
        matrix_results = cached_matrix_similarity_results(compartment)
        heatmap_path = plot_matrix_similarity_heatmap(
            matrix_results,
            figure_family_dir(gallery_dir, DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME),
            output_name=f"06_matrix_similarity_heatmap_{gallery_compartment_suffix(compartment)}.svg",
            title=f"Derived matrix similarity state-state r - {gallery_compartment_title(compartment)}",
            compartment_filter=compartment,
        )
        rep = None
        if matrix_results.get("matrix_similarity"):
            rep = sorted(
                matrix_results["matrix_similarity"],
                key=lambda row: (
                    str(row.get("animal_id", "")),
                    str(row.get("exp_id", "")),
                    str(row.get("global_dendrite_id", "")),
                    str(row.get("state_a", "")),
                    str(row.get("state_b", "")),
                ),
            )[0]
        append_entry(
            "matrix_similarity_heatmap",
            gallery_compartment_suffix(compartment),
            heatmap_path,
            compartment=compartment,
            animal_id=None if rep is None else rep.get("animal_id"),
            exp_id=None if rep is None else rep.get("exp_id"),
            global_dendrite_id=None if rep is None else rep.get("global_dendrite_id"),
        )
    dist_results = results.get("matrix_similarity", [])
    for compartment in matrix_similarity_output_compartments(dist_results):
        compartment_results = cached_matrix_similarity_results(compartment)
        dist_path = plot_matrix_similarity_distribution(
            compartment_results,
            figure_family_dir(gallery_dir, DEFAULT_MATRIX_SIMILARITY_FIGURES_DIRNAME),
            output_name=f"06_matrix_similarity_distribution_{gallery_compartment_suffix(compartment)}.svg",
            title=f"Spine-spine coefficient distributions - {gallery_compartment_title(compartment)}",
            compartment_filter=compartment,
        )
        rep = select_representative_trace_record(cache, compartment_filter=compartment, require_spine=True)
        append_entry(
            "matrix_similarity_distribution",
            gallery_compartment_suffix(compartment),
            dist_path,
            compartment=compartment,
            animal_id=None if rep is None else rep["animal_id"],
            exp_id=None if rep is None else rep["exp_id"],
            global_dendrite_id=None if rep is None else rep["global_dendrite_id"],
        )
    # Quiet-awake-movies spine coactivity checkpoint examples.
    anchor_rows = results.get("spine_coactivity", {}).get("table_rows", [])
    if spine_coactivity_anchor_compartments is None:
        spine_coactivity_anchor_compartments = spine_coactivity_anchor_state_compartments(anchor_rows)
    for compartment in spine_coactivity_anchor_compartments:
        scope_results = results if compartment is None else cached_spine_coactivity_results(compartment)
        distribution_path = plot_spine_coactivity_distribution_figure(
            scope_results,
            spine_coactivity_figure_dir(gallery_dir, "distribution"),
            output_name=spine_coactivity_pair_state_output_name("anchor_distribution", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
            title=f"Quiet awake movies coactive-pair distribution - {gallery_compartment_title(compartment)}",
            compartment_filter=compartment,
            state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        rep = None
        append_entry(
            "spine_coactivity_anchor_distribution",
            spine_coactivity_pair_state_scope_name(SPINE_COACTIVITY_ANCHOR_STATE, True) + f"_{gallery_compartment_suffix(compartment)}",
            distribution_path,
            compartment=compartment,
        )
        heatmap_path = plot_spine_coactivity_pair_state_heatmap_figure(
            scope_results,
            spine_coactivity_figure_dir(gallery_dir, "pair_state_heatmap"),
            output_name=spine_coactivity_pair_state_output_name("pair_state_heatmap", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
            title=f"Quiet awake movies coactive pairs across states - {gallery_compartment_title(compartment)}",
            compartment_filter=compartment,
            anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        rep = None
        spine_rows = scope_results.get("spine_coactivity", {}).get("pair_state_rows", []) if isinstance(scope_results.get("spine_coactivity", {}), dict) else []
        if spine_rows:
            rep = sorted(
                spine_rows,
                key=lambda row: (
                    str(row.get("animal_id", "")),
                    str(row.get("exp_id", "")),
                    str(row.get("global_dendrite_id", "")),
                    str(row.get("state", "")),
                    str(row.get("global_pair_id", "")),
                ),
            )[0]
        append_entry(
            "spine_coactivity_pair_state_heatmap",
            spine_coactivity_pair_state_scope_name(SPINE_COACTIVITY_ANCHOR_STATE, True) + f"_{gallery_compartment_suffix(compartment)}",
            heatmap_path,
            compartment=compartment,
            animal_id=None if rep is None else rep.get("animal_id"),
            exp_id=None if rep is None else rep.get("exp_id"),
            global_dendrite_id=None if rep is None else rep.get("global_dendrite_id"),
        )
        summary_path = plot_spine_coactivity_pair_state_summary_figure(
            scope_results,
            spine_coactivity_figure_dir(gallery_dir, "pair_state_summary"),
            output_name=spine_coactivity_pair_state_output_name("pair_state_summary", SPINE_COACTIVITY_ANCHOR_STATE, compartment, True),
            title=f"Quiet awake movies coactive-pair summary - {gallery_compartment_title(compartment)}",
            compartment_filter=compartment,
            anchor_state_filter=SPINE_COACTIVITY_ANCHOR_STATE,
            coactive_only=True,
        )
        append_entry(
            "spine_coactivity_pair_state_summary",
            spine_coactivity_pair_state_scope_name(SPINE_COACTIVITY_ANCHOR_STATE, True) + f"_{gallery_compartment_suffix(compartment)}",
            summary_path,
            compartment=compartment,
            animal_id=None if rep is None else rep.get("animal_id"),
            exp_id=None if rep is None else rep.get("exp_id"),
            global_dendrite_id=None if rep is None else rep.get("global_dendrite_id"),
        )
    # Mixed-model checkpoint examples.
    run_params = results.get("run_parameters", {}) if isinstance(results.get("run_parameters"), dict) else {}
    comparison_preset_name = str(run_params.get("comparison_preset_name") or "default")
    generate_shared_general_figures = bool(run_params.get("generate_shared_general_figures", True))
    shared_mixed_model_dir = None
    if generate_shared_general_figures:
        base_root = gallery_dir
        shared_mixed_model_dir = ensure_dir(base_root / DEFAULT_SHARED_FIGURES_DIRNAME / DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
    mixed_model_gallery_dir = figure_family_dir(gallery_dir, DEFAULT_MIXED_MODEL_FIGURES_DIRNAME)
    for spec in [item for item in mixed_model_branch_render_specs(results, review=False) if item.get("plotter") is plot_mixed_model_contrasts_checkpoint]:
        scope = str(spec["scope"])
        is_shared_scope = mixed_model_scope_is_shared(scope)
        if is_shared_scope and shared_mixed_model_dir is None:
            continue
        target_dir = shared_mixed_model_dir if is_shared_scope else mixed_model_gallery_dir
        path = spec["plotter"](
            results,
            figure_nested_dir(target_dir, scope),
            scope=scope,
            output_name=spec["output_name"],
            title=spec["title"],
            model_key=str(spec.get("model_key", "mixed_model")),
        )
        append_entry("mixed_model_contrasts", scope, path, scope=scope)
    # Direct trial-type comparison checkpoint examples.
    path = plot_direct_trial_type_distribution_figure(
        results,
        figure_family_dir(gallery_dir, DEFAULT_DIRECT_TRIAL_TYPE_FIGURES_DIRNAME),
        output_name="08_direct_trial_type_distribution_all.svg",
        title="Direct trial-type comparison - video means by state",
    )
    append_entry("direct_trial_type_distribution", "all", path, scope="direct_trial_type_comparison")
    path = plot_direct_trial_type_state_comparison_figure(
        results,
        figure_family_dir(gallery_dir, DEFAULT_DIRECT_TRIAL_TYPE_FIGURES_DIRNAME),
        output_name="08_direct_trial_type_state_comparison_all.svg",
        title="Direct trial-type comparison - state pair scatter",
    )
    append_entry("direct_trial_type_state_comparison", "all", path, scope="direct_trial_type_comparison")
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(),
        "gallery_dir": str(gallery_dir),
        "n_files": int(len(entries)),
        "entries": entries,
    }
    manifest_path = gallery_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(jsonable(manifest), handle, indent=2, sort_keys=True)
    return {"manifest_path": str(manifest_path), "entries": entries, "files": files}
def find_first_key(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None
def numeric_array(value: Any, dtype=float) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object:
        try:
            arr = np.asarray(value, dtype=dtype)
        except Exception:
            return None
    try:
        return np.asarray(arr, dtype=dtype)
    except Exception:
        return None
def fill_nan_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if mask.all():
        return x
    if mask.sum() == 0:
        return np.zeros_like(x)
    idx = np.arange(x.size)
    filled = np.interp(idx, idx[mask], x[mask])
    return filled
def moving_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))
def robust_sigma(x: np.ndarray) -> float:
    mad = moving_mad(x)
    if not np.isfinite(mad) or mad == 0:
        std = float(np.nanstd(x))
        return std if np.isfinite(std) and std > 0 else 1.0
    return 1.4826 * mad
def interpolate_series(target_t: np.ndarray, source_t: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    target_t = np.asarray(target_t, dtype=float)
    source_t = np.asarray(source_t, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    if target_t.size == 0 or source_t.size == 0 or source_y.size == 0:
        return np.full(target_t.shape, np.nan, dtype=float)
    valid = np.isfinite(source_t) & np.isfinite(source_y)
    if valid.sum() == 0:
        return np.full(target_t.shape, np.nan, dtype=float)
    source_t = source_t[valid]
    source_y = source_y[valid]
    order = np.argsort(source_t)
    source_t = source_t[order]
    source_y = source_y[order]
    if source_t.size == 1:
        return np.full(target_t.shape, float(source_y[0]), dtype=float)
    return np.interp(target_t, source_t, source_y, left=float(source_y[0]), right=float(source_y[-1]))
def estimate_sampling_rate(t: np.ndarray) -> Optional[float]:
    t = np.asarray(t, dtype=float)
    if t.size < 3:
        return None
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return 1.0 / median_dt
def robust_bisquare_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-9, tuning_constant: float = 4.685) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"intercept": float("nan"), "alpha": float("nan"), "n": int(x.size), "scale": float("nan")}
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    scale = robust_sigma(y - X @ beta)
    for _ in range(max_iter):
        residuals = y - X @ beta
        scale = robust_sigma(residuals)
        if not np.isfinite(scale) or scale <= 0:
            break
        u = residuals / (tuning_constant * scale)
        weights = np.zeros_like(u)
        inside = np.abs(u) < 1
        weights[inside] = (1 - u[inside] ** 2) ** 2
        if np.all(weights == 0):
            break
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w
        try:
            beta_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return {
        "intercept": float(beta[0]),
        "alpha": float(beta[1]),
        "n": int(x.size),
        "scale": float(scale),
    }
def _canonical_event_detection_method(method: Any) -> str:
    text = str(method or "").strip().lower()
    if text in EVENT_DETECTION_METHODS:
        return text
    raise ValueError(f"Unknown event detection method: {method}")


def _event_detection_source_series(trace: np.ndarray, method: str) -> Tuple[np.ndarray, str]:
    method = _canonical_event_detection_method(method)
    trace = np.asarray(trace, dtype=float)
    if method == "derivative":
        return np.diff(trace), "first_derivative"
    return trace, "trace"


def _event_detection_runs(series: np.ndarray, threshold: float, min_consecutive_frames: int, *, index_offset: int = 0) -> List[Tuple[int, int]]:
    above = np.isfinite(series) & (series > threshold)
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(above):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_consecutive_frames:
                runs.append((start + index_offset, idx + index_offset))
            start = None
    if start is not None and series.size - start >= min_consecutive_frames:
        runs.append((start + index_offset, series.size + index_offset))
    return runs


def detect_events(
    trace: np.ndarray,
    min_consecutive_frames: int = 3,
    sigma_factor: float = 3.0,
    threshold: Optional[float] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    method = get_active_event_detection_method(method)
    trace = np.asarray(trace, dtype=float)
    source_series, source_series_name = _event_detection_source_series(trace, method)
    trace_median = float(np.nanmedian(trace)) if trace.size else float("nan")
    if source_series.size == 0:
        return {
            "method": method,
            "source_series": source_series_name,
            "trace_median": trace_median,
            "baseline_median": trace_median if method == "amplitude" else float("nan"),
            "noise_std": float("nan"),
            "threshold": float("nan") if threshold is None else float(threshold),
            "event_count": 0,
            "event_runs": [],
            "event_frequency_per_min": float("nan"),
            "active": False,
        }
    if method == "amplitude":
        centered = source_series - np.nanmedian(source_series)
        baseline_median = float(np.nanmedian(source_series))
        index_offset = 0
    else:
        centered = source_series - np.nanmedian(source_series)
        baseline_median = float(np.nanmedian(source_series))
        index_offset = 1
    noise_std = robust_sigma(centered)
    active_threshold = float(threshold) if threshold is not None and np.isfinite(threshold) else float(sigma_factor * noise_std)
    runs = _event_detection_runs(source_series, active_threshold, min_consecutive_frames, index_offset=index_offset)
    return {
        "method": method,
        "source_series": source_series_name,
        "trace_median": trace_median,
        "baseline_median": baseline_median,
        "noise_std": float(noise_std),
        "threshold": float(active_threshold),
        "event_count": int(len(runs)),
        "event_runs": runs,
        "event_frequency_per_min": float("nan"),
        "active": bool(len(runs) >= 3),
    }


def interval_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return (t >= start) & (t < end)


def estimate_trace_duration_seconds(time: np.ndarray) -> float:
    try:
        time = np.asarray(time, dtype=float).ravel()
    except Exception:
        return float("nan")
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return float("nan")
    diffs = np.diff(finite)
    valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    step = float(np.nanmedian(valid_diffs)) if valid_diffs.size else float("nan")
    if not np.isfinite(step) or step <= 0:
        return float(finite.size)
    return float(max(finite[-1] - finite[0] + step, step))


def event_frequency_per_minute(event_count: int, duration_seconds: float) -> float:
    if duration_seconds is None or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        return float("nan")
    return float(event_count) * 60.0 / float(duration_seconds)


def estimate_trace_step_seconds(time: np.ndarray) -> float:
    try:
        time = np.asarray(time, dtype=float).ravel()
    except Exception:
        return float("nan")
    finite = time[np.isfinite(time)]
    if finite.size <= 1:
        return float("nan")
    diffs = np.diff(finite)
    valid_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if valid_diffs.size == 0:
        return float("nan")
    return float(np.nanmedian(valid_diffs))


def _build_event_detection_info(
    trace: np.ndarray,
    time: Optional[np.ndarray],
    *,
    method: Optional[str] = None,
    min_consecutive_frames: int = 3,
    sigma_factor: float = 3.0,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    method = get_active_event_detection_method(method)
    trace = np.asarray(trace, dtype=float)
    event_info = detect_events(
        trace,
        min_consecutive_frames=min_consecutive_frames,
        sigma_factor=sigma_factor,
        threshold=threshold,
        method=method,
    )
    fallback_time = np.arange(trace.size, dtype=float)
    duration_seconds = estimate_trace_duration_seconds(time if time is not None else fallback_time)
    if not np.isfinite(duration_seconds):
        duration_seconds = estimate_trace_duration_seconds(fallback_time)
    event_info["duration_seconds"] = float(duration_seconds)
    event_info["event_frequency_per_min"] = event_frequency_per_minute(int(event_info.get("event_count", 0) or 0), duration_seconds)
    event_info["min_consecutive_frames"] = int(min_consecutive_frames)
    event_info["sigma_factor"] = float(sigma_factor)
    event_info["method"] = method
    return event_info


def build_event_info(
    trace: np.ndarray,
    time: Optional[np.ndarray] = None,
    *,
    method: Optional[str] = None,
    include_all_methods: bool = True,
) -> Dict[str, Any]:
    trace = np.asarray(trace, dtype=float)
    method = get_active_event_detection_method(method)
    event_info = _build_event_detection_info(trace, time, method=method)
    event_info["primary_method"] = method
    event_info["event_detection_methods"] = list(EVENT_DETECTION_METHODS)
    if include_all_methods:
        event_info["methods"] = {
            method_name: _build_event_detection_info(trace, time, method=method_name)
            for method_name in EVENT_DETECTION_METHODS
        }
    return event_info


def build_masked_event_info(
    trace: np.ndarray,
    time: Optional[np.ndarray],
    mask: np.ndarray,
    *,
    method: Optional[str] = None,
    threshold: Optional[float] = None,
    min_consecutive_frames: int = 3,
    include_all_methods: bool = True,
) -> Dict[str, Any]:
    trace = np.asarray(trace, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    method = get_active_event_detection_method(method)
    if trace.size == 0 or mask.size != trace.size or not np.any(mask):
        event_info = detect_events(
            np.asarray([], dtype=float),
            min_consecutive_frames=min_consecutive_frames,
            threshold=threshold,
            method=method,
        )
        event_info["duration_seconds"] = float("nan")
        event_info["event_frequency_per_min"] = float("nan")
        event_info["min_consecutive_frames"] = int(min_consecutive_frames)
        event_info["sigma_factor"] = 3.0
        event_info["primary_method"] = method
        event_info["event_detection_methods"] = list(EVENT_DETECTION_METHODS)
        if include_all_methods:
            event_info["methods"] = {
                method_name: _build_event_detection_info(
                    np.asarray([], dtype=float),
                    None,
                    method=method_name,
                    min_consecutive_frames=min_consecutive_frames,
                    threshold=threshold,
                )
                for method_name in EVENT_DETECTION_METHODS
            }
        return event_info
    masked_trace = trace[mask]
    event_info = _build_event_detection_info(
        masked_trace,
        time,
        method=method,
        min_consecutive_frames=min_consecutive_frames,
        threshold=threshold,
    )
    step_seconds = estimate_trace_step_seconds(time if time is not None else np.arange(trace.size, dtype=float))
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        duration_seconds = float(masked_trace.size)
    else:
        duration_seconds = float(np.isfinite(masked_trace).sum() * step_seconds)
    event_info["duration_seconds"] = float(duration_seconds)
    event_info["event_frequency_per_min"] = event_frequency_per_minute(int(event_info.get("event_count", 0) or 0), duration_seconds)
    event_info["min_consecutive_frames"] = int(min_consecutive_frames)
    event_info["sigma_factor"] = 3.0
    event_info["primary_method"] = method
    event_info["event_detection_methods"] = list(EVENT_DETECTION_METHODS)
    if include_all_methods:
        event_info["methods"] = {
            method_name: build_masked_event_info(
                trace,
                time,
                mask,
                method=method_name,
                threshold=threshold,
                min_consecutive_frames=min_consecutive_frames,
                include_all_methods=False,
            )
            for method_name in EVENT_DETECTION_METHODS
        }
    return event_info


def build_state_masked_event_info(
    trace: np.ndarray,
    time: Optional[np.ndarray],
    mask: np.ndarray,
    full_event_info: Optional[Dict[str, Any]] = None,
    *,
    method: Optional[str] = None,
    min_consecutive_frames: int = 3,
) -> Dict[str, Any]:
    source_info = full_event_info if isinstance(full_event_info, dict) else {}
    selected_method = get_active_event_detection_method(method or source_info.get("method") or source_info.get("primary_method"))
    threshold = as_float(source_info.get("threshold"))
    return build_masked_event_info(
        trace,
        time,
        mask,
        method=selected_method,
        threshold=threshold,
        min_consecutive_frames=min_consecutive_frames,
    )


def event_run_onsets_match(run_a: Tuple[int, int], run_b: Tuple[int, int]) -> bool:
    return int(run_a[0]) == int(run_b[0])


def event_run_overlaps(run_a: Tuple[int, int], run_b: Tuple[int, int]) -> bool:
    return event_run_onsets_match(run_a, run_b)


def _annotate_spine_event_info_for_method(
    spine_event_info: Dict[str, Any],
    dendrite_event_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return shared_annotate_spine_event_info(spine_event_info, dendrite_event_info)


def annotate_spine_event_info(spine_event_info: Dict[str, Any], dendrite_event_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return shared_annotate_spine_event_info(spine_event_info, dendrite_event_info)
def extract_movie_feature_prefixes(columns: Sequence[str]) -> List[str]:
    prefixes = set()
    for column in columns:
        match = re.match(r"^(F\d+)_", column)
        if match:
            prefixes.add(match.group(1))
    return sorted(prefixes, key=lambda x: int(x[1:]))
def movie_feature_blocks(row: Dict[str, str], columns: Sequence[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for prefix in extract_movie_feature_prefixes(columns):
        if row.get(f"{prefix}_type") != "movie":
            continue
        name = row.get(f"{prefix}_name")
        if not name:
            continue
        blocks.append(
            {
                "prefix": prefix,
                "name": name,
                "onset": as_float(row.get(f"{prefix}_onset")),
                "duration": as_float(row.get(f"{prefix}_duration")),
                "speed": as_float(row.get(f"{prefix}_speed")),
                "loop": as_int(row.get(f"{prefix}_loop")),
            }
        )
    return blocks
def classify_movie_name(feature_name: str) -> str:
    name = str(feature_name)
    if name == BLANK_MOVIE_PATH:
        return "blank"
    if name.startswith(GRATING_PREFIX):
        return "grating"
    if name.startswith(ZEBRA_PREFIX):
        return "zebra"
    return "movies"
def normalize_movie_clip_id(feature_name: Any) -> str:
    text = str(feature_name or "").strip()
    if not text:
        return "unknown_video"
    base = PureWindowsPath(text).name or text
    base = re.sub(r"\.[A-Za-z0-9]+$", "", base).strip()
    return base or text
def detect_trial_state_label(
    row: Dict[str, str],
    columns: Sequence[str],
    exp_time: np.ndarray,
    wheel_interp: Optional[np.ndarray],
    locomotion_threshold: float,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    blocks = movie_feature_blocks(row, columns)
    debug: Dict[str, Any] = {"movie_feature_count": len(blocks), "ambiguous": False}
    if len(blocks) == 0:
        return None, None, debug
    if len(blocks) > 1:
        debug["ambiguous"] = True
        return None, None, debug
    category = classify_movie_name(blocks[0]["name"])
    start = as_float(row.get("time"))
    duration = as_float(row.get("duration"))
    if start is None or duration is None:
        return category, None, debug
    end = start + duration
    mask = interval_mask(exp_time, start, end)
    score = float("nan")
    if wheel_interp is not None and mask.any():
        score = float(np.nanmedian(np.abs(wheel_interp[mask])))
    if not np.isfinite(score):
        locomotion_state = "quiet"
    else:
        locomotion_state = "active" if score >= locomotion_threshold else "quiet"
    label = f"{locomotion_state}_{category}"
    debug.update(
        {
            "category": category,
            "trial_start": start,
            "trial_end": end,
            "wheel_score": score,
            "locomotion_threshold": locomotion_threshold,
            "state_label": label,
        }
    )
    return category, label, debug
def extract_calcium_bundle(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    bundle = read_pickle(path)
    if isinstance(bundle, dict):
        t = find_first_key(bundle, ["t", "time", "timeline"])
        dff = find_first_key(bundle, ["dF", "dff", "dF/F", "df", "signal"])
        if dff is None:
            raise KeyError(f"Cannot find dF/F array in {path}")
        dff_arr = np.asarray(dff, dtype=float)
        if t is None:
            t_arr = np.arange(dff_arr.shape[-1], dtype=float)
        else:
            t_arr = np.asarray(t, dtype=float)
        return t_arr, dff_arr, bundle
    if isinstance(bundle, np.ndarray):
        if bundle.ndim == 2:
            return np.arange(bundle.shape[1], dtype=float), np.asarray(bundle, dtype=float), {"dF": bundle}
        raise ValueError(f"Unexpected calcium bundle shape in {path}: {bundle.shape}")
    raise TypeError(f"Unexpected calcium bundle type in {path}: {type(bundle).__name__}")
def extract_series_bundle(path: Path, signal_priority: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    bundle = read_pickle(path)
    if isinstance(bundle, dict):
        t = find_first_key(bundle, ["t", "time", "timeline"])
        series = find_first_key(bundle, signal_priority)
        if series is None:
            numeric_candidates = [v for v in bundle.values() if isinstance(v, (list, tuple, np.ndarray))]
            if not numeric_candidates:
                raise KeyError(f"Cannot find usable series in {path}")
            series = numeric_candidates[0]
        series_arr = np.asarray(series, dtype=float)
        if t is None:
            t_arr = np.arange(series_arr.size, dtype=float)
        else:
            t_arr = np.asarray(t, dtype=float)
        return t_arr, series_arr, bundle
    if isinstance(bundle, np.ndarray):
        bundle = np.asarray(bundle, dtype=float)
        if bundle.ndim == 1:
            return np.arange(bundle.size, dtype=float), bundle, {"series": bundle}
        if bundle.ndim == 2 and bundle.shape[0] == 2:
            return np.asarray(bundle[0], dtype=float), np.asarray(bundle[1], dtype=float), {"t": bundle[0], "series": bundle[1]}
        raise ValueError(f"Unexpected series bundle shape in {path}: {bundle.shape}")
    raise TypeError(f"Unexpected series bundle type in {path}: {type(bundle).__name__}")
def extract_sleep_state_bundle(path: Path) -> Dict[str, Any]:
    bundle = read_pickle(path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Unexpected sleep_state bundle type in {path}: {type(bundle).__name__}")
    if "state_10hz" not in bundle or "state_10hz_t" not in bundle:
        raise KeyError(f"sleep_state bundle missing state_10hz/state_10hz_t in {path}")
    return bundle
def extract_cut_neural_bundle(
    path: Path,
    preferred_keys: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    bundle = read_pickle(path)
    if isinstance(bundle, dict):
        t = find_first_key(bundle, ["t", "time", "timeline"])
        key_candidates = list(preferred_keys or []) + ["dF", "dff", "dF/F", "df", "signal"]
        dff = find_first_key(bundle, key_candidates)
        if dff is None:
            numeric_candidates = [v for v in bundle.values() if isinstance(v, (list, tuple, np.ndarray))]
            if not numeric_candidates:
                raise KeyError(f"Cannot find dF/F array in {path}")
            dff = numeric_candidates[0]
        dff_arr = np.asarray(dff, dtype=float)
        if dff_arr.ndim != 3:
            raise ValueError(f"Expected 3D cut neural array in {path}, got {dff_arr.shape}")
        if t is None:
            t_arr = np.arange(dff_arr.shape[-1], dtype=float)
        else:
            t_arr = np.asarray(t, dtype=float)
        return t_arr, dff_arr, bundle
    if isinstance(bundle, np.ndarray):
        arr = np.asarray(bundle, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D cut neural array in {path}, got {arr.shape}")
        return np.arange(arr.shape[-1], dtype=float), arr, {"dF": arr}
    raise TypeError(f"Unexpected cut neural bundle type in {path}: {type(bundle).__name__}")
def signature_for_file(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
def locate_conversion_file(
    repo_base: Path,
    animal_id: str,
    exp_id: str,
    prefer_same_day_source: bool = False,
) -> Tuple[Optional[Path], Optional[str], bool]:
    primary_exp_root = resolve_repo_root(repo_base, animal_id, exp_id)
    same_day_candidates: List[Path] = []
    animal_root = repo_base / animal_id
    date_prefix = derive_date(exp_id)
    if animal_root.exists():
        for sibling in sorted(animal_root.iterdir()):
            if sibling.is_dir() and sibling.name.startswith(date_prefix + "_") and sibling.name != exp_id:
                same_day_candidates.append(sibling)
    candidates: List[Path] = same_day_candidates + [primary_exp_root] if prefer_same_day_source else [primary_exp_root] + same_day_candidates
    def detect(path: Path) -> Optional[Path]:
        primary = path / "suite2p" / "SpinesGUI"
        fallback = path / "suite2p_combined" / "SpinesGUI"
        for spines_dir in [primary, fallback]:
            if not spines_dir.exists():
                continue
            for filename in [DEND_AXON_CONVERSION_FILENAME, NORMAL_CONVERSION_FILENAME]:
                candidate = spines_dir / filename
                if candidate.exists():
                    return candidate
        return None
    for idx, candidate_root in enumerate(candidates):
        conv = detect(candidate_root)
        if conv is not None:
            return conv, candidate_root.name, idx > 0
    return None, None, False
def load_conversion_library(path: Path) -> Dict[Any, Any]:
    raw = np.load(path, allow_pickle=True)
    if hasattr(raw, "item"):
        raw = raw.item()
    if not isinstance(raw, dict):
        raise TypeError(f"Conversion library at {path} is not a dictionary")
    return raw
def normalize_conversion_entry(general_roi_id: Any, entry: Dict[str, Any], mode: str) -> Dict[str, Any]:
    roi_type = find_first_key(entry, ["roi-type", "roi_type"])
    if roi_type is None:
        raise KeyError(f"ROI entry {general_roi_id} missing roi-type")
    roi_type = list(roi_type)
    code = as_int(roi_type[0])
    conversion_index = find_first_key(entry, ["conversion_index", "conversion index"])
    conversion_index = as_int(conversion_index)
    plane = find_first_key(entry, ["plane"])
    conversion = find_first_key(entry, ["conversion"])
    if conversion is None:
        conversion = [None, None]
    conversion = list(conversion) if isinstance(conversion, (list, tuple, np.ndarray)) else [None, None]
    result: Dict[str, Any] = {
        "general_roi_id": int(general_roi_id) if str(general_roi_id).isdigit() else str(general_roi_id),
        "mode": mode,
        "roi_type_code": code,
        "conversion_index": conversion_index,
        "plane": as_int(plane),
        "plane_roi_id": as_int(conversion[1]) if len(conversion) > 1 else None,
        "conversion_plane": as_int(conversion[0]) if len(conversion) > 0 else None,
        "raw_entry": entry,
    }
    if mode == "normal":
        result["cell_id"] = as_int(roi_type[1]) if len(roi_type) > 1 else None
        result["dendrite_id"] = as_int(roi_type[2]) if len(roi_type) > 2 else None
        result["spine_id"] = as_int(roi_type[3]) if len(roi_type) > 3 else None
        result["is_dendrite"] = code == 1
        result["is_spine"] = code == 2
        result["anchor_id"] = f"cell{result['cell_id']}" if result["cell_id"] is not None else f"roi{result['general_roi_id']}"
    elif mode == "dendrite_axon":
        result["dendrite_id"] = as_int(roi_type[1]) if len(roi_type) > 1 else None
        result["spine_id"] = as_int(roi_type[2]) if len(roi_type) > 2 else None
        result["axon_id"] = as_int(roi_type[3]) if len(roi_type) > 3 else None
        result["bouton_id"] = as_int(roi_type[4]) if len(roi_type) > 4 else None
        result["cell_id"] = None
        result["is_dendrite"] = code == 0
        result["is_spine"] = code == 1
        result["anchor_id"] = f"roi{result['general_roi_id']}"
    else:
        raise ValueError(f"Unknown conversion mode: {mode}")
    return result
def normalize_conversion_library(library: Dict[Any, Any], mode: str) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for general_roi_id, entry in library.items():
        normalized[str(general_roi_id)] = normalize_conversion_entry(general_roi_id, entry, mode)
    return normalized
def determine_conversion_mode(path: Path) -> str:
    if path.name == DEND_AXON_CONVERSION_FILENAME:
        return "dendrite_axon"
    if path.name == NORMAL_CONVERSION_FILENAME:
        return "normal"
    raise ValueError(f"Unknown conversion filename: {path.name}")
def find_source_signal_key(bundle: Dict[str, Any]) -> Optional[str]:
    for key in ["dF", "dff", "dF/F", "df", "signal"]:
        if key in bundle:
            return key
    return None
def build_state_masks_movie(
    exp_time: np.ndarray,
    trial_rows: List[Dict[str, str]],
    columns: Sequence[str],
    wheel_time: Optional[np.ndarray],
    wheel_speed: Optional[np.ndarray],
    sleep_state: Optional[Dict[str, Any]],
    locomotion_threshold: float,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], Optional[np.ndarray]]:
    masks = {label: np.zeros(exp_time.shape, dtype=bool) for label in MOVIE_STATE_LABELS}
    wheel_interp = None
    sleep_codes_on_time: Optional[np.ndarray] = None
    if sleep_state is not None:
        sleep_t = np.asarray(sleep_state["state_10hz_t"], dtype=float)
        sleep_codes = np.asarray(sleep_state["state_10hz"], dtype=float)
        sleep_inside = (exp_time >= sleep_t.min()) & (exp_time <= sleep_t.max())
        sleep_codes_on_time = np.full(exp_time.shape, -1, dtype=int)
        if sleep_inside.any():
            interpolated_sleep = np.interp(exp_time[sleep_inside], sleep_t, sleep_codes)
            sleep_codes_on_time[sleep_inside] = np.rint(interpolated_sleep).astype(int)
    if sleep_codes_on_time is None:
        sleep_codes_on_time = np.full(exp_time.shape, -1, dtype=int)
    if wheel_time is not None and wheel_speed is not None:
        wheel_interp = interpolate_series(exp_time, wheel_time, wheel_speed)
    trial_meta: List[Dict[str, Any]] = []
    for trial_index, row in enumerate(trial_rows):
        category, state_label, debug = detect_trial_state_label(row, columns, exp_time, wheel_interp, locomotion_threshold)
        start = debug.get("trial_start")
        end = debug.get("trial_end")
        if state_label is None or start is None or end is None:
            if debug.get("ambiguous"):
                trial_meta.append(
                    {
                        "trial_index": trial_index,
                        "warning": "multiple movie features detected; trial skipped",
                        "trial_row": row,
                    }
                )
            continue
        trial_mask = interval_mask(exp_time, start, end)
        sleep_state_label = None
        if sleep_codes_on_time.size and np.any(trial_mask):
            trial_codes = sleep_codes_on_time[trial_mask]
            trial_codes = trial_codes[np.isfinite(trial_codes)]
            if trial_codes.size:
                codes, counts = np.unique(trial_codes.astype(int), return_counts=True)
                best = int(codes[int(np.argmax(counts))])
                sleep_state_label = SLEEP_STATE_MAP.get(best)
        if sleep_state_label is None:
            sleep_state_label = "quiet_awake" if str(state_label).startswith("quiet") else "active_awake"
        movie_type = category or "movies"
        combined_label = combined_movie_state_label(sleep_state_label, movie_type)
        if combined_label not in masks:
            masks[combined_label] = np.zeros(exp_time.shape, dtype=bool)
        masks[combined_label] |= trial_mask
        trial_meta.append(
            {
                "trial_index": trial_index,
                "category": movie_type,
                "sleep_state_label": sleep_state_label,
                "state_label": combined_label,
                "movie_state_label": combined_label,
                "sleep_state": sleep_state_label,
                "movie_trial_type": movie_type,
                "sleep_code": int(next((code for code, label in SLEEP_STATE_MAP.items() if label == sleep_state_label), -1)),
                "wheel_score": debug.get("wheel_score"),
                "locomotion_threshold": locomotion_threshold,
                "start": start,
                "end": end,
                "duration": debug.get("trial_end", 0.0) - debug.get("trial_start", 0.0) if start is not None and end is not None else None,
            }
        )
    return masks, trial_meta, wheel_interp
def build_state_masks_sleep(exp_time: np.ndarray, sleep_state: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    sleep_t = np.asarray(sleep_state["state_10hz_t"], dtype=float)
    sleep_codes = np.asarray(sleep_state["state_10hz"], dtype=float)
    inside = (exp_time >= sleep_t.min()) & (exp_time <= sleep_t.max())
    codes = np.full(exp_time.shape, -1, dtype=int)
    if inside.any():
        interpolated = np.interp(exp_time[inside], sleep_t, sleep_codes)
        codes[inside] = np.rint(interpolated).astype(int)
    masks = {label: codes == code for code, label in SLEEP_STATE_MAP.items()}
    meta = {
        "sleep_state_keys": sorted(list(sleep_state.keys())),
        "state_labels": dict(SLEEP_STATE_MAP),
        "state_10hz_t": sleep_t,
        "state_10hz": sleep_codes,
        "state_codes_on_calcium_time": codes,
    }
    return masks, meta
def choose_locomotion_threshold(
    explicit_threshold: Optional[float],
    sleep_state_thresholds: Sequence[float],
    wheel_interp: Optional[np.ndarray],
) -> float:
    if explicit_threshold is not None and np.isfinite(explicit_threshold):
        return float(explicit_threshold)
    valid_thresholds = [float(v) for v in sleep_state_thresholds if np.isfinite(v)]
    if valid_thresholds:
        return float(np.median(valid_thresholds))
    if wheel_interp is not None and wheel_interp.size:
        return float(np.nanmedian(np.abs(wheel_interp)) + DEFAULT_Locomotion_THRESHOLD_FRACTION * robust_sigma(wheel_interp))
    return 0.0
def get_compartment_tag(exp_id: str, basal_expids: Sequence[str], apical_expids: Sequence[str], sleep_expids: Sequence[str]) -> str:
    if exp_id in basal_expids:
        return "basal"
    if exp_id in apical_expids:
        return "apical"
    if exp_id in sleep_expids:
        return "sleep"
    return "movie"
def resolve_experiment_compartment(
    exp_id: str,
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
    sleep_expids: Sequence[str],
    same_day_source_exp_id: Optional[str] = None,
) -> str:
    compartment = get_compartment_tag(exp_id, basal_expids, apical_expids, sleep_expids)
    if exp_id not in sleep_expids:
        return compartment
    if same_day_source_exp_id is not None:
        source_compartment = get_compartment_tag(same_day_source_exp_id, basal_expids, apical_expids, sleep_expids)
        if source_compartment in {"basal", "apical"}:
            return source_compartment
    date_prefix = derive_date(exp_id)
    basal_same_day = any(derive_date(candidate) == date_prefix for candidate in basal_expids)
    apical_same_day = any(derive_date(candidate) == date_prefix for candidate in apical_expids)
    if basal_same_day and not apical_same_day:
        return "basal"
    if apical_same_day and not basal_same_day:
        return "apical"
    return compartment
def experiment_source_paths(repo_base: Path, exp_id: str, channel: int) -> Dict[str, Optional[Path]]:
    animal_id = derive_animal_id(exp_id)
    exp_root = resolve_repo_root(repo_base, animal_id, exp_id)
    recordings = exp_root / "recordings"
    cut = exp_root / "cut"
    sleep_score = exp_root / "sleep_score"
    return {
        "exp_root": exp_root,
        "recordings": recordings,
        "cut": cut,
        "sleep_score": sleep_score,
        "calcium": recordings / f"s2p_ch{channel}.pickle",
        "wheel": recordings / "wheel.pickle",
        "eye_left": recordings / "dlcEyeLeft_resampled.pickle",
        "eye_right": recordings / "dlcEyeRight_resampled.pickle",
        "sleep_state": sleep_score / "sleep_state.pickle",
        "trial_csv": exp_root / f"{exp_id}_all_trials.csv",
        "cut_spikes": cut / f"s2p_ch{channel}_Spikes_cut.pickle",
    }


def experiment_source_signature(exp_paths: Mapping[str, Optional[Path]], conversion_path: Optional[Path], compartment: str, channel: int) -> str:
    return stable_hash(
        {
            "calcium": signature_for_file(exp_paths["calcium"]),
            "wheel": signature_for_file(exp_paths["wheel"]),
            "eye_left": signature_for_file(exp_paths["eye_left"]),
            "eye_right": signature_for_file(exp_paths["eye_right"]),
            "sleep_state": signature_for_file(exp_paths["sleep_state"]),
            "cut_neural": signature_for_file(Path(exp_paths["cut"]) / f"s2p_ch{channel}_dF_cut.pickle" if exp_paths.get("cut") is not None else None),
            "trial_csv": signature_for_file(exp_paths["trial_csv"]),
            "conversion": signature_for_file(conversion_path) if conversion_path is not None else None,
            "compartment": compartment,
            "channel": channel,
        }
    )
def select_pupil_series(bundle: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if bundle is None:
        return None, None, None
    if not isinstance(bundle, dict):
        return None, None, None
    t = find_first_key(bundle, ["t", "time", "timeline"])
    if t is None:
        return None, None, None
    t_arr = np.asarray(t, dtype=float)
    priority = [
        ("pupil_diameter", "pupil_diameter"),
        ("diameter", "diameter"),
        ("pupil_area", "pupil_area"),
        ("area", "area"),
        ("pupil_speed", "pupil_speed"),
        ("speed", "speed"),
    ]
    for key, label in priority:
        if key in bundle:
            return t_arr, np.asarray(bundle[key], dtype=float), label
    for key, value in bundle.items():
        if key == "t":
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 1:
                return t_arr, arr, str(key)
    return None, None, None
def save_npz_cache(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, cache=np.array([cacheable(payload)], dtype=object))


def analysis_cache_meta_hash(meta: Any) -> str:
    return stable_hash(cacheable(meta))


def analysis_table_cache_meta_hash(meta: Any) -> str:
    return analysis_cache_meta_hash(meta)


def analysis_table_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_analysis_tables_cache.npz")


def analysis_results_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_analysis_results_cache.npz")


def analysis_day_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_analysis_day_cache.npz")


FAMILY_RESULT_CACHE_STAGES = (
    "visual_response",
    "state",
    "direct_trial_type_comparison",
    "mixed_model",
    "spine_coactivity",
    "correlation",
    "matrix_similarity",
)


def family_results_cache_dir(cache_path: Path) -> Path:
    return cache_path.parent / "results"


def family_results_cache_path(cache_path: Path, stage: str) -> Path:
    return family_results_cache_dir(cache_path) / f"{cache_path.stem}_{stage}_results_cache.npz"


def family_results_cache_index(cache_path: Path) -> Dict[str, str]:
    return {stage: str(family_results_cache_path(cache_path, stage)) for stage in FAMILY_RESULT_CACHE_STAGES}


def family_results_cache_stage_for_selection(selected_families: Optional[Sequence[str]]) -> str:
    if isinstance(selected_families, str):
        raw_values = [part.strip() for part in selected_families.split(",") if part.strip()]
    else:
        raw_values = [str(value).strip() for value in (selected_families or []) if str(value).strip()]
    selected = set(raw_values)
    if not selected:
        selected = set(FAMILY_RESULT_CACHE_STAGES[1:])
    if "basal_apical" in selected:
        selected.add("state")
    for stage in reversed(FAMILY_RESULT_CACHE_STAGES[1:]):
        if stage in selected:
            return stage
    return FAMILY_RESULT_CACHE_STAGES[-1]


def save_family_results_cache(
    cache_path: Path,
    stage: str,
    results: Dict[str, Any],
    *,
    base_meta: Dict[str, Any],
) -> Path:
    family_meta = dict(base_meta)
    family_meta["family_result_stage"] = stage
    payload = {
        "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
        "meta": cacheable(family_meta),
        "meta_hash": analysis_cache_meta_hash(family_meta),
        "analysis_results": cacheable(analysis_results_cache_payload(results)),
    }
    path = family_results_cache_path(cache_path, stage)
    save_npz_cache(path, payload)
    return path


def source_cache_signature(source_cache: Dict[str, Any]) -> str:
    return stable_hash(
        {
            str(exp_id): str(exp_meta.get("source_signature", ""))
            for exp_id, exp_meta in sorted(dict(source_cache.get("experiments", {})).items())
        }
    )


def analysis_day_cache_meta(
    source_cache: Dict[str, Any],
    analysis_tables: Optional[Dict[str, Any]] = None,
    *,
    analysis_unit: str = "day",
) -> Dict[str, Any]:
    source_config = dict(source_cache.get("config", {}))
    return {
        "analysis_cache_schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "analysis_unit": str(analysis_unit),
        "source_config_hash": str(source_cache.get("config_hash", "")),
        "source_signature": source_cache_signature(source_cache),
        "analysis_config_hash": stable_hash({**source_config, "analysis_unit": str(analysis_unit)}),
    }


def save_analysis_day_cache(path: Path, analysis_cache: Dict[str, Any], *, meta: Dict[str, Any]) -> Path:
    payload = {
        "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "meta": cacheable(meta),
        "meta_hash": analysis_cache_meta_hash(meta),
        "analysis_cache": cacheable(analysis_cache),
    }
    save_npz_cache(path, payload)
    return path


def load_analysis_day_cache(
    path: Path,
    *,
    expected_meta: Optional[Dict[str, Any]] = None,
    ignore_meta_keys: Optional[Sequence[str]] = None,
    rebuild: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if rebuild:
        step_message("rebuilding day-pooled analysis cache")
        return None, "rebuild_requested"
    if not path.exists():
        step_message("rebuilding day-pooled analysis cache")
        return None, "missing"
    try:
        cache = load_npz_cache(path)
    except Exception:
        step_message("rebuilding day-pooled analysis cache")
        return None, "unreadable"
    if not isinstance(cache, dict):
        step_message("rebuilding day-pooled analysis cache")
        return None, "invalid_payload"
    if cache.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
        step_message("rebuilding day-pooled analysis cache")
        return None, "schema_mismatch"
    if expected_meta is not None:
        ignore_keys = {str(key) for key in (ignore_meta_keys or []) if str(key)}
        filtered_expected_meta = {key: value for key, value in expected_meta.items() if key not in ignore_keys}
        saved_meta = cache.get("meta", {})
        if not isinstance(saved_meta, dict):
            saved_meta = {}
        filtered_saved_meta = {key: value for key, value in saved_meta.items() if key not in ignore_keys}
        if analysis_cache_meta_hash(filtered_saved_meta) != analysis_cache_meta_hash(filtered_expected_meta):
            step_message("day-pooled analysis cache meta mismatch")
            differing_keys = [
                key
                for key in sorted(set(filtered_saved_meta) | set(filtered_expected_meta))
                if filtered_saved_meta.get(key) != filtered_expected_meta.get(key)
            ]
            if differing_keys:
                step_message(f"meta mismatch keys: {', '.join(differing_keys)}")
            return None, "meta_mismatch"
    analysis_cache = cache.get("analysis_cache")
    if not isinstance(analysis_cache, dict):
        step_message("rebuilding day-pooled analysis cache")
        return None, "invalid_analysis_cache"
    return cache, "ok"


def load_analysis_tables_cache(path: Path, *, rebuild: bool = False) -> Optional[Dict[str, Any]]:
    if rebuild or not path.exists():
        step_message("rebuilding analysis-tables cache")
        return None
    try:
        cache = load_npz_cache(path)
    except Exception:
        step_message("rebuilding analysis-tables cache")
        return None
    if not isinstance(cache, dict):
        step_message("rebuilding analysis-tables cache")
        return None
    if cache.get("schema_version") != ANALYSIS_TABLE_CACHE_SCHEMA_VERSION:
        step_message("rebuilding analysis-tables cache")
        return None
    tables = cache.get("analysis_tables")
    if not isinstance(tables, dict):
        step_message("rebuilding analysis-tables cache")
        return None
    return cache


def save_analysis_tables_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_npz_cache(path, payload)


def load_analysis_results_cache(
    path: Path,
    *,
    expected_meta: Optional[Dict[str, Any]] = None,
    ignore_meta_keys: Optional[Sequence[str]] = None,
    rebuild: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if rebuild:
        step_message("rebuilding analysis-results cache")
        return None, "rebuild_requested"
    if not path.exists():
        step_message("rebuilding analysis-results cache")
        return None, "missing"
    try:
        cache = load_npz_cache(path)
    except Exception:
        step_message("rebuilding analysis-results cache")
        return None, "unreadable"
    if not isinstance(cache, dict):
        step_message("rebuilding analysis-results cache")
        return None, "invalid_payload"
    if cache.get("schema_version") != ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION:
        step_message("rebuilding analysis-results cache")
        return None, "schema_mismatch"
    if expected_meta is not None:
        ignore_keys = {str(key) for key in (ignore_meta_keys or []) if str(key)}
        filtered_expected_meta = {key: value for key, value in expected_meta.items() if key not in ignore_keys}
        expected_hash = analysis_cache_meta_hash(filtered_expected_meta)

        saved_meta = cache.get("meta", {})
        if not isinstance(saved_meta, dict):
            saved_meta = {}
        filtered_saved_meta = {key: value for key, value in saved_meta.items() if key not in ignore_keys}
        saved_hash = analysis_cache_meta_hash(filtered_saved_meta)

        if saved_hash != expected_hash:
            step_message("analysis-results cache meta mismatch")
            step_message(f"saved meta_hash:   {saved_hash}")
            step_message(f"expected meta_hash:{expected_hash}")
            differing_keys = [
                key
                for key in sorted(set(filtered_saved_meta) | set(filtered_expected_meta))
                if filtered_saved_meta.get(key) != filtered_expected_meta.get(key)
            ]
            if differing_keys:
                step_message(f"meta mismatch keys: {', '.join(differing_keys)}")
            for key in differing_keys:
                saved_value = filtered_saved_meta.get(key)
                expected_value = filtered_expected_meta.get(key)
                step_message(
                    f"meta mismatch for {key!r}: "
                    f"saved={saved_value!r} expected={expected_value!r}"
                )
            return None, "meta_mismatch"
    results = cache.get("analysis_results")
    if not isinstance(results, dict):
        step_message("rebuilding analysis-results cache")
        return None, "invalid_results"
    return cache, "ok"


def save_analysis_results_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_npz_cache(path, payload)


def analysis_results_cache_payload(results: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(results)
    for key in [
        "analysis_cache_summary",
        "source_cache_summary",
        "cache_summary",
        "analysis_report_path",
        "output_artifacts",
        "figure_files",
        "review_figure_files",
        "checkpoint_gallery",
        "shared_shuffle_cache",
        "state_coverage",
    ]:
        payload.pop(key, None)
    return payload


def load_cached_analysis_table(
    cache: Dict[str, Any],
    table_name: str,
    *,
    expected_meta: Optional[Dict[str, Any]] = None,
    rebuild: bool = False,
) -> Optional[Dict[str, Any]]:
    if rebuild:
        return None
    tables = cache.get("analysis_tables", {})
    if not isinstance(tables, dict):
        return None
    entry = tables.get(table_name)
    if not isinstance(entry, dict):
        return None
    if entry.get("schema_version") != ANALYSIS_TABLE_CACHE_SCHEMA_VERSION:
        return None
    if expected_meta is not None:
        expected_hash = analysis_table_cache_meta_hash(expected_meta)
        if str(entry.get("meta_hash") or "") != expected_hash:
            return None
    rows = entry.get("table_rows")
    checks = entry.get("table_checks")
    if not isinstance(rows, list) or not isinstance(checks, dict):
        return None
    return entry


def store_cached_analysis_table(
    cache: Dict[str, Any],
    table_name: str,
    table_rows: Sequence[Dict[str, Any]],
    table_checks: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tables = cache.setdefault("analysis_tables", {})
    entry: Dict[str, Any] = {
        "schema_version": ANALYSIS_TABLE_CACHE_SCHEMA_VERSION,
        "meta": cacheable(meta or {}),
        "meta_hash": analysis_table_cache_meta_hash(meta or {}),
        "table_rows": cacheable(list(table_rows)),
        "table_checks": cacheable(table_checks),
    }
    if summary is not None:
        entry["summary"] = cacheable(summary)
    tables[table_name] = entry
    return entry
def ensure_numpy_pickle_compatibility() -> None:
    """Expose NumPy's legacy pickle module paths when loading cached object arrays."""
    try:
        import importlib
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return
    # NumPy 2 pickles may reference `numpy._core`; older environments only ship `numpy.core`.
    sys.modules.setdefault("numpy._core", numpy_core)
    for submodule in ("multiarray", "numeric", "umath", "_multiarray_umath", "overrides", "fromnumeric"):
        try:
            module = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)
ensure_numpy_pickle_compatibility()
def load_npz_cache(path: Path) -> Dict[str, Any]:
    ensure_numpy_pickle_compatibility()
    try:
        loaded = np.load(path, allow_pickle=True)
        if "cache" not in loaded:
            raise KeyError(f"{path} does not contain a 'cache' entry")
        cache_obj = loaded["cache"]
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        ensure_numpy_pickle_compatibility()
        loaded = np.load(path, allow_pickle=True)
        if "cache" not in loaded:
            raise KeyError(f"{path} does not contain a 'cache' entry")
        cache_obj = loaded["cache"]
    if isinstance(cache_obj, np.ndarray) and cache_obj.dtype == object:
        item = cache_obj.item()
        if isinstance(item, dict):
            return item
    if isinstance(cache_obj, np.ndarray) and cache_obj.size == 1:
        item = cache_obj.reshape(()).item()
        if isinstance(item, dict):
            return item
    raise TypeError(f"Could not recover cache dictionary from {path}")
def shared_shuffle_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_shuffle_cache.npz")
def array_signature(array: Any) -> Optional[str]:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.size == 0:
        return stable_hash({"shape": list(arr.shape), "dtype": str(arr.dtype), "empty": True})
    try:
        contiguous = np.ascontiguousarray(arr)
        digest = hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()
    except Exception:
        digest = stable_hash({"shape": list(arr.shape), "dtype": str(arr.dtype), "values": arr.tolist()})
    return stable_hash({"shape": list(arr.shape), "dtype": str(arr.dtype), "digest": digest})
def shared_shuffle_key(metadata: Dict[str, Any]) -> str:
    return stable_hash(metadata)
def load_shared_shuffle_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        cache = load_npz_cache(path)
    except Exception:
        return None
    if not isinstance(cache, dict):
        return None
    if cache.get("schema_version") != SHARED_SHUFFLE_CACHE_SCHEMA_VERSION:
        return None
    return cache
def save_shared_shuffle_cache(path: Path, payload: Dict[str, Any]) -> None:
    save_npz_cache(path, payload)
def build_shared_shuffle_cache_key(
    *,
    family: str,
    signal: str,
    analysis_unit: str,
    animal_id: str,
    day_id: str,
    source_id: str,
    vector_length: int,
    state_label: Optional[str] = None,
    mask_signature: Optional[str] = None,
) -> str:
    return shared_shuffle_key(
        {
            "family": family,
            "signal": signal,
            "analysis_unit": analysis_unit,
            "animal_id": animal_id,
            "day_id": day_id,
            "source_id": source_id,
            "vector_length": int(vector_length),
            "state_label": state_label,
            "mask_signature": mask_signature,
        }
    )
def build_shared_shuffle_entry(key: str, vector_length: int, shuffle_n: int) -> Dict[str, Any]:
    if vector_length <= 1 or shuffle_n <= 0:
        shifts = np.zeros(0, dtype=np.int32)
    else:
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        shifts = rng.integers(1, vector_length, size=int(shuffle_n), dtype=np.int32)
    return {
        "key": key,
        "vector_length": int(vector_length),
        "shuffle_n": int(shuffle_n),
        "shifts": np.asarray(shifts, dtype=np.int32),
    }
def ensure_shared_shuffle_entry(
    shared_shuffle_cache: Optional[Dict[str, Any]],
    key: str,
    vector_length: int,
    shuffle_n: int,
) -> Optional[Dict[str, Any]]:
    if shared_shuffle_cache is None:
        return None
    entries = shared_shuffle_cache.setdefault("entries", {})
    entry = entries.get(key)
    if (
        entry is None
        or int(entry.get("vector_length", -1)) != int(vector_length)
        or int(entry.get("shuffle_n", -1)) != int(shuffle_n)
    ):
        entry = build_shared_shuffle_entry(key, vector_length, shuffle_n)
        entries[key] = entry
    return entry
def build_shared_shuffle_cache(
    cache: Dict[str, Any],
    shuffle_n: int,
    *,
    state_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    entries: Dict[str, Dict[str, Any]] = {}
    state_labels = [str(state) for state in (state_labels or []) if str(state)]
    analysis_unit = str(cache.get("analysis_unit", "source"))
    for animal_id in sorted(animals):
        animal_entry = animals[animal_id]
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            for day_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                exp_meta = experiments.get(day_id)
                if exp_meta is None:
                    continue
                day_id = str(day_id)
                d_trace = np.asarray(d_obs.get("trace"), dtype=float)
                if d_trace.size > 1 and np.any(np.isfinite(d_trace)):
                    key = build_shared_shuffle_cache_key(
                        family="correlation",
                        signal="dendrite_trace",
                        analysis_unit=analysis_unit,
                        animal_id=str(animal_id),
                        day_id=day_id,
                        source_id=str(global_dendrite_id),
                        vector_length=int(d_trace.size),
                    )
                    entries[key] = build_shared_shuffle_entry(key, int(d_trace.size), shuffle_n)
                for spine_id in d_obs.get("spine_ids", []):
                    s_obs = dendrite_record.get("spines", {}).get(spine_id, {}).get("observations", {}).get(day_id)
                    if s_obs is None:
                        continue
                    spine_trace = np.asarray(s_obs.get("trace"), dtype=float)
                    if spine_trace.size > 1 and np.any(np.isfinite(spine_trace)):
                        key = build_shared_shuffle_cache_key(
                            family="correlation",
                            signal="spine_trace",
                            analysis_unit=analysis_unit,
                            animal_id=str(animal_id),
                            day_id=day_id,
                            source_id=str(spine_id),
                            vector_length=int(spine_trace.size),
                        )
                        entries[key] = build_shared_shuffle_entry(key, int(spine_trace.size), shuffle_n)
                    spine_specific = np.asarray(s_obs.get("spine_specific"), dtype=float)
                    if spine_specific.size > 1 and np.any(np.isfinite(spine_specific)):
                        key = build_shared_shuffle_cache_key(
                            family="correlation",
                            signal="spine_specific",
                            analysis_unit=analysis_unit,
                            animal_id=str(animal_id),
                            day_id=day_id,
                            source_id=str(spine_id),
                            vector_length=int(spine_specific.size),
                        )
                        entries[key] = build_shared_shuffle_entry(key, int(spine_specific.size), shuffle_n)
                    if not state_labels or spine_specific.size <= 1:
                        continue
                    for state_label in state_labels:
                        mask = exp_meta.get("state_masks", {}).get(state_label)
                        if mask is None:
                            continue
                        mask = np.asarray(mask, dtype=bool)
                        if mask.shape != spine_specific.shape or not np.any(mask):
                            continue
                        state_trace = np.asarray(spine_specific[mask], dtype=float)
                        if state_trace.size <= 1 or not np.any(np.isfinite(state_trace)):
                            continue
                        key = build_shared_shuffle_cache_key(
                            family="coactivity",
                            signal="spine_specific_state",
                            analysis_unit=analysis_unit,
                            animal_id=str(animal_id),
                            day_id=day_id,
                            source_id=str(spine_id),
                            vector_length=int(state_trace.size),
                            state_label=str(state_label),
                            mask_signature=array_signature(mask),
                        )
                        entries[key] = build_shared_shuffle_entry(key, int(state_trace.size), shuffle_n)
    signature = stable_hash(
        {
            "analysis_unit": analysis_unit,
            "shuffle_n": int(shuffle_n),
            "state_labels": list(state_labels),
            "experiment_signatures": {
                str(exp_id): str(exp_meta.get("source_signature"))
                for exp_id, exp_meta in sorted(experiments.items())
            },
        }
    )
    return {
        "schema_version": SHARED_SHUFFLE_CACHE_SCHEMA_VERSION,
        "analysis_unit": analysis_unit,
        "shuffle_n": int(shuffle_n),
        "state_labels": list(state_labels),
        "signature": signature,
        "entries": entries,
    }
def load_or_build_shared_shuffle_cache(
    cache: Dict[str, Any],
    shuffle_n: int,
    *,
    state_labels: Optional[Sequence[str]] = None,
    cache_path: Optional[Path] = None,
    rebuild: bool = False,
) -> Tuple[Dict[str, Any], Optional[Path], bool]:
    shuffle_path = None if cache_path is None else shared_shuffle_cache_path(cache_path)
    existing = None
    if shuffle_path is not None and shuffle_path.exists() and not rebuild:
        existing = load_shared_shuffle_cache(shuffle_path)
    target = build_shared_shuffle_cache(cache, shuffle_n, state_labels=state_labels)
    if existing and existing.get("signature") == target["signature"] and int(existing.get("shuffle_n", -1)) == int(shuffle_n):
        return existing, shuffle_path, False
    step_message("rebuilding shared circular-shift cache")
    if shuffle_path is not None:
        save_shared_shuffle_cache(shuffle_path, target)
    return target, shuffle_path, True
def strip_gabor_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key in ("gabor_detail", "gabor_detail_source", "gabor_response_source"):
            value.pop(key, None)
        for child in value.values():
            strip_gabor_fields(child)
    elif isinstance(value, list):
        for child in value:
            strip_gabor_fields(child)
def global_dendrite_id(animal_id: str, date: str, *args: Any) -> str:
    """Build a unique dendrite key while staying compatible with the older call signature."""
    if len(args) == 2:
        anchor_id, dendrite_id = args
    elif len(args) == 3:
        _, anchor_id, dendrite_id = args
    else:
        raise TypeError("global_dendrite_id expects (animal_id, date, anchor_id, dendrite_id) or the legacy compartment-inclusive form")
    return f"{animal_id}|{date}|{anchor_id}|d{int(dendrite_id)}"
def global_spine_id(animal_id: str, date: str, *args: Any) -> str:
    """Build a unique spine key while staying compatible with the older call signature."""
    if len(args) == 3:
        anchor_id, dendrite_id, spine_id = args
    elif len(args) == 4:
        _, anchor_id, dendrite_id, spine_id = args
    else:
        raise TypeError("global_spine_id expects (animal_id, date, anchor_id, dendrite_id, spine_id) or the legacy compartment-inclusive form")
    return f"{animal_id}|{date}|{anchor_id}|d{int(dendrite_id)}|s{int(spine_id)}"
def process_experiment(
    repo_base: Path,
    exp_id: str,
    channel: int,
    movie_expids: Sequence[str],
    sleep_expids: Sequence[str],
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
    explicit_locomotion_threshold: Optional[float],
) -> Dict[str, Any]:
    # Load one experiment and normalize everything into the cache schema used downstream.
    animal_id = derive_animal_id(exp_id)
    exp_paths = experiment_source_paths(repo_base, exp_id, channel)
    exp_root = exp_paths["exp_root"]
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {exp_root}")
    if not exp_paths["calcium"].exists():
        raise FileNotFoundError(f"Missing calcium file: {exp_paths['calcium']}")
    # The continuous dF/F matrix is the primary signal for regression and comparisons.
    exp_time, calcium_matrix, calcium_bundle = extract_calcium_bundle(exp_paths["calcium"])
    calcium_matrix = np.asarray(calcium_matrix, dtype=float)
    if calcium_matrix.ndim != 2:
        raise ValueError(f"Expected 2D dF/F matrix in {exp_paths['calcium']}, got {calcium_matrix.shape}")
    source_signal_key = find_source_signal_key(calcium_bundle)
    # Behavioral and pupil traces are optional, but when present they are aligned to the experiment timebase.
    wheel_time = wheel_speed = wheel_bundle = None
    if exp_paths["wheel"].exists():
        wheel_time, wheel_speed, wheel_bundle = extract_series_bundle(exp_paths["wheel"], ["speed", "wheel", "motion", "velocity"])
    else:
        wheel_bundle = None
    eye_left = eye_right = None
    eye_left_time = eye_left_series = None
    eye_right_time = eye_right_series = None
    if exp_paths["eye_left"].exists():
        eye_left_time, eye_left_series, eye_left = extract_series_bundle(
            exp_paths["eye_left"],
            ["pupil_diameter", "diameter", "pupil_area", "area", "pupil_speed", "speed"],
        )
    if exp_paths["eye_right"].exists():
        eye_right_time, eye_right_series, eye_right = extract_series_bundle(
            exp_paths["eye_right"],
            ["pupil_diameter", "diameter", "pupil_area", "area", "pupil_speed", "speed"],
        )
    pupil_time = pupil_series = pupil_source = None
    if eye_left_time is not None and eye_left_series is not None and eye_right_time is not None and eye_right_series is not None:
        left_interp = interpolate_series(exp_time, eye_left_time, eye_left_series)
        right_interp = interpolate_series(exp_time, eye_right_time, eye_right_series)
        pupil_time = exp_time
        pupil_series = np.nanmean(np.vstack([left_interp, right_interp]), axis=0)
        pupil_source = "left_right_average"
    elif eye_left_time is not None and eye_left_series is not None:
        pupil_time = exp_time
        pupil_series = interpolate_series(exp_time, eye_left_time, eye_left_series)
        pupil_source = "left_eye"
    elif eye_right_time is not None and eye_right_series is not None:
        pupil_time = exp_time
        pupil_series = interpolate_series(exp_time, eye_right_time, eye_right_series)
        pupil_source = "right_eye"
    # Sleep-state files are used for sleep analyses and, when available, can also refine movie-state labeling.
    sleep_state = None
    sleep_alert = None
    if exp_id in sleep_expids or exp_id in movie_expids:
        if exp_paths["sleep_state"].exists():
            sleep_state = extract_sleep_state_bundle(exp_paths["sleep_state"])
        else:
            if exp_id in sleep_expids:
                sleep_alert = f"[ALERT] sleep_state.pickle missing for {exp_id}; skipping sleep-state analyses."
            else:
                sleep_alert = f"[ALERT] sleep_state.pickle missing for {exp_id}; movie-state trials will keep their trial-type labels only."
            eprint(sleep_alert)
    trial_rows: List[Dict[str, str]] = []
    if exp_paths["trial_csv"].exists():
        trial_rows = read_csv_rows(exp_paths["trial_csv"])
    cut_neural_path = exp_paths["cut"] / f"s2p_ch{channel}_dF_cut.pickle"
    cut_neural_time = None
    cut_neural = None
    cut_neural_bundle = None
    if cut_neural_path.exists():
        cut_neural_time, cut_neural, cut_neural_bundle = extract_cut_neural_bundle(cut_neural_path)
    # SpinesGUI conversion libraries define the dendrite/spine hierarchy and can fall back to a same-day match.
    conversion_path, conversion_source_exp, used_fallback = locate_conversion_file(
        repo_base,
        animal_id,
        exp_id,
        prefer_same_day_source=exp_id in sleep_expids,
    )
    conversion_mode = "soma_only"
    conversion_library: Dict[str, Dict[str, Any]] = {}
    conversion_alert = None
    if conversion_path is not None:
        conversion_mode = determine_conversion_mode(conversion_path)
        conversion_library = normalize_conversion_library(load_conversion_library(conversion_path), conversion_mode)
    else:
        conversion_source_exp = None
        used_fallback = False
        conversion_alert = f"[ALERT] No SpinesGUI conversion library found for {exp_id}; treating it as soma-only."
        eprint(conversion_alert)
    # Group the experiment by compartment so later comparisons can split basal, apical, movie, and sleep data.
    # Sleep experiments inherit basal/apical from the same-day source when possible so anatomy stays separated.
    compartment = resolve_experiment_compartment(
        exp_id,
        basal_expids,
        apical_expids,
        sleep_expids,
        same_day_source_exp_id=conversion_source_exp,
    )
    # Prefer an explicit sleep-state threshold when one is stored with the experiment.
    sleep_thresholds: List[float] = []
    if sleep_state is not None and "locomotion_threshold" in sleep_state:
        threshold = as_float(sleep_state["locomotion_threshold"])
        if threshold is not None:
            sleep_thresholds.append(threshold)
    locomotion_threshold = choose_locomotion_threshold(explicit_locomotion_threshold, sleep_thresholds, None)
    # Build one mask per requested state so the same experiment can be reused by multiple analyses.
    state_masks: Dict[str, np.ndarray] = {label: np.zeros(exp_time.shape, dtype=bool) for label in ALL_REQUESTED_STATES}
    trial_meta: List[Dict[str, Any]] = []
    wheel_interp = None
    if wheel_time is not None and wheel_speed is not None:
        wheel_interp = interpolate_series(exp_time, wheel_time, wheel_speed)
    if exp_id in movie_expids and trial_rows:
        if explicit_locomotion_threshold is None and not sleep_thresholds and wheel_interp is not None:
            locomotion_threshold = choose_locomotion_threshold(None, [], wheel_interp)
        state_masks_movie, trial_meta, wheel_interp = build_state_masks_movie(
            exp_time,
            trial_rows,
            trial_rows[0].keys() if trial_rows else [],
            wheel_time,
            wheel_speed,
            sleep_state,
            locomotion_threshold,
        )
        for key, mask in state_masks_movie.items():
            state_masks[key] = mask
    if sleep_state is not None:
        state_masks_sleep, sleep_meta = build_state_masks_sleep(exp_time, sleep_state)
        state_masks.update(state_masks_sleep)
    trial_state_indices: Dict[str, List[int]] = defaultdict(list)
    for meta in trial_meta:
        trial_index = meta.get("trial_index")
        state_label = meta.get("state_label")
        if trial_index is None or state_label is None:
            continue
        trial_state_indices[str(state_label)].append(int(trial_index))
    trial_state_counts = {state: len(indices) for state, indices in trial_state_indices.items()}
    # Keep a normalized experiment record for cache reuse and downstream summaries.
    exp_meta = {
        "animal_id": animal_id,
        "exp_id": exp_id,
        "date": derive_date(exp_id),
        "compartment": compartment,
        "repo_root": str(exp_root),
        "source_paths": {k: str(v) if v is not None else None for k, v in exp_paths.items()},
        "source_signal_key": source_signal_key,
        "channel": channel,
        "analysis_mode": "soma_only" if conversion_path is None else ("dendrite_spine" if conversion_mode == "normal" else conversion_mode),
        "conversion": {
            "path": str(conversion_path) if conversion_path is not None else None,
            "source_exp_id": conversion_source_exp,
            "used_fallback": used_fallback,
            "mode": conversion_mode,
            "present": bool(conversion_path is not None),
        },
        "time": np.asarray(exp_time, dtype=np.float32),
        "wheel": {
            "time": np.asarray(wheel_time, dtype=np.float32) if wheel_time is not None else None,
            "speed": np.asarray(wheel_speed, dtype=np.float32) if wheel_speed is not None else None,
            "interpolated": np.asarray(wheel_interp, dtype=np.float32) if wheel_interp is not None else None,
        },
        "pupil": {
            "time": np.asarray(pupil_time, dtype=np.float32) if pupil_time is not None else None,
            "series": np.asarray(pupil_series, dtype=np.float32) if pupil_series is not None else None,
            "source": pupil_source,
        },
        "sleep_state": {
            "available": bool(sleep_state is not None),
            "alert": sleep_alert,
            "bundle": cacheable(sleep_state) if sleep_state is not None else None,
        },
        "cut": {
            "available": bool(cut_neural is not None),
            "path": str(cut_neural_path) if cut_neural_path.exists() else None,
            "shape": list(cut_neural.shape) if cut_neural is not None else None,
            "time": np.asarray(cut_neural_time, dtype=np.float32) if cut_neural_time is not None else None,
            "bundle_keys": sorted(list(cut_neural_bundle.keys())) if isinstance(cut_neural_bundle, dict) else [],
            "trial_state_counts": trial_state_counts,
            "trial_state_labels": [meta.get("state_label") for meta in trial_meta if meta.get("state_label") is not None],
        },
        "state_masks": {state: np.asarray(mask, dtype=bool) for state, mask in state_masks.items()},
        "trial_rows": trial_rows,
        "trial_meta": trial_meta,
        "locomotion_threshold": float(locomotion_threshold),
        "alerts": [alert for alert in [sleep_alert, conversion_alert] if alert],
        "conversion_library_summary": {
            "mode": conversion_mode,
            "roi_count": len(conversion_library),
            "used_fallback": bool(used_fallback),
            "source_exp_id": conversion_source_exp,
            "present": bool(conversion_path is not None),
        },
        "source_signature": None,
    }
    if wheel_interp is None and exp_id in movie_expids and trial_rows:
        # Fall back to the trial average of raw wheel if we can only access the bundle and not the interpolated trace.
        pass
    calcium_signature = signature_for_file(exp_paths["calcium"])
    wheel_signature = signature_for_file(exp_paths["wheel"])
    eye_left_signature = signature_for_file(exp_paths["eye_left"])
    eye_right_signature = signature_for_file(exp_paths["eye_right"])
    sleep_signature = signature_for_file(exp_paths["sleep_state"])
    trial_signature = signature_for_file(exp_paths["trial_csv"])
    conv_signature = signature_for_file(conversion_path) if conversion_path is not None else None
    exp_meta["source_signature"] = experiment_source_signature(exp_paths, conversion_path, compartment, channel)
    animal_cache = {
        "animal_id": animal_id,
        "exp_meta": exp_meta,
        "dendrites": {},
    }
    for roi_key, entry in conversion_library.items():
        if entry["conversion_index"] is None:
            continue
        if not entry["is_dendrite"]:
            continue
        conversion_index = entry["conversion_index"]
        if conversion_index < 0 or conversion_index >= calcium_matrix.shape[0]:
            continue
        dend_trace = np.asarray(calcium_matrix[conversion_index], dtype=float)
        dend_trace_hp = np.asarray(dend_trace, dtype=float)
        g_d_id = global_dendrite_id(
            animal_id,
            exp_meta["date"],
            compartment,
            entry["anchor_id"],
            entry["dendrite_id"] if entry["dendrite_id"] is not None else conversion_index,
        )
        dend_record = animal_cache["dendrites"].setdefault(
            g_d_id,
            {
                "global_dendrite_id": g_d_id,
                "animal_id": animal_id,
                "compartment": compartment,
                "date": exp_meta["date"],
                "observations": {},
                "local_ids": {},
                "spines": {},
            },
        )
        obs = dend_record["observations"].setdefault(
            exp_id,
            {
                "exp_id": exp_id,
                "compartment": compartment,
                "time": np.asarray(exp_time, dtype=np.float32),
                "trace": np.asarray(dend_trace, dtype=np.float32),
                "trace_hp": np.asarray(dend_trace_hp, dtype=np.float32),
                "spine_ids": [],
            },
        )
        if cut_neural is not None and trial_state_indices:
            dend_cut_state_means: Dict[str, float] = {}
            if conversion_index < cut_neural.shape[0]:
                dend_cut_matrix = np.asarray(cut_neural[conversion_index], dtype=float)
                for state_label, trial_indices in trial_state_indices.items():
                    valid_trials = [idx for idx in trial_indices if 0 <= idx < dend_cut_matrix.shape[0]]
                    if not valid_trials:
                        continue
                    trial_values = np.nanmean(np.asarray(dend_cut_matrix[valid_trials], dtype=float), axis=1)
                    if trial_values.size:
                        dend_cut_state_means[state_label] = float(np.nanmean(trial_values))
            if dend_cut_state_means:
                obs["cut_state_means"] = dend_cut_state_means
        obs["local_ids"] = {
            "general_roi_id": entry["general_roi_id"],
            "conversion_index": entry["conversion_index"],
            "plane": entry["plane"],
            "plane_roi_id": entry["plane_roi_id"],
            "mode": entry["mode"],
            "cell_id": entry.get("cell_id"),
            "dendrite_id": entry.get("dendrite_id"),
        }
        obs["event_info"] = build_event_info(dend_trace, exp_time)
        dend_record["local_ids"][exp_id] = obs["local_ids"]
        spine_entries = [e for e in conversion_library.values() if e["is_spine"] and e["dendrite_id"] == entry["dendrite_id"]]
        for spine_entry in spine_entries:
            spine_conversion_index = spine_entry["conversion_index"]
            if spine_conversion_index is None or spine_conversion_index < 0 or spine_conversion_index >= calcium_matrix.shape[0]:
                continue
            spine_trace = np.asarray(calcium_matrix[spine_conversion_index], dtype=float)
            spine_trace_hp = np.asarray(spine_trace, dtype=float)
            alpha_fit = robust_bisquare_fit(dend_trace_hp, spine_trace_hp)
            alpha = alpha_fit["alpha"]
            spine_specific = spine_trace_hp - alpha * dend_trace_hp
            event_info = annotate_spine_event_info(build_event_info(spine_specific, exp_time), obs.get("event_info"))
            g_s_id = global_spine_id(
                animal_id,
                exp_meta["date"],
                compartment,
                spine_entry["anchor_id"],
                spine_entry["dendrite_id"] if spine_entry["dendrite_id"] is not None else entry["dendrite_id"],
                spine_entry["spine_id"] if spine_entry["spine_id"] is not None else spine_conversion_index,
            )
            spine_record = dend_record["spines"].setdefault(
                g_s_id,
                {
                    "global_spine_id": g_s_id,
                    "animal_id": animal_id,
                    "compartment": compartment,
                    "date": exp_meta["date"],
                    "observations": {},
                    "local_ids": {},
                },
            )
            spine_obs = spine_record["observations"].setdefault(
                exp_id,
                {
                    "exp_id": exp_id,
                    "compartment": compartment,
                    "time": np.asarray(exp_time, dtype=np.float32),
                    "trace": np.asarray(spine_trace, dtype=np.float32),
                    "trace_hp": np.asarray(spine_trace_hp, dtype=np.float32),
                    "alpha": float(alpha),
                    "spine_specific": np.asarray(spine_specific, dtype=np.float32),
                    "event_info": event_info,
                    "dendrite_event_info": dict(obs.get("event_info") or {}),
                    "fit": alpha_fit,
                    "local_ids": {},
                },
            )
            if cut_neural is not None and trial_state_indices and spine_conversion_index < cut_neural.shape[0]:
                spine_cut_matrix = np.asarray(cut_neural[spine_conversion_index], dtype=float)
                if conversion_index < cut_neural.shape[0]:
                    dend_cut_matrix = np.asarray(cut_neural[conversion_index], dtype=float)
                    spine_cut_specific = spine_cut_matrix - alpha * dend_cut_matrix
                    spine_cut_state_means: Dict[str, float] = {}
                    for state_label, trial_indices in trial_state_indices.items():
                        valid_trials = [idx for idx in trial_indices if 0 <= idx < spine_cut_specific.shape[0]]
                        if not valid_trials:
                            continue
                        trial_values = np.nanmean(np.asarray(spine_cut_specific[valid_trials], dtype=float), axis=1)
                        if trial_values.size:
                            spine_cut_state_means[state_label] = float(np.nanmean(trial_values))
                    if spine_cut_state_means:
                        spine_obs["cut_state_means"] = spine_cut_state_means
            spine_obs["local_ids"] = {
                "general_roi_id": spine_entry["general_roi_id"],
                "conversion_index": spine_entry["conversion_index"],
                "plane": spine_entry["plane"],
                "plane_roi_id": spine_entry["plane_roi_id"],
                "mode": spine_entry["mode"],
                "cell_id": spine_entry.get("cell_id"),
                "dendrite_id": spine_entry.get("dendrite_id"),
                "spine_id": spine_entry.get("spine_id"),
            }
            spine_record["local_ids"][exp_id] = spine_obs["local_ids"]
            obs["spine_ids"].append(g_s_id)
    return animal_cache
def merge_animal_cache(target: Dict[str, Any], update: Dict[str, Any]) -> None:
    if "animals" not in target:
        target["animals"] = {}
    animal_id = update["animal_id"]
    animal_entry = target["animals"].setdefault(
        animal_id,
        {
            "animal_id": animal_id,
            "dendrites": {},
        },
    )
    exp_meta = update["exp_meta"]
    target.setdefault("experiments", {})[exp_meta["exp_id"]] = exp_meta
    for g_d_id, dend_record in update["dendrites"].items():
        current = animal_entry["dendrites"].setdefault(
            g_d_id,
            {
                "global_dendrite_id": g_d_id,
                "animal_id": animal_id,
                "compartment": dend_record["compartment"],
                "date": dend_record["date"],
                "observations": {},
                "local_ids": {},
                "spines": {},
            },
        )
        current["compartment"] = dend_record["compartment"]
        current["date"] = dend_record["date"]
        for exp_id, obs in dend_record["observations"].items():
            current["observations"][exp_id] = obs
        for exp_id, local in dend_record["local_ids"].items():
            current["local_ids"][exp_id] = local
        for g_s_id, spine_record in dend_record["spines"].items():
            s_current = current["spines"].setdefault(
                g_s_id,
                {
                    "global_spine_id": g_s_id,
                    "animal_id": animal_id,
                    "compartment": spine_record["compartment"],
                    "date": spine_record["date"],
                    "observations": {},
                    "local_ids": {},
                },
            )
            s_current["compartment"] = spine_record["compartment"]
            s_current["date"] = spine_record["date"]
            for exp_id, s_obs in spine_record["observations"].items():
                s_current["observations"][exp_id] = s_obs
        for exp_id, local in spine_record["local_ids"].items():
            s_current["local_ids"][exp_id] = local
def make_day_id(animal_id: str, date: str, compartment: str) -> str:
    return f"{animal_id}|{date}|{compartment}"
def grouped_experiments_by_day(cache: Dict[str, Any]) -> Dict[Tuple[str, str, str], List[str]]:
    grouped: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for exp_id, exp_meta in cache.get("experiments", {}).items():
        animal_id = str(exp_meta.get("animal_id") or derive_animal_id(exp_id))
        date = str(exp_meta.get("date") or derive_date(exp_id))
        compartment = str(exp_meta.get("compartment") or "other")
        grouped[(animal_id, date, compartment)].append(exp_id)
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key])
    return dict(sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])))
def stitch_day_time(exp_ids: Sequence[str], cache: Dict[str, Any]) -> np.ndarray:
    segments: List[np.ndarray] = []
    offset = 0.0
    for exp_id in exp_ids:
        exp_meta = cache.get("experiments", {}).get(exp_id)
        if exp_meta is None:
            continue
        time = np.asarray(exp_meta.get("time"), dtype=float).ravel()
        if time.size == 0:
            continue
        rel_time = time - float(time[0])
        if segments:
            rel_time = rel_time + offset
        segments.append(rel_time)
        diffs = np.diff(rel_time)
        positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        step = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
        offset = float(rel_time[-1] + step)
    return np.concatenate(segments) if segments else np.asarray([], dtype=float)
def stitch_day_series(
    exp_ids: Sequence[str],
    cache: Dict[str, Any],
    getter,
    *,
    fill_value: Any = np.nan,
    dtype: Any = float,
) -> np.ndarray:
    segments: List[np.ndarray] = []
    for exp_id in exp_ids:
        exp_meta = cache.get("experiments", {}).get(exp_id)
        if exp_meta is None:
            continue
        time = np.asarray(exp_meta.get("time"), dtype=float).ravel()
        if time.size == 0:
            continue
        values = getter(exp_id, exp_meta)
        if values is None:
            arr = np.full(time.shape, fill_value, dtype=dtype)
        else:
            arr = np.asarray(values, dtype=dtype).ravel()
            if arr.size < time.size:
                padded = np.full(time.shape, fill_value, dtype=dtype)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > time.size:
                arr = arr[: time.size]
        segments.append(arr)
    return np.concatenate(segments) if segments else np.asarray([], dtype=dtype)
def sleep_state_bundle_has_rem(bundle: Any) -> bool:
    if not isinstance(bundle, dict):
        return False
    for key in ["state_codes_on_calcium_time", "state_10hz", "state_codes"]:
        values = bundle.get(key)
        if values is None:
            continue
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size and np.any(np.rint(arr).astype(int) == STATE_LABEL_TO_CODE["rem"]):
            return True
    labels = bundle.get("state_labels")
    if isinstance(labels, dict) and any(str(label).lower() == "rem" for label in labels.values()):
        return True
    return False
def build_day_pooled_cache(cache: Dict[str, Any], analysis_tables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw_experiments = cache.get("experiments", {})
    day_groups = grouped_experiments_by_day(cache)
    movie_expids = set(parse_list_argument(cache.get("config", {}).get("movie_expids")))
    sleep_expids = set(parse_list_argument(cache.get("config", {}).get("sleep_expids")))
    analysis_cache: Dict[str, Any] = {
        "schema_version": cache.get("schema_version"),
        "config": dict(cache.get("config", {})),
        "config_hash": stable_hash({**dict(cache.get("config", {})), "analysis_unit": "day"}),
        "animals": {},
        "experiments": {},
        "alerts": list(cache.get("alerts", [])),
        "demo_truth": cache.get("demo_truth"),
        "analysis_unit": "day",
        "analysis_tables": cacheable(analysis_tables if isinstance(analysis_tables, dict) else (cache.get("analysis_tables", {}) if isinstance(cache.get("analysis_tables", {}), dict) else {})),
        "source_cache_summary": summarize_cache(cache),
        "day_pooling": {
            "rem_gate_skipped_days": [],
            "rem_gate_skipped_exp_ids": [],
        },
    }
    rem_gate_skipped_days: List[str] = []
    rem_gate_skipped_exp_ids: List[List[str]] = []
    for day_idx, ((animal_id, date, compartment), exp_ids) in enumerate(day_groups.items(), start=1):
        step_progress(day_idx, len(day_groups), label=f"{animal_id} | {date} | {compartment}")
        day_id = make_day_id(animal_id, date, compartment)
        representative_exp_id = exp_ids[0]
        representative_meta = dict(raw_experiments.get(representative_exp_id, {}))
        day_time = stitch_day_time(exp_ids, cache)
        sleep_day_exp_ids = [exp_id for exp_id in exp_ids if exp_id in sleep_expids]
        if sleep_day_exp_ids:
            sleep_bundles = [raw_experiments.get(exp_id, {}).get("sleep_state", {}).get("bundle") for exp_id in sleep_day_exp_ids]
            has_rem = any(sleep_state_bundle_has_rem(bundle) for bundle in sleep_bundles if bundle is not None)
            if not has_rem:
                alert = (
                    f"[ALERT] Skipping pooled sleep day {day_id} because no contributing sleep expID contains REM: "
                    f"{', '.join(sleep_day_exp_ids)}"
                )
                eprint(alert)
                analysis_cache.setdefault("alerts", []).append(alert)
                rem_gate_skipped_days.append(day_id)
                rem_gate_skipped_exp_ids.append(list(sleep_day_exp_ids))
                continue
        state_masks = {
            state_label: stitch_day_series(
                exp_ids,
                cache,
                lambda exp_id, exp_meta, state_label=state_label: exp_meta.get("state_masks", {}).get(state_label),
                fill_value=False,
                dtype=bool,
            )
            for state_label in ALL_REQUESTED_STATES
        }
        locomotion_thresholds = [
            float(value)
            for value in [as_float(raw_experiments.get(exp_id, {}).get("locomotion_threshold")) for exp_id in exp_ids]
            if value is not None and np.isfinite(value)
        ]
        trial_state_counts: Dict[str, int] = defaultdict(int)
        for exp_id in exp_ids:
            cut_meta = raw_experiments.get(exp_id, {}).get("cut", {}) or {}
            for state_label, count in cut_meta.get("trial_state_counts", {}).items():
                trial_state_counts[str(state_label)] += int(count)
        day_meta = {
            "animal_id": animal_id,
            "day_id": day_id,
            "day_pair_id": make_day_id(animal_id, date, "paired"),
            "exp_id": day_id,
            "date": date,
            "compartment": compartment,
            "repo_root": representative_meta.get("repo_root"),
            "source_paths": representative_meta.get("source_paths"),
            "source_signal_key": representative_meta.get("source_signal_key"),
            "channel": representative_meta.get("channel"),
            "conversion": representative_meta.get("conversion"),
            "time": np.asarray(day_time, dtype=np.float32),
            "wheel": {
                "time": np.asarray(day_time, dtype=np.float32),
                "speed": stitch_day_series(
                    exp_ids,
                    cache,
                    lambda exp_id, exp_meta: exp_meta.get("wheel", {}).get("speed"),
                    fill_value=np.nan,
                    dtype=float,
                ).astype(np.float32, copy=False),
                "interpolated": stitch_day_series(
                    exp_ids,
                    cache,
                    lambda exp_id, exp_meta: exp_meta.get("wheel", {}).get("interpolated"),
                    fill_value=np.nan,
                    dtype=float,
                ).astype(np.float32, copy=False),
            },
            "pupil": {
                "time": np.asarray(day_time, dtype=np.float32),
                "series": stitch_day_series(
                    exp_ids,
                    cache,
                    lambda exp_id, exp_meta: exp_meta.get("pupil", {}).get("series"),
                    fill_value=np.nan,
                    dtype=float,
                ).astype(np.float32, copy=False),
                "source": representative_meta.get("pupil", {}).get("source"),
            },
            "sleep_state": {
                "available": bool(any(raw_experiments.get(exp_id, {}).get("sleep_state", {}).get("available") for exp_id in exp_ids)),
                "alert": None,
                "bundle": None,
            },
            "cut": {
                "available": bool(any((raw_experiments.get(exp_id, {}).get("cut", {}) or {}).get("available") for exp_id in exp_ids)),
                "path": representative_meta.get("source_paths", {}).get("cut"),
                "shape": representative_meta.get("cut", {}).get("shape"),
                "time": np.asarray(day_time, dtype=np.float32),
                "bundle_keys": representative_meta.get("cut", {}).get("bundle_keys", []),
                "trial_state_counts": dict(sorted(trial_state_counts.items())),
                "trial_state_labels": sorted(
                    {
                        label
                        for exp_id in exp_ids
                        for label in (raw_experiments.get(exp_id, {}).get("cut", {}) or {}).get("trial_state_labels", [])
                        if label is not None
                    }
                ),
            },
            "state_masks": {state: np.asarray(mask, dtype=bool) for state, mask in state_masks.items()},
            "trial_rows": [],
            "trial_meta": [],
            "locomotion_threshold": float(np.nanmedian(locomotion_thresholds)) if locomotion_thresholds else float("nan"),
            "alerts": [],
            "conversion_library_summary": representative_meta.get("conversion_library_summary", {}),
            "source_signature": stable_hash(
                {
                    "analysis_unit": "day",
                    "animal_id": animal_id,
                    "date": date,
                    "compartment": compartment,
                    "source_signatures": [str(raw_experiments.get(exp_id, {}).get("source_signature", "")) for exp_id in exp_ids],
                }
            ),
            "source_exp_ids": list(exp_ids),
            "representative_exp_id": representative_exp_id,
            "day_family": (
                "pooled"
                if any(exp_id in movie_expids for exp_id in exp_ids) and any(exp_id in sleep_expids for exp_id in exp_ids)
                else "sleep"
                if any(exp_id in sleep_expids for exp_id in exp_ids)
                else "movie"
                if any(exp_id in movie_expids for exp_id in exp_ids)
                else "other"
            ),
        }
        analysis_cache["experiments"][day_id] = day_meta
        animal_entry = analysis_cache["animals"].setdefault(
            animal_id,
            {
                "animal_id": animal_id,
                "dendrites": {},
            },
        )
        raw_animal_entry = cache.get("animals", {}).get(animal_id, {})
        for global_dendrite_id, raw_dendrite_record in raw_animal_entry.get("dendrites", {}).items():
            if not any(exp_id in raw_dendrite_record.get("observations", {}) for exp_id in exp_ids):
                continue
            current_dendrite = animal_entry["dendrites"].setdefault(
                global_dendrite_id,
                {
                    "global_dendrite_id": global_dendrite_id,
                    "animal_id": animal_id,
                    "compartment": compartment,
                    "date": date,
                    "observations": {},
                    "local_ids": {},
                    "spines": {},
                },
            )
            current_dendrite["compartment"] = compartment
            current_dendrite["date"] = date
            current_dendrite["local_ids"][day_id] = dict(raw_dendrite_record.get("local_ids", {}).get(representative_exp_id, {}))
            current_dendrite["observations"][day_id] = {
                "exp_id": day_id,
                "day_id": day_id,
                "day_pair_id": make_day_id(animal_id, date, "paired"),
                "compartment": compartment,
                "time": np.asarray(day_time, dtype=np.float32),
                "trace": stitch_day_series(
                    exp_ids,
                    cache,
                    lambda exp_id, exp_meta, raw_dendrite_record=raw_dendrite_record: raw_dendrite_record.get("observations", {}).get(exp_id, {}).get("trace"),
                    fill_value=np.nan,
                    dtype=float,
                ).astype(np.float32, copy=False),
                "trace_hp": stitch_day_series(
                    exp_ids,
                    cache,
                    lambda exp_id, exp_meta, raw_dendrite_record=raw_dendrite_record: raw_dendrite_record.get("observations", {}).get(exp_id, {}).get("trace_hp"),
                    fill_value=np.nan,
                    dtype=float,
                ).astype(np.float32, copy=False),
                "spine_ids": [],
                "source_exp_ids": list(exp_ids),
                "representative_exp_id": representative_exp_id,
                "local_ids": dict(raw_dendrite_record.get("local_ids", {}).get(representative_exp_id, {})),
            }
            current_dendrite["observations"][day_id]["event_info"] = build_event_info(
                current_dendrite["observations"][day_id]["trace"],
                current_dendrite["observations"][day_id]["time"],
            )
            for spine_id, raw_spine_record in raw_dendrite_record.get("spines", {}).items():
                if not any(exp_id in raw_spine_record.get("observations", {}) for exp_id in exp_ids):
                    continue
                dendrite_event_info = current_dendrite["observations"][day_id].get("event_info")
                current_spine_record = current_dendrite["spines"].setdefault(
                    spine_id,
                    {
                        "global_spine_id": spine_id,
                        "animal_id": animal_id,
                        "compartment": compartment,
                        "date": date,
                        "observations": {},
                        "local_ids": {},
                    },
                )
                current_spine_record["compartment"] = compartment
                current_spine_record["date"] = date
                alpha_values = [
                    float(raw_spine_record.get("observations", {}).get(exp_id, {}).get("alpha"))
                    for exp_id in exp_ids
                    if raw_spine_record.get("observations", {}).get(exp_id, {}).get("alpha") is not None
                ]
                current_spine_record["local_ids"][day_id] = dict(raw_spine_record.get("local_ids", {}).get(representative_exp_id, {}))
                spine_specific = stitch_day_series(
                    exp_ids,
                    cache,
                    lambda exp_id, exp_meta, raw_spine_record=raw_spine_record: raw_spine_record.get("observations", {}).get(exp_id, {}).get("spine_specific"),
                    fill_value=np.nan,
                    dtype=float,
                ).astype(np.float32, copy=False)
                current_spine_record["observations"][day_id] = {
                    "exp_id": day_id,
                    "day_id": day_id,
                    "day_pair_id": make_day_id(animal_id, date, "paired"),
                    "compartment": compartment,
                    "time": np.asarray(day_time, dtype=np.float32),
                    "trace": stitch_day_series(
                        exp_ids,
                        cache,
                        lambda exp_id, exp_meta, raw_spine_record=raw_spine_record: raw_spine_record.get("observations", {}).get(exp_id, {}).get("trace"),
                        fill_value=np.nan,
                        dtype=float,
                    ).astype(np.float32, copy=False),
                    "trace_hp": stitch_day_series(
                        exp_ids,
                        cache,
                        lambda exp_id, exp_meta, raw_spine_record=raw_spine_record: raw_spine_record.get("observations", {}).get(exp_id, {}).get("trace_hp"),
                        fill_value=np.nan,
                        dtype=float,
                    ).astype(np.float32, copy=False),
                    "alpha": float(np.nanmean(alpha_values)) if alpha_values else float("nan"),
                    "spine_specific": spine_specific,
                    "event_info": annotate_spine_event_info(
                        build_event_info(
                            spine_specific,
                            day_time,
                        ),
                        dendrite_event_info,
                    ),
                    "dendrite_event_info": dict(dendrite_event_info or {}),
                    "fit": {},
                    "source_exp_ids": list(exp_ids),
                    "representative_exp_id": representative_exp_id,
                    "local_ids": dict(raw_spine_record.get("local_ids", {}).get(representative_exp_id, {})),
                }
                if spine_id not in current_dendrite["observations"][day_id]["spine_ids"]:
                    current_dendrite["observations"][day_id]["spine_ids"].append(spine_id)
    analysis_cache["day_pooling"] = {
        "rem_gate_skipped_days": list(rem_gate_skipped_days),
        "rem_gate_skipped_exp_ids": list(rem_gate_skipped_exp_ids),
        "rem_gate_skipped_count": int(len(rem_gate_skipped_days)),
        "rem_gate_total_day_groups": int(len(day_groups)),
        "rem_gate_sleep_day_groups": int(sum(1 for exp_ids in day_groups.values() if any(exp_id in sleep_expids for exp_id in exp_ids))),
        "rem_gate_kept_count": int(len(day_groups) - len(rem_gate_skipped_days)),
    }
    return analysis_cache
def values_from_observation(trace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if trace is None or mask is None:
        return np.array([], dtype=float)
    trace = np.asarray(trace, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if trace.size == 0 or mask.size == 0:
        return np.array([], dtype=float)
    if trace.shape[0] != mask.shape[0]:
        raise ValueError("Trace and mask must have the same length")
    return trace[mask]
def fisher_mean(rs: Sequence[float]) -> float:
    arr = np.asarray([r for r in rs if np.isfinite(r)], dtype=float)
    if arr.size == 0:
        return float("nan")
    clipped = np.clip(arr, -0.999999, 0.999999)
    return float(np.tanh(np.nanmean(np.arctanh(clipped))))
def normalize_state_value(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
def _state_summary_filter_sets(
    dendrite_ids_filter: Optional[Sequence[str]],
    spine_ids_filter: Optional[Sequence[str]],
) -> Tuple[Optional[set[str]], Optional[set[str]]]:
    dendrite_id_filter_set = None
    if dendrite_ids_filter is not None:
        dendrite_id_filter_set = {str(dendrite_id) for dendrite_id in dendrite_ids_filter if str(dendrite_id)}
    spine_id_filter_set = None
    if spine_ids_filter is not None:
        spine_id_filter_set = {str(spine_id) for spine_id in spine_ids_filter if str(spine_id)}
    return dendrite_id_filter_set, spine_id_filter_set


def _collect_state_summary_values(
    cache: Dict[str, Any],
    metric_kind: str,
    state_labels: Sequence[str],
    compartment_filter: Optional[str] = None,
    subject_key: str = "day_id",
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
    *,
    by_dendrite: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    ordered_state_labels = list(dict.fromkeys([str(state) for state in state_labels if state is not None and str(state).strip()]))
    by_state: Dict[str, Dict[str, List[float]]] = {state: defaultdict(list) for state in ordered_state_labels}
    if not ordered_state_labels:
        return by_state

    dendrite_id_filter_set, spine_id_filter_set = _state_summary_filter_sets(dendrite_ids_filter, spine_ids_filter)
    animals = cache.get("animals", {})
    experiments = cache.get("experiments", {})
    spine_metric_kinds = {
        "spine_specific_mean",
        "spine_event_frequency_per_min",
        "coincident_event_frequency_per_min",
        "noncoincident_event_frequency_per_min",
    }

    for animal_id, animal_entry in animals.items():
        dendrites = animal_entry.get("dendrites", {})
        if not isinstance(dendrites, dict):
            continue
        for dendrite_id, dendrite_record in dendrites.items():
            if dendrite_id_filter_set is not None and str(dendrite_id) not in dendrite_id_filter_set:
                continue
            observations = dendrite_record.get("observations", {})
            if not isinstance(observations, dict):
                continue
            spines = dendrite_record.get("spines", {})
            if not isinstance(spines, dict):
                spines = {}
            for exp_id, d_obs in observations.items():
                d_compartment = observation_compartment(cache, exp_id, d_obs)
                if compartment_filter is not None and d_compartment != compartment_filter:
                    continue
                exp_meta = experiments.get(exp_id)
                if exp_meta is None:
                    continue
                state_masks = exp_meta.get("state_masks", {})
                if not isinstance(state_masks, dict):
                    state_masks = {}
                selected_spine_ids = None
                if spine_id_filter_set is not None:
                    selected_spine_ids = [
                        spine_id
                        for spine_id in d_obs.get("spine_ids", [])
                        if str(spine_id) in spine_id_filter_set
                    ]
                    if not selected_spine_ids:
                        continue
                subject_id = str(exp_id) if by_dendrite else str(d_obs.get(subject_key) or exp_id or animal_id)
                d_trace = d_obs.get("trace")
                d_time = d_obs.get("time")
                d_event_info = d_obs.get("event_info") or {}
                d_cut_means = d_obs.get("cut_state_means") if metric_kind == "dendrite_mean" else None
                if metric_kind in spine_metric_kinds:
                    spine_iterable = selected_spine_ids if selected_spine_ids is not None else d_obs.get("spine_ids", [])
                    spine_items: List[Tuple[str, Dict[str, Any]]] = []
                    for spine_id in spine_iterable:
                        spine_entry = spines.get(spine_id, {})
                        if not isinstance(spine_entry, dict):
                            continue
                        s_obs = spine_entry.get("observations", {}).get(exp_id)
                        if s_obs is None:
                            continue
                        s_compartment = observation_compartment(cache, exp_id, s_obs)
                        if compartment_filter is not None and s_compartment != compartment_filter:
                            continue
                        spine_items.append((str(spine_id), s_obs))
                else:
                    spine_items = []

                for state_label in ordered_state_labels:
                    mask = state_masks.get(state_label)
                    has_mask = mask is not None and np.any(mask)
                    if metric_kind == "dendrite_mean":
                        if isinstance(d_cut_means, dict) and state_label in d_cut_means and np.isfinite(d_cut_means[state_label]):
                            by_state[state_label][subject_id].append(float(d_cut_means[state_label]))
                            continue
                        if not has_mask:
                            continue
                        values = values_from_observation(d_obs.get("trace"), mask)
                        if values.size:
                            by_state[state_label][subject_id].append(float(np.nanmean(values)))
                    elif metric_kind == "spine_specific_mean":
                        if not spine_items:
                            continue
                        spine_values: List[float] = []
                        for _, s_obs in spine_items:
                            cut_means = s_obs.get("cut_state_means")
                            if isinstance(cut_means, dict) and state_label in cut_means and np.isfinite(cut_means[state_label]):
                                spine_values.append(float(cut_means[state_label]))
                                continue
                            if not has_mask:
                                continue
                            values = values_from_observation(s_obs.get("spine_specific"), mask)
                            if values.size:
                                spine_values.append(float(np.nanmean(values)))
                        if spine_values:
                            by_state[state_label][subject_id].append(float(np.nanmean(spine_values)))
                    elif metric_kind == "dendrite_event_frequency_per_min":
                        if not has_mask:
                            continue
                        event_info = build_state_masked_event_info(d_trace, d_time, mask, d_event_info)
                        freq = as_float(event_info.get("event_frequency_per_min"))
                        if freq is not None and np.isfinite(freq):
                            by_state[state_label][subject_id].append(float(freq))
                    elif metric_kind in {"spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"}:
                        if not has_mask or not spine_items:
                            continue
                        spine_values: List[float] = []
                        for _, s_obs in spine_items:
                            spine_full_info = s_obs.get("event_info") or {}
                            spine_event_info = build_state_masked_event_info(
                                s_obs.get("trace"),
                                s_obs.get("time"),
                                mask,
                                spine_full_info,
                            )
                            if metric_kind == "spine_event_frequency_per_min":
                                freq = as_float(spine_event_info.get("spine_event_frequency_per_min", spine_event_info.get("event_frequency_per_min")))
                            else:
                                dendrite_event_info = build_state_masked_event_info(
                                    d_obs.get("trace"),
                                    d_obs.get("time"),
                                    mask,
                                    d_event_info,
                                )
                                event_info = annotate_spine_event_info(spine_event_info, dendrite_event_info)
                                freq = as_float(event_info.get(metric_kind))
                            if freq is not None and np.isfinite(freq):
                                spine_values.append(float(freq))
                        if spine_values:
                            by_state[state_label][subject_id].append(float(np.nanmean(spine_values)))
                    else:
                        raise ValueError(f"Unknown metric_kind: {metric_kind}")
    return by_state


def per_experiment_state_metrics(
    cache: Dict[str, Any],
    metric_kind: str,
    state_label: str,
    compartment_filter: Optional[str] = None,
    subject_key: str = "day_id",
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    return _collect_state_summary_values(
        cache,
        metric_kind,
        [state_label],
        compartment_filter,
        subject_key=subject_key,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    ).get(state_label, {})


def paired_comparison(
    values_by_state: Dict[str, Dict[str, List[float]]],
    state_a: str,
    state_b: str,
    metric_name: str,
    shuffle_n: int,
) -> Dict[str, Any]:
    subjects = sorted(set(values_by_state[state_a]).intersection(values_by_state[state_b]))
    a = np.asarray([normalize_state_value(values_by_state[state_a][s]) for s in subjects], dtype=float)
    b = np.asarray([normalize_state_value(values_by_state[state_b][s]) for s in subjects], dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    subjects = [s for s, keep in zip(subjects, mask) if keep]
    a = a[mask]
    b = b[mask]
    if a.size < 2 or b.size < 2:
        return {
            "metric": metric_name,
            "state_a": state_a,
            "state_b": state_b,
            "n_subjects": int(a.size),
            "paired": True,
            "test_choice": "insufficient_data",
        }
    diffs = a - b
    shapiro_p = float(stats.shapiro(diffs).pvalue) if 3 <= diffs.size <= 5000 else float("nan")
    is_normal = np.isfinite(shapiro_p) and shapiro_p > 0.05
    if is_normal:
        test_choice = "paired_t"
        classical = stats.ttest_rel(a, b, nan_policy="omit")
        classical_stat = float(classical.statistic)
        classical_p = float(classical.pvalue)
    else:
        test_choice = "wilcoxon"
        try:
            classical = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            classical_stat = float(classical.statistic)
            classical_p = float(classical.pvalue)
        except Exception:
            classical_stat = float("nan")
            classical_p = float("nan")
    observed_effect = float(np.nanmean(diffs))
    null = []
    rng = np.random.default_rng(12345)
    for _ in range(shuffle_n):
        signs = rng.choice([-1.0, 1.0], size=diffs.size)
        null.append(float(np.nanmean(diffs * signs)))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed_effect)) + 1) / (len(null) + 1))
    levene_p = float(stats.levene(a, b).pvalue) if a.size >= 3 and b.size >= 3 else float("nan")
    return {
        "metric": metric_name,
        "state_a": state_a,
        "state_b": state_b,
        "paired": True,
        "n_subjects": int(a.size),
        "subjects": subjects,
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "effect_size": observed_effect,
        "test_choice": test_choice,
        "classical_stat": classical_stat,
        "classical_p": classical_p,
        "shuffle_p": shuffle_p,
        "normality_p": shapiro_p,
        "variance_p": levene_p,
    }
def independent_comparison(
    values_by_state: Dict[str, Dict[str, List[float]]],
    state_a: str,
    state_b: str,
    metric_name: str,
    shuffle_n: int,
) -> Dict[str, Any]:
    a = np.asarray([normalize_state_value(v) for v in values_by_state[state_a].values()], dtype=float)
    b = np.asarray([normalize_state_value(v) for v in values_by_state[state_b].values()], dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {
            "metric": metric_name,
            "state_a": state_a,
            "state_b": state_b,
            "n_subjects": int(min(a.size, b.size)),
            "paired": False,
            "test_choice": "insufficient_data",
        }
    shapiro_a = float(stats.shapiro(a).pvalue) if 3 <= a.size <= 5000 else float("nan")
    shapiro_b = float(stats.shapiro(b).pvalue) if 3 <= b.size <= 5000 else float("nan")
    is_normal = (not np.isfinite(shapiro_a) or shapiro_a > 0.05) and (not np.isfinite(shapiro_b) or shapiro_b > 0.05)
    if is_normal:
        test_choice = "ttest_ind"
        classical = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        classical_stat = float(classical.statistic)
        classical_p = float(classical.pvalue)
    else:
        test_choice = "mannwhitneyu"
        classical = stats.mannwhitneyu(a, b, alternative="two-sided")
        classical_stat = float(classical.statistic)
        classical_p = float(classical.pvalue)
    observed_effect = float(np.nanmean(a) - np.nanmean(b))
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(12345)
    null = []
    for _ in range(shuffle_n):
        perm = rng.permutation(pooled.size)
        null_a = pooled[perm[:n_a]]
        null_b = pooled[perm[n_a:]]
        null.append(float(np.nanmean(null_a) - np.nanmean(null_b)))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed_effect)) + 1) / (len(null) + 1))
    levene_p = float(stats.levene(a, b).pvalue) if a.size >= 3 and b.size >= 3 else float("nan")
    return {
        "metric": metric_name,
        "state_a": state_a,
        "state_b": state_b,
        "paired": False,
        "n_subjects": int(min(a.size, b.size)),
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "effect_size": observed_effect,
        "test_choice": test_choice,
        "classical_stat": classical_stat,
        "classical_p": classical_p,
        "shuffle_p": shuffle_p,
        "normality_p_a": shapiro_a,
        "normality_p_b": shapiro_b,
        "variance_p": levene_p,
    }
def summarize_state_values(
    cache: Dict[str, Any],
    metric_kind: str,
    state_labels: Sequence[str],
    compartment_filter: Optional[str] = None,
    subject_key: str = "day_id",
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, List[float]]]:
    cache_key = _state_summary_cache_key(
        "values",
        cache,
        metric_kind=metric_kind,
        state_labels=state_labels,
        compartment_filter=compartment_filter,
        subject_key=subject_key,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    )
    cached = _STATE_SUMMARY_VALUES_CACHE.get(cache_key)
    if cached is None:
        cached = _state_summary_cache_store(cache).get(cache_key)
        if isinstance(cached, dict):
            _STATE_SUMMARY_VALUES_CACHE[cache_key] = cached
            return cached
    else:
        return cached
    by_state = _collect_state_summary_values(
        cache,
        metric_kind,
        state_labels,
        compartment_filter,
        subject_key=subject_key,
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    )
    _STATE_SUMMARY_VALUES_CACHE[cache_key] = by_state
    _state_summary_cache_store(cache)[cache_key] = by_state
    return by_state


def pairwise_state_comparisons(
    cache: Dict[str, Any],
    metric_kind: str,
    state_labels: Sequence[str],
    shuffle_n: int,
    compartment_filter: Optional[str] = None,
    dendrite_ids_filter: Optional[Sequence[str]] = None,
    spine_ids_filter: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    values_by_state = summarize_state_values(
        cache,
        metric_kind,
        state_labels,
        compartment_filter,
        subject_key="day_id",
        dendrite_ids_filter=dendrite_ids_filter,
        spine_ids_filter=spine_ids_filter,
    )
    comparisons: List[Dict[str, Any]] = []
    for state_a, state_b in combinations(state_labels, 2):
        comparisons.append(paired_comparison(values_by_state, state_a, state_b, metric_kind, shuffle_n))
    return comparisons
def basal_apical_comparison(
    cache: Dict[str, Any],
    metric_kind: str,
    state_label: str,
    shuffle_n: int,
    dendrite_ids_filter_by_compartment: Optional[Dict[str, Sequence[str]]] = None,
    spine_ids_filter_by_compartment: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    basal_values = summarize_state_values(
        cache,
        metric_kind,
        [state_label],
        compartment_filter="basal",
        subject_key="day_pair_id",
        dendrite_ids_filter=(dendrite_ids_filter_by_compartment or {}).get("basal"),
        spine_ids_filter=(spine_ids_filter_by_compartment or {}).get("basal"),
    )[state_label]
    apical_values = summarize_state_values(
        cache,
        metric_kind,
        [state_label],
        compartment_filter="apical",
        subject_key="day_pair_id",
        dendrite_ids_filter=(dendrite_ids_filter_by_compartment or {}).get("apical"),
        spine_ids_filter=(spine_ids_filter_by_compartment or {}).get("apical"),
    )[state_label]
    subjects = sorted(set(basal_values).intersection(apical_values))
    if subjects:
        paired_payload = {
            state_label + "_basal": basal_values,
            state_label + "_apical": apical_values,
        }
        # Reuse the paired test machinery by mapping the two compartments to pseudo-states.
        comparison = paired_comparison(
            {state_label + "_basal": basal_values, state_label + "_apical": apical_values},
            state_label + "_basal",
            state_label + "_apical",
            metric_kind,
            shuffle_n,
        )
        comparison["comparison"] = "basal_vs_apical"
        comparison["state"] = state_label
        return comparison
    # Fallback to independent testing if there is no matched subject intersection.
    comparison = independent_comparison(
        {state_label + "_basal": basal_values, state_label + "_apical": apical_values},
        state_label + "_basal",
        state_label + "_apical",
        metric_kind,
        shuffle_n,
    )
    comparison["comparison"] = "basal_vs_apical"
    comparison["state"] = state_label
    return comparison


def _split_group_metadata_from_membership_rows(membership_rows: Sequence[Mapping[str, Any]]) -> tuple[List[str], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    split_group_order: List[str] = []
    subject_ids_by_group: Dict[str, List[str]] = defaultdict(list)
    split_group_meta: Dict[str, Dict[str, Any]] = {}
    for row in membership_rows or []:
        if not isinstance(row, Mapping):
            continue
        split_group = str(row.get("group") or "").strip()
        subject_id = str(row.get("subject_id") or "").strip()
        if not split_group or not subject_id:
            continue
        if split_group not in split_group_order:
            split_group_order.append(split_group)
        subject_ids_by_group[split_group].append(subject_id)
        meta = split_group_meta.setdefault(split_group, {"split_group": split_group})
        display = str(row.get("group_display") or split_group).strip() or split_group
        if display and "split_group_display" not in meta:
            meta["split_group_display"] = display
        color = str(row.get("group_color") or "").strip()
        if color and "split_group_color" not in meta:
            meta["split_group_color"] = color
        rank_value = row.get("rank")
        try:
            rank_float = float(rank_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rank_float):
            continue
        current_rank = meta.get("split_group_rank")
        if current_rank is None or rank_float < float(current_rank):
            meta["split_group_rank"] = rank_float
    for split_group in split_group_order:
        split_group_meta.setdefault(split_group, {"split_group": split_group, "split_group_display": split_group})
    return split_group_order, subject_ids_by_group, split_group_meta


def _with_split_group_metadata(row: Mapping[str, Any], split_group: str, split_group_meta: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    payload = dict(row)
    payload["split_group"] = split_group
    meta = split_group_meta.get(split_group, {})
    display = meta.get("split_group_display")
    if display is not None:
        payload["split_group_display"] = display
    color = meta.get("split_group_color")
    if color is not None:
        payload["split_group_color"] = color
    rank = meta.get("split_group_rank")
    if rank is not None:
        payload["split_group_rank"] = rank
    return payload


def _split_aware_state_comparison_rows(
    cache: Dict[str, Any],
    state_comparison_states: Sequence[str] | None,
    basal_apical_states: Sequence[str] | None,
    shuffle_n: int,
    membership_rows: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_group_order, subject_ids_by_group, split_group_meta = _split_group_metadata_from_membership_rows(membership_rows)
    if not split_group_order:
        return [], []
    state_rows: List[Dict[str, Any]] = []
    basal_apical_rows: List[Dict[str, Any]] = []
    metrics = [
        "dendrite_mean",
        "spine_specific_mean",
        "dendrite_event_frequency_per_min",
        "spine_event_frequency_per_min",
        "coincident_event_frequency_per_min",
        "noncoincident_event_frequency_per_min",
    ]
    state_labels = list(state_comparison_states or [])
    basal_apical_labels = list(basal_apical_states or [])
    for split_group in split_group_order:
        subject_ids = list(dict.fromkeys(subject_ids_by_group.get(split_group, [])))
        if not subject_ids:
            continue
        for metric in metrics:
            for row in pairwise_state_comparisons(
                cache,
                metric,
                state_labels,
                shuffle_n,
                dendrite_ids_filter=subject_ids,
            ):
                state_rows.append(_with_split_group_metadata(row, split_group, split_group_meta))
        for state in basal_apical_labels:
            for metric in metrics:
                basal_apical_rows.append(
                    _with_split_group_metadata(
                        basal_apical_comparison(
                            cache,
                            metric,
                            state,
                            shuffle_n,
                            dendrite_ids_filter_by_compartment={"basal": subject_ids, "apical": subject_ids},
                        ),
                        split_group,
                        split_group_meta,
                    )
                )
    return state_rows, basal_apical_rows

def collect_state_vectors_for_dendrite(
    cache: Dict[str, Any],
    dendrite_record: Dict[str, Any],
    exp_id: str,
    state_label: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    d_obs = dendrite_record["observations"].get(exp_id)
    if d_obs is None:
        return np.array([], dtype=float), []
    exp_meta = cache["experiments"][exp_id]
    mask = exp_meta["state_masks"].get(state_label)
    if mask is None or not np.any(mask):
        return np.array([], dtype=float), []
    state_trace = d_obs["trace"][mask]
    spine_vectors: List[np.ndarray] = []
    for spine_id in d_obs["spine_ids"]:
        s_obs = dendrite_record["spines"][spine_id]["observations"].get(exp_id)
        if s_obs is None:
            continue
        spine_vectors.append(np.asarray(s_obs["spine_specific"][mask], dtype=float))
    return np.asarray(state_trace, dtype=float), spine_vectors
def correlation_matrix(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(vectors) < 2:
        return None
    min_len = min(vec.size for vec in vectors)
    if min_len < 2:
        return None
    arr = np.vstack([np.asarray(vec[:min_len], dtype=float) for vec in vectors])
    if arr.shape[1] < 2:
        return None
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(arr)
    return corr
def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix is None or matrix.size == 0:
        return np.array([], dtype=float)
    idx = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[idx], dtype=float)
def shuffle_matrix_similarity(vectors_a: List[np.ndarray], vectors_b: List[np.ndarray], shuffle_n: int) -> Tuple[float, float, float]:
    matrix_a = correlation_matrix(vectors_a)
    matrix_b = correlation_matrix(vectors_b)
    if matrix_a is None or matrix_b is None:
        return float("nan"), float("nan"), float("nan")
    tri_a = upper_triangle_values(matrix_a)
    tri_b = upper_triangle_values(matrix_b)
    mask = np.isfinite(tri_a) & np.isfinite(tri_b)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    observed = float(stats.pearsonr(tri_a[mask], tri_b[mask]).statistic)
    combined = vectors_a + vectors_b
    n_a = len(vectors_a)
    rng = np.random.default_rng(12345)
    null = []
    for _ in range(shuffle_n):
        perm = rng.permutation(len(combined))
        group_a = [combined[i] for i in perm[:n_a]]
        group_b = [combined[i] for i in perm[n_a:]]
        m_a = correlation_matrix(group_a)
        m_b = correlation_matrix(group_b)
        if m_a is None or m_b is None:
            continue
        tri_a_s = upper_triangle_values(m_a)
        tri_b_s = upper_triangle_values(m_b)
        mask_s = np.isfinite(tri_a_s) & np.isfinite(tri_b_s)
        if mask_s.sum() < 2:
            continue
        null.append(float(stats.pearsonr(tri_a_s[mask_s], tri_b_s[mask_s]).statistic))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)) if null else float("nan")
    return observed, shuffle_p, float(np.nanmean(null)) if null else float("nan")
def correlation_analysis_for_observation(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    shuffle_n: int,
    use_circular_shift: bool = True,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    shared_shuffle_key: Optional[str] = None,
) -> Dict[str, Any]:
    a = np.asarray(trace_a, dtype=float)
    b = np.asarray(trace_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or b.size < 3:
        return {"r": float("nan"), "classical_p": float("nan"), "shuffle_p": float("nan"), "n": int(a.size)}
    classical = stats.pearsonr(a, b)
    observed = float(classical.statistic)
    classical_p = float(classical.pvalue)
    null = []
    shifts = None
    if use_circular_shift and shared_shuffle_cache is not None and shared_shuffle_key is not None:
        entry = ensure_shared_shuffle_entry(shared_shuffle_cache, shared_shuffle_key, int(b.size), int(shuffle_n))
        if entry is not None:
            shifts = np.asarray(entry.get("shifts"), dtype=np.int32)
    if use_circular_shift and b.size > 3:
        if shifts is not None and shifts.size > 0:
            for shift in shifts:
                shifted = np.roll(b, int(shift))
                null.append(float(stats.pearsonr(a, shifted).statistic))
        else:
            rng = np.random.default_rng(12345)
            for _ in range(shuffle_n):
                shift = int(rng.integers(1, b.size))
                shifted = np.roll(b, shift)
                null.append(float(stats.pearsonr(a, shifted).statistic))
    else:
        rng = np.random.default_rng(12345)
        for _ in range(shuffle_n):
            perm = rng.permutation(b.size)
            null.append(float(stats.pearsonr(a, b[perm]).statistic))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))
    return {
        "r": observed,
        "classical_p": classical_p,
        "shuffle_p": shuffle_p,
        "n": int(a.size),
    }
def _spine_coactivity_pair_id(day_id: str, global_dendrite_id: str, global_spine_id_1: str, global_spine_id_2: str) -> str:
    return f"{day_id}|{global_dendrite_id}|{global_spine_id_1}|{global_spine_id_2}"
def _spine_coactivity_compute_pair_row(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    mask: np.ndarray,
    shuffle_n: int,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    shared_shuffle_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int, Optional[str]]:
    values_a = values_from_observation(trace_a, mask)
    values_b = values_from_observation(trace_b, mask)
    finite_mask = np.isfinite(values_a) & np.isfinite(values_b)
    values_a = np.asarray(values_a[finite_mask], dtype=float)
    values_b = np.asarray(values_b[finite_mask], dtype=float)
    n_frames = int(values_a.size)
    if n_frames < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_frames, "insufficient_samples"
    if not np.any(np.isfinite(values_a)) or not np.any(np.isfinite(values_b)):
        return float("nan"), float("nan"), float("nan"), float("nan"), n_frames, "no_finite_overlap"
    if np.nanstd(values_a) <= 0 or np.nanstd(values_b) <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_frames, "constant_trace"
    shared_shuffle_key = None
    if shared_shuffle_cache is not None and shared_shuffle_meta is not None:
        shared_shuffle_key = build_shared_shuffle_cache_key(
            family=str(shared_shuffle_meta.get("family", "coactivity")),
            signal=str(shared_shuffle_meta.get("signal", "spine_specific_state")),
            analysis_unit=str(shared_shuffle_meta.get("analysis_unit", "day")),
            animal_id=str(shared_shuffle_meta.get("animal_id", "unknown")),
            day_id=str(shared_shuffle_meta.get("day_id", "unknown")),
            source_id=str(shared_shuffle_meta.get("source_id", "unknown")),
            vector_length=int(n_frames),
            state_label=shared_shuffle_meta.get("state_label"),
            mask_signature=shared_shuffle_meta.get("mask_signature"),
        )
    analysis = correlation_analysis_for_observation(
        values_a,
        values_b,
        shuffle_n,
        use_circular_shift=True,
        shared_shuffle_cache=shared_shuffle_cache,
        shared_shuffle_key=shared_shuffle_key,
    )
    r_value = as_float(analysis.get("r"))
    classical_p = as_float(analysis.get("classical_p"))
    shuffle_p = as_float(analysis.get("shuffle_p"))
    if r_value is None or not np.isfinite(r_value):
        return float("nan"), float("nan"), float("nan"), float("nan"), n_frames, "pearson_failed"
    z_value = float(np.arctanh(np.clip(float(r_value), -0.999999, 0.999999)))
    return float(r_value), z_value, float(classical_p) if classical_p is not None else float("nan"), float(shuffle_p) if shuffle_p is not None else float("nan"), n_frames, None
def build_spine_coactivity_table(
    cache: Dict[str, Any],
    state_labels: Sequence[str],
    shuffle_n: int,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    rows: List[Dict[str, Any]] = []
    state_labels = [str(state) for state in state_labels if str(state)]
    state_order = {state: idx for idx, state in enumerate(state_labels)}
    analysis_unit = str(cache.get("analysis_unit", "day"))
    skip_counts: Dict[str, int] = defaultdict(int)
    tested_pairs = 0
    valid_pairs = 0
    jobs: List[Tuple[str, str, Dict[str, Any], str, Dict[str, Any], Dict[str, Any], List[str], List[Tuple[str, str]]]] = []
    for animal_id in sorted(animals):
        animal_entry = animals[animal_id]
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            for day_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                exp_meta = experiments.get(day_id)
                if exp_meta is None:
                    continue
                spine_ids = [str(spine_id) for spine_id in d_obs.get("spine_ids", [])]
                if len(spine_ids) < 2:
                    continue
                spine_pairs = list(combinations(sorted(spine_ids), 2))
                if not spine_pairs:
                    continue
                jobs.append((animal_id, str(global_dendrite_id), dendrite_record, day_id, d_obs, exp_meta, spine_ids, spine_pairs))
    total_jobs = len(jobs)
    with step_scope("spine coactivity pair table", total=total_jobs if total_jobs else None):
        for idx, (animal_id, global_dendrite_id, dendrite_record, day_id, d_obs, exp_meta, spine_ids, spine_pairs) in enumerate(jobs, start=1):
            if total_jobs:
                step_progress(idx, total_jobs, label=f"{animal_id} | {global_dendrite_id} | {day_id}")
            compartment = observation_compartment(cache, day_id, d_obs)
            d_time = np.asarray(d_obs.get("time"), dtype=float)
            state_masks = {str(state_label): np.asarray(mask, dtype=bool) for state_label, mask in exp_meta.get("state_masks", {}).items()}
            state_mask_signatures = {state_label: array_signature(mask) for state_label, mask in state_masks.items()}
            spines_by_id = dendrite_record.get("spines", {})
            day_spine_traces: Dict[str, Any] = {}
            for spine_id in spine_ids:
                spine_record = spines_by_id.get(spine_id, {})
                s_obs = spine_record.get("observations", {}).get(day_id)
                if s_obs is None:
                    day_spine_traces[spine_id] = None
                    continue
                spine_trace = s_obs.get("spine_specific")
                if spine_trace is None:
                    day_spine_traces[spine_id] = None
                    continue
                day_spine_traces[spine_id] = np.asarray(spine_trace, dtype=float)
            for state_label in state_labels:
                mask = state_masks.get(state_label)
                state_index = int(state_order.get(state_label, -1))
                if mask is None:
                    for pair_index, (spine_id_1, spine_id_2) in enumerate(spine_pairs, start=1):
                        rows.append(
                            {
                                "analysis": "spine_coactivity",
                                "animal_id": animal_id,
                                "day_id": day_id,
                                "exp_id": day_id,
                                "global_dendrite_id": global_dendrite_id,
                                "compartment": compartment,
                                "state": state_label,
                                "state_order": state_index,
                                "pair_index": pair_index,
                                "global_spine_id_1": spine_id_1,
                                "global_spine_id_2": spine_id_2,
                                "global_pair_id": _spine_coactivity_pair_id(day_id, str(global_dendrite_id), spine_id_1, spine_id_2),
                                "n_frames": 0,
                                "coactivity_r": float("nan"),
                                "coactivity_z": float("nan"),
                                "coactive": False,
                                "status": "missing_state",
                                "skip_reason": "missing_state_mask",
                            }
                        )
                        skip_counts["missing_state_mask"] += 1
                    continue
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != d_time.shape or not np.any(mask):
                    for pair_index, (spine_id_1, spine_id_2) in enumerate(spine_pairs, start=1):
                        rows.append(
                            {
                                "analysis": "spine_coactivity",
                                "animal_id": animal_id,
                                "day_id": day_id,
                                "exp_id": day_id,
                                "global_dendrite_id": global_dendrite_id,
                                "compartment": compartment,
                                "state": state_label,
                                "state_order": state_index,
                                "pair_index": pair_index,
                                "global_spine_id_1": spine_id_1,
                                "global_spine_id_2": spine_id_2,
                                "global_pair_id": _spine_coactivity_pair_id(day_id, str(global_dendrite_id), spine_id_1, spine_id_2),
                                "n_frames": 0,
                                "coactivity_r": float("nan"),
                                "coactivity_z": float("nan"),
                                "coactive": False,
                                "status": "missing_state",
                                "skip_reason": "empty_state_mask",
                            }
                        )
                        skip_counts["empty_state_mask"] += 1
                    continue
                mask_signature = state_mask_signatures.get(state_label)
                shared_shuffle_meta_base = {
                    "family": "coactivity",
                    "signal": "spine_specific_state",
                    "analysis_unit": analysis_unit,
                    "animal_id": animal_id,
                    "day_id": day_id,
                    "state_label": state_label,
                    "mask_signature": mask_signature,
                }
                for pair_index, (spine_id_1, spine_id_2) in enumerate(spine_pairs, start=1):
                    tested_pairs += 1
                    trace_a = day_spine_traces.get(spine_id_1)
                    trace_b = day_spine_traces.get(spine_id_2)
                    pair_id = _spine_coactivity_pair_id(day_id, str(global_dendrite_id), spine_id_1, spine_id_2)
                    base_row = {
                        "analysis": "spine_coactivity",
                        "animal_id": animal_id,
                        "day_id": day_id,
                        "exp_id": day_id,
                        "global_dendrite_id": global_dendrite_id,
                        "compartment": compartment,
                        "state": state_label,
                        "state_order": state_index,
                        "pair_index": pair_index,
                        "global_spine_id_1": spine_id_1,
                        "global_spine_id_2": spine_id_2,
                        "global_pair_id": pair_id,
                    }
                    if trace_a is None or trace_b is None:
                        rows.append({**base_row, "n_frames": 0, "coactivity_r": float("nan"), "coactivity_z": float("nan"), "coactive": False, "status": "missing_spine_observation", "skip_reason": "missing_spine_observation"})
                        skip_counts["missing_spine_observation"] += 1
                        continue
                    if trace_a.size == 0 or trace_b.size == 0:
                        rows.append({**base_row, "n_frames": 0, "coactivity_r": float("nan"), "coactivity_z": float("nan"), "coactive": False, "status": "empty_trace", "skip_reason": "empty_trace"})
                        skip_counts["empty_trace"] += 1
                        continue
                    shared_shuffle_meta = {**shared_shuffle_meta_base, "source_id": spine_id_2}
                    r_value, z_value, classical_p, shuffle_p, n_frames, skip_reason = _spine_coactivity_compute_pair_row(
                        trace_a,
                        trace_b,
                        mask,
                        shuffle_n,
                        shared_shuffle_cache=shared_shuffle_cache,
                        shared_shuffle_meta=shared_shuffle_meta,
                    )
                    if skip_reason is not None:
                        rows.append({**base_row, "n_frames": n_frames, "coactivity_r": r_value, "coactivity_z": z_value, "coactive": False, "shuffle_significant": False, "status": "skipped", "skip_reason": skip_reason, "classical_p": float("nan"), "shuffle_p": float("nan"), "shuffle_n_requested": int(shuffle_n), "shuffle_n_success": 0})
                        skip_counts[skip_reason] += 1
                        continue
                    rows.append({**base_row, "n_frames": n_frames, "coactivity_r": r_value, "coactivity_z": z_value, "coactive": bool(r_value > 0.0), "shuffle_significant": bool(np.isfinite(shuffle_p) and shuffle_p < REPORT_SIGNIFICANCE_ALPHA), "status": "ok", "skip_reason": None, "classical_p": classical_p, "shuffle_p": shuffle_p, "shuffle_n_requested": int(shuffle_n), "shuffle_n_success": int(shuffle_n) if shuffle_n > 0 else 0})
                    valid_pairs += 1
    table_checks = {
        "n_rows": int(len(rows)),
        "n_ok_rows": int(sum(1 for row in rows if row.get("status") == "ok")),
        "n_pairs_tested": int(tested_pairs),
        "n_pairs_valid": int(valid_pairs),
        "skip_counts": dict(skip_counts),
        "state_labels": list(state_labels),
    }
    return rows, table_checks
def summarize_spine_coactivity_table(rows: Sequence[Dict[str, Any]], state_labels: Sequence[str]) -> Dict[str, Any]:
    state_labels = [str(state) for state in state_labels if str(state)]
    compartment_order = {"basal": 0, "apical": 1}
    valid_rows = [row for row in rows if row.get("status") == "ok" and np.isfinite(as_float(row.get("coactivity_z")))]
    state_summary_rows: List[Dict[str, Any]] = []
    pair_summary_rows: List[Dict[str, Any]] = []
    animal_state_rows: List[Dict[str, Any]] = []
    state_agreement_rows: List[Dict[str, Any]] = []
    compartment_summary_rows: List[Dict[str, Any]] = []
    by_comp_state: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_comp_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_comp_animal_state: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    pair_state_vectors: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for row in valid_rows:
        compartment = str(row.get("compartment", "unknown"))
        state = str(row.get("state", "unknown"))
        animal_id = str(row.get("animal_id", "unknown"))
        pair_id = str(row.get("global_pair_id", "unknown"))
        by_comp_state[(compartment, state)].append(row)
        by_comp_pair[(compartment, pair_id)].append(row)
        by_comp_animal_state[(compartment, state, animal_id)].append(float(row.get("coactivity_z")))
        pair_state_vectors[(compartment, pair_id)][state] = float(row.get("coactivity_z"))
    for compartment, state in sorted(by_comp_state, key=lambda item: (compartment_order.get(item[0], 99), state_labels.index(item[1]) if item[1] in state_labels else 999, item[1])):
        group_rows = by_comp_state[(compartment, state)]
        r_values = np.asarray([as_float(row.get("coactivity_r")) for row in group_rows if np.isfinite(as_float(row.get("coactivity_r")))], dtype=float)
        z_values = np.asarray([as_float(row.get("coactivity_z")) for row in group_rows if np.isfinite(as_float(row.get("coactivity_z")))], dtype=float)
        if r_values.size == 0:
            continue
        state_summary_rows.append(
            {
                "compartment": compartment,
                "state": state,
                "tested_pairs": int(r_values.size),
                "mean_coactivity_r": float(np.nanmean(r_values)),
                "median_coactivity_r": float(np.nanmedian(r_values)),
                "std_coactivity_r": float(np.nanstd(r_values)),
                "sem_coactivity_r": float(np.nanstd(r_values) / math.sqrt(r_values.size)) if r_values.size > 0 else float("nan"),
                "mean_coactivity_z": float(np.nanmean(z_values)) if z_values.size else float("nan"),
                "positive_fraction": float(np.mean(r_values > 0.0)),
            }
        )
    compartment_mean_profiles: Dict[str, Dict[str, float]] = defaultdict(dict)
    for compartment in sorted({row.get("compartment", "unknown") for row in valid_rows}, key=lambda value: compartment_order.get(str(value), 99)):
        for state in state_labels:
            state_rows = [row for row in valid_rows if str(row.get("compartment", "unknown")) == compartment and str(row.get("state", "")) == state]
            z_values = np.asarray([as_float(row.get("coactivity_z")) for row in state_rows if np.isfinite(as_float(row.get("coactivity_z")))], dtype=float)
            if z_values.size:
                compartment_mean_profiles[compartment][state] = float(np.nanmean(z_values))
    for (compartment, pair_id), group_rows in sorted(by_comp_pair.items(), key=lambda item: (compartment_order.get(item[0][0], 99), item[0][1])):
        state_values = {str(row.get("state")): float(row.get("coactivity_z")) for row in group_rows if np.isfinite(as_float(row.get("coactivity_z")))}
        r_values = np.asarray([as_float(row.get("coactivity_r")) for row in group_rows if np.isfinite(as_float(row.get("coactivity_r")))], dtype=float)
        z_values = np.asarray([as_float(row.get("coactivity_z")) for row in group_rows if np.isfinite(as_float(row.get("coactivity_z")))], dtype=float)
        if z_values.size == 0:
            continue
        pair_summary = {
            "compartment": compartment,
            "global_pair_id": pair_id,
            "animal_id": str(group_rows[0].get("animal_id", "unknown")),
            "day_id": str(group_rows[0].get("day_id", group_rows[0].get("exp_id", "unknown"))),
            "global_dendrite_id": str(group_rows[0].get("global_dendrite_id", "unknown")),
            "global_spine_id_1": str(group_rows[0].get("global_spine_id_1", "unknown")),
            "global_spine_id_2": str(group_rows[0].get("global_spine_id_2", "unknown")),
            "tested_states": int(z_values.size),
            "mean_coactivity_r": float(np.nanmean(r_values)) if r_values.size else float("nan"),
            "mean_coactivity_z": float(np.nanmean(z_values)),
            "std_coactivity_r": float(np.nanstd(r_values)) if r_values.size else float("nan"),
            "coactivity_r_range": float(np.nanmax(r_values) - np.nanmin(r_values)) if r_values.size else float("nan"),
            "positive_state_fraction": float(np.mean(r_values > 0.0)) if r_values.size else float("nan"),
        }
        mean_profile = compartment_mean_profiles.get(compartment, {})
        pair_vec = []
        profile_vec = []
        for state in state_labels:
            if state in state_values and state in mean_profile:
                pair_vec.append(state_values[state])
                profile_vec.append(mean_profile[state])
        pair_vec_arr = np.asarray(pair_vec, dtype=float)
        profile_vec_arr = np.asarray(profile_vec, dtype=float)
        if pair_vec_arr.size >= 2 and np.nanstd(pair_vec_arr) > 0 and np.nanstd(profile_vec_arr) > 0:
            pair_summary["profile_similarity_r"] = float(stats.pearsonr(pair_vec_arr, profile_vec_arr).statistic)
        else:
            pair_summary["profile_similarity_r"] = float("nan")
        pair_summary["profile_similarity_n_states"] = int(pair_vec_arr.size)
        pair_summary_rows.append(pair_summary)
    for (compartment, state, animal_id), z_values in sorted(by_comp_animal_state.items(), key=lambda item: (compartment_order.get(item[0][0], 99), state_labels.index(item[0][1]) if item[0][1] in state_labels else 999, item[0][2])):
        arr = np.asarray(z_values, dtype=float)
        if arr.size == 0:
            continue
        animal_state_rows.append(
            {
                "compartment": compartment,
                "state": state,
                "animal_id": animal_id,
                "mean_coactivity_z": float(np.nanmean(arr)),
                "mean_coactivity_r": float(np.tanh(np.nanmean(arr))),
                "tested_pairs": int(arr.size),
            }
        )
    animal_state_mean_lookup: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    animal_state_mean_r_lookup: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for row in animal_state_rows:
        compartment = str(row["compartment"])
        state = str(row["state"])
        animal_id = str(row["animal_id"])
        animal_state_mean_lookup[(compartment, state)][animal_id] = float(row["mean_coactivity_z"])
        animal_state_mean_r_lookup[(compartment, state)][animal_id] = float(row["mean_coactivity_r"])
    for compartment in sorted({row.get("compartment", "unknown") for row in animal_state_rows}, key=lambda value: compartment_order.get(str(value), 99)):
        for state_a, state_b in combinations(state_labels, 2):
            animals_a = animal_state_mean_lookup.get((compartment, state_a), {})
            animals_b = animal_state_mean_lookup.get((compartment, state_b), {})
            common_animals = sorted(set(animals_a).intersection(animals_b))
            if len(common_animals) < 2:
                continue
            a = np.asarray([animals_a[animal] for animal in common_animals], dtype=float)
            b = np.asarray([animals_b[animal] for animal in common_animals], dtype=float)
            if np.nanstd(a) <= 0 or np.nanstd(b) <= 0:
                agreement_r = float("nan")
            else:
                agreement_r = float(stats.pearsonr(a, b).statistic)
            a_sign = np.asarray([animal_state_mean_r_lookup[(compartment, state_a)][animal] for animal in common_animals], dtype=float)
            b_sign = np.asarray([animal_state_mean_r_lookup[(compartment, state_b)][animal] for animal in common_animals], dtype=float)
            same_sign_fraction = float(np.mean(np.sign(a_sign) == np.sign(b_sign))) if common_animals else float("nan")
            state_agreement_rows.append(
                {
                    "compartment": compartment,
                    "state_a": state_a,
                    "state_b": state_b,
                    "tested_animals": int(len(common_animals)),
                    "agreement_r": agreement_r,
                    "same_sign_fraction": same_sign_fraction,
                    "mean_a_z": float(np.nanmean(a)),
                    "mean_b_z": float(np.nanmean(b)),
                }
            )
    for compartment in sorted({row.get("compartment", "unknown") for row in pair_summary_rows}, key=lambda value: compartment_order.get(str(value), 99)):
        comp_pairs = [row for row in pair_summary_rows if str(row.get("compartment", "unknown")) == compartment]
        comp_agreement = [row for row in state_agreement_rows if str(row.get("compartment", "unknown")) == compartment]
        compartment_summary_rows.append(
            {
                "compartment": compartment,
                "n_pairs": int(len(comp_pairs)),
                "n_animals": int(len({str(row.get("animal_id")) for row in comp_pairs})),
                "mean_pair_coactivity_r": float(np.nanmean([as_float(row.get("mean_coactivity_r")) for row in comp_pairs])) if comp_pairs else float("nan"),
                "mean_positive_state_fraction": float(np.nanmean([as_float(row.get("positive_state_fraction")) for row in comp_pairs])) if comp_pairs else float("nan"),
                "mean_profile_similarity_r": float(np.nanmean([value for value in (as_float(row.get("profile_similarity_r")) for row in comp_pairs) if value is not None])) if any(as_float(row.get("profile_similarity_r")) is not None for row in comp_pairs) else float("nan"),
                "mean_state_agreement_r": float(np.nanmean([value for value in (as_float(row.get("agreement_r")) for row in comp_agreement) if value is not None])) if any(as_float(row.get("agreement_r")) is not None for row in comp_agreement) else float("nan"),
            }
        )
    overall_summary_rows = list(compartment_summary_rows)
    return {
        "state_summary_rows": state_summary_rows,
        "pair_summary_rows": pair_summary_rows,
        "animal_state_rows": animal_state_rows,
        "state_agreement_rows": state_agreement_rows,
        "compartment_summary_rows": compartment_summary_rows,
        "overall_summary_rows": overall_summary_rows,
    }


def _spine_coactivity_pair_property_row(cache: Dict[str, Any], row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    state = canonical_state_label(row.get("state"))
    animal_id = str(row.get("animal_id") or "").strip()
    day_id = str(row.get("day_id") or row.get("exp_id") or "").strip()
    global_dendrite_id = str(row.get("global_dendrite_id") or "").strip()
    global_pair_id = str(row.get("global_pair_id") or "").strip()
    if not state or not animal_id or not day_id or not global_dendrite_id or not global_pair_id:
        return None

    experiments = cache.get("experiments", {})
    exp_meta = experiments.get(day_id)
    if exp_meta is None:
        return None
    time = np.asarray(exp_meta.get("time"), dtype=float)
    state_mask = np.asarray(exp_meta.get("state_masks", {}).get(state), dtype=bool)
    if state_mask.shape != time.shape or not np.any(state_mask):
        return None

    animals = cache.get("animals", {})
    animal_entry = animals.get(animal_id)
    if not isinstance(animal_entry, dict):
        return None
    dendrite_record = animal_entry.get("dendrites", {}).get(global_dendrite_id)
    if not isinstance(dendrite_record, dict):
        return None
    spines_by_id = dendrite_record.get("spines", {})
    if not isinstance(spines_by_id, dict):
        return None

    activity_values: List[float] = []
    frequency_values: List[float] = []
    for spine_id in (str(row.get("global_spine_id_1") or "").strip(), str(row.get("global_spine_id_2") or "").strip()):
        if not spine_id:
            return None
        spine_entry = spines_by_id.get(spine_id, {})
        if not isinstance(spine_entry, dict):
            return None
        s_obs = spine_entry.get("observations", {}).get(day_id)
        if s_obs is None:
            return None
        cut_means = s_obs.get("cut_state_means")
        activity_value = as_float(cut_means.get(state)) if isinstance(cut_means, dict) else None
        if activity_value is None or not np.isfinite(activity_value):
            activity_trace = values_from_observation(s_obs.get("spine_specific"), state_mask)
            activity_trace = activity_trace[np.isfinite(activity_trace)]
            activity_value = float(np.nanmean(activity_trace)) if activity_trace.size else float("nan")
        if activity_value is not None and np.isfinite(activity_value):
            activity_values.append(float(activity_value))
        spine_event_info = build_state_masked_event_info(
            s_obs.get("trace"),
            s_obs.get("time"),
            state_mask,
            s_obs.get("event_info") or {},
        )
        frequency_value = as_float(spine_event_info.get("spine_event_frequency_per_min", spine_event_info.get("event_frequency_per_min")))
        if frequency_value is not None and np.isfinite(frequency_value):
            frequency_values.append(float(frequency_value))

    if len(activity_values) < 2 or len(frequency_values) < 2:
        return None

    coactivity_r = as_float(row.get("coactivity_r"))
    if coactivity_r is None or not np.isfinite(coactivity_r):
        return None

    n_frames = as_float(row.get("n_frames"))
    payload = dict(row)
    payload.update(
        {
            "subject_id": global_pair_id,
            "state_duration_s": float(n_frames) if n_frames is not None and np.isfinite(n_frames) else float("nan"),
            "state_n_frames": int(n_frames) if n_frames is not None and np.isfinite(n_frames) else 0,
            "mean_spine_specific_mean": float(np.nanmean(activity_values)),
            "mean_spine_event_frequency_per_min": float(np.nanmean(frequency_values)),
        }
    )
    return payload



def build_spine_coactivity_property_split_results(
    cache: Dict[str, Any],
    spine_coactivity_results: Mapping[str, Any],
    state_labels: Sequence[str],
    shuffle_n: int,
) -> Dict[str, Any]:
    table_rows: List[Dict[str, Any]] = []
    for row in spine_coactivity_results.get("table_rows", []):
        if str(row.get("status")) != "ok":
            continue
        coactivity_value = as_float(row.get("coactivity_r"))
        if coactivity_value is None or not np.isfinite(coactivity_value):
            continue
        table_rows.append(dict(row))
    enriched_rows: List[Dict[str, Any]] = []
    for row in table_rows:
        enriched_row = _spine_coactivity_pair_property_row(cache, row)
        if enriched_row is not None:
            enriched_rows.append(enriched_row)

    sleep_expids = cache.get("config", {}).get("sleep_expids")
    compartments = spine_coactivity_output_compartments(enriched_rows)
    bundles: List[Dict[str, Any]] = []
    for compartment in compartments:
        compartment_rows = [row for row in enriched_rows if str(row.get("compartment") or "") == compartment]
        if not compartment_rows:
            continue
        for basis_name in ("all", "nrem", "rem"):
            bundle = build_roi_split_results(
                compartment_rows,
                roi_type="spine_coactivity_property",
                split_name="activity_frequency",
                score_column="mean_spine_specific_mean",
                secondary_score_column="mean_spine_event_frequency_per_min",
                response_columns=("coactivity_r",),
                subject_key="global_pair_id",
                compartment=compartment,
                selected_states=state_labels,
                state_order=state_labels,
                shuffle_n=shuffle_n,
                sleep_expids=sleep_expids,
                branch_name="spine_coactivity_property_split",
                basis_name=basis_name,
                split_mode="quadrant",
            )
            if bundle.get("counts", {}).get("n_membership_rows", 0):
                bundles.append(bundle)

    subject_state_rows: List[Dict[str, Any]] = []
    seen_subject_state_keys: set[tuple[str, str, str, str, str, str]] = set()
    for bundle in bundles:
        for row in bundle.get("subject_state_rows", []):
            key = (
                str(row.get("branch_name") or ""),
                str(row.get("basis_name") or ""),
                str(row.get("roi_type") or ""),
                str(row.get("compartment") or ""),
                str(row.get("subject_id") or ""),
                str(row.get("state") or ""),
            )
            if key in seen_subject_state_keys:
                continue
            seen_subject_state_keys.add(key)
            subject_state_rows.append(dict(row))

    result = {
        "branches": {},
        "bundles": bundles,
        "subject_state_rows": subject_state_rows,
        "membership_rows": [row for bundle in bundles for row in bundle.get("membership_rows", [])],
        "comparison_rows": [row for bundle in bundles for row in bundle.get("comparison_rows", [])],
        "summary_rows": [row for bundle in bundles for row in bundle.get("summary_rows", [])],
        "counts": {
            "subject_state_rows": int(len(subject_state_rows)),
            "membership_rows": int(sum(len(bundle.get("membership_rows", [])) for bundle in bundles)),
            "comparison_rows": int(sum(len(bundle.get("comparison_rows", [])) for bundle in bundles)),
            "summary_rows": int(sum(len(bundle.get("summary_rows", [])) for bundle in bundles)),
            "bundles": int(len(bundles)),
            "branches": int(len({str(bundle.get("branch_name") or "") for bundle in bundles if str(bundle.get("branch_name") or "")})),
            "basis_leaves": int(len({(str(bundle.get("branch_name") or ""), str(bundle.get("basis_name") or "")) for bundle in bundles if str(bundle.get("branch_name") or "") and str(bundle.get("basis_name") or "")})),
        },
    }
    for bundle in bundles:
        branch_key = str(bundle.get("branch_name") or bundle.get("split_name") or "").strip() or "split"
        basis_key = str(bundle.get("basis_name") or "all").strip() or "all"
        result["branches"].setdefault(branch_key, {})[basis_key] = bundle
    return result


def run_spine_coactivity_analysis(
    cache: Dict[str, Any],
    shuffle_n: int,
    state_comparison_states: Optional[Sequence[str]],
    basal_apical_states: Optional[Sequence[str]],
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    fit_spine_coactivity_mixed_model: bool = False,
    mixed_model_contrast_p_source: str = "classical",
    spine_coactivity_abs_threshold: float = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
) -> Dict[str, Any]:
    cache_config = cache.get("config", {})
    if not isinstance(cache_config, dict):
        cache_config = {}
    analysis_state_set = list(dict.fromkeys([str(state) for state in (state_comparison_states or []) + (basal_apical_states or [])]))
    if not analysis_state_set:
        analysis_state_set = list(PRIMARY_QUIET_STATES)
    analysis_states = [state for state in ALL_REQUESTED_STATES if state in analysis_state_set]
    if not analysis_states:
        analysis_states = list(PRIMARY_QUIET_STATES)
    rebuild_requested = bool(cache_config.get("analysis_tables_rebuild")) or bool(cache_config.get("rebuild"))
    cached_table = load_cached_analysis_table(
        cache,
        "spine_coactivity_table",
        expected_meta={
            "analysis_unit": str(cache.get("analysis_unit", "day")),
            "state_labels": list(analysis_states),
            "shuffle_n": int(shuffle_n),
            "shared_shuffle_cache_present": bool(shared_shuffle_cache),
        },
        rebuild=rebuild_requested,
    )
    if cached_table is not None:
        table_rows = list(cached_table["table_rows"])
        table_checks = dict(cached_table["table_checks"])
        summary = cached_table.get("summary")
        if not isinstance(summary, dict) or not summary:
            with step_scope("spine coactivity summary assembly"):
                summary = summarize_spine_coactivity_table(table_rows, analysis_states)
    else:
        table_rows, table_checks = build_spine_coactivity_table(cache, analysis_states, shuffle_n, shared_shuffle_cache=shared_shuffle_cache)
        with step_scope("spine coactivity summary assembly"):
            summary = summarize_spine_coactivity_table(table_rows, analysis_states)
        store_cached_analysis_table(
            cache,
            "spine_coactivity_table",
            table_rows,
            table_checks,
            meta={
                "analysis_unit": str(cache.get("analysis_unit", "day")),
                "state_labels": list(analysis_states),
                "shuffle_n": int(shuffle_n),
                "shared_shuffle_cache_present": bool(shared_shuffle_cache),
            },
            summary=summary,
        )
    outputs: Dict[str, Any] = {
        "available": bool(MixedLM is not None),
        "enabled": bool(fit_spine_coactivity_mixed_model),
        "p_value_source": normalize_mixed_model_contrast_p_source(mixed_model_contrast_p_source),
        "p_value_source_requested": normalize_mixed_model_contrast_p_source(mixed_model_contrast_p_source),
        "alerts": [],
        "table_checks": table_checks,
        "table_rows": table_rows,
        "pair_state_rows": [
            row
            for row in table_rows
            if str(row.get("status")) == "ok" and (lambda value: value is not None and np.isfinite(value))(as_float(row.get("coactivity_r")))
        ],
        "rows": table_rows,
        "state_summary_rows": summary["state_summary_rows"],
        "pair_summary_rows": summary["pair_summary_rows"],
        "animal_state_rows": summary["animal_state_rows"],
        "state_agreement_rows": summary["state_agreement_rows"],
        "compartment_summary_rows": summary["compartment_summary_rows"],
        "overall_summary_rows": summary["overall_summary_rows"],
        "summary_rows": {"coactivity_r": []},
        "contrast_rows": [],
        "designs": {},
        "model_equations": {},
        "tested_terms": {},
        "tested_contrasts": {},
        "selection": {
            "state_comparison_states": list(state_comparison_states) if state_comparison_states is not None else list(PRIMARY_QUIET_STATES),
            "basal_apical_states": list(basal_apical_states) if basal_apical_states is not None else list(DEFAULT_BASAL_APICAL_STATES),
            "dendrite_response_cohort": cache_config.get("dendrite_response_cohort", DEFAULT_DENDRITE_RESPONSE_COHORT),
            "spine_coactivity_anchor_state": SPINE_COACTIVITY_ANCHOR_STATE,
            "spine_coactivity_abs_threshold": float(spine_coactivity_abs_threshold),
            "spine_coactivity_selection_rule": spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold),
            "spine_coactivity_selection_field": SPINE_COACTIVITY_QUIET_ANCHOR_SELECTION_FIELD,
        },
    }
    with step_scope("spine coactivity property split assembly"):
        outputs["property_split"] = build_spine_coactivity_property_split_results(
            cache,
            outputs,
            analysis_states,
            shuffle_n,
        )
    if not fit_spine_coactivity_mixed_model:
        outputs["model"] = {}
        return outputs
    if MixedLM is None:
        outputs["alerts"].append("[ALERT] statsmodels is unavailable, so the spine coactivity mixed-model analysis was skipped.")
        outputs["model"] = {}
        return outputs
    state_pairs = []
    for state_a, state_b in combinations(outputs["selection"]["state_comparison_states"], 2):
        state_pairs.append({"kind": "state_pair", "state_a": state_a, "state_b": state_b})
    basal_apical_pairs = []
    for state in outputs["selection"]["basal_apical_states"]:
        basal_apical_pairs.append({"kind": "basal_apical", "state": state})
    with step_scope("spine coactivity mixed model fit"):
        coactivity_result = run_mixed_model_family(
            table_rows,
            "coactivity_r",
            "all_state",
            state_pairs + basal_apical_pairs,
            shuffle_n,
            alerts=outputs["alerts"],
            vc_level_keys=["global_pair_id"],
            p_value_source=outputs["p_value_source"],
        )
    outputs["summary_rows"]["coactivity_r"] = coactivity_result.get("summary_rows", [])
    outputs["contrast_rows"] = list(coactivity_result.get("contrast_rows", []))
    outputs["p_value_source"] = coactivity_result.get("p_value_source", outputs["p_value_source"])
    if coactivity_result.get("design") is not None:
        design = coactivity_result["design"]
        outputs["designs"]["coactivity_r"] = {
            "response": "coactivity_r",
            "scope": "all_state",
            "p_value_source": coactivity_result.get("p_value_source", outputs["p_value_source"]),
            "state_levels": list(design.get("state_levels", [])),
            "state_reference": design.get("state_reference"),
            "compartment_levels": list(design.get("compartment_levels", [])),
            "compartment_reference": design.get("compartment_reference"),
            "include_compartment": bool(design.get("include_compartment", False)),
            "interaction_compartments": list(design.get("interaction_compartments", [])),
            "covariate_specs": list(design.get("covariate_specs", [])),
            "fixed_effect_names": list(design.get("fixed_effect_names", [])),
            "random_structure_name": coactivity_result.get("fit", {}).get("random_structure_name"),
            "fit_method": coactivity_result.get("fit", {}).get("fit_method"),
            "converged": bool(coactivity_result.get("fit", {}).get("converged", False)),
            "vc_level_keys": ["global_pair_id"],
        }
        outputs["model_equations"]["coactivity_r"] = coactivity_result.get("equation")
        outputs["tested_terms"]["coactivity_r"] = list(coactivity_result.get("tested_terms", []))
        outputs["tested_contrasts"]["coactivity_r"] = list(coactivity_result.get("tested_contrasts", []))
    outputs["model"] = {k: v for k, v in coactivity_result.items() if k not in {"model", "fit"}}
    return outputs
def process_spine_coactivity_only(
    cache: Dict[str, Any],
    shuffle_n: int,
    state_comparison_states: Optional[Sequence[str]] = None,
    basal_apical_states: Optional[Sequence[str]] = None,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    figure_root: Optional[Path] = None,
    fit_spine_coactivity_mixed_model: bool = False,
    mixed_model_contrast_p_source: str = "classical",
    spine_coactivity_abs_threshold: float = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
) -> Dict[str, Any]:
    # Standalone entry point that only builds the spine-coactivity analysis.
    results: Dict[str, Any] = {
        "state_comparisons": [],
        "basal_apical_comparisons": [],
        "correlations": [],
        "matrix_similarity": [],
        "state_summaries": {},
        "state_dendrite_summaries": {},
        "demo_validation": [],
        "alerts": list(cache.get("alerts", [])),
        "state_coverage": [],
        "mixed_model": {},
        "mixed_model_selected_state": {},
        "direct_trial_type_comparison": {},
        "spine_coactivity": {},
        "spine_coactivity_model": {},
    }
    with step_scope("spine coactivity analysis"):
        spine_coactivity_results = run_spine_coactivity_analysis(
            cache,
            shuffle_n=shuffle_n,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            shared_shuffle_cache=shared_shuffle_cache,
            fit_spine_coactivity_mixed_model=fit_spine_coactivity_mixed_model,
            mixed_model_contrast_p_source=mixed_model_contrast_p_source,
            spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
        )
    results["spine_coactivity"] = {k: v for k, v in spine_coactivity_results.items() if k != "model"}
    results["spine_coactivity_model"] = {
        "available": spine_coactivity_results.get("available", False),
        "enabled": spine_coactivity_results.get("enabled", False),
        "p_value_source": spine_coactivity_results.get("p_value_source", "classical"),
        "p_value_source_requested": spine_coactivity_results.get("p_value_source_requested", "classical"),
        "alerts": list(spine_coactivity_results.get("alerts", [])),
        "summary_rows": {"coactivity_r": list(spine_coactivity_results.get("summary_rows", {}).get("coactivity_r", []))},
        "contrast_rows": list(spine_coactivity_results.get("contrast_rows", [])),
        "designs": spine_coactivity_results.get("designs", {}),
        "model_equations": spine_coactivity_results.get("model_equations", {}),
        "tested_terms": spine_coactivity_results.get("tested_terms", {}),
        "tested_contrasts": spine_coactivity_results.get("tested_contrasts", {}),
        "selection": spine_coactivity_results.get("selection", {}),
    }
    results["alerts"].extend(spine_coactivity_results.get("alerts", []))
    if output_dir is not None:
        with step_scope("figure generation: spine_coactivity"):
            render_analysis_family_figures(output_dir, results, cache, "spine_coactivity", figure_root=figure_root)
    results["analysis_mode"] = "spine_coactivity_only"
    return results

def build_mixed_model_roi_split_results(
    cache: Dict[str, Any],
    mixed_model_results: Mapping[str, Any],
    state_comparison_states: Sequence[str] | None,
    shuffle_n: int,
) -> Dict[str, Any]:
    roi_split_bundles: List[Dict[str, Any]] = []
    mixed_model_table_rows = list(mixed_model_results.get("table_rows", []))
    sleep_expids = cache.get("config", {}).get("sleep_expids")
    comparison_states = list(state_comparison_states or mixed_model_results.get("selection", {}).get("state_comparison_states", []) or [])
    split_basis_names = ("all", "nrem", "rem")
    split_specs = (
        ("activity_split", "activity", "binary"),
        ("frequency_split", "frequency", "binary"),
        ("activity_frequency_split", "activity_frequency", "quadrant"),
    )
    for compartment in ["basal", "apical"]:
        compartment_rows = [row for row in mixed_model_table_rows if str(row.get("compartment") or "") == compartment]
        if not compartment_rows:
            continue
        for roi_type, subject_key, response_columns in (
            ("dendrite", "global_dendrite_id", ("mean_dendrite_activity", "dendrite_event_frequency_per_min")),
            ("spine", "global_dendrite_id", ("mean_spine_activity_per_dendrite", "spine_event_frequency_per_min")),
        ):
            primary_score_column = response_columns[0]
            secondary_score_column = response_columns[1]
            for branch_name, split_name, split_mode in split_specs:
                score_column = secondary_score_column if branch_name == "frequency_split" else primary_score_column
                split_secondary_score = secondary_score_column if branch_name == "activity_frequency_split" else None
                for basis_name in split_basis_names:
                    bundle = build_roi_split_results(
                        compartment_rows,
                        roi_type=roi_type,
                        split_name=split_name,
                        branch_name=branch_name,
                        basis_name=basis_name,
                        score_column=score_column,
                        secondary_score_column=split_secondary_score,
                        response_columns=response_columns,
                        subject_key=subject_key,
                        compartment=compartment,
                        selected_states=comparison_states,
                        state_order=comparison_states,
                        shuffle_n=shuffle_n,
                        split_mode=split_mode,
                        sleep_expids=sleep_expids,
                    )
                    if bundle.get("counts", {}).get("n_membership_rows", 0):
                        roi_split_bundles.append(bundle)
    roi_split_subject_state_rows: List[Dict[str, Any]] = []
    seen_roi_split_subject_state_keys: set[tuple[str, str, str, str, str, str]] = set()
    for bundle in roi_split_bundles:
        for row in bundle.get("subject_state_rows", []):
            key = (
                str(row.get("branch_name") or ""),
                str(row.get("basis_name") or ""),
                str(row.get("roi_type") or ""),
                str(row.get("compartment") or ""),
                str(row.get("subject_id") or ""),
                str(row.get("state") or ""),
            )
            if key in seen_roi_split_subject_state_keys:
                continue
            seen_roi_split_subject_state_keys.add(key)
            roi_split_subject_state_rows.append(dict(row))
    roi_split_results = {
        "branches": {},
        "bundles": roi_split_bundles,
        "subject_state_rows": roi_split_subject_state_rows,
        "membership_rows": [row for bundle in roi_split_bundles for row in bundle.get("membership_rows", [])],
        "comparison_rows": [row for bundle in roi_split_bundles for row in bundle.get("comparison_rows", [])],
        "summary_rows": [row for bundle in roi_split_bundles for row in bundle.get("summary_rows", [])],
        "counts": {
            "subject_state_rows": int(len(roi_split_subject_state_rows)),
            "membership_rows": int(sum(len(bundle.get("membership_rows", [])) for bundle in roi_split_bundles)),
            "comparison_rows": int(sum(len(bundle.get("comparison_rows", [])) for bundle in roi_split_bundles)),
            "summary_rows": int(sum(len(bundle.get("summary_rows", [])) for bundle in roi_split_bundles)),
            "bundles": int(len(roi_split_bundles)),
            "branches": int(len({str(bundle.get("branch_name") or "") for bundle in roi_split_bundles if str(bundle.get("branch_name") or "")})),
            "basis_leaves": int(len({(str(bundle.get("branch_name") or ""), str(bundle.get("basis_name") or "")) for bundle in roi_split_bundles if str(bundle.get("branch_name") or "") and str(bundle.get("basis_name") or "")})),
        },
    }
    for bundle in roi_split_bundles:
        branch_key = str(bundle.get("branch_name") or bundle.get("split_name") or "").strip() or "split"
        basis_key = str(bundle.get("basis_name") or "all").strip() or "all"
        roi_split_results["branches"].setdefault(branch_key, {})[basis_key] = bundle
    return roi_split_results

def process_mixed_model_only(
    cache: Dict[str, Any],
    shuffle_n: int,
    state_comparison_states: Optional[Sequence[str]] = None,
    basal_apical_states: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
    figure_root: Optional[Path] = None,
    mixed_model_contrast_p_source: str = "classical",
) -> Dict[str, Any]:
    # Standalone entry point that only builds the main mixed-model branch.
    results: Dict[str, Any] = {
        "state_comparisons": [],
        "basal_apical_comparisons": [],
        "correlations": [],
        "matrix_similarity": [],
        "state_summaries": {},
        "state_dendrite_summaries": {},
        "demo_validation": [],
        "alerts": list(cache.get("alerts", [])),
        "state_coverage": [],
        "mixed_model": {},
        "mixed_model_selected_state": {},
        "direct_trial_type_comparison": {},
        "spine_coactivity": {},
        "spine_coactivity_model": {},
    }
    with step_scope("mixed model analysis"):
        mixed_model_results = run_mixed_model_analysis(
            cache,
            shuffle_n=shuffle_n,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            mixed_model_contrast_p_source=mixed_model_contrast_p_source,
            source_cache=cache,
        )
    results["mixed_model"] = mixed_model_results.get("selected_state", {})
    results["mixed_model_selected_state"] = mixed_model_results.get("selected_state", {})
    results["mixed_model_visual_response_responsive"] = mixed_model_results.get("mixed_model_visual_response_responsive", {})
    results["mixed_model_visual_response_nonresponsive"] = mixed_model_results.get("mixed_model_visual_response_nonresponsive", {})
    results["mixed_model_visual_response_responsive_selected_state"] = mixed_model_results.get("mixed_model_visual_response_responsive_selected_state", {}) or results.get("mixed_model_visual_response_responsive", {})
    results["mixed_model_visual_response_nonresponsive_selected_state"] = mixed_model_results.get("mixed_model_visual_response_nonresponsive_selected_state", {}) or results.get("mixed_model_visual_response_nonresponsive", {})
    results["mixed_model_visual_response_responsive"] = mixed_model_results.get("mixed_model_visual_response_responsive", {})
    results["mixed_model_visual_response_nonresponsive"] = mixed_model_results.get("mixed_model_visual_response_nonresponsive", {})
    results["mixed_model_visual_response_responsive_selected_state"] = mixed_model_results.get("mixed_model_visual_response_responsive_selected_state", {}) or results.get("mixed_model_visual_response_responsive", {})
    results["mixed_model_visual_response_nonresponsive_selected_state"] = mixed_model_results.get("mixed_model_visual_response_nonresponsive_selected_state", {}) or results.get("mixed_model_visual_response_nonresponsive", {})
    results["alerts"].extend(mixed_model_results.get("alerts", []))
    results["demo_validation"].extend(mixed_model_results.get("validation_rows", []))
    results["roi_split"] = build_mixed_model_roi_split_results(
        cache,
        mixed_model_results,
        state_comparison_states,
        shuffle_n,
    )
    split_membership_rows = results["roi_split"].get("membership_rows", []) if isinstance(results.get("roi_split"), dict) else []
    split_state_comparisons, split_basal_apical_comparisons = _split_aware_state_comparison_rows(
        cache,
        state_comparison_states,
        basal_apical_states,
        shuffle_n,
        split_membership_rows,
    )
    if split_state_comparisons:
        results["state_comparisons"] = split_state_comparisons
    if split_basal_apical_comparisons:
        results["basal_apical_comparisons"] = split_basal_apical_comparisons

    if output_dir is not None:
        with step_scope("figure generation: mixed_model"):
            render_analysis_family_figures(output_dir, results, cache, "mixed_model", figure_root=figure_root)
    results["analysis_mode"] = "mixed_model_only"
    return results
def _direct_trial_type_trial_response(cut_arr: np.ndarray, roi_index: int, trial_index: int) -> Tuple[float, int]:
    if cut_arr is None:
        return float("nan"), 0
    arr = np.asarray(cut_arr, dtype=float)
    if arr.ndim != 3:
        return float("nan"), 0
    if roi_index < 0 or roi_index >= arr.shape[0]:
        return float("nan"), 0
    if trial_index < 0 or trial_index >= arr.shape[1]:
        return float("nan"), 0
    trial_values = np.asarray(arr[roi_index, trial_index], dtype=float).ravel()
    trial_values = trial_values[np.isfinite(trial_values)]
    if trial_values.size == 0:
        return float("nan"), 0
    return float(np.nanmean(trial_values)), int(trial_values.size)
def build_direct_trial_type_comparison_table(cache: Dict[str, Any], state_labels: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    movie_expids = set(parse_list_argument(cache.get("config", {}).get("movie_expids")))
    if not movie_expids:
        movie_expids = {str(exp_id) for exp_id, exp_meta in experiments.items() if exp_meta.get("trial_rows")}
    state_labels = [str(state) for state in state_labels if str(state) and str(state) in MOVIE_STATE_LABELS]
    state_labels = [state for state in MOVIE_STATE_LABELS if state in set(state_labels)]
    state_order = {state: idx for idx, state in enumerate(state_labels)}
    movie_trial_types = infer_movie_trial_types_from_states(state_labels)
    candidate_rows: List[Dict[str, Any]] = []
    skip_counts: Dict[str, int] = defaultdict(int)
    cut_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    trial_rows_cache: Dict[str, List[Mapping[str, Any]]] = {}
    trial_meta_cache: Dict[str, List[Mapping[str, Any]]] = {}
    jobs: List[Tuple[str, str, Dict[str, Any], Dict[str, Any], str]] = []
    for animal_id in sorted(animals):
        animal_entry = animals[animal_id]
        for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            for exp_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                if exp_id not in movie_expids:
                    continue
                exp_meta = experiments.get(exp_id)
                if exp_meta is None:
                    continue
                if not exp_meta.get("trial_rows") or not exp_meta.get("trial_meta"):
                    skip_counts["missing_trial_data"] += 1
                    continue
                roi_index = as_int(d_obs.get("local_ids", {}).get("conversion_index"))
                if roi_index is None:
                    skip_counts["missing_roi_index"] += 1
                    continue
                jobs.append((animal_id, str(global_dendrite_id), dendrite_record, d_obs, exp_id))
    total_jobs = len(jobs)
    with step_scope("direct trial-type trial table", total=total_jobs if total_jobs else None):
        for idx, (animal_id, global_dendrite_id, dendrite_record, d_obs, exp_id) in enumerate(jobs, start=1):
            if total_jobs:
                step_progress(idx, total_jobs, label=f"{animal_id} | {global_dendrite_id} | {exp_id}")
            exp_meta = experiments.get(exp_id)
            if exp_meta is None:
                skip_counts["missing_experiment"] += 1
                continue
            cut_dir = Path(str(exp_meta.get("source_paths", {}).get("cut") or ""))
            cut_path = cut_dir / f"s2p_ch{int(cache.get('config', {}).get('channel', DEFAULT_CHANNEL))}_dF_cut.pickle"
            if exp_id not in cut_cache:
                if not cut_path.exists():
                    skip_counts["missing_cut_bundle"] += 1
                    continue
                try:
                    _cut_time, cut_arr, _ = extract_cut_neural_bundle(cut_path)
                    cut_arr = np.asarray(cut_arr, dtype=float)
                    finite_mask = np.isfinite(cut_arr)
                    cut_counts = finite_mask.sum(axis=2)
                    cut_sums = np.where(finite_mask, cut_arr, 0.0).sum(axis=2)
                    cut_means = np.divide(
                        cut_sums,
                        cut_counts,
                        out=np.full(cut_sums.shape, np.nan, dtype=float),
                        where=cut_counts > 0,
                    )
                    cut_cache[exp_id] = (cut_means, cut_counts)
                except Exception:
                    skip_counts["unreadable_cut_bundle"] += 1
                    continue
            cut_means, cut_counts = cut_cache[exp_id]
            roi_index = as_int(d_obs.get("local_ids", {}).get("conversion_index"))
            if roi_index is None:
                skip_counts["missing_roi_index"] += 1
                continue
            trial_rows = trial_rows_cache.get(exp_id)
            if trial_rows is None:
                trial_rows = list(exp_meta.get("trial_rows", []) or [])
                trial_rows_cache[exp_id] = trial_rows
            trial_meta = trial_meta_cache.get(exp_id)
            if trial_meta is None:
                trial_meta = [
                    meta
                    for meta in exp_meta.get("trial_meta", []) or []
                    if isinstance(meta, dict) and meta.get("trial_index") is not None and meta.get("state_label") is not None
                ]
                trial_meta_cache[exp_id] = trial_meta
            if not trial_meta:
                skip_counts["missing_trial_meta"] += 1
                continue
            compartment = observation_compartment(cache, exp_id, d_obs)
            for meta in trial_meta:
                state_label = str(meta.get("state_label"))
                category = str(meta.get("category") or "")
                if state_label not in state_order:
                    skip_counts["excluded_state"] += 1
                    continue
                if movie_trial_types and category not in movie_trial_types:
                    skip_counts["excluded_movie_trial_type"] += 1
                    continue
                trial_index = as_int(meta.get("trial_index"))
                if trial_index is None or trial_index < 0 or trial_index >= len(trial_rows) or trial_index >= cut_means.shape[1]:
                    skip_counts["invalid_trial_index"] += 1
                    continue
                if roi_index < 0 or roi_index >= cut_means.shape[0]:
                    skip_counts["missing_roi_index"] += 1
                    continue
                trial_row = trial_rows[trial_index]
                video_name = str(trial_row.get("F1_name") or "")
                video_id = normalize_movie_clip_id(video_name)
                trial_mean_response = float(cut_means[roi_index, trial_index])
                n_frames = int(cut_counts[roi_index, trial_index])
                if not np.isfinite(trial_mean_response):
                    skip_counts["empty_trial_response"] += 1
                    continue
                candidate_rows.append(
                    {
                        "analysis": "direct_trial_type_comparison",
                        "animal_id": animal_id,
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "date": exp_meta.get("date"),
                        "global_dendrite_id": global_dendrite_id,
                        "roi_index": roi_index,
                        "compartment": compartment,
                        "state": state_label,
                        "state_order": state_order[state_label],
                        "movie_trial_type": category,
                        "video_id": video_id,
                        "video_name": video_name,
                        "trial_index": trial_index,
                        "trial_mean_response": float(trial_mean_response),
                        "n_frames": int(n_frames),
                        "status": "ok",
                    }
                )
    animal_video_state_presence: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in candidate_rows:
        animal_video_state_presence[(str(row.get("animal_id")), str(row.get("video_id")))].add(str(row.get("state")))
    included_groups = {key for key, states in animal_video_state_presence.items() if len(states) >= 2}
    filtered_rows = [row for row in candidate_rows if (str(row.get("animal_id")), str(row.get("video_id"))) in included_groups]
    excluded_trial_rows = len(candidate_rows) - len(filtered_rows)
    skip_counts["single_state_video_groups"] = int(sum(1 for states in animal_video_state_presence.values() if len(states) < 2))
    skip_counts["excluded_trial_rows_single_state_video"] = int(excluded_trial_rows)
    table_checks = {
        "n_rows": int(len(filtered_rows)),
        "n_candidate_rows": int(len(candidate_rows)),
        "n_animals": int(len({str(row.get("animal_id")) for row in filtered_rows})),
        "n_videos": int(len({str(row.get("video_id")) for row in filtered_rows})),
        "n_pairable_video_groups": int(len(included_groups)),
        "skip_counts": dict(skip_counts),
        "state_labels": list(state_labels),
        "movie_trial_types": list(movie_trial_types),
    }
    return filtered_rows, table_checks
def _direct_trial_type_split_group_value(row: Mapping[str, Any]) -> str | None:
    split_group = str(row.get("split_group") or "").strip()
    return split_group or None


def _direct_trial_type_split_group_order(rows: Sequence[Mapping[str, Any]]) -> List[str | None]:
    split_groups: List[str | None] = []
    has_unassigned = False
    for row in rows or []:
        split_group = _direct_trial_type_split_group_value(row)
        if split_group is None:
            has_unassigned = True
            continue
        if split_group not in split_groups:
            split_groups.append(split_group)
    if not split_groups:
        return [None]
    if has_unassigned:
        split_groups.append(None)
    return split_groups


def summarize_direct_trial_type_table(
    rows: Sequence[Dict[str, Any]],
    state_labels: Sequence[str],
    shuffle_n: int,
    split_group_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    state_labels = [str(state) for state in state_labels if str(state)]
    state_order = {state: idx for idx, state in enumerate(state_labels)}
    has_split_groups = any(_direct_trial_type_split_group_value(row) is not None for row in rows)
    split_group_order = list(split_group_order) if split_group_order else _direct_trial_type_split_group_order(rows)
    if has_split_groups and any(_direct_trial_type_split_group_value(row) is None for row in rows) and None not in split_group_order:
        split_group_order = list(split_group_order) + [None]
    split_order_lookup = {split_group: idx for idx, split_group in enumerate(split_group_order)}
    split_meta: Dict[str, Dict[str, Any]] = {}
    if has_split_groups:
        for row in rows:
            split_group = _direct_trial_type_split_group_value(row)
            if split_group is None:
                continue
            payload = split_meta.setdefault(split_group, {"split_group": split_group})
            display = str(row.get("split_group_display") or split_group).strip() or split_group
            if display and "split_group_display" not in payload:
                payload["split_group_display"] = display
            color = str(row.get("split_group_color") or "").strip()
            if color and "split_group_color" not in payload:
                payload["split_group_color"] = color
            rank_value = row.get("split_group_rank")
            try:
                rank_float = float(rank_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(rank_float):
                continue
            current_rank = payload.get("split_group_rank")
            if current_rank is None or rank_float < float(current_rank):
                payload["split_group_rank"] = rank_float
    animal_video_state_groups: Dict[Tuple[str, str, str, str | None], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split_group = _direct_trial_type_split_group_value(row) if has_split_groups else None
        animal_video_state_groups[(str(row.get("animal_id")), str(row.get("video_id")), str(row.get("state")), split_group)].append(dict(row))
    animal_video_state_rows: List[Dict[str, Any]] = []
    for (animal_id, video_id, state, split_group), group_rows in sorted(
        animal_video_state_groups.items(),
        key=lambda item: (
            split_order_lookup.get(item[0][3], len(split_group_order)),
            item[0][0],
            item[0][1],
            state_order.get(item[0][2], 999),
        ),
    ):
        values = np.asarray([as_float(row.get("trial_mean_response")) for row in group_rows if np.isfinite(as_float(row.get("trial_mean_response")))], dtype=float)
        if values.size == 0:
            continue
        payload = {
            "animal_id": animal_id,
            "video_id": video_id,
            "state": state,
            "state_display": state_display_label(state),
            "state_color": state_display_color(state),
            "mean_response": float(np.nanmean(values)),
            "median_response": float(np.nanmedian(values)),
            "sem_response": float(np.nanstd(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan"),
            "n_rows": int(len(group_rows)),
            "n_trials": int(len({(str(row.get("exp_id")), int(as_int(row.get("trial_index")) or -1)) for row in group_rows})),
            "n_dendrite_observations": int(len({str(row.get("global_dendrite_id")) for row in group_rows})),
            "n_days": int(len({str(row.get("day_id")) for row in group_rows})),
            "video_name": str(group_rows[0].get("video_name") or ""),
            "movie_trial_type": str(group_rows[0].get("movie_trial_type") or ""),
            "compartment": str(group_rows[0].get("compartment") or "unknown"),
        }
        if has_split_groups:
            payload["split_group"] = split_group
            payload["split_group_display"] = None
            payload["split_group_color"] = None
            payload["split_group_rank"] = None
            if split_group is not None:
                split_group_meta = split_meta.get(split_group, {})
                if split_group_meta.get("split_group_display") is not None:
                    payload["split_group_display"] = split_group_meta.get("split_group_display")
                if split_group_meta.get("split_group_color") is not None:
                    payload["split_group_color"] = split_group_meta.get("split_group_color")
                if split_group_meta.get("split_group_rank") is not None:
                    payload["split_group_rank"] = split_group_meta.get("split_group_rank")
        animal_video_state_rows.append(payload)
    video_state_groups: Dict[Tuple[str, str, str, str | None], List[Dict[str, Any]]] = defaultdict(list)
    for row in animal_video_state_rows:
        video_state_groups[(str(row.get("video_id")), str(row.get("state")), str(row.get("split_group")) if has_split_groups else None)].append(dict(row))
    video_state_rows: List[Dict[str, Any]] = []
    for (video_id, state, split_group_key), group_rows in sorted(
        video_state_groups.items(),
        key=lambda item: (
            split_order_lookup.get(item[0][2] if has_split_groups else None, len(split_group_order)),
            item[0][0],
            state_order.get(item[0][1], 999),
        ),
    ):
        values = np.asarray([value for value in (as_float(row.get("mean_response")) for row in group_rows) if value is not None], dtype=float)
        if values.size == 0:
            continue
        payload = {
            "video_id": video_id,
            "state": state,
            "state_display": state_display_label(state),
            "state_color": state_display_color(state),
            "mean_response": float(np.nanmean(values)),
            "median_response": float(np.nanmedian(values)),
            "sem_response": float(np.nanstd(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan"),
            "n_animals": int(len({str(row.get("animal_id")) for row in group_rows})),
            "n_animal_video_rows": int(len(group_rows)),
            "animal_ids": sorted({str(row.get("animal_id")) for row in group_rows}),
            "movie_trial_type": str(group_rows[0].get("movie_trial_type") or ""),
        }
        if has_split_groups:
            payload["split_group"] = split_group_key if split_group_key not in {"None", "", None} else None
            payload["split_group_display"] = None
            payload["split_group_color"] = None
            payload["split_group_rank"] = None
            if payload["split_group"] is not None:
                split_group_meta = split_meta.get(str(payload["split_group"]), {})
                if split_group_meta.get("split_group_display") is not None:
                    payload["split_group_display"] = split_group_meta.get("split_group_display")
                if split_group_meta.get("split_group_color") is not None:
                    payload["split_group_color"] = split_group_meta.get("split_group_color")
                if split_group_meta.get("split_group_rank") is not None:
                    payload["split_group_rank"] = split_group_meta.get("split_group_rank")
        video_state_rows.append(payload)
    state_summary_rows: List[Dict[str, Any]] = []
    state_summary_split_groups = split_group_order if has_split_groups else [None]
    for split_group in state_summary_split_groups:
        split_rows = [row for row in video_state_rows if (_direct_trial_type_split_group_value(row) if has_split_groups else None) == split_group] if has_split_groups else list(video_state_rows)
        if not split_rows:
            continue
        for state in state_labels:
            state_rows = [row for row in split_rows if str(row.get("state")) == state]
            values = np.asarray([value for value in (as_float(row.get("mean_response")) for row in state_rows) if value is not None], dtype=float)
            if values.size == 0:
                continue
            payload = {
                "state": state,
                "state_display": state_display_label(state),
                "state_color": state_display_color(state),
                "tested_videos": int(values.size),
                "mean_response": float(np.nanmean(values)),
                "median_response": float(np.nanmedian(values)),
                "sem_response": float(np.nanstd(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan"),
                "positive_fraction": float(np.mean(values > 0.0)),
                "n_animals": int(len({animal_id for row in state_rows for animal_id in row.get("animal_ids", [])})),
            }
            if has_split_groups:
                payload["split_group"] = split_group
                payload["split_group_display"] = None
                payload["split_group_color"] = None
                payload["split_group_rank"] = None
                if split_group is not None:
                    split_group_meta = split_meta.get(split_group, {})
                    if split_group_meta.get("split_group_display") is not None:
                        payload["split_group_display"] = split_group_meta.get("split_group_display")
                    if split_group_meta.get("split_group_color") is not None:
                        payload["split_group_color"] = split_group_meta.get("split_group_color")
                    if split_group_meta.get("split_group_rank") is not None:
                        payload["split_group_rank"] = split_group_meta.get("split_group_rank")
            state_summary_rows.append(payload)
    pair_rows: List[Dict[str, Any]] = []
    pair_group_order = split_group_order if has_split_groups else [None]
    for split_group in pair_group_order:
        split_rows = [row for row in video_state_rows if (_direct_trial_type_split_group_value(row) if has_split_groups else None) == split_group] if has_split_groups else list(video_state_rows)
        if not split_rows:
            continue
        values_by_state: Dict[str, Dict[str, List[float]]] = {state: {} for state in state_labels}
        for row in split_rows:
            state = str(row.get("state"))
            video_id = str(row.get("video_id"))
            values_by_state.setdefault(state, {})[video_id] = [float(row.get("mean_response"))]
        for state_a, state_b in combinations(state_labels, 2):
            comparison = paired_comparison(values_by_state, state_a, state_b, "direct_trial_type", shuffle_n)
            if comparison.get("subjects"):
                a_values = np.asarray([values_by_state[state_a][subject][0] for subject in comparison["subjects"]], dtype=float)
                b_values = np.asarray([values_by_state[state_b][subject][0] for subject in comparison["subjects"]], dtype=float)
                if a_values.size >= 2 and np.nanstd(a_values) > 0 and np.nanstd(b_values) > 0:
                    agreement_r = float(stats.pearsonr(a_values, b_values).statistic)
                else:
                    agreement_r = float("nan")
                comparison["agreement_r"] = agreement_r
                comparison["agreement_n"] = int(a_values.size)
                comparison["agreement_p"] = float(stats.pearsonr(a_values, b_values).pvalue) if a_values.size >= 2 and np.nanstd(a_values) > 0 and np.nanstd(b_values) > 0 else float("nan")
            comparison["analysis"] = "direct_trial_type_comparison"
            comparison["comparison"] = "state_pair"
            comparison["n_videos"] = int(comparison.get("n_subjects", 0))
            comparison["video_ids"] = list(comparison.get("subjects", []))
            if has_split_groups:
                comparison["split_group"] = split_group
                comparison["split_group_display"] = None
                comparison["split_group_color"] = None
                comparison["split_group_rank"] = None
                if split_group is not None:
                    split_group_meta = split_meta.get(split_group, {})
                    if split_group_meta.get("split_group_display") is not None:
                        comparison["split_group_display"] = split_group_meta.get("split_group_display")
                    if split_group_meta.get("split_group_color") is not None:
                        comparison["split_group_color"] = split_group_meta.get("split_group_color")
                    if split_group_meta.get("split_group_rank") is not None:
                        comparison["split_group_rank"] = split_group_meta.get("split_group_rank")
            pair_rows.append(comparison)
    pairable_videos = sorted({(str(row.get("split_group")) if has_split_groups else "", str(row.get("video_id"))) for row in video_state_rows if str(row.get("video_id"))})
    tested_animals = sorted({str(row.get("animal_id")) for row in animal_video_state_rows if str(row.get("animal_id"))})
    state_pair_count = len(pair_rows)
    significant_pairs = sum(1 for row in pair_rows if is_significant_row(row, p_key="shuffle_p"))
    _effect_sizes = [value for value in (as_float(row.get("effect_size")) for row in pair_rows) if value is not None]
    mean_effect_size = float(np.nanmean(_effect_sizes)) if _effect_sizes else float("nan")
    _agreement_rs = [value for value in (as_float(row.get("agreement_r")) for row in pair_rows) if value is not None]
    mean_agreement_r = float(np.nanmean(_agreement_rs)) if _agreement_rs else float("nan")
    overall_summary_rows = [
        {
            "tested_trial_rows": int(len(rows)),
            "tested_animal_video_state_rows": int(len(animal_video_state_rows)),
            "tested_video_state_rows": int(len(video_state_rows)),
            "tested_videos": int(len(pairable_videos)),
            "tested_animals": int(len(tested_animals)),
            "tested_state_pairs": int(state_pair_count),
            "significant_state_pairs": int(significant_pairs),
            "mean_effect_size": float(mean_effect_size),
            "mean_agreement_r": float(mean_agreement_r),
            "state_labels": list(state_labels),
        }
    ]
    if has_split_groups:
        overall_summary_rows[0]["n_split_groups"] = int(len([split_group for split_group in split_group_order if split_group is not None]))
        overall_summary_rows[0]["split_groups"] = [split_group for split_group in split_group_order if split_group is not None]
    return {
        "animal_video_state_rows": animal_video_state_rows,
        "video_state_rows": video_state_rows,
        "state_summary_rows": state_summary_rows,
        "state_pair_rows": pair_rows,
        "overall_summary_rows": overall_summary_rows,
    }


def run_direct_trial_type_comparison(
    cache: Dict[str, Any],
    state_comparison_states: Optional[Sequence[str]],
    shuffle_n: int,
    split_membership_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    selection = cache.get("config", {}) if isinstance(cache.get("config", {}), dict) else {}
    selected_states = [str(state) for state in (state_comparison_states or []) if str(state) in MOVIE_STATE_LABELS]
    if not selected_states:
        return {
            "available": False,
            "alerts": ["[ALERT] direct trial-type comparison was skipped because no movie state labels were selected."],
            "selection": {"state_labels": [], "movie_trial_types": []},
            "table_checks": {"n_rows": 0, "n_candidate_rows": 0, "skip_counts": {"no_movie_state_labels": 1}},
            "table_rows": [],
            "animal_video_state_rows": [],
            "video_state_rows": [],
            "state_summary_rows": [],
            "state_pair_rows": [],
            "overall_summary_rows": [],
        }
    rebuild_requested = bool(cache.get("config", {}).get("analysis_tables_rebuild")) or bool(cache.get("config", {}).get("rebuild"))
    table_meta = {
        "analysis": "direct_trial_type_comparison",
        "state_labels": list(selected_states),
        "movie_trial_types": list(infer_movie_trial_types_from_states(selected_states)),
        "analysis_unit": str(cache.get("analysis_unit", "day")),
        "source_config_hash": str(cache.get("config_hash", "")),
    }
    cached_table = load_cached_analysis_table(cache, "direct_trial_type_table", expected_meta=table_meta, rebuild=rebuild_requested)
    if cached_table is not None:
        table_rows = list(cached_table.get("table_rows", []))
        table_checks = dict(cached_table.get("table_checks", {}))
    else:
        table_rows, table_checks = build_direct_trial_type_comparison_table(cache, selected_states)
        store_cached_analysis_table(cache, "direct_trial_type_table", table_rows, table_checks, meta=table_meta)
    split_group_order: List[str | None] = []
    if split_membership_rows:
        split_group_order, _, _ = _split_group_metadata_from_membership_rows(split_membership_rows)
        table_rows = annotate_rows_with_split_group(table_rows, split_membership_rows)
    if not split_group_order:
        split_group_order = _direct_trial_type_split_group_order(table_rows)
    summary = summarize_direct_trial_type_table(table_rows, selected_states, shuffle_n, split_group_order=split_group_order)
    state_pair_rows = list(summary.get("state_pair_rows", []))
    video_state_rows = list(summary.get("video_state_rows", []))
    animal_video_state_rows = list(summary.get("animal_video_state_rows", []))
    state_summary_rows = list(summary.get("state_summary_rows", []))
    overall_summary_rows = list(summary.get("overall_summary_rows", []))
    direct_trial_type_states = list(table_checks.get("state_labels", selected_states))
    direct_movie_trial_types = infer_movie_trial_types_from_states(direct_trial_type_states)
    if any(split_group is not None for split_group in split_group_order):
        table_checks = dict(table_checks)
        table_checks["split_groups"] = [split_group for split_group in split_group_order if split_group is not None]
        table_checks["n_split_groups"] = int(len([split_group for split_group in split_group_order if split_group is not None]))
    return {
        "available": bool(table_rows) or bool(state_pair_rows),
        "alerts": [],
        "selection": {
            "state_labels": direct_trial_type_states,
            "movie_trial_types": direct_movie_trial_types,
            "state_mode": cache.get("config", {}).get("state_mode"),
        },
        "table_checks": table_checks,
        "table_rows": table_rows,
        "animal_video_state_rows": animal_video_state_rows,
        "video_state_rows": video_state_rows,
        "state_summary_rows": state_summary_rows,
        "state_pair_rows": state_pair_rows,
        "overall_summary_rows": overall_summary_rows,
    }
def build_mixed_model_table(cache: Dict[str, Any], source_cache: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Turn the normalized cache into one long dendrite-state table that the mixed models can read directly.
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    table_rows: List[Dict[str, Any]] = []
    spine_mean_errors: List[float] = []
    visual_response_summary = classify_visual_responsive_dendrites(cache, source_cache=source_cache)
    visual_response_lookup = {
        str(row.get("global_dendrite_id")): str(row.get("cohort") or "nonresponsive")
        for row in visual_response_summary.get("rows", [])
        if isinstance(row, dict) and str(row.get("global_dendrite_id") or "")
    }
    for animal_id in sorted(animals):
        animal_entry = animals[animal_id]
        for global_dend_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
            for day_id, d_obs in sorted(dendrite_record.get("observations", {}).items()):
                exp_meta = experiments.get(day_id)
                if exp_meta is None:
                    continue
                exp_time = np.asarray(d_obs.get("time"), dtype=float)
                sampling_rate = estimate_sampling_rate(exp_time)
                state_family = str(exp_meta.get("day_family") or "pooled")
                wheel_interp = np.asarray(exp_meta.get("wheel", {}).get("interpolated"), dtype=float)
                pupil_series = np.asarray(exp_meta.get("pupil", {}).get("series"), dtype=float)
                for state_label in ALL_REQUESTED_STATES:
                    mask = exp_meta.get("state_masks", {}).get(state_label)
                    if mask is None:
                        continue
                    mask = np.asarray(mask, dtype=bool)
                    if mask.shape != exp_time.shape or not np.any(mask):
                        continue
                    dend_values = values_from_observation(d_obs["trace"], mask)
                    dend_values = dend_values[np.isfinite(dend_values)]
                    if dend_values.size == 0:
                        continue
                    mean_dendrite_activity = float(np.nanmean(dend_values))
                    spine_means: List[float] = []
                    for spine_id in d_obs.get("spine_ids", []):
                        s_obs = dendrite_record.get("spines", {}).get(spine_id, {}).get("observations", {}).get(day_id)
                        if s_obs is None:
                            continue
                        spine_values = values_from_observation(s_obs["spine_specific"], mask)
                        spine_values = spine_values[np.isfinite(spine_values)]
                        if spine_values.size:
                            spine_means.append(float(np.nanmean(spine_values)))
                    mean_spine_activity_per_dendrite = float(np.nanmean(spine_means)) if spine_means else float("nan")
                    if spine_means and np.isfinite(mean_spine_activity_per_dendrite):
                        spine_mean_errors.append(float(abs(mean_spine_activity_per_dendrite - float(np.nanmean(spine_means)))))
                    dendrite_event_info = build_state_masked_event_info(d_obs.get("trace"), exp_time, mask, d_obs.get("event_info"))
                    dendrite_event_frequency_per_min = as_float(dendrite_event_info.get("event_frequency_per_min"))
                    spine_event_frequencies: List[float] = []
                    coincident_event_frequencies: List[float] = []
                    noncoincident_event_frequencies: List[float] = []
                    for spine_id in d_obs.get("spine_ids", []):
                        s_obs = dendrite_record.get("spines", {}).get(spine_id, {}).get("observations", {}).get(day_id)
                        if s_obs is None:
                            continue
                        spine_full_info = s_obs.get("event_info") or {}
                        spine_event_info = build_state_masked_event_info(
                            s_obs.get("trace"),
                            exp_time,
                            mask,
                            spine_full_info,
                        )
                        spine_freq = as_float(spine_event_info.get("spine_event_frequency_per_min", spine_event_info.get("event_frequency_per_min")))
                        if spine_freq is not None and np.isfinite(spine_freq):
                            spine_event_frequencies.append(float(spine_freq))
                        annotated_event_info = annotate_spine_event_info(spine_event_info, dendrite_event_info)
                        coincident_freq = as_float(annotated_event_info.get("coincident_event_frequency_per_min"))
                        noncoincident_freq = as_float(annotated_event_info.get("noncoincident_event_frequency_per_min"))
                        if coincident_freq is not None and np.isfinite(coincident_freq):
                            coincident_event_frequencies.append(float(coincident_freq))
                        if noncoincident_freq is not None and np.isfinite(noncoincident_freq):
                            noncoincident_event_frequencies.append(float(noncoincident_freq))
                    dendrite_event_frequency_per_min = float(dendrite_event_frequency_per_min) if dendrite_event_frequency_per_min is not None and np.isfinite(dendrite_event_frequency_per_min) else float("nan")
                    spine_event_frequency_per_min = float(np.nanmean(spine_event_frequencies)) if spine_event_frequencies else float("nan")
                    coincident_event_frequency_per_min = float(np.nanmean(coincident_event_frequencies)) if coincident_event_frequencies else float("nan")
                    noncoincident_event_frequency_per_min = float(np.nanmean(noncoincident_event_frequencies)) if noncoincident_event_frequencies else float("nan")
                    locomotion_mean = float("nan")
                    if wheel_interp is not None and wheel_interp.size == exp_time.size:
                        locomotion_values = np.abs(np.asarray(wheel_interp, dtype=float)[mask])
                        locomotion_values = locomotion_values[np.isfinite(locomotion_values)]
                        if locomotion_values.size:
                            locomotion_mean = float(np.nanmean(locomotion_values))
                    pupil_mean = float("nan")
                    if pupil_series is not None and pupil_series.size == exp_time.size:
                        pupil_values = np.asarray(pupil_series, dtype=float)[mask]
                        pupil_values = pupil_values[np.isfinite(pupil_values)]
                        if pupil_values.size:
                            pupil_mean = float(np.nanmean(pupil_values))
                    n_frames = int(np.count_nonzero(mask))
                    state_duration_s = float(n_frames / sampling_rate) if sampling_rate is not None and sampling_rate > 0 else float("nan")
                    table_rows.append(
                        {
                            "animal_id": animal_id,
                            "exp_id": day_id,
                            "day_id": day_id,
                            "day_pair_id": str(d_obs.get("day_pair_id") or exp_meta.get("day_pair_id") or make_day_id(animal_id, str(exp_meta.get("date")), "paired")),
                            "date": exp_meta.get("date"),
                            "global_dendrite_id": global_dend_id,
                            "compartment": observation_compartment(cache, day_id, d_obs),
                            "state": state_label,
                            "visual_response_classifier_type": "dendrite",
                            "visual_response_cohort": visual_response_lookup.get(str(global_dend_id), "nonresponsive"),
                            "visual_response_category": visual_response_lookup.get(str(global_dend_id), "nonresponsive"),
                            "state_family": state_family,
                            "state_duration_s": state_duration_s,
                            "state_n_frames": n_frames,
                            "mean_dendrite_activity": mean_dendrite_activity,
                            "mean_spine_activity_per_dendrite": mean_spine_activity_per_dendrite,
                            "dendrite_event_frequency_per_min": dendrite_event_frequency_per_min,
                            "spine_event_frequency_per_min": spine_event_frequency_per_min,
                            "coincident_event_frequency_per_min": coincident_event_frequency_per_min,
                            "noncoincident_event_frequency_per_min": noncoincident_event_frequency_per_min,
                            "n_spines": int(len(spine_means)),
                            "locomotion_mean": locomotion_mean,
                            "pupil_mean": pupil_mean,
                        }
                    )
    checks = {
        "n_rows": int(len(table_rows)),
        "mean_spine_activity_per_dendrite_max_abs_error": float(np.nanmax(spine_mean_errors)) if spine_mean_errors else float("nan"),
    }
    return table_rows, checks
def build_mixed_model_design(
    rows: List[Dict[str, Any]],
    response: str,
    scope: str,
    state_order: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    if scope not in {"all_state", "selected_state"}:
        raise ValueError(f"Unknown mixed-model scope: {scope}")
    def _is_finite_numeric(value: Any) -> bool:
        coerced = as_float(value)
        return coerced is not None and bool(np.isfinite(coerced))
    working_rows = list(rows)
    working_rows = [row for row in working_rows if _is_finite_numeric(row.get(response))]
    if not working_rows:
        return None
    ordered_states = [canonical_state_label(state) for state in (state_order if state_order is not None else ALL_REQUESTED_STATES)]
    ordered_states = list(dict.fromkeys(state for state in ordered_states if str(state).strip()))
    state_levels = [state for state in ordered_states if any(canonical_state_label(row.get("state")) == state for row in working_rows)]
    if not state_levels:
        return None
    if "quiet_awake_blank" in state_levels and state_levels[0] != "quiet_awake_blank":
        state_levels = ["quiet_awake_blank"] + [state for state in state_levels if state != "quiet_awake_blank"]
    state_reference = "quiet_awake_blank" if "quiet_awake_blank" in state_levels else ("quiet_awake" if "quiet_awake" in state_levels else state_levels[0])
    compartment_levels = ordered_compartment_levels([row.get("compartment") for row in working_rows])
    compartment_reference: Optional[str] = None
    include_compartment = len(compartment_levels) >= 2 and any(level in {"basal", "apical"} for level in compartment_levels)
    interaction_compartments: List[str] = []
    if include_compartment:
        compartment_reference = "basal" if "basal" in compartment_levels else compartment_levels[0]
        interaction_compartments = [compartment for compartment in compartment_levels if compartment != compartment_reference]
    visual_response_levels = [str(value) for value in dict.fromkeys(str(row.get("visual_response_cohort") or "nonresponsive") for row in working_rows)]
    visual_response_reference: Optional[str] = None
    include_visual_response = len(visual_response_levels) >= 2
    if include_visual_response:
        visual_response_reference = "nonresponsive" if "nonresponsive" in visual_response_levels else visual_response_levels[0]

    split_group_levels = [str(value) for value in dict.fromkeys(str(row.get("split_group") or "") for row in working_rows) if str(value).strip()]
    split_group_reference: Optional[str] = None
    include_split_group = len(split_group_levels) >= 2
    interaction_split_groups: List[str] = []
    split_group_labels: Dict[str, str] = {}
    split_group_colors: Dict[str, str] = {}
    if include_split_group:
        split_group_rank: Dict[str, float] = {}
        for row in working_rows:
            split_group = str(row.get("split_group") or "").strip()
            if not split_group:
                continue
            label = str(row.get("split_group_display") or row.get("group_display") or split_group).strip()
            if label and split_group not in split_group_labels:
                split_group_labels[split_group] = label
            color = str(row.get("split_group_color") or row.get("group_color") or "").strip()
            if color and split_group not in split_group_colors:
                split_group_colors[split_group] = color
            rank_value = row.get("split_group_rank")
            try:
                rank_float = float(rank_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(rank_float):
                continue
            current = split_group_rank.get(split_group)
            if current is None or rank_float < current:
                split_group_rank[split_group] = rank_float
        if split_group_rank:
            split_group_levels = sorted(split_group_levels, key=lambda group: (split_group_rank.get(group, float("inf")), group))
        split_group_reference = split_group_levels[0] if split_group_levels else None
        interaction_split_groups = [group for group in split_group_levels if group != split_group_reference]

    covariate_specs: List[Dict[str, Any]] = []
    if response == "mean_spine_activity_per_dendrite":
        raw = np.asarray([math.log1p(float(row.get("n_spines", 0) or 0.0)) for row in working_rows], dtype=float)
        finite = np.isfinite(raw)
        if finite.sum() >= 2:
            center = float(np.nanmean(raw[finite]))
            centered = np.asarray(raw - center, dtype=float)
            centered[~finite] = 0.0
            if np.nanstd(centered[finite]) > 0:
                covariate_specs.append({"name": "log1p_n_spines", "center": center, "values": centered})
    fixed_effect_names = ["Intercept"]
    fixed_effect_names.extend([f"state[{state}]" for state in state_levels[1:]])
    if include_compartment:
        fixed_effect_names.extend([f"compartment[{compartment}]" for compartment in interaction_compartments])
        for state in state_levels[1:]:
            fixed_effect_names.extend([f"state[{state}]:compartment[{compartment}]" for compartment in interaction_compartments])
    if include_split_group:
        fixed_effect_names.extend([f"split_group[{split_group}]" for split_group in interaction_split_groups])
        for state in state_levels[1:]:
            fixed_effect_names.extend([f"state[{state}]:split_group[{split_group}]" for split_group in interaction_split_groups])
    if include_visual_response:
        fixed_effect_names.extend([f"visual_response_cohort[{cohort}]" for cohort in visual_response_levels if cohort != visual_response_reference])
    fixed_effect_names.extend([spec["name"] for spec in covariate_specs])
    X_rows: List[List[float]] = []
    y: List[float] = []
    blocks: List[Tuple[str, str]] = []
    for index, row in enumerate(working_rows):
        y_value = as_float(row.get(response))
        if y_value is None or not np.isfinite(y_value):
            continue
        row_values: List[float] = [1.0]
        for state in state_levels[1:]:
            row_values.append(1.0 if row.get("state") == state else 0.0)
        if include_compartment:
            compartment_dummies: Dict[str, float] = {}
            for compartment in interaction_compartments:
                compartment_dummy = 1.0 if row.get("compartment") == compartment else 0.0
                compartment_dummies[compartment] = compartment_dummy
                row_values.append(compartment_dummy)
            for state in state_levels[1:]:
                state_dummy = 1.0 if row.get("state") == state else 0.0
                for compartment in interaction_compartments:
                    row_values.append(state_dummy * compartment_dummies[compartment])
        if include_split_group:
            split_group_value = str(row.get("split_group") or "").strip()
            split_group_dummies: Dict[str, float] = {}
            for split_group in interaction_split_groups:
                split_group_dummy = 1.0 if split_group_value == split_group else 0.0
                split_group_dummies[split_group] = split_group_dummy
                row_values.append(split_group_dummy)
            for state in state_levels[1:]:
                state_dummy = 1.0 if row.get("state") == state else 0.0
                for split_group in interaction_split_groups:
                    row_values.append(state_dummy * split_group_dummies[split_group])
        if include_visual_response:
            cohort_value = str(row.get("visual_response_cohort") or "nonresponsive")
            for cohort in visual_response_levels:
                if cohort == visual_response_reference:
                    continue
                row_values.append(1.0 if cohort_value == cohort else 0.0)
        for spec in covariate_specs:
            row_values.append(float(spec["values"][index]))
        X_rows.append(row_values)
        y.append(float(y_value))
        blocks.append((str(row.get("animal_id")), str(row.get("day_id") or row.get("exp_id"))))
    if not y:
        return None
    X = np.asarray(X_rows, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    group_labels = np.asarray([block[0] for block in blocks], dtype=object)
    design = {
        "response": response,
        "scope": scope,
        "rows": [working_rows[idx] for idx in range(len(working_rows)) if _is_finite_numeric(working_rows[idx].get(response))],
        "y": y_arr,
        "X": X,
        "fixed_effect_names": fixed_effect_names,
        "state_levels": state_levels,
        "state_reference": state_reference,
        "compartment_levels": compartment_levels,
        "compartment_reference": compartment_reference,
        "interaction_compartments": interaction_compartments,
        "include_compartment": include_compartment,
        "split_group_levels": split_group_levels,
        "split_group_reference": split_group_reference,
        "interaction_split_groups": interaction_split_groups,
        "split_group_labels": split_group_labels,
        "split_group_colors": split_group_colors,
        "include_split_group": include_split_group,
        "visual_response_levels": visual_response_levels,
        "visual_response_reference": visual_response_reference,
        "include_visual_response": include_visual_response,
        "covariate_specs": covariate_specs,
        "groups": group_labels,
        "blocks": blocks,
        "n_obs": int(y_arr.size),
        "n_animals": int(len({row[0] for row in blocks})),
        "n_sessions": int(len({row[1] for row in blocks})),
        "n_dendrites": int(len({str(row.get("global_dendrite_id")) for row in working_rows if _is_finite_numeric(row.get(response))})),
    }
    return design

def make_variance_component_dict(rows: List[Dict[str, Any]], level_key: str, group_key: str = "animal_id") -> Dict[str, np.ndarray]:
    component: Dict[str, np.ndarray] = {}
    grouped_rows: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_rows[str(row.get(group_key))].append(index)
    for group_value, indices in grouped_rows.items():
        levels = [str(rows[index].get(level_key)) for index in indices]
        unique_levels = sorted(set(levels))
        if not unique_levels:
            continue
        level_to_index = {level: idx for idx, level in enumerate(unique_levels)}
        matrix = np.zeros((len(indices), len(unique_levels)), dtype=float)
        for row_index, index in enumerate(indices):
            matrix[row_index, level_to_index[str(rows[index].get(level_key))]] = 1.0
        component[group_value] = matrix
    return component
@dataclass
class FixedEffectFallbackResult:
    fe_params: np.ndarray
    cov_matrix: np.ndarray
    converged: bool = True
    method_name: str = "ols_fallback"
    fallback_reason: str = ""
    def cov_params(self) -> np.ndarray:
        return self.cov_matrix
    @property
    def params(self) -> np.ndarray:
        return self.fe_params
def design_requires_fixed_effect_fallback(design: Dict[str, Any]) -> Optional[str]:
    X = np.asarray(design.get("X"), dtype=float)
    y = np.asarray(design.get("y"), dtype=float)
    if X.size == 0 or y.size == 0:
        return "empty design"
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return "invalid design matrix"
    if X.shape[0] <= X.shape[1]:
        return f"too few rows for {X.shape[1]} fixed-effect parameters"
    try:
        rank = int(np.linalg.matrix_rank(X))
    except Exception:
        rank = 0
    if rank < X.shape[1]:
        return f"fixed-effect matrix rank deficient ({rank}/{X.shape[1]})"
    groups = np.asarray(design.get("groups"), dtype=object)
    if np.unique(groups).size < 2:
        return "insufficient animal groups"
    return None
def build_fixed_effect_fallback_result(design: Dict[str, Any], reason: str) -> Dict[str, Any]:
    X = np.asarray(design["X"], dtype=float)
    y = np.asarray(design["y"], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    df_resid = max(int(y.size - np.linalg.matrix_rank(X)), 1)
    sigma2 = float(np.nansum(resid ** 2) / df_resid)
    cov_matrix = sigma2 * np.linalg.pinv(X.T @ X)
    result = FixedEffectFallbackResult(
        fe_params=np.asarray(beta, dtype=float),
        cov_matrix=np.asarray(cov_matrix, dtype=float),
        converged=True,
        method_name="ols_fallback",
        fallback_reason=reason,
    )
    return {
        "result": result,
        "fit_method": "ols_fallback",
        "converged": True,
        "warning_messages": [],
        "warning_count": 0,
        "fallback_reason": reason,
        "random_structure": {
            "include_animal": False,
            "include_session": False,
            "include_dendrite": False,
        },
        "random_structure_name": "ols_fallback",
        "design": design,
    }
def fit_mixedlm_fixed_structure(
    design: Dict[str, Any],
    include_session: bool,
    include_dendrite: bool,
    vc_level_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if MixedLM is None:
        raise RuntimeError("statsmodels is not available in this environment")
    exog_vc: Dict[str, Dict[str, np.ndarray]] = {}
    rows = design["rows"]
    if vc_level_keys is not None:
        for level_key in vc_level_keys:
            component = make_variance_component_dict(rows, str(level_key), group_key="animal_id")
            if component and any(matrix.shape[1] > 0 for matrix in component.values()):
                exog_vc[str(level_key)] = component
    else:
        if include_session:
            session_component = make_variance_component_dict(rows, "day_id")
            if session_component and any(matrix.shape[1] > 0 for matrix in session_component.values()):
                exog_vc["session"] = session_component
        if include_dendrite:
            dendrite_component = make_variance_component_dict(rows, "global_dendrite_id")
            if dendrite_component and any(matrix.shape[1] > 0 for matrix in dendrite_component.values()):
                exog_vc["dendrite"] = dendrite_component
    fallback_reason = design_requires_fixed_effect_fallback(design)
    if fallback_reason is not None:
        return build_fixed_effect_fallback_result(design, fallback_reason)
    fit_methods = ["lbfgs", "powell", "cg"]
    last_result = None
    last_method = None
    last_exception: Optional[Exception] = None
    for method in fit_methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = MixedLM(
                    endog=design["y"],
                    exog=design["X"],
                    groups=design["groups"],
                    exog_re=np.ones((design["y"].shape[0], 1), dtype=float),
                    exog_vc=exog_vc if exog_vc else None,
                )
                result = model.fit(reml=True, method=method, disp=False, maxiter=200)
            last_result = result
            last_method = method
            if bool(getattr(result, "converged", False)):
                break
        except Exception as exc:
            last_exception = exc
            continue
    if last_result is None:
        fallback_reason = str(last_exception) if last_exception is not None else "MixedLM fit returned no result"
        return build_fixed_effect_fallback_result(design, fallback_reason)
    return {
        "result": last_result,
        "fit_method": last_method,
        "converged": bool(getattr(last_result, "converged", False)),
        "warning_messages": [],
        "warning_count": 0,
        "random_structure": {
            "include_animal": True,
            "include_session": bool(include_session),
            "include_dendrite": bool(include_dendrite),
            "vc_level_keys": list(vc_level_keys) if vc_level_keys is not None else [],
        },
        "random_structure_name": f"mixedlm_{last_method}",
        "design": design,
    }
def fit_mixedlm_with_fallback(
    design: Dict[str, Any],
    alerts: Optional[List[str]] = None,
    vc_level_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    attempts = [
        ("animal+day+dendrite", True, True),
        ("animal+dendrite", False, True),
        ("animal_only", False, False),
    ]
    if MixedLM is None:
        message = "[ALERT] statsmodels is unavailable, so mixed-model fits were skipped."
        if alerts is not None:
            alerts.append(message)
        return {
            "status": "skipped",
            "alert": message,
            "design": design,
            "fit_result": None,
            "summary_rows": [],
            "random_structure": None,
            "converged": False,
            "fit_method": None,
        }
    last_exception: Optional[Exception] = None
    best_result: Optional[Dict[str, Any]] = None
    best_score: Tuple[int, int, int] = (-1, -10**9, -10**9)
    for structure_index, (structure_name, include_session, include_dendrite) in enumerate(attempts):
        try:
            attempt = fit_mixedlm_fixed_structure(design, include_session, include_dendrite, vc_level_keys=vc_level_keys)
            if attempt.get("fit_method") != "ols_fallback":
                attempt["random_structure_name"] = structure_name
            attempt["design"] = design
            score = (
                1 if attempt.get("converged", False) else 0,
                -int(attempt.get("warning_count", 0)),
                -int(structure_index),
            )
            if score > best_score:
                best_score = score
                best_result = attempt
        except Exception as exc:
            last_exception = exc
            if alerts is not None:
                alerts.append(f"[ALERT] MixedLM fit attempt failed for {design['response']} ({design['scope']}), structure {structure_name}: {exc}")
            continue
    if best_result is not None:
        if best_result.get("random_structure_name") == "ols_fallback" and alerts is not None:
            reason = str(best_result.get("fallback_reason", "mixed model failure"))
            alerts.append(
                f"[ALERT] MixedLM failed for {design['response']} ({design['scope']}); using a fixed-effect fallback instead ({reason})."
            )
        elif not best_result.get("converged", False) and alerts is not None:
            alerts.append(
                f"[ALERT] MixedLM did not fully converge for {design['response']} ({design['scope']}); using {best_result['random_structure_name']}."
            )
        return best_result
    return build_fixed_effect_fallback_result(
        design,
        f"all mixed-model attempts failed: {last_exception}" if last_exception is not None else "all mixed-model attempts failed",
    )
def mixed_model_design_row(design: Dict[str, Any], state: str, compartment: Optional[str] = None, split_group: Optional[str] = None) -> np.ndarray:
    row = np.zeros(len(design["fixed_effect_names"]), dtype=float)
    row[0] = 1.0
    index = 1
    for state_name in design["state_levels"][1:]:
        row[index] = 1.0 if state == state_name else 0.0
        index += 1
    if design.get("include_compartment"):
        interaction_compartments = list(design.get("interaction_compartments", []))
        compartment_dummies: Dict[str, float] = {}
        for compartment_name in interaction_compartments:
            compartment_dummy = 1.0 if compartment == compartment_name else 0.0
            compartment_dummies[compartment_name] = compartment_dummy
            row[index] = compartment_dummy
            index += 1
        for state_name in design["state_levels"][1:]:
            state_dummy = 1.0 if state == state_name else 0.0
            for compartment_name in interaction_compartments:
                row[index] = state_dummy * compartment_dummies[compartment_name]
                index += 1
    if design.get("include_split_group"):
        interaction_split_groups = list(design.get("interaction_split_groups", []))
        split_group_value = str(split_group if split_group is not None else design.get("split_group_reference") or "").strip()
        split_group_dummies: Dict[str, float] = {}
        for split_group_name in interaction_split_groups:
            split_group_dummy = 1.0 if split_group_value == split_group_name else 0.0
            split_group_dummies[split_group_name] = split_group_dummy
            row[index] = split_group_dummy
            index += 1
        for state_name in design["state_levels"][1:]:
            state_dummy = 1.0 if state == state_name else 0.0
            for split_group_name in interaction_split_groups:
                row[index] = state_dummy * split_group_dummies[split_group_name]
                index += 1
    if design.get("include_visual_response"):
        visual_response_reference = str(design.get("visual_response_reference") or "nonresponsive")
        for cohort in design.get("visual_response_levels", []):
            if cohort == visual_response_reference:
                continue
            row[index] = 1.0 if str(design.get("visual_response_cohort") or "nonresponsive") == cohort else 0.0
            index += 1
    for spec in design["covariate_specs"]:
        row[index] = 0.0
        index += 1
    return row

def mixed_model_equation_string(design: Dict[str, Any], fit_info: Dict[str, Any]) -> str:
    fixed_terms = ["state"]
    if design.get("include_compartment"):
        fixed_terms.extend(["compartment", "state:compartment"])
    if design.get("include_split_group"):
        fixed_terms.extend(["split_group", "state:split_group"])
    if design.get("include_visual_response"):
        fixed_terms.append("visual_response_cohort")
    fixed_terms.extend([spec["name"] for spec in design.get("covariate_specs", [])])
    random_bits: List[str] = []
    if fit_info.get("random_structure_name") == "ols_fallback":
        random_bits.append("random effects: none (OLS fallback)")
    else:
        random_structure = fit_info.get("random_structure") or {}
        if random_structure.get("include_animal", False):
            random_bits.append("(1 | animal_id)")
        if random_structure.get("include_session", False):
            random_bits.append("(1 | animal_id:day_id)")
        if random_structure.get("include_dendrite", False):
            random_bits.append("(1 | animal_id:global_dendrite_id)")
        for level_key in random_structure.get("vc_level_keys", []) or []:
            random_bits.append(f"(1 | {level_key})")
        if not random_bits:
            random_bits.append("random effects: none")
    random_desc = " + ".join(random_bits)
    response = str(design.get("response", "response"))
    fixed_desc = " + ".join(fixed_terms)
    reference_desc = []
    if design.get("state_reference") is not None:
        reference_desc.append(f"state_ref={design.get('state_reference')}")
    if design.get("compartment_reference") is not None:
        reference_desc.append(f"compartment_ref={design.get('compartment_reference')}")
    if design.get("split_group_reference") is not None:
        reference_desc.append(f"split_group_ref={design.get('split_group_reference')}")
    reference_text = f" ({', '.join(reference_desc)})" if reference_desc else ""
    return f"{response} ~ {fixed_desc} | {random_desc}{reference_text}"

def contrast_from_result(result: Any, design: Dict[str, Any], contrast_spec: Dict[str, Any]) -> Dict[str, Any]:
    beta = np.asarray(result.fe_params, dtype=float).ravel()
    cov = np.asarray(result.cov_params(), dtype=float)
    cov_fe = cov[: beta.size, : beta.size]
    if contrast_spec["kind"] == "state_pair":
        compartment_reference = design.get("compartment_reference") if design.get("include_compartment") else None
        row_a = mixed_model_design_row(design, str(contrast_spec["state_a"]), compartment_reference)
        row_b = mixed_model_design_row(design, str(contrast_spec["state_b"]), compartment_reference)
        contrast_row = row_a - row_b
        contrast_name = f"{contrast_spec['state_a']} - {contrast_spec['state_b']}"
    elif contrast_spec["kind"] == "split_group_pair":
        if not design.get("include_split_group"):
            raise ValueError("split_group_pair contrast requested without split-group factor")
        state = str(contrast_spec.get("state") or design.get("state_reference") or design.get("state_levels", [""])[0])
        compartment_reference = design.get("compartment_reference") if design.get("include_compartment") else None
        group_a = str(contrast_spec.get("group_a") or contrast_spec.get("split_group_a") or "")
        group_b = str(contrast_spec.get("group_b") or contrast_spec.get("split_group_b") or "")
        if not group_a or not group_b:
            raise ValueError("split_group_pair contrast requested without both group levels")
        row_a = mixed_model_design_row(design, state, compartment_reference, split_group=group_a)
        row_b = mixed_model_design_row(design, state, compartment_reference, split_group=group_b)
        contrast_row = row_a - row_b
        contrast_name = f"{group_a} - {group_b} @ {state}"
    elif contrast_spec["kind"] == "basal_apical":
        row_apical = mixed_model_design_row(design, str(contrast_spec["state"]), "apical")
        row_basal = mixed_model_design_row(design, str(contrast_spec["state"]), "basal")
        contrast_row = row_apical - row_basal
        contrast_name = f"{contrast_spec['state']} : apical - basal"
    elif contrast_spec["kind"] == "visual_response_cohort":
        if not design.get("include_visual_response"):
            raise ValueError("visual_response_cohort contrast requested without visual-response factor")
        contrast_row = np.zeros_like(beta)
        target_term = None
        for term in design.get("fixed_effect_names", []):
            if term.startswith("visual_response_cohort["):
                target_term = term
                break
        if target_term is None:
            raise ValueError("visual_response_cohort contrast requested but no cohort term was fitted")
        contrast_row[design["fixed_effect_names"].index(target_term)] = 1.0 if "responsive" in target_term else -1.0
        contrast_name = "responsive - nonresponsive"
    else:
        raise ValueError(f"Unknown contrast kind: {contrast_spec['kind']}")
    estimate = float(np.dot(contrast_row, beta))
    variance = float(np.dot(contrast_row, np.dot(cov_fe, contrast_row)))
    se = math.sqrt(variance) if np.isfinite(variance) and variance > 0 else float("nan")
    z_value = float(estimate / se) if np.isfinite(se) and se > 0 else float("nan")
    classical_p = float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan")
    return {
        "contrast_name": contrast_name,
        "estimate": estimate,
        "se": se,
        "z": z_value,
        "classical_p": classical_p,
    }
def shuffle_state_labels_within_blocks(rows: List[Dict[str, Any]], rng: np.random.Generator) -> List[Dict[str, Any]]:
    grouped_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[(str(row.get("animal_id")), str(row.get("day_id") or row.get("exp_id")))].append(index)
    shuffled = [dict(row) for row in rows]
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        states = [rows[index].get("state") for index in indices]
        permuted = list(rng.permutation(states))
        for index, state in zip(indices, permuted):
            shuffled[index]["state"] = state
    return shuffled
def summarize_mixed_model_result(
    fit_info: Dict[str, Any],
    design: Dict[str, Any],
) -> List[Dict[str, Any]]:
    result = fit_info.get("result")
    if result is None:
        return []
    beta = np.asarray(result.fe_params, dtype=float).ravel()
    cov = np.asarray(result.cov_params(), dtype=float)
    cov_fe = cov[: beta.size, : beta.size]
    rows: List[Dict[str, Any]] = []
    for index, term in enumerate(design["fixed_effect_names"]):
        estimate = float(beta[index])
        variance = float(cov_fe[index, index]) if index < cov_fe.shape[0] else float("nan")
        se = math.sqrt(variance) if np.isfinite(variance) and variance > 0 else float("nan")
        z_value = float(estimate / se) if np.isfinite(se) and se > 0 else float("nan")
        p_value = float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan")
        rows.append(
            {
                "response": design["response"],
                "scope": design["scope"],
                "model_name": f"{design['scope']}_{design['response']}",
                "term": term,
                "estimate": estimate,
                "se": se,
                "z": z_value,
                "p_value": p_value,
                "random_structure": fit_info.get("random_structure_name"),
                "fit_method": fit_info.get("fit_method"),
                "converged": bool(fit_info.get("converged", False)),
                "n_obs": int(design.get("n_obs", 0)),
                "n_animals": int(design.get("n_animals", 0)),
                "n_sessions": int(design.get("n_sessions", 0)),
                "n_dendrites": int(design.get("n_dendrites", 0)),
                "state_reference": design.get("state_reference"),
                "compartment_reference": design.get("compartment_reference"),
                "visual_response_reference": design.get("visual_response_reference"),
            }
        )
    return rows
def run_mixed_model_family(
    table_rows: List[Dict[str, Any]],
    response: str,
    scope: str,
    contrast_specs: List[Dict[str, Any]],
    shuffle_n: int,
    alerts: Optional[List[str]] = None,
    vc_level_keys: Optional[Sequence[str]] = None,
    state_order: Optional[Sequence[str]] = None,
    state_filter: Optional[Sequence[str]] = None,
    p_value_source: str = "classical",
) -> Dict[str, Any]:
    if scope not in {"all_state", "selected_state"}:
        raise ValueError(f"Unknown mixed-model scope: {scope}")
    requested_p_value_source = normalize_mixed_model_contrast_p_source(p_value_source)
    effective_p_value_source = requested_p_value_source
    if requested_p_value_source == "shuffle" and int(shuffle_n) <= 0:
        if alerts is not None:
            alerts.append(
                f"[ALERT] Mixed-model shuffle p-values were requested for {response} ({scope}) but shuffle_n <= 0; using classical p-values instead."
            )
        effective_p_value_source = "classical"
    working_rows = list(table_rows)
    if state_filter is not None:
        state_filter_set = {str(state).strip() for state in state_filter if state is not None and str(state).strip()}
        working_rows = [row for row in working_rows if str(row.get("state")) in state_filter_set]
    design = build_mixed_model_design(working_rows, response, scope, state_order=state_order)
    if design is None:
        return {
            "response": response,
            "scope": scope,
            "p_value_source": effective_p_value_source,
            "p_value_source_requested": requested_p_value_source,
            "summary_rows": [],
            "contrast_rows": [],
            "design": None,
            "fit": None,
            "equation": None,
            "tested_terms": [],
            "tested_contrasts": [],
        }
    valid_contrast_specs: List[Dict[str, Any]] = []
    for contrast_spec in contrast_specs:
        if contrast_spec["kind"] == "state_pair":
            state_a = str(contrast_spec.get("state_a"))
            state_b = str(contrast_spec.get("state_b"))
            missing = [state for state in [state_a, state_b] if state not in design["state_levels"]]
            if missing:
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Skipping mixed-model state contrast for {response} ({scope}) because the model does not contain: {', '.join(missing)}"
                    )
                continue
            valid_contrast_specs.append(contrast_spec)
        elif contrast_spec["kind"] == "basal_apical":
            if not design.get("include_compartment"):
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Skipping mixed-model basal/apical contrast for {response} ({scope}) because the model does not have both basal and apical rows."
                    )
                continue
            state = str(contrast_spec.get("state"))
            if state not in design["state_levels"]:
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Skipping mixed-model basal/apical contrast for {response} ({scope}) because the model does not contain state {state}."
                    )
                continue
            valid_contrast_specs.append(contrast_spec)
        elif contrast_spec["kind"] == "split_group_pair":
            if not design.get("include_split_group"):
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Skipping mixed-model split-group contrast for {response} ({scope}) because the model does not contain a split-group factor."
                    )
                continue
            group_a = str(contrast_spec.get("group_a") or contrast_spec.get("split_group_a") or "")
            group_b = str(contrast_spec.get("group_b") or contrast_spec.get("split_group_b") or "")
            missing_groups = [group for group in [group_a, group_b] if group and group not in design.get("split_group_levels", [])]
            if missing_groups:
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Skipping mixed-model split-group contrast for {response} ({scope}) because the model does not contain: {', '.join(missing_groups)}"
                    )
                continue
            state = str(contrast_spec.get("state") or "")
            if state and state not in design["state_levels"]:
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Skipping mixed-model split-group contrast for {response} ({scope}) because the model does not contain state {state}."
                    )
                continue
            valid_contrast_specs.append(contrast_spec)
        elif contrast_spec["kind"] == "visual_response_cohort":
            if not design.get("include_visual_response"):
                continue
            valid_contrast_specs.append(contrast_spec)
        else:
            valid_contrast_specs.append(contrast_spec)
    fit_info = fit_mixedlm_with_fallback(design, alerts=alerts, vc_level_keys=vc_level_keys)
    if fit_info.get("status") == "skipped" or fit_info.get("result") is None:
        return {
            "response": response,
            "scope": scope,
            "p_value_source": effective_p_value_source,
            "p_value_source_requested": requested_p_value_source,
            "summary_rows": [],
            "contrast_rows": [],
            "design": design,
            "fit": fit_info,
            "equation": mixed_model_equation_string(design, fit_info),
            "tested_terms": list(design.get("fixed_effect_names", [])),
            "tested_contrasts": [contrast_spec.get("kind", "unknown") for contrast_spec in valid_contrast_specs],
        }
    summary_rows = summarize_mixed_model_result(fit_info, design)
    contrast_rows: List[Dict[str, Any]] = []
    for contrast_spec in valid_contrast_specs:
        observed = contrast_from_result(fit_info["result"], design, contrast_spec)
        active_p = observed["classical_p"]
        shuffle_n_requested = int(shuffle_n) if effective_p_value_source == "shuffle" else 0
        shuffle_n_success = 0
        if effective_p_value_source == "shuffle":
            null_effects: List[float] = []
            rng = np.random.default_rng(12345)
            for _ in range(int(shuffle_n)):
                shuffled_rows = shuffle_state_labels_within_blocks(working_rows, rng)
                shuffled_design = build_mixed_model_design(shuffled_rows, response, scope, state_order=design.get("state_levels"))
                if shuffled_design is None:
                    continue
                shuffled_fit = fit_mixedlm_with_fallback(shuffled_design, alerts=None, vc_level_keys=vc_level_keys)
                if shuffled_fit.get("result") is None:
                    continue
                try:
                    shuffled_contrast = contrast_from_result(shuffled_fit["result"], shuffled_design, contrast_spec)
                except Exception:
                    continue
                shuffled_effect = as_float(shuffled_contrast.get("estimate"))
                if shuffled_effect is None or not np.isfinite(shuffled_effect):
                    continue
                null_effects.append(float(shuffled_effect))
            shuffle_n_success = len(null_effects)
            if null_effects:
                observed_effect = abs(float(observed["estimate"]))
                active_p = float((np.sum(np.abs(null_effects) >= observed_effect) + 1) / (len(null_effects) + 1))
            else:
                if alerts is not None:
                    alerts.append(
                        f"[ALERT] Mixed-model shuffle p-value requested for {response} ({scope}) but no successful shuffle refits were available; using classical p-values instead."
                    )
                active_p = observed["classical_p"]
                effective_p_value_source = "classical"
        observed_row = {
            "response": response,
            "scope": scope,
            "contrast_type": contrast_spec["kind"],
            "contrast_name": observed["contrast_name"],
            "state_a": contrast_spec.get("state_a"),
            "state_b": contrast_spec.get("state_b"),
            "state": contrast_spec.get("state"),
            "estimate": observed["estimate"],
            "se": observed["se"],
            "z": observed["z"],
            "classical_p": observed["classical_p"],
            "shuffle_p": active_p,
            "p_value_source": effective_p_value_source,
            "shuffle_n_requested": shuffle_n_requested,
            "shuffle_n_success": shuffle_n_success,
            "random_structure": fit_info.get("random_structure_name"),
            "fit_method": fit_info.get("fit_method"),
            "converged": bool(fit_info.get("converged", False)),
            "model_name": f"{scope}_{response}",
        }
        contrast_rows.append(observed_row)
    return {
        "response": response,
        "scope": scope,
        "p_value_source": effective_p_value_source,
        "p_value_source_requested": requested_p_value_source,
        "summary_rows": summary_rows,
        "contrast_rows": contrast_rows,
        "design": design,
        "fit": fit_info,
        "equation": mixed_model_equation_string(design, fit_info),
        "tested_terms": list(design.get("fixed_effect_names", [])),
        "tested_contrasts": [
            (
                f"{contrast_spec['kind']}:{contrast_spec.get('state_a')}:{contrast_spec.get('state_b')}"
                if contrast_spec["kind"] == "state_pair"
                else f"{contrast_spec['kind']}:{contrast_spec.get('state')}"
                if contrast_spec["kind"] == "basal_apical"
                else "visual_response_cohort:responsive_vs_nonresponsive"
                if contrast_spec["kind"] == "visual_response_cohort"
                else f"split_group_pair:{contrast_spec.get('state')}:{contrast_spec.get('group_a')}:{contrast_spec.get('group_b')}"
                if contrast_spec["kind"] == "split_group_pair"
                else contrast_spec["kind"]
            )
            for contrast_spec in valid_contrast_specs
        ],
    }

def run_mixed_model_analysis(
    cache: Dict[str, Any],
    shuffle_n: int,
    state_comparison_states: Optional[Sequence[str]],
    basal_apical_states: Optional[Sequence[str]],
    mixed_model_contrast_p_source: str = "classical",
    source_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # The mixed-model layer uses a long dendrite-by-state table and reports the fitted-test p-values directly.
    rebuild_requested = bool(cache.get("config", {}).get("analysis_tables_rebuild")) or bool(cache.get("config", {}).get("rebuild"))
    cached_table = load_cached_analysis_table(
        cache,
        "mixed_model_table",
        expected_meta={
            "analysis_unit": str(cache.get("analysis_unit", "day")),
            "visual_response_classifier_type": "dendrite",
            "visual_response_classifier_method": VISUAL_RESPONSE_CLASSIFIER_METHOD,
            "visual_response_classifier_version": VISUAL_RESPONSE_CLASSIFIER_VERSION,
            "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
            "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        },
        rebuild=rebuild_requested,
    )
    if cached_table is not None:
        table_rows = list(cached_table["table_rows"])
        table_checks = dict(cached_table["table_checks"])
    else:
        table_rows, table_checks = build_mixed_model_table(cache, source_cache=source_cache)
        store_cached_analysis_table(
            cache,
            "mixed_model_table",
            table_rows,
            table_checks,
            meta={
                "analysis_unit": str(cache.get("analysis_unit", "day")),
                "visual_response_classifier_type": "dendrite",
                "visual_response_classifier_method": VISUAL_RESPONSE_CLASSIFIER_METHOD,
                "visual_response_classifier_version": VISUAL_RESPONSE_CLASSIFIER_VERSION,
                "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
                "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
            },
        )
    alerts: List[str] = []
    state_comparison_states = list(state_comparison_states) if state_comparison_states is not None else list(PRIMARY_QUIET_STATES)
    basal_apical_states = list(basal_apical_states) if basal_apical_states is not None else list(DEFAULT_BASAL_APICAL_STATES)
    p_value_source = normalize_mixed_model_contrast_p_source(mixed_model_contrast_p_source)
    def empty_branch(state_pair_states: Sequence[str], basal_apical_state_subset: Sequence[str], visual_response_cohort: Optional[str] = None) -> Dict[str, Any]:
        return {
            "available": bool(MixedLM is not None),
            "p_value_source": p_value_source,
            "p_value_source_requested": p_value_source,
            "summary_rows": {
                "mean_dendrite_activity": [],
                "mean_spine_activity_per_dendrite": [],
                "dendrite_event_frequency_per_min": [],
                "spine_event_frequency_per_min": [],
                "coincident_event_frequency_per_min": [],
                "noncoincident_event_frequency_per_min": [],
            },
            "contrast_rows": [],
            "validation_rows": [],
            "designs": {},
            "model_equations": {},
            "tested_terms": {},
            "tested_contrasts": {},
            "selection": {
                "state_comparison_states": list(state_pair_states),
                "basal_apical_states": list(basal_apical_state_subset),
                "visual_response_cohort": visual_response_cohort,
            },
        }
    def build_branch(
        scope: str,
        *,
        row_filter_states: Sequence[str],
        state_order: Sequence[str],
        state_pair_states: Sequence[str],
        basal_apical_state_subset: Sequence[str],
        visual_response_cohort: Optional[str] = None,
        include_validation: bool = False,
    ) -> Dict[str, Any]:
        branch = empty_branch(state_pair_states, basal_apical_state_subset, visual_response_cohort=visual_response_cohort)
        state_filter = {str(state) for state in row_filter_states if state is not None and str(state).strip()}
        filtered_rows = [row for row in table_rows if str(row.get("state")) in state_filter]
        if visual_response_cohort not in (None, "all"):
            filtered_rows = [row for row in filtered_rows if str(row.get("visual_response_cohort") or "nonresponsive") == str(visual_response_cohort)]
        branch_state_order = mixed_model_visual_response_state_order(state_order, visual_response_cohort)
        branch_state_pair_states = mixed_model_visual_response_state_order(state_pair_states, visual_response_cohort)
        branch_basal_apical_states = mixed_model_visual_response_state_order(basal_apical_state_subset, visual_response_cohort)
        if scope == "selected_state":
            present_states = [state for state in state_order if any(str(row.get("state")) == state for row in filtered_rows)]
            if len(present_states) < 2:
                message = (
                    f"[ALERT] Skipping selected-state mixed-model branch because the selected state set only contains "
                    f"{len(present_states)} usable state(s): {format_report_list(present_states)}"
                )
                alerts.append(message)
                return branch
            branch_state_order = present_states
        else:
            branch_state_order = list(dict.fromkeys(state_order))
        state_pairs = [{"kind": "state_pair", "state_a": state_a, "state_b": state_b} for state_a, state_b in combinations(branch_state_pair_states, 2)]
        basal_apical_pairs = [{"kind": "basal_apical", "state": state} for state in branch_basal_apical_states]
        visual_response_pairs = [{"kind": "visual_response_cohort"}] if any(str(row.get("visual_response_cohort") or "nonresponsive") == "responsive" for row in filtered_rows) and any(str(row.get("visual_response_cohort") or "nonresponsive") == "nonresponsive" for row in filtered_rows) else []
        contrast_specs = state_pairs + basal_apical_pairs + visual_response_pairs
        for response in ["mean_dendrite_activity", "mean_spine_activity_per_dendrite", "dendrite_event_frequency_per_min", "spine_event_frequency_per_min", "coincident_event_frequency_per_min", "noncoincident_event_frequency_per_min"]:
            result = run_mixed_model_family(
                filtered_rows,
                response,
                scope,
                contrast_specs,
                shuffle_n,
                alerts=alerts,
                state_order=branch_state_order,
                p_value_source=p_value_source,
            )
            branch["summary_rows"][response].extend(result.get("summary_rows", []))
            branch["contrast_rows"].extend(result.get("contrast_rows", []))
            branch["p_value_source"] = result.get("p_value_source", p_value_source)
            branch["p_value_source_requested"] = result.get("p_value_source_requested", p_value_source)
            if result.get("design") is not None:
                design = result["design"]
                branch["designs"][response] = {
                    "response": response,
                    "scope": scope,
                    "p_value_source": result.get("p_value_source", p_value_source),
                    "state_levels": list(design.get("state_levels", [])),
                    "state_reference": design.get("state_reference"),
                    "compartment_levels": list(design.get("compartment_levels", [])),
                    "compartment_reference": design.get("compartment_reference"),
                    "include_compartment": bool(design.get("include_compartment", False)),
                    "interaction_compartments": list(design.get("interaction_compartments", [])),
                    "covariate_specs": list(design.get("covariate_specs", [])),
                    "fixed_effect_names": list(design.get("fixed_effect_names", [])),
                    "random_structure_name": result.get("fit", {}).get("random_structure_name"),
                    "fit_method": result.get("fit", {}).get("fit_method"),
                    "converged": bool(result.get("fit", {}).get("converged", False)),
                }
                branch["model_equations"][response] = result.get("equation")
                branch["tested_terms"][response] = list(result.get("tested_terms", []))
                branch["tested_contrasts"][response] = list(result.get("tested_contrasts", []))
            if include_validation:
                demo_truth = cache.get("demo_truth")
                if demo_truth:
                    expected_contrasts = demo_truth.get("expected_mixed_model_contrasts", [])
                    if expected_contrasts:
                        lookup = {
                            (row.get("response"), row.get("scope"), row.get("contrast_type"), row.get("state_a"), row.get("state_b"), row.get("state")): row
                            for row in branch["contrast_rows"]
                        }
                        for expected in expected_contrasts:
                            key = (
                                expected.get("response"),
                                expected.get("scope"),
                                expected.get("contrast_type"),
                                expected.get("state_a"),
                                expected.get("state_b"),
                                expected.get("state"),
                            )
                            observed_row = lookup.get(key)
                            if observed_row is None:
                                continue
                            expected_effect = float(expected.get("expected_effect", float("nan")))
                            observed_effect = float(observed_row.get("estimate", float("nan")))
                            if np.isfinite(expected_effect) and np.isfinite(observed_effect):
                                branch["validation_rows"].append(
                                    {
                                        "check": "mixed_model_contrast",
                                        "response": expected.get("response"),
                                        "scope": expected.get("scope"),
                                        "contrast_type": expected.get("contrast_type"),
                                        "state_a": expected.get("state_a"),
                                        "state_b": expected.get("state_b"),
                                        "state": expected.get("state"),
                                        "expected_effect": expected_effect,
                                        "observed_effect": observed_effect,
                                        "abs_error": float(abs(observed_effect - expected_effect)),
                                    }
                                )
        return branch
    selected_state_order = [state for state in state_comparison_states if state in ALL_REQUESTED_STATES]
    selected_basal_apical_states = [state for state in basal_apical_states if state in selected_state_order]
    selected_state_branch = build_branch(
        "selected_state",
        row_filter_states=selected_state_order,
        state_order=selected_state_order,
        state_pair_states=selected_state_order,
        basal_apical_state_subset=selected_basal_apical_states,
        include_validation=True,
    )
    visual_response_responsive_selected_branch = build_branch(
        "selected_state",
        row_filter_states=selected_state_order,
        state_order=selected_state_order,
        state_pair_states=selected_state_order,
        basal_apical_state_subset=selected_basal_apical_states,
        visual_response_cohort="responsive",
        include_validation=False,
    )
    visual_response_nonresponsive_selected_branch = build_branch(
        "selected_state",
        row_filter_states=selected_state_order,
        state_order=selected_state_order,
        state_pair_states=selected_state_order,
        basal_apical_state_subset=selected_basal_apical_states,
        visual_response_cohort="nonresponsive",
        include_validation=False,
    )
    return {
        "available": bool(MixedLM is not None),
        "alerts": list(dict.fromkeys(alerts)),
        "table_checks": table_checks,
        "table_rows": table_rows,
        "selection": {
            "state_comparison_states": list(state_comparison_states),
            "basal_apical_states": list(basal_apical_states),
            "p_value_source": p_value_source,
            "visual_response_cohorts": ["responsive", "nonresponsive"],
        },
        "p_value_source": p_value_source,
        "p_value_source_requested": p_value_source,
        "selected_state": selected_state_branch,
        "mixed_model": selected_state_branch,
        "mixed_model_selected_state": selected_state_branch,
        "mixed_model_visual_response_responsive": visual_response_responsive_selected_branch,
        "mixed_model_visual_response_nonresponsive": visual_response_nonresponsive_selected_branch,
        "mixed_model_visual_response_responsive_selected_state": visual_response_responsive_selected_branch,
        "mixed_model_visual_response_nonresponsive_selected_state": visual_response_nonresponsive_selected_branch,
        "validation_rows": list(selected_state_branch.get("validation_rows", [])),
    }
def process_cached_analysis(
    cache: Dict[str, Any],
    shuffle_n: int,
    state_comparison_states: Optional[Sequence[str]] = None,
    basal_apical_states: Optional[Sequence[str]] = None,
    source_cache: Optional[Dict[str, Any]] = None,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    figure_root: Optional[Path] = None,
    fit_spine_coactivity_mixed_model: bool = False,
    mixed_model_contrast_p_source: str = "classical",
    spine_coactivity_abs_threshold: float = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
) -> Dict[str, Any]:
    # Turn the cached experiments into group summaries, correlations, and matrix comparisons.
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    state_comparison_states = list(state_comparison_states) if state_comparison_states is not None else list(PRIMARY_QUIET_STATES)
    basal_apical_states = list(basal_apical_states) if basal_apical_states is not None else list(DEFAULT_BASAL_APICAL_STATES)
    results: Dict[str, Any] = {
        "state_comparisons": [],
        "basal_apical_comparisons": [],
        "correlations": [],
        "matrix_similarity": [],
        "state_summaries": {},
        "state_dendrite_summaries": {},
        "demo_validation": [],
        "alerts": list(cache.get("alerts", [])),
        "state_coverage": [],
        "mixed_model": {},
        "mixed_model_selected_state": {},
        "direct_trial_type_comparison": {},
        "spine_coactivity": {},
        "spine_coactivity_model": {},
    }
    analysis_unit = str(cache.get("analysis_unit", "day"))
    if output_dir is not None:
        cleanup_stale_state_coverage_artifacts(output_dir)
    visual_response_summary = classify_visual_responsive_dendrites(cache, source_cache=source_cache)
    visual_response_state_summaries = build_visual_response_dendrite_summary_results(cache, state_comparison_states, visual_response_summary)
    spine_visual_response_summary = classify_visual_responsive_spines(cache, source_cache=source_cache)
    spine_visual_response_state_summaries = build_visual_response_spine_state_summary_results(cache, state_comparison_states, spine_visual_response_summary)
    results["dendrite_visual_response"] = visual_response_summary
    results["dendrite_visual_response_state_summaries"] = visual_response_state_summaries
    results["spine_visual_response"] = spine_visual_response_summary
    results["spine_visual_response_state_summaries"] = spine_visual_response_state_summaries
    with step_scope("state summary metrics"):
        # Build per-state summaries first because several later outputs reuse the same numbers.
        state_metric_names = [
            "dendrite_mean",
            "spine_specific_mean",
            "dendrite_event_frequency_per_min",
            "spine_event_frequency_per_min",
            "coincident_event_frequency_per_min",
            "noncoincident_event_frequency_per_min",
        ]
        state_metric_values = {
            metric_name: summarize_state_values(cache, metric_name, state_comparison_states)
            for metric_name in state_metric_names
        }
        state_metric_dendrite_values = {
            metric_name: summarize_state_values_by_dendrite(cache, metric_name, state_comparison_states)
            for metric_name in state_metric_names
        }
        for metric_name in state_metric_names:
            results["state_summaries"][metric_name] = state_metric_values[metric_name]
            results["state_dendrite_summaries"][metric_name] = state_metric_dendrite_values[metric_name]
    # Count how much time and how many trials each experiment contributes to every state.
    with step_scope("state coverage"):
        for idx, (exp_id, exp_meta) in enumerate(experiments.items(), start=1):
            step_progress(idx, len(experiments), label=str(exp_id))
            time = np.asarray(exp_meta.get("time"), dtype=float)
            sampling_rate = estimate_sampling_rate(time)
            coverage: Dict[str, Any] = {
                "exp_id": exp_id,
                "day_id": exp_id,
                "animal_id": exp_meta.get("animal_id"),
                "compartment": exp_meta.get("compartment"),
                "sampling_rate_hz": float(sampling_rate) if sampling_rate is not None else float("nan"),
            }
            for state_label in ALL_REQUESTED_STATES:
                mask = exp_meta.get("state_masks", {}).get(state_label)
                n_frames = int(np.count_nonzero(mask)) if mask is not None else 0
                coverage[f"{state_label}_frames"] = n_frames
                if sampling_rate is not None and sampling_rate > 0:
                    coverage[f"{state_label}_seconds"] = float(n_frames / sampling_rate)
                else:
                    coverage[f"{state_label}_seconds"] = float("nan")
            cut_meta = exp_meta.get("cut", {})
            for state_label, count in cut_meta.get("trial_state_counts", {}).items():
                coverage[f"{state_label}_trials"] = int(count)
            results["state_coverage"].append(coverage)
    if output_dir is not None:
        with step_scope("figure generation: state"):
            render_analysis_family_figures(output_dir, results, cache, "state", figure_root=figure_root)
    # Main group-level comparisons requested by the analysis plan.
    with step_scope("pairwise state comparisons", total=6):
        for idx, metric in enumerate([
            "dendrite_mean",
            "spine_specific_mean",
            "dendrite_event_frequency_per_min",
            "spine_event_frequency_per_min",
            "coincident_event_frequency_per_min",
            "noncoincident_event_frequency_per_min",
        ], start=1):
            step_progress(idx, 6, label=str(metric))
            results["state_comparisons"].extend(pairwise_state_comparisons(cache, metric, state_comparison_states, shuffle_n))
    # Basal/apical comparisons are run state-by-state so each condition stays interpretable.
    with step_scope("basal/apical comparisons", total=len(basal_apical_states)):
        for idx, state in enumerate(basal_apical_states, start=1):
            step_progress(idx, len(basal_apical_states), label=str(state))
            for metric in [
                "dendrite_mean",
                "spine_specific_mean",
                "dendrite_event_frequency_per_min",
                "spine_event_frequency_per_min",
                "coincident_event_frequency_per_min",
                "noncoincident_event_frequency_per_min",
            ]:
                results["basal_apical_comparisons"].append(basal_apical_comparison(cache, metric, state, shuffle_n))
    if output_dir is not None:
        with step_scope("figure generation: basal_apical"):
            render_analysis_family_figures(output_dir, results, cache, "basal_apical", figure_root=figure_root)
    # The mixed-model layer uses a long dendrite-by-state table and reports the fitted-test p-values directly.
    with step_scope("mixed model analysis"):
        mixed_model_results = run_mixed_model_analysis(
            cache,
            shuffle_n=shuffle_n,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            mixed_model_contrast_p_source=mixed_model_contrast_p_source,
            source_cache=source_cache,
        )
    results["mixed_model"] = mixed_model_results.get("selected_state", {})
    results["mixed_model_selected_state"] = mixed_model_results.get("selected_state", {})
    results["mixed_model_visual_response_responsive"] = mixed_model_results.get("mixed_model_visual_response_responsive", {})
    results["mixed_model_visual_response_nonresponsive"] = mixed_model_results.get("mixed_model_visual_response_nonresponsive", {})
    results["mixed_model_visual_response_responsive_selected_state"] = mixed_model_results.get("mixed_model_visual_response_responsive_selected_state", {}) or results.get("mixed_model_visual_response_responsive", {})
    results["mixed_model_visual_response_nonresponsive_selected_state"] = mixed_model_results.get("mixed_model_visual_response_nonresponsive_selected_state", {}) or results.get("mixed_model_visual_response_nonresponsive", {})
    results["alerts"].extend(mixed_model_results.get("alerts", []))
    results["demo_validation"].extend(mixed_model_results.get("validation_rows", []))
    results["roi_split"] = build_mixed_model_roi_split_results(
        cache,
        mixed_model_results,
        state_comparison_states,
        shuffle_n,
    )
    with step_scope("direct trial-type comparison"):
        direct_trial_type_results = run_direct_trial_type_comparison(
            cache,
            state_comparison_states,
            shuffle_n,
            results["roi_split"].get("membership_rows", []) if isinstance(results.get("roi_split"), dict) else [],
        )
    results["direct_trial_type_comparison"] = direct_trial_type_results
    results["alerts"].extend(direct_trial_type_results.get("alerts", []))
    if output_dir is not None:
        with step_scope("figure generation: direct_trial_type_comparison"):
            render_analysis_family_figures(output_dir, results, cache, "direct_trial_type_comparison", figure_root=figure_root)
    if output_dir is not None:
        with step_scope("figure generation: mixed_model"):
            render_analysis_family_figures(output_dir, results, cache, "mixed_model", figure_root=figure_root)
    with step_scope("spine coactivity analysis"):
        spine_coactivity_results = run_spine_coactivity_analysis(
            cache,
            shuffle_n=shuffle_n,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            shared_shuffle_cache=shared_shuffle_cache,
            fit_spine_coactivity_mixed_model=fit_spine_coactivity_mixed_model,
            mixed_model_contrast_p_source=mixed_model_contrast_p_source,
            spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
        )
    results["spine_coactivity"] = {k: v for k, v in spine_coactivity_results.items() if k != "model"}
    results["spine_coactivity_model"] = {
        "available": spine_coactivity_results.get("available", False),
        "enabled": spine_coactivity_results.get("enabled", False),
        "alerts": list(spine_coactivity_results.get("alerts", [])),
        "summary_rows": {"coactivity_r": list(spine_coactivity_results.get("summary_rows", {}).get("coactivity_r", []))},
        "contrast_rows": list(spine_coactivity_results.get("contrast_rows", [])),
        "designs": spine_coactivity_results.get("designs", {}),
        "model_equations": spine_coactivity_results.get("model_equations", {}),
        "tested_terms": spine_coactivity_results.get("tested_terms", {}),
        "tested_contrasts": spine_coactivity_results.get("tested_contrasts", {}),
        "selection": spine_coactivity_results.get("selection", {}),
    }
    results["alerts"].extend(spine_coactivity_results.get("alerts", []))
    if output_dir is not None:
        with step_scope("figure generation: spine_coactivity"):
            render_analysis_family_figures(output_dir, results, cache, "spine_coactivity", figure_root=figure_root)
    # Correlate the same traces against wheel and pupil signals when those signals exist.
    dendrite_observations: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for animal_id, animal_entry in animals.items():
        for dendrite_id, dendrite_record in animal_entry["dendrites"].items():
            for exp_id, d_obs in dendrite_record["observations"].items():
                dendrite_observations.append((animal_id, dendrite_id, dendrite_record, d_obs))
    with step_scope("correlations", total=len(dendrite_observations)):
        for idx, (animal_id, dendrite_id, dendrite_record, d_obs) in enumerate(dendrite_observations, start=1):
            step_progress(idx, len(dendrite_observations), label=str(dendrite_id))
            exp_id = str(d_obs.get("exp_id") or "")
            exp_meta = experiments[exp_id]
            wheel = exp_meta["wheel"]
            pupil = exp_meta["pupil"]
            if wheel["interpolated"] is not None:
                wheel_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="dendrite_trace",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=dendrite_id,
                    vector_length=int(np.asarray(d_obs["trace"], dtype=float).size),
                )
                corr = correlation_analysis_for_observation(
                    d_obs["trace"],
                    wheel["interpolated"],
                    shuffle_n,
                    shared_shuffle_cache=shared_shuffle_cache,
                    shared_shuffle_key=wheel_key,
                )
                results["correlations"].append(
                    {
                        "analysis": "dendrite_wheel",
                        "animal_id": animal_id,
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "global_dendrite_id": dendrite_id,
                        "compartment": observation_compartment(cache, exp_id, d_obs),
                        **corr,
                    }
                )
            if pupil["series"] is not None:
                pupil_interp = pupil["series"]
                if pupil["time"] is not None and not np.array_equal(pupil["time"], d_obs["time"]):
                    pupil_interp = interpolate_series(d_obs["time"], pupil["time"], pupil["series"])
                pupil_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="dendrite_trace",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=dendrite_id,
                    vector_length=int(np.asarray(d_obs["trace"], dtype=float).size),
                )
                corr = correlation_analysis_for_observation(
                    d_obs["trace"],
                    pupil_interp,
                    shuffle_n,
                    shared_shuffle_cache=shared_shuffle_cache,
                    shared_shuffle_key=pupil_key,
                )
                results["correlations"].append(
                    {
                        "analysis": "dendrite_pupil",
                        "animal_id": animal_id,
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "global_dendrite_id": dendrite_id,
                        "compartment": observation_compartment(cache, exp_id, d_obs),
                        **corr,
                    }
                )
            for spine_id in d_obs["spine_ids"]:
                s_obs = dendrite_record["spines"][spine_id]["observations"].get(exp_id)
                if s_obs is None:
                    continue
                spine_raw_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="spine_trace",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=spine_id,
                    vector_length=int(np.asarray(s_obs["trace_hp"], dtype=float).size),
                )
                spine_specific_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="spine_specific",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=spine_id,
                    vector_length=int(np.asarray(s_obs["spine_specific"], dtype=float).size),
                )
                corr_raw = correlation_analysis_for_observation(
                    s_obs["trace_hp"],
                    d_obs["trace"],
                    shuffle_n,
                    shared_shuffle_cache=shared_shuffle_cache,
                    shared_shuffle_key=spine_raw_key,
                )
                corr_specific = correlation_analysis_for_observation(
                    s_obs["spine_specific"],
                    d_obs["trace"],
                    shuffle_n,
                    shared_shuffle_cache=shared_shuffle_cache,
                    shared_shuffle_key=spine_specific_key,
                )
                results["correlations"].append(
                    {
                        "analysis": "spine_dendrite_raw",
                        "animal_id": animal_id,
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "global_dendrite_id": dendrite_id,
                        "global_spine_id": spine_id,
                        "compartment": observation_compartment(cache, exp_id, s_obs),
                        **corr_raw,
                    }
                )
                results["correlations"].append(
                    {
                        "analysis": "spine_dendrite_specific",
                        "animal_id": animal_id,
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "global_dendrite_id": dendrite_id,
                        "global_spine_id": spine_id,
                        "compartment": observation_compartment(cache, exp_id, s_obs),
                        **corr_specific,
                    }
                )
    roi_split_membership_rows = list((results.get("roi_split", {}) if isinstance(results.get("roi_split", {}), dict) else {}).get("membership_rows", []))
    if roi_split_membership_rows:
        results["correlations"] = annotate_rows_with_split_group(results.get("correlations", []), roi_split_membership_rows)
    if output_dir is not None:
        with step_scope("figure generation: correlation"):
            render_analysis_family_figures(output_dir, results, cache, "correlation", figure_root=figure_root)
    # Spine-spine matrix comparisons ask whether the correlation structure changes across states.
    matrix_observations: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for animal_id, animal_entry in animals.items():
        for dendrite_id, dendrite_record in animal_entry["dendrites"].items():
            for exp_id, d_obs in dendrite_record["observations"].items():
                matrix_observations.append((animal_id, dendrite_id, dendrite_record, d_obs))
    with step_scope("matrix similarity", total=len(matrix_observations)):
        for idx, (animal_id, dendrite_id, dendrite_record, d_obs) in enumerate(matrix_observations, start=1):
            step_progress(idx, len(matrix_observations), label=str(dendrite_id))
            exp_id = str(d_obs.get("exp_id") or "")
            exp_meta = experiments[exp_id]
            if len(d_obs["spine_ids"]) < 2:
                continue
            state_vectors: Dict[str, List[np.ndarray]] = {}
            for state_label in [k for k in state_comparison_states if k in exp_meta["state_masks"]]:
                mask = exp_meta["state_masks"].get(state_label)
                if mask is None or not np.any(mask):
                    continue
                vectors = []
                for spine_id in d_obs["spine_ids"]:
                    s_obs = dendrite_record["spines"][spine_id]["observations"].get(exp_id)
                    if s_obs is None:
                        continue
                    vec = np.asarray(s_obs["spine_specific"][mask], dtype=float)
                    if vec.size:
                        vectors.append(vec)
                if len(vectors) >= 2:
                    state_vectors[state_label] = vectors
            for state_a, state_b in combinations(sorted(state_vectors), 2):
                observed, shuffle_p, null_mean = shuffle_matrix_similarity(state_vectors[state_a], state_vectors[state_b], shuffle_n)
                results["matrix_similarity"].append(
                    {
                        "animal_id": animal_id,
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "global_dendrite_id": dendrite_id,
                        "compartment": observation_compartment(cache, exp_id, d_obs),
                        "state_a": state_a,
                        "state_b": state_b,
                        "matrix_similarity_r": observed,
                        "shuffle_p": shuffle_p,
                        "shuffle_null_mean": null_mean,
                        "n_spines": int(len(state_vectors[state_a])),
                    }
                )
    if roi_split_membership_rows:
        results["matrix_similarity"] = annotate_rows_with_split_group(results.get("matrix_similarity", []), roi_split_membership_rows)
    if output_dir is not None:
        with step_scope("figure generation: matrix_similarity"):
            render_analysis_family_figures(output_dir, results, cache, "matrix_similarity", figure_root=figure_root)
    # The demo records planted alpha values so we can verify that the regression recovers them.
    demo_truth = cache.get("demo_truth")
    if demo_truth:
        source_experiments = (source_cache or cache).get("experiments", {})
        for expected in demo_truth.get("expected_alphas", []):
            g_spine_id = expected["global_spine_id"]
            exp_id = expected["exp_id"]
            source_exp_meta = source_experiments.get(exp_id)
            observed = None
            if source_exp_meta is not None:
                animal_id = str(source_exp_meta.get("animal_id") or derive_animal_id(exp_id))
                date = str(source_exp_meta.get("date") or derive_date(exp_id))
                compartment = str(source_exp_meta.get("compartment") or "other")
                day_id = make_day_id(animal_id, date, compartment)
            else:
                day_id = exp_id
            for animal_id, animal_entry in animals.items():
                for dendrite_id, dendrite_record in animal_entry["dendrites"].items():
                    if g_spine_id not in dendrite_record["spines"]:
                        continue
                    spine_record = dendrite_record["spines"][g_spine_id]
                    if day_id not in spine_record["observations"]:
                        continue
                    observed = spine_record["observations"][day_id]["alpha"]
                    break
            if observed is not None:
                results["demo_validation"].append(
                    {
                        "exp_id": exp_id,
                        "global_spine_id": g_spine_id,
                        "expected_alpha": float(expected["alpha"]),
                        "observed_alpha": float(observed),
                        "abs_error": float(abs(observed - expected["alpha"])),
                    }
                )
    return results
def write_analysis_report(
    report_path: Path,
    output_dir: Path,
    results: Dict[str, Any],
    analysis_cache: Dict[str, Any],
    source_cache: Dict[str, Any],
    cache_path: Path,
    artifact_paths: Sequence[str],
) -> None:
    config = results.get("config", {})
    runtime = results.get("run_parameters", {})
    selection = results.get("analysis_state_selection", {})
    cache_summary = results.get("analysis_cache_summary", results.get("cache_summary", {}))
    source_cache_summary = results.get("source_cache_summary", {})
    shared_shuffle_cache = results.get("shared_shuffle_cache", {})
    demo_validation = list(results.get("demo_validation", []))
    alerts = list(dict.fromkeys(results.get("alerts", [])))
    generated_at = datetime.datetime.now().isoformat(timespec="seconds")
    mode = "demo" if analysis_cache.get("demo_truth") else "real"
    report_artifacts = [report_relative_path(path, output_dir) for path in artifact_paths]
    cache_rel = report_relative_path(cache_path, output_dir)
    report_rel = report_relative_path(report_path, output_dir)
    for extra in [cache_rel, report_rel]:
        if extra not in report_artifacts:
            report_artifacts.append(extra)
    lines: List[str] = []
    def append_section(title: str) -> None:
        if lines:
            lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
    def append_kv(label: str, value: Any) -> None:
        lines.append(f"- {label}: {value}")
    def format_percent(value: Any) -> str:
        number = as_float(value)
        if number is None or not np.isfinite(number):
            return "n/a"
        return f"{number:.1f}%"
    def format_observation_label(row: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ["animal_id", "day_id", "global_dendrite_id", "global_spine_id", "compartment"]:
            value = row.get(key)
            if value is None:
                continue
            parts.append(str(value))
        if "day_id" not in row and row.get("exp_id") is not None:
            parts.insert(1 if parts else 0, str(row.get("exp_id")))
        return " / ".join(parts) if parts else "unknown"
    def format_state_row(row: Dict[str, Any]) -> str:
        return (
            f"{row.get('metric')} | {row.get('state_a')} vs {row.get('state_b')} | "
            f"effect={format_report_number(row.get('effect_size'))} | "
            f"shuffle_p={format_report_pvalue(row.get('shuffle_p'))} | "
            f"classical_p={format_report_pvalue(row.get('classical_p'))} | "
            f"test={row.get('test_choice', 'n/a')} | n={row.get('n_subjects', 'n/a')}"
        )
    def format_basal_row(row: Dict[str, Any]) -> str:
        return (
            f"{row.get('metric')} | {row.get('state')} | "
            f"effect={format_report_number(row.get('effect_size'))} | "
            f"shuffle_p={format_report_pvalue(row.get('shuffle_p'))} | "
            f"classical_p={format_report_pvalue(row.get('classical_p'))} | "
            f"test={row.get('test_choice', 'n/a')} | n={row.get('n_subjects', 'n/a')}"
        )
    def format_roi_split_row(row: Dict[str, Any]) -> str:
        return (
            f"{row.get('branch_name') or row.get('split_name')} | {row.get('basis_name')} | {row.get('roi_type')} | {row.get('compartment')} | {row.get('split_name')} | {row.get('window')} | "
            f"{row.get('response_column')} | {row.get('state')} | "
            f"effect={format_report_number(row.get('effect_size'))} | "
            f"shuffle_p={format_report_pvalue(row.get('shuffle_p'))} | "
            f"classical_p={format_report_pvalue(row.get('classical_p'))} | "
            f"test={row.get('test_choice', 'n/a')} | n={row.get('n_subjects', 'n/a')}"
        )
    def format_direct_trial_row(row: Dict[str, Any]) -> str:
        return (
            f"{row.get('state_a')} vs {row.get('state_b')} | "
            f"videos={row.get('n_videos', 'n/a')} | "
            f"effect={format_report_number(row.get('effect_size'))} | "
            f"shuffle_p={format_report_pvalue(row.get('shuffle_p'))} | "
            f"classical_p={format_report_pvalue(row.get('classical_p'))} | "
            f"agreement_r={format_report_number(row.get('agreement_r'))} | "
            f"test={row.get('test_choice', 'n/a')}"
        )
    def format_mixed_model_summary_row(row: Dict[str, Any]) -> str:
        term = str(row.get("term", "n/a"))
        if ":" in term:
            term = f"{term} [interaction]"
        return (
            f"{row.get('model_name')} | {term} | "
            f"estimate={format_report_number(row.get('estimate'))} | "
            f"p={format_report_pvalue(row.get('p_value'))} | "
            f"random={row.get('random_structure', 'n/a')} | fit={row.get('fit_method', 'n/a')}"
        )
    def format_mixed_model_contrast_row(row: Dict[str, Any]) -> str:
        p_source = normalize_mixed_model_contrast_p_source(row.get("p_value_source"))
        p_label = mixed_model_contrast_p_label(p_source)
        return (
            f"{row.get('model_name')} | {row.get('contrast_name')} | "
            f"estimate={format_report_number(row.get('estimate'))} | "
            f"{p_label}={format_report_pvalue(row.get('shuffle_p'))} | "
            f"random={row.get('random_structure', 'n/a')} | fit={row.get('fit_method', 'n/a')}"
        )
    def correlation_family_label(analysis: Any) -> str:
        analysis_text = str(analysis or "correlation")
        return {
            "dendrite_wheel": "dendrite activity vs wheel",
            "dendrite_pupil": "dendrite activity vs pupil",
            "spine_dendrite_raw": "spine-specific activity vs dendrite activity (raw)",
            "spine_dendrite_specific": "spine-specific activity vs dendrite activity (specific)",
        }.get(analysis_text, analysis_text.replace("_", " vs "))
    def summarize_correlation_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        has_split_group = any(str(row.get("split_group") or "").strip() for row in rows)
        for row in rows:
            analysis = str(row.get("analysis", "correlation"))
            compartment = str(row.get("compartment", "unknown"))
            split_group = str(row.get("split_group") or "").strip() if has_split_group else ""
            grouped[(analysis, compartment, split_group)].append(dict(row))
        summary_rows: List[Dict[str, Any]] = []
        for (analysis, compartment, split_group), grouped_rows in sorted(grouped.items()):
            tested = len(grouped_rows)
            significant_rows = [row for row in grouped_rows if is_significant_row(row)]
            payload: Dict[str, Any] = {
                "analysis": analysis,
                "compartment": compartment,
                "tested_dendrite_observations": tested,
                "significant_dendrite_observations": len(significant_rows),
                "percent_significant": 100.0 * len(significant_rows) / tested if tested else float("nan"),
            }
            if has_split_group:
                first_row = grouped_rows[0]
                payload.update(
                    {
                        "split_group": split_group or None,
                        "split_group_display": str(first_row.get("split_group_display") or split_group).strip() or None,
                        "split_group_color": str(first_row.get("split_group_color") or "").strip() or None,
                        "split_group_rank": first_row.get("split_group_rank"),
                    }
                )
            summary_rows.append(payload)
        return summary_rows
    def summarize_matrix_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        has_split_group = any(str(row.get("split_group") or "").strip() for row in rows)
        for row in rows:
            compartment = str(row.get("compartment", "unknown"))
            if compartment not in {"basal", "apical"}:
                continue
            state_a = str(row.get("state_a", "")).strip()
            state_b = str(row.get("state_b", "")).strip()
            if not state_a or not state_b:
                continue
            pair = tuple(sorted((state_a, state_b)))
            split_group = str(row.get("split_group") or "").strip() if has_split_group else ""
            grouped[(compartment, pair[0], pair[1], split_group)].append(dict(row))
        compartment_order = {"basal": 0, "apical": 1}
        summary_rows: List[Dict[str, Any]] = []
        for (compartment, state_a, state_b, split_group), grouped_rows in sorted(
            grouped.items(),
            key=lambda item: (
                compartment_order.get(item[0][0], 99),
                item[0][1],
                item[0][2],
                item[0][3],
            ),
        ):
            tested_rows: List[Dict[str, Any]] = []
            positive_significant = 0
            negative_significant = 0
            non_significant = 0
            for row in grouped_rows:
                r_value = as_float(row.get("matrix_similarity_r"))
                p_value = as_float(row.get("shuffle_p"))
                if r_value is None or p_value is None or not np.isfinite(r_value) or not np.isfinite(p_value):
                    continue
                tested_rows.append(row)
                if p_value < REPORT_SIGNIFICANCE_ALPHA and r_value > 0:
                    positive_significant += 1
                elif p_value < REPORT_SIGNIFICANCE_ALPHA and r_value < 0:
                    negative_significant += 1
                else:
                    non_significant += 1
            tested = len(tested_rows)
            if tested == 0:
                continue
            significant = positive_significant + negative_significant
            payload: Dict[str, Any] = {
                "compartment": compartment,
                "state_a": state_a,
                "state_b": state_b,
                "tested_observations": tested,
                "positive_significant": positive_significant,
                "negative_significant": negative_significant,
                "non_significant": non_significant,
                "significant_observations": significant,
                "percent_significant": 100.0 * significant / tested if tested else float("nan"),
            }
            if has_split_group:
                first_row = grouped_rows[0]
                payload.update(
                    {
                        "split_group": split_group or None,
                        "split_group_display": str(first_row.get("split_group_display") or split_group).strip() or None,
                        "split_group_color": str(first_row.get("split_group_color") or "").strip() or None,
                        "split_group_rank": first_row.get("split_group_rank"),
                    }
                )
            summary_rows.append(payload)
        return summary_rows
    def summarize_mixed_model_section(mixed_model_data: Dict[str, Any]) -> Dict[str, Any]:
        summary_rows = mixed_model_data.get("summary_rows", {})
        fixed_rows = list(summary_rows.get("mean_dendrite_activity", [])) + list(summary_rows.get("mean_spine_activity_per_dendrite", []))
        contrast_rows = list(mixed_model_data.get("contrast_rows", []))
        designs = mixed_model_data.get("designs", {}) if isinstance(mixed_model_data.get("designs", {}), dict) else {}
        model_equations = mixed_model_data.get("model_equations", {}) if isinstance(mixed_model_data.get("model_equations", {}), dict) else {}
        return {
            "tested_terms": len(fixed_rows),
            "significant_terms": sum(1 for row in fixed_rows if is_significant_row(row, p_key="p_value")),
            "tested_contrasts": len(contrast_rows),
            "p_value_source": mixed_model_data.get("p_value_source", "classical"),
            "significant_contrasts": sum(1 for row in contrast_rows if is_significant_row(row, p_key="shuffle_p")),
            "model_enabled": bool(summary_rows) or bool(contrast_rows) or bool(designs),
            "fallback_used": any(str(design.get("random_structure_name")) == "ols_fallback" for design in designs.values()),
            "model_equations": model_equations,
            "designs": designs,
            "tested_terms_by_response": mixed_model_data.get("tested_terms", {}),
            "tested_contrasts_by_response": mixed_model_data.get("tested_contrasts", {}),
            "fixed_rows": fixed_rows,
            "contrast_rows": contrast_rows,
            "summary_rows": summary_rows,
        }
    def summarize_direct_trial_type_section(direct_trial_type_data: Dict[str, Any]) -> Dict[str, Any]:
        state_summary_rows = list(direct_trial_type_data.get("state_summary_rows", []))
        state_pair_rows = list(direct_trial_type_data.get("state_pair_rows", []))
        animal_video_state_rows = list(direct_trial_type_data.get("animal_video_state_rows", []))
        video_state_rows = list(direct_trial_type_data.get("video_state_rows", []))
        overall_summary_rows = list(direct_trial_type_data.get("overall_summary_rows", []))
        overall = overall_summary_rows[0] if overall_summary_rows else {}
        return {
            "tested_trials": int(overall.get("tested_trial_rows", len(direct_trial_type_data.get("table_rows", [])))),
            "tested_animal_video_state_rows": int(overall.get("tested_animal_video_state_rows", len(animal_video_state_rows))),
            "tested_video_state_rows": int(overall.get("tested_video_state_rows", len(video_state_rows))),
            "tested_videos": int(overall.get("tested_videos", len({str(row.get("video_id")) for row in video_state_rows}))),
            "tested_animals": int(overall.get("tested_animals", len({str(row.get("animal_id")) for row in animal_video_state_rows}))),
            "tested_state_pairs": int(overall.get("tested_state_pairs", len(state_pair_rows))),
            "significant_state_pairs": int(overall.get("significant_state_pairs", sum(1 for row in state_pair_rows if is_significant_row(row, p_key="shuffle_p")))),
            "mean_effect_size": as_float(overall.get("mean_effect_size")),
            "mean_agreement_r": as_float(overall.get("mean_agreement_r")),
            "state_summary_rows": state_summary_rows,
            "state_pair_rows": state_pair_rows,
            "animal_video_state_rows": animal_video_state_rows,
            "video_state_rows": video_state_rows,
            "overall_summary_rows": overall_summary_rows,
        }
    def summarize_spine_coactivity_section(spine_coactivity_data: Dict[str, Any], spine_coactivity_model: Dict[str, Any]) -> Dict[str, Any]:
        pair_summary_rows = list(spine_coactivity_data.get("pair_summary_rows", []))
        state_summary_rows = list(spine_coactivity_data.get("state_summary_rows", []))
        state_agreement_rows = list(spine_coactivity_data.get("state_agreement_rows", []))
        compartment_summary_rows = list(spine_coactivity_data.get("compartment_summary_rows", []))
        property_split_data = spine_coactivity_data.get("property_split", {}) if isinstance(spine_coactivity_data.get("property_split", {}), dict) else {}
        property_split_bundles = list(property_split_data.get("bundles", [])) if isinstance(property_split_data, dict) else []
        property_split_subject_rows = list(property_split_data.get("subject_state_rows", [])) if isinstance(property_split_data, dict) else []
        property_split_comparisons = list(property_split_data.get("comparison_rows", [])) if isinstance(property_split_data, dict) else []
        property_split_summary_rows = list(property_split_data.get("summary_rows", [])) if isinstance(property_split_data, dict) else []
        model_summary_rows = list((spine_coactivity_model.get("summary_rows", {}) or {}).get("coactivity_r", []))
        contrast_rows = list(spine_coactivity_model.get("contrast_rows", []))
        return {
            "tested_pairs": len(pair_summary_rows),
            "tested_state_rows": len(state_summary_rows),
            "tested_contrasts": len(contrast_rows),
            "p_value_source": spine_coactivity_model.get("p_value_source", "classical"),
            "significant_contrasts": sum(1 for row in contrast_rows if is_significant_row(row, p_key="shuffle_p")),
            "mean_state_agreement_r": float(np.nanmean([as_float(row.get("mean_state_agreement_r")) for row in compartment_summary_rows if np.isfinite(as_float(row.get("mean_state_agreement_r")))])) if any(np.isfinite(as_float(row.get("mean_state_agreement_r"))) for row in compartment_summary_rows) else float("nan"),
            "mean_positive_state_fraction": float(np.nanmean([as_float(row.get("mean_positive_state_fraction")) for row in compartment_summary_rows if np.isfinite(as_float(row.get("mean_positive_state_fraction")))])) if any(np.isfinite(as_float(row.get("mean_positive_state_fraction"))) for row in compartment_summary_rows) else float("nan"),
            "mean_profile_similarity_r": float(np.nanmean([as_float(row.get("mean_profile_similarity_r")) for row in compartment_summary_rows if np.isfinite(as_float(row.get("mean_profile_similarity_r")))])) if any(np.isfinite(as_float(row.get("mean_profile_similarity_r"))) for row in compartment_summary_rows) else float("nan"),
            "property_split_bundles": len(property_split_bundles),
            "property_split_subject_rows": len(property_split_subject_rows),
            "property_split_comparisons": len(property_split_comparisons),
            "property_split_summary_rows": len(property_split_summary_rows),
            "model_enabled": bool(spine_coactivity_model.get("enabled", False)),
            "model_equations": spine_coactivity_model.get("model_equations", {}),
            "designs": spine_coactivity_model.get("designs", {}),
            "summary_rows": state_summary_rows,
            "pair_summary_rows": pair_summary_rows,
            "state_agreement_rows": state_agreement_rows,
            "compartment_summary_rows": compartment_summary_rows,
            "contrast_rows": contrast_rows,
            "model_summary_rows": model_summary_rows,
            "property_split_comparison_rows": property_split_comparisons,
        }
    def collect_quality_summary(
            source_cache_obj: Dict[str, Any],
            analysis_cache_obj: Dict[str, Any],
            results_obj: Dict[str, Any],
            mixed_summary: Dict[str, Any],
        ) -> Dict[str, Any]:
            experiments = source_cache_obj.get("experiments", {})
            missing_wheel = sorted(
                exp_id
                for exp_id, exp_meta in experiments.items()
                if exp_meta.get("wheel", {}).get("interpolated") is None
            )
            missing_pupil = sorted(
                exp_id
                for exp_id, exp_meta in experiments.items()
                if exp_meta.get("pupil", {}).get("series") is None
            )
            insufficient_spines = 0
            for animal_entry in analysis_cache_obj.get("animals", {}).values():
                for dendrite_record in animal_entry.get("dendrites", {}).values():
                    for d_obs in dendrite_record.get("observations", {}).values():
                        if len(d_obs.get("spine_ids", [])) < 2:
                            insufficient_spines += 1
            missing_sleep = sorted(
                {
                    match.group(1)
                    for alert in results_obj.get("alerts", [])
                    for match in [re.search(r"sleep_state\.pickle missing for ([^;]+);", str(alert))]
                    if match
                }
            )
            skipped_state_terms = sorted(
                {
                    match.group(1)
                    for alert in results_obj.get("alerts", [])
                    for match in [re.search(r"does not contain: (.+)$", str(alert))]
                    if match
                }
            )
            mixed_fallback_reasons = sorted(
                {
                    match.group(1)
                    for alert in results_obj.get("alerts", [])
                    for match in [re.search(r"using a fixed-effect fallback instead \((.+)\)\.", str(alert))]
                    if match
                }
            )
            return {
                "missing_sleep": missing_sleep,
                "missing_wheel": missing_wheel,
                "missing_pupil": missing_pupil,
                "insufficient_spines": insufficient_spines,
                "skipped_states": skipped_state_terms,
                "mixed_fallback_reasons": mixed_fallback_reasons,
                "mixed_fallback_used": mixed_summary.get("fallback_used", False),
            }
    def collect_event_summary(analysis_cache_obj: Dict[str, Any]) -> Dict[str, Any]:
        compartment_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        dendrite_obs_total = 0
        spine_obs_total = 0
        for animal_entry in analysis_cache_obj.get("animals", {}).values():
            for dendrite_record in animal_entry.get("dendrites", {}).values():
                for d_obs in dendrite_record.get("observations", {}).values():
                    event_info = d_obs.get("event_info") or {}
                    freq = as_float(event_info.get("event_frequency_per_min"))
                    if freq is None or not np.isfinite(freq):
                        continue
                    compartment = str(d_obs.get("compartment", dendrite_record.get("compartment", "unknown")))
                    compartment_stats[compartment]["dendrite_event_frequency_per_min"].append(float(freq))
                    dendrite_obs_total += 1
                for spine_record in dendrite_record.get("spines", {}).values():
                    for s_obs in spine_record.get("observations", {}).values():
                        event_info = s_obs.get("event_info") or {}
                        freq = as_float(event_info.get("event_frequency_per_min"))
                        if freq is None or not np.isfinite(freq):
                            continue
                        compartment = str(s_obs.get("compartment", spine_record.get("compartment", dendrite_record.get("compartment", "unknown"))))
                        compartment_stats[compartment]["spine_event_frequency_per_min"].append(float(freq))
                        coincident_freq = as_float(event_info.get("coincident_event_frequency_per_min"))
                        noncoincident_freq = as_float(event_info.get("noncoincident_event_frequency_per_min"))
                        coincident_fraction = as_float(event_info.get("coincident_event_fraction"))
                        if coincident_freq is not None and np.isfinite(coincident_freq):
                            compartment_stats[compartment]["coincident_event_frequency_per_min"].append(float(coincident_freq))
                        if noncoincident_freq is not None and np.isfinite(noncoincident_freq):
                            compartment_stats[compartment]["noncoincident_event_frequency_per_min"].append(float(noncoincident_freq))
                        if coincident_fraction is not None and np.isfinite(coincident_fraction):
                            compartment_stats[compartment]["coincident_event_fraction"].append(float(coincident_fraction))
                        spine_obs_total += 1
        compartment_order = {"basal": 0, "apical": 1}
        compartment_rows: List[Dict[str, Any]] = []
        for compartment in sorted(compartment_stats, key=lambda value: compartment_order.get(value, 99)):
            stats = compartment_stats[compartment]
            compartment_rows.append(
                {
                    "compartment": compartment,
                    "n_dendrite_observations": int(len(stats.get("dendrite_event_frequency_per_min", []))),
                    "mean_dendrite_event_frequency_per_min": float(np.nanmean(stats.get("dendrite_event_frequency_per_min", []))) if stats.get("dendrite_event_frequency_per_min") else float("nan"),
                    "n_spine_observations": int(len(stats.get("spine_event_frequency_per_min", []))),
                    "mean_spine_event_frequency_per_min": float(np.nanmean(stats.get("spine_event_frequency_per_min", []))) if stats.get("spine_event_frequency_per_min") else float("nan"),
                    "mean_coincident_event_frequency_per_min": float(np.nanmean(stats.get("coincident_event_frequency_per_min", []))) if stats.get("coincident_event_frequency_per_min") else float("nan"),
                    "mean_noncoincident_event_frequency_per_min": float(np.nanmean(stats.get("noncoincident_event_frequency_per_min", []))) if stats.get("noncoincident_event_frequency_per_min") else float("nan"),
                    "mean_coincident_event_fraction": float(np.nanmean(stats.get("coincident_event_fraction", []))) if stats.get("coincident_event_fraction") else float("nan"),
                }
            )
        all_dendrite_freqs = [freq for stats in compartment_stats.values() for freq in stats.get("dendrite_event_frequency_per_min", [])]
        all_spine_freqs = [freq for stats in compartment_stats.values() for freq in stats.get("spine_event_frequency_per_min", [])]
        all_coincident_freqs = [freq for stats in compartment_stats.values() for freq in stats.get("coincident_event_frequency_per_min", [])]
        all_noncoincident_freqs = [freq for stats in compartment_stats.values() for freq in stats.get("noncoincident_event_frequency_per_min", [])]
        all_coincident_fractions = [freq for stats in compartment_stats.values() for freq in stats.get("coincident_event_fraction", [])]
        return {
            "tested_dendrite_observations": int(dendrite_obs_total),
            "tested_spine_observations": int(spine_obs_total),
            "mean_dendrite_event_frequency_per_min": float(np.nanmean(all_dendrite_freqs)) if all_dendrite_freqs else float("nan"),
            "mean_spine_event_frequency_per_min": float(np.nanmean(all_spine_freqs)) if all_spine_freqs else float("nan"),
            "mean_coincident_event_frequency_per_min": float(np.nanmean(all_coincident_freqs)) if all_coincident_freqs else float("nan"),
            "mean_noncoincident_event_frequency_per_min": float(np.nanmean(all_noncoincident_freqs)) if all_noncoincident_freqs else float("nan"),
            "mean_coincident_event_fraction": float(np.nanmean(all_coincident_fractions)) if all_coincident_fractions else float("nan"),
            "compartment_summary_rows": compartment_rows,
        }
    def strongest_significant_result(
        state_rows: Sequence[Dict[str, Any]],
        basal_rows: Sequence[Dict[str, Any]],
        correlation_rows: Sequence[Dict[str, Any]],
        matrix_rows: Sequence[Dict[str, Any]],
        mixed_summary: Dict[str, Any],
    ) -> str:
        candidates: List[Tuple[float, str]] = []
        def add_candidate(row: Dict[str, Any], p_key: str, label: str, effect_key: str = "effect_size") -> None:
            try:
                p_value = float(row.get(p_key, float("nan")))
            except Exception:
                return
            if not np.isfinite(p_value) or p_value >= REPORT_SIGNIFICANCE_ALPHA:
                return
            try:
                effect = float(row.get(effect_key, row.get("estimate", float("nan"))))
            except Exception:
                effect = float("nan")
            effect_text = format_report_number(effect) if np.isfinite(effect) else "n/a"
            candidates.append((p_value, f"{label} | effect={effect_text} | p={format_report_pvalue(p_value)}"))
        for row in state_rows:
            add_candidate(row, "shuffle_p", f"state comparison {row.get('metric')} {row.get('state_a')} vs {row.get('state_b')}")
        for row in basal_rows:
            add_candidate(row, "shuffle_p", f"basal/apical {row.get('metric')} {row.get('state')}")
        for row in correlation_rows:
            add_candidate(row, "shuffle_p", f"correlation {correlation_family_label(row.get('analysis'))} {format_observation_label(row)}", effect_key="r")
        for row in matrix_rows:
            add_candidate(
                row,
                "shuffle_p",
                f"matrix similarity {row.get('compartment', 'unknown')} {row.get('global_dendrite_id')} {row.get('state_a')} vs {row.get('state_b')}",
                effect_key="matrix_similarity_r",
            )
        for row in direct_trial_type_summary.get("state_pair_rows", []):
            add_candidate(
                row,
                "shuffle_p",
                f"direct trial-type {row.get('state_a')} vs {row.get('state_b')}",
                effect_key="effect_size",
            )
        for row in mixed_summary.get("fixed_rows", []):
            add_candidate(row, "p_value", f"mixed model {row.get('model_name')} {row.get('term')}", effect_key="estimate")
        for row in mixed_summary.get("contrast_rows", []):
            add_candidate(row, "shuffle_p", f"mixed contrast {row.get('model_name')} {row.get('contrast_name')}", effect_key="estimate")
        coactivity_model = results.get("spine_coactivity_model", {})
        if isinstance(coactivity_model, dict):
            for row in coactivity_model.get("contrast_rows", []):
                add_candidate(row, "shuffle_p", f"spine coactivity {row.get('model_name')} {row.get('contrast_name')}", effect_key="estimate")
        for row in spine_coactivity_summary.get("property_split_comparison_rows", []):
            add_candidate(
                row,
                "shuffle_p",
                f"spine coactivity property split {row.get('branch_name')} {row.get('basis_name')} {row.get('compartment')} {row.get('comparison')}",
                effect_key="effect_size",
            )
        if not candidates:
            return "none"
        _, label = min(candidates, key=lambda item: item[0])
        return label
    def render_state_section(
        title: str,
        rows: Sequence[Dict[str, Any]],
        row_formatter,
        *,
        group_key_fn=None,
        p_key: str = "shuffle_p",
        p_label: Optional[str] = None,
        tested_label: str = "tested",
    ) -> None:
        append_section(title)
        tested_rows = [dict(row) for row in rows]
        significant_rows = [row for row in tested_rows if is_significant_row(row, p_key=p_key)]
        append_kv(tested_label, len(tested_rows))
        p_label = p_label or ("shuffle_p" if p_key == "shuffle_p" else p_key)
        append_kv(f"significant ({p_label} < {REPORT_SIGNIFICANCE_ALPHA:g})", len(significant_rows))
        if not significant_rows:
            lines.append("- no significant results")
            return
        if group_key_fn is None:
            for row in sort_rows_by_shuffle_p(significant_rows, p_key=p_key):
                lines.append(f"- {row_formatter(row)}")
            return
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in sort_rows_by_shuffle_p(significant_rows, p_key=p_key):
            grouped[str(group_key_fn(row))].append(row)
        for group_name in sorted(grouped):
            lines.append(f"- {group_name}")
            for row in grouped[group_name]:
                lines.append(f"  - {row_formatter(row)}")
    def render_correlation_summary_section(summary_rows: Sequence[Dict[str, Any]]) -> None:
        append_section("Correlations")
        if not summary_rows:
            lines.append("- none")
            return
        total_tested = sum(int(row.get("tested_dendrite_observations", 0)) for row in summary_rows)
        total_significant = sum(int(row.get("significant_dendrite_observations", 0)) for row in summary_rows)
        append_kv("tested day observations", total_tested)
        append_kv("significant day observations", total_significant)
        append_kv("percent significant", format_percent(100.0 * total_significant / total_tested if total_tested else float("nan")))
        for row in sorted(
            summary_rows,
            key=lambda item: (
                str(item.get("analysis", "")),
                str(item.get("compartment", "")),
            ),
        ):
            lines.append(
                f"- {correlation_family_label(row.get('analysis'))} | {row.get('compartment')} | "
                f"{row.get('tested_dendrite_observations')} tested day observations | "
                f"{row.get('significant_dendrite_observations')} significant | "
                f"{format_percent(row.get('percent_significant'))}"
            )
    def render_matrix_summary_section(summary_rows: Sequence[Dict[str, Any]]) -> None:
        append_section("Spine-spine matrix similarity")
        if not summary_rows:
            lines.append("- none")
            return
        total_tested = sum(int(row.get("tested_observations", 0)) for row in summary_rows)
        total_positive = sum(int(row.get("positive_significant", 0)) for row in summary_rows)
        total_negative = sum(int(row.get("negative_significant", 0)) for row in summary_rows)
        total_non_significant = sum(int(row.get("non_significant", 0)) for row in summary_rows)
        total_significant = total_positive + total_negative
        append_kv("tested day observations", total_tested)
        append_kv("positive significant", total_positive)
        append_kv("negative significant", total_negative)
        append_kv("non-significant", total_non_significant)
        append_kv("percent significant", format_percent(100.0 * total_significant / total_tested if total_tested else float("nan")))
        lines.append(
            f"- legend: positive significant means matrix_similarity_r > 0 and shuffle_p < {REPORT_SIGNIFICANCE_ALPHA:g}; "
            f"negative significant means matrix_similarity_r < 0 and shuffle_p < {REPORT_SIGNIFICANCE_ALPHA:g}; "
            "non-significant means everything else"
        )
        for compartment in ["basal", "apical"]:
            compartment_rows = [row for row in summary_rows if str(row.get("compartment", "unknown")) == compartment]
            if not compartment_rows:
                continue
            lines.append(f"- {compartment}")
            for row in compartment_rows:
                lines.append(
                    f"  - {row.get('state_a')} vs {row.get('state_b')} | "
                    f"{row.get('tested_observations')} tested day observations | "
                    f"{row.get('positive_significant')} positive significant | "
                    f"{row.get('negative_significant')} negative significant | "
                    f"{row.get('non_significant')} non-significant | "
                    f"{format_percent(row.get('percent_significant'))} significant"
                )
    def render_validation_summary(validation_rows: Sequence[Dict[str, Any]], mixed_summary: Dict[str, Any]) -> None:
        append_section("Validation / tests")
        if validation_rows:
            abs_errors = [as_float(row.get("abs_error")) for row in validation_rows]
            finite_errors = [(idx, value) for idx, value in enumerate(abs_errors) if value is not None and np.isfinite(value)]
            max_error = max((value for _, value in finite_errors), default=float("nan"))
            worst_row = validation_rows[max(finite_errors, key=lambda item: item[1])[0]] if finite_errors else None
            n_checks = len(validation_rows)
            append_kv("demo validation rows", n_checks)
            append_kv("max_abs_error", format_report_number(max_error))
            if worst_row is not None:
                append_kv("worst_check", worst_row.get("check", "validation"))
            append_kv("detail", "see demo_validation.csv for row-level checks")
        else:
            append_kv("demo validation rows", 0)
            append_kv("max_abs_error", "n/a")
            append_kv("detail", "no demo validation rows in this run")
        if mixed_summary.get("table_checks"):
            checks = mixed_summary.get("table_checks", {})
            append_kv(
                "mixed-model table check",
                format_report_number(checks.get("mean_spine_activity_per_dendrite_max_abs_error")),
            )
    state_rows = list(results.get("state_comparisons", []))
    basal_rows = list(results.get("basal_apical_comparisons", []))
    correlation_rows = list(results.get("correlations", []))
    matrix_rows = list(results.get("matrix_similarity", []))
    mixed_model = results.get("mixed_model", {})
    mixed_model = mixed_model if isinstance(mixed_model, dict) else {}
    mixed_model_selected = results.get("mixed_model_selected_state", {})
    mixed_model_selected = mixed_model_selected if isinstance(mixed_model_selected, dict) else {}
    direct_trial_type = results.get("direct_trial_type_comparison", {})
    direct_trial_type = direct_trial_type if isinstance(direct_trial_type, dict) else {}
    mixed_summary = summarize_mixed_model_section(mixed_model)
    mixed_selected_summary = summarize_mixed_model_section(mixed_model_selected)
    roi_split = results.get("roi_split", {})
    roi_split = roi_split if isinstance(roi_split, dict) else {}
    mixed_combined_summary = {
        "fallback_used": bool(mixed_summary.get("fallback_used")) or bool(mixed_selected_summary.get("fallback_used")),
        "fixed_rows": list(mixed_summary.get("fixed_rows", [])) + list(mixed_selected_summary.get("fixed_rows", [])),
        "contrast_rows": list(mixed_summary.get("contrast_rows", [])) + list(mixed_selected_summary.get("contrast_rows", [])),
    }
    direct_trial_type_summary = summarize_direct_trial_type_section(direct_trial_type)
    spine_coactivity = results.get("spine_coactivity", {})
    spine_coactivity = spine_coactivity if isinstance(spine_coactivity, dict) else {}
    spine_coactivity_model = results.get("spine_coactivity_model", {})
    spine_coactivity_model = spine_coactivity_model if isinstance(spine_coactivity_model, dict) else {}
    spine_coactivity_summary = summarize_spine_coactivity_section(spine_coactivity, spine_coactivity_model)
    correlation_summary_rows = summarize_correlation_rows(correlation_rows)
    matrix_summary_rows = summarize_matrix_rows(matrix_rows)
    quality_summary = collect_quality_summary(source_cache, analysis_cache, results, mixed_combined_summary)
    event_summary = collect_event_summary(analysis_cache)
    state_test_pairs = [f"{state_a} vs {state_b}" for state_a, state_b in combinations(selection.get("state_comparison_states") or [], 2)]
    basal_apical_test_labels = [f"{state}: apical - basal" for state in (selection.get("basal_apical_states") or [])]
    strong_result = strongest_significant_result(state_rows, basal_rows, correlation_rows, matrix_rows, mixed_combined_summary)
    lines.append("Sleep Dendrite/Spine Analysis Report")
    lines.append("====================================")
    lines.append(f"Generated at: {generated_at}")
    lines.append(f"Run mode: {mode}")
    lines.append(f"Output directory: {output_dir}")
    lines.append(f"Cache file: {cache_rel}")
    analysis_tables_cache_rel = results.get("run_parameters", {}).get("analysis_tables_cache_path")
    analysis_results_cache_rel = results.get("run_parameters", {}).get("analysis_results_cache_path")
    lines.append(
        f"Analysis tables cache: {report_relative_path(Path(analysis_tables_cache_rel), output_dir) if analysis_tables_cache_rel else 'n/a'}"
    )
    lines.append(
        f"Analysis results cache: {report_relative_path(Path(analysis_results_cache_rel), output_dir) if analysis_results_cache_rel else 'n/a'}"
    )
    lines.append(f"Report file: {report_rel}")
    append_section("Executive summary")
    append_kv("animals loaded", cache_summary.get("n_animals", "n/a"))
    append_kv("days loaded", cache_summary.get("n_days", cache_summary.get("n_experiments", "n/a")))
    if source_cache_summary:
        append_kv("source experiments loaded", source_cache_summary.get("n_experiments", "n/a"))
    append_kv(
        "required sleep files missing",
        f"yes ({format_report_list(quality_summary.get('missing_sleep'))})"
        if quality_summary.get("missing_sleep")
        else "no",
    )
    append_kv(
        "mixed-model fallback used",
        "yes" if mixed_combined_summary.get("fallback_used") else "no",
    )
    append_kv(
        "spine coactivity summary",
        f"{spine_coactivity_summary.get('tested_pairs', 0)} pairs, {spine_coactivity_summary.get('tested_state_rows', 0)} state rows, "
        f"{spine_coactivity_summary.get('tested_contrasts', 0)} contrasts, "
        f"mean agreement r={format_report_number(spine_coactivity_summary.get('mean_state_agreement_r'))}",
    )
    append_kv("strongest significant result", strong_result)
    append_section("Results at a glance")
    append_kv(
        "state comparisons",
        f"{len(state_rows)} tested day comparisons, {sum(1 for row in state_rows if is_significant_row(row))} significant, "
        f"{format_percent(100.0 * sum(1 for row in state_rows if is_significant_row(row)) / len(state_rows) if state_rows else float('nan'))}",
    )
    append_kv(
        "basal-vs-apical",
        f"{len(basal_rows)} tested day comparisons, {sum(1 for row in basal_rows if is_significant_row(row))} significant, "
        f"{format_percent(100.0 * sum(1 for row in basal_rows if is_significant_row(row)) / len(basal_rows) if basal_rows else float('nan'))}",
    )
    append_kv(
        "direct trial-type comparison",
        f"{direct_trial_type_summary.get('tested_videos', 0)} tested videos, "
        f"{direct_trial_type_summary.get('tested_state_pairs', 0)} tested state pairs, "
        f"{direct_trial_type_summary.get('significant_state_pairs', 0)} significant, "
        f"mean agreement r={format_report_number(direct_trial_type_summary.get('mean_agreement_r'))}",
    )
    total_corr_tested = sum(int(row.get("tested_dendrite_observations", 0)) for row in correlation_summary_rows)
    total_corr_sig = sum(int(row.get("significant_dendrite_observations", 0)) for row in correlation_summary_rows)
    append_kv(
        "correlations",
        f"{total_corr_tested} tested day observations, {total_corr_sig} significant, "
        f"{format_percent(100.0 * total_corr_sig / total_corr_tested if total_corr_tested else float('nan'))}",
    )
    total_matrix_tested = sum(int(row.get("tested_observations", 0)) for row in matrix_summary_rows)
    total_matrix_positive = sum(int(row.get("positive_significant", 0)) for row in matrix_summary_rows)
    total_matrix_negative = sum(int(row.get("negative_significant", 0)) for row in matrix_summary_rows)
    total_matrix_non_significant = sum(int(row.get("non_significant", 0)) for row in matrix_summary_rows)
    total_matrix_sig = total_matrix_positive + total_matrix_negative
    append_kv(
        "spine-spine matrix",
        f"{total_matrix_tested} tested day observations, {total_matrix_positive} positive significant, "
        f"{total_matrix_negative} negative significant, {total_matrix_non_significant} non-significant, "
        f"{format_percent(100.0 * total_matrix_sig / total_matrix_tested if total_matrix_tested else float('nan'))} significant",
    )
    append_kv(
        "mixed model (selected_state)",
        f"{mixed_summary.get('tested_terms', 0)} tested day-level terms, {mixed_summary.get('significant_terms', 0)} significant, "
        f"fallback={'yes' if mixed_summary.get('fallback_used') else 'no'}, "
        f"model={'enabled' if mixed_summary.get('model_enabled') else 'disabled'}",
    )
    append_kv(
        "mixed model (selected_state)",
        f"{mixed_selected_summary.get('tested_terms', 0)} tested day-level terms, {mixed_selected_summary.get('significant_terms', 0)} significant, "
        f"fallback={'yes' if mixed_selected_summary.get('fallback_used') else 'no'}, "
        f"model={'enabled' if mixed_selected_summary.get('model_enabled') else 'disabled'}",
    )
    append_kv(
        "spine coactivity",
        f"{spine_coactivity_summary.get('tested_pairs', 0)} pairs, {spine_coactivity_summary.get('tested_state_rows', 0)} state rows, "
        f"{spine_coactivity_summary.get('significant_contrasts', 0)} significant contrasts, "
        f"mean agreement r={format_report_number(spine_coactivity_summary.get('mean_state_agreement_r'))}, "
        "model=" + ("enabled" if spine_coactivity_summary.get('model_enabled') else "disabled"),
    )
    append_section("Run metadata")
    append_kv("user_id", config.get("user_id", "n/a"))
    append_kv("repo_base", config.get("repo_base", "n/a"))
    append_kv("analysis_unit", analysis_cache.get("analysis_unit", "source"))
    append_kv("channel", runtime.get("channel", config.get("channel", "n/a")))
    append_kv("shuffle_n", runtime.get("shuffle_n", config.get("shuffle_n", "n/a")))
    append_kv("cpu_thread_limit", runtime.get("cpu_thread_limit", config.get("cpu_thread_limit", DEFAULT_CPU_THREAD_LIMIT)))
    append_kv("shared_shuffle_cache", shared_shuffle_cache.get("path", "n/a"))
    append_kv("shared_shuffle_cache_reused", shared_shuffle_cache.get("reused", "n/a"))
    append_kv("shared_shuffle_cache_entries", shared_shuffle_cache.get("entry_count", "n/a"))
    locomotion_threshold = runtime.get("locomotion_threshold", config.get("locomotion_threshold"))
    spine_coactivity_abs_threshold = runtime.get("spine_coactivity_abs_threshold", config.get("spine_coactivity_abs_threshold", DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD))
    append_kv("locomotion_threshold", "auto" if locomotion_threshold is None else format_report_number(locomotion_threshold))
    append_kv("spine_coactivity_abs_threshold", f"abs(coactivity_r) >= {format_report_number(spine_coactivity_abs_threshold)}")
    append_kv("movie_expids", format_report_list(config.get("movie_expids")))
    append_kv("sleep_expids", format_report_list(config.get("sleep_expids")))
    append_kv("basal_expids", format_report_list(config.get("basal_expids")))
    append_kv("apical_expids", format_report_list(config.get("apical_expids")))
    append_kv("state_mode", selection.get("state_mode", "n/a"))
    append_kv("state_mode_source", selection.get("state_mode_source", "n/a"))
    append_kv("movie_trial_types", format_report_list(selection.get("movie_trial_types")))
    append_kv("movie_trial_types_source", selection.get("movie_trial_types_source", "n/a"))
    append_kv("compare_states", format_report_list(selection.get("compare_states")))
    append_kv("state_comparison_states", format_report_list(selection.get("state_comparison_states")))
    append_kv("basal_apical_states", format_report_list(selection.get("basal_apical_states")))
    append_kv("event_detection_method", config.get("event_detection_method", "derivative"))
    append_kv("visual_response_metric", config.get("visual_response_metric", "mean"))
    if cache_summary:
        append_kv("n_animals", cache_summary.get("n_animals", "n/a"))
        append_kv("n_days", cache_summary.get("n_days", cache_summary.get("n_experiments", "n/a")))
        append_kv("n_global_dendrites", cache_summary.get("n_global_dendrites", "n/a"))
        append_kv("n_global_spines", cache_summary.get("n_global_spines", "n/a"))
        append_kv("n_dendrite_observations", cache_summary.get("n_dendrite_observations", "n/a"))
    append_section("Model diagnostics")
    for branch_name, branch_summary in [("selected_state", mixed_selected_summary)]:
        if branch_summary.get("model_equations"):
            lines.append(f"- {branch_name}")
            for response_name, equation in sorted(branch_summary.get("model_equations", {}).items()):
                design = branch_summary.get("designs", {}).get(response_name, {})
                lines.append(f"  - {response_name}: {equation}")
                lines.append(f"    - random structure: {design.get('random_structure_name', 'n/a')}")
                lines.append(f"    - fit method: {design.get('fit_method', 'n/a')}")
                lines.append(f"    - converged: {'yes' if design.get('converged') else 'no'}")
                lines.append(f"    - fallback: {'yes' if design.get('random_structure_name') == 'ols_fallback' else 'no'}")
        else:
            lines.append(f"- {branch_name}: unavailable")
    append_section("Spine coactivity")
    append_kv("tested pairs", spine_coactivity_summary.get("tested_pairs", 0))
    append_kv("tested state rows", spine_coactivity_summary.get("tested_state_rows", 0))
    append_kv("tested contrasts", spine_coactivity_summary.get("tested_contrasts", 0))
    append_kv("significant contrasts", spine_coactivity_summary.get("significant_contrasts", 0))
    append_kv("mean state agreement", format_report_number(spine_coactivity_summary.get("mean_state_agreement_r")))
    append_kv("mean positive-state fraction", format_report_number(spine_coactivity_summary.get("mean_positive_state_fraction")))
    append_kv("mean profile similarity", format_report_number(spine_coactivity_summary.get("mean_profile_similarity_r")))
    append_kv(
        "spine coactivity property split",
        f"{spine_coactivity_summary.get('property_split_bundles', 0)} bundles, {spine_coactivity_summary.get('property_split_subject_rows', 0)} pair-state rows, {spine_coactivity_summary.get('property_split_comparisons', 0)} comparisons",
    )
    append_kv("quiet-anchor selection", f"shuffle_significant and abs(coactivity_r) >= {format_report_number(spine_coactivity_abs_threshold)}")
    for row in spine_coactivity_summary.get("compartment_summary_rows", []):
        lines.append(
            f"- {row.get('compartment')}: pairs={row.get('n_pairs', 'n/a')} | animals={row.get('n_animals', 'n/a')} | "
            f"pair r={format_report_number(row.get('mean_pair_coactivity_r'))} | agreement r={format_report_number(row.get('mean_state_agreement_r'))} | "
            f"positive fraction={format_report_number(row.get('mean_positive_state_fraction'))}"
        )
    append_section("Calcium events")
    append_kv("dendrite observations", event_summary.get("tested_dendrite_observations", 0))
    append_kv("spine observations", event_summary.get("tested_spine_observations", 0))
    append_kv("mean dendrite event frequency", format_report_number(event_summary.get("mean_dendrite_event_frequency_per_min")))
    append_kv("mean spine event frequency", format_report_number(event_summary.get("mean_spine_event_frequency_per_min")))
    append_kv("mean coincident spine event frequency", format_report_number(event_summary.get("mean_coincident_event_frequency_per_min")))
    append_kv("mean noncoincident spine event frequency", format_report_number(event_summary.get("mean_noncoincident_event_frequency_per_min")))
    append_kv("mean coincident fraction", format_report_number(event_summary.get("mean_coincident_event_fraction")))
    for row in event_summary.get("compartment_summary_rows", []):
        lines.append(
            f"- {row.get('compartment')}: dendrites={row.get('n_dendrite_observations', 'n/a')} | "
            f"spines={row.get('n_spine_observations', 'n/a')} | dendrite freq={format_report_number(row.get('mean_dendrite_event_frequency_per_min'))} | "
            f"spine freq={format_report_number(row.get('mean_spine_event_frequency_per_min'))} | coincident freq={format_report_number(row.get('mean_coincident_event_frequency_per_min'))} | "
            f"noncoincident freq={format_report_number(row.get('mean_noncoincident_event_frequency_per_min'))} | coincident fraction={format_report_number(row.get('mean_coincident_event_fraction'))}"
        )
    append_section("Significance legend")
    lines.append(f"- significance threshold: shuffle_p < {REPORT_SIGNIFICANCE_ALPHA:g}")
    lines.append("- state comparisons, basal-vs-apical comparisons, correlations, and spine-spine matrix summaries use shuffle p-values as the primary significance criterion")
    lines.append("- mixed-model fixed effects use classical p-values as the primary significance criterion")
    lines.append(f"- mixed-model contrasts use the configured p-value source: selected_state={mixed_selected_summary.get('p_value_source', 'classical')}")
    append_section("Tests performed")
    append_kv(
        "state comparison tests",
        (
            "dendrite_mean: "
            + (format_report_list(state_test_pairs) if state_test_pairs else "none")
            + "; spine_specific_mean: "
            + (format_report_list(state_test_pairs) if state_test_pairs else "none")
            + "; dendrite_event_frequency_per_min: "
            + (format_report_list(state_test_pairs) if state_test_pairs else "none")
            + "; spine_event_frequency_per_min: "
            + (format_report_list(state_test_pairs) if state_test_pairs else "none")
            + "; coincident_event_frequency_per_min: "
            + (format_report_list(state_test_pairs) if state_test_pairs else "none")
            + "; noncoincident_event_frequency_per_min: "
            + (format_report_list(state_test_pairs) if state_test_pairs else "none")
        ),
    )
    append_kv(
        "basal/apical tests",
        (
            "dendrite_mean: "
            + (format_report_list(basal_apical_test_labels) if basal_apical_test_labels else "none")
            + "; spine_specific_mean: "
            + (format_report_list(basal_apical_test_labels) if basal_apical_test_labels else "none")
        ),
    )
    append_kv(
        "correlation tests",
        "dendrite vs wheel, dendrite vs pupil, spine vs dendrite (raw), spine vs dendrite (specific)",
    )
    append_kv("matrix similarity tests", format_report_list(state_test_pairs) if state_test_pairs else "none")
    append_kv("direct trial-type fixed results", f"{direct_trial_type_summary.get('tested_videos', 0)} videos, {direct_trial_type_summary.get('tested_state_pairs', 0)} state pairs")
    append_kv("mixed-model fixed-effect terms", "see Model diagnostics and the Mixed-model fixed effects sections for selected_state")
    append_kv("mixed-model contrasts", "state_pair and basal-vs-apical contrasts from the selected_state branch")
    append_kv(
        "mixed-model p-value source",
        f"selected_state={mixed_selected_summary.get('p_value_source', 'classical')}",
    )
    append_kv(
        "spine coactivity mixed-model p-value source",
        spine_coactivity_summary.get('p_value_source', 'classical') if spine_coactivity_summary.get('model_enabled') else 'disabled',
    )
    lines.append(f"- quiet-awake-movies anchor selections use shuffle_significant and abs(coactivity_r) >= {format_report_number(spine_coactivity_abs_threshold)}")
    for branch_name, branch_summary in [("selected_state", mixed_selected_summary)]:
        for response_name, terms in sorted((branch_summary.get("tested_terms_by_response") or {}).items()):
            lines.append(f"- {branch_name} {response_name}: {format_report_list(terms) if terms else 'none'}")
    append_section("Quality / exclusions")
    append_kv("missing sleep_state.pickle", format_report_list(quality_summary.get("missing_sleep")) if quality_summary.get("missing_sleep") else "none")
    append_kv("missing wheel traces", format_report_list(quality_summary.get("missing_wheel")) if quality_summary.get("missing_wheel") else "none")
    append_kv("missing pupil traces", format_report_list(quality_summary.get("missing_pupil")) if quality_summary.get("missing_pupil") else "none")
    append_kv("insufficient spine observations", quality_summary.get("insufficient_spines", 0))
    append_kv("REM-gated pooled days skipped", analysis_cache.get("day_pooling", {}).get("rem_gate_skipped_count", 0))
    append_kv("direct trial-type excluded rows", direct_trial_type.get("table_checks", {}).get("n_candidate_rows", 0) - direct_trial_type.get("table_checks", {}).get("n_rows", 0))
    append_kv("skipped states / contrast terms", format_report_list(quality_summary.get("skipped_states")) if quality_summary.get("skipped_states") else "none")
    append_kv("mixed-model fallback reasons", format_report_list(quality_summary.get("mixed_fallback_reasons")) if quality_summary.get("mixed_fallback_reasons") else "none")
    append_kv("mixed-model fallback used", "yes" if quality_summary.get("mixed_fallback_used") else "no")
    append_section("Outputs written")
    table_artifacts = [
        artifact
        for artifact in report_artifacts
        if artifact not in {cache_rel, report_rel}
        and not artifact.startswith("figures/")
        and not artifact.startswith("checkpoint_examples/")
        and not artifact.startswith("review_figures/")
    ]
    figure_artifacts = [artifact for artifact in report_artifacts if artifact.startswith("figures/")]
    checkpoint_gallery_artifacts = [artifact for artifact in report_artifacts if artifact.startswith("checkpoint_examples/")]
    review_figure_artifacts = [artifact for artifact in report_artifacts if artifact.startswith("review_figures/")]
    cache_artifacts = [artifact for artifact in report_artifacts if artifact in {cache_rel, report_rel}]
    if table_artifacts:
        lines.append("- analysis tables:")
        for artifact in table_artifacts:
            lines.append(f"  - {artifact}")
    if figure_artifacts:
        lines.append("- figures:")
        for artifact in figure_artifacts:
            lines.append(f"  - {artifact}")
    if checkpoint_gallery_artifacts:
        lines.append("- checkpoint gallery:")
        for artifact in checkpoint_gallery_artifacts:
            lines.append(f"  - {artifact}")
    if review_figure_artifacts:
        lines.append("- review figures:")
        for artifact in review_figure_artifacts:
            lines.append(f"  - {artifact}")
    if cache_artifacts:
        lines.append("- cache/report:")
        for artifact in cache_artifacts:
            lines.append(f"  - {artifact}")
    if not (table_artifacts or figure_artifacts or checkpoint_gallery_artifacts or review_figure_artifacts or cache_artifacts):
        lines.append("- none")
    render_state_section(
        "Pairwise state comparisons",
        state_rows,
        format_state_row,
        group_key_fn=lambda row: str(row.get("metric", "unknown")),
        tested_label="tested day comparisons",
    )
    render_state_section(
        "Basal-vs-apical comparisons",
        basal_rows,
        format_basal_row,
        group_key_fn=lambda row: str(row.get("metric", "unknown")),
        tested_label="tested day comparisons",
    )
    render_state_section(
        "Calcium event-frequency state comparisons",
        [row for row in state_rows if str(row.get("metric")) in {
            "dendrite_event_frequency_per_min",
            "spine_event_frequency_per_min",
            "coincident_event_frequency_per_min",
            "noncoincident_event_frequency_per_min",
        }],
        format_state_row,
        group_key_fn=lambda row: str(row.get("metric", "unknown")),
        tested_label="tested day comparisons",
    )
    roi_split_rows = list(roi_split.get("comparison_rows", []))
    if roi_split_rows:
        append_section("ROI split comparisons")
        append_kv("tested comparisons", len(roi_split_rows))
        append_kv("tested subject-state rows", len(roi_split.get("subject_state_rows", [])))
        append_kv("tested split bundles", len(roi_split.get("bundles", [])))
        append_kv("branches", format_report_list(sorted({str(row.get("branch_name") or "") for row in roi_split_rows if str(row.get("branch_name") or "")})))
        append_kv("bases", format_report_list(sorted({str(row.get("basis_name") or "") for row in roi_split_rows if str(row.get("basis_name") or "")})))
        append_kv("roi types", format_report_list(sorted({str(row.get("roi_type") or "") for row in roi_split_rows if str(row.get("roi_type") or "")})))
        append_kv("split names", format_report_list(sorted({str(row.get("split_name") or "") for row in roi_split_rows if str(row.get("split_name") or "")})))
        append_kv("compartments", format_report_list(sorted({str(row.get("compartment") or "") for row in roi_split_rows if str(row.get("compartment") or "")})))
        append_kv("significant comparisons", sum(1 for row in roi_split_rows if is_significant_row(row, p_key="shuffle_p")))
        render_state_section(
            "ROI split group comparisons",
            roi_split_rows,
            format_roi_split_row,
            group_key_fn=lambda row: f"{row.get('branch_name', 'unknown')} | {row.get('basis_name', 'unknown')} | {row.get('roi_type', 'unknown')} | {row.get('compartment', 'unknown')} | {row.get('split_name', 'unknown')} | {row.get('window', 'unknown')}",
            p_key="shuffle_p",
            tested_label="tested comparisons",
        )
    append_section("Direct trial-type comparison")
    append_kv("tested videos", direct_trial_type_summary.get('tested_videos', 0))
    append_kv("tested animals", direct_trial_type_summary.get('tested_animals', 0))
    append_kv("tested animal-video-state rows", direct_trial_type_summary.get('tested_animal_video_state_rows', 0))
    append_kv("tested state pairs", direct_trial_type_summary.get('tested_state_pairs', 0))
    append_kv("significant state pairs", direct_trial_type_summary.get('significant_state_pairs', 0))
    append_kv("mean effect size", format_report_number(direct_trial_type_summary.get('mean_effect_size')))
    append_kv("mean video agreement r", format_report_number(direct_trial_type_summary.get('mean_agreement_r')))
    render_state_section(
        "Direct trial-type state comparisons",
        direct_trial_type_summary.get('state_pair_rows', []),
        format_direct_trial_row,
        group_key_fn=lambda row: f"{row.get('state_a', 'unknown')} vs {row.get('state_b', 'unknown')}",
        p_key="shuffle_p",
        tested_label="tested state pairs",
    )
    render_correlation_summary_section(correlation_summary_rows)
    render_matrix_summary_section(matrix_summary_rows)
    if mixed_summary.get("model_enabled"):
        render_state_section(
            "Mixed-model fixed effects - selected_state",
            mixed_summary.get("summary_rows", {}).get("mean_dendrite_activity", [])
            + mixed_summary.get("summary_rows", {}).get("mean_spine_activity_per_dendrite", []),
            format_mixed_model_summary_row,
            group_key_fn=lambda row: str(row.get("model_name", "unknown")),
            p_key="p_value",
            tested_label="tested terms",
        )
        render_state_section(
            "Mixed-model contrasts - selected_state",
            mixed_summary.get("contrast_rows", []),
            format_mixed_model_contrast_row,
            group_key_fn=lambda row: str(row.get("response", "unknown")),
            p_key="shuffle_p",
            p_label=f"{mixed_summary.get('p_value_source', 'classical')} p",
            tested_label="tested contrasts",
        )
    if mixed_selected_summary.get("model_enabled"):
        render_state_section(
            "Mixed-model fixed effects - selected_state",
            mixed_selected_summary.get("summary_rows", {}).get("mean_dendrite_activity", [])
            + mixed_selected_summary.get("summary_rows", {}).get("mean_spine_activity_per_dendrite", []),
            format_mixed_model_summary_row,
            group_key_fn=lambda row: str(row.get("model_name", "unknown")),
            p_key="p_value",
            tested_label="tested terms",
        )
        render_state_section(
            "Mixed-model contrasts - selected_state",
            mixed_selected_summary.get("contrast_rows", []),
            format_mixed_model_contrast_row,
            group_key_fn=lambda row: str(row.get("response", "unknown")),
            p_key="shuffle_p",
            p_label=f"{mixed_selected_summary.get('p_value_source', 'classical')} p",
            tested_label="tested contrasts",
        )
    render_validation_summary(demo_validation, mixed_summary)
    append_section("Alerts")
    if alerts:
        for alert in alerts:
            lines.append(f"- {alert}")
    else:
        lines.append("- none")
    write_text_report(report_path, lines)
def write_analysis_outputs(
    output_dir: Path,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    *,
    source_cache: Optional[Dict[str, Any]] = None,
    figure_root: Optional[Path] = None,
    plots_only: bool = False,
    include_supporting_figures: bool = True,
    branch_first_figures: bool = False,
    branch_first_output_root: Optional[Path] = None,
) -> List[str]:
    ensure_dir(output_dir)
    written_artifacts: List[str] = []
    if include_supporting_figures:
        # Save figures first so the JSON report can include their exact file paths.
        step_message("figure generation starting; this may take a while")
        with step_scope("figure generation"):
            figure_files = generate_analysis_figures(output_dir, results, cache, figure_root=figure_root)
        results["figure_files"] = figure_files
        for path in figure_files:
            written_artifacts.append(report_relative_path(path, output_dir))
        step_message("figure generation complete: %d file(s)" % len(figure_files))
        step_message("review figure generation starting")
        with step_scope("review figure generation"):
            review_figure_files = generate_review_figures(output_dir, results, cache, review_root=DEFAULT_REVIEW_FIGURES_DIR)
        results["review_figure_files"] = review_figure_files
        for path in review_figure_files:
            written_artifacts.append(report_relative_path(path, ROOT_DIR))
        step_message("review figure generation complete: %d file(s)" % len(review_figure_files))
        run_params = results.get("run_parameters", {}) if isinstance(results.get("run_parameters"), dict) else {}
        comparison_preset_name = str(run_params.get("comparison_preset_name") or "default")
        generate_shared_general_figures = bool(run_params.get("generate_shared_general_figures", True))
        shared_general_root = None
        if generate_shared_general_figures:
            base_root = Path(figure_root) if figure_root is not None else (output_dir / "figures")
            shared_general_root = ensure_dir(base_root / DEFAULT_SHARED_FIGURES_DIRNAME)
        if source_cache is not None and shared_general_root is not None:
            step_message("visual response figure generation starting")
            with step_scope("visual response figure generation"):
                visual_response_figure_files = render_cached_visual_response_figures(
                    output_dir,
                    results,
                    cache,
                    source_cache,
                    figure_root=shared_general_root,
                )
            results["visual_response_figure_files"] = visual_response_figure_files
            for path in visual_response_figure_files:
                written_artifacts.append(report_relative_path(path, output_dir))
            step_message("visual response figure generation complete: %d file(s)" % len(visual_response_figure_files))
            step_message("event detection example gallery starting")
            with step_scope("event detection example gallery"):
                event_example_gallery = generate_event_detection_example_gallery(cache, shared_general_root)
            results["event_example_gallery"] = event_example_gallery
            for path in event_example_gallery:
                written_artifacts.append(report_relative_path(path, output_dir))
            step_message("event detection example gallery complete: %d file(s)" % len(event_example_gallery))
        else:
            results["visual_response_figure_files"] = []
            results["event_example_gallery"] = []
        step_message("checkpoint gallery generation starting")
        with step_scope("checkpoint gallery"):
            checkpoint_gallery = generate_checkpoint_gallery(output_dir, cache, results)
        results["checkpoint_gallery"] = checkpoint_gallery
        if checkpoint_gallery.get("manifest_path"):
            written_artifacts.append(report_relative_path(checkpoint_gallery["manifest_path"], output_dir))
        for path in checkpoint_gallery.get("files", []):
            written_artifacts.append(report_relative_path(path, output_dir))
        step_message("checkpoint gallery complete: %d file(s)" % len(checkpoint_gallery.get("files", [])))
        if branch_first_figures:
            from analysis.shared.analysis_families.mixed_model import run_split_family

            branch_root = Path(branch_first_output_root) if branch_first_output_root is not None else output_dir
            mixed_model_results = results.get("mixed_model_selected_state") or results.get("mixed_model") or {}
            sleep_expids = results.get("analysis_sleep_expids", []) if isinstance(results.get("analysis_sleep_expids"), list) else []
            analysis_state_selection = results.get("analysis_state_selection", {}) if isinstance(results.get("analysis_state_selection", {}), dict) else {}
            analysis_state_comparison_states = list(analysis_state_selection.get("state_comparison_states") or [])
            shuffle_n = int(run_params.get("shuffle_n", DEFAULT_SHUFFLES) or DEFAULT_SHUFFLES)
            mixed_model_contrast_p_source = str(results.get("mixed_model_contrast_p_source") or run_params.get("mixed_model_contrast_p_source") or "classical")
            split_response_columns = list((mixed_model_results.get("summary_rows") or {}).keys()) if isinstance(mixed_model_results, dict) else []
            for branch_name, basis_name in iter_branch_basis_leaves(ANALYSIS_BRANCHES, ANALYSIS_BASES):
                leaf_results = scoped_branch_results(results, branch_name=branch_name, basis_name=basis_name, sleep_expids=sleep_expids)
                leaf_split_mixed_model = {}
                try:
                    leaf_split_mixed_model = run_split_family(
                        mixed_model_results.get("table_rows", []) if isinstance(mixed_model_results, dict) else [],
                        leaf_results.get("roi_split", {}).get("membership_rows", []) if isinstance(leaf_results.get("roi_split", {}), dict) else [],
                        response_columns=split_response_columns,
                        state_comparison_states=list(leaf_results.get("analysis_state_selection", {}).get("state_comparison_states", analysis_state_comparison_states)),
                        shuffle_n=shuffle_n,
                        mixed_model_contrast_p_source=mixed_model_contrast_p_source,
                        state_filter=list(leaf_results.get("analysis_state_selection", {}).get("state_comparison_states", analysis_state_comparison_states)),
                        vc_level_keys=None,
                    )
                except Exception:
                    logger.exception("Failed to build split mixed-model leaf results for %s/%s", branch_name, basis_name)
                    leaf_split_mixed_model = {}
                if leaf_split_mixed_model.get("available"):
                    leaf_results["mixed_model"] = leaf_split_mixed_model
                    leaf_results["mixed_model_selected_state"] = leaf_split_mixed_model
                leaf_output_dir = branch_leaf_root(branch_root, branch_name, basis_name)
                leaf_figure_root = branch_leaf_figure_root(branch_root, branch_name, basis_name)
                step_message(f"branch-first figure generation: {branch_name}/{basis_name}")
                with step_scope(f"branch-first figure generation: {branch_name}/{basis_name}"):
                    leaf_written = write_analysis_outputs(
                        leaf_output_dir,
                        leaf_results,
                        cache,
                        source_cache=source_cache,
                        figure_root=leaf_figure_root,
                        plots_only=True,
                        include_supporting_figures=include_supporting_figures,
                        branch_first_figures=False,
                        branch_first_output_root=branch_root,
                    )
                for path in leaf_written:
                    leaf_path = Path(leaf_output_dir) / str(path)
                    written_artifacts.append(report_relative_path(leaf_path, output_dir))
    else:
        results["figure_files"] = []
        results["review_figure_files"] = []
        results["visual_response_figure_files"] = []
        results["checkpoint_gallery"] = {}
        results["event_example_gallery"] = []
    if plots_only:
        results["output_root"] = str(output_dir)
        results["output_artifacts"] = list(dict.fromkeys(written_artifacts))
        write_manifest(output_dir, jsonable(results))
        return list(dict.fromkeys(written_artifacts))
    json_path = output_dir / "analysis_results.json"
    with json_path.open("w") as handle:
        json.dump(jsonable(results), handle, indent=2, sort_keys=True)
    written_artifacts.append(report_relative_path(json_path, output_dir))
    state_rows: List[Dict[str, Any]] = []
    for row in results.get("state_comparisons", []):
        state_rows.append(row)
    for row in results.get("basal_apical_comparisons", []):
        state_rows.append(row)
    if state_rows:
        fieldnames = sorted({key for row in state_rows for key in row.keys()})
        state_csv = output_dir / "state_comparisons.csv"
        write_csv_rows(state_csv, state_rows, fieldnames)
        written_artifacts.append(report_relative_path(state_csv, output_dir))
    if results.get("correlations"):
        fieldnames = sorted({key for row in results["correlations"] for key in row.keys()})
        correlations_csv = output_dir / "correlations.csv"
        write_csv_rows(correlations_csv, results["correlations"], fieldnames)
        written_artifacts.append(report_relative_path(correlations_csv, output_dir))
    if results.get("matrix_similarity"):
        fieldnames = sorted({key for row in results["matrix_similarity"] for key in row.keys()})
        matrix_csv = output_dir / "matrix_similarity.csv"
        write_csv_rows(matrix_csv, results["matrix_similarity"], fieldnames)
        written_artifacts.append(report_relative_path(matrix_csv, output_dir))
    if results.get("demo_validation"):
        fieldnames = sorted({key for row in results["demo_validation"] for key in row.keys()})
        demo_csv = output_dir / "demo_validation.csv"
        write_csv_rows(demo_csv, results["demo_validation"], fieldnames)
        written_artifacts.append(report_relative_path(demo_csv, output_dir))
    direct_trial_type = results.get("direct_trial_type_comparison", {})
    direct_trial_type = direct_trial_type if isinstance(direct_trial_type, dict) else {}
    if direct_trial_type:
        if direct_trial_type.get("table_rows"):
            fieldnames = sorted({key for row in direct_trial_type["table_rows"] for key in row.keys()})
            direct_table_csv = output_dir / "direct_trial_type_table.csv"
            write_csv_rows(direct_table_csv, direct_trial_type["table_rows"], fieldnames)
            written_artifacts.append(report_relative_path(direct_table_csv, output_dir))
        for key, output_name in [
            ("animal_video_state_rows", "direct_trial_type_animal_video_state_summary.csv"),
            ("video_state_rows", "direct_trial_type_video_state_summary.csv"),
            ("state_summary_rows", "direct_trial_type_state_summary.csv"),
            ("state_pair_rows", "direct_trial_type_state_comparisons.csv"),
            ("overall_summary_rows", "direct_trial_type_overall_summary.csv"),
        ]:
            rows = direct_trial_type.get(key, [])
            if not rows:
                continue
            fieldnames = sorted({field for row in rows for field in row.keys()})
            csv_path = output_dir / output_name
            write_csv_rows(csv_path, rows, fieldnames)
            written_artifacts.append(report_relative_path(csv_path, output_dir))
    mixed_model = results.get("mixed_model", {})
    mixed_model_selected = results.get("mixed_model_selected_state", {})
    mixed_model_visual_response_responsive = results.get("mixed_model_visual_response_responsive", {})
    mixed_model_visual_response_nonresponsive = results.get("mixed_model_visual_response_nonresponsive", {})
    mixed_model_visual_response_responsive_selected_state = results.get("mixed_model_visual_response_responsive_selected_state", {}) or results.get("mixed_model_visual_response_responsive", {})
    mixed_model_visual_response_nonresponsive_selected_state = results.get("mixed_model_visual_response_nonresponsive_selected_state", {}) or results.get("mixed_model_visual_response_nonresponsive", {})
    mixed_model_branches = [
        ("mixed_model", mixed_model, ""),
        ("mixed_model_selected_state", mixed_model_selected, "selected_state"),
        ("mixed_model_visual_response_responsive", mixed_model_visual_response_responsive, "visual_response_responsive_selected_state"),
        ("mixed_model_visual_response_nonresponsive", mixed_model_visual_response_nonresponsive, "visual_response_nonresponsive_selected_state"),
        ("mixed_model_visual_response_responsive_selected_state", mixed_model_visual_response_responsive_selected_state, "visual_response_responsive_selected_state"),
        ("mixed_model_visual_response_nonresponsive_selected_state", mixed_model_visual_response_nonresponsive_selected_state, "visual_response_nonresponsive_selected_state"),
    ]
    for branch_key, branch_data, branch_suffix in mixed_model_branches:
        if not isinstance(branch_data, dict) or not branch_data:
            continue
        if branch_key == "mixed_model" and branch_data.get("table_rows"):
            fieldnames = sorted({key for row in branch_data["table_rows"] for key in row.keys()})
            mixed_table_csv = output_dir / "mixed_model_table.csv"
            write_csv_rows(mixed_table_csv, branch_data["table_rows"], fieldnames)
            written_artifacts.append(report_relative_path(mixed_table_csv, output_dir))
        summary_rows = branch_data.get("summary_rows", {})
        for response_name, rows in summary_rows.items():
            if not rows:
                continue
            fieldnames = sorted({key for row in rows for key in row.keys()})
            if branch_key == "mixed_model":
                output_name = f"mixed_model_summary_{response_name}.csv"
            else:
                output_name = f"mixed_model_{branch_suffix}_summary_{response_name}.csv"
            summary_csv = output_dir / output_name
            write_csv_rows(summary_csv, rows, fieldnames)
            written_artifacts.append(report_relative_path(summary_csv, output_dir))
        if branch_data.get("contrast_rows"):
            fieldnames = sorted({key for row in branch_data["contrast_rows"] for key in row.keys()})
            if branch_key == "mixed_model":
                output_name = "mixed_model_contrasts.csv"
            else:
                output_name = f"mixed_model_contrasts_{branch_suffix}.csv"
            contrast_csv = output_dir / output_name
            write_csv_rows(contrast_csv, branch_data["contrast_rows"], fieldnames)
            written_artifacts.append(report_relative_path(contrast_csv, output_dir))
    roi_split = results.get("roi_split", {})
    if isinstance(roi_split, dict):
        for key, output_name in [
            ("subject_state_rows", "roi_split_subject_state.csv"),
            ("membership_rows", "roi_split_membership.csv"),
            ("comparison_rows", "roi_split_comparisons.csv"),
            ("summary_rows", "roi_split_summary.csv"),
        ]:
            rows = roi_split.get(key, [])
            if not rows:
                continue
            fieldnames = sorted({field for row in rows for field in row.keys()})
            csv_path = output_dir / output_name
            write_csv_rows(csv_path, rows, fieldnames)
            written_artifacts.append(report_relative_path(csv_path, output_dir))
    spine_coactivity = results.get("spine_coactivity", {})
    if isinstance(spine_coactivity, dict):
        property_split = spine_coactivity.get("property_split", {})
        if isinstance(property_split, dict):
            for key, output_name in [
                ("subject_state_rows", "spine_coactivity_property_split_subject_state.csv"),
                ("membership_rows", "spine_coactivity_property_split_membership.csv"),
                ("comparison_rows", "spine_coactivity_property_split_comparisons.csv"),
                ("summary_rows", "spine_coactivity_property_split_summary.csv"),
            ]:
                rows = property_split.get(key, [])
                if not rows:
                    continue
                fieldnames = sorted({field for row in rows for field in row.keys()})
                csv_path = output_dir / output_name
                write_csv_rows(csv_path, rows, fieldnames)
                written_artifacts.append(report_relative_path(csv_path, output_dir))
    if isinstance(spine_coactivity, dict):
        if spine_coactivity.get("table_rows"):
            fieldnames = sorted({key for row in spine_coactivity["table_rows"] for key in row.keys()})
            coactivity_table_csv = output_dir / "spine_coactivity_table.csv"
            write_csv_rows(coactivity_table_csv, spine_coactivity["table_rows"], fieldnames)
            written_artifacts.append(report_relative_path(coactivity_table_csv, output_dir))
        if spine_coactivity.get("pair_state_rows"):
            fieldnames = sorted({key for row in spine_coactivity["pair_state_rows"] for key in row.keys()})
            pair_state_csv = output_dir / "spine_coactivity_pair_state_rows.csv"
            write_csv_rows(pair_state_csv, spine_coactivity["pair_state_rows"], fieldnames)
            written_artifacts.append(report_relative_path(pair_state_csv, output_dir))
        for key, output_name in [
            ("state_summary_rows", "spine_coactivity_state_summary.csv"),
            ("pair_summary_rows", "spine_coactivity_pair_summary.csv"),
            ("animal_state_rows", "spine_coactivity_animal_state_summary.csv"),
            ("state_agreement_rows", "spine_coactivity_state_agreement.csv"),
            ("compartment_summary_rows", "spine_coactivity_compartment_summary.csv"),
        ]:
            rows = spine_coactivity.get(key, [])
            if not rows:
                continue
            fieldnames = sorted({field for row in rows for field in row.keys()})
            csv_path = output_dir / output_name
            write_csv_rows(csv_path, rows, fieldnames)
            written_artifacts.append(report_relative_path(csv_path, output_dir))
    spine_coactivity_model = results.get("spine_coactivity_model", {})
    spine_coactivity_model = spine_coactivity_model if isinstance(spine_coactivity_model, dict) else {}
    if isinstance(spine_coactivity_model, dict):
        if spine_coactivity_model.get("summary_rows", {}).get("coactivity_r"):
            rows = spine_coactivity_model["summary_rows"]["coactivity_r"]
            fieldnames = sorted({key for row in rows for key in row.keys()})
            csv_path = output_dir / "spine_coactivity_model_summary_coactivity_r.csv"
            write_csv_rows(csv_path, rows, fieldnames)
            written_artifacts.append(report_relative_path(csv_path, output_dir))
        if spine_coactivity_model.get("contrast_rows"):
            fieldnames = sorted({key for row in spine_coactivity_model["contrast_rows"] for key in row.keys()})
            csv_path = output_dir / "spine_coactivity_model_contrasts.csv"
            write_csv_rows(csv_path, spine_coactivity_model["contrast_rows"], fieldnames)
            written_artifacts.append(report_relative_path(csv_path, output_dir))
    return list(dict.fromkeys(written_artifacts))


def write_poster_ready_figures(
    output_dir: Path,
    cache: Dict[str, Any],
    source_cache: Dict[str, Any],
    results: Dict[str, Any],
    analysis_families: Optional[Sequence[str]] = None,
) -> List[str]:
    from posters.sleep_dendrite_spine_poster_figure import (
        DEFAULT_HEIGHT_CM as MIXED_POSTER_HEIGHT_CM,
        DEFAULT_OUTPUT_STEM as MIXED_POSTER_OUTPUT_STEM,
        DEFAULT_WIDTH_CM as MIXED_POSTER_WIDTH_CM,
        write_mixed_model_poster_figure,
    )
    from posters.sleep_dendrite_spine_spine_coactivity_poster_figure import (
        DEFAULT_SPINE_COACTIVITY_HEIGHT_CM,
        DEFAULT_SPINE_COACTIVITY_OUTPUT_STEM,
        DEFAULT_SPINE_COACTIVITY_WIDTH_CM,
        write_spine_coactivity_poster_figure,
    )
    from analysis.shared.plots.poster_ready import (
        _assign_visual_response_cohorts,
        _compartment_comparison_state_labels_from_comparison_rows,
        _poster_mixed_model_significant_states,
        _select_mixed_model_rows,
        _significant_state_labels_from_comparison_rows,
        write_blank_movie_state_boxplot_figure,
        write_state_mixed_model_poster_figure,
        write_visual_response_poster_figure,
    )

    poster_output_dir = ensure_dir(ROOT_DIR / "results" / "poster_ready")
    poster_result_root = Path(results.get("output_root") or output_dir)

    def _load_preset_csv_rows(preset_name: str, csv_name: str) -> List[Dict[str, Any]]:
        return load_comparison_preset_csv_rows(poster_result_root, preset_name, csv_name, logger=logger)

    written: List[str] = []
    selected_families = set(str(family) for family in (analysis_families or []) if str(family))
    allow_all_families = not selected_families

    if allow_all_families or "mixed_model" in selected_families:
        with step_scope("poster figure generation: mixed_model"):
            mixed_path = write_mixed_model_poster_figure(
            cache,
            poster_output_dir,
            results=results,
            output_stem=MIXED_POSTER_OUTPUT_STEM,
            width_cm=float(MIXED_POSTER_WIDTH_CM),
            height_cm=float(MIXED_POSTER_HEIGHT_CM),
        )
        written.append(report_relative_path(mixed_path, output_dir))

    if allow_all_families or "spine_coactivity" in selected_families:
        with step_scope("poster figure generation: spine_coactivity"):
            coactivity_path = write_spine_coactivity_poster_figure(
            cache,
            poster_output_dir,
            results=results,
            output_stem=DEFAULT_SPINE_COACTIVITY_OUTPUT_STEM,
            width_cm=float(DEFAULT_SPINE_COACTIVITY_WIDTH_CM),
            height_cm=float(DEFAULT_SPINE_COACTIVITY_HEIGHT_CM),
        )
        written.append(report_relative_path(coactivity_path, output_dir))

    if allow_all_families or "poster_ready" in selected_families or not selected_families:
        with step_scope("poster figure generation: shared poster ready"):
            def _combined_state_values(cohort_payload: Any, metric_key: str) -> Dict[str, List[float]]:
                combined: Dict[str, List[float]] = {}
                if not isinstance(cohort_payload, dict):
                    return combined
                for compartment_payload in cohort_payload.values():
                    if not isinstance(compartment_payload, dict):
                        continue
                    metric_summary = compartment_payload.get("state_summaries", {}).get(metric_key, {})
                    if not isinstance(metric_summary, dict):
                        continue
                    for state, by_subject in metric_summary.items():
                        state_values = flatten_state_summary_values(by_subject)
                        arr = np.asarray(state_values, dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size:
                            combined.setdefault(str(state), []).extend([float(value) for value in arr])
                return combined

            def _state_values_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
                grouped: Dict[str, List[float]] = {}
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
                    grouped.setdefault(state, []).append(value_f)
                return grouped

            def _state_sample_sizes_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
                grouped: Dict[str, set[str]] = {}
                for row in rows:
                    state = canonical_state_label(row.get("state") or row.get("state_label") or row.get("state_display") or "")
                    if not state:
                        continue
                    roi = str(
                        row.get("global_soma_id")
                        or row.get("global_bouton_id")
                        or row.get("roi_key")
                        or row.get("global_spine_id")
                        or row.get("global_dendrite_id")
                        or row.get("roi_id")
                        or row.get("soma_id")
                        or row.get("bouton_id")
                        or row.get("roi_index")
                        or row.get("entity_id")
                        or ""
                    ).strip()
                    if not roi:
                        continue
                    grouped.setdefault(state, set()).add(roi)
                return {state: len(roi_ids) for state, roi_ids in grouped.items()}

            def _entity_state_values(entity_key: str, metric_key: str) -> Dict[str, Dict[str, List[float]]]:
                summary = results.get(f"{entity_key}_visual_response_state_summaries", {})
                if not isinstance(summary, dict):
                    return {"responsive": {}, "nonresponsive": {}}
                if entity_key != "dendrite":
                    return {
                        "responsive": _combined_state_values(summary.get("responsive", {}), metric_key),
                        "nonresponsive": _combined_state_values(summary.get("nonresponsive", {}), metric_key),
                    }
                def _compartment_state_values(cohort_payload: Any) -> Dict[str, List[float]]:
                    combined: Dict[str, List[float]] = {}
                    if not isinstance(cohort_payload, dict):
                        return combined
                    for compartment, compartment_payload in cohort_payload.items():
                        if not isinstance(compartment_payload, dict):
                            continue
                        metric_summary = compartment_payload.get("state_summaries", {}).get(metric_key, {})
                        if not isinstance(metric_summary, dict):
                            continue
                        for state, by_subject in metric_summary.items():
                            state_values = flatten_state_summary_values(by_subject)
                            arr = np.asarray(state_values, dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if arr.size:
                                combined.setdefault(f"{compartment}_{state}", []).extend([float(value) for value in arr])
                    return combined
                return {
                    "responsive": _compartment_state_values(summary.get("responsive", {})),
                    "nonresponsive": _compartment_state_values(summary.get("nonresponsive", {})),
                }

            def _collapse_dendrite_state_values(state_map: Dict[str, List[float]]) -> Dict[str, List[float]]:
                collapsed: Dict[str, List[float]] = {}
                for state, values in state_map.items():
                    state_key = canonical_state_label(state)
                    for prefix in ("basal_", "apical_"):
                        if state_key.startswith(prefix):
                            state_key = state_key[len(prefix):]
                            break
                    collapsed.setdefault(state_key, []).extend(list(values))
                return collapsed

            def _dendrite_visual_summary_state_values(metric_key: str) -> Dict[str, Dict[str, List[float]]]:
                summary = results.get("dendrite_visual_response_state_summaries", {})
                if not isinstance(summary, dict):
                    return {"responsive": {}, "nonresponsive": {}}
                combined: Dict[str, Dict[str, List[float]]] = {"responsive": {}, "nonresponsive": {}}
                for cohort in ("responsive", "nonresponsive"):
                    cohort_payload = summary.get(cohort, {})
                    if not isinstance(cohort_payload, dict):
                        continue
                    for compartment, compartment_payload in cohort_payload.items():
                        if not isinstance(compartment_payload, dict):
                            continue
                        metric_summary = compartment_payload.get("state_summaries", {}).get(metric_key, {})
                        if not isinstance(metric_summary, dict):
                            continue
                        for state, by_subject in metric_summary.items():
                            state_values = flatten_state_summary_values(by_subject)
                            arr = np.asarray(state_values, dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if arr.size:
                                combined[cohort].setdefault(f"{compartment}_{canonical_state_label(state)}", []).extend([float(value) for value in arr])
                return combined

            poster_state_order = selected_mixed_model_state_labels(results) or ["quiet_awake_blank", "quiet_awake_movies", "quiet_awake", "nrem", "rem"]
            blank_state_order = ["quiet_awake_blank", "nrem_blank", "rem_blank"]
            movie_state_order = ["quiet_awake_movies", "nrem_movies", "rem_movies"]

            mixed_model_contrast_p_source = str(results.get("mixed_model_contrast_p_source") or "classical")
            entity_specs = [
                {"entity_key": "dendrite", "entity_label": "dendrite", "metric_key": "dendrite_mean", "visual_payload": results.get("dendrite_visual_response", {}), "responsive_mixed": results.get("mixed_model_visual_response_responsive_selected_state", {}) or results.get("mixed_model_visual_response_responsive", {}), "nonresponsive_mixed": results.get("mixed_model_visual_response_nonresponsive_selected_state", {}) or results.get("mixed_model_visual_response_nonresponsive", {}), "visual_compartments": ("basal", "apical")},
                {"entity_key": "spine", "entity_label": "spine", "metric_key": "spine_specific_mean", "visual_payload": results.get("spine_visual_response", {}), "responsive_mixed": results.get("mixed_model_visual_response_responsive_selected_state", {}) or results.get("mixed_model_visual_response_responsive", {}), "nonresponsive_mixed": results.get("mixed_model_visual_response_nonresponsive_selected_state", {}) or results.get("mixed_model_visual_response_nonresponsive", {}), "visual_compartments": (None,)},
            ]
            for spec in entity_specs:
                entity_key = str(spec["entity_key"])
                entity_label = str(spec["entity_label"])
                metric_key = str(spec["metric_key"])
                visual_payload = spec["visual_payload"]
                responsive_mixed = spec["responsive_mixed"]
                nonresponsive_mixed = spec["nonresponsive_mixed"]
                visual_compartments = tuple(spec.get("visual_compartments") or (None,))
                entity_root = ensure_dir(poster_output_dir / entity_key)
                visual_dir = ensure_dir(entity_root / "visual_response")
                mixed_dir = ensure_dir(entity_root / "mixed_model")
                blank_dir = ensure_dir(entity_root / "blank_movie_states")
                rows = list(visual_payload.get("rows", [])) if isinstance(visual_payload, dict) else []
                blank_preset_activity_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("blank_state_comparisons", "state_activity_by_experiment.csv"), rows)
                movie_preset_activity_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("movies_state_comparisons", "state_activity_by_experiment.csv"), rows)
                blank_preset_comparison_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("blank_state_comparisons", "state_comparisons_movie.csv"), rows)
                movie_preset_comparison_rows = _assign_visual_response_cohorts(_load_preset_csv_rows("movies_state_comparisons", "state_comparisons_movie.csv"), rows)

                def _debug_preset_loaded(preset_name: str, csv_name: str, preset_rows: Sequence[Mapping[str, Any]]) -> None:
                    state_keys = list(dict.fromkeys(canonical_state_label(row.get("state") or row.get("state_label") or row.get("state_display") or "") for row in preset_rows if canonical_state_label(row.get("state") or row.get("state_label") or row.get("state_display") or "")))
                    compartment_keys = list(dict.fromkeys(canonical_state_label(row.get("compartment") or row.get("output_compartment") or "") for row in preset_rows if canonical_state_label(row.get("compartment") or row.get("output_compartment") or "")))
                    cohort_keys = list(dict.fromkeys(str(row.get("cohort") or "").strip().lower() for row in preset_rows if str(row.get("cohort") or "").strip()))
                    print(f"[poster-debug] loaded preset {preset_name}/{csv_name}: rows={len(preset_rows)} states={state_keys} compartments={compartment_keys} cohorts={cohort_keys}", file=sys.stderr)

                _debug_preset_loaded("blank_state_comparisons", "state_activity_by_experiment.csv", blank_preset_activity_rows)
                _debug_preset_loaded("movies_state_comparisons", "state_activity_by_experiment.csv", movie_preset_activity_rows)
                _debug_preset_loaded("blank_state_comparisons", "state_comparisons_movie.csv", blank_preset_comparison_rows)
                _debug_preset_loaded("movies_state_comparisons", "state_comparisons_movie.csv", movie_preset_comparison_rows)

                def _poster_compartment_matches(row_value: Any, candidates: Sequence[Any]) -> bool:
                    value = canonical_state_label(row_value)
                    if not value:
                        return False
                    for candidate in candidates:
                        candidate_key = canonical_state_label(candidate)
                        if not candidate_key:
                            continue
                        if value == candidate_key or value.startswith(f"{candidate_key}_") or candidate_key.startswith(f"{value}_"):
                            return True
                    return False

                if entity_key == "dendrite":
                    raw_blank_compartments = list(dict.fromkeys(str(row.get("compartment") or "").strip().lower() for row in blank_preset_activity_rows if str(row.get("compartment") or "").strip()))
                    raw_movie_compartments = list(dict.fromkeys(str(row.get("compartment") or "").strip().lower() for row in movie_preset_activity_rows if str(row.get("compartment") or "").strip()))
                    print(f"[poster-debug] dendrite raw blank preset compartments = {raw_blank_compartments}", file=sys.stderr)
                    print(f"[poster-debug] dendrite raw movie preset compartments = {raw_movie_compartments}", file=sys.stderr)
                    print(f"[poster-debug] dendrite preset counts blank_activity={len(blank_preset_activity_rows)} movie_activity={len(movie_preset_activity_rows)} blank_comparisons={len(blank_preset_comparison_rows)} movie_comparisons={len(movie_preset_comparison_rows)}", file=sys.stderr)
                preset_compartments = {str(compartment).strip().lower() for compartment in visual_compartments if str(compartment).strip() and str(compartment).strip().lower() != "none"}
                if not preset_compartments:
                    preset_compartments = {entity_key}
                if entity_key == "dendrite":
                    preset_compartments.update({"dendrite", "basal_dendrite", "apical_dendrite"})
                if rows:
                    if entity_key == "dendrite":
                        for compartment in visual_compartments:
                            compartment_rows = [row for row in rows if str(row.get("compartment") or "") == str(compartment)]
                            if not compartment_rows:
                                continue
                            compartment_dir = ensure_dir(visual_dir / str(compartment))
                            visual_path = write_visual_response_poster_figure(
                                output_dir=compartment_dir,
                                entity_label=f"{entity_label} {compartment}",
                                visual_response_rows=compartment_rows,
                                cache=cache,
                                source_cache=source_cache,
                                kind=entity_key,
                                output_stem=f"{entity_key}_{compartment}_visual_response_poster_ready",
                            )
                            if visual_path:
                                written.append(report_relative_path(Path(visual_path), output_dir))
                    else:
                        visual_path = write_visual_response_poster_figure(
                            output_dir=visual_dir,
                            entity_label=entity_label,
                            visual_response_rows=rows,
                            cache=cache,
                            source_cache=source_cache,
                            kind=entity_key,
                            output_stem=f"{entity_key}_visual_response_poster_ready",
                        )
                        if visual_path:
                            written.append(report_relative_path(Path(visual_path), output_dir))
                state_values = _entity_state_values(entity_key, metric_key)
                if entity_key == "dendrite":
                    dendrite_summary_state_values = _dendrite_visual_summary_state_values(metric_key)
                    if dendrite_summary_state_values.get("responsive") or dendrite_summary_state_values.get("nonresponsive"):
                        state_values = dendrite_summary_state_values
                if state_values.get("responsive") or state_values.get("nonresponsive"):
                    dendrite_state_order = poster_state_order
                    if entity_key == "dendrite":
                        dendrite_state_order = [f"basal_{state}" for state in poster_state_order] + [f"apical_{state}" for state in poster_state_order]
                    mixed_path = write_state_mixed_model_poster_figure(
                        output_dir=mixed_dir,
                        entity_label=entity_key,
                        responsive_state_values=state_values.get("responsive", {}),
                        nonresponsive_state_values=state_values.get("nonresponsive", {}),
                        mixed_model_rows={
                            "responsive": responsive_mixed,
                            "nonresponsive": nonresponsive_mixed,
                        },
                        state_order=dendrite_state_order,
                        output_stem=f"{entity_key}_state_mixed_model_poster_ready",
                        title="Quiet blank vs sleep states",
                        preferred_response_keys=(("mean_dendrite_activity", "mean") if entity_key == "dendrite" else ("mean_spine_activity_per_dendrite", "mean", "mean_dendrite_activity")),
                        mixed_model_contrast_p_source=mixed_model_contrast_p_source,
                    )
                    if mixed_path:
                        written.append(report_relative_path(Path(mixed_path), output_dir))
                blank_rows = [row for row in blank_preset_activity_rows if _poster_compartment_matches(row.get("compartment"), preset_compartments) and str(row.get("cohort") or "").strip().lower() == "responsive"]
                movie_rows = [row for row in movie_preset_activity_rows if _poster_compartment_matches(row.get("compartment"), preset_compartments) and str(row.get("cohort") or "").strip().lower() == "responsive"]
                non_blank_rows = [row for row in blank_preset_activity_rows if _poster_compartment_matches(row.get("compartment"), preset_compartments) and str(row.get("cohort") or "").strip().lower() != "responsive"]
                non_movie_rows = [row for row in movie_preset_activity_rows if _poster_compartment_matches(row.get("compartment"), preset_compartments) and str(row.get("cohort") or "").strip().lower() != "responsive"]
                responsive_state_values_for_blank = state_values.get("responsive", {})
                nonresponsive_state_values_for_blank = state_values.get("nonresponsive", {})
                if entity_key == "dendrite":
                    blank_values = responsive_state_values_for_blank
                    movie_values = responsive_state_values_for_blank
                    non_blank_values = nonresponsive_state_values_for_blank
                    non_movie_values = nonresponsive_state_values_for_blank
                else:
                    blank_values = _state_values_from_rows(blank_rows) if blank_rows else {state: values for state, values in responsive_state_values_for_blank.items() if state in blank_state_order}
                    movie_values = _state_values_from_rows(movie_rows) if movie_rows else {state: values for state, values in responsive_state_values_for_blank.items() if state in movie_state_order}
                    non_blank_values = _state_values_from_rows(non_blank_rows) if non_blank_rows else {state: values for state, values in nonresponsive_state_values_for_blank.items() if state in blank_state_order}
                    non_movie_values = _state_values_from_rows(non_movie_rows) if non_movie_rows else {state: values for state, values in nonresponsive_state_values_for_blank.items() if state in movie_state_order}
                debug_blank_summary = {
                    "entity": entity_key,
                    "blank_rows": len(blank_rows),
                    "movie_rows": len(movie_rows),
                    "non_blank_rows": len(non_blank_rows),
                    "non_movie_rows": len(non_movie_rows),
                    "responsive_blank_states": list(blank_values.keys()),
                    "responsive_movie_states": list(movie_values.keys()),
                    "nonresponsive_blank_states": list(non_blank_values.keys()),
                    "nonresponsive_movie_states": list(non_movie_values.keys()),
                }
                print(f"[poster-debug] blank/movie selection {debug_blank_summary}", file=sys.stderr)
                if entity_key == "dendrite":
                    print(f"[poster-debug] dendrite blank/movie source values responsive={list(blank_values.keys()) + list(movie_values.keys())} nonresponsive={list(non_blank_values.keys()) + list(non_movie_values.keys())}", file=sys.stderr)
                blank_sample_sizes = _state_sample_sizes_from_rows(blank_rows) if blank_rows else {}
                movie_sample_sizes = _state_sample_sizes_from_rows(movie_rows) if movie_rows else {}
                non_blank_sample_sizes = _state_sample_sizes_from_rows(non_blank_rows) if non_blank_rows else {}
                non_movie_sample_sizes = _state_sample_sizes_from_rows(non_movie_rows) if non_movie_rows else {}
                has_blank_movie_values = bool(blank_values or movie_values or non_blank_values or non_movie_values)
                if not has_blank_movie_values:
                    print(f"[poster-debug] skipping blank/movie plot for {entity_key}: no selected values after filtering", file=sys.stderr)
                if has_blank_movie_values:
                    responsive_significant_states = set()
                    nonresponsive_significant_states = set()
                    responsive_selected_rows = _select_mixed_model_rows(
                        responsive_mixed,
                        preferred_response_keys=("mean_dendrite_activity", "mean") if entity_key == "dendrite" else ("mean_spine_activity_per_dendrite", "mean", "mean_dendrite_activity"),
                    )
                    nonresponsive_selected_rows = _select_mixed_model_rows(
                        nonresponsive_mixed,
                        preferred_response_keys=("mean_dendrite_activity", "mean") if entity_key == "dendrite" else ("mean_spine_activity_per_dendrite", "mean", "mean_dendrite_activity"),
                    )
                    responsive_significant_states.update(_poster_mixed_model_significant_states(responsive_selected_rows))
                    nonresponsive_significant_states.update(_poster_mixed_model_significant_states(nonresponsive_selected_rows))
                    responsive_compartment_significant_states = set()
                    nonresponsive_compartment_significant_states = set()
                    if entity_key == "dendrite":
                        responsive_compartment_significant_states.update(_compartment_comparison_state_labels_from_comparison_rows(blank_preset_comparison_rows + movie_preset_comparison_rows))
                        nonresponsive_compartment_significant_states.update(_compartment_comparison_state_labels_from_comparison_rows(blank_preset_comparison_rows + movie_preset_comparison_rows))
                    else:
                        responsive_significant_states.update(_significant_state_labels_from_comparison_rows(blank_preset_comparison_rows + movie_preset_comparison_rows))
                        nonresponsive_significant_states.update(_significant_state_labels_from_comparison_rows(blank_preset_comparison_rows + movie_preset_comparison_rows))
                    state_path = write_blank_movie_state_boxplot_figure(
                        output_dir=blank_dir,
                        entity_label=entity_key,
                        responsive_blank_values=blank_values,
                        responsive_movie_values=movie_values,
                        nonresponsive_blank_values=non_blank_values,
                        nonresponsive_movie_values=non_movie_values,
                        blank_state_order=blank_state_order,
                        movie_state_order=movie_state_order,
                        responsive_blank_sample_sizes=blank_sample_sizes,
                        responsive_movie_sample_sizes=movie_sample_sizes,
                        nonresponsive_blank_sample_sizes=non_blank_sample_sizes,
                        nonresponsive_movie_sample_sizes=non_movie_sample_sizes,
                        responsive_significant_states=sorted(responsive_significant_states),
                        nonresponsive_significant_states=sorted(nonresponsive_significant_states),
                        responsive_compartment_significant_states=sorted(responsive_compartment_significant_states),
                        nonresponsive_compartment_significant_states=sorted(nonresponsive_compartment_significant_states),
                        output_stem=f"{entity_key}_blank_movie_states_poster_ready",
                        title="Blank vs movie states",
                    )
                    print(f"[poster-debug] wrote blank/movie plot for {entity_key}: {state_path if state_path else None}", file=sys.stderr)
                    if state_path:
                        written.append(report_relative_path(Path(state_path), output_dir))
    return written

def load_or_build_cache(
    repo_base: Path,
    movie_expids: Sequence[str],
    sleep_expids: Sequence[str],
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
    channel: int,
    explicit_locomotion_threshold: Optional[float],
    cache_path: Path,
    rebuild: bool,
    validate_existing_cache: bool = True,
) -> Dict[str, Any]:
    # Reuse the saved cache when the requested experiment set and source files have not changed.
    requested_config = {
        "repo_base": str(repo_base),
        "movie_expids": sorted(set(movie_expids)),
        "sleep_expids": sorted(set(sleep_expids)),
        "basal_expids": sorted(set(basal_expids)),
        "apical_expids": sorted(set(apical_expids)),
        "channel": channel,
        "explicit_locomotion_threshold": explicit_locomotion_threshold,
    }
    requested_hash = stable_hash(requested_config)
    existing: Optional[Dict[str, Any]] = None
    if cache_path.exists() and not rebuild:
        try:
            existing = load_npz_cache(cache_path)
        except Exception as exc:
            eprint(f"[ALERT] Could not load existing cache at {cache_path}: {exc}. Rebuilding.")
            existing = None
    existing_has_content = False
    if existing:
        existing_summary = summarize_cache(existing)
        existing_has_content = bool(existing_summary.get("n_animals", 0)) or bool(existing_summary.get("n_experiments", 0))
    if existing and existing.get("schema_version") == CACHE_SCHEMA_VERSION and existing.get("config_hash") == requested_hash:
        if validate_existing_cache:
            # Touch each stored experiment just far enough to see whether any source file changed.
            source_stale = False
            validation_items = sorted(existing.get("experiments", {}).items())
            with step_scope("validate existing cache", total=len(validation_items)):
                for idx, (exp_id, exp_meta) in enumerate(validation_items, start=1):
                    step_progress(idx, len(validation_items), label=str(exp_id))
                    try:
                        exp_paths = experiment_source_paths(repo_base, exp_id, channel)
                        conversion_path, conversion_source_exp, _ = locate_conversion_file(
                            repo_base,
                            derive_animal_id(exp_id),
                            exp_id,
                            prefer_same_day_source=exp_id in sleep_expids,
                        )
                        compartment = resolve_experiment_compartment(
                            exp_id,
                            basal_expids,
                            apical_expids,
                            sleep_expids,
                            same_day_source_exp_id=conversion_source_exp,
                        )
                        signature = experiment_source_signature(exp_paths, conversion_path, compartment, channel)
                        if signature != exp_meta.get("source_signature"):
                            source_stale = True
                            break
                    except Exception:
                        source_stale = True
                        break
            if not source_stale and existing_has_content:
                step_message("reusing existing cache")
                strip_gabor_fields(existing)
                return existing
            if not source_stale and not existing_has_content:
                step_message("existing cache was empty; rebuilding from source experiments")
        else:
            if existing_has_content:
                step_message("reusing existing cache without validation")
                strip_gabor_fields(existing)
                return existing
            step_message("existing cache was empty; rebuilding from source experiments")
    # Anything stale or missing gets rebuilt into a fresh cache dictionary.
    step_message("rebuilding cache from source experiments")
    cache: Dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "config": requested_config,
        "config_hash": requested_hash,
        "animals": {},
        "experiments": {},
        "alerts": [],
        "demo_truth": None,
    }
    source_items = sorted(set(movie_expids) | set(sleep_expids))
    with step_scope("process source experiments", total=len(source_items)):
        for idx, exp_id in enumerate(source_items, start=1):
            step_progress(idx, len(source_items), label=str(exp_id))
            try:
                processed = process_experiment(
                    repo_base=repo_base,
                    exp_id=exp_id,
                    channel=channel,
                    movie_expids=movie_expids,
                    sleep_expids=sleep_expids,
                    basal_expids=basal_expids,
                    apical_expids=apical_expids,
                    explicit_locomotion_threshold=explicit_locomotion_threshold,
                )
            except FileNotFoundError as exc:
                alert = f"[ALERT] Skipping {exp_id}: {exc}"
                eprint(alert)
                cache["alerts"].append(alert)
                continue
            except Exception as exc:
                alert = f"[ALERT] Failed to process {exp_id}: {exc}"
                eprint(alert)
                cache["alerts"].append(alert)
                continue
            merge_animal_cache(cache, processed)
            if processed["exp_meta"]["alerts"]:
                cache["alerts"].extend(processed["exp_meta"]["alerts"])
    strip_gabor_fields(cache)
    save_npz_cache(cache_path, cache)
    return cache
def summarize_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    animals = cache.get("animals", {})
    experiments = cache.get("experiments", {})
    analysis_unit = str(cache.get("analysis_unit", "source"))
    dendrites = 0
    spines = 0
    observations = 0
    for animal_entry in animals.values():
        dendrites += len(animal_entry["dendrites"])
        for dendrite_record in animal_entry["dendrites"].values():
            observations += len(dendrite_record["observations"])
            spines += len(dendrite_record["spines"])
    return {
        "analysis_unit": analysis_unit,
        "n_animals": len(animals),
        "n_experiments": len(experiments),
        "n_days": len(experiments) if analysis_unit == "day" else None,
        "n_global_dendrites": dendrites,
        "n_global_spines": spines,
        "n_dendrite_observations": observations,
        "alerts": cache.get("alerts", []),
    }
def demo_event_trace(t: np.ndarray, event_times: Sequence[float], amplitude: float = 1.0, width: float = 1.0) -> np.ndarray:
    trace = np.zeros_like(t, dtype=float)
    for center in event_times:
        trace += amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
    return trace
def create_demo_conversion_library(mode: str = "normal") -> Dict[int, Dict[str, Any]]:
    library: Dict[int, Dict[str, Any]] = {}
    if mode == "normal":
        # Two cells, two dendrites per cell, three spines per dendrite.
        library[1] = {"roi-type": [0, 1, 0, 0], "conversion index": 0, "plane": 0, "conversion": [0, 0]}
        library[2] = {"roi-type": [0, 2, 0, 0], "conversion index": 1, "plane": 0, "conversion": [0, 1]}
        library[10] = {"roi-type": [1, 1, 1, 0], "conversion index": 2, "plane": 0, "conversion": [0, 2]}
        library[11] = {"roi-type": [1, 1, 2, 0], "conversion index": 3, "plane": 0, "conversion": [0, 3]}
        library[20] = {"roi-type": [2, 1, 1, 1], "conversion index": 4, "plane": 0, "conversion": [0, 4]}
        library[21] = {"roi-type": [2, 1, 1, 2], "conversion index": 5, "plane": 0, "conversion": [0, 5]}
        library[22] = {"roi-type": [2, 1, 1, 3], "conversion index": 6, "plane": 0, "conversion": [0, 6]}
        library[30] = {"roi-type": [2, 1, 2, 1], "conversion index": 7, "plane": 0, "conversion": [0, 7]}
        library[31] = {"roi-type": [2, 1, 2, 2], "conversion index": 8, "plane": 0, "conversion": [0, 8]}
        library[32] = {"roi-type": [2, 1, 2, 3], "conversion index": 9, "plane": 0, "conversion": [0, 9]}
    else:
        library[10] = {"roi-type": [0, 1, 0, 0, 0], "conversion index": 0, "plane": 0, "conversion": [0, 0]}
        library[11] = {"roi-type": [0, 2, 0, 0, 0], "conversion index": 1, "plane": 0, "conversion": [0, 1]}
        library[20] = {"roi-type": [1, 1, 1, 0, 0], "conversion index": 2, "plane": 0, "conversion": [0, 2]}
        library[21] = {"roi-type": [1, 1, 2, 0, 0], "conversion index": 3, "plane": 0, "conversion": [0, 3]}
        library[30] = {"roi-type": [1, 1, 1, 0, 0], "conversion index": 4, "plane": 0, "conversion": [0, 4]}
    return library
def generate_demo_state_traces(t: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    quiet = demo_event_trace(t, [18, 57, 97], amplitude=0.75, width=1.0)
    active = demo_event_trace(t, [30, 75, 112], amplitude=0.35, width=0.8)
    base = 0.15 * np.sin(2 * np.pi * t / 60.0) + 0.05 * np.sin(2 * np.pi * t / 17.0)
    dend = base + quiet + 0.3 * active + 0.05 * rng.normal(size=t.size)
    return {"dend": dend}
def default_demo_state_offsets() -> Dict[str, float]:
    return {
        "quiet_awake_blank": -0.04,
        "active_awake_blank": 0.02,
        "nrem_blank": -0.14,
        "rem_blank": -0.08,
        "quiet_awake_gratings": 0.00,
        "active_awake_gratings": 0.06,
        "nrem_gratings": -0.02,
        "rem_gratings": 0.03,
        "quiet_awake_zebras": 0.10,
        "active_awake_zebras": 0.16,
        "nrem_zebras": 0.08,
        "rem_zebras": 0.13,
        "quiet_awake_movies": 0.32,
        "active_awake_movies": 0.38,
        "nrem_movies": 0.28,
        "rem_movies": 0.34,
        "quiet_awake": -0.04,
        "active_awake": 0.02,
        "nrem": -0.14,
        "rem": -0.08,
    }
def default_demo_compartment_offsets() -> Dict[str, float]:
    return {
        "basal": 0.12,
        "apical": -0.10,
        "sleep": 0.0,
        "movie": 0.0,
        "other": 0.0,
    }
def default_demo_experiments() -> List[Dict[str, Any]]:
    return [
        {
            "exp_id": "2025-01-01_01_ESRCDEMO",
            "animal_id": "ESRCDEMO",
            "keep_conversion": True,
            "mode": "normal",
            "state_kind": "basal",
            "include_in_movie": True,
            "include_in_sleep": False,
            "compartment": "basal",
            "make_sleep_state": True,
            "omit_sleep_state": False,
            "seed": 1,
        },
        {
            "exp_id": "2025-01-01_02_ESRCDEMO",
            "animal_id": "ESRCDEMO",
            "keep_conversion": False,
            "mode": "normal",
            "state_kind": "apical",
            "include_in_movie": True,
            "include_in_sleep": False,
            "compartment": "apical",
            "make_sleep_state": True,
            "omit_sleep_state": False,
            "seed": 2,
        },
        {
            "exp_id": "2025-01-01_03_ESRCDEMO",
            "animal_id": "ESRCDEMO",
            "keep_conversion": False,
            "mode": "normal",
            "state_kind": "sleep",
            "include_in_movie": False,
            "include_in_sleep": True,
            "compartment": "sleep",
            "make_sleep_state": True,
            "omit_sleep_state": False,
            "seed": 3,
        },
        {
            "exp_id": "2025-01-02_01_ESRCDEMO",
            "animal_id": "ESRCDEMO",
            "keep_conversion": True,
            "mode": "normal",
            "state_kind": "basal",
            "include_in_movie": True,
            "include_in_sleep": False,
            "compartment": "basal",
            "make_sleep_state": True,
            "omit_sleep_state": False,
            "seed": 4,
        },
        {
            "exp_id": "2025-01-02_02_ESRCDEMO",
            "animal_id": "ESRCDEMO",
            "keep_conversion": False,
            "mode": "normal",
            "state_kind": "sleep",
            "include_in_movie": False,
            "include_in_sleep": True,
            "compartment": "sleep",
            "make_sleep_state": True,
            "omit_sleep_state": True,
            "seed": 5,
        },
    ]
def default_demo_trial_specs() -> List[Dict[str, Any]]:
    return [
        {
            "category": "blank",
            "name": BLANK_MOVIE_PATH,
            "repeats": 3,
            "active_repeats": [1],
        },
        {
            "category": "grating",
            "name": GRATING_PREFIX + r"\00001",
            "repeats": 3,
            "active_repeats": [1],
        },
        {
            "category": "zebra",
            "name": ZEBRA_PREFIX + r"\02001",
            "repeats": 3,
            "active_repeats": [1],
        },
        {
            "category": "movies",
            "name": r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03001",
            "repeats": 3,
            "active_repeats": [1, 2],
        },
    ]
def build_demo_mixed_model_truth(
    state_offsets: Dict[str, float],
    compartment_offsets: Dict[str, float],
) -> List[Dict[str, Any]]:
    truth: List[Dict[str, Any]] = []
    quiet_states = [state for state in PRIMARY_QUIET_STATES if state in state_offsets]
    for state_a, state_b in combinations(quiet_states, 2):
        expected_effect = float(state_offsets.get(state_a, 0.0) - state_offsets.get(state_b, 0.0))
        truth.append(
            {
                "response": "mean_dendrite_activity",
                "scope": "all_state",
                "contrast_type": "state_pair",
                "state_a": state_a,
                "state_b": state_b,
                "expected_effect": expected_effect,
            }
        )
        truth.append(
            {
                "response": "mean_spine_activity_per_dendrite",
                "scope": "all_state",
                "contrast_type": "state_pair",
                "state_a": state_a,
                "state_b": state_b,
                "expected_effect": expected_effect,
            }
        )
    apical_minus_basal = float(compartment_offsets.get("apical", 0.0) - compartment_offsets.get("basal", 0.0))
    for state in [state for state in DEFAULT_BASAL_APICAL_STATES if state in state_offsets]:
        truth.append(
            {
                "response": "mean_dendrite_activity",
                "scope": "all_state",
                "contrast_type": "basal_apical",
                "state": state,
                "expected_effect": apical_minus_basal,
            }
        )
        truth.append(
            {
                "response": "mean_spine_activity_per_dendrite",
                "scope": "all_state",
                "contrast_type": "basal_apical",
                "state": state,
                "expected_effect": apical_minus_basal,
            }
        )
    return truth
def write_demo_experiment(
    repo_base: Path,
    exp_id: str,
    animal_id: str,
    keep_conversion: bool,
    mode: str,
    state_kind: str,
    rng: np.random.Generator,
    expected_alphas: List[Dict[str, Any]],
    make_sleep_state: bool = True,
    omit_sleep_state: bool = False,
    experiment_spec: Optional[Dict[str, Any]] = None,
) -> None:
    # Write one synthetic experiment directory with traces, trials, behavior, and optional sleep metadata.
    spec = experiment_spec or {}
    exp_root = ensure_dir(repo_base / animal_id / exp_id)
    recordings = ensure_dir(exp_root / "recordings")
    cut_dir = ensure_dir(exp_root / "cut")
    sleep_score = ensure_dir(exp_root / "sleep_score")
    suite2p_spines = ensure_dir(exp_root / "suite2p" / "SpinesGUI")
    # The demo timeline is configurable so different synthetic shapes are easy to test.
    t_start = as_float(spec.get("t_start")) or 0.0
    t_end = as_float(spec.get("t_end")) or 120.0
    dt = as_float(spec.get("dt")) or 0.1
    t = np.arange(t_start, t_end, dt, dtype=float)
    n_rois = int(spec.get("n_rois", 10))
    dend_noise_scale = as_float(spec.get("dend_noise_scale")) or 0.03
    spine_noise_scale = as_float(spec.get("spine_noise_scale")) or 0.03
    soma_noise_scale = as_float(spec.get("soma_noise_scale")) or 0.05
    pupil_noise_scale = as_float(spec.get("pupil_noise_scale")) or 0.05
    wheel_noise_scale = as_float(spec.get("wheel_noise_scale")) or 0.08
    wheel_motion_scale = as_float(spec.get("wheel_motion_scale")) or 0.8
    pupil_scale = as_float(spec.get("pupil_scale")) or 0.6
    base_dend_1_scale = as_float(spec.get("base_dend_1_scale")) or 0.2
    base_dend_2_scale = as_float(spec.get("base_dend_2_scale")) or -0.15
    calcium = np.zeros((n_rois, t.size), dtype=float)
    base_dend_1 = base_dend_1_scale * np.sin(2 * np.pi * t / 40.0) + dend_noise_scale * rng.normal(size=t.size)
    base_dend_2 = base_dend_2_scale * np.cos(2 * np.pi * t / 55.0) + dend_noise_scale * rng.normal(size=t.size)
    calcium[2] = base_dend_1
    calcium[3] = base_dend_2
    default_alpha_truths = {
        4: 0.62,
        5: 0.55,
        6: 0.71,
        7: 0.64,
        8: 0.58,
        9: 0.67,
    }
    alpha_truths = {int(k): float(v) for k, v in (spec.get("alpha_truths") or default_alpha_truths).items()}
    default_state_event_times = {
        4: [20, 48, 84],
        5: [26, 60, 98],
        6: [32, 70, 106],
        7: [14, 52, 92],
        8: [35, 78, 111],
        9: [41, 84, 114],
    }
    state_event_times = {
        int(k): [float(x) for x in v]
        for k, v in (spec.get("state_event_times") or default_state_event_times).items()
    }
    trial_specs = spec.get("trial_specs") or default_demo_trial_specs()
    trial_start = as_float(spec.get("trial_start")) or 5.0
    trial_duration = as_float(spec.get("trial_duration")) or 5.0
    trial_gap = as_float(spec.get("trial_gap")) or 0.0
    state_offsets = {str(k): float(v) for k, v in (spec.get("state_offsets") or default_demo_state_offsets()).items()}
    compartment_offsets = {str(k): float(v) for k, v in (spec.get("compartment_offsets") or default_demo_compartment_offsets()).items()}
    compartment_offset_value = float(compartment_offsets.get(state_kind, compartment_offsets.get("sleep" if state_kind == "sleep" else state_kind, 0.0)))
    shared_effect_trace = np.zeros_like(t, dtype=float)
    trial_cursor = trial_start
    for trial_spec in trial_specs:
        category = str(trial_spec.get("category", "movies"))
        repeats = int(trial_spec.get("repeats", 1))
        active_repeats = {int(v) for v in trial_spec.get("active_repeats", [])}
        active_pattern = trial_spec.get("active_pattern")
        repeat_duration = as_float(trial_spec.get("duration")) or trial_duration
        repeat_gap = as_float(trial_spec.get("gap")) or trial_gap
        for repeat in range(repeats):
            if isinstance(active_pattern, list) and repeat < len(active_pattern):
                active = bool(active_pattern[repeat])
            elif active_repeats:
                active = repeat in active_repeats
            else:
                active = (repeat % 2 == 1) or (category == "movies" and repeat == 2)
            state_label = combined_movie_state_label("active_awake" if active else "quiet_awake", category)
            offset = float(state_offsets.get(state_label, 0.0))
            shared_effect_trace += offset * interval_mask(t, trial_cursor, trial_cursor + repeat_duration)
            trial_cursor += repeat_duration + repeat_gap
    calcium[2] = base_dend_1 + shared_effect_trace + compartment_offset_value
    calcium[3] = base_dend_2 + shared_effect_trace + compartment_offset_value
    for idx in sorted(alpha_truths):
        if idx >= n_rois:
            continue
        spine_specific = demo_event_trace(t, state_event_times[idx], amplitude=0.5 + 0.08 * (idx - 4), width=0.9)
        spine_specific = spine_specific + shared_effect_trace + compartment_offset_value
        if idx in [4, 5, 6]:
            calcium[idx] = alpha_truths[idx] * calcium[2] + spine_specific + spine_noise_scale * rng.normal(size=t.size)
        else:
            calcium[idx] = alpha_truths[idx] * calcium[3] + spine_specific + spine_noise_scale * rng.normal(size=t.size)
    calcium[0] = soma_noise_scale * rng.normal(size=t.size)
    calcium[1] = soma_noise_scale * rng.normal(size=t.size)
    write_pickle(recordings / "s2p_ch0.pickle", {"t": t, "dF": calcium, "OriginalSuite2pCellIDs": np.arange(n_rois)})
    # Trial rows emulate the movie stimulus table read by the real data loader.
    trial_rows: List[Dict[str, Any]] = []
    trial_specs = spec.get("trial_specs") or default_demo_trial_specs()
    trial_start = as_float(spec.get("trial_start")) or 5.0
    trial_duration = as_float(spec.get("trial_duration")) or 5.0
    trial_gap = as_float(spec.get("trial_gap")) or 0.0
    for trial_spec in trial_specs:
        category = str(trial_spec.get("category", "movies"))
        name = str(trial_spec.get("name", r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03001"))
        repeats = int(trial_spec.get("repeats", 1))
        active_repeats = {int(v) for v in trial_spec.get("active_repeats", [])}
        active_pattern = trial_spec.get("active_pattern")
        repeat_duration = as_float(trial_spec.get("duration")) or trial_duration
        repeat_gap = as_float(trial_spec.get("gap")) or trial_gap
        extra_features = trial_spec.get("extra_features", [])
        for repeat in range(repeats):
            if isinstance(active_pattern, list) and repeat < len(active_pattern):
                active = bool(active_pattern[repeat])
            elif active_repeats:
                active = repeat in active_repeats
            else:
                active = (repeat % 2 == 1) or (category == "movies" and repeat == 2)
            row = {
                "time": f"{trial_start:.3f}",
                "duration": f"{repeat_duration:.3f}",
                "F1_type": "movie",
                "F1_name": name,
                "F1_onset": "0",
                "F1_duration": f"{repeat_duration:.3f}",
                "F1_speed": "1",
                "F1_loop": "0",
            }
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
            trial_start += repeat_duration + repeat_gap
    if len(trial_rows) >= 2 and spec.get("add_ambiguous_feature", True):
        # A deliberately ambiguous second feature exercises the parser's disambiguation logic.
        trial_rows[1]["F2_type"] = "movie"
        trial_rows[1]["F2_name"] = r"D:\bonsai_resources\all_movie_clips_bv_sets\007\03002"
        trial_rows[1]["F2_onset"] = "0"
        trial_rows[1]["F2_duration"] = "5"
        trial_rows[1]["F2_speed"] = "1"
        trial_rows[1]["F2_loop"] = "0"
    trial_fieldnames = sorted({key for row in trial_rows for key in row.keys() if not key.startswith("_")}, key=lambda x: (0 if x == "time" else 1, x))
    cleaned_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in trial_rows]
    write_csv_rows(exp_root / f"{exp_id}_all_trials.csv", cleaned_rows, trial_fieldnames)
    # Wheel and pupil traces are coupled so locomotion thresholds have a known effect.
    wheel_t = t.copy()
    wheel_speed = wheel_noise_scale * rng.normal(size=t.size)
    wheel_active_segments = spec.get(
        "wheel_active_segments",
        [
            (5, 10),
            (15, 20),
            (25, 30),
            (35, 40),
            (45, 50),
            (60, 65),
            (75, 80),
            (95, 100),
        ],
    )
    for start, end in wheel_active_segments:
        mask = interval_mask(wheel_t, start, end)
        wheel_speed[mask] += wheel_motion_scale + 0.1 * rng.normal(size=mask.sum())
    write_pickle(recordings / "wheel.pickle", {"t": wheel_t, "speed": wheel_speed})
    pupil = 1.5 + pupil_scale * np.tanh(wheel_speed) + pupil_noise_scale * rng.normal(size=t.size)
    write_pickle(recordings / "dlcEyeLeft_resampled.pickle", {"t": t, "pupil_diameter": pupil, "speed": wheel_speed})
    write_pickle(recordings / "dlcEyeRight_resampled.pickle", {"t": t, "pupil_diameter": pupil + 0.02 * rng.normal(size=t.size), "speed": wheel_speed})
    # Cut files mirror the continuous traces in trial-aligned form.
    if spec.get("make_cut_files", True):
        cut_time = np.linspace(0.0, trial_duration, int(spec.get("cut_points", 50)), dtype=float)
        cut_calcium = np.zeros((n_rois, len(cleaned_rows), cut_time.size), dtype=float)
        cut_wheel = np.zeros((len(cleaned_rows), cut_time.size), dtype=float)
        for trial_index, row in enumerate(cleaned_rows):
            trial_onset = as_float(row.get("time"))
            if trial_onset is None:
                continue
            absolute_time = trial_onset + cut_time
            cut_wheel[trial_index] = interpolate_series(absolute_time, wheel_t, wheel_speed)
            for roi_idx in range(n_rois):
                cut_calcium[roi_idx, trial_index] = interpolate_series(absolute_time, t, calcium[roi_idx])
        write_pickle(
            cut_dir / "s2p_ch0_dF_cut.pickle",
            {
                "t": cut_time,
                "dF": cut_calcium,
                "trial_index": np.arange(len(cleaned_rows), dtype=int),
                "trial_rows": cleaned_rows,
            },
        )
        write_pickle(
            cut_dir / "wheel.pickle",
            {
                "t": cut_time,
                "speed": cut_wheel,
                "trial_index": np.arange(len(cleaned_rows), dtype=int),
            },
        )
    # Sleep-state metadata is optional in the demo so missing-file alerts can be tested.
    if make_sleep_state:
        state_codes = np.zeros_like(t, dtype=int)
        sleep_segments = spec.get(
            "sleep_state_segments",
            [
                {"start": 0.0, "end": 30.0, "code": 1},
                {"start": 30.0, "end": 60.0, "code": 0},
                {"start": 60.0, "end": 90.0, "code": 2},
                {"start": 90.0, "end": float(t[-1]) + dt, "code": 3},
            ],
        )
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
        sleep_state = {
            "state_10hz_t": t,
            "state_10hz": state_codes,
            "state_epoch_t": t[::10],
            "state_epoch": state_codes[::10],
            "epoch_t": t[::10],
            "state_labels": SLEEP_STATE_MAP,
            "locomotion_threshold": as_float(spec.get("sleep_locomotion_threshold")) or 0.35,
            "emg_rms_10hz": np.abs(rng.normal(size=t.size)),
            "emg_rms_10hz_t": t,
            "wheel_10hz": wheel_speed,
                "wheel_10hz_t": t,
        }
        if not omit_sleep_state:
            write_pickle(sleep_score / "sleep_state.pickle", sleep_state)
    # Conversion libraries use the same structure as the real SpinesGUI files.
    if keep_conversion:
        conversion = create_demo_conversion_library(mode=mode)
        np.save(suite2p_spines / (DEND_AXON_CONVERSION_FILENAME if mode == "dendrite_axon" else NORMAL_CONVERSION_FILENAME), conversion, allow_pickle=True)
    # Record the planted alpha values so the demo can verify recovery later.
    for spine_idx, alpha in alpha_truths.items():
        exp_key = exp_id
        dendrite_parent = 1 if spine_idx in [4, 5, 6] else 2
        expected_alphas.append(
            {
                "exp_id": exp_key,
                "global_spine_id": global_spine_id(
                    animal_id,
                    derive_date(exp_id),
                    "sleep" if state_kind == "sleep" else state_kind,
                    "cell1",
                    dendrite_parent,
                    spine_idx - 3,
                ),
                "alpha": float(alpha),
            }
        )
def build_demo_repository(base_dir: Path, demo_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Build or rebuild the synthetic repository tree from the requested demo recipe.
    spec = demo_spec or {}
    repo_subdir = str(spec.get("repo_subdir", "demo_repository"))
    repo_base = base_dir / repo_subdir
    if repo_base.exists():
        shutil.rmtree(repo_base)
    ensure_dir(repo_base)
    # Each experiment entry can override almost every demo characteristic.
    experiments = spec.get("experiments") or default_demo_experiments()
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("demo_spec must provide a non-empty experiments list")
    state_offsets = {str(k): float(v) for k, v in (spec.get("state_offsets") or default_demo_state_offsets()).items()}
    compartment_offsets = {str(k): float(v) for k, v in (spec.get("compartment_offsets") or default_demo_compartment_offsets()).items()}
    expected_alphas: List[Dict[str, Any]] = []
    for experiment_spec in experiments:
        merged_experiment_spec = {k: v for k, v in spec.items() if k != "experiments"}
        merged_experiment_spec.update(experiment_spec)
        exp_id = str(experiment_spec["exp_id"])
        animal_id = str(merged_experiment_spec.get("animal_id", "ESRCDEMO"))
        keep_conversion = bool(merged_experiment_spec.get("keep_conversion", True))
        mode = str(merged_experiment_spec.get("mode", "normal"))
        state_kind = str(merged_experiment_spec.get("state_kind", merged_experiment_spec.get("compartment", "basal")))
        seed = merged_experiment_spec.get("seed")
        if seed is None:
            seed = int(stable_hash({"exp_id": exp_id})[:8], 16)
        rng = np.random.default_rng(int(seed))
        write_demo_experiment(
            repo_base,
            exp_id,
            animal_id,
            keep_conversion=keep_conversion,
            mode=mode,
            state_kind=state_kind,
            rng=rng,
            expected_alphas=expected_alphas,
            make_sleep_state=bool(merged_experiment_spec.get("make_sleep_state", True)),
            omit_sleep_state=bool(merged_experiment_spec.get("omit_sleep_state", False)),
            experiment_spec=merged_experiment_spec,
        )
    # The expID lists can be supplied explicitly, or derived from the experiment roles.
    if "movie_expids" in spec:
        movie_expids = [str(v) for v in spec.get("movie_expids", [])]
    else:
        movie_expids = [str(exp["exp_id"]) for exp in experiments if exp.get("include_in_movie", exp.get("compartment") in {"basal", "apical", "movie"})]
    if "sleep_expids" in spec:
        sleep_expids = [str(v) for v in spec.get("sleep_expids", [])]
    else:
        sleep_expids = [str(exp["exp_id"]) for exp in experiments if exp.get("include_in_sleep", exp.get("compartment") == "sleep")]
    if "basal_expids" in spec:
        basal_expids = [str(v) for v in spec.get("basal_expids", [])]
    else:
        basal_expids = [str(exp["exp_id"]) for exp in experiments if exp.get("compartment") == "basal"]
    if "apical_expids" in spec:
        apical_expids = [str(v) for v in spec.get("apical_expids", [])]
    else:
        apical_expids = [str(exp["exp_id"]) for exp in experiments if exp.get("compartment") == "apical"]
    demo_config = {
        "user_id": str(spec.get("user_id", "demo_user")),
        "repo_base": str(repo_base),
        "movie_expids": movie_expids,
        "sleep_expids": sleep_expids,
        "basal_expids": basal_expids,
        "apical_expids": apical_expids,
        "channel": int(spec.get("channel", 0)),
        "locomotion_threshold": as_float(spec.get("locomotion_threshold")) or 0.35,
        "expected_alphas": expected_alphas,
        "expected_mixed_model_contrasts": build_demo_mixed_model_truth(state_offsets, compartment_offsets),
        "demo_spec": spec,
    }
    return demo_config
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
def resolve_state_selection(values: Any, default_states: Sequence[str], allowed_states: Sequence[str], label: str) -> List[str]:
    if values is None:
        return list(default_states)
    selected = parse_list_argument(values)
    if not selected:
        return []
    allowed = set(allowed_states)
    unknown = [state for state in selected if state not in allowed]
    if unknown:
        raise SystemExit(
            f"Unknown state label(s) in {label}: {', '.join(unknown)}. "
            f"Allowed labels are: {', '.join(allowed_states)}"
        )
    deduped: List[str] = []
    seen = set()
    for state in selected:
        if state in seen:
            continue
        deduped.append(state)
        seen.add(state)
    return deduped
def normalize_state_mode(value: Any, default: Optional[str] = "quiet") -> Optional[str]:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text not in STATE_MODE_CHOICES:
        raise SystemExit(f"Unknown state_mode: {value}. Allowed values are: {', '.join(STATE_MODE_CHOICES)}")
    return text
def normalize_movie_trial_types(values: Any) -> List[str]:
    selected = parse_list_argument(values)
    if not selected:
        return []
    allowed = set(MOVIE_TRIAL_TYPES)
    unknown = [trial for trial in selected if trial not in allowed]
    if unknown:
        raise SystemExit(
            f"Unknown movie trial type(s): {', '.join(unknown)}. Allowed values are: {', '.join(MOVIE_TRIAL_TYPES)}"
        )
    deduped: List[str] = []
    seen = set()
    for trial in selected:
        if trial in seen:
            continue
        deduped.append(trial)
        seen.add(trial)
    return deduped
def infer_state_mode_from_states(states: Sequence[str]) -> str:
    labels = [str(state) for state in states if state is not None and str(state).strip()]
    has_quiet = any(label.startswith("quiet_") for label in labels)
    has_active = any(label.startswith("active_") for label in labels)
    if has_quiet and has_active:
        return "all"
    if has_active:
        return "active"
    return "quiet"
def infer_movie_trial_types_from_states(states: Sequence[str]) -> List[str]:
    labels = {str(state) for state in states if state is not None and str(state).strip()}
    selected: List[str] = []
    for trial_type, state_labels in MOVIE_TRIAL_TYPE_TO_STATE_LABELS.items():
        if any(label in labels for label in state_labels):
            selected.append(trial_type)
    return selected
def build_state_mode_state_selection(
    state_mode: str,
    movie_trial_types: Sequence[str],
    *,
    include_movie_states: bool,
    include_sleep_states: bool,
) -> List[str]:
    selected: List[str] = []
    sleep_mode_states = STATE_MODE_SLEEP_LABELS[state_mode]
    if include_movie_states:
        for trial_type in movie_trial_types:
            for sleep_label in sleep_mode_states:
                selected.append(combined_movie_state_label(sleep_label, trial_type))
    if include_sleep_states:
        selected.extend(sleep_mode_states)
    deduped: List[str] = []
    seen = set()
    for state in selected:
        if state in seen:
            continue
        deduped.append(state)
        seen.add(state)
    return deduped
def resolve_analysis_state_selections(
    config: Dict[str, Any],
    *,
    movie_expids: Optional[Sequence[str]] = None,
    sleep_expids: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    compare_states_raw = config.get("compare_states")
    compare_states = parse_list_argument(compare_states_raw) if compare_states_raw is not None else []
    explicit_state_comparison_raw = config.get("state_comparison_states")
    explicit_basal_apical_raw = config.get("basal_apical_states")
    if explicit_state_comparison_raw is not None and explicit_basal_apical_raw is None:
        explicit_basal_apical_raw = explicit_state_comparison_raw
    elif explicit_basal_apical_raw is not None and explicit_state_comparison_raw is None:
        explicit_state_comparison_raw = explicit_basal_apical_raw
    state_mode_raw = config.get("state_mode")
    movie_trial_types_raw = config.get("movie_trial_types")
    selection_alerts: List[str] = []
    movie_expids = list(movie_expids or [])
    sleep_expids = list(sleep_expids or [])
    movie_present = bool(movie_expids)
    sleep_present = bool(sleep_expids)
    has_new_state_selection = state_mode_raw is not None or movie_trial_types_raw is not None
    use_legacy_shortcut = compare_states_raw is not None and not has_new_state_selection and explicit_state_comparison_raw is None and explicit_basal_apical_raw is None
    use_legacy_default = not has_new_state_selection and compare_states_raw is None and explicit_state_comparison_raw is None and explicit_basal_apical_raw is None
    if use_legacy_shortcut:
        state_mode = normalize_state_mode(infer_state_mode_from_states(compare_states), default="quiet") or "quiet"
        movie_trial_types = normalize_movie_trial_types(infer_movie_trial_types_from_states(compare_states))
        if explicit_state_comparison_raw is None:
            explicit_state_comparison_raw = compare_states
        if explicit_basal_apical_raw is None:
            explicit_basal_apical_raw = compare_states
        state_mode_source = "compare_states"
        movie_trial_types_source = "compare_states"
    elif use_legacy_default:
        state_mode = "quiet"
        movie_trial_types = ["blank", "movies"]
        state_mode_source = "default"
        movie_trial_types_source = "default"
    else:
        state_mode = normalize_state_mode(state_mode_raw, default="quiet") or "quiet"
        movie_trial_types = normalize_movie_trial_types(movie_trial_types_raw)
        state_mode_source = "config" if state_mode_raw is not None else "default"
        movie_trial_types_source = "config" if movie_trial_types_raw is not None else "default"
    if explicit_state_comparison_raw is None:
        state_comparison_states = build_state_mode_state_selection(
            state_mode,
            movie_trial_types,
            include_movie_states=movie_present,
            include_sleep_states=sleep_present,
        )
        if movie_present and not movie_trial_types and has_new_state_selection:
            selection_alerts.append(
                "[ALERT] movie_expids are present but movie_trial_types is empty or missing; "
                "movie state labels were not added to state_comparison_states."
            )
    else:
        state_comparison_states = resolve_state_selection(
            explicit_state_comparison_raw,
            PRIMARY_QUIET_STATES,
            ALL_REQUESTED_STATES,
            "state_comparison_states",
        )
    basal_apical_states = resolve_state_selection(
        explicit_basal_apical_raw,
        DEFAULT_BASAL_APICAL_STATES,
        ALL_REQUESTED_STATES,
        "basal_apical_states",
    )
    selection_meta = {
        "compare_states": list(compare_states) if compare_states_raw is not None else None,
        "state_mode": state_mode,
        "movie_trial_types": list(movie_trial_types) if movie_trial_types else None,
        "state_mode_source": state_mode_source,
        "movie_trial_types_source": movie_trial_types_source,
        "alerts": selection_alerts,
    }
    return state_comparison_states, basal_apical_states, selection_meta
def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open() as handle:
        return json.load(handle)
def merge_cli_config(cli: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(file_config)
    for key, value in cli.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        merged[key] = value
    return merged
def run_comparison_preset_subprocesses(config: Dict[str, Any]) -> bool:
    preset_names = parse_list_argument(config.get("comparison_preset_names"))
    presets = normalize_comparison_presets(config.get("comparison_presets"))
    plan = build_comparison_preset_batch_plan(
        presets,
        selected_names=preset_names,
        poster_ready_only=bool(config.get("poster_ready_only")),
        poster_required_names=POSTER_REQUIRED_COMPARISON_PRESETS,
    )
    if not plan.presets:
        return False

    base_output_dir = resolve_repo_path(config.get("output_dir") or DEFAULT_RESULTS_DIR, REPO_ROOT)
    shared_cache_path = resolve_repo_path(config.get("cache_path") or (base_output_dir / DEFAULT_CACHE_DIRNAME / DEFAULT_CACHE_NAME), REPO_ROOT)

    child_script = Path(__file__).resolve()
    preset_configs: Dict[str, Dict[str, Any]] = {}
    for preset_index, (preset_name, overrides) in enumerate(plan.presets):
        safe_name = safe_filename_component(preset_name)
        preset_output_dir = base_output_dir / safe_name
        preset_cache_path = preset_output_dir / DEFAULT_CACHE_DIRNAME / f"{shared_cache_path.stem}_{safe_name}.npz"
        preset_config = copy.deepcopy(config)
        preset_config.pop("comparison_presets", None)
        preset_config.pop("comparison_preset_names", None)
        preset_config.pop("comparison_preset_name", None)
        preset_config.pop("analysis_run_cache_path", None)
        preset_config.pop("analysis_results_cache_path", None)
        preset_config.update(overrides)
        preset_config["comparison_preset_name"] = preset_name
        preset_config["output_dir"] = str(preset_output_dir)
        preset_config["cache_path"] = str(shared_cache_path)
        preset_config["analysis_run_cache_path"] = str(preset_cache_path)
        preset_config["analysis_results_cache_path"] = None
        generate_once = preset_index == 0
        preset_rebuild = bool(config.get("rebuild")) if generate_once else False
        preset_config["rebuild"] = preset_rebuild
        preset_config["source_cache_rebuild"] = preset_rebuild
        preset_config["analysis_tables_rebuild"] = preset_rebuild
        preset_config["analysis_results_rebuild"] = True
        preset_config["shared_shuffle_cache_rebuild"] = preset_rebuild
        preset_config["branch_first_output_root"] = str(preset_output_dir)
        preset_config["branch_first_figures"] = True
        if bool(preset_config.get("plots_only")):
            preset_results_cache_path = analysis_results_cache_path(preset_cache_path)
            preset_family_stage = family_results_cache_stage_for_selection(preset_config.get("analysis_families"))
            preset_family_cache_path = family_results_cache_path(preset_cache_path, preset_family_stage)
            if not preset_results_cache_path.exists() and not preset_family_cache_path.exists():
                print(
                    "[comparison preset] "
                    f"{preset_name}: plots_only cache missing; rebuilding analysis results for this preset",
                    file=sys.stderr,
                )
                preset_config["plots_only"] = False
        preset_config["plots_only_include_supporting_figures"] = True
        preset_config["generate_poster_ready_figures"] = False
        preset_config["generate_shared_general_figures"] = True if generate_once else False

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            temp_config_path = Path(handle.name)
            json.dump(preset_config, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        try:
            print(
                f"[comparison preset] {preset_name} -> {preset_output_dir}",
                file=sys.stderr,
            )
            subprocess.run([sys.executable, str(child_script), "--config", str(temp_config_path)], check=True)
        finally:
            try:
                temp_config_path.unlink()
            except FileNotFoundError:
                pass
        preset_configs[preset_name] = preset_config

    if plan.reference_preset_name in preset_configs:
        final_config = copy.deepcopy(preset_configs[plan.reference_preset_name])
        final_config["plots_only"] = True
        final_config["poster_ready_only"] = bool(config.get("poster_ready_only"))
        final_config["generate_poster_ready_figures"] = True
        final_config["plots_only_include_supporting_figures"] = False
        final_config["generate_shared_general_figures"] = False
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            temp_config_path = Path(handle.name)
            json.dump(final_config, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        try:
            print(
                f"[comparison preset] poster readback -> {plan.reference_preset_name}",
                file=sys.stderr,
            )
            subprocess.run([sys.executable, str(child_script), "--config", str(temp_config_path)], check=True)
        finally:
            try:
                temp_config_path.unlink()
            except FileNotFoundError:
                pass
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh sleep dendrite/spine analysis pipeline")
    parser.add_argument("--config", type=Path, help="Optional JSON configuration file")
    parser.add_argument("--user-id")
    parser.add_argument("--repo-base", type=Path)
    parser.add_argument("--movie-expids", nargs="*")
    parser.add_argument("--sleep-expids", nargs="*")
    parser.add_argument("--basal-expids", nargs="*")
    parser.add_argument("--apical-expids", nargs="*")
    parser.add_argument(
        "--compare-states",
        nargs="*",
        help="Compatibility shortcut that can still fill both state_comparison_states and basal_apical_states",
    )
    parser.add_argument(
        "--state-mode",
        help="State mode used to derive the comparison list: all, quiet, or active",
    )
    parser.add_argument(
        "--movie-trial-types",
        nargs="*",
        help="Explicit movie trial types to include when deriving the comparison list: blank, grating, zebra, movies",
    )
    parser.add_argument(
        "--state-comparison-states",
        nargs="*",
        help="Optional manual override for the pairwise state comparison analysis",
    )
    parser.add_argument(
        "--basal-apical-states",
        nargs="*",
        help="Optional subset of movie state labels for the basal-vs-apical comparison analysis",
    )
    parser.add_argument(
        "--dendrite-response-cohort",
        choices=list(DENDRITE_RESPONSE_COHORTS),
        help="Select which dendrite cohort to render in poster-ready basal/apical panels: all, responsive, or nonresponsive",
    )
    parser.add_argument(
        "--spine-coactivity-anchor-state",
        help="Anchor state used to select spine-pair comparison plots and basal-vs-apical distributions",
    )
    parser.add_argument(
        "--skip-source-cache-validation",
        action="store_true",
        help="Reuse an existing source cache without validating each experiment against source files",
    )
    parser.add_argument("--fit-spine-coactivity-mixed-model", action="store_true", help="Enable the optional mixed-model inference layer for spine coactivity")
    parser.add_argument("--spine-coactivity-only", action="store_true", help="Skip the main state/correlation/matrix analyses and run only spine coactivity")
    parser.add_argument("--mixed-model-only", action="store_true", help="Skip the state/correlation/matrix analyses and run only the main mixed-model branch")
    parser.add_argument("--plots-only", action="store_true", help="Reuse saved caches and only generate plots, without rewriting CSV/JSON/report artifacts")
    parser.add_argument(
        "--poster-ready-only",
        action="store_true",
        help="Run only the reduced analysis set needed for poster-ready figures and skip the standard figure pass",
    )
    parser.add_argument(
        "--mixed-model-contrast-p-source",
        choices=["classical", "shuffle"],
        help="Choose the p-value source for mixed-model contrasts: classical or shuffle",
    )
    parser.add_argument(
        "--analysis-families",
        nargs="*",
        help="Optional subset of analysis families to select: state, basal_apical, direct_trial_type_comparison, correlation, matrix_similarity, mixed_model, spine_coactivity",
    )
    parser.add_argument(
        "--comparison-presets",
        nargs="*",
        help="Optional subset of preset names to run from a comparison_presets config block",
    )
    parser.add_argument("--channel", type=int)
    parser.add_argument("--locomotion-threshold", type=float)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--analysis-tables-cache-path", type=Path)
    parser.add_argument("--analysis-results-cache-path", type=Path)
    parser.add_argument("--source-cache-rebuild", action="store_true")
    parser.add_argument("--analysis-tables-rebuild", action="store_true")
    parser.add_argument("--analysis-results-rebuild", action="store_true")
    parser.add_argument("--shared-shuffle-cache-rebuild", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shuffle-n", type=int)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-spec", type=Path, help="Optional JSON file describing custom demo characteristics")
    return parser
def main(argv: Optional[Sequence[str]] = None) -> int:
    # Parse the config, merge overrides, then run the cache build and analysis stages.
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    file_config = load_config_file(args.config)
    cli_config = {
        "user_id": args.user_id,
        "repo_base": str(args.repo_base) if args.repo_base else None,
        "movie_expids": parse_list_argument(args.movie_expids),
        "sleep_expids": parse_list_argument(args.sleep_expids),
        "basal_expids": parse_list_argument(args.basal_expids),
        "apical_expids": parse_list_argument(args.apical_expids),
        "compare_states": parse_list_argument(args.compare_states),
        "state_mode": args.state_mode,
        "movie_trial_types": parse_list_argument(args.movie_trial_types),
        "state_comparison_states": parse_list_argument(args.state_comparison_states),
        "basal_apical_states": parse_list_argument(args.basal_apical_states),
        "dendrite_response_cohort": args.dendrite_response_cohort,
        "spine_coactivity_anchor_state": args.spine_coactivity_anchor_state,
        "source_cache_validate": False if args.skip_source_cache_validation else None,
        "fit_spine_coactivity_mixed_model": True if args.fit_spine_coactivity_mixed_model else None,
        "spine_coactivity_only": True if args.spine_coactivity_only else None,
        "mixed_model_only": True if args.mixed_model_only else None,
        "mixed_model_contrast_p_source": args.mixed_model_contrast_p_source,
        "analysis_families": parse_list_argument(args.analysis_families) or None,
        "comparison_preset_names": parse_list_argument(args.comparison_presets) or None,
        "plots_only": True if args.plots_only else None,
        "poster_ready_only": True if args.poster_ready_only else None,
        "demo": True if args.demo else None,
        "channel": args.channel,
        "shuffle_n": args.shuffle_n,
        "locomotion_threshold": args.locomotion_threshold,
        "cache_path": str(args.cache_path) if args.cache_path else None,
        "analysis_tables_cache_path": str(args.analysis_tables_cache_path) if args.analysis_tables_cache_path else None,
        "analysis_results_cache_path": str(args.analysis_results_cache_path) if args.analysis_results_cache_path else None,
        "source_cache_rebuild": True if args.source_cache_rebuild else None,
        "analysis_tables_rebuild": True if args.analysis_tables_rebuild else None,
        "analysis_results_rebuild": True if args.analysis_results_rebuild else None,
        "shared_shuffle_cache_rebuild": True if args.shared_shuffle_cache_rebuild else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "rebuild": True if args.rebuild else None,
    }
    config = merge_cli_config(cli_config, file_config)
    if config.get("comparison_presets"):
        if run_comparison_preset_subprocesses(config):
            return 0
    global SPINE_COACTIVITY_ANCHOR_STATE
    configured_anchor_state = canonical_state_label(str(config.get("spine_coactivity_anchor_state") or SPINE_COACTIVITY_ANCHOR_STATE))
    if configured_anchor_state not in ALL_REQUESTED_STATES:
        raise SystemExit(f"Unknown spine coactivity anchor state: {configured_anchor_state}. Allowed values are: {', '.join(ALL_REQUESTED_STATES)}")
    SPINE_COACTIVITY_ANCHOR_STATE = configured_anchor_state
    # Apply explicit script defaults only after merging config and CLI values.
    # That keeps `--config` as the main place to set the run, while still
    # allowing command-line overrides when you need them.
    for key, value in USER_EDITABLE_DEFAULTS.items():
        if key not in config or config[key] is None:
            config[key] = value
    if bool(config.get("poster_ready_only")):
        config["analysis_families"] = ["mixed_model", "spine_coactivity"]
    cpu_thread_limit = apply_cpu_thread_limit(config.get("cpu_thread_limit"))
    from analysis_families.core import normalize_analysis_families
    config["analysis_families"] = normalize_analysis_families(config.get("analysis_families"))
    if bool(config.get("demo")) or args.demo:
        # Demo mode first materializes a fake repository, then reroutes the analysis there.
        demo_output_dir = resolve_repo_path(config.get("output_dir") or DEFAULT_RESULTS_DIR, REPO_ROOT)
        demo_base = demo_output_dir
        demo_spec = None
        if args.demo_spec is not None:
            demo_spec = load_config_file(args.demo_spec)
        elif isinstance(config.get("demo_spec"), dict):
            demo_spec = config.get("demo_spec")
        elif isinstance(config.get("demo_spec"), str):
            demo_spec_path = Path(str(config.get("demo_spec")))
            if demo_spec_path.exists():
                demo_spec = load_config_file(demo_spec_path)
        demo_config = build_demo_repository(demo_base, demo_spec=demo_spec)
        config = merge_cli_config(
            {
                "user_id": demo_config["user_id"],
                "repo_base": demo_config["repo_base"],
                "movie_expids": demo_config["movie_expids"],
                "sleep_expids": demo_config["sleep_expids"],
                "basal_expids": demo_config["basal_expids"],
                "apical_expids": demo_config["apical_expids"],
                "channel": demo_config["channel"],
                "locomotion_threshold": demo_config["locomotion_threshold"],
            },
            config,
        )
        config["demo_truth"] = {
            "expected_alphas": demo_config["expected_alphas"],
            "expected_mixed_model_contrasts": demo_config["expected_mixed_model_contrasts"],
        }
    with step_scope("config / path setup"):
        # These expID lists define the source sessions that will be pooled into the analysis cache.
        user_id = config.get("user_id")
        if not user_id:
            raise SystemExit("Missing user_id. Provide --user-id or config file entry.")
        repo_base = Path(config.get("repo_base") or f"/home/{user_id}/data/Repository")
        movie_expids = parse_list_argument(config.get("movie_expids"))
        sleep_expids = parse_list_argument(config.get("sleep_expids"))
        basal_expids = parse_list_argument(config.get("basal_expids"))
        apical_expids = parse_list_argument(config.get("apical_expids"))
        state_comparison_states, basal_apical_states, selection_meta = resolve_analysis_state_selections(
            config,
            movie_expids=movie_expids,
            sleep_expids=sleep_expids,
        )
        channel = int(config.get("channel") or DEFAULT_CHANNEL)
        shuffle_n = int(config.get("shuffle_n") or DEFAULT_SHUFFLES)
        mixed_model_contrast_p_source = normalize_mixed_model_contrast_p_source(config.get("mixed_model_contrast_p_source"))
        event_detection_method = set_active_event_detection_method(config.get("event_detection_method"))
        visual_response_metric = set_active_visual_response_metric(config.get("visual_response_metric"))
        spine_coactivity_abs_threshold = normalize_non_negative_float(
            config.get("spine_coactivity_abs_threshold"),
            DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
            "spine_coactivity_abs_threshold",
        )
        rebuild = bool(config.get("rebuild"))
        source_cache_validate = bool(config.get("source_cache_validate"))
        source_cache_rebuild = bool(config.get("source_cache_rebuild")) or rebuild
        analysis_tables_rebuild = bool(config.get("analysis_tables_rebuild")) or rebuild
        analysis_results_rebuild = bool(config.get("analysis_results_rebuild")) or rebuild
        shared_shuffle_cache_rebuild = bool(config.get("shared_shuffle_cache_rebuild")) or rebuild
        output_dir = resolve_repo_path(config.get("output_dir") or DEFAULT_RESULTS_DIR, REPO_ROOT)
        cache_path = resolve_repo_path(config.get("cache_path") or (ensure_dir(output_dir / DEFAULT_CACHE_DIRNAME) / DEFAULT_CACHE_NAME), REPO_ROOT)
        analysis_run_cache_path = resolve_repo_path(config.get("analysis_run_cache_path") or cache_path, REPO_ROOT)
        analysis_tables_cache_file = resolve_repo_path(config.get("analysis_tables_cache_path") or analysis_table_cache_path(cache_path), REPO_ROOT)
        analysis_results_cache_file = resolve_repo_path(config.get("analysis_results_cache_path") or analysis_results_cache_path(analysis_run_cache_path), REPO_ROOT)
        figure_output_dir = resolve_repo_path(config.get("figure_output_dir"), REPO_ROOT) if config.get("figure_output_dir") else None
        plots_only = bool(config.get("plots_only"))
        if plots_only:
            source_cache_rebuild = False
            analysis_tables_rebuild = False
            analysis_results_rebuild = False
            shared_shuffle_cache_rebuild = False
        step_message(
            f"RUN FLAGS: rebuild={rebuild}, plots_only={plots_only}, "
            f"analysis_results_rebuild={analysis_results_rebuild}"
        )
        ensure_dir(output_dir)
    
    
    # Build or reuse the cache first; every later output comes from this normalized data structure.
    with step_scope("cache load or rebuild"):
        if plots_only:
            if not cache_path.exists():
                raise SystemExit(f"plots_only requires an existing cache file at {cache_path}")
            try:
                source_cache = load_npz_cache(cache_path)
            except Exception as exc:
                raise SystemExit(f"plots_only could not load cached source data from {cache_path}: {exc}") from exc
            if not isinstance(source_cache, dict):
                raise SystemExit(f"plots_only expected a cached source dictionary at {cache_path}")
        else:
            if not source_cache_validate:
                step_message("skipping source-cache validation")
            source_cache = load_or_build_cache(
                repo_base=repo_base,
                movie_expids=movie_expids,
                sleep_expids=sleep_expids,
                basal_expids=basal_expids,
                apical_expids=apical_expids,
                channel=channel,
                explicit_locomotion_threshold=config.get("locomotion_threshold"),
                cache_path=cache_path,
                rebuild=source_cache_rebuild,
                validate_existing_cache=source_cache_validate,
            )
    if config.get("demo_truth"):
        source_cache["demo_truth"] = config["demo_truth"]
        save_npz_cache(cache_path, source_cache)
    with step_scope("analysis-table cache load"):
        analysis_tables_cache = load_analysis_tables_cache(analysis_tables_cache_file, rebuild=analysis_tables_rebuild)
    analysis_tables = analysis_tables_cache.get("analysis_tables", {}) if isinstance(analysis_tables_cache, dict) else None
    analysis_cache_file = analysis_day_cache_path(cache_path)
    analysis_cache_expected_meta = analysis_day_cache_meta(source_cache, analysis_tables, analysis_unit="day")
    analysis_meta_ignore_keys = ("analysis_tables_signature",)
    analysis_cache_rebuild = bool(rebuild or source_cache_rebuild or analysis_tables_rebuild)
    with step_scope("day-level cache construction"):
        analysis_cache_payload, analysis_cache_status = load_analysis_day_cache(
            analysis_cache_file,
            expected_meta=analysis_cache_expected_meta,
            ignore_meta_keys=analysis_meta_ignore_keys,
            rebuild=analysis_cache_rebuild,
        )
        if analysis_cache_status == "ok" and isinstance(analysis_cache_payload, dict):
            analysis_cache = dict(analysis_cache_payload.get("analysis_cache", {}))
        else:
            analysis_cache = build_day_pooled_cache(
                source_cache,
                analysis_tables=analysis_tables,
            )
            save_analysis_day_cache(analysis_cache_file, analysis_cache, meta=analysis_cache_expected_meta)
    shuffle_state_labels = list(
        dict.fromkeys(
            [state for state in (state_comparison_states or []) + (basal_apical_states or []) if state]
        )
    )
    if not shuffle_state_labels:
        shuffle_state_labels = list(PRIMARY_QUIET_STATES)
    shared_shuffle_cache = None
    shared_shuffle_cache_file = None
    shared_shuffle_cache_rebuilt = False
    if not plots_only:
        with step_scope("shared circular-shift cache"):
            shared_shuffle_cache, shared_shuffle_cache_file, shared_shuffle_cache_rebuilt = load_or_build_shared_shuffle_cache(
                analysis_cache,
                shuffle_n,
                state_labels=shuffle_state_labels,
                cache_path=analysis_run_cache_path,
                rebuild=shared_shuffle_cache_rebuild,
            )
    else:
        shared_shuffle_cache = None
    from analysis_families.core import run_cached_analysis
    source_signature = source_cache_signature(source_cache)
    analysis_results_meta = {
        "analysis_unit": str(analysis_cache.get("analysis_unit", "day")),
        "analysis_cache_schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "source_config_hash": str(source_cache.get("config_hash", "")),
        "source_signature": source_signature,
        "analysis_config_hash": str(analysis_cache.get("config_hash", "")),
        "state_comparison_states": list(state_comparison_states or []),
        "basal_apical_states": list(basal_apical_states or []),
        "spine_coactivity_anchor_state": SPINE_COACTIVITY_ANCHOR_STATE,
        "spine_coactivity_abs_threshold": spine_coactivity_abs_threshold,
        "event_detection_method": event_detection_method,
        "visual_response_metric": visual_response_metric,
        "state_mode": selection_meta.get("state_mode"),
        "movie_trial_types": list(selection_meta.get("movie_trial_types") or []),
        "compare_states": list(selection_meta.get("compare_states") or []) if selection_meta.get("compare_states") is not None else None,
        "mixed_model_contrast_p_source": mixed_model_contrast_p_source,
        "analysis_families": list(config.get("analysis_families") or []),
        "shuffle_n": int(shuffle_n),
        "comparison_preset_name": str(config.get("comparison_preset_name") or "default"),
        "dendrite_response_cohort": str(config.get("dendrite_response_cohort", DEFAULT_DENDRITE_RESPONSE_COHORT) or DEFAULT_DENDRITE_RESPONSE_COHORT),
        "spine_visual_response_cohort": str(config.get("dendrite_response_cohort", DEFAULT_DENDRITE_RESPONSE_COHORT) or DEFAULT_DENDRITE_RESPONSE_COHORT),
        "dendrite_visual_response_classifier_type": "dendrite",
        "spine_visual_response_classifier_type": "spine",
        "visual_response_classifier_method": VISUAL_RESPONSE_CLASSIFIER_METHOD,
        "visual_response_classifier_version": VISUAL_RESPONSE_CLASSIFIER_VERSION,
        "visual_response_covariate": VISUAL_RESPONSE_COVARIATE_NAME,
        "visual_response_trial_types": list(VISUAL_RESPONSE_VISUAL_TRIAL_TYPES),
        "visual_response_blank_trial_type": VISUAL_RESPONSE_BLANK_TRIAL_TYPE,
        "fit_spine_coactivity_mixed_model": bool(config.get("fit_spine_coactivity_mixed_model")),
        "spine_coactivity_only": bool(config.get("spine_coactivity_only")),
        "mixed_model_only": bool(config.get("mixed_model_only")),
        "shared_shuffle_signature": str(shared_shuffle_cache.get("signature", "")) if isinstance(shared_shuffle_cache, dict) else "",
        "shared_shuffle_shuffle_n": int(shared_shuffle_cache.get("shuffle_n", shuffle_n)) if isinstance(shared_shuffle_cache, dict) else int(shuffle_n),
        "analysis_run_cache_path": str(analysis_run_cache_path),
    }
    plots_only_ignore_meta_keys = (
        "shared_shuffle_signature",
        "shared_shuffle_shuffle_n",
        "dendrite_response_cohort",
        "spine_visual_response_cohort",
        "dendrite_visual_response_classifier_type",
        "spine_visual_response_classifier_type",
        "visual_response_classifier_method",
        "visual_response_classifier_version",
        "visual_response_covariate",
        "visual_response_trial_types",
        "visual_response_blank_trial_type",
    ) if plots_only else None
    results_meta_ignore_keys = analysis_meta_ignore_keys + tuple(plots_only_ignore_meta_keys or ())
    family_result_stage = family_results_cache_stage_for_selection(config.get("analysis_families"))
    family_results_cache_file = family_results_cache_path(analysis_run_cache_path, family_result_stage)
    analysis_results_cache_file_for_run = analysis_results_cache_file
    analysis_results_cache, analysis_results_cache_status = load_analysis_results_cache(
            analysis_results_cache_file,
            expected_meta=analysis_results_meta,
            ignore_meta_keys=results_meta_ignore_keys,
            rebuild=False if plots_only else analysis_results_rebuild,
        )
    if plots_only and analysis_results_cache_status == "meta_mismatch":
        try:
            fallback_cache = load_npz_cache(analysis_results_cache_file)
        except Exception:
            fallback_cache = None
        if isinstance(fallback_cache, dict) and fallback_cache.get("schema_version") == ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION and isinstance(fallback_cache.get("analysis_results"), dict):
            step_message("plots_only cache metadata differed, but the saved analysis-results cache is structurally valid; reusing it for figure regeneration")
            analysis_results_cache = fallback_cache
            analysis_results_cache_status = "ok"
    if plots_only and analysis_results_cache_status != "ok":
        family_expected_meta = dict(analysis_results_meta)
        family_expected_meta["family_result_stage"] = family_result_stage
        family_results_cache, family_results_cache_status = load_analysis_results_cache(
            family_results_cache_file,
            expected_meta=family_expected_meta,
            ignore_meta_keys=results_meta_ignore_keys,
            rebuild=False,
        )
        if family_results_cache_status == "meta_mismatch":
            try:
                fallback_cache = load_npz_cache(family_results_cache_file)
            except Exception:
                fallback_cache = None
            if isinstance(fallback_cache, dict) and fallback_cache.get("schema_version") == ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION and isinstance(fallback_cache.get("analysis_results"), dict):
                step_message(f"plots_only family cache metadata differed, but the saved family-results cache is structurally valid; reusing it for figure regeneration: {family_results_cache_file}")
                family_results_cache = fallback_cache
                family_results_cache_status = "ok"
        if family_results_cache_status == "ok":
            step_message(f"plots_only family-results cache reused for figure regeneration: {family_results_cache_file}")
            analysis_results_cache = family_results_cache
            analysis_results_cache_status = "ok"
            analysis_results_cache_file_for_run = family_results_cache_file
    if plots_only and analysis_results_cache_status != "ok":
        raise SystemExit(f"plots_only requires a valid compatible analysis results cache at {analysis_results_cache_file} or family cache at {family_results_cache_file}; status was {analysis_results_cache_status}")
    source_summary = summarize_cache(source_cache)
    analysis_summary = summarize_cache(analysis_cache)
    # Print a compact cache summary so users can see what was loaded before the full analysis runs.
    info(
        json.dumps(
            jsonable(
                {
                    "analysis": analysis_summary,
                    "source": source_summary,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    # The cached analysis produces the final JSON, CSV, and figure outputs.
    for alert in selection_meta.get("alerts", []):
        eprint(alert)
    with step_scope("analysis families"):
        if plots_only and analysis_results_cache_status == "ok" and analysis_results_cache is not None:
            results = dict(analysis_results_cache.get("analysis_results", {}))
        elif plots_only:
            raise SystemExit(
                f"plots_only requires a valid compatible analysis-results cache at "
                f"{analysis_results_cache_file}; status was {analysis_results_cache_status}"
            )
        elif analysis_results_cache is not None:
            results = dict(analysis_results_cache.get("analysis_results", {}))
        elif bool(config.get("spine_coactivity_only")):
            results = run_cached_analysis(
                analysis_cache,
                shuffle_n=shuffle_n,
                state_comparison_states=state_comparison_states,
                basal_apical_states=basal_apical_states,
                source_cache=source_cache,
                shared_shuffle_cache=shared_shuffle_cache,
                output_dir=output_dir,
                figure_root=figure_output_dir,
                fit_spine_coactivity_mixed_model=bool(config.get("fit_spine_coactivity_mixed_model")),
                mixed_model_contrast_p_source=str(config.get("mixed_model_contrast_p_source") or "classical"),
                spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
                analysis_families=["spine_coactivity"],
                analysis_results_meta=analysis_results_meta,
                cache_path=analysis_run_cache_path,
            )
        elif bool(config.get("mixed_model_only")):
            results = run_cached_analysis(
                analysis_cache,
                shuffle_n=shuffle_n,
                state_comparison_states=state_comparison_states,
                basal_apical_states=basal_apical_states,
                source_cache=source_cache,
                shared_shuffle_cache=shared_shuffle_cache,
                output_dir=output_dir,
                figure_root=figure_output_dir,
                fit_spine_coactivity_mixed_model=bool(config.get("fit_spine_coactivity_mixed_model")),
                mixed_model_contrast_p_source=str(config.get("mixed_model_contrast_p_source") or "classical"),
                spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
                analysis_families=["mixed_model"],
                analysis_results_meta=analysis_results_meta,
                cache_path=analysis_run_cache_path,
            )
        else:
            results = run_cached_analysis(
                analysis_cache,
                shuffle_n=shuffle_n,
                state_comparison_states=state_comparison_states,
                basal_apical_states=basal_apical_states,
                source_cache=source_cache,
                shared_shuffle_cache=shared_shuffle_cache,
                output_dir=output_dir,
                figure_root=figure_output_dir,
                fit_spine_coactivity_mixed_model=bool(config.get("fit_spine_coactivity_mixed_model")),
                mixed_model_contrast_p_source=str(config.get("mixed_model_contrast_p_source") or "classical"),
                spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
                analysis_families=config.get("analysis_families"),
                analysis_results_meta=analysis_results_meta,
                cache_path=analysis_run_cache_path,
            )
    results.setdefault("alerts", []).extend(selection_meta.get("alerts", []))
    for alert in dict.fromkeys(results.get("alerts", []) + results.get("mixed_model", {}).get("alerts", [])):
        eprint(alert)
    results["analysis_cache_summary"] = analysis_summary
    results["source_cache_summary"] = source_summary
    results["cache_summary"] = analysis_summary
    results["analysis_unit"] = analysis_cache.get("analysis_unit", "day")
    if "analysis_mode" not in results:
        if bool(config.get("spine_coactivity_only")):
            results["analysis_mode"] = "spine_coactivity_only"
        elif bool(config.get("mixed_model_only")):
            results["analysis_mode"] = "mixed_model_only"
        elif bool(config.get("poster_ready_only")):
            results["analysis_mode"] = "poster_ready_only"
        else:
            results["analysis_mode"] = "full"
    results["config"] = source_cache.get("config", {})
    results["job_spec"] = AnalysisJobSpec(
        pipeline=str(config.get("analysis_name") or "dendrites_pipeline"),
        split_type="batches",
        state_basis="overall",
        analysis_type=str(config.get("comparison_preset_name") or "default"),
        cohort="all",
    ).as_dict()
    results["run_parameters"] = {
        "channel": channel,
        "shuffle_n": shuffle_n,
        "cpu_thread_limit": cpu_thread_limit,
        "locomotion_threshold": config.get("locomotion_threshold"),
        "spine_coactivity_abs_threshold": spine_coactivity_abs_threshold,
        "rebuild": rebuild,
        "source_cache_rebuild": source_cache_rebuild,
        "analysis_tables_rebuild": analysis_tables_rebuild,
        "analysis_results_rebuild": analysis_results_rebuild,
        "shared_shuffle_cache_rebuild": shared_shuffle_cache_rebuild,
        "spine_coactivity_only": bool(config.get("spine_coactivity_only")),
        "mixed_model_only": bool(config.get("mixed_model_only")),
        "mixed_model_contrast_p_source": mixed_model_contrast_p_source,
        "analysis_families": list(config.get("analysis_families") or []),
        "comparison_preset_name": str(config.get("comparison_preset_name") or "default"),
        "generate_shared_general_figures": bool(config.get("generate_shared_general_figures", True)),
        "poster_ready_only": bool(config.get("poster_ready_only")),
        "analysis_run_cache_path": str(analysis_run_cache_path),
        "state_mode": selection_meta.get("state_mode"),
        "movie_trial_types": selection_meta.get("movie_trial_types"),
        "compare_states": selection_meta.get("compare_states"),
        "spine_coactivity_anchor_state": SPINE_COACTIVITY_ANCHOR_STATE,
        "source_cache_validate": source_cache_validate,
        "output_dir": str(output_dir),
        "cache_path": str(cache_path),
        "analysis_tables_cache_path": str(analysis_tables_cache_file),
        "analysis_results_cache_path": str(analysis_results_cache_file_for_run),
    }
    results["analysis_state_selection"] = {
        "compare_states": selection_meta.get("compare_states"),
        "comparison_preset_name": str(config.get("comparison_preset_name") or "default"),
        "state_mode": selection_meta.get("state_mode"),
        "mixed_model_contrast_p_source": mixed_model_contrast_p_source,
        "movie_trial_types": selection_meta.get("movie_trial_types"),
        "dendrite_response_cohort": str(config.get("dendrite_response_cohort", DEFAULT_DENDRITE_RESPONSE_COHORT) or DEFAULT_DENDRITE_RESPONSE_COHORT),
        "spine_visual_response_cohort": str(config.get("dendrite_response_cohort", DEFAULT_DENDRITE_RESPONSE_COHORT) or DEFAULT_DENDRITE_RESPONSE_COHORT),
        "spine_coactivity_anchor_state": SPINE_COACTIVITY_ANCHOR_STATE,
        "spine_coactivity_abs_threshold": spine_coactivity_abs_threshold,
        "spine_coactivity_selection_rule": spine_coactivity_anchor_selection_text(spine_coactivity_abs_threshold),
        "state_comparison_states": state_comparison_states,
        "basal_apical_states": basal_apical_states,
        "state_mode_source": selection_meta.get("state_mode_source"),
        "movie_trial_types_source": selection_meta.get("movie_trial_types_source"),
        "alerts": list(selection_meta.get("alerts", [])),
    }
    results["family_result_cache_index"] = family_results_cache_index(analysis_run_cache_path)
    # Save the analysis-results cache before figure generation so `plots_only` can still reuse it
    # even if a later plot or poster step fails.
    early_analysis_results_payload = {
        "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
        "meta": cacheable(analysis_results_meta),
        "meta_hash": analysis_cache_meta_hash(analysis_results_meta),
        "analysis_results": cacheable(analysis_results_cache_payload(results)),
    }
    save_analysis_results_cache(analysis_results_cache_file, early_analysis_results_payload)
    with step_scope("analysis outputs"):
        written_artifacts = write_analysis_outputs(
            output_dir,
            results,
            analysis_cache,
            source_cache=source_cache,
            figure_root=figure_output_dir,
            plots_only=plots_only,
            include_supporting_figures=bool(config.get("plots_only_include_supporting_figures", not plots_only)) and not bool(config.get("poster_ready_only")),
            branch_first_figures=bool(config.get("branch_first_figures", True)),
            branch_first_output_root=resolve_repo_path(config.get("branch_first_output_root"), REPO_ROOT) if config.get("branch_first_output_root") else None,
        )
    if bool(config.get("generate_poster_ready_figures", True)):
        with step_scope("poster figure generation"):
            written_artifacts.extend(
                write_poster_ready_figures(
                    output_dir,
                    analysis_cache,
                    source_cache,
                    results,
                    analysis_families=None if bool(config.get("poster_ready_only")) else config.get("analysis_families"),
                )
            )
    if isinstance(analysis_cache.get(STATE_SUMMARY_PAYLOAD_CACHE_KEY), dict):
        save_analysis_day_cache(analysis_cache_file, analysis_cache, meta=analysis_cache_expected_meta)
    results["shared_shuffle_cache"] = {
        "path": str(shared_shuffle_cache_file) if shared_shuffle_cache_file is not None else None,
        "reused": not bool(shared_shuffle_cache_rebuilt),
        "shuffle_n": shuffle_n,
        "entry_count": int(len(shared_shuffle_cache.get("entries", {}))) if isinstance(shared_shuffle_cache, dict) else 0,
    }
    report_path: Optional[Path] = None
    if plots_only:
        results["analysis_mode"] = "plots_only"
        results["output_artifacts"] = list(dict.fromkeys(written_artifacts))
        results.pop("analysis_report_path", None)
    else:
        report_path = output_dir / "analysis_report.txt"
        results["analysis_report_path"] = str(report_path)
        results["output_artifacts"] = list(
            dict.fromkeys(
                written_artifacts
                + [report_relative_path(cache_path, output_dir), report_relative_path(report_path, output_dir)]
            )
        )
        with step_scope("report / artifact writing"):
            write_analysis_report(report_path, output_dir, results, analysis_cache, source_cache, cache_path, results["output_artifacts"])
        source_cache_payload = dict(source_cache)
        source_cache_payload.pop("analysis_tables", None)
        source_cache_payload.pop("analysis_results", None)
        save_npz_cache(cache_path, source_cache_payload)
    results["output_artifacts"] = list(dict.fromkeys(list(results.get("output_artifacts", [])) + collect_output_artifacts(output_dir)))
    write_manifest(output_dir, jsonable(results))
    poster_ready_outputs = [path for path in written_artifacts if "/poster_ready/" in str(path)]
    if poster_ready_outputs:
        step_message(f"poster-ready outputs ({len(poster_ready_outputs)}):")
        for artifact_path in poster_ready_outputs:
            print(f"  - {artifact_path}")
        analysis_tables_payload = {
            "schema_version": ANALYSIS_TABLE_CACHE_SCHEMA_VERSION,
            "meta": cacheable({
                "analysis_unit": str(analysis_cache.get("analysis_unit", "day")),
                "source_config_hash": str(source_cache.get("config_hash", "")),
            }),
            "meta_hash": analysis_cache_meta_hash({
                "analysis_unit": str(analysis_cache.get("analysis_unit", "day")),
                "source_config_hash": str(source_cache.get("config_hash", "")),
            }),
            "analysis_tables": cacheable(analysis_cache.get("analysis_tables", {}) if isinstance(analysis_cache.get("analysis_tables", {}), dict) else {}),
        }
        save_analysis_tables_cache(analysis_tables_cache_file, analysis_tables_payload)
        analysis_results_payload = {
            "schema_version": ANALYSIS_RESULTS_CACHE_SCHEMA_VERSION,
            "meta": cacheable(analysis_results_meta),
            "meta_hash": analysis_cache_meta_hash(analysis_results_meta),
            "analysis_results": cacheable(analysis_results_cache_payload(results)),
        }
        save_analysis_results_cache(analysis_results_cache_file, analysis_results_payload)
        if shared_shuffle_cache_file is not None and isinstance(shared_shuffle_cache, dict):
            save_shared_shuffle_cache(shared_shuffle_cache_file, shared_shuffle_cache)
        info(f"Cache saved to: {cache_path}")
        if report_path is not None:
            info(f"Report saved to: {report_path}")
        else:
            info(f"Report not rewritten in plots-only mode; existing report remains at: {output_dir / 'analysis_report.txt'}")
        info(f"Checkpoint gallery saved to: {output_dir / DEFAULT_CHECKPOINT_GALLERY_DIRNAME}")
        info(f"Review figures saved to: {output_dir / DEFAULT_REVIEW_FIGURES_DIRNAME}")
        info(f"Results saved to: {output_dir}")
    run_issues = list(dict.fromkeys(results.get("alerts", []) + results.get("mixed_model", {}).get("alerts", [])))
    if run_issues:
        info(f"Issues encountered ({len(run_issues)}):")
        for issue in run_issues:
            info(f"  - {issue}")
    else:
        info("Issues encountered: none")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
