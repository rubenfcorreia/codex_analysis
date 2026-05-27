#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Polygon, Rectangle
except Exception:  # pragma: no cover - matplotlib is required for the real run
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
    compose_svg_figure,
    configure_poster_matplotlib,
    rasterize_svg_to_png,
    save_figure,
    set_hour_ticks,
    set_sparse_numeric_ticks,
)

from sleep_dendrite_spine_pipeline import (
    as_float,
    cleanup_roi_detail_figures,
    derive_animal_id,
    derive_date,
    determine_conversion_mode,
    ensure_dir,
    format_dendrite_display_name,
    load_conversion_library,
    load_npz_cache,
    locate_conversion_file,
    step_message,
    step_progress,
    step_scope,
    safe_filename_component,
)

if plt is not None:
    configure_poster_matplotlib()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "sleep_dendrite_spine_example_config.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "main_pipeline"
DEFAULT_CACHE_PATH = DEFAULT_OUTPUT_DIR / "sleep_dendrite_spine_cache.npz"
DEFAULT_FIGURE_PREFIX = "full_day_summary"
DEFAULT_DPI = POSTER_DPI

# Keep the requested labels first-class in the legend and timeline.
BROAD_STATE_ORDER = [
    "blank",
    "movie",
    "zebra",
    "grating",
    "quiet_awake",
    "active_awake",
    "nrem",
    "rem",
]
BROAD_STATE_COLORS = {
    "blank": "#d0d0d0",
    "movie": "#4c78a8",
    "zebra": "#f58518",
    "grating": "#54a24b",
    "quiet_awake": "#72b7b2",
    "active_awake": "#e45756",
    "nrem": "#b279a2",
    "rem": "#ffbf00",
}
BROAD_STATE_TO_CODE = {label: idx for idx, label in enumerate(BROAD_STATE_ORDER)}
CODE_TO_BROAD_STATE = {idx: label for label, idx in BROAD_STATE_TO_CODE.items()}

MOVIE_SESSION_LABELS = {
    "blank": ("quiet_blank", "active_blank"),
    "movie": ("quiet_movies", "active_movies"),
    "zebra": ("quiet_zebra", "active_zebra"),
    "grating": ("quiet_grating", "active_grating"),
}
SLEEP_SESSION_LABELS = {
    "quiet_awake": ("quiet_awake",),
    "active_awake": ("active_awake",),
    "nrem": ("nrem",),
    "rem": ("rem",),
}


@dataclass
class SessionSummary:
    exp_id: str
    offset_bins: int
    n_bins: int
    x_hours: np.ndarray
    label_codes: np.ndarray
    compartment: str


@dataclass
class DaySummary:
    animal_id: str
    date: str
    exp_ids: List[str]
    sessions: List[SessionSummary]
    total_bins: int
    state_codes: np.ndarray
    x_hours: np.ndarray
    boundary_hours: np.ndarray


def load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def parse_list_argument(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def format_compartment_label(compartment: Any) -> str:
    text = str(compartment or "").strip().lower()
    if text in {"basal", "apical"}:
        return text.capitalize()
    if text:
        return text.capitalize()
    return "Unknown"


def normalized_compartment_folder(compartment: Any) -> str:
    text = str(compartment or "").strip().lower()
    if text in {"basal", "apical"}:
        return text
    return "other"


def extract_dendrite_token(global_dendrite_id: str) -> str:
    parts = [part for part in str(global_dendrite_id).split("|") if part]
    if not parts:
        return "unknown"
    return parts[-1]


def build_figure_save_path(output_dir: Path, animal_id: str, date: str, compartment: Any, global_dendrite_id: str) -> Path:
    animal_slug = safe_filename_component(animal_id)
    compartment_slug = safe_filename_component(normalized_compartment_folder(compartment))
    date_slug = safe_filename_component(date)
    dendrite_slug = safe_filename_component(extract_dendrite_token(global_dendrite_id))
    figure_dir = output_dir / "figures" / animal_slug / compartment_slug / date_slug
    figure_name = f"{animal_slug}_{compartment_slug}_{date_slug}_{dendrite_slug}.svg"
    return figure_dir / figure_name


def build_roi_detail_save_path(output_dir: Path, animal_id: str, date: str, compartment: Any, global_dendrite_id: str) -> Path:
    animal_slug = safe_filename_component(animal_id)
    compartment_slug = safe_filename_component(normalized_compartment_folder(compartment))
    date_slug = safe_filename_component(date)
    dendrite_slug = safe_filename_component(extract_dendrite_token(global_dendrite_id))
    figure_dir = output_dir / animal_slug / compartment_slug / date_slug
    figure_name = f"{animal_slug}_{compartment_slug}_{date_slug}_{dendrite_slug}_detail.svg"
    return figure_dir / figure_name


def configured_compartment_for_exp_id(
    exp_id: str,
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
) -> Optional[str]:
    if exp_id in basal_expids:
        return "basal"
    if exp_id in apical_expids:
        return "apical"
    return None


def warn_if_compartment_mismatch(
    exp_id: str,
    expected_compartment: Optional[str],
    cached_compartment: Any,
) -> None:
    if expected_compartment is None:
        return
    cached_raw = str(cached_compartment or "").strip()
    cached_normalized = normalized_compartment_folder(cached_raw)
    if cached_normalized == expected_compartment:
        return
    warnings.warn(
        (
            f"Compartment mismatch for {exp_id}: config expects {expected_compartment}, "
            f"but the cache labels it as {cached_raw or 'missing'} "
            f"(saved folder would be {cached_normalized}). Rebuild the cache if the config was updated."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


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


def as_bool_array(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=bool)
    arr = np.asarray(value, dtype=bool).ravel()
    if arr.size < length:
        out = np.zeros(length, dtype=bool)
        out[: arr.size] = arr
        return out
    return arr[:length]


def combine_masks(state_masks: Dict[str, Any], keys: Sequence[str], length: int) -> np.ndarray:
    combined = np.zeros(length, dtype=bool)
    for key in keys:
        combined |= as_bool_array(state_masks.get(key), length)
    return combined


def build_sample_labels(exp_meta: Dict[str, Any]) -> np.ndarray:
    t = np.asarray(exp_meta.get("time"), dtype=float).ravel()
    if t.size == 0:
        return np.asarray([], dtype=object)
    length = t.size
    labels = np.full(length, "blank", dtype=object)
    state_masks = exp_meta.get("state_masks", {}) or {}
    compartment = str(exp_meta.get("compartment") or "")
    movie_present = any(combine_masks(state_masks, keys, length).any() for keys in MOVIE_SESSION_LABELS.values())
    sleep_present = any(combine_masks(state_masks, keys, length).any() for keys in SLEEP_SESSION_LABELS.values())
    use_sleep_labels = compartment == "sleep" or (sleep_present and not movie_present)
    if use_sleep_labels:
        for broad_label, keys in SLEEP_SESSION_LABELS.items():
            labels[combine_masks(state_masks, keys, length)] = broad_label
    else:
        for broad_label, keys in MOVIE_SESSION_LABELS.items():
            if broad_label == "blank":
                labels[combine_masks(state_masks, keys, length)] = "blank"
            else:
                labels[combine_masks(state_masks, keys, length)] = broad_label
    return labels


def bin_mode_labels(labels: np.ndarray, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(n_bins, "blank", dtype=object)
    for bin_no in range(n_bins):
        idx = np.flatnonzero(bin_index == bin_no)
        if idx.size == 0:
            continue
        values = labels[idx]
        values = values[np.asarray([str(v).strip() != "" for v in values], dtype=bool)]
        if values.size == 0:
            continue
        counts = Counter(values.tolist())
        out[bin_no] = counts.most_common(1)[0][0]
    return out


def bin_mean(values: np.ndarray, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return np.full(n_bins, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.full(n_bins, np.nan, dtype=float)
    sums = np.bincount(bin_index[valid], weights=values[valid], minlength=n_bins)
    counts = np.bincount(bin_index[valid], minlength=n_bins)
    out = np.full(n_bins, np.nan, dtype=float)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def make_session_summary(exp_id: str, exp_meta: Dict[str, Any], offset_bins: int) -> SessionSummary:
    t = np.asarray(exp_meta.get("time"), dtype=float).ravel()
    if t.size == 0:
        return SessionSummary(
            exp_id=exp_id,
            offset_bins=offset_bins,
            n_bins=1,
            x_hours=np.asarray([offset_bins + 0.5], dtype=float) / 3600.0,
            label_codes=np.asarray([BROAD_STATE_TO_CODE["blank"]], dtype=int),
            compartment=str(exp_meta.get("compartment") or ""),
        )

    rel_t = t - float(t[0])
    if rel_t.size == 0:
        rel_t = np.asarray([0.0], dtype=float)
    n_bins = max(1, int(np.ceil(float(rel_t[-1]))))
    bin_index = np.clip(np.floor(rel_t).astype(int), 0, n_bins - 1)
    sample_labels = build_sample_labels(exp_meta)
    sample_labels, bin_index = align_length(sample_labels, bin_index)
    label_bins = bin_mode_labels(sample_labels, bin_index.astype(int), n_bins)
    label_codes = np.asarray([BROAD_STATE_TO_CODE.get(str(label), BROAD_STATE_TO_CODE["blank"]) for label in label_bins], dtype=int)
    x_hours = (offset_bins + np.arange(n_bins, dtype=float) + 0.5) / 3600.0
    return SessionSummary(
        exp_id=exp_id,
        offset_bins=offset_bins,
        n_bins=n_bins,
        x_hours=x_hours,
        label_codes=label_codes,
        compartment=str(exp_meta.get("compartment") or ""),
    )


def build_day_summary(animal_id: str, date: str, exp_ids: Sequence[str], cache: Dict[str, Any]) -> DaySummary:
    exp_ids = list(exp_ids)
    offset_bins = 0
    sessions: List[SessionSummary] = []
    state_codes: List[np.ndarray] = []
    for exp_id in exp_ids:
        exp_meta = cache["experiments"].get(exp_id)
        if exp_meta is None:
            continue
        session = make_session_summary(exp_id, exp_meta, offset_bins)
        sessions.append(session)
        state_codes.append(session.label_codes)
        offset_bins += session.n_bins
    total_bins = offset_bins
    if total_bins == 0:
        total_bins = 1
    stitched_state_codes = np.concatenate(state_codes) if state_codes else np.asarray([BROAD_STATE_TO_CODE["blank"]], dtype=int)
    x_hours = (np.arange(stitched_state_codes.size, dtype=float) + 0.5) / 3600.0
    boundary_hours = np.asarray([s.offset_bins / 3600.0 for s in sessions[1:]], dtype=float) if len(sessions) > 1 else np.asarray([], dtype=float)
    return DaySummary(
        animal_id=animal_id,
        date=date,
        exp_ids=exp_ids,
        sessions=sessions,
        total_bins=total_bins,
        state_codes=stitched_state_codes,
        x_hours=x_hours,
        boundary_hours=boundary_hours,
    )


def grouped_day_expids(config: Dict[str, Any], cache: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    movie_expids = parse_list_argument(config.get("movie_expids"))
    sleep_expids = parse_list_argument(config.get("sleep_expids"))
    selected = sorted(set(movie_expids) | set(sleep_expids))
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for exp_id in selected:
        exp_meta = cache.get("experiments", {}).get(exp_id)
        if exp_meta is None:
            continue
        animal_id = str(exp_meta.get("animal_id") or derive_animal_id(exp_id))
        date = str(exp_meta.get("date") or derive_date(exp_id))
        groups[(animal_id, date)].append(exp_id)
    for key in list(groups.keys()):
        groups[key] = sorted(groups[key])
    return dict(sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])))


def describe_day_group_mismatch(
    config: Dict[str, Any],
    cache: Dict[str, Any],
    *,
    config_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    preview_limit: int = 5,
) -> str:
    requested = sorted(set(parse_list_argument(config.get("movie_expids"))) | set(parse_list_argument(config.get("sleep_expids"))))
    available = sorted(str(exp_id) for exp_id in cache.get("experiments", {}).keys())
    cache_config = cache.get("config", {}) or {}
    cache_requested = sorted(
        set(parse_list_argument(cache_config.get("movie_expids"))) | set(parse_list_argument(cache_config.get("sleep_expids")))
    )

    def preview(items: Sequence[str]) -> str:
        if not items:
            return "none"
        head = ", ".join(items[:preview_limit])
        if len(items) <= preview_limit:
            return head
        return f"{head}, ... (+{len(items) - preview_limit} more)"

    lines = ["No configured expIDs were found in the cache."]
    if config_path is not None:
        lines.append(f"Config file: {config_path}")
    if cache_path is not None:
        lines.append(f"Cache file: {cache_path}")
    lines.append(f"Requested expIDs ({len(requested)}): {preview(requested)}")
    lines.append(f"Cache expIDs ({len(available)}): {preview(available)}")
    if requested and available and not (set(requested) & set(available)):
        lines.append("The config expIDs and cache expIDs do not overlap.")
    if cache_requested and cache_requested != requested:
        lines.append(f"Cache was built for expIDs ({len(cache_requested)}): {preview(cache_requested)}")
    repo_base = cache_config.get("repo_base")
    if repo_base:
        lines.append(f"Cache repo_base: {repo_base}")
    lines.append("Use a matching --config/--cache-path pair, or rebuild the cache from the desired data.")
    return "\n".join(lines)


def selected_dendrite_records(cache: Dict[str, Any], animal_id: str, date: str, exp_ids: Sequence[str]) -> List[Tuple[str, Dict[str, Any]]]:
    exp_id_set = set(exp_ids)
    animal_entry = cache.get("animals", {}).get(animal_id, {})
    records: List[Tuple[str, Dict[str, Any]]] = []
    for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
        if str(dendrite_record.get("date") or "") != date:
            continue
        obs_ids = [exp_id for exp_id in dendrite_record.get("observations", {}) if exp_id in exp_id_set]
        if obs_ids:
            records.append((global_dendrite_id, dendrite_record))
    return records


def choose_representative_exp_id(dendrite_record: Dict[str, Any], day_exp_ids: Sequence[str]) -> Optional[str]:
    day_exp_ids = list(day_exp_ids)
    observations = dendrite_record.get("observations", {})
    spine_records = dendrite_record.get("spines", {})
    for exp_id in day_exp_ids:
        if exp_id not in observations:
            continue
        if any(exp_id in spine.get("observations", {}) for spine in spine_records.values()):
            return exp_id
    for exp_id in day_exp_ids:
        if exp_id in observations:
            return exp_id
    return None


def resolve_conversion_path(exp_meta: Dict[str, Any], animal_id: str, exp_id: str, repo_base: Optional[Path]) -> Path:
    conversion = exp_meta.get("conversion", {}) or {}
    path_value = conversion.get("path")
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path
    if repo_base is not None:
        conv_path, _, _ = locate_conversion_file(repo_base, animal_id, exp_id)
        if conv_path is not None:
            return conv_path
    raise FileNotFoundError(f"Could not resolve a SpinesGUI conversion file for {exp_id}")


def resolve_ops_path(exp_root: Path, plane: int) -> Path:
    candidate_paths = [
        exp_root / "suite2p" / f"plane{plane}" / "ops.npy",
        exp_root / "suite2p" / "ch2" / f"plane{plane}" / "ops.npy",
        exp_root / "ch2" / "suite2p" / f"plane{plane}" / "ops.npy",
        exp_root / "suite2p_combined" / f"plane{plane}" / "ops.npy",
        exp_root / "suite2p_combined" / "ch2" / f"plane{plane}" / "ops.npy",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ops.npy for {exp_root} plane {plane}")


def load_mean_image(ops_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    ops = np.load(ops_path, allow_pickle=True).item()
    mean_img = ops.get("meanImg")
    if mean_img is None or np.asarray(mean_img).ndim != 2 or not np.isfinite(np.asarray(mean_img)).any():
        mean_img = ops.get("meanImgE")
    if mean_img is None:
        raise KeyError(f"No meanImg/meanImgE array found in {ops_path}")
    return np.asarray(mean_img, dtype=float), ops


def load_stat_path(ops_path: Path) -> Optional[Path]:
    stat_path = ops_path.with_name("stat.npy")
    return stat_path if stat_path.exists() else None


def extract_roi_coordinates(
    raw_conversion_entry: Dict[str, Any],
    stat_entry: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    coords = raw_conversion_entry.get("ROI coordinates")
    if coords is not None:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.size > 0:
            return arr
    if stat_entry is not None:
        xpix = stat_entry.get("xpix")
        ypix = stat_entry.get("ypix")
        if xpix is not None and ypix is not None:
            x = np.asarray(xpix, dtype=float).ravel()
            y = np.asarray(ypix, dtype=float).ravel()
            if x.size and y.size and x.size == y.size:
                return np.column_stack([x, y])
    return None


def load_contour_data(
    raw_library: Dict[Any, Any],
    general_roi_id: Any,
    plane_roi_id: Optional[int],
    stat_path: Optional[Path],
) -> Optional[np.ndarray]:
    raw_entry = raw_library.get(general_roi_id)
    if raw_entry is None:
        raw_entry = raw_library.get(int(general_roi_id)) if str(general_roi_id).isdigit() else None
    if raw_entry is None:
        return None
    stat_entry = None
    if stat_path is not None and plane_roi_id is not None:
        stat = np.load(stat_path, allow_pickle=True)
        if 0 <= plane_roi_id < stat.shape[0]:
            stat_entry = stat[plane_roi_id]
    return extract_roi_coordinates(raw_entry, stat_entry)


def assign_contour_colors(n_items: int) -> List[str]:
    palette = plt.get_cmap("tab10") if plt is not None else None
    if palette is None:
        return ["#333333"] * n_items
    return [palette(i % palette.N) for i in range(n_items)]


def choose_roi_label_anchor(
    coords: np.ndarray,
    image_shape: Sequence[int],
    occupied_label_boxes: Sequence[Tuple[float, float, float, float]],
    nearby_boxes: Sequence[Tuple[float, float, float, float]] = (),
    label_text: str = "",
) -> Tuple[float, float, str, str]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.size == 0:
        return 0.5, 0.5, "center", "center"

    centroid = np.nanmean(coords, axis=0)
    min_x = float(np.nanmin(coords[:, 0]))
    max_x = float(np.nanmax(coords[:, 0]))
    min_y = float(np.nanmin(coords[:, 1]))
    max_y = float(np.nanmax(coords[:, 1]))
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    base_pad = max(5.5, 0.28 * max(width, height))
    far_pad = max(base_pad * 1.55, base_pad + 6.0)
    farther_pad = max(far_pad * 1.25, far_pad + 5.0)
    image_w = float(image_shape[1]) if len(image_shape) > 1 else 1.0
    image_h = float(image_shape[0]) if len(image_shape) > 0 else 1.0
    margin = max(4.0, base_pad * 0.4)
    display_label = str(label_text or "").replace("_", " ").strip()
    label_chars = max(len(display_label), 1)
    label_fontsize = float(POSTER_NOTE_SIZE)
    label_width = max(26.0, 0.85 * label_fontsize * label_chars + 10.0)
    label_height = max(15.0, 1.45 * label_fontsize)

    try:
        from matplotlib.path import Path as MplPath
    except Exception:  # pragma: no cover - matplotlib is required for real figure generation
        MplPath = None
    contour_path = MplPath(np.vstack([coords, coords[:1]])) if MplPath is not None else None

    def clamp_point(x: float, y: float) -> Tuple[float, float]:
        return (
            float(np.clip(x, margin, max(image_w - margin, margin))),
            float(np.clip(y, margin, max(image_h - margin, margin))),
        )

    def estimate_label_box(x: float, y: float, ha: str, va: str) -> Tuple[float, float, float, float]:
        if ha == "left":
            min_x_box, max_x_box = x, x + label_width
        elif ha == "right":
            min_x_box, max_x_box = x - label_width, x
        else:
            min_x_box, max_x_box = x - label_width / 2.0, x + label_width / 2.0

        if va == "bottom":
            min_y_box, max_y_box = y - label_height, y
        elif va == "top":
            min_y_box, max_y_box = y, y + label_height
        else:
            min_y_box, max_y_box = y - label_height / 2.0, y + label_height / 2.0

        return min_x_box, min_y_box, max_x_box, max_y_box

    def boxes_overlap(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
        pad: float = 0.0,
    ) -> bool:
        a_min_x, a_min_y, a_max_x, a_max_y = box_a
        b_min_x, b_min_y, b_max_x, b_max_y = box_b
        return not (
            a_max_x <= b_min_x - pad
            or b_max_x <= a_min_x - pad
            or a_max_y <= b_min_y - pad
            or b_max_y <= a_min_y - pad
        )

    def box_to_box_clearance(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        a_min_x, a_min_y, a_max_x, a_max_y = box_a
        b_min_x, b_min_y, b_max_x, b_max_y = box_b
        dx = max(b_min_x - a_max_x, a_min_x - b_max_x, 0.0)
        dy = max(b_min_y - a_max_y, a_min_y - b_max_y, 0.0)
        if dx == 0.0 and dy == 0.0:
            overlap_x = min(a_max_x, b_max_x) - max(a_min_x, b_min_x)
            overlap_y = min(a_max_y, b_max_y) - max(a_min_y, b_min_y)
            return -float(min(max(overlap_x, 0.0), max(overlap_y, 0.0)))
        return float(np.hypot(dx, dy))

    side_space = {
        "right": max(image_w - margin - max_x, 0.0),
        "left": max(min_x - margin, 0.0),
        "top": max(min_y - margin, 0.0),
        "bottom": max(image_h - margin - max_y, 0.0),
    }
    side_order = sorted(side_space, key=side_space.get, reverse=True)
    pads = [base_pad, max(base_pad * 1.35, base_pad + 4.0), far_pad, farther_pad]
    orthogonal_offsets = [
        0.0,
        label_height * 1.1,
        -label_height * 1.1,
        label_height * 2.2,
        -label_height * 2.2,
    ]

    def make_candidate(side: str, pad: float, shift: float) -> Tuple[float, float, str, str]:
        if side == "right":
            return max_x + pad, centroid[1] + shift, "left", "center"
        if side == "left":
            return min_x - pad, centroid[1] + shift, "right", "center"
        if side == "top":
            return centroid[0] + shift, min_y - pad, "center", "bottom"
        return centroid[0] + shift, max_y + pad, "center", "top"

    candidate_records: List[Tuple[Tuple[float, float, float, float, float, float, float, float], float, float, str, str]] = []
    for side_index, side in enumerate(side_order):
        for pad_index, pad in enumerate(pads):
            for shift_index, shift in enumerate(orthogonal_offsets):
                x, y, ha, va = make_candidate(side, pad, shift)
                x, y = clamp_point(float(x), float(y))
                label_box = estimate_label_box(x, y, ha, va)
                sample_points = [
                    (label_box[0], label_box[1]),
                    (label_box[0], label_box[3]),
                    (label_box[2], label_box[1]),
                    (label_box[2], label_box[3]),
                    ((label_box[0] + label_box[2]) / 2.0, (label_box[1] + label_box[3]) / 2.0),
                ]
                label_overlap_count = sum(1 for box in occupied_label_boxes if boxes_overlap(label_box, box, pad=3.0))
                nearby_overlap_count = sum(1 for box in nearby_boxes if boxes_overlap(label_box, box, pad=4.5))
                inside_contour = 1 if contour_path is not None and any(contour_path.contains_point(pt) for pt in sample_points) else 0
                out_of_bounds = 1 if (
                    label_box[0] < 0.0
                    or label_box[1] < 0.0
                    or label_box[2] > image_w
                    or label_box[3] > image_h
                ) else 0
                min_clearance = min(
                    [box_to_box_clearance(label_box, box) for box in occupied_label_boxes] +
                    [box_to_box_clearance(label_box, box) for box in nearby_boxes],
                    default=float("inf"),
                )
                border_clearance = min(
                    x - margin,
                    image_w - margin - x,
                    y - margin,
                    image_h - margin - y,
                )
                centroid_distance = abs(x - centroid[0]) + abs(y - centroid[1])
                score = (
                    -float(label_overlap_count),
                    -float(nearby_overlap_count),
                    -float(inside_contour),
                    -float(out_of_bounds),
                    float(min_clearance),
                    float(border_clearance),
                    -float(side_index),
                    -float(pad_index),
                    -float(shift_index),
                    -float(centroid_distance),
                )
                if label_overlap_count == 0 and nearby_overlap_count == 0 and inside_contour == 0 and out_of_bounds == 0:
                    return x, y, ha, va
                candidate_records.append((score, x, y, ha, va))

    if candidate_records:
        candidate_records.sort(reverse=True)
        _, x, y, ha, va = candidate_records[0]
        return x, y, ha, va

    image_center = np.asarray([image_w / 2.0, image_h / 2.0], dtype=float)
    direction = np.asarray(centroid, dtype=float) - image_center
    norm = float(np.hypot(direction[0], direction[1]))
    if not np.isfinite(norm) or norm <= 1e-9:
        direction = np.asarray([1.0, -1.0], dtype=float)
        norm = float(np.hypot(direction[0], direction[1]))
    unit = direction / norm
    fallback_x, fallback_y = clamp_point(float(centroid[0] + unit[0] * farther_pad), float(centroid[1] + unit[1] * farther_pad))
    ha = "left" if unit[0] >= 0 else "right"
    va = "bottom" if unit[1] <= 0 else "top"
    return fallback_x, fallback_y, ha, va


def choose_dendrite_label_anchor(
    coords: np.ndarray,
    image_shape: Sequence[int],
    occupied_label_boxes: Sequence[Tuple[float, float, float, float]],
    nearby_boxes: Sequence[Tuple[float, float, float, float]] = (),
    label_text: str = "",
) -> Tuple[float, float, str, str]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.size == 0:
        return 0.5, 0.5, "center", "center"

    centroid = np.nanmean(coords, axis=0)
    min_x = float(np.nanmin(coords[:, 0]))
    max_x = float(np.nanmax(coords[:, 0]))
    min_y = float(np.nanmin(coords[:, 1]))
    max_y = float(np.nanmax(coords[:, 1]))
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    image_w = float(image_shape[1]) if len(image_shape) > 1 else 1.0
    image_h = float(image_shape[0]) if len(image_shape) > 0 else 1.0
    margin = max(4.0, 0.24 * max(width, height))
    label_fontsize = float(POSTER_NOTE_SIZE + 2)
    display_label = str(label_text or "").replace("_", " ").strip()
    label_chars = max(len(display_label), 1)
    label_width = max(28.0, 0.86 * label_fontsize * label_chars + 10.0)
    label_height = max(16.0, 1.42 * label_fontsize)
    pad = max(4.0, 0.22 * max(width, height))
    mid_shift = label_height * 0.55
    side_shift = label_height * 0.9

    try:
        from matplotlib.path import Path as MplPath
    except Exception:  # pragma: no cover - matplotlib is required for real figure generation
        MplPath = None
    contour_path = MplPath(np.vstack([coords, coords[:1]])) if MplPath is not None else None

    def clamp_point(x: float, y: float) -> Tuple[float, float]:
        return (
            float(np.clip(x, margin, max(image_w - margin, margin))),
            float(np.clip(y, margin, max(image_h - margin, margin))),
        )

    def estimate_label_box(x: float, y: float, ha: str, va: str) -> Tuple[float, float, float, float]:
        if ha == "left":
            min_x_box, max_x_box = x, x + label_width
        elif ha == "right":
            min_x_box, max_x_box = x - label_width, x
        else:
            min_x_box, max_x_box = x - label_width / 2.0, x + label_width / 2.0

        if va == "bottom":
            min_y_box, max_y_box = y - label_height, y
        elif va == "top":
            min_y_box, max_y_box = y, y + label_height
        else:
            min_y_box, max_y_box = y - label_height / 2.0, y + label_height / 2.0

        return min_x_box, min_y_box, max_x_box, max_y_box

    def boxes_overlap(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
        pad: float = 0.0,
    ) -> bool:
        a_min_x, a_min_y, a_max_x, a_max_y = box_a
        b_min_x, b_min_y, b_max_x, b_max_y = box_b
        return not (
            a_max_x <= b_min_x - pad
            or b_max_x <= a_min_x - pad
            or a_max_y <= b_min_y - pad
            or b_max_y <= a_min_y - pad
        )

    def box_to_box_clearance(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        a_min_x, a_min_y, a_max_x, a_max_y = box_a
        b_min_x, b_min_y, b_max_x, b_max_y = box_b
        dx = max(b_min_x - a_max_x, a_min_x - b_max_x, 0.0)
        dy = max(b_min_y - a_max_y, a_min_y - b_max_y, 0.0)
        if dx == 0.0 and dy == 0.0:
            overlap_x = min(a_max_x, b_max_x) - max(a_min_x, b_min_x)
            overlap_y = min(a_max_y, b_max_y) - max(a_min_y, b_min_y)
            return -float(min(max(overlap_x, 0.0), max(overlap_y, 0.0)))
        return float(np.hypot(dx, dy))

    candidate_records: List[Tuple[Tuple[float, float, float, float, float, float, float, float, float], float, float, str, str]] = []
    vertical_major_axis = height >= width
    if vertical_major_axis:
        candidate_specs = [
            (centroid[0], min_y - pad, "center", "bottom", 0.0),
            (centroid[0], max_y + pad, "center", "top", 0.0),
            (centroid[0] + label_width * 0.18, min_y - pad, "center", "bottom", 0.0),
            (centroid[0] - label_width * 0.18, min_y - pad, "center", "bottom", 0.0),
            (centroid[0] + label_width * 0.18, max_y + pad, "center", "top", 0.0),
            (centroid[0] - label_width * 0.18, max_y + pad, "center", "top", 0.0),
            (max_x + pad, centroid[1], "left", "center", 0.0),
            (min_x - pad, centroid[1], "right", "center", 0.0),
            (max_x + pad, centroid[1] + side_shift, "left", "center", 0.0),
            (max_x + pad, centroid[1] - side_shift, "left", "center", 0.0),
            (min_x - pad, centroid[1] + side_shift, "right", "center", 0.0),
            (min_x - pad, centroid[1] - side_shift, "right", "center", 0.0),
        ]
    else:
        candidate_specs = [
            (min_x - pad, centroid[1], "right", "center", 0.0),
            (max_x + pad, centroid[1], "left", "center", 0.0),
            (min_x - pad, centroid[1] + label_height * 0.18, "right", "center", 0.0),
            (min_x - pad, centroid[1] - label_height * 0.18, "right", "center", 0.0),
            (max_x + pad, centroid[1] + label_height * 0.18, "left", "center", 0.0),
            (max_x + pad, centroid[1] - label_height * 0.18, "left", "center", 0.0),
            (centroid[0], min_y - pad, "center", "bottom", 0.0),
            (centroid[0], max_y + pad, "center", "top", 0.0),
            (centroid[0] + mid_shift, min_y - pad, "center", "bottom", 0.0),
            (centroid[0] - mid_shift, min_y - pad, "center", "bottom", 0.0),
            (centroid[0] + mid_shift, max_y + pad, "center", "top", 0.0),
            (centroid[0] - mid_shift, max_y + pad, "center", "top", 0.0),
        ]

    for rank, (x, y, ha, va, _) in enumerate(candidate_specs):
        x, y = clamp_point(float(x), float(y))
        label_box = estimate_label_box(x, y, ha, va)
        sample_points = [
            (label_box[0], label_box[1]),
            (label_box[0], label_box[3]),
            (label_box[2], label_box[1]),
            (label_box[2], label_box[3]),
            ((label_box[0] + label_box[2]) / 2.0, (label_box[1] + label_box[3]) / 2.0),
        ]
        label_overlap_count = sum(1 for box in occupied_label_boxes if boxes_overlap(label_box, box, pad=3.5))
        nearby_overlap_count = sum(1 for box in nearby_boxes if boxes_overlap(label_box, box, pad=5.0))
        inside_contour = 1 if contour_path is not None and any(contour_path.contains_point(pt) for pt in sample_points) else 0
        out_of_bounds = 1 if (
            label_box[0] < 0.0
            or label_box[1] < 0.0
            or label_box[2] > image_w
            or label_box[3] > image_h
        ) else 0
        min_clearance = min(
            [box_to_box_clearance(label_box, box) for box in occupied_label_boxes] +
            [box_to_box_clearance(label_box, box) for box in nearby_boxes],
            default=float("inf"),
        )
        border_clearance = min(
            x - margin,
            image_w - margin - x,
            y - margin,
            image_h - margin - y,
        )
        centroid_distance = abs(x - centroid[0]) + abs(y - centroid[1])
        score = (
            -float(label_overlap_count),
            -float(nearby_overlap_count),
            -float(inside_contour),
            -float(out_of_bounds),
            float(min_clearance),
            float(border_clearance),
            -float(rank),
            -float(centroid_distance),
            -float(label_width * label_height),
        )
        if label_overlap_count == 0 and nearby_overlap_count == 0 and inside_contour == 0 and out_of_bounds == 0:
            return x, y, ha, va
        candidate_records.append((score, x, y, ha, va))

    if candidate_records:
        candidate_records.sort(reverse=True)
        _, x, y, ha, va = candidate_records[0]
        return x, y, ha, va

    image_center = np.asarray([image_w / 2.0, image_h / 2.0], dtype=float)
    direction = np.asarray(centroid, dtype=float) - image_center
    norm = float(np.hypot(direction[0], direction[1]))
    if not np.isfinite(norm) or norm <= 1e-9:
        direction = np.asarray([1.0, -1.0], dtype=float)
        norm = float(np.hypot(direction[0], direction[1]))
    unit = direction / norm
    fallback_x, fallback_y = clamp_point(float(centroid[0] + unit[0] * pad), float(centroid[1] + unit[1] * pad))
    ha = "left" if unit[0] >= 0 else "right"
    va = "bottom" if unit[1] <= 0 else "top"
    return fallback_x, fallback_y, ha, va



def plot_roi_overlays(
    ax: Any,
    mean_img: np.ndarray,
    contours: List[Dict[str, Any]],
    title: str,
    show_labels: bool = True,
    title_pad: float = 8.0,
    title_fontsize: float = POSTER_TITLE_SIZE,
    image_anchor: str = "C",
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable")
    try:
        from matplotlib import patheffects as pe
    except Exception:  # pragma: no cover - path effects are optional
        pe = None
    finite = mean_img[np.isfinite(mean_img)]
    if finite.size:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
    else:
        vmin, vmax = 0.0, 1.0
    ax.set_anchor(image_anchor)
    ax.imshow(mean_img, cmap="gray", origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
    colors = assign_contour_colors(len(contours))
    contour_boxes: List[Tuple[float, float, float, float]] = []
    for overlay in contours:
        coords = np.asarray(overlay["coords"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.size == 0:
            contour_boxes.append((0.0, 0.0, 0.0, 0.0))
            continue
        contour_boxes.append((
            float(np.nanmin(coords[:, 0])),
            float(np.nanmin(coords[:, 1])),
            float(np.nanmax(coords[:, 0])),
            float(np.nanmax(coords[:, 1])),
        ))
    occupied_label_boxes: List[Tuple[float, float, float, float]] = []

    def estimate_label_box(x: float, y: float, ha: str, va: str, label_text: str) -> Tuple[float, float, float, float]:
        display_label = str(label_text or "").replace("_", " ").strip()
        label_chars = max(len(display_label), 1)
        label_width = max(26.0, 0.85 * float(POSTER_NOTE_SIZE) * label_chars + 10.0)
        label_height = max(15.0, 1.45 * float(POSTER_NOTE_SIZE))
        if ha == "left":
            min_x_box, max_x_box = x, x + label_width
        elif ha == "right":
            min_x_box, max_x_box = x - label_width, x
        else:
            min_x_box, max_x_box = x - label_width / 2.0, x + label_width / 2.0
        if va == "bottom":
            min_y_box, max_y_box = y - label_height, y
        elif va == "top":
            min_y_box, max_y_box = y, y + label_height
        else:
            min_y_box, max_y_box = y - label_height / 2.0, y + label_height / 2.0
        return min_x_box, min_y_box, max_x_box, max_y_box

    for idx, overlay in enumerate(contours):
        coords = np.asarray(overlay["coords"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.size == 0:
            continue
        closed = np.vstack([coords, coords[:1]])
        color = overlay.get("color") or colors[idx]
        linewidth = max(float(overlay.get("linewidth", 1.8)), 2.0)
        outline = Polygon(closed, closed=True, fill=False, edgecolor="black", linewidth=linewidth + 1.5, alpha=0.65)
        ax.add_patch(outline)
        patch = Polygon(closed, closed=True, fill=False, edgecolor=color, linewidth=linewidth)
        if pe is not None:
            patch.set_path_effects([pe.withStroke(linewidth=linewidth + 1.0, foreground="black")])
        ax.add_patch(patch)
        label = str(overlay.get("label") or "")
        if show_labels and label:
            nearby_boxes = [box for box_idx, box in enumerate(contour_boxes) if box_idx != idx]
            if overlay.get("kind") == "dendrite":
                label_x, label_y, label_ha, label_va = choose_dendrite_label_anchor(
                    coords,
                    mean_img.shape,
                    occupied_label_boxes,
                    nearby_boxes,
                    label_text=label,
                )
                label_fontsize = POSTER_NOTE_SIZE + 2
            else:
                label_x, label_y, label_ha, label_va = choose_roi_label_anchor(
                    coords,
                    mean_img.shape,
                    occupied_label_boxes,
                    nearby_boxes,
                    label_text=label,
                )
                label_fontsize = POSTER_NOTE_SIZE
            label_box = estimate_label_box(label_x, label_y, label_ha, label_va, label)
            occupied_label_boxes.append(label_box)
            text_kwargs = {
                "x": float(label_x),
                "y": float(label_y),
                "s": label,
                "color": color,
                "fontsize": label_fontsize,
                "ha": label_ha,
                "va": label_va,
                "fontweight": "bold",
                "clip_on": True,
            }
            if pe is not None:
                text_kwargs["path_effects"] = [pe.withStroke(linewidth=3.0 if overlay.get("kind") == "dendrite" else 2.5, foreground="black")]
            ax.text(**text_kwargs)
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


def add_broad_state_key(
    ax: Any,
    state_order: Sequence[str] = BROAD_STATE_ORDER,
    show_title: bool = False,
) -> None:
    if plt is None:
        return
    try:
        from matplotlib import patheffects as pe
    except Exception:  # pragma: no cover - path effects are optional
        pe = None
    ax.set_axis_off()
    label_fontsize = max(7.0, POSTER_FONT_SIZE - 8)
    entries = [state for state in state_order if state in BROAD_STATE_COLORS]
    if not entries:
        return
    # Keep the key compact and readable as a single column next to the state timeline.
    n_cols = 1
    n_rows = int(np.ceil(len(entries) / float(n_cols)))
    top = 0.99 if show_title else 0.995
    bottom = 0.02
    row_step = (top - bottom) / max(n_rows - 1, 1)
    col_x = [0.03]
    label_x = [0.18]
    swatch_width = 0.10
    swatch_height = 0.038
    for idx, state in enumerate(entries):
        col = idx // n_rows
        row = idx % n_rows
        if col >= len(col_x):
            break
        y = top - row * row_step
        x0 = col_x[col]
        color = BROAD_STATE_COLORS[state]
        ax.add_patch(
            Rectangle(
                (x0, y - swatch_height / 2.0),
                swatch_width,
                swatch_height,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        display_label = state.replace("_", " ") if "_" in state else state
        text_kwargs = {
            "x": label_x[col],
            "y": y,
            "s": display_label,
            "transform": ax.transAxes,
            "ha": "left",
            "va": "center",
            "fontsize": label_fontsize,
            "color": color,
            "fontweight": "bold",
            "clip_on": False,
        }
        if pe is not None:
            text_kwargs["path_effects"] = [pe.withStroke(linewidth=1.2, foreground="white")]
        ax.text(**text_kwargs)


def read_pickle_file(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def ordered_unique_labels(rows: Sequence[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    seen = set()
    for row in rows:
        label = str(row.get("state_label") or row.get("category") or "").strip()
        if not label:
            label = "unknown"
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def summarize_state_responses(
    rows: Sequence[Dict[str, Any]],
    responses: np.ndarray,
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    labels = ordered_unique_labels(rows)
    grouped: Dict[str, List[float]] = {label: [] for label in labels}
    for row, response in zip(rows, np.asarray(responses, dtype=float)):
        if not np.isfinite(response):
            continue
        label = str(row.get("state_label") or row.get("category") or "").strip() or "unknown"
        grouped.setdefault(label, []).append(float(response))
    means = np.asarray(
        [float(np.nanmean(np.asarray(grouped.get(label, []), dtype=float))) if grouped.get(label) else float("nan") for label in labels],
        dtype=float,
    )
    box_values = {label: np.asarray(grouped.get(label, []), dtype=float) for label in labels}
    return labels, means, box_values


def plot_state_row(
    fig: Any,
    axes: Sequence[Any],
    roi_label: str,
    rows: Sequence[Dict[str, Any]],
    responses: np.ndarray,
    color: Any,
    cut_state_means: Optional[Dict[str, Any]] = None,
) -> None:
    ax_state, ax_box, ax_trace = axes
    if cut_state_means:
        labels = list(cut_state_means.keys())
        values = np.asarray([as_float(cut_state_means.get(label)) for label in labels], dtype=float)
        labels = [label for label, value in zip(labels, values) if np.isfinite(value)]
        values = values[np.isfinite(values)]
    else:
        labels, values, grouped = summarize_state_responses(rows, responses)
        values = np.asarray(values, dtype=float)
        grouped = {label: np.asarray(grouped.get(label, []), dtype=float) for label in labels}
    if not labels:
        ax_state.text(0.5, 0.5, "No state labels", ha="center", va="center", transform=ax_state.transAxes, fontsize=POSTER_NOTE_SIZE)
        ax_state.set_axis_off()
        ax_box.text(0.5, 0.5, "No state labels", ha="center", va="center", transform=ax_box.transAxes, fontsize=POSTER_NOTE_SIZE)
        ax_box.set_axis_off()
        ax_trace.text(0.5, 0.5, "No state labels", ha="center", va="center", transform=ax_trace.transAxes, fontsize=POSTER_NOTE_SIZE)
        ax_trace.set_axis_off()
        return

    if cut_state_means:
        ax_state.bar(np.arange(len(labels)), values, color=color, alpha=0.8)
        ax_state.set_xticks(np.arange(len(labels)))
        ax_state.set_xticklabels(labels, rotation=35, ha="right", fontsize=POSTER_FONT_SIZE)
    else:
        ax_state.bar(np.arange(len(labels)), values, color=color, alpha=0.8)
        ax_state.set_xticks(np.arange(len(labels)))
        ax_state.set_xticklabels(labels, rotation=35, ha="right", fontsize=POSTER_FONT_SIZE)
    ax_state.set_ylabel("mean resp.", fontsize=POSTER_LABEL_SIZE)
    ax_state.set_title("state means", fontsize=POSTER_TITLE_SIZE)
    ax_state.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(ax_state, axis="y", nbins=4)

    box_groups: List[np.ndarray] = []
    box_labels: List[str] = []
    if cut_state_means:
        for idx, label in enumerate(labels):
            value = float(values[idx]) if idx < len(values) and np.isfinite(values[idx]) else float("nan")
            if np.isfinite(value):
                box_groups.append(np.asarray([value], dtype=float))
                box_labels.append(label)
    else:
        for label in labels:
            label_rows = [row for row in rows if str(row.get("state_label") or row.get("category") or "").strip() == label]
            group_vals = []
            for row, response in zip(label_rows, np.asarray(responses, dtype=float)):
                if np.isfinite(response):
                    group_vals.append(float(response))
            group_vals = np.asarray(group_vals, dtype=float)
            group_vals = group_vals[np.isfinite(group_vals)]
            if group_vals.size:
                box_groups.append(group_vals)
                box_labels.append(label)
    if box_groups:
        bp = ax_box.boxplot(box_groups, vert=True, labels=box_labels, patch_artist=True)
        for patch in bp.get("boxes", []):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax_box.set_ylabel("response", fontsize=POSTER_LABEL_SIZE)
        ax_box.set_title("state distribution", fontsize=POSTER_TITLE_SIZE)
        ax_box.tick_params(axis="x", labelsize=POSTER_FONT_SIZE, rotation=35)
        ax_box.tick_params(axis="y", labelsize=POSTER_FONT_SIZE)
        set_sparse_numeric_ticks(ax_box, axis="y", nbins=4)
    else:
        ax_box.text(0.5, 0.5, "No response distributions", ha="center", va="center", transform=ax_box.transAxes, fontsize=POSTER_NOTE_SIZE)
        ax_box.set_axis_off()

    if cut_state_means:
        trace_x = np.arange(len(labels), dtype=float)
        trace_y = values
        ax_trace.plot(trace_x, trace_y, color=color, linewidth=1.2, marker="o", markersize=4.0)
        ax_trace.set_xticks(trace_x)
        ax_trace.set_xticklabels(labels, rotation=35, ha="right", fontsize=POSTER_FONT_SIZE)
        ax_trace.set_title("state response", fontsize=POSTER_TITLE_SIZE)
        ax_trace.set_xlabel("state", fontsize=POSTER_LABEL_SIZE)
    else:
        sample_indices = np.arange(len(responses), dtype=float)
        ax_trace.plot(sample_indices, responses, color=color, linewidth=1.2, marker="o", markersize=4.0)
        ax_trace.set_title("response trace", fontsize=POSTER_TITLE_SIZE)
        ax_trace.set_xlabel("sample #", fontsize=POSTER_LABEL_SIZE)
    ax_trace.set_ylabel("mean resp.", fontsize=POSTER_LABEL_SIZE)
    ax_trace.tick_params(axis="both", labelsize=POSTER_FONT_SIZE)
    set_sparse_numeric_ticks(ax_trace, axis="y", nbins=4)
    if not cut_state_means:
        set_sparse_numeric_ticks(ax_trace, axis="x", nbins=4)
    if rows:
        labels_for_rows = [str(row.get("state_label") or row.get("category") or "").strip() or "unknown" for row in rows]
        for idx, label in enumerate(labels_for_rows):
            ax_trace.text(
                idx,
                1.02,
                label,
                transform=ax_trace.get_xaxis_transform(),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=POSTER_NOTE_SIZE - 2,
                color="#666666",
            )

    finite = np.asarray(responses, dtype=float)
    finite = finite[np.isfinite(finite)]
    response_label = f"{roi_label}"
    if finite.size:
        response_label += f"  mean={np.nanmean(finite):.3f}  max={np.nanmax(finite):.3f}"
    ax_state.text(
        0.02,
        1.04,
        response_label,
        transform=ax_state.transAxes,
        ha="left",
        va="bottom",
        fontsize=POSTER_NOTE_SIZE,
        fontweight="bold",
        color=color,
    )


def detail_label_from_local_ids(local_ids: Dict[str, Any], fallback: str, is_child: bool) -> str:
    if not is_child:
        dendrite_id = local_ids.get("dendrite_id")
        cell_id = local_ids.get("cell_id")
        if dendrite_id is not None:
            return f"D{int(dendrite_id)}"
        if cell_id is not None:
            return f"Cell{int(cell_id)}"
        return fallback
    spine_id = local_ids.get("spine_id")
    if spine_id is not None:
        return f"S{int(spine_id)}"
    axon_id = local_ids.get("axon_id")
    if axon_id is not None:
        return f"A{int(axon_id)}"
    bouton_id = local_ids.get("bouton_id")
    if bouton_id is not None:
        return f"B{int(bouton_id)}"
    child_index = local_ids.get("child_index")
    if child_index is not None:
        return f"C{int(child_index)}"
    return fallback


def summarize_roi_detail_rows(
    cache: Dict[str, Any],
    animal_id: str,
    global_dendrite_id: str,
    dendrite_record: Dict[str, Any],
    preferred_exp_ids: Optional[Sequence[str]] = None,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]]:
    observations = dendrite_record.get("observations", {})
    candidate_exp_ids = list(preferred_exp_ids or [])
    if not candidate_exp_ids:
        candidate_exp_ids = sorted(observations.keys())
    representative_exp_id = None
    fallback_exp_id = None
    for exp_id in candidate_exp_ids:
        if exp_id not in observations:
            continue
        fallback_exp_id = fallback_exp_id or exp_id
        exp_meta = cache.get("experiments", {}).get(exp_id)
        if exp_meta is None:
            continue
        representative_exp_id = exp_id
        break
    if representative_exp_id is None:
        representative_exp_id = fallback_exp_id
    if representative_exp_id is None and observations:
        representative_exp_id = sorted(observations.keys())[0]
    if representative_exp_id is None:
        return None
    exp_meta = cache.get("experiments", {}).get(representative_exp_id)
    if exp_meta is None:
        return None
    representative_obs = dendrite_record.get("observations", {}).get(representative_exp_id)
    if representative_obs is None:
        return None
    compartment = str(
        representative_obs.get("compartment")
        or dendrite_record.get("compartment")
        or exp_meta.get("compartment")
        or ""
    ).strip()
    rows = list(exp_meta.get("trial_rows") or [])
    return representative_exp_id, exp_meta, representative_obs, compartment, rows


def plot_roi_detail_figure(
    cache: Dict[str, Any],
    animal_id: str,
    global_dendrite_id: str,
    dendrite_record: Dict[str, Any],
    output_path: Path,
    repo_base: Optional[Path],
    dpi: int,
    preferred_exp_ids: Optional[Sequence[str]] = None,
    date: Optional[str] = None,
) -> Optional[str]:
    if plt is None:
        return None
    detail = summarize_roi_detail_rows(cache, animal_id, global_dendrite_id, dendrite_record, preferred_exp_ids=preferred_exp_ids)
    if detail is None:
        return None
    representative_exp_id, exp_meta, representative_obs, dendrite_compartment, rows = detail

    conversion_path = resolve_conversion_path(exp_meta, animal_id, representative_exp_id, repo_base)
    exp_root = conversion_path.parents[2]
    local_ids = representative_obs.get("local_ids", {})
    plane = int(local_ids.get("plane") or 0)
    ops_path = resolve_ops_path(exp_root, plane)
    mean_img, _ops = load_mean_image(ops_path)
    stat_path = load_stat_path(ops_path)
    raw_conversion = load_conversion_library(conversion_path)

    parent_coords = load_contour_data(
        raw_conversion,
        local_ids.get("general_roi_id"),
        local_ids.get("plane_roi_id"),
        stat_path,
    )
    group_items: List[Dict[str, Any]] = []
    parent_label = detail_label_from_local_ids(local_ids, "parent", is_child=False)
    group_items.append(
        {
            "label": parent_label,
            "coords": parent_coords,
            "color": "#e41a1c",
            "linewidth": 2.4,
            "observation": representative_obs,
            "is_child": False,
            "local_ids": local_ids,
        }
    )

    child_records = sorted(dendrite_record.get("spines", {}).items())
    child_palette = assign_contour_colors(max(len(child_records), 1))
    for child_idx, (child_global_id, child_record) in enumerate(child_records):
        child_obs = child_record.get("observations", {}).get(representative_exp_id)
        if child_obs is None:
            continue
        child_local_ids = child_obs.get("local_ids", {})
        child_coords = load_contour_data(
            raw_conversion,
            child_local_ids.get("general_roi_id"),
            child_local_ids.get("plane_roi_id"),
            stat_path,
        )
        if child_coords is None:
            continue
        child_label = detail_label_from_local_ids(child_local_ids, str(child_global_id), is_child=True)
        group_items.append(
            {
                "label": child_label,
                "coords": child_coords,
                "color": child_palette[child_idx % len(child_palette)],
                "linewidth": 1.8,
                "observation": child_obs,
                "is_child": True,
                "local_ids": child_local_ids,
            }
        )

    if not group_items:
        return None

    for item in group_items:
        observation = item["observation"]
        state_means = observation.get("cut_state_means")
        if isinstance(state_means, dict) and state_means:
            item["cut_state_means"] = state_means

    fig_height = min(max(7.4, 1.92 * len(group_items) + 4.8), 11.4)
    detail_cols = 3
    fig_width = min(max(POSTER_WIDE_FIGSIZE[0] + 1.2, 12.2), 13.6)
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.22, 2.58], wspace=0.18)
    ax_image = fig.add_subplot(outer[0, 0])
    ax_image.set_gid("meanImg")
    right = outer[0, 1].subgridspec(len(group_items), 1, hspace=1.18)

    plot_roi_overlays(
        ax_image,
        mean_img,
        [dict(item, coords=item["coords"]) for item in group_items],
        "meanImg",
        title_pad=2.0,
        title_fontsize=max(20, POSTER_TITLE_SIZE - 4),
    )
    legend_handles = [
        Patch(facecolor=item["color"], edgecolor="none", label=str(item["label"]))
        for item in group_items
        if str(item.get("label") or "").strip()
    ]
    ax_image.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="none",
        fontsize=POSTER_LEGEND_SIZE,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )

    for row_index, item in enumerate(group_items):
        row_grid = right[row_index, 0].subgridspec(1, detail_cols, wspace=0.52)
        axes = [fig.add_subplot(row_grid[0, col_index]) for col_index in range(detail_cols)]
        rows_for_item = list(rows)
        state_means = item.get("cut_state_means")
        if isinstance(state_means, dict) and state_means:
            responses = np.asarray([as_float(state_means.get(label)) for label in state_means.keys()], dtype=float)
        else:
            responses = np.full(len(rows_for_item), np.nan, dtype=float)
        while len(axes) < 3:
            axes.append(fig.add_subplot(row_grid[0, len(axes)]))
        plot_state_row(
            fig,
            axes[:3],
            item["label"],
            rows_for_item,
            responses,
            item["color"],
            cut_state_means=item.get("cut_state_means"),
        )
        axes[0].text(
            0.98,
            0.96,
            item["label"],
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=POSTER_NOTE_SIZE,
            color=item["color"],
            fontweight="bold",
        )

    detail_title = format_dendrite_display_name(animal_id, dendrite_compartment, local_ids.get("dendrite_id"))
    fig.suptitle(detail_title, fontsize=max(20, POSTER_TITLE_SIZE - 2), y=0.995)
    fig.text(
        0.5,
        0.952,
        str(date or representative_exp_id),
        ha="center",
        va="top",
        fontsize=POSTER_NOTE_SIZE + 4,
        color="#444444",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    ensure_dir(output_path.parent)
    save_figure(fig, output_path, dpi=dpi, extra_formats=())
    composed_svg_path = output_path.with_suffix(".svg")
    compose_svg_figure(
        composed_svg_path,
        [meanimg_panel_path.with_suffix(".svg"), detail_panel_path.with_suffix(".svg")],
        layout="horizontal",
        title=day_title,
    )
    rasterize_svg_to_png(composed_svg_path, output_path, dpi=dpi)
    return str(output_path)


def generate_roi_detail_figures(
    output_dir: Path,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    figure_root: Optional[Path] = None,
    repo_base: Optional[Path] = None,
    dpi: int = DEFAULT_DPI,
) -> List[str]:
    if plt is None:
        return []
    config = results.get("config", {}) or {}
    fig_dir = ensure_dir(Path(figure_root) if figure_root is not None else (output_dir / "figures"))
    day_groups = grouped_day_expids(config, cache)
    saved: List[str] = []
    targets: List[Tuple[str, str, str, Dict[str, Any]]] = []
    if day_groups:
        for (animal_id, date), exp_ids in day_groups.items():
            for global_dendrite_id, dendrite_record in selected_dendrite_records(cache, animal_id, date, exp_ids):
                targets.append((animal_id, date, global_dendrite_id, dendrite_record, list(exp_ids)))
    else:
        for animal_id, animal_entry in cache.get("animals", {}).items():
            for global_dendrite_id, dendrite_record in sorted(animal_entry.get("dendrites", {}).items()):
                date = str(dendrite_record.get("date") or "")
                if not date:
                    continue
                targets.append((str(animal_id), date, str(global_dendrite_id), dendrite_record, []))
    with step_scope("day ROI detail figures", total=len(targets)):
        for idx, (animal_id, date, global_dendrite_id, dendrite_record, preferred_exp_ids) in enumerate(targets, start=1):
            step_progress(idx, len(targets), label=f"{animal_id} | {date} | {global_dendrite_id}")
            summary = summarize_roi_detail_rows(cache, animal_id, global_dendrite_id, dendrite_record, preferred_exp_ids=preferred_exp_ids)
            if summary is None:
                continue
            _exp_id, _exp_meta, _representative_obs, compartment, _rows = summary
            figure_path = build_roi_detail_save_path(fig_dir, animal_id, date, compartment, global_dendrite_id)
            try:
                saved_path = plot_roi_detail_figure(
                    cache,
                    animal_id,
                    global_dendrite_id,
                    dendrite_record,
                    figure_path,
                    repo_base,
                    dpi,
                    preferred_exp_ids=preferred_exp_ids,
                    date=date,
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to create ROI detail figure for {animal_id} {date} {global_dendrite_id}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if saved_path:
                saved.append(saved_path)
    return saved


def add_session_boundaries(ax: Any, boundary_hours: np.ndarray) -> None:
    for boundary in boundary_hours:
        ax.axvline(float(boundary), color="white", linewidth=2.0, alpha=0.95, zorder=5)
        ax.axvline(float(boundary), color="black", linewidth=0.8, alpha=0.75, zorder=6)


def format_hours_axis(ax: Any, total_hours: float) -> None:
    set_hour_ticks(ax, total_hours, labelsize=POSTER_FONT_SIZE)


def apply_shared_time_axis(ax: Any, total_hours: float, show_labels: bool = True) -> None:
    if not np.isfinite(total_hours) or total_hours <= 0:
        total_hours = 1.0
    ax.set_xlim(0.0, total_hours)
    ax.margins(x=0.0)
    format_hours_axis(ax, total_hours)
    ax.tick_params(axis="x", labelbottom=show_labels)


def render_meanimg_panel_figure(
    mean_img: np.ndarray,
    contours: List[Dict[str, Any]],
    title: str,
    title_pad: float = 2.0,
    title_fontsize: float = max(20, POSTER_TITLE_SIZE - 4),
) -> Any:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate figures")
    fig = plt.figure(figsize=(5.4, 5.8))
    ax = fig.add_subplot(1, 1, 1)
    plot_roi_overlays(
        ax,
        mean_img,
        contours,
        title,
        show_labels=True,
        title_pad=title_pad,
        title_fontsize=title_fontsize,
    )
    fig.tight_layout(pad=0.12)
    return fig


def render_day_detail_stack_figure(
    day_summary: DaySummary,
    dend_trace: np.ndarray,
    spine_heatmap: np.ndarray,
    spine_labels: Sequence[str],
    dendrite_color: str,
    spine_colors: Sequence[Any],
) -> Any:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate figures")

    total_hours = day_summary.total_bins / 3600.0 if day_summary.total_bins > 0 else 1.0
    if not np.isfinite(total_hours) or total_hours <= 0:
        total_hours = 1.0

    heatmap_height = max(2.2, 0.45 * max(1, spine_heatmap.shape[0]))
    fig_width = min(max(POSTER_WIDE_FIGSIZE[0] + 2.2, 12.0), 15.6)
    fig_height = min(max(POSTER_WIDE_FIGSIZE[1] + 0.9, 7.2 + 0.24 * max(0, spine_heatmap.shape[0] - 1)), 9.8)
    fig = plt.figure(figsize=(fig_width, fig_height))
    right = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.62, 0.90],
        height_ratios=[0.72, 1.95, heatmap_height],
        hspace=0.84,
        wspace=0.05,
    )
    ax_timeline = fig.add_subplot(right[0, 0])
    ax_state_key = fig.add_subplot(right[0, 1])
    ax_trace = fig.add_subplot(right[1, 0], sharex=ax_timeline)
    ax_heatmap = fig.add_subplot(right[2, 0], sharex=ax_timeline)

    state_matrix = day_summary.state_codes[np.newaxis, :]
    cmap = ListedColormap([BROAD_STATE_COLORS[label] for label in BROAD_STATE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(BROAD_STATE_ORDER) + 0.5, 1.0), cmap.N)

    ax_timeline.imshow(
        state_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[0.0, total_hours, 0.0, 1.0],
        origin="lower",
    )
    add_session_boundaries(ax_timeline, day_summary.boundary_hours)
    ax_timeline.set_yticks([])
    apply_shared_time_axis(ax_timeline, total_hours, show_labels=False)

    for session_index, session in enumerate(day_summary.sessions, start=1):
        mid_hour = (session.offset_bins + session.n_bins / 2.0) / 3600.0
        ax_timeline.text(
            mid_hour,
            1.08,
            f"Session {session_index}",
            transform=ax_timeline.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=max(11, POSTER_NOTE_SIZE - 6),
            fontweight="bold",
            rotation=0,
        )

    ax_trace.plot(day_summary.x_hours, dend_trace, color="#222222", linewidth=0.9)
    add_session_boundaries(ax_trace, day_summary.boundary_hours)
    apply_shared_time_axis(ax_trace, total_hours, show_labels=False)
    ax_trace.set_ylabel("dF/F", fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=-1)
    ax_trace.yaxis.label.set_color(dendrite_color)
    ax_trace.yaxis.set_tick_params(labelsize=max(10, POSTER_FONT_SIZE - 3), pad=0.25)
    set_sparse_numeric_ticks(ax_trace, axis="y", nbins=4)
    ax_trace.set_title("Dendrite activity", fontsize=max(18, POSTER_TITLE_SIZE - 4), pad=4)
    ax_trace.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.6)

    if spine_heatmap.shape[0] > 0:
        clipped_heatmap = np.asarray(spine_heatmap, dtype=float)
        clipped_heatmap = np.where(np.isfinite(clipped_heatmap), np.clip(clipped_heatmap, 0.0, None), np.nan)
        finite = clipped_heatmap[np.isfinite(clipped_heatmap)]
        if finite.size:
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        else:
            vmax = 1.0
        masked_heatmap = np.ma.masked_invalid(clipped_heatmap)
        im = ax_heatmap.imshow(
            masked_heatmap,
            aspect="auto",
            interpolation="nearest",
            cmap="binary",
            vmin=0.0,
            vmax=vmax,
            extent=[0.0, total_hours, float(spine_heatmap.shape[0]), 0.0],
            origin="upper",
        )
        add_session_boundaries(ax_heatmap, day_summary.boundary_hours)
        ax_heatmap.set_yticks(np.arange(spine_heatmap.shape[0]) + 0.5)
        ax_heatmap.set_yticklabels(spine_labels, fontsize=POSTER_FONT_SIZE)
        for tick_label, color in zip(ax_heatmap.get_yticklabels(), spine_colors):
            tick_label.set_color(color)
        apply_shared_time_axis(ax_heatmap, total_hours, show_labels=True)
        ax_heatmap.set_ylabel("spines", fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=-1)
        ax_heatmap.set_title("Spine activity heatmap", fontsize=max(18, POSTER_TITLE_SIZE - 4), pad=5)
    else:
        ax_heatmap.text(
            0.5,
            0.5,
            "No spines with day observations",
            transform=ax_heatmap.transAxes,
            ha="center",
            va="center",
            fontsize=POSTER_TITLE_SIZE,
        )
        ax_heatmap.set_axis_off()

    ax_heatmap.set_xlabel("stitched time (h)", fontsize=POSTER_LABEL_SIZE)

    ax_state_key.axis("off")
    add_broad_state_key(ax_state_key, show_title=False)

    if spine_heatmap.shape[0] > 0:
        heatmap_box = ax_heatmap.get_position()
        cbar_width = 0.012
        cbar_pad = 0.006
        cbar_left = min(heatmap_box.x1 + cbar_pad, 0.985 - cbar_width)
        cbar_ax = fig.add_axes([cbar_left, heatmap_box.y0, cbar_width, heatmap_box.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)

    fig.tight_layout(rect=[0.0, 0.0, 0.985, 0.88])
    return fig


def stitch_binned_series(
    cache: Dict[str, Any],
    day_summary: DaySummary,
    dendrite_record: Dict[str, Any],
    field_name: str,
) -> np.ndarray:
    stitched = np.full(day_summary.total_bins, np.nan, dtype=float)
    for session in day_summary.sessions:
        exp_id = session.exp_id
        exp_meta = cache["experiments"].get(exp_id)
        if exp_meta is None:
            continue
        observation = dendrite_record.get("observations", {}).get(exp_id)
        if observation is None:
            continue
        values = observation.get(field_name)
        if values is None:
            continue
        t = np.asarray(exp_meta.get("time"), dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        t, values = align_length(t, values)
        if t.size == 0 or values.size == 0:
            continue
        rel_t = t - float(t[0])
        n_bins = session.n_bins
        bin_index = np.clip(np.floor(rel_t).astype(int), 0, n_bins - 1)
        binned = bin_mean(values, bin_index, n_bins)
        start = session.offset_bins
        end = start + n_bins
        stitched[start:end] = binned
    return stitched


def stitch_spine_heatmap(
    cache: Dict[str, Any],
    day_summary: DaySummary,
    dendrite_record: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    spine_items: List[Tuple[int, str, Dict[str, Any]]] = []
    for spine_global_id, spine_record in dendrite_record.get("spines", {}).items():
        best_exp = None
        best_local_ids = None
        for session in day_summary.sessions:
            exp_id = session.exp_id
            spine_obs = spine_record.get("observations", {}).get(exp_id)
            if spine_obs is None:
                continue
            best_exp = exp_id
            best_local_ids = spine_obs.get("local_ids", {})
            break
        if best_exp is None:
            continue
        spine_id = int(best_local_ids.get("spine_id") or 0)
        spine_items.append((spine_id, spine_global_id, spine_record))
    spine_items.sort(key=lambda item: (item[0], item[1]))

    if not spine_items:
        return np.empty((0, day_summary.total_bins), dtype=float), []

    rows: List[np.ndarray] = []
    labels: List[str] = []
    for spine_id, spine_global_id, spine_record in spine_items:
        stitched = np.full(day_summary.total_bins, np.nan, dtype=float)
        for session in day_summary.sessions:
            exp_id = session.exp_id
            exp_meta = cache["experiments"].get(exp_id)
            if exp_meta is None:
                continue
            spine_obs = spine_record.get("observations", {}).get(exp_id)
            if spine_obs is None:
                continue
            values = spine_obs.get("spine_specific")
            if values is None:
                continue
            t = np.asarray(exp_meta.get("time"), dtype=float).ravel()
            values = np.asarray(values, dtype=float).ravel()
            t, values = align_length(t, values)
            if t.size == 0 or values.size == 0:
                continue
            rel_t = t - float(t[0])
            n_bins = session.n_bins
            bin_index = np.clip(np.floor(rel_t).astype(int), 0, n_bins - 1)
            binned = bin_mean(values, bin_index, n_bins)
            start = session.offset_bins
            end = start + n_bins
            stitched[start:end] = binned
        rows.append(stitched)
        labels.append(f"S{spine_id}")
    if not rows:
        return np.empty((0, day_summary.total_bins), dtype=float), []
    return np.vstack(rows), labels


def plot_day_figure(
    cache: Dict[str, Any],
    day_summary: DaySummary,
    animal_id: str,
    global_dendrite_id: str,
    dendrite_record: Dict[str, Any],
    output_path: Path,
    repo_base: Optional[Path],
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate figures")

    representative_exp_id = choose_representative_exp_id(dendrite_record, day_summary.exp_ids)
    if representative_exp_id is None:
        raise ValueError(f"No representative expID found for {global_dendrite_id}")

    exp_meta = cache["experiments"][representative_exp_id]
    representative_obs = dendrite_record["observations"][representative_exp_id]
    dendrite_compartment = str(
        representative_obs.get("compartment")
        or dendrite_record.get("compartment")
        or exp_meta.get("compartment")
        or ""
    ).strip()
    conversion_path = resolve_conversion_path(exp_meta, animal_id, representative_exp_id, repo_base)
    exp_root = conversion_path.parents[2]
    local_ids = dendrite_record["observations"][representative_exp_id]["local_ids"]
    plane = int(local_ids.get("plane") or 0)
    ops_path = resolve_ops_path(exp_root, plane)
    mean_img, _ops = load_mean_image(ops_path)
    stat_path = load_stat_path(ops_path)
    raw_conversion = load_conversion_library(conversion_path)
    conversion_mode = determine_conversion_mode(conversion_path)

    dend_local = local_ids
    dend_general_roi_id = dend_local.get("general_roi_id")
    dend_plane_roi_id = dend_local.get("plane_roi_id")
    dend_coords = load_contour_data(raw_conversion, dend_general_roi_id, dend_plane_roi_id, stat_path)

    spine_overlays: List[Dict[str, Any]] = []
    for spine_global_id, spine_record in sorted(dendrite_record.get("spines", {}).items()):
        spine_obs = spine_record.get("observations", {}).get(representative_exp_id)
        if spine_obs is None:
            continue
        spine_local = spine_obs.get("local_ids", {})
        spine_coords = load_contour_data(
            raw_conversion,
            spine_local.get("general_roi_id"),
            spine_local.get("plane_roi_id"),
            stat_path,
        )
        if spine_coords is None:
            continue
        spine_id = spine_local.get("spine_id")
        label = f"S{spine_id}" if spine_id is not None else str(spine_global_id)
        spine_overlays.append(
            {
                "kind": "spine",
                "label": label,
                "coords": spine_coords,
                "color": None,
            }
        )

    contours: List[Dict[str, Any]] = []
    if dend_coords is not None:
        dend_label = f"D{int(dend_local.get('dendrite_id') or 0)}"
        contours.append({"kind": "dendrite", "label": dend_label, "coords": dend_coords, "color": "#e41a1c"})
    contours.extend(spine_overlays)
    contour_palette = assign_contour_colors(len(contours))
    contour_colors: List[Any] = []
    for idx, overlay in enumerate(contours):
        color = overlay.get("color") or contour_palette[idx]
        overlay["color"] = color
        contour_colors.append(color)
    dendrite_color = contour_colors[0] if contour_colors else "#e41a1c"
    spine_colors = contour_colors[1:] if len(contour_colors) > 1 else []

    meanimg_panel_title = "meanImg"
    meanimg_panel_path = output_path.with_name(f"{output_path.stem}_meanImg_panel.svg")
    standalone_meanimg_fig = render_meanimg_panel_figure(mean_img, contours, meanimg_panel_title)
    save_figure(standalone_meanimg_fig, meanimg_panel_path, dpi=dpi)

    state_matrix = day_summary.state_codes[np.newaxis, :]
    cmap = ListedColormap([BROAD_STATE_COLORS[label] for label in BROAD_STATE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(BROAD_STATE_ORDER) + 0.5, 1.0), cmap.N)

    dend_trace = stitch_binned_series(cache, day_summary, dendrite_record, "trace")
    spine_heatmap, spine_labels = stitch_spine_heatmap(cache, day_summary, dendrite_record)

    total_hours = day_summary.total_bins / 3600.0 if day_summary.total_bins > 0 else 1.0
    if not np.isfinite(total_hours) or total_hours <= 0:
        total_hours = 1.0

    heatmap_height = max(2.2, 0.45 * max(1, spine_heatmap.shape[0]))
    # Give the meanImg more horizontal room without shrinking the right-hand stack.
    fig_width = min(max(POSTER_WIDE_FIGSIZE[0] + 5.4, 17.8), 18.4)
    fig_height = min(max(POSTER_WIDE_FIGSIZE[1] + 0.8, 7.2 + 0.24 * max(0, spine_heatmap.shape[0] - 1)), 9.4)
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.75, 3.28],
        wspace=0.16,
    )
    ax_image = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(
        3,
        2,
        width_ratios=[2.62, 0.90],
        height_ratios=[0.72, 1.95, heatmap_height],
        hspace=0.84,
        wspace=0.05,
    )
    ax_timeline = fig.add_subplot(right[0, 0])
    ax_state_key = fig.add_subplot(right[0, 1])
    ax_trace = fig.add_subplot(right[1, 0], sharex=ax_timeline)
    ax_heatmap = fig.add_subplot(right[2, 0], sharex=ax_timeline)

    ax_timeline.imshow(
        state_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[0.0, total_hours, 0.0, 1.0],
        origin="lower",
    )
    add_session_boundaries(ax_timeline, day_summary.boundary_hours)
    ax_timeline.set_yticks([])
    apply_shared_time_axis(ax_timeline, total_hours, show_labels=False)

    for session_index, session in enumerate(day_summary.sessions, start=1):
        mid_hour = (session.offset_bins + session.n_bins / 2.0) / 3600.0
        ax_timeline.text(
            mid_hour,
            1.08,
            f"Session {session_index}",
            transform=ax_timeline.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=max(11, POSTER_NOTE_SIZE - 6),
            fontweight="bold",
            rotation=0,
        )

    image_title = "meanImg"
    plot_roi_overlays(
        ax_image,
        mean_img,
        contours,
        image_title,
        show_labels=True,
        title_pad=2.0,
        title_fontsize=max(20, POSTER_TITLE_SIZE - 4),
    )

    ax_trace.plot(day_summary.x_hours, dend_trace, color="#222222", linewidth=0.9)
    add_session_boundaries(ax_trace, day_summary.boundary_hours)
    apply_shared_time_axis(ax_trace, total_hours, show_labels=False)
    ax_trace.set_ylabel("dF/F", fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=-1)
    ax_trace.yaxis.label.set_color(dendrite_color)
    ax_trace.yaxis.set_tick_params(labelsize=max(10, POSTER_FONT_SIZE - 3), pad=0.25)
    set_sparse_numeric_ticks(ax_trace, axis="y", nbins=4)
    ax_trace.set_title("Dendrite activity", fontsize=max(18, POSTER_TITLE_SIZE - 4), pad=4)
    ax_trace.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.6)

    if spine_heatmap.shape[0] > 0:
        clipped_heatmap = np.asarray(spine_heatmap, dtype=float)
        clipped_heatmap = np.where(np.isfinite(clipped_heatmap), np.clip(clipped_heatmap, 0.0, None), np.nan)
        finite = clipped_heatmap[np.isfinite(clipped_heatmap)]
        if finite.size:
            vmax = float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        else:
            vmax = 1.0
        masked_heatmap = np.ma.masked_invalid(clipped_heatmap)
        im = ax_heatmap.imshow(
            masked_heatmap,
            aspect="auto",
            interpolation="nearest",
            cmap="binary",
            vmin=0.0,
            vmax=vmax,
            extent=[0.0, total_hours, float(spine_heatmap.shape[0]), 0.0],
            origin="upper",
        )
        add_session_boundaries(ax_heatmap, day_summary.boundary_hours)
        ax_heatmap.set_yticks(np.arange(spine_heatmap.shape[0]) + 0.5)
        ax_heatmap.set_yticklabels(spine_labels, fontsize=POSTER_FONT_SIZE)
        for tick_label, color in zip(ax_heatmap.get_yticklabels(), spine_colors):
            tick_label.set_color(color)
        apply_shared_time_axis(ax_heatmap, total_hours, show_labels=True)
        ax_heatmap.set_ylabel("spines", fontsize=max(11, POSTER_LABEL_SIZE - 6), labelpad=-1)
        ax_heatmap.set_title("Spine activity heatmap", fontsize=max(18, POSTER_TITLE_SIZE - 4), pad=5)
    else:
        ax_heatmap.text(
            0.5,
            0.5,
            "No spines with day observations",
            transform=ax_heatmap.transAxes,
            ha="center",
            va="center",
            fontsize=POSTER_TITLE_SIZE,
        )
        ax_heatmap.set_axis_off()

    ax_heatmap.set_xlabel("stitched time (h)", fontsize=POSTER_LABEL_SIZE)

    day_title = format_dendrite_display_name(animal_id, dendrite_compartment, dend_local.get("dendrite_id"))
    fig.suptitle(day_title, fontsize=max(20, POSTER_TITLE_SIZE - 2), x=0.53, y=0.965)
    fig.tight_layout(rect=[0.0, 0.0, 0.985, 0.88])
    add_broad_state_key(ax_state_key, show_title=False)

    if spine_heatmap.shape[0] > 0:
        heatmap_box = ax_heatmap.get_position()
        cbar_width = 0.012
        cbar_pad = 0.006
        cbar_left = min(heatmap_box.x1 + cbar_pad, 0.985 - cbar_width)
        cbar_ax = fig.add_axes([cbar_left, heatmap_box.y0, cbar_width, heatmap_box.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)

    ensure_dir(output_path.parent)
    save_figure(fig, output_path, dpi=dpi)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate full-day dendrite/spine figure summaries from the cached analysis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Config JSON with the movie/sleep expID lists")
    parser.add_argument("--cache-path", type=Path, help="Optional override for the cached NPZ file")
    parser.add_argument("--output-dir", type=Path, help="Optional override for the analysis output directory")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Figure DPI for saved PNGs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if plt is None:
        raise SystemExit("matplotlib is required to generate the dendrite figures")

    config = load_json_config(args.config)
    user_id = str(config.get("user_id") or "")
    repo_base = Path(config.get("repo_base") or (f"/home/{user_id}/data/Repository" if user_id else "/home/rubencorreia/data/Repository"))
    output_dir = Path(args.output_dir or config.get("output_dir") or DEFAULT_OUTPUT_DIR)
    cache_path = Path(args.cache_path or config.get("cache_path") or DEFAULT_CACHE_PATH)
    basal_expids = set(parse_list_argument(config.get("basal_expids")))
    apical_expids = set(parse_list_argument(config.get("apical_expids")))
    ensure_dir(output_dir)
    with step_scope("load day cache"):
        cache = load_npz_cache(cache_path)

    day_groups = grouped_day_expids(config, cache)
    if not day_groups:
        raise SystemExit(
            describe_day_group_mismatch(
                config,
                cache,
                config_path=args.config,
                cache_path=cache_path,
            )
        )

    written: List[str] = []
    warned_mismatch_expids: set[str] = set()
    with step_scope("day figure generation", total=len(day_groups)):
        for day_idx, ((animal_id, date), exp_ids) in enumerate(day_groups.items(), start=1):
            step_progress(day_idx, len(day_groups), label=f"{animal_id} | {date}")
            day_summary = build_day_summary(animal_id, date, exp_ids, cache)
            dendrite_records = selected_dendrite_records(cache, animal_id, date, exp_ids)
            with step_scope("day dendrites", total=len(dendrite_records)):
                for dend_idx, (global_dendrite_id, dendrite_record) in enumerate(dendrite_records, start=1):
                    step_progress(dend_idx, len(dendrite_records), label=str(global_dendrite_id))
                    representative_exp_id = choose_representative_exp_id(dendrite_record, day_summary.exp_ids)
                    representative_obs = (
                        dendrite_record.get("observations", {}).get(representative_exp_id, {}) if representative_exp_id is not None else {}
                    )
                    if representative_exp_id is not None and representative_exp_id not in warned_mismatch_expids:
                        expected_compartment = configured_compartment_for_exp_id(
                            representative_exp_id,
                            basal_expids,
                            apical_expids,
                        )
                        cached_compartment = (
                            representative_obs.get("compartment")
                            or dendrite_record.get("compartment")
                            or cache.get("experiments", {}).get(representative_exp_id, {}).get("compartment")
                            or ""
                        )
                        warn_if_compartment_mismatch(representative_exp_id, expected_compartment, cached_compartment)
                        if expected_compartment is not None:
                            warned_mismatch_expids.add(representative_exp_id)
                    dendrite_compartment = str(
                        representative_obs.get("compartment")
                        or dendrite_record.get("compartment")
                        or cache.get("experiments", {}).get(representative_exp_id or "", {}).get("compartment")
                        or ""
                    ).strip()
                    figure_path = build_figure_save_path(output_dir, animal_id, day_summary.date, dendrite_compartment, global_dendrite_id)
                    plot_day_figure(cache, day_summary, animal_id, global_dendrite_id, dendrite_record, figure_path, repo_base, args.dpi)
                    written.append(str(figure_path))
                    print(f"[saved] {figure_path}")

    with step_scope("cleanup ROI detail figures"):
        removed = cleanup_roi_detail_figures(output_dir / "figures")
        if removed:
            step_message(f"removed {len(removed)} stale ROI detail PNG/SVG files")

    print(json.dumps({"n_figures": len(written), "output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
