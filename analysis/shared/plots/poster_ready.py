from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

from analysis.shared.shared_calcium_response import build_state_masks_movie, choose_locomotion_threshold, extract_cut_neural_bundle, find_first_key, load_visual_response_cut_data, read_pickle, visual_response_trial_group
from analysis.compartment_common import pick_state_bundle
from poster_plotting import (
    POSTER_FONT_SIZE,
    POSTER_LABEL_SIZE,
    POSTER_LEGEND_SIZE,
    POSTER_NOTE_SIZE,
    POSTER_TITLE_SIZE,
    configure_poster_matplotlib,
    save_figure,
)

if plt is not None:
    configure_poster_matplotlib()


FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 7.5
MIXED_MODEL_HEIGHT_CM = 8.3
VISUAL_RESPONSE_WIDTH_CM = 19.0
VISUAL_RESPONSE_HEIGHT_CM = 6.5

FIGURE_TITLE_FS = POSTER_TITLE_SIZE
FIGURE_LABEL_FS = POSTER_LABEL_SIZE
FIGURE_TICK_FS = POSTER_FONT_SIZE
FIGURE_NOTE_FS = POSTER_NOTE_SIZE
FIGURE_LEGEND_FS = POSTER_LEGEND_SIZE

RESPONSIVE_COLOR = "#4C72B0"
NONRESPONSIVE_COLOR = "#DD8452"
MIXED_MODEL_COLOR = "#1F77B4"
BOX_COLORS = {
    "responsive": "#4C72B0",
    "nonresponsive": "#DD8452",
    "quiet_awake_blank": "#ff7f0e",
    "nrem_blank": "#2ca02c",
    "rem_blank": "#d62728",
    "quiet_awake": "#ff7f0e",
    "nrem": "#2ca02c",
    "rem": "#d62728",
    "quiet_awake_movies": "#ff7f0e",
    "nrem_movies": "#2ca02c",
    "rem_movies": "#d62728",
    "basal": "#4C72B0",
    "apical": "#DD8452",
}

FIGURE_2_STATE_ORDER = ["quiet_awake_blank", "nrem_blank", "rem_blank"]
FIGURE_3_BLANK_STATE_ORDER = ["quiet_awake_blank", "nrem_blank", "rem_blank"]
FIGURE_3_MOVIE_STATE_ORDER = ["quiet_awake_movies", "nrem_movies", "rem_movies"]


def cm_to_inch(value_cm: float) -> float:
    return float(value_cm) / 2.54


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rows(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        if value and all(isinstance(v, Mapping) for v in value.values()):
            return [dict(row) for row in value.values() if isinstance(row, Mapping)]
        return [dict(value)]
    return []


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def _coerce_bool(value: Any) -> bool:
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


def _row_roi_lookup_key(row: Mapping[str, Any]) -> tuple[str, str]:
    compartment = str(row.get("compartment") or "").strip().lower()
    if compartment == "soma":
        global_id = row.get("global_soma_id")
    elif compartment == "bouton":
        global_id = row.get("global_bouton_id")
    else:
        global_id = row.get("global_soma_id") or row.get("global_bouton_id")
    return compartment, str(global_id).strip() if global_id is not None else ""


def _assign_visual_response_cohorts(rows: Sequence[Mapping[str, Any]], visual_response_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup: Dict[tuple[str, str], str] = {}
    for row in visual_response_rows:
        key = _row_roi_lookup_key(row)
        if _coerce_bool(row.get("responsive", False)) or str(row.get("cohort") or "").strip().lower() == "responsive":
            lookup[key] = "responsive"
        elif key not in lookup:
            lookup[key] = "nonresponsive"
    assigned: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        row_copy["cohort"] = lookup.get(_row_roi_lookup_key(row_copy), "nonresponsive")
        assigned.append(row_copy)
    return assigned


def _unit_cohort_lookup(visual_response_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in visual_response_rows:
        unit_id = str(row.get("unit_id") or "").strip()
        if not unit_id:
            continue
        cohort = "responsive" if _coerce_bool(row.get("responsive", False)) or str(row.get("cohort") or "").strip().lower() == "responsive" else "nonresponsive"
        lookup[unit_id] = cohort
    return lookup


def assign_pairwise_visual_response_cohorts(rows: Sequence[Mapping[str, Any]], visual_response_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = _unit_cohort_lookup(visual_response_rows)
    assigned: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        left_cohort = lookup.get(str(row_copy.get("left_unit_id") or "").strip(), "mixed")
        right_cohort = lookup.get(str(row_copy.get("right_unit_id") or "").strip(), "mixed")
        row_copy["left_member_cohort"] = left_cohort
        row_copy["right_member_cohort"] = right_cohort
        row_copy["cohort"] = left_cohort if left_cohort == right_cohort and left_cohort in {"responsive", "nonresponsive"} else "mixed"
        assigned.append(row_copy)
    return assigned


def split_rows_by_cohort(rows: Sequence[Mapping[str, Any]], *, cohort_key: str = "cohort") -> Dict[str, list[dict[str, Any]]]:
    grouped: Dict[str, list[dict[str, Any]]] = {"all": [] , "responsive": [], "nonresponsive": []}
    for row in rows:
        row_copy = dict(row)
        grouped["all"].append(row_copy)
        cohort = str(row_copy.get(cohort_key) or "").strip().lower()
        if cohort in {"responsive", "nonresponsive"}:
            grouped[cohort].append(row_copy)
    return grouped


def _poster_mixed_model_term_kind(term: str) -> str:
    term = str(term or "")
    if term == "Intercept":
        return "intercept"
    if ":" in term:
        return "interaction"
    if term.startswith("state["):
        return "state"
    if term.startswith("compartment["):
        return "compartment"
    return "covariate"


def _poster_mixed_model_term_value_label(term: str) -> str:
    text = str(term or "")
    if text.startswith("state[") and text.endswith("]"):
        return text[len("state[") : -1]
    if text.startswith("compartment[") and text.endswith("]"):
        return text[len("compartment[") : -1]
    return text


def _poster_mixed_model_term_label(term: str) -> str:
    term = str(term or "")
    if term == "Intercept":
        return term
    if ":" in term:
        return " x ".join(_poster_mixed_model_term_label(part) for part in term.split(":"))
    kind = _poster_mixed_model_term_kind(term)
    value = _poster_mixed_model_term_value_label(term)
    if kind == "state":
        return _state_display_label(value)
    if kind == "compartment":
        return _state_display_label(value)
    return _state_display_label(term)


def _mixed_model_term_state_keys(term: str) -> set[str]:
    keys: set[str] = set()
    for part in str(term or "").split(":"):
        part = str(part or "")
        if part.startswith("state[") and part.endswith("]"):
            keys.add(normalize_state_label(part[len("state[") : -1]))
        elif part.startswith("state:"):
            keys.add(normalize_state_label(part.split(":", 1)[1]))
        elif part.startswith("state_"):
            keys.add(normalize_state_label(part[len("state_") :]))
    return keys


def _forest_row_label(row: Mapping[str, Any]) -> str:
    term = str(row.get("term") or "")
    if term:
        return _poster_mixed_model_term_label(term)
    state_a = row.get("state_a")
    state_b = row.get("state_b")
    contrast_type = str(row.get("contrast_type") or "")
    state_value = row.get("state")
    if state_a is not None or state_b is not None:
        left = _state_display_label(state_a)
        right = _state_display_label(state_b)
        if contrast_type == "basal_apical":
            return f"{left} x {right}".strip()
        return f"{left} - {right}".strip()
    if state_value is not None and contrast_type == "basal_apical":
        return f"{_state_display_label(state_value)} X Basal - Apical"
    contrast_name = str(row.get("contrast_name") or "")
    if contrast_name:
        return _state_display_label(contrast_name)
    return ""


def _state_display_label(state_label: Any) -> str:
    canonical = str(state_label or "").strip().lower().replace("-", "_")
    while "__" in canonical:
        canonical = canonical.replace("__", "_")
    parts = [part for part in canonical.split("_") if part]
    if not parts:
        return ""
    if len(parts) == 1 and parts[0] == "nrem":
        return "NREM"
    if len(parts) == 1 and parts[0] == "rem":
        return "REM"
    if parts[0] == "nrem":
        tail = " ".join(part.capitalize() for part in parts[1:])
        return f"NREM {tail}".strip()
    if parts[0] == "rem":
        tail = " ".join(part.capitalize() for part in parts[1:])
        return f"REM {tail}".strip()
    if len(parts) >= 2 and parts[0] in {"quiet", "active"} and parts[1] == "awake":
        head = f"{parts[0].capitalize()} Awake"
        tail = " ".join(part.capitalize() for part in parts[2:])
        return f"{head} {tail}".strip()
    return " ".join(part.capitalize() for part in parts)


def _canonical_state_key(state_label: Any) -> str:
    canonical = str(state_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in canonical:
        canonical = canonical.replace("__", "_")
    return canonical


def normalize_state_label(state_label: Any) -> str:
    return _canonical_state_key(state_label)


def _state_compare_key(state_label: Any) -> str:
    key = _canonical_state_key(state_label)
    for prefix in ("basal_", "apical_"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _state_matches_any(state_label: Any, candidates: Sequence[str]) -> bool:
    state_key = _canonical_state_key(state_label)
    state_compare = _state_compare_key(state_label)
    candidate_keys = {_canonical_state_key(candidate) for candidate in candidates if str(candidate).strip()}
    candidate_compares = {_state_compare_key(candidate) for candidate in candidates if str(candidate).strip()}
    return bool(state_key in candidate_keys or state_compare in candidate_compares)


def selected_states_present(state_values: Mapping[str, Sequence[float]], state_order: Sequence[str]) -> list[str]:
    return [state for state in state_order if _finite_array(state_values.get(state, [])).size]


def filter_mixed_model_terms_to_states(rows: Sequence[Mapping[str, Any]], allowed_states: Sequence[str]) -> list[dict[str, Any]]:
    allowed = {normalize_state_label(state) for state in allowed_states if str(state).strip()}
    allowed_compares = {_state_compare_key(state) for state in allowed_states if str(state).strip()}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        term = str(row.get("term") or row.get("contrast_name") or "")
        kind = _poster_mixed_model_term_kind(term) if term else "covariate"
        if kind not in {"intercept", "state", "compartment", "interaction"}:
            continue
        state_keys = set(_mixed_model_term_state_keys(term))
        compare_keys: set[str] = set()
        for key in ("state", "state_a", "state_b", "state_display", "state_label", "state_a_display", "state_b_display"):
            value = row.get(key)
            if value is not None:
                state_keys.add(normalize_state_label(value))
                compare_keys.add(_state_compare_key(value))
        if kind in {"intercept", "compartment"}:
            filtered.append(dict(row))
            continue
        if not state_keys:
            continue
        if state_keys.intersection(allowed) or compare_keys.intersection(allowed_compares):
            filtered.append(dict(row))
    return filtered


def _state_value_map_from_rows(rows: Sequence[Mapping[str, Any]], *, value_key: str | Sequence[str] = "mean") -> Dict[str, list[float]]:
    grouped: Dict[str, list[float]] = {}
    value_keys = (value_key,) if isinstance(value_key, str) else tuple(str(key) for key in value_key)
    for row in rows:
        state = _canonical_state_key(row.get("state") or row.get("state_label") or row.get("state_display") or "")
        if not state:
            continue
        value = None
        for key in value_keys:
            candidate = row.get(key)
            if candidate is not None:
                value = candidate
                break
        if value is None:
            continue
        try:
            value_f = float(value)
        except Exception:
            continue
        if not np.isfinite(value_f):
            continue
        grouped.setdefault(state, []).append(value_f)
    return grouped


def _mean_and_sem(values: Sequence[float]) -> tuple[float, float]:
    arr = _finite_array(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem


def _pick_exemplar_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    responsive = [dict(row) for row in rows if _coerce_bool(row.get("responsive", False)) and np.isfinite(float(row.get("delta", float("nan"))))]
    nonresponsive = [dict(row) for row in rows if not _coerce_bool(row.get("responsive", False)) and np.isfinite(float(row.get("delta", float("nan"))))]
    responsive_row = None
    if responsive:
        responsive_row = max(responsive, key=lambda row: float(row.get("delta", float("nan"))))
    nonresponsive_row = None
    if nonresponsive:
        deltas = np.asarray([float(row.get("delta", float("nan"))) for row in nonresponsive], dtype=float)
        finite = deltas[np.isfinite(deltas)]
        if finite.size:
            target = float(np.nanmedian(finite))
            nonresponsive_row = min(nonresponsive, key=lambda row: abs(float(row.get("delta", float("nan"))) - target))
    return responsive_row, nonresponsive_row


def _visual_response_entity_id(row: Mapping[str, Any]) -> str:
    compartment = str(row.get("compartment") or "").strip().lower()
    if compartment == "soma":
        value = row.get("global_soma_id")
    elif compartment == "bouton":
        value = row.get("global_bouton_id")
    else:
        value = row.get("global_soma_id") or row.get("global_bouton_id")
    return str(value).strip() if value is not None else ""


def _canonicalize_visual_response_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entity_id = _visual_response_entity_id(row)
        if not entity_id:
            continue
        grouped.setdefault(entity_id, []).append(dict(row))
    canonical_rows: list[dict[str, Any]] = []
    for entity_id, members in grouped.items():
        row = dict(members[0])
        responsive = any(_coerce_bool(member.get("responsive", False)) or str(member.get("cohort") or "").strip().lower() == "responsive" for member in members)
        row["responsive"] = responsive
        row["cohort"] = "responsive" if responsive else "nonresponsive"
        row["entity_id"] = entity_id
        canonical_rows.append(row)
    return canonical_rows


def _visual_response_unique_entity_count(rows: Sequence[Mapping[str, Any]]) -> int:
    canonical_rows = _canonicalize_visual_response_rows(rows)
    if canonical_rows:
        return int(len(canonical_rows))
    unique_ids: set[str] = set()
    fallback = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entity_id = _visual_response_entity_id(row)
        if entity_id:
            unique_ids.add(entity_id)
        else:
            fallback += 1
    return int(len(unique_ids) if unique_ids else fallback)


def _load_visual_response_plot_data(response_row: Mapping[str, Any], *, locomotion_threshold: float | None = None) -> Optional[Dict[str, Any]]:
    source_path = response_row.get("source_path")
    if not source_path:
        return None
    source_path = Path(str(source_path))
    if not source_path.exists():
        return None
    exp_root = source_path.parent.parent if len(source_path.parents) >= 2 else None
    if exp_root is None or not exp_root.exists():
        return None
    match = None
    for token in (source_path.name, source_path.stem):
        if "ch" in token:
            match = token
            break
    if match is None:
        return None
    channel = None
    for chunk in source_path.name.split("_"):
        if chunk.startswith("ch"):
            try:
                channel = int(chunk[2:].split(".")[0])
            except Exception:
                channel = None
            break
    if channel is None:
        return None
    try:
        _, state_bundle = pick_state_bundle(exp_root, str(response_row.get("mode") or "movie"))
    except Exception:
        return None
    trial_rows = state_bundle.get("rows", []) if isinstance(state_bundle, Mapping) else []
    cut_data = load_visual_response_cut_data(exp_root, channel, trial_rows, locomotion_threshold=locomotion_threshold)
    if not cut_data:
        return None
    cut_time = np.asarray(cut_data.get("cut_time"), dtype=float)
    cut_neural = np.asarray(cut_data.get("cut_neural"), dtype=float)
    trial_meta = cut_data.get("trial_meta", []) if isinstance(cut_data, dict) else []
    roi_index = int(response_row.get("roi_index", -1))
    if roi_index < 0 or roi_index >= cut_neural.shape[0] or cut_time.size == 0:
        return None
    trial_matrix = np.asarray(cut_neural[roi_index], dtype=float)
    visual_traces: list[np.ndarray] = []
    blank_traces: list[np.ndarray] = []
    visual_values: list[float] = []
    blank_values: list[float] = []
    for meta in trial_meta:
        if not isinstance(meta, Mapping):
            continue
        trial_index = meta.get("trial_index")
        if trial_index is None:
            continue
        try:
            trial_index = int(trial_index)
        except Exception:
            continue
        if trial_index < 0 or trial_index >= trial_matrix.shape[0]:
            continue
        group = visual_response_trial_group(meta.get("state_label"))
        if group is None:
            continue
        trace = np.asarray(trial_matrix[trial_index], dtype=float)
        if not np.isfinite(trace).any():
            continue
        stim_mask = np.isfinite(trace) & np.isfinite(cut_time) & (cut_time >= 0)
        duration = meta.get("duration")
        try:
            duration_f = float(duration) if duration is not None else None
        except Exception:
            duration_f = None
        if duration_f is not None and np.isfinite(duration_f):
            stim_mask &= cut_time < duration_f
        if not np.any(stim_mask):
            continue
        stimulus = float(np.nanmean(trace[stim_mask]))
        if group == "visual":
            visual_traces.append(trace)
            visual_values.append(stimulus)
        else:
            blank_traces.append(trace)
            blank_values.append(stimulus)
    if not visual_values or not blank_values:
        return None
    visual_arr = np.asarray(visual_values, dtype=float)
    blank_arr = np.asarray(blank_values, dtype=float)
    durations = []
    for meta in trial_meta:
        if not isinstance(meta, Mapping):
            continue
        try:
            duration_f = float(meta.get("duration")) if meta.get("duration") is not None else float("nan")
        except Exception:
            duration_f = float("nan")
        if np.isfinite(duration_f) and duration_f > 0:
            durations.append(duration_f)
    if durations:
        event_duration = float(np.nanmax(np.asarray(durations, dtype=float)))
    else:
        event_duration = 30.0
    return {
        "cut_time": cut_time,
        "event_onset": 0.0,
        "event_duration": event_duration,
        "visual_traces": visual_traces,
        "blank_traces": blank_traces,
        "visual_values": visual_arr,
        "blank_values": blank_arr,
        "visual_mean_trace": np.asarray(np.nanmean(np.asarray(visual_traces, dtype=float), axis=0), dtype=float),
        "blank_mean_trace": np.asarray(np.nanmean(np.asarray(blank_traces, dtype=float), axis=0), dtype=float),
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=FIGURE_TICK_FS)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _set_boxplot_colors(bp: Mapping[str, Any], colors: Sequence[str]) -> None:
    for element_name in ("boxes", "whiskers", "caps", "medians"):
        for artist in bp.get(element_name, []):
            if element_name == "boxes":
                continue
            artist.set_color("#555555")
    for patch, color in zip(bp.get("boxes", []), colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.9)


def _boxplot(
    ax: plt.Axes,
    state_values: Mapping[str, Sequence[float]],
    state_order: Sequence[str],
    *,
    title: str,
    ylabel: str,
    cohort_label: str | None = None,
    significance_flags: Sequence[bool] | None = None,
    sample_sizes: Mapping[str, int] | None = None,
    horizontal: bool = False,
    box_width: float = 0.56,
    bracket_step_scale: float | None = None,
    bracket_height_scale: float | None = None,
    bracket_text_scale: float | None = None,
    extent_padding_scale: float | None = None,
    horizontal_extent_padding_scale: float | None = None,
) -> None:
    series = []
    labels = []
    positions = []
    colors = []
    sample_counts = []
    present_states = []
    present_flags = []
    flags = list(significance_flags) if significance_flags is not None else None
    sig_extent = None
    ordered_states = list(state_order)[::-1] if horizontal else list(state_order)
    tick_positions = [float(idx) for idx in range(1, len(ordered_states) + 1)]
    tick_labels = [_state_display_label(state) for state in ordered_states]
    for idx, state in enumerate(ordered_states, start=1):
        arr = _finite_array(state_values.get(state, []))
        if arr.size == 0:
            continue
        present_states.append(state)
        labels.append(_state_display_label(state))
        positions.append(float(idx))
        colors.append(_poster_label_color(state))
        sample_counts.append(int(sample_sizes.get(state, int(arr.size)) if sample_sizes is not None else int(arr.size)))
        series.append(arr)
        if flags is not None:
            orig_index = state_order.index(state) if state in state_order else idx - 1
            present_flags.append(bool(flags[orig_index]) if orig_index < len(flags) else False)
    if series:
        bp = ax.boxplot(series, positions=positions, widths=box_width, patch_artist=True, showfliers=False, vert=not horizontal)
        for patch, color in zip(bp.get("boxes", []), colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#444444")
            patch.set_alpha(0.9)
        for line in bp.get("whiskers", []) + bp.get("caps", []):
            line.set_color("#555555")
        for median in bp.get("medians", []):
            median.set_color("#222222")
            median.set_linewidth(1.5)
        rng = np.random.default_rng(7)
        for xpos, state in zip(positions, present_states):
            arr = _finite_array(state_values.get(state, []))
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            if horizontal:
                ax.scatter(arr, np.full(arr.size, xpos) + jitter, s=14, alpha=0.45, color=BOX_COLORS.get(state, "#7f7f7f"), edgecolor="none")
            else:
                ax.scatter(np.full(arr.size, xpos) + jitter, arr, s=14, alpha=0.45, color=BOX_COLORS.get(state, "#7f7f7f"), edgecolor="none")
        finite = np.concatenate(series)
        finite = finite[np.isfinite(finite)]
        if finite.size and present_flags:
            pad = max(0.03 * float(np.ptp(finite)), 0.02)
            if horizontal:
                x = float(np.nanmax(finite)) + pad
                sig_extent = x
                for ypos, state, is_sig in zip(positions, present_states, present_flags):
                    if is_sig:
                        ax.text(x + max(0.002 * float(np.ptp(finite)), 0.0015), ypos, "*", ha="left", va="center", fontsize=FIGURE_NOTE_FS, color="#8b0000", fontweight="bold")
            else:
                y = float(np.nanmax(finite)) + pad
                sig_extent = y
                for xpos, state, is_sig in zip(positions, present_states, present_flags):
                    if is_sig:
                        ax.text(xpos, y, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS, color="#8b0000", fontweight="bold")
        if horizontal:
            ax.set_ylim(0.5, max(float(len(ordered_states)) + 0.5, 1.5))
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=FIGURE_TICK_FS)
            ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
            ax.set_ylabel("")
        else:
            ax.set_xlim(0.5, max(float(len(ordered_states)) + 0.5, 1.5))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=25, ha="right")
            ax.set_ylabel(ylabel, fontsize=FIGURE_LABEL_FS)
    else:
        if horizontal:
            ax.set_ylim(0.5, 1.5)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
            ax.set_ylabel("")
        else:
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylabel(ylabel, fontsize=FIGURE_LABEL_FS)
    if sig_extent is not None:
        if horizontal:
            current_left, current_right = ax.get_xlim()
            horizontal_pad_scale = horizontal_extent_padding_scale if horizontal_extent_padding_scale is not None else 0.025
            ax.set_xlim(current_left, max(current_right, sig_extent + max(horizontal_pad_scale * abs(sig_extent), 0.015)))
        else:
            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(current_bottom, max(current_top, sig_extent + max((extent_padding_scale if extent_padding_scale is not None else 0.04) * abs(sig_extent), 0.03)))
    y0, y1 = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    y_range = max(float(y1 - y0), 1e-6)
    x_range = max(float(x1 - x0), 1e-6)
    for pos, state, arr_size in zip(positions, present_states, sample_counts):
        if horizontal:
            arr = _finite_array(state_values.get(state, []))
            annotate_x = float(np.nanmax(arr)) if arr.size else x1
            ax.text(annotate_x + max(0.02 * x_range, 0.04), pos, f"n={arr_size}", ha="left", va="center", fontsize=FIGURE_NOTE_FS - 1, color=_poster_label_color(state), clip_on=False)
        else:
            arr = _finite_array(state_values.get(state, []))
            annotate_y = min(float(np.nanmax(arr)) + 0.03 * y_range, float(y1) - 0.01 * y_range) if arr.size else float(y1) - 0.01 * y_range
            ax.text(pos, annotate_y, f"n={arr_size}", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS - 1, color=_poster_label_color(state), clip_on=False)
    ax.set_title(title, fontsize=FIGURE_TITLE_FS, pad=4)
    ax.grid(axis="both" if horizontal else "y", alpha=0.22)
    if cohort_label:
        ax.text(0.02, 0.98, cohort_label, transform=ax.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")
    if present_flags and len(present_states) >= 2:
        comparison_targets = [(state, pos) for state, pos, is_sig in zip(present_states[1:], positions[1:], present_flags[1:]) if is_sig]
        if comparison_targets:
            reference_pos = positions[0]
            if horizontal:
                x0, x1 = ax.get_xlim()
                x_range = x1 - x0 if np.isfinite(x1 - x0) and (x1 - x0) > 0 else 1.0
                anchor = float(sig_extent if sig_extent is not None else x1)
                bracket_base = anchor + max(0.003 * x_range, 0.002)
                bracket_step = max((bracket_step_scale if bracket_step_scale is not None else 0.006) * x_range, 0.002)
                text_offset = max((bracket_text_scale if bracket_text_scale is not None else 0.0015) * x_range, 0.0015)
                bracket_height = max((bracket_height_scale if bracket_height_scale is not None else 0.0035) * x_range, 0.002)
                top_needed = bracket_base + len(comparison_targets) * bracket_step + bracket_height + text_offset + 0.003 * x_range
                if top_needed > x1:
                    ax.set_xlim(x0, top_needed)
                for level, (state, ypos) in enumerate(comparison_targets):
                    x = bracket_base + level * bracket_step
                    ax.plot([x, x + bracket_height, x + bracket_height, x], [reference_pos, reference_pos, ypos, ypos], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
            else:
                y0, y1 = ax.get_ylim()
                y_range = y1 - y0 if np.isfinite(y1 - y0) and (y1 - y0) > 0 else 1.0
                anchor = float(sig_extent if sig_extent is not None else y1)
                bracket_base = anchor + max(0.003 * y_range, 0.002)
                bracket_step = max((bracket_step_scale if bracket_step_scale is not None else 0.006) * y_range, 0.002)
                text_offset = max((bracket_text_scale if bracket_text_scale is not None else 0.0015) * y_range, 0.0015)
                bracket_height = max((bracket_height_scale if bracket_height_scale is not None else 0.0035) * y_range, 0.002)
                top_needed = bracket_base + len(comparison_targets) * bracket_step + bracket_height + text_offset + 0.003 * y_range
                if top_needed > y1:
                    ax.set_ylim(y0, top_needed)
                for level, (state, xpos) in enumerate(comparison_targets):
                    y = bracket_base + level * bracket_step
                    ax.plot([reference_pos, reference_pos, xpos, xpos], [y, y + bracket_height, y + bracket_height, y], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
    if present_flags and any(present_flags):
        ax.text(0.98, 0.98, "* significant comparisons", transform=ax.transAxes, ha="right", va="top", fontsize=FIGURE_NOTE_FS, color="#5a1a1a", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d0d0d0", alpha=0.88))
    if not series:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=FIGURE_NOTE_FS, color="#666666")
    _style_axes(ax)


def _forest_panel(ax: plt.Axes, rows: Sequence[Mapping[str, Any]], *, title: str, ylabel: str = "Estimate (95% CI)", p_value_key: str = "p_value", show_legend: bool = False, color_fn: Any = None) -> None:
    rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not rows:
        ax.set_axis_off()
        return
    ordered = [row for row in rows if row.get("term") is not None or row.get("contrast_name") is not None or row.get("state_a") is not None or row.get("state_b") is not None or row.get("state") is not None]
    if not ordered:
        ax.set_axis_off()
        return
    y_positions = np.arange(len(ordered))[::-1]
    estimates = []
    bounds = [0.0]
    for row in ordered:
        est = row.get("estimate")
        se = row.get("se")
        try:
            est_f = float(est)
        except Exception:
            est_f = float("nan")
        try:
            se_f = float(se)
        except Exception:
            se_f = float("nan")
        estimates.append((est_f, se_f))
        if np.isfinite(est_f):
            if np.isfinite(se_f):
                ci = 1.96 * se_f
                bounds.extend([est_f - ci, est_f + ci])
            else:
                bounds.append(est_f)
    ax.axvline(0.0, color="#333333", linewidth=1)
    for y_pos, row, (est_f, se_f) in zip(y_positions, ordered, estimates):
        if not np.isfinite(est_f):
            continue
        ci = 1.96 * se_f if np.isfinite(se_f) else float("nan")
        label = _forest_row_label(row)
        kind = _poster_mixed_model_term_kind(label) if label.startswith(("state[", "compartment[")) or label == "Intercept" else "interaction" if " X " in label else "covariate"
        default_color = {
            "intercept": "#7f7f7f",
            "state": "#1f77b4",
            "compartment": "#2ca02c",
            "interaction": "#ff7f0e",
            "covariate": "#9467bd",
        }.get(kind, "#7f7f7f")
        color = color_fn(row, label, kind) if callable(color_fn) else default_color
        if np.isfinite(ci):
            ax.errorbar(est_f, y_pos, xerr=ci, fmt="none", ecolor=color, elinewidth=1.4, capsize=3)
        ax.scatter(est_f, y_pos, s=50, color=color, edgecolor="#222222", linewidth=0.8, zorder=3)
        p_value = row.get(p_value_key)
        if p_value is not None:
            try:
                p_value_f = float(p_value)
            except Exception:
                p_value_f = float("nan")
        else:
            p_value_f = float("nan")
        if np.isfinite(p_value_f) and p_value_f < 0.05:
            ax.scatter(est_f, y_pos, s=105, marker="*", color="#111111", zorder=4)
    labels = []
    for row in ordered:
        label = _forest_row_label(row)
        labels.append(_poster_mixed_model_term_label(label) if label.startswith(("state[", "compartment[", "Intercept")) else label)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=FIGURE_TICK_FS)
    ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
    ax.set_title(title, fontsize=FIGURE_TITLE_FS, pad=4)
    ax.grid(axis="x", alpha=0.22)
    finite_bounds = np.asarray(bounds, dtype=float)
    finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
    if finite_bounds.size:
        lo = float(np.nanmin(finite_bounds))
        hi = float(np.nanmax(finite_bounds))
        pad = max(0.12 * max(hi - lo, 1e-6), 0.05)
        ax.set_xlim(lo - pad, hi + pad)
    if show_legend:
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#1F77B4", markeredgecolor="white", markersize=7, label="estimate"),
            Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=9, label="p < 0.05"),
        ]
        ax.legend(handles=legend_handles, frameon=False, fontsize=FIGURE_NOTE_FS, loc="upper right")
    _style_axes(ax)




def _normalize_mixed_model_contrast_p_source(value: Any) -> str:
    text = str(value or "classical").strip().lower()
    return "shuffle" if text == "shuffle" else "classical"


def _combined_limits(*series: Sequence[float]) -> tuple[float, float]:
    finite_parts = []
    for values in series:
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_parts.append(arr)
    if not finite_parts:
        return float("nan"), float("nan")
    finite = np.concatenate(finite_parts)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    pad = max(0.05 * max(hi - lo, 1e-6), 0.05)
    return lo - pad, hi + pad


def _visual_response_poster_shared_limits(*plot_data_items: Optional[Mapping[str, Any]]) -> tuple[float, float] | None:
    finite_series: list[np.ndarray] = []
    for plot_data in plot_data_items:
        if not isinstance(plot_data, Mapping):
            continue
        for key in ("blank_mean_trace", "visual_mean_trace", "blank_values", "visual_values"):
            series = plot_data.get(key)
            if series is None:
                continue
            arr = np.asarray(series, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                finite_series.append(arr)
    if not finite_series:
        return None
    lo, hi = _combined_limits(*finite_series)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return float(lo), float(hi)


def _resolve_visual_response_source_exp_id(source_cache: Optional[Mapping[str, Any]], exp_id: str, observation: Optional[Mapping[str, Any]] = None) -> str:
    if not isinstance(source_cache, Mapping):
        return exp_id
    experiments = source_cache.get("experiments", {})
    if exp_id in experiments:
        return exp_id
    if isinstance(observation, Mapping):
        for candidate in (observation.get("representative_exp_id"), *(observation.get("source_exp_ids", []) or [])):
            candidate = str(candidate or "")
            if candidate and candidate in experiments:
                return candidate
    return exp_id


def _load_visual_response_cut_data_from_source_cache(
    source_cache: Optional[Mapping[str, Any]],
    exp_id: str,
    cut_cache: dict[str, Optional[dict[str, Any]]],
    observation: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if exp_id in cut_cache:
        return cut_cache[exp_id]
    if not isinstance(source_cache, Mapping):
        cut_cache[exp_id] = None
        return None
    resolved_exp_id = _resolve_visual_response_source_exp_id(source_cache, exp_id, observation)
    exp_meta = source_cache.get("experiments", {}).get(resolved_exp_id, {})
    source_paths = exp_meta.get("source_paths", {}) if isinstance(exp_meta, Mapping) else {}
    exp_root = Path(str(source_paths.get("exp_root") or ""))
    cut_dir = Path(str(source_paths.get("cut") or (exp_root / "cut" if exp_root else "")))
    if not cut_dir.exists():
        cut_cache[exp_id] = None
        return None
    config = source_cache.get("config", {}) if isinstance(source_cache.get("config", {}), Mapping) else {}
    channel_value = config.get("channel") if isinstance(config, Mapping) else None
    channel = int(channel_value) if channel_value is not None else 1
    selected_path = None
    for candidate_path in (
        exp_root / "cut_intertrials" / f"s2p_ch{channel}_dF_cut.pickle",
        exp_root / "cut_with_intertrials" / f"s2p_ch{channel}_dF_cut.pickle",
    ):
        if candidate_path.exists():
            selected_path = candidate_path
            break
    if selected_path is None:
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
        cut_cache[exp_id] = None
        return None
    wheel_time = find_first_key(wheel_bundle, ["t", "time", "timestamps"]) if isinstance(wheel_bundle, Mapping) else None
    wheel_speed = find_first_key(wheel_bundle, ["speed", "wheel", "motion", "velocity"]) if isinstance(wheel_bundle, Mapping) else None
    wheel_interp = None
    if wheel_time is not None and wheel_speed is not None:
        wheel_time_arr = np.asarray(wheel_time, dtype=float).ravel()
        wheel_speed_arr = np.asarray(wheel_speed, dtype=float).ravel()
        common_len = min(wheel_time_arr.size, wheel_speed_arr.size)
        if common_len >= 2:
            wheel_time_arr = wheel_time_arr[:common_len]
            wheel_speed_arr = wheel_speed_arr[:common_len]
            wheel_interp = np.asarray(np.interp(cut_time, wheel_time_arr, wheel_speed_arr), dtype=float)
    trial_meta = [dict(meta) for meta in exp_meta.get("trial_meta", []) if isinstance(meta, Mapping)]
    payload = {
        "cut_time": np.asarray(cut_time, dtype=float),
        "cut_neural": np.asarray(cut_neural, dtype=float),
        "trial_meta": trial_meta,
        "source_label": "cut_intertrials" if (exp_root / "cut_intertrials" / f"s2p_ch{channel}_dF_cut.pickle").exists() else "cut_with_intertrials",
        "source_path": str(selected_path),
    }
    cut_cache[exp_id] = payload
    return payload


def _visual_response_entity_observation(
    cache: Mapping[str, Any],
    kind: str,
    response_row: Mapping[str, Any],
) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    animal_id = str(response_row.get("animal_id") or "")
    if not animal_id:
        return None, None, None, None
    animals = cache.get("animals", {}) if isinstance(cache, Mapping) else {}
    animal_entry = animals.get(animal_id, {}) if isinstance(animals, Mapping) else {}
    if kind == "dendrite":
        entity_id = str(response_row.get("global_dendrite_id") or response_row.get("dendrite_id") or "")
        dendrite_record = animal_entry.get("dendrites", {}).get(entity_id) if isinstance(animal_entry, Mapping) else None
        if not isinstance(dendrite_record, Mapping):
            return None, None, None, None
        observations = dendrite_record.get("observations", {}) if isinstance(dendrite_record, Mapping) else {}
        if not isinstance(observations, Mapping) or not observations:
            return None, None, dendrite_record, None
        exp_id, observation = next(iter(sorted(observations.items())))
        return str(exp_id), observation if isinstance(observation, Mapping) else None, dendrite_record, None
    if kind == "spine":
        entity_id = str(response_row.get("global_spine_id") or response_row.get("spine_id") or "")
        dendrites = animal_entry.get("dendrites", {}) if isinstance(animal_entry, Mapping) else {}
        if not isinstance(dendrites, Mapping):
            return None, None, None, None
        for dendrite_record in dendrites.values():
            if not isinstance(dendrite_record, Mapping):
                continue
            spine_record = dendrite_record.get("spines", {}).get(entity_id) if isinstance(dendrite_record.get("spines", {}), Mapping) else None
            if not isinstance(spine_record, Mapping):
                continue
            observations = spine_record.get("observations", {}) if isinstance(spine_record, Mapping) else {}
            if not isinstance(observations, Mapping) or not observations:
                return None, None, dendrite_record, None
            exp_id, observation = next(iter(sorted(observations.items())))
            parent_observation = dendrite_record.get("observations", {}).get(exp_id) if isinstance(dendrite_record.get("observations", {}), Mapping) else None
            return str(exp_id), observation if isinstance(observation, Mapping) else None, dendrite_record, parent_observation if isinstance(parent_observation, Mapping) else None
        return None, None, None, None
    entity_key = f"{kind}s"
    entity_id = str(
        response_row.get("unit_id")
        or response_row.get(f"global_{kind}_id")
        or response_row.get(f"{kind}_id")
        or response_row.get("roi_id")
        or response_row.get("roi_index")
        or ""
    )
    entity_record = animal_entry.get(entity_key, {}).get(entity_id) if isinstance(animal_entry, Mapping) else None
    if not isinstance(entity_record, Mapping):
        return None, None, None, None
    observations = entity_record.get("observations", {}) if isinstance(entity_record, Mapping) else {}
    if not isinstance(observations, Mapping) or not observations:
        return None, None, entity_record, None
    exp_id, observation = next(iter(sorted(observations.items())))
    return str(exp_id), observation if isinstance(observation, Mapping) else None, entity_record, None


def _visual_response_entity_plot_data(
    source_cache: Optional[Mapping[str, Any]],
    kind: str,
    exp_id: str,
    cut_cache: dict[str, Optional[dict[str, Any]]],
    observation: Mapping[str, Any],
    parent_dendrite_observation: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    exp_id = str(exp_id or "")
    if not exp_id:
        return None
    cut_data = _load_visual_response_cut_data_from_source_cache(source_cache, exp_id, cut_cache, observation) if source_cache else None
    if not cut_data:
        return None
    roi_index_value = observation.get("local_ids", {}).get("conversion_index")
    try:
        roi_index = int(roi_index_value)
    except Exception:
        roi_index = -1
    if roi_index < 0:
        return None
    cut_neural = np.asarray(cut_data.get("cut_neural"), dtype=float)
    cut_time = np.asarray(cut_data.get("cut_time"), dtype=float)
    trial_meta = cut_data.get("trial_meta", []) if isinstance(cut_data, dict) else []
    if roi_index >= cut_neural.shape[0] or cut_time.size == 0:
        return None
    if kind == "spine":
        if parent_dendrite_observation is None:
            return None
        dendrite_index_value = parent_dendrite_observation.get("local_ids", {}).get("conversion_index")
        try:
            dendrite_index = int(dendrite_index_value)
        except Exception:
            dendrite_index = -1
        alpha = parent_dendrite_observation.get("alpha")
        try:
            alpha_f = float(alpha) if alpha is not None else None
        except Exception:
            alpha_f = None
        if dendrite_index < 0 or alpha_f is None or dendrite_index >= cut_neural.shape[0]:
            return None
        trial_matrix = np.asarray(cut_neural[roi_index] - alpha_f * cut_neural[dendrite_index], dtype=float)
    else:
        trial_matrix = np.asarray(cut_neural[roi_index], dtype=float)
    visual_traces: list[np.ndarray] = []
    blank_traces: list[np.ndarray] = []
    visual_values: list[float] = []
    blank_values: list[float] = []
    for meta in trial_meta:
        if not isinstance(meta, Mapping):
            continue
        trial_label = str(meta.get("state_label") or "")
        group = visual_response_trial_group(trial_label)
        if group is None:
            continue
        trial_index = meta.get("trial_index")
        try:
            trial_index = int(trial_index)
        except Exception:
            continue
        if trial_index < 0 or trial_index >= trial_matrix.shape[0]:
            continue
        trial_trace = np.asarray(trial_matrix[trial_index], dtype=float)
        if not np.isfinite(trial_trace).any():
            continue
        duration = meta.get("duration")
        try:
            duration_f = float(duration) if duration is not None else None
        except Exception:
            duration_f = None
        stim_mask = np.isfinite(trial_trace) & np.isfinite(cut_time) & (cut_time >= 0)
        if duration_f is not None and np.isfinite(duration_f):
            stim_mask &= cut_time < duration_f
        if not np.any(stim_mask):
            continue
        stimulus = float(np.nanmean(trial_trace[stim_mask]))
        if group == "visual":
            visual_traces.append(trial_trace)
            visual_values.append(stimulus)
        else:
            blank_traces.append(trial_trace)
            blank_values.append(stimulus)
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
        "visual_label": "movies",
        "blank_label": "blank",
    }


def _single_exemplar_panel(
    ax_blank: plt.Axes,
    ax_movie: plt.Axes,
    ax_box: plt.Axes,
    response_row: Mapping[str, Any],
    *,
    cache: Optional[Mapping[str, Any]] = None,
    source_cache: Optional[Mapping[str, Any]] = None,
    kind: str = "dendrite",
    title: str,
    panel_label: str | None = None,
    show_movie_ylabel: bool = False,
    show_box_ylabel: bool = False,
    shared_y_limits: tuple[float, float] | None = None,
) -> None:
    plot_data = _load_visual_response_plot_data(response_row)
    lookup_cache = source_cache if isinstance(source_cache, Mapping) else cache
    if not plot_data and lookup_cache is not None:
        exp_id, observation, _, parent_dendrite_observation = _visual_response_entity_observation(lookup_cache, kind, response_row)
        if exp_id is not None and observation is not None:
            plot_data = _visual_response_entity_plot_data(source_cache or lookup_cache, kind, exp_id, {}, observation, parent_dendrite_observation)
    if not plot_data:
        print(f"[ALERT] poster visual-response traces unavailable for {response_row.get('source_path') or response_row.get('global_dendrite_id') or response_row.get('global_spine_id') or response_row.get('roi_id') or 'unknown row'}", file=sys.stderr)
        ax_blank.set_axis_off()
        ax_movie.set_axis_off()
        ax_box.set_axis_off()
        return
    cut_time = np.asarray(plot_data["cut_time"], dtype=float)
    visual_mean_trace = np.asarray(plot_data["visual_mean_trace"], dtype=float)
    blank_mean_trace = np.asarray(plot_data["blank_mean_trace"], dtype=float)
    visual_values = np.asarray(plot_data["visual_values"], dtype=float)
    blank_values = np.asarray(plot_data["blank_values"], dtype=float)
    event_onset = float(plot_data.get("event_onset", 0.0))
    event_duration = float(plot_data.get("event_duration", float("nan")))
    if cut_time.size == 0 or visual_mean_trace.size == 0 or blank_mean_trace.size == 0:
        ax_blank.set_axis_off()
        ax_movie.set_axis_off()
        ax_box.set_axis_off()
        return
    if np.isfinite(event_duration) and event_duration > event_onset:
        for ax in (ax_blank, ax_movie):
            ax.axvspan(event_onset, event_duration, color="#E5E7EB", alpha=0.45, zorder=0)
            ax.axvline(event_onset, color="#111111", linestyle="--", linewidth=1.0, alpha=0.8, zorder=4)
            ax.axvline(event_duration, color="#111111", linestyle=":", linewidth=1.0, alpha=0.8, zorder=4)
            ax.text((event_onset + event_duration) / 2.0, 0.98, "event", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")
    if panel_label:
        ax_blank.text(-0.25, 0.5, panel_label, transform=ax_blank.transAxes, rotation=90, ha="right", va="center", fontsize=FIGURE_NOTE_FS, color="#222222", fontweight="bold", clip_on=False)
    ax_blank.plot(cut_time, blank_mean_trace, color="#7F8790", linewidth=2.6, zorder=3)
    ax_blank.set_title("Blank traces", fontsize=FIGURE_TITLE_FS, pad=4)
    ax_blank.set_xlabel("Time (s)", fontsize=FIGURE_LABEL_FS)
    ax_blank.set_ylabel("dF/F", fontsize=FIGURE_LABEL_FS)
    ax_blank.grid(axis="y", alpha=0.2)
    ax_blank.text(0.02, 0.98, f"movie nonresponsive: {len(blank_values)} trials", transform=ax_blank.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")
    ax_movie.plot(cut_time, visual_mean_trace, color="#D97706", linewidth=2.6, zorder=3)
    ax_movie.set_title("Movies traces", fontsize=FIGURE_TITLE_FS, pad=4)
    ax_movie.set_xlabel("Time (s)", fontsize=FIGURE_LABEL_FS)
    ax_movie.set_ylabel("dF/F", fontsize=FIGURE_LABEL_FS)
    ax_movie.grid(axis="y", alpha=0.2)
    ax_movie.text(0.02, 0.98, f"movie responsive: {len(visual_values)} trials", transform=ax_movie.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")
    if not show_movie_ylabel:
        ax_movie.set_ylabel("")
        ax_movie.tick_params(axis="y", labelleft=False)
    data = [blank_values, visual_values]
    bp = ax_box.boxplot(data, positions=[1.0, 2.0], widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(bp.get("boxes", []), ["#7F8790", "#D97706"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
        patch.set_alpha(0.95)
    for whisker in bp.get("whiskers", []):
        whisker.set_color("#555555")
    for cap in bp.get("caps", []):
        cap.set_color("#555555")
    for median in bp.get("medians", []):
        median.set_color("#222222")
        median.set_linewidth(1.5)
    rng = np.random.default_rng(7)
    for xpos, values, color in [(1.0, blank_values, "#7F8790"), (2.0, visual_values, "#D97706")]:
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax_box.scatter(np.full(values.size, xpos) + jitter, values, s=12, alpha=0.42, color=color, edgecolor="none")
    ax_box.set_xticks([1.0, 2.0])
    ax_box.set_xticklabels(["blank", "movies"])
    ax_box.set_ylabel("Mean cut-stimulus activity", fontsize=FIGURE_LABEL_FS)
    ax_box.set_xlabel("Condition", fontsize=FIGURE_LABEL_FS)
    ax_box.set_title("Blank vs movies", fontsize=FIGURE_TITLE_FS, pad=4)
    ax_box.grid(axis="y", alpha=0.2)
    if not show_box_ylabel:
        ax_box.set_ylabel("")
        ax_box.tick_params(axis="y", labelleft=False)
    all_values = np.concatenate([blank_values, visual_values])
    finite = all_values[np.isfinite(all_values)]
    if shared_y_limits is not None and np.all(np.isfinite(shared_y_limits)):
        y_low, y_high = float(shared_y_limits[0]), float(shared_y_limits[1])
    elif finite.size:
        y_low, y_high = _combined_limits(blank_mean_trace, visual_mean_trace, blank_values, visual_values)
    else:
        y_low, y_high = float("nan"), float("nan")
    if np.isfinite(y_low) and np.isfinite(y_high):
        ax_blank.set_ylim(y_low, y_high)
        ax_movie.set_ylim(y_low, y_high)
        ax_box.set_ylim(y_low, y_high)
    ax_box.text(0.02, 0.98, f"n={int(blank_values.size)} blank, n={int(visual_values.size)} visual", transform=ax_box.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")
    p_value = None
    for key in ("shuffle_p", "adjusted_pvalue", "p_value", "raw_pvalue"):
        candidate = response_row.get(key)
        try:
            candidate_f = float(candidate) if candidate is not None else float("nan")
        except Exception:
            candidate_f = float("nan")
        if np.isfinite(candidate_f):
            p_value = candidate_f
            break
    if p_value is None:
        ttest = stats.ttest_ind(np.asarray(visual_values, dtype=float), np.asarray(blank_values, dtype=float), equal_var=False, nan_policy="omit")
        p_value = float(ttest.pvalue) if np.isfinite(ttest.pvalue) else float("nan")
    if bool(response_row.get("significant", False)) or (np.isfinite(p_value) and p_value < 0.05):
        finite = np.concatenate([blank_values, visual_values])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y = float(np.nanmax(finite)) + max(0.05 * float(np.ptp(finite)), 0.05)
            ax_box.plot([1.0, 1.0, 2.0, 2.0], [y * 0.98, y, y, y * 0.98], color="#8b0000", linewidth=1.2)
            ax_box.text(1.5, y, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS, color="#8b0000", fontweight="bold")
            ax_box.text(0.98, 0.98, "* significant comparisons", transform=ax_box.transAxes, ha="right", va="top", fontsize=FIGURE_NOTE_FS, color="#5a1a1a", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d0d0d0", alpha=0.88))
    _style_axes(ax_blank)
    _style_axes(ax_movie)
    _style_axes(ax_box)
def write_correlation_poster_figure(
    *,
    output_dir: Path | str,
    entity_label: str,
    correlation_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    output_stem: Optional[str] = None,
    title: str = "State correlation summaries",
    figure_label: Optional[str] = None,
) -> Optional[str]:
    if plt is None:
        return None
    rows = [dict(row) for row in correlation_rows if isinstance(row, Mapping)]
    if not rows:
        return None
    state_values = _state_value_map_from_rows(rows, value_key=("mean_corr", "corr", "mean"))
    if not state_values:
        return None
    state_order = [state for state in ["quiet_awake_blank", "quiet_awake_movies", "quiet_awake", "nrem", "rem", "nrem_blank", "rem_blank", "nrem_movies", "rem_movies"] if state in state_values]
    extras = sorted(state for state in state_values.keys() if state not in state_order)
    state_order.extend(extras)
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_state_summary_boxplots_correlation_poster_ready"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(MIXED_MODEL_HEIGHT_CM)), constrained_layout=False)
    ax = fig.add_subplot(111)
    significant_states = _significant_state_labels_from_comparison_rows(comparison_rows or [])
    _boxplot(
        ax,
        state_values,
        state_order,
        title=title,
        ylabel="Correlation",
        cohort_label=entity_label,
        significance_flags=[state in significant_states for state in state_order],
        sample_sizes={state: int(len(_finite_array(values))) for state, values in state_values.items()},
        box_width=0.72,
        bracket_step_scale=0.010,
        bracket_height_scale=0.0045,
        bracket_text_scale=0.002,
        extent_padding_scale=0.03,
        horizontal=True,
    )
    label = figure_label if figure_label is not None else entity_label.capitalize()
    fig.suptitle(f"{label} {title.lower()}", fontsize=FIGURE_TITLE_FS, y=0.965)
    output_path = out_dir / f"{stem}.svg"
    return _write_figure(fig, output_path)


def _write_blank_movie_and_correlation_panel(
    ax: plt.Axes,
    *,
    entity_label: str,
    correlation_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    title: str = "State correlation summaries",
) -> bool:
    rows = [dict(row) for row in correlation_rows if isinstance(row, Mapping)]
    if not rows:
        ax.set_axis_off()
        return False
    state_values = _state_value_map_from_rows(rows, value_key=("mean_corr", "corr", "mean"))
    if not state_values:
        ax.set_axis_off()
        return False
    state_order = [state for state in ["quiet_awake_blank", "quiet_awake_movies", "quiet_awake", "nrem", "rem", "nrem_blank", "rem_blank", "nrem_movies", "rem_movies"] if state in state_values]
    extras = sorted(state for state in state_values.keys() if state not in state_order)
    state_order.extend(extras)
    significant_states = _significant_state_labels_from_comparison_rows(comparison_rows or [])
    _boxplot(
        ax,
        state_values,
        state_order,
        title=title,
        ylabel="Correlation",
        cohort_label=entity_label,
        significance_flags=[state in significant_states for state in state_order],
        sample_sizes={state: int(len(_finite_array(values))) for state, values in state_values.items()},
        box_width=0.72,
        bracket_step_scale=0.007,
        bracket_height_scale=0.003,
        bracket_text_scale=0.0012,
        extent_padding_scale=0.02,
        horizontal=True,
    )
    return True


def write_blank_movie_and_correlation_poster_figure(
    *,
    output_dir: Path | str,
    entity_label: str,
    responsive_blank_values: Mapping[str, Sequence[float]],
    responsive_movie_values: Mapping[str, Sequence[float]],
    nonresponsive_blank_values: Mapping[str, Sequence[float]],
    nonresponsive_movie_values: Mapping[str, Sequence[float]],
    blank_state_order: Sequence[str],
    movie_state_order: Sequence[str],
    responsive_blank_sample_sizes: Mapping[str, int] | None = None,
    responsive_movie_sample_sizes: Mapping[str, int] | None = None,
    nonresponsive_blank_sample_sizes: Mapping[str, int] | None = None,
    nonresponsive_movie_sample_sizes: Mapping[str, int] | None = None,
    responsive_significant_states: Sequence[str] | None = None,
    nonresponsive_significant_states: Sequence[str] | None = None,
    responsive_comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    nonresponsive_comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    responsive_compartment_significant_states: Sequence[str] | None = None,
    nonresponsive_compartment_significant_states: Sequence[str] | None = None,
    correlation_rows: Sequence[Mapping[str, Any]] | None = None,
    correlation_comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    output_stem: Optional[str] = None,
    title: str = "Blank vs movie states",
    correlation_title: str = "State correlation summaries",
) -> Optional[str]:
    if plt is None:
        return None
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_blank_movie_states_with_correlation_poster_ready"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(8.0)), constrained_layout=False)
    outer = fig.add_gridspec(3, 2, left=0.07, right=0.985, top=0.93, bottom=0.11, wspace=0.16, hspace=0.28, height_ratios=[1.0, 1.0, 0.88])
    ax_resp_blank = fig.add_subplot(outer[0, 0])
    ax_nonresp_blank = fig.add_subplot(outer[0, 1], sharey=ax_resp_blank)
    ax_resp_movie = fig.add_subplot(outer[1, 0])
    ax_nonresp_movie = fig.add_subplot(outer[1, 1], sharey=ax_resp_movie)
    ax_corr = fig.add_subplot(outer[2, :])
    responsive_sig = {str(state) for state in (responsive_significant_states or [])}
    nonresponsive_sig = {str(state) for state in (nonresponsive_significant_states or [])}
    responsive_compartment_sig = {str(state) for state in (responsive_compartment_significant_states or [])}
    nonresponsive_compartment_sig = {str(state) for state in (nonresponsive_compartment_significant_states or [])}
    if responsive_comparison_rows:
        responsive_sig.update(_significant_state_labels_from_comparison_rows(responsive_comparison_rows))
    if nonresponsive_comparison_rows:
        nonresponsive_sig.update(_significant_state_labels_from_comparison_rows(nonresponsive_comparison_rows))
    blank_state_order = list(dict.fromkeys([str(state) for state in blank_state_order] + [str(state) for state in responsive_blank_values.keys()] + [str(state) for state in nonresponsive_blank_values.keys()]))
    movie_state_order = list(dict.fromkeys([str(state) for state in movie_state_order] + [str(state) for state in responsive_movie_values.keys()] + [str(state) for state in nonresponsive_movie_values.keys()]))
    print(f"[poster] {entity_label} blank/movie source states: blank={blank_state_order}; movie={movie_state_order}", file=sys.stderr)
    dendrite_grouped = str(entity_label).strip().lower() == "dendrite" or any(str(key).startswith(("basal_", "apical_")) for key in list(responsive_blank_values) + list(responsive_movie_values) + list(nonresponsive_blank_values) + list(nonresponsive_movie_values))

    def _split_compartment_values(values: Mapping[str, Sequence[float]]) -> tuple[dict[str, Sequence[float]], dict[str, Sequence[float]], dict[str, Sequence[float]]]:
        basal: dict[str, Sequence[float]] = {}
        apical: dict[str, Sequence[float]] = {}
        other: dict[str, Sequence[float]] = {}
        for key, arr in values.items():
            state_key = _canonical_state_key(key)
            if state_key.startswith("basal_"):
                basal[state_key[len("basal_"):]] = arr
            elif state_key.startswith("apical_"):
                apical[state_key[len("apical_"):]] = arr
            else:
                other[state_key] = arr
        return basal, apical, other

    def _project_dendrite_panel_values(values: Mapping[str, Sequence[float]], panel_kind: str) -> dict[str, list[float]]:
        projected: dict[str, list[float]] = {}
        if panel_kind not in {"blank", "movie"}:
            return {str(key): list(_finite_array(arr)) for key, arr in values.items()}
        exact_suffix = {
            "blank": {"quiet_awake_blank", "nrem_blank", "rem_blank"},
            "movie": {"quiet_awake_movies", "nrem_movies", "rem_movies"},
        }[panel_kind]
        fallback_suffix = {
            "blank": {"quiet_awake": "quiet_awake_blank", "nrem": "nrem_blank", "rem": "rem_blank"},
            "movie": {"quiet_awake": "quiet_awake_movies", "nrem": "nrem_movies", "rem": "rem_movies"},
        }[panel_kind]
        for key, arr in values.items():
            state_key = _canonical_state_key(key)
            if state_key.startswith("basal_") or state_key.startswith("apical_"):
                state_key = state_key.split("_", 1)[1]
            if state_key in exact_suffix:
                projected.setdefault(state_key, []).extend(_finite_array(arr).tolist())
            elif state_key in fallback_suffix:
                projected.setdefault(fallback_suffix[state_key], []).extend(_finite_array(arr).tolist())
        return projected

    def _grouped_panel(
        ax: plt.Axes,
        basal_values: Mapping[str, Sequence[float]],
        apical_values: Mapping[str, Sequence[float]],
        state_order: Sequence[str],
        *,
        title_text: str,
        cohort_label: str,
        significant_states: Sequence[str],
        compartment_significant_states: Sequence[str],
        sample_sizes_basal: Mapping[str, int] | None = None,
        sample_sizes_apical: Mapping[str, int] | None = None,
    ) -> None:
        rng = np.random.default_rng(7)
        basal_state_map = {state: _finite_array(basal_values.get(state, [])) for state in state_order}
        apical_state_map = {state: _finite_array(apical_values.get(state, [])) for state in state_order}
        all_data: list[np.ndarray] = []
        for compartment, summary_map, color, offset in (("basal", basal_state_map, BOX_COLORS.get("basal", "#4C72B0"), -0.18), ("apical", apical_state_map, BOX_COLORS.get("apical", "#DD8452"), 0.18)):
            positions: list[float] = []
            data: list[np.ndarray] = []
            for idx, state in enumerate(state_order, start=1):
                arr = summary_map.get(state, np.asarray([], dtype=float))
                if arr.size:
                    positions.append(float(idx) + offset)
                    data.append(arr)
            if not data:
                continue
            bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False, vert=False)
            _set_boxplot_colors(bp, [color] * len(data))
            for pos, arr in zip(positions, data):
                jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                ax.scatter(arr, np.full(arr.size, pos) + jitter, s=14, alpha=0.48, color=color, edgecolor="none")
            all_data.extend(data)
        ax.set_yticks(range(1, len(state_order) + 1))
        ax.set_yticklabels([_state_display_label(state) for state in state_order], fontsize=FIGURE_TICK_FS)
        ax.set_xlabel("Mean response", fontsize=FIGURE_LABEL_FS)
        ax.set_ylabel("State", fontsize=FIGURE_LABEL_FS)
        ax.set_title(title_text, fontsize=FIGURE_TITLE_FS, pad=4)
        ax.grid(axis="x", alpha=0.25)
        finite = np.concatenate(all_data) if all_data else np.asarray([], dtype=float)
        finite = finite[np.isfinite(finite)] if finite.size else finite
        if finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            pad = max(0.12 * max(hi - lo, 1e-6), 0.05)
            ax.set_xlim(lo - pad, hi + pad)
            x0, x1 = ax.get_xlim()
            x_range = max(x1 - x0, 1e-6)
            count_x = x1 + max(0.02 * x_range, 0.05)
            bracket_x = x1 + max(0.045 * x_range, 0.08)
            bracket_dx = max(0.02 * x_range, 0.035)
            bracket_step = max(0.028 * x_range, 0.05)
            sig_states = [state for state in state_order if _state_compare_key(state) in {_state_compare_key(s) for s in significant_states}]
            comp_sig_states = [state for state in state_order if _state_compare_key(state) in {_state_compare_key(s) for s in compartment_significant_states}]
            for level, state in enumerate(comp_sig_states):
                center = float(state_order.index(state) + 1)
                y_low = center - 0.18
                y_high = center + 0.18
                x = bracket_x + level * bracket_step
                ax.plot([x, x + bracket_dx, x + bracket_dx, x], [y_low, y_low, y_high, y_high], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
                ax.text(x + bracket_dx * 0.5, y_high + 0.06, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS + 2, color="#8b0000", fontweight="bold", clip_on=False)
            for compartment, summary_map, color, offset in (("basal", basal_state_map, BOX_COLORS.get("basal", "#4C72B0"), -0.18), ("apical", apical_state_map, BOX_COLORS.get("apical", "#DD8452"), 0.18)):
                sample_map = sample_sizes_basal if compartment == "basal" else sample_sizes_apical
                for idx, state in enumerate(state_order, start=1):
                    arr = summary_map.get(state, np.asarray([], dtype=float))
                    if not arr.size:
                        continue
                    sample_n = int(sample_map.get(state, int(arr.size))) if sample_map is not None else int(arr.size)
                    ax.text(count_x, float(idx) + offset, f"n={sample_n}", color=color, fontsize=FIGURE_NOTE_FS - 1, ha="left", va="center", clip_on=False)
            for level, state in enumerate(sig_states):
                y = float(state_order.index(state) + 1)
                x = bracket_x + level * bracket_step
                ax.plot([x, x + bracket_dx, x + bracket_dx, x], [y, y, y + 0.35, y + 0.35], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
                ax.text(x + bracket_dx * 0.5, y + 0.35, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS + 2, color="#8b0000", fontweight="bold", clip_on=False)
        legend_handles = [
            Line2D([0], [0], color=BOX_COLORS.get("basal", "#4C72B0"), marker="s", linestyle="", markersize=8, label="Basal"),
            Line2D([0], [0], color=BOX_COLORS.get("apical", "#DD8452"), marker="s", linestyle="", markersize=8, label="Apical"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FIGURE_LEGEND_FS)
        ax.text(0.02, 0.98, cohort_label, transform=ax.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")

    if dendrite_grouped:
        resp_basal_blank, resp_apical_blank, resp_other_blank = _split_compartment_values(responsive_blank_values)
        resp_basal_movie, resp_apical_movie, resp_other_movie = _split_compartment_values(responsive_movie_values)
        nonresp_basal_blank, nonresp_apical_blank, nonresp_other_blank = _split_compartment_values(nonresponsive_blank_values)
        nonresp_basal_movie, nonresp_apical_movie, nonresp_other_movie = _split_compartment_values(nonresponsive_movie_values)
        blank_state_order = ["quiet_awake_blank", "nrem_blank"] if "rem_blank" not in blank_state_order else ["quiet_awake_blank", "nrem_blank", "rem_blank"]
        movie_state_order = ["quiet_awake_movies", "nrem_movies"] if "rem_movies" not in movie_state_order else ["quiet_awake_movies", "nrem_movies", "rem_movies"]
        _grouped_panel(
            ax_resp_blank,
            _project_dendrite_panel_values({**resp_basal_blank, **resp_other_blank}, "blank"),
            _project_dendrite_panel_values({**resp_apical_blank, **resp_other_blank}, "blank"),
            blank_state_order,
            title_text="responsive blank states",
            cohort_label="responsive",
            significant_states=[state for state in responsive_sig],
            compartment_significant_states=[state for state in responsive_compartment_sig],
            sample_sizes_basal=responsive_blank_sample_sizes,
            sample_sizes_apical=responsive_blank_sample_sizes,
        )
        _grouped_panel(
            ax_nonresp_blank,
            _project_dendrite_panel_values({**nonresp_basal_blank, **nonresp_other_blank}, "blank"),
            _project_dendrite_panel_values({**nonresp_apical_blank, **nonresp_other_blank}, "blank"),
            blank_state_order,
            title_text="nonresponsive blank states",
            cohort_label="nonresponsive",
            significant_states=[state for state in nonresponsive_sig],
            compartment_significant_states=[state for state in nonresponsive_compartment_sig],
            sample_sizes_basal=nonresponsive_blank_sample_sizes,
            sample_sizes_apical=nonresponsive_blank_sample_sizes,
        )
        _grouped_panel(
            ax_resp_movie,
            _project_dendrite_panel_values({**resp_basal_movie, **resp_other_movie}, "movie"),
            _project_dendrite_panel_values({**resp_apical_movie, **resp_other_movie}, "movie"),
            movie_state_order,
            title_text="responsive movie states",
            cohort_label="responsive",
            significant_states=[state for state in responsive_sig],
            compartment_significant_states=[state for state in responsive_compartment_sig],
            sample_sizes_basal=responsive_movie_sample_sizes,
            sample_sizes_apical=responsive_movie_sample_sizes,
        )
        _grouped_panel(
            ax_nonresp_movie,
            _project_dendrite_panel_values({**nonresp_basal_movie, **nonresp_other_movie}, "movie"),
            _project_dendrite_panel_values({**nonresp_apical_movie, **nonresp_other_movie}, "movie"),
            movie_state_order,
            title_text="nonresponsive movie states",
            cohort_label="nonresponsive",
            significant_states=[state for state in nonresponsive_sig],
            compartment_significant_states=[state for state in nonresponsive_compartment_sig],
            sample_sizes_basal=nonresponsive_movie_sample_sizes,
            sample_sizes_apical=nonresponsive_movie_sample_sizes,
        )
    else:
        _boxplot(
            ax_resp_blank,
            responsive_blank_values,
            blank_state_order,
            title="responsive blank states",
            ylabel="Mean response",
            cohort_label="responsive",
            significance_flags=[state in responsive_sig for state in blank_state_order],
            sample_sizes=responsive_blank_sample_sizes or {state: int(len(_finite_array(responsive_blank_values.get(state, [])))) for state in blank_state_order},
            horizontal=True,
        )
        _boxplot(
            ax_nonresp_blank,
            nonresponsive_blank_values,
            blank_state_order,
            title="nonresponsive blank states",
            ylabel="Mean response",
            cohort_label="nonresponsive",
            significance_flags=[state in nonresponsive_sig for state in blank_state_order],
            sample_sizes=nonresponsive_blank_sample_sizes,
            horizontal=True,
        )
        _boxplot(
            ax_resp_movie,
            responsive_movie_values,
            movie_state_order,
            title="responsive movie states",
            ylabel="Mean response",
            cohort_label="responsive",
            significance_flags=[state in responsive_sig for state in movie_state_order],
            sample_sizes=responsive_movie_sample_sizes,
            horizontal=True,
        )
        _boxplot(
            ax_nonresp_movie,
            nonresponsive_movie_values,
            movie_state_order,
            title="nonresponsive movie states",
            ylabel="Mean response",
            cohort_label="nonresponsive",
            significance_flags=[state in nonresponsive_sig for state in movie_state_order],
            sample_sizes=nonresponsive_movie_sample_sizes,
            horizontal=True,
        )
    for ax in (ax_nonresp_blank, ax_nonresp_movie):
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")
    for ax in (ax_resp_blank, ax_nonresp_blank):
        ax.tick_params(axis="x", labelbottom=False)

    corr_rows = [dict(row) for row in (correlation_rows or []) if isinstance(row, Mapping)]
    if corr_rows:
        corr_state_values = _state_value_map_from_rows(corr_rows, value_key=("mean_corr", "corr", "mean"))
        if corr_state_values:
            corr_state_order = [state for state in ["quiet_awake_blank", "nrem_blank", "quiet_awake_movies", "nrem_movies", "quiet_awake", "nrem"] if state in corr_state_values]
            corr_extras = sorted(state for state in corr_state_values.keys() if state not in corr_state_order)
            corr_state_order.extend(corr_extras)
            corr_sig = _significant_state_labels_from_comparison_rows(correlation_comparison_rows or [])
            _boxplot(
                ax_corr,
                corr_state_values,
                corr_state_order,
                title=correlation_title,
                ylabel="Correlation",
                cohort_label=entity_label,
                significance_flags=[state in corr_sig for state in corr_state_order],
                sample_sizes={state: int(len(_finite_array(values))) for state, values in corr_state_values.items()},
                box_width=0.72,
                bracket_step_scale=0.004,
                bracket_height_scale=0.002,
                bracket_text_scale=0.0008,
                extent_padding_scale=0.02,
                horizontal_extent_padding_scale=0.008,
                horizontal=True,
            )
        else:
            ax_corr.set_axis_off()
    else:
        ax_corr.set_axis_off()

    output_path = out_dir / f"{stem}.svg"
    fig.savefig(output_path, format="svg", dpi=300)
    if plt is not None:
        plt.close(fig)
    return str(output_path)


def _select_mixed_model_rows(mixed_model_branch: Any, preferred_response_keys: Sequence[str] | None = None) -> list[dict[str, Any]]:
    if isinstance(mixed_model_branch, Mapping):
        summary_rows = mixed_model_branch.get("summary_rows")
        if isinstance(summary_rows, Mapping):
            if preferred_response_keys is None:
                preferred_keys = ("mean_activity", "mean_dendrite_activity", "mean_spine_activity_per_dendrite", "mean")
            else:
                preferred_keys = tuple(dict.fromkeys(("mean_activity",) + tuple(str(key) for key in preferred_response_keys)))
            for preferred in preferred_keys:
                rows = summary_rows.get(preferred)
                if isinstance(rows, list) and rows:
                    return [dict(row) for row in rows if isinstance(row, Mapping)]
            for rows in summary_rows.values():
                if isinstance(rows, list) and rows:
                    return [dict(row) for row in rows if isinstance(row, Mapping)]
        if isinstance(mixed_model_branch.get("summary_rows"), list):
            return [dict(row) for row in mixed_model_branch.get("summary_rows", []) if isinstance(row, Mapping)]
    return []


def _write_figure(fig: plt.Figure, output_path: Path) -> str:
    save_figure(fig, output_path, extra_formats=())
    return str(output_path)


def _poster_mixed_model_significant_states(rows: Sequence[Mapping[str, Any]], *, p_value_key: str = "p_value") -> set[str]:
    significant: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        term = str(row.get("term") or "")
        if not term.startswith("state["):
            continue
        p_value = row.get(p_value_key)
        try:
            p_value_f = float(p_value) if p_value is not None else float("nan")
        except Exception:
            p_value_f = float("nan")
        if not np.isfinite(p_value_f) or p_value_f >= 0.05:
            continue
        label = term[len("state[") :]
        if label.endswith("]"):
            label = label[:-1]
        if ":" in label:
            label = label.split(":", 1)[0]
        significant.add(label)
    return significant


def _significant_state_labels_from_comparison_rows(rows: Sequence[Mapping[str, Any]], *, p_value_keys: Sequence[str] = ("shuffle_p", "p_value", "adjusted_pvalue", "raw_pvalue")) -> set[str]:
    significant: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        p_value_f = float("nan")
        for key in p_value_keys:
            candidate = row.get(key)
            try:
                candidate_f = float(candidate) if candidate is not None else float("nan")
            except Exception:
                candidate_f = float("nan")
            if np.isfinite(candidate_f):
                p_value_f = candidate_f
                break
        if not np.isfinite(p_value_f) or p_value_f >= 0.05:
            continue
        for key in ("state", "state_label", "state_display", "state_a", "state_a_display", "state_b", "state_b_display"):
            state = _canonical_state_key(row.get(key))
            if state:
                significant.add(state)
    return significant



def _compartment_comparison_state_labels_from_comparison_rows(rows: Sequence[Mapping[str, Any]], *, p_value_keys: Sequence[str] = ("shuffle_p", "p_value", "adjusted_pvalue", "raw_pvalue")) -> set[str]:
    significant: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        comparison = _canonical_state_key(row.get("comparison") or row.get("contrast_type") or row.get("contrast_name") or "")
        if comparison not in {"basal_vs_apical", "apical_vs_basal"}:
            continue
        p_value_f = float("nan")
        for key in p_value_keys:
            candidate = row.get(key)
            try:
                candidate_f = float(candidate) if candidate is not None else float("nan")
            except Exception:
                candidate_f = float("nan")
            if np.isfinite(candidate_f):
                p_value_f = candidate_f
                break
        if not np.isfinite(p_value_f) or p_value_f >= 0.05:
            continue
        state = _canonical_state_key(row.get("state") or row.get("state_label") or row.get("state_display") or "")
        if state:
            significant.add(state)
            continue
        for key in ("state_a", "state_b"):
            candidate = _canonical_state_key(row.get(key))
            if not candidate:
                continue
            for prefix in ("basal_", "apical_"):
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix):]
                    significant.add(candidate)
                    break
    return significant


def _poster_label_color(label: Any) -> str:
    key = _canonical_state_key(label)
    if key in {"basal", "apical"}:
        return BOX_COLORS.get(key, "#444444")
    if key.startswith("basal_"):
        return BOX_COLORS.get("basal", "#4C72B0")
    if key.startswith("apical_"):
        return BOX_COLORS.get("apical", "#DD8452")
    if key.startswith("quiet"):
        return BOX_COLORS.get("quiet_awake", "#ff7f0e")
    if key.startswith("nrem"):
        return BOX_COLORS.get("nrem", "#2ca02c")
    if key.startswith("rem"):
        return BOX_COLORS.get("rem", "#d62728")
    if key.startswith("active"):
        return "#1f77b4"
    return BOX_COLORS.get(key, "#444444")


def write_visual_response_poster_figure(
    *,
    output_dir: Path | str,
    entity_label: str,
    visual_response_rows: Sequence[Mapping[str, Any]],
    cache: Optional[Mapping[str, Any]] = None,
    source_cache: Optional[Mapping[str, Any]] = None,
    kind: str | None = None,
    output_stem: Optional[str] = None,
    locomotion_threshold: float | None = None,
) -> Optional[str]:
    if plt is None:
        return None
    rows = _canonicalize_visual_response_rows([dict(row) for row in visual_response_rows if isinstance(row, Mapping)])
    if not rows:
        return None
    responsive_row, nonresponsive_row = _pick_exemplar_rows(rows)
    if responsive_row is None or nonresponsive_row is None:
        return None
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_visual_response_poster_ready"
    fig = plt.figure(figsize=(cm_to_inch(VISUAL_RESPONSE_WIDTH_CM), cm_to_inch(VISUAL_RESPONSE_HEIGHT_CM)), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.9, 0.9], height_ratios=[1, 1], left=0.055, right=0.985, top=0.92, bottom=0.20, wspace=0.18, hspace=0.42)
    resp_grid = outer[0, 0].subgridspec(1, 3, width_ratios=[1.08, 1.08, 0.92], wspace=0.18)
    nonresp_grid = outer[1, 0].subgridspec(1, 3, width_ratios=[1.08, 1.08, 0.92], wspace=0.18)
    ax_resp_blank = fig.add_subplot(resp_grid[0, 0])
    ax_resp_movie = fig.add_subplot(resp_grid[0, 1], sharey=ax_resp_blank)
    ax_resp_box = fig.add_subplot(resp_grid[0, 2], sharey=ax_resp_blank)
    ax_nonresp_blank = fig.add_subplot(nonresp_grid[0, 0])
    ax_nonresp_movie = fig.add_subplot(nonresp_grid[0, 1], sharey=ax_nonresp_blank)
    ax_nonresp_box = fig.add_subplot(nonresp_grid[0, 2], sharey=ax_nonresp_blank)
    ax_pie = fig.add_subplot(outer[:, 1])
    poster_kind = str(kind or entity_label or "dendrite")
    responsive_plot_data = _load_visual_response_plot_data(responsive_row)
    nonresponsive_plot_data = _load_visual_response_plot_data(nonresponsive_row)
    shared_y_limits = _visual_response_poster_shared_limits(responsive_plot_data, nonresponsive_plot_data)
    _single_exemplar_panel(ax_resp_blank, ax_resp_movie, ax_resp_box, responsive_row, cache=cache, source_cache=source_cache, kind=poster_kind, title="responsive example", panel_label="responsive example", show_movie_ylabel=False, show_box_ylabel=False, shared_y_limits=shared_y_limits)
    _single_exemplar_panel(ax_nonresp_blank, ax_nonresp_movie, ax_nonresp_box, nonresponsive_row, cache=cache, source_cache=source_cache, kind=poster_kind, title="nonresponsive example", panel_label="nonresponsive example", show_movie_ylabel=False, show_box_ylabel=False, shared_y_limits=shared_y_limits)
    fig.text(0.017, 0.735, "responsive example", rotation=90, ha="right", va="center", fontsize=FIGURE_NOTE_FS, color="#222222", fontweight="bold")
    fig.text(0.017, 0.335, "nonresponsive example", rotation=90, ha="right", va="center", fontsize=FIGURE_NOTE_FS, color="#222222", fontweight="bold")
    for ax in (ax_resp_movie, ax_resp_box, ax_nonresp_movie, ax_nonresp_box):
        ax.tick_params(axis="y", labelleft=False)
    responsive_rows = [row for row in rows if _coerce_bool(row.get("responsive", False))]
    nonresponsive_rows = [row for row in rows if not _coerce_bool(row.get("responsive", False))]
    counts = np.asarray([
        _visual_response_unique_entity_count(responsive_rows),
        _visual_response_unique_entity_count(nonresponsive_rows),
    ], dtype=float)
    labels = ["movie responsive", "movie nonresponsive"]
    colors = [RESPONSIVE_COLOR, NONRESPONSIVE_COLOR]
    total = float(np.sum(counts)) if np.sum(counts) > 0 else 1.0
    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        textprops={"fontsize": FIGURE_NOTE_FS},
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
    )
    for text in texts + autotexts:
        text.set_fontsize(FIGURE_NOTE_FS)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RESPONSIVE_COLOR, markeredgecolor="white", markersize=8, label="movie responsive"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NONRESPONSIVE_COLOR, markeredgecolor="white", markersize=8, label="movie nonresponsive"),
    ]
    ax_pie.legend(handles=legend_handles, frameon=False, fontsize=FIGURE_NOTE_FS, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    ax_pie.set_title(f"{entity_label.capitalize()} visual responsiveness", fontsize=FIGURE_TITLE_FS, pad=8)
    ax_pie.text(0.5, -0.08, f"n={int(total)}", transform=ax_pie.transAxes, ha="center", va="top", fontsize=FIGURE_NOTE_FS)
    fig.suptitle(f"{entity_label.capitalize()} visual response", fontsize=FIGURE_TITLE_FS, y=0.965)
    output_path = out_dir / f"{stem}.svg"
    return _write_figure(fig, output_path)
def write_state_mixed_model_poster_figure(
    *,
    output_dir: Path | str,
    entity_label: str,
    responsive_state_values: Mapping[str, Sequence[float]],
    nonresponsive_state_values: Mapping[str, Sequence[float]],
    mixed_model_rows: Sequence[Mapping[str, Any]] | Any,
    state_order: Sequence[str],
    output_stem: Optional[str] = None,
    title: str = "Quiet blank vs sleep states",
    preferred_response_keys: Sequence[str] | None = None,
    mixed_model_contrast_p_source: str = "classical",
    responsive_state_sample_sizes: Mapping[str, int] | None = None,
    nonresponsive_state_sample_sizes: Mapping[str, int] | None = None,
) -> Optional[str]:
    if plt is None:
        return None

    def _state_series_map(values: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        return {str(key): [float(value) for value in series if value is not None] for key, series in values.items() if isinstance(series, Sequence)}

    def _split_dendrite_state_values(values: Mapping[str, Sequence[float]]) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        basal: dict[str, list[float]] = {}
        apical: dict[str, list[float]] = {}
        other: dict[str, list[float]] = {}
        for state, series in values.items():
            key = str(state)
            arr = [float(value) for value in series if value is not None]
            if key.startswith("basal_"):
                basal[key[len("basal_"):]] = arr
            elif key.startswith("apical_"):
                apical[key[len("apical_"):]] = arr
            else:
                other[key] = arr
        return basal, apical, other

    def _grouped_compartment_boxplot(
        ax: plt.Axes,
        *,
        basal_values: Mapping[str, Sequence[float]],
        apical_values: Mapping[str, Sequence[float]],
        state_order: Sequence[str],
        title: str,
        ylabel: str,
        cohort_label: str,
        significant_states: Sequence[str] | None = None,
    ) -> None:
        rng = np.random.default_rng(7)
        basal_state_map = {state: _finite_array(basal_values.get(state, [])) for state in state_order}
        apical_state_map = {state: _finite_array(apical_values.get(state, [])) for state in state_order}
        all_data: list[np.ndarray] = []
        compartment_specs = [
            ("basal", basal_state_map, "#4C72B0", -0.18),
            ("apical", apical_state_map, "#DD8452", 0.18),
        ]
        for compartment, summary_map, color, offset in compartment_specs:
            positions: list[float] = []
            data: list[np.ndarray] = []
            for idx, state in enumerate(state_order, start=1):
                arr = summary_map.get(state, np.asarray([], dtype=float))
                if arr.size:
                    positions.append(float(idx) + offset)
                    data.append(arr)
            if not data:
                continue
            bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False, vert=False)
            _set_boxplot_colors(bp, [color] * len(data))
            for pos, arr in zip(positions, data):
                jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                ax.scatter(arr, np.full(arr.size, pos) + jitter, s=14, alpha=0.48, color=color, edgecolor="none")
            all_data.extend(data)

        state_labels = [_state_display_label(state) for state in state_order]
        ax.set_yticks(range(1, len(state_order) + 1))
        ax.set_yticklabels(state_labels, fontsize=FIGURE_TICK_FS)
        ax.set_xlabel(ylabel, fontsize=FIGURE_LABEL_FS)
        ax.set_ylabel("State", fontsize=FIGURE_LABEL_FS)
        ax.set_title(title, fontsize=FIGURE_TITLE_FS, pad=4)
        ax.grid(axis="x", alpha=0.25)

        finite = np.concatenate(all_data) if all_data else np.asarray([], dtype=float)
        finite = finite[np.isfinite(finite)] if finite.size else finite
        if finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            pad = max(0.12 * max(hi - lo, 1e-6), 0.05)
            ax.set_xlim(lo - pad, hi + pad)
            x0, x1 = ax.get_xlim()
            x_range = max(x1 - x0, 1e-6)
            count_x = x1 + max(0.02 * x_range, 0.05)
            bracket_x = x1 + max(0.045 * x_range, 0.08)
            bracket_dx = max(0.02 * x_range, 0.035)
            bracket_step = max(0.028 * x_range, 0.05)
            significant = {_state_compare_key(state) for state in (significant_states or [])}
            sig_states = [state for state in state_order if _state_compare_key(state) in significant]
            for compartment, summary_map, color, offset in compartment_specs:
                for idx, state in enumerate(state_order, start=1):
                    arr = summary_map.get(state, np.asarray([], dtype=float))
                    if not arr.size:
                        continue
                    ax.text(count_x, float(idx) + offset, f"n={int(arr.size)}", color=color, fontsize=FIGURE_NOTE_FS - 1, ha="left", va="center", clip_on=False)
            for level, state in enumerate(sig_states):
                y = float(state_order.index(state) + 1)
                x = bracket_x + level * bracket_step
                ax.plot([x, x + bracket_dx, x + bracket_dx, x], [y, y, y + 0.35, y + 0.35], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
                ax.text(x + bracket_dx * 0.5, y + 0.35, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS + 2, color="#8b0000", fontweight="bold", clip_on=False)

        legend_handles = [
            Line2D([0], [0], color="#4C72B0", marker="s", linestyle="", markersize=8, label="Basal"),
            Line2D([0], [0], color="#DD8452", marker="s", linestyle="", markersize=8, label="Apical"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FIGURE_LEGEND_FS)
        ax.text(0.02, 0.98, cohort_label, transform=ax.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")

    base_states = ["quiet_awake_blank", "quiet_awake_movies", "quiet_awake", "nrem", "rem"]
    resp = _state_series_map(responsive_state_values)
    nonresp = _state_series_map(nonresponsive_state_values)
    if not resp and not nonresp:
        return None

    dendrite_prefixed = any(str(state).startswith(("basal_", "apical_")) for state in state_order) or any(str(k).startswith(("basal_", "apical_")) for k in resp) or any(str(k).startswith(("basal_", "apical_")) for k in nonresp)
    forest_color_fn: Any = None
    if dendrite_prefixed:
        def _dendrite_forest_color(row: Mapping[str, Any], label: str, kind: str) -> str:
            label_l = str(label or "").lower()
            if "apical" in label_l:
                return BOX_COLORS.get("apical", "#DD8452")
            if "basal" in label_l:
                return BOX_COLORS.get("basal", "#4C72B0")
            if kind == "state":
                return "#1f77b4"
            if kind == "compartment":
                return "#1f77b4"
            if kind == "interaction":
                return BOX_COLORS.get("apical", "#DD8452")
            if kind == "covariate":
                return "#1f77b4"
            return {
                "intercept": "#7f7f7f",
            }.get(kind, "#1f77b4")
        forest_color_fn = _dendrite_forest_color
    if dendrite_prefixed:
        resp_basal, resp_apical, resp_other = _split_dendrite_state_values(resp)
        nonresp_basal, nonresp_apical, nonresp_other = _split_dendrite_state_values(nonresp)
        candidate_order: list[str] = []
        for candidate in list(state_order) + list(resp.keys()) + list(nonresp.keys()):
            base = str(candidate)
            for prefix in ("basal_", "apical_"):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            if not base:
                continue
            if base in candidate_order:
                continue
            candidate_order.append(base)
        if not candidate_order:
            candidate_order = list(dict.fromkeys([_state_compare_key(state) for state in state_order]))
        present_state_order = list(dict.fromkeys(candidate_order))
    else:
        present_state_order = list(dict.fromkeys([str(state) for state in state_order if str(state).strip()]))
        if not present_state_order:
            present_state_order = list(dict.fromkeys([state for state in list(resp.keys()) + list(nonresp.keys()) if str(state).strip()]))
        if not present_state_order:
            present_state_order = list(base_states)

    forest_rows_by_cohort: dict[str, list[dict[str, Any]]] = {}
    if isinstance(mixed_model_rows, Mapping) and ("responsive" in mixed_model_rows or "nonresponsive" in mixed_model_rows):
        rows_by_cohort = {}
        for cohort, branch in mixed_model_rows.items():
            if str(cohort) not in {"responsive", "nonresponsive"} or not isinstance(branch, Mapping):
                continue
            selected_rows = _select_mixed_model_rows(branch, preferred_response_keys=preferred_response_keys)
            filtered_rows = filter_mixed_model_terms_to_states(selected_rows, present_state_order)
            rows_by_cohort[str(cohort)] = filtered_rows
            forest_rows_by_cohort[str(cohort)] = filtered_rows
    else:
        selected_rows = _select_mixed_model_rows(mixed_model_rows, preferred_response_keys=preferred_response_keys)
        filtered_rows = filter_mixed_model_terms_to_states(selected_rows, present_state_order)
        rows_by_cohort = {"responsive": filtered_rows, "nonresponsive": filtered_rows}
        forest_rows_by_cohort = {"responsive": filtered_rows, "nonresponsive": filtered_rows}

    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_state_mixed_model_poster_ready"
    p_source = _normalize_mixed_model_contrast_p_source(mixed_model_contrast_p_source)
    p_source_label = "shuffle p" if p_source == "shuffle" else "classical p"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(MIXED_MODEL_HEIGHT_CM)), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.92, bottom=0.11, wspace=0.24, hspace=0.24, height_ratios=[0.95, 1.05])
    print(f"[poster] {entity_label} selected boxplot states = {list(present_state_order)}", file=sys.stderr)
    if dendrite_prefixed:
        resp_basal, resp_apical, _ = _split_dendrite_state_values(resp)
        nonresp_basal, nonresp_apical, _ = _split_dendrite_state_values(nonresp)
        boxplot_state_order = list(reversed(present_state_order))

        cohort_boxplot_data = [
            ("responsive", resp_basal, resp_apical, 0),
            ("nonresponsive", nonresp_basal, nonresp_apical, 1),
        ]
        for cohort_label, basal_values, apical_values, col_index in cohort_boxplot_data:
            ax_box = fig.add_subplot(outer[0, col_index])
            ax_forest = fig.add_subplot(outer[1, col_index])
            box_rows = rows_by_cohort.get(cohort_label, [])
            forest_rows = forest_rows_by_cohort.get(cohort_label, []) or box_rows
            if not box_rows:
                box_rows = rows_by_cohort.get("responsive", []) or rows_by_cohort.get("nonresponsive", [])
            if not forest_rows:
                forest_rows = box_rows
            forest_labels = {_state_compare_key(_forest_row_label(row)) for row in forest_rows if _forest_row_label(row)}
            box_labels = {_state_compare_key(_forest_row_label(row)) for row in box_rows if _forest_row_label(row)}
            state_forest_labels = {label for label in forest_labels if label}
            state_box_labels = {label for label in box_labels if label}
            if state_forest_labels and not state_forest_labels.issubset(state_box_labels):
                print(f"[ALERT] poster forest labels extend beyond boxplot labels for {entity_label} {cohort_label}: {sorted(state_forest_labels - state_box_labels)}", file=sys.stderr)
            raw_terms = [str(row.get('term') or row.get('contrast_name') or '') for row in forest_rows]
            forest_main_effect_states = sorted({_state_compare_key(_forest_row_label(row)) for row in forest_rows if str(row.get('term') or '').startswith('state[')})
            forest_interaction_states = sorted({_state_compare_key(_forest_row_label(row)) for row in forest_rows if _poster_mixed_model_term_kind(str(row.get('term') or '')) == 'interaction'})
            dropped_terms = [term for term in raw_terms if term not in {str(row.get("term") or "") for row in forest_rows}]
            print(f"[poster] {entity_label} {cohort_label} forest raw terms = {raw_terms}", file=sys.stderr)
            print(f"[poster] {entity_label} {cohort_label} forest main-effect states = {forest_main_effect_states}", file=sys.stderr)
            print(f"[poster] {entity_label} {cohort_label} forest interaction states = {forest_interaction_states}", file=sys.stderr)
            print(f"[poster] {entity_label} {cohort_label} dropped forest terms = {dropped_terms}", file=sys.stderr)
            significant_states = _poster_mixed_model_significant_states(box_rows)
            _grouped_compartment_boxplot(
                ax_box,
                basal_values=basal_values,
                apical_values=apical_values,
                state_order=boxplot_state_order,
                title=f"{cohort_label} {title.lower()}",
                ylabel="Mean response",
                cohort_label=cohort_label,
                significant_states=sorted(significant_states),
            )
            _forest_panel(ax_forest, forest_rows, title=f"{cohort_label} mixed model ({p_source_label})", show_legend=True, color_fn=forest_color_fn)
            ax_forest.text(0.98, 0.02, p_source_label, transform=ax_forest.transAxes, ha="right", va="bottom", fontsize=FIGURE_NOTE_FS, color="#444444")
    else:
        cohort_specs = [("responsive", resp, 0), ("nonresponsive", nonresp, 1)]
        for cohort_label, values, col_index in cohort_specs:
            ax_box = fig.add_subplot(outer[0, col_index])
            ax_forest = fig.add_subplot(outer[1, col_index])
            box_rows = rows_by_cohort.get(cohort_label, [])
            forest_rows = forest_rows_by_cohort.get(cohort_label, []) or box_rows
            if not box_rows:
                box_rows = rows_by_cohort.get("responsive", []) or rows_by_cohort.get("nonresponsive", [])
            if not forest_rows:
                forest_rows = box_rows
            forest_labels = {_state_compare_key(_forest_row_label(row)) for row in forest_rows if _forest_row_label(row)}
            box_labels = {_state_compare_key(_forest_row_label(row)) for row in box_rows if _forest_row_label(row)}
            state_forest_labels = {label for label in forest_labels if label}
            state_box_labels = {label for label in box_labels if label}
            if state_forest_labels and not state_forest_labels.issubset(state_box_labels):
                print(f"[ALERT] poster forest labels extend beyond boxplot labels for {entity_label} {cohort_label}: {sorted(state_forest_labels - state_box_labels)}", file=sys.stderr)
            raw_terms = [str(row.get('term') or row.get('contrast_name') or '') for row in forest_rows]
            forest_main_effect_states = sorted({_state_compare_key(_forest_row_label(row)) for row in forest_rows if str(row.get('term') or '').startswith('state[')})
            forest_interaction_states = sorted({_state_compare_key(_forest_row_label(row)) for row in forest_rows if _poster_mixed_model_term_kind(str(row.get('term') or '')) == 'interaction'})
            dropped_terms = [term for term in raw_terms if term not in {str(row.get("term") or "") for row in forest_rows}]
            print(f"[poster] {entity_label} {cohort_label} forest raw terms = {raw_terms}", file=sys.stderr)
            print(f"[poster] {entity_label} {cohort_label} forest main-effect states = {forest_main_effect_states}", file=sys.stderr)
            print(f"[poster] {entity_label} {cohort_label} forest interaction states = {forest_interaction_states}", file=sys.stderr)
            print(f"[poster] {entity_label} {cohort_label} dropped forest terms = {dropped_terms}", file=sys.stderr)
            significant_states = _poster_mixed_model_significant_states(box_rows)
            _boxplot(
                ax_box,
                values,
                present_state_order,
                title=f"{cohort_label} {title.lower()}",
                ylabel="Mean response",
                cohort_label=cohort_label,
                significance_flags=[_state_matches_any(state, sorted(significant_states)) for state in present_state_order],
                sample_sizes=(responsive_state_sample_sizes if cohort_label == "responsive" else nonresponsive_state_sample_sizes) or {state: int(len(_finite_array(values.get(state, [])))) for state in present_state_order},
                horizontal=True,
            )
            _forest_panel(ax_forest, forest_rows, title=f"{cohort_label} mixed model ({p_source_label})", show_legend=True, color_fn=forest_color_fn)
            ax_forest.text(0.98, 0.02, p_source_label, transform=ax_forest.transAxes, ha="right", va="bottom", fontsize=FIGURE_NOTE_FS, color="#444444")

    fig.suptitle(f"{entity_label.capitalize()} {title}", fontsize=FIGURE_TITLE_FS, y=0.985)
    output_path = out_dir / f"{stem}.svg"
    return _write_figure(fig, output_path)

def write_blank_movie_state_boxplot_figure(
    *,
    output_dir: Path | str,
    entity_label: str,
    responsive_blank_values: Mapping[str, Sequence[float]],
    responsive_movie_values: Mapping[str, Sequence[float]],
    nonresponsive_blank_values: Mapping[str, Sequence[float]],
    nonresponsive_movie_values: Mapping[str, Sequence[float]],
    blank_state_order: Sequence[str],
    movie_state_order: Sequence[str],
    responsive_blank_sample_sizes: Mapping[str, int] | None = None,
    responsive_movie_sample_sizes: Mapping[str, int] | None = None,
    nonresponsive_blank_sample_sizes: Mapping[str, int] | None = None,
    nonresponsive_movie_sample_sizes: Mapping[str, int] | None = None,
    responsive_significant_states: Sequence[str] | None = None,
    nonresponsive_significant_states: Sequence[str] | None = None,
    responsive_comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    nonresponsive_comparison_rows: Sequence[Mapping[str, Any]] | None = None,
    responsive_compartment_significant_states: Sequence[str] | None = None,
    nonresponsive_compartment_significant_states: Sequence[str] | None = None,
    output_stem: Optional[str] = None,
    title: str = "Blank vs movie states",
) -> Optional[str]:
    if plt is None:
        return None
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_blank_movie_states_poster_ready"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(FIGURE_HEIGHT_CM)), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.93, bottom=0.11, wspace=0.16, hspace=0.30)
    ax_resp_blank = fig.add_subplot(outer[0, 0])
    ax_nonresp_blank = fig.add_subplot(outer[0, 1], sharey=ax_resp_blank)
    ax_resp_movie = fig.add_subplot(outer[1, 0])
    ax_nonresp_movie = fig.add_subplot(outer[1, 1], sharey=ax_resp_movie)
    responsive_sig = {str(state) for state in (responsive_significant_states or [])}
    nonresponsive_sig = {str(state) for state in (nonresponsive_significant_states or [])}
    responsive_compartment_sig = {str(state) for state in (responsive_compartment_significant_states or [])}
    nonresponsive_compartment_sig = {str(state) for state in (nonresponsive_compartment_significant_states or [])}
    if responsive_comparison_rows:
        responsive_sig.update(_significant_state_labels_from_comparison_rows(responsive_comparison_rows))
    if nonresponsive_comparison_rows:
        nonresponsive_sig.update(_significant_state_labels_from_comparison_rows(nonresponsive_comparison_rows))
    blank_state_order = list(dict.fromkeys([str(state) for state in blank_state_order] + [str(state) for state in responsive_blank_values.keys()] + [str(state) for state in nonresponsive_blank_values.keys()]))
    movie_state_order = list(dict.fromkeys([str(state) for state in movie_state_order] + [str(state) for state in responsive_movie_values.keys()] + [str(state) for state in nonresponsive_movie_values.keys()]))
    print(f"[poster] {entity_label} blank/movie source states: blank={blank_state_order}; movie={movie_state_order}", file=sys.stderr)
    dendrite_grouped = str(entity_label).strip().lower() == "dendrite" or any(str(key).startswith(("basal_", "apical_")) for key in list(responsive_blank_values) + list(responsive_movie_values) + list(nonresponsive_blank_values) + list(nonresponsive_movie_values))

    def _split_compartment_values(values: Mapping[str, Sequence[float]]) -> tuple[dict[str, Sequence[float]], dict[str, Sequence[float]], dict[str, Sequence[float]]]:
        basal: dict[str, Sequence[float]] = {}
        apical: dict[str, Sequence[float]] = {}
        other: dict[str, Sequence[float]] = {}
        for key, arr in values.items():
            state_key = _canonical_state_key(key)
            if state_key.startswith("basal_"):
                basal[state_key[len("basal_"):]] = arr
            elif state_key.startswith("apical_"):
                apical[state_key[len("apical_"):]] = arr
            else:
                other[state_key] = arr
        return basal, apical, other

    def _project_dendrite_panel_values(values: Mapping[str, Sequence[float]], panel_kind: str) -> dict[str, list[float]]:
        projected: dict[str, list[float]] = {}
        if panel_kind not in {"blank", "movie"}:
            return {str(key): list(_finite_array(arr)) for key, arr in values.items()}
        exact_suffix = {
            "blank": {"quiet_awake_blank", "nrem_blank", "rem_blank"},
            "movie": {"quiet_awake_movies", "nrem_movies", "rem_movies"},
        }[panel_kind]
        fallback_suffix = {
            "blank": {"quiet_awake": "quiet_awake_blank", "nrem": "nrem_blank", "rem": "rem_blank"},
            "movie": {"quiet_awake": "quiet_awake_movies", "nrem": "nrem_movies", "rem": "rem_movies"},
        }[panel_kind]
        for key, arr in values.items():
            state_key = _canonical_state_key(key)
            if state_key.startswith("basal_") or state_key.startswith("apical_"):
                state_key = state_key.split("_", 1)[1]
            if state_key in exact_suffix:
                projected.setdefault(state_key, []).extend(_finite_array(arr).tolist())
            elif state_key in fallback_suffix:
                projected.setdefault(fallback_suffix[state_key], []).extend(_finite_array(arr).tolist())
        return projected

    def _grouped_panel(
        ax: plt.Axes,
        basal_values: Mapping[str, Sequence[float]],
        apical_values: Mapping[str, Sequence[float]],
        state_order: Sequence[str],
        *,
        title_text: str,
        cohort_label: str,
        significant_states: Sequence[str],
        compartment_significant_states: Sequence[str],
        sample_sizes_basal: Mapping[str, int] | None = None,
        sample_sizes_apical: Mapping[str, int] | None = None,
    ) -> None:
        rng = np.random.default_rng(7)
        basal_state_map = {state: _finite_array(basal_values.get(state, [])) for state in state_order}
        apical_state_map = {state: _finite_array(apical_values.get(state, [])) for state in state_order}
        all_data: list[np.ndarray] = []
        for compartment, summary_map, color, offset in (("basal", basal_state_map, BOX_COLORS.get("basal", "#4C72B0"), -0.18), ("apical", apical_state_map, BOX_COLORS.get("apical", "#DD8452"), 0.18)):
            positions: list[float] = []
            data: list[np.ndarray] = []
            for idx, state in enumerate(state_order, start=1):
                arr = summary_map.get(state, np.asarray([], dtype=float))
                if arr.size:
                    positions.append(float(idx) + offset)
                    data.append(arr)
            if not data:
                continue
            bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=False, vert=False)
            _set_boxplot_colors(bp, [color] * len(data))
            for pos, arr in zip(positions, data):
                jitter = rng.uniform(-0.08, 0.08, size=arr.size)
                ax.scatter(arr, np.full(arr.size, pos) + jitter, s=14, alpha=0.48, color=color, edgecolor="none")
            all_data.extend(data)
        ax.set_yticks(range(1, len(state_order) + 1))
        ax.set_yticklabels([_state_display_label(state) for state in state_order], fontsize=FIGURE_TICK_FS)
        ax.set_xlabel("Mean response", fontsize=FIGURE_LABEL_FS)
        ax.set_ylabel("State", fontsize=FIGURE_LABEL_FS)
        ax.set_title(title_text, fontsize=FIGURE_TITLE_FS, pad=4)
        ax.grid(axis="x", alpha=0.25)
        finite = np.concatenate(all_data) if all_data else np.asarray([], dtype=float)
        finite = finite[np.isfinite(finite)] if finite.size else finite
        if finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            pad = max(0.12 * max(hi - lo, 1e-6), 0.05)
            ax.set_xlim(lo - pad, hi + pad)
            x0, x1 = ax.get_xlim()
            x_range = max(x1 - x0, 1e-6)
            count_x = x1 + max(0.02 * x_range, 0.05)
            bracket_x = x1 + max(0.045 * x_range, 0.08)
            bracket_dx = max(0.02 * x_range, 0.035)
            bracket_step = max(0.028 * x_range, 0.05)
            sig_states = [state for state in state_order if _state_compare_key(state) in {_state_compare_key(s) for s in significant_states}]
            comp_sig_states = [state for state in state_order if _state_compare_key(state) in {_state_compare_key(s) for s in compartment_significant_states}]
            for level, state in enumerate(comp_sig_states):
                center = float(state_order.index(state) + 1)
                y_low = center - 0.18
                y_high = center + 0.18
                x = bracket_x + level * bracket_step
                ax.plot([x, x + bracket_dx, x + bracket_dx, x], [y_low, y_low, y_high, y_high], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
                ax.text(x + bracket_dx * 0.5, y_high + 0.06, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS + 2, color="#8b0000", fontweight="bold", clip_on=False)
            for compartment, summary_map, color, offset in (("basal", basal_state_map, BOX_COLORS.get("basal", "#4C72B0"), -0.18), ("apical", apical_state_map, BOX_COLORS.get("apical", "#DD8452"), 0.18)):
                sample_map = sample_sizes_basal if compartment == "basal" else sample_sizes_apical
                for idx, state in enumerate(state_order, start=1):
                    arr = summary_map.get(state, np.asarray([], dtype=float))
                    if not arr.size:
                        continue
                    sample_n = int(sample_map.get(state, int(arr.size))) if sample_map is not None else int(arr.size)
                    ax.text(count_x, float(idx) + offset, f"n={sample_n}", color=color, fontsize=FIGURE_NOTE_FS - 1, ha="left", va="center", clip_on=False)
            for level, state in enumerate(sig_states):
                y = float(state_order.index(state) + 1)
                x = bracket_x + level * bracket_step
                ax.plot([x, x + bracket_dx, x + bracket_dx, x], [y, y, y + 0.35, y + 0.35], color="#444444", linewidth=1.0, clip_on=False, zorder=5)
                ax.text(x + bracket_dx * 0.5, y + 0.35, "*", ha="center", va="bottom", fontsize=FIGURE_NOTE_FS + 2, color="#8b0000", fontweight="bold", clip_on=False)
        legend_handles = [
            Line2D([0], [0], color=BOX_COLORS.get("basal", "#4C72B0"), marker="s", linestyle="", markersize=8, label="Basal"),
            Line2D([0], [0], color=BOX_COLORS.get("apical", "#DD8452"), marker="s", linestyle="", markersize=8, label="Apical"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FIGURE_LEGEND_FS)
        ax.text(0.02, 0.98, cohort_label, transform=ax.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")

    if dendrite_grouped:
        resp_basal_blank, resp_apical_blank, resp_other_blank = _split_compartment_values(responsive_blank_values)
        resp_basal_movie, resp_apical_movie, resp_other_movie = _split_compartment_values(responsive_movie_values)
        nonresp_basal_blank, nonresp_apical_blank, nonresp_other_blank = _split_compartment_values(nonresponsive_blank_values)
        nonresp_basal_movie, nonresp_apical_movie, nonresp_other_movie = _split_compartment_values(nonresponsive_movie_values)
        blank_state_order = ["quiet_awake_blank", "nrem_blank", "rem_blank"]
        movie_state_order = ["quiet_awake_movies", "nrem_movies", "rem_movies"]
        _grouped_panel(
            ax_resp_blank,
            _project_dendrite_panel_values({**resp_basal_blank, **resp_other_blank}, "blank"),
            _project_dendrite_panel_values({**resp_apical_blank, **resp_other_blank}, "blank"),
            blank_state_order,
            title_text="responsive blank states",
            cohort_label="responsive",
            significant_states=[state for state in responsive_sig],
            compartment_significant_states=[state for state in responsive_compartment_sig],
            sample_sizes_basal=responsive_blank_sample_sizes,
            sample_sizes_apical=responsive_blank_sample_sizes,
        )
        _grouped_panel(
            ax_nonresp_blank,
            _project_dendrite_panel_values({**nonresp_basal_blank, **nonresp_other_blank}, "blank"),
            _project_dendrite_panel_values({**nonresp_apical_blank, **nonresp_other_blank}, "blank"),
            blank_state_order,
            title_text="nonresponsive blank states",
            cohort_label="nonresponsive",
            significant_states=[state for state in nonresponsive_sig],
            compartment_significant_states=[state for state in nonresponsive_compartment_sig],
            sample_sizes_basal=nonresponsive_blank_sample_sizes,
            sample_sizes_apical=nonresponsive_blank_sample_sizes,
        )
        _grouped_panel(
            ax_resp_movie,
            _project_dendrite_panel_values({**resp_basal_movie, **resp_other_movie}, "movie"),
            _project_dendrite_panel_values({**resp_apical_movie, **resp_other_movie}, "movie"),
            movie_state_order,
            title_text="responsive movie states",
            cohort_label="responsive",
            significant_states=[state for state in responsive_sig],
            compartment_significant_states=[state for state in responsive_compartment_sig],
            sample_sizes_basal=responsive_movie_sample_sizes,
            sample_sizes_apical=responsive_movie_sample_sizes,
        )
        _grouped_panel(
            ax_nonresp_movie,
            _project_dendrite_panel_values({**nonresp_basal_movie, **nonresp_other_movie}, "movie"),
            _project_dendrite_panel_values({**nonresp_apical_movie, **nonresp_other_movie}, "movie"),
            movie_state_order,
            title_text="nonresponsive movie states",
            cohort_label="nonresponsive",
            significant_states=[state for state in nonresponsive_sig],
            compartment_significant_states=[state for state in nonresponsive_compartment_sig],
            sample_sizes_basal=nonresponsive_movie_sample_sizes,
            sample_sizes_apical=nonresponsive_movie_sample_sizes,
        )
    else:
        _boxplot(
            ax_resp_blank,
            responsive_blank_values,
            blank_state_order,
            title="responsive blank states",
            ylabel="Mean response",
            cohort_label="responsive",
            significance_flags=[state in responsive_sig for state in blank_state_order],
            sample_sizes=responsive_blank_sample_sizes or {state: int(len(_finite_array(responsive_blank_values.get(state, [])))) for state in blank_state_order},
            horizontal=True,
        )
        _boxplot(
            ax_nonresp_blank,
            nonresponsive_blank_values,
            blank_state_order,
            title="nonresponsive blank states",
            ylabel="Mean response",
            cohort_label="nonresponsive",
            significance_flags=[state in nonresponsive_sig for state in blank_state_order],
            sample_sizes=nonresponsive_blank_sample_sizes,
            horizontal=True,
        )
        _boxplot(
            ax_resp_movie,
            responsive_movie_values,
            movie_state_order,
            title="responsive movie states",
            ylabel="Mean response",
            cohort_label="responsive",
            significance_flags=[state in responsive_sig for state in movie_state_order],
            sample_sizes=responsive_movie_sample_sizes,
            horizontal=True,
        )
        _boxplot(
            ax_nonresp_movie,
            nonresponsive_movie_values,
            movie_state_order,
            title="nonresponsive movie states",
            ylabel="Mean response",
            cohort_label="nonresponsive",
            significance_flags=[state in nonresponsive_sig for state in movie_state_order],
            sample_sizes=nonresponsive_movie_sample_sizes,
            horizontal=True,
        )
    for ax in (ax_nonresp_blank, ax_nonresp_movie):
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")
    for ax in (ax_resp_blank, ax_nonresp_blank):
        ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle(f"{entity_label.capitalize()} {title}", fontsize=FIGURE_TITLE_FS, y=0.985)
    output_path = out_dir / f"{stem}.svg"
    return _write_figure(fig, output_path)


__all__ = [
    "assign_pairwise_visual_response_cohorts",
    "split_rows_by_cohort",
    "write_visual_response_poster_figure",
    "write_state_mixed_model_poster_figure",
    "write_blank_movie_state_boxplot_figure",
    "write_blank_movie_and_correlation_poster_figure",
]
