from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.shared.shared_calcium_response import load_visual_response_cut_data, visual_response_trial_group
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


FIGURE_WIDTH_CM = 35.0
FIGURE_HEIGHT_CM = 29.0

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
    "quiet_awake_blank": "#4C72B0",
    "nrem_blank": "#55A868",
    "rem_blank": "#C44E52",
    "quiet_awake_movies": "#4C72B0",
    "nrem_movies": "#55A868",
    "rem_movies": "#C44E52",
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


def _state_value_map_from_rows(rows: Sequence[Mapping[str, Any]], *, value_key: str = "mean") -> Dict[str, list[float]]:
    grouped: Dict[str, list[float]] = {}
    for row in rows:
        state = str(row.get("state") or row.get("state_label") or "").strip()
        if not state:
            continue
        value = row.get(value_key)
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
    responsive = [dict(row) for row in rows if bool(row.get("responsive", False)) and np.isfinite(float(row.get("delta", float("nan"))))]
    nonresponsive = [dict(row) for row in rows if not bool(row.get("responsive", False)) and np.isfinite(float(row.get("delta", float("nan"))))]
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
    return {
        "cut_time": cut_time,
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


def _boxplot(ax: plt.Axes, state_values: Mapping[str, Sequence[float]], state_order: Sequence[str], *, title: str, ylabel: str, cohort_label: str | None = None) -> None:
    series = []
    labels = []
    positions = []
    colors = []
    for idx, state in enumerate(state_order, start=1):
        arr = _finite_array(state_values.get(state, []))
        labels.append(state.replace("_", " "))
        positions.append(float(idx))
        colors.append(BOX_COLORS.get(state, "#7f7f7f"))
        if arr.size == 0:
            continue
        series.append(arr)
    if series:
        present_positions = [pos for pos, state in zip(positions, state_order) if _finite_array(state_values.get(state, [])).size > 0]
        present_colors = [BOX_COLORS.get(state, "#7f7f7f") for state in state_order if _finite_array(state_values.get(state, [])).size > 0]
        bp = ax.boxplot(series, positions=present_positions, widths=0.56, patch_artist=True, showfliers=False)
        for patch, color in zip(bp.get("boxes", []), present_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#444444")
            patch.set_alpha(0.9)
        for line in bp.get("whiskers", []) + bp.get("caps", []):
            line.set_color("#555555")
        for median in bp.get("medians", []):
            median.set_color("#222222")
            median.set_linewidth(1.5)
        rng = np.random.default_rng(7)
        for xpos, state in zip(present_positions, [state for state in state_order if _finite_array(state_values.get(state, [])).size > 0]):
            arr = _finite_array(state_values.get(state, []))
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            ax.scatter(np.full(arr.size, xpos) + jitter, arr, s=14, alpha=0.45, color=BOX_COLORS.get(state, "#7f7f7f"), edgecolor="none")
    ax.set_xlim(0.5, max(float(len(state_order)) + 0.5, 1.5))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=FIGURE_LABEL_FS)
    ax.set_title(title, fontsize=FIGURE_TITLE_FS, pad=4)
    ax.grid(axis="y", alpha=0.22)
    if cohort_label:
        ax.text(0.02, 0.98, cohort_label, transform=ax.transAxes, ha="left", va="top", fontsize=FIGURE_NOTE_FS, color="#444444")
    if not series:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=FIGURE_NOTE_FS, color="#666666")
    _style_axes(ax)


def _forest_panel(ax: plt.Axes, rows: Sequence[Mapping[str, Any]], *, title: str, ylabel: str = "Estimate (95% CI)", p_value_key: str = "p_value") -> None:
    rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not rows:
        ax.set_axis_off()
        return
    ordered = [row for row in rows if row.get("term") is not None]
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
        color = MIXED_MODEL_COLOR if str(row.get("term") or "").startswith("state[") else "#7f7f7f"
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
    labels = [str(row.get("term") or "").replace("state[", "").replace("]", "").replace(":compartment[apical]", " apical") for row in ordered]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=FIGURE_TICK_FS - 1)
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


def _single_exemplar_panel(ax_blank: plt.Axes, ax_movie: plt.Axes, ax_box: plt.Axes, response_row: Mapping[str, Any], *, title: str) -> None:
    plot_data = _load_visual_response_plot_data(response_row)
    if not plot_data:
        ax_blank.set_axis_off()
        ax_movie.set_axis_off()
        ax_box.set_axis_off()
        return
    cut_time = np.asarray(plot_data["cut_time"], dtype=float)
    visual_mean_trace = np.asarray(plot_data["visual_mean_trace"], dtype=float)
    blank_mean_trace = np.asarray(plot_data["blank_mean_trace"], dtype=float)
    visual_values = np.asarray(plot_data["visual_values"], dtype=float)
    blank_values = np.asarray(plot_data["blank_values"], dtype=float)
    if cut_time.size == 0 or visual_mean_trace.size == 0 or blank_mean_trace.size == 0:
        ax_blank.set_axis_off()
        ax_movie.set_axis_off()
        ax_box.set_axis_off()
        return

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

    data = [blank_values, visual_values]
    bp = ax_box.boxplot(data, positions=[1.0, 2.0], widths=0.58, patch_artist=True, showfliers=False)
    for patch, color in zip(bp.get("boxes", []), ["#D9D9D9", "#F8C38B"]):
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
    for xpos, values, color in [(1.0, blank_values, "#6B6B6B"), (2.0, visual_values, "#C96E00")]:
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax_box.scatter(np.full(values.size, xpos) + jitter, values, s=12, alpha=0.42, color=color, edgecolor="none")
    ax_box.set_xticks([1.0, 2.0])
    ax_box.set_xticklabels(["blank", "movies"])
    ax_box.set_ylabel("Mean cut-stimulus activity", fontsize=FIGURE_LABEL_FS)
    ax_box.set_xlabel("Condition", fontsize=FIGURE_LABEL_FS)
    ax_box.set_title("Blank vs movies", fontsize=FIGURE_TITLE_FS, pad=4)
    ax_box.grid(axis="y", alpha=0.2)
    all_values = np.concatenate([blank_values, visual_values])
    finite = all_values[np.isfinite(all_values)]
    if finite.size:
        y_low, y_high = _combined_limits(blank_mean_trace, visual_mean_trace, blank_values, visual_values)
        ax_blank.set_ylim(y_low, y_high)
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
            ax_box.text(1.5, y, "*", ha="center", va="bottom", fontsize=24, color="#8b0000", fontweight="bold")

    _style_axes(ax_blank)
    _style_axes(ax_movie)
    _style_axes(ax_box)
def _select_mixed_model_rows(mixed_model_branch: Any, preferred_response_keys: Sequence[str] | None = None) -> list[dict[str, Any]]:
    if isinstance(mixed_model_branch, Mapping):
        summary_rows = mixed_model_branch.get("summary_rows")
        if isinstance(summary_rows, Mapping):
            preferred_keys = tuple(str(key) for key in (preferred_response_keys or ("mean_dendrite_activity", "mean_spine_activity_per_dendrite", "mean")))
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




def write_visual_response_poster_figure(
    *,
    output_dir: Path | str,
    entity_label: str,
    visual_response_rows: Sequence[Mapping[str, Any]],
    output_stem: Optional[str] = None,
    locomotion_threshold: float | None = None,
) -> Optional[str]:
    if plt is None:
        return None
    rows = [dict(row) for row in visual_response_rows if isinstance(row, Mapping)]
    if not rows:
        return None
    responsive_row, nonresponsive_row = _pick_exemplar_rows(rows)
    if responsive_row is None or nonresponsive_row is None:
        return None
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_visual_response_poster_ready"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(FIGURE_HEIGHT_CM)), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.85, 0.85], height_ratios=[1, 1], left=0.06, right=0.985, top=0.95, bottom=0.08, wspace=0.20, hspace=0.24)
    resp_grid = outer[0, 0].subgridspec(1, 3, width_ratios=[1.05, 1.05, 0.95], wspace=0.22)
    nonresp_grid = outer[1, 0].subgridspec(1, 3, width_ratios=[1.05, 1.05, 0.95], wspace=0.22)
    ax_resp_blank = fig.add_subplot(resp_grid[0, 0])
    ax_resp_movie = fig.add_subplot(resp_grid[0, 1], sharey=ax_resp_blank)
    ax_resp_box = fig.add_subplot(resp_grid[0, 2], sharey=ax_resp_blank)
    ax_nonresp_blank = fig.add_subplot(nonresp_grid[0, 0])
    ax_nonresp_movie = fig.add_subplot(nonresp_grid[0, 1], sharey=ax_nonresp_blank)
    ax_nonresp_box = fig.add_subplot(nonresp_grid[0, 2], sharey=ax_nonresp_blank)
    ax_pie = fig.add_subplot(outer[:, 1])
    _single_exemplar_panel(ax_resp_blank, ax_resp_movie, ax_resp_box, responsive_row, title="movie responsive exemplar")
    _single_exemplar_panel(ax_nonresp_blank, ax_nonresp_movie, ax_nonresp_box, nonresponsive_row, title="movie nonresponsive exemplar")
    counts = np.asarray([
        sum(bool(row.get("responsive", False)) for row in rows),
        sum(not bool(row.get("responsive", False)) for row in rows),
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
    ax_pie.set_title(f"{entity_label.capitalize()} visual responsiveness", fontsize=FIGURE_TITLE_FS, pad=8)
    ax_pie.text(0.5, -0.08, f"n={int(total)}", transform=ax_pie.transAxes, ha="center", va="top", fontsize=FIGURE_NOTE_FS)
    fig.suptitle(f"{entity_label.capitalize()} visual response", fontsize=POSTER_TITLE_SIZE + 1, y=0.988)
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
) -> Optional[str]:
    if plt is None:
        return None
    resp = {str(k): list(v) for k, v in responsive_state_values.items()}
    nonresp = {str(k): list(v) for k, v in nonresponsive_state_values.items()}
    if not resp and not nonresp:
        return None
    if isinstance(mixed_model_rows, Mapping) and ("responsive" in mixed_model_rows or "nonresponsive" in mixed_model_rows):
        rows_by_cohort = {
            str(cohort): _select_mixed_model_rows(branch, preferred_response_keys=preferred_response_keys)
            for cohort, branch in mixed_model_rows.items()
            if str(cohort) in {"responsive", "nonresponsive"}
        }
    else:
        selected_rows = _select_mixed_model_rows(mixed_model_rows, preferred_response_keys=preferred_response_keys)
        rows_by_cohort = {"responsive": selected_rows, "nonresponsive": selected_rows}
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_state_mixed_model_poster_ready"
    p_source = _normalize_mixed_model_contrast_p_source(mixed_model_contrast_p_source)
    p_source_label = "shuffle p" if p_source == "shuffle" else "classical p"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(20.5)), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.92, bottom=0.11, wspace=0.20, hspace=0.28, height_ratios=[0.92, 1.08])
    cohort_specs = [("responsive", resp, 0), ("nonresponsive", nonresp, 1)]
    for cohort_label, values, col_index in cohort_specs:
        ax_box = fig.add_subplot(outer[0, col_index])
        ax_forest = fig.add_subplot(outer[1, col_index])
        _boxplot(ax_box, values, state_order, title=f"{cohort_label} {title.lower()}", ylabel="Mean response", cohort_label=cohort_label)
        cohort_rows = rows_by_cohort.get(cohort_label, [])
        if not cohort_rows:
            cohort_rows = rows_by_cohort.get("responsive", []) or rows_by_cohort.get("nonresponsive", [])
        _forest_panel(ax_forest, cohort_rows, title=f"{cohort_label} mixed model ({p_source_label})")
        ax_forest.text(0.98, 0.02, p_source_label, transform=ax_forest.transAxes, ha="right", va="bottom", fontsize=FIGURE_NOTE_FS, color="#444444")
    fig.suptitle(f"{entity_label.capitalize()} {title}", fontsize=POSTER_TITLE_SIZE + 1, y=0.985)
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
    output_stem: Optional[str] = None,
    title: str = "Blank vs movie states",
) -> Optional[str]:
    if plt is None:
        return None
    out_dir = _ensure_dir(Path(output_dir))
    stem = output_stem or f"{entity_label}_blank_movie_states_poster_ready"
    fig = plt.figure(figsize=(cm_to_inch(FIGURE_WIDTH_CM), cm_to_inch(18.5)), constrained_layout=False)
    outer = fig.add_gridspec(2, 2, left=0.07, right=0.985, top=0.93, bottom=0.11, wspace=0.15, hspace=0.28)
    panels = [
        ("responsive", responsive_blank_values, responsive_movie_values, 0),
        ("nonresponsive", nonresponsive_blank_values, nonresponsive_movie_values, 1),
    ]
    for cohort_label, blank_values, movie_values, row_index in panels:
        ax_blank = fig.add_subplot(outer[row_index, 0])
        ax_movie = fig.add_subplot(outer[row_index, 1])
        _boxplot(ax_blank, blank_values, blank_state_order, title=f"{cohort_label} blank states", ylabel="Mean response", cohort_label=cohort_label)
        _boxplot(ax_movie, movie_values, movie_state_order, title=f"{cohort_label} movie states", ylabel="Mean response")
    fig.suptitle(f"{entity_label.capitalize()} {title}", fontsize=POSTER_TITLE_SIZE + 1, y=0.985)
    output_path = out_dir / f"{stem}.svg"
    return _write_figure(fig, output_path)


__all__ = [
    "write_visual_response_poster_figure",
    "write_state_mixed_model_poster_figure",
    "write_blank_movie_state_boxplot_figure",
]
