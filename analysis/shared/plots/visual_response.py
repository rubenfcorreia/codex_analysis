from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.compartment_common import pick_state_bundle
from analysis.shared.shared_calcium_response import load_visual_response_cut_data, visual_response_trial_group
from analysis.main_pipeline.sleep_dendrite_spine_pipeline import (
    plot_visual_response_boxplot_figure,
    visual_response_figure_output_dir,
)


def _infer_source_parts(response_row: Mapping[str, Any]) -> tuple[Optional[Path], Optional[int]]:
    source_path = response_row.get("source_path")
    if not source_path:
        return None, None
    path = Path(str(source_path))
    match = re.search(r"ch(\d+)", path.name)
    channel = int(match.group(1)) if match else None
    exp_root = path.parent.parent if len(path.parents) >= 2 else None
    return exp_root, channel


def _load_visual_response_plot_data(response_row: Mapping[str, Any], *, locomotion_threshold: float | None = None) -> Optional[Dict[str, Any]]:
    exp_root, channel = _infer_source_parts(response_row)
    if exp_root is None or channel is None or not exp_root.exists():
        return None
    mode = str(response_row.get("mode") or "")
    try:
        _, state_bundle = pick_state_bundle(exp_root, mode)
    except Exception:
        return None
    trial_rows = state_bundle.get("rows", []) if isinstance(state_bundle, Mapping) else []
    cut_data = load_visual_response_cut_data(exp_root, channel, trial_rows, locomotion_threshold=locomotion_threshold)
    if not cut_data:
        return None
    roi_index = int(response_row.get("roi_index", -1))
    cut_neural = np.asarray(cut_data.get("cut_neural"), dtype=float)
    cut_time = np.asarray(cut_data.get("cut_time"), dtype=float)
    trial_meta = cut_data.get("trial_meta", []) if isinstance(cut_data, dict) else []
    if roi_index < 0 or roi_index >= cut_neural.shape[0] or cut_time.size == 0:
        return None
    trial_matrix = np.asarray(cut_neural[roi_index], dtype=float)
    visual_traces: List[np.ndarray] = []
    blank_traces: List[np.ndarray] = []
    visual_values: List[float] = []
    blank_values: List[float] = []
    for meta in trial_meta:
        if not isinstance(meta, Mapping):
            continue
        trial_index = int(meta.get("trial_index")) if meta.get("trial_index") is not None else None
        if trial_index is None or trial_index < 0 or trial_index >= trial_matrix.shape[0]:
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
    return {
        "cut_time": cut_time,
        "visual_traces": visual_traces,
        "blank_traces": blank_traces,
        "visual_values": np.asarray(visual_values, dtype=float),
        "blank_values": np.asarray(blank_values, dtype=float),
        "visual_mean_trace": np.asarray(np.nanmean(np.asarray(visual_traces, dtype=float), axis=0), dtype=float),
        "blank_mean_trace": np.asarray(np.nanmean(np.asarray(blank_traces, dtype=float), axis=0), dtype=float),
    }


def plot_visual_response_entity_figure(
    response_row: Mapping[str, Any],
    fig_dir: Path | str,
    *,
    cohort_label: str = "all",
    kind: str = "soma",
) -> Optional[str]:
    if plt is None:
        return None
    plot_data = _load_visual_response_plot_data(response_row)
    if not plot_data:
        return None
    cut_time = np.asarray(plot_data["cut_time"], dtype=float)
    visual_traces = [np.asarray(trace, dtype=float) for trace in plot_data["visual_traces"]]
    blank_traces = [np.asarray(trace, dtype=float) for trace in plot_data["blank_traces"]]
    visual_values = np.asarray(plot_data["visual_values"], dtype=float)
    blank_values = np.asarray(plot_data["blank_values"], dtype=float)
    visual_mean_trace = np.asarray(plot_data["visual_mean_trace"], dtype=float)
    blank_mean_trace = np.asarray(plot_data["blank_mean_trace"], dtype=float)
    if cut_time.size == 0 or visual_mean_trace.size == 0 or blank_mean_trace.size == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), gridspec_kw={"width_ratios": [1.05, 1.05, 0.95]})
    blank_ax, movie_ax, box_ax = axes

    for trace in blank_traces:
        blank_ax.plot(cut_time, trace, color="#9AA0A6", linewidth=0.7, alpha=0.10, zorder=1)
    blank_ax.plot(cut_time, blank_mean_trace, color="#7F8790", linewidth=2.6, zorder=3)
    blank_ax.set_title("Blank traces", fontsize=14)
    blank_ax.set_xlabel("Time (s)")
    blank_ax.set_ylabel("dF/F")
    blank_ax.grid(axis="y", alpha=0.2)
    blank_ax.text(0.02, 0.98, f"trials: blank={len(blank_traces)}", transform=blank_ax.transAxes, ha="left", va="top", fontsize=9, color="#444444")

    for trace in visual_traces:
        movie_ax.plot(cut_time, trace, color="#F58518", linewidth=0.7, alpha=0.10, zorder=1)
    movie_ax.plot(cut_time, visual_mean_trace, color="#D97706", linewidth=2.6, zorder=3)
    movie_ax.set_title("Movies traces", fontsize=14)
    movie_ax.set_xlabel("Time (s)")
    movie_ax.set_ylabel("dF/F")
    movie_ax.grid(axis="y", alpha=0.2)
    movie_ax.text(0.02, 0.98, f"trials: movies={len(visual_traces)}", transform=movie_ax.transAxes, ha="left", va="top", fontsize=9, color="#444444")

    data = [blank_values, visual_values]
    bp = box_ax.boxplot(data, positions=[1.0, 2.0], widths=0.58, patch_artist=True, showfliers=False)
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
        box_ax.scatter(np.full(values.size, xpos) + jitter, values, s=12, alpha=0.42, color=color, edgecolor="none")
    box_ax.set_xticks([1.0, 2.0])
    box_ax.set_xticklabels(["blank", "movies"])
    box_ax.set_ylabel("Mean cut-stimulus activity")
    box_ax.set_title("Blank vs movies")
    box_ax.grid(axis="y", alpha=0.2)
    all_values = np.concatenate([blank_values, visual_values])
    finite = all_values[np.isfinite(all_values)]
    if finite.size:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        pad = max(0.05 * (high - low), 0.05)
        box_ax.set_ylim(low - pad, high + pad)
    box_ax.text(0.02, 0.98, f"n={int(blank_values.size)} blank, n={int(visual_values.size)} visual", transform=box_ax.transAxes, ha="left", va="top", fontsize=9, color="#444444")
    ttest = stats.ttest_ind(np.asarray(visual_values, dtype=float), np.asarray(blank_values, dtype=float), equal_var=False, nan_policy="omit")
    p_value = float(ttest.pvalue) if np.isfinite(ttest.pvalue) else float("nan")
    if bool(response_row.get("significant", False)) or (np.isfinite(p_value) and p_value < 0.05):
        finite = np.concatenate([blank_values, visual_values])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y = float(np.nanmax(finite)) + max(0.05 * float(np.ptp(finite)), 0.05)
            box_ax.plot([1.0, 1.0, 2.0, 2.0], [y * 0.98, y, y, y * 0.98], color="#8b0000", linewidth=1.2)
            box_ax.text(1.5, y, "*", ha="center", va="bottom", fontsize=24, color="#8b0000", fontweight="bold")

    cohort_text = f" ({cohort_label})" if cohort_label and cohort_label != "all" else ""
    title_prefix = "Soma" if kind == "soma" else "Bouton"
    fig.suptitle(f"{title_prefix} {response_row.get('animal_id', '')} {response_row.get('roi_index', '')}{cohort_text}", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.84, wspace=0.36)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"{response_row.get('animal_id', 'animal')}_{response_row.get('roi_index', 'roi')}_{cohort_label}_blank_vs_movies.svg"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def render_visual_response_entity_figures(
    response_rows: Sequence[Mapping[str, Any]],
    fig_dir: Path | str,
    *,
    cohort_label: str = "all",
    kind: str = "soma",
) -> List[str]:
    saved: List[str] = []
    for row in response_rows:
        output = plot_visual_response_entity_figure(row, fig_dir, cohort_label=cohort_label, kind=kind)
        if output:
            saved.append(output)
    return saved


__all__ = [
    "plot_visual_response_boxplot_figure",
    "plot_visual_response_entity_figure",
    "visual_response_figure_output_dir",
    "render_visual_response_entity_figures",
]
