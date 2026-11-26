"""General initial-analysis figures, including state-overlap summaries."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from analysis.cache_builders.overlap_plots import venn_plot
from analysis.shared.state_utils import state_display_color, state_display_label

_STATE_ORDER = [
    "quiet_awake_blank", "nrem_blank", "rem_blank",
    "quiet_awake_movies", "nrem_movies", "rem_movies",
    "quiet_awake_gratings", "nrem_gratings", "rem_gratings",
    "quiet_awake_zebras", "nrem_zebras", "rem_zebras",
    "quiet_awake", "nrem", "rem", "active_awake",
]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _bar(counts: Mapping[str, int], title: str, path: Path, colors: Mapping[str, str] | None = None) -> None:
    labels = list(counts); values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, min(13, 4 + 0.28 * len(labels))), 4.5))
    ax.barh(labels[::-1], values[::-1], color=[(colors or {}).get(label, "#4c78a8") for label in labels[::-1]])
    ax.set_title(title); ax.set_xlabel("ROI count"); ax.grid(axis="x", alpha=0.2)
    _save(fig, path)


def _ordered_states(states: Sequence[str]) -> list[str]:
    present = {str(state) for state in states}
    return [state for state in _STATE_ORDER if state in present] + sorted(present.difference(_STATE_ORDER))


def _state_sets(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in records:
        key = str(row.get("roi_key"))
        states = row.get("preferred_states", [])
        if isinstance(states, str):
            states = [states]
        for state in states or []:
            result.setdefault(str(state), set()).add(key)
    return result


def _overlap(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    sets = _state_sets(records)
    states = _ordered_states(list(sets))
    if not states:
        return
    if len(states) <= 3:
        venn_plot({state: sets[state] for state in states}, path, colors=[state_display_color(state) for state in sorted(states)])
    else:
        combinations_count = Counter(tuple(sorted(str(state) for state in (row.get("preferred_states") or []))) for row in records)
        combinations_count.pop((), None)
        _bar({" ∩ ".join(key): value for key, value in combinations_count.most_common(20)}, "Preferred-state overlap (UpSet)", path)


def _numeric_plot(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    pairs = [(float(row["activity_mean"]), float(row["frequency_mean"])) for row in records if row.get("activity_mean") is not None and row.get("frequency_mean") is not None]
    activity = [pair[0] for pair in pairs]
    frequency = [pair[1] for pair in pairs]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(activity, frequency, s=20, alpha=0.75)
    ax.set_xlabel("Mean activity"); ax.set_ylabel("Mean event frequency"); ax.set_title("Activity–frequency relationship")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _category(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("roi_type") or "unknown"),
        str(row.get("compartment") or "unspecified"),
        str(row.get("visual_class") or "unknown"),
    )


def _sort_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda row: (_category(row), str(row.get("preferred_state") or ""), str(row.get("roi_key") or "")))


def _add_heatmap_groups(ax: plt.Axes, records: Sequence[Mapping[str, Any]]) -> None:
    groups = []
    start = 0
    previous = None
    for index, row in enumerate(records):
        current = _category(row)
        if previous is not None and current != previous:
            groups.append((start, index - 1, previous)); start = index
        previous = current
    if previous is not None:
        groups.append((start, len(records) - 1, previous))
    for start, end, category in groups:
        if start:
            ax.axhline(start - 0.5, color="white", linewidth=1.5)
        midpoint = (start + end) / 2
        ax.text(-0.02, midpoint, f"{category[0]} / {category[1]} / {category[2]}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    ax.set_yticks([])


def _heatmap(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    ordered_records = _sort_records(records)
    states = _ordered_states([str(state) for row in ordered_records for state in (row.get("state_activity") or {})])
    if not states or not ordered_records:
        return
    values = np.full((len(ordered_records), len(states)), np.nan)
    for i, row in enumerate(ordered_records):
        for j, state in enumerate(states):
            value = (row.get("state_activity") or {}).get(state)
            if value is not None:
                try:
                    values[i, j] = float(value)
                except (TypeError, ValueError):
                    pass
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(states) + 4), max(5, min(16, 3 + len(records) * 0.025))))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest", cmap=cmap)
    labels = [state_display_label(state) for state in states]
    ax.set_xlabel("State"); ax.set_ylabel("ROI (grouped by type, compartment, and visual response)")
    ax.set_xticks(range(len(states)), labels, rotation=35, ha="right")
    for tick, state in zip(ax.get_xticklabels(), states):
        tick.set_color(state_display_color(state)); tick.set_fontweight("bold")
    _add_heatmap_groups(ax, ordered_records)
    fig.colorbar(image, ax=ax, label="Mean activity"); _save(fig, path)


def _frequency_heatmap(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    states = sorted({str(state) for row in records for state in (row.get("state_frequency") or {})})
    if not states or not records:
        return
    values = np.full((len(records), len(states)), np.nan)
    for i, row in enumerate(records):
        for j, state in enumerate(states):
            value = (row.get("state_frequency") or {}).get(state)
            if value is not None:
                values[i, j] = float(value)
    fig, ax = plt.subplots(figsize=(7, max(4, min(12, len(records) * 0.18))))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="magma")
    labels = [state_display_label(state) for state in states]
    ax.set_xlabel("State"); ax.set_ylabel("ROI"); ax.set_xticks(range(len(states)), labels, rotation=35, ha="right")
    for tick, state in zip(ax.get_xticklabels(), states):
        tick.set_color(state_display_color(state)); tick.set_fontweight("bold")
    fig.colorbar(image, ax=ax, label="Event frequency / min"); _save(fig, path)


def _visual_category_plot(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, str], Counter[str]] = {}
    for row in records:
        roi_type, compartment, _ = _category(row)
        grouped.setdefault((roi_type, compartment), Counter())[str(row.get("visual_class") or "unknown")] += 1
    if not grouped:
        return
    categories = list(grouped)
    classes = ["visual", "non_visual", "unknown"]
    totals = [sum(grouped[key].values()) for key in categories]
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(categories) + 4), 5.5))
    x = np.arange(len(categories)); bottom = np.zeros(len(categories))
    colors = {"visual": "#2ca02c", "non_visual": "#7f7f7f", "unknown": "#d9d9d9"}
    for cls in classes:
        values = np.array([grouped[key][cls] for key in categories], dtype=float)
        ax.bar(x, values, bottom=bottom, label=cls.replace("_", " "), color=colors[cls])
        bottom += values
    for index, total in enumerate(totals):
        ax.text(index, total + max(totals) * 0.02, f"n={total}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"{roi_type}\\n{compartment}" for roi_type, compartment in categories])
    ax.set_ylabel("ROI count"); ax.set_title("Visual responsiveness by ROI category")
    ax.legend(frameon=False, ncol=3); ax.grid(axis="y", alpha=0.2)
    _save(fig, path)

def _preferred_state_proportion_plot(records: Sequence[Mapping[str, Any]], roi_type: str, compartment: str, path: Path, preferred_state_views: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> None:
    views = preferred_state_views or {"combined": records}
    view_names = ["combined", "movie", "sleep"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, view_name in zip(axes, view_names):
        source = views.get(view_name, [])
        category = [row for row in source if str(row.get("roi_type") or "unknown") == roi_type and str(row.get("compartment") or "unspecified") == compartment]
        eligible = [row for row in category if str(row.get("preferred_state_class") or "unknown") != "unknown"]
        unknown = len(category) - len(eligible)
        if not eligible:
            ax.text(0.5, 0.5, "No preferred-state data", ha="center", va="center")
            ax.set_axis_off()
            continue
        counts = Counter(str(row.get("preferred_state_class")) for row in eligible)
        labels = _ordered_states([label for label in counts if label != "tie"])
        if "tie" in counts: labels.append("tie")
        proportions = np.array([100.0 * counts[label] / len(eligible) for label in labels])
        colors = [state_display_color(label) if label != "tie" else "#7f7f7f" for label in labels]
        ax.barh(labels[::-1], proportions[::-1], color=colors[::-1])
        for label, proportion in zip(labels, proportions):
            ax.text(min(98, proportion + 1.0), label, f"{proportion:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Eligible ROIs (%)")
        ax.set_title(f"{view_name.title()} (n={len(eligible)}, unknown={unknown})")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"Preferred-state proportions: {compartment} {roi_type}", y=1.02)
    fig.tight_layout()
    _save(fig, path)




def _state_coverage_plot(records: Sequence[Mapping[str, Any]], path: Path, preferred_state_views: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> None:
    views = preferred_state_views or {"combined": records}
    view_names = ["combined", "movie", "sleep"]
    categories = [("dendrite", "apical"), ("dendrite", "basal"), ("spine", "apical"), ("spine", "basal")]
    category_labels = [f"{roi} {comp}" for roi, comp in categories]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    plotted = False
    for ax, view_name in zip(axes, view_names):
        source = list(views.get(view_name, []))
        states = _ordered_states([str(state) for row in source for state in (row.get("state_activity") or {})])
        if not source or not states:
            ax.text(0.5, 0.5, "No state-activity data", ha="center", va="center")
            ax.set_axis_off()
            continue
        plotted = True
        x = np.arange(len(states))
        for category, label in zip(categories, category_labels):
            category_rows = [row for row in source if (str(row.get("roi_type") or "unknown"), str(row.get("compartment") or "unspecified")) == category]
            if not category_rows:
                continue
            coverage = []
            for state in states:
                valid = 0
                for row in category_rows:
                    value = (row.get("state_activity") or {}).get(state)
                    try:
                        valid += int(value is not None and np.isfinite(float(value)))
                    except (TypeError, ValueError):
                        pass
                coverage.append(100.0 * valid / len(category_rows))
            ax.plot(x, coverage, marker="o", linewidth=1.8, markersize=4, label=label)
        ax.set_title(view_name.title())
        ax.set_xticks(x, [state_display_label(state) for state in states], rotation=45, ha="right")
        for tick, state in zip(ax.get_xticklabels(), states):
            tick.set_color(state_display_color(state)); tick.set_fontweight("bold")
        ax.set_ylim(0, 105); ax.set_xlabel("State"); ax.grid(axis="y", alpha=0.25)
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel("ROIs with valid activity (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("State-activity coverage by ROI category", y=1.02)
    fig.tight_layout()
    _save(fig, path)


def write_initial_analysis_figures(records: Sequence[Mapping[str, Any]], output_dir: Path, preferred_state_views: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    visual = Counter(str(row.get("visual_class") or "unknown") for row in records)
    preferred = Counter(str(row.get("preferred_state_class") or "unknown") for row in records)
    preferred_colors = {label: state_display_color(label) if label not in {"unknown", "tie"} else "#7f7f7f" for label in preferred}
    for filename, counts, title in (("visual_response_counts.png", visual, "Visual response classification"), ("preferred_state_counts.png", preferred, "Preferred-state classification")):
        path = output_dir / filename; _bar(counts, title, path, preferred_colors if "preferred" in filename else None); paths.append(str(path))
    category_path = output_dir / "visual_response_by_roi_category.png"; _visual_category_plot(records, category_path)
    if category_path.exists(): paths.append(str(category_path))
    category_order = [
        ("dendrite", "apical"), ("dendrite", "basal"),
        ("spine", "apical"), ("spine", "basal"),
    ]
    for roi_type, compartment in category_order:
        category_path = output_dir / f"preferred_state_proportions_{roi_type}_{compartment}.png"
        _preferred_state_proportion_plot(records, roi_type, compartment, category_path, preferred_state_views=preferred_state_views)
        paths.append(str(category_path))
    overlap = output_dir / "preferred_state_overlap.png"; _overlap((preferred_state_views or {}).get("combined", records), overlap)
    if overlap.exists(): paths.append(str(overlap))
    activity = output_dir / "activity_frequency_scatter.png"; _numeric_plot(records, activity); paths.append(str(activity))
    coverage = output_dir / "state_activity_coverage.png"; _state_coverage_plot(records, coverage, preferred_state_views=preferred_state_views)
    if coverage.exists(): paths.append(str(coverage))
    return paths
