"""General initial-analysis figures, including state-overlap summaries."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from analysis.cache_builders.overlap_plots import venn_plot


def _bar(counts: Mapping[str, int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(counts); values = [counts[label] for label in labels]
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title); ax.set_ylabel("ROI count"); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


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
    states = sorted(sets)
    if not states:
        return
    if len(states) <= 3:
        venn_plot(sets, path)
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


def _heatmap(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    states = sorted({str(state) for row in records for state in (row.get("state_activity") or {})})
    if not states or not records:
        return
    values = np.full((len(records), len(states)), np.nan)
    for i, row in enumerate(records):
        for j, state in enumerate(states):
            value = (row.get("state_activity") or {}).get(state)
            if value is not None:
                values[i, j] = float(value)
    fig, ax = plt.subplots(figsize=(7, max(4, min(12, len(records) * 0.18))))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("State"); ax.set_ylabel("ROI"); ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
    fig.colorbar(image, ax=ax, label="Activity"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


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
    ax.set_xlabel("State"); ax.set_ylabel("ROI"); ax.set_xticks(range(len(states)), states, rotation=35, ha="right")
    fig.colorbar(image, ax=ax, label="Event frequency / min"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_initial_analysis_figures(records: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    visual = Counter(str(row.get("visual_class") or "unknown") for row in records)
    preferred = Counter(str(row.get("preferred_state_class") or "unknown") for row in records)
    for filename, counts, title in (("visual_response_counts.png", visual, "Visual response classification"), ("preferred_state_counts.png", preferred, "Preferred-state classification")):
        path = output_dir / filename; _bar(counts, title, path); paths.append(str(path))
    overlap = output_dir / "preferred_state_overlap.png"; _overlap(records, overlap)
    if overlap.exists(): paths.append(str(overlap))
    activity = output_dir / "activity_frequency_scatter.png"; _numeric_plot(records, activity); paths.append(str(activity))
    heatmap = output_dir / "state_activity_heatmap.png"; _heatmap(records, heatmap)
    if heatmap.exists(): paths.append(str(heatmap))
    frequency_heatmap = output_dir / "state_frequency_heatmap.png"; _frequency_heatmap(records, frequency_heatmap)
    if frequency_heatmap.exists(): paths.append(str(frequency_heatmap))
    return paths
