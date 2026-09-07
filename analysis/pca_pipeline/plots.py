from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analysis.compartment_common import ensure_dir


def save_pca_figures(root: Path, label: str, result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    out = ensure_dir(root)
    scores = np.asarray(result["scores"], dtype=float)
    variance = np.asarray(result["explained_variance_ratio"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, variance.size + 1), np.cumsum(variance), marker="o")
    ax.set(xlabel="Principal component", ylabel="Cumulative explained variance", ylim=(0, 1.05), title=label)
    fig.tight_layout()
    fig.savefig(out / f"{label}_explained_variance.png", dpi=160)
    plt.close(fig)
    if scores.shape[1] < 2:
        return
    for color_key, filename in (("state", "state"), ("visual_condition", "visual_condition")):
        labels = np.asarray([str(row.get(color_key, "unknown")) for row in rows])
        fig, ax = plt.subplots(figsize=(6, 5))
        for value in sorted(set(labels)):
            idx = labels == value
            ax.scatter(scores[idx, 0], scores[idx, 1], s=8, alpha=0.45, label=value)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{label}: {color_key}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{label}_PC1_PC2_{filename}.png", dpi=160)
        plt.close(fig)
    for color_key, filename in (("locomotion", "locomotion"), ("pupil_size", "pupil")):
        values = np.asarray([float(row.get(color_key, np.nan)) for row in rows])
        if not np.isfinite(values).any():
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_values = np.where(np.isfinite(values), values, np.nanmedian(values))
        scatter = ax.scatter(scores[:, 0], scores[:, 1], c=plot_values, s=8, alpha=0.5, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label=color_key)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{label}: {color_key}")
        fig.tight_layout()
        fig.savefig(out / f"{label}_PC1_PC2_{filename}.png", dpi=160)
        plt.close(fig)
