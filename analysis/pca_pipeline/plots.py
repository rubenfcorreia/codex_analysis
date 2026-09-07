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
    cumulative_variance = np.cumsum(variance)
    display_count = int(np.searchsorted(cumulative_variance, 0.5, side="left") + 1) if np.any(cumulative_variance >= 0.5) else variance.size
    display_count = max(1, min(display_count, variance.size))
    display_variance = variance[:display_count]
    display_cumulative = cumulative_variance[:display_count]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, display_count + 1), display_cumulative * 100.0, marker="o")
    ax.axhline(50.0, color="#e45756", linestyle="--", linewidth=1)
    ax.set(xlabel="Principal component", ylabel="Cumulative explained variance (%)", ylim=(0, 100), title=f"{label}: variance to 50%")
    fig.tight_layout()
    fig.savefig(out / f"{label}_explained_variance.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    pc_numbers = np.arange(1, display_count + 1)
    ax.bar(pc_numbers, display_variance * 100.0, color="#4c78a8", alpha=0.75, label="Individual")
    ax.plot(pc_numbers, display_cumulative * 100.0, color="#e45756", marker="o", label="Cumulative")
    ax.axhline(50.0, color="#e45756", linestyle="--", linewidth=1)
    ax.set(xlabel="Number of PCs", ylabel="Explained variance (%)", title=f"{label}: variance to 50%")
    ax.set_xticks(pc_numbers)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{label}_variance_by_PC.png", dpi=160)
    plt.close(fig)
    if scores.shape[1] < 2:
        return
    n_components = min(scores.shape[1], variance.size)
    fig, axes = plt.subplots(n_components, n_components, figsize=(2.4 * n_components, 2.4 * n_components), squeeze=False)
    for row_idx in range(n_components):
        for col_idx in range(n_components):
            ax = axes[row_idx, col_idx]
            if row_idx == col_idx:
                ax.hist(scores[:, col_idx], bins=30, color="#4c78a8", alpha=0.8)
                ax.set_title(f"PC{col_idx + 1}: {variance[col_idx] * 100:.1f}%", fontsize=9)
            else:
                ax.scatter(scores[:, col_idx], scores[:, row_idx], s=4, alpha=0.35, color="#4c78a8", rasterized=True)
            if row_idx == n_components - 1:
                ax.set_xlabel(f"PC{col_idx + 1}")
            if col_idx == 0:
                ax.set_ylabel(f"PC{row_idx + 1}")
            ax.tick_params(labelsize=7)
    fig.suptitle(f"{label}: all principal components", y=1.0)
    fig.tight_layout()
    fig.savefig(out / f"{label}_all_PCs.png", dpi=160)
    plt.close(fig)
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
