from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .figure_io import save_figure


def plot_preferred_state_concordance(results: Dict[str, Any], fig_dir: Path, *, output_name: str = "preferred_state_concordance.svg", title: str = "Dendrite–spine preferred-state concordance") -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    family = results.get("preferred_state_concordance", {})
    rows = family.get("confusion_rows", []) if isinstance(family, dict) else []
    dendrite_rows = family.get("dendrite_rows", []) if isinstance(family, dict) else []
    states = list(family.get("state_labels", [])) if isinstance(family, dict) else []
    states = states or sorted({str(row.get("dendrite_preferred_state")) for row in rows} | {str(row.get("spine_preferred_state")) for row in rows})
    if not states:
        return None
    matrix = np.zeros((len(states), len(states)), dtype=int)
    index = {state: i for i, state in enumerate(states)}
    for row in rows:
        d, s = str(row.get("dendrite_preferred_state")), str(row.get("spine_preferred_state"))
        if d in index and s in index:
            matrix[index[d], index[s]] = int(row.get("n_spines", 0) or 0)
    fractions = [float(row["matching_spine_fraction"]) for row in dendrite_rows if row.get("matching_spine_fraction") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks(range(len(states)), states, rotation=45, ha="right")
    axes[0].set_yticks(range(len(states)), states)
    axes[0].set_xlabel("Spine preferred state")
    axes[0].set_ylabel("Dendrite preferred state")
    for i in range(len(states)):
        for j in range(len(states)):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=axes[0], label="Spines")
    axes[1].hist(fractions, bins=np.linspace(0, 1, 11), color="#4477AA", edgecolor="white")
    axes[1].set(xlim=(0, 1), xlabel="Matching-spine fraction per dendrite", ylabel="Dendrites")
    fig.suptitle(title)
    return save_figure(fig, Path(fig_dir) / output_name)


__all__ = ["plot_preferred_state_concordance"]
