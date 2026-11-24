"""Dependency-free Venn/UpSet-style overlap renderers."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def venn_plot(state_sets: Mapping[str, set[str]], path: Path) -> None:
    states = sorted(state_sets)
    if len(states) > 3:
        raise ValueError("venn_plot supports at most three states")
    fig, ax = plt.subplots(figsize=(6, 5))
    positions = {2: [(0.38, 0.5), (0.62, 0.5)], 3: [(0.38, 0.62), (0.62, 0.62), (0.5, 0.36)], 1: [(0.5, 0.5)]}[len(states)]
    colors = ("#4c78a8", "#f58518", "#54a24b")
    for index, state in enumerate(states):
        circle = Circle(positions[index], 0.28, alpha=0.35, color=colors[index])
        ax.add_patch(circle)
        ax.text(positions[index][0], positions[index][1] + 0.33, state, ha="center", weight="bold")
    combinations_count = {}
    for size in range(1, len(states) + 1):
        for combo in combinations(states, size):
            included = set.intersection(*(state_sets[state] for state in combo))
            excluded = set.union(*(state_sets[state] for state in states if state not in combo), set())
            combinations_count[combo] = len(included - excluded)
    labels = {1: [(0.25, 0.5)], 2: [(0.5, 0.5)], 3: [(0.5, 0.54)]}
    for combo, count in combinations_count.items():
        if len(combo) == 1:
            index = states.index(combo[0]); x, y = positions[index][0] - 0.1, positions[index][1]
        elif len(combo) == 2:
            x, y = 0.5, 0.5
        else:
            x, y = 0.5, 0.55
        ax.text(x, y, str(count), ha="center", va="center", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title("Preferred-state overlap")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
