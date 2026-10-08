from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def generate_preview(recipe: Dict[str, Any], truth: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        (output_dir / "preview_unavailable.txt").write_text("matplotlib is unavailable; truth manifests remain authoritative.\n", encoding="utf-8")
        return
    states = list(recipe["states"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    ax = axes[0, 0]
    x = np.arange(len(states))
    ax.bar(x - 0.18, [recipe["dendrites"]["basal"]["activity_scale"]] * len(states), width=0.18, label="basal")
    ax.bar(x, [recipe["dendrites"]["apical"]["activity_scale"]] * len(states), width=0.18, label="apical")
    ax.bar(x + 0.18, [recipe["somas"]["soma_1"]["activity_scale"]] * len(states), width=0.18, label="soma 1")
    ax.set_xticks(x, states, rotation=25)
    ax.set_title("Expected activity scales")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.bar(x - 0.15, [recipe["dendrites"]["basal"]["frequency_scale"]] * len(states), width=0.15, label="basal")
    ax.bar(x, [recipe["dendrites"]["apical"]["frequency_scale"]] * len(states), width=0.15, label="apical")
    ax.bar(x + 0.15, [recipe["somas"]["soma_2"]["frequency_scale"]] * len(states), width=0.15, label="soma 2")
    ax.set_xticks(x, states, rotation=25)
    ax.set_title("Expected frequency scales")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "expected_state_scales.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["soma_1", "soma_2", "bouton_1", "bouton_2", "bouton_3"]
    ax.axis("off")
    ax.text(0.05, 0.80, "soma_1  ─────────  bouton_1", fontsize=14)
    ax.text(0.05, 0.55, "soma_2  ─────────  bouton_3", fontsize=14)
    ax.text(0.05, 0.30, "bouton_2  independent control", fontsize=14)
    ax.set_title("Expected soma/bouton topology")
    fig.savefig(output_dir / "expected_topology.svg", bbox_inches="tight")
    plt.close(fig)
    (output_dir / "expected_topology.json").write_text(json.dumps({"labels": labels, "topology": truth.get("topology", {})}, indent=2) + "\n", encoding="utf-8")
