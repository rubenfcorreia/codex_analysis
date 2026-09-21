from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from analysis.shared.plots.figure_io import save_figure


def _model(results: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = results.get(key, {}) if isinstance(results, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _summary_rows(results: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    summary = _model(results, key).get("summary_rows", {})
    if not isinstance(summary, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for response, values in summary.items():
        if isinstance(values, list):
            for row in values:
                if isinstance(row, Mapping):
                    rows.append({**dict(row), "response": row.get("response") or response})
    return rows


def plot_mixed_model_forest_figure(results: Dict[str, Any], fig_dir: Path, output_name: str = "mixed_model_forest.svg", title: Optional[str] = None, model_key: str = "mixed_model") -> Optional[str]:
    rows = _summary_rows(results, model_key)
    if not rows:
        return None
    labels = [str(row.get("term") or row.get("contrast") or row.get("response") or "effect") for row in rows]
    estimates = [float(row.get("estimate", row.get("mean", np.nan))) for row in rows]
    finite = np.isfinite(estimates)
    if not finite.any():
        return None
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.28 * int(finite.sum()) + 1.5)))
    y = np.arange(len(estimates))[finite]
    ax.axvline(0.0, color="#888888", linewidth=0.8)
    ax.scatter(np.asarray(estimates)[finite], y, color="#4C78A8", s=24)
    ax.set_yticks(y); ax.set_yticklabels(np.asarray(labels, dtype=object)[finite]); ax.set_title(title or "Mixed-model effects"); ax.set_xlabel("Estimate")
    fig.tight_layout()
    output = Path(fig_dir) / output_name
    save_figure(fig, output, extra_formats=())
    plt.close(fig)
    return str(output)


def plot_mixed_model_predicted_means_figure(results: Dict[str, Any], fig_dir: Path, output_name: str = "mixed_model_predicted_means.svg", title: Optional[str] = None, model_key: str = "mixed_model") -> Optional[str]:
    rows = _summary_rows(results, model_key)
    if not rows:
        return None
    labels = [str(row.get("state") or row.get("term") or row.get("response") or "state") for row in rows]
    values = [float(row.get("predicted_mean", row.get("mean", row.get("estimate", np.nan)))) for row in rows]
    finite = np.isfinite(values)
    if not finite.any():
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(np.arange(int(finite.sum())), np.asarray(values)[finite], color="#4C78A8")
    ax.set_xticks(np.arange(int(finite.sum()))); ax.set_xticklabels(np.asarray(labels, dtype=object)[finite], rotation=35, ha="right")
    ax.set_title(title or "Mixed-model predicted means"); ax.set_ylabel("Predicted mean")
    fig.tight_layout()
    output = Path(fig_dir) / output_name
    save_figure(fig, output, extra_formats=())
    plt.close(fig)
    return str(output)


def plot_mixed_model_contrasts_checkpoint(results: Dict[str, Any], fig_dir: Path, scope: str, output_name: Optional[str] = None, title: Optional[str] = None, model_key: str = "mixed_model") -> Optional[str]:
    model = _model(results, model_key)
    rows = [dict(row) for row in model.get("contrast_rows", []) if isinstance(row, Mapping) and (not scope or str(row.get("scope")) == str(scope))]
    if not rows:
        return None
    labels = [str(row.get("contrast_name") or row.get("contrast_type") or row.get("response") or "contrast") for row in rows]
    values = [float(row.get("estimate", row.get("effect_size", np.nan))) for row in rows]
    finite = np.isfinite(values)
    if not finite.any():
        return None
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.28 * int(finite.sum()) + 1.5)))
    ax.axvline(0.0, color="#888888", linewidth=0.8)
    ax.barh(np.arange(int(finite.sum())), np.asarray(values)[finite], color="#72B7B2")
    ax.set_yticks(np.arange(int(finite.sum()))); ax.set_yticklabels(np.asarray(labels, dtype=object)[finite]); ax.set_title(title or "Mixed-model contrasts"); ax.set_xlabel("Estimate")
    fig.tight_layout()
    output = Path(fig_dir) / (output_name or "mixed_model_contrasts.svg")
    save_figure(fig, output, extra_formats=())
    plt.close(fig)
    return str(output)
