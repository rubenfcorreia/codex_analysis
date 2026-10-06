from __future__ import annotations

from typing import Any, Dict

import numpy as np

REPORT_SIGNIFICANCE_ALPHA = 0.05


def resolve_inferential_p_value(row: Dict[str, Any], requested_key: str = "declared") -> tuple[float, str]:
    source = str(row.get("p_value_source") or "").strip().lower()
    source_keys = {
        "circular_shift": "shuffle_p",
        "shuffle": "shuffle_p",
        "model_wald": "p_value",
        "vector_label_permutation": "p_value",
        "classical": "p_value",
        "none_descriptive": "",
        "descriptive": "",
    }
    key = source_keys.get(source, requested_key if requested_key != "declared" else "")
    if not key:
        return float("nan"), source or "unavailable"
    try:
        adjusted = float(row.get("adjusted_pvalue", float("nan")))
    except (TypeError, ValueError):
        adjusted = float("nan")
    if np.isfinite(adjusted):
        return adjusted, (source or key) + "_adjusted"
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        value = float("nan")
    return value, source or key


def is_significant_row(row: Dict[str, Any], alpha: float = REPORT_SIGNIFICANCE_ALPHA, p_key: str = "shuffle_p") -> bool:
    declared_source = str(row.get("p_value_source") or "").strip().lower()
    if declared_source and p_key == "shuffle_p" and declared_source not in {"shuffle", "circular_shift"}:
        return False
    if declared_source and p_key in {"p_value", "classical_p"} and declared_source not in {"classical", "model_wald", "vector_label_permutation"}:
        return False
    try:
        p_value = float(row.get(p_key, float("nan")))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(p_value) and p_value < alpha)


def apply_bh_fdr_rows(rows, *, p_key: str = "p_value", adjusted_key: str = "adjusted_pvalue") -> None:
    """Apply BH-FDR within each declared correction family in-place."""
    families = {}
    for index, row in enumerate(rows):
        family = str(row.get("correction_family") or "unavailable")
        try:
            p_value = float(row.get(p_key, float("nan")))
        except (TypeError, ValueError):
            p_value = float("nan")
        if np.isfinite(p_value):
            families.setdefault(family, []).append((index, p_value))
    for members in families.values():
        ordered = sorted(members, key=lambda item: item[1])
        running = 1.0
        adjusted = {}
        count = len(ordered)
        for rank in range(count, 0, -1):
            index, p_value = ordered[rank - 1]
            running = min(running, p_value * count / rank)
            adjusted[index] = min(running, 1.0)
        for index, _ in members:
            rows[index][adjusted_key] = float(adjusted[index])
            rows[index]["correction_method"] = "bh_fdr"


__all__ = ["REPORT_SIGNIFICANCE_ALPHA", "apply_bh_fdr_rows", "is_significant_row", "resolve_inferential_p_value"]
