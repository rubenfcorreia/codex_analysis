from __future__ import annotations

from typing import Any, Dict

import numpy as np

REPORT_SIGNIFICANCE_ALPHA = 0.05


def is_significant_row(row: Dict[str, Any], alpha: float = REPORT_SIGNIFICANCE_ALPHA, p_key: str = "shuffle_p") -> bool:
    try:
        p_value = float(row.get(p_key, float("nan")))
    except Exception:
        return False
    return bool(np.isfinite(p_value) and p_value < alpha)


__all__ = ["REPORT_SIGNIFICANCE_ALPHA", "is_significant_row"]
