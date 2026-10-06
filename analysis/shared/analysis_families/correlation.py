from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

SUPPORTED_CORRELATION_METHODS = {"pearson"}
SUPPORTED_CORRELATION_INFERENCE = {"circular_shift", "none"}


def validate_correlation_settings(method: str = "pearson", inference: str = "circular_shift") -> tuple[str, str]:
    method = str(method or "pearson").strip().lower()
    inference = str(inference or "circular_shift").strip().lower()
    if method not in SUPPORTED_CORRELATION_METHODS:
        raise ValueError(f"Unsupported correlation_method={method!r}; supported methods: pearson")
    if inference not in SUPPORTED_CORRELATION_INFERENCE:
        raise ValueError(f"Unsupported correlation_inference={inference!r}; supported methods: circular_shift, none")
    return method, inference


def _fisher_ci(r_value: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r_value) or n <= 3 or abs(r_value) >= 1.0:
        return float("nan"), float("nan")
    z = float(np.arctanh(np.clip(r_value, -0.999999, 0.999999)))
    half = 1.96 / float(np.sqrt(n - 3))
    return tuple(float(value) for value in (np.tanh(z - half), np.tanh(z + half)))


def correlation_analysis_for_observation(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    shuffle_n: int,
    use_circular_shift: bool = True,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    shared_shuffle_key: Optional[str] = None,
    *,
    correlation_method: str = "pearson",
    correlation_inference: Optional[str] = None,
    shuffle_seed: int = 12345,
    min_shift_frames: int = 1,
) -> Dict[str, Any]:
    method, configured_inference = validate_correlation_settings(
        correlation_method,
        correlation_inference or ("circular_shift" if use_circular_shift else "none"),
    )
    a = np.asarray(trace_a, dtype=float).reshape(-1)
    b = np.asarray(trace_b, dtype=float).reshape(-1)
    usable = min(a.size, b.size)
    a, b = a[:usable], b[:usable]
    fixed_mask = np.isfinite(a) & np.isfinite(b)
    n_valid = int(fixed_mask.sum())
    base = {
        "correlation_method": method,
        "p_value_source": "circular_shift" if configured_inference == "circular_shift" else "none_descriptive",
        "null_model": configured_inference,
        "shuffle_seed": int(shuffle_seed),
        "shuffle_n_requested": int(shuffle_n) if configured_inference == "circular_shift" else 0,
        "shuffle_n_success": 0,
        "n_valid_samples": n_valid,
        "inferential_unit": "temporal_circular_shift" if configured_inference == "circular_shift" else "descriptive",
        "correction_family": "correlation_time_series",
    }
    if n_valid < 3:
        return {**base, "r": float("nan"), "effect_size": float("nan"), "classical_p": float("nan"), "shuffle_p": float("nan"), "p_value": float("nan"), "lower_ci": float("nan"), "upper_ci": float("nan"), "n": n_valid, "status": "insufficient_data"}

    observed_result = stats.pearsonr(a[fixed_mask], b[fixed_mask])
    observed = float(observed_result.statistic)
    lower_ci, upper_ci = _fisher_ci(observed, n_valid)
    null: list[float] = []
    if configured_inference == "circular_shift" and int(shuffle_n) > 0 and usable > 1:
        shifts = None
        if shared_shuffle_cache is not None and shared_shuffle_key is not None:
            entry = shared_shuffle_cache.get("entries", {}).get(shared_shuffle_key)
            if isinstance(entry, dict):
                shifts = np.asarray(entry.get("shifts", []), dtype=int)
        if shifts is None or shifts.size == 0:
            rng = np.random.default_rng(int(shuffle_seed))
            minimum = max(1, min(int(min_shift_frames), usable - 1))
            if minimum >= usable:
                shifts = np.array([], dtype=int)
            else:
                shifts = rng.integers(minimum, usable, size=int(shuffle_n))
        for shift in shifts[: int(shuffle_n)]:
            shifted = np.roll(b, int(shift))
            shift_mask = np.isfinite(a) & np.isfinite(shifted)
            if int(shift_mask.sum()) < 3:
                continue
            null.append(float(stats.pearsonr(a[shift_mask], shifted[shift_mask]).statistic))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)) if null else float("nan")
    return {
        **base,
        "r": observed,
        "effect_size": observed,
        "classical_p": float(observed_result.pvalue),
        "shuffle_p": shuffle_p,
        "p_value": shuffle_p if configured_inference == "circular_shift" else float("nan"),
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "n": n_valid,
        "shuffle_n_success": int(len(null)),
        "status": "ok" if configured_inference == "none" or null else "insufficient_null_samples",
    }


__all__ = ["SUPPORTED_CORRELATION_INFERENCE", "SUPPORTED_CORRELATION_METHODS", "correlation_analysis_for_observation", "validate_correlation_settings"]
