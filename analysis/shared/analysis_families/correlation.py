from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import time
from pathlib import Path
import sys

import numpy as np
from scipy import stats

from analysis.shared.cache_utils import array_signature, stable_hash
from analysis.shared.shuffle_dataset import ShuffleDataset

SUPPORTED_CORRELATION_METHODS = {"pearson"}
SUPPORTED_CORRELATION_INFERENCE = {"circular_shift", "none"}
SUPPORTED_NULL_MODELS = {"mask_aware", "full_trace"}


def _record_shuffle_dataset_usage(shared_shuffle_cache: Optional[Dict[str, Any]], *, path: str) -> None:
    if not isinstance(shared_shuffle_cache, dict):
        return
    usage = shared_shuffle_cache.setdefault("shuffle_dataset_usage", {"opened": False, "read_calls": 0, "signals_read": 0, "path": str(path)})
    usage["path"] = str(path)
    if not bool(usage.get("opened")):
        usage["opened"] = True
        print(f"SHUFFLE DATASET: consumer open | null_model=full_trace | path={path}", file=sys.stderr)


def validate_correlation_settings(method: str = "pearson", inference: str = "circular_shift") -> tuple[str, str]:
    method = str(method or "pearson").strip().lower()
    inference = str(inference or "circular_shift").strip().lower()
    if method not in SUPPORTED_CORRELATION_METHODS:
        raise ValueError(f"Unsupported correlation_method={method!r}; supported methods: pearson")
    if inference not in SUPPORTED_CORRELATION_INFERENCE:
        raise ValueError(f"Unsupported correlation_inference={inference!r}; supported methods: circular_shift, none")
    return method, inference


def validate_null_model(null_model: str = "mask_aware") -> str:
    """Validate the inferential null: mask-aware shuffles retained state samples, while full-trace shuffles complete traces before masking."""
    value = str(null_model or "mask_aware").strip().lower()
    if value not in SUPPORTED_NULL_MODELS:
        raise ValueError(f"Unsupported null_model={value!r}; supported models: mask_aware, full_trace")
    return value


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
    signal_keys: Optional[Tuple[Optional[str], Optional[str]]] = None,
    *,
    correlation_method: str = "pearson",
    correlation_inference: Optional[str] = None,
    shuffle_seed: int = 12345,
    min_shift_frames: int = 1,
    null_model: Optional[str] = None,
) -> Dict[str, Any]:
    method, configured_inference = validate_correlation_settings(
        correlation_method,
        correlation_inference or ("circular_shift" if use_circular_shift else "none"),
    )
    configured_null_model = validate_null_model(null_model or (shared_shuffle_cache or {}).get("null_model", "mask_aware"))
    a = np.asarray(trace_a, dtype=float).reshape(-1)
    b = np.asarray(trace_b, dtype=float).reshape(-1)
    usable = min(a.size, b.size)
    a, b = a[:usable], b[:usable]
    fixed_mask = np.isfinite(a) & np.isfinite(b)
    n_valid = int(fixed_mask.sum())
    base = {
        "correlation_method": method,
        "p_value_source": "circular_shift" if configured_inference == "circular_shift" else "none_descriptive",
        "null_model": configured_null_model,
        "shuffle_seed": int(shuffle_seed),
        "shuffle_n_requested": int(shuffle_n) if configured_inference == "circular_shift" else 0,
        "shuffle_n_success": 0,
        "n_valid_samples": n_valid,
        "inferential_unit": "temporal_circular_shift" if configured_inference == "circular_shift" else "descriptive",
        "correction_family": "correlation_time_series",
    }
    if signal_keys and isinstance(shared_shuffle_cache, dict) and shared_shuffle_cache.get("shuffle_dataset_path"):
        return correlation_analysis_for_batch(
            [(a, b)], int(shuffle_n), shared_shuffle_cache=shared_shuffle_cache,
            shared_shuffle_keys=[shared_shuffle_key], signal_keys=[signal_keys], result_keys=[shared_shuffle_key],
            correlation_method=method, correlation_inference=configured_inference,
            shuffle_seed=shuffle_seed, min_shift_frames=min_shift_frames, null_model=configured_null_model,
        )[0]
    null_cache = shared_shuffle_cache if isinstance(shared_shuffle_cache, dict) and shared_shuffle_key else None
    null_cache_key = None
    if null_cache is not None:
        null_cache_key = stable_hash({
            "schema": 1,
            "shared_shuffle_key": str(shared_shuffle_key),
            "trace_a": array_signature(a),
            "trace_b": array_signature(b),
            "finite_mask": array_signature(fixed_mask),
            "vector_length": int(usable),
            "correlation_method": method,
            "correlation_inference": configured_inference,
            "shuffle_n": int(shuffle_n),
            "shuffle_seed": int(shuffle_seed),
            "min_shift_frames": int(min_shift_frames),
            "null_model": configured_null_model,
        })
        null_results = null_cache.setdefault("null_results", {})
        cached_result = null_results.get(null_cache_key)
        cache_stats = null_cache.setdefault("null_result_stats", {"hits": 0, "misses": 0, "builds": 0})
        if isinstance(cached_result, dict):
            cache_stats["hits"] = int(cache_stats.get("hits", 0)) + 1
            return dict(cached_result)
        cache_stats["misses"] = int(cache_stats.get("misses", 0)) + 1

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        if null_cache is not None and null_cache_key is not None:
            null_cache.setdefault("null_results", {})[null_cache_key] = dict(result)
            stats = null_cache.setdefault("null_result_stats", {"hits": 0, "misses": 0, "builds": 0})
            stats["builds"] = int(stats.get("builds", 0)) + 1
        return result

    if n_valid < 3:
        return _finish({**base, "r": float("nan"), "effect_size": float("nan"), "classical_p": float("nan"), "shuffle_p": float("nan"), "p_value": float("nan"), "lower_ci": float("nan"), "upper_ci": float("nan"), "n": n_valid, "status": "insufficient_data"})

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
        shifts = np.asarray(shifts[: int(shuffle_n)], dtype=int)
        if shifts.size:
            # Evaluate all circular shifts in one bounded batch.  The
            # indexing is identical to np.roll(b, shift), while avoiding a
            # Python/scipy call for every null sample.
            indices = (np.arange(usable, dtype=int)[None, :] - shifts[:, None]) % usable
            shifted = b[indices]
            valid = np.isfinite(a)[None, :] & np.isfinite(shifted)
            counts = valid.sum(axis=1)
            a_values = np.where(valid, a[None, :], 0.0)
            b_values = np.where(valid, shifted, 0.0)
            a_mean = np.divide(a_values.sum(axis=1), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
            b_mean = np.divide(b_values.sum(axis=1), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
            a_centered = np.where(valid, a[None, :] - a_mean[:, None], 0.0)
            b_centered = np.where(valid, shifted - b_mean[:, None], 0.0)
            numerator = np.sum(a_centered * b_centered, axis=1)
            denominator = np.sqrt(np.sum(a_centered * a_centered, axis=1) * np.sum(b_centered * b_centered, axis=1))
            values = np.divide(numerator, denominator, out=np.full(shifts.shape, np.nan, dtype=float), where=(counts >= 3) & (denominator > 0))
            null.extend(float(value) for value in values if np.isfinite(value))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)) if null else float("nan")
    return _finish({
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
    })



def _batch_null_cache_key(
    a: np.ndarray,
    b: np.ndarray,
    fixed_mask: np.ndarray,
    usable: int,
    shared_shuffle_key: Optional[str],
    result_key: Optional[str],
    method: str,
    configured_inference: str,
    shuffle_n: int,
    shuffle_seed: int,
    min_shift_frames: int,
    null_model: str,
    mask_signature: Optional[str],
) -> Optional[str]:
    if not shared_shuffle_key:
        return None
    return stable_hash({
        "schema": 2,
        "shared_shuffle_key": str(shared_shuffle_key),
        "pair_identity": str(result_key) if result_key is not None else None,
        "trace_a": array_signature(a),
        "trace_b": array_signature(b),
        "finite_mask": array_signature(fixed_mask),
        "vector_length": int(usable),
        "correlation_method": method,
        "correlation_inference": configured_inference,
        "shuffle_n": int(shuffle_n),
        "shuffle_seed": int(shuffle_seed),
        "min_shift_frames": int(min_shift_frames),
        "null_model": str(null_model),
        "mask_signature": mask_signature,
    })


def _batch_shift_schedule(
    shared_shuffle_cache: Optional[Dict[str, Any]],
    shared_shuffle_key: Optional[str],
    usable: int,
    shuffle_n: int,
    shuffle_seed: int,
    min_shift_frames: int,
) -> np.ndarray:
    schedule_stats = None
    if isinstance(shared_shuffle_cache, dict):
        schedule_stats = shared_shuffle_cache.setdefault("schedule_stats", {"hits": 0, "misses": 0, "builds": 0})
    if shared_shuffle_cache is not None and shared_shuffle_key:
        entry = shared_shuffle_cache.setdefault("entries", {}).get(shared_shuffle_key)
        if isinstance(entry, dict):
            existing = np.asarray(entry.get("shifts", []), dtype=int)
            if existing.size >= shuffle_n:
                if schedule_stats is not None:
                    schedule_stats["hits"] = int(schedule_stats.get("hits", 0)) + 1
                return existing[:shuffle_n]
    if schedule_stats is not None:
        schedule_stats["misses"] = int(schedule_stats.get("misses", 0)) + 1
    minimum = max(1, min(int(min_shift_frames), usable - 1))
    if minimum >= usable or shuffle_n <= 0:
        shifts = np.array([], dtype=int)
    else:
        shifts = np.random.default_rng(int(shuffle_seed)).integers(minimum, usable, size=int(shuffle_n))
    if shared_shuffle_cache is not None and shared_shuffle_key:
        shared_shuffle_cache.setdefault("entries", {})[shared_shuffle_key] = {
            "key": str(shared_shuffle_key),
            "vector_length": int(usable),
            "shuffle_n": int(shuffle_n),
            "shuffle_seed": int(shuffle_seed),
            "min_shift_frames": int(min_shift_frames),
            "shifts": shifts,
        }
    return shifts


def correlation_analysis_for_batch(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    shuffle_n: int,
    *,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    shared_shuffle_keys: Optional[Sequence[Optional[str]]] = None,
    signal_keys: Optional[Sequence[Tuple[Any, Any]]] = None,
    signal_masks: Optional[Sequence[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    result_keys: Optional[Sequence[Optional[str]]] = None,
    correlation_method: str = "pearson",
    correlation_inference: Optional[str] = None,
    shuffle_seed: int = 12345,
    min_shift_frames: int = 1,
    null_model: Optional[str] = None,
    max_batch_size: int = 64,
    shift_block_size: int = 8,
) -> List[Dict[str, Any]]:
    """Calculate exact per-pair correlations in bounded vectorized batches.

    The null values are never pooled: each pair receives its own null vector
    and p-value. Compatible pairs only share the seeded shift schedule and
    matrix execution. ``max_batch_size`` bounds the temporary ``pairs x time``
    arrays held in memory.
    """
    method, configured_inference = validate_correlation_settings(
        correlation_method,
        correlation_inference or "circular_shift",
    )
    configured_null_model = validate_null_model(null_model or (shared_shuffle_cache or {}).get("null_model", "mask_aware"))
    keys = list(shared_shuffle_keys or [None] * len(pairs))
    if len(keys) != len(pairs):
        raise ValueError("shared_shuffle_keys must have one entry per pair")
    result_key_list = list(result_keys or [None] * len(pairs))
    if len(result_key_list) != len(pairs):
        raise ValueError("result_keys must have one entry per pair")
    signal_key_list = list(signal_keys or [(None, None)] * len(pairs))
    if len(signal_key_list) != len(pairs):
        raise ValueError("signal_keys must have one entry per pair")
    signal_mask_list = list(signal_masks or [(None, None)] * len(pairs))
    if len(signal_mask_list) != len(pairs):
        raise ValueError("signal_masks must have one entry per pair")
    batch_size = max(1, int(max_batch_size))
    results: List[Optional[Dict[str, Any]]] = [None] * len(pairs)
    pending: List[Dict[str, Any]] = []
    dataset = None
    if configured_null_model == "full_trace" and isinstance(shared_shuffle_cache, dict) and shared_shuffle_cache.get("shuffle_dataset_path"):
        dataset_path = str(shared_shuffle_cache["shuffle_dataset_path"])
        metadata = dict(shared_shuffle_cache.get("shuffle_dataset_metadata") or {})
        try:
            if not Path(dataset_path).exists():
                raise FileNotFoundError(f"SHUFFLE DATASET: validation failure | reason=missing | action=reject | path={dataset_path}")
            dataset = ShuffleDataset(dataset_path, metadata, readonly=True)
            _record_shuffle_dataset_usage(shared_shuffle_cache, path=dataset_path)
        except (OSError, ValueError, RuntimeError):
            if Path(dataset_path).exists():
                raise
            dataset = None
    cache_stats = None
    if isinstance(shared_shuffle_cache, dict):
        cache_stats = shared_shuffle_cache.setdefault("null_result_stats", {"hits": 0, "misses": 0, "builds": 0})
    if dataset is not None:
        requested_keys = [key for pair_keys in signal_key_list for key in pair_keys if key is not None]
        dataset.validate_requested_keys(requested_keys)
    for index, (trace_a, trace_b) in enumerate(pairs):
        a = np.asarray(trace_a, dtype=float).reshape(-1)
        b = np.asarray(trace_b, dtype=float).reshape(-1)
        usable = min(a.size, b.size)
        a = a[:usable]
        b = b[:usable]
        fixed_mask = np.isfinite(a) & np.isfinite(b)
        n_valid = int(fixed_mask.sum())
        base = {
            "correlation_method": method,
            "p_value_source": "circular_shift" if configured_inference == "circular_shift" else "none_descriptive",
            "null_model": configured_null_model,
            "shuffle_seed": int(shuffle_seed),
            "shuffle_n_requested": int(shuffle_n) if configured_inference == "circular_shift" else 0,
            "shuffle_n_success": 0,
            "n_valid_samples": n_valid,
            "inferential_unit": "temporal_circular_shift" if configured_inference == "circular_shift" else "descriptive",
            "correction_family": "correlation_time_series",
        }
        mask_signature = stable_hash({
            "a": array_signature(signal_mask_list[index][0]),
            "b": array_signature(signal_mask_list[index][1]),
        })
        cache_key = _batch_null_cache_key(a, b, fixed_mask, usable, keys[index], result_key_list[index], method, configured_inference, shuffle_n, shuffle_seed, min_shift_frames, configured_null_model, mask_signature)
        if isinstance(shared_shuffle_cache, dict) and cache_key is not None:
            cached = shared_shuffle_cache.setdefault("null_results", {}).get(cache_key)
            if isinstance(cached, dict):
                cache_stats["hits"] = int(cache_stats.get("hits", 0))
                cache_stats["hits"] += 1
                results[index] = dict(cached)
                continue
            cache_stats["misses"] = int(cache_stats.get("misses", 0)) + 1
        if n_valid < 3:
            result = {**base, "r": float("nan"), "effect_size": float("nan"), "classical_p": float("nan"), "shuffle_p": float("nan"), "p_value": float("nan"), "lower_ci": float("nan"), "upper_ci": float("nan"), "n": n_valid, "status": "insufficient_data"}
            results[index] = result
            if isinstance(shared_shuffle_cache, dict) and cache_key is not None:
                shared_shuffle_cache.setdefault("null_results", {})[cache_key] = dict(result)
                cache_stats["builds"] = int(cache_stats.get("builds", 0)) + 1
            continue
        observed_result = stats.pearsonr(a[fixed_mask], b[fixed_mask])
        observed = float(observed_result.statistic)
        lower_ci, upper_ci = _fisher_ci(observed, n_valid)
        pending.append({
            "index": index,
            "a": a,
            "b": b,
            "fixed_mask": fixed_mask,
            "usable": usable,
            "observed": observed,
            "classical_p": float(observed_result.pvalue),
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "base": base,
            "cache_key": cache_key,
            "shifts": _batch_shift_schedule(shared_shuffle_cache, keys[index], usable, int(shuffle_n), int(shuffle_seed), int(min_shift_frames)),
            "signal_a_key": signal_key_list[index][0],
            "signal_b_key": signal_key_list[index][1],
            "signal_masks": signal_mask_list[index],
        })

    batch_diagnostics = None
    if isinstance(shared_shuffle_cache, dict):
        batch_diagnostics = shared_shuffle_cache.setdefault(
            "null_batch_diagnostics",
            {"null_model": configured_null_model, "batches": 0, "pairs": 0, "elapsed_s": 0.0, "max_batch_size": 0, "shift_block_size": int(shift_block_size), "vector_lengths": {}},
        )
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for item in pending:
        group_key = (
            int(item["usable"]),
            str(item["base"]["correlation_method"]),
            str(item["base"]["null_model"]),
            int(shuffle_n),
            int(shuffle_seed),
            int(min_shift_frames),
            () if dataset is not None else tuple(np.asarray(item["shifts"], dtype=int).tolist()),
        )
        groups.setdefault(group_key, []).append(item)

    for group_items in groups.values():
        for start in range(0, len(group_items), batch_size):
            chunk = group_items[start:start + batch_size]
            chunk_started = time.perf_counter()
            shifts = np.asarray(chunk[0]["shifts"], dtype=int)
            a_stack = np.stack([item["a"] for item in chunk], axis=0)
            b_stack = np.stack([item["b"] for item in chunk], axis=0)
            null_values: List[List[float]] = [[] for _ in chunk]
            block_size = max(1, int(shift_block_size))
            if configured_null_model == "full_trace" and dataset is not None and all(item.get("signal_a_key") and item.get("signal_b_key") for item in chunk):
                # A batch often contains the same soma/behaviour signal in many
                # pairs. Keep this cache bounded to the current batch so repeated
                # pairs reuse one HDF5 decompression without retaining the full
                # dataset in memory.
                loaded_signal_arrays = {}
                def _load_signal(signal_key, signal_mask):
                    keys = signal_key if isinstance(signal_key, (tuple, list)) else (signal_key,)
                    canonical_keys = tuple(str(key) for key in keys)
                    if canonical_keys not in loaded_signal_arrays:
                        arrays = [dataset.read_signal(key, int(shuffle_n)) for key in canonical_keys]
                        if isinstance(shared_shuffle_cache, dict):
                            usage = shared_shuffle_cache.setdefault("shuffle_dataset_usage", {})
                            usage["signals_read"] = int(usage.get("signals_read", 0)) + len(canonical_keys)
                            usage["read_calls"] = int(usage.get("read_calls", 0)) + 1
                        loaded_signal_arrays[canonical_keys] = (
                            np.nanmean(np.stack(arrays, axis=0), axis=0) if len(arrays) > 1 else arrays[0]
                        )
                    array = loaded_signal_arrays[canonical_keys]
                    if signal_mask is not None:
                        mask = np.asarray(signal_mask, dtype=bool).reshape(-1)
                        usable_mask = mask[:array.shape[1]]
                        array = array[:, usable_mask]
                    return array
                a_null_stack = np.stack([_load_signal(item["signal_a_key"], item["signal_masks"][0]) for item in chunk], axis=0)
                b_null_stack = np.stack([_load_signal(item["signal_b_key"], item["signal_masks"][1]) for item in chunk], axis=0)
                null_count = min(int(shuffle_n), a_null_stack.shape[1], b_null_stack.shape[1])
                for shift_start in range(0, null_count, block_size):
                    a_null = a_null_stack[:, shift_start:shift_start + block_size, :]
                    b_null = b_null_stack[:, shift_start:shift_start + block_size, :]
                    valid = np.isfinite(a_null) & np.isfinite(b_null)
                    counts = valid.sum(axis=2)
                    a_values = np.where(valid, a_null, 0.0)
                    b_values = np.where(valid, b_null, 0.0)
                    a_mean = np.divide(a_values.sum(axis=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
                    b_mean = np.divide(b_values.sum(axis=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
                    a_centered = np.where(valid, a_null - a_mean[:, :, None], 0.0)
                    b_centered = np.where(valid, b_null - b_mean[:, :, None], 0.0)
                    numerator = np.sum(a_centered * b_centered, axis=2)
                    denominator = np.sqrt(np.sum(a_centered * a_centered, axis=2) * np.sum(b_centered * b_centered, axis=2))
                    values = np.divide(numerator, denominator, out=np.full((len(chunk), a_null.shape[1]), np.nan, dtype=float), where=(counts >= 3) & (denominator > 0))
                    for local_index, pair_values in enumerate(values):
                        null_values[local_index].extend(float(value) for value in pair_values if np.isfinite(value))
            else:
                for shift_start in range(0, len(shifts), block_size):
                    shift_block = shifts[shift_start:shift_start + block_size]
                    shifted = np.stack([
                        b_stack[:, (np.arange(a_stack.shape[1], dtype=int) - int(shift)) % a_stack.shape[1]
                    ] for shift in shift_block], axis=1)
                    valid = np.isfinite(a_stack)[:, None, :] & np.isfinite(shifted)
                    counts = valid.sum(axis=2)
                    a_values = np.where(valid, a_stack[:, None, :], 0.0)
                    b_values = np.where(valid, shifted, 0.0)
                    a_mean = np.divide(a_values.sum(axis=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
                    b_mean = np.divide(b_values.sum(axis=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
                    a_centered = np.where(valid, a_stack[:, None, :] - a_mean[:, :, None], 0.0)
                    b_centered = np.where(valid, shifted - b_mean[:, :, None], 0.0)
                    numerator = np.sum(a_centered * b_centered, axis=2)
                    denominator = np.sqrt(np.sum(a_centered * a_centered, axis=2) * np.sum(b_centered * b_centered, axis=2))
                    values = np.divide(numerator, denominator, out=np.full((len(chunk), len(shift_block)), np.nan, dtype=float), where=(counts >= 3) & (denominator > 0))
                    for local_index, pair_values in enumerate(values):
                        null_values[local_index].extend(float(value) for value in pair_values if np.isfinite(value))
            if batch_diagnostics is not None:
                batch_diagnostics["batches"] = int(batch_diagnostics.get("batches", 0)) + 1
                batch_diagnostics["pairs"] = int(batch_diagnostics.get("pairs", 0)) + len(chunk)
                batch_diagnostics["elapsed_s"] = float(batch_diagnostics.get("elapsed_s", 0.0)) + (time.perf_counter() - chunk_started)
                batch_diagnostics["max_batch_size"] = max(int(batch_diagnostics.get("max_batch_size", 0)), len(chunk))
                lengths = batch_diagnostics.setdefault("vector_lengths", {})
                label = str(int(a_stack.shape[1]))
                lengths[label] = int(lengths.get(label, 0)) + len(chunk)
            for item, null in zip(chunk, null_values):
                observed = float(item["observed"])
                shuffle_p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)) if null else float("nan")
                result = {
                    **item["base"],
                    "r": observed,
                    "effect_size": observed,
                    "classical_p": item["classical_p"],
                    "shuffle_p": shuffle_p,
                    "p_value": shuffle_p if configured_inference == "circular_shift" else float("nan"),
                    "lower_ci": item["lower_ci"],
                    "upper_ci": item["upper_ci"],
                    "n": int(item["fixed_mask"].sum()),
                    "shuffle_n_success": int(len(null)),
                    "status": "ok" if configured_inference == "none" or null else "insufficient_null_samples",
                }
                results[item["index"]] = result
                if isinstance(shared_shuffle_cache, dict) and item["cache_key"] is not None:
                    shared_shuffle_cache.setdefault("null_results", {})[item["cache_key"]] = dict(result)
                    cache_stats["builds"] = int(cache_stats.get("builds", 0)) + 1
    if dataset is not None:
        dataset.close(complete=True)
    return [result if result is not None else {} for result in results]


__all__ = ["SUPPORTED_CORRELATION_INFERENCE", "SUPPORTED_CORRELATION_METHODS", "SUPPORTED_NULL_MODELS", "correlation_analysis_for_observation", "correlation_analysis_for_batch", "validate_correlation_settings", "validate_null_model"]
