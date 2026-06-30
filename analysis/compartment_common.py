"""Shared helpers for compartment comparison pipelines.

This module intentionally stays lightweight so both the spine/dendrite and
soma/bouton pipelines can reuse the same experiment loading, state handling,
and summary helpers without pulling in a large orchestration script.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:  # pragma: no cover - imported in the live repo when available
    from analysis.main_pipeline.sleep_dendrite_spine_pipeline import (  # type: ignore
        build_state_masks_movie,
        build_state_masks_sleep,
        canonical_state_label,
        ensure_dir,
        estimate_sampling_rate,
        grouped_experiments_by_day,
        interpolate_series,
        make_day_id,
        resolve_analysis_state_selections,
        safe_filename_component,
        state_display_color,
        state_display_label,
        state_family_label,
        write_csv_rows,
        write_json_file,
    )
except Exception:  # pragma: no cover - local fallback for portability
    def ensure_dir(path: Path | str) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def safe_filename_component(value: Any) -> str:
        text = str(value).strip()
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("._") or "value"


    def canonical_state_label(label: Any) -> str:
        return safe_filename_component(label).lower()


    def state_family_label(label: Any) -> str:
        return canonical_state_label(label)


    def state_display_label(label: Any) -> str:
        return str(label).replace("_", " ").strip().title()


    def state_display_color(label: Any) -> str:
        palette = {
            "all": "#4c78a8",
            "run": "#f58518",
            "running": "#f58518",
            "still": "#54a24b",
            "quiet": "#72b7b2",
            "nrem": "#b279a2",
            "rem": "#e45756",
            "wake": "#ff9da6",
            "active": "#9d755d",
            "inactive": "#bab0ab",
        }
        return palette.get(canonical_state_label(label), "#4c78a8")


    def write_csv_rows(path: Path | str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})


    def write_json_file(path: Path | str, payload: Mapping[str, Any]) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")


    def estimate_sampling_rate(t: Sequence[float] | np.ndarray) -> float:
        t = np.asarray(t, dtype=float)
        if t.size < 2:
            return float("nan")
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return float("nan")
        return float(1.0 / np.median(dt))


    def interpolate_series(
        source_t: Sequence[float] | np.ndarray,
        source_y: Sequence[float] | np.ndarray,
        target_t: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        source_t = np.asarray(source_t, dtype=float)
        source_y = np.asarray(source_y, dtype=float)
        target_t = np.asarray(target_t, dtype=float)
        if source_t.size == 0 or source_y.size == 0:
            return np.full_like(target_t, np.nan, dtype=float)
        valid = np.isfinite(source_t) & np.isfinite(source_y)
        if valid.sum() < 2:
            return np.full_like(target_t, np.nan, dtype=float)
        return np.interp(target_t, source_t[valid], source_y[valid], left=np.nan, right=np.nan)


    def make_day_id(animal_id: str, date: str) -> str:
        return f"{safe_filename_component(animal_id)}_{safe_filename_component(date)}"


    def grouped_experiments_by_day(expids: Sequence[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for expid in expids:
            date = derive_date(expid)
            animal = derive_animal_id(expid)
            groups.setdefault(make_day_id(animal, date), []).append(expid)
        return groups


    def resolve_analysis_state_selections(config: Mapping[str, Any], mode: str) -> List[str]:
        if mode == "movie":
            return list(config.get("movie_states", ["running", "still", "all"]))
        return list(config.get("sleep_states", ["nrem", "rem", "wake", "all"]))


    def build_state_masks_movie(state_bundle: Mapping[str, Any], selected_states: Sequence[str]) -> Dict[str, np.ndarray]:
        # Fallback movie masks: treat explicit trial labels when present, otherwise
        # expose a conservative "all" mask.
        n = len(next(iter(state_bundle.values()))) if state_bundle else 0
        base = np.ones(n, dtype=bool)
        masks = {"all": base}
        for state in selected_states:
            if state != "all":
                masks[state] = base.copy()
        return masks


    def build_state_masks_sleep(state_bundle: Mapping[str, Any], selected_states: Sequence[str]) -> Dict[str, np.ndarray]:
        n = len(next(iter(state_bundle.values()))) if state_bundle else 0
        base = np.ones(n, dtype=bool)
        masks = {"all": base}
        for state in selected_states:
            if state != "all":
                masks[state] = base.copy()
        return masks




def normalize_comparison_presets(raw: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if raw is None:
        return []

    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                raise SystemExit("comparison_presets entries must be mappings with a name field")
            name = entry.get("name")
            if name is None or not str(name).strip():
                raise SystemExit("comparison_presets entries must include a non-empty name")
            overrides = {key: value for key, value in entry.items() if key != "name"}
            items.append((str(name), overrides))
    else:
        raise SystemExit("comparison_presets must be a mapping or a list of named preset mappings")

    presets: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()
    for name, overrides in items:
        preset_name = str(name).strip()
        if not preset_name:
            raise SystemExit("comparison_presets contains an empty preset name")
        if preset_name in seen:
            raise SystemExit(f"comparison_presets contains a duplicate preset name: {preset_name}")
        if not isinstance(overrides, Mapping):
            raise SystemExit(f"comparison preset '{preset_name}' must map to a JSON object of overrides")
        presets.append((preset_name, dict(overrides)))
        seen.add(preset_name)
    return presets


def filter_comparison_presets(
    presets: Sequence[Tuple[str, Dict[str, Any]]],
    selected_names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not selected_names:
        return list(presets)

    selected = [str(name).strip() for name in selected_names if str(name).strip()]
    if not selected:
        return list(presets)
    selected_set = set(selected)
    available = {name for name, _ in presets}
    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(
            f"Unknown comparison preset(s): {', '.join(missing)}. Available presets are: {', '.join(name for name, _ in presets)}"
        )
    return [(name, overrides) for name, overrides in presets if name in selected_set]

def read_pickle(path: Path | str) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def resolve_repo_root(start: Path | None = None) -> Path:
    start = Path(start or __file__).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() or (candidate / "analysis").exists():
            return candidate
    return start.parent


def derive_animal_id(expid: str) -> str:
    match = re.search(r"_([A-Za-z0-9]+)$", expid)
    if match:
        return match.group(1)
    return expid.split("_")[-1]


def derive_date(expid: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})_", expid)
    if match:
        return match.group(1)
    return expid[:10]


def stable_hash(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_npz_cache(path: Path | str) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_npz_cache(path: Path | str, **arrays: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def cacheable(*parts: Any) -> str:
    return stable_hash(parts)


def stitch_day_time(exp_times: Sequence[np.ndarray], exp_offsets: Sequence[float] | None = None) -> np.ndarray:
    if not exp_times:
        return np.array([], dtype=float)
    if exp_offsets is None:
        exp_offsets = [0.0] * len(exp_times)
    stitched: List[np.ndarray] = []
    running = 0.0
    for t, offset in zip(exp_times, exp_offsets):
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            continue
        stitched.append(t + running + offset)
        running = stitched[-1][-1] if stitched[-1].size else running
    return np.concatenate(stitched) if stitched else np.array([], dtype=float)


def stitch_day_series(series_list: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(series) for series in series_list if np.asarray(series).size]
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays, axis=-1)


@dataclass(frozen=True)
class LoadedBundle:
    path: Path
    data: Dict[str, Any]

    @property
    def t(self) -> np.ndarray:
        for key in ("t", "time", "timestamps"):
            if key in self.data:
                return np.asarray(self.data[key], dtype=float)
        return np.array([], dtype=float)

    def matrix(self, preferred_keys: Sequence[str] = ("dF", "Spikes", "F")) -> np.ndarray:
        for key in preferred_keys:
            if key in self.data:
                return as_2d_matrix(self.data[key])
        numeric = [value for value in self.data.values() if isinstance(value, (list, tuple, np.ndarray))]
        if numeric:
            return as_2d_matrix(numeric[0])
        return np.zeros((0, self.t.size), dtype=float)

    def roi_ids(self) -> List[int]:
        for key in ("OriginalSuite2pCellIDs", "cell_ids", "roi_ids"):
            if key in self.data:
                value = np.asarray(self.data[key]).ravel().tolist()
                return [int(v) for v in value]
        return list(range(self.matrix().shape[0]))


def as_2d_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim >= 2:
        return arr.reshape(arr.shape[0], -1)
    return arr.reshape(0, 0)


def load_continuous_bundle(path: Path | str) -> LoadedBundle:
    return LoadedBundle(path=Path(path), data=read_pickle(path))


def find_first_key(bundle: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in bundle:
            return bundle[key]
    return default


def weighted_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def summarize_vector(values: Sequence[float] | np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def pairwise_correlation(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def lagged_correlation(
    x_t: Sequence[float] | np.ndarray,
    x: Sequence[float] | np.ndarray,
    y_t: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    lags_s: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x_t = np.asarray(x_t, dtype=float)
    x = np.asarray(x, dtype=float)
    y_t = np.asarray(y_t, dtype=float)
    y = np.asarray(y, dtype=float)
    lags_s = np.asarray(lags_s, dtype=float)
    corrs = np.full(lags_s.shape, np.nan, dtype=float)
    for idx, lag in enumerate(lags_s):
        shifted = interpolate_series(y_t + lag, y, x_t)
        corrs[idx] = pairwise_correlation(x, shifted)
    return lags_s, corrs


def summarize_lag_scan(lags_s: np.ndarray, corrs: np.ndarray) -> Dict[str, float]:
    if lags_s.size == 0 or corrs.size == 0 or np.all(~np.isfinite(corrs)):
        return {
            "zero_lag_corr": float("nan"),
            "best_corr": float("nan"),
            "best_lag_s": float("nan"),
            "mean_corr": float("nan"),
            "area_abs_corr": float("nan"),
        }
    finite = np.isfinite(corrs)
    zero_idx = int(np.argmin(np.abs(lags_s))) if lags_s.size else 0
    best_idx = int(np.nanargmax(corrs))
    area = float(np.trapz(np.nan_to_num(np.abs(corrs), nan=0.0), lags_s))
    return {
        "zero_lag_corr": float(corrs[zero_idx]) if finite[zero_idx] else float("nan"),
        "best_corr": float(corrs[best_idx]),
        "best_lag_s": float(lags_s[best_idx]),
        "mean_corr": float(np.nanmean(corrs)),
        "area_abs_corr": area,
    }


def ensure_monotonic_time(t: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(t, dtype=float)
    if arr.size < 2:
        return arr
    order = np.argsort(arr)
    return arr[order]


def family_output_dir(root: Path | str, family: str, division: str) -> Path:
    return ensure_dir(Path(root) / division / canonical_state_label(family))


def pick_state_bundle(exp_root: Path, mode: str) -> Tuple[Path, Mapping[str, Any]]:
    if mode == "movie":
        candidates = sorted(exp_root.glob("*_all_trials.csv"))
        if not candidates:
            raise FileNotFoundError(f"No movie trial CSV found in {exp_root}")
        path = candidates[0]
        return path, {"path": str(path), "rows": read_csv_rows(path)}
    if mode == "sleep":
        path = exp_root / "sleep_score" / "sleep_state.pickle"
        if not path.exists():
            raise FileNotFoundError(f"No sleep state bundle found at {path}")
        return path, read_pickle(path)
    raise ValueError(f"Unknown mode {mode!r}")


def pick_channel_bundle(exp_root: Path, channel: int) -> LoadedBundle:
    path = exp_root / "recordings" / f"s2p_ch{channel}.pickle"
    if not path.exists():
        raise FileNotFoundError(f"Missing channel bundle: {path}")
    return load_continuous_bundle(path)


def experiment_root_from_expid(repo_root: Path, expid: str, animal_id: str | None = None) -> Path:
    animal_id = animal_id or derive_animal_id(expid)
    candidate = Path("/home/rubencorreia/data/Repository") / animal_id / expid
    if candidate.exists():
        return candidate
    alt = repo_root / "data" / "Repository" / animal_id / expid
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not locate experiment root for {expid}")


def roi_series(bundle: LoadedBundle, metric: str = "dF") -> np.ndarray:
    return bundle.matrix(preferred_keys=(metric, "Spikes", "F"))


def mean_trace(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.array([], dtype=float)
    return np.nanmean(matrix, axis=0)

