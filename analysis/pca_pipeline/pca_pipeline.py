from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from analysis.compartment_common import ensure_dir, read_pickle
from analysis.dendrites_pipeline.dendrites_pipeline import extract_series_bundle
from analysis.shared.analysis_families.core import build_experiment_context, shared_time_axis
from analysis.shared.analysis_families.state import state_masks_for_context
from analysis.shared.state_utils import derive_animal_id, derive_date, make_day_id

from .plots import save_pca_figures

LOGGER = logging.getLogger(__name__)


@dataclass
class PCAConfig:
    movie_expids: Sequence[str]
    sleep_expids: Sequence[str]
    output_root: Path
    window_s: float = 1.0
    min_valid_fraction: float = 0.8
    min_roi_std: float = 1e-9
    n_components: int = 5
    soma_channel: int = 1
    bouton_channel: int = 0
    selected_states: Sequence[str] = ("quiet_awake", "nrem", "rem", "running", "still", "all")
    include_spines: bool = False
    spine_sources: Sequence[Mapping[str, Any]] = ()
    metric: str = "dF"
    thread_limit: int = 1
    auto_discover_expids: bool = True
    source_configs: Sequence[str] = ("analysis/soma_bouton_pipeline/soma_bouton_pipeline_config.json", "analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], repo_root: Path) -> "PCAConfig":
        root = Path(value.get("output_root", "results/pca_pipeline"))
        if not root.is_absolute():
            root = repo_root / root
        return cls(
            movie_expids=tuple(value.get("movie_expids", [])),
            sleep_expids=tuple(value.get("sleep_expids", [])),
            output_root=root,
            window_s=float(value.get("window_s", 1.0)),
            min_valid_fraction=float(value.get("min_valid_fraction", 0.8)),
            min_roi_std=float(value.get("min_roi_std", 1e-9)),
            n_components=int(value.get("n_components", 5)),
            soma_channel=int(value.get("soma_channel", 1)),
            bouton_channel=int(value.get("bouton_channel", 0)),
            selected_states=tuple(value.get("selected_states", ("quiet_awake", "nrem", "rem", "running", "still", "all"))),
            include_spines=bool(value.get("include_spines", False)),
            spine_sources=tuple(value.get("spine_sources", [])),
            metric=str(value.get("metric", "dF")),
            thread_limit=max(1, int(value.get("thread_limit", 1))),
            auto_discover_expids=bool(value.get("auto_discover_expids", True)),
            source_configs=tuple(value.get("source_configs", ("analysis/soma_bouton_pipeline/soma_bouton_pipeline_config.json", "analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json"))),
        )


def _source_type_for_expid(expid: str, config: PCAConfig, repo_root: Path) -> str:
    matches: List[str] = []
    for raw_path in config.source_configs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            continue
        try:
            with path.open() as handle:
                source = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if expid in source.get("movie_expids", []) or expid in source.get("sleep_expids", []):
            matches.append("soma/bouton" if "soma_bouton" in path.name else "dendrite/spine")
    return " and ".join(dict.fromkeys(matches)) or "unknown"


def discover_expids(config: PCAConfig, repo_root: Path) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Read experiment IDs from the existing analysis configs."""
    movies: List[str] = []
    sleeps: List[str] = []
    for raw_path in config.source_configs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            LOGGER.warning("Configured source config does not exist: %s", path)
            continue
        try:
            with path.open() as handle:
                source = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read source config %s: %s", path, exc)
            continue
        source_movies = [str(item) for item in source.get("movie_expids", []) if str(item).strip()]
        source_sleeps = [str(item) for item in source.get("sleep_expids", []) if str(item).strip()]
        movies.extend(source_movies)
        sleeps.extend(source_sleeps)
        source_label = "soma/bouton" if "soma_bouton" in path.name else "dendrite/spine"
        LOGGER.info("Found %d movie and %d sleep expIDs from %s config", len(source_movies), len(source_sleeps), source_label)
    return tuple(dict.fromkeys(movies)), tuple(dict.fromkeys(sleeps))


def make_window_matrix(matrix: np.ndarray, time: np.ndarray, window_s: float = 1.0, min_valid_fraction: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return window means as [window, ROI], window starts, and valid-window mask."""
    values = np.asarray(matrix, dtype=float)
    time = np.asarray(time, dtype=float).ravel()
    if values.ndim != 2 or time.size != values.shape[1] or time.size < 2 or window_s <= 0:
        return np.empty((0, 0)), np.empty(0), np.empty(0, dtype=bool)
    starts = np.arange(time[0], time[-1], window_s)
    rows: List[np.ndarray] = []
    kept_starts: List[float] = []
    valid_rows: List[bool] = []
    for start in starts:
        idx = (time >= start) & (time < start + window_s)
        if not idx.any():
            continue
        chunk = values[:, idx]
        valid_fraction = np.mean(np.isfinite(chunk), axis=1)
        rows.append(np.nanmean(chunk, axis=1))
        kept_starts.append(float(start))
        valid_rows.append(bool(np.all(valid_fraction >= min_valid_fraction)))
    if not rows:
        return np.empty((0, values.shape[0])), np.empty(0), np.empty(0, dtype=bool)
    return np.asarray(rows), np.asarray(kept_starts), np.asarray(valid_rows, dtype=bool)


def fit_population_pca(window_values: np.ndarray, *, n_components: int = 5, min_roi_std: float = 1e-9, thread_limit: int = 1) -> Dict[str, np.ndarray]:
    """Fit PCA after per-ROI z-scoring; rows are observations and columns are ROIs."""
    values = np.asarray(window_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires at least two observations")
    finite = np.all(np.isfinite(values), axis=0)
    means = np.nanmean(values[:, finite], axis=0) if finite.any() else np.empty(0)
    stds = np.nanstd(values[:, finite], axis=0) if finite.any() else np.empty(0)
    keep = finite.copy()
    keep[finite] = stds > min_roi_std
    if keep.sum() < 2:
        raise ValueError("PCA requires at least two variable, finite ROIs")
    standardized = (values[:, keep] - means[stds > min_roi_std]) / stds[stds > min_roi_std]
    components = min(int(n_components), standardized.shape[0], standardized.shape[1])
    from sklearn.decomposition import PCA
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=max(1, int(thread_limit))):
            model = PCA(n_components=components).fit(standardized)
    except ImportError:
        model = PCA(n_components=components).fit(standardized)
    scores = model.transform(standardized)
    components_matrix = model.components_
    explained_ratio = model.explained_variance_ratio_
    return {
        "scores": scores,
        "components": components_matrix,
        "explained_variance_ratio": explained_ratio,
        "roi_keep": keep,
        "roi_mean": means[stds > min_roi_std],
        "roi_std": stds[stds > min_roi_std],
    }


def _interp_series(path: Path, target_time: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    if not path.exists():
        return np.full(target_time.shape, np.nan)
    try:
        source_t, source_y, _ = extract_series_bundle(path, keys)
        source_t = np.asarray(source_t, dtype=float).ravel()
        source_y = np.asarray(source_y, dtype=float).ravel()
        n = min(source_t.size, source_y.size)
        valid = np.isfinite(source_t[:n]) & np.isfinite(source_y[:n])
        if valid.sum() < 2:
            return np.full(target_time.shape, np.nan)
        return np.interp(target_time, source_t[:n][valid], source_y[:n][valid], left=np.nan, right=np.nan)
    except Exception as exc:
        LOGGER.warning("Could not load metadata %s: %s", path, exc)
        return np.full(target_time.shape, np.nan)


def _metadata_for_context(ctx: Any, time: np.ndarray, selected_states: Sequence[str]) -> Dict[str, np.ndarray]:
    state_masks = state_masks_for_context(ctx, selected_states)
    metadata: Dict[str, np.ndarray] = {
        "state": np.full(time.shape, "unlabeled", dtype=object),
        "locomotion": np.full(time.shape, np.nan),
        "pupil_size": np.full(time.shape, np.nan),
        "visual_condition": np.full(time.shape, "unknown", dtype=object),
    }
    for state, mask in state_masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.size == time.size:
            metadata["state"][mask] = str(state)
    wheel = ctx.exp_root / "recordings" / "wheel.pickle"
    metadata["locomotion"] = _interp_series(wheel, time, ("speed", "wheel", "motion", "velocity"))
    for name in ("dlcEyeLeft_resampled.pickle", "dlcEyeRight_resampled.pickle"):
        pupil_path = ctx.exp_root / "recordings" / name
        for keys in (("pupil_size", "pupil_diameter", "pupil_area", "radius"), ("diameter", "area")):
            pupil = _interp_series(pupil_path, time, keys)
            if np.isfinite(pupil).any():
                current = metadata["pupil_size"]
                stacked = np.vstack([current, pupil])
                counts = np.sum(np.isfinite(stacked), axis=0)
                totals = np.nansum(stacked, axis=0)
                metadata["pupil_size"] = np.divide(totals, counts, out=np.full(counts.shape, np.nan, dtype=float), where=counts > 0)
                break
    metadata["visual_condition"][:] = "visual" if ctx.mode == "movie" else "no_visual"
    return metadata


def _window_nanmean(values: np.ndarray, time: np.ndarray, start: float, window_s: float) -> float:
    left = int(np.searchsorted(time, start, side="left"))
    right = int(np.searchsorted(time, start + window_s, side="left"))
    chunk = np.asarray(values[left:right], dtype=float)
    finite = chunk[np.isfinite(chunk)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _rows_for_pca(ctx: Any, compartment: str, matrix: np.ndarray, time: np.ndarray, cfg: PCAConfig) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    windows, starts, valid_windows = make_window_matrix(matrix, time, cfg.window_s, cfg.min_valid_fraction)
    if windows.size == 0:
        return [], {}
    windows = windows[valid_windows]
    starts = starts[valid_windows]
    metadata = _metadata_for_context(ctx, time, cfg.selected_states)
    result = fit_population_pca(windows, n_components=cfg.n_components, min_roi_std=cfg.min_roi_std, thread_limit=cfg.thread_limit)
    rows: List[Dict[str, Any]] = []
    for row_idx, start in enumerate(starts):
        row: Dict[str, Any] = {
            "animal_id": ctx.animal_id,
            "day_id": ctx.day_id,
            "expid": ctx.expid,
            "session_id": ctx.expid,
            "compartment": compartment,
            "window_start": float(start),
            "window_end": float(start + cfg.window_s),
            "state": str(metadata["state"][min(int(np.searchsorted(time, start)), time.size - 1)]),
            "locomotion": _window_nanmean(metadata["locomotion"], time, start, cfg.window_s),
            "pupil_size": _window_nanmean(metadata["pupil_size"], time, start, cfg.window_s),
            "visual_condition": str(metadata["visual_condition"][0]),
        }
        for pc_idx, score in enumerate(result["scores"][row_idx], start=1):
            row[f"PC{pc_idx}"] = float(score)
        rows.append(row)
    return rows, result


def _group_pca_rows(contexts: Sequence[Any], compartment: str, cfg: PCAConfig) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]] | None:
    if not contexts:
        return [], {}
    first_bundle = contexts[0].soma if compartment == "soma" else contexts[0].bouton
    reference_ids = first_bundle.roi_ids()
    window_parts: List[np.ndarray] = []
    start_parts: List[np.ndarray] = []
    context_parts: List[Any] = []
    for ctx in contexts:
        bundle = ctx.soma if compartment == "soma" else ctx.bouton
        if bundle.roi_ids() != reference_ids:
            LOGGER.warning("[%s] ROI registration mismatch for %s; using session-level PCA", ctx.day_id, compartment)
            return None
        time = shared_time_axis(ctx)
        windows, starts, valid = make_window_matrix(bundle.matrix(preferred_keys=(cfg.metric, "Spikes", "F")), time, cfg.window_s, cfg.min_valid_fraction)
        if windows.size:
            window_parts.append(windows[valid])
            start_parts.append(starts[valid])
            context_parts.extend([ctx] * int(valid.sum()))
    if not window_parts:
        return [], {}
    windows = np.vstack(window_parts)
    starts = np.concatenate(start_parts)
    result = fit_population_pca(windows, n_components=cfg.n_components, min_roi_std=cfg.min_roi_std, thread_limit=cfg.thread_limit)
    rows: List[Dict[str, Any]] = []
    offset = 0
    for ctx in contexts:
        count = sum(1 for item in context_parts if item is ctx)
        if not count:
            continue
        time = shared_time_axis(ctx)
        metadata = _metadata_for_context(ctx, time, cfg.selected_states)
        for row_idx in range(offset, offset + count):
            start = float(starts[row_idx])
            rows.append({
                "animal_id": ctx.animal_id, "day_id": ctx.day_id, "expid": ctx.expid, "session_id": ctx.expid,
                "compartment": compartment, "window_start": start, "window_end": start + cfg.window_s,
                "state": str(metadata["state"][min(int(np.searchsorted(time, start)), time.size - 1)]),
                "locomotion": _window_nanmean(metadata["locomotion"], time, start, cfg.window_s),
                "pupil_size": _window_nanmean(metadata["pupil_size"], time, start, cfg.window_s),
                "visual_condition": str(metadata["visual_condition"][0]),
                **{f"PC{pc_idx}": float(score) for pc_idx, score in enumerate(result["scores"][row_idx], start=1)},
            })
        offset += count
    return rows, result


def _spine_source_rows(source: Mapping[str, Any], contexts: Mapping[str, Any], cfg: PCAConfig, repo_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    source_path = Path(str(source.get("path", "")))
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Missing spine source: {source_path}")
    data = np.load(source_path, allow_pickle=True)
    matrix = np.asarray(data[str(source.get("matrix_key", "matrix"))], dtype=float)
    time = np.asarray(data[str(source.get("time_key", "time"))], dtype=float).ravel()
    animal_id = str(source.get("animal_id", ""))
    day_id = str(source.get("day_id", ""))
    expid = str(source.get("expid", ""))
    if not animal_id or not day_id or not expid:
        raise ValueError("Each spine source requires animal_id, day_id, and expid")
    ctx = contexts.get(expid)
    if ctx is None or ctx.day_id != day_id or ctx.animal_id != animal_id:
        raise ValueError(f"Spine source {source_path} has no matching configured same-day context")
    rows, result = _rows_for_pca(ctx, "spine", matrix, time, cfg)
    parent_dendrite = source.get("parent_dendrite_id", "")
    parent_soma = source.get("parent_soma_id", "")
    for row in rows:
        row["parent_dendrite_id"] = str(parent_dendrite)
        row["parent_soma_id"] = str(parent_soma)
    return rows, result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path, repo_root: Path) -> PCAConfig:
    with path.open() as handle:
        return PCAConfig.from_mapping(json.load(handle), repo_root)


def run_pca_pipeline(config: PCAConfig, repo_root: Path) -> Dict[str, Any]:
    result_root = ensure_dir(config.output_root)
    ensure_dir(result_root / "cache")
    all_rows: List[Dict[str, Any]] = []
    contexts: Dict[str, Any] = {}
    figure_payloads: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    skipped: List[Dict[str, str]] = []
    movie_expids = tuple(config.movie_expids)
    sleep_expids = tuple(config.sleep_expids)
    if config.auto_discover_expids and (not movie_expids or not sleep_expids):
        discovered_movies, discovered_sleeps = discover_expids(config, repo_root)
        movie_expids = movie_expids or discovered_movies
        sleep_expids = sleep_expids or discovered_sleeps
        LOGGER.info("Using %d movie and %d sleep experiment IDs from existing configs", len(movie_expids), len(sleep_expids))
    experiments = [(str(expid), "movie") for expid in movie_expids] + [(str(expid), "sleep") for expid in sleep_expids]
    for expid, mode in experiments:
        try:
            ctx = build_experiment_context(expid, mode, config.soma_channel, config.bouton_channel, repo_root=repo_root)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            source_type = _source_type_for_expid(expid, config, repo_root)
            LOGGER.warning("[%s] skip %s (%s: no soma/bouton channels)", make_day_id(derive_animal_id(expid), derive_date(expid)), expid, source_type)
            skipped.append({"expid": expid, "mode": mode, "source_type": source_type, "reason": str(exc)})
            continue
        contexts[expid] = ctx
        LOGGER.info("[%s] loaded %s session", ctx.day_id, expid)
    grouped_contexts: Dict[str, List[Any]] = {}
    for ctx in contexts.values():
        grouped_contexts.setdefault(ctx.day_id, []).append(ctx)
    for day_id, day_contexts in grouped_contexts.items():
        LOGGER.info("[%s] pooling %d registered sessions", day_id, len(day_contexts))
        for compartment in ("soma", "bouton"):
            grouped_result = _group_pca_rows(day_contexts, compartment, config)
            if grouped_result is None:
                for ctx in day_contexts:
                    rows, result = _rows_for_pca(ctx, compartment, (ctx.soma if compartment == "soma" else ctx.bouton).matrix(preferred_keys=(config.metric, "Spikes", "F")), shared_time_axis(ctx), config)
                    all_rows.extend(rows)
                    if result: figure_payloads.append((f"{ctx.day_id}_{ctx.expid}_{compartment}", result, rows))
            else:
                rows, result = grouped_result
                all_rows.extend(rows)
                if result: figure_payloads.append((f"{day_id}_{compartment}", result, rows))
    if config.include_spines:
        for source in config.spine_sources:
            rows, result = _spine_source_rows(source, contexts, config, repo_root)
            all_rows.extend(rows)
            if result:
                figure_payloads.append((f'{source.get("day_id", "unknown")}_spine', result, rows))
    _write_csv(result_root / "csv" / "pca_scores.csv", all_rows)
    if skipped:
        by_type = {}
        for item in skipped:
            by_type[item["source_type"]] = by_type.get(item["source_type"], 0) + 1
        LOGGER.info("Skipped %d sessions: %s", len(skipped), ", ".join(f"{count} {kind}" for kind, count in sorted(by_type.items())))
    summary = {
        "n_rows": len(all_rows),
        "days": sorted({row["day_id"] for row in all_rows}),
        "same_day_groups": sorted({row["day_id"] for row in all_rows}),
        "compartments": sorted({row["compartment"] for row in all_rows}),
        "skipped_experiments": skipped,
    }
    _write_csv(result_root / "csv" / "pca_summary.csv", [{key: value if not isinstance(value, list) else json.dumps(value) for key, value in summary.items()}])
    for label, result, rows in figure_payloads:
        save_pca_figures(result_root / "figures", label, result, rows)
    with (result_root / "cache" / "run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exploratory same-day PCA of soma and bouton activity.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    repo_root = Path(__file__).resolve().parents[2]
    summary = run_pca_pipeline(_load_config(args.config, repo_root), repo_root)
    LOGGER.info("PCA complete: %s", summary)


if __name__ == "__main__":
    main()
