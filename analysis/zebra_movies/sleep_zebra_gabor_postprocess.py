#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
CODEX_ROOT = SCRIPT_DIR.parents[2]
REPO_ROOT = SCRIPT_DIR.parents[1]
MAIN_PIPELINE_DIR = SCRIPT_DIR.parent / "main_pipeline"
for extra_path in (REPO_ROOT, MAIN_PIPELINE_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from sleep_dendrite_spine_pipeline import (
    DEFAULT_CACHE_NAME,
    derive_animal_id,
    derive_date,
    ensure_dir,
    eprint,
    extract_cut_neural_bundle,
    jsonable,
    load_npz_cache,
    make_day_id,
    read_csv_rows,
    report_relative_path,
    safe_filename_component,
    step_progress,
    step_scope,
)
from sleep_zebra_gabor_detail import DEFAULT_STIMULUS_SOURCE_ROOT, compute_movie_gabor_summaries

DEFAULT_RESULTS_DIR = CODEX_ROOT / "results" / "zebra_movies"
DEFAULT_GABOR_DIRNAME = "gabor"
DETAILS_SUBDIR = "experiments"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the movie-only Gabor post-process as a separate sidecar analysis."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_RESULTS_DIR / DEFAULT_CACHE_NAME,
        help="Path to the main pipeline cache (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the sidecar bundle. Defaults to the cache parent or the cache's configured output_dir.",
    )
    parser.add_argument(
        "--stimulus-source-root",
        type=Path,
        default=DEFAULT_STIMULUS_SOURCE_ROOT,
        help="Root directory containing the movie frame clips used by the Gabor helper.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Frame sampling stride forwarded to compute_movie_gabor_summaries.",
    )
    parser.add_argument(
        "--movie-expids",
        nargs="*",
        default=None,
        help="Optional subset of movie expIDs to process. Defaults to the cache config movie list.",
    )
    return parser


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)


def compact_string_list(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
    return list(dict.fromkeys(out))


def resolve_movie_expids(cache: Dict[str, Any], override: Optional[Sequence[str]] = None) -> List[str]:
    if override:
        return compact_string_list(override)
    config = cache.get("config", {}) or {}
    movie_expids = compact_string_list(config.get("movie_expids") or [])
    if movie_expids:
        return movie_expids
    experiments = cache.get("experiments", {}) or {}
    return sorted(
        exp_id
        for exp_id, exp_meta in experiments.items()
        if str((exp_meta or {}).get("compartment") or "").strip().lower() == "movie"
    )


def load_trial_rows(exp_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    trial_rows = exp_meta.get("trial_rows")
    if isinstance(trial_rows, list) and trial_rows:
        return [dict(row) for row in trial_rows if isinstance(row, dict)]
    source_paths = exp_meta.get("source_paths", {}) or {}
    trial_csv = source_paths.get("trial_csv")
    if trial_csv:
        trial_csv_path = Path(str(trial_csv))
        if trial_csv_path.exists():
            return [dict(row) for row in read_csv_rows(trial_csv_path)]
    return []


def resolve_response_bundle(exp_meta: Dict[str, Any], channel: Optional[int]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    alerts: List[str] = []
    source_paths = exp_meta.get("source_paths", {}) or {}
    response_candidates: List[Tuple[Path, Sequence[str], str]] = []
    cut_spikes_text = source_paths.get("cut_spikes")
    if cut_spikes_text:
        response_candidates.append(
            (Path(str(cut_spikes_text)), ["Spikes", "spikes", "Spike", "spike"], "cut_spikes")
        )
    cut_root_text = source_paths.get("cut")
    if cut_root_text:
        cut_root = Path(str(cut_root_text))
        if cut_root.is_dir() and channel is not None:
            response_candidates.extend(
                [
                    (cut_root / f"s2p_ch{int(channel)}_Spikes_cut.pickle", ["Spikes", "spikes", "Spike", "spike"], "cut_spikes"),
                    (cut_root / f"s2p_ch{int(channel)}_dF_cut.pickle", ["dF", "dff", "dF/F", "df", "signal"], "cut"),
                ]
            )
        else:
            response_candidates.append((cut_root, ["dF", "dff", "dF/F", "df", "signal"], "cut"))

    for response_path, preferred_keys, source_label in response_candidates:
        if not response_path.exists() or response_path.is_dir():
            continue
        try:
            response_time, response_cut, response_bundle = extract_cut_neural_bundle(response_path, preferred_keys=preferred_keys)
        except Exception as exc:
            alerts.append(f"Could not load {source_label} at {response_path}: {exc}")
            continue
        bundle_keys = sorted(list(response_bundle.keys())) if isinstance(response_bundle, dict) else []
        signal_key = next(
            (
                key
                for key in ["Spikes", "spikes", "Spike", "spike", "dF", "dff", "dF/F", "df", "signal"]
                if key in bundle_keys
            ),
            None,
        )
        if signal_key is None and bundle_keys:
            signal_key = str(bundle_keys[0])
        return {
            "response_path": str(response_path),
            "response_file_kind": source_label,
            "response_signal_key": signal_key or "unknown",
            "response_time": response_time,
            "response_cut": response_cut,
            "response_bundle_keys": bundle_keys,
        }, alerts
    alerts.append("No usable cut_spikes or cut file was found")
    return None, alerts


def build_detail_path(output_dir: Path, animal_id: str, date: str, exp_id: str) -> Path:
    animal_slug = safe_filename_component(animal_id)
    date_slug = safe_filename_component(date)
    exp_slug = safe_filename_component(exp_id)
    detail_dir = output_dir / DEFAULT_GABOR_DIRNAME / DETAILS_SUBDIR / animal_slug / date_slug
    return detail_dir / f"{exp_slug}_gabor_detail.json"


def process_movie_experiment(
    cache: Dict[str, Any],
    exp_id: str,
    stimulus_source_root: Path,
    frame_stride: int,
    output_dir: Path,
) -> Tuple[Dict[str, Any], List[str]]:
    experiments = cache.get("experiments", {}) or {}
    exp_meta = experiments.get(exp_id)
    if exp_meta is None:
        alert = f"Movie expID {exp_id} is not present in the cache"
        return {
            "exp_id": exp_id,
            "available": False,
            "alerts": [alert],
        }, [alert]

    animal_id = str(exp_meta.get("animal_id") or derive_animal_id(exp_id))
    date = str(exp_meta.get("date") or derive_date(exp_id))
    compartment = str(exp_meta.get("compartment") or "movie")
    day_id = str(exp_meta.get("day_id") or make_day_id(animal_id, date, compartment))
    detail_path = build_detail_path(output_dir, animal_id, date, exp_id)

    trial_rows = load_trial_rows(exp_meta)
    channel = None
    try:
        channel = int((cache.get("config", {}) or {}).get("channel"))
    except Exception:
        channel = None

    response_bundle, bundle_alerts = resolve_response_bundle(exp_meta, channel)
    alerts = list(bundle_alerts)

    gabor_summary: Dict[str, Any]
    if response_bundle is None:
        gabor_summary = {
            "available": False,
            "response_source": "missing",
            "response_signal_key": None,
            "grid_shape": [],
            "sigmas": [],
            "theta_radians": [],
            "visual_coverage": [],
            "roi_summaries": {},
            "clip_count": 0,
            "trial_count": len(trial_rows),
            "sample_count": 0,
        }
    else:
        try:
            gabor_summary = compute_movie_gabor_summaries(
                trial_rows,
                response_bundle["response_cut"],
                clip_source_root=stimulus_source_root,
                response_source=str(response_bundle["response_signal_key"] or response_bundle["response_file_kind"]),
                frame_stride=max(int(frame_stride), 1),
            )
        except Exception as exc:
            alerts.append(f"Failed to compute Gabor summaries for {exp_id}: {exc}")
            gabor_summary = {
                "available": False,
                "response_source": str(response_bundle["response_signal_key"] or response_bundle["response_file_kind"]),
                "response_signal_key": response_bundle.get("response_signal_key"),
                "grid_shape": [],
                "sigmas": [],
                "theta_radians": [],
                "visual_coverage": [],
                "roi_summaries": {},
                "clip_count": 0,
                "trial_count": len(trial_rows),
                "sample_count": 0,
            }

    detail_payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exp_id": exp_id,
        "animal_id": animal_id,
        "date": date,
        "day_id": day_id,
        "compartment": compartment,
        "trial_row_count": len(trial_rows),
        "trial_rows_present": bool(trial_rows),
        "response_path": response_bundle.get("response_path") if response_bundle else None,
        "response_file_kind": response_bundle.get("response_file_kind") if response_bundle else "missing",
        "response_signal_key": response_bundle.get("response_signal_key") if response_bundle else None,
        "gabor_summary": gabor_summary,
        "alerts": alerts,
    }
    write_json_file(detail_path, detail_payload)

    manifest_entry = {
        "exp_id": exp_id,
        "animal_id": animal_id,
        "date": date,
        "day_id": day_id,
        "compartment": compartment,
        "detail_path": report_relative_path(detail_path, output_dir),
        "response_path": response_bundle.get("response_path") if response_bundle else None,
        "response_file_kind": response_bundle.get("response_file_kind") if response_bundle else "missing",
        "response_signal_key": response_bundle.get("response_signal_key") if response_bundle else None,
        "available": bool(gabor_summary.get("available")),
        "trial_count": int(gabor_summary.get("trial_count", len(trial_rows)) or 0),
        "clip_count": int(gabor_summary.get("clip_count", 0) or 0),
        "sample_count": int(gabor_summary.get("sample_count", 0) or 0),
        "alerts": alerts,
    }
    return manifest_entry, alerts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cache_path = args.cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file does not exist: {cache_path}")

    with step_scope("load Gabor cache"):
        cache = load_npz_cache(cache_path)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(cache_path.parent or DEFAULT_RESULTS_DIR)
    gabor_root = ensure_dir(output_dir / DEFAULT_GABOR_DIRNAME)
    ensure_dir(gabor_root)
    detail_root = ensure_dir(gabor_root / DETAILS_SUBDIR)

    movie_expids = resolve_movie_expids(cache, args.movie_expids)
    manifest_entries: List[Dict[str, Any]] = []
    all_alerts: List[str] = list(cache.get("alerts", []) or [])
    status_counts: Counter[str] = Counter()

    with step_scope("movie Gabor post-process", total=len(movie_expids)):
        for idx, exp_id in enumerate(movie_expids, start=1):
            step_progress(idx, len(movie_expids), label=str(exp_id))
            manifest_entry, alerts = process_movie_experiment(
                cache=cache,
                exp_id=exp_id,
                stimulus_source_root=Path(args.stimulus_source_root),
                frame_stride=int(args.frame_stride),
                output_dir=output_dir,
            )
            manifest_entries.append(manifest_entry)
            status = "available" if manifest_entry.get("available") else "empty"
            if manifest_entry.get("alerts"):
                status = "alert"
            status_counts[status] += 1
            all_alerts.extend(manifest_entry.get("alerts", []))

    all_alerts = list(dict.fromkeys(str(alert) for alert in all_alerts if str(alert).strip()))
    generated_at = datetime.now().isoformat(timespec="seconds")

    manifest_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "gabor_root": str(gabor_root),
        "detail_root": str(detail_root),
        "movie_expids": movie_expids,
        "n_movie_expids": len(movie_expids),
        "status_counts": dict(sorted(status_counts.items())),
        "experiments": manifest_entries,
        "alerts": all_alerts,
    }
    manifest_path = gabor_root / "manifest.json"
    write_json_file(manifest_path, manifest_payload)

    summary_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "stimulus_source_root": str(args.stimulus_source_root),
        "movie_expids": movie_expids,
        "n_requested_movie_expids": len(movie_expids),
        "n_available": sum(1 for entry in manifest_entries if entry.get("available")),
        "n_with_alerts": sum(1 for entry in manifest_entries if entry.get("alerts")),
        "status_counts": dict(sorted(status_counts.items())),
        "manifest_path": report_relative_path(manifest_path, output_dir),
        "detail_root": report_relative_path(detail_root, output_dir),
        "experiments": {
            entry["exp_id"]: {
                "animal_id": entry["animal_id"],
                "date": entry["date"],
                "day_id": entry["day_id"],
                "compartment": entry["compartment"],
                "available": entry["available"],
                "trial_count": entry["trial_count"],
                "clip_count": entry["clip_count"],
                "sample_count": entry["sample_count"],
                "detail_path": entry["detail_path"],
                "response_path": entry["response_path"],
                "response_file_kind": entry["response_file_kind"],
                "response_signal_key": entry["response_signal_key"],
            }
            for entry in manifest_entries
        },
        "alerts": all_alerts,
    }
    summary_path = gabor_root / "gabor_summary.json"
    write_json_file(summary_path, summary_payload)

    print(
        json.dumps(
            {
                "cache_path": str(cache_path),
                "output_dir": str(output_dir),
                "gabor_root": str(gabor_root),
                "n_movie_expids": len(movie_expids),
                "n_available": summary_payload["n_available"],
                "n_with_alerts": summary_payload["n_with_alerts"],
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if all_alerts:
        for alert in all_alerts:
            text = str(alert).strip()
            if text.startswith("[ALERT] "):
                text = text[len("[ALERT] ") :]
            eprint(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
