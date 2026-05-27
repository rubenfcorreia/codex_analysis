#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODEX_ROOT = SCRIPT_DIR.parents[2]
REPO_ROOT = SCRIPT_DIR.parents[1]
MAIN_PIPELINE_DIR = SCRIPT_DIR.parent / "main_pipeline"
for extra_path in (REPO_ROOT, MAIN_PIPELINE_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from sleep_dendrite_spine_day_figures import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DPI,
    build_day_summary,
    build_figure_save_path,
    configured_compartment_for_exp_id,
    describe_day_group_mismatch,
    grouped_day_expids,
    plot_day_figure,
    selected_dendrite_records,
    warn_if_compartment_mismatch,
)
from sleep_dendrite_spine_pipeline import (
    DEND_AXON_CONVERSION_FILENAME,
    DEFAULT_CACHE_NAME,
    NORMAL_CONVERSION_FILENAME,
    USER_EDITABLE_DEFAULTS,
    build_arg_parser as build_pipeline_arg_parser,
    build_demo_repository,
    derive_animal_id,
    derive_date,
    determine_conversion_mode,
    cleanup_roi_detail_figures,
    ensure_dir,
    load_config_file,
    load_or_build_cache,
    locate_conversion_file,
    merge_cli_config,
    parse_list_argument,
    resolve_repo_root,
    step_progress,
    step_message,
    step_scope,
)
from sleep_dendrite_spine_day_figures import choose_representative_exp_id
from sleep_zebra_movies_assets import (
    DEFAULT_GABOR_LIBRARY_PATH,
    DEFAULT_STIMULUS_CACHE_ROOT,
    DEFAULT_STIMULUS_SOURCE_ROOT,
    prepare_zebra_stimulus_assets,
)

DEFAULT_RESULTS_DIR = CODEX_ROOT / "results" / "zebra_movies"


def coerce_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_plane_index(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        if part.startswith("plane") and part[5:].isdigit():
            return int(part[5:])
    return None


def scan_conversion_roles(exp_root: Path) -> List[Dict[str, Any]]:
    roles: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for spines_dir in sorted(exp_root.rglob("SpinesGUI")):
        if not spines_dir.is_dir():
            continue
        for filename in [DEND_AXON_CONVERSION_FILENAME, NORMAL_CONVERSION_FILENAME]:
            candidate = spines_dir / filename
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str in seen_paths:
                continue
            seen_paths.add(candidate_str)
            roles.append(
                {
                    "path": candidate_str,
                    "mode": determine_conversion_mode(candidate),
                }
            )
    return roles


def scan_ops_layouts(exp_root: Path) -> List[Dict[str, Any]]:
    layouts: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for ops_path in sorted(exp_root.rglob("ops.npy")):
        if "suite2p" not in ops_path.parts:
            continue
        ops_path_str = str(ops_path)
        if ops_path_str in seen_paths:
            continue
        seen_paths.add(ops_path_str)
        try:
            ops = np.load(ops_path, allow_pickle=True).item()
        except Exception:
            continue
        if not isinstance(ops, dict):
            continue
        layouts.append(
            {
                "path": ops_path_str,
                "plane": extract_plane_index(ops_path),
                "nchannels": coerce_optional_int(ops.get("nchannels")),
                "functional_chan": coerce_optional_int(ops.get("functional_chan")),
                "nplanes": coerce_optional_int(ops.get("nplanes")),
            }
        )
    return layouts


def discover_experiment_mode(
    repo_base: Path,
    animal_id: str,
    exp_id: str,
    sleep_expids: Sequence[str],
) -> Dict[str, Any]:
    exp_root = resolve_repo_root(repo_base, animal_id, exp_id)
    primary_conversion_path, conversion_source_exp, used_fallback = locate_conversion_file(
        repo_base,
        animal_id,
        exp_id,
        prefer_same_day_source=exp_id in set(sleep_expids),
    )

    if primary_conversion_path is None:
        roi_mode = "soma_only"
        primary_conversion_mode: Optional[str] = None
    else:
        primary_conversion_mode = determine_conversion_mode(primary_conversion_path)
        roi_mode = "dendrite_spine" if primary_conversion_mode == "normal" else "dendrite_axon"

    if exp_root.exists():
        conversion_roles = scan_conversion_roles(exp_root)
        ops_layouts = scan_ops_layouts(exp_root)
    else:
        conversion_roles = []
        ops_layouts = []

    unique_conversion_modes = {role["mode"] for role in conversion_roles}
    unique_ops_layouts = {
        (layout["nchannels"], layout["functional_chan"], layout["nplanes"])
        for layout in ops_layouts
        if any(value is not None for value in (layout["nchannels"], layout["functional_chan"], layout["nplanes"]))
    }
    layout_mode = "mixed" if len(unique_conversion_modes) > 1 or len(unique_ops_layouts) > 1 else "single"
    analysis_mode = "mixed" if layout_mode == "mixed" else roi_mode

    return {
        "exp_id": exp_id,
        "animal_id": animal_id,
        "date": derive_date(exp_id),
        "repo_root": str(exp_root),
        "analysis_mode": analysis_mode,
        "roi_mode": roi_mode,
        "layout_mode": layout_mode,
        "primary_conversion_path": str(primary_conversion_path) if primary_conversion_path is not None else None,
        "primary_conversion_mode": primary_conversion_mode,
        "conversion_source_exp_id": conversion_source_exp,
        "used_fallback": bool(used_fallback),
        "conversion_roles": conversion_roles,
        "ops_layouts": ops_layouts,
    }


def discover_requested_experiments(config: Dict[str, Any], repo_base: Path) -> List[Dict[str, Any]]:
    movie_expids = parse_list_argument(config.get("movie_expids"))
    sleep_expids = parse_list_argument(config.get("sleep_expids"))
    requested_expids = sorted(set(movie_expids) | set(sleep_expids))
    discoveries: List[Dict[str, Any]] = []
    for exp_id in requested_expids:
        animal_id = derive_animal_id(exp_id)
        try:
            discovery = discover_experiment_mode(repo_base, animal_id, exp_id, sleep_expids)
        except Exception as exc:
            discovery = {
                "exp_id": exp_id,
                "animal_id": animal_id,
                "date": derive_date(exp_id),
                "repo_root": str(resolve_repo_root(repo_base, animal_id, exp_id)),
                "analysis_mode": "missing",
                "roi_mode": "missing",
                "layout_mode": "missing",
                "primary_conversion_path": None,
                "primary_conversion_mode": None,
                "conversion_source_exp_id": None,
                "used_fallback": False,
                "conversion_roles": [],
                "ops_layouts": [],
                "error": str(exc),
            }
        discoveries.append(discovery)
    return discoveries


def summarize_discoveries(discoveries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_requested_experiments": len(discoveries),
        "analysis_mode_counts": dict(sorted(Counter(str(item.get("analysis_mode") or "unknown") for item in discoveries).items())),
        "roi_mode_counts": dict(sorted(Counter(str(item.get("roi_mode") or "unknown") for item in discoveries).items())),
        "layout_mode_counts": dict(sorted(Counter(str(item.get("layout_mode") or "unknown") for item in discoveries).items())),
    }


def print_discovery_report(discoveries: Sequence[Dict[str, Any]]) -> None:
    for item in discoveries:
        exp_id = str(item.get("exp_id") or "unknown")
        analysis_mode = str(item.get("analysis_mode") or "unknown")
        roi_mode = str(item.get("roi_mode") or "unknown")
        layout_mode = str(item.get("layout_mode") or "unknown")
        conversion_path = item.get("primary_conversion_path")
        source_exp = item.get("conversion_source_exp_id")
        extra_bits = []
        if conversion_path:
            extra_bits.append(f"conversion={conversion_path}")
        if source_exp and source_exp != exp_id:
            extra_bits.append(f"source={source_exp}")
        if item.get("ops_layouts"):
            extra_bits.append(f"ops={len(item['ops_layouts'])}")
        suffix = f" ({', '.join(extra_bits)})" if extra_bits else ""
        print(f"[roi] {exp_id}: analysis_mode={analysis_mode}, roi_mode={roi_mode}, layout_mode={layout_mode}{suffix}")
    print(json.dumps(summarize_discoveries(discoveries), indent=2, sort_keys=True))


def build_run_config(args: Any) -> Dict[str, Any]:
    file_config = load_config_file(args.config)
    cli_config = {
        "user_id": args.user_id,
        "repo_base": str(args.repo_base) if args.repo_base else None,
        "movie_expids": parse_list_argument(args.movie_expids),
        "sleep_expids": parse_list_argument(args.sleep_expids),
        "basal_expids": parse_list_argument(args.basal_expids),
        "apical_expids": parse_list_argument(args.apical_expids),
        "compare_states": parse_list_argument(args.compare_states),
        "state_comparison_states": parse_list_argument(args.state_comparison_states),
        "basal_apical_states": parse_list_argument(args.basal_apical_states),
        "channel": args.channel,
        "shuffle_n": args.shuffle_n,
        "high_pass_hz": args.high_pass_hz,
        "locomotion_threshold": args.locomotion_threshold,
        "cache_path": str(args.cache_path) if args.cache_path else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "rebuild": True if args.rebuild else None,
        "figure_dpi": args.figure_dpi,
        "stimulus_source_root": str(args.stimulus_source_root) if args.stimulus_source_root else None,
        "stimulus_cache_root": str(args.stimulus_cache_root) if args.stimulus_cache_root else None,
        "gabor_save_path": str(args.gabor_save_path) if args.gabor_save_path else None,
        "build_full_gabor_library": True if args.build_full_gabor_library else None,
    }
    config = merge_cli_config(cli_config, file_config)
    for key, value in USER_EDITABLE_DEFAULTS.items():
        if key not in config or config[key] is None:
            config[key] = value
    return config


def maybe_build_demo_repository(args: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    if not args.demo:
        return config

    demo_output_dir = Path(args.output_dir) if args.output_dir is not None else DEFAULT_RESULTS_DIR
    demo_base = demo_output_dir
    demo_spec = None
    if args.demo_spec is not None:
        demo_spec = load_config_file(args.demo_spec)
    elif isinstance(config.get("demo_spec"), dict):
        demo_spec = config.get("demo_spec")
    elif isinstance(config.get("demo_spec"), str):
        demo_spec_path = Path(str(config.get("demo_spec")))
        if demo_spec_path.exists():
            demo_spec = load_config_file(demo_spec_path)

    demo_config = build_demo_repository(demo_base, demo_spec=demo_spec)
    merged = dict(config)
    merged["user_id"] = demo_config["user_id"]
    merged["repo_base"] = demo_config["repo_base"]
    merged["movie_expids"] = demo_config["movie_expids"]
    merged["sleep_expids"] = demo_config["sleep_expids"]
    merged["basal_expids"] = demo_config["basal_expids"]
    merged["apical_expids"] = demo_config["apical_expids"]
    merged["channel"] = demo_config["channel"]
    merged["locomotion_threshold"] = demo_config["locomotion_threshold"]
    return merged


def generate_day_figures(
    cache: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    repo_base: Path,
    dpi: int,
) -> List[str]:
    day_groups = grouped_day_expids(config, cache)
    if not day_groups:
        raise SystemExit(
            describe_day_group_mismatch(
                config,
                cache,
                config_path=args.config,
                cache_path=cache_path,
            )
        )

    written: List[str] = []
    warned_mismatch_expids: set[str] = set()
    with step_scope("zebra day figures", total=len(day_groups)):
        for day_idx, ((animal_id, date), exp_ids) in enumerate(day_groups.items(), start=1):
            step_progress(day_idx, len(day_groups), label=f"{animal_id} | {date}")
            day_summary = build_day_summary(animal_id, date, exp_ids, cache)
            dendrite_records = selected_dendrite_records(cache, animal_id, date, exp_ids)
            if not dendrite_records:
                print(f"[skip] {animal_id} | {date}: no dendrite/spine hierarchy found")
                continue

            with step_scope("zebra dendrites", total=len(dendrite_records)):
                for dend_idx, (global_dendrite_id, dendrite_record) in enumerate(dendrite_records, start=1):
                    step_progress(dend_idx, len(dendrite_records), label=str(global_dendrite_id))
                    representative_exp_id = choose_representative_exp_id(dendrite_record, day_summary.exp_ids)
                    representative_obs = (
                        dendrite_record.get("observations", {}).get(representative_exp_id, {}) if representative_exp_id is not None else {}
                    )
                    if representative_exp_id is not None and representative_exp_id not in warned_mismatch_expids:
                        expected_compartment = configured_compartment_for_exp_id(
                            representative_exp_id,
                            set(parse_list_argument(config.get("basal_expids"))),
                            set(parse_list_argument(config.get("apical_expids"))),
                        )
                        cached_compartment = (
                            representative_obs.get("compartment")
                            or dendrite_record.get("compartment")
                            or cache.get("experiments", {}).get(representative_exp_id, {}).get("compartment")
                            or ""
                        )
                        warn_if_compartment_mismatch(representative_exp_id, expected_compartment, cached_compartment)
                        if expected_compartment is not None:
                            warned_mismatch_expids.add(representative_exp_id)

                    dendrite_compartment = str(
                        representative_obs.get("compartment")
                        or dendrite_record.get("compartment")
                        or cache.get("experiments", {}).get(representative_exp_id or "", {}).get("compartment")
                        or ""
                    ).strip()
                    figure_path = build_figure_save_path(output_dir, animal_id, day_summary.date, dendrite_compartment, global_dendrite_id)
                    try:
                        plot_day_figure(cache, day_summary, animal_id, global_dendrite_id, dendrite_record, figure_path, repo_base, dpi)
                    except FileNotFoundError as exc:
                        print(f"[skip] {animal_id} | {day_summary.date} | {global_dendrite_id}: {exc}")
                        continue
                    written.append(str(figure_path))
                    print(f"[saved] {figure_path}")
    return written


def build_arg_parser() -> Any:
    parser = build_pipeline_arg_parser()
    parser.description = "ROI-aware zebra movie wrapper that refreshes the cache and writes dendrite/spine figures"
    parser.set_defaults(config=DEFAULT_CONFIG_PATH)
    parser.add_argument("--figure-dpi", type=int, default=None, help="Figure DPI for the per-dendrite PNG and SVG outputs")
    parser.add_argument(
        "--stimulus-source-root",
        type=Path,
        default=None,
        help="Root directory that contains the zebra clip folders from all_movie_clips_bv_sets",
    )
    parser.add_argument(
        "--stimulus-cache-root",
        type=Path,
        default=None,
        help="Root directory for the local zebra stimulus caches",
    )
    parser.add_argument(
        "--gabor-save-path",
        type=Path,
        default=None,
        help="Target path for the WavEn-compatible Gabor library file",
    )
    parser.add_argument(
        "--build-full-gabor-library",
        action="store_true",
        help="Materialize the dense WavEn Gabor library instead of writing only the manifest",
    )
    return parser


def prepare_stimulus_assets(config: Dict[str, Any], discoveries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    exp_roots = []
    for item in discoveries:
        repo_root = item.get("repo_root")
        if not repo_root or item.get("analysis_mode") == "missing":
            continue
        exp_roots.append(Path(str(repo_root)))
    if not exp_roots:
        return None

    stimulus_source_root = Path(config.get("stimulus_source_root") or DEFAULT_STIMULUS_SOURCE_ROOT)
    stimulus_cache_root = Path(config.get("stimulus_cache_root") or DEFAULT_STIMULUS_CACHE_ROOT)
    gabor_save_path = Path(config.get("gabor_save_path") or DEFAULT_GABOR_LIBRARY_PATH)
    build_full_gabor_library = bool(config.get("build_full_gabor_library"))

    prep = prepare_zebra_stimulus_assets(
        exp_roots=exp_roots,
        stimulus_source_root=stimulus_source_root,
        stimulus_cache_root=stimulus_cache_root,
        gabor_save_path=gabor_save_path,
        build_full_gabor_library=build_full_gabor_library,
    )
    summary = {
        "stimulus_source_root": str(prep.stimulus_source_root),
        "stimulus_cache_root": str(prep.stimulus_cache_root),
        "encoded_video_cache_root": str(prep.encoded_video_cache_root),
        "gabor_library_path": str(prep.gabor_library_path),
        "gabor_manifest_path": str(prep.gabor_manifest_path),
        "gabor_materialized": bool(prep.gabor_materialized),
        "movie_clip_count": prep.n_clips,
        "movie_clip_reused": prep.n_reused,
        "movie_clip_rendered": prep.n_rendered,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = build_run_config(args)
    config = maybe_build_demo_repository(args, config)

    user_id = str(config.get("user_id") or "")
    if not user_id:
        raise SystemExit("Missing user_id. Provide --user-id or config file entry.")

    repo_base = Path(config.get("repo_base") or f"/home/{user_id}/data/Repository")
    output_dir = Path(args.output_dir) if args.output_dir is not None else DEFAULT_RESULTS_DIR
    cache_path = Path(args.cache_path) if args.cache_path is not None else (output_dir / DEFAULT_CACHE_NAME)
    figure_dpi = int(config.get("figure_dpi") or DEFAULT_DPI)
    config["output_dir"] = str(output_dir)
    config["cache_path"] = str(cache_path)
    ensure_dir(output_dir)
    ensure_dir(cache_path.parent)

    with step_scope("discover zebra inputs"):
        discoveries = discover_requested_experiments(config, repo_base)
    print_discovery_report(discoveries)

    with step_scope("prepare zebra assets"):
        prepare_stimulus_assets(config, discoveries)

    movie_expids = parse_list_argument(config.get("movie_expids"))
    sleep_expids = parse_list_argument(config.get("sleep_expids"))
    basal_expids = parse_list_argument(config.get("basal_expids"))
    apical_expids = parse_list_argument(config.get("apical_expids"))
    channel = int(config.get("channel") or 0)
    high_pass_hz = float(config.get("high_pass_hz") or 0.02)
    rebuild = bool(config.get("rebuild"))

    with step_scope("cache load or rebuild"):
        source_cache = load_or_build_cache(
            repo_base=repo_base,
            movie_expids=movie_expids,
            sleep_expids=sleep_expids,
            basal_expids=basal_expids,
            apical_expids=apical_expids,
            channel=channel,
            high_pass_hz=high_pass_hz,
            explicit_locomotion_threshold=config.get("locomotion_threshold"),
            cache_path=cache_path,
            rebuild=rebuild,
        )

    cache = source_cache
    with step_scope("zebra day figure pipeline"):
        written_figures = generate_day_figures(cache, config, output_dir, repo_base, figure_dpi)
    with step_scope("cleanup ROI detail figures"):
        removed = cleanup_roi_detail_figures(output_dir / "figures")
        if removed:
            step_message(f"removed {len(removed)} stale ROI detail PNG/SVG files")

    print(
        json.dumps(
            {
                "n_figures": len(written_figures),
                "output_dir": str(output_dir),
                "cache_path": str(cache_path),
                "figure_dpi": figure_dpi,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
