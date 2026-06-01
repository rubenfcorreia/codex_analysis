#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sleep_dendrite_spine_poster_figure import (
    DEFAULT_SPINE_COACTIVITY_HEIGHT_CM,
    DEFAULT_SPINE_COACTIVITY_OUTPUT_STEM,
    DEFAULT_SPINE_COACTIVITY_WIDTH_CM,
    _save_svg_figure_exact,
    build_spine_coactivity_poster_figure,
    ensure_dir,
    load_analysis_results_payload,
    load_cache,
    set_svg_physical_size,
)

DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "results" / "main_pipeline" / "cache" / "sleep_dendrite_spine_cache.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "poster_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the poster-ready spine-coactivity figure.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the raw sleep_dendrite_spine_cache.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the poster-ready SVG will be saved.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_SPINE_COACTIVITY_OUTPUT_STEM,
        help="Base filename stem for the exported figure.",
    )
    parser.add_argument(
        "--width-cm",
        type=float,
        default=DEFAULT_SPINE_COACTIVITY_WIDTH_CM,
        help="Target figure width in centimeters.",
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=DEFAULT_SPINE_COACTIVITY_HEIGHT_CM,
        help="Target figure height in centimeters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = load_cache(args.cache_path)
    results = load_analysis_results_payload()
    if not isinstance(results, dict) or not results:
        raise RuntimeError("No analysis_results payload was available for the spine-coactivity poster.")
    output_dir = ensure_dir(args.output_dir)
    output_stem = str(args.output_stem).strip() or DEFAULT_SPINE_COACTIVITY_OUTPUT_STEM

    # Generate the requested sidecar SVGs first so they exist independently.
    from sleep_dendrite_spine_poster_figure import render_spine_coactivity_component_svgs

    render_spine_coactivity_component_svgs(results, output_dir)
    figure, _, _ = build_spine_coactivity_poster_figure(
        cache,
        results,
        width_cm=float(args.width_cm),
        height_cm=float(args.height_cm),
    )
    svg_path = output_dir / f"{output_stem}.svg"
    _save_svg_figure_exact(figure, svg_path)
    set_svg_physical_size(svg_path, float(args.width_cm))
    print(f"Saved SVG: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
