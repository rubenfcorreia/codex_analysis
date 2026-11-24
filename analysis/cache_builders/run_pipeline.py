"""Run the complete staged cache dependency graph for one pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.cache_builders.build_classification_cache import build_classification_cache
from analysis.cache_builders.build_initial_analysis import build_initial_analysis
from analysis.cache_builders.build_processed_cache import build_processed_cache
from analysis.cache_builders.build_source_cache import build_source_cache
from analysis.cache_builders.common import cache_root
from analysis.cache_builders.run_pipeline_stage import run_stage


def _payload_path(config: dict[str, Any], stage: str) -> Path:
    return cache_root(config, stage) / f"{stage}_cache.npz"


def run_pipeline(config_path: Path, pipeline: str) -> dict[str, Any]:
    source_manifest = build_source_cache(config_path, pipeline)
    with config_path.open() as handle:
        config = json.load(handle)
    source_path = Path(str(source_manifest["records"]["source_cache_path"]))
    processed_path = _payload_path(config, "processed")
    initial_path = _payload_path(config, "initial_analysis")
    classification_path = _payload_path(config, "classification")
    processed_manifest = build_processed_cache(config_path, pipeline, source_path, processed_path)
    initial_manifest = build_initial_analysis(config_path, pipeline, processed_path, initial_path)
    classification_manifest = build_classification_cache(config_path, pipeline, initial_path, classification_path)
    analysis_result = run_stage(
        pipeline,
        "for_analysis",
        config_path,
        overrides={
            "cache_path": str(source_path),
            "classification_cache_path": str(classification_path),
            "processed_cache_path": str(processed_path),
            "initial_analysis_cache_path": str(initial_path),
        },
    )
    results_result = run_stage(pipeline, "results", config_path)
    return {
        "pipeline": pipeline,
        "stages": {
            "source": source_manifest,
            "processed": processed_manifest,
            "initial_analysis": initial_manifest,
            "classification": classification_manifest,
            "for_analysis": analysis_result,
            "results": results_result,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.config, args.pipeline), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
