"""Create a manifest for a staged cache artifact.

This transitional command is useful while existing pipeline-specific builders
are being moved behind the staged interface.  It never invents analysis data;
it records the upstream dependency and fails if the upstream artifact is
missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cache_builders.common import CACHE_STAGES, cache_manifest_path, dependency_signature, load_json_manifest, save_json_manifest, stage_manifest
from analysis.shared.cache_config import normalize_staged_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--stage", choices=CACHE_STAGES[1:], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open() as handle:
        config = normalize_staged_config(json.load(handle), pipeline=args.pipeline)
    upstream = load_json_manifest(args.upstream_manifest)
    if upstream is None:
        raise SystemExit(f"Missing upstream cache manifest: {args.upstream_manifest}")
    manifest = stage_manifest(
        pipeline=args.pipeline,
        stage=args.stage,
        config_signature=dependency_signature(config, args.stage),
        upstream_signature=dependency_signature(upstream),
        active_expids=list(upstream.get("active_expids", [])),
        records={"status": "manifest_only_transitional", "upstream_manifest": str(args.upstream_manifest)},
    )
    save_json_manifest(cache_manifest_path({**config, "cache_paths": {args.stage: str(args.config.parent / "cache" / args.stage)}}, args.stage), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
