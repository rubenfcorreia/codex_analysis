"""Build shuffle artifacts through the pipeline's real analysis path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cache_builders.run_pipeline_stage import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.pipeline, "for_analysis", args.config), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
