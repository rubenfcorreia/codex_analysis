"""Run a real legacy pipeline stage through the staged-cache interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from analysis.shared.cache_config import normalize_staged_config


def run_stage(pipeline: str, stage: str, config_path: Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with config_path.open() as handle:
        raw = json.load(handle)
    config = normalize_staged_config(raw, pipeline=pipeline, base_dir=config_path.parent)
    if overrides:
        config.update(dict(overrides))
    if stage not in {"for_analysis", "results"}:
        raise ValueError("Real execution is currently supported for analysis and results stages")
    if stage == "results":
        config["plots_only"] = True
    if pipeline == "soma_bouton":
        from analysis.soma_bouton_pipeline import soma_bouton_pipeline as impl
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=str(config_path.parent), delete=False) as handle:
            json.dump(config, handle, indent=2, default=str)
            normalized_path = Path(handle.name)
        try:
            config = impl.load_config(normalized_path)
            if stage == "results":
                config["plots_only"] = True
            return impl.run_pipeline(config)
        finally:
            normalized_path.unlink(missing_ok=True)
    if pipeline == "dendrites":
        from analysis.dendrites_pipeline import dendrites_pipeline as impl
        # The legacy dendrites entry point imports analysis_families as a script-local package.
        # Expose its directory when the staged runner imports it as a package.
        import sys
        module_dir = str(Path(impl.__file__).resolve().parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        # The dendrite CLI owns its default-merging and path validation.
        argv = ["--config", str(config_path)]
        if stage == "results":
            argv.append("--plots-only")
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(config, handle, indent=2, default=str)
            normalized_path = handle.name
        try:
            argv = ["--config", normalized_path]
            if stage == "results":
                argv.append("--plots-only")
            status = impl.main(argv)
        finally:
            Path(normalized_path).unlink(missing_ok=True)
        return {"pipeline": pipeline, "stage": stage, "exit_code": int(status or 0)}
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--stage", choices=("for_analysis", "results"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage(args.pipeline, args.stage, args.config), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
