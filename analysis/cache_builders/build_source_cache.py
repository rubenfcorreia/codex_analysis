"""Build or validate the pipeline source cache as a standalone command.

This is a compatibility boundary around the existing experiment loaders.  It
does not duplicate scientific preprocessing; the staged manifest records which
existing source cache was produced and with which configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from analysis.cache_builders.common import cache_manifest_path, cache_root, dependency_signature, save_json_manifest, stage_manifest
from analysis.shared.cache_config import normalize_staged_config
from analysis.cache_builders.incremental_source import load_or_build_incremental
from analysis.cache_builders.incremental_soma_source import load_or_build_incremental as load_or_build_soma_incremental


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def build_source_cache(config_path: Path, pipeline: str) -> Dict[str, Any]:
    raw = _load_json(config_path)
    config = normalize_staged_config(raw, pipeline=pipeline, base_dir=config_path.parent)
    if pipeline == "dendrites":
        from analysis.dendrites_pipeline import dendrites_pipeline as impl

        merged = dict(impl.USER_EDITABLE_DEFAULTS)
        merged.update(config)
        config = normalize_staged_config(merged, pipeline=pipeline)
        impl.apply_cpu_thread_limit(config.get("cpu_thread_limit", 1))
        repo_base = impl.resolve_repo_path(config.get("repo_base") or (Path.home() / "data" / "Repository"), impl.REPO_ROOT)
        output_dir = impl.resolve_repo_path(config.get("output_dir") or impl.DEFAULT_RESULTS_DIR, impl.REPO_ROOT)
        cache_path = impl.resolve_repo_path(config.get("cache_path") or (cache_root(config, "source") / "source_cache.npz"), impl.REPO_ROOT)
        cache = load_or_build_incremental(impl, 
            repo_base=repo_base,
            movie_expids=config.get("movie_expids", []),
            sleep_expids=config.get("sleep_expids", []),
            basal_expids=config.get("basal_expids", []),
            apical_expids=config.get("apical_expids", []),
            channel=int(config.get("channel", 0)),
            explicit_locomotion_threshold=config.get("locomotion_threshold"),
            cache_path=cache_path,
            rebuild=bool(config.get("rebuild") or config.get("source_cache_rebuild")),
            validate_existing_cache=bool(config.get("source_cache_validate", True)),
            force_rebuild_expids=config.get("rebuild_expids", []),
        )
        expids = list(cache.get("experiments", {}).keys())
        signature = dependency_signature(cache.get("config_hash"), cache.get("experiments", {}))
    elif pipeline == "soma_bouton":
        from analysis.soma_bouton_pipeline import soma_bouton_pipeline as impl

        config = impl.load_config(config_path)
        impl._apply_cpu_thread_limit(config.get("cpu_thread_limit", 1))
        repo_base = impl.resolve_repo_path(config.get("repo_base") or (Path.home() / "data" / "Repository"), Path(__file__).resolve().parents[2])
        result_root = impl.resolve_repo_path(config.get("result_root") or "results/soma_bouton_pipeline", repo_base)
        cache_path = impl.resolve_repo_path(config.get("cache_path") or (cache_root(config, "source") / "source_cache.npz"), repo_base)
        expids_by_mode = {mode: list(dict.fromkeys(str(value) for value in config.get(f"{mode}_expids", []) if str(value))) for mode in ("movie", "sleep")}
        cache = load_or_build_soma_incremental(impl, config, repo_base, cache_path, expids_by_mode, rebuild=bool(config.get("rebuild") or config.get("source_cache_rebuild")), force_rebuild_expids=config.get("rebuild_expids", []))
        expids = [str(row.get("exp_id") or row.get("expid")) for row in cache.get("experiments", []) if row.get("exp_id") or row.get("expid")]
        signature = str(cache.get("source_signature") or dependency_signature(cache.get("metadata"), cache.get("experiments")))
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    manifest = stage_manifest(
        pipeline=pipeline,
        stage="source",
        config_signature=dependency_signature(config),
        upstream_signature=None,
        active_expids=expids,
        records={"source_cache_path": str(cache_path), "source_signature": signature},
    )
    save_json_manifest(cache_manifest_path({**config, "cache_paths": {"source": cache_path.parent}}, "source"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("dendrites", "soma_bouton"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_source_cache(args.config, args.pipeline)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
