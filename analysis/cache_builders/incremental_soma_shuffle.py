"""Selective reuse for the soma/bouton shared permutation cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from analysis.shared.cache_utils import load_npz_cache, save_npz_cache


def load_or_build(
    legacy_loader: Callable[..., tuple[dict[str, Any], bool, str]],
    path: Path,
    *,
    metadata: dict[str, Any],
    rebuild: bool,
    metadata_hash: Callable[[Mapping[str, Any]], str],
) -> tuple[dict[str, Any], bool, str]:
    """Reuse old entries when only the requested experiment set changed."""
    if not rebuild and path.exists():
        try:
            old = load_npz_cache(path)
            old_meta = old.get("meta", {})
            variable = {"movie_expids", "sleep_expids"}
            stable_old = {key: value for key, value in old_meta.items() if key not in variable}
            stable_new = {key: value for key, value in metadata.items() if key not in variable}
            if stable_old == stable_new and isinstance(old.get("entries"), dict):
                old["meta"] = metadata
                old["meta_hash"] = metadata_hash(metadata)
                old.setdefault("reuse_stats", {})["experiment_set_changed"] = old_meta.get("movie_expids") != metadata.get("movie_expids") or old_meta.get("sleep_expids") != metadata.get("sleep_expids")
                save_npz_cache(path, old)
                return old, False, "reused_incremental"
        except Exception:
            pass
    return legacy_loader(path, metadata=metadata, rebuild=rebuild)
