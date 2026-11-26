"""Incremental source-cache orchestration for the legacy dendrite loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.shared.cache_utils import load_npz_cache, save_npz_cache


def _remove_experiment(cache: dict[str, Any], exp_id: str) -> None:
    cache.get("experiments", {}).pop(exp_id, None)
    for animal in list(cache.get("animals", {}).values()):
        for dendrite_id, dendrite in list(animal.get("dendrites", {}).items()):
            dendrite.get("observations", {}).pop(exp_id, None)
            dendrite.get("local_ids", {}).pop(exp_id, None)
            for spine_id, spine in list(dendrite.get("spines", {}).items()):
                spine.get("observations", {}).pop(exp_id, None)
                spine.get("local_ids", {}).pop(exp_id, None)
                if not spine.get("observations") and not spine.get("local_ids"):
                    dendrite["spines"].pop(spine_id, None)
            if not dendrite.get("observations") and not dendrite.get("spines"):
                animal["dendrites"].pop(dendrite_id, None)
        if not animal.get("dendrites"):
            cache["animals"].pop(animal.get("animal_id"), None)


def load_or_build_incremental(
    impl: Any,
    *,
    repo_base: Path,
    movie_expids: Sequence[str],
    sleep_expids: Sequence[str],
    basal_expids: Sequence[str],
    apical_expids: Sequence[str],
    channel: int,
    explicit_locomotion_threshold: float | None,
    cache_path: Path,
    rebuild: bool = False,
    validate_existing_cache: bool = True,
    force_rebuild_expids: Sequence[str] = (),
) -> dict[str, Any]:
    requested = {str(value) for value in (*movie_expids, *sleep_expids)}
    requested_config = {
        "repo_base": str(repo_base), "movie_expids": sorted(set(movie_expids)),
        "sleep_expids": sorted(set(sleep_expids)), "basal_expids": sorted(set(basal_expids)),
        "apical_expids": sorted(set(apical_expids)), "channel": channel,
        "explicit_locomotion_threshold": explicit_locomotion_threshold,
    }
    old = None if rebuild or not cache_path.exists() else load_npz_cache(cache_path)
    old_config = old.get("config", {}) if isinstance(old, dict) else {}
    stable = isinstance(old, dict) and old.get("schema_version") == impl.CACHE_SCHEMA_VERSION and all(
        old_config.get(key) == requested_config.get(key)
        for key in requested_config
    )
    cache = old if stable else {"schema_version": impl.CACHE_SCHEMA_VERSION, "animals": {}, "experiments": {}, "alerts": [], "demo_truth": None}
    cache["config"] = requested_config
    cache["config_hash"] = impl.stable_hash(requested_config)
    for exp_id in list(cache.get("experiments", {})):
        if str(exp_id) not in requested:
            _remove_experiment(cache, str(exp_id))
    rebuild_ids = set(requested) - set(str(value) for value in cache.get("experiments", {}))
    rebuild_ids.update(str(value) for value in force_rebuild_expids if str(value) in requested)
    if validate_existing_cache and stable:
        for exp_id, metadata in list(cache.get("experiments", {}).items()):
            if str(exp_id) not in requested:
                continue
            try:
                paths = impl.experiment_source_paths(repo_base, exp_id, channel)
                conversion, source_exp, _ = impl.locate_conversion_file(repo_base, impl.derive_animal_id(exp_id), exp_id, prefer_same_day_source=exp_id in set(sleep_expids))
                compartment = impl.resolve_experiment_compartment(exp_id, basal_expids, apical_expids, sleep_expids, same_day_source_exp_id=source_exp)
                signature = impl.experiment_source_signature(paths, conversion, compartment, channel)
                if signature != metadata.get("source_signature"):
                    rebuild_ids.add(str(exp_id))
            except Exception:
                rebuild_ids.add(str(exp_id))
    for exp_id in sorted(rebuild_ids):
        _remove_experiment(cache, exp_id)
        try:
            processed = impl.process_experiment(
                repo_base=repo_base, exp_id=exp_id, channel=channel, movie_expids=movie_expids,
                sleep_expids=sleep_expids, basal_expids=basal_expids, apical_expids=apical_expids,
                explicit_locomotion_threshold=explicit_locomotion_threshold,
            )
        except Exception as exc:
            cache.setdefault("alerts", []).append(f"[ALERT] Failed to process {exp_id}: {exc}")
            continue
        impl.merge_animal_cache(cache, processed)
        cache.setdefault("alerts", []).extend(processed.get("exp_meta", {}).get("alerts", []))
    impl.strip_gabor_fields(cache)
    save_npz_cache(cache_path, cache)
    return cache
