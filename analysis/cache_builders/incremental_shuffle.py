"""Reuse compatible shuffle entries when the experiment set changes."""

from __future__ import annotations

from typing import Any, Mapping


def reuse_compatible_entries(existing: Mapping[str, Any] | None, target: dict[str, Any]) -> dict[str, Any]:
    """Keep old deterministic shuffle vectors for still-valid keys.

    The target cache defines the active key set, so removed experiments are
    pruned automatically. Entries are reused only when their vector length and
    replicate count still match the requested computation.
    """
    old_entries = existing.get("entries", {}) if isinstance(existing, Mapping) else {}
    if not isinstance(old_entries, Mapping):
        return target
    reused = 0
    for key, entry in list(target.get("entries", {}).items()):
        old = old_entries.get(key)
        if isinstance(old, Mapping) and int(old.get("vector_length", -1)) == int(entry.get("vector_length", -2)) and int(old.get("shuffle_n", -1)) == int(entry.get("shuffle_n", -2)):
            target["entries"][key] = dict(old)
            reused += 1
    target["reuse_stats"] = {"reused_entries": reused, "new_entries": max(0, len(target.get("entries", {})) - reused)}
    return target
