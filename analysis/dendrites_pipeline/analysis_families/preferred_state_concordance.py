from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from analysis.shared.plots.dendrite_plot_support import observation_compartment


def preferred_state(state_means: Mapping[str, Any] | None, state_order: Sequence[str]) -> Optional[str]:
    """Return the first state with the largest finite response mean."""
    if not isinstance(state_means, Mapping):
        return None
    candidates = []
    for order, state in enumerate(state_order):
        try:
            value = float(state_means.get(state))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            candidates.append((value, -order, str(state)))
    if not candidates:
        return None
    return max(candidates)[2]


def _in_scope(exp_id: str, experiments: Mapping[str, Any], scope_keys: Optional[Sequence[str]]) -> bool:
    allowed = {str(value) for value in (scope_keys or [])}
    if not allowed:
        return True
    date = experiments.get(exp_id, {}).get("date") if isinstance(experiments.get(exp_id), Mapping) else None
    return f"{exp_id}|{exp_id}" in allowed or f"{exp_id}|{date or exp_id}" in allowed


def run_preferred_state_concordance_analysis(
    cache: Mapping[str, Any],
    *,
    state_comparison_states: Sequence[str],
    scope_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    states = [str(state) for state in state_comparison_states if str(state)]
    experiments = cache.get("experiments", {})
    table_rows = []
    dendrite_groups = defaultdict(list)
    alerts = []

    for animal_id, animal_entry in (cache.get("animals", {}) or {}).items():
        for dendrite_id, dendrite_record in (animal_entry.get("dendrites", {}) or {}).items():
            for exp_id, dendrite_obs in (dendrite_record.get("observations", {}) or {}).items():
                exp_id = str(dendrite_obs.get("exp_id") or exp_id)
                if not _in_scope(exp_id, experiments, scope_keys):
                    continue
                dendrite_state = preferred_state(dendrite_obs.get("cut_state_means"), states)
                if dendrite_state is None:
                    continue
                compartment = observation_compartment(cache, exp_id, dendrite_obs)
                for spine_id in dendrite_obs.get("spine_ids", ()):
                    spine_record = (dendrite_record.get("spines", {}) or {}).get(spine_id, {})
                    spine_obs = (spine_record.get("observations", {}) or {}).get(exp_id)
                    if not isinstance(spine_obs, Mapping):
                        continue
                    spine_state = preferred_state(spine_obs.get("cut_state_means"), states)
                    if spine_state is None:
                        continue
                    row = {
                        "analysis": "preferred_state_concordance",
                        "animal_id": str(animal_id),
                        "exp_id": exp_id,
                        "day_id": exp_id,
                        "compartment": compartment,
                        "global_dendrite_id": str(dendrite_id),
                        "global_spine_id": str(spine_id),
                        "dendrite_preferred_state": dendrite_state,
                        "spine_preferred_state": spine_state,
                        "preferred_state_match": bool(dendrite_state == spine_state),
                    }
                    table_rows.append(row)
                    dendrite_groups[(str(animal_id), exp_id, str(dendrite_id), compartment)].append(row)

    dendrite_rows = []
    confusion = defaultdict(int)
    for (animal_id, exp_id, dendrite_id, compartment), rows in sorted(dendrite_groups.items()):
        matches = sum(bool(row["preferred_state_match"]) for row in rows)
        dendrite_state = rows[0]["dendrite_preferred_state"]
        dendrite_row = {
            "analysis": "preferred_state_concordance",
            "animal_id": animal_id,
            "exp_id": exp_id,
            "day_id": exp_id,
            "compartment": compartment,
            "global_dendrite_id": dendrite_id,
            "dendrite_preferred_state": dendrite_state,
            "n_spines": len(rows),
            "n_matching_spines": matches,
            "matching_spine_fraction": matches / len(rows),
        }
        dendrite_rows.append(dendrite_row)
        for row in rows:
            confusion[(row["dendrite_preferred_state"], row["spine_preferred_state"])] += 1

    total = len(table_rows)
    summary = {
        "n_spines": total,
        "n_dendrites": len(dendrite_rows),
        "n_matching_spines": sum(bool(row["preferred_state_match"]) for row in table_rows),
        "spine_match_fraction": (sum(bool(row["preferred_state_match"]) for row in table_rows) / total) if total else float("nan"),
        "dendrite_match_fraction_mean": float(np.mean([row["matching_spine_fraction"] for row in dendrite_rows])) if dendrite_rows else float("nan"),
    }
    confusion_rows = [
        {"dendrite_preferred_state": dendrite_state, "spine_preferred_state": spine_state, "n_spines": count}
        for (dendrite_state, spine_state), count in sorted(confusion.items())
    ]
    if not table_rows:
        alerts.append("No dendrite/spine pairs had valid preferred-state labels.")
    return {"table_rows": table_rows, "dendrite_rows": dendrite_rows, "confusion_rows": confusion_rows, "summary": summary, "state_labels": states, "alerts": alerts}


__all__ = ["preferred_state", "run_preferred_state_concordance_analysis"]
