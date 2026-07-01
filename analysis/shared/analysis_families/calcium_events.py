from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def calcium_event_summary_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    grouped: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("day_id") or ""),
            str(row.get("state") or ""),
            str(row.get("compartment") or ""),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (day_id, state, compartment), members in grouped.items():
        event_frequency = np.asarray([float(row.get("event_frequency_per_min", float("nan"))) for row in members], dtype=float)
        event_count = np.asarray([float(row.get("event_count", float("nan"))) for row in members], dtype=float)
        activity = np.asarray([float(row.get("mean", float("nan"))) for row in members], dtype=float)
        summary_rows.append(
            {
                "day_id": day_id,
                "state": state,
                "compartment": compartment,
                "n_rows": int(len(members)),
                "mean_activity": float(np.nanmean(activity)) if activity.size else float("nan"),
                "mean_event_frequency_per_min": float(np.nanmean(event_frequency)) if event_frequency.size else float("nan"),
                "mean_event_count": float(np.nanmean(event_count)) if event_count.size else float("nan"),
            }
        )
    return summary_rows


def build_calcium_event_family_results(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    return {
        "available": bool(row_list),
        "rows": row_list,
        "summary_rows": calcium_event_summary_rows(row_list),
        "counts": {
            "all": int(len(row_list)),
            "soma": int(sum(str(row.get("compartment")) == "soma" for row in row_list)),
            "bouton": int(sum(str(row.get("compartment")) == "bouton" for row in row_list)),
        },
    }


def run_family(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return build_calcium_event_family_results(rows)


__all__ = [
    "build_calcium_event_family_results",
    "calcium_event_summary_rows",
    "run_family",
]
