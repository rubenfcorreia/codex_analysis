from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from analysis.shared.shared_calcium_response import (
    DEFAULT_VISUAL_RESPONSE_COHORT,
    VISUAL_RESPONSE_COHORTS,
    get_active_visual_response_metric,
    summarize_visual_response_entity_rows,
)


def visual_response_day_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    grouped: Dict[tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("day_id") or ""),
            str(row.get("mode") or ""),
            str(row.get("compartment") or ""),
            str(row.get("cohort") or "nonresponsive"),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (day_id, mode, compartment, cohort), members in grouped.items():
        visual = np.asarray([float(row.get("mean_visual", float("nan"))) for row in members], dtype=float)
        blank = np.asarray([float(row.get("mean_blank", float("nan"))) for row in members], dtype=float)
        delta = np.asarray([float(row.get("delta", float("nan"))) for row in members], dtype=float)
        responsive = sum(bool(row.get("responsive", False)) for row in members)
        summary_rows.append(
            {
                "day_id": day_id,
                "mode": mode,
                "compartment": compartment,
                "cohort": cohort,
                "n_rois": int(len(members)),
                "n_responsive": int(responsive),
                "responsive_fraction": float(responsive / len(members)) if members else float("nan"),
                "mean_visual": float(np.nanmean(visual)) if visual.size else float("nan"),
                "mean_blank": float(np.nanmean(blank)) if blank.size else float("nan"),
                "mean_delta": float(np.nanmean(delta)) if delta.size else float("nan"),
            }
        )
    return summary_rows


def build_visual_response_family_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort: str = DEFAULT_VISUAL_RESPONSE_COHORT,
    response_metric: Optional[str] = None,
) -> Dict[str, Any]:
    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    summary = summarize_visual_response_entity_rows(row_list)
    metric = get_active_visual_response_metric(response_metric or summary.get("response_metric"))

    cohort_counts = {label: 0 for label in VISUAL_RESPONSE_COHORTS}
    cohort_rows: Dict[str, List[Dict[str, Any]]] = {label: [] for label in VISUAL_RESPONSE_COHORTS}
    by_compartment: Dict[str, List[Dict[str, Any]]] = {}
    for row in row_list:
        label = str(row.get("cohort") or DEFAULT_VISUAL_RESPONSE_COHORT)
        if label not in cohort_rows:
            continue
        row_copy = dict(row)
        cohort_rows[label].append(row_copy)
        cohort_counts[label] += 1
        compartment = str(row_copy.get("compartment") or "all")
        by_compartment.setdefault(compartment, []).append(row_copy)

    selected_rows = cohort_rows.get(cohort, row_list) if cohort in cohort_rows else row_list
    selected_responsive_rows = [dict(row) for row in selected_rows if bool(row.get("responsive", False))]
    selected_nonresponsive_rows = [dict(row) for row in selected_rows if not bool(row.get("responsive", False))]
    selected_by_compartment = {
        compartment: [
            dict(row) for row in rows_for_compartment
            if cohort == "all" or str(row.get("cohort") or DEFAULT_VISUAL_RESPONSE_COHORT) == cohort
        ]
        for compartment, rows_for_compartment in by_compartment.items()
    }

    return {
        "available": bool(row_list),
        "cohort": cohort,
        "response_metric": metric,
        "summary": summary,
        "rows": row_list,
        "day_rows": visual_response_day_rows(row_list),
        "cohort_rows": cohort_rows,
        "by_compartment": by_compartment,
        "selected_by_compartment": selected_by_compartment,
        "cohort_counts": cohort_counts,
        "selected_rows": selected_rows,
        "responsive_rows": selected_responsive_rows,
        "nonresponsive_rows": selected_nonresponsive_rows,
        "counts": {
            "all": int(len(row_list)),
            "responsive": int(sum(bool(row.get("responsive", False)) for row in row_list)),
            "nonresponsive": int(sum(not bool(row.get("responsive", False)) for row in row_list)),
        },
        "counts_by_compartment": {
            compartment: {
                "all": int(len(rows_for_compartment)),
                "responsive": int(sum(bool(row.get("responsive", False)) for row in rows_for_compartment)),
                "nonresponsive": int(sum(not bool(row.get("responsive", False)) for row in rows_for_compartment)),
            }
            for compartment, rows_for_compartment in by_compartment.items()
        },
    }


def run_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort: str = DEFAULT_VISUAL_RESPONSE_COHORT,
    response_metric: Optional[str] = None,
) -> Dict[str, Any]:
    return build_visual_response_family_results(rows, cohort=cohort, response_metric=response_metric)


__all__ = [
    "build_visual_response_family_results",
    "run_family",
    "visual_response_day_rows",
]
