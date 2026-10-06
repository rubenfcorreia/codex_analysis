from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from analysis.shared.shared_calcium_response import (
    DEFAULT_VISUAL_RESPONSE_COHORT,
    VISUAL_RESPONSE_COHORTS,
    get_active_visual_response_metric,
    summarize_visual_response_entity_rows,
)


def _visual_response_entity_id(row: Mapping[str, Any]) -> str:
    compartment = str(row.get("compartment") or "").strip().lower()
    if compartment == "soma":
        value = row.get("global_soma_id")
    elif compartment == "bouton":
        value = row.get("global_bouton_id")
    else:
        value = (row.get("global_soma_id") or row.get("global_bouton_id")
                 or row.get("global_dendrite_id") or row.get("global_spine_id"))
    return str(value).strip() if value is not None else ""


def _coerce_visual_response_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.bool_)):
        try:
            return bool(int(value))
        except Exception:
            return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "on"}


def _canonicalize_visual_response_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate within analysis scope, never across modes or days."""
    grouped: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entity_id = _visual_response_entity_id(row)
        if not entity_id:
            continue
        key = (
            str(row.get("day_id") or row.get("expid") or ""),
            str(row.get("mode") or ""),
            str(row.get("compartment") or ""),
            entity_id,
        )
        grouped.setdefault(key, []).append(dict(row))
    canonical_rows: List[Dict[str, Any]] = []
    for (_, _, _, entity_id), members in grouped.items():
        row = dict(members[0])
        responsive = any(_coerce_visual_response_bool(member.get("responsive", False)) or str(member.get("cohort") or "").strip().lower() == "responsive" for member in members)
        row["responsive"] = responsive
        row["cohort"] = "responsive" if responsive else "nonresponsive"
        row["entity_id"] = entity_id
        canonical_rows.append(row)
    return canonical_rows


def _apply_bh_fdr(rows: List[Dict[str, Any]]) -> None:
    """Apply Benjamini-Hochberg correction independently per response family."""
    families: Dict[tuple[str, str, str], List[int]] = {}
    for index, row in enumerate(rows):
        try:
            p_value = float(row.get("raw_pvalue", float("nan")))
        except (TypeError, ValueError):
            p_value = float("nan")
        if np.isfinite(p_value):
            key = (str(row.get("mode") or ""), str(row.get("compartment") or ""), str(row.get("response_metric") or ""))
            families.setdefault(key, []).append(index)
    for indices in families.values():
        ordered = sorted(indices, key=lambda index: float(rows[index]["raw_pvalue"]))
        m = len(ordered)
        adjusted = [1.0] * m
        running = 1.0
        for rank in range(m, 0, -1):
            p_value = float(rows[ordered[rank - 1]]["raw_pvalue"])
            running = min(running, p_value * m / rank)
            adjusted[rank - 1] = min(running, 1.0)
        for index, adjusted_p in zip(ordered, adjusted):
            row = rows[index]
            delta = float(row.get("delta", float("nan")))
            row["adjusted_pvalue"] = float(adjusted_p)
            row["significant"] = bool(adjusted_p < 0.05)
            row["responsive"] = bool(row["significant"] and np.isfinite(delta) and delta > 0)
            row["cohort"] = "responsive" if row["responsive"] else "nonresponsive"
            row["star"] = "*" if row["significant"] else ""


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
                "mean_visual": float(np.nanmean(visual)) if np.isfinite(visual).any() else float("nan"),
                "mean_blank": float(np.nanmean(blank)) if np.isfinite(blank).any() else float("nan"),
                "mean_delta": float(np.nanmean(delta)) if np.isfinite(delta).any() else float("nan"),
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
    deduped_rows = _canonicalize_visual_response_rows(row_list)
    _apply_bh_fdr(deduped_rows)
    summary = summarize_visual_response_entity_rows(deduped_rows)
    metric = get_active_visual_response_metric(response_metric or summary.get("response_metric"))

    cohort_counts = {label: 0 for label in VISUAL_RESPONSE_COHORTS}
    cohort_rows: Dict[str, List[Dict[str, Any]]] = {label: [] for label in VISUAL_RESPONSE_COHORTS}
    by_compartment: Dict[str, List[Dict[str, Any]]] = {}
    for row in deduped_rows:
        label = str(row.get("cohort") or DEFAULT_VISUAL_RESPONSE_COHORT)
        if label not in cohort_rows:
            continue
        row_copy = dict(row)
        cohort_rows[label].append(row_copy)
        cohort_counts[label] += 1
        compartment = str(row_copy.get("compartment") or "all")
        by_compartment.setdefault(compartment, []).append(row_copy)

    selected_rows = cohort_rows.get(cohort, deduped_rows) if cohort in cohort_rows else deduped_rows
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
        "available": bool(deduped_rows),
        "cohort": cohort,
        "response_metric": metric,
        "summary": summary,
        "rows": deduped_rows,
        "day_rows": visual_response_day_rows(deduped_rows),
        "cohort_rows": cohort_rows,
        "by_compartment": by_compartment,
        "selected_by_compartment": selected_by_compartment,
        "cohort_counts": cohort_counts,
        "selected_rows": selected_rows,
        "responsive_rows": selected_responsive_rows,
        "nonresponsive_rows": selected_nonresponsive_rows,
        "counts": {
            "all": int(len(deduped_rows)),
            "responsive": int(sum(bool(row.get("responsive", False)) for row in deduped_rows)),
            "nonresponsive": int(sum(not bool(row.get("responsive", False)) for row in deduped_rows)),
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


def apply_visual_response_fdr(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return ROI rows with family-wise BH-FDR classification applied."""
    output = [dict(row) for row in rows if isinstance(row, Mapping)]
    _apply_bh_fdr(output)
    return output


def run_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort: str = DEFAULT_VISUAL_RESPONSE_COHORT,
    response_metric: Optional[str] = None,
) -> Dict[str, Any]:
    return build_visual_response_family_results(rows, cohort=cohort, response_metric=response_metric)


__all__ = [
    "apply_visual_response_fdr",
    "build_visual_response_family_results",
    "run_family",
    "visual_response_day_rows",
]
