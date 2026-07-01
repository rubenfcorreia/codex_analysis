from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from analysis.main_pipeline.sleep_dendrite_spine_pipeline import run_mixed_model_family



def _mixed_model_table_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    table_rows: List[Dict[str, Any]] = []
    for row in rows:
        compartment = str(row.get("compartment") or "")
        if compartment not in {"soma", "bouton"}:
            continue
        table_rows.append({
            "animal_id": row.get("animal_id"),
            "day_id": row.get("day_id"),
            "expid": row.get("expid"),
            "mode": row.get("mode"),
            "state": row.get("state"),
            "compartment": "basal" if compartment == "soma" else "apical",
            "visual_response_cohort": str(row.get("cohort") or "nonresponsive"),
            "mean_activity": float(row.get("mean", float("nan"))),
            "event_frequency_per_min": float(row.get("event_frequency_per_min", float("nan"))),
        })
    return table_rows


def _contrast_specs(state_comparison_states: Sequence[str], basal_apical_states: Sequence[str]) -> List[Dict[str, Any]]:
    state_pairs = [
        {"kind": "state_pair", "state_a": state_a, "state_b": state_b}
        for state_a, state_b in combinations([state for state in state_comparison_states if state], 2)
    ]
    basal_apical_pairs = [{"kind": "basal_apical", "state": state} for state in basal_apical_states if state]
    return state_pairs + basal_apical_pairs


def _run_branch(
    table_rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    state_order: Sequence[str],
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    p_value_source: str,
) -> Dict[str, Any]:
    responses = ["mean_activity", "event_frequency_per_min"]
    contrast_specs = _contrast_specs(state_comparison_states, basal_apical_states)
    branch: Dict[str, Any] = {
        "available": bool(table_rows),
        "p_value_source": p_value_source,
        "p_value_source_requested": p_value_source,
        "summary_rows": {},
        "contrast_rows": [],
        "designs": {},
        "model_equations": {},
        "tested_terms": {},
        "tested_contrasts": {},
        "selection": {
            "state_comparison_states": list(state_comparison_states),
            "basal_apical_states": list(basal_apical_states),
            "visual_response_cohort": None,
        },
        "validation_rows": [],
        "alerts": [],
    }
    alerts: List[str] = branch["alerts"]
    for response in responses:
        result = run_mixed_model_family(
            list(table_rows),
            response,
            scope,
            contrast_specs,
            shuffle_n,
            alerts=alerts,
            state_order=state_order,
            p_value_source=p_value_source,
        )
        branch["summary_rows"][response] = list(result.get("summary_rows", []))
        branch["contrast_rows"].extend(result.get("contrast_rows", []))
        if result.get("design") is not None:
            branch["designs"][response] = result.get("design")
        branch["model_equations"][response] = result.get("equation")
        branch["tested_terms"][response] = list(result.get("tested_terms", []))
        branch["tested_contrasts"][response] = list(result.get("tested_contrasts", []))
        branch["p_value_source"] = result.get("p_value_source", p_value_source)
        branch["p_value_source_requested"] = result.get("p_value_source_requested", p_value_source)
        branch["validation_rows"].extend(result.get("validation_rows", []))
    return branch


def run_family(
    activity_rows: Sequence[Mapping[str, Any]],
    *,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    mixed_model_contrast_p_source: str = "classical",
) -> Dict[str, Any]:
    table_rows = _mixed_model_table_from_rows(activity_rows)
    state_order = [state for state in state_comparison_states if state]
    if not state_order:
        state_order = sorted({str(row.get("state") or "") for row in table_rows if row.get("state")})
    selected_state_rows = [row for row in table_rows if str(row.get("state") or "") in set(state_order)]
    return {
        "all_state": _run_branch(
            table_rows,
            scope="all_state",
            state_order=state_order,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            shuffle_n=shuffle_n,
            p_value_source=mixed_model_contrast_p_source,
        ),
        "selected_state": _run_branch(
            selected_state_rows,
            scope="selected_state",
            state_order=state_order,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            shuffle_n=shuffle_n,
            p_value_source=mixed_model_contrast_p_source,
        ),
    }

__all__ = ["run_family"]
