from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from analysis.dendrites_pipeline.dendrites_pipeline import run_mixed_model_family
from analysis.shared.roi_split import annotate_rows_with_split_group
from analysis.shared.state_utils import canonical_state_label



def _mixed_model_table_from_rows(rows: Sequence[Mapping[str, Any]], compartment: Optional[str] = None) -> List[Dict[str, Any]]:
    table_rows: List[Dict[str, Any]] = []
    compartment_filter = str(compartment or "").strip().lower() or None
    for row in rows:
        row_compartment = str(row.get("compartment") or "").strip().lower()
        if row_compartment not in {"soma", "bouton"}:
            continue
        if compartment_filter is not None and row_compartment != compartment_filter:
            continue
        roi_id = row.get("roi_id")
        if roi_id is None or str(roi_id).strip() == "":
            roi_id = row.get(f"{row_compartment}_id")
        if roi_id is None or str(roi_id).strip() == "":
            roi_id = row.get("roi_index")
        if row_compartment == "soma":
            unit_id = str(row.get("global_soma_id") or "").strip()
        else:
            unit_id = str(row.get("global_bouton_id") or "").strip()
        if not unit_id:
            continue
        table_rows.append({
            "animal_id": row.get("animal_id"),
            "day_id": row.get("day_id"),
            "expid": row.get("expid"),
            "mode": row.get("mode"),
            "state": row.get("state"),
            "compartment": row_compartment,
            "channel": row.get("channel"),
            "roi_id": roi_id,
            "unit_id": unit_id,
            "subject_id": unit_id,
            "soma_id": row.get("soma_id"),
            "bouton_id": row.get("bouton_id"),
            "soma_unit_id": row.get("soma_unit_id"),
            "bouton_unit_id": row.get("bouton_unit_id"),
            "global_soma_id": row.get("global_soma_id"),
            "global_bouton_id": row.get("global_bouton_id"),
            "roi_key": row.get("roi_key") or unit_id,
            "roi_index": row.get("roi_index"),
            "visual_response_cohort": str(row.get("cohort") or "nonresponsive"),
            "mean_activity": float(row.get("mean", float("nan"))),
            "event_frequency_per_min": float(row.get("event_frequency_per_min", float("nan"))),
        })
    return table_rows


def _contrast_specs(state_comparison_states: Sequence[str], include_visual_response: bool) -> List[Dict[str, Any]]:
    state_pairs = [
        {"kind": "state_pair", "state_a": state_a, "state_b": state_b}
        for state_a, state_b in combinations([state for state in state_comparison_states if state], 2)
    ]
    visual_response_pairs = [{"kind": "visual_response_cohort"}] if include_visual_response else []
    return state_pairs + visual_response_pairs


def _run_branch(
    table_rows: Sequence[Mapping[str, Any]],
    *,
    compartment: str,
    scope: str,
    state_order: Sequence[str],
    state_comparison_states: Sequence[str],
    shuffle_n: int,
    p_value_source: str,
    state_filter: Sequence[str] | None = None,
    vc_level_keys: Sequence[str] | None = ("unit_id",),
) -> Dict[str, Any]:
    responses = ["mean_activity", "event_frequency_per_min"]
    include_visual_response = any(str(row.get("visual_response_cohort") or "nonresponsive") == "responsive" for row in table_rows) and any(str(row.get("visual_response_cohort") or "nonresponsive") == "nonresponsive" for row in table_rows)
    contrast_specs = _contrast_specs(state_comparison_states, include_visual_response)
    branch: Dict[str, Any] = {
        "available": bool(table_rows),
        "compartment": compartment,
        "p_value_source": p_value_source,
        "p_value_source_requested": p_value_source,
        "summary_rows": {},
        "contrast_rows": [],
        "designs": {},
        "model_equations": {},
        "tested_terms": {},
        "tested_contrasts": {},
        "selection": {
            "compartment": compartment,
            "state_comparison_states": list(state_comparison_states),
            "visual_response_cohort": None,
        },
        "validation_rows": [],
        "alerts": [],
    }
    alerts: List[str] = branch["alerts"]
    working_rows = list(table_rows)
    if state_filter is not None:
        state_filter_set = {str(state).strip() for state in state_filter if state is not None and str(state).strip()}
        working_rows = [row for row in working_rows if str(row.get("state")) in state_filter_set]
    if not working_rows:
        return branch
    if not state_order:
        state_order = sorted({str(row.get("state") or "") for row in working_rows if row.get("state")})
    valid_state_order = [state for state in state_order if any(str(row.get("state")) == state for row in working_rows)]
    branch_state_order = valid_state_order or list(state_order)
    for response in responses:
        result = run_mixed_model_family(
            list(working_rows),
            response,
            scope,
            contrast_specs,
            shuffle_n,
            alerts=alerts,
            vc_level_keys=vc_level_keys,
            state_order=branch_state_order,
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
    basal_apical_states: Sequence[str] | None = None,
    shuffle_n: int,
    mixed_model_contrast_p_source: str = "classical",
    vc_level_keys: Sequence[str] | None = ("unit_id",),
) -> Dict[str, Any]:
    del basal_apical_states
    compartments = [compartment for compartment in ("soma", "bouton") if any(str(row.get("compartment") or "").strip().lower() == compartment for row in activity_rows)]
    if not compartments:
        compartments = ["all"]
    state_order = [state for state in state_comparison_states if state]
    if not state_order:
        state_order = sorted({str(row.get("state") or "") for row in activity_rows if row.get("state")})
    results: Dict[str, Any] = {}
    for compartment in compartments:
        compartment_rows = _mixed_model_table_from_rows(activity_rows, None if compartment == "all" else compartment)
        if not compartment_rows:
            continue
        results[compartment] = {
            "selected_state": _run_branch(
                compartment_rows,
                compartment=compartment,
                scope="selected_state",
                state_order=state_order,
                state_comparison_states=state_comparison_states,
                shuffle_n=shuffle_n,
                p_value_source=mixed_model_contrast_p_source,
                state_filter=state_comparison_states,
                vc_level_keys=vc_level_keys,
            ),
        }
    return results


def run_split_family(
    table_rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    *,
    response_columns: Sequence[str],
    state_comparison_states: Sequence[str],
    shuffle_n: int,
    mixed_model_contrast_p_source: str = "classical",
    scope: str = "selected_state",
    state_filter: Sequence[str] | None = None,
    vc_level_keys: Sequence[str] | None = ("unit_id",),
    split_group_column: str = "split_group",
) -> Dict[str, Any]:
    annotated_rows = annotate_rows_with_split_group(table_rows, membership_rows, group_column=split_group_column)
    working_rows = [row for row in annotated_rows if str(row.get(split_group_column) or "").strip()]
    if state_filter is not None:
        state_filter_set = {str(state).strip() for state in state_filter if state is not None and str(state).strip()}
        working_rows = [row for row in working_rows if str(row.get("state")) in state_filter_set]
    response_names = [str(column).strip() for column in response_columns if str(column).strip()]
    p_value_source = str(mixed_model_contrast_p_source or "classical").strip().lower()
    if p_value_source not in {"classical", "shuffle"}:
        p_value_source = "classical"
    branch: Dict[str, Any] = {
        "available": bool(working_rows) and bool(response_names),
        "p_value_source": p_value_source,
        "p_value_source_requested": p_value_source,
        "summary_rows": {},
        "contrast_rows": [],
        "designs": {},
        "model_equations": {},
        "tested_terms": {},
        "tested_contrasts": {},
        "selection": {
            "state_comparison_states": [canonical_state_label(state) for state in state_comparison_states if canonical_state_label(state)],
            "split_group_column": split_group_column,
            "split_group_levels": [],
            "split_group_reference": None,
        },
        "validation_rows": [],
        "alerts": [],
        "table_rows": working_rows,
    }
    alerts: List[str] = branch["alerts"]
    if not working_rows or not response_names:
        return branch
    state_order = [canonical_state_label(state) for state in state_comparison_states if canonical_state_label(state)]
    if not state_order:
        state_order = [canonical_state_label(row.get("state")) for row in working_rows if canonical_state_label(row.get("state"))]
    state_order = list(dict.fromkeys(state_order))
    if not state_order:
        state_order = sorted({str(row.get("state") or "") for row in working_rows if row.get("state")})
    valid_state_order = [state for state in state_order if any(str(row.get("state")) == state for row in working_rows)]
    branch_state_order = valid_state_order or list(state_order)

    split_group_rank: Dict[str, float] = {}
    split_group_levels: List[str] = []
    for row in working_rows:
        group = str(row.get(split_group_column) or "").strip()
        if not group:
            continue
        if group not in split_group_levels:
            split_group_levels.append(group)
        rank_value = row.get(f"{split_group_column}_rank")
        try:
            rank_float = float(rank_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rank_float):
            continue
        current = split_group_rank.get(group)
        if current is None or rank_float < current:
            split_group_rank[group] = rank_float
    if split_group_rank:
        split_group_levels = sorted(split_group_levels, key=lambda group: (split_group_rank.get(group, float("inf")), group))
    branch["selection"]["split_group_levels"] = list(split_group_levels)
    branch["selection"]["split_group_reference"] = split_group_levels[0] if split_group_levels else None

    include_visual_response = any(str(row.get("visual_response_cohort") or "nonresponsive") == "responsive" for row in working_rows) and any(str(row.get("visual_response_cohort") or "nonresponsive") == "nonresponsive" for row in working_rows)
    contrast_specs = _contrast_specs(branch_state_order, include_visual_response)
    if len(split_group_levels) == 2:
        reference_group = split_group_levels[0]
        comparison_group = split_group_levels[1]
        for state in branch_state_order:
            contrast_specs.append({
                "kind": "split_group_pair",
                "state": state,
                "group_a": comparison_group,
                "group_b": reference_group,
            })
    compartment_levels = {
        str(row.get("compartment") or "").strip().lower()
        for row in working_rows
        if str(row.get("compartment") or "").strip()
    }
    if {"basal", "apical"}.issubset(compartment_levels):
        for state in branch_state_order:
            contrast_specs.append({
                "kind": "basal_apical",
                "state": state,
            })

    for response in response_names:
        result = run_mixed_model_family(
            list(working_rows),
            response,
            scope,
            contrast_specs,
            shuffle_n,
            alerts=alerts,
            vc_level_keys=vc_level_keys,
            state_order=branch_state_order,
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


__all__ = ["run_family", "run_split_family"]
