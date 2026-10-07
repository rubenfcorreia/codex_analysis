from __future__ import annotations

from itertools import combinations
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from scipy import stats
from analysis.shared.roi_split import annotate_rows_with_split_group
from analysis.shared.state_utils import canonical_state_label


def run_mixed_model_family(
    rows, response, scope, contrast_specs, shuffle_n, *, alerts=None, vc_level_keys=None, state_order=None, p_value_source="classical"
):
    """Fit the canonical animal-clustered mixed model for one response."""
    del vc_level_keys
    alerts = alerts if alerts is not None else []
    requested_source = str(p_value_source or "classical").strip().lower()
    effective_source = requested_source if requested_source == "classical" else "classical"
    if requested_source != "classical":
        alerts.append(f"[ALERT] Mixed-model shuffle p-values are not implemented for {response} ({scope}); using classical model p-values.")
    try:
        import pandas as pd
        from patsy import build_design_matrices
        from scipy import stats
        from statsmodels.regression.mixed_linear_model import MixedLM
    except Exception as exc:
        alerts.append(f"[ALERT] Mixed-model dependencies unavailable for {response} ({scope}): {exc}")
        return {"summary_rows": [], "contrast_rows": [], "design": {}, "equation": None, "tested_terms": [], "tested_contrasts": [], "validation_rows": [], "p_value_source": effective_source, "p_value_source_requested": requested_source, "fit": {"converged": False, "fit_method": "unavailable", "error": str(exc)}}

    records = []
    for row in rows:
        try:
            value = float(row.get(response))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        item = dict(row)
        item["response"] = value
        item["state"] = str(item.get("state") or "missing")
        item["animal_key"] = str(item.get("animal_id") or item.get("day_id") or "missing_animal")
        item["day_key"] = str(item.get("day_id") or item.get("expid") or "missing_day")
        item["unit_key"] = str(item.get("unit_id") or item.get("subject_id") or item.get("roi_key") or item["day_key"])
        item["visual_response_cohort"] = str(item.get("visual_response_cohort") or "nonresponsive")
        item["split_group"] = str(item.get("split_group") or "missing")
        item["compartment"] = str(item.get("compartment") or "all")
        records.append(item)
    if not records:
        return {"summary_rows": [], "contrast_rows": [], "design": {}, "equation": None, "tested_terms": [], "tested_contrasts": [], "validation_rows": [], "p_value_source": effective_source, "p_value_source_requested": requested_source, "fit": {"converged": False, "fit_method": "no_data"}}

    frame = pd.DataFrame(records)
    state_levels = [str(state) for state in (state_order or []) if str(state) in set(frame["state"])]
    state_levels += [state for state in sorted(frame["state"].unique()) if state not in state_levels]
    formula_terms = ["C(state)"]
    if frame["visual_response_cohort"].nunique() > 1:
        formula_terms.append("C(visual_response_cohort)")
        formula_terms.append("C(state):C(visual_response_cohort)")
    if frame["split_group"].nunique() > 1:
        formula_terms.append("C(split_group)")
        formula_terms.append("C(state):C(split_group)")
    if frame["compartment"].nunique() > 1:
        formula_terms.append("C(compartment)")
        formula_terms.append("C(state):C(compartment)")
    formula = "response ~ " + " + ".join(formula_terms)
    vc_formula = {"day": "0 + C(day_key)", "unit": "0 + C(unit_key)"}
    fit = None
    fit_method = "lbfgs"
    fit_error = None
    warning_messages = []
    try:
        model = MixedLM.from_formula(formula, groups="animal_key", re_formula="1", vc_formula=vc_formula, data=frame)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            warning_messages.extend(str(item.message) for item in caught)
        if not bool(getattr(fit, "converged", False)):
            fit_method = "powell"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(reml=False, method="powell", maxiter=300, disp=False)
                warning_messages.extend(str(item.message) for item in caught)
    except Exception as exc:
        fit_error = str(exc)
        # Small or perfectly balanced synthetic/cohort subsets can make a
        # variance component unidentifiable.  Retain the animal random
        # intercept and the identifiable ROI component rather than silently
        # reverting to an independent test.
        try:
            fit_method = "lbfgs_unit_fallback"
            model = MixedLM.from_formula(formula, groups="animal_key", re_formula="1", vc_formula={"unit": "0 + C(unit_key)"}, data=frame)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
                warning_messages.extend(str(item.message) for item in caught)
            if not bool(getattr(fit, "converged", False)):
                fit_method = "powell_unit_fallback"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = model.fit(reml=False, method="powell", maxiter=300, disp=False)
                    warning_messages.extend(str(item.message) for item in caught)
        except Exception as fallback_exc:
            fit_error = f"{exc}; fallback: {fallback_exc}"
            alerts.append(f"[ALERT] Mixed-model fit failed for {response} ({scope}): {fit_error}")

    singular_or_non_pd = any(any(token in message.lower() for token in ("not positive definite", "boundary", "singular")) for message in warning_messages)
    inferential_fit_ok = bool(fit is not None and getattr(fit, "converged", False) and not singular_or_non_pd)
    if singular_or_non_pd:
        alerts.append(f"[ALERT] Mixed-model fit for {response} ({scope}) emitted singular/boundary diagnostics; inferential contrasts are suppressed.")

    summary_rows = []
    for state in state_levels:
        subset = frame.loc[frame["state"] == state, "response"]
        if len(subset):
            summary_rows.append({"response": response, "scope": scope, "state": state, "mean": float(subset.mean()), "estimate": float(subset.mean()), "n": int(len(subset)), "n_animals": int(frame.loc[frame["state"] == state, "animal_key"].nunique()), "n_days": int(frame.loc[frame["state"] == state, "day_key"].nunique())})

    def prediction_row(state, *, cohort=None, split_group=None, compartment=None):
        row = {column: frame.iloc[0][column] for column in frame.columns}
        row["state"] = str(state)
        if cohort is not None:
            row["visual_response_cohort"] = str(cohort)
        if split_group is not None:
            row["split_group"] = str(split_group)
        if compartment is not None:
            row["compartment"] = str(compartment)
        return pd.DataFrame([row])

    def contrast(label, left_row, right_row, spec):
        if not inferential_fit_ok:
            return {"response": response, "scope": scope, "contrast_name": label, "estimate": float("nan"), "p_value": float("nan"), "p_value_source": effective_source, "status": "fit_failed_or_diagnostic_warning", **spec}
        try:
            left = build_design_matrices([fit.model.data.design_info], left_row, return_type="dataframe")[0].to_numpy(dtype=float)[0]
            right = build_design_matrices([fit.model.data.design_info], right_row, return_type="dataframe")[0].to_numpy(dtype=float)[0]
            vector = left - right
            estimate = float(vector @ fit.fe_params.to_numpy())
            covariance = np.asarray(fit.cov_params(), dtype=float)
            covariance = covariance[: len(vector), : len(vector)]
            se = float(np.sqrt(max(0.0, vector @ covariance @ vector)))
            z_value = estimate / se if se > 0 else float("nan")
            p_value = float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan")
            ci = 1.96 * se if np.isfinite(se) else float("nan")
            return {"response": response, "scope": scope, "contrast_name": label, "estimate": estimate, "standard_error": se, "lower_ci": estimate - ci if np.isfinite(ci) else float("nan"), "upper_ci": estimate + ci if np.isfinite(ci) else float("nan"), "p_value": p_value, "classical_p": p_value, "p_value_source": effective_source, "status": "ok", **spec}
        except Exception as exc:
            alerts.append(f"[ALERT] Mixed-model contrast failed for {response} ({label}): {exc}")
            return {"response": response, "scope": scope, "contrast_name": label, "estimate": float("nan"), "p_value": float("nan"), "p_value_source": effective_source, "status": "contrast_failed", **spec}

    contrast_rows = []
    reference_cohort = sorted(frame["visual_response_cohort"].unique())[0]
    reference_split = sorted(frame["split_group"].unique())[0]
    reference_compartment = sorted(frame["compartment"].unique())[0]
    for spec in contrast_specs or []:
        kind = spec.get("kind")
        if kind == "state_pair":
            a, b = str(spec.get("state_a")), str(spec.get("state_b"))
            if a in state_levels and b in state_levels:
                contrast_rows.append(contrast(f"{a} vs {b}", prediction_row(a), prediction_row(b), {"contrast_type": kind, "state_a": a, "state_b": b}))
        elif kind == "visual_response_cohort" and frame["visual_response_cohort"].nunique() > 1:
            levels = sorted(frame["visual_response_cohort"].unique())
            contrast_rows.append(contrast(f"{levels[1]} vs {levels[0]}", prediction_row(state_levels[0], cohort=levels[1]), prediction_row(state_levels[0], cohort=levels[0]), {"contrast_type": kind, "cohort_a": levels[1], "cohort_b": levels[0]}))
        elif kind == "split_group_pair" and frame["split_group"].nunique() > 1:
            contrast_rows.append(contrast(f"{spec.get('group_a')} vs {spec.get('group_b')}", prediction_row(spec.get("state"), split_group=spec.get("group_a")), prediction_row(spec.get("state"), split_group=spec.get("group_b")), {"contrast_type": kind, **spec}))
        elif kind == "basal_apical" and {"basal", "apical"}.issubset(set(frame["compartment"])):
            contrast_rows.append(contrast("apical vs basal", prediction_row(spec.get("state"), compartment="apical"), prediction_row(spec.get("state"), compartment="basal"), {"contrast_type": kind, **spec}))

    tested_terms = []
    if inferential_fit_ok:
        for term, p_value in fit.pvalues.items():
            tested_terms.append({"term": str(term), "estimate": float(fit.params.get(term, np.nan)), "p_value": float(p_value) if np.isfinite(p_value) else float("nan")})
    design = {"state_levels": state_levels, "formula": formula, "random_effect_group": "animal_key", "variance_components": list(vc_formula), "n_rows": int(len(frame)), "n_animals": int(frame["animal_key"].nunique()), "n_days": int(frame["day_key"].nunique()), "n_units": int(frame["unit_key"].nunique())}
    fit_payload = {"converged": bool(inferential_fit_ok), "fit_method": fit_method if fit is not None else "failed", "fallback_used": bool(fit is not None and "fallback" in fit_method), "singular_or_non_pd": bool(singular_or_non_pd), "warning_messages": warning_messages, "error": fit_error, "aic": float(fit.aic) if fit is not None and np.isfinite(fit.aic) else float("nan"), "bic": float(fit.bic) if fit is not None and np.isfinite(fit.bic) else float("nan")}
    return {"summary_rows": summary_rows, "contrast_rows": contrast_rows, "design": design, "equation": formula, "tested_terms": tested_terms, "tested_contrasts": contrast_rows, "validation_rows": [], "p_value_source": effective_source, "p_value_source_requested": requested_source, "fit": fit_payload}

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
