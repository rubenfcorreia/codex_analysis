from __future__ import annotations

from typing import List, Optional, Sequence, Mapping


def normalize_analysis_families(
    values: Optional[Sequence[str] | str],
    *,
    allowed_families: Sequence[str],
) -> List[str]:
    allowed = [str(family).strip() for family in allowed_families if str(family).strip()]
    if values is None:
        return list(allowed)
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",") if part.strip()]
    else:
        raw = [str(value).strip() for value in values if str(value).strip()]
    if not raw:
        return list(allowed)
    selected: List[str] = []
    for family in allowed:
        if family in raw and family not in selected:
            selected.append(family)
    unknown = [family for family in raw if family not in allowed]
    if unknown:
        raise SystemExit(
            f"Unknown analysis family/families: {', '.join(unknown)}. Allowed values are: {', '.join(allowed)}"
        )
    return selected


def analysis_families_to_text(
    values: Optional[Sequence[str] | str],
    *,
    allowed_families: Sequence[str],
) -> str:
    return ",".join(normalize_analysis_families(values, allowed_families=allowed_families))


_FAMILY_RUNNERS = {}


def register_family(name: str, runner):
    key = str(name).strip().lower()
    if not key:
        raise ValueError("analysis family name cannot be empty")
    _FAMILY_RUNNERS[key] = runner
    return runner


def family_names() -> List[str]:
    return sorted(_FAMILY_RUNNERS)


def run_family(name: str, *args, **kwargs):
    key = str(name).strip().lower()
    try:
        runner = _FAMILY_RUNNERS[key]
    except KeyError as exc:
        available = ", ".join(family_names()) or "none"
        raise KeyError(f"Unknown analysis family {name!r}; registered families: {available}") from exc
    return runner(*args, **kwargs)


def run_analysis_family(name, inputs, scope, config=None):
    from .contracts import AnalysisFamilyResult, AnalysisScope
    from . import calcium_events, coincidence, mixed_model, state, visual_response
    config = dict(config or {})
    rows = inputs.get("rows", []) if isinstance(inputs, dict) else inputs
    rows = [dict(row) for row in (rows or []) if isinstance(row, Mapping)]
    if not isinstance(scope, AnalysisScope):
        scope = AnalysisScope(**dict(scope))
    key = str(name).strip().lower()
    if key == "state":
        raw = state.build_state_family_results(rows, list(scope.selected_states), int(config.get("shuffle_n", 0) or 0))
    elif key == "mixed_model":
        raw = mixed_model.run_family(rows, state_comparison_states=list(scope.selected_states), shuffle_n=int(config.get("shuffle_n", 0) or 0), mixed_model_contrast_p_source=str(config.get("mixed_model_contrast_p_source", "classical")))
    elif key == "calcium_events":
        raw = calcium_events.run_family(rows)
    elif key == "coincidence":
        raw = coincidence.run_family(rows)
    elif key == "visual_response":
        raw = visual_response.run_family(rows, cohort=str(config.get("cohort", "all")), response_metric=config.get("response_metric"))
    else:
        runner = _FAMILY_RUNNERS.get(key)
        if runner is None:
            raise KeyError(f"Unknown analysis family {name!r}")
        raw = runner(rows, scope=scope, config=config)
    raw = dict(raw or {})
    result_rows = raw.get("rows", raw.get("table_rows", rows))
    if not isinstance(result_rows, list):
        result_rows = []
    return AnalysisFamilyResult(family=key, scope=scope, tables=raw, rows=[dict(row) for row in result_rows if isinstance(row, Mapping)], metadata={"contract_version": "1"}, alerts=list(raw.get("alerts", [])))
