from __future__ import annotations

from typing import Any, Mapping, Sequence

from analysis.shared.state_utils import canonical_state_label

SPLIT_BRANCHES = frozenset({"activity_split", "frequency_split", "activity_frequency_split"})


def _row_scope_matches(row: Mapping[str, Any], branch: str, basis: str) -> bool:
    row_branch = canonical_state_label(row.get("analysis_branch_name") or row.get("branch_name"))
    row_basis = canonical_state_label(row.get("analysis_basis_name") or row.get("basis_name"))
    return (not row_branch or row_branch == canonical_state_label(branch)) and (not row_basis or row_basis == canonical_state_label(basis))


def scope_split_rows(rows: Sequence[Mapping[str, Any]], *, branch: str, basis: str, selected_states: Sequence[str] = ()) -> list[dict[str, Any]]:
    selected = {canonical_state_label(state) for state in selected_states if canonical_state_label(state)}
    scoped: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not _row_scope_matches(row, branch, basis):
            continue
        state = canonical_state_label(row.get("state") or row.get("state_label"))
        if selected and state and state not in selected:
            continue
        copied = dict(row)
        copied["analysis_branch_name"] = canonical_state_label(branch)
        copied["analysis_basis_name"] = canonical_state_label(basis)
        if str(copied.get("split_group") or "").strip():
            scoped.append(copied)
    return scoped


def split_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values = []
    for row in rows:
        value = row.get("split_group") or row.get("group") or row.get("split_group_display") or row.get("group_display")
        text = str(value or "").strip()
        if text:
            values.append(text)
    return tuple(dict.fromkeys(values))


def require_split_groups(rows: Sequence[Mapping[str, Any]], *, branch: str, basis: str, minimum: int = 2) -> tuple[str, ...]:
    groups = split_groups(rows)
    if canonical_state_label(branch) in SPLIT_BRANCHES and len(groups) < minimum:
        raise ValueError(f"{branch}/{basis} requires at least {minimum} split groups; found {list(groups)}")
    return groups


__all__ = ["SPLIT_BRANCHES", "require_split_groups", "scope_split_rows", "split_groups"]
