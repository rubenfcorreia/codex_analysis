from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from analysis.shared.roi_split import ROI_SPLIT_BASES, ROI_SPLIT_BRANCHES
from analysis.shared.state_utils import canonical_state_label, safe_filename_component

ANALYSIS_BRANCHES: Tuple[str, ...] = tuple(ROI_SPLIT_BRANCHES)
ANALYSIS_BASES: Tuple[str, ...] = tuple(ROI_SPLIT_BASES)


def iter_branch_basis_leaves(
    branches: Sequence[str] | None = None,
    bases: Sequence[str] | None = None,
) -> Iterator[Tuple[str, str]]:
    branch_names = tuple(str(branch).strip().lower() for branch in (branches or ANALYSIS_BRANCHES) if str(branch).strip())
    basis_names = tuple(str(basis).strip().lower() for basis in (bases or ANALYSIS_BASES) if str(basis).strip())
    for branch_name in branch_names:
        for basis_name in basis_names:
            yield branch_name, basis_name


def branch_leaf_root(result_root: Path | str, branch_name: Any, basis_name: Any, preset_name: Any | None = None) -> Path:
    root = Path(result_root) / safe_filename_component(branch_name)
    preset_text = str(preset_name or "").strip()
    if preset_text:
        root /= safe_filename_component(preset_text)
    return root / safe_filename_component(basis_name)


def branch_leaf_figure_root(result_root: Path | str, branch_name: Any, basis_name: Any, preset_name: Any | None = None) -> Path:
    return branch_leaf_root(result_root, branch_name, basis_name, preset_name=preset_name) / "figures"


def basis_state_labels(basis_name: Any) -> List[str]:
    basis = canonical_state_label(basis_name)
    if basis in {"nrem", "rem"}:
        return [basis]
    return []


def scoped_analysis_state_selection(selection: Mapping[str, Any] | None, basis_name: Any) -> Dict[str, Any]:
    scoped = dict(selection or {})
    labels = basis_state_labels(basis_name)
    if labels:
        scoped["state_comparison_states"] = list(labels)
        scoped["basal_apical_states"] = list(labels)
    return scoped


def _row_state_labels(row: Mapping[str, Any]) -> List[str]:
    labels: List[str] = []
    for key in (
        "state",
        "state_a",
        "state_b",
        "state_label",
        "state_display",
        "comparison_state",
        "selected_state",
    ):
        value = row.get(key)
        if value is None:
            continue
        text = canonical_state_label(value)
        if not text:
            continue
        if text not in labels:
            labels.append(text)
    return labels


def scope_rows_for_basis(
    rows: Sequence[Mapping[str, Any]] | None,
    basis_name: Any,
    *,
    sleep_expids: Sequence[Any] | None = None,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    basis = canonical_state_label(basis_name)
    if basis not in {"nrem", "rem"}:
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    sleep_ids = {str(expid).strip() for expid in (sleep_expids or []) if str(expid).strip()}
    scoped: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        labels = _row_state_labels(row)
        if labels and not all(label == basis or label.startswith(f"{basis}_") for label in labels):
            continue
        if sleep_ids:
            expid = str(row.get("expid") or row.get("exp_id") or "").strip()
            if expid and expid not in sleep_ids:
                continue
        scoped.append(dict(row))
    return scoped


def select_roi_split_leaf(
    roi_split_results: Mapping[str, Any] | None,
    *,
    branch_name: Any,
    basis_name: Any,
) -> Dict[str, Any]:
    if not isinstance(roi_split_results, Mapping):
        return {
            "branches": {},
            "bundles": [],
            "subject_state_rows": [],
            "membership_rows": [],
            "comparison_rows": [],
            "summary_rows": [],
            "counts": {
                "subject_state_rows": 0,
                "membership_rows": 0,
                "comparison_rows": 0,
                "summary_rows": 0,
                "bundles": 0,
                "branches": 0,
                "basis_leaves": 0,
            },
        }

    branch_text = str(branch_name or "").strip().lower()
    basis_text = canonical_state_label(basis_name)
    bundles = [
        dict(bundle)
        for bundle in roi_split_results.get("bundles", [])
        if isinstance(bundle, Mapping)
        and str(bundle.get("branch_name") or "").strip().lower() == branch_text
        and canonical_state_label(bundle.get("basis_name")) == basis_text
    ]
    subject_state_rows = [dict(row) for bundle in bundles for row in bundle.get("subject_state_rows", []) if isinstance(row, Mapping)]
    membership_rows = [dict(row) for bundle in bundles for row in bundle.get("membership_rows", []) if isinstance(row, Mapping)]
    comparison_rows = [dict(row) for bundle in bundles for row in bundle.get("comparison_rows", []) if isinstance(row, Mapping)]
    summary_rows = [dict(row) for bundle in bundles for row in bundle.get("summary_rows", []) if isinstance(row, Mapping)]
    branches: Dict[str, Dict[str, Any]] = {}
    if bundles:
        branches.setdefault(branch_text, {})[basis_text] = bundles
    return {
        "branches": branches,
        "bundles": bundles,
        "subject_state_rows": subject_state_rows,
        "membership_rows": membership_rows,
        "comparison_rows": comparison_rows,
        "summary_rows": summary_rows,
        "counts": {
            "subject_state_rows": int(len(subject_state_rows)),
            "membership_rows": int(len(membership_rows)),
            "comparison_rows": int(len(comparison_rows)),
            "summary_rows": int(len(summary_rows)),
            "bundles": int(len(bundles)),
            "branches": int(1 if bundles else 0),
            "basis_leaves": int(1 if bundles else 0),
        },
    }


def scoped_branch_results(
    results: Mapping[str, Any],
    *,
    branch_name: Any,
    basis_name: Any,
    sleep_expids: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    scoped = dict(results)
    scoped["analysis_branch_name"] = str(branch_name or "").strip().lower()
    scoped["analysis_basis_name"] = canonical_state_label(basis_name)
    scoped["analysis_scope"] = {
        "branch_name": scoped["analysis_branch_name"],
        "basis_name": scoped["analysis_basis_name"],
    }
    scoped["analysis_state_selection"] = scoped_analysis_state_selection(
        results.get("analysis_state_selection") if isinstance(results, Mapping) else None,
        basis_name,
    )
    scoped["roi_split"] = select_roi_split_leaf(
        results.get("roi_split") if isinstance(results, Mapping) else None,
        branch_name=branch_name,
        basis_name=basis_name,
    )
    scoped["analysis_sleep_expids"] = [str(expid) for expid in (sleep_expids or []) if str(expid).strip()]
    return scoped


__all__ = [
    "ANALYSIS_BASES",
    "ANALYSIS_BRANCHES",
    "basis_state_labels",
    "branch_leaf_figure_root",
    "branch_leaf_root",
    "iter_branch_basis_leaves",
    "scope_rows_for_basis",
    "scoped_analysis_state_selection",
    "scoped_branch_results",
    "select_roi_split_leaf",
]
