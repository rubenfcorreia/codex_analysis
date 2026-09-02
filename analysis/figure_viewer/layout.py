from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from analysis.figure_viewer.catalog import filter_records, unique_values
from analysis.figure_viewer.models import FigureFilterState, FigureRecord

HIERARCHY_FIELDS: Tuple[str, ...] = (
    "pipeline",
    "preset",
    "split",
    "basis",
    "family",
    "compartment",
    "cohort",
    "scope",
)

FIELD_LABELS: Mapping[str, str] = {
    "pipeline": "Pipeline",
    "preset": "Preset",
    "split": "Split",
    "basis": "Basis",
    "family": "Family",
    "compartment": "Compartment",
    "cohort": "Cohort",
    "scope": "Scope",
}


@dataclass(frozen=True)
class SlotSelection:
    pipeline: str = ""
    preset: str = ""
    split: str = ""
    basis: str = ""
    family: str = ""
    compartment: str = ""
    cohort: str = ""
    scope: str = ""
    figure_key: str = ""
    initialized: bool = False

    def normalized(self) -> "SlotSelection":
        return SlotSelection(
            pipeline=str(self.pipeline or "").strip(),
            preset=str(self.preset or "").strip(),
            split=str(self.split or "").strip(),
            basis=str(self.basis or "").strip(),
            family=str(self.family or "").strip(),
            compartment=str(self.compartment or "").strip(),
            cohort=str(self.cohort or "").strip(),
            scope=str(self.scope or "").strip(),
            figure_key=str(self.figure_key or "").strip(),
            initialized=bool(self.initialized),
        )

    def filter_state(self) -> FigureFilterState:
        normalized = self.normalized()
        return FigureFilterState(
            pipeline=normalized.pipeline,
            preset=normalized.preset,
            split=normalized.split,
            basis=normalized.basis,
            family=normalized.family,
            cohort=normalized.cohort,
            scope=normalized.scope,
        )


@dataclass
class BrowserNode:
    name: str
    path_parts: Tuple[str, ...]
    record: FigureRecord | None = None
    children: Dict[str, "BrowserNode"] = field(default_factory=dict)
    figure_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.record is not None


def repo_relative_parts(path: Path, repo_root: Path) -> Tuple[str, ...]:
    try:
        return tuple(path.resolve().relative_to(repo_root.resolve()).parts)
    except Exception:
        return tuple(path.parts)


def build_results_index(records: Sequence[FigureRecord], repo_root: Path) -> BrowserNode:
    root = BrowserNode(name="results", path_parts=("results",))
    for record in records:
        parts = repo_relative_parts(record.preview_path, repo_root)
        if not parts or parts[0] != "results":
            continue
        node = root
        node.figure_count += 1
        for part in parts[1:]:
            child = node.children.get(part)
            if child is None:
                child = BrowserNode(name=part, path_parts=node.path_parts + (part,))
                node.children[part] = child
            node = child
            node.figure_count += 1
        node.record = record
    return root


def browser_children(node: BrowserNode) -> List[BrowserNode]:
    return sorted(node.children.values(), key=lambda child: (child.is_leaf, child.name.lower()))


def browser_node_label(node: BrowserNode) -> str:
    if node.record is not None:
        return node.record.preview_path.name
    if node.path_parts == ("results",):
        return "results"
    if node.figure_count:
        return f"{node.name} ({node.figure_count})"
    return node.name


def prefix_filter_state(selection: SlotSelection, stop_field: str | None = None) -> FigureFilterState:
    selection = selection.normalized()
    values: Dict[str, str] = {}
    for field_name in HIERARCHY_FIELDS:
        if stop_field == field_name:
            break
        value = getattr(selection, field_name)
        if value:
            values[field_name] = value
    return FigureFilterState(**values)


def field_options(records: Sequence[FigureRecord], selection: SlotSelection, field_name: str) -> List[str]:
    return unique_values(filter_records(records, prefix_filter_state(selection, field_name)), field_name)


def selection_from_record(record: FigureRecord) -> SlotSelection:
    return SlotSelection(
        pipeline=record.pipeline,
        preset=record.preset,
        split=record.split,
        basis=record.basis,
        family=record.family,
        compartment=record.compartment,
        cohort=record.cohort,
        scope=record.scope,
        figure_key=record.figure_key,
        initialized=True,
    )


def selection_with_field(selection: SlotSelection, field_name: str, value: str) -> SlotSelection:
    if field_name not in HIERARCHY_FIELDS:
        raise ValueError(f"Unknown field: {field_name}")
    selection = selection.normalized()
    values = {field: getattr(selection, field) for field in HIERARCHY_FIELDS}
    values[field_name] = str(value or "").strip()
    if not values[field_name] and not any(values[field] for field in HIERARCHY_FIELDS):
        return SlotSelection()
    return SlotSelection(
        pipeline=values["pipeline"],
        preset=values["preset"],
        split=values["split"],
        basis=values["basis"],
        family=values["family"],
        compartment=values["compartment"],
        cohort=values["cohort"],
        scope=values["scope"],
        figure_key="",
        initialized=True,
    )
def resolve_selection(
    records: Sequence[FigureRecord],
    selection: SlotSelection,
    *,
    preserve_figure_key: str = "",
) -> Tuple[SlotSelection, List[FigureRecord]]:
    selection = selection.normalized()
    if not selection.initialized and not any(getattr(selection, field) for field in HIERARCHY_FIELDS) and not selection.figure_key:
        return selection, []

    cleaned_values: Dict[str, str] = {}
    for field_name in HIERARCHY_FIELDS:
        current_value = getattr(selection, field_name)
        if not current_value:
            break
        prefix_selection = SlotSelection(**cleaned_values, initialized=bool(cleaned_values))
        options = field_options(records, prefix_selection, field_name)
        if current_value not in options:
            break
        cleaned_values[field_name] = current_value

    candidate_records = filter_records(records, FigureFilterState(**cleaned_values))
    selected_record: FigureRecord | None = None
    if preserve_figure_key:
        selected_record = next((record for record in candidate_records if record.figure_key == preserve_figure_key), None)
    if selected_record is None and selection.figure_key:
        selected_record = next((record for record in candidate_records if record.figure_key == selection.figure_key), None)
    if selected_record is None and len(candidate_records) == 1:
        selected_record = candidate_records[0]

    return (
        SlotSelection(
            **cleaned_values,
            figure_key=selected_record.figure_key if selected_record else "",
            initialized=bool(selection.initialized or selection.figure_key or cleaned_values or selected_record is not None),
        ),
        candidate_records,
    )
def comparison_signature(records: Sequence[FigureRecord | None]) -> str:
    return "|".join(record.figure_key for record in records if record is not None and record.figure_key)


def comparison_label(records: Sequence[FigureRecord | None]) -> str:
    labels = [
        record.display_label or record.title or record.preview_path.name
        for record in records
        if record is not None
    ]
    return " / ".join(labels) if labels else "Comparison"
