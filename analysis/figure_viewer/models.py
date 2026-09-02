from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class FigureFilterState:
    pipeline: str = ""
    preset: str = ""
    split: str = ""
    basis: str = ""
    family: str = ""
    compartment: str = ""
    cohort: str = ""
    scope: str = ""
    search: str = ""

    def normalized(self) -> "FigureFilterState":
        return FigureFilterState(
            pipeline=str(self.pipeline).strip(),
            preset=str(self.preset).strip(),
            split=str(self.split).strip(),
            basis=str(self.basis).strip(),
            family=str(self.family).strip(),
            compartment=str(self.compartment).strip(),
            cohort=str(self.cohort).strip(),
            scope=str(self.scope).strip(),
            search=str(self.search).strip(),
        )


@dataclass(frozen=True)
class FigureRecord:
    figure_key: str
    display_label: str
    title: str
    preview_path: Path
    comparison_key: str = ""
    comparison_label: str = ""
    source_paths: Tuple[Path, ...] = field(default_factory=tuple)
    source_kinds: Tuple[str, ...] = field(default_factory=tuple)
    pipeline: str = ""
    preset: str = ""
    split: str = ""
    basis: str = ""
    family: str = ""
    cohort: str = ""
    scope: str = ""
    compartment: str = ""
    variant: str = ""
    source_root: str = ""
    manifest_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    search_text: str = ""
    sort_key: Tuple[str, ...] = field(default_factory=tuple)

    def as_context(self) -> Dict[str, Any]:
        return {
            "figure_key": self.figure_key,
            "display_label": self.display_label,
            "title": self.title,
            "comparison_key": self.comparison_key,
            "comparison_label": self.comparison_label,
            "preview_path": str(self.preview_path),
            "source_paths": [str(path) for path in self.source_paths],
            "source_kinds": list(self.source_kinds),
            "pipeline": self.pipeline,
            "preset": self.preset,
            "split": self.split,
            "basis": self.basis,
            "family": self.family,
            "cohort": self.cohort,
            "scope": self.scope,
            "compartment": self.compartment,
            "variant": self.variant,
            "source_root": self.source_root,
            "manifest_path": self.manifest_path,
            "metadata": dict(self.metadata),
        }
