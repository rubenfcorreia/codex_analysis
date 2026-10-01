from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AnalysisScope:
    pipeline: str
    preset: str = ""
    branch: str = ""
    basis: str = ""
    selected_states: tuple[str, ...] = ()
    split_definition: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "preset": self.preset,
            "analysis_branch_name": self.branch,
            "analysis_basis_name": self.basis,
            "selected_states": list(self.selected_states),
            "split_definition": self.split_definition,
        }


@dataclass(frozen=True)
class FigureContext:
    output_root: Any
    scope: AnalysisScope
    family: str
    stem_prefix: str = ""


@dataclass
class AnalysisFamilyResult:
    family: str
    scope: AnalysisScope
    rows: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "scope": self.scope.as_dict(),
            "rows": list(self.rows),
            "tables": dict(self.tables),
            "metadata": dict(self.metadata),
            "alerts": list(self.alerts),
        }


def scope_metadata(scope: AnalysisScope | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(scope, AnalysisScope):
        return scope.as_dict()
    return {str(key): value for key, value in scope.items()}


def attach_scope(rows: Sequence[Mapping[str, Any]], scope: AnalysisScope | Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = scope_metadata(scope)
    return [{**dict(row), **{key: value for key, value in metadata.items() if value not in ("", [], ())}} for row in rows]


def validate_family_payload(payload: Mapping[str, Any]) -> None:
    family = str(payload.get("family") or "").strip()
    scope = payload.get("scope")
    if not family or not isinstance(scope, Mapping):
        raise ValueError("analysis-family payload requires family and scope metadata")
    if not str(scope.get("pipeline") or "").strip():
        raise ValueError("analysis-family scope requires pipeline")


__all__ = ["AnalysisFamilyResult", "AnalysisScope", "FigureContext", "attach_scope", "scope_metadata", "validate_family_payload"]
