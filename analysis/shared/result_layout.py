"""Canonical lifecycle-based result paths for analysis pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any


@dataclass(frozen=True)
class ResultLayout:
    """Stable analysis outputs and disposable visual outputs for one run."""

    run_root: Path
    analysis_root: Path
    figure_root: Path

    @property
    def cache_root(self) -> Path:
        return self.analysis_root / "cache"

    @property
    def statistics_root(self) -> Path:
        return self.analysis_root / "statistics"

    @property
    def reports_root(self) -> Path:
        return self.analysis_root / "reports"

    @property
    def manifests_root(self) -> Path:
        return self.analysis_root / "manifests"

    def ensure_stable(self) -> None:
        for path in (self.analysis_root, self.cache_root, self.statistics_root, self.reports_root, self.manifests_root):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_figures(self) -> None:
        self.figure_root.mkdir(parents=True, exist_ok=True)


def resolve_result_layout(
    config: Mapping[str, Any],
    *,
    root_key: str = "output_dir",
    legacy_root_key: str = "result_root",
    repo_root: Path | None = None,
) -> ResultLayout:
    """Resolve new sibling roots while honoring explicit path overrides."""
    raw_root = config.get(root_key) or config.get(legacy_root_key) or "results"
    run_root = Path(raw_root)
    if repo_root is not None and not run_root.is_absolute():
        run_root = repo_root / run_root
    raw_analysis = config.get("analysis_output_dir")
    raw_figures = config.get("figure_output_dir")
    analysis_root = Path(raw_analysis) if raw_analysis else run_root / "analysis"
    figure_root = Path(raw_figures) if raw_figures else run_root / "figures"
    if repo_root is not None:
        if not analysis_root.is_absolute():
            analysis_root = repo_root / analysis_root
        if not figure_root.is_absolute():
            figure_root = repo_root / figure_root
    return ResultLayout(run_root.resolve(), analysis_root.resolve(), figure_root.resolve())


__all__ = ["ResultLayout", "resolve_result_layout"]
