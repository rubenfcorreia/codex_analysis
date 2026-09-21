from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from analysis.shared.branch_tree import branch_leaf_root


def comparison_figure_root(config: Mapping[str, Any], *, repo_root: Path | None = None) -> Path | None:
    raw = config.get("branch_first_output_root") or config.get("comparison_output_root")
    if not raw:
        return None
    root = Path(str(raw))
    if repo_root is not None and not root.is_absolute():
        root = repo_root / root
    return root.resolve()


def comparison_leaf_figure_root(config: Mapping[str, Any], *, branch: str, basis: str, repo_root: Path | None = None) -> Path | None:
    root = comparison_figure_root(config, repo_root=repo_root)
    return branch_leaf_root(root, branch, basis) if root is not None else None


def shared_figure_root(pipeline_root: Path | str) -> Path:
    return Path(pipeline_root) / "general" / "figures"


def standalone_figure_root(run_root: Path | str) -> Path:
    return Path(run_root) / "figures"


__all__ = ["comparison_figure_root", "comparison_leaf_figure_root", "shared_figure_root", "standalone_figure_root"]
