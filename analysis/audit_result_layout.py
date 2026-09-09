"""Read-only checks for comparison result-folder layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from analysis.shared.branch_tree import ANALYSIS_BRANCHES


def find_illegal_figure_dirs(results_root: Path | str) -> List[Path]:
    """Return comparison roots containing a non-leaf ``figures`` directory."""
    root = Path(results_root)
    if not root.exists():
        return []
    branches = set(ANALYSIS_BRANCHES)
    return [
        figure_dir
        for figure_dir in sorted(path for path in root.rglob("figures") if path.is_dir())
        if any((figure_dir.parent / branch).is_dir() for branch in branches)
    ]


def audit_roots(results_roots: Iterable[Path | str]) -> List[Path]:
    findings: List[Path] = []
    for root in results_roots:
        findings.extend(find_illegal_figure_dirs(root))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", nargs="*", type=Path, default=[Path("results")])
    args = parser.parse_args()
    findings = audit_roots(args.results_root)
    if findings:
        print("Illegal comparison figure directories:")
        for path in findings:
            print(path)
        return 1
    print("No illegal comparison figure directories found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
