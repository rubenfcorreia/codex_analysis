"""Audit generated figure layout, manifests, and split-branch completeness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.branch_tree import ANALYSIS_BRANCHES

FIGURE_SUFFIXES = {".svg", ".png", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in FIGURE_SUFFIXES]


def find_illegal_figure_dirs(results_root: Path | str) -> List[Path]:
    root = Path(results_root)
    if not root.exists():
        return []
    branches = set(ANALYSIS_BRANCHES)
    findings: List[Path] = []
    comparison_roots = {
        path
        for branch in branches
        for path in root.rglob(branch)
        if path.is_dir()
    }
    comparison_roots = {path.parent for path in comparison_roots}
    for path in sorted(root.rglob("figures")):
        if not path.is_dir():
            continue
        parts = path.relative_to(root).parts
        if parts == ("general", "figures"):
            continue
        # A top-level figures directory is valid for a standalone run. It is
        # legacy only when the same result root also contains comparison
        # branches, where figures must live in the branch/basis leaf.
        if parts == ("figures",):
            if comparison_roots:
                findings.append(path)
            continue
        if path.parent in comparison_roots or set(parts).intersection(branches):
            findings.append(path)
    return findings


def find_manifest_issues(results_root: Path | str) -> List[str]:
    root = Path(results_root)
    issues: List[str] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text())
        except Exception as exc:
            issues.append(f"{manifest_path}: invalid JSON ({exc})")
            continue
        output_root = Path(payload.get("output_root") or manifest_path.parent)
        if not output_root.is_absolute():
            output_root = (manifest_path.parent / output_root).resolve()
        artifacts = payload.get("output_artifacts", [])
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            candidate = Path(str(artifact))
            if not candidate.is_absolute():
                candidate = output_root / candidate
            try:
                candidate.resolve().relative_to(output_root.resolve())
            except ValueError:
                issues.append(f"{manifest_path}: artifact outside output root: {artifact}")
                continue
            if not candidate.exists():
                issues.append(f"{manifest_path}: missing artifact: {artifact}")
    return issues


def summarize_results(results_root: Path | str) -> Dict[str, object]:
    root = Path(results_root)
    files = _files(root)
    by_suffix: Dict[str, int] = {}
    by_scope: Dict[str, Dict[str, int]] = {}
    for path in files:
        by_suffix[path.suffix.lower()] = by_suffix.get(path.suffix.lower(), 0) + 1
        parts = path.relative_to(root).parts
        if len(parts) >= 3 and parts[1] in ANALYSIS_BRANCHES:
            scope = "/".join(parts[:3])
            family = parts[3] if len(parts) > 3 else "root"
        elif parts[:2] == ("general", "figures"):
            scope, family = "general", parts[2] if len(parts) > 2 else "root"
        else:
            scope, family = "standalone", parts[0] if parts else "root"
        entry = by_scope.setdefault(f"{scope}/{family}", {"figure_files": 0, "figure_bytes": 0})
        entry["figure_files"] += 1
        entry["figure_bytes"] += path.stat().st_size
    return {"figure_files": len(files), "figure_bytes": sum(p.stat().st_size for p in files), "by_suffix": by_suffix, "by_scope": by_scope}


def audit_roots(results_roots: Iterable[Path | str]) -> List[str]:
    findings: List[str] = []
    for root_value in results_roots:
        root = Path(root_value)
        findings.extend(str(path) for path in find_illegal_figure_dirs(root))
        findings.extend(find_manifest_issues(root))
        findings.extend(find_duplicate_logical_figures(root))
        findings.extend(find_split_group_issues(root))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", nargs="+", type=Path)
    args = parser.parse_args()
    findings = audit_roots(args.results_root)
    for root in args.results_root:
        print(f"{root}: {summarize_results(root)}")
    if findings:
        print("Audit findings:")
        print("\n".join(findings))
        return 1
    print("No figure-layout or manifest issues found.")
    return 0



def find_duplicate_logical_figures(results_root: Path | str) -> List[str]:
    root = Path(results_root)
    figures = _files(root)
    findings: List[str] = []
    for path in figures:
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if "figures" not in parts:
            continue
        parts.remove("figures")
        canonical = root.joinpath(*parts)
        if canonical.exists() and canonical != path:
            findings.append(f"{path}: duplicate logical figure also exists at {canonical}")
    return findings


def find_split_group_issues(results_root: Path | str) -> List[str]:
    root = Path(results_root)
    findings: List[str] = []
    split_branches = {"activity_split", "frequency_split", "activity_frequency_split"}
    group_tokens = ("high", "low", "more active", "less active", "higher frequency", "lower frequency")
    for branch in split_branches:
        for leaf in sorted(path for path in root.rglob(branch) if path.is_dir()):
            for basis_dir in sorted(leaf.iterdir()):
                if not basis_dir.is_dir():
                    continue
                figure_files = _files(basis_dir)
                if not figure_files:
                    continue
                roi_split_files = [path for path in figure_files if path.name.lower().startswith("roi_split")]
                if not roi_split_files:
                    findings.append(f"{basis_dir}: split leaf has no ROI-split artifact")
                    continue
                text = "\n".join(path.read_text(errors="ignore").lower() for path in roi_split_files)
                present = {token for token in group_tokens if token in text}
                cached_groups = set()
                for cache_path in root.rglob("*analysis_results_cache.npz"):
                    try:
                        payload = np.load(cache_path, allow_pickle=True)["cache"].item()
                        roi_split = payload.get("analysis_results", {}).get("roi_split", {})
                        branch_payload = roi_split.get("branches", {}).get(branch, {}).get(basis_dir.name, {})
                        for row in branch_payload.get("subject_state_rows", []):
                            group = row.get("split_group") or row.get("group")
                            if group:
                                cached_groups.add(str(group).lower())
                    except (KeyError, OSError, ValueError, TypeError, AttributeError):
                        continue
                if len(cached_groups) >= 2:
                    continue
                if len(present) < 2:
                    findings.append(f"{basis_dir}: split artifacts do not expose two group labels: {sorted(present)}")
    return findings

if __name__ == "__main__":
    raise SystemExit(main())
