"""Filesystem and log checks for isolated analysis-pipeline audits."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_EMPTY_PATTERNS = (
    "no qualifying records",
    "no usable selected state",
    "no valid within-experiment state transitions",
    "no rows were available",
)
ERROR_PATTERNS = ("FAIL", "Traceback", "ERROR", "outside", "invalid", "missing", "refusing")


def classify_audit_lines(lines: Iterable[str]) -> dict[str, list[str]]:
    """Classify audit log lines without treating expected sparsity as a defect."""
    expected: list[str] = []
    failures: list[str] = []
    for raw in lines:
        line = str(raw).rstrip()
        lowered = line.lower()
        if any(pattern in lowered for pattern in EXPECTED_EMPTY_PATTERNS):
            expected.append(line)
        elif any(pattern.lower() in lowered for pattern in ERROR_PATTERNS):
            failures.append(line)
    return {"expected_empty": expected, "failures": failures}


def validate_audit_filesystem(root: Path | str) -> dict[str, Any]:
    """Validate isolated audit artifacts and canonical analysis/figure placement."""
    root = Path(root).resolve()
    analysis_root = root / "analysis"
    figures_root = root / "figures"
    image_suffixes = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"}
    analysis_images = [p for p in analysis_root.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes]
    figure_images = [p for p in figures_root.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes]
    manifests = list(root.rglob("manifest.json"))
    missing: list[str] = []
    outside: list[str] = []
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text())
        except Exception:
            missing.append(str(manifest_path))
            continue
        declared_root = Path(str(payload.get("output_root") or manifest_path.parent)).resolve()
        for artifact in payload.get("output_artifacts", []) or []:
            candidate = Path(str(artifact))
            if not candidate.is_absolute():
                candidate = (declared_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if not candidate.exists():
                missing.append(str(candidate))
            try:
                candidate.relative_to(declared_root)
            except ValueError:
                outside.append(str(candidate))
    return {
        "analysis_image_count": len(analysis_images),
        "figure_image_count": len(figure_images),
        "manifest_count": len(manifests),
        "missing_manifest_artifacts": sorted(set(missing)),
        "manifest_artifacts_outside_root": sorted(set(outside)),
        "analysis_images": [str(path) for path in analysis_images],
    }
