from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from analysis.shared.state_utils import ensure_dir, safe_filename_component
@dataclass(frozen=True)
class AnalysisJobSpec:
    pipeline: str
    split_type: str
    state_basis: str
    analysis_type: str
    cohort: str = 'all'
    def normalized(self) -> 'AnalysisJobSpec':
        return AnalysisJobSpec(
            pipeline=safe_filename_component(self.pipeline),
            split_type=safe_filename_component(self.split_type),
            state_basis=safe_filename_component(self.state_basis),
            analysis_type=safe_filename_component(self.analysis_type),
            cohort=safe_filename_component(self.cohort),
        )
    def as_dict(self) -> Dict[str, str]:
        spec = self.normalized()
        return {
            'pipeline': spec.pipeline,
            'split_type': spec.split_type,
            'state_basis': spec.state_basis,
            'analysis_type': spec.analysis_type,
            'cohort': spec.cohort,
        }
    def key(self) -> str:
        spec = self.normalized()
        return '/'.join([spec.pipeline, spec.split_type, spec.state_basis, spec.analysis_type, spec.cohort])
def job_root(base_root: Path | str, spec: AnalysisJobSpec) -> Path:
    normalized = spec.normalized()
    return (
        Path(base_root)
        / normalized.pipeline
        / normalized.split_type
        / normalized.state_basis
        / normalized.analysis_type
        / normalized.cohort
    )
def manifest_path(output_root: Path | str) -> Path:
    return Path(output_root) / 'summary' / 'manifest.json'
def validate_manifest_payload(output_root: Path | str, manifest: Mapping[str, Any]) -> None:
    storage_root = Path(output_root).resolve()
    root = storage_root
    declared_root = manifest.get("output_root")
    if declared_root:
        resolved_declared = Path(str(declared_root)).resolve()
        # Standalone runs may store analysis/summary/manifest.json while
        # figures live in a sibling figures directory. Accept that one
        # declared run root, but never accept an unrelated path.
        try:
            storage_root.relative_to(resolved_declared)
            root = resolved_declared
        except ValueError:
            try:
                resolved_declared.relative_to(storage_root)
            except ValueError as exc:
                raise ValueError(f"manifest output_root {resolved_declared} is outside {storage_root}") from exc
    artifacts = manifest.get("output_artifacts", [])
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ValueError("manifest output_artifacts must be a sequence")
    for artifact in artifacts:
        candidate = Path(str(artifact))
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest artifact {candidate} is outside {root}") from exc
        if not candidate.exists():
            raise FileNotFoundError(f"manifest artifact does not exist: {candidate}")


def write_manifest(output_root: Path | str, manifest: Mapping[str, Any]) -> Path:
    path = manifest_path(output_root)
    ensure_dir(path.parent)
    payload = dict(manifest)
    if "output_artifacts" in payload:
        declared_root = Path(str(payload.get("output_root") or output_root)).resolve()
        normalized_artifacts = []
        for artifact in payload.get("output_artifacts", []):
            text = str(artifact)
            candidate = Path(text)
            if not candidate.is_absolute():
                rooted_candidate = declared_root / candidate
                if not rooted_candidate.exists() and candidate.exists():
                    rooted_candidate = candidate.resolve()
                candidate = rooted_candidate
            try:
                normalized_artifacts.append(str(candidate.resolve().relative_to(declared_root)))
            except ValueError:
                # Leave invalid paths untouched so validation raises the
                # actionable outside-root error below.
                normalized_artifacts.append(text)
        payload["output_artifacts"] = normalized_artifacts
        validate_manifest_payload(output_root, payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path
def load_manifest(output_root: Path | str) -> Optional[Dict[str, Any]]:
    root = Path(output_root)
    candidate_paths = [
        manifest_path(root),
        root / 'manifest.json',
        root / 'analysis' / 'manifest.json',
        root / 'analysis' / 'manifests' / 'manifest.json',
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    return None
def _normalize_output_artifact_path(root: Path, artifact: Any) -> str:
    text = str(artifact or '').strip()
    if not text:
        return ''
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError:
            return text.replace('\\', '/').strip()
    return str(candidate).replace('\\', '/').strip()


def _scan_output_artifacts(output_root: Path | str) -> List[str]:
    root = Path(output_root)
    if not root.exists():
        return []
    artifacts: List[str] = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root)).replace('\\', '/').strip()
        if relative in {'manifest.json', 'summary/manifest.json'}:
            continue
        artifacts.append(relative)
    return list(dict.fromkeys(artifacts))


def collect_output_artifacts(
    output_root: Path | str,
    tracked_artifacts: Sequence[str] | None = None,
    *,
    validate: bool = False,
) -> List[str]:
    root = Path(output_root)
    if tracked_artifacts is None:
        return _scan_output_artifacts(root)
    artifacts = [
        artifact
        for artifact in (_normalize_output_artifact_path(root, item) for item in tracked_artifacts)
        if artifact and artifact not in {'manifest.json', 'summary/manifest.json'}
    ]
    artifacts = list(dict.fromkeys(artifacts))
    if not validate:
        return artifacts
    scanned = _scan_output_artifacts(root)
    return list(dict.fromkeys(artifacts + [artifact for artifact in scanned if artifact not in artifacts]))


def _artifact_matches(relative_path: str, candidates: Sequence[str]) -> bool:
    relative = relative_path.replace('\\', '/').strip()
    if not relative:
        return False
    for candidate in candidates:
        candidate_text = str(candidate).replace('\\', '/').strip()
        if not candidate_text:
            continue
        if relative == candidate_text or relative.endswith('/' + candidate_text) or relative.endswith(candidate_text):
            return True
    return False
def manifest_artifact_path(
    manifest: Mapping[str, Any] | None,
    output_root: Path | str,
    *relative_candidates: str,
) -> Optional[Path]:
    if not isinstance(manifest, Mapping):
        manifest = {}
    root = Path(output_root)
    artifacts = manifest.get('output_artifacts', [])
    if isinstance(artifacts, Sequence):
        for artifact in artifacts:
            artifact_text = str(artifact or '').strip()
            if not artifact_text:
                continue
            if _artifact_matches(artifact_text, relative_candidates):
                candidate = Path(artifact_text)
                return candidate if candidate.is_absolute() else root / candidate
    for candidate_text in relative_candidates:
        candidate = root / candidate_text
        if candidate.exists():
            return candidate
    return None
def manifest_output_root(manifest: Mapping[str, Any] | None, fallback_root: Path | str | None = None) -> Path:
    if isinstance(manifest, Mapping):
        output_root = manifest.get('output_root')
        if output_root:
            return Path(output_root)
    if fallback_root is not None:
        return Path(fallback_root)
    return Path('.')
