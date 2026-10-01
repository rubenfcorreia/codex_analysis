from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from analysis.figure_viewer.models import FigureFilterState, FigureRecord
from analysis.shared.result_manifest import load_manifest
from analysis.shared.state_utils import canonical_state_label


DEFAULT_RESULTS_DEPTH = 8
CATALOG_CACHE_VERSION = 5
CATALOG_CACHE_NAME = "catalog.json"
IMAGE_SUFFIXES = {".png", ".svg"}
KNOWN_COHORTS = {
    "all",
    "responsive",
    "nonresponsive",
    "nrem",
    "rem",
    "quiet_awake",
    "quiet_awake_blank",
    "quiet_awake_movies",
    "active_awake",
    "mixed",
}
KNOWN_COMPARTMENTS = {
    "soma",
    "bouton",
    "dendrite",
    "spine",
    "axon",
}

KNOWN_COMPARTMENT_ALIASES = {
    "dendrites": "dendrite",
    "spines": "spine",
    "somas": "soma",
    "boutons": "bouton",
    "axons": "axon",
}

KNOWN_DENDRITE_REGIONS = {
    "basal",
    "apical",
    "basal_vs_apical",
}

KNOWN_REGION_TYPES = {
    "dendritic",
    "pairwise",
}


def catalog_cache_path(repo_root: Path | str) -> Path:
    return Path(repo_root).resolve() / ".figure_viewer" / CATALOG_CACHE_NAME


def _record_from_cache(payload: Mapping[str, Any]) -> FigureRecord | None:
    try:
        preview_path = Path(str(payload["preview_path"]))
        source_paths = tuple(Path(str(path)) for path in payload.get("source_paths", []))
        return FigureRecord(
            figure_key=str(payload["figure_key"]),
            display_label=str(payload.get("display_label", "")),
            title=str(payload.get("title", "")),
            preview_path=preview_path,
            comparison_key=str(payload.get("comparison_key", "")),
            comparison_label=str(payload.get("comparison_label", "")),
            source_paths=source_paths,
            source_kinds=tuple(str(kind) for kind in payload.get("source_kinds", [])),
            pipeline=str(payload.get("pipeline", "")),
            preset=str(payload.get("preset", "")),
            split=str(payload.get("split", "")),
            basis=str(payload.get("basis", "")),
            family=str(payload.get("family", "")),
            cohort=str(payload.get("cohort", "")),
            scope=str(payload.get("scope", "")),
            compartment=str(payload.get("compartment", "")),
            dendrite_region=str(payload.get("dendrite_region", "")),
            region_type=str(payload.get("region_type", "")),
            metric=str(payload.get("metric", "")),
            variant=str(payload.get("variant", "")),
            source_root=str(payload.get("source_root", "")),
            manifest_path=str(payload.get("manifest_path", "")),
            metadata=dict(payload.get("metadata", {})),
            search_text=str(payload.get("search_text", "")),
            sort_key=tuple(str(item) for item in payload.get("sort_key", [])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_catalog_cache(repo_root: Path | str) -> List[FigureRecord]:
    path = catalog_cache_path(repo_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != CATALOG_CACHE_VERSION:
            return []
        records = [_record_from_cache(item) for item in payload.get("records", [])]
        if not records or any(record is None for record in records):
            return []
        return [record for record in records if record is not None and record.preview_path.exists()]
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return []


def save_catalog_cache(repo_root: Path | str, records: Sequence[FigureRecord]) -> None:
    path = catalog_cache_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CATALOG_CACHE_VERSION,
        "records": [record.as_context() | {
            "search_text": record.search_text,
            "sort_key": list(record.sort_key),
        } for record in records],
    }
    fd, temporary_name = tempfile.mkstemp(prefix="catalog-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class CatalogScanProgress:
    phase: str
    current: int
    total: int
    message: str = ""


def _emit_progress(
    progress_callback: Callable[[CatalogScanProgress], None] | None,
    phase: str,
    current: int,
    total: int,
    message: str = "",
) -> None:
    if progress_callback is not None:
        progress_callback(CatalogScanProgress(phase=phase, current=current, total=total, message=message))


def _humanize(text: Any) -> str:
    cleaned = str(text or "").replace("_", " ").replace("-", " ").strip()
    return " ".join(part for part in cleaned.split() if part)


def _normalize_path_text(path_text: Any) -> str:
    return str(path_text or "").replace("\\", "/").strip("/")


def _result_parts(path: Path, repo_root: Path) -> List[str]:
    try:
        parts = list(path.resolve().relative_to(repo_root.resolve()).parts)
    except Exception:
        parts = list(path.parts)
    if parts and parts[0] == "results":
        return parts[1:]
    return parts


def _metric_from_filename(filename: str, context: Mapping[str, str]) -> str:
    stem = canonical_state_label(Path(filename).stem)
    for prefix in ("state_summary_boxplots", "state_summary", "summary_boxplots"):
        if stem.startswith(prefix + "_"):
            stem = stem[len(prefix) + 1:]
            break
    # Remove compound region labels before tokenizing so `basal_vs_apical`
    # cannot leave a stray `vs` in the metric name.
    for region in sorted(KNOWN_DENDRITE_REGIONS, key=len, reverse=True):
        stem = stem.replace(region, "_")
    tokens = [token for token in stem.split("_") if token]
    excluded = (
        {canonical_state_label(value) for value in KNOWN_COHORTS}
        | {canonical_state_label(value) for value in KNOWN_COMPARTMENTS}
        | {canonical_state_label(value) for value in KNOWN_COMPARTMENT_ALIASES}
        | {canonical_state_label(value) for value in KNOWN_DENDRITE_REGIONS}
        | {canonical_state_label(context.get("cohort"))}
        | {canonical_state_label(context.get("compartment"))}
        | {canonical_state_label(context.get("dendrite_region"))}
        | {"state_summary", "boxplots"}
    )
    tokens = [token for token in tokens if token not in excluded]
    return "_".join(tokens)


def _path_context(path: Path, repo_root: Path) -> Dict[str, str]:
    parts = _result_parts(path, repo_root)
    context = {
        "pipeline": "",
        "preset": "",
        "split": "",
        "basis": "",
        "family": "",
        "cohort": "",
        "scope": "",
        "compartment": "",
        "dendrite_region": "",
        "region_type": "",
        "metric": "",
        "variant": "",
    }
    if not parts:
        return context

    context["pipeline"] = str(parts[0])
    if len(parts) >= 2:
        context["preset"] = str(parts[1])

    filename = Path(parts[-1]).stem
    if parts[0] == "review_figures":
        context["family"] = str(parts[1]) if len(parts) >= 2 else ""
        semantic_parts = list(parts[2:-1])
    elif "figures" in parts:
        figures_index = parts.index("figures")
        context["split"] = str(parts[2]) if len(parts) >= 3 else ""
        context["basis"] = str(parts[3]) if len(parts) >= 4 else ""
        context["family"] = str(parts[figures_index + 1]) if figures_index + 1 < len(parts) - 1 else ""
        semantic_parts = list(parts[figures_index + 2 : -1])
    elif "checkpoint_examples" in parts:
        checkpoint_index = parts.index("checkpoint_examples")
        context["split"] = str(parts[2]) if len(parts) >= 3 else ""
        context["basis"] = str(parts[3]) if len(parts) >= 4 else ""
        context["family"] = str(parts[checkpoint_index + 1]) if checkpoint_index + 1 < len(parts) - 1 else ""
        semantic_parts = list(parts[checkpoint_index + 2 : -1])
    elif parts[0] == "poster_ready":
        context["family"] = str(parts[2]) if len(parts) >= 3 else ""
        semantic_parts = list(parts[3:-1])
    else:
        context["split"] = str(parts[2]) if len(parts) >= 3 else ""
        context["basis"] = str(parts[3]) if len(parts) >= 4 else ""
        semantic_parts = list(parts[4:-1])
        if semantic_parts:
            first = canonical_state_label(semantic_parts[0])
            if first not in KNOWN_COHORTS and first not in KNOWN_COMPARTMENTS:
                context["family"] = str(semantic_parts.pop(0))

    remaining: List[str] = []
    for item in semantic_parts:
        label = canonical_state_label(item)
        label = KNOWN_COMPARTMENT_ALIASES.get(label, label)
        if not context["cohort"] and label in KNOWN_COHORTS:
            context["cohort"] = label
        elif not context["compartment"] and label in KNOWN_COMPARTMENTS:
            context["compartment"] = label
        elif not context["dendrite_region"] and label in KNOWN_DENDRITE_REGIONS:
            context["dendrite_region"] = label
        elif not context["region_type"] and label in KNOWN_REGION_TYPES:
            context["region_type"] = label
        else:
            remaining.append(str(item))
    context["scope"] = "/".join(remaining)

    normalized_filename = canonical_state_label(filename)
    if not context["compartment"]:
        for label in sorted(KNOWN_COMPARTMENTS, key=len, reverse=True):
            if canonical_state_label(label) in normalized_filename:
                context["compartment"] = label
                break
    if not context["dendrite_region"]:
        for label in sorted(KNOWN_DENDRITE_REGIONS, key=len, reverse=True):
            if canonical_state_label(label) in normalized_filename:
                context["dendrite_region"] = label
                break
    if not context["region_type"]:
        for label in sorted(KNOWN_REGION_TYPES, key=len, reverse=True):
            if canonical_state_label(label) in normalized_filename:
                context["region_type"] = label
                break
    context["metric"] = _metric_from_filename(filename, context)
    context["variant"] = canonical_state_label(semantic_parts[-1] if semantic_parts else filename)
    return context


def _context_from_output_root(output_root: Path, repo_root: Path) -> Dict[str, str]:
    parts = _result_parts(output_root, repo_root)
    context = {
        "pipeline": "",
        "preset": "",
        "split": "",
        "basis": "",
        "family": "",
        "cohort": "",
        "scope": "",
        "compartment": "",
        "dendrite_region": "",
        "region_type": "",
        "metric": "",
        "variant": "",
    }
    if len(parts) >= 1:
        context["pipeline"] = str(parts[0])
    if len(parts) >= 2:
        context["preset"] = str(parts[1])
    if len(parts) >= 3:
        context["split"] = str(parts[2])
    if len(parts) >= 4:
        context["basis"] = str(parts[3])
    if len(parts) >= 5:
        context["cohort"] = str(parts[4])
    if len(parts) >= 6:
        context["scope"] = "/".join(str(part) for part in parts[5:])
    return context


def _summary_context(manifest: Mapping[str, Any], output_root: Path, repo_root: Path) -> Dict[str, str]:
    context = _context_from_output_root(output_root, repo_root)
    job_spec = manifest.get("job_spec") if isinstance(manifest.get("job_spec"), Mapping) else {}
    analysis_scope = manifest.get("analysis_scope") if isinstance(manifest.get("analysis_scope"), Mapping) else {}

    if isinstance(job_spec, Mapping):
        if job_spec.get("pipeline"):
            context["pipeline"] = str(job_spec.get("pipeline"))
        if job_spec.get("analysis_type"):
            context["preset"] = str(job_spec.get("analysis_type"))
        if job_spec.get("cohort"):
            context["cohort"] = str(job_spec.get("cohort"))

    for field_name in ("pipeline", "preset", "split", "basis", "family", "cohort", "scope", "compartment", "dendrite_region", "region_type", "metric"):
        if manifest.get(field_name):
            context[field_name] = str(manifest[field_name])

    branch_name = manifest.get("analysis_branch_name") or analysis_scope.get("branch_name")
    basis_name = manifest.get("analysis_basis_name") or analysis_scope.get("basis_name")
    if branch_name and not context["split"]:
        context["split"] = str(branch_name)
    if basis_name and not context["basis"]:
        context["basis"] = str(basis_name)
    return context


def _figure_title_from_path(path: Path) -> str:
    return _humanize(path.stem)


def _display_label(title: str, context: Mapping[str, str]) -> str:
    scope = context.get("scope") or context.get("compartment") or context.get("cohort") or context.get("variant") or ""
    bits = [
        context.get("preset", ""),
        context.get("split", ""),
        context.get("basis", ""),
        context.get("family", ""),
        scope,
    ]
    bits = [bit for bit in bits if bit]
    suffix = " / ".join(bits)
    return f"{title} — {suffix}" if suffix else title


def _comparison_key(context: Mapping[str, str], title: str) -> str:
    family = canonical_state_label(context.get("family"))
    title_key = canonical_state_label(title)
    if family and title_key:
        return f"{family}::{title_key}"
    return family or title_key or canonical_state_label(context.get("variant")) or canonical_state_label(context.get("scope"))


def _comparison_label(context: Mapping[str, str], title: str) -> str:
    family = _humanize(context.get("family"))
    bits = [family, title]
    bits = [bit for bit in bits if bit]
    return " / ".join(bits) if bits else title


def _search_text(title: str, context: Mapping[str, str], path: Path, source_kinds: Sequence[str]) -> str:
    parts = [title, _comparison_label(context, title), _comparison_key(context, title), str(path), *context.values(), *source_kinds]
    return " ".join(str(piece).lower() for piece in parts if piece is not None and str(piece).strip())


def _merge_builder(
    builder: MutableMapping[str, Any],
    *,
    path: Path,
    preview_path: Path,
    source_kind: str,
    context: Mapping[str, str],
    title: str,
    manifest_path: Path | None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    builder.setdefault("figure_key", path.with_suffix("").resolve().as_posix())
    current_preview = builder.get("preview_path")
    if current_preview is None:
        builder["preview_path"] = preview_path
    else:
        current_preview = Path(current_preview)
        if current_preview.suffix.lower() != ".svg" and preview_path.suffix.lower() == ".svg":
            builder["preview_path"] = preview_path
    builder.setdefault("title", title)
    builder.setdefault("source_paths", set()).add(path)
    builder.setdefault("source_kinds", set()).add(source_kind)
    builder.setdefault("manifest_paths", set())
    if manifest_path is not None:
        builder["manifest_paths"].add(manifest_path)
    builder.setdefault("metadata", {})
    if metadata:
        for key, value in metadata.items():
            if value in (None, "", [], {}):
                continue
            if key not in builder["metadata"]:
                builder["metadata"][key] = value
    for key, value in context.items():
        if value and not builder.get(key):
            builder[key] = value
    comparison_key = _comparison_key(builder, builder.get("title", title))
    builder["comparison_key"] = comparison_key
    builder["comparison_label"] = _comparison_label(builder, builder.get("title", title))
    builder["display_label"] = _display_label(builder.get("title", title), builder)
    builder["search_text"] = _search_text(builder.get("title", title), builder, path, tuple(sorted(builder["source_kinds"])))
    builder["sort_key"] = (
        builder.get("pipeline", ""),
        builder.get("preset", ""),
        builder.get("split", ""),
        builder.get("basis", ""),
        builder.get("family", ""),
        builder.get("cohort", ""),
        builder.get("scope", ""),
        builder.get("title", ""),
    )


def _candidate_path(root: Path, relative_text: str) -> Path:
    relative = Path(_normalize_path_text(relative_text))
    return relative if relative.is_absolute() else (root / relative)


def _depth_within_figures(relative_text: str) -> int:
    parts = _normalize_path_text(relative_text).split("/")
    if "figures" not in parts:
        return 999
    return len(parts) - parts.index("figures") - 1


def _summary_candidates(manifest: Mapping[str, Any], output_root: Path, depth_limit: int) -> Iterable[Tuple[Path, Path, str, Dict[str, Any]]]:
    artifacts = manifest.get("output_artifacts", [])
    if not isinstance(artifacts, Sequence):
        return []
    grouped: Dict[str, Dict[str, Path]] = {}
    artifact_metadata: Dict[str, Dict[str, Any]] = {}
    for artifact in artifacts:
        artifact_payload = artifact if isinstance(artifact, Mapping) else {}
        rel_text = _normalize_path_text(artifact_payload.get("file") if artifact_payload else artifact)
        if not rel_text.lower().endswith(tuple(IMAGE_SUFFIXES)):
            continue
        if "figures" not in rel_text.split("/"):
            continue
        if _depth_within_figures(rel_text) > depth_limit:
            continue
        path = _candidate_path(output_root, rel_text)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.with_suffix("").resolve().as_posix()
        grouped.setdefault(key, {})[path.suffix.lower()] = path
        if artifact_payload:
            artifact_metadata.setdefault(key, {}).update(artifact_payload)
    for key, renditions in grouped.items():
        preview_path = renditions.get(".svg") or renditions.get(".png") or next(iter(renditions.values()))
        for path in renditions.values():
            yield path, preview_path, "summary_manifest", artifact_metadata.get(key, {})


def _checkpoint_candidates(checkpoint_root: Path, manifest: Mapping[str, Any], repo_root: Path) -> Iterable[Tuple[Path, Path, str, Dict[str, str], Dict[str, Any]]]:
    entries = manifest.get("entries", [])
    if not isinstance(entries, Sequence):
        return []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        rel_text = _normalize_path_text(entry.get("file"))
        if not rel_text.lower().endswith(tuple(IMAGE_SUFFIXES)):
            continue
        path = _candidate_path(checkpoint_root.parent, rel_text)
        if not path.exists():
            continue
        context = _path_context(path, repo_root)
        checkpoint_name = canonical_state_label(entry.get("checkpoint") or context.get("family") or "checkpoint")
        context["family"] = checkpoint_name
        context["scope"] = str(entry.get("scope") or context.get("scope") or "")
        variant = str(entry.get("variant") or "").strip()
        if variant:
            context["variant"] = canonical_state_label(variant)
        compartment = str(entry.get("compartment") or "").strip()
        if compartment:
            context["compartment"] = canonical_state_label(compartment)
        dendrite_region = str(entry.get("dendrite_region") or entry.get("dendrite_type") or "").strip()
        if dendrite_region:
            context["dendrite_region"] = canonical_state_label(dendrite_region)
        region_type = str(entry.get("region_type") or entry.get("orientation") or "").strip()
        if region_type:
            context["region_type"] = canonical_state_label(region_type)
        metric = str(entry.get("metric") or entry.get("analysis_type") or "").strip()
        if metric:
            context["metric"] = canonical_state_label(metric)
        cohort = str(entry.get("cohort") or entry.get("variant") or "").strip()
        if cohort and not context.get("cohort"):
            context["cohort"] = canonical_state_label(cohort)
        title = str(entry.get("title") or _figure_title_from_path(path)).strip() or _figure_title_from_path(path)
        metadata = {
            "checkpoint": entry.get("checkpoint"),
            "scope": entry.get("scope"),
            "variant": entry.get("variant"),
            "animal_id": entry.get("animal_id"),
            "exp_id": entry.get("exp_id"),
            "global_dendrite_id": entry.get("global_dendrite_id"),
            "global_spine_id": entry.get("global_spine_id"),
        }
        yield path, path, "checkpoint_manifest", context, {k: v for k, v in metadata.items() if v not in (None, "", [], {})}


def _review_candidates(review_root: Path) -> Iterable[Tuple[Path, Path, str, Dict[str, str], Dict[str, Any]]]:
    if not review_root.exists():
        return []
    for path in sorted(review_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            parts = list(path.resolve().relative_to(review_root.resolve()).parts)
        except Exception:
            parts = list(path.parts)
        context = _path_context(Path(review_root.name, *parts), review_root.parent)
        title = _figure_title_from_path(path)
        metadata = {"review_root": str(review_root)}
        yield path, path, "review_figures", context, metadata




def discover_figure_records(
    *,
    repo_root: Path | str | None = None,
    include_review_figures: bool = True,
    summary_depth_limit: int = DEFAULT_RESULTS_DEPTH,
    progress_callback: Callable[[CatalogScanProgress], None] | None = None,
) -> List[FigureRecord]:
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    results_root = repo_root / "results"
    review_root = repo_root / "review_figures"
    builders: Dict[str, MutableMapping[str, Any]] = {}

    summary_manifests: List[Path] = []
    checkpoint_manifests: List[Path] = []
    review_candidates: List[Tuple[Path, Path, str, Dict[str, str], Dict[str, Any]]] = []

    if results_root.exists():
        summary_manifests = sorted(set(
            results_root.rglob("summary/manifest.json")
        ) | set(results_root.rglob("analysis/manifest.json")) | set(results_root.rglob("analysis/manifests/manifest.json")))
        checkpoint_manifests = sorted(
            set(results_root.rglob("checkpoint_examples/manifest.json"))
            | set(results_root.rglob("figures/checkpoint/manifest.json"))
        )
    if include_review_figures and review_root.exists():
        review_candidates = list(_review_candidates(review_root))

    # Manifests provide metadata, but some pipelines publish figures without one.
    # New-layout filesystem fallback is restricted to disposable figure trees.
    direct_result_files: List[Path] = []
    if results_root.exists():
        direct_result_files = sorted(
            path
            for path in results_root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and not {part.lower() for part in path.parts} & {"cache", "entities", "statistics", "reports", "manifests"}
            )
        )

    manifested_paths: set[Path] = set()

    for manifest_file in summary_manifests:
        manifest = load_manifest(manifest_file.parent.parent)
        if not isinstance(manifest, Mapping):
            continue
        output_root = Path(manifest.get("output_root") or manifest_file.parent.parent)
        manifested_paths.update(
            source_path.resolve()
            for source_path, _, _, _ in _summary_candidates(
                manifest,
                output_root,
                summary_depth_limit,
            )
        )

    direct_result_files = [
        path for path in direct_result_files
        if path.resolve() not in manifested_paths
    ]

    total_steps = (
        len(summary_manifests)
        + len(checkpoint_manifests)
        + len(review_candidates)
        + len(direct_result_files)
    )
    processed_steps = 0

    if progress_callback is not None:
        message_bits = [f"{len(summary_manifests)} summary manifests", f"{len(checkpoint_manifests)} checkpoint manifests"]
        if include_review_figures:
            message_bits.append(f"{len(review_candidates)} review figures")
        _emit_progress(progress_callback, "Scanning results/", 0, total_steps, "Locating " + ", ".join(message_bits) + ".")
        if total_steps == 0:
            _emit_progress(progress_callback, "Scanning results/", 0, 0, "No result figures found.")

    if results_root.exists():
        for manifest_file in summary_manifests:
            manifest = load_manifest(manifest_file.parent.parent)
            if not isinstance(manifest, Mapping):
                continue
            output_root = Path(manifest.get("output_root") or manifest_file.parent.parent)
            output_context = _summary_context(manifest, output_root, repo_root)
            manifest_meta = {
                "output_root": str(output_root),
                "manifest_path": str(manifest_file),
                "comparison_preset_name": manifest.get("comparison_preset_name"),
                "analysis_branch_name": manifest.get("analysis_branch_name"),
                "analysis_basis_name": manifest.get("analysis_basis_name"),
                "job_spec": dict(manifest.get("job_spec", {})) if isinstance(manifest.get("job_spec"), Mapping) else {},
            }
            for source_path, preview_path, source_kind, artifact_metadata in _summary_candidates(manifest, output_root, summary_depth_limit):
                context = dict(output_context)
                for field_name in ("pipeline", "preset", "split", "basis", "family", "cohort", "compartment", "dendrite_region", "region_type", "metric"):
                    value = artifact_metadata.get(field_name) or artifact_metadata.get("analysis_type" if field_name == "metric" else field_name)
                    if value:
                        context[field_name] = canonical_state_label(value)
                try:
                    rel_parts = list(source_path.resolve().relative_to(results_root.resolve()).parts)
                except Exception:
                    rel_parts = list(source_path.parts)
                path_context = _path_context(Path(*rel_parts), results_root)
                for key, value in path_context.items():
                    if value:
                        context[key] = value
                # Explicit artifact metadata has precedence over path inference.
                for field_name in ("pipeline", "preset", "split", "basis", "family", "cohort", "compartment", "dendrite_region", "region_type", "metric"):
                    value = artifact_metadata.get(field_name) or artifact_metadata.get("analysis_type" if field_name == "metric" else field_name)
                    if value:
                        context[field_name] = canonical_state_label(value)
                title = _figure_title_from_path(source_path)
                key = source_path.with_suffix("").resolve().as_posix()
                _merge_builder(
                    builders.setdefault(key, {}),
                    path=source_path,
                    preview_path=preview_path,
                    source_kind=source_kind,
                    context=context,
                    title=title,
                    manifest_path=manifest_file,
                    metadata=manifest_meta,
                )
            processed_steps += 1
            if progress_callback is not None:
                try:
                    relative_text = manifest_file.resolve().relative_to(repo_root.resolve()).as_posix()
                except Exception:
                    relative_text = manifest_file.as_posix()
                _emit_progress(
                    progress_callback,
                    "Processing summary manifests",
                    processed_steps,
                    total_steps,
                    relative_text,
                )

        for checkpoint_manifest_file in checkpoint_manifests:
            checkpoint_root = checkpoint_manifest_file.parent
            manifest = load_manifest(checkpoint_root)
            if not isinstance(manifest, Mapping):
                continue
            checkpoint_output_root = checkpoint_root.parent
            output_context = _context_from_output_root(checkpoint_output_root, repo_root)
            manifest_meta = {
                "output_root": str(checkpoint_output_root),
                "manifest_path": str(checkpoint_manifest_file),
                "gallery_dir": str(checkpoint_root),
            }
            for source_path, preview_path, source_kind, entry_context, metadata in _checkpoint_candidates(checkpoint_root, manifest, repo_root):
                context = dict(output_context)
                context.update(entry_context)
                title = _figure_title_from_path(source_path)
                key = source_path.with_suffix("").resolve().as_posix()
                _merge_builder(
                    builders.setdefault(key, {}),
                    path=source_path,
                    preview_path=preview_path,
                    source_kind=source_kind,
                    context=context,
                    title=title,
                    manifest_path=checkpoint_manifest_file,
                    metadata={**manifest_meta, **metadata},
                )
            processed_steps += 1
            if progress_callback is not None:
                try:
                    relative_text = checkpoint_manifest_file.resolve().relative_to(repo_root.resolve()).as_posix()
                except Exception:
                    relative_text = checkpoint_manifest_file.as_posix()
                _emit_progress(
                    progress_callback,
                    "Processing checkpoint manifests",
                    processed_steps,
                    total_steps,
                    relative_text,
                )

        for source_path in direct_result_files:
            context = _path_context(source_path, repo_root)
            title = _figure_title_from_path(source_path)
            key = source_path.with_suffix("").resolve().as_posix()
            _merge_builder(
                builders.setdefault(key, {}),
                path=source_path,
                preview_path=source_path,
                source_kind="results_filesystem",
                context=context,
                title=title,
                manifest_path=None,
                metadata={"output_root": str(results_root)},
            )
            processed_steps += 1
            if progress_callback is not None:
                _emit_progress(
                    progress_callback,
                    "Indexing result files",
                    processed_steps,
                    total_steps,
                    source_path.relative_to(repo_root).as_posix(),
                )

    if include_review_figures and review_candidates:
        for source_path, preview_path, source_kind, context, metadata in review_candidates:
            title = _figure_title_from_path(source_path)
            key = source_path.with_suffix("").resolve().as_posix()
            _merge_builder(
                builders.setdefault(key, {}),
                path=source_path,
                preview_path=preview_path,
                source_kind=source_kind,
                context=context,
                title=title,
                manifest_path=None,
                metadata=metadata,
            )
            processed_steps += 1
            if progress_callback is not None:
                try:
                    relative_text = source_path.resolve().relative_to(repo_root.resolve()).as_posix()
                except Exception:
                    relative_text = source_path.as_posix()
                _emit_progress(
                    progress_callback,
                    "Processing review figures",
                    processed_steps,
                    total_steps,
                    relative_text,
                )

    records: List[FigureRecord] = []
    for builder in builders.values():
        source_paths = tuple(sorted((Path(path) for path in builder.get("source_paths", set())), key=lambda path: path.as_posix()))
        source_kinds = tuple(sorted(str(kind) for kind in builder.get("source_kinds", set())))
        record = FigureRecord(
            figure_key=str(builder.get("figure_key", "")),
            display_label=str(builder.get("display_label", builder.get("title", ""))),
            title=str(builder.get("title", "")),
            preview_path=Path(builder.get("preview_path")),
            comparison_key=str(builder.get("comparison_key", "")),
            comparison_label=str(builder.get("comparison_label", "")),
            source_paths=source_paths,
            source_kinds=source_kinds,
            pipeline=str(builder.get("pipeline", "")),
            preset=str(builder.get("preset", "")),
            split=str(builder.get("split", "")),
            basis=str(builder.get("basis", "")),
            family=str(builder.get("family", "")),
            cohort=str(builder.get("cohort", "")),
            scope=str(builder.get("scope", "")),
            compartment=str(builder.get("compartment", "")),
            dendrite_region=str(builder.get("dendrite_region", "")),
            region_type=str(builder.get("region_type", "")),
            metric=str(builder.get("metric", "")),
            variant=str(builder.get("variant", "")),
            source_root=str(builder.get("metadata", {}).get("output_root", "") or builder.get("metadata", {}).get("review_root", "")),
            manifest_path=str(builder.get("metadata", {}).get("manifest_path", "")),
            metadata=dict(builder.get("metadata", {})),
            search_text=str(builder.get("search_text", "")),
            sort_key=tuple(str(item) for item in builder.get("sort_key", ())),
        )
        if record.preview_path.exists():
            records.append(record)

    records.sort(key=lambda record: record.sort_key)
    if progress_callback is not None:
        _emit_progress(
            progress_callback,
            "Finalizing figure list",
            total_steps,
            total_steps,
            f"Indexed {len(records)} figures.",
        )
    return records


def filter_records(records: Sequence[FigureRecord], filters: FigureFilterState) -> List[FigureRecord]:
    normalized = filters.normalized()
    query = normalized.search.lower()
    filtered: List[FigureRecord] = []
    for record in records:
        if normalized.pipeline and record.pipeline != normalized.pipeline:
            continue
        if normalized.preset and record.preset != normalized.preset:
            continue
        if normalized.split and record.split != normalized.split:
            continue
        if normalized.basis and record.basis != normalized.basis:
            continue
        if normalized.family and record.family != normalized.family:
            continue
        if normalized.compartment and record.compartment != normalized.compartment:
            continue
        if normalized.dendrite_region and record.dendrite_region != normalized.dendrite_region:
            continue
        if normalized.region_type and record.region_type != normalized.region_type:
            continue
        if normalized.metric and record.metric != normalized.metric:
            continue
        if normalized.cohort and record.cohort != normalized.cohort:
            continue
        if normalized.scope and record.scope != normalized.scope:
            continue
        if query and query not in record.search_text:
            continue
        filtered.append(record)
    return filtered


def unique_values(records: Sequence[FigureRecord], field_name: str) -> List[str]:
    values = {
        str(getattr(record, field_name) or "").strip()
        for record in records
        if str(getattr(record, field_name) or "").strip()
    }
    return sorted(values, key=lambda value: value.lower())


def group_records(records: Sequence[FigureRecord]) -> Dict[str, List[FigureRecord]]:
    grouped: Dict[str, List[FigureRecord]] = {}
    for record in records:
        key = record.comparison_key or record.figure_key
        grouped.setdefault(key, []).append(record)
    for group in grouped.values():
        group.sort(key=lambda record: record.sort_key)
    return grouped


def comparison_group_label(record: FigureRecord) -> str:
    return record.comparison_label or record.display_label or record.title
