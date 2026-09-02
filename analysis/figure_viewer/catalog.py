from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from analysis.figure_viewer.models import FigureFilterState, FigureRecord
from analysis.shared.result_manifest import load_manifest
from analysis.shared.state_utils import canonical_state_label


DEFAULT_RESULTS_DEPTH = 8
IMAGE_SUFFIXES = {".png", ".svg"}
KNOWN_COHORTS = {
    "all",
    "responsive",
    "nonresponsive",
    "basal",
    "apical",
    "nrem",
    "rem",
    "quiet_awake",
    "quiet_awake_blank",
    "quiet_awake_movies",
    "active_awake",
    "mixed",
}
KNOWN_COMPARTMENTS = {
    "all",
    "basal",
    "apical",
    "soma",
    "bouton",
    "dendrite",
    "spine",
    "axon",
}


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
        "variant": "",
    }
    if not parts:
        return context
    if len(parts) >= 1:
        context["pipeline"] = str(parts[0])
    if len(parts) >= 2:
        context["preset"] = str(parts[1])

    if parts[0] == "review_figures":
        if len(parts) >= 4:
            context["split"] = str(parts[2])
        if len(parts) >= 5:
            context["basis"] = str(parts[3])
        if len(parts) >= 2:
            context["family"] = str(parts[1])
        scope_parts = list(parts[2:-1])
    elif "figures" in parts:
        figures_index = parts.index("figures")
        if figures_index + 1 < len(parts):
            context["family"] = str(parts[figures_index + 1])
        scope_parts = list(parts[figures_index + 2 : -1])
    elif "checkpoint_examples" in parts:
        checkpoint_index = parts.index("checkpoint_examples")
        if checkpoint_index + 1 < len(parts):
            context["family"] = str(parts[checkpoint_index + 1])
        scope_parts = list(parts[checkpoint_index + 2 : -1])
    else:
        if len(parts) >= 3 and parts[2] not in {"figures", "checkpoint_examples"}:
            context["split"] = str(parts[2])
        if len(parts) >= 4 and parts[3] not in {"figures", "checkpoint_examples"}:
            context["basis"] = str(parts[3])
        scope_parts = list(parts[1:-1])

    if scope_parts:
        context["scope"] = "/".join(scope_parts)
        for item in scope_parts:
            label = canonical_state_label(item)
            if not context["cohort"] and label in KNOWN_COHORTS:
                context["cohort"] = label
            if not context["compartment"] and label in KNOWN_COMPARTMENTS:
                context["compartment"] = label
        context["variant"] = canonical_state_label(scope_parts[-1])
    if not context["variant"]:
        context["variant"] = canonical_state_label(Path(parts[-1]).stem)
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
        if job_spec.get("pipeline") and not context["pipeline"]:
            context["pipeline"] = str(job_spec.get("pipeline"))
        if job_spec.get("analysis_type") and not context["preset"]:
            context["preset"] = str(job_spec.get("analysis_type"))
        if job_spec.get("cohort") and not context["cohort"]:
            context["cohort"] = str(job_spec.get("cohort"))

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


def _summary_candidates(manifest: Mapping[str, Any], output_root: Path, depth_limit: int) -> Iterable[Tuple[Path, Path, str]]:
    artifacts = manifest.get("output_artifacts", [])
    if not isinstance(artifacts, Sequence):
        return []
    grouped: Dict[str, Dict[str, Path]] = {}
    for artifact in artifacts:
        rel_text = _normalize_path_text(artifact)
        if not rel_text.lower().endswith(tuple(IMAGE_SUFFIXES)):
            continue
        if "figures" not in rel_text.split("/"):
            continue
        if _depth_within_figures(rel_text) > depth_limit:
            continue
        path = _candidate_path(output_root, rel_text)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        grouped.setdefault(path.with_suffix("").resolve().as_posix(), {})[path.suffix.lower()] = path
    for renditions in grouped.values():
        preview_path = renditions.get(".svg") or renditions.get(".png") or next(iter(renditions.values()))
        for path in renditions.values():
            yield path, preview_path, "summary_manifest"


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
        summary_manifests = sorted(results_root.rglob("summary/manifest.json"))
        checkpoint_manifests = sorted(results_root.rglob("checkpoint_examples/manifest.json"))
    if include_review_figures and review_root.exists():
        review_candidates = list(_review_candidates(review_root))

    # Manifests provide metadata, but some pipelines publish figures without one.
    # The results tree itself is the authoritative fallback for those outputs.
    direct_result_files: List[Path] = []
    if results_root.exists():
        direct_result_files = sorted(
            path
            for path in results_root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and not {part.lower() for part in path.parts} & {"cache", "entities"}
            )
        )

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
            for source_path, preview_path, source_kind in _summary_candidates(manifest, output_root, summary_depth_limit):
                context = dict(output_context)
                try:
                    rel_parts = list(source_path.resolve().relative_to(results_root.resolve()).parts)
                except Exception:
                    rel_parts = list(source_path.parts)
                path_context = _path_context(Path(*rel_parts), results_root)
                for key, value in path_context.items():
                    if value:
                        context[key] = value
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
