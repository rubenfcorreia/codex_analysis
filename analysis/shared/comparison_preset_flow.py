from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analysis.compartment_common import filter_comparison_presets, read_csv_rows

POSTER_REQUIRED_COMPARISON_PRESETS: Tuple[str, ...] = (
    "blank_state_comparisons",
    "movies_state_comparisons",
    "all_requested_comparisons",
)

POSTER_REFERENCE_COMPARISON_PRESETS: Tuple[str, ...] = (
    "all_requested_comparisons",
    "movies_state_comparisons",
    "blank_state_comparisons",
)


@dataclass(frozen=True)
class ComparisonPresetBatchPlan:
    presets: List[Tuple[str, Dict[str, Any]]]
    reference_preset_name: str


def _unique_ordered_presets(presets: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    ordered: List[Tuple[str, Dict[str, Any]]] = []
    seen: set[str] = set()
    for preset_name, overrides in presets:
        if preset_name in seen:
            continue
        ordered.append((preset_name, dict(overrides)))
        seen.add(preset_name)
    return ordered


def build_comparison_preset_batch_plan(
    presets: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    selected_names: Optional[Sequence[str]] = None,
    poster_ready_only: bool = False,
    poster_required_names: Sequence[str] = POSTER_REQUIRED_COMPARISON_PRESETS,
    poster_reference_names: Sequence[str] = POSTER_REFERENCE_COMPARISON_PRESETS,
) -> ComparisonPresetBatchPlan:
    available_presets = _unique_ordered_presets(presets)
    selected_presets = filter_comparison_presets(available_presets, selected_names)
    selected_names_set = {name for name, _ in selected_presets}
    required_names_set = {str(name).strip() for name in poster_required_names if str(name).strip()}

    if poster_ready_only:
        selected_presets = [preset for preset in available_presets if preset[0] in required_names_set]
    else:
        selected_presets = list(selected_presets)
        for preset_name, overrides in available_presets:
            if preset_name in required_names_set and preset_name not in selected_names_set:
                selected_presets.append((preset_name, dict(overrides)))

    if not selected_presets:
        return ComparisonPresetBatchPlan([], "")

    plan_names = [preset_name for preset_name, _ in selected_presets]
    reference_name = next((name for name in poster_reference_names if name in plan_names), plan_names[-1])
    return ComparisonPresetBatchPlan(selected_presets, reference_name)


def _candidate_preset_roots(batch_result_root: Path, preset_name: str) -> List[Path]:
    batch_result_root = Path(batch_result_root)
    roots = [batch_result_root]
    if batch_result_root.name != preset_name:
        roots.append(batch_result_root.parent)
    return roots


def load_comparison_preset_csv_rows(
    batch_result_root: Path,
    preset_name: str,
    csv_name: str,
    *,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    candidate_paths: List[Path] = []
    for root in _candidate_preset_roots(Path(batch_result_root), preset_name):
        preset_root = root if root.name == preset_name else root / preset_name
        candidate_paths.append(preset_root / "csv" / csv_name)
        candidate_paths.append(preset_root / csv_name)
        if csv_name == "state_comparisons_movie.csv":
            candidate_paths.append(preset_root / "state_comparisons.csv")
    for csv_path in dict.fromkeys(candidate_paths):
        if csv_path.exists():
            return read_csv_rows(csv_path)
    if logger is not None:
        logger.warning("Missing preset CSV %s; tried %s", csv_name, ", ".join(str(path) for path in candidate_paths))
    return []


def load_all_requested_comparison_rows(
    batch_result_root: Path,
    csv_name: str,
    *,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    return load_comparison_preset_csv_rows(batch_result_root, "all_requested_comparisons", csv_name, logger=logger)


def load_comparison_preset_manifest(
    batch_result_root: Path,
    preset_name: str,
    *,
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    candidate_paths: List[Path] = []
    for root in _candidate_preset_roots(Path(batch_result_root), preset_name):
        preset_root = root if root.name == preset_name else root / preset_name
        candidate_paths.append(preset_root / "summary" / "manifest.json")
        candidate_paths.append(preset_root / "manifest.json")
    for manifest_path in dict.fromkeys(candidate_paths):
        if manifest_path.exists():
            import json

            with manifest_path.open() as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
    if logger is not None:
        logger.warning("Missing preset manifest for %s; tried %s", preset_name, ", ".join(str(path) for path in candidate_paths))
    return None
