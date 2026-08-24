
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analysis.compartment_common import filter_comparison_presets, read_csv_rows
from analysis.shared.result_manifest import load_manifest, manifest_artifact_path, manifest_path

POSTER_REQUIRED_COMPARISON_PRESETS: Tuple[str, ...] = (
    'blank_state_comparisons',
    'movies_state_comparisons',
    'all_requested_comparisons',
)

POSTER_REFERENCE_COMPARISON_PRESETS: Tuple[str, ...] = (
    'all_requested_comparisons',
    'movies_state_comparisons',
    'blank_state_comparisons',
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
        return ComparisonPresetBatchPlan([], '')

    plan_names = [preset_name for preset_name, _ in selected_presets]
    reference_name = next((name for name in poster_reference_names if name in plan_names), plan_names[-1])
    return ComparisonPresetBatchPlan(selected_presets, reference_name)


def _preset_name_from_manifest(manifest: Mapping[str, Any]) -> str:
    run_parameters = manifest.get('run_parameters', {})
    if isinstance(run_parameters, Mapping):
        preset_name = str(run_parameters.get('comparison_preset_name') or '').strip()
        if preset_name:
            return preset_name
    preset_name = str(manifest.get('comparison_preset_name') or '').strip()
    return preset_name


def _preset_csv_relative_candidates(csv_name: str) -> List[str]:
    candidates = [f'csv/{csv_name}', csv_name]
    if csv_name == 'state_comparisons_movie.csv':
        candidates.extend(['csv/state_comparisons.csv', 'state_comparisons.csv'])
    return list(dict.fromkeys(candidates))


def load_comparison_preset_csv_rows(
    batch_result_root: Path,
    preset_name: str,
    csv_name: str,
    *,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    batch_result_root = Path(batch_result_root)
    manifest = load_manifest(batch_result_root)
    if isinstance(manifest, Mapping):
        manifest_preset_name = _preset_name_from_manifest(manifest)
        if manifest_preset_name and manifest_preset_name != preset_name and logger is not None:
            logger.warning(
                'Preset manifest mismatch for %s: expected %s, found %s',
                batch_result_root,
                preset_name,
                manifest_preset_name,
            )
        for relative_candidate in _preset_csv_relative_candidates(csv_name):
            csv_path = manifest_artifact_path(manifest, batch_result_root, relative_candidate)
            if csv_path is not None and csv_path.exists():
                return read_csv_rows(csv_path)
    candidate_paths: List[Path] = []
    for relative_candidate in _preset_csv_relative_candidates(csv_name):
        candidate_paths.append(batch_result_root / relative_candidate)
    for csv_path in dict.fromkeys(candidate_paths):
        if csv_path.exists():
            return read_csv_rows(csv_path)
    if logger is not None:
        logger.warning('Missing preset CSV %s; tried %s', csv_name, ', '.join(str(path) for path in candidate_paths))
    return []


def load_all_requested_comparison_rows(
    batch_result_root: Path,
    csv_name: str,
    *,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    return load_comparison_preset_csv_rows(batch_result_root, 'all_requested_comparisons', csv_name, logger=logger)


def load_comparison_preset_manifest(
    batch_result_root: Path,
    preset_name: str,
    *,
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    batch_result_root = Path(batch_result_root)
    manifest = load_manifest(batch_result_root)
    if isinstance(manifest, Mapping):
        manifest_preset_name = _preset_name_from_manifest(manifest)
        if manifest_preset_name and manifest_preset_name != preset_name and logger is not None:
            logger.warning(
                'Preset manifest mismatch for %s: expected %s, found %s',
                batch_result_root,
                preset_name,
                manifest_preset_name,
            )
        return dict(manifest)
    if logger is not None:
        logger.warning(
            'Missing preset manifest for %s; tried %s',
            preset_name,
            ', '.join(str(path) for path in [manifest_path(batch_result_root), batch_result_root / 'manifest.json']),
        )
    return None
