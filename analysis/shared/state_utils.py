from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path: Path | str, repo_root: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def safe_filename_component(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._") or "value"


def canonical_state_label(label: Any) -> str:
    return safe_filename_component(label).lower()


def state_family_label(label: Any) -> str:
    canonical = canonical_state_label(label)
    if canonical in {"all", "overall", "total"}:
        return "all"
    if canonical.startswith("active_awake"):
        return "active_awake"
    if canonical.startswith("quiet"):
        return "quiet_awake"
    if canonical.startswith("nrem"):
        return "nrem"
    if canonical.startswith("rem"):
        return "rem"
    return canonical


def state_display_label(label: Any) -> str:
    return str(label).replace("_", " ").strip().title()


def state_display_color(label: Any) -> str:
    palette = {
        "all": "#4c78a8",
        "active_awake": "#4c78a8",
        "quiet_awake": "#f58518",
        "run": "#f58518",
        "running": "#f58518",
        "still": "#54a24b",
        "quiet": "#f58518",
        "nrem": "#54a24b",
        "rem": "#e45756",
        "wake": "#ff9da6",
        "active": "#9d755d",
        "inactive": "#bab0ab",
    }
    return palette.get(state_family_label(label), "#4c78a8")


def combined_movie_state_label(sleep_label: Any, trial_type: Any) -> str:
    parts = [canonical_state_label(sleep_label), canonical_state_label(trial_type)]
    return "_".join(part for part in parts if part)


def make_day_id(animal_id: str, date: str) -> str:
    return f"{safe_filename_component(animal_id)}_{safe_filename_component(date)}"


def derive_animal_id(expid: str) -> str:
    match = re.search(r"_([A-Za-z0-9]+)$", expid)
    if match:
        return match.group(1)
    return expid.split("_")[-1]


def derive_date(expid: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})_", expid)
    if match:
        return match.group(1)
    return expid[:10]


def grouped_experiments_by_day(expids: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for expid in expids:
        date = derive_date(expid)
        animal = derive_animal_id(expid)
        groups.setdefault(make_day_id(animal, date), []).append(expid)
    return groups


def resolve_analysis_state_selections(config: Mapping[str, Any], mode: str) -> List[str]:
    explicit_state_comparison = config.get("state_comparison_states")
    explicit_basal_apical = config.get("basal_apical_states")
    if explicit_state_comparison is not None or explicit_basal_apical is not None:
        states = explicit_state_comparison if explicit_state_comparison is not None else explicit_basal_apical
        return [str(state) for state in states if str(state)]
    if mode == "movie":
        return list(config.get("movie_states", ["running", "still", "all"]))
    return list(config.get("sleep_states", ["nrem", "rem", "wake", "all"]))


__all__ = [
    "combined_movie_state_label",
    "canonical_state_label",
    "derive_animal_id",
    "derive_date",
    "ensure_dir",
    "grouped_experiments_by_day",
    "make_day_id",
    "resolve_analysis_state_selections",
    "resolve_repo_path",
    "safe_filename_component",
    "state_display_color",
    "state_display_label",
    "state_family_label",
]
