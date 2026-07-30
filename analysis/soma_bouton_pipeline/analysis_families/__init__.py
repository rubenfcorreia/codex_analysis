from __future__ import annotations

from typing import List

from analysis.shared.analysis_families.registry import analysis_families_to_text as _shared_analysis_families_to_text
from analysis.shared.analysis_families.registry import normalize_analysis_families as _shared_normalize_analysis_families

ANALYSIS_FAMILIES: List[str] = [
    "state",
    "mixed_model",
    "calcium_events",
    "visual_response",
    "coincidence",
]


def normalize_analysis_families(values):
    return _shared_normalize_analysis_families(values, allowed_families=ANALYSIS_FAMILIES)


def analysis_families_to_text(values):
    return _shared_analysis_families_to_text(values, allowed_families=ANALYSIS_FAMILIES)


__all__ = ["ANALYSIS_FAMILIES", "analysis_families_to_text", "normalize_analysis_families"]
