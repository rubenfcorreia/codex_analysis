from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Sequence

MODULE_DIR = Path(__file__).resolve().parent
PARENT_DIR = MODULE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

ANALYSIS_FAMILIES: List[str] = [
    "state",
    "basal_apical",
    "direct_trial_type_comparison",
    "correlation",
    "matrix_similarity",
    "mixed_model",
    "spine_coactivity",
]


def normalize_analysis_families(values: Optional[Sequence[str] | str]) -> List[str]:
    if values is None:
        return list(ANALYSIS_FAMILIES)
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",") if part.strip()]
    else:
        raw = [str(value).strip() for value in values if str(value).strip()]
    if not raw:
        return list(ANALYSIS_FAMILIES)
    selected: List[str] = []
    for family in ANALYSIS_FAMILIES:
        if family in raw and family not in selected:
            selected.append(family)
    unknown = [family for family in raw if family not in ANALYSIS_FAMILIES]
    if unknown:
        raise SystemExit(
            f"Unknown analysis family/families: {', '.join(unknown)}. Allowed values are: {', '.join(ANALYSIS_FAMILIES)}"
        )
    return selected


def analysis_families_to_text(values: Optional[Sequence[str] | str]) -> str:
    return ",".join(normalize_analysis_families(values))
