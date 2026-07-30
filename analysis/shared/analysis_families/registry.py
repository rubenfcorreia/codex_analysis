from __future__ import annotations

from typing import List, Optional, Sequence


def normalize_analysis_families(
    values: Optional[Sequence[str] | str],
    *,
    allowed_families: Sequence[str],
) -> List[str]:
    allowed = [str(family).strip() for family in allowed_families if str(family).strip()]
    if values is None:
        return list(allowed)
    if isinstance(values, str):
        raw = [part.strip() for part in values.split(",") if part.strip()]
    else:
        raw = [str(value).strip() for value in values if str(value).strip()]
    if not raw:
        return list(allowed)
    selected: List[str] = []
    for family in allowed:
        if family in raw and family not in selected:
            selected.append(family)
    unknown = [family for family in raw if family not in allowed]
    if unknown:
        raise SystemExit(
            f"Unknown analysis family/families: {', '.join(unknown)}. Allowed values are: {', '.join(allowed)}"
        )
    return selected


def analysis_families_to_text(
    values: Optional[Sequence[str] | str],
    *,
    allowed_families: Sequence[str],
) -> str:
    return ",".join(normalize_analysis_families(values, allowed_families=allowed_families))
