from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    from .core import run_state_family
except ImportError:
    from core import run_state_family


def run_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    run_state_family(
        cache,
        results,
        state_comparison_states=state_comparison_states,
        basal_apical_states=basal_apical_states,
        shuffle_n=shuffle_n,
        output_dir=output_dir,
        figure_root=figure_root,
    )
