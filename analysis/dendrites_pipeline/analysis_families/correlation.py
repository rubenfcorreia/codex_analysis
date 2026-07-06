from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    from .core import run_correlation_family
except ImportError:
    from core import run_correlation_family


def run_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    shuffle_n: int,
    shared_shuffle_cache: Optional[Dict[str, Any]],
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    run_correlation_family(
        cache,
        results,
        state_comparison_states=state_comparison_states,
        shuffle_n=shuffle_n,
        shared_shuffle_cache=shared_shuffle_cache,
        output_dir=output_dir,
        figure_root=figure_root,
    )
