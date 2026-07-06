from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    from .core import run_mixed_model_family_block
except ImportError:
    from core import run_mixed_model_family_block


def run_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    mixed_model_contrast_p_source: str,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    run_mixed_model_family_block(
        cache,
        results,
        state_comparison_states=state_comparison_states,
        basal_apical_states=basal_apical_states,
        shuffle_n=shuffle_n,
        mixed_model_contrast_p_source=mixed_model_contrast_p_source,
        output_dir=output_dir,
        figure_root=figure_root,
    )
