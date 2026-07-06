from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    from .core import run_spine_coactivity_family_block
except ImportError:
    from core import run_spine_coactivity_family_block


def run_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    shared_shuffle_cache: Optional[Dict[str, Any]],
    fit_spine_coactivity_mixed_model: bool,
    mixed_model_contrast_p_source: str,
    spine_coactivity_abs_threshold: float,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    run_spine_coactivity_family_block(
        cache,
        results,
        state_comparison_states=state_comparison_states,
        basal_apical_states=basal_apical_states,
        shuffle_n=shuffle_n,
        shared_shuffle_cache=shared_shuffle_cache,
        fit_spine_coactivity_mixed_model=fit_spine_coactivity_mixed_model,
        mixed_model_contrast_p_source=mixed_model_contrast_p_source,
        spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
        output_dir=output_dir,
        figure_root=figure_root,
    )
