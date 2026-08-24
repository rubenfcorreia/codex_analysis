from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.shared.analysis_families.registry import analysis_families_to_text as _shared_analysis_families_to_text
from analysis.shared.analysis_families.registry import normalize_analysis_families as _shared_normalize_analysis_families
from analysis.dendrites_pipeline.dendrites_pipeline import (
    ALL_REQUESTED_STATES,
    DEFAULT_BASAL_APICAL_STATES,
    DENDRITE_RESPONSE_COHORTS,
    DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
    PRIMARY_QUIET_STATES,
    analysis_cache_meta_hash,
    as_float,
    basal_apical_comparison,
    build_shared_shuffle_cache_key,
    build_state_summary_gallery_results,
    build_visual_response_dendrite_summary_results,
    build_visual_response_spine_summary_results,
    build_visual_response_spine_state_summary_results,
    build_visual_response_spine_state_summary_results,
    classify_visual_responsive_dendrites,
    classify_visual_responsive_spines,
    plot_state_summary_figure,
    plot_visual_response_boxplot_figure,
    plot_visual_response_entity_figure,
    cleanup_stale_state_coverage_artifacts,
    correlation_analysis_for_observation,
    derive_animal_id,
    derive_date,
    estimate_sampling_rate,
    format_dendrite_display_name,
    is_significant_row,
    load_cached_analysis_table,
    make_day_id,
    observation_compartment,
    interpolate_series,
    make_day_id,
    shuffle_matrix_similarity,
    derive_animal_id,
    derive_date,
    pairwise_state_comparisons,
    render_analysis_family_figures,
    save_family_results_cache,
    run_direct_trial_type_comparison,
    run_mixed_model_analysis,
    run_spine_coactivity_analysis,
    selected_basal_apical_state_labels,
    selected_matrix_state_labels,
    shuffle_matrix_similarity,
    spine_coactivity_anchor_selection_text,
    step_message,
    step_progress,
    step_scope,
    summarize_state_values,
    summarize_state_values_by_dendrite,
    summarize_cache,
    selected_matrix_state_labels,
    state_summary_figure_dir,
    state_summary_y_limits,
    visual_response_figure_output_dir,
    build_filtered_matrix_similarity_results,
    matrix_similarity_output_compartments,
    build_filtered_spine_coactivity_results,
    spine_coactivity_output_compartments,
    spine_coactivity_anchor_state_compartments,
    visual_response_dendrite_ids,
)

ANALYSIS_FAMILIES: List[str] = [
    "state",
    "basal_apical",
    "direct_trial_type_comparison",
    "correlation",
    "matrix_similarity",
    "mixed_model",
    "spine_coactivity",
]


def normalize_analysis_families(values):
    return _shared_normalize_analysis_families(values, allowed_families=ANALYSIS_FAMILIES)


def analysis_families_to_text(values):
    return _shared_analysis_families_to_text(values, allowed_families=ANALYSIS_FAMILIES)


def _base_results(cache: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "state_comparisons": [],
        "basal_apical_comparisons": [],
        "correlations": [],
        "matrix_similarity": [],
        "state_summaries": {},
        "state_dendrite_summaries": {},
        "demo_validation": [],
        "alerts": list(cache.get("alerts", [])),
        "state_coverage": [],
        "mixed_model": {},
        "mixed_model_selected_state": {},
        "mixed_model_visual_response_responsive": {},
        "mixed_model_visual_response_nonresponsive": {},
        "mixed_model_visual_response_responsive_selected_state": {},
        "mixed_model_visual_response_nonresponsive_selected_state": {},
        "direct_trial_type_comparison": {},
        "spine_coactivity": {},
        "spine_coactivity_model": {},

    }


def _visual_response_count_text(summary: Dict[str, Any]) -> str:
    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    total_responsive = 0
    total_nonresponsive = 0
    total_tested = 0
    parts: List[str] = []
    for compartment in ("basal", "apical"):
        compartment_counts = counts.get(compartment, {}) if isinstance(counts, dict) else {}
        responsive = int(compartment_counts.get("responsive", 0) or 0)
        nonresponsive = int(compartment_counts.get("nonresponsive", 0) or 0)
        tested = int(compartment_counts.get("tested", 0) or 0)
        total_responsive += responsive
        total_nonresponsive += nonresponsive
        total_tested += tested
        if tested > 0:
            parts.append(f"{compartment}: {responsive}/{tested} responsive")
    if parts:
        return f"{total_responsive}/{total_tested} responsive ({', '.join(parts)})"
    return f"{total_responsive}/{total_tested} responsive"


def prepare_visual_response_cohorts(
    cache: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    source_cache: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> Dict[str, Any]:
    dendrite_visual_response = classify_visual_responsive_dendrites(cache, source_cache=source_cache)
    step_message(f"visual response dendrites: {_visual_response_count_text(dendrite_visual_response)}")
    dendrite_visual_response_state_summaries = build_visual_response_dendrite_summary_results(
        cache,
        state_comparison_states,
        dendrite_visual_response,
    )
    spine_visual_response = classify_visual_responsive_spines(cache, source_cache=source_cache)
    step_message(f"visual response spines: {_visual_response_count_text(spine_visual_response)}")
    spine_visual_response_state_summaries = build_visual_response_spine_state_summary_results(
        cache,
        state_comparison_states,
        spine_visual_response,
    )
    if output_dir is not None:
        fig_root = Path(figure_root) if figure_root is not None else (Path(output_dir) / "figures")
        fig_root.mkdir(parents=True, exist_ok=True)
        cut_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        with step_scope("visual response figures"):
            for kind, response_summary in (("dendrites", dendrite_visual_response), ("spines", spine_visual_response)):
                if not isinstance(response_summary, dict):
                    continue
                rows = response_summary.get("rows", []) if isinstance(response_summary, dict) else []
                for cohort in DENDRITE_RESPONSE_COHORTS:
                    cohort_rows = rows if cohort == "all" else [row for row in rows if isinstance(row, dict) and str(row.get("cohort") or "all") == cohort]
                    cohort_dir = visual_response_figure_output_dir(fig_root, kind, cohort)
                    entity_dir = cohort_dir / "entities"
                    finite_pairs = 0
                    saved_entities = 0
                    for row in cohort_rows:
                        blank_value = row.get("mean_blank")
                        visual_value = row.get("mean_visual")
                        if blank_value is None or visual_value is None:
                            continue
                        try:
                            blank_value = float(blank_value)
                            visual_value = float(visual_value)
                        except Exception:
                            continue
                        if np.isfinite(blank_value) and np.isfinite(visual_value):
                            finite_pairs += 1
                        output_path = plot_visual_response_entity_figure(
                            row,
                            cache,
                            source_cache,
                            entity_dir,
                            kind=kind[:-1],
                            cohort_label=cohort,
                            cut_cache=cut_cache,
                        )
                        if output_path:
                            saved_entities += 1
                    output_path = plot_visual_response_boxplot_figure(
                        response_summary,
                        cohort_dir,
                        output_name="visual_response_blank_vs_movies.svg",
                        title=f"{kind[:-1].capitalize()} visual response - {cohort.capitalize()}",
                        cohort_label=cohort,
                        kind=kind,
                    )
                    if output_path:
                        pass
                    else:
                        step_message(f"no visual response figure for {kind}/{cohort} (rows={len(cohort_rows)}, finite_pairs={finite_pairs}, entities={saved_entities})")
    return {
        "dendrite_visual_response": dendrite_visual_response,
        "dendrite_visual_response_state_summaries": dendrite_visual_response_state_summaries,
        "spine_visual_response": spine_visual_response,
        "spine_visual_response_state_summaries": spine_visual_response_state_summaries,
    }


def _needs_visual_response_cohorts(selected_families: Sequence[str]) -> bool:
    selected = {str(family) for family in selected_families}
    return bool(selected.intersection({"state", "mixed_model"}))


def run_state_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    experiments = cache.get("experiments", {})
    state_metric_names = [
        "dendrite_mean",
        "spine_specific_mean",
        "dendrite_event_frequency_per_min",
        "spine_event_frequency_per_min",
        "coincident_event_frequency_per_min",
        "noncoincident_event_frequency_per_min",
    ]
    with step_scope("state summary metrics"):
        state_metric_values = {metric_name: summarize_state_values(cache, metric_name, state_comparison_states) for metric_name in state_metric_names}
        state_metric_dendrite_values = {metric_name: summarize_state_values_by_dendrite(cache, metric_name, state_comparison_states) for metric_name in state_metric_names}
        for metric_name in state_metric_names:
            results["state_summaries"][metric_name] = state_metric_values[metric_name]
            results["state_dendrite_summaries"][metric_name] = state_metric_dendrite_values[metric_name]
    with step_scope("state coverage"):
        for idx, (exp_id, exp_meta) in enumerate(experiments.items(), start=1):
            step_progress(idx, len(experiments), label=str(exp_id))
            time = np.asarray(exp_meta.get("time"), dtype=float)
            sampling_rate = estimate_sampling_rate(time)
            coverage: Dict[str, Any] = {
                "exp_id": exp_id,
                "day_id": exp_id,
                "animal_id": exp_meta.get("animal_id"),
                "compartment": exp_meta.get("compartment"),
                "sampling_rate_hz": float(sampling_rate) if sampling_rate is not None else float("nan"),
            }
            for state_label in ALL_REQUESTED_STATES:
                mask = exp_meta.get("state_masks", {}).get(state_label)
                n_frames = int(np.count_nonzero(mask)) if mask is not None else 0
                coverage[f"{state_label}_frames"] = n_frames
                coverage[f"{state_label}_seconds"] = float(n_frames / sampling_rate) if sampling_rate is not None and sampling_rate > 0 else float("nan")
            cut_meta = exp_meta.get("cut", {})
            for state_label, count in cut_meta.get("trial_state_counts", {}).items():
                coverage[f"{state_label}_trials"] = int(count)
            results["state_coverage"].append(coverage)
    with step_scope("pairwise state comparisons", total=6):
        for idx, metric in enumerate(state_metric_names, start=1):
            step_progress(idx, 6, label=str(metric))
            results["state_comparisons"].extend(pairwise_state_comparisons(cache, metric, state_comparison_states, shuffle_n))
    with step_scope("basal/apical comparisons", total=len(basal_apical_states)):
        for idx, state in enumerate(basal_apical_states, start=1):
            step_progress(idx, len(basal_apical_states), label=str(state))
            for metric in state_metric_names:
                results["basal_apical_comparisons"].append(basal_apical_comparison(cache, metric, state, shuffle_n))
    if output_dir is not None:
        with step_scope("figure generation: state"):
            render_analysis_family_figures(
                output_dir,
                results,
                cache,
                "state",
                figure_root=figure_root,
            )

        _render_visual_response_state_summary_figures(
            output_dir,
            results,
            cache,
            figure_root=figure_root,
        )

def _render_visual_response_state_summary_figures(
    output_dir: Any,
    results: Dict[str, Any],
    cache: Dict[str, Any],
    *,
    figure_root: Optional[Any] = None,
) -> None:
    fig_root = Path(figure_root) if figure_root is not None else (Path(output_dir) / "figures")
    state_groups = (
        ("selected_states", selected_matrix_state_labels(results)),
        ("all_states", list(ALL_REQUESTED_STATES)),
    )
    cohort_specs = (
        ("dendrites", "dendrite_visual_response", "global_dendrite_id", "dendrite"),
        ("spines", "spine_visual_response", "global_spine_id", "spine"),
    )

    for state_group, state_labels in state_groups:
        summary_root = state_summary_figure_dir(fig_root, state_group)
        for kind, source_key, entity_id_key, entity_kind in cohort_specs:
            response_summary = results.get(source_key, {})
            if not isinstance(response_summary, dict) or not response_summary:
                step_message(f"no visual-response state summaries found for {kind} ({state_group})")
                continue
            rows = response_summary.get("rows", []) if isinstance(response_summary, dict) else []
            if not isinstance(rows, list) or not rows:
                step_message(f"no visual-response rows found for {kind} ({state_group})")
                continue
            for cohort in ("responsive", "nonresponsive"):
                cohort_rows = [
                    row
                    for row in rows
                    if isinstance(row, dict) and str(row.get("cohort") or "all") == cohort
                ]
                entity_ids = sorted(
                    {
                        str(row.get(entity_id_key))
                        for row in cohort_rows
                        if row.get(entity_id_key) is not None and str(row.get(entity_id_key)).strip()
                    }
                )
                if not entity_ids:
                    step_message(f"no visual-response state boxplots for {kind}/{cohort} ({state_group}): empty cohort")
                    continue
                if kind == "dendrites":
                    cohort_results = build_state_summary_gallery_results(
                        cache,
                        state_labels,
                        None,
                        dendrite_ids_filter=entity_ids,
                    )
                else:
                    cohort_results = build_state_summary_gallery_results(
                        cache,
                        state_labels,
                        None,
                        spine_ids_filter=entity_ids,
                    )
                if not isinstance(cohort_results, dict) or not cohort_results:
                    step_message(f"no visual-response state boxplots for {kind}/{cohort} ({state_group}): empty summary")
                    continue
                output_path = plot_state_summary_figure(
                    cohort_results,
                    summary_root,
                    output_name=f"visual_response_state_summary_{entity_kind}_{cohort}.svg",
                    title=f"Visual response state summary - {kind.capitalize()} ({cohort}, {state_group})",
                    state_labels=state_labels,
                    y_limits=state_summary_y_limits(cache, state_labels),
                    cohort_label=cohort,
                    state_group=state_group,
                )
                if not output_path:
                    step_message(f"no visual-response state boxplots for {kind}/{cohort} ({state_group}): plotter returned no output")

def run_direct_trial_type_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    shuffle_n: int,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    roi_split = results.get("roi_split", {})
    split_membership_rows = roi_split.get("membership_rows", []) if isinstance(roi_split, dict) else []
    with step_scope("direct trial-type comparison"):
        direct_trial_type_results = run_direct_trial_type_comparison(cache, state_comparison_states, shuffle_n, split_membership_rows)
    results["direct_trial_type_comparison"] = direct_trial_type_results
    results["alerts"].extend(direct_trial_type_results.get("alerts", []))
    if output_dir is not None:
        with step_scope("figure generation: direct_trial_type_comparison"):
            render_analysis_family_figures(output_dir, results, cache, "direct_trial_type_comparison", figure_root=figure_root)

def run_mixed_model_family_block(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    basal_apical_states: Sequence[str],
    shuffle_n: int,
    mixed_model_contrast_p_source: str,
    source_cache: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    with step_scope("mixed model analysis"):
        mixed_model_results = run_mixed_model_analysis(
            cache,
            shuffle_n=shuffle_n,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            mixed_model_contrast_p_source=mixed_model_contrast_p_source,
            source_cache=source_cache,
        )
    results["mixed_model"] = mixed_model_results.get("all_state", {})
    results["mixed_model_selected_state"] = mixed_model_results.get("selected_state", {})

    for key in (
        "mixed_model_visual_response_responsive",
        "mixed_model_visual_response_nonresponsive",
        "mixed_model_visual_response_responsive_selected_state",
        "mixed_model_visual_response_nonresponsive_selected_state",
    ):
        results[key] = mixed_model_results.get(key, {})

    for key in (
        "mixed_model_visual_response_responsive",
        "mixed_model_visual_response_nonresponsive",
        "mixed_model_visual_response_responsive_selected_state",
        "mixed_model_visual_response_nonresponsive_selected_state",
    ):
        value = results.get(key, {})
        if not isinstance(value, dict) or not value:
            step_message(f"{key}: empty")
            continue

        summary_rows = value.get("summary_rows", {})
        contrast_rows = value.get("contrast_rows", [])

        if isinstance(summary_rows, dict):
            n_summary = sum(len(rows) for rows in summary_rows.values())
        elif isinstance(summary_rows, list):
            n_summary = len(summary_rows)
        else:
            n_summary = 0

        n_contrasts = len(contrast_rows) if isinstance(contrast_rows, list) else 0

        step_message(f"{key}: summary_rows={n_summary}, contrast_rows={n_contrasts}")

    results["alerts"].extend(mixed_model_results.get("alerts", []))
    results["demo_validation"].extend(mixed_model_results.get("validation_rows", []))
    if output_dir is not None:
        with step_scope("figure generation: mixed_model"):
            render_analysis_family_figures(output_dir, results, cache, "mixed_model", figure_root=figure_root)


def run_spine_coactivity_family_block(
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
    with step_scope("spine coactivity analysis"):
        spine_coactivity_results = run_spine_coactivity_analysis(
            cache,
            shuffle_n=shuffle_n,
            state_comparison_states=state_comparison_states,
            basal_apical_states=basal_apical_states,
            shared_shuffle_cache=shared_shuffle_cache,
            fit_spine_coactivity_mixed_model=fit_spine_coactivity_mixed_model,
            mixed_model_contrast_p_source=mixed_model_contrast_p_source,
            spine_coactivity_abs_threshold=spine_coactivity_abs_threshold,
        )
    results["spine_coactivity"] = {k: v for k, v in spine_coactivity_results.items() if k != "model"}
    results["spine_coactivity_model"] = {
        "available": spine_coactivity_results.get("available", False),
        "enabled": spine_coactivity_results.get("enabled", False),
        "alerts": list(spine_coactivity_results.get("alerts", [])),
        "summary_rows": {"coactivity_r": list(spine_coactivity_results.get("summary_rows", {}).get("coactivity_r", []))},
        "contrast_rows": list(spine_coactivity_results.get("contrast_rows", [])),
        "designs": spine_coactivity_results.get("designs", {}),
        "model_equations": spine_coactivity_results.get("model_equations", {}),
        "tested_terms": spine_coactivity_results.get("tested_terms", {}),
        "tested_contrasts": spine_coactivity_results.get("tested_contrasts", {}),
        "selection": spine_coactivity_results.get("selection", {}),
    }
    results["alerts"].extend(spine_coactivity_results.get("alerts", []))
    if output_dir is not None:
        with step_scope("figure generation: spine_coactivity"):
            render_analysis_family_figures(output_dir, results, cache, "spine_coactivity", figure_root=figure_root)


def run_correlation_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    shuffle_n: int,
    shared_shuffle_cache: Optional[Dict[str, Any]],
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    analysis_unit = str(cache.get("analysis_unit", "day"))
    dendrite_observations: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for animal_id, animal_entry in animals.items():
        for dendrite_id, dendrite_record in animal_entry["dendrites"].items():
            for exp_id, d_obs in dendrite_record["observations"].items():
                dendrite_observations.append((animal_id, dendrite_id, dendrite_record, d_obs))
    with step_scope("correlations", total=len(dendrite_observations)):
        for idx, (animal_id, dendrite_id, dendrite_record, d_obs) in enumerate(dendrite_observations, start=1):
            step_progress(idx, len(dendrite_observations), label=str(dendrite_id))
            exp_id = str(d_obs.get("exp_id") or "")
            exp_meta = experiments[exp_id]
            wheel = exp_meta["wheel"]
            pupil = exp_meta["pupil"]
            if wheel["interpolated"] is not None:
                wheel_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="dendrite_trace",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=dendrite_id,
                    vector_length=int(np.asarray(d_obs["trace"], dtype=float).size),
                )
                corr = correlation_analysis_for_observation(
                    d_obs["trace"],
                    wheel["interpolated"],
                    shuffle_n,
                    shared_shuffle_cache=shared_shuffle_cache,
                    shared_shuffle_key=wheel_key,
                )
                results["correlations"].append({"analysis": "dendrite_wheel", "animal_id": animal_id, "exp_id": exp_id, "day_id": exp_id, "global_dendrite_id": dendrite_id, "compartment": observation_compartment(cache, exp_id, d_obs), **corr})
            if pupil["series"] is not None:
                pupil_interp = pupil["series"]
                if pupil["time"] is not None and not np.array_equal(pupil["time"], d_obs["time"]):
                    pupil_interp = interpolate_series(d_obs["time"], pupil["time"], pupil["series"])
                pupil_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="dendrite_trace",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=dendrite_id,
                    vector_length=int(np.asarray(d_obs["trace"], dtype=float).size),
                )
                corr = correlation_analysis_for_observation(
                    d_obs["trace"],
                    pupil_interp,
                    shuffle_n,
                    shared_shuffle_cache=shared_shuffle_cache,
                    shared_shuffle_key=pupil_key,
                )
                results["correlations"].append({"analysis": "dendrite_pupil", "animal_id": animal_id, "exp_id": exp_id, "day_id": exp_id, "global_dendrite_id": dendrite_id, "compartment": observation_compartment(cache, exp_id, d_obs), **corr})
            for spine_id in d_obs["spine_ids"]:
                s_obs = dendrite_record["spines"][spine_id]["observations"].get(exp_id)
                if s_obs is None:
                    continue
                spine_raw_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="spine_trace_hp",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=spine_id,
                    vector_length=int(np.asarray(s_obs["trace_hp"], dtype=float).size),
                )
                spine_specific_key = build_shared_shuffle_cache_key(
                    family="correlation",
                    signal="spine_specific",
                    analysis_unit=analysis_unit,
                    animal_id=animal_id,
                    day_id=exp_id,
                    source_id=spine_id,
                    vector_length=int(np.asarray(s_obs["spine_specific"], dtype=float).size),
                )
                corr_raw = correlation_analysis_for_observation(s_obs["trace_hp"], d_obs["trace"], shuffle_n, shared_shuffle_cache=shared_shuffle_cache, shared_shuffle_key=spine_raw_key)
                corr_specific = correlation_analysis_for_observation(s_obs["spine_specific"], d_obs["trace"], shuffle_n, shared_shuffle_cache=shared_shuffle_cache, shared_shuffle_key=spine_specific_key)
                results["correlations"].append({"analysis": "spine_dendrite_raw", "animal_id": animal_id, "exp_id": exp_id, "day_id": exp_id, "global_dendrite_id": dendrite_id, "global_spine_id": spine_id, "compartment": observation_compartment(cache, exp_id, s_obs), **corr_raw})
                results["correlations"].append({"analysis": "spine_dendrite_specific", "animal_id": animal_id, "exp_id": exp_id, "day_id": exp_id, "global_dendrite_id": dendrite_id, "global_spine_id": spine_id, "compartment": observation_compartment(cache, exp_id, s_obs), **corr_specific})
    if output_dir is not None:
        with step_scope("figure generation: correlation"):
            render_analysis_family_figures(output_dir, results, cache, "correlation", figure_root=figure_root)


def run_matrix_similarity_family(
    cache: Dict[str, Any],
    results: Dict[str, Any],
    *,
    state_comparison_states: Sequence[str],
    shuffle_n: int,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
) -> None:
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    matrix_observations: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for animal_id, animal_entry in animals.items():
        for dendrite_id, dendrite_record in animal_entry["dendrites"].items():
            for exp_id, d_obs in dendrite_record["observations"].items():
                matrix_observations.append((animal_id, dendrite_id, dendrite_record, d_obs))
    with step_scope("matrix similarity", total=len(matrix_observations)):
        for idx, (animal_id, dendrite_id, dendrite_record, d_obs) in enumerate(matrix_observations, start=1):
            step_progress(idx, len(matrix_observations), label=str(dendrite_id))
            exp_id = str(d_obs.get("exp_id") or "")
            exp_meta = experiments[exp_id]
            if len(d_obs["spine_ids"]) < 2:
                continue
            state_vectors: Dict[str, List[np.ndarray]] = {}
            for state_label in [k for k in state_comparison_states if k in exp_meta["state_masks"]]:
                mask = exp_meta["state_masks"].get(state_label)
                if mask is None or not np.any(mask):
                    continue
                vectors = []
                for spine_id in d_obs["spine_ids"]:
                    s_obs = dendrite_record["spines"][spine_id]["observations"].get(exp_id)
                    if s_obs is None:
                        continue
                    vec = np.asarray(s_obs["spine_specific"][mask], dtype=float)
                    if vec.size:
                        vectors.append(vec)
                if len(vectors) >= 2:
                    state_vectors[state_label] = vectors
            for state_a, state_b in combinations(sorted(state_vectors), 2):
                observed, shuffle_p, null_mean = shuffle_matrix_similarity(state_vectors[state_a], state_vectors[state_b], shuffle_n)
                results["matrix_similarity"].append({"animal_id": animal_id, "exp_id": exp_id, "day_id": exp_id, "global_dendrite_id": dendrite_id, "compartment": observation_compartment(cache, exp_id, d_obs), "state_a": state_a, "state_b": state_b, "matrix_similarity_r": observed, "shuffle_p": shuffle_p, "shuffle_null_mean": null_mean, "n_spines": int(len(state_vectors[state_a]))})
    if output_dir is not None:
        with step_scope("figure generation: matrix_similarity"):
            render_analysis_family_figures(output_dir, results, cache, "matrix_similarity", figure_root=figure_root)


def run_cached_analysis(
    cache: Dict[str, Any],
    shuffle_n: int,
    *,
    state_comparison_states: Optional[Sequence[str]] = None,
    basal_apical_states: Optional[Sequence[str]] = None,
    source_cache: Optional[Dict[str, Any]] = None,
    shared_shuffle_cache: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Any] = None,
    figure_root: Optional[Any] = None,
    fit_spine_coactivity_mixed_model: bool = False,
    mixed_model_contrast_p_source: str = "classical",
    spine_coactivity_abs_threshold: float = DEFAULT_SPINE_COACTIVITY_ABS_THRESHOLD,
    analysis_families: Optional[Sequence[str]] = None,
    analysis_results_meta: Optional[Dict[str, Any]] = None,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    selected_families = normalize_analysis_families(analysis_families)
    experiments = cache.get("experiments", {})
    animals = cache.get("animals", {})
    state_comparison_states = list(state_comparison_states) if state_comparison_states is not None else list(PRIMARY_QUIET_STATES)
    basal_apical_states = list(basal_apical_states) if basal_apical_states is not None else list(DEFAULT_BASAL_APICAL_STATES)
    results = _base_results(cache)
    analysis_unit = str(cache.get("analysis_unit", "day"))
    results["analysis_mode"] = ",".join(selected_families) if set(selected_families) != set(ANALYSIS_FAMILIES) else "full"
    if output_dir is not None:
        cleanup_stale_state_coverage_artifacts(output_dir)

    if _needs_visual_response_cohorts(selected_families):
        # Prepare visual-response cohorts before family dispatch so downstream analyses can reuse
        # all/responsive/nonresponsive splits.
        with step_scope("visual response cohorts"):
            results.update(
                prepare_visual_response_cohorts(
                    cache,
                    state_comparison_states=state_comparison_states,
                    source_cache=source_cache,
                    output_dir=output_dir,
                    figure_root=figure_root,
                )
            )
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "visual_response", results, base_meta=analysis_results_meta)

        for key in (
            "dendrite_visual_response_state_summaries",
            "spine_visual_response_state_summaries",
        ):
            value = results.get(key, {})
            if not isinstance(value, dict) or not value:
                step_message(f"{key}: empty")
                continue

            step_message(f"{key}: top-level keys={list(value.keys())[:10]}")
    if "state" in selected_families:
        run_state_family(cache, results, state_comparison_states=state_comparison_states, basal_apical_states=basal_apical_states, shuffle_n=shuffle_n, output_dir=output_dir, figure_root=figure_root)
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "state", results, base_meta=analysis_results_meta)
    if "mixed_model" in selected_families:
        run_mixed_model_family_block(cache, results, state_comparison_states=state_comparison_states, basal_apical_states=basal_apical_states, shuffle_n=shuffle_n, mixed_model_contrast_p_source=mixed_model_contrast_p_source, source_cache=source_cache, output_dir=output_dir, figure_root=figure_root)
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "mixed_model", results, base_meta=analysis_results_meta)
    if "direct_trial_type_comparison" in selected_families:
        run_direct_trial_type_family(cache, results, state_comparison_states=state_comparison_states, shuffle_n=shuffle_n, output_dir=output_dir, figure_root=figure_root)
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "direct_trial_type_comparison", results, base_meta=analysis_results_meta)
    if "spine_coactivity" in selected_families:
        run_spine_coactivity_family_block(cache, results, state_comparison_states=state_comparison_states, basal_apical_states=basal_apical_states, shuffle_n=shuffle_n, shared_shuffle_cache=shared_shuffle_cache, fit_spine_coactivity_mixed_model=fit_spine_coactivity_mixed_model, mixed_model_contrast_p_source=mixed_model_contrast_p_source, spine_coactivity_abs_threshold=spine_coactivity_abs_threshold, output_dir=output_dir, figure_root=figure_root)
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "spine_coactivity", results, base_meta=analysis_results_meta)
    if "correlation" in selected_families:
        run_correlation_family(cache, results, state_comparison_states=state_comparison_states, shuffle_n=shuffle_n, shared_shuffle_cache=shared_shuffle_cache, output_dir=output_dir, figure_root=figure_root)
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "correlation", results, base_meta=analysis_results_meta)
    if "matrix_similarity" in selected_families:
        run_matrix_similarity_family(cache, results, state_comparison_states=state_comparison_states, shuffle_n=shuffle_n, output_dir=output_dir, figure_root=figure_root)
        if cache_path is not None and analysis_results_meta is not None:
            save_family_results_cache(cache_path, "matrix_similarity", results, base_meta=analysis_results_meta)
    if source_cache is not None:
        demo_truth = cache.get("demo_truth")
        if demo_truth:
            source_experiments = source_cache.get("experiments", {})
            for expected in demo_truth.get("expected_alphas", []):
                g_spine_id = expected["global_spine_id"]
                exp_id = expected["exp_id"]
                source_exp_meta = source_experiments.get(exp_id)
                observed = None
                if source_exp_meta is not None:
                    animal_id = str(source_exp_meta.get("animal_id") or derive_animal_id(exp_id))
                    date = str(source_exp_meta.get("date") or derive_date(exp_id))
                    compartment = str(source_exp_meta.get("compartment") or "other")
                    day_id = make_day_id(animal_id, date, compartment)
                else:
                    day_id = exp_id
                for animal_id, animal_entry in animals.items():
                    for dendrite_id, dendrite_record in animal_entry["dendrites"].items():
                        if g_spine_id not in dendrite_record["spines"]:
                            continue
                        spine_record = dendrite_record["spines"][g_spine_id]
                        if day_id not in spine_record["observations"]:
                            continue
                        observed = spine_record["observations"][day_id]["alpha"]
                        break
                if observed is not None:
                    results["demo_validation"].append({"exp_id": exp_id, "global_spine_id": g_spine_id, "expected_alpha": float(expected["alpha"]), "observed_alpha": float(observed), "abs_error": float(abs(observed - expected["alpha"]))})
    return results
