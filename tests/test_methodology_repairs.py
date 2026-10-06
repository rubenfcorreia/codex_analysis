from __future__ import annotations

import numpy as np

from analysis.shared.analysis_families.common_helpers import interpolate_series
from analysis.shared.analysis_families.mixed_model import run_mixed_model_family
from analysis.shared.analysis_families.state import _summarize_roi_trace
from analysis.shared.analysis_families.visual_response import apply_visual_response_fdr


def test_roi_summary_is_trace_specific():
    mask = np.array([True, True, False])
    first = _summarize_roi_trace(np.array([1.0, 3.0, 9.0]), mask)
    second = _summarize_roi_trace(np.array([10.0, 14.0, 2.0]), mask)
    assert first["mean"] == 2.0
    assert second["mean"] == 12.0
    assert first["mean"] != second["mean"]


def test_auxiliary_interpolation_does_not_extrapolate():
    values = interpolate_series(np.array([-1.0, 0.5, 2.0]), np.array([0.0, 1.0]), np.array([2.0, 4.0]))
    assert np.isnan(values[0])
    assert values[1] == 3.0
    assert np.isnan(values[2])


def test_visual_response_fdr_is_applied_across_family():
    rows = [
        {"mode": "movie", "compartment": "soma", "response_metric": "mean", "raw_pvalue": 0.001, "delta": 1.0},
        {"mode": "movie", "compartment": "soma", "response_metric": "mean", "raw_pvalue": 0.20, "delta": 1.0},
    ]
    result = apply_visual_response_fdr(rows)
    assert result[0]["adjusted_pvalue"] == 0.002
    assert result[0]["responsive"] is True
    assert result[1]["responsive"] is False


def test_mixed_model_returns_real_fit_and_state_contrast():
    rows = []
    for animal_index in range(4):
        for roi_index in range(2):
            for state, value in (("quiet_awake", 0.0), ("nrem", 1.0)):
                rows.append({
                    "animal_id": f"animal{animal_index}",
                    "day_id": f"animal{animal_index}_day1",
                    "unit_id": f"animal{animal_index}_roi{roi_index}",
                    "state": state,
                    "mean_activity": value + animal_index * 0.01,
                })
    result = run_mixed_model_family(
        rows,
        "mean_activity",
        "selected_state",
        [{"kind": "state_pair", "state_a": "nrem", "state_b": "quiet_awake"}],
        100,
        state_order=["quiet_awake", "nrem"],
    )
    assert result["fit"]["fit_method"] in {"lbfgs", "powell", "lbfgs_unit_fallback", "powell_unit_fallback"}
    assert result["design"]["random_effect_group"] == "animal_key"
    assert result["contrast_rows"]
    contrast = result["contrast_rows"][0]
    if result["fit"]["converged"]:
        assert contrast["status"] == "ok"
        assert contrast["estimate"] > 0.5
    else:
        assert contrast["status"] == "fit_failed_or_diagnostic_warning"
        assert not np.isfinite(contrast["p_value"])


def test_correlation_settings_and_temporal_null_metadata():
    from analysis.shared.analysis_families.correlation import correlation_analysis_for_observation, validate_correlation_settings
    trace = np.sin(np.linspace(0, 20, 200))
    result = correlation_analysis_for_observation(trace, trace + 0.1 * np.cos(np.linspace(0, 20, 200)), 32, shuffle_seed=7)
    assert result["correlation_method"] == "pearson"
    assert result["p_value_source"] == "circular_shift"
    assert result["shuffle_seed"] == 7
    assert result["shuffle_n_requested"] == 32
    assert result["shuffle_n_success"] == 32
    assert np.isfinite(result["lower_ci"])
    with np.testing.assert_raises(ValueError):
        validate_correlation_settings("spearman", "circular_shift")


def test_missing_frames_do_not_change_time_axis_for_shuffles():
    from analysis.shared.analysis_families.correlation import correlation_analysis_for_observation
    trace = np.sin(np.linspace(0, 12, 120))
    partner = np.cos(np.linspace(0, 12, 120))
    trace[20:25] = np.nan
    result = correlation_analysis_for_observation(trace, partner, 16, shuffle_seed=11)
    assert result["n_valid_samples"] == 115
    assert result["shuffle_n_success"] == 16


def test_matrix_similarity_wrapper_returns_canonical_metadata_shape():
    from analysis.shared.analysis_families.matrix_similarity import shuffle_matrix_similarity
    vectors_a = [np.array([0.0, 1.0, 3.0, 2.0]), np.array([1.0, 2.0, 0.0, 4.0]), np.array([2.0, 0.0, 4.0, 1.0])]
    vectors_b = [np.array([0.0, 2.0, 1.0, 3.0]), np.array([2.0, 1.0, 3.0, 0.0]), np.array([1.0, 3.0, 0.0, 2.0])]
    result = shuffle_matrix_similarity(vectors_a, vectors_b, 8, shuffle_seed=3)
    assert len(result) == 7
    assert result[3] == 8
    assert result[4] >= 2


def test_shuffle_cache_key_changes_with_statistical_settings():
    from analysis.shared.cache_utils import build_shared_shuffle_cache_key
    base = dict(family="correlation", signal="wheel", analysis_unit="day", animal_id="a", day_id="d", source_id="r", vector_length=100)
    first = build_shared_shuffle_cache_key(**base, shuffle_n=10, shuffle_seed=1, min_shift_frames=1)
    second = build_shared_shuffle_cache_key(**base, shuffle_n=20, shuffle_seed=1, min_shift_frames=1)
    third = build_shared_shuffle_cache_key(**base, shuffle_n=10, shuffle_seed=2, min_shift_frames=1)
    assert first != second
    assert first != third


def test_declared_p_value_source_rejects_missing_or_incompatible_values():
    from analysis.shared.statistics import resolve_inferential_p_value
    assert np.isnan(resolve_inferential_p_value({"p_value": 0.001})[0])
    assert np.isnan(resolve_inferential_p_value({"p_value_source": "none_descriptive", "p_value": 0.001})[0])
    assert resolve_inferential_p_value({"p_value_source": "vector_label_permutation", "p_value": 0.02})[0] == 0.02
    assert resolve_inferential_p_value({"p_value_source": "circular_shift", "shuffle_p": 0.02, "adjusted_pvalue": 0.04})[0] == 0.04


def test_identical_transition_values_are_explicitly_non_significant():
    from analysis.shared.state_transitions import paired_transition_summaries
    rows = [
        {"scope": "test", "window_mode": "fixed", "metric": "m", "compartment": "all", "state_before": "a", "state_after": "b", "pre_value": 1.0, "post_value": 1.0, "animal_id": "a", "entity_id": "r1"},
        {"scope": "test", "window_mode": "fixed", "metric": "m", "compartment": "all", "state_before": "a", "state_after": "b", "pre_value": 1.0, "post_value": 1.0, "animal_id": "b", "entity_id": "r2"},
    ]
    result = paired_transition_summaries(rows)
    assert result[0]["paired_test_status"] == "identical_values"
    assert result[0]["paired_pvalue"] == 1.0
