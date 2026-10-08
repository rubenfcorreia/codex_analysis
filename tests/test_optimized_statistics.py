import numpy as np
from scipy import stats

from analysis.shared.analysis_families.correlation import correlation_analysis_for_observation
from analysis.shared.analysis_families.matrix_similarity import matrix_similarity_analysis
from analysis.compartment_common import interpolate_series, lagged_correlation, pairwise_correlation


def _reference_null(a, b, shifts):
    values = []
    for shift in shifts:
        shifted = np.roll(b, int(shift))
        mask = np.isfinite(a) & np.isfinite(shifted)
        if int(mask.sum()) < 3:
            continue
        values.append(float(stats.pearsonr(a[mask], shifted[mask]).statistic))
    return values


def test_vectorized_circular_shift_preserves_null_p_value():
    rng = np.random.default_rng(11)
    a = rng.normal(size=80)
    b = rng.normal(size=80)
    a[[3, 31]] = np.nan
    b[[9, 47]] = np.nan
    shifts = np.array([1, 7, 13, 22, 35], dtype=int)
    expected = _reference_null(a, b, shifts)
    result = correlation_analysis_for_observation(
        a,
        b,
        len(shifts),
        shared_shuffle_cache={"entries": {"pair": {"shifts": shifts}}},
        shared_shuffle_key="pair",
    )
    observed = float(result["r"])
    expected_p = (sum(abs(value) >= abs(observed) for value in expected) + 1) / (len(expected) + 1)
    assert result["shuffle_n_success"] == len(expected)
    assert np.isclose(result["shuffle_p"], expected_p)


def test_matrix_similarity_shuffle_is_deterministic():
    rng = np.random.default_rng(12)
    vectors_a = [rng.normal(size=40) for _ in range(4)]
    vectors_b = [rng.normal(size=40) for _ in range(4)]
    first = matrix_similarity_analysis(vectors_a, vectors_b, 12, shuffle_seed=19)
    second = matrix_similarity_analysis(vectors_a, vectors_b, 12, shuffle_seed=19)
    assert first == second


def test_vectorized_lag_scan_matches_legacy_interpolation():
    rng = np.random.default_rng(13)
    time = np.arange(120, dtype=float) * 0.1
    x = rng.normal(size=time.size)
    y = rng.normal(size=time.size)
    x[5] = np.nan
    y[18] = np.nan
    lags = np.arange(-1.0, 1.01, 0.1)
    _, actual = lagged_correlation(time, x, time, y, lags)
    expected = np.asarray([pairwise_correlation(x, interpolate_series(time + lag, y, time)) for lag in lags])
    assert np.allclose(actual, expected, equal_nan=True, atol=1e-12)


def test_shared_shuffle_cache_reports_reuse_and_invalidation(tmp_path):
    from analysis.soma_bouton_pipeline.soma_bouton_pipeline import _load_or_build_shared_permutation_cache

    path = tmp_path / "shuffle.npz"
    metadata = {"seed": 1, "n": 4}
    _, rebuilt, status = _load_or_build_shared_permutation_cache(path, metadata=metadata, rebuild=False)
    assert rebuilt and status == "missing"
    _, rebuilt, status = _load_or_build_shared_permutation_cache(path, metadata=metadata, rebuild=False)
    assert not rebuilt and status == "reused"
    _, rebuilt, status = _load_or_build_shared_permutation_cache(path, metadata={"seed": 2, "n": 4}, rebuild=False)
    assert rebuilt and status == "meta_mismatch"


def test_plot_profiles_preserve_full_and_disable_analysis_only():
    from analysis.shared.plot_profiles import apply_plot_profile

    full = apply_plot_profile({})
    assert full["plot_profile"] == "full"
    assert "poster_ready_only" not in full
    analysis_only = apply_plot_profile({"plot_profile": "analysis_only"})
    assert analysis_only["poster_ready_only"] is True
    assert analysis_only["generate_poster_ready_figures"] is False
    poster_only = apply_plot_profile({"plot_profile": "poster_only"})
    assert poster_only["poster_ready_only"] is True
    assert poster_only["generate_poster_ready_figures"] is True


def test_runtime_diagnostics_reports_no_figure_leak(tmp_path):
    import matplotlib.pyplot as plt
    from analysis.shared.runtime_diagnostics import finish, snapshot

    start = snapshot(tmp_path)
    figure = plt.figure()
    figure.savefig(tmp_path / "figure.png")
    plt.close(figure)
    result = finish(start, tmp_path)
    assert result["figure_leak_count"] == 0
    assert result["output_file_count_delta"] == 1


def test_analysis_cache_reports_corrupt_and_schema_mismatch(tmp_path):
    from analysis.shared.analysis_cache import load_analysis_results_cache
    from analysis.shared.cache_utils import save_npz_cache

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an npz")
    _, status = load_analysis_results_cache(corrupt)
    assert status == "unreadable"
    schema = tmp_path / "schema.npz"
    save_npz_cache(schema, {"schema_version": 999, "analysis_results": {}})
    _, status = load_analysis_results_cache(schema)
    assert status == "schema_mismatch"


def test_cache_records_include_payload_rows_sizes_and_invalidation_reason(tmp_path):
    from analysis.shared.cache_utils import save_npz_cache
    from analysis.shared.runtime_diagnostics import annotate_cache_records

    cache_path = tmp_path / "tables.npz"
    save_npz_cache(cache_path, {"schema_version": 1, "analysis_tables": {"activity_rows": [{"id": 1}, {"id": 2}]}})
    records = annotate_cache_records(
        {"analysis_tables": {"path": str(cache_path), "status": "reused", "key": "abc"}},
        timings={"analysis_tables": 0.25},
    )
    assert len(records) == 1
    assert records[0]["file_size_bytes"] > 0
    assert records[0]["load_elapsed_s"] == 0.25
    assert records[0]["row_counts"]["analysis_tables.activity_rows"] == 2

    missing = annotate_cache_records(
        {"source": {"path": str(tmp_path / "missing.npz"), "status": "meta_mismatch"}}
    )
    assert missing[0]["invalidation_reason"] == "meta_mismatch"


def test_benchmark_snapshot_contains_pipeline_diagnostics(tmp_path):
    from analysis.demo_pipeline.demo_pipeline import _benchmark_snapshot

    manifest = {
        "runtime_diagnostics": {
            "start": {"open_figures": 0, "peak_rss_bytes": 10},
            "end": {"open_figures": 0, "peak_rss_bytes": 20, "file_count": 2, "output_bytes": 30},
            "figure_leak_count": 0,
        },
        "stage_timings": [{"name": "cache load", "elapsed_s": 0.5}],
        "cache_summary": {"source": {"path": str(tmp_path / "source.npz"), "status": "reused"}},
    }
    snapshot = _benchmark_snapshot(tmp_path, {"demo": manifest})
    assert snapshot["peak_rss_bytes"] == 20
    assert snapshot["figure_leak_count"] == 0
    assert snapshot["pipelines"]["demo"]["stage_totals_s"]["cache"] == 0.5


def test_exact_null_result_cache_reuses_only_identical_traces():
    from analysis.shared.analysis_families.correlation import correlation_analysis_for_observation

    a = np.linspace(0.0, 1.0, 40)
    b = np.sin(np.linspace(0.0, 3.0, 40))
    cache = {"entries": {"pair-state": {"shifts": np.array([1, 3, 7, 11], dtype=int)}}}
    first = correlation_analysis_for_observation(
        a, b, 4, shared_shuffle_cache=cache, shared_shuffle_key="pair-state"
    )
    second = correlation_analysis_for_observation(
        a.copy(), b.copy(), 4, shared_shuffle_cache=cache, shared_shuffle_key="pair-state"
    )
    assert first == second
    assert cache["null_result_stats"]["hits"] == 1
    assert cache["null_result_stats"]["builds"] == 1

    changed_trace = a.copy()
    changed_trace[0] += 0.5
    changed = correlation_analysis_for_observation(
        changed_trace, b, 4, shared_shuffle_cache=cache, shared_shuffle_key="pair-state"
    )
    assert changed["r"] != first["r"]
    assert cache["null_result_stats"]["misses"] == 2
