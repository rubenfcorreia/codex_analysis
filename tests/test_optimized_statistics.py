import numpy as np
from scipy import stats

from analysis.shared.analysis_families.correlation import correlation_analysis_for_observation
from analysis.shared.analysis_families.matrix_similarity import matrix_similarity_analysis


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
