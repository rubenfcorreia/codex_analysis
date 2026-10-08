from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


def correlation_matrix(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(vectors) < 2:
        return None
    min_len = min(np.asarray(vec).size for vec in vectors)
    if min_len < 2:
        return None
    arr = np.vstack([np.asarray(vec[:min_len], dtype=float) for vec in vectors])
    with np.errstate(invalid="ignore"):
        return np.corrcoef(arr)


def upper_triangle_values(matrix: Optional[np.ndarray]) -> np.ndarray:
    if matrix is None or matrix.size == 0:
        return np.array([], dtype=float)
    idx = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[idx], dtype=float)


def matrix_similarity_analysis(
    vectors_a: List[np.ndarray],
    vectors_b: List[np.ndarray],
    shuffle_n: int,
    *,
    shuffle_seed: int = 12345,
) -> Tuple[float, float, float, int, int, float, float]:
    matrix_a = correlation_matrix(vectors_a)
    matrix_b = correlation_matrix(vectors_b)
    if matrix_a is None or matrix_b is None:
        return (float("nan"),) * 3 + (0, 0, float("nan"), float("nan"))
    tri_a = upper_triangle_values(matrix_a)
    tri_b = upper_triangle_values(matrix_b)
    mask = np.isfinite(tri_a) & np.isfinite(tri_b)
    n_pairs = int(mask.sum())
    if n_pairs < 2:
        return (float("nan"),) * 3 + (0, n_pairs, float("nan"), float("nan"))
    observed = float(stats.pearsonr(tri_a[mask], tri_b[mask]).statistic)
    if n_pairs > 3 and np.isfinite(observed) and abs(observed) < 1:
        z = float(np.arctanh(observed))
        half = 1.96 / float(np.sqrt(n_pairs - 3))
        lower_ci, upper_ci = float(np.tanh(z - half)), float(np.tanh(z + half))
    else:
        lower_ci = upper_ci = float("nan")
    combined = vectors_a + vectors_b
    n_a = len(vectors_a)
    rng = np.random.default_rng(int(shuffle_seed))
    null = []
    # With equal-length vectors, every shuffled group correlation matrix is a
    # submatrix of this one combined matrix.  This preserves the original
    # values and removes repeated O(n^2) correlation calculations.
    lengths = [np.asarray(vector).size for vector in combined]
    combined_matrix = correlation_matrix(combined) if lengths and len(set(lengths)) == 1 else None
    for _ in range(max(0, int(shuffle_n))):
        perm = rng.permutation(len(combined))
        if combined_matrix is not None:
            idx_a, idx_b = perm[:n_a], perm[n_a:]
            tri_a_s = upper_triangle_values(combined_matrix[np.ix_(idx_a, idx_a)])
            tri_b_s = upper_triangle_values(combined_matrix[np.ix_(idx_b, idx_b)])
        else:
            group_a = [combined[i] for i in perm[:n_a]]
            group_b = [combined[i] for i in perm[n_a:]]
            tri_a_s, tri_b_s = upper_triangle_values(correlation_matrix(group_a)), upper_triangle_values(correlation_matrix(group_b))
        mask_s = np.isfinite(tri_a_s) & np.isfinite(tri_b_s)
        if int(mask_s.sum()) < 2:
            continue
        null.append(float(stats.pearsonr(tri_a_s[mask_s], tri_b_s[mask_s]).statistic))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)) if null else float("nan")
    null_mean = float(np.nanmean(null)) if null else float("nan")
    return observed, shuffle_p, null_mean, int(len(null)), n_pairs, lower_ci, upper_ci


def shuffle_matrix_similarity(vectors_a: List[np.ndarray], vectors_b: List[np.ndarray], shuffle_n: int, *, shuffle_seed: int = 12345):
    return matrix_similarity_analysis(vectors_a, vectors_b, shuffle_n, shuffle_seed=shuffle_seed)


__all__ = ["correlation_matrix", "upper_triangle_values", "matrix_similarity_analysis", "shuffle_matrix_similarity"]
