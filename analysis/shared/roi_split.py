
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats

from analysis.shared.state_utils import canonical_state_label, state_display_color, state_display_label

WINDOW_LABELS: Tuple[str, ...] = ('overall', 'nrem', 'rem')
WINDOW_DISPLAY_LABELS = {
    'overall': 'Overall',
    'nrem': 'NREM',
    'rem': 'REM',
}


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return float(result) if np.isfinite(result) else float('nan')


def estimate_frame_duration_seconds(time: Sequence[Any]) -> float:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    finite = time_arr[np.isfinite(time_arr)]
    if finite.size < 2:
        return 1.0
    diffs = np.diff(finite)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    dt = float(np.nanmedian(diffs))
    return dt if np.isfinite(dt) and dt > 0 else 1.0


def summarize_mask_duration(time: Sequence[Any], mask: Sequence[Any]) -> Tuple[int, float]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    usable = min(time_arr.size, mask_arr.size)
    if usable <= 0:
        return 0, 0.0
    mask_arr = mask_arr[:usable]
    n_frames = int(np.count_nonzero(mask_arr))
    if n_frames <= 0:
        return 0, 0.0
    frame_duration = estimate_frame_duration_seconds(time_arr[:usable])
    return n_frames, float(n_frames * frame_duration)


def _stable_unique_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values: List[str] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        values.append(text)
    unique = list(dict.fromkeys(values))
    if len(unique) == 1:
        return unique[0]
    return None


def _row_weight(row: Mapping[str, Any], duration_column: str, n_frames_column: str) -> float:
    duration = _as_float(row.get(duration_column))
    if np.isfinite(duration) and duration > 0:
        return float(duration)
    n_frames = _as_float(row.get(n_frames_column))
    if np.isfinite(n_frames) and n_frames > 0:
        return float(n_frames)
    return 1.0


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return float('nan')
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not np.any(mask):
        finite = values_arr[np.isfinite(values_arr)]
        return float(np.nanmean(finite)) if finite.size else float('nan')
    values_arr = values_arr[mask]
    weights_arr = weights_arr[mask]
    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0:
        finite = values_arr[np.isfinite(values_arr)]
        return float(np.nanmean(finite)) if finite.size else float('nan')
    return float(np.average(values_arr, weights=weights_arr))


def _aggregate_metric(rows: Sequence[Mapping[str, Any]], metric_column: str, duration_column: str, n_frames_column: str) -> Tuple[float, float]:
    values: List[float] = []
    weights: List[float] = []
    for row in rows:
        value = _as_float(row.get(metric_column))
        if not np.isfinite(value):
            continue
        weight = _row_weight(row, duration_column, n_frames_column)
        if not np.isfinite(weight) or weight <= 0:
            continue
        values.append(value)
        weights.append(weight)
    return _weighted_mean(values, weights), float(sum(weights)) if weights else float('nan')


def _window_family_label(state: Any) -> str:
    canonical = canonical_state_label(state)
    if not canonical:
        return ''
    if canonical.startswith('nrem'):
        return 'nrem'
    if canonical.startswith('rem'):
        return 'rem'
    return canonical


def _state_matches_window(state: str, window: str) -> bool:
    window_key = canonical_state_label(window)
    state_key = canonical_state_label(state)
    if not state_key:
        return False
    if window_key == 'overall':
        return state_key != 'all'
    return _window_family_label(state_key) == window_key


def _ordered_unique_states(states: Sequence[str], state_order: Sequence[str] | None = None) -> List[str]:
    ordered = [canonical_state_label(state) for state in states if canonical_state_label(state)]
    deduped = list(dict.fromkeys(ordered))
    if not state_order:
        return deduped
    order_map = {canonical_state_label(state): idx for idx, state in enumerate(state_order) if canonical_state_label(state)}
    return sorted(deduped, key=lambda state: (order_map.get(state, len(order_map)), state))


def _window_state_labels(
    available_states: Sequence[str],
    window: str,
    state_order: Sequence[str] | None = None,
) -> List[str]:
    ordered_states = _ordered_unique_states(available_states, state_order=state_order)
    if not ordered_states:
        return []
    window_key = canonical_state_label(window)
    if window_key == 'overall':
        non_all = [state for state in ordered_states if state != 'all']
        return non_all or [state for state in ordered_states if state == 'all']
    return [state for state in ordered_states if _state_matches_window(state, window_key)]


def _subject_id_for_row(row: Mapping[str, Any], subject_key: str) -> str:
    preferred_keys = [
        subject_key,
        'unit_id',
        'roi_key',
        'global_dendrite_id',
        'global_soma_id',
        'global_bouton_id',
        'global_spine_id',
        'roi_id',
    ]
    for key in preferred_keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    roi_index = row.get('roi_index')
    if roi_index is not None and str(roi_index).strip():
        return str(roi_index)
    return ''


def _aggregate_subject_state_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    subject_key: str,
    roi_type: str,
    compartment: str | None,
    score_column: str,
    response_columns: Sequence[str],
    state_column: str,
    duration_column: str,
    n_frames_column: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        state = canonical_state_label(row.get(state_column))
        subject_id = _subject_id_for_row(row, subject_key)
        if not state or not subject_id:
            continue
        grouped[(subject_id, state)].append(row)

    subject_state_rows: List[Dict[str, Any]] = []
    for (subject_id, state), group_rows in grouped.items():
        payload: Dict[str, Any] = {
            'roi_type': roi_type,
            'subject_key': subject_key,
            'subject_id': subject_id,
            'state': state,
            'state_display': state_display_label(state),
            'state_color': state_display_color(state),
            'state_family': _window_family_label(state),
            'compartment': compartment if compartment is not None else str(group_rows[0].get('compartment') or ''),
            'n_rows': int(len(group_rows)),
            'n_recordings': int(len({
                str(row.get('day_id') or row.get('expid') or '')
                for row in group_rows
                if str(row.get('day_id') or row.get('expid') or '').strip()
            })),
            'state_n_frames': 0,
            'state_duration_s': 0.0,
            'score_column': score_column,
        }
        n_frames_total = 0
        duration_total = 0.0
        for row in group_rows:
            n_frames = _as_float(row.get(n_frames_column))
            if np.isfinite(n_frames) and n_frames > 0:
                n_frames_total += int(n_frames)
            duration = _as_float(row.get(duration_column))
            if np.isfinite(duration) and duration > 0:
                duration_total += float(duration)
            elif np.isfinite(n_frames) and n_frames > 0:
                duration_total += float(n_frames)
        payload['state_n_frames'] = int(n_frames_total)
        payload['state_duration_s'] = float(duration_total)
        animal_id = _stable_unique_value(group_rows, 'animal_id')
        if animal_id is not None:
            payload['animal_id'] = animal_id
        score_value, _ = _aggregate_metric(group_rows, score_column, duration_column, n_frames_column)
        payload['score_value'] = float(score_value)
        for column in response_columns:
            value, _ = _aggregate_metric(group_rows, column, duration_column, n_frames_column)
            payload[column] = float(value)
        subject_state_rows.append(payload)

    return subject_state_rows


def _aggregate_subject_window_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_column: str,
    response_columns: Sequence[str],
    duration_column: str,
    n_frames_column: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        'n_states': int(len(rows)),
        'state_duration_s': 0.0,
        'state_n_frames': 0,
    }
    n_frames_total = 0
    duration_total = 0.0
    for row in rows:
        n_frames = _as_float(row.get(n_frames_column))
        if np.isfinite(n_frames) and n_frames > 0:
            n_frames_total += int(n_frames)
        duration = _as_float(row.get(duration_column))
        if np.isfinite(duration) and duration > 0:
            duration_total += float(duration)
        elif np.isfinite(n_frames) and n_frames > 0:
            duration_total += float(n_frames)
    payload['state_n_frames'] = int(n_frames_total)
    payload['state_duration_s'] = float(duration_total)
    score_value, _ = _aggregate_metric(rows, score_column, duration_column, n_frames_column)
    payload['score_value'] = float(score_value)
    for column in response_columns:
        value, _ = _aggregate_metric(rows, column, duration_column, n_frames_column)
        payload[column] = float(value)
    return payload


def _compare_independent_groups(
    a_values: Sequence[float],
    b_values: Sequence[float],
    *,
    metric_name: str,
    shuffle_n: int,
) -> Dict[str, Any]:
    a = np.asarray([_as_float(value) for value in a_values], dtype=float)
    b = np.asarray([_as_float(value) for value in b_values], dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {
            'metric': metric_name,
            'paired': False,
            'test_choice': 'insufficient_data',
            'n_subjects': int(min(a.size, b.size)),
            'n_a': int(a.size),
            'n_b': int(b.size),
        }
    shapiro_a = float(stats.shapiro(a).pvalue) if 3 <= a.size <= 5000 else float('nan')
    shapiro_b = float(stats.shapiro(b).pvalue) if 3 <= b.size <= 5000 else float('nan')
    is_normal = (not np.isfinite(shapiro_a) or shapiro_a > 0.05) and (not np.isfinite(shapiro_b) or shapiro_b > 0.05)
    if is_normal:
        test_choice = 'ttest_ind'
        classical = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        classical_stat = float(classical.statistic)
        classical_p = float(classical.pvalue)
    else:
        test_choice = 'mannwhitneyu'
        classical = stats.mannwhitneyu(a, b, alternative='two-sided')
        classical_stat = float(classical.statistic)
        classical_p = float(classical.pvalue)
    observed_effect = float(np.nanmean(a) - np.nanmean(b))
    rng = np.random.default_rng(12345)
    null: List[float] = []
    if shuffle_n > 0 and a.size + b.size > 0:
        pooled = np.concatenate([a, b])
        n_a = int(a.size)
        for _ in range(int(shuffle_n)):
            perm = rng.permutation(pooled.size)
            null_a = pooled[perm[:n_a]]
            null_b = pooled[perm[n_a:]]
            null.append(float(np.nanmean(null_a) - np.nanmean(null_b)))
    shuffle_p = float((np.sum(np.abs(null) >= abs(observed_effect)) + 1) / (len(null) + 1)) if null else float('nan')
    variance_p = float(stats.levene(a, b).pvalue) if a.size >= 3 and b.size >= 3 else float('nan')
    return {
        'metric': metric_name,
        'paired': False,
        'test_choice': test_choice,
        'n_subjects': int(min(a.size, b.size)),
        'n_a': int(a.size),
        'n_b': int(b.size),
        'mean_a': float(np.nanmean(a)),
        'mean_b': float(np.nanmean(b)),
        'effect_size': observed_effect,
        'classical_stat': classical_stat,
        'classical_p': classical_p,
        'shuffle_p': shuffle_p,
        'normality_p_a': shapiro_a,
        'normality_p_b': shapiro_b,
        'variance_p': variance_p,
    }


def build_roi_split_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    roi_type: str,
    split_name: str,
    score_column: str,
    response_columns: Sequence[str],
    subject_key: str,
    compartment: str | None = None,
    selected_states: Sequence[str] | None = None,
    state_order: Sequence[str] | None = None,
    state_column: str = 'state',
    duration_column: str = 'state_duration_s',
    n_frames_column: str = 'state_n_frames',
    shuffle_n: int = 1000,
    windows: Sequence[str] = WINDOW_LABELS,
) -> Dict[str, Any]:
    selected_state_set = {
        canonical_state_label(state)
        for state in (selected_states or [])
        if canonical_state_label(state)
    }
    compartment_label = str(compartment or '').strip().lower() or None
    filtered_rows: List[Mapping[str, Any]] = []
    for row in rows:
        if compartment_label is not None and str(row.get('compartment') or '').strip().lower() != compartment_label:
            continue
        state = canonical_state_label(row.get(state_column))
        if not state:
            continue
        if selected_state_set and state not in selected_state_set:
            continue
        filtered_rows.append(row)

    subject_state_rows = _aggregate_subject_state_rows(
        filtered_rows,
        subject_key=subject_key,
        roi_type=roi_type,
        compartment=compartment_label,
        score_column=score_column,
        response_columns=response_columns,
        state_column=state_column,
        duration_column=duration_column,
        n_frames_column=n_frames_column,
    )
    subject_state_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row['subject_id']), str(row['state'])): dict(row)
        for row in subject_state_rows
    }
    available_states = [str(row['state']) for row in subject_state_rows]
    subject_ids = sorted({str(row['subject_id']) for row in subject_state_rows if str(row.get('subject_id') or '').strip()})


    membership_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    window_response_rows: List[Dict[str, Any]] = []
    window_response_comparison_rows: List[Dict[str, Any]] = []

    for window in windows:
        window_key = canonical_state_label(window)
        window_states = _window_state_labels(available_states, window_key, state_order=state_order)
        if not window_states:
            continue
        ranked_subjects: List[Dict[str, Any]] = []
        for subject_id in subject_ids:
            state_rows = [subject_state_lookup.get((subject_id, state)) for state in window_states]
            state_rows = [row for row in state_rows if row is not None]
            if not state_rows:
                continue
            values: List[float] = []
            weights: List[float] = []
            for row in state_rows:
                value = _as_float(row.get(score_column))
                if not np.isfinite(value):
                    continue
                weight = _as_float(row.get('state_duration_s'))
                if not np.isfinite(weight) or weight <= 0:
                    weight = _as_float(row.get('state_n_frames'))
                if not np.isfinite(weight) or weight <= 0:
                    weight = 1.0
                values.append(value)
                weights.append(weight)
            if not values:
                continue
            ranked_subjects.append(
                {
                    'subject_id': subject_id,
                    'score': _weighted_mean(values, weights),
                    'n_states': int(len(state_rows)),
                    'state_duration_s': float(sum(weights)),
                    'state_n_frames': int(sum(int(n_frames) for n_frames in [_as_float(row.get('state_n_frames')) for row in state_rows] if np.isfinite(n_frames) and n_frames > 0)),
                }
            )
        ranked_subjects = [row for row in ranked_subjects if np.isfinite(_as_float(row.get('score')))]
        ranked_subjects.sort(key=lambda row: (-_as_float(row.get('score')), str(row.get('subject_id'))))
        n_ranked = len(ranked_subjects)
        more_count = (n_ranked + 1) // 2
        less_count = n_ranked - more_count
        group_lookup: Dict[str, str] = {}
        for rank, row in enumerate(ranked_subjects, start=1):
            group = 'more_active' if rank <= more_count else 'less_active'
            subject_id = str(row.get('subject_id'))
            group_lookup[subject_id] = group
            membership_rows.append(
                {
                    'roi_type': roi_type,
                    'compartment': compartment_label or '',
                    'split_name': split_name,
                    'score_column': score_column,
                    'window': window_key,
                    'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'subject_id': subject_id,
                    'group': group,
                    'rank': int(rank),
                    'score': float(row.get('score', float('nan'))),
                    'n_states': int(row.get('n_states', 0)),
                    'state_duration_s': float(row.get('state_duration_s', float('nan'))),
                    'state_n_frames': int(row.get('state_n_frames', 0)),
                    'n_subjects': int(n_ranked),
                    'n_more_active': int(more_count),
                    'n_less_active': int(less_count),
                }
            )

        ranked_scores = [float(row.get('score', float('nan'))) for row in ranked_subjects if np.isfinite(_as_float(row.get('score')))]
        summary_rows.append(
            {
                'roi_type': roi_type,
                'compartment': compartment_label or '',
                'split_name': split_name,
                'score_column': score_column,
                'window': window_key,
                'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                'n_subjects': int(n_ranked),
                'n_more_active': int(more_count),
                'n_less_active': int(less_count),
                'n_state_rows': int(sum(len([1 for state in window_states if (subject_id, state) in subject_state_lookup]) for subject_id in subject_ids)),
                'n_states': int(len(window_states)),
                'window_states': ','.join(window_states),
                'mean_score': float(np.nanmean(ranked_scores)) if ranked_scores else float('nan'),
                'median_score': float(np.nanmedian(ranked_scores)) if ranked_scores else float('nan'),
                'std_score': float(np.nanstd(ranked_scores, ddof=1)) if len(ranked_scores) > 1 else 0.0 if ranked_scores else float('nan'),
                'min_score': float(np.nanmin(ranked_scores)) if ranked_scores else float('nan'),
                'max_score': float(np.nanmax(ranked_scores)) if ranked_scores else float('nan'),
            }
        )

        window_subject_rows: List[Dict[str, Any]] = []
        for rank, row in enumerate(ranked_subjects, start=1):
            subject_id = str(row.get('subject_id'))
            state_rows = [subject_state_lookup.get((subject_id, state)) for state in window_states]
            state_rows = [item for item in state_rows if item is not None]
            if not state_rows:
                continue
            group = group_lookup.get(subject_id)
            if group not in {'more_active', 'less_active'}:
                continue
            payload = _aggregate_subject_window_rows(
                state_rows,
                score_column=score_column,
                response_columns=response_columns,
                duration_column=duration_column,
                n_frames_column=n_frames_column,
            )
            if not payload:
                continue
            payload.update(
                {
                    'roi_type': roi_type,
                    'compartment': compartment_label or '',
                    'split_name': split_name,
                    'score_column': score_column,
                    'window': window_key,
                    'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'state': window_key,
                    'state_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'state_color': state_display_color(window_key),
                    'subject_id': subject_id,
                    'group': group,
                    'rank': int(rank),
                    'n_states': int(len(state_rows)),
                    'window_n_subjects': int(n_ranked),
                    'window_n_more_active': int(more_count),
                    'window_n_less_active': int(less_count),
                }
            )
            window_subject_rows.append(payload)
            window_response_rows.append(dict(payload))

        for response_column in response_columns:
            more_values: List[float] = []
            less_values: List[float] = []
            for subject_row in window_subject_rows:
                value = _as_float(subject_row.get(response_column))
                if not np.isfinite(value):
                    continue
                group = str(subject_row.get('group') or '').strip().lower()
                if group == 'more_active':
                    more_values.append(value)
                elif group == 'less_active':
                    less_values.append(value)
            comparison = _compare_independent_groups(more_values, less_values, metric_name=response_column, shuffle_n=shuffle_n)
            comparison.update(
                {
                    'comparison': 'more_active_vs_less_active',
                    'roi_type': roi_type,
                    'compartment': compartment_label or '',
                    'split_name': split_name,
                    'score_column': score_column,
                    'response_column': response_column,
                    'window': window_key,
                    'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'state': window_key,
                    'state_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'state_color': state_display_color(window_key),
                    'group_a': 'more_active',
                    'group_b': 'less_active',
                    'state_a': 'more_active',
                    'state_b': 'less_active',
                    'state_a_display': 'More Active',
                    'state_b_display': 'Less Active',
                    'n_more_active': int(len(more_values)),
                    'n_less_active': int(len(less_values)),
                    'window_n_subjects': int(n_ranked),
                    'window_n_more_active': int(more_count),
                    'window_n_less_active': int(less_count),
                }
            )
            window_response_comparison_rows.append(comparison)

        for response_column in response_columns:
            for state in window_states:
                more_values: List[float] = []
                less_values: List[float] = []
                for subject_id in subject_ids:
                    row = subject_state_lookup.get((subject_id, state))
                    if row is None:
                        continue
                    value = _as_float(row.get(response_column))
                    if not np.isfinite(value):
                        continue
                    group = group_lookup.get(subject_id)
                    if group == 'more_active':
                        more_values.append(value)
                    elif group == 'less_active':
                        less_values.append(value)
                comparison = _compare_independent_groups(more_values, less_values, metric_name=response_column, shuffle_n=shuffle_n)
                comparison.update(
                    {
                        'comparison': 'more_active_vs_less_active',
                        'roi_type': roi_type,
                        'compartment': compartment_label or '',
                        'split_name': split_name,
                        'score_column': score_column,
                        'response_column': response_column,
                        'window': window_key,
                        'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                        'state': state,
                        'state_display': state_display_label(state),
                        'state_color': state_display_color(state),
                        'group_a': 'more_active',
                        'group_b': 'less_active',
                        'n_more_active': int(len(more_values)),
                        'n_less_active': int(len(less_values)),
                        'window_n_subjects': int(n_ranked),
                        'window_n_more_active': int(more_count),
                        'window_n_less_active': int(less_count),
                    }
                )
                comparison_rows.append(comparison)

    sort_key = lambda row: (
        row.get('roi_type', ''),
        row.get('compartment', ''),
        row.get('split_name', ''),
        row.get('window', ''),
        row.get('response_column', row.get('metric', '')),
        row.get('state', ''),
        row.get('subject_id', ''),
    )
    subject_state_rows.sort(key=lambda row: (row.get('subject_id', ''), row.get('state', '')))
    window_response_rows.sort(key=lambda row: (row.get('roi_type', ''), row.get('compartment', ''), row.get('split_name', ''), row.get('window', ''), row.get('subject_id', '')))
    window_response_comparison_rows.sort(key=lambda row: (row.get('roi_type', ''), row.get('compartment', ''), row.get('split_name', ''), row.get('window', ''), row.get('response_column', '')))
    membership_rows.sort(key=sort_key)
    comparison_rows.sort(key=sort_key)
    summary_rows.sort(key=lambda row: (row.get('roi_type', ''), row.get('compartment', ''), row.get('split_name', ''), row.get('window', '')))
    return {
        'roi_type': roi_type,
        'compartment': compartment_label or '',
        'split_name': split_name,
        'score_column': score_column,
        'response_columns': list(response_columns),
        'selected_states': list(selected_states or []),
        'state_order': list(state_order or []),
        'subject_state_rows': subject_state_rows,
        'window_response_rows': window_response_rows,
        'window_response_comparison_rows': window_response_comparison_rows,
        'membership_rows': membership_rows,
        'comparison_rows': comparison_rows,
        'summary_rows': summary_rows,
        'counts': {
            'n_input_rows': int(len(filtered_rows)),
            'n_subject_state_rows': int(len(subject_state_rows)),
            'n_window_response_rows': int(len(window_response_rows)),
            'n_window_response_comparison_rows': int(len(window_response_comparison_rows)),
            'n_membership_rows': int(len(membership_rows)),
            'n_comparison_rows': int(len(comparison_rows)),
            'n_summary_rows': int(len(summary_rows)),
        },
    }


__all__ = [
    'WINDOW_LABELS',
    'WINDOW_DISPLAY_LABELS',
    'build_roi_split_results',
    'estimate_frame_duration_seconds',
    'summarize_mask_duration',
]
