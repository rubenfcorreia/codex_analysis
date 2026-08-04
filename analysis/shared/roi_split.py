
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
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

ROI_SPLIT_BRANCHES: Tuple[str, ...] = (
    'pooled',
    'activity_split',
    'frequency_split',
    'activity_frequency_split',
)
ROI_SPLIT_BASES: Tuple[str, ...] = ('all', 'nrem', 'rem')
ROI_SPLIT_ACTIVITY_BINARY_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ('more_active', 'More active', '#4c78a8'),
    ('less_active', 'Less active', '#f58518'),
)
ROI_SPLIT_FREQUENCY_BINARY_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ('higher_frequency', 'Higher frequency', '#4c78a8'),
    ('lower_frequency', 'Lower frequency', '#f58518'),
)
ROI_SPLIT_BINARY_GROUPS: Tuple[Tuple[str, str, str], ...] = ROI_SPLIT_ACTIVITY_BINARY_GROUPS
ROI_SPLIT_QUADRANT_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ('high_activity_high_frequency', 'High activity / high frequency', '#4c78a8'),
    ('high_activity_low_frequency', 'High activity / low frequency', '#72b7b2'),
    ('low_activity_high_frequency', 'Low activity / high frequency', '#f58518'),
    ('low_activity_low_frequency', 'Low activity / low frequency', '#e45756'),
)


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


def _row_session_id(row: Mapping[str, Any]) -> str:
    for key in ('expid', 'day_id'):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ''


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




def _basis_label(basis_name: Any) -> str:
    basis_key = canonical_state_label(basis_name)
    if basis_key in {'all', 'overall'}:
        return 'all'
    if basis_key in {'nrem', 'rem'}:
        return basis_key
    return basis_key or 'all'


def split_group_specs_for_branch(split_name: Any, split_mode: Any) -> List[Dict[str, str]]:
    split_key = canonical_state_label(split_mode)
    if split_key in {'activity_frequency', 'activity_frequency_split', 'quadrant'}:
        return [
            {'group': group, 'label': label, 'color': color}
            for group, label, color in ROI_SPLIT_QUADRANT_GROUPS
        ]
    split_name_key = canonical_state_label(split_name)
    if split_name_key in {'frequency', 'frequency_split', 'event_frequency', 'event_frequency_split', 'firing_rate'}:
        return [
            {'group': group, 'label': label, 'color': color}
            for group, label, color in ROI_SPLIT_FREQUENCY_BINARY_GROUPS
        ]
    return [
        {'group': group, 'label': label, 'color': color}
        for group, label, color in ROI_SPLIT_ACTIVITY_BINARY_GROUPS
    ]


def _group_display_map(group_specs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    return {
        str(spec.get('group') or '').strip().lower(): {
            'label': str(spec.get('label') or '').strip(),
            'color': str(spec.get('color') or '#4c78a8').strip(),
        }
        for spec in group_specs
        if str(spec.get('group') or '').strip()
    }


def _group_summary_string(group_specs: Sequence[Mapping[str, Any]], group_counts: Mapping[str, int]) -> str:
    parts: List[str] = []
    for spec in group_specs:
        group = str(spec.get('group') or '').strip()
        if not group:
            continue
        parts.append(f"{group}:{int(group_counts.get(group, 0))}")
    return ';'.join(parts)


def annotate_rows_with_split_group(
    rows: Sequence[Mapping[str, Any]],
    membership_rows: Sequence[Mapping[str, Any]],
    *,
    group_column: str = 'split_group',
) -> List[Dict[str, Any]]:
    membership_lookup: Dict[str, Dict[str, Any]] = {}
    for row in membership_rows or []:
        if not isinstance(row, Mapping):
            continue
        subject_id = str(row.get('subject_id') or '').strip()
        group = str(row.get('group') or '').strip()
        if not subject_id or not group:
            continue
        membership_lookup[subject_id] = {
            'group': group,
            'group_display': str(row.get('group_display') or group).strip(),
            'group_color': str(row.get('group_color') or '').strip(),
            'group_rank': row.get('rank'),
        }

    annotated: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        payload = dict(row)
        subject_id = str(
            row.get('subject_id')
            or row.get('unit_id')
            or row.get('global_soma_id')
            or row.get('global_bouton_id')
            or row.get('global_dendrite_id')
            or row.get('global_spine_id')
            or row.get('roi_id')
            or row.get('roi_key')
            or ''
        ).strip()
        group_info = membership_lookup.get(subject_id)
        if group_info:
            payload[group_column] = group_info['group']
            payload[f'{group_column}_display'] = group_info['group_display']
            if group_info['group_color']:
                payload[f'{group_column}_color'] = group_info['group_color']
            group_rank = group_info.get('group_rank')
            if group_rank is not None:
                payload[f'{group_column}_rank'] = group_rank
        annotated.append(payload)
    return annotated


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
    sleep_expids: Sequence[str] | None = None,
    branch_name: str | None = None,
    basis_name: str = 'all',
    secondary_score_column: str | None = None,
    split_mode: str = 'binary',
) -> Dict[str, Any]:
    selected_state_set = {
        canonical_state_label(state)
        for state in (selected_states or [])
        if canonical_state_label(state)
    }
    sleep_expid_set = {
        str(expid).strip()
        for expid in (sleep_expids or [])
        if str(expid).strip()
    }
    compartment_label = str(compartment or '').strip().lower() or None
    branch_label = canonical_state_label(branch_name)
    basis_key = _basis_label(basis_name)
    split_key = canonical_state_label(split_name) or 'split'
    split_mode_key = canonical_state_label(split_mode) or 'binary'
    group_specs = split_group_specs_for_branch(split_key, split_mode_key)
    group_meta = _group_display_map(group_specs)
    primary_response_columns = [str(column).strip() for column in response_columns if str(column).strip()]
    aggregation_columns = list(dict.fromkeys(primary_response_columns + [column for column in [secondary_score_column] if column]))

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
        response_columns=aggregation_columns,
        state_column=state_column,
        duration_column=duration_column,
        n_frames_column=n_frames_column,
    )
    for row in subject_state_rows:
        row.update(
            {
                'branch_name': branch_label or '',
                'basis_name': basis_key,
                'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
                'split_name': split_key,
                'split_mode': split_mode_key,
            }
        )

    ranking_rows = list(filtered_rows)
    if basis_key in {'nrem', 'rem'} and sleep_expid_set:
        ranking_rows = [row for row in ranking_rows if _row_session_id(row) in sleep_expid_set]
    if basis_key not in {'all', 'overall', 'nrem', 'rem'}:
        ranking_rows = [row for row in ranking_rows if canonical_state_label(row.get(state_column)) == basis_key]

    ranking_subject_state_rows = _aggregate_subject_state_rows(
        ranking_rows,
        subject_key=subject_key,
        roi_type=roi_type,
        compartment=compartment_label,
        score_column=score_column,
        response_columns=aggregation_columns,
        state_column=state_column,
        duration_column=duration_column,
        n_frames_column=n_frames_column,
    )
    ranking_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row['subject_id']), str(row['state'])): dict(row)
        for row in ranking_subject_state_rows
    }
    ranking_subject_ids = sorted({str(row['subject_id']) for row in ranking_subject_state_rows if str(row.get('subject_id') or '').strip()})
    ranking_states = _window_state_labels([str(row['state']) for row in ranking_subject_state_rows], 'overall' if basis_key == 'all' else basis_key, state_order=state_order)

    ranked_subjects: List[Dict[str, Any]] = []
    for subject_id in ranking_subject_ids:
        state_rows = [ranking_lookup.get((subject_id, state)) for state in ranking_states]
        state_rows = [row for row in state_rows if row is not None]
        if not state_rows:
            continue
        score_value, _ = _aggregate_metric(state_rows, score_column, duration_column, n_frames_column)
        if not np.isfinite(score_value):
            continue
        n_frames_total = 0
        duration_total = 0.0
        for row in state_rows:
            n_frames = _as_float(row.get(n_frames_column))
            if np.isfinite(n_frames) and n_frames > 0:
                n_frames_total += int(n_frames)
            duration = _as_float(row.get(duration_column))
            if np.isfinite(duration) and duration > 0:
                duration_total += float(duration)
            elif np.isfinite(n_frames) and n_frames > 0:
                duration_total += float(n_frames)
        payload: Dict[str, Any] = {
            'subject_id': subject_id,
            'score': float(score_value),
            'n_states': int(len(state_rows)),
            'state_duration_s': float(duration_total),
            'state_n_frames': int(n_frames_total),
        }
        if secondary_score_column:
            secondary_value, _ = _aggregate_metric(state_rows, secondary_score_column, duration_column, n_frames_column)
            payload['secondary_score'] = float(secondary_value)
        ranked_subjects.append(payload)

    group_lookup: Dict[str, str] = {}
    if len(group_specs) == 2:
        ranked_subjects = [row for row in ranked_subjects if np.isfinite(_as_float(row.get('score')))]
        ranked_subjects.sort(key=lambda row: (-_as_float(row.get('score')), str(row.get('subject_id'))))
        more_count = (len(ranked_subjects) + 1) // 2
        for index, row in enumerate(ranked_subjects, start=1):
            group = str(group_specs[0]['group']) if index <= more_count else str(group_specs[1]['group'])
            row['group'] = group
            row['rank'] = int(index)
            group_lookup[str(row['subject_id'])] = group
    else:
        ranked_subjects = [row for row in ranked_subjects if np.isfinite(_as_float(row.get('score'))) and np.isfinite(_as_float(row.get('secondary_score')))]
        ranked_subjects.sort(key=lambda row: (-_as_float(row.get('score')), -_as_float(row.get('secondary_score')), str(row.get('subject_id'))))
        if ranked_subjects:
            score_median = float(np.nanmedian([_as_float(row.get('score')) for row in ranked_subjects]))
            secondary_median = float(np.nanmedian([_as_float(row.get('secondary_score')) for row in ranked_subjects]))
            for index, row in enumerate(ranked_subjects, start=1):
                score_high = _as_float(row.get('score')) >= score_median
                secondary_high = _as_float(row.get('secondary_score')) >= secondary_median
                if score_high and secondary_high:
                    group = str(group_specs[0]['group'])
                elif score_high and not secondary_high:
                    group = str(group_specs[1]['group'])
                elif not score_high and secondary_high:
                    group = str(group_specs[2]['group'])
                else:
                    group = str(group_specs[3]['group'])
                row['group'] = group
                row['rank'] = int(index)
                group_lookup[str(row['subject_id'])] = group

    group_counts: Dict[str, int] = {str(spec['group']): 0 for spec in group_specs}
    for row in ranked_subjects:
        group = str(row.get('group') or '')
        if group in group_counts:
            group_counts[group] += 1

    membership_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    window_response_rows: List[Dict[str, Any]] = []
    window_response_comparison_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    pairwise_groups = [(str(left['group']), str(right['group'])) for left_index, left in enumerate(group_specs) for right in group_specs[left_index + 1:]]

    for index, row in enumerate(ranked_subjects, start=1):
        subject_id = str(row.get('subject_id'))
        group = str(row.get('group') or '')
        if not subject_id or not group:
            continue
        payload: Dict[str, Any] = {
            'roi_type': roi_type,
            'compartment': compartment_label or '',
            'branch_name': branch_label or '',
            'basis_name': basis_key,
            'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
            'split_name': split_key,
            'split_mode': split_mode_key,
            'score_column': score_column,
            'secondary_score_column': secondary_score_column or '',
            'subject_id': subject_id,
            'group': group,
            'group_display': group_meta.get(group, {}).get('label', group),
            'group_color': group_meta.get(group, {}).get('color', '#4c78a8'),
            'rank': int(index),
            'score': float(row.get('score', float('nan'))),
            'state_duration_s': float(row.get('state_duration_s', float('nan'))),
            'state_n_frames': int(row.get('state_n_frames', 0)),
            'n_states': int(row.get('n_states', 0)),
            'n_groups': int(len(group_specs)),
        }
        if 'secondary_score' in row:
            payload['secondary_score'] = float(row.get('secondary_score', float('nan')))
        for spec in group_specs:
            group_name = str(spec['group'])
            payload[f'n_{group_name}'] = int(group_counts.get(group_name, 0))
        membership_rows.append(payload)

    for window in windows:
        window_key = canonical_state_label(window)
        window_rows = filtered_rows
        if window_key in {'nrem', 'rem'} and sleep_expid_set:
            window_rows = [row for row in filtered_rows if _row_session_id(row) in sleep_expid_set]
        window_subject_state_rows = _aggregate_subject_state_rows(
            window_rows,
            subject_key=subject_key,
            roi_type=roi_type,
            compartment=compartment_label,
            score_column=score_column,
            response_columns=aggregation_columns,
            state_column=state_column,
            duration_column=duration_column,
            n_frames_column=n_frames_column,
        )
        window_subject_state_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {
            (str(row['subject_id']), str(row['state'])): dict(row)
            for row in window_subject_state_rows
        }
        window_subject_ids = sorted({str(row['subject_id']) for row in window_subject_state_rows if str(row.get('subject_id') or '').strip()})
        window_states = _window_state_labels([str(row['state']) for row in window_subject_state_rows], window_key, state_order=state_order)
        if not window_states:
            continue
        window_subject_rows: List[Dict[str, Any]] = []
        for index, ranked_row in enumerate(ranked_subjects, start=1):
            subject_id = str(ranked_row.get('subject_id'))
            group = group_lookup.get(subject_id)
            if group not in group_meta:
                continue
            state_rows = [window_subject_state_lookup.get((subject_id, state)) for state in window_states]
            state_rows = [item for item in state_rows if item is not None]
            if not state_rows:
                continue
            payload = _aggregate_subject_window_rows(
                state_rows,
                score_column=score_column,
                response_columns=aggregation_columns,
                duration_column=duration_column,
                n_frames_column=n_frames_column,
            )
            if not payload:
                continue
            if secondary_score_column:
                secondary_value, _ = _aggregate_metric(state_rows, secondary_score_column, duration_column, n_frames_column)
                payload[secondary_score_column] = float(secondary_value)
            payload.update(
                {
                    'roi_type': roi_type,
                    'compartment': compartment_label or '',
                    'branch_name': branch_label or '',
                    'basis_name': basis_key,
                    'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
                    'split_name': split_key,
                    'split_mode': split_mode_key,
                    'state': window_key,
                    'state_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'state_color': state_display_color(window_key),
                    'window': window_key,
                    'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                    'subject_id': subject_id,
                    'group': group,
                    'group_display': group_meta.get(group, {}).get('label', group),
                    'group_color': group_meta.get(group, {}).get('color', '#4c78a8'),
                    'rank': int(index),
                    'n_states': int(len(state_rows)),
                    'window_n_subjects': int(len(ranked_subjects)),
                    'window_n_groups': int(len(group_specs)),
                }
            )
            for spec in group_specs:
                group_name = str(spec['group'])
                payload[f'n_{group_name}'] = int(group_counts.get(group_name, 0))
            window_subject_rows.append(payload)
            window_response_rows.append(dict(payload))

        for response_column in primary_response_columns:
            for group_a, group_b in pairwise_groups:
                a_values: List[float] = []
                b_values: List[float] = []
                for subject_row in window_subject_rows:
                    value = _as_float(subject_row.get(response_column))
                    if not np.isfinite(value):
                        continue
                    group = str(subject_row.get('group') or '')
                    if group == group_a:
                        a_values.append(value)
                    elif group == group_b:
                        b_values.append(value)
                comparison = _compare_independent_groups(a_values, b_values, metric_name=response_column, shuffle_n=shuffle_n)
                comparison.update(
                    {
                        'comparison': f'{group_a}_vs_{group_b}',
                        'roi_type': roi_type,
                        'compartment': compartment_label or '',
                        'branch_name': branch_label or '',
                        'basis_name': basis_key,
                        'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
                        'split_name': split_key,
                        'split_mode': split_mode_key,
                        'score_column': score_column,
                        'secondary_score_column': secondary_score_column or '',
                        'response_column': response_column,
                        'window': window_key,
                        'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                        'state': window_key,
                        'state_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                        'state_color': state_display_color(window_key),
                        'group_a': group_a,
                        'group_b': group_b,
                        'state_a': group_a,
                        'state_b': group_b,
                        'state_a_display': group_meta.get(group_a, {}).get('label', group_a),
                        'state_b_display': group_meta.get(group_b, {}).get('label', group_b),
                        'group_a_display': group_meta.get(group_a, {}).get('label', group_a),
                        'group_b_display': group_meta.get(group_b, {}).get('label', group_b),
                        'n_subjects': int(min(len(a_values), len(b_values))),
                        'n_groups': int(len(group_specs)),
                        'window_n_subjects': int(len(ranked_subjects)),
                        'window_n_groups': int(len(group_specs)),
                    }
                )
                window_response_comparison_rows.append(comparison)

        for response_column in primary_response_columns:
            for state in window_states:
                for group_a, group_b in pairwise_groups:
                    a_values = []
                    b_values = []
                    for subject_id in window_subject_ids:
                        row = window_subject_state_lookup.get((subject_id, state))
                        if row is None:
                            continue
                        value = _as_float(row.get(response_column))
                        if not np.isfinite(value):
                            continue
                        group = group_lookup.get(subject_id)
                        if group == group_a:
                            a_values.append(value)
                        elif group == group_b:
                            b_values.append(value)
                    comparison = _compare_independent_groups(a_values, b_values, metric_name=response_column, shuffle_n=shuffle_n)
                    comparison.update(
                        {
                            'comparison': f'{group_a}_vs_{group_b}',
                            'roi_type': roi_type,
                            'compartment': compartment_label or '',
                            'branch_name': branch_label or '',
                            'basis_name': basis_key,
                            'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
                            'split_name': split_key,
                            'split_mode': split_mode_key,
                            'score_column': score_column,
                            'secondary_score_column': secondary_score_column or '',
                            'response_column': response_column,
                            'window': window_key,
                            'window_display': WINDOW_DISPLAY_LABELS.get(window_key, state_display_label(window_key)),
                            'state': state,
                            'state_display': state_display_label(state),
                            'state_color': state_display_color(state),
                            'group_a': group_a,
                            'group_b': group_b,
                            'state_a': group_a,
                            'state_b': group_b,
                            'state_a_display': group_meta.get(group_a, {}).get('label', group_a),
                            'state_b_display': group_meta.get(group_b, {}).get('label', group_b),
                            'group_a_display': group_meta.get(group_a, {}).get('label', group_a),
                            'group_b_display': group_meta.get(group_b, {}).get('label', group_b),
                            'n_subjects': int(min(len(a_values), len(b_values))),
                            'n_groups': int(len(group_specs)),
                            'window_n_subjects': int(len(ranked_subjects)),
                            'window_n_groups': int(len(group_specs)),
                        }
                    )
                    comparison_rows.append(comparison)

        ranked_scores = [_as_float(row.get('score')) for row in ranked_subjects if np.isfinite(_as_float(row.get('score')))]
        summary_row: Dict[str, Any] = {
            'roi_type': roi_type,
            'compartment': compartment_label or '',
            'branch_name': branch_label or '',
            'basis_name': basis_key,
            'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
            'split_name': split_key,
            'split_mode': split_mode_key,
            'score_column': score_column,
            'secondary_score_column': secondary_score_column or '',
            'window': basis_key,
            'window_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
            'n_subjects': int(len(ranked_subjects)),
            'n_groups': int(len(group_specs)),
            'group_labels': ','.join(str(spec['label']) for spec in group_specs),
            'group_series': ','.join(str(spec['group']) for spec in group_specs),
            'group_colors': ','.join(str(spec['color']) for spec in group_specs),
            'group_counts': _group_summary_string(group_specs, group_counts),
            'n_state_rows': int(sum(len([1 for state in window_states if (subject_id, state) in window_subject_state_lookup]) for subject_id in window_subject_ids)),
            'n_states': int(len(window_states)),
            'window_states': ','.join(window_states),
            'mean_score': float(np.nanmean(ranked_scores)) if ranked_scores else float('nan'),
            'median_score': float(np.nanmedian(ranked_scores)) if ranked_scores else float('nan'),
            'std_score': float(np.nanstd(ranked_scores, ddof=1)) if len(ranked_scores) > 1 else 0.0 if ranked_scores else float('nan'),
            'min_score': float(np.nanmin(ranked_scores)) if ranked_scores else float('nan'),
            'max_score': float(np.nanmax(ranked_scores)) if ranked_scores else float('nan'),
        }
        for spec in group_specs:
            group_name = str(spec['group'])
            summary_row[f'n_{group_name}'] = int(group_counts.get(group_name, 0))
        summary_rows.append(summary_row)
    subject_state_rows = annotate_rows_with_split_group(subject_state_rows, membership_rows)

    sort_key = lambda row: (
        row.get('branch_name', ''),
        row.get('basis_name', ''),
        row.get('roi_type', ''),
        row.get('compartment', ''),
        row.get('window', ''),
        row.get('response_column', row.get('metric', '')),
        row.get('state', ''),
        row.get('subject_id', ''),
        row.get('group', ''),
    )
    subject_state_rows.sort(key=lambda row: (row.get('branch_name', ''), row.get('basis_name', ''), row.get('subject_id', ''), row.get('state', '')))
    window_response_rows.sort(key=lambda row: (row.get('branch_name', ''), row.get('basis_name', ''), row.get('roi_type', ''), row.get('compartment', ''), row.get('window', ''), row.get('subject_id', '')))
    window_response_comparison_rows.sort(key=lambda row: (row.get('branch_name', ''), row.get('basis_name', ''), row.get('roi_type', ''), row.get('compartment', ''), row.get('window', ''), row.get('response_column', ''), row.get('group_a', ''), row.get('group_b', '')))
    membership_rows.sort(key=sort_key)
    comparison_rows.sort(key=sort_key)
    summary_rows.sort(key=lambda row: (row.get('branch_name', ''), row.get('basis_name', ''), row.get('roi_type', ''), row.get('compartment', ''), row.get('window', '')))
    return {
        'roi_type': roi_type,
        'compartment': compartment_label or '',
        'branch_name': branch_label or '',
        'basis_name': basis_key,
        'basis_display': WINDOW_DISPLAY_LABELS.get(basis_key, state_display_label(basis_key)),
        'split_name': split_key,
        'split_mode': split_mode_key,
        'response_columns': list(primary_response_columns),
        'selected_states': list(selected_states or []),
        'state_order': list(state_order or []),
        'group_labels': [str(spec['label']) for spec in group_specs],
        'group_series': [str(spec['group']) for spec in group_specs],
        'group_colors': [str(spec['color']) for spec in group_specs],
        'subject_state_rows': subject_state_rows,
        'window_response_rows': window_response_rows,
        'window_response_comparison_rows': window_response_comparison_rows,
        'membership_rows': membership_rows,
        'comparison_rows': comparison_rows,
        'summary_rows': summary_rows,
        'counts': {
            'n_input_rows': int(len(filtered_rows)),
            'n_basis_rows': int(len(ranking_rows)),
            'n_subject_state_rows': int(len(subject_state_rows)),
            'n_window_response_rows': int(len(window_response_rows)),
            'n_window_response_comparison_rows': int(len(window_response_comparison_rows)),
            'n_membership_rows': int(len(membership_rows)),
            'n_comparison_rows': int(len(comparison_rows)),
            'n_summary_rows': int(len(summary_rows)),
            'n_groups': int(len(group_specs)),
        },
    }


__all__ = [
    'ROI_SPLIT_BASES',
    'ROI_SPLIT_BRANCHES',
    'WINDOW_LABELS',
    'WINDOW_DISPLAY_LABELS',
    'build_roi_split_results',
    'estimate_frame_duration_seconds',
    'split_group_specs_for_branch',
    'summarize_mask_duration',
]
