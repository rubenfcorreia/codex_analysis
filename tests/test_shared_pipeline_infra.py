from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.compartment_common import LoadedBundle
from analysis.dendrites_pipeline.analysis_families import normalize_analysis_families as main_normalize
from analysis.shared.analysis_families.core import ExperimentContext
from analysis.shared.analysis_families.pairwise import build_pairwise_correlation_rows, pairwise_correlation_summary_rows, pairwise_member_from_trace
from analysis.shared.plots.poster_ready import assign_pairwise_visual_response_cohorts, split_rows_by_cohort
from analysis.shared.comparison_preset_flow import (
    POSTER_REQUIRED_COMPARISON_PRESETS,
    build_comparison_preset_batch_plan,
    load_comparison_preset_csv_rows,
)
from analysis.shared.cache_utils import (
    analysis_day_cache_path,
    analysis_results_cache_path,
    analysis_table_cache_path,
    build_pairwise_correlation_cache_key,
    family_results_cache_path,
    load_family_results_cache,
    save_family_results_cache,
    shared_shuffle_cache_path,
)
from analysis.shared.state_utils import grouped_experiments_by_day, make_day_id, resolve_repo_path
from analysis.soma_bouton_pipeline import soma_bouton_pipeline as soma_pipeline
from analysis.soma_bouton_pipeline.plots import plot_state_correlation
from analysis.soma_bouton_pipeline.analysis_families import normalize_analysis_families as soma_normalize
from analysis.shared.analysis_families.registry import normalize_analysis_families as shared_normalize


def _synthetic_context(day_id: str = "mouse1_2024-01-01") -> ExperimentContext:
    soma_bundle = LoadedBundle(
        path=Path(f"/tmp/{day_id}_soma.pkl"),
        data={"t": np.array([0.0, 1.0, 2.0, 3.0]), "dF": np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])},
    )
    bouton_bundle = LoadedBundle(
        path=Path(f"/tmp/{day_id}_bouton.pkl"),
        data={"t": np.array([0.0, 1.0, 2.0, 3.0]), "dF": np.array([[1.0, 1.0, 2.0, 2.0], [4.0, 3.0, 2.0, 1.0]])},
    )
    return ExperimentContext(
        expid=f"{day_id}_exp",
        mode="sleep",
        exp_root=Path(f"/tmp/{day_id}"),
        animal_id="mouse1",
        date="2024-01-01",
        day_id=day_id,
        soma_channel=1,
        bouton_channel=0,
        soma=soma_bundle,
        bouton=bouton_bundle,
        state_bundle={},
        state_bundle_path=Path(f"/tmp/{day_id}_state.pkl"),
    )


def test_cache_paths_are_stage_scoped() -> None:
    base = Path('/tmp/example/dendrites_cache.npz')
    assert analysis_table_cache_path(base).name == 'dendrites_cache_analysis_tables_cache.npz'
    assert analysis_results_cache_path(base).name == 'dendrites_cache_analysis_results_cache.npz'
    assert analysis_day_cache_path(base).name == 'dendrites_cache_analysis_day_cache.npz'
    assert shared_shuffle_cache_path(base).name == 'dendrites_cache_shuffle_cache.npz'
    assert family_results_cache_path(base, 'state').name.endswith('_state_results_cache.npz')
    assert family_results_cache_path(base, 'mixed_model').name.endswith('_mixed_model_results_cache.npz')
    assert family_results_cache_path(base, 'pairwise_correlation').name.endswith('_pairwise_correlation_results_cache.npz')


def test_shuffle_cache_keys_scope_roi_ids_by_day() -> None:
    from analysis.shared.cache_utils import build_shared_shuffle_cache_key

    key_a = build_shared_shuffle_cache_key(
        family='state',
        signal='activity',
        analysis_unit='day',
        animal_id='mouse1',
        day_id='mouse1_2024-01-01',
        source_id='roi-17',
        vector_length=100,
    )
    key_b = build_shared_shuffle_cache_key(
        family='state',
        signal='activity',
        analysis_unit='day',
        animal_id='mouse1',
        day_id='mouse1_2024-01-02',
        source_id='roi-17',
        vector_length=100,
    )
    assert key_a != key_b


def test_pairwise_cache_key_scopes_same_roi_ids_by_day() -> None:
    key_a = build_pairwise_correlation_cache_key(
        family='pairwise_correlation',
        comparison_name='soma_pairwise',
        analysis_unit='day',
        animal_id='mouse1',
        day_id='mouse1_2024-01-01',
        mode='sleep',
        left_compartment='soma',
        left_channel=1,
        right_compartment='soma',
        right_channel=1,
        pair_mode='within_compartment',
        selected_states=['sleep'],
        source_signature={'roi_id': 17},
    )
    key_b = build_pairwise_correlation_cache_key(
        family='pairwise_correlation',
        comparison_name='soma_pairwise',
        analysis_unit='day',
        animal_id='mouse1',
        day_id='mouse1_2024-01-02',
        mode='sleep',
        left_compartment='soma',
        left_channel=1,
        right_compartment='soma',
        right_channel=1,
        pair_mode='within_compartment',
        selected_states=['sleep'],
        source_signature={'roi_id': 17},
    )
    assert key_a != key_b


def test_pairwise_roi_ids_do_not_collide_across_days() -> None:
    ctx_a = _synthetic_context('mouse1_2024-01-01')
    ctx_b = _synthetic_context('mouse1_2024-01-02')
    masks = {'sleep': np.array([True, True, True, True])}
    members_a = [
        pairwise_member_from_trace(ctx=ctx_a, compartment='soma', channel=1, roi_index=0, roi_id=17, trace=[1.0, 2.0, 3.0, 4.0]),
        pairwise_member_from_trace(ctx=ctx_a, compartment='soma', channel=1, roi_index=1, roi_id=18, trace=[2.0, 3.0, 4.0, 5.0]),
    ]
    members_b = [
        pairwise_member_from_trace(ctx=ctx_b, compartment='soma', channel=1, roi_index=0, roi_id=17, trace=[1.0, 2.0, 3.0, 4.0]),
        pairwise_member_from_trace(ctx=ctx_b, compartment='soma', channel=1, roi_index=1, roi_id=18, trace=[2.0, 3.0, 4.0, 5.0]),
    ]
    rows_a = build_pairwise_correlation_rows(ctx_a, masks, comparison_name='soma_pairwise', left_members=members_a)
    rows_b = build_pairwise_correlation_rows(ctx_b, masks, comparison_name='soma_pairwise', left_members=members_b)
    assert rows_a[0]['left_unit_id'] != rows_b[0]['left_unit_id']
    assert rows_a[0]['pair_unit_id'] != rows_b[0]['pair_unit_id']


def test_pairwise_rows_split_by_visual_response_cohort() -> None:
    visual_response_rows = [
        {'compartment': 'soma', 'unit_id': 'soma-responsive', 'responsive': True},
        {'compartment': 'soma', 'unit_id': 'soma-nonresponsive', 'responsive': False},
        {'compartment': 'bouton', 'unit_id': 'bouton-responsive', 'responsive': True},
        {'compartment': 'bouton', 'unit_id': 'bouton-nonresponsive', 'responsive': False},
    ]
    pairwise_rows = [
        {'left_unit_id': 'soma-responsive', 'right_unit_id': 'bouton-responsive'},
        {'left_unit_id': 'soma-nonresponsive', 'right_unit_id': 'bouton-nonresponsive'},
        {'left_unit_id': 'soma-responsive', 'right_unit_id': 'bouton-nonresponsive'},
    ]
    assigned = assign_pairwise_visual_response_cohorts(pairwise_rows, visual_response_rows)
    grouped = split_rows_by_cohort(assigned)
    assert len(grouped['all']) == 3
    assert len(grouped['responsive']) == 1
    assert len(grouped['nonresponsive']) == 1
    assert {row['cohort'] for row in assigned} == {'responsive', 'nonresponsive', 'mixed'}


def test_shared_pairwise_helper_smoke_same_day() -> None:
    ctx = _synthetic_context('mouse1_2024-01-01')
    masks = {'sleep': np.array([True, True, True, True])}
    members = [
        pairwise_member_from_trace(ctx=ctx, compartment='soma', channel=1, roi_index=0, roi_id=17, trace=[1.0, 2.0, 3.0, 4.0]),
        pairwise_member_from_trace(ctx=ctx, compartment='soma', channel=1, roi_index=1, roi_id=18, trace=[2.0, 4.0, 6.0, 8.0]),
    ]
    rows = build_pairwise_correlation_rows(ctx, masks, comparison_name='soma_pairwise', left_members=members)
    summary = pairwise_correlation_summary_rows(rows)
    assert len(rows) == 1
    assert rows[0]['day_id'] == ctx.day_id
    assert rows[0]['pair_mode'] == 'within_compartment'
    assert np.isclose(rows[0]['corr'], 1.0)
    assert len(summary) == 1
    assert summary[0]['n_pairs'] == 1
    assert np.isclose(summary[0]['mean_corr'], 1.0)


def test_pairwise_family_cache_round_trip(tmp_path: Path) -> None:
    base_cache = tmp_path / 'analysis_cache.npz'
    cache_path = family_results_cache_path(base_cache, 'pairwise_correlation')
    ctx = _synthetic_context('mouse1_2024-01-01')
    masks = {'sleep': np.array([True, True, True, True])}
    members = [
        pairwise_member_from_trace(ctx=ctx, compartment='soma', channel=1, roi_index=0, roi_id=17, trace=[1.0, 2.0, 3.0, 4.0]),
        pairwise_member_from_trace(ctx=ctx, compartment='soma', channel=1, roi_index=1, roi_id=18, trace=[2.0, 4.0, 6.0, 8.0]),
    ]
    rows = build_pairwise_correlation_rows(ctx, masks, comparison_name='soma_pairwise', left_members=members)
    summary = pairwise_correlation_summary_rows(rows)
    base_meta = {
        'analysis_name': 'soma_bouton_pipeline',
        'comparison_preset_name': 'default',
        'state_modes': ['sleep'],
        'selected_states_by_mode': {'sleep': ['sleep']},
        'soma_channel': 1,
        'bouton_channel': 0,
        'pairwise_correlation_schema_version': 1,
    }
    save_family_results_cache(
        base_cache,
        'pairwise_correlation',
        {
            'correlation_rows': rows,
            'soma_pairwise_rows': rows,
            'bouton_pairwise_rows': [],
            'correlation_summary_rows': summary,
            'soma_pairwise_summary_rows': summary,
            'bouton_pairwise_summary_rows': [],
        },
        base_meta=base_meta,
    )
    loaded, status = load_family_results_cache(
        cache_path,
        expected_meta={**base_meta, 'family_result_stage': 'pairwise_correlation'},
    )
    assert status == 'ok'
    assert loaded is not None
    assert loaded['analysis_results']['correlation_rows'] == rows


def test_soma_pipeline_reuses_pairwise_family_cache(tmp_path: Path, monkeypatch) -> None:
    ctx = _synthetic_context('mouse1_2024-01-01')
    sleep_mask = {'sleep': np.array([True, True, True, True])}
    soma_a = pairwise_member_from_trace(ctx=ctx, compartment='soma', channel=1, roi_index=0, roi_id=17, trace=[1.0, 2.0, 3.0, 4.0])
    soma_b = pairwise_member_from_trace(ctx=ctx, compartment='soma', channel=1, roi_index=1, roi_id=18, trace=[4.0, 3.0, 2.0, 1.0])
    bouton_a = pairwise_member_from_trace(ctx=ctx, compartment='bouton', channel=0, roi_index=0, roi_id=27, trace=[1.0, 1.0, 2.0, 2.0])
    bouton_b = pairwise_member_from_trace(ctx=ctx, compartment='bouton', channel=0, roi_index=1, roi_id=28, trace=[2.0, 3.0, 4.0, 5.0])
    cached_rows = {
        'correlation_rows': build_pairwise_correlation_rows(ctx, sleep_mask, comparison_name='bouton_soma', left_members=[soma_a], right_members=[bouton_a]),
        'soma_pairwise_rows': build_pairwise_correlation_rows(ctx, sleep_mask, comparison_name='soma_pairwise', left_members=[soma_a, soma_b]),
        'bouton_pairwise_rows': build_pairwise_correlation_rows(ctx, sleep_mask, comparison_name='bouton_pairwise', left_members=[bouton_a, bouton_b]),
    }

    result_root = tmp_path / 'results'
    cache_root = tmp_path / 'cache'
    pairwise_cache_calls = []

    def fake_load_analysis_results_cache(*args, **kwargs):
        return None, 'missing'

    def fake_load_family_results_cache(path, *, expected_meta=None, rebuild=False):
        pairwise_cache_calls.append((Path(path), dict(expected_meta or {}), bool(rebuild)))
        return {'analysis_results': dict(cached_rows)}, 'ok'

    def fake_build_experiment_context(*args, **kwargs):
        expid = kwargs.get('expid') if 'expid' in kwargs else args[0]
        mode = kwargs.get('mode') if 'mode' in kwargs else args[1]
        soma_channel = kwargs.get('soma_channel') if 'soma_channel' in kwargs else args[2]
        bouton_channel = kwargs.get('bouton_channel') if 'bouton_channel' in kwargs else args[3]
        return SimpleNamespace(
            expid=expid,
            mode=mode,
            animal_id='mouse1',
            date='2024-01-01',
            day_id='mouse1_2024-01-01',
            soma_channel=soma_channel,
            bouton_channel=bouton_channel,
        )

    monkeypatch.setattr(soma_pipeline, 'load_analysis_results_cache', fake_load_analysis_results_cache)
    monkeypatch.setattr(soma_pipeline, 'load_family_results_cache', fake_load_family_results_cache)
    monkeypatch.setattr(soma_pipeline, 'build_experiment_context', fake_build_experiment_context)
    monkeypatch.setattr(soma_pipeline, 'activity_rows_for_context', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'experiment_summary_row', lambda ctx: {'expid': ctx.expid, 'mode': ctx.mode, 'day_id': ctx.day_id})
    monkeypatch.setattr(soma_pipeline, 'bouton_soma_correlation_rows', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pairwise should come from cache')))
    monkeypatch.setattr(soma_pipeline, 'soma_pairwise_correlation_rows', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pairwise should come from cache')))
    monkeypatch.setattr(soma_pipeline, 'bouton_pairwise_correlation_rows', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pairwise should come from cache')))
    monkeypatch.setattr(soma_pipeline, 'lag_scan_rows', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'lag_summary_rows', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'state_summary_rows', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'state_comparison_rows', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'run_mixed_model_family', lambda *args, **kwargs: {})
    monkeypatch.setattr(soma_pipeline, 'run_visual_response_family', lambda *args, **kwargs: {'available': False, 'rows': []})
    monkeypatch.setattr(soma_pipeline, 'shared_visual_response_day_rows', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'plot_state_activity', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'plot_state_correlation', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'plot_lag_heatmap', lambda *args, **kwargs: [])
    monkeypatch.setattr(soma_pipeline, 'resolve_analysis_state_selections', lambda *args, **kwargs: ['sleep'])

    config = {
        'analysis_name': 'soma_bouton_pipeline',
        'result_root': str(result_root),
        'cache_root': str(cache_root),
        'movie_expids': [],
        'sleep_expids': ['mouse1_sleep_2024-01-01'],
        'soma_channel': 1,
        'bouton_channel': 0,
        'rebuild': False,
        'plots_only': False,
        'poster_ready_only': False,
        'comparison_preset_name': 'default',
        'state_mode': 'sleep',
        'visual_response_cohort': 'all',
    }

    manifest = soma_pipeline.run_pipeline(config)
    assert manifest['loaded_from'] == 'rebuild'
    assert pairwise_cache_calls
    assert pairwise_cache_calls[0][1]['family_result_stage'] == 'pairwise_correlation'
    assert manifest['counts']['correlation_rows'] == len(cached_rows['correlation_rows'])
    assert manifest['counts']['soma_pairwise_correlation_rows'] == len(cached_rows['soma_pairwise_rows'])
    assert manifest['counts']['bouton_pairwise_correlation_rows'] == len(cached_rows['bouton_pairwise_rows'])


def test_day_grouping_uses_animal_and_date() -> None:
    assert make_day_id('Mouse A', '2024-01-05') == 'Mouse_A_2024-01-05'
    grouped = grouped_experiments_by_day(['2024-01-05_mouseA', '2024-01-05_mouseB', '2024-01-06_mouseA'])
    assert len(grouped) == 3
    assert grouped['mouseA_2024-01-05'] == ['2024-01-05_mouseA']
    assert grouped['mouseB_2024-01-05'] == ['2024-01-05_mouseB']
    assert grouped['mouseA_2024-01-06'] == ['2024-01-06_mouseA']


def test_repo_relative_paths_anchor_to_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    relative = 'results/soma_bouton_pipeline/cache/source_cache.npz'
    absolute = Path('/tmp/example/source_cache.npz')

    assert resolve_repo_path(relative, repo_root) == repo_root / relative
    assert resolve_repo_path(absolute, repo_root) == absolute


def test_analysis_family_registry_is_shared_across_pipeline_packages() -> None:
    expected = ['state', 'mixed_model']
    assert shared_normalize(expected, allowed_families=['state', 'mixed_model']) == expected
    assert main_normalize(expected) == expected
    assert soma_normalize(expected) == expected


def test_shared_comparison_preset_plan_includes_required_presets() -> None:
    presets = [
        ('movies_state_comparisons', {'movie': True}),
        ('blank_state_comparisons', {'blank': True}),
        ('all_requested_comparisons', {'all': True}),
        ('unused_preset', {'unused': True}),
    ]
    normal_plan = build_comparison_preset_batch_plan(
        presets,
        selected_names=['movies_state_comparisons'],
        poster_ready_only=False,
        poster_required_names=POSTER_REQUIRED_COMPARISON_PRESETS,
    )
    assert [name for name, _ in normal_plan.presets] == [
        'movies_state_comparisons',
        'blank_state_comparisons',
        'all_requested_comparisons',
    ]
    assert normal_plan.reference_preset_name == 'all_requested_comparisons'

    poster_plan = build_comparison_preset_batch_plan(
        presets,
        selected_names=['movies_state_comparisons'],
        poster_ready_only=True,
        poster_required_names=POSTER_REQUIRED_COMPARISON_PRESETS,
    )
    assert [name for name, _ in poster_plan.presets] == [
        'movies_state_comparisons',
        'blank_state_comparisons',
        'all_requested_comparisons',
    ]
    assert poster_plan.reference_preset_name == 'all_requested_comparisons'



def test_shared_comparison_preset_csv_loader_reads_sibling_results(tmp_path: Path) -> None:
    batch_root = tmp_path / 'results'
    csv_dir = batch_root / 'blank_state_comparisons' / 'csv'
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / 'state_activity_by_experiment.csv'
    csv_path.write_text('expid,state,mean\nexp1,sleep,1.5\n')

    rows = load_comparison_preset_csv_rows(batch_root, 'blank_state_comparisons', 'state_activity_by_experiment.csv')
    assert rows == [{'expid': 'exp1', 'state': 'sleep', 'mean': '1.5'}]



def test_state_correlation_plot_helper_supports_custom_labels(tmp_path: Path) -> None:
    rows = [
        {'state': 'sleep', 'state_display': 'Sleep', 'mean_corr': 0.5, 'day_id': 'mouse1_2024-01-01'},
        {'state': 'sleep', 'state_display': 'Sleep', 'mean_corr': 0.25, 'day_id': 'mouse1_2024-01-02'},
    ]
    outputs = plot_state_correlation(
        rows,
        tmp_path,
        cohort_label='soma_pairwise',
        title='Soma-soma correlation',
        output_stem='soma_pairwise_state_summary_boxplots_correlation',
    )
    assert outputs
    assert {path.name for path in outputs} == {
        'soma_pairwise_state_summary_boxplots_correlation.png',
        'soma_pairwise_state_summary_boxplots_correlation.svg',
    }
