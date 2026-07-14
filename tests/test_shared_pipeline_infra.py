from __future__ import annotations

from pathlib import Path

from analysis.dendrites_pipeline.analysis_families import normalize_analysis_families as main_normalize
from analysis.shared.analysis_families.registry import normalize_analysis_families as shared_normalize
from analysis.shared.cache_utils import analysis_day_cache_path, analysis_results_cache_path, analysis_table_cache_path, build_shared_shuffle_cache_key, family_results_cache_index, shared_shuffle_cache_path
from analysis.shared.state_utils import grouped_experiments_by_day, make_day_id, resolve_repo_path
from analysis.soma_bouton_pipeline.analysis_families import normalize_analysis_families as soma_normalize


def test_cache_paths_are_stage_scoped() -> None:
    base = Path('/tmp/example/dendrites_cache.npz')
    assert analysis_table_cache_path(base).name == 'dendrites_cache_analysis_tables_cache.npz'
    assert analysis_results_cache_path(base).name == 'dendrites_cache_analysis_results_cache.npz'
    assert analysis_day_cache_path(base).name == 'dendrites_cache_analysis_day_cache.npz'
    assert shared_shuffle_cache_path(base).name == 'dendrites_cache_shuffle_cache.npz'
    index = family_results_cache_index(base)
    assert index['state'].endswith('_state_results_cache.npz')
    assert index['mixed_model'].endswith('_mixed_model_results_cache.npz')


def test_shuffle_cache_keys_scope_roi_ids_by_day() -> None:
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
