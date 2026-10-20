from pathlib import Path

import numpy as np

from analysis.shared.analysis_families.correlation import correlation_analysis_for_batch
from analysis.shared.shuffle_dataset import ShuffleDataset, build_shuffle_dataset_once


def test_persisted_shuffle_dataset_has_independent_reproducible_replicates(tmp_path: Path):
    path = tmp_path / "shuffle.h5"
    metadata = {"pipeline": "test", "source_signature": "source", "shuffle_n": 6}
    build_shuffle_dataset_once(path, metadata, {"left": np.arange(24.0), "right": np.roll(np.arange(24.0), 3)}, shuffle_n=6, seed=7, pipeline="test", preset_count=2)
    cache = {"shuffle_dataset_path": str(path), "shuffle_dataset_metadata": {}}
    rows = correlation_analysis_for_batch(
        [(np.arange(24.0), np.roll(np.arange(24.0), 3))],
        6,
        shared_shuffle_cache=cache,
        shared_shuffle_keys=["pair"],
        signal_keys=[("left", "right")],
        result_keys=["pair-state"],
        correlation_inference="circular_shift",
        max_batch_size=1,
        shift_block_size=2,
    )
    assert rows[0]["shuffle_n_success"] == 6
    with ShuffleDataset(path, {}, readonly=True) as dataset:
        left = dataset.read_signal("left")
        right = dataset.read_signal("right")
        assert left.shape == (6, 24)
        assert right.shape == (6, 24)
        assert not np.array_equal(left[0], right[0])


def test_shuffle_dataset_metadata_mismatch_is_rejected(tmp_path: Path):
    path = tmp_path / "shuffle.h5"
    metadata = {"pipeline": "test", "source_signature": "source", "shuffle_n": 3}
    build_shuffle_dataset_once(path, metadata, {"roi": np.arange(10.0)}, shuffle_n=3, seed=7, pipeline="test", preset_count=1)
    try:
        ShuffleDataset(path, {**metadata, "source_signature": "changed"})
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("stale shuffle metadata must be rejected")


def test_pipeline_dataset_builds_once_and_logs_reuse(tmp_path: Path):
    path = tmp_path / "pipeline.h5"
    logs = []
    signals = {"animal|day|soma|ch1|0": np.arange(12.0), "animal|day|bouton|ch0|2": np.arange(12.0) + 1}
    first = build_shuffle_dataset_once(path, {"source": "x"}, signals, shuffle_n=4, seed=11, pipeline="test", preset_count=3, log=logs.append)
    second = build_shuffle_dataset_once(path, {"source": "x"}, signals, shuffle_n=4, seed=11, pipeline="test", preset_count=3, log=logs.append)
    assert first["status"] == "built"
    assert second["status"] == "reused"
    assert first["sha256"] == second["sha256"]
    assert any("build start | pipeline=test | presets=3" in line for line in logs)
    assert any("source inventory | signals=2 | unique_keys=2 | duplicate_keys=0" in line for line in logs)
    assert any("build complete" in line for line in logs)
    with ShuffleDataset(path, {}, readonly=True) as dataset:
        assert set(dataset.signal_keys()) == set(signals)
        assert all(dataset.read_signal(key).shape == (4, 12) for key in signals)
        assert not any(any(token in key for token in ("preset", "state", "comparison", "cohort", "pair")) for key in dataset.signal_keys())


def test_stale_signal_inventory_is_rebuilt(tmp_path: Path):
    path = tmp_path / "shuffle.h5"
    signals = {"left": np.arange(12.0), "right": np.arange(12.0) + 1}
    first = build_shuffle_dataset_once(path, {"source": "v1"}, signals, shuffle_n=4, seed=7, pipeline="test", preset_count=1)
    changed = {"left": np.arange(12.0) + 10, "right": signals["right"], "new": np.arange(12.0) + 2}
    second = build_shuffle_dataset_once(path, {"source": "v2"}, changed, shuffle_n=3, seed=9, pipeline="test", preset_count=1)
    assert first["status"] == "built"
    assert second["status"] == "built"
    with ShuffleDataset(path, {}, readonly=True) as dataset:
        assert set(dataset.signal_keys()) == set(changed)
        assert dataset.read_signal("left").shape == (3, 12)


def test_missing_requested_signal_fails_before_correlation(tmp_path: Path):
    path = tmp_path / "shuffle.h5"
    build_shuffle_dataset_once(path, {"source": "x"}, {"left": np.arange(12.0), "right": np.arange(12.0)}, shuffle_n=4, seed=7, pipeline="test", preset_count=1)
    cache = {"shuffle_dataset_path": str(path), "shuffle_dataset_metadata": {}}
    try:
        correlation_analysis_for_batch(
            [(np.arange(12.0), np.arange(12.0))], 4, shared_shuffle_cache=cache,
            shared_shuffle_keys=["pair"], signal_keys=[("missing", "right")],
            result_keys=["pair-state"], correlation_inference="circular_shift",
            null_model="full_trace",
        )
    except KeyError as exc:
        message = str(exc)
        assert "missing" in message and str(path) in message and "rebuild_dataset" in message
    else:
        raise AssertionError("missing shuffle signals must fail before correlation")


def test_replicate_metadata_mismatch_is_rebuilt(tmp_path: Path):
    path = tmp_path / "shuffle.h5"
    signals = {"roi": np.arange(12.0)}
    build_shuffle_dataset_once(path, {"source": "x"}, signals, shuffle_n=4, seed=7, pipeline="test", preset_count=1)
    import h5py
    with h5py.File(path, "a") as handle:
        handle["signals"][next(iter(handle["signals"]))].attrs["shuffle_n"] = 999
    rebuilt = build_shuffle_dataset_once(path, {"source": "x"}, signals, shuffle_n=4, seed=7, pipeline="test", preset_count=1)
    assert rebuilt["status"] == "built"
    with ShuffleDataset(path, {}, readonly=True) as dataset:
        assert dataset.read_signal("roi").shape == (4, 12)


def test_grouped_signal_keys_are_flattened_for_validation(tmp_path: Path):
    path = tmp_path / "shuffle.h5"
    build_shuffle_dataset_once(path, {"source": "x"}, {"left": np.arange(12.0), "right": np.arange(12.0)}, shuffle_n=4, seed=7, pipeline="test", preset_count=1)
    cache = {"shuffle_dataset_path": str(path), "shuffle_dataset_metadata": {}}
    rows = correlation_analysis_for_batch(
        [(np.arange(12.0), np.arange(12.0))], 4, shared_shuffle_cache=cache,
        shared_shuffle_keys=["pair"], signal_keys=[(("left",), ("right",))],
        result_keys=["pair-state"], correlation_inference="circular_shift",
    )
    assert rows[0]["shuffle_n_success"] == 4


def test_mask_aware_cached_matches_legacy_for_noncontiguous_mask(tmp_path: Path):
    path = tmp_path / "full_trace.h5"
    a = np.sin(np.arange(40.0) / 3.0)
    b = np.roll(a, 4) + 0.1 * np.cos(np.arange(40.0))
    mask = np.array([(index % 5) < 2 for index in range(a.size)], dtype=bool)
    build_shuffle_dataset_once(
        path, {"source": "test"}, {"a": a, "b": b}, shuffle_n=10,
        seed=7, pipeline="test", preset_count=1, log=lambda _: None,
    )
    common = dict(
        pairs=[(a[mask], b[mask])],
        shuffle_n=10,
        shared_shuffle_keys=["pair"],
        signal_keys=[("a", "b")],
        signal_masks=[(mask, mask)],
        result_keys=["pair-state"],
        correlation_inference="circular_shift",
        shuffle_seed=123,
        max_batch_size=1,
        shift_block_size=2,
    )
    legacy_cache = {"null_model": "mask_aware", "entries": {}, "null_results": {}}
    cached_cache = {
        "null_model": "mask_aware",
        "entries": {},
        "null_results": {},
        "shuffle_dataset_path": str(path),
        "shuffle_dataset_metadata": {},
    }
    legacy = correlation_analysis_for_batch(shared_shuffle_cache=legacy_cache, **common)[0]
    cached = correlation_analysis_for_batch(shared_shuffle_cache=cached_cache, **common)[0]
    assert np.isclose(cached["r"], legacy["r"], equal_nan=True)
    assert np.isclose(cached["shuffle_p"], legacy["shuffle_p"], equal_nan=True)
    assert cached["shuffle_n_success"] == legacy["shuffle_n_success"] == 10


def test_full_trace_model_is_explicit_and_can_differ_from_mask_aware(tmp_path: Path):
    path = tmp_path / "full_trace.h5"
    a = np.sin(np.arange(40.0) / 3.0)
    b = np.roll(a, 4) + 0.1 * np.cos(np.arange(40.0))
    mask = np.array([(index % 5) < 2 for index in range(a.size)], dtype=bool)
    build_shuffle_dataset_once(
        path, {"source": "test"}, {"a": a, "b": b}, shuffle_n=10,
        seed=7, pipeline="test", preset_count=1, log=lambda _: None,
    )
    common = dict(
        pairs=[(a[mask], b[mask])],
        shuffle_n=10,
        shared_shuffle_keys=["pair"],
        signal_keys=[("a", "b")],
        signal_masks=[(mask, mask)],
        result_keys=["pair-state"],
        correlation_inference="circular_shift",
        shuffle_seed=3,
        max_batch_size=1,
        shift_block_size=2,
    )
    mask_aware = correlation_analysis_for_batch(
        shared_shuffle_cache={"null_model": "mask_aware", "entries": {}, "null_results": {}, "shuffle_dataset_path": str(path), "shuffle_dataset_metadata": {}},
        null_model="mask_aware", **common,
    )[0]
    full_trace = correlation_analysis_for_batch(
        shared_shuffle_cache={"null_model": "full_trace", "entries": {}, "null_results": {}, "shuffle_dataset_path": str(path), "shuffle_dataset_metadata": {}},
        null_model="full_trace", **common,
    )[0]
    assert mask_aware["null_model"] == "mask_aware"
    assert full_trace["null_model"] == "full_trace"
    assert not np.isclose(mask_aware["shuffle_p"], full_trace["shuffle_p"])
