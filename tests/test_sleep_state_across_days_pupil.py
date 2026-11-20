from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.sleep_state_across_days import sleep_state_across_days as sleepmod


def test_movie_trial_pupil_splits_blank_and_movie(monkeypatch, tmp_path):
    summary = SimpleNamespace(exp_ids=["2026-01-01_01_MOUSE001"], animal_id="MOUSE001", category="movie")
    trial_rows = [
        {"exp_id": summary.exp_ids[0], "animal_id": "MOUSE001", "date": "2026-01-01", "category": "movie", "trial_index": 0, "trial_category": "blank", "video_id": "00000", "trial_time_s": 0.0, "trial_end_s": 2.0},
        {"exp_id": summary.exp_ids[0], "animal_id": "MOUSE001", "date": "2026-01-01", "category": "movie", "trial_index": 1, "trial_category": "movies", "video_id": "00001", "trial_time_s": 2.0, "trial_end_s": 4.0},
        {"exp_id": summary.exp_ids[0], "animal_id": "MOUSE001", "date": "2026-01-01", "category": "movie", "trial_index": 2, "trial_category": "zebra", "video_id": "00002", "trial_time_s": 4.0, "trial_end_s": 5.0},
    ]
    bundle = {"state_10hz_t": np.arange(5, dtype=float), "state_10hz": np.zeros(5, dtype=int)}
    monkeypatch.setattr(sleepmod, "resolve_repo_root", lambda repo_base, animal_id, exp_id: tmp_path)
    monkeypatch.setattr(sleepmod, "load_sleep_state_bundle", lambda exp_root, cache=None: (bundle, tmp_path / "sleep_state.pickle", None))
    monkeypatch.setattr(sleepmod, "load_session_pupil_series", lambda exp_root, category, cache=None: (np.arange(5, dtype=float), np.arange(1, 6, dtype=float), "eye.pickle"))

    rows, pooled, checks = sleepmod.summarize_movie_trial_pupil(
        [summary], trial_rows, tmp_path, animal_pupil_normalization={"MOUSE001": {"center": 0.0, "scale": 1.0}}
    )

    assert [row["trial_category"] for row in rows] == ["blank", "movies"]
    assert [row["pupil_sample_count"] for row in rows] == [2, 2]
    assert np.allclose([row["pupil_mean_diameter"] for row in rows], [1.5, 3.5])
    assert np.allclose([row["pupil_mean_z"] for row in rows], [1.5, 3.5])
    assert {row["trial_category"]: row["n_trials"] for row in pooled} == {"blank": 1, "movies": 1}
    assert checks["n_blank_trials_with_pupil"] == 1
    assert checks["n_movie_trials_with_pupil"] == 1
