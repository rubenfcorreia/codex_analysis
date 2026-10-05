from pathlib import Path
from types import SimpleNamespace
import csv
import json

import numpy as np

from analysis.shared.progression import run_dendrite_spine_progression, run_soma_bouton_progression


class _Bundle:
    def __init__(self, matrix, time):
        self._matrix = np.asarray(matrix, dtype=float)
        self.t = np.asarray(time, dtype=float)

    def matrix(self):
        return self._matrix


def test_soma_progression_uses_experiment_level_means(tmp_path: Path) -> None:
    contexts = [
        SimpleNamespace(
            expid="2026-01-01_01_A", animal_id="A", date="2026-01-01", mode="movie",
            state_bundle={"rows": [{"time": 0, "duration": 2, "F1_type": "movie", "F1_name": r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00000"}]},
            soma=_Bundle([[1, 3, 5, 7]], [0, 1, 2, 3]),
            bouton=_Bundle([[2, 4, 6, 8]], [0, 1, 2, 3]),
        ),
        SimpleNamespace(
            expid="2026-01-02_01_A", animal_id="A", date="2026-01-02", mode="sleep",
            state_bundle={}, soma=_Bundle([[4, 4]], [0, 1]), bouton=_Bundle([[6, 6]], [0, 1]),
        ),
    ]
    result = run_soma_bouton_progression(contexts, tmp_path / "general" / "progression", {"enabled": True, "blank_bin_s": 1})
    assert result["blank_rows"] == 4
    assert result["sleep_rows"] == 4
    assert (tmp_path / "general" / "progression" / "soma" / "figures" / "soma_progression.svg").exists()


def test_dendrite_progression_keeps_raw_and_specific_spine_means(tmp_path: Path) -> None:
    expid = "2026-01-01_01_A"
    obs = {"time": np.arange(3), "trace": np.array([1., 2., 3.])}
    spine_obs = {"time": np.arange(3), "trace": np.array([2., 3., 4.]), "spine_specific": np.array([.5, 1., 1.5])}
    cache = {
        "config": {"movie_expids": [expid]},
        "experiments": {expid: {"date": "2026-01-01", "trial_meta": [{"time": 0, "duration": 2, "F1_type": "movie", "F1_name": r"D:\bonsai_resources\all_movie_clips_bv_sets\007\00000"}]}},
        "animals": {"A": {"dendrites": {"d1": {"observations": {expid: obs}, "spines": {"s1": {"observations": {expid: spine_obs}}}}}}},
    }
    result = run_dendrite_spine_progression(cache, tmp_path / "general" / "progression", {"enabled": True})
    assert result["blank_rows"] == 6
    assert (tmp_path / "general" / "progression" / "dendrite" / "blank_trial_progression.csv").exists()
    assert (tmp_path / "general" / "progression" / "spine" / "figures" / "spine_progression.svg").exists()
    assert (tmp_path / "general" / "progression" / "spine" / "figures" / "spine_raw_progression.svg").exists()
    assert (tmp_path / "general" / "progression" / "spine" / "figures" / "spine_spine_specific_progression.svg").exists()


def test_sleep_progression_concatenates_same_day_and_truncates_to_shortest_day(tmp_path: Path) -> None:
    def context(expid, date, values, times):
        return SimpleNamespace(
            expid=expid, animal_id="A", date=date, mode="sleep", state_bundle={},
            soma=_Bundle([values], times), bouton=_Bundle([[value * 2 for value in values]], times),
        )

    contexts = [
        context("2026-01-01_02_A", "2026-01-01", [3, 3], [0, 1]),
        context("2026-01-01_01_A", "2026-01-01", [1, 1], [0, 1]),
        context("2026-01-02_01_A", "2026-01-02", [10, 10, 10], [0, 1, 2]),
    ]
    root = tmp_path / "general" / "progression"
    result = run_soma_bouton_progression(contexts, root, {"enabled": True, "sleep_time_step_s": 1})
    assert result["sleep_rows"] == 12
    with (root / "soma" / "sleep_expid_progression.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    day_one = [row for row in rows if row["date"] == "2026-01-01"]
    assert [float(row["elapsed_time_s"]) for row in day_one] == [0.0, 1.0, 2.0]
    assert json.loads(day_one[0]["source_expids"]) == ["2026-01-01_01_A", "2026-01-01_02_A"]
    assert json.loads(day_one[0]["segment_boundaries"])[1]["start_s"] == 1.0
