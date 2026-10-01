from pathlib import Path
from types import SimpleNamespace

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
            state_bundle={"rows": [{"state_label": "blank", "start": 0, "duration": 2}]},
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
    assert result["sleep_rows"] == 2
    assert (tmp_path / "general" / "progression" / "soma" / "figures" / "soma_progression.svg").exists()


def test_dendrite_progression_keeps_raw_and_specific_spine_means(tmp_path: Path) -> None:
    expid = "2026-01-01_01_A"
    obs = {"time": np.arange(3), "trace": np.array([1., 2., 3.])}
    spine_obs = {"time": np.arange(3), "trace": np.array([2., 3., 4.]), "spine_specific": np.array([.5, 1., 1.5])}
    cache = {
        "config": {"movie_expids": [expid]},
        "experiments": {expid: {"date": "2026-01-01", "trial_meta": [{"state_label": "blank", "start": 0, "end": 2}]}},
        "animals": {"A": {"dendrites": {"d1": {"observations": {expid: obs}, "spines": {"s1": {"observations": {expid: spine_obs}}}}}}},
    }
    result = run_dendrite_spine_progression(cache, tmp_path / "general" / "progression", {"enabled": True})
    assert result["blank_rows"] == 6
    assert (tmp_path / "general" / "progression" / "dendrite" / "blank_trial_progression.csv").exists()
    assert (tmp_path / "general" / "progression" / "spine" / "figures" / "spine_progression.svg").exists()
