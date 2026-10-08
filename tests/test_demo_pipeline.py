import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from analysis.demo_pipeline.demo_pipeline import build_demo
from analysis.demo_pipeline.recipe_schema import default_recipe, validate_recipe
from analysis.demo_pipeline.validation import validate_demo


def test_demo_is_deterministic_and_validates(tmp_path):
    recipe = default_recipe()
    recipe["mode"] = "fast"
    recipe["replicates"] = 1
    recipe["analysis"]["generate_figures"] = False
    first = build_demo(recipe, tmp_path / "first")
    second = build_demo(recipe, tmp_path / "second")
    assert first == second
    first_trace = next((tmp_path / "first").rglob("recordings/s2p_ch0.pickle"))
    second_trace = next((tmp_path / "second").rglob("recordings/s2p_ch0.pickle"))
    with first_trace.open("rb") as handle:
        first_data = pickle.load(handle)
    with second_trace.open("rb") as handle:
        second_data = pickle.load(handle)
    assert np.array_equal(first_data["dF"], second_data["dF"])
    assert validate_demo(tmp_path / "first")["passed"]


def test_recipe_rejects_duplicate_states():
    recipe = default_recipe()
    recipe["states"] = ["nrem", "nrem"]
    with pytest.raises(ValueError, match="unique"):
        validate_recipe(recipe)


def test_recipe_rejects_invalid_topology():
    recipe = default_recipe()
    recipe["boutons"]["bouton_1"]["coupled_to"] = "missing_soma"
    with pytest.raises(ValueError, match="unknown soma"):
        validate_recipe(recipe)
