from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_STATES = ["active_awake", "quiet_awake", "nrem", "rem"]


def default_recipe() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "full",
        "states": list(DEFAULT_STATES),
        "replicates": 4,
        "seed": 12345,
        "duration_s": 120.0,
        "dt_s": 0.1,
        "shuffle_n": 20,
        "output_repo_subdir": "data/Repository",
        "dendrites": {
            "basal": {"activity_scale": 2.0, "frequency_scale": 2.0},
            "apical": {"activity_scale": 0.5, "frequency_scale": 0.5},
        },
        "somas": {
            "soma_1": {"activity_scale": 3.0, "frequency_scale": 3.0},
            "soma_2": {"activity_scale": 1.0 / 3.0, "frequency_scale": 1.0 / 3.0},
        },
        "boutons": {
            "bouton_1": {"coupled_to": "soma_1"},
            "bouton_2": {"coupled_to": None},
            "bouton_3": {"coupled_to": "soma_2"},
        },
        "trace_style": {
            "baseline": 0.05,
            "event_amplitude": 1.0,
            "event_width_s": 0.35,
            "noise": 0.0,
            "base_events_per_state": 3,
        },
        "analysis": {
            "run_dendrites": True,
            "run_soma_bouton": True,
            "generate_figures": True,
        },
    }


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_recipe(path: Path | str | None = None) -> Dict[str, Any]:
    recipe = default_recipe()
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("Demo recipe must be a JSON object")
        recipe = _merge(recipe, loaded)
    validate_recipe(recipe)
    return recipe


def validate_recipe(recipe: Dict[str, Any]) -> None:
    states = recipe.get("states")
    if not isinstance(states, list) or not states or len(set(states)) != len(states):
        raise ValueError("states must be a non-empty list of unique labels")
    if int(recipe.get("replicates", 0)) <= 0:
        raise ValueError("replicates must be positive")
    if float(recipe.get("duration_s", 0)) <= 0 or float(recipe.get("dt_s", 0)) <= 0:
        raise ValueError("duration_s and dt_s must be positive")
    for compartment in ("basal", "apical"):
        values = recipe.get("dendrites", {}).get(compartment, {})
        for field in ("activity_scale", "frequency_scale"):
            if float(values.get(field, 0)) <= 0:
                raise ValueError(f"dendrites.{compartment}.{field} must be positive")
    somas = recipe.get("somas", {})
    if set(somas) != {"soma_1", "soma_2"}:
        raise ValueError("somas must contain exactly soma_1 and soma_2")
    for name, values in somas.items():
        for field in ("activity_scale", "frequency_scale"):
            if float(values.get(field, 0)) <= 0:
                raise ValueError(f"somas.{name}.{field} must be positive")
    boutons = recipe.get("boutons", {})
    if set(boutons) != {"bouton_1", "bouton_2", "bouton_3"}:
        raise ValueError("boutons must contain exactly bouton_1, bouton_2, and bouton_3")
    for name, values in boutons.items():
        target = values.get("coupled_to")
        if target is not None and target not in somas:
            raise ValueError(f"{name}.coupled_to refers to unknown soma {target!r}")


def recipe_hash(recipe: Dict[str, Any]) -> str:
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def resolved_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    validate_recipe(recipe)
    result = copy.deepcopy(recipe)
    result["recipe_hash"] = recipe_hash(recipe)
    return result
