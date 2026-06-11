from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.main_pipeline import sleep_dendrite_spine_pipeline as pipeline


STATE_LABELS = ["quiet_awake_movies", "quiet_awake_blank"]
METRIC_KEYS = {
    "dendrite_mean",
    "spine_specific_mean",
    "dendrite_event_frequency_per_min",
    "spine_event_frequency_per_min",
    "coincident_event_frequency_per_min",
    "noncoincident_event_frequency_per_min",
}


def _make_observation(compartment: str, movie_value: float, blank_value: float) -> dict:
    time = np.linspace(0.0, 19.0, 20, dtype=float)
    trace = np.zeros_like(time)
    return {
        "compartment": compartment,
        "time": time,
        "trace": trace,
        "trace_hp": trace,
        "spine_ids": [],
        "cut_state_means": {
            "quiet_awake_movies": float(movie_value),
            "quiet_awake_blank": float(blank_value),
        },
        "event_info": pipeline.build_event_info(trace, time),
    }


def _build_cache() -> dict:
    experiments: dict = {}
    animals = {
        "animal_1": {
            "dendrites": {
                "basal_responsive": {"observations": {}, "spines": {}},
                "basal_nonresponsive": {"observations": {}, "spines": {}},
                "apical_responsive": {"observations": {}, "spines": {}},
                "apical_nonresponsive": {"observations": {}, "spines": {}},
            }
        }
    }
    experiment_specs = [
        ("basal_exp_1", "basal"),
        ("basal_exp_2", "basal"),
        ("basal_exp_3", "basal"),
        ("apical_exp_1", "apical"),
        ("apical_exp_2", "apical"),
        ("apical_exp_3", "apical"),
    ]
    for exp_id, compartment in experiment_specs:
        experiments[exp_id] = {
            "compartment": compartment,
            "state_masks": {
                "quiet_awake_movies": np.ones(20, dtype=bool),
                "quiet_awake_blank": np.ones(20, dtype=bool),
            },
        }

    observations = {
        "basal_responsive": {
            "basal_exp_1": _make_observation("basal", 1.00, 0.10),
            "basal_exp_2": _make_observation("basal", 1.10, 0.15),
            "basal_exp_3": _make_observation("basal", 0.95, 0.20),
        },
        "basal_nonresponsive": {
            "basal_exp_1": _make_observation("basal", 0.20, 0.60),
            "basal_exp_2": _make_observation("basal", 0.25, 0.55),
            "basal_exp_3": _make_observation("basal", 0.15, 0.65),
        },
        "apical_responsive": {
            "apical_exp_1": _make_observation("apical", 1.20, 0.05),
            "apical_exp_2": _make_observation("apical", 1.30, 0.10),
            "apical_exp_3": _make_observation("apical", 1.10, 0.08),
        },
        "apical_nonresponsive": {
            "apical_exp_1": _make_observation("apical", 0.30, 0.70),
            "apical_exp_2": _make_observation("apical", 0.25, 0.60),
            "apical_exp_3": _make_observation("apical", 0.35, 0.80),
        },
    }
    for dendrite_id, dendrite_observations in observations.items():
        animal_dendrite = animals["animal_1"]["dendrites"][dendrite_id]
        animal_dendrite["observations"].update(dendrite_observations)
        for obs in dendrite_observations.values():
            obs["spine_ids"] = []

    return {
        "animals": animals,
        "experiments": experiments,
        "config": {},
        "alerts": [],
    }


def test_visual_response_classification_and_filtered_summaries() -> None:
    cache = _build_cache()
    response_summary = pipeline.classify_visual_responsive_dendrites(cache)

    row_lookup = {row["global_dendrite_id"]: row for row in response_summary["rows"]}
    assert row_lookup["basal_responsive"]["responsive"] is True
    assert row_lookup["basal_nonresponsive"]["responsive"] is False
    assert row_lookup["apical_responsive"]["responsive"] is True
    assert row_lookup["apical_nonresponsive"]["responsive"] is False
    assert response_summary["cohort_ids"]["basal"]["responsive"] == ["basal_responsive"]
    assert response_summary["cohort_ids"]["basal"]["nonresponsive"] == ["basal_nonresponsive"]
    assert response_summary["cohort_ids"]["apical"]["responsive"] == ["apical_responsive"]
    assert response_summary["cohort_ids"]["apical"]["nonresponsive"] == ["apical_nonresponsive"]

    responsive_basal = pipeline.build_state_summary_gallery_results(
        cache,
        STATE_LABELS,
        compartment_filter="basal",
        dendrite_ids_filter=response_summary["cohort_ids"]["basal"]["responsive"],
    )
    nonresponsive_basal = pipeline.build_state_summary_gallery_results(
        cache,
        STATE_LABELS,
        compartment_filter="basal",
        dendrite_ids_filter=response_summary["cohort_ids"]["basal"]["nonresponsive"],
    )
    responsive_apical = pipeline.build_state_summary_gallery_results(
        cache,
        STATE_LABELS,
        compartment_filter="apical",
        dendrite_ids_filter=response_summary["cohort_ids"]["apical"]["responsive"],
    )
    nonresponsive_apical = pipeline.build_state_summary_gallery_results(
        cache,
        STATE_LABELS,
        compartment_filter="apical",
        dendrite_ids_filter=response_summary["cohort_ids"]["apical"]["nonresponsive"],
    )

    for bundle in [responsive_basal, nonresponsive_basal, responsive_apical, nonresponsive_apical]:
        assert set(bundle["state_summaries"].keys()) == METRIC_KEYS
        assert set(bundle["state_dendrite_summaries"].keys()) == METRIC_KEYS

    assert set(responsive_basal["state_dendrite_summaries"]["dendrite_mean"]["quiet_awake_movies"].keys()) == {"basal_responsive"}
    assert set(nonresponsive_basal["state_dendrite_summaries"]["dendrite_mean"]["quiet_awake_movies"].keys()) == {"basal_nonresponsive"}
    assert set(responsive_apical["state_dendrite_summaries"]["dendrite_mean"]["quiet_awake_movies"].keys()) == {"apical_responsive"}
    assert set(nonresponsive_apical["state_dendrite_summaries"]["dendrite_mean"]["quiet_awake_movies"].keys()) == {"apical_nonresponsive"}

    assert responsive_basal["state_summaries"]["dendrite_mean"]["quiet_awake_movies"]
    assert nonresponsive_basal["state_summaries"]["dendrite_mean"]["quiet_awake_movies"]
    assert responsive_apical["state_summaries"]["dendrite_mean"]["quiet_awake_movies"]
    assert nonresponsive_apical["state_summaries"]["dendrite_mean"]["quiet_awake_movies"]
