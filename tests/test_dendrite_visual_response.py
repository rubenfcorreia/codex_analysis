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


def _make_spine_observation(compartment: str, movie_value: float, blank_value: float) -> dict:
    time = np.linspace(0.0, 19.0, 20, dtype=float)
    trace = np.zeros_like(time)
    trace[4:7] = 1.0
    return {
        "compartment": compartment,
        "day_id": None,
        "exp_id": None,
        "time": time,
        "trace": trace,
        "trace_hp": trace,
        "spine_specific": trace,
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

    dendrite_specs = {
        "basal_responsive": {
            "basal_exp_1": (1.00, 0.10),
            "basal_exp_2": (1.10, 0.15),
            "basal_exp_3": (0.95, 0.20),
        },
        "basal_nonresponsive": {
            "basal_exp_1": (0.20, 0.60),
            "basal_exp_2": (0.25, 0.55),
            "basal_exp_3": (0.15, 0.65),
        },
        "apical_responsive": {
            "apical_exp_1": (1.20, 0.05),
            "apical_exp_2": (1.30, 0.10),
            "apical_exp_3": (1.10, 0.08),
        },
        "apical_nonresponsive": {
            "apical_exp_1": (0.30, 0.70),
            "apical_exp_2": (0.25, 0.60),
            "apical_exp_3": (0.35, 0.80),
        },
    }
    for dendrite_id, dendrite_observations in dendrite_specs.items():
        animal_dendrite = animals["animal_1"]["dendrites"][dendrite_id]
        for exp_id, (movie_value, blank_value) in dendrite_observations.items():
            d_obs = _make_observation("basal" if dendrite_id.startswith("basal") else "apical", movie_value, blank_value)
            spine_id = f"{dendrite_id}|{exp_id}|s1"
            d_obs["spine_ids"] = [spine_id]
            animal_dendrite["observations"][exp_id] = d_obs
            animal_dendrite["spines"][spine_id] = {
                "observations": {
                    exp_id: _make_spine_observation("basal" if dendrite_id.startswith("basal") else "apical", movie_value, blank_value)
                }
            }
            animal_dendrite["spines"][spine_id]["observations"][exp_id]["day_id"] = exp_id
            animal_dendrite["spines"][spine_id]["observations"][exp_id]["exp_id"] = exp_id

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


def test_event_info_records_parallel_detection_methods_for_dendrite_and_spine() -> None:
    time = np.arange(30, dtype=float)
    dendrite_trace = np.concatenate([
        np.zeros(8, dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.zeros(3, dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.zeros(13, dtype=float),
    ])
    spine_trace = np.concatenate([
        np.zeros(8, dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.zeros(4, dtype=float),
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.zeros(12, dtype=float),
    ])

    dendrite_event_info = pipeline.build_event_info(dendrite_trace, time)
    spine_event_info = pipeline.annotate_spine_event_info(pipeline.build_event_info(spine_trace, time), dendrite_event_info)

    for event_info in [dendrite_event_info, spine_event_info]:
        assert event_info["method"] == "derivative"
        assert event_info["primary_method"] == "derivative"
        assert set(event_info["event_detection_methods"]) == {"amplitude", "derivative"}
        assert set(event_info["methods"].keys()) == {"amplitude", "derivative"}
        assert event_info["methods"]["derivative"]["method"] == "derivative"
        assert event_info["methods"]["amplitude"]["method"] == "amplitude"
        assert np.isfinite(event_info["threshold"])
        assert np.isfinite(event_info["methods"]["derivative"]["threshold"])
        assert np.isfinite(event_info["methods"]["amplitude"]["threshold"])

    assert np.isfinite(spine_event_info["coincident_event_frequency_per_min"])
    assert np.isfinite(spine_event_info["methods"]["derivative"]["coincident_event_frequency_per_min"])
    assert np.isfinite(spine_event_info["methods"]["amplitude"]["coincident_event_frequency_per_min"])


def test_state_summary_outputs_use_family_and_cohort_subfolders(tmp_path: Path) -> None:
    cache = _build_cache()
    response_summary = pipeline.classify_visual_responsive_dendrites(cache)
    output_dir = pipeline.state_summary_figure_dir(tmp_path / "figures")

    overview_results = pipeline.build_state_summary_gallery_results(cache, STATE_LABELS, None)
    overview_path = pipeline.plot_state_summary_figure(
        overview_results,
        output_dir,
        output_name="state_summary_boxplots.svg",
        title="Selected-state summary distributions - All compartments",
        state_labels=STATE_LABELS,
        cohort_label="all",
    )
    assert overview_path is not None
    assert Path(overview_path).parent == output_dir / "dendrites" / "selected_states" / "all"
    assert (output_dir / "dendrites" / "selected_states" / "all" / "state_summary_boxplots_dendrite_mean.svg").exists()
    assert (output_dir / "spines" / "selected_states" / "all" / "state_summary_boxplots_spine_specific_mean.svg").exists()

    responsive_results = pipeline.build_state_summary_gallery_results(
        cache,
        STATE_LABELS,
        compartment_filter="basal",
        dendrite_ids_filter=response_summary["cohort_ids"]["basal"]["responsive"],
    )
    responsive_path = pipeline.plot_state_summary_figure(
        responsive_results,
        output_dir,
        output_name="state_summary_boxplots_basal_responsive.svg",
        title="Selected-state summary distributions - Basal responsive",
        state_labels=STATE_LABELS,
        cohort_label="responsive",
    )
    assert responsive_path is not None
    assert Path(responsive_path).parent == output_dir / "dendrites" / "selected_states" / "responsive"
    assert (output_dir / "dendrites" / "selected_states" / "responsive" / "state_summary_boxplots_basal_responsive_dendrite_mean.svg").exists()
    assert (output_dir / "spines" / "selected_states" / "responsive" / "state_summary_boxplots_basal_responsive_spine_specific_mean.svg").exists()
