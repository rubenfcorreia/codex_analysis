from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.main_pipeline import sleep_dendrite_spine_pipeline as pipeline


def _make_trace(event_windows: list[tuple[int, int]], n: int = 24, baseline: float = 0.0, peak: float = 1.0) -> np.ndarray:
    trace = np.full(n, baseline, dtype=float)
    for start, end in event_windows:
        trace[start:end] = peak
    return trace


def _make_event_info(time: np.ndarray, runs: list[tuple[int, int]], threshold: float = 0.5) -> dict:
    duration_seconds = float(time[-1] - time[0] + 1.0) if time.size else float("nan")
    event_count = int(len(runs))
    return {
        "noise_std": 1.0,
        "threshold": float(threshold),
        "event_count": event_count,
        "event_runs": [(int(start), int(end)) for start, end in runs],
        "event_frequency_per_min": pipeline.event_frequency_per_minute(event_count, duration_seconds) if event_count else float("nan"),
        "active": bool(event_count >= 3),
        "duration_seconds": duration_seconds,
        "min_consecutive_frames": 3,
        "sigma_factor": 3.0,
    }


def _build_cache() -> dict:
    time = np.arange(24, dtype=float)
    dendrite_runs = [(4, 7), (14, 17)]
    spine_runs = [(4, 7), (10, 13)]
    dendrite_trace = _make_trace(dendrite_runs, n=time.size, peak=1.0)
    spine_trace = _make_trace(spine_runs, n=time.size, peak=1.0)
    dendrite_event_info = _make_event_info(time, dendrite_runs)
    spine_event_info = pipeline.annotate_spine_event_info(_make_event_info(time, spine_runs), dendrite_event_info)

    valid_day_id = "animal_a|2024-06-01|basal"
    invalid_day_id = "animal_a|2024-06-02|basal"

    animals = {
        "animal_a": {
            "dendrites": {
                "animal_a|2024-06-01|basal|d1": {
                    "observations": {
                        valid_day_id: {
                            "compartment": "basal",
                            "day_id": valid_day_id,
                            "exp_id": valid_day_id,
                            "time": time,
                            "trace": dendrite_trace,
                            "event_info": dendrite_event_info,
                            "spine_ids": ["animal_a|2024-06-01|basal|d1|s1"],
                        }
                    },
                    "spines": {
                        "animal_a|2024-06-01|basal|d1|s1": {
                            "observations": {
                                valid_day_id: {
                                    "compartment": "basal",
                                    "day_id": valid_day_id,
                                    "exp_id": valid_day_id,
                                    "time": time,
                                    "trace": spine_trace,
                                    "trace_hp": spine_trace,
                                    "spine_specific": spine_trace,
                                    "event_info": spine_event_info,
                                    "dendrite_event_info": dendrite_event_info,
                                }
                            }
                        }
                    },
                },
                "animal_a|2024-06-02|basal|d2": {
                    "observations": {
                        invalid_day_id: {
                            "compartment": "basal",
                            "day_id": invalid_day_id,
                            "exp_id": invalid_day_id,
                            "time": np.array([], dtype=float),
                            "trace": np.array([], dtype=float),
                            "event_info": pipeline.build_event_info(np.array([], dtype=float), np.array([], dtype=float)),
                            "spine_ids": ["animal_a|2024-06-02|basal|d2|s1"],
                        }
                    },
                    "spines": {
                        "animal_a|2024-06-02|basal|d2|s1": {
                            "observations": {
                                invalid_day_id: {
                                    "compartment": "basal",
                                    "day_id": invalid_day_id,
                                    "exp_id": invalid_day_id,
                                    "time": np.array([], dtype=float),
                                    "trace": np.array([], dtype=float),
                                    "trace_hp": np.array([], dtype=float),
                                    "spine_specific": np.array([], dtype=float),
                                    "event_info": pipeline.build_event_info(np.array([], dtype=float), np.array([], dtype=float)),
                                    "dendrite_event_info": pipeline.build_event_info(np.array([], dtype=float), np.array([], dtype=float)),
                                }
                            }
                        }
                    },
                },
            }
        }
    }
    return {"animals": animals, "experiments": {}, "config": {}, "alerts": []}


def _all_axis_text(fig: object) -> list[str]:
    return [text.get_text() for ax in fig.axes for text in ax.texts]


def _all_line_labels(fig: object) -> list[str]:
    return [line.get_label() for ax in fig.axes for line in ax.lines]


def _count_threshold_lines(fig: object, threshold: float) -> int:
    count = 0
    for ax in fig.axes:
        if any(np.allclose(line.get_ydata(), threshold) for line in ax.lines):
            count += 1
    return count


def test_event_detection_example_figure_draws_annotated_contact_sheet() -> None:
    time = np.arange(24, dtype=float)
    trace = _make_trace([(4, 7), (14, 17)], n=time.size, peak=1.0)
    event_info = _make_event_info(time, [(4, 7), (14, 17)])

    fig = pipeline._build_event_detection_example_figure(
        time=time,
        trace=trace,
        event_info=event_info,
        title="Dendrite example",
        trace_label="Dendrite dF/F",
        trace_kind="dendrite",
    )
    assert fig is not None
    assert len(fig.axes) == 10

    threshold = float(event_info["threshold"])
    assert _count_threshold_lines(fig, threshold) == 10
    assert any("event" in text.lower() for text in _all_axis_text(fig))
    assert "Dendrite dF/F" in _all_line_labels(fig)

    if pipeline.plt is not None:
        pipeline.plt.close(fig)


def test_spine_example_figure_includes_dendrite_context_and_coincidence_marker() -> None:
    time = np.arange(24, dtype=float)
    dendrite_runs = [(4, 7), (14, 17)]
    spine_runs = [(4, 7), (10, 13)]
    dendrite_trace = _make_trace(dendrite_runs, n=time.size, peak=1.0)
    spine_trace = _make_trace(spine_runs, n=time.size, peak=1.0)
    dendrite_event_info = _make_event_info(time, dendrite_runs)
    spine_event_info = pipeline.annotate_spine_event_info(_make_event_info(time, spine_runs), dendrite_event_info)

    fig = pipeline._build_event_detection_example_figure(
        time=time,
        trace=spine_trace,
        event_info=spine_event_info,
        title="Spine example",
        trace_label="Spine-specific dF/F",
        trace_kind="spine",
        dendrite_event_info=dendrite_event_info,
        dendrite_trace=dendrite_trace,
        dendrite_time=time,
    )
    assert fig is not None
    assert len(fig.axes) == 10

    threshold = float(spine_event_info["threshold"])
    assert _count_threshold_lines(fig, threshold) == 10
    assert "Spine-specific dF/F" in _all_line_labels(fig)
    assert "Dendrite dF/F" in _all_line_labels(fig)
    assert any("coincident" in text.lower() for text in _all_axis_text(fig))
    assert any("noncoincident" in text.lower() for text in _all_axis_text(fig))

    if pipeline.plt is not None:
        pipeline.plt.close(fig)


def test_event_detection_gallery_writes_per_roi_outputs_and_skips_invalid(tmp_path: Path) -> None:
    cache = _build_cache()
    figure_root = tmp_path / "figures"

    saved = pipeline.generate_event_detection_example_gallery(cache, figure_root)

    assert len(saved) == 2
    saved_paths = [Path(path) for path in saved]
    assert all(path.exists() for path in saved_paths)
    assert all("event_examples" in path.parts for path in saved_paths)
    assert all("animal_a" in path.parts for path in saved_paths)
    assert all("basal" in path.parts for path in saved_paths)
    assert all("2024-06-01" in path.parts for path in saved_paths)
    assert any(path.name.endswith("dendrite_event_example.svg") for path in saved_paths)
    assert any(path.name.endswith("spine_event_example.svg") for path in saved_paths)
