from pathlib import Path

from analysis.shared.plots.poster_ready import (
    paired_state_values_from_rows,
    write_paired_state_summary_figure,
)


def _rows(states):
    return [
        {"global_soma_id": entity, "state": state, "mean": value}
        for entity, state, value in states
    ]


def test_paired_state_values_intersects_entities_and_averages_duplicates():
    rows = _rows(
        [
            ("a", "quiet_awake_blank", 1.0),
            ("a", "quiet_awake_blank", 3.0),
            ("a", "nrem_blank", 2.0),
            ("b", "quiet_awake_blank", 5.0),
            ("b", "nrem_blank", 6.0),
            ("c", "quiet_awake_blank", 7.0),
            ("c", "rem_blank", 9.0),
        ]
    )
    values, sample_sizes = paired_state_values_from_rows(
        rows,
        state_order=["quiet_awake_blank", "nrem_blank"],
        entity_id_column="global_soma_id",
    )
    assert values == {
        "quiet_awake_blank": [2.0, 5.0],
        "nrem_blank": [2.0, 6.0],
    }
    assert sample_sizes == {"quiet_awake_blank": 2, "nrem_blank": 2}


def test_paired_state_values_supports_three_state_movie_intersection():
    rows = _rows(
        [
            ("a", "quiet_awake_movies", 1.0),
            ("a", "nrem_movies", 2.0),
            ("a", "rem_movies", 3.0),
            ("b", "quiet_awake_movies", 4.0),
            ("b", "nrem_movies", 5.0),
        ]
    )
    values, sample_sizes = paired_state_values_from_rows(
        rows,
        state_order=["quiet_awake_movies", "nrem_movies", "rem_movies"],
        entity_id_column="global_soma_id",
    )
    assert values == {
        "quiet_awake_movies": [1.0],
        "nrem_movies": [2.0],
        "rem_movies": [3.0],
    }
    assert sample_sizes == {
        "quiet_awake_movies": 1,
        "nrem_movies": 1,
        "rem_movies": 1,
    }


def test_paired_state_summary_writer_writes_separate_svg(tmp_path: Path):
    output = write_paired_state_summary_figure(
        output_dir=tmp_path / "paired_state_summary",
        entity_label="bouton",
        state_values={"quiet_awake_movies": [1.0, 2.0], "nrem_movies": [2.0, 3.0]},
        state_order=["quiet_awake_movies", "nrem_movies"],
        sample_sizes={"quiet_awake_movies": 2, "nrem_movies": 2},
        output_stem="bouton_movies_quiet_awake_vs_nrem_paired_state_summary",
        title="Movies quiet awake vs nrem",
    )
    assert output is not None
    assert Path(output).parent.name == "paired_state_summary"
    assert Path(output).name == "bouton_movies_quiet_awake_vs_nrem_paired_state_summary.svg"
    assert Path(output).exists()


def test_paired_state_summary_writer_supports_basal_apical_panels(tmp_path: Path):
    output = write_paired_state_summary_figure(
        output_dir=tmp_path,
        entity_label="dendrite",
        state_values={},
        state_order=["quiet_awake_blank", "nrem_blank"],
        panel_values={
            "basal": {"quiet_awake_blank": [1.0], "nrem_blank": [2.0]},
            "apical": {"quiet_awake_blank": [3.0], "nrem_blank": [4.0]},
        },
        panel_sample_sizes={
            "basal": {"quiet_awake_blank": 1, "nrem_blank": 1},
            "apical": {"quiet_awake_blank": 1, "nrem_blank": 1},
        },
        output_stem="dendrite_blank_quiet_awake_vs_nrem_paired_state_summary",
        title="Blank quiet awake vs nrem",
    )
    assert output is not None
    assert Path(output).exists()


def test_paired_state_summary_writer_skips_empty_intersection(tmp_path: Path):
    output = write_paired_state_summary_figure(
        output_dir=tmp_path,
        entity_label="soma",
        state_values={"quiet_awake_blank": [], "nrem_blank": []},
        state_order=["quiet_awake_blank", "nrem_blank"],
        sample_sizes={"quiet_awake_blank": 0, "nrem_blank": 0},
        output_stem="soma_blank_quiet_awake_vs_nrem_paired_state_summary",
        title="Blank quiet awake vs nrem",
    )
    assert output is None
    assert not list(tmp_path.glob("*.svg"))
