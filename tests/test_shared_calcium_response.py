from __future__ import annotations

import numpy as np

from analysis.shared import shared_calcium_response as response


def test_visual_response_uses_cut_trial_index_and_video_id(monkeypatch, tmp_path):
    cut_dir = tmp_path / "cut_intertrials"
    cut_dir.mkdir()
    (cut_dir / "s2p_ch1_dF_cut.pickle").write_bytes(b"placeholder")

    prefix = "D:" + chr(92) + "bonsai_resources" + chr(92) + "all_movie_clips_bv_sets" + chr(92) + "007" + chr(92)
    rows = [
        {"time": "1000.0", "duration": "3", "F1_type": "movie", "F1_name": prefix + "00000"},
        {"time": "2000.0", "duration": "3", "F1_type": "movie", "F1_name": prefix + "01001"},
        {"time": "3000.0", "duration": "3", "F1_type": "movie", "F1_name": prefix + "02001"},
        {"time": "4000.0", "duration": "3", "F1_type": "movie", "F1_name": prefix + "00277"},
    ]
    cut_time = np.asarray([-1.0, 0.0, 1.0, 2.0])
    cut_neural = np.asarray([
        [[1.0, 2.0, 2.0, 2.0], [3.0, 4.0, 4.0, 4.0], [5.0, 6.0, 6.0, 6.0], [7.0, 8.0, 8.0, 8.0]],
    ])
    monkeypatch.setattr(response, "extract_cut_neural_bundle", lambda path: (cut_time, cut_neural, {}))

    loaded = response.load_visual_response_cut_data(tmp_path, 1, rows)
    assert loaded is not None
    metadata = loaded["trial_meta"]
    assert [item["trial_index"] for item in metadata] == [0, 1, 2, 3]
    assert [item["category"] for item in metadata] == ["blank", "grating", "zebra", "movies"]
    assert all(item["metadata_source"] == "cut_trial_index" for item in metadata)
    assert [item["video_id"].rsplit("\\", 1)[-1] for item in metadata] == ["00000", "01001", "02001", "00277"]

    mean_rows = response.visual_response_trial_rows(cut_neural[0], cut_time, metadata, response_metric="mean")
    assert {item["group"] for item in mean_rows} == {"blank", "visual"}
    assert {item["trial_label"] for item in mean_rows} == {"blank", "grating", "zebra", "movies"}
    assert {item["response"] for item in mean_rows} == {2.0, 4.0, 6.0, 8.0}

    event_rows = response.visual_response_trial_rows(cut_neural[0], cut_time, metadata, response_metric="calcium_events")
    assert len(event_rows) == 4
    assert {item["group"] for item in event_rows} == {"blank", "visual"}
