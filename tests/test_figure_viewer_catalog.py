from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.figure_viewer.app import FigureViewerApp
from analysis.figure_viewer.catalog import discover_figure_records, filter_records, group_records, unique_values
from analysis.figure_viewer.layout import SlotSelection, browser_children, build_results_index, comparison_signature, resolve_selection, selection_from_record, selection_with_field
from analysis.figure_viewer.models import FigureFilterState, FigureRecord


def _write_svg(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='120' height='80'><rect width='120' height='80' fill='white'/><text x='8' y='44'>{label}</text></svg>"
    )
    return path


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_discover_figure_records_indexes_summary_checkpoint_and_review(tmp_path: Path) -> None:
    repo_root = tmp_path

    summary_output_root = repo_root / "results" / "demo_pipeline" / "demo_preset" / "activity_split" / "nrem"
    summary_svg = _write_svg(summary_output_root / "figures" / "state_summary" / "dendrites" / "selected_states" / "all" / "state_summary_boxplots_dendrite_mean.svg", "summary")
    summary_png = _write_svg(summary_output_root / "figures" / "state_summary" / "dendrites" / "selected_states" / "all" / "state_summary_boxplots_dendrite_mean.png", "summary png")
    _write_manifest(
        summary_output_root / "summary" / "manifest.json",
        {
            "output_root": str(summary_output_root),
            "analysis_branch_name": "activity_split",
            "analysis_basis_name": "nrem",
            "analysis_scope": {"branch_name": "activity_split", "basis_name": "nrem"},
            "job_spec": {
                "pipeline": "demo_pipeline",
                "analysis_type": "demo_preset",
                "cohort": "all",
            },
            "output_artifacts": [
                "figures/state_summary/dendrites/selected_states/all/state_summary_boxplots_dendrite_mean.png",
                "figures/state_summary/dendrites/selected_states/all/state_summary_boxplots_dendrite_mean.svg",
            ],
        },
    )

    checkpoint_output_root = repo_root / "results" / "demo_pipeline" / "checkpoint_demo"
    checkpoint_svg = _write_svg(checkpoint_output_root / "checkpoint_examples" / "state_summary" / "dendrites" / "selected_states" / "all" / "checkpoint_state_summary.svg", "checkpoint")
    _write_manifest(
        checkpoint_output_root / "checkpoint_examples" / "manifest.json",
        {
            "entries": [
                {
                    "checkpoint": "state_summary",
                    "file": "checkpoint_examples/state_summary/dendrites/selected_states/all/checkpoint_state_summary.svg",
                    "title": "Checkpoint state summary",
                    "scope": "dendrites/selected_states/all",
                    "variant": "all",
                }
            ],
            "gallery_dir": str(checkpoint_output_root / "checkpoint_examples"),
            "generated_at": "2026-08-31T12:00:00",
            "n_files": 1,
        },
    )

    review_svg = _write_svg(repo_root / "review_figures" / "mixed_model" / "review_selected_state.svg", "review")

    records = discover_figure_records(repo_root=repo_root)
    assert len(records) == 3
    summary_record = next(record for record in records if record.preview_path == summary_svg)
    checkpoint_record = next(record for record in records if record.preview_path == checkpoint_svg)
    review_record = next(record for record in records if record.preview_path == review_svg)

    assert summary_record.preview_path.suffix == ".svg"
    assert summary_record.pipeline == "demo_pipeline"
    assert summary_record.preset == "demo_preset"
    assert summary_record.split == "activity_split"
    assert summary_record.basis == "nrem"
    assert summary_record.family == "state_summary"
    assert summary_record.comparison_key
    assert summary_record.comparison_label.startswith("state summary")
    assert checkpoint_record.pipeline == "demo_pipeline"
    assert checkpoint_record.preset == "checkpoint_demo"
    assert checkpoint_record.family == "state_summary"
    assert review_record.pipeline == "review_figures"
    assert review_record.family == "mixed_model"

    grouped = group_records(records)
    assert summary_record.comparison_key in grouped
    assert len(grouped[summary_record.comparison_key]) == 1
    assert unique_values(records, "pipeline") == ["demo_pipeline", "review_figures"]


def test_discover_figure_records_emits_scan_progress(tmp_path: Path) -> None:
    repo_root = tmp_path

    output_root = repo_root / "results" / "demo_pipeline" / "demo_preset" / "activity_split" / "nrem"
    _write_svg(output_root / "figures" / "state_summary" / "dendrites" / "selected_states" / "all" / "state_summary_boxplots_dendrite_mean.svg", "summary")
    _write_manifest(
        output_root / "summary" / "manifest.json",
        {
            "output_root": str(output_root),
            "analysis_branch_name": "activity_split",
            "analysis_basis_name": "nrem",
            "analysis_scope": {"branch_name": "activity_split", "basis_name": "nrem"},
            "job_spec": {
                "pipeline": "demo_pipeline",
                "analysis_type": "demo_preset",
                "cohort": "all",
            },
            "output_artifacts": [
                "figures/state_summary/dendrites/selected_states/all/state_summary_boxplots_dendrite_mean.svg",
            ],
        },
    )

    events = []
    records = discover_figure_records(repo_root=repo_root, progress_callback=events.append)

    assert len(records) == 1
    assert len(events) >= 3
    assert events[0].phase == "Scanning results/"
    assert events[0].current == 0
    assert events[0].total == 1
    assert events[0].message.startswith("Locating ")
    assert any(event.phase == "Processing summary manifests" for event in events)
    assert events[-1].phase == "Finalizing figure list"
    assert events[-1].current == events[-1].total == 1
    assert events[-1].message == "Indexed 1 figures."


def test_filter_records_and_comparison_grouping() -> None:
    record_a = FigureRecord(
        figure_key="/tmp/a",
        display_label="A",
        title="State summary boxplots",
        preview_path=Path("/tmp/a.svg"),
        comparison_key="state_summary::state_summary_boxplots",
        comparison_label="state_summary / State summary boxplots",
        pipeline="demo_pipeline",
        preset="preset_a",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="all",
        scope="dendrites/all",
        search_text="state summary boxplots demo_pipeline preset_a activity_split nrem",
        sort_key=("demo_pipeline", "preset_a", "activity_split", "nrem", "state_summary", "all", "dendrites/all", "State summary boxplots"),
    )
    record_b = FigureRecord(
        figure_key="/tmp/b",
        display_label="B",
        title="State summary boxplots",
        preview_path=Path("/tmp/b.svg"),
        comparison_key="state_summary::state_summary_boxplots",
        comparison_label="state_summary / State summary boxplots",
        pipeline="demo_pipeline",
        preset="preset_b",
        split="frequency_split",
        basis="nrem",
        family="state_summary",
        cohort="all",
        scope="dendrites/all",
        search_text="state summary boxplots demo_pipeline preset_b frequency_split nrem",
        sort_key=("demo_pipeline", "preset_b", "frequency_split", "nrem", "state_summary", "all", "dendrites/all", "State summary boxplots"),
    )
    filtered = filter_records([record_a, record_b], FigureFilterState(split="activity_split", search="boxplots"))
    assert filtered == [record_a]
    grouped = group_records([record_a, record_b])
    assert list(grouped) == ["state_summary::state_summary_boxplots"]
    assert grouped["state_summary::state_summary_boxplots"] == [record_a, record_b]


def _manual_record(*, figure_key: str, display_label: str, title: str, preview_path: Path, pipeline: str, preset: str, split: str, basis: str, family: str, cohort: str, scope: str) -> FigureRecord:
    return FigureRecord(
        figure_key=figure_key,
        display_label=display_label,
        title=title,
        preview_path=preview_path,
        pipeline=pipeline,
        preset=preset,
        split=split,
        basis=basis,
        family=family,
        cohort=cohort,
        scope=scope,
    )


def test_pipeline_selection_does_not_auto_fill_downstream_fields() -> None:
    record_a = _manual_record(
        figure_key="figure-a",
        display_label="Figure A",
        title="State summary",
        preview_path=Path("/tmp/a.svg"),
        pipeline="demo_pipeline",
        preset="preset_a",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="all",
        scope="dendrites/all",
    )
    record_b = _manual_record(
        figure_key="figure-b",
        display_label="Figure B",
        title="State summary",
        preview_path=Path("/tmp/b.svg"),
        pipeline="demo_pipeline",
        preset="preset_b",
        split="frequency_split",
        basis="rem",
        family="mixed_model",
        cohort="responsive",
        scope="dendrites/responsive",
    )

    selection = selection_with_field(SlotSelection(), "pipeline", "demo_pipeline")
    resolved, candidate_records = resolve_selection([record_a, record_b], selection)

    assert resolved.pipeline == "demo_pipeline"
    assert resolved.preset == ""
    assert resolved.split == ""
    assert resolved.basis == ""
    assert resolved.family == ""
    assert resolved.cohort == ""
    assert resolved.scope == ""
    assert resolved.figure_key == ""
    assert candidate_records == [record_a, record_b]


def test_results_tree_follows_folder_hierarchy_and_excludes_review_figures_when_requested(tmp_path: Path) -> None:
    repo_root = tmp_path

    output_root = repo_root / "results" / "demo_pipeline" / "demo_preset" / "activity_split" / "nrem"
    summary_svg = _write_svg(output_root / "figures" / "state_summary" / "dendrites" / "selected_states" / "all" / "state_summary_boxplots_dendrite_mean.svg", "summary")
    _write_manifest(
        output_root / "summary" / "manifest.json",
        {
            "output_root": str(output_root),
            "analysis_branch_name": "activity_split",
            "analysis_basis_name": "nrem",
            "analysis_scope": {"branch_name": "activity_split", "basis_name": "nrem"},
            "job_spec": {
                "pipeline": "demo_pipeline",
                "analysis_type": "demo_preset",
                "cohort": "all",
            },
            "output_artifacts": [
                "figures/state_summary/dendrites/selected_states/all/state_summary_boxplots_dendrite_mean.svg",
            ],
        },
    )
    _write_svg(repo_root / "review_figures" / "mixed_model" / "review_selected_state.svg", "review")

    records = discover_figure_records(repo_root=repo_root, include_review_figures=False)
    assert len(records) == 1

    root = build_results_index(records, repo_root)
    assert root.name == "results"
    assert root.figure_count == 1

    node = root
    for expected_name in [
        "demo_pipeline",
        "demo_preset",
        "activity_split",
        "nrem",
        "figures",
        "state_summary",
        "dendrites",
        "selected_states",
        "all",
        "state_summary_boxplots_dendrite_mean.svg",
    ]:
        children = browser_children(node)
        assert len(children) == 1
        node = children[0]
        assert node.name == expected_name

    assert node.is_leaf
    assert node.record is not None
    assert node.record.preview_path == summary_svg


def test_slot_selection_changes_in_place_when_changing_cohort() -> None:
    record_all = _manual_record(
        figure_key="figure-all",
        display_label="State summary / all",
        title="State summary",
        preview_path=Path("/tmp/all.svg"),
        pipeline="demo_pipeline",
        preset="demo_preset",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="all",
        scope="dendrites/selected_states/all",
    )
    record_responsive = _manual_record(
        figure_key="figure-responsive",
        display_label="State summary / responsive",
        title="State summary",
        preview_path=Path("/tmp/responsive.svg"),
        pipeline="demo_pipeline",
        preset="demo_preset",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="responsive",
        scope="dendrites/selected_states/responsive",
    )

    selection = selection_from_record(record_all)
    selection = selection_with_field(selection, "cohort", "responsive")
    resolved, candidate_records = resolve_selection(
        [record_all, record_responsive],
        selection,
        preserve_figure_key=record_all.figure_key,
    )

    assert resolved.cohort == "responsive"
    assert resolved.figure_key == record_responsive.figure_key
    assert candidate_records == [record_responsive]
    assert comparison_signature([record_all, record_responsive]) == "figure-all|figure-responsive"
    assert comparison_signature([record_responsive, record_all]) == "figure-responsive|figure-all"



def test_switching_one_field_preserves_other_valid_fields() -> None:
    record_all = _manual_record(
        figure_key="figure-all",
        display_label="State summary / all",
        title="State summary",
        preview_path=Path("/tmp/all.svg"),
        pipeline="demo_pipeline",
        preset="demo_preset",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="all",
        scope="dendrites/selected_states",
    )
    record_responsive = _manual_record(
        figure_key="figure-responsive",
        display_label="State summary / responsive",
        title="State summary",
        preview_path=Path("/tmp/responsive.svg"),
        pipeline="demo_pipeline",
        preset="demo_preset",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="responsive",
        scope="dendrites/selected_states",
    )

    selection = selection_from_record(record_all)
    selection = selection_with_field(selection, "cohort", "responsive")

    assert selection.pipeline == record_all.pipeline
    assert selection.preset == record_all.preset
    assert selection.split == record_all.split
    assert selection.basis == record_all.basis
    assert selection.family == record_all.family
    assert selection.scope == record_all.scope

    resolved, candidate_records = resolve_selection(
        [record_all, record_responsive],
        selection,
        preserve_figure_key=record_all.figure_key,
    )

    assert resolved.pipeline == record_all.pipeline
    assert resolved.preset == record_all.preset
    assert resolved.split == record_all.split
    assert resolved.basis == record_all.basis
    assert resolved.family == record_all.family
    assert resolved.cohort == "responsive"
    assert resolved.scope == record_all.scope
    assert resolved.figure_key == record_responsive.figure_key
    assert candidate_records == [record_responsive]


def test_load_record_into_slot_prefers_empty_slot_before_creating_a_new_one() -> None:
    class DummySlot:
        def __init__(self, index: int, record: FigureRecord | None = None) -> None:
            self.index = index
            self.record = record
            self.loaded: list[FigureRecord] = []

        def load_record(self, record: FigureRecord) -> None:
            self.record = record
            self.loaded.append(record)

    app = FigureViewerApp.__new__(FigureViewerApp)
    app.slots = [
        DummySlot(
            0,
            _manual_record(
                figure_key="figure-a",
                display_label="Figure A",
                title="Figure A",
                preview_path=Path("/tmp/a.svg"),
                pipeline="demo_pipeline",
                preset="preset_a",
                split="activity_split",
                basis="nrem",
                family="state_summary",
                cohort="all",
                scope="dendrites/all",
            ),
        ),
        DummySlot(index=1),
    ]
    app.active_slot_index = 0
    status_messages: list[str] = []
    changed_slots: list[object] = []

    app.set_active_slot = lambda index: setattr(app, "active_slot_index", index)
    app.slot_did_change = lambda slot: changed_slots.append(slot)
    app.set_status = lambda message: status_messages.append(message)

    def add_slot(initial_record=None):  # noqa: ARG001 - mirrors the real method signature
        slot = DummySlot(index=len(app.slots))
        app.slots.append(slot)
        return slot

    app.add_slot = add_slot

    record_b = _manual_record(
        figure_key="figure-b",
        display_label="Figure B",
        title="Figure B",
        preview_path=Path("/tmp/b.svg"),
        pipeline="demo_pipeline",
        preset="preset_b",
        split="activity_split",
        basis="nrem",
        family="state_summary",
        cohort="responsive",
        scope="dendrites/responsive",
    )
    FigureViewerApp.load_record_into_slot(app, record_b)

    assert app.slots[1].record == record_b
    assert app.active_slot_index == 1
    assert changed_slots == [app.slots[1]]
    assert status_messages[-1] == "Added Figure B to Slot 2."

    app.active_slot_index = 0
    app.slots[1].record = record_b
    record_c = _manual_record(
        figure_key="figure-c",
        display_label="Figure C",
        title="Figure C",
        preview_path=Path("/tmp/c.svg"),
        pipeline="demo_pipeline",
        preset="preset_c",
        split="frequency_split",
        basis="rem",
        family="mixed_model",
        cohort="all",
        scope="dendrites/all",
    )
    FigureViewerApp.load_record_into_slot(app, record_c)

    assert len(app.slots) == 3
    assert app.slots[2].record == record_c
    assert app.active_slot_index == 2
    assert status_messages[-1] == "Added Figure C to Slot 3."


def test_add_slot_honors_requested_side_when_arranging_new_panels() -> None:
    from analysis.figure_viewer import app as viewer_app
    from types import SimpleNamespace

    class DummyVar:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

        def set(self, value: str) -> None:
            self.value = value

    class DummyWidget:
        def __init__(self) -> None:
            self.grid_calls = []
            self.row_configs = []
            self.column_configs = []
            self.destroyed = False

        def grid(self, **kwargs) -> None:
            self.grid_calls.append(kwargs)

        def rowconfigure(self, index: int, **kwargs) -> None:
            self.row_configs.append((index, kwargs))

        def columnconfigure(self, index: int, **kwargs) -> None:
            self.column_configs.append((index, kwargs))

        def destroy(self) -> None:
            self.destroyed = True

        def tkraise(self) -> None:
            pass

    class DummySlotView:
        def __init__(self, app, parent, index: int) -> None:
            self.app = app
            self.parent = parent
            self.index = index
            self.frame = DummyWidget()
            self.record = None

        def load_record(self, record: FigureRecord) -> None:
            self.record = record

        def sync_to_records(self, **_kwargs) -> None:
            pass

        def refresh_header(self) -> None:
            pass

    fake_app = FigureViewerApp.__new__(FigureViewerApp)
    fake_app.slots = []
    fake_app.slot_rows = []
    fake_app.slot_row_frames = []
    fake_app._row_column_extents = {}
    fake_app._slot_row_extent = -1
    fake_app.slot_side_var = DummyVar("Top")
    fake_app.slots_container = SimpleNamespace(inner=DummyWidget(), tkraise=lambda: None)
    fake_app.slots_placeholder = DummyWidget()
    fake_app.active_slot_index = None
    refresh_calls: list[bool] = []
    fake_app.set_active_slot = lambda index: setattr(fake_app, "active_slot_index", index)
    fake_app.comparison_panel = SimpleNamespace(refresh=lambda: refresh_calls.append(True))
    fake_app.set_status = lambda _message: None
    fake_app._create_slot_row_frame = lambda: DummyWidget()

    original_slot_view = viewer_app.SlotView
    viewer_app.SlotView = DummySlotView
    try:
        first = FigureViewerApp.add_slot(fake_app)
        assert fake_app.slot_rows == [[first]]
        assert fake_app.slots == [first]
        assert fake_app.active_slot_index == 0

        fake_app.slot_side_var.set("Top")
        second = FigureViewerApp.add_slot(fake_app)
        assert fake_app.slot_rows == [[second], [first]]
        assert fake_app.slots == [second, first]
        assert fake_app.active_slot_index == 0

        fake_app.slot_side_var.set("Right")
        third = FigureViewerApp.add_slot(fake_app)
        assert fake_app.slot_rows == [[second, third], [first]]
        assert fake_app.slots == [second, third, first]
        assert fake_app.active_slot_index == 1
        assert len(refresh_calls) == 3
    finally:
        viewer_app.SlotView = original_slot_view
