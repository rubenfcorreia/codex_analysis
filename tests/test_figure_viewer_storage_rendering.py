from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.figure_viewer.rendering import image_for_preview
from analysis.figure_viewer.storage import NoteStore


def _write_svg(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='120'><rect width='200' height='120' fill='white'/><text x='10' y='60'>{label}</text></svg>"
    )
    return path


def test_svg_preview_renders_to_image(tmp_path: Path) -> None:
    svg_path = _write_svg(tmp_path / "preview.svg", "svg preview")
    image = image_for_preview(svg_path)
    assert image.mode == "RGBA"
    assert image.width > 0
    assert image.height > 0


def test_svg_preview_falls_back_to_png_without_svg_backend(tmp_path: Path, monkeypatch) -> None:
    svg_path = _write_svg(tmp_path / "preview.svg", "svg preview")
    png_path = svg_path.with_suffix(".png")
    Image.new("RGBA", (180, 90), "white").save(png_path)

    import analysis.figure_viewer.rendering as rendering

    monkeypatch.setattr(rendering, "_HAS_CAIROSVG", False)
    monkeypatch.setattr(rendering, "_HAS_REPORTLAB", False)

    image = image_for_preview(svg_path)
    assert image.mode == "RGBA"
    assert image.size == (180, 90)


def test_note_store_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "notes.sqlite"
    store = NoteStore(db_path)
    try:
        store.add_note(
            scope="image",
            scope_key="figure-1",
            note_text="First note",
            context={"figure_key": "figure-1", "kind": "image"},
        )
        store.add_note(
            scope="comparison",
            scope_key="comparison-1",
            note_text="Comparison note",
            context={"comparison_key": "comparison-1", "kind": "comparison"},
        )
        image_notes = store.list_notes(scope="image", scope_key="figure-1")
        comparison_notes = store.list_notes(scope="comparison", scope_key="comparison-1")
        assert len(image_notes) == 1
        assert image_notes[0]["note_text"] == "First note"
        assert image_notes[0]["context"]["kind"] == "image"
        assert len(comparison_notes) == 1
        assert comparison_notes[0]["note_text"] == "Comparison note"
    finally:
        store.close()
