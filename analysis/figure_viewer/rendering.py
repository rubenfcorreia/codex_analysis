from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

_HAS_CAIROSVG = find_spec("cairosvg") is not None
_HAS_REPORTLAB = find_spec("reportlab") is not None and find_spec("svglib") is not None


def _load_raster_image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGBA")


def _load_svg_via_cairosvg(path: Path) -> Image.Image:
    from cairosvg import svg2png

    png_bytes = svg2png(bytestring=path.read_bytes(), url=str(path))
    return Image.open(BytesIO(png_bytes)).convert("RGBA")


def _load_svg_via_reportlab(path: Path) -> Image.Image:
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg

    drawing = svg2rlg(str(path))
    if drawing is None:
        raise ValueError(f"Could not parse SVG: {path}")
    return renderPM.drawToPIL(drawing).convert("RGBA")


def _svg_preview_fallback(path: Path) -> Image.Image | None:
    png_path = path.with_suffix(".png")
    if png_path.exists():
        return _load_raster_image(png_path)
    return None


def _load_svg_image(path: Path) -> Image.Image:
    errors = []
    if _HAS_CAIROSVG:
        try:
            return _load_svg_via_cairosvg(path)
        except Exception as exc:  # pragma: no cover - backend specific
            errors.append(f"cairosvg: {exc}")
    if _HAS_REPORTLAB:
        try:
            return _load_svg_via_reportlab(path)
        except Exception as exc:  # pragma: no cover - backend specific
            errors.append(f"reportlab: {exc}")
    fallback = _svg_preview_fallback(path)
    if fallback is not None:
        return fallback
    if not errors:
        errors.append("No SVG rasterizer is installed. Install `reportlab` or `cairosvg`.")
    raise RuntimeError("; ".join(errors))


def _placeholder_image(message: str, size: Tuple[int, int] = (960, 540)) -> Image.Image:
    image = Image.new("RGBA", size, "#f7f0e8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = max(size[0] - 40, 40)
    lines = []
    for paragraph in str(message).splitlines() or ["Unable to render figure"]:
        lines.extend(wrap(paragraph, width=80) or [""])
    if not lines:
        lines = ["Unable to render figure"]
    line_height = 14
    total_height = line_height * len(lines)
    y = max((size[1] - total_height) // 2, 20)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = max((size[0] - text_width) // 2, 20)
        draw.text((x, y), line, fill="#7a3e00", font=font)
        y += line_height
    return image


@lru_cache(maxsize=256)
def load_source_image(path_text: str, mtime_ns: int) -> Image.Image:
    path = Path(path_text)
    try:
        suffix = path.suffix.lower()
        if suffix == ".svg":
            return _load_svg_image(path)
        return _load_raster_image(path)
    except Exception as exc:  # pragma: no cover - exercised through GUI and smoke tests
        return _placeholder_image(f"{path.name}\n{exc}")


def image_for_preview(path: Path) -> Image.Image:
    if not path.exists():
        return _placeholder_image(f"{path.name}\nMissing file")
    if path.suffix.lower() == ".svg" and not (_HAS_CAIROSVG or _HAS_REPORTLAB):
        fallback = _svg_preview_fallback(path)
        if fallback is not None:
            fallback_path = path.with_suffix(".png")
            stat = fallback_path.stat()
            return load_source_image(str(fallback_path), int(stat.st_mtime_ns)).copy()
    stat = path.stat()
    return load_source_image(str(path), int(stat.st_mtime_ns)).copy()
