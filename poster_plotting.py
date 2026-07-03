from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import math
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Sequence

import numpy as np
import warnings

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except Exception:  # pragma: no cover - matplotlib is required for real figure generation
    matplotlib = None
    plt = None
    MaxNLocator = None

try:
    import cairosvg
except Exception:  # pragma: no cover - cairosvg is optional for SVG rasterization
    cairosvg = None

try:
    from svgutils.compose import Figure as SVGFigure
    from svgutils.compose import SVG as SVGPanel
    from svgutils.compose import Text as SVGText
except Exception:  # pragma: no cover - svgutils is optional for SVG composition
    SVGFigure = None
    SVGPanel = None
    SVGText = None


POSTER_DPI = 300


class PosterSize(int):
    """Integer-like poster text size that never shrinks below its floor."""

    def __new__(cls, value: int, floor: int) -> "PosterSize":
        obj = int.__new__(cls, int(value))
        obj._floor = int(floor)
        return obj

    def _clamp(self, value: float) -> "PosterSize":
        return PosterSize(max(self._floor, int(round(float(value)))), self._floor)

    def __add__(self, other: object) -> "PosterSize":
        return self._clamp(int(self) + float(other))

    def __radd__(self, other: object) -> "PosterSize":
        return self._clamp(float(other) + int(self))

    def __sub__(self, other: object) -> "PosterSize":
        return self._clamp(int(self) - float(other))

    def __rsub__(self, other: object) -> "PosterSize":
        return self._clamp(float(other) - int(self))

    def __mul__(self, other: object) -> "PosterSize":
        return self._clamp(int(self) * float(other))

    def __rmul__(self, other: object) -> "PosterSize":
        return self._clamp(float(other) * int(self))

    def __truediv__(self, other: object) -> "PosterSize":
        return self._clamp(int(self) / float(other))

    def __rtruediv__(self, other: object) -> "PosterSize":
        return self._clamp(float(other) / int(self))


POSTER_FONT_SIZE = PosterSize(9, 9)
POSTER_LABEL_SIZE = PosterSize(11, 11)
POSTER_TITLE_SIZE = PosterSize(12, 12)
POSTER_SUPTITLE_SIZE = PosterSize(12, 12)
POSTER_LEGEND_SIZE = PosterSize(9, 9)
POSTER_NOTE_SIZE = PosterSize(9, 9)
POSTER_SINGLE_FIGSIZE = (7.8, 5.4)
POSTER_DOUBLE_FIGSIZE = (10.4, 5.8)
POSTER_WIDE_FIGSIZE = (11.2, 6.8)
POSTER_DENSE_FIGSIZE = (10.6, 7.2)
POSTER_LINEWIDTH = 1.8
POSTER_MARKERSIZE = 5.0
POSTER_TICK_WIDTH = 1.0
POSTER_TICK_LENGTH = 5.0



SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def _parse_svg_dimension(value: Any) -> float:
    try:
        text = str(value).strip().lower().replace("px", "")
        if not text:
            return float("nan")
        return float(text)
    except Exception:
        return float("nan")


def _load_svg_panel(path: Path) -> tuple[ET.Element, float, float]:
    tree = ET.parse(str(path))
    root = tree.getroot()
    width = _parse_svg_dimension(root.attrib.get("width"))
    height = _parse_svg_dimension(root.attrib.get("height"))
    if not np.isfinite(width) or not np.isfinite(height):
        view_box = str(root.attrib.get("viewBox") or "").strip().replace(",", " ")
        parts = [part for part in view_box.split() if part]
        if len(parts) == 4:
            try:
                width = float(parts[2])
                height = float(parts[3])
            except Exception:
                width = float("nan")
                height = float("nan")
    if not np.isfinite(width):
        width = 0.0
    if not np.isfinite(height):
        height = 0.0
    return root, width, height


def _plan_grid_layout(n_panels: int, grid_shape: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    n_panels = max(1, int(n_panels))
    if grid_shape is None:
        ncols = max(1, int(math.ceil(math.sqrt(n_panels))))
        nrows = int(math.ceil(n_panels / ncols))
        return nrows, ncols
    nrows = max(1, int(grid_shape[0]))
    ncols = max(1, int(grid_shape[1]))
    if nrows * ncols < n_panels:
        nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def configure_poster_matplotlib() -> None:
    if matplotlib is None:
        return
    matplotlib.rcParams.update(
        {
            "font.size": POSTER_FONT_SIZE,
            "axes.titlesize": POSTER_TITLE_SIZE,
            "axes.titlepad": 8.0,
            "axes.labelsize": POSTER_LABEL_SIZE,
            "axes.labelpad": 6.0,
            "xtick.labelsize": POSTER_FONT_SIZE,
            "ytick.labelsize": POSTER_FONT_SIZE,
            "xtick.major.size": POSTER_TICK_LENGTH,
            "ytick.major.size": POSTER_TICK_LENGTH,
            "xtick.major.width": POSTER_TICK_WIDTH,
            "ytick.major.width": POSTER_TICK_WIDTH,
            "xtick.minor.size": POSTER_TICK_LENGTH * 0.65,
            "ytick.minor.size": POSTER_TICK_LENGTH * 0.65,
            "legend.fontsize": POSTER_LEGEND_SIZE,
            "legend.frameon": False,
            "figure.titlesize": POSTER_SUPTITLE_SIZE,
            "figure.titleweight": "bold",
            "svg.fonttype": "none",
            "lines.linewidth": POSTER_LINEWIDTH,
            "lines.markersize": POSTER_MARKERSIZE,
        }
    )


def save_figure(
    fig: Any,
    path: Path,
    dpi: int = POSTER_DPI,
    extra_formats: Sequence[str] = ("svg",),
    pad_inches: float = 0.22,
) -> List[Path]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    svg_path = output_path if output_path.suffix.lower() == ".svg" else stem.with_suffix(".svg")

    fig.savefig(svg_path, format="svg", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)

    if plt is not None:
        plt.close(fig)
    return [svg_path]


def rasterize_svg_to_png(svg_path: Path, png_path: Path, dpi: int = POSTER_DPI) -> Optional[Path]:
    if cairosvg is None:
        warnings.warn(
            "cairosvg is unavailable, so the final PNG could not be rasterized from the composed SVG.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    svg_path = Path(svg_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=dpi)
    return png_path


def _apply_svg_viewbox_trim(
    panel: ET.Element,
    *,
    panel_width: float,
    panel_height: float,
    trim: tuple[float, float, float, float] | None,
) -> None:
    """Trim an SVG viewBox before nesting it, making the content larger.

    trim is (left, top, right, bottom), each as a fraction of the current viewBox.
    Example: (0.03, 0.08, 0.03, 0.05) trims 3% left/right, 8% top, 5% bottom.
    """
    view_box = str(panel.attrib.get("viewBox") or "").strip().replace(",", " ")
    parts = [part for part in view_box.split() if part]

    if len(parts) == 4:
        try:
            x0, y0, width, height = [float(part) for part in parts]
        except ValueError:
            x0, y0, width, height = 0.0, 0.0, float(panel_width), float(panel_height)
    else:
        x0, y0, width, height = 0.0, 0.0, float(panel_width), float(panel_height)

    if width <= 0.0 or height <= 0.0:
        x0, y0, width, height = 0.0, 0.0, float(panel_width), float(panel_height)

    if trim is not None:
        left, top, right, bottom = [max(0.0, float(value)) for value in trim]
        left = min(left, 0.45)
        top = min(top, 0.45)
        right = min(right, 0.45)
        bottom = min(bottom, 0.45)

        new_x0 = x0 + left * width
        new_y0 = y0 + top * height
        new_width = width * max(0.05, 1.0 - left - right)
        new_height = height * max(0.05, 1.0 - top - bottom)

        x0, y0, width, height = new_x0, new_y0, new_width, new_height

    panel.attrib["viewBox"] = f"{x0:g} {y0:g} {width:g} {height:g}"


def compose_svg_figure_fit_to_boxes(
    output_path: Path,
    component_paths: Sequence[Path],
    boxes: Sequence[tuple[float, float, float, float]],
    canvas_width: float,
    canvas_height: float,
    *,
    physical_width: str | None = None,
    physical_height: str | None = None,
    preserve_aspect_ratio: str = "xMidYMid meet",
    viewbox_trims: Sequence[tuple[float, float, float, float] | None] | None = None,
) -> Path | None:
    """Compose SVG panels by scaling each panel into an explicit output box.

    Each input SVG is nested inside a fixed rectangle on the output canvas.
    Optional viewBox trims remove internal whitespace from source SVGs before
    fitting, so plots occupy more of their allotted poster quadrant.
    """
    component_paths = [Path(path) for path in component_paths]
    if not component_paths:
        return None
    if len(component_paths) != len(boxes):
        raise ValueError(
            f"Expected one box per SVG panel, got {len(boxes)} boxes for "
            f"{len(component_paths)} panels."
        )

    if viewbox_trims is None:
        viewbox_trims = [None] * len(component_paths)
    elif len(viewbox_trims) != len(component_paths):
        raise ValueError(
            f"Expected one viewBox trim per SVG panel, got {len(viewbox_trims)} trims for "
            f"{len(component_paths)} panels."
        )

    svg_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(physical_width or canvas_width),
            "height": str(physical_height or canvas_height),
            "viewBox": f"0 0 {canvas_width:g} {canvas_height:g}",
        },
    )

    for index, (path, box, trim) in enumerate(zip(component_paths, boxes, viewbox_trims), start=1):
        panel, panel_width, panel_height = _load_svg_panel(path)
        x, y, width, height = [float(value) for value in box]
        if width <= 0.0 or height <= 0.0:
            continue

        _apply_svg_viewbox_trim(
            panel,
            panel_width=panel_width,
            panel_height=panel_height,
            trim=trim,
        )

        panel.attrib["x"] = f"{x:g}"
        panel.attrib["y"] = f"{y:g}"
        panel.attrib["width"] = f"{width:g}"
        panel.attrib["height"] = f"{height:g}"
        panel.attrib["preserveAspectRatio"] = preserve_aspect_ratio
        panel.attrib.pop("transform", None)

        panel_group = ET.SubElement(
            svg_root,
            f"{{{SVG_NS}}}g",
            {"id": f"panel_{index:02d}_{path.stem}"},
        )
        panel_group.append(panel)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg_root).write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path

def compose_svg_figure(
    output_path: Path,
    component_paths: Sequence[Path],
    layout: str = "horizontal",
    gap: float = 30.0,
    margin: float = 36.0,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    title_size: int = POSTER_SUPTITLE_SIZE,
    subtitle_size: int = POSTER_NOTE_SIZE + 2,
    grid_shape: Optional[tuple[int, int]] = None,
) -> Optional[Path]:
    layout = str(layout).lower().strip()
    component_paths = [Path(path) for path in component_paths]
    if not component_paths:
        return None

    panels: List[tuple[Any, float, float]] = []
    for path in component_paths:
        panel, width, height = _load_svg_panel(path)
        panels.append((panel, width, height))
    if not panels:
        return None

    panel_widths = [float(panel_width) for _, panel_width, _ in panels]
    panel_heights = [float(panel_height) for _, _, panel_height in panels]
    title_block = 0.0
    if title:
        title_block += 42.0
    if subtitle:
        title_block += 28.0

    if layout == "vertical":
        content_width = max(panel_widths)
        content_height = sum(panel_heights) + gap * max(0, len(panels) - 1)
        canvas_width = content_width + 2.0 * margin
        canvas_height = content_height + 2.0 * margin + title_block
    elif layout == "grid":
        nrows, ncols = _plan_grid_layout(len(panels), grid_shape)
        cell_widths = [0.0] * ncols
        cell_heights = [0.0] * nrows
        for index, (_, panel_width, panel_height) in enumerate(panels):
            row = index // ncols
            col = index % ncols
            if row >= nrows:
                break
            cell_widths[col] = max(cell_widths[col], panel_width)
            cell_heights[row] = max(cell_heights[row], panel_height)
        content_width = sum(cell_widths) + gap * max(0, ncols - 1)
        content_height = sum(cell_heights) + gap * max(0, nrows - 1)
        canvas_width = content_width + 2.0 * margin
        canvas_height = content_height + 2.0 * margin + title_block
    else:
        content_width = sum(panel_widths) + gap * max(0, len(panels) - 1)
        content_height = max(panel_heights)
        canvas_width = content_width + 2.0 * margin
        canvas_height = content_height + 2.0 * margin + title_block

    svg_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(canvas_width),
            "height": str(canvas_height),
            "viewBox": f"0 0 {canvas_width} {canvas_height}",
        },
    )
    if title:
        title_el = ET.SubElement(
            svg_root,
            f"{{{SVG_NS}}}text",
            {
                "x": str(canvas_width / 2.0),
                "y": "26",
                "font-size": str(title_size),
                "font-weight": "bold",
                "text-anchor": "middle",
            },
        )
        title_el.text = title
    if subtitle:
        subtitle_y = 26.0 if not title else 47.0
        subtitle_el = ET.SubElement(
            svg_root,
            f"{{{SVG_NS}}}text",
            {
                "x": str(canvas_width / 2.0),
                "y": str(subtitle_y),
                "font-size": str(subtitle_size),
                "text-anchor": "middle",
                "fill": "#444444",
            },
        )
        subtitle_el.text = subtitle
    if layout == "vertical":
        y = margin + title_block
        for index, (panel, panel_width, panel_height) in enumerate(panels, start=1):
            x = margin + (content_width - panel_width) / 2.0
            panel.attrib["x"] = str(x)
            panel.attrib["y"] = str(y)
            panel_id = f"panel_{index:02d}_{component_paths[index - 1].stem}"
            panel_group = ET.SubElement(svg_root, f"{{{SVG_NS}}}g", {"id": panel_id})
            panel_group.append(panel)
            y += panel_height + gap
    elif layout == "grid":
        nrows, ncols = _plan_grid_layout(len(panels), grid_shape)
        cell_widths = [0.0] * ncols
        cell_heights = [0.0] * nrows
        for index, (_, panel_width, panel_height) in enumerate(panels):
            row = index // ncols
            col = index % ncols
            if row >= nrows:
                break
            cell_widths[col] = max(cell_widths[col], panel_width)
            cell_heights[row] = max(cell_heights[row], panel_height)
        row_offsets: List[float] = []
        running_y = margin + title_block
        for row_height in cell_heights:
            row_offsets.append(running_y)
            running_y += row_height + gap
        col_offsets: List[float] = []
        running_x = margin
        for col_width in cell_widths:
            col_offsets.append(running_x)
            running_x += col_width + gap
        for index, (panel, panel_width, panel_height) in enumerate(panels, start=1):
            row = (index - 1) // ncols
            col = (index - 1) % ncols
            if row >= nrows:
                break
            x = col_offsets[col] + (cell_widths[col] - panel_width) / 2.0
            y = row_offsets[row] + (cell_heights[row] - panel_height) / 2.0
            panel.attrib["x"] = str(x)
            panel.attrib["y"] = str(y)
            panel_id = f"panel_{index:02d}_{component_paths[index - 1].stem}"
            panel_group = ET.SubElement(svg_root, f"{{{SVG_NS}}}g", {"id": panel_id})
            panel_group.append(panel)
    else:
        x = margin
        for index, (panel, panel_width, panel_height) in enumerate(panels, start=1):
            y = margin + title_block + (content_height - panel_height) / 2.0
            panel.attrib["x"] = str(x)
            panel.attrib["y"] = str(y)
            panel_id = f"panel_{index:02d}_{component_paths[index - 1].stem}"
            panel_group = ET.SubElement(svg_root, f"{{{SVG_NS}}}g", {"id": panel_id})
            panel_group.append(panel)
            x += panel_width + gap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg_root).write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path


def set_sparse_numeric_ticks(ax: Any, axis: str = "both", nbins: int = 5, integer: bool = False) -> None:
    if MaxNLocator is None:
        return
    if axis in {"x", "both"}:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max(2, nbins), integer=integer))
    if axis in {"y", "both"}:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(2, nbins), integer=integer))


def set_sparse_colorbar_ticks(cbar: Any, nbins: int = 5) -> None:
    if MaxNLocator is None:
        return
    cbar.locator = MaxNLocator(nbins=max(2, nbins))
    cbar.update_ticks()


def set_hour_ticks(ax: Any, total_hours: float, labelsize: int = POSTER_FONT_SIZE) -> None:
    if not np.isfinite(total_hours) or total_hours <= 0:
        total_hours = 1.0
    if total_hours <= 2.5:
        tick_step = 0.5
    elif total_hours <= 5.0:
        tick_step = 1.0
    elif total_hours <= 10.0:
        tick_step = 2.0
    elif total_hours <= 20.0:
        tick_step = 4.0
    else:
        tick_step = 6.0
    ticks = np.arange(0.0, total_hours + 1e-6, tick_step)
    if ticks.size < 2:
        ticks = np.asarray([0.0, total_hours], dtype=float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks], fontsize=labelsize)
