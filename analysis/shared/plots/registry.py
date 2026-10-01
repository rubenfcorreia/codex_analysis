from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

_PLOT_FAMILIES: dict[str, Callable[..., list[Path]]] = {}


def register_plot_family(name: str, plotter: Callable[..., list[Path]]):
    key = str(name).strip().lower()
    if not key:
        raise ValueError("plot family name cannot be empty")
    _PLOT_FAMILIES[key] = plotter
    return plotter


def plot_analysis_family(name: str, result: Any, context: Any, **kwargs: Any) -> list[Path]:
    key = str(name).strip().lower()
    try:
        plotter = _PLOT_FAMILIES[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PLOT_FAMILIES)) or "none"
        raise KeyError(f"Unknown plot family {name!r}; registered families: {available}") from exc
    paths = plotter(result, context, **kwargs)
    return [Path(path) for path in (paths or [])]


def plot_family_names() -> list[str]:
    return sorted(_PLOT_FAMILIES)


__all__ = ["plot_analysis_family", "plot_family_names", "register_plot_family"]
