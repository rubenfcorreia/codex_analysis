from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def save_figure(fig: Any, path: Path | str, *, dpi: int = 300, extra_formats: Sequence[str] = ("svg",), **kwargs: Any) -> Path:
    """Save a matplotlib figure through the shared output boundary."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, **kwargs)
    for format_name in extra_formats:
        suffix = str(format_name).lstrip(".")
        alternate = output.with_suffix(f".{suffix}")
        if alternate == output:
            continue
        fig.savefig(alternate, dpi=dpi, **kwargs)
    return output


__all__ = ["save_figure"]
