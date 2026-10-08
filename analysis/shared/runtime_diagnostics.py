"""Low-overhead runtime diagnostics shared by analysis entrypoints."""

from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import Any, Dict


def _file_stats(root: Path) -> tuple[int, int]:
    count = 0
    size = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                count += 1
                try:
                    size += path.stat().st_size
                except OSError:
                    pass
    return count, size


def snapshot(root: Path) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        open_figures = len(plt.get_fignums())
    except Exception:
        open_figures = None
    files, bytes_written = _file_stats(Path(root))
    return {
        "timestamp": time.time(),
        "open_figures": open_figures,
        "file_count": files,
        "output_bytes": bytes_written,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def finish(start: Dict[str, Any], root: Path) -> Dict[str, Any]:
    end = snapshot(Path(root))
    return {
        "elapsed_s": float(end["timestamp"] - start["timestamp"]),
        "start": start,
        "end": end,
        "figure_leak_count": (end["open_figures"] - start["open_figures"])
        if start.get("open_figures") is not None and end.get("open_figures") is not None else None,
        "output_file_count_delta": int(end["file_count"] - start["file_count"]),
        "output_bytes_delta": int(end["output_bytes"] - start["output_bytes"]),
    }
