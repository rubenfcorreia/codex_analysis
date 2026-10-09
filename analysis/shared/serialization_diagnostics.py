"""Low-overhead serialization diagnostics for pipeline output writers."""

from __future__ import annotations

import resource
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


_LOCK = threading.Lock()
_ACTIVE: Optional["SerializationCollector"] = None


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if value < 10**10 else value


class SerializationCollector:
    """Collect additive timing records without changing artifact formats."""

    def __init__(self) -> None:
        self.records: list[Dict[str, Any]] = []

    def record(
        self,
        *,
        path: Path | str,
        artifact: str | None = None,
        format_name: str = "unknown",
        elapsed_s: float = 0.0,
        rows: int | None = None,
        columns: int | None = None,
        cache_status: str | None = None,
        schema_version: Any = None,
        status: str = "completed",
        error: str | None = None,
        rss_before: int | None = None,
        rss_after: int | None = None,
    ) -> Dict[str, Any]:
        target = Path(path)
        try:
            output_bytes = int(target.stat().st_size) if target.is_file() else 0
        except OSError:
            output_bytes = 0
        record: Dict[str, Any] = {
            "artifact": artifact or target.name,
            "format": format_name,
            "path": str(target),
            "elapsed_s": float(elapsed_s),
            "rows": int(rows) if rows is not None else None,
            "columns": int(columns) if columns is not None else None,
            "output_bytes": output_bytes,
            "throughput_bytes_per_s": output_bytes / float(elapsed_s) if elapsed_s > 0 else None,
            "cache_status": cache_status,
            "schema_version": schema_version,
            "status": status,
            "error": error,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
        }
        with _LOCK:
            self.records.append(record)
        return record


def set_active_collector(collector: Optional[SerializationCollector]) -> None:
    global _ACTIVE
    with _LOCK:
        _ACTIVE = collector


def active_collector() -> Optional[SerializationCollector]:
    with _LOCK:
        return _ACTIVE


def record_serialization(**kwargs: Any) -> None:
    collector = active_collector()
    if collector is not None:
        collector.record(**kwargs)


def rss_bytes() -> int:
    return _rss_bytes()


def timed_serialization(
    *,
    path: Path | str,
    artifact: str | None,
    format_name: str,
    rows: int | None = None,
    columns: int | None = None,
    cache_status: str | None = None,
    schema_version: Any = None,
):
    """Return a small context manager for instrumenting non-CSV writes."""

    class _Timer:
        def __enter__(self):
            self.started = time.perf_counter()
            self.rss_before = rss_bytes()
            return self

        def __exit__(self, exc_type, exc, tb):
            record_serialization(
                path=path,
                artifact=artifact,
                format_name=format_name,
                elapsed_s=time.perf_counter() - self.started,
                rows=rows,
                columns=columns,
                cache_status=cache_status,
                schema_version=schema_version,
                status="failed" if exc is not None else "completed",
                error=str(exc) if exc is not None else None,
                rss_before=self.rss_before,
                rss_after=rss_bytes(),
            )
            return False

    return _Timer()


__all__ = [
    "SerializationCollector",
    "active_collector",
    "record_serialization",
    "rss_bytes",
    "set_active_collector",
    "timed_serialization",
]
