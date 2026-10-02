"""Small, pipeline-agnostic logging and timing helpers.

The helper deliberately knows nothing about analysis families.  Pipelines own
their task names and may attach pipeline-specific metadata to each scope.
"""

from __future__ import annotations

import datetime as _datetime
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional


@dataclass
class StepFrame:
    name: str
    task: Optional[str]
    index: Optional[int]
    total: Optional[int]
    started_at: float


_STEP_STACK: List[StepFrame] = []
_STAGE_TIMINGS: List[Dict[str, Any]] = []
_PIPELINE = "pipeline"
_PRESET = "default"
_LAST_PROGRESS: Dict[str, float] = {}


def reset_stage_timings(*, pipeline: str = "pipeline", preset: str = "default") -> None:
    """Start a fresh timing collection for one pipeline/preset run."""
    global _PIPELINE, _PRESET
    _STEP_STACK.clear()
    _STAGE_TIMINGS.clear()
    _LAST_PROGRESS.clear()
    _PIPELINE = str(pipeline)
    _PRESET = str(preset)


def get_stage_timings() -> List[Dict[str, Any]]:
    return [dict(entry) for entry in _STAGE_TIMINGS]


def current_step_prefix() -> str:
    timestamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    identity = f"{_PIPELINE}/{_PRESET}"
    if not _STEP_STACK:
        return f"[{timestamp} {identity}]"
    frame = _STEP_STACK[-1]
    progress = ""
    if frame.index is not None and frame.total is not None:
        progress = f" {frame.index}/{frame.total}"
    elif frame.index is not None:
        progress = f" {frame.index}"
    return f"[{timestamp} {identity}{progress}]"


def _display_name(name: str, task: Optional[str]) -> str:
    return f"{task}: {name}" if task else str(name)


def eprint(*args: Any) -> None:
    message = " ".join(str(arg) for arg in args)
    prefix = current_step_prefix()
    if not message.startswith(prefix):
        message = f"{prefix} {message}"
    print(message, file=sys.stderr, flush=True)


def info(*args: Any) -> None:
    print(*args, flush=True)


def step_message(message: str) -> None:
    eprint(message)


@contextmanager
def step_scope(
    name: str,
    *,
    task: Optional[str] = None,
    index: Optional[int] = None,
    total: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:
    frame = StepFrame(
        name=str(name),
        task=str(task) if task else None,
        index=index,
        total=total,
        started_at=time.perf_counter(),
    )
    _STEP_STACK.append(frame)
    display_name = _display_name(frame.name, frame.task)
    step_message(f"START {display_name}")
    status = "completed"
    error_message: Optional[str] = None
    try:
        yield
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        elapsed = time.perf_counter() - frame.started_at
        step_message(f"FAIL {display_name} ({elapsed:.1f}s): {exc}")
        raise
    else:
        elapsed = time.perf_counter() - frame.started_at
        step_message(f"DONE {display_name} ({elapsed:.1f}s)")
    finally:
        elapsed = time.perf_counter() - frame.started_at
        _STEP_STACK.pop()
        record: Dict[str, Any] = {
            "pipeline": _PIPELINE,
            "comparison_preset_name": _PRESET,
            "name": frame.name,
            "task": frame.task,
            "index": frame.index,
            "total": frame.total,
            "elapsed_s": float(elapsed),
            "status": status,
            "error": error_message,
        }
        if metadata:
            record["metadata"] = dict(metadata)
        _STAGE_TIMINGS.append(record)


def step_progress(
    current: int,
    total: int,
    label: Optional[str] = None,
    *,
    every: Optional[int] = None,
) -> None:
    """Emit useful loop progress without flooding the terminal."""
    if total <= 0:
        return
    interval = every if every is not None else max(1, total // 20)
    key = "/".join((str(_PIPELINE), str(_PRESET), "/".join(frame.name for frame in _STEP_STACK)))
    previous = _LAST_PROGRESS.get(key, 0.0)
    if current not in (1, total) and current - previous < interval:
        return
    _LAST_PROGRESS[key] = float(current)
    detail = f"{current}/{total}"
    if label:
        detail = f"{detail} {label}"
    step_message(f"PROGRESS {detail}")


__all__ = [
    "current_step_prefix",
    "eprint",
    "get_stage_timings",
    "info",
    "reset_stage_timings",
    "step_message",
    "step_progress",
    "step_scope",
]
