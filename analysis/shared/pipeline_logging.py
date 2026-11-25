"""Small, pipeline-agnostic logging and timing helpers.

The helper deliberately knows nothing about analysis families.  Pipelines own
their task names and may attach pipeline-specific metadata to each scope.
"""

from __future__ import annotations

import datetime as _datetime
import logging
import traceback
import sys
import time
from contextlib import contextmanager
from pathlib import Path
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
_LOGGER: Optional[logging.Logger] = None
_LOG_PATH: Optional[Path] = None


class _PipelineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{timestamp} {record.levelname} [{record.name}] {record.getMessage()}"


def configure_run_logging(pipeline: str, *, log_level: str = "INFO", quiet: bool = False, log_root: Path | str = "logs", run_timestamp: Optional[str] = None) -> Path:
    global _LOGGER, _LOG_PATH
    requested_level = str(log_level).upper()
    if requested_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Unsupported log level: {log_level}")
    if _LOGGER is not None:
        _LOGGER.setLevel(getattr(logging, requested_level))
        for handler in _LOGGER.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING if quiet else getattr(logging, requested_level))
        return _LOG_PATH or Path(log_root) / str(pipeline) / "run.log"
    timestamp = run_timestamp or _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(log_root) / str(pipeline)
    output_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = output_dir / f"{timestamp}.log"
    _LOGGER = logging.getLogger(f"pipeline.{pipeline}")
    _LOGGER.setLevel(logging.DEBUG)
    _LOGGER.propagate = False
    formatter = _PipelineFormatter()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING if quiet else getattr(logging, requested_level))
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(console)
    _LOGGER.addHandler(file_handler)
    return _LOG_PATH


def log_path() -> Optional[Path]:
    return _LOG_PATH


def log_event(level: str, event: str, **fields: Any) -> None:
    message = str(event)
    if fields:
        message += " " + " ".join(f"{key}={value}" for key, value in fields.items())
    if _LOGGER is None:
        print(message, file=sys.stderr, flush=True)
    else:
        _LOGGER.log(getattr(logging, str(level).upper(), logging.INFO), message)


def log_exception(event: str, exc: BaseException, **fields: Any) -> None:
    log_event("ERROR", event, **fields, error=str(exc))
    if _LOGGER is not None:
        _LOGGER.error("traceback:\n%s", "".join(traceback.format_exception(exc)))


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
    if _LOGGER is None:
        print(message, file=sys.stderr, flush=True)
    else:
        _LOGGER.info(message)


def info(*args: Any) -> None:
    message = " ".join(str(arg) for arg in args)
    if _LOGGER is None:
        print(message, flush=True)
    else:
        _LOGGER.info(message)


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
    "configure_run_logging",
    "eprint",
    "get_stage_timings",
    "info",
    "log_event",
    "log_exception",
    "log_path",
    "reset_stage_timings",
    "step_message",
    "step_progress",
    "step_scope",
]
