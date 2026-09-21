from __future__ import annotations

from typing import Any, Optional

DEFAULT_EVENT_DETECTION_METHOD = "derivative"
EVENT_DETECTION_METHODS = ("amplitude", "derivative")

DEFAULT_VISUAL_RESPONSE_METRIC = "mean"
VISUAL_RESPONSE_METRICS = ("mean", "calcium_events")

_ACTIVE_EVENT_DETECTION_METHOD: str = DEFAULT_EVENT_DETECTION_METHOD
_ACTIVE_VISUAL_RESPONSE_METRIC: str = DEFAULT_VISUAL_RESPONSE_METRIC


def _normalize_choice(value: Any, *, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in allowed:
        return text
    raise ValueError(f"Unknown value: {value}")


def normalize_event_detection_method(value: Any, default: str = DEFAULT_EVENT_DETECTION_METHOD) -> str:
    return _normalize_choice(value, allowed=EVENT_DETECTION_METHODS, default=default)


def normalize_visual_response_metric(value: Any, default: str = DEFAULT_VISUAL_RESPONSE_METRIC) -> str:
    text = str(value or "").strip().lower()
    if text in {"events", "calcium_event", "calcium_events", "event_frequency", "event_frequency_per_min"}:
        return "calcium_events"
    if text in {"mean", "average"}:
        return "mean"
    if not text:
        return default
    raise ValueError(f"Unknown visual response metric: {value}")


def set_active_event_detection_method(value: Any) -> str:
    global _ACTIVE_EVENT_DETECTION_METHOD
    _ACTIVE_EVENT_DETECTION_METHOD = normalize_event_detection_method(value)
    return _ACTIVE_EVENT_DETECTION_METHOD


def set_active_visual_response_metric(value: Any) -> str:
    global _ACTIVE_VISUAL_RESPONSE_METRIC
    _ACTIVE_VISUAL_RESPONSE_METRIC = normalize_visual_response_metric(value)
    return _ACTIVE_VISUAL_RESPONSE_METRIC


def get_active_event_detection_method(value: Optional[Any] = None) -> str:
    if value is None:
        return _ACTIVE_EVENT_DETECTION_METHOD
    return normalize_event_detection_method(value)


def get_active_visual_response_metric(value: Optional[Any] = None) -> str:
    if value is None:
        return _ACTIVE_VISUAL_RESPONSE_METRIC
    return normalize_visual_response_metric(value)


def visual_response_metric_field(metric: Any) -> str:
    normalized = normalize_visual_response_metric(metric)
    return "event_frequency_per_min" if normalized == "calcium_events" else "mean"


def visual_response_metric_label(metric: Any) -> str:
    normalized = normalize_visual_response_metric(metric)
    return "Calcium events" if normalized == "calcium_events" else "Mean activity"
