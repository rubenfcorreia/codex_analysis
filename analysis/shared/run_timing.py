"""Process-level wall-clock timing shared by the analysis pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json
import time


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunTimer:
    pipeline: str
    config_path: Optional[str] = None
    output_root: Optional[str] = None
    mode: Optional[str] = None
    preset_names: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now_iso)
    _started_monotonic: float = field(default_factory=time.perf_counter, repr=False)

    def finish(self, *, exit_status: str, error: Optional[BaseException] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pipeline": self.pipeline,
            "config_path": self.config_path,
            "output_root": self.output_root,
            "mode": self.mode,
            "preset_names": list(self.preset_names),
            "run_started_at": self.started_at,
            "run_finished_at": utc_now_iso(),
            "wall_time_s": round(time.perf_counter() - self._started_monotonic, 6),
            "exit_status": exit_status,
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
        return payload


def write_timing_report(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def attach_timing_to_outputs(root: Path, payload: Mapping[str, Any]) -> None:
    """Persist timing beside run outputs and augment any existing manifests/reports."""
    root = Path(root)
    write_timing_report(root / "run_timing.json", payload)
    timing_path = root / "timing_report.json"
    timing_payload: Dict[str, Any] = {}
    if timing_path.exists():
        try:
            loaded = json.loads(timing_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                timing_payload = loaded
        except (OSError, ValueError, TypeError):
            timing_payload = {}
    timing_payload.update(dict(payload))
    write_timing_report(timing_path, timing_payload)
    for manifest_path in (root / "manifest.json", root / "summary" / "manifest.json", root / "analysis" / "general" / "manifest.json", root / "analysis" / "general" / "summary" / "manifest.json"):
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        manifest["run_timing"] = dict(payload)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    batch_manifest_path = root / "batch_manifest.json"
    if not batch_manifest_path.exists() and payload.get("scope") == "preset_batch":
        batch_manifest_path.write_text(json.dumps({
            "output_root": str(root),
            "run_timing": dict(payload),
            "preset_names": list(payload.get("resolved_preset_names") or payload.get("preset_names") or []),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = root / "analysis_report.txt"
    if report_path.exists():
        lines = report_path.read_text(encoding="utf-8", errors="replace")
        marker = "\nRun timing\n==========\n"
        if "\nRun timing\n" not in lines:
            timing_lines = [
                marker,
                f"run_started_at: {payload.get('run_started_at')}\n",
                f"run_finished_at: {payload.get('run_finished_at')}\n",
                f"wall_time_s: {payload.get('wall_time_s')}\n",
                f"exit_status: {payload.get('exit_status')}\n",
            ]
            if payload.get("error_type"):
                timing_lines.append(f"error: {payload.get('error_type')}: {payload.get('error_message')}\n")
            report_path.write_text(lines.rstrip() + "\n" + "".join(timing_lines), encoding="utf-8")
