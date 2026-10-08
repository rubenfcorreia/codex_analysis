from __future__ import annotations

import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict


def session_name(output_dir: Path, pipeline: str) -> str:
    run_id = output_dir.name.replace("-", "_").replace(" ", "_")
    return f"demo_{run_id}_{pipeline}"


def _metadata_path(output_dir: Path) -> Path:
    return Path(output_dir) / "demo_run_metadata.json"


def _read_metadata(output_dir: Path) -> Dict[str, Any]:
    path = _metadata_path(output_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def _write_metadata(output_dir: Path, payload: Dict[str, Any]) -> None:
    _metadata_path(output_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10))


def _find_entry(payload: Dict[str, Any], session: str) -> Dict[str, Any] | None:
    for entry in reversed(payload.get("tmux", [])):
        if entry.get("session") == session:
            return entry
    return None


def launch(output_dir: Path, pipeline: str) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    config = output_dir / "configs" / f"{pipeline}_config.json"
    if not config.exists():
        raise FileNotFoundError(config)
    script_dir = "dendrites_pipeline" if pipeline == "dendrites" else "soma_bouton_pipeline"
    script = "dendrites_pipeline.py" if pipeline == "dendrites" else "soma_bouton_pipeline.py"
    command = ["python3", str(Path(__file__).resolve().parents[2] / "analysis" / script_dir / script), "--config", str(config)]
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{pipeline}.log"
    session = session_name(output_dir, pipeline)
    if subprocess.run(["tmux", "has-session", "-t", session], check=False).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    shell_command = (
        "set -o pipefail; "
        + " ".join(shlex.quote(item) for item in command)
        + " 2>&1 | tee -a "
        + shlex.quote(str(log_path))
        + "; exit " + "$" + "{PIPESTATUS[0]}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command], check=True)
    metadata = {
        "session": session,
        "pipeline": pipeline,
        "command": command,
        "cwd": str(Path(__file__).resolve().parents[2]),
        "log_path": str(log_path),
        "started_at": dt.datetime.now().isoformat(),
        "status": "running",
    }
    existing = _read_metadata(output_dir)
    existing.setdefault("tmux", []).append(metadata)
    _write_metadata(output_dir, existing)
    return metadata


def status(output_dir: Path, pipeline: str) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    session = session_name(output_dir, pipeline)
    alive = subprocess.run(["tmux", "has-session", "-t", session], check=False).returncode == 0
    payload = _read_metadata(output_dir)
    entry = _find_entry(payload, session) or {"session": session, "pipeline": pipeline}
    if alive:
        entry["status"] = "running"
    else:
        if entry.get("status") == "running":
            entry["status"] = "completed"
        if "ended_at" not in entry:
            entry["ended_at"] = dt.datetime.now().isoformat()
            for candidate in reversed(payload.get("tmux", [])):
                if candidate.get("session") == session:
                    candidate.update(entry)
                    break
            _write_metadata(output_dir, payload)
    return {**entry, "running": alive}


def stop(output_dir: Path, pipeline: str) -> None:
    output_dir = Path(output_dir).resolve()
    session = session_name(output_dir, pipeline)
    subprocess.run(["tmux", "kill-session", "-t", session], check=False)
    payload = _read_metadata(output_dir)
    entry = _find_entry(payload, session)
    if entry is not None:
        entry.update({"status": "cancelled", "ended_at": dt.datetime.now().isoformat()})
        _write_metadata(output_dir, payload)
