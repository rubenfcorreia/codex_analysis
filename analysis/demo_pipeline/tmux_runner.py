from __future__ import annotations

import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence


def session_name(output_dir: Path, pipeline: str) -> str:
    run_id = output_dir.name.replace("-", "_").replace(" ", "_")
    return f"demo_{run_id}_{pipeline}"


def launch(output_dir: Path, pipeline: str) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    config = output_dir / "configs" / f"{pipeline}_config.json"
    if not config.exists():
        raise FileNotFoundError(config)
    script = "dendrites_pipeline.py" if pipeline == "dendrites" else "soma_bouton_pipeline.py"
    command = ["python3", str(Path(__file__).resolve().parents[2] / "analysis" / ("dendrites_pipeline" if pipeline == "dendrites" else "soma_bouton_pipeline") / script), "--config", str(config)]
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{pipeline}.log"
    session = session_name(output_dir, pipeline)
    shell_command = " ".join(shlex.quote(item) for item in command) + " 2>&1 | tee -a " + shlex.quote(str(log_path))
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command], check=True)
    metadata = {"session": session, "pipeline": pipeline, "command": command, "log_path": str(log_path), "started_at": dt.datetime.now().isoformat()}
    path = output_dir / "demo_run_metadata.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.setdefault("tmux", []).append(metadata)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return metadata


def status(output_dir: Path, pipeline: str) -> Dict[str, Any]:
    session = session_name(Path(output_dir), pipeline)
    result = subprocess.run(["tmux", "has-session", "-t", session], check=False)
    return {"session": session, "running": result.returncode == 0}


def stop(output_dir: Path, pipeline: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session_name(Path(output_dir), pipeline)], check=False)
