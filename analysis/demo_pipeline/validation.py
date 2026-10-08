from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_truth(output_dir: Path) -> Dict[str, Any]:
    return json.loads((Path(output_dir) / "demo_truth.json").read_text(encoding="utf-8"))


def _check(condition: bool, name: str, details: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(condition), "details": details}


def validate_demo(output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    truth = _load_truth(output_dir)
    checks: List[Dict[str, Any]] = []
    checks.append(_check(bool(truth.get("states")), "states_present", str(truth.get("states"))))
    checks.append(_check(len(truth.get("dendrites", [])) > 0, "dendrites_present", str(len(truth.get("dendrites", [])))))
    checks.append(_check(len(truth.get("soma_boutons", [])) > 0, "soma_boutons_present", str(len(truth.get("soma_boutons", [])))))
    topology = truth.get("topology", {})
    checks.extend([
        _check(topology.get("bouton_1", {}).get("coupled_to") == "soma_1", "bouton_1_mapping", "bouton_1 -> soma_1"),
        _check(topology.get("bouton_2", {}).get("coupled_to") is None, "bouton_2_control", "bouton_2 independent"),
        _check(topology.get("bouton_3", {}).get("coupled_to") == "soma_2", "bouton_3_mapping", "bouton_3 -> soma_2"),
    ])
    for item in truth.get("dendrites", []):
        compartment = item["compartment"]
        expected = 2.0 if compartment == "basal" else 0.5
        checks.append(_check(np.isclose(item["activity_scale"], expected), f"{compartment}_activity_scale", str(item["activity_scale"])))
        checks.append(_check(np.isclose(item["frequency_scale"], expected), f"{compartment}_frequency_scale", str(item["frequency_scale"])))
    for item in truth.get("soma_boutons", []):
        checks.append(_check(np.isclose(float(item["somas"]["soma_1"]["activity_scale"]), 3.0), "soma_1_activity_scale", "3.0"))
        checks.append(_check(np.isclose(float(item["somas"]["soma_2"]["activity_scale"]), 1.0 / 3.0), "soma_2_activity_scale", "1/3"))
    passed = all(check["passed"] for check in checks)
    result = {"passed": passed, "checks": checks, "n_checks": len(checks)}
    validation_dir = output_dir / "observed_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (validation_dir / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "passed", "details"])
        writer.writeheader()
        writer.writerows(checks)
    (output_dir / "demo_validation_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
