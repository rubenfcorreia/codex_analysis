#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is required for the real run
    plt = None

from poster_plotting import POSTER_DPI

MAIN_PIPELINE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_PIPELINE_DIR))
DEFAULT_CACHE_PATH = ROOT_DIR / "results" / "main_pipeline" / "cache" / "sleep_dendrite_spine_cache.npz"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "poster_ready"


def cm_to_inch(value_cm: float) -> float:
    return float(value_cm) / 2.54


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_numpy_pickle_compatibility() -> None:
    """Expose NumPy's legacy pickle module paths when loading cached object arrays."""
    try:
        import importlib

        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    for submodule in ("multiarray", "numeric", "umath", "_multiarray_umath", "overrides", "fromnumeric"):
        try:
            module = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)


def load_npz_cache(path: Path) -> Dict[str, Any]:
    ensure_numpy_pickle_compatibility()
    try:
        loaded = np.load(path, allow_pickle=True)
        if "cache" not in loaded:
            raise KeyError(f"{path} does not contain a 'cache' entry")
        cache_obj = loaded["cache"]
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        ensure_numpy_pickle_compatibility()
        loaded = np.load(path, allow_pickle=True)
        if "cache" not in loaded:
            raise KeyError(f"{path} does not contain a 'cache' entry")
        cache_obj = loaded["cache"]
    if isinstance(cache_obj, np.ndarray) and cache_obj.dtype == object:
        item = cache_obj.item()
        if isinstance(item, dict):
            return item
    if isinstance(cache_obj, np.ndarray) and cache_obj.size == 1:
        item = cache_obj.reshape(()).item()
        if isinstance(item, dict):
            return item
    raise TypeError(f"Could not recover cache dictionary from {path}")


def load_cache(path: Path) -> Dict[str, Any]:
    try:
        cache = load_npz_cache(path)
        if not isinstance(cache, dict):
            raise TypeError(f"Expected a cache dictionary from {path}")
        return cache
    except Exception:
        json_path = ROOT_DIR / "results" / "main_pipeline" / "analysis_results.json"
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text())
                if isinstance(payload, dict):
                    return {
                        "config": dict(payload.get("config", {})) if isinstance(payload.get("config"), dict) else {},
                        "analysis_unit": payload.get("analysis_unit"),
                        "alerts": list(payload.get("alerts", [])) if isinstance(payload.get("alerts", []), list) else [],
                        "run_parameters": dict(payload.get("run_parameters", {})) if isinstance(payload.get("run_parameters"), dict) else {},
                    }
            except Exception:
                pass
        return {}


def _load_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def load_analysis_results_payload() -> Dict[str, Any]:
    json_path = ROOT_DIR / "results" / "main_pipeline" / "analysis_results.json"
    payload: Dict[str, Any] = {}
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text())
            if isinstance(loaded, dict):
                payload = dict(loaded)
        except Exception:
            pass
    spine_coactivity = payload.get("spine_coactivity")
    if isinstance(spine_coactivity, dict) and spine_coactivity.get("table_rows"):
        return payload

    coactivity_base = ROOT_DIR / "results" / "main_pipeline"
    table_rows = _load_csv_rows(coactivity_base / "spine_coactivity_table.csv")
    if not table_rows:
        cache_path = coactivity_base / "cache" / "sleep_dendrite_spine_cache_analysis_results_cache.npz"
        if cache_path.exists():
            try:
                payload = load_npz_cache(cache_path)
                if isinstance(payload, dict):
                    if isinstance(payload.get("analysis_results"), dict):
                        return dict(payload.get("analysis_results", {}))
                    return payload
            except Exception:
                pass
        return payload if isinstance(payload, dict) else {}

    coactivity_payload = dict(spine_coactivity) if isinstance(spine_coactivity, dict) else {}
    coactivity_payload["table_rows"] = table_rows
    coactivity_payload.setdefault("pair_state_rows", list(table_rows))

    state_summary_rows = _load_csv_rows(coactivity_base / "spine_coactivity_state_summary.csv")
    if state_summary_rows:
        coactivity_payload.setdefault("state_summary_rows", state_summary_rows)
    pair_summary_rows = _load_csv_rows(coactivity_base / "spine_coactivity_pair_summary.csv")
    if pair_summary_rows:
        coactivity_payload.setdefault("pair_summary_rows", pair_summary_rows)
    compartment_summary_rows = _load_csv_rows(coactivity_base / "spine_coactivity_compartment_summary.csv")
    if compartment_summary_rows:
        coactivity_payload.setdefault("compartment_summary_rows", compartment_summary_rows)
    state_agreement_rows = _load_csv_rows(coactivity_base / "spine_coactivity_state_agreement.csv")
    if state_agreement_rows:
        coactivity_payload.setdefault("state_agreement_rows", state_agreement_rows)

    payload["spine_coactivity"] = coactivity_payload
    return payload


def _svg_dimension_to_float(value: Any) -> float:
    text = str(value).strip().lower().replace("px", "").replace("pt", "")
    try:
        return float(text)
    except Exception:
        return float("nan")


def set_svg_physical_size(svg_path: Path, width_cm: float) -> None:
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    width = _svg_dimension_to_float(root.attrib.get("width"))
    height = _svg_dimension_to_float(root.attrib.get("height"))
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0:
        return
    aspect = height / width
    root.attrib["width"] = f"{float(width_cm):.4f}cm"
    root.attrib["height"] = f"{float(width_cm) * aspect:.4f}cm"
    tree.write(str(svg_path), encoding="utf-8", xml_declaration=True)


def save_svg_figure_exact(fig: Any, path: Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=POSTER_DPI, bbox_inches=None, pad_inches=0)
    if plt is not None:
        plt.close(fig)
    return output_path


_save_svg_figure_exact = save_svg_figure_exact
