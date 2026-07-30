from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from analysis.compartment_common import (
    LoadedBundle,
    experiment_root_from_expid,
    find_first_key,
    mean_trace,
    pick_channel_bundle,
    pick_state_bundle,
    resolve_repo_root,
    roi_series,
    summarize_vector,
)
from analysis.shared.state_utils import derive_animal_id, derive_date, safe_filename_component


@dataclass(frozen=True)
class ExperimentContext:
    expid: str
    mode: str
    exp_root: Path
    animal_id: str
    date: str
    day_id: str
    soma_channel: int
    bouton_channel: int
    soma: LoadedBundle
    bouton: LoadedBundle
    state_bundle: Mapping[str, Any]
    state_bundle_path: Path


def build_experiment_context(expid: str, mode: str, soma_channel: int, bouton_channel: int, repo_root: Path | None = None) -> ExperimentContext:
    repo_root = repo_root or resolve_repo_root()
    animal_id = derive_animal_id(expid)
    date = derive_date(expid)
    exp_root = experiment_root_from_expid(repo_root, expid, animal_id=animal_id)
    state_bundle_path, state_bundle = pick_state_bundle(exp_root, mode)
    soma = pick_channel_bundle(exp_root, soma_channel)
    bouton = pick_channel_bundle(exp_root, bouton_channel)
    day_id = f"{safe_filename_component(animal_id)}_{safe_filename_component(date)}"
    return ExperimentContext(
        expid=expid,
        mode=mode,
        exp_root=exp_root,
        animal_id=animal_id,
        date=date,
        day_id=day_id,
        soma_channel=int(soma_channel),
        bouton_channel=int(bouton_channel),
        soma=soma,
        bouton=bouton,
        state_bundle=state_bundle,
        state_bundle_path=state_bundle_path,
    )


def experiment_summary_row(ctx: ExperimentContext) -> Dict[str, Any]:
    return {
        "expid": ctx.expid,
        "mode": ctx.mode,
        "animal_id": ctx.animal_id,
        "date": ctx.date,
        "day_id": ctx.day_id,
        "soma_channel": int(ctx.soma_channel),
        "bouton_channel": int(ctx.bouton_channel),
        "exp_root": str(ctx.exp_root),
        "state_bundle_path": str(ctx.state_bundle_path),
        "n_soma_roi": int(ctx.soma.matrix().shape[0]),
        "n_bouton_roi": int(ctx.bouton.matrix().shape[0]),
        "soma_samples": int(ctx.soma.t.size),
        "bouton_samples": int(ctx.bouton.t.size),
        "soma_sampling_hz": float(np.nan_to_num(1.0 / np.median(np.diff(ctx.soma.t))) if ctx.soma.t.size > 1 else np.nan),
        "bouton_sampling_hz": float(np.nan_to_num(1.0 / np.median(np.diff(ctx.bouton.t))) if ctx.bouton.t.size > 1 else np.nan),
    }


def shared_time_axis(ctx: ExperimentContext) -> np.ndarray:
    t = ctx.soma.t if ctx.soma.t.size >= ctx.bouton.t.size else ctx.bouton.t
    if t.size == 0:
        t = ctx.soma.t if ctx.soma.t.size else ctx.bouton.t
    return np.asarray(t, dtype=float)


def _make_scoped_unit_id(
    *,
    animal_id: Any = None,
    day_id: Any = None,
    compartment: Any = None,
    channel: Any = None,
    roi_id: Any = None,
) -> str:
    parts = [
        str(animal_id).strip() if animal_id is not None else "",
        str(day_id).strip() if day_id is not None else "",
        str(compartment).strip().lower() if compartment is not None else "",
        f"ch{int(channel)}" if channel is not None and str(channel).strip() != "" else "",
        str(roi_id).strip() if roi_id is not None and str(roi_id).strip() != "" else "",
    ]
    return "|".join(part for part in parts if part)


def make_global_soma_id(*, animal_id: Any = None, day_id: Any = None, channel: Any = None, roi_id: Any = None) -> str:
    return _make_scoped_unit_id(animal_id=animal_id, day_id=day_id, compartment="soma", channel=channel, roi_id=roi_id)


def make_global_bouton_id(*, animal_id: Any = None, day_id: Any = None, channel: Any = None, roi_id: Any = None) -> str:
    return _make_scoped_unit_id(animal_id=animal_id, day_id=day_id, compartment="bouton", channel=channel, roi_id=roi_id)


def make_unit_id(
    *,
    animal_id: Any = None,
    expid: Any = None,
    day_id: Any = None,
    compartment: Any = None,
    channel: Any = None,
    roi_id: Any = None,
    roi_index: Any = None,
) -> str:
    del expid, roi_index
    return _make_scoped_unit_id(
        animal_id=animal_id,
        day_id=day_id,
        compartment=compartment,
        channel=channel,
        roi_id=roi_id,
    )


def summarize_activity(matrix: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    if matrix.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    if mask.size == 0:
        masked = matrix
    else:
        masked = matrix[:, mask]
    flattened = masked.reshape(-1)
    return summarize_vector(flattened)

__all__ = ["ExperimentContext", "build_experiment_context", "experiment_summary_row", "shared_time_axis", "summarize_activity", "make_unit_id", "make_global_soma_id", "make_global_bouton_id"]
