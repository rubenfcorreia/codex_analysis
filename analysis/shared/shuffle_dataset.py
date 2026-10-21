"""Persistent, provenance-aware full-trace shuffle datasets.

The dataset is deliberately separate from the object result caches.  It stores
one independently circular-shuffled trace matrix per signal, so correlation
nulls can reuse the same shuffled ROI traces across all pairs and states.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

SCHEMA_VERSION = 1
MASK_AWARE_SCHEMA_VERSION = 1


def _key_id(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()


def _attrs_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _signal_shifts(key: str, length: int, count: int, seed: int, minimum: int) -> np.ndarray:
    if length <= 1 or count <= 0:
        return np.zeros(0, dtype=np.int32)
    minimum = max(1, min(int(minimum), length - 1))
    if minimum >= length:
        return np.zeros(0, dtype=np.int32)
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(local_seed).integers(minimum, length, size=count, dtype=np.int32)


def estimate_dataset_bytes(signal_lengths: Mapping[str, int], shuffle_n: int, dtype: str = "float32") -> int:
    return int(sum(max(0, int(length)) for length in signal_lengths.values()) * max(0, int(shuffle_n)) * np.dtype(dtype).itemsize)


class ShuffleDataset:
    """Small HDF5-backed store with atomic creation and strict metadata checks."""

    def __init__(self, path: Path | str, metadata: Mapping[str, Any], *, rebuild: bool = False, readonly: bool = False):
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised by environment validation
            raise RuntimeError("h5py is required for the full-trace shuffle dataset") from exc
        self._h5py = h5py
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.metadata["schema_version"] = SCHEMA_VERSION
        self.meta_hash = hashlib.sha256(_attrs_json(self.metadata).encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.readonly = bool(readonly)
        self._handle = None
        if self.path.exists() and not rebuild:
            try:
                with h5py.File(self.path, "r") as handle:
                    if readonly and not metadata:
                        self.meta_hash = str(handle.attrs.get("meta_hash", ""))
                    elif str(handle.attrs.get("meta_hash", "")) != self.meta_hash:
                        raise ValueError("shuffle dataset metadata mismatch")
                    if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                        raise ValueError("shuffle dataset schema mismatch")
                    if str(handle.attrs.get("complete", "false")).lower() != "true":
                        raise ValueError(f"SHUFFLE DATASET: incomplete build rejected | path={self.path}")
            except (OSError, ValueError):
                raise

    def _open(self):
        if self._handle is None:
            self._handle = self._h5py.File(self.path, "r" if self.readonly else "a")
            if self.readonly:
                return self._handle
            self._handle.attrs.setdefault("schema_version", SCHEMA_VERSION)
            self._handle.attrs.setdefault("meta_hash", self.meta_hash)
            self._handle.attrs.setdefault("metadata", _attrs_json(self.metadata))
            self._handle.attrs["complete"] = "false"
            self._handle.require_group("signals")
        return self._handle

    def ensure_signal(self, key: str, trace: np.ndarray, shuffle_n: int, *, seed: int, min_shift_frames: int = 1) -> str:
        if self.readonly:
            raise RuntimeError("cannot add signals to a read-only shuffle dataset")
        trace = np.asarray(trace, dtype=np.float32).reshape(-1)
        handle = self._open()
        signal_id = _key_id(key)
        group = handle["signals"].require_group(signal_id)
        if "traces" in group:
            dataset = group["traces"]
            if tuple(dataset.shape) != (int(shuffle_n), int(trace.size)):
                raise ValueError(f"shuffle signal shape mismatch for {key}")
            return signal_id
        required_bytes = int(shuffle_n) * int(trace.size) * np.dtype("float32").itemsize
        free_bytes = int(shutil.disk_usage(self.path.parent).free)
        if required_bytes > free_bytes:
            raise OSError(f"insufficient disk space for shuffle signal {key}: need {required_bytes} bytes, have {free_bytes}")
        shifts = _signal_shifts(key, trace.size, int(shuffle_n), int(seed), int(min_shift_frames))
        traces = np.empty((int(shuffle_n), int(trace.size)), dtype=np.float32)
        for index, shift in enumerate(shifts):
            traces[index] = np.roll(trace, int(shift))
        group.create_dataset(
            "traces", data=traces, chunks=(max(1, min(8, int(shuffle_n))), max(1, min(4096, int(trace.size)))), compression="gzip", compression_opts=4,
        )
        group.attrs["key"] = str(key)
        group.attrs["length"] = int(trace.size)
        group.attrs["shuffle_n"] = int(shuffle_n)
        group.attrs["seed"] = int(seed)
        group.attrs["min_shift_frames"] = int(min_shift_frames)
        group.attrs["source_signature"] = hashlib.sha256(np.ascontiguousarray(trace).view(np.uint8).tobytes()).hexdigest()
        handle.flush()
        return signal_id

    def has_signal(self, key: str) -> bool:
        return _key_id(key) in self._open()["signals"]

    def signal_keys(self) -> list[str]:
        return [str(group.attrs.get("key", "")) for group in self._open()["signals"].values()]

    def validate_signal_inventory(
        self,
        signals: Mapping[str, np.ndarray],
        *,
        shuffle_n: int,
        seed: int,
        min_shift_frames: int,
    ) -> None:
        """Validate every persisted signal against the current source inventory."""
        handle = self._open()
        expected = {str(key): np.asarray(value, dtype=np.float32).reshape(-1) for key, value in signals.items()}
        saved = {str(group.attrs.get("key", "")): group for group in handle["signals"].values()}
        if set(saved) != set(expected):
            missing = sorted(set(expected) - set(saved))
            extra = sorted(set(saved) - set(expected))
            raise ValueError(
                "shuffle dataset signal inventory mismatch "
                f"(missing={missing[:3]}, extra={extra[:3]})"
            )
        for key, trace in expected.items():
            group = saved[key]
            dataset = group.get("traces")
            if dataset is None or tuple(dataset.shape) != (int(shuffle_n), int(trace.size)):
                raise ValueError(f"shuffle signal shape mismatch for {key}")
            if int(group.attrs.get("shuffle_n", -1)) != int(shuffle_n):
                raise ValueError(f"shuffle signal replicate mismatch for {key}")
            if int(group.attrs.get("seed", -1)) != int(seed):
                raise ValueError(f"shuffle signal seed mismatch for {key}")
            if int(group.attrs.get("min_shift_frames", -1)) != int(min_shift_frames):
                raise ValueError(f"shuffle signal minimum-shift mismatch for {key}")
            source_signature = hashlib.sha256(np.ascontiguousarray(trace).view(np.uint8).tobytes()).hexdigest()
            if str(group.attrs.get("source_signature", "")) != source_signature:
                raise ValueError(f"shuffle signal source mismatch for {key}")

    def validate_requested_keys(self, keys: list[str] | tuple[str, ...]) -> None:
        """Fail early with requested canonical keys and the dataset path."""
        def flatten(value):
            if isinstance(value, (tuple, list)):
                for item in value:
                    yield from flatten(item)
            else:
                yield str(value)

        requested = list(flatten(keys))
        missing = [key for key in requested if not self.has_signal(key)]
        if missing:
            raise KeyError(
                "SHUFFLE DATASET: missing signal(s) before correlation | "
                f"path={self.path} | keys={missing[:5]} | "
                "action=rebuild_dataset_from_current_source_cache"
            )

    def read_signal(self, key: str, count: Optional[int] = None) -> np.ndarray:
        handle = self._open()
        group = handle["signals"][_key_id(key)]
        dataset = group["traces"]
        return np.asarray(dataset[: int(count) if count is not None else None], dtype=float)

    def close(self, *, complete: bool = True) -> None:
        if self._handle is not None:
            if not self.readonly:
                self._handle.attrs["complete"] = "true" if complete else "false"
                self._handle.flush()
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "ShuffleDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(complete=exc_type is None)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_shuffle_dataset_once(
    path: Path | str,
    metadata: Mapping[str, Any],
    signals: Mapping[str, np.ndarray],
    *,
    shuffle_n: int,
    seed: int,
    min_shift_frames: int = 1,
    pipeline: str,
    preset_count: int,
    log=print,
) -> dict[str, Any]:
    """Build or validate one complete dataset before preset analysis starts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = {str(key): np.asarray(value, dtype=float).reshape(-1) for key, value in signals.items()}
    duplicate_count = len(canonical) - len(set(canonical))
    log(f"SHUFFLE DATASET: build start | pipeline={pipeline} | presets={int(preset_count)}")
    log(f"SHUFFLE DATASET: source inventory | signals={len(signals)} | unique_keys={len(canonical)} | duplicate_keys={duplicate_count}")
    if duplicate_count:
        raise ValueError(f"duplicate canonical shuffle keys: {duplicate_count}")
    metadata = dict(metadata)
    source_digest = hashlib.sha256()
    for key, trace in sorted(canonical.items()):
        source_digest.update(str(key).encode("utf-8"))
        source_digest.update(np.ascontiguousarray(trace, dtype=np.float32).view(np.uint8).tobytes())
    metadata.update({"shuffle_n": int(shuffle_n), "shuffle_seed": int(seed), "min_shift_frames": int(min_shift_frames), "pipeline": str(pipeline), "source_signature": source_digest.hexdigest()})
    required = estimate_dataset_bytes({key: value.size for key, value in canonical.items()}, shuffle_n)
    free = shutil.disk_usage(target.parent).free
    log(f"SHUFFLE DATASET: storage estimate | required_bytes={required} | free_bytes={free}")
    if required > free:
        raise OSError(f"insufficient disk space for {pipeline} shuffle dataset: need {required}, free {free}")
    if target.exists():
        try:
            with ShuffleDataset(target, metadata, readonly=True) as existing:
                existing.validate_signal_inventory(canonical, shuffle_n=int(shuffle_n), seed=int(seed), min_shift_frames=int(min_shift_frames))
                payload = {"path": str(target), "sha256": _file_sha256(target), "signals": len(canonical), "replicates": int(shuffle_n), "source_signature": str(metadata.get("source_signature", "")), "status": "reused"}
                log(f"SHUFFLE DATASET: build complete | path={target} | signals={len(canonical)} | replicates={int(shuffle_n)} | sha256={payload['sha256']}")
                return payload
        except (OSError, ValueError, KeyError) as exc:
            log(f"SHUFFLE DATASET: validation failure | reason=stale_or_incompatible | action=rebuild | detail={exc}")
    temporary = target.with_name(f".{target.name}.building")
    if temporary.exists():
        log(f"SHUFFLE DATASET: incomplete build rejected | path={temporary}")
        temporary.unlink()
    dataset = ShuffleDataset(temporary, metadata, rebuild=True)
    dataset._open()
    try:
        for index, (key, trace) in enumerate(sorted(canonical.items()), start=1):
            dataset.ensure_signal(key, trace, int(shuffle_n), seed=int(seed), min_shift_frames=int(min_shift_frames))
            if index == 1 or index == len(canonical) or index % 25 == 0:
                log(f"SHUFFLE DATASET: signal build progress | {index}/{len(canonical)} | key={key}")
        dataset.close(complete=True)
        os.replace(temporary, target)
    except Exception:
        dataset.close(complete=False)
        raise
    payload = {"path": str(target), "sha256": _file_sha256(target), "signals": len(canonical), "replicates": int(shuffle_n), "source_signature": str(metadata.get("source_signature", "")), "status": "built"}
    log(f"SHUFFLE DATASET: build complete | path={target} | signals={len(canonical)} | replicates={int(shuffle_n)} | sha256={payload['sha256']}")
    return payload


def prepare_shuffle_dataset(path: Path | str, metadata: Mapping[str, Any], *, rebuild: bool = False) -> ShuffleDataset:
    """Open an existing valid dataset or atomically create a new one."""
    target = Path(path)
    if target.exists() and not rebuild:
        return ShuffleDataset(target, metadata)
    temporary = target.with_name(f".{target.name}.building")
    if temporary.exists():
        temporary.unlink()
    dataset = ShuffleDataset(temporary, metadata, rebuild=True)
    original_close = dataset.close

    def close_and_publish(*, complete: bool = True) -> None:
        original_close(complete=complete)
        if complete:
            os.replace(temporary, target)

    dataset.close = close_and_publish  # type: ignore[method-assign]
    return dataset


def validate_shuffle_preset_overrides(presets, *, log=print) -> None:
    """Reject per-preset shuffle settings; one pipeline-wide artifact owns them."""
    names = {
        "shuffle_n", "correlation_shuffle_n", "pairwise_correlation_shuffle_n",
        "correlation_shuffle_seed", "correlation_min_shift_frames",
        "correlation_method", "correlation_inference",
        "pairwise_correlation_inference",
    }
    for preset_name, overrides in presets:
        for parameter in sorted(names.intersection(dict(overrides))):
            log(f"SHUFFLE DATASET: preset override rejected | preset={preset_name} | differing_parameter={parameter}")
            raise ValueError(f"preset-level shuffle override rejected: {preset_name}:{parameter}")



def _mask_entry_id(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

def build_mask_aware_shuffle_dataset_once(path: Path | str, metadata: Mapping[str, Any], entries: Mapping[str, np.ndarray], *, shuffle_n: int, seed: int, min_shift_frames: int = 1, pipeline: str, log=print) -> dict[str, Any]:
    """Eagerly build/reuse state-masked shuffled vectors in one atomic HDF5 file."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    canonical = {str(key): np.asarray(value, dtype=np.float32).reshape(-1) for key, value in entries.items()}
    meta = dict(metadata); meta.update({"schema_version": MASK_AWARE_SCHEMA_VERSION, "pipeline": str(pipeline), "shuffle_n": int(shuffle_n), "shuffle_seed": int(seed), "min_shift_frames": int(min_shift_frames), "entry_signature": hashlib.sha256("".join(f"{k}:{hashlib.sha256(v.tobytes()).hexdigest()}" for k, v in sorted(canonical.items())).encode()).hexdigest()})
    meta_hash = hashlib.sha256(_attrs_json(meta).encode()).hexdigest()
    required = estimate_dataset_bytes({k: v.size for k, v in canonical.items()}, shuffle_n)
    log(f"MASK-AWARE DATASET: build start | pipeline={pipeline} | entries={len(canonical)} | required_bytes={required}")
    if target.exists():
        try:
            import h5py
            with h5py.File(target, "r") as h:
                if h.attrs.get("meta_hash", "") == meta_hash and int(h.attrs.get("schema_version", -1)) == MASK_AWARE_SCHEMA_VERSION and str(h.attrs.get("complete", "false")).lower() == "true":
                    log(f"MASK-AWARE DATASET: reuse | path={target} | entries={len(canonical)}")
                    return {"path": str(target), "status": "reused", "entries": len(canonical), "meta_hash": meta_hash}
        except (OSError, ValueError):
            log("MASK-AWARE DATASET: validation failure | reason=stale_or_incompatible | action=rebuild")
    import h5py
    temporary = target.with_name(f".{target.name}.building")
    if temporary.exists(): temporary.unlink()
    with h5py.File(temporary, "w") as h:
        h.attrs["schema_version"] = MASK_AWARE_SCHEMA_VERSION; h.attrs["meta_hash"] = meta_hash; h.attrs["metadata"] = _attrs_json(meta); h.attrs["complete"] = "false"
        group = h.create_group("entries")
        for index, (key, trace) in enumerate(sorted(canonical.items()), 1):
            shifts = _signal_shifts(key, trace.size, int(shuffle_n), int(seed), int(min_shift_frames))
            values = np.stack([np.roll(trace, int(shift)) for shift in shifts], axis=0) if shifts.size else np.empty((0, trace.size), dtype=np.float32)
            item = group.create_group(_mask_entry_id(key)); item.create_dataset("traces", data=values, compression="gzip", compression_opts=4); item.attrs["key"] = key; item.attrs["source_signature"] = hashlib.sha256(trace.tobytes()).hexdigest(); item.attrs["length"] = trace.size
            if index == 1 or index == len(canonical) or index % 100 == 0: log(f"MASK-AWARE DATASET: progress | {index}/{len(canonical)} | key={key}")
        h.attrs["complete"] = "true"
    os.replace(temporary, target)
    return {"path": str(target), "status": "built", "entries": len(canonical), "meta_hash": meta_hash}

class MaskAwareShuffleDataset:
    def __init__(self, path: Path | str):
        import h5py
        self.path = Path(path); self._handle = h5py.File(self.path, "r")
        if int(self._handle.attrs.get("schema_version", -1)) != MASK_AWARE_SCHEMA_VERSION or str(self._handle.attrs.get("complete", "false")).lower() != "true": raise ValueError(f"invalid mask-aware dataset: {self.path}")
    def read_signal(self, key: str, count: Optional[int] = None) -> np.ndarray:
        group = self._handle["entries"][_mask_entry_id(key)]
        return np.asarray(group["traces"][: int(count) if count is not None else None], dtype=float)
    def close(self) -> None:
        self._handle.close()

__all__ = ["SCHEMA_VERSION", "ShuffleDataset", "MaskAwareShuffleDataset", "estimate_dataset_bytes", "prepare_shuffle_dataset", "build_shuffle_dataset_once", "build_mask_aware_shuffle_dataset_once", "validate_shuffle_preset_overrides"]
