#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_STIMULUS_SOURCE_ROOT = Path("/data/Remote_Repository/bv_resources/all_movie_clips_bv_sets")
DEFAULT_STIMULUS_CACHE_ROOT = Path("/home/rubencorreia/data/zebra_movies")
DEFAULT_VIDEO_CACHE_ROOT = DEFAULT_STIMULUS_CACHE_ROOT / "encoded_movies"
DEFAULT_GABOR_LIBRARY_PATH = DEFAULT_STIMULUS_CACHE_ROOT / "gabors_library.npy"
DEFAULT_STIMULUS_MANIFEST_NAME = "stimulus_cache_manifest.json"
DEFAULT_GABOR_MANIFEST_NAME = "gabor_library_manifest.json"
DEFAULT_MOVIE_FPS = 30
DEFAULT_GABOR_PARAMS: Dict[str, Any] = {
    "N_thetas": 8,
    "Sigmas": [2, 3, 4, 5, 6, 8],
    "Frequencies": [0.015, 0.04, 0.07, 0.1],
    "Phases": [0, 90],
    "NX": 135,
    "NY": 54,
}

_TRIAL_CSV_RE = re.compile(r".*_all_trials\.csv$", re.IGNORECASE)
_REMOTE_CLIP_RE = re.compile(r"all_movie_clips_bv_sets[\\/](.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class EncodedClipResult:
    source_clip_root: Path
    cached_video_path: Path
    manifest_path: Path
    frame_count: int
    frame_size: Tuple[int, int]
    codec: str
    reused: bool


@dataclass(frozen=True)
class ZebraStimulusPrepResult:
    stimulus_source_root: Path
    stimulus_cache_root: Path
    encoded_video_cache_root: Path
    gabor_library_path: Path
    gabor_manifest_path: Path
    clip_results: Tuple[EncodedClipResult, ...]
    gabor_materialized: bool
    gabor_manifest_only: bool

    @property
    def n_clips(self) -> int:
        return len(self.clip_results)

    @property
    def n_reused(self) -> int:
        return sum(1 for result in self.clip_results if result.reused)

    @property
    def n_rendered(self) -> int:
        return sum(1 for result in self.clip_results if not result.reused)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def find_trial_csv(exp_root: Path) -> Path:
    candidates = [path for path in exp_root.glob("*_all_trials.csv") if _TRIAL_CSV_RE.match(path.name)]
    if not candidates:
        raise FileNotFoundError(f"No trial CSV found under {exp_root}")
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the shortest filename if multiple CSV exports are present.
    return sorted(candidates, key=lambda item: (len(item.name), item.name))[0]


def collect_movie_clip_names(exp_root: Path) -> List[str]:
    trial_csv = find_trial_csv(exp_root)
    clip_names: List[str] = []
    seen: set[str] = set()
    with trial_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return clip_names
        name_fields = [field for field in reader.fieldnames if field.endswith("_name")]
        for row in reader:
            for field in name_fields:
                name = str(row.get(field) or "").strip()
                if not name:
                    continue
                type_field = field[:-5] + "_type"
                stim_type = str(row.get(type_field) or "").strip().lower()
                if stim_type and stim_type not in {"movie", "zebra", "clip"}:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                clip_names.append(name)
    return clip_names


def resolve_remote_clip_path(name: str, source_root: Path = DEFAULT_STIMULUS_SOURCE_ROOT) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate

    normalized = str(name).strip().replace("\\", "/")
    if normalized.lower().startswith(str(source_root).replace("\\", "/").lower()):
        candidate = Path(normalized)
        if candidate.exists():
            return candidate
        relative = candidate.relative_to(source_root)
        if len(relative.parts) >= 3:
            for index in range(1, len(relative.parts) - 1):
                part = relative.parts[index]
                if not part.isdigit():
                    continue
                collapsed = source_root / Path(*relative.parts[:index], *relative.parts[index + 1 :])
                if collapsed.exists():
                    return collapsed
        return candidate

    match = _REMOTE_CLIP_RE.search(normalized)
    if match:
        relative = Path(match.group(1))
        candidate = source_root / relative
        if candidate.exists():
            return candidate
        if len(relative.parts) >= 3:
            for index in range(1, len(relative.parts) - 1):
                part = relative.parts[index]
                if not part.isdigit():
                    continue
                collapsed = source_root / Path(*relative.parts[:index], *relative.parts[index + 1 :])
                if collapsed.exists():
                    return collapsed
        return candidate

    suffix = normalized.lstrip("/")
    relative = Path(suffix)
    candidate = source_root / relative
    if candidate.exists():
        return candidate
    if len(relative.parts) >= 3:
        for index in range(1, len(relative.parts) - 1):
            part = relative.parts[index]
            if not part.isdigit():
                continue
            collapsed = source_root / Path(*relative.parts[:index], *relative.parts[index + 1 :])
            if collapsed.exists():
                return collapsed
    return candidate


def discover_clip_roots(exp_roots: Sequence[Path], source_root: Path = DEFAULT_STIMULUS_SOURCE_ROOT) -> List[Path]:
    clip_roots: List[Path] = []
    seen: set[str] = set()
    for exp_root in exp_roots:
        if not exp_root.exists():
            continue
        try:
            clip_names = collect_movie_clip_names(exp_root)
        except FileNotFoundError:
            continue
        for clip_name in clip_names:
            clip_root = resolve_remote_clip_path(clip_name, source_root=source_root)
            clip_key = str(clip_root)
            if clip_key in seen:
                continue
            seen.add(clip_key)
            clip_roots.append(clip_root)
    return sorted(clip_roots, key=lambda path: str(path))


def _frame_paths(clip_root: Path) -> List[Path]:
    frame_paths = sorted(clip_root.glob("frame-*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame-*.jpg files found under {clip_root}")
    return frame_paths


def _frame_metadata(frame_paths: Sequence[Path]) -> Tuple[int, Tuple[int, int], int]:
    from PIL import Image

    first = frame_paths[0]
    with Image.open(first) as image:
        width, height = image.size
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in frame_paths)
    return len(frame_paths), (width, height), latest_mtime_ns


def _manifest_is_fresh(
    manifest_path: Path,
    video_path: Path,
    clip_root: Path,
    frame_count: int,
    frame_size: Tuple[int, int],
    latest_mtime_ns: int,
) -> bool:
    if not manifest_path.exists() or not video_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return False
    return (
        str(manifest.get("source_clip_root")) == str(clip_root)
        and int(manifest.get("frame_count", -1)) == int(frame_count)
        and tuple(manifest.get("frame_size", [])) == tuple(frame_size)
        and int(manifest.get("source_latest_mtime_ns", -1)) == int(latest_mtime_ns)
        and str(manifest.get("video_path")) == str(video_path)
    )


def _encode_clip_with_codec(
    clip_root: Path,
    output_path: Path,
    codec: str,
    fps: int,
    frame_paths: Sequence[Path],
    frame_size: Tuple[int, int],
) -> bool:
    import cv2

    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        writer.release()
        return False

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame {frame_path}")
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return True


def encode_clip_folder(
    clip_root: Path,
    cache_root: Path = DEFAULT_VIDEO_CACHE_ROOT,
    source_root: Path = DEFAULT_STIMULUS_SOURCE_ROOT,
    fps: int = DEFAULT_MOVIE_FPS,
) -> EncodedClipResult:
    frame_paths = _frame_paths(clip_root)
    frame_count, frame_size, latest_mtime_ns = _frame_metadata(frame_paths)
    relative_clip = Path(*clip_root.relative_to(source_root).parts) if clip_root.is_relative_to(source_root) else Path(clip_root.name)
    base_output_path = cache_root / relative_clip
    ensure_dir(base_output_path.parent)
    ensure_dir(cache_root)

    candidates = [("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]
    last_error: Optional[Exception] = None

    for codec, extension in candidates:
        output_path = base_output_path.with_suffix(extension)
        manifest_path = output_path.with_name(output_path.name + ".json")
        if _manifest_is_fresh(manifest_path, output_path, clip_root, frame_count, frame_size, latest_mtime_ns):
            manifest = load_json(manifest_path)
            return EncodedClipResult(
                source_clip_root=clip_root,
                cached_video_path=output_path,
                manifest_path=manifest_path,
                frame_count=frame_count,
                frame_size=frame_size,
                codec=str(manifest.get("codec") or codec),
                reused=True,
            )

        try:
            if output_path.exists():
                output_path.unlink()
            if _encode_clip_with_codec(clip_root, output_path, codec, fps, frame_paths, frame_size):
                manifest = {
                    "version": 1,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_clip_root": str(clip_root),
                    "source_stimulus_root": str(source_root),
                    "video_path": str(output_path),
                    "frame_count": frame_count,
                    "frame_size": list(frame_size),
                    "source_latest_mtime_ns": latest_mtime_ns,
                    "fps": fps,
                    "codec": codec,
                }
                save_json(manifest_path, manifest)
                return EncodedClipResult(
                    source_clip_root=clip_root,
                    cached_video_path=output_path,
                    manifest_path=manifest_path,
                    frame_count=frame_count,
                    frame_size=frame_size,
                    codec=codec,
                    reused=False,
                )
        except Exception as exc:  # pragma: no cover - rare codec failures
            last_error = exc
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            continue

    if last_error is not None:
        raise RuntimeError(f"Failed to encode {clip_root} into a video cache") from last_error
    raise RuntimeError(f"Failed to encode {clip_root} into a video cache")


def _gabor_library_shape(n_thetas: int, sigmas: Sequence[float], frequencies: Sequence[float], phases: Sequence[float], nx: int, ny: int) -> Tuple[int, int, int, int, int, int, int]:
    return (nx, ny, n_thetas, len(tuple(sigmas)), len(tuple(frequencies)), len(tuple(phases)), nx * ny)


def estimate_dense_gabor_library_bytes(
    n_thetas: int,
    sigmas: Sequence[float],
    frequencies: Sequence[float],
    phases: Sequence[float],
    nx: int,
    ny: int,
    dtype: np.dtype = np.dtype(np.float16),
) -> int:
    shape = _gabor_library_shape(n_thetas, sigmas, frequencies, phases, nx, ny)
    return int(np.prod(shape, dtype=np.int64) * dtype.itemsize)


def _make_gabor_filter(
    i: int,
    j: int,
    angle: float,
    sigma: float,
    phase: float,
    frequency: float,
    lx: int,
    ly: int,
) -> np.ndarray:
    from skimage.filters import gabor_kernel

    backgrd = np.zeros((lx, ly))
    gk = gabor_kernel(
        frequency=frequency,
        theta=angle,
        sigma_x=sigma,
        sigma_y=sigma,
        offset=phase,
    )
    canvas = np.ones((lx + (2 * gk.shape[0]), ly + (2 * gk.shape[1])))
    canvas[gk.shape[0] : gk.shape[0] + lx, gk.shape[1] : gk.shape[1] + ly] = backgrd
    dp = (gk.shape[0] - 1) / 2
    x = i + gk.shape[0]
    y = j + gk.shape[1]
    canvas[int(x - dp) : int(x + dp + 1), int(y - dp) : int(y + dp + 1)] = gk.real
    backgrd = canvas[gk.shape[0] : gk.shape[0] + lx, gk.shape[1] : gk.shape[1] + ly]
    return backgrd.T.astype(np.float16)


def build_dense_gabor_library(
    save_path: Path,
    n_thetas: int,
    sigmas: Sequence[float],
    frequencies: Sequence[float],
    phases: Sequence[float],
    nx: int,
    ny: int,
    allow_large: bool = False,
    size_limit_bytes: int = 8 * 1024**3,
) -> Path:
    estimate = estimate_dense_gabor_library_bytes(n_thetas, sigmas, frequencies, phases, nx, ny)
    if estimate > size_limit_bytes and not allow_large:
        raise RuntimeError(
            "Dense Gabor library would be very large "
            f"({estimate / 1024**3:.1f} GiB). "
            "Set allow_large=True to materialize it."
        )

    import numpy.lib.format as npformat

    ensure_dir(save_path.parent)
    xs = np.arange(nx)
    ys = np.arange(ny)
    thetas = np.array([(i * np.pi) / n_thetas for i in range(n_thetas)])
    sigmas_arr = np.array(sigmas)
    frequencies_arr = np.array(frequencies)
    phases_arr = np.array(phases)

    shape = _gabor_library_shape(n_thetas, sigmas_arr, frequencies_arr, phases_arr, nx, ny)
    library = npformat.open_memmap(save_path, mode="w+", dtype=np.float16, shape=shape)
    for x in xs:
        for y in ys:
            for theta_index, theta in enumerate(thetas):
                for sigma_index, sigma in enumerate(sigmas_arr):
                    for frequency_index, frequency in enumerate(frequencies_arr):
                        for phase_index, phase in enumerate(phases_arr):
                            library[x, y, theta_index, sigma_index, frequency_index, phase_index] = _make_gabor_filter(
                                int(x),
                                int(y),
                                float(theta),
                                float(sigma),
                                float(phase),
                                float(frequency),
                                nx,
                                ny,
                            )
    library.flush()
    return save_path


def build_gabor_manifest(
    cache_root: Path = DEFAULT_STIMULUS_CACHE_ROOT,
    gabor_save_path: Path = DEFAULT_GABOR_LIBRARY_PATH,
    allow_full_materialization: bool = False,
    size_limit_bytes: int = 8 * 1024**3,
) -> Dict[str, Any]:
    ensure_dir(cache_root)
    ensure_dir(gabor_save_path.parent)
    manifest_path = cache_root / DEFAULT_GABOR_MANIFEST_NAME
    estimate = estimate_dense_gabor_library_bytes(
        int(DEFAULT_GABOR_PARAMS["N_thetas"]),
        DEFAULT_GABOR_PARAMS["Sigmas"],
        DEFAULT_GABOR_PARAMS["Frequencies"],
        DEFAULT_GABOR_PARAMS["Phases"],
        int(DEFAULT_GABOR_PARAMS["NX"]),
        int(DEFAULT_GABOR_PARAMS["NY"]),
    )
    materialized = False
    if allow_full_materialization:
        build_dense_gabor_library(
            gabor_save_path,
            int(DEFAULT_GABOR_PARAMS["N_thetas"]),
            DEFAULT_GABOR_PARAMS["Sigmas"],
            DEFAULT_GABOR_PARAMS["Frequencies"],
            DEFAULT_GABOR_PARAMS["Phases"],
            int(DEFAULT_GABOR_PARAMS["NX"]),
            int(DEFAULT_GABOR_PARAMS["NY"]),
            allow_large=True,
            size_limit_bytes=size_limit_bytes,
        )
        materialized = True

    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gabor_defaults": jsonable(DEFAULT_GABOR_PARAMS),
        "gabor_library_path": str(gabor_save_path),
        "gabor_library_exists": gabor_save_path.exists(),
        "gabor_library_materialized": materialized,
        "estimated_dense_library_bytes": estimate,
    }
    save_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
    }


def prepare_zebra_stimulus_assets(
    exp_roots: Sequence[Path],
    stimulus_source_root: Path = DEFAULT_STIMULUS_SOURCE_ROOT,
    stimulus_cache_root: Path = DEFAULT_STIMULUS_CACHE_ROOT,
    gabor_save_path: Path = DEFAULT_GABOR_LIBRARY_PATH,
    build_full_gabor_library: bool = False,
) -> ZebraStimulusPrepResult:
    ensure_dir(stimulus_cache_root)
    ensure_dir(gabor_save_path.parent)

    clip_roots = discover_clip_roots(exp_roots, source_root=stimulus_source_root)
    video_cache_root = stimulus_cache_root / "encoded_movies"
    existing_clip_roots = [clip_root for clip_root in clip_roots if clip_root.exists()]
    clip_results = tuple(
        encode_clip_folder(clip_root, cache_root=video_cache_root, source_root=stimulus_source_root)
        for clip_root in existing_clip_roots
    )
    session_manifest_path = stimulus_cache_root / DEFAULT_STIMULUS_MANIFEST_NAME
    save_json(
        session_manifest_path,
        {
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stimulus_source_root": str(stimulus_source_root),
            "stimulus_cache_root": str(stimulus_cache_root),
            "gabor_library_path": str(gabor_save_path),
            "movie_clip_count": len(clip_results),
            "missing_clip_roots": [str(path) for path in clip_roots if not path.exists()],
            "movie_clip_paths": [str(result.source_clip_root) for result in clip_results],
            "cached_video_paths": [str(result.cached_video_path) for result in clip_results],
            "reused_cached_videos": [bool(result.reused) for result in clip_results],
        },
    )
    gabor_manifest = build_gabor_manifest(
        cache_root=stimulus_cache_root,
        gabor_save_path=gabor_save_path,
        allow_full_materialization=build_full_gabor_library,
    )

    return ZebraStimulusPrepResult(
        stimulus_source_root=stimulus_source_root,
        stimulus_cache_root=stimulus_cache_root,
        encoded_video_cache_root=video_cache_root,
        gabor_library_path=gabor_save_path,
        gabor_manifest_path=Path(str(gabor_manifest["manifest_path"])),
        clip_results=clip_results,
        gabor_materialized=bool(gabor_manifest.get("gabor_library_materialized")),
        gabor_manifest_only=not bool(gabor_manifest.get("gabor_library_materialized")),
    )
