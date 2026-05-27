#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None

import numpy as np

try:
    import matplotlib.image as mpimg
except Exception:  # pragma: no cover - optional dependency guard
    mpimg = None

try:
    from scipy.ndimage import zoom as ndi_zoom
except Exception:  # pragma: no cover - optional dependency guard
    ndi_zoom = None

try:
    from scipy.signal import fftconvolve
except Exception:  # pragma: no cover - optional dependency guard
    fftconvolve = None

try:
    from skimage.filters import gabor
except Exception:  # pragma: no cover - optional dependency guard
    gabor = None

from sleep_zebra_movies_assets import DEFAULT_STIMULUS_SOURCE_ROOT, resolve_remote_clip_path

DEFAULT_GABOR_GRID_SHAPE = (18, 9)
DEFAULT_GABOR_SIGMAS: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
DEFAULT_GABOR_THETA_COUNT = 8
DEFAULT_VISUAL_COVERAGE = (-135.0, 45.0, 34.0, -34.0)
DEFAULT_ANALYSIS_COVERAGE = (-135.0, 0.0, 34.0, -34.0)


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return float(value)
        except Exception:
            return None
    return None


def extract_movie_name_from_row(row: Dict[str, Any]) -> Optional[str]:
    movie_names: List[str] = []
    for field, value in row.items():
        if not field.endswith("_name"):
            continue
        prefix = field[:-5]
        stim_type = str(row.get(f"{prefix}_type") or "").strip().lower()
        if stim_type and stim_type not in {"movie", "zebra", "clip"}:
            continue
        name = str(value or "").strip()
        if name:
            movie_names.append(name)
    if not movie_names:
        return None
    if len(movie_names) > 1:
        # The lab data-access guidance says we should not guess when multiple movie
        # features are present on a single trial.
        raise ValueError(
            "Multiple movie features found on one trial row; "
            f"cannot disambiguate {movie_names!r}"
        )
    return movie_names[0]


def group_movie_trials(trial_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for trial_index, row in enumerate(trial_rows):
        try:
            movie_name = extract_movie_name_from_row(row)
        except ValueError:
            continue
        if not movie_name:
            continue
        grouped[movie_name].append(int(trial_index))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _gabor_frequency_for_sigma(sigma: float) -> float:
    # Match the linear sigma-frequency relation used by the WavEn GUI when
    # frequency is not specified independently.
    frequency = (-0.016 * float(sigma)) + 0.148
    return float(max(0.01, frequency))


def _fallback_gabor_response(frame: np.ndarray, theta: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    if fftconvolve is None:
        raise RuntimeError("scipy.signal.fftconvolve is required to compute fallback Gabor summaries")
    frequency = _gabor_frequency_for_sigma(float(sigma))
    radius = max(3, int(np.ceil(4.0 * float(sigma))))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    x_theta = xx * cos_theta + yy * sin_theta
    y_theta = -xx * sin_theta + yy * cos_theta
    gaussian = np.exp(-0.5 * (np.square(x_theta) + np.square(y_theta)) / max(float(sigma) ** 2, 1e-6))
    phase = 2.0 * np.pi * frequency * x_theta
    real_kernel = gaussian * np.cos(phase)
    imag_kernel = gaussian * np.sin(phase)
    real_kernel = real_kernel - float(np.mean(real_kernel))
    imag_kernel = imag_kernel - float(np.mean(imag_kernel))
    frame = np.asarray(frame, dtype=np.float32)
    real = fftconvolve(frame, real_kernel, mode="same")
    imag = fftconvolve(frame, imag_kernel, mode="same")
    return np.asarray(real, dtype=np.float32), np.asarray(imag, dtype=np.float32)


def _load_downsampled_frame(frame_path: Path, grid_shape: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read stimulus frame: {frame_path}")
        if gray.shape != grid_shape:
            gray = cv2.resize(
                gray,
                (grid_shape[1], grid_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        frame = gray.astype(np.float32, copy=False) / 255.0
        return frame
    if mpimg is not None and ndi_zoom is not None:
        gray = mpimg.imread(str(frame_path))
        if gray is None:
            raise FileNotFoundError(f"Could not read stimulus frame: {frame_path}")
        gray = np.asarray(gray, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray[..., :3].mean(axis=-1)
        if np.nanmax(gray) > 1.0:
            gray = gray / 255.0
        if gray.shape != grid_shape:
            zoom_y = float(grid_shape[0]) / float(gray.shape[0])
            zoom_x = float(grid_shape[1]) / float(gray.shape[1])
            gray = ndi_zoom(gray, (zoom_y, zoom_x), order=1)
        return gray.astype(np.float32, copy=False)
    raise RuntimeError("cv2 or matplotlib.image/scipy.ndimage is required to compute Gabor summaries")


def _clip_frame_paths(clip_root: Path) -> List[Path]:
    frame_paths = sorted(clip_root.glob("frame-*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame-*.jpg files found under {clip_root}")
    return frame_paths


def _compute_frame_feature_map(
    frame: np.ndarray,
    theta_radians: Sequence[float],
    sigmas: Sequence[float],
) -> np.ndarray:
    feature_map = np.empty(
        frame.shape + (len(tuple(theta_radians)), len(tuple(sigmas))),
        dtype=np.float32,
    )
    for theta_index, theta in enumerate(theta_radians):
        for sigma_index, sigma in enumerate(sigmas):
            if gabor is not None:
                real, imag = gabor(
                    frame,
                    frequency=_gabor_frequency_for_sigma(float(sigma)),
                    theta=float(theta),
                    sigma_x=float(sigma),
                    sigma_y=float(sigma),
                )
            else:
                real, imag = _fallback_gabor_response(frame, float(theta), float(sigma))
            feature_map[:, :, theta_index, sigma_index] = (
                np.asarray(real, dtype=np.float32) ** 2
                + np.asarray(imag, dtype=np.float32) ** 2
            )
    return feature_map


def _finalize_correlations(
    sum_xy: np.ndarray,
    sum_x: np.ndarray,
    sum_x2: np.ndarray,
    sum_y: np.ndarray,
    sum_y2: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    if n_samples <= 1:
        return np.zeros_like(sum_xy, dtype=np.float32)
    n = float(n_samples)
    mean_x = sum_x / n
    mean_y = sum_y / n
    numerator = (sum_xy / n) - mean_x[None, ...] * mean_y[:, None, None, None, None]
    var_x = np.maximum((sum_x2 / n) - np.square(mean_x), 0.0)
    var_y = np.maximum((sum_y2 / n) - np.square(mean_y), 0.0)
    denom = np.sqrt(var_x[None, ...] * var_y[:, None, None, None, None])
    corr = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denom, out=corr, where=denom > 0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32, copy=False)


def summarize_gabor_matrix(
    gabor_matrix: np.ndarray,
    theta_radians: Sequence[float],
    sigmas: Sequence[float],
    visual_coverage: Sequence[float] = DEFAULT_VISUAL_COVERAGE,
) -> Dict[str, Any]:
    matrix = np.asarray(gabor_matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 4:
        raise ValueError(f"Expected a 4D Gabor matrix, got {matrix.shape}")
    if not np.isfinite(matrix).any():
        matrix = np.zeros_like(matrix, dtype=float)

    abs_matrix = np.abs(matrix)
    if np.isfinite(abs_matrix).any():
        best = np.unravel_index(int(np.nanargmax(abs_matrix)), abs_matrix.shape)
    else:
        best = (0, 0, 0, 0)
    x_index, y_index, orientation_index, size_index = map(int, best)

    cc_xy = matrix[:, :, orientation_index, size_index]
    cc_o = matrix[x_index, y_index, :, :]

    try:
        u, _, vh = np.linalg.svd(cc_xy, full_matrices=False)
        if u.shape[1] >= 2 and vh.shape[0] >= 2:
            sign = 1.0
            dominant_v = vh[1]
            if dominant_v.size and dominant_v[np.argmax(np.abs(dominant_v))] < 0:
                sign = -1.0
            azimuth_tuning = sign * dominant_v[::-1]
            elevation_tuning = sign * u[:, 1]
        else:
            raise np.linalg.LinAlgError("Not enough singular vectors")
    except Exception:
        azimuth_tuning = np.nanmean(cc_xy, axis=0)
        elevation_tuning = np.nanmean(cc_xy, axis=1)

    orientation_tuning = np.append(cc_o[:, size_index], cc_o[0, size_index])
    size_tuning = cc_o[orientation_index, :]

    xM, xm, yM, ym = [float(v) for v in visual_coverage]
    size_labels = [float(2.0 * float(sigma)) for sigma in sigmas]
    orientation_labels = [float(np.rad2deg(theta)) for theta in theta_radians]
    orientation_labels = orientation_labels + [180.0]

    return {
        "kind": "gabor",
        "rf2d": cc_xy.T.astype(np.float32, copy=False),
        "azimuth_tuning": np.asarray(azimuth_tuning, dtype=np.float32),
        "elevation_tuning": np.asarray(elevation_tuning, dtype=np.float32),
        "orientation_tuning": np.asarray(orientation_tuning, dtype=np.float32),
        "size_tuning": np.asarray(size_tuning, dtype=np.float32),
        "best_indices": [x_index, y_index, orientation_index, size_index],
        "peak_value": float(np.nanmax(abs_matrix)) if np.isfinite(abs_matrix).any() else 0.0,
        "visual_coverage": [xM, xm, yM, ym],
        "grid_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "theta_radians": [float(theta) for theta in theta_radians],
        "theta_degrees": orientation_labels,
        "sigmas": [float(sigma) for sigma in sigmas],
        "size_labels": size_labels,
    }


def compute_movie_gabor_summaries(
    trial_rows: Sequence[Dict[str, Any]],
    response_cut: np.ndarray,
    *,
    clip_source_root: Path = DEFAULT_STIMULUS_SOURCE_ROOT,
    grid_shape: Tuple[int, int] = DEFAULT_GABOR_GRID_SHAPE,
    sigmas: Sequence[float] = DEFAULT_GABOR_SIGMAS,
    theta_count: int = DEFAULT_GABOR_THETA_COUNT,
    visual_coverage: Sequence[float] = DEFAULT_VISUAL_COVERAGE,
    response_source: str = "auto",
    frame_stride: int = 4,
) -> Dict[str, Any]:
    if response_cut is None:
        raise ValueError("response_cut is required")

    response_cut = np.asarray(response_cut, dtype=float)
    if response_cut.ndim != 3:
        raise ValueError(f"Expected a 3D response cut array, got {response_cut.shape}")

    grouped_trials = group_movie_trials(trial_rows)
    if not grouped_trials:
        return {
            "available": False,
            "response_source": response_source,
            "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "sigmas": [float(sigma) for sigma in sigmas],
            "theta_radians": [float(i * np.pi / theta_count) for i in range(theta_count)],
            "roi_summaries": {},
            "clip_count": 0,
            "trial_count": 0,
        }

    n_rois, n_trials, n_frames = response_cut.shape
    theta_radians = [float(i * np.pi / theta_count) for i in range(theta_count)]
    n_sigmas = len(tuple(sigmas))

    sum_xy = np.zeros((n_rois, grid_shape[0], grid_shape[1], theta_count, n_sigmas), dtype=np.float64)
    sum_x = np.zeros((grid_shape[0], grid_shape[1], theta_count, n_sigmas), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    sum_y = np.zeros(n_rois, dtype=np.float64)
    sum_y2 = np.zeros(n_rois, dtype=np.float64)
    total_samples = 0

    used_clip_count = 0
    used_trial_count = 0
    for movie_name, trial_indices in grouped_trials.items():
        clip_root = resolve_remote_clip_path(movie_name, source_root=clip_source_root)
        try:
            frame_paths = _clip_frame_paths(clip_root)
        except FileNotFoundError:
            continue
        clip_frames = min(len(frame_paths), n_frames)
        if clip_frames <= 0:
            continue
        used_clip_count += 1
        used_trial_count += len(trial_indices)
        trial_indices = [index for index in trial_indices if 0 <= index < n_trials]
        if not trial_indices:
            continue
        sample_step = max(int(frame_stride), 1)
        sampled_frame_paths = frame_paths[:clip_frames:sample_step]
        if not sampled_frame_paths:
            continue
        response_group = np.asarray(response_cut[:, trial_indices, :clip_frames:sample_step], dtype=np.float64)
        if response_group.ndim != 3:
            continue
        sampled_frame_count = min(len(sampled_frame_paths), response_group.shape[2])
        if sampled_frame_count <= 0:
            continue
        sampled_frame_paths = sampled_frame_paths[:sampled_frame_count]
        response_group = response_group[:, :, :sampled_frame_count]

        clip_resp_sum = response_group.sum(axis=1)
        clip_resp_sq_sum = np.square(response_group).sum(axis=(1, 2))
        sum_y += clip_resp_sum.sum(axis=1)
        sum_y2 += clip_resp_sq_sum

        for frame_index, frame_path in enumerate(sampled_frame_paths[:sampled_frame_count]):
            frame = _load_downsampled_frame(frame_path, grid_shape)
            feature_map = _compute_frame_feature_map(frame, theta_radians, sigmas)
            frame_resp = clip_resp_sum[:, frame_index]
            sum_xy += frame_resp[:, None, None, None, None] * feature_map[None, ...]
            sum_x += feature_map * len(trial_indices)
            sum_x2 += np.square(feature_map) * len(trial_indices)
            total_samples += len(trial_indices)

    correlations = _finalize_correlations(sum_xy, sum_x, sum_x2, sum_y, sum_y2, total_samples)
    roi_summaries: Dict[str, Any] = {}
    for roi_index in range(correlations.shape[0]):
        summary = summarize_gabor_matrix(
            correlations[roi_index],
            theta_radians=theta_radians,
            sigmas=sigmas,
            visual_coverage=visual_coverage,
        )
        summary["response_source"] = response_source
        roi_summaries[str(roi_index)] = summary

    return {
        "available": bool(roi_summaries),
        "response_source": response_source,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "sigmas": [float(sigma) for sigma in sigmas],
        "theta_radians": theta_radians,
        "visual_coverage": [float(v) for v in visual_coverage],
        "roi_summaries": roi_summaries,
        "clip_count": int(used_clip_count),
        "trial_count": int(used_trial_count),
        "sample_count": int(total_samples),
    }


def lookup_roi_summary(
    gabor_detail: Optional[Dict[str, Any]],
    local_ids: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(gabor_detail, dict):
        return None
    roi_summaries = gabor_detail.get("roi_summaries")
    if not isinstance(roi_summaries, dict) or not roi_summaries:
        return None

    keys: List[str] = []
    conversion_index = local_ids.get("conversion_index")
    if conversion_index is not None:
        keys.append(str(int(conversion_index)))
    cell_id = local_ids.get("cell_id")
    if cell_id is not None:
        keys.append(str(int(cell_id)))
    general_roi_id = local_ids.get("general_roi_id")
    if general_roi_id is not None:
        keys.append(str(general_roi_id))
    plane = local_ids.get("plane")
    plane_roi_id = local_ids.get("plane_roi_id")
    if plane is not None and plane_roi_id is not None:
        keys.append(f"plane{int(plane)}:{int(plane_roi_id)}")
        keys.append(f"{int(plane)}:{int(plane_roi_id)}")
    for key in keys:
        if key in roi_summaries:
            summary = roi_summaries[key]
            if isinstance(summary, dict):
                return summary
    return None
