#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


ANALYSES = [
    {
        "folder": "sleep_fraction_gt_0_5",
        "csv_name": "movie_video_level_summary.csv",
        "metric_column": "mean_sleep_fraction",
        "rank_column": "sleep_fraction_rank",
        "label": "sleep_fraction",
    },
    {
        "folder": "awake_fraction_gt_0_5",
        "csv_name": "movie_video_awake_fraction_summary.csv",
        "metric_column": "mean_awake_fraction",
        "rank_column": "awake_fraction_rank",
        "label": "awake_fraction",
    },
    {
        "folder": "post_clip_wake_up_gt_0_5",
        "csv_name": "movie_video_post_clip_wake_up_summary.csv",
        "metric_column": "mean_post_clip_active_wake_fraction",
        "rank_column": "post_clip_wake_up_rank",
        "label": "post_clip_wake_up",
    },
    {
        "folder": "onset_awake_increase_gt_0_5",
        "csv_name": "movie_video_onset_awake_increase_summary.csv",
        "metric_column": "mean_onset_minus_prev_trial_tail_awake_fraction",
        "rank_column": "onset_awake_increase_rank",
        "label": "onset_awake_increase",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cinematic clip .npy files to mp4 for the thresholded "
            "movie analysis results."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/rubencorreia/code/codex_analysis/results/sleep_state_across_days"),
        help="Directory containing the per-video CSV results.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("/home/adamranson/data/vid_for_decoder/cinematic_clips"),
        help="Directory containing the source .npy clip files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root folder for exported videos. Defaults to "
            "<results-dir>/video_exports."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Strict cutoff applied to the selected metric values.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate to use for the restored mp4 files.",
    )
    return parser.parse_args()


def load_selected_rows(
    csv_path: Path,
    metric_column: str,
    rank_column: str,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_value_raw = row.get(metric_column, "")
            if metric_value_raw in ("", None):
                continue
            try:
                metric_value = float(metric_value_raw)
            except ValueError:
                continue
            if metric_value <= threshold:
                continue
            rank_raw = row.get(rank_column, "")
            try:
                rank_value = int(float(rank_raw))
            except ValueError:
                rank_value = 10**9
            rows.append(
                {
                    "rank": rank_value,
                    "metric_value": metric_value,
                    "trial_category": row.get("trial_category", ""),
                    "video_id": str(row.get("video_id", "")).strip(),
                    "video_name": row.get("video_name", ""),
                    "n_trials": row.get("n_trials", ""),
                    "n_animals": row.get("n_animals", ""),
                    "n_expids": row.get("n_expids", ""),
                    "n_after_blank_or_zebra_trials": row.get(
                        "n_after_blank_or_zebra_trials", ""
                    ),
                }
            )
    rows.sort(key=lambda item: (item["rank"], item["video_id"]))
    return rows


def clip_to_frames(clip: np.ndarray) -> Iterable[np.ndarray]:
    if clip.ndim != 3:
        raise ValueError(f"Expected a 3D clip array, got shape {clip.shape!r}")
    if clip.shape[0] == 0 or clip.shape[1] == 0 or clip.shape[2] == 0:
        raise ValueError(f"Empty clip array: shape {clip.shape!r}")

    for frame_index in range(clip.shape[2]):
        frame = np.asarray(clip[:, :, frame_index])
        if np.issubdtype(frame.dtype, np.floating):
            frame_u8 = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
        yield np.repeat(frame_u8[:, :, None], 3, axis=2)


def export_mp4(source_npy: Path, output_mp4: Path, fps: int) -> None:
    clip = np.load(source_npy, mmap_mode="r")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_mp4,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for frame in clip_to_frames(clip):
            writer.append_data(frame)


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "analysis",
        "metric_label",
        "threshold",
        "export_status",
        "rank",
        "metric_value",
        "trial_category",
        "video_id",
        "video_name",
        "n_trials",
        "n_animals",
        "n_expids",
        "n_after_blank_or_zebra_trials",
        "source_npy",
        "output_mp4",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_analysis(
    *,
    analysis: dict[str, str],
    results_dir: Path,
    clips_dir: Path,
    output_root: Path,
    threshold: float,
    fps: int,
) -> dict[str, object]:
    csv_path = results_dir / analysis["csv_name"]
    selected_rows = load_selected_rows(
        csv_path,
        analysis["metric_column"],
        analysis["rank_column"],
        threshold,
    )

    analysis_dir = output_root / analysis["folder"]
    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    exported = 0
    missing = 0
    for item in selected_rows:
        video_id = item["video_id"]
        source_npy = clips_dir / f"{video_id}.npy"
        output_mp4 = analysis_dir / f"{video_id}.mp4"
        if not source_npy.exists():
            missing += 1
            continue
        export_mp4(source_npy, output_mp4, fps=fps)
        manifest_rows.append(
            {
                "analysis": analysis["folder"],
                "metric_label": analysis["label"],
                "threshold": threshold,
                "export_status": "exported",
                "rank": item["rank"],
                "metric_value": item["metric_value"],
                "trial_category": item["trial_category"],
                "video_id": video_id,
                "video_name": item["video_name"],
                "n_trials": item["n_trials"],
                "n_animals": item["n_animals"],
                "n_expids": item["n_expids"],
                "n_after_blank_or_zebra_trials": item["n_after_blank_or_zebra_trials"],
                "source_npy": str(source_npy),
                "output_mp4": str(output_mp4),
            }
        )
        exported += 1

    manifest_path = analysis_dir / "selection_manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    return {
        "analysis": analysis["folder"],
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "exported": exported,
        "missing": missing,
        "selected": len(selected_rows),
        "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    clips_dir = args.clips_dir
    output_root = args.output_root or (results_dir / "video_exports")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for analysis in ANALYSES:
        summary_rows.append(
            export_analysis(
                analysis=analysis,
                results_dir=results_dir,
                clips_dir=clips_dir,
                output_root=output_root,
                threshold=args.threshold,
                fps=args.fps,
            )
        )

    summary_path = output_root / "export_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "analysis",
                "csv_path",
                "manifest_path",
                "exported",
                "missing",
                "selected",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote export summary to {summary_path}")
    for row in summary_rows:
        print(
            f"{row['analysis']}: selected={row['selected']} exported={row['exported']} missing={row['missing']} manifest={row['manifest_path']}"
        )


if __name__ == "__main__":
    main()
