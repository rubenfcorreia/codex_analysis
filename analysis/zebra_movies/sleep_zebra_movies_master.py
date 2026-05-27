#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
CODEX_ROOT = SCRIPT_DIR.parents[2]
REPO_ROOT = SCRIPT_DIR.parents[1]
MAIN_PIPELINE_DIR = SCRIPT_DIR.parent / "main_pipeline"
for extra_path in (REPO_ROOT, MAIN_PIPELINE_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from sleep_dendrite_spine_pipeline import DEFAULT_CACHE_NAME, step_scope

DEFAULT_RESULTS_DIR = CODEX_ROOT / "results" / "zebra_movies"

WRAPPER_SCRIPT = SCRIPT_DIR / "sleep_zebra_movies_roi_wrapper.py"
GABOR_SCRIPT = SCRIPT_DIR / "sleep_zebra_gabor_postprocess.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Zebra movie workflow: ROI wrapper first, then the separate Gabor post-process."
    )
    parser.add_argument("--skip-wrapper", action="store_true", help="Skip the ROI wrapper step.")
    parser.add_argument("--skip-gabor", action="store_true", help="Skip the Gabor post-process step.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory forwarded to the wrapper and used for the Gabor cache bundle.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Override the cache path forwarded to the wrapper and used by the Gabor post-process.",
    )
    return parser


def extract_option(argv: Sequence[str], option_name: str) -> Optional[str]:
    for index, token in enumerate(argv):
        if token == option_name and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith(option_name + "="):
            return token.split("=", 1)[1]
    return None


def append_override(cmd: List[str], option_name: str, value: Optional[Path]) -> List[str]:
    if value is None:
        return cmd
    return cmd + [option_name, str(value)]


def run_command(cmd: Sequence[str]) -> None:
    print("[run] " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args, wrapper_args = parser.parse_known_args(argv)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir_text = extract_option(wrapper_args, "--output-dir") or str(DEFAULT_RESULTS_DIR)
        output_dir = Path(output_dir_text)

    cache_path = args.cache_path
    if cache_path is None:
        cache_path_text = extract_option(wrapper_args, "--cache-path") or str(output_dir / DEFAULT_CACHE_NAME)
        cache_path = Path(cache_path_text)

    if not args.skip_wrapper:
        with step_scope("zebra ROI wrapper"):
            wrapper_cmd: List[str] = [sys.executable, str(WRAPPER_SCRIPT), *wrapper_args]
            wrapper_cmd = append_override(wrapper_cmd, "--output-dir", output_dir)
            wrapper_cmd = append_override(wrapper_cmd, "--cache-path", cache_path)
            run_command(wrapper_cmd)

    if not args.skip_gabor:
        with step_scope("zebra Gabor post-process"):
            gabor_cmd: List[str] = [sys.executable, str(GABOR_SCRIPT), "--cache-path", str(cache_path), "--output-dir", str(output_dir)]
            stimulus_source_root = extract_option(wrapper_args, "--stimulus-source-root")
            if stimulus_source_root:
                gabor_cmd += ["--stimulus-source-root", stimulus_source_root]
            run_command(gabor_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
