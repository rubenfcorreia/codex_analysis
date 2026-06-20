# Sleep Dendrite/Spine Pipeline

This directory is organized by workflow.

## Launchers

| Folder | Purpose |
| --- | --- |
| `analysis/` | Runnable scripts grouped by workflow, with `main_pipeline/` split into subfolders and separate launchers for `visual_response/` and `sleep_state_across_days/`. |

## Docs

| Folder | Purpose |
| --- | --- |
| `docs/main_pipeline/` | Main dendrite/spine analysis, folder layout, and demo builder notes. |
| `docs/visual_response/` | Visual-response launcher and movie-style metric notes. |
| `docs/methods/` | Repo-wide methods page for metrics, calculations, and statistical tests. |
| `docs/sleep_state_across_days/` | Sleep-state-only across-days notes. |

## Configs

| File | Purpose |
| --- | --- |
| `analysis/main_pipeline/sleep_dendrite_spine_example_config.json` | Example config for the main pipeline. |
| `analysis/main_pipeline/sleep_dendrite_spine_custom_demo_spec.json` | Demo recipe for the main pipeline. |
| `analysis/sleep_state_across_days/sleep_state_across_days_config.json` | Example config for the sleep-state workflow. |
