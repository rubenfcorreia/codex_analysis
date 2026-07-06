# Sleep Dendrite/Spine Pipeline

This directory is organized by workflow.

## Launchers

| Folder | Purpose |
| --- | --- |
| [analysis/README.md](analysis/README.md) | Runnable scripts grouped by workflow, with shared cache/state/family helpers in `shared/`, the dendrite/spine pipeline under `dendrites_pipeline/`, the deprecated historical archive under `deprecated/main_pipeline/`, and separate launchers for `visual_response/` and `sleep_state_across_days/`. |

## Docs

| Folder | Purpose |
| --- | --- |
| [docs/dendrites_pipeline/README.md](docs/dendrites_pipeline/README.md) | Main dendrite/spine analysis, folder layout, shared helper notes, and demo builder notes. |
| [docs/deprecated/main_pipeline/README.md](docs/deprecated/main_pipeline/README.md) | Deprecated historical archive for the old main dendrite/spine pipeline docs. |
| [docs/visual_response/README.md](docs/visual_response/README.md) | Visual-response launcher and movie-style metric notes. |
| [docs/methods/README.md](docs/methods/README.md) | Repo-wide methods page for metrics, calculations, and statistical tests. |
| [docs/sleep_state_across_days/README.md](docs/sleep_state_across_days/README.md) | Sleep-state-only across-days notes. |

## Configs

| File | Purpose |
| --- | --- |
| [analysis/dendrites_pipeline/README.md](analysis/dendrites_pipeline/README.md) | Main dendrite/spine pipeline entrypoint and workflow notes. |
| `analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json` | Example config for the main pipeline. |
| `analysis/dendrites_pipeline/sleep_dendrite_spine_custom_demo_spec.json` | Demo recipe for the main pipeline. |
| `analysis/sleep_state_across_days/sleep_state_across_days_config.json` | Example config for the sleep-state workflow. |
