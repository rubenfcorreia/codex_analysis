# codex_analysis index

A compact navigation map for the repo, organized the same way as the README files, with a deeper tree for `analysis/`.

## Root
- `README.md` - repo overview and workflow map.
- `INDEX.md` - this navigation file.
- `analysis/` - runnable analysis entrypoints, grouped by workflow.
- `docs/` - prose documentation and methods notes.
- `tests/` - smoke tests and validation helpers.

## Analysis Tree

### `analysis/main_pipeline/`
Main dendrite/spine workflow. The README describes this as a top-level driver plus smaller analysis-family modules.

- `README.md`
- `sleep_dendrite_spine_pipeline.py`
- `sleep_dendrite_spine_example_config.json`
- `sleep_dendrite_spine_custom_demo_spec.json`
- `analysis_families/`
  - `__init__.py`
  - `core.py`
  - `state.py`
  - `basal_apical.py`
  - `direct_trial_type_comparison.py`
  - `correlation.py`
  - `matrix_similarity.py`
  - `mixed_model.py`
  - `spine_coactivity.py`
- `demo/`
  - `__init__.py`
  - `sleep_demo_builder.py`
- `figures/`
  - `__init__.py`
  - `sleep_dendrite_spine_day_figures.py`
- `posters/`
  - `__init__.py`
  - `sleep_dendrite_spine_poster_common.py`
  - `sleep_dendrite_spine_poster_figure.py`
  - `sleep_dendrite_spine_spine_coactivity_poster_figure.py`

### `analysis/visual_response/`
Movie visual-response workflow and poster-ready visual-response plots.

- `README.md`
- `movie_visual_response.py`
- `movie_visual_response_config.json`
- `poster_ready_visual_response.py`

### `analysis/sleep_state_across_days/`
Sleep-state-only across-days workflow.

- `README.md`
- `sleep_state_across_days.py`
- `export_thresholded_cinematic_clips.py`
- `sleep_state_across_days_config.json`

### `analysis/zebra_movies/`
Present in the tree, but no tracked runnable scripts were found when this index was refreshed.

## Docs
- `docs/main_pipeline/` - main pipeline notes.
- `docs/methods/` - methods documentation.
- `docs/sleep_state_across_days/` - across-days documentation.

## Tests
- `tests/` - validation and smoke checks.

## Where To Start
- To change the main pipeline, start in `analysis/main_pipeline/sleep_dendrite_spine_pipeline.py`.
- To change one analysis family, start in `analysis/main_pipeline/analysis_families/`.
- To change movie visual-response plotting, start in `analysis/visual_response/`.
- To change sleep-state across-days behavior, start in `analysis/sleep_state_across_days/`.
- To change documentation, start in `docs/`.

## Token-Efficient Reading Order
1. Read the relevant workflow README first.
2. Open only the entrypoint for the task.
3. Open the matching subfolder or family module next.
4. Check the config file and one or two nearby helpers if needed.

## Notes
- This index is intentionally narrow: it mirrors the README structure rather than trying to summarize every file.
- The repo is workflow-oriented, so the README files are the best first map before diving into code.
