# Main Pipeline Layout

The main dendrite/spine workflow is now split into purpose-specific folders under `analysis/main_pipeline/`.
The top-level driver stays at the root so it remains the single orchestrator.

## Top-Level Driver

- `sleep_dendrite_spine_pipeline.py`

## Analysis Families

- `analysis_families/core.py`
- `analysis_families/state.py`
- `analysis_families/basal_apical.py`
- `analysis_families/direct_trial_type_comparison.py`
- `analysis_families/correlation.py`
- `analysis_families/matrix_similarity.py`
- `analysis_families/mixed_model.py`
- `analysis_families/spine_coactivity.py`

## Figure and Demo Scripts

- `figures/sleep_dendrite_spine_day_figures.py`
- `demo/sleep_demo_builder.py`
- `posters/sleep_dendrite_spine_poster_common.py`
- `posters/sleep_dendrite_spine_poster_figure.py`
- `posters/sleep_dendrite_spine_spine_coactivity_poster_figure.py`

## Config Files

- `sleep_dendrite_spine_example_config.json`
- `sleep_dendrite_spine_custom_demo_spec.json`

## Example

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/main_pipeline/sleep_dendrite_spine_pipeline.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/main_pipeline/sleep_dendrite_spine_example_config.json
```
