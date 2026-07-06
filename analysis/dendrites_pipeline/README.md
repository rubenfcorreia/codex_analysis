# Main Pipeline Layout

The main dendrite/spine workflow is now split into purpose-specific folders under `analysis/dendrites_pipeline/`.
The top-level driver stays at the root so it remains the single orchestrator.

## Top-Level Driver

- `dendrites_pipeline.py`

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

## Visual Response

- The main pipeline now writes dendrite and spine visual-response summaries and boxplots under `results/dendrites_pipeline/figures/visual_response/`.
- Dendrite responsiveness is computed from dendrite cut activity only.
- Spine responsiveness is computed from spine-specific cut activity only, not from the parent dendrite label.
- The spine-specific signal is the residual after subtracting the fitted dendritic component from the spine trace, then restricting to the cut stimulus-period data.
- Visual-response metrics use the stimulus-period cut activity from `cut_intertrials/` when available, with `cut_with_intertrials/` as a fallback.
- If both `cut_intertrials/` and `cut_with_intertrials/` are missing, the pipeline prints an alert instead of silently mixing in a different cut bundle.

## Config Files

- `sleep_dendrite_spine_example_config.json`
- `sleep_dendrite_spine_custom_demo_spec.json`

## Example

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/dendrites_pipeline.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json
```
