# Dendrites Pipeline Layout

The dendrites workflow is now split into purpose-specific folders under `analysis/dendrites_pipeline/`.
The top-level driver stays at the root so it remains the single orchestrator, and comparison-preset batches now defer the shared poster/readback step until all required presets have finished. Shared helpers that are used by multiple workflows live in `analysis/shared/`, including the ROI split helper that powers the more-active vs less-active comparisons shared with soma/bouton.

See also: [../../README.md](../../README.md), [../../docs/dendrites_pipeline/README.md](../../docs/dendrites_pipeline/README.md), [../../docs/deprecated/main_pipeline/README.md](../../docs/deprecated/main_pipeline/README.md).

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

## Shared Helpers

- `../shared/comparison_preset_flow.py`
- `../shared/roi_split.py` - shared more-active vs less-active split helper used by the dendrites and soma/bouton pipelines
- `../shared/plots/roi_split.py` - shared ROI split figure renderer used by the dendrites and soma/bouton pipelines

## Figure and Demo Scripts

- `figures/sleep_dendrite_spine_day_figures.py`
- `demo/sleep_demo_builder.py`
- `posters/sleep_dendrite_spine_poster_common.py`
- `posters/sleep_dendrite_spine_poster_figure.py`
- `posters/sleep_dendrite_spine_spine_coactivity_poster_figure.py`

## Visual Response

- The dendrites pipeline now writes dendrite and spine visual-response summaries and boxplots under `results/dendrites_pipeline/figures/visual_response/`.
- Dendrite responsiveness is computed from dendrite cut activity only.
- Spine responsiveness is computed from spine-specific cut activity only, not from the parent dendrite label.
- The spine-specific signal is the residual after subtracting the fitted dendritic component from the spine trace, then restricting to the cut stimulus-period data.
- Visual-response metrics use the stimulus-period cut activity from `cut_intertrials/` when available, with `cut_with_intertrials/` as a fallback.
- If both `cut_intertrials/` and `cut_with_intertrials/` are missing, the pipeline prints an alert instead of silently mixing in a different cut bundle.

## ROI Split Figures

- The main pipeline writes ROI split figures under `results/dendrites_pipeline/figures/roi_split/<roi_type>/<compartment>/<split_name>/roi_split_<roi_type>_<compartment>_<split_name>.svg|png`.
- The shared renderer is implemented in `analysis/shared/plots/roi_split.py` and is reused by the soma/bouton pipeline.

## Config Files

- `sleep_dendrite_spine_example_config.json`
- `sleep_dendrite_spine_custom_demo_spec.json`

## Example

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/dendrites_pipeline.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json
```
