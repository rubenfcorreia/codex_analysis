# Dendrites Pipeline Layout

The dendrites workflow is now split into purpose-specific folders under `analysis/dendrites_pipeline/`.
The top-level driver stays at the root so it remains the single orchestrator, and comparison-preset batches now defer the shared poster/readback step until all required presets have finished. Shared helpers that are used by multiple workflows live in `analysis/shared/`, including the branch-aware ROI split helper that powers the split-first activity and frequency comparisons shared with soma/bouton.

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
- `../shared/roi_split.py` - shared branch-aware ROI split helper used by the dendrites and soma/bouton pipelines
- `../shared/plots/roi_split.py` - shared ROI split figure renderer used by the dendrites and soma/bouton pipelines

## Figure and Demo Scripts

- `figures/sleep_dendrite_spine_day_figures.py`
- `demo/sleep_demo_builder.py`
- `posters/sleep_dendrite_spine_poster_common.py`
- `posters/sleep_dendrite_spine_poster_figure.py`
- `posters/sleep_dendrite_spine_spine_coactivity_poster_figure.py`

## Visual Response

- The dendrites pipeline now writes dendrite and spine visual-response summaries and boxplots under `results/dendrites_pipeline/<branch>/<basis>/figures/visual_response/` for the branch-first leaf trees.
- Dendrite responsiveness is computed from dendrite cut activity only.
- Spine responsiveness is computed from spine-specific cut activity only, not from the parent dendrite label.
- The spine-specific signal is the residual after subtracting the fitted dendritic component from the spine trace, then restricting to the cut stimulus-period data.
- Visual-response metrics use the stimulus-period cut activity from `cut_intertrials/` when available, with `cut_with_intertrials/` as a fallback.
- If both `cut_intertrials/` and `cut_with_intertrials/` are missing, the pipeline prints an alert instead of silently mixing in a different cut bundle.

## Progression and State-Transition Outputs

When `progression_analysis.enabled` is true, the pipeline writes dendrite and spine mean dF/F progression outputs under `results/dendrites_pipeline/general/progression/`. Each native compartment contains `blank_trial_progression.csv`, `sleep_expid_progression.csv`, `progression_summary.csv`, and a figure with blank-trial and continuous sleep-time panels. Blank trials are aligned to onset and summarized in one-second bins by default. Sleep expIDs from the same `(animal_id, date)` are ordered and concatenated without gaps into one animal-day replicate; traces are interpolated at `sleep_time_step_s` (one second by default) and truncated to the shortest valid day. The spine output includes separate raw-spine dF/F and spine-specific residual signals; plotted values are means with SEM across animal-days.

State-transition CSVs and figures are kept under `results/dendrites_pipeline/general/state_transitions/`. Their scope, window mode, and metric selection remain controlled by `transition_analysis`.
## State-transition analysis

The transition family analyzes dendrite, spine-specific, event-frequency, coincident, and noncoincident metrics around within-recording state changes. It uses the shared implementation in `analysis/shared/state_transitions.py`, so its aggregation rules match the soma/bouton pipeline.

Primary transition results use `aggregation_level=expday`. Repeated transitions are first averaged within each dendrite or spine and experiment-day, then entities are averaged within that experiment-day. Each `expID/day` contributes one paired before/after summary and one aligned trace. Figures show experiment-day traces and the grand mean ± SEM, with counts for experiment-days, animals, entities, and raw transitions.

The transition CSV outputs are grouped by state scope and window mode:

- `state_transition_events_<scope>_<window_mode>.csv` preserves raw entity-by-transition rows.
- `state_transition_comparisons_<scope>_<window_mode>.csv` contains experiment-day-balanced summaries and paired statistics.
- `state_transition_pooled_comparisons_<scope>_<window_mode>.csv` contains event-pooled diagnostic summaries.

Figures are written under `state_transitions/<scope>/<window_mode>/`; aligned trace filenames end in `_expday_trace.svg`. Primary paired statistics use experiment-days as the independent replicates.

## Paired State Summary Figures

Blank and movie comparison presets generate paired-only state summaries for Quiet Awake vs NREM and Quiet Awake vs NREM vs REM. The paired intersection is calculated independently for each ROI and state set, and duplicate entity/state observations are averaged before plotting.

Dendrite outputs are written under the poster-ready paired_state_summary/ directory with separate Basal and Apical panels. Spine outputs use the same paired-state filtering but remain single-panel.

## ROI Split Figures

- The main pipeline writes ROI split figures under `results/dendrites_pipeline/<branch>/<basis>/figures/roi_split/<roi_type>/<compartment>/roi_split_<roi_type>_<compartment>_<split_name>_<basis_name>.svg|png`, where `branch` is one of `activity_split`, `frequency_split`, or `activity_frequency_split`, `basis` is one of `all`, `nrem`, or `rem`, and the same split membership is also carried into the mixed-model leaf fits.
- The shared renderer is implemented in `analysis/shared/plots/roi_split.py` and is reused by the soma/bouton pipeline.

## Config Files

- `sleep_dendrite_spine_example_config.json`
- `sleep_dendrite_spine_custom_demo_spec.json`

Correlation settings are explicit in the example and poster-ready configs:

- `correlation_method`: currently `pearson`; unsupported methods are rejected.
- `correlation_inference`: `circular_shift` for time-series null inference or `none` for descriptive output.
- `correlation_shuffle_n`, `correlation_shuffle_seed`, and `correlation_min_shift_frames` control the circular-shift null.

Correlation and coactivity outputs retain classical p-values only as diagnostics when a shuffle/null inference is configured. Reports and figures use the stored `p_value_source`; missing or incompatible sources are reported as unavailable. Mixed-model contrasts use model-based Wald inference and do not run shuffles. Active caches are invalidated when the methodology version or statistical configuration changes.

## Example

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/dendrites_pipeline.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json
```
