# Analysis Methods

This page explains how the main analysis metrics are computed in `codex_analysis` and how the statistical tests are applied.
It focuses on the dendrite/spine pipeline, then summarizes the sleep-state and zebra-movie side workflows that share the same repository.

See also: [../../README.md](../../README.md), [../../analysis/README.md](../../analysis/README.md), [../dendrites_pipeline/README.md](../dendrites_pipeline/README.md), [../visual_response/README.md](../visual_response/README.md), [../sleep_state_across_days/README.md](../sleep_state_across_days/README.md).

Visual-response note: the main pipeline now treats dendrite and spine visual-response metrics as separate cohorts, uses movie-style blank-versus-movies comparisons, and reads those metrics from cut stimulus-period data, preferring `cut_intertrials/` and falling back to `cut_with_intertrials/`.

## Main Dendrite/Spine Pipeline

### Preprocessing

- The pipeline starts from the recorded `dF/F` traces and keeps them in raw form for the main preprocessing path.
- Event detection is run directly on the cached trace being analyzed. The primary detector uses the positive first derivative of `dF/F`, centers that derivative per trace, and counts threshold crossings that stay above `+3 SD` for at least the minimum consecutive-frame window.
- The legacy amplitude detector is still cached in parallel under `event_info["methods"]["amplitude"]`; that branch uses the trace median as its baseline and a noise estimate from the raw trace itself.
- Event frequency is computed from the event count and the analyzed duration, so state-masked event rates simply reuse the same detector on the masked trace segment.
- Dendrite traces are used as raw dF/F for dendrite activity metrics.
- Spine-specific activity is computed with robust regression by subtracting the fitted dendrite contribution from the spine trace.
- In code terms, the backbone is `spine_specific = spine_trace - alpha * dendrite_trace`.
- No high-pass filter is applied to the spine-specific signal before event detection or summary calculations.
- The spine event and coactivity metrics use that residual when they need a spine-specific signal, while the raw trace is still retained for trace plots and other comparisons.
- When the same-day SpinesGUI conversion is missing for an experiment, the pipeline falls back to a same-day experiment from the same animal.
- Movie experiments keep their trial-type masks, and the state-comparison list is derived from `state_mode` plus the explicit `movie_trial_types` selection.
- When a `sleep_state.pickle` bundle is present the sleep-scoring labels are added too, so quiet periods can be split into `quiet_awake`, `nrem`, and `rem`.
- Those same masks are reused for the state-comparison, basal-vs-apical, spine coactivity, and mixed-model branches so every family sees the same pooled day-level observations.
- Observations are pooled into day-level analysis units before the summary statistics and mixed models are fit.
- Visual-response cohorts are built from cut stimulus-period activity, using `cut_intertrials/` when available and `cut_with_intertrials/` as the fallback source.
- Dendrite responsiveness is computed from dendrite cut activity only.
- Spine responsiveness is computed from spine-specific cut activity only.
- The classifier then assigns each unit to the `all`, `responsive`, or `nonresponsive` cohort for downstream plots and summaries.

### State Groups and Responsiveness Cohorts

- State groups are built from the experiment metadata before the day-level summaries are computed.
- Movie experiments use the trial table plus the locomotion threshold and any available sleep scoring to build the per-trial masks.
- Sleep experiments add the scored quiet-awake, NREM, and REM masks directly from `sleep_state.pickle` when that bundle exists.
- The state masks are then pooled at the day level, so repeated source sessions from the same day share the same state labels in the downstream tables.
- The visual-response cohort is separate from the state masks.
- It uses the cut stimulus-period activity only, with `cut_intertrials/` preferred and `cut_with_intertrials/` used as a fallback.
- Dendrite responsiveness is computed from the dendrite cut trace, while spine responsiveness is computed from the spine-specific residual trace.
- The resulting `all`, `responsive`, and `nonresponsive` cohorts are reused by the visual-response figures and the summary tables.

### Summary Metrics

- `state_comparisons.csv` and the paired state summary figures report mean activity by requested state.
- `roi_split_subject_state.csv`, `roi_split_membership.csv`, `roi_split_comparisons.csv`, and `roi_split_summary.csv` report more-active vs less-active ROI groups, ranked globally across pooled eligible ROIs with duration-weighted scores. The helper runs both activity-derived and event-frequency-derived rankings and compares the same response metrics within overall, NREM, and REM windows.
- `basal_apical_comparisons.csv` and the basal/apical figures compare the same metric between basal and apical compartments.
- Correlation analyses report Pearson `r` for:
  - dendrite activity vs wheel motion
  - dendrite activity vs pupil trace
  - spine-specific activity vs dendrite activity
- Spine-spine matrix similarity compares how the correlation structure of spine-specific activity changes across states.
- Spine coactivity computes Pearson `r` for every unordered spine-pair within each dendrite and within each state.
- The model-facing response is named `coactivity_r`, while the pair table also keeps `Fisher z` values as `coactivity_z` for optional transformed analyses.
- The coactivity summaries use `coactive = r > 0` as a descriptive persistence label.
- The mixed-model inference layer for coactivity is optional and only runs when `fit_spine_coactivity_mixed_model` is enabled. Its contrast p-values follow `mixed_model_contrast_p_source`, which defaults to `classical` and can be switched to `shuffle` when you want the shuffle-refit null instead.

### Statistical Tests

- Pairwise state and basal/apical comparisons use a normality screen on the paired values:
  - paired t-test when the paired differences look approximately normal
  - Wilcoxon signed-rank test otherwise
- The paired comparisons also compute a shuffle null by sign-flipping the within-subject differences.
- Unpaired comparisons use:
  - Welch's t-test when the data look approximately normal
  - Mann-Whitney U otherwise
- ROI split comparisons use independent-group tests on the more-active and less-active ROI groups: Welch's t-test when both groups look approximately normal, Mann-Whitney U otherwise, plus a shuffle null from permuting the group labels.
- Correlation analyses use Pearson `r` plus:
  - the classical `pearsonr` p-value
  - a shuffle p-value from circular-shift or permutation nulls
- The circular-shift nulls used by the trace-based correlation and coactivity families are built from a shared cache so the same surrogate shifts can be reused across those analyses.
- Spine-spine matrix similarity uses Pearson `r` between the upper triangles of state-specific correlation matrices, with a shuffle null made by reassigning spine vectors across the two state groups.
- Spine coactivity uses Pearson `r` on state-masked `spine_specific` traces and a circular-shift shuffle null.
- The repository treats `shuffle_p < 0.05` as the primary significance rule for the state, ROI split, correlation, matrix, and coactivity families.
- For visual-response boxplots, compare the mean cut-period activity during blank trials against the mean cut-period activity during movie trials, and only use the `cut_with_intertrials/` bundle for that metric.

### Mixed-Model Layer

- The main mixed model is split into two branches:
  - `all_state`, which uses the full requested-state table
  - `selected_state`, which re-fits the same model on the `state_comparison_states` subset
- The `state_comparison_states` list is now built primarily from `state_mode` and `movie_trial_types`, with `compare_states` kept only as a compatibility shortcut.
- Fixed effects include:
  - state
  - compartment
  - state × compartment interaction
  - any additional covariates carried by the design
- The random-effects structure starts with an animal intercept and then tries richer structures with day/session and dendrite terms when the design supports them.
- If `MixedLM` is unavailable or the fit fails to converge cleanly, the pipeline falls back to a fixed-effect least-squares approximation.
- Fixed-effect rows in `mixed_model_summary_*.csv` and `mixed_model_selected_state_summary_*.csv` report the estimate, standard error, z score, and classical p-value.
- Contrast rows in `mixed_model_contrasts.csv` and `mixed_model_contrasts_selected_state.csv` use the classical p-value as the primary test and add a shuffle p-value as robustness check. When the coactivity mixed model is disabled, those contrast files are simply not written.
- The mixed-model shuffle procedure permutes state labels within animal × day blocks before refitting the same model. The same state-label shuffle is used for both the `all_state` and `selected_state` branches.

### Main Pipeline Outputs

- `state_comparisons.csv`
- `roi_split_subject_state.csv`
- `roi_split_membership.csv`
- `roi_split_comparisons.csv`
- `roi_split_summary.csv`
- `correlations.csv`
- `matrix_similarity.csv`
- `mixed_model_summary_mean_dendrite_activity.csv`
- `mixed_model_summary_mean_spine_activity_per_dendrite.csv`
- `mixed_model_contrasts.csv`
- `mixed_model_selected_state_summary_mean_dendrite_activity.csv`
- `mixed_model_selected_state_summary_mean_spine_activity_per_dendrite.csv`
- `mixed_model_contrasts_selected_state.csv`
- `spine_coactivity_table.csv`
- `spine_coactivity_state_summary.csv`
- `spine_coactivity_pair_summary.csv`
- `spine_coactivity_animal_state_summary.csv`
- `spine_coactivity_state_agreement.csv`
- `spine_coactivity_compartment_summary.csv`
- `spine_coactivity_model_summary_coactivity_r.csv`
- `spine_coactivity_model_contrasts.csv`

## Sleep-State Across Days

- `sleep_state_across_days.py` reads only `sleep_score/sleep_state.pickle`.
- It computes state fraction, bout count, total state time, and mean bout duration for active wake, quiet wake, NREM, and REM.
- Same-day expIDs are pooled within `(animal_id, date, category)` before the figures and tables are written.
- The combined figures compare animals by within-animal day index so day-to-day trends are easier to read.
- The workflow writes CSV tables plus stacked-area, probability-vs-time, REM-latency, REM-fraction, and composition figures.
- This workflow is descriptive rather than inferential: it summarizes sleep-state structure and transitions rather than running the main pipeline's hypothesis tests.

## How The Pipeline Treats Significance

- Use `shuffle_p < 0.05` for the state comparisons, correlations, matrix similarity, and spine coactivity families.
- When `movie_expids` are present, `movie_trial_types` should be set explicitly so the compare-state list can include the intended movie categories.
- Use classical p-values for mixed-model fixed effects and mixed-model contrasts in both mixed-model branches.
- Treat `coactive = r > 0` as a descriptive flag only.
- Reuse the shared circular-shift null cache for the correlation/coactivity families, but keep state comparisons, mixed models, and matrix similarity on their own null models.
- Keep the metric choice and the null model together when interpreting a figure:
  - Pearson `r` shows effect direction and strength
  - classical p-values come from the parametric test
  - shuffle p-values come from the pipeline's permutation or circular-shift nulls
