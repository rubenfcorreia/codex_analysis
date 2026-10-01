# Main Dendrite/Spine Pipeline

Use `analysis/dendrites_pipeline/dendrites_pipeline.py` when you want the full dendrite/spine analysis from `dF/F` traces.
The day-figure helper, demo builder, and poster scripts now live in dedicated subfolders, so the workflow is easier to navigate and split into smaller pieces. The workflow keeps its caches under the dendrites results tree, so reruns with unchanged inputs can reuse the same pipeline-local intermediates. The split-first analysis now carries the ROI split membership into the mixed-model leaves, instead of using it only for separate summary tables.

See also: [../../README.md](../../README.md), [../../analysis/README.md](../../analysis/README.md), [../visual_response/README.md](../visual_response/README.md), [../methods/README.md](../methods/README.md), [../sleep_state_across_days/README.md](../sleep_state_across_days/README.md), [../deprecated/main_pipeline/README.md](../deprecated/main_pipeline/README.md).

## Current Workflow

1. Load the config and resolve repository, cache, output, and figure paths.
2. Build or reuse the source cache, analysis-table cache, analysis-results cache, and shared shuffle cache so the pipeline can reuse prior work when the source data and settings match. These caches stay inside the dendrites results tree, so repeated runs only reuse dendrites-specific intermediates.
3. Normalize the data to day-level analysis units.
   - Multiple source expIDs from the same day are pooled into one day-level unit.
   - The pooled unit keeps the source provenance, but downstream summaries are reported per day rather than per raw session.
   - Dendrite and spine IDs are normalized to pooled identifiers so repeated references to the same biological unit stay stable across caches and plots.
4. Build the state masks that power the state-group analyses.
   - Movie experiments derive trial-by-trial masks from the trial table plus the locomotion and sleep metadata.
   - When a `sleep_state.pickle` bundle is available, the quiet-state labels are refined into `quiet_awake`, `nrem`, and `rem`.
   - These masks are reused by the state-comparison, ROI split, basal-vs-apical, spine coactivity, and mixed-model branches so every family sees the same pooled day-level observations.
5. Prepare visual-response cohorts when the selected families need them.
   - Dendrite responsiveness is computed from dendrite cut activity only.
   - Spine responsiveness is computed from spine-specific cut activity only.
   - The classifier compares visual stimulus trials against blank trials using the cut stimulus-period data, preferring `cut_intertrials/` and falling back to `cut_with_intertrials/` for the visual-response metric inputs.
   - This produces the `all`, `responsive`, and `nonresponsive` cohorts that downstream families can reuse immediately.
   - Runs that only ask for a family such as `spine_coactivity` skip this cut-bundle work entirely, while `state` and `mixed_model` runs still prepare the cohorts because they need them.
6. Run the analysis families through the top-level driver.
   - Each family reads the normalized day-level cache rather than raw source files.
   - Family-specific summaries, comparisons, and figures operate on the already-pooled units and selected cohorts.
7. Write the CSV tables, JSON summaries, SVG figures, checkpoint gallery, and run report.
   - Output paths remain organized under the main pipeline results tree so `plots_only` can regenerate figures from compatible caches without recomputing the underlying analyses.

## Script Map

| Script | What it does |
| --- | --- |
| `analysis/dendrites_pipeline/dendrites_pipeline.py` | Main analysis CLI for real data and demo runs. |
| `analysis/dendrites_pipeline/figures/sleep_dendrite_spine_day_figures.py` | Rebuilds the day-level figures and checkpoint gallery from an existing cache. |
| `analysis/dendrites_pipeline/demo/sleep_demo_builder.py` | Builds and validates the synthetic demo repository. |
| `analysis/dendrites_pipeline/posters/` | Poster-generation scripts and shared poster helpers. |
| `analysis/dendrites_pipeline/analysis_families/` | Family-specific analysis runners and shared dispatcher logic. |

## Visual Response

- The main pipeline writes dendrite and spine visual-response summaries under `results/dendrites_pipeline/<branch>/<basis>/figures/visual_response/`, where `branch` is one of `pooled`, `activity_split`, `frequency_split`, or `activity_frequency_split` and `basis` is one of `all`, `nrem`, or `rem`.
- ROI split figures are written under `results/dendrites_pipeline/<branch>/<basis>/figures/roi_split/<roi_type>/<compartment>/roi_split_<roi_type>_<compartment>_<split_name>_<basis_name>.svg|png`, where `branch` is one of `activity_split`, `frequency_split`, or `activity_frequency_split` and `basis` is one of `all`, `nrem`, or `rem`.
- The shared renderer lives in `analysis/shared/plots/roi_split.py` and is reused by the soma/bouton pipeline.
- Dendrite responsiveness is computed from dendrite cut activity only.
- Spine responsiveness is computed from spine-specific cut activity only, not from the parent dendrite label.
- The spine-specific signal is the residual after subtracting the fitted dendritic component from the spine trace, then restricting to the cut stimulus-period data.
- Those metrics use the stimulus-period cut activity from `cut_intertrials/` when available, with `cut_with_intertrials/` as a fallback.
- If both `cut_intertrials/` and `cut_with_intertrials/` are missing, the loader prints an alert and skips the visual-response metric for that experiment.

## ROI Split And Mixed Models

- The branch-first split helper ranks pooled eligible ROIs globally with duration-weighted scores, then builds three split scopes: `all`, `NREM`, and `REM`.
- `activity_split` uses `more_active` / `less_active`; `frequency_split` uses `higher_frequency` / `lower_frequency`; `activity_frequency_split` uses the four activity-by-frequency quadrants.
- `NREM` and `REM` splits are computed from sleep-session rows only, so sleep and movie data are not mixed when the split is derived.
- The same split membership is then passed into the branch-first mixed-model leaves as a `split_group` factor, which lets the model compare the split categories within each state.

## Preprocessing

The preprocessing stage keeps event detection tied to the cached `dF/F` traces rather than to a separate high-pass filtered signal.

- The primary detector is the positive first derivative of the recorded trace. The derivative series is centered per trace and thresholded at `+3 SD`, and an event is counted when that derivative run stays above threshold for the minimum consecutive-frame window.
- The legacy amplitude detector is still stored in parallel under `event_info["methods"]["amplitude"]`; that branch uses the raw-trace median as its baseline and a noise estimate from the raw trace itself.
- Event frequency is the event count divided by the effective duration of the analyzed trace or state mask.
- For spines, the pipeline fits the spine trace against the paired dendrite trace with a robust bisquare regression and keeps the residual as the spine-specific signal.
- That residual is `spine_specific = spine_trace - alpha * dendrite_trace`, where `alpha` is the fitted dendritic contribution.
- No high-pass filter is applied to the spine-specific activity before event detection or state averaging.
- State masks come from the experiment metadata: movie trials use the trial table and locomotion/sleep labeling, and sleep recordings add the scored quiet-awake, NREM, and REM masks when the sleep bundle is present.
- Visual-response cohorts are computed after the state masks exist, using cut stimulus-period activity from `cut_intertrials/` when possible and `cut_with_intertrials/` as the fallback source.
- Dendrite responsiveness is measured from dendrite cut activity only, while spine responsiveness is measured from spine-specific cut activity only.
- The response classifier then assigns each unit to the `all`, `responsive`, or `nonresponsive` cohort for reuse by downstream summaries and figures.
- State-summary and mixed-model summaries reuse the cached event-info records, so they inherit whichever detector method is stored as the primary one for that observation.

## Where To Set Inputs

1. Edit `analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json` for the easiest real-data workflow.
2. Pass `--config <file>` when you run the script.
3. Override any field on the CLI if you need a one-off change.
4. If you want a single script-level place to edit defaults, use `USER_EDITABLE_DEFAULTS` near the top of `analysis/dendrites_pipeline/dendrites_pipeline.py`.
5. By default, results are written to `/home/rubencorreia/code/codex_analysis/results/dendrites_pipeline/`.

The main fields you will usually set are:

- `user_id`
- `repo_base`
- `movie_expids`
- `sleep_expids`
- `basal_expids`
- `apical_expids`
- `state_mode`
- `movie_trial_types`
- `compare_states`
- `state_comparison_states`
- `basal_apical_states`
- `channel`
- `shuffle_n`
- `locomotion_threshold`
- `output_dir`
- `cache_path` (defaults to `output_dir/cache/`)
- `analysis_tables_cache_path`
- `analysis_results_cache_path`
- `rebuild`
- `source_cache_rebuild`
- `analysis_tables_rebuild`
- `analysis_results_rebuild`
- `shared_shuffle_cache_rebuild`
- `stimulus_source_root`
- `stimulus_cache_root`

Figure exports are SVG-only, with poster-friendly text sizes and normal poster-panel sizing.
If you override the DPI, it applies only to any rasterized content embedded inside the SVG.

## Real Data

The repository layout is resolved as:

`/home/<user_id>/data/Repository/<animal_id>/<exp_id>`

Use the example config as a starting point:

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/dendrites_pipeline.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json
```

You can still override the expID lists on the command line; they are treated as source-session provenance and then pooled by day:

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/dendrites_pipeline.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_example_config.json \
  --movie-expids 2025-11-05_01_ESRC020 2026-01-26_01_ESRC023
```

## Threshold Precedence

1. `locomotion_threshold` from your config or CLI, if you set it
2. `sleep_state.pickle`'s `locomotion_threshold`, when that file exists
3. A wheel-derived fallback for movie experiments when neither of the above is available

The state-comparison list is now driven primarily by `state_mode` and `movie_trial_types`.
Set `state_mode` to `all`, `quiet`, or `active`, and give `movie_trial_types` an explicit subset of `blank`, `grating`, `zebra`, and `movies` for movie experiments.
For sleep-only runs, `state_mode` alone is enough; for mixed movie/sleep runs, the movie trial types decide which quiet/active movie labels are added to the comparison list.
The legacy `compare_states` field still works as a compatibility shortcut, but the new fields are the primary path.

For example:

```json
{
  "state_mode": "quiet",
  "movie_trial_types": ["blank", "movies"]
}
```

Use `state_comparison_states` if you want to override the derived list completely.
Use `basal_apical_states` the same way if you want to reduce the state list used for the basal-vs-apical comparison.
The main pipeline summary boxplots now follow those selected state lists, while the separate day-figure script still uses the full state set for its per-dendrite views.

Sleep expIDs keep their anatomical basal/apical assignment from the same-day source when one exists, so the compartment-separated plots still reflect the underlying dendrite type even for sleep sessions.
The downstream summaries pool same-day source sessions into day-level analysis units, so the report and plots count days rather than raw expIDs.

## Demo Mode

The default demo builds a synthetic repository with sensible defaults:

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/dendrites_pipeline.py \
  --demo \
  --output-dir /tmp/sleep_codex_demo
```

The custom demo recipe is a compact builder recipe:

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/demo/sleep_demo_builder.py build \
  --recipe /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_custom_demo_spec.json \
  --output-dir /tmp/sleep_codex_custom_demo

python3 /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/demo/sleep_demo_builder.py validate \
  --recipe /home/rubencorreia/code/codex_analysis/analysis/dendrites_pipeline/sleep_dendrite_spine_custom_demo_spec.json \
  --output-dir /tmp/sleep_codex_custom_demo
```

If you omit `--output-dir` when using `demo/sleep_demo_builder.py`, it writes under `/home/rubencorreia/code/codex_analysis/results/dendrites_pipeline/demo/`.

The recipe usually only needs:

- `repo_subdir`
- `user_id`
- `analysis_families`
- `stimulus_source_root`
- `experiments`

Each experiment entry can also toggle:

- `kind`
- `compartment`
- `mode`
- `layout`
- `responses`
- `seed`

The builder infers `trial_specs` from the response kind, so you usually only need to override them when you want a custom trial table.
`validate` rebuilds the synthetic repository and runs the full analysis pipeline in one step.
Demo figures are saved under `figures/demo/` inside the chosen output directory.

## What The Pipeline Does

- Uses already-calculated `dF/F` traces from the recordings.
- Builds unique global dendrite and spine IDs within each animal.
- Uses same-day SpinesGUI conversion fallback when needed.
- Computes spine-specific activity with robust regression.
- Splits movie data into quiet and active states.
- Fits the mixed-model summaries when the design is well-behaved, and falls back to a fixed-effect least-squares approximation when the mixed-model design is singular or the optimizer cannot converge cleanly.
- Builds branch-aware ROI split comparisons from the pooled day-level observations, using both activity-derived and event-frequency-derived rankings plus the exploratory activity×frequency quadrant split.
- Uses `sleep_state.pickle` for sleep analysis and never uses `sleep_state_sim.pickle`.
- Alerts and skips sleep-state analysis if `sleep_state.pickle` is missing.
- Saves a reloadable source cache, analysis-table cache, analysis-results cache, and shared shuffle cache.

## Methods

For the metric definitions and statistical tests behind these outputs, see the repo-wide methods page at [docs/methods/README.md](../methods/README.md).
It covers the preprocessing, state comparisons, ROI split comparisons, correlations, spine coactivity, matrix similarity, and mixed-model families.

## Analysis Logic Tree

1. Inputs and config
   - read the expID lists, state selections, channel, shuffle count, thresholds, demo flag, optional coactivity-model flag, optional coactivity-only flag, optional mixed-model-only flag, and output paths
   - pool same-day observations into day-level analysis units before summary statistics are computed
2. Cache and normalization
   - resolve repository roots
   - load or rebuild the source cache, the analysis-table cache, the analysis-results cache, and the shared shuffle cache as needed
   - normalize dendrite/spine IDs and compartment labels
   - keep source expIDs as provenance while the main analysis runs on day IDs
3. State splitting
   - movie trials become quiet and active movie/blank/grating/zebra states
   - sleep experiments use `sleep_state.pickle` to define quiet awake, active awake, NREM, and REM
  - movie experiments keep their trial-type masks, and when `sleep_state.pickle` is present they also add the sleep-scoring labels so quiet periods can be split into `quiet_awake`, `nrem`, and `rem` without losing `quiet_movies` / `active_movies`
4. Summary statistics and correlations
   - compute per-state dendrite and spine means on the pooled day units
   - test dendrite-wheel and dendrite-pupil correlations
   - test spine-dendrite coupling and spine-spine matrix similarity
5. Mixed-model layer
   - fit a single all-state model with state, compartment, and state × compartment terms
   - use the interaction terms as the primary basal-vs-apical test
   - keep shuffle-based p-values as a robustness check for the post-fit contrasts
   - choose `mixed_model_contrast_p_source: classical` for the faster model p-values, or `shuffle` to restore shuffle-refit contrasts
   - the spine-coactivity mixed model is optional and is controlled by `fit_spine_coactivity_mixed_model`
   - `spine_coactivity_only: true` (or `--spine-coactivity-only`) skips the main state/correlation/matrix analyses and reruns just the spine coactivity branch from the cache
   - `mixed_model_only: true` (or `--mixed-model-only`) skips the other analyses and reruns just the main mixed-model branch from the cache
   - demo-mode runs can be enabled from the config with `demo: true` or from the CLI with `--demo`
6. Outputs and reports
   - write the JSON cache summary, CSV tables, figures, checkpoint gallery, and the run report text file

## How To Read The Report

The generated `analysis_report.txt` is summary-first.

1. Start with `Executive summary`
   - tells you how many animals and day-level analysis units were loaded
   - also shows the source-expID count for provenance
   - flags missing sleep files and mixed-model fallback
   - points to the strongest significant result
2. Check `Results at a glance`
   - gives the tested vs significant counts and percentages for each analysis family
3. Read `ROI split comparisons`
   - shows the branch-first ROI split analysis across `all`, `NREM`, and `REM`; activity branches use `more_active` / `less_active`, frequency branches use `higher_frequency` / `lower_frequency`, and the exploratory branch uses the activity-by-frequency quadrants
4. Read `Spine-spine matrix similarity`
   - shows the basal/apical split and the positive-significant / negative-significant / non-significant counts for each selected state pair
5. Read `Model diagnostics`
   - shows the exact mixed-model equation and the random-effect structure that was actually used
6. Review `Quality / exclusions`
   - lists missing inputs, skipped states, insufficient-spine cases, and fallback reasons
7. Use the CSV files for row-level detail
   - the report summarizes the run
   - `state_comparisons.csv`, `roi_split_*.csv`, `correlations.csv`, `matrix_similarity.csv`, and `mixed_model_*.csv` hold the full rows

## Outputs

The run writes a compressed cache plus analysis files to the output directory.
If you do not set `output_dir`, the script uses `/home/rubencorreia/code/codex_analysis/results/dendrites_pipeline/`.

Typical outputs are:

- `sleep_dendrite_spine_cache.npz`
- `analysis_results.json`
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
- `demo_validation.csv` when running demo mode
- `analysis_report.txt` - summary-first run report
- `figures/basal_apical_summary.svg`
- `figures/correlation_summary.svg`
- `figures/matrix_similarity_heatmap_basal.svg`
- `figures/matrix_similarity_heatmap_apical.svg`
- `figures/state_summary/state_summary_boxplots.svg`
- `figures/state_summary/state_summary_boxplots_basal.svg`
- `figures/state_summary/state_summary_boxplots_apical.svg`
- `figures/visual_response/dendrites/<cohort>/visual_response_blank_vs_movies.svg`
- `figures/visual_response/spines/<cohort>/visual_response_blank_vs_movies.svg`
- `figures/state_summary/state_summary_boxplots_basal_vs_apical.svg`
- `figures/state_summary/state_summary_boxplots_components/` and matching `*_components/` folders for the per-panel SVG pieces used to build the combined SVGs
- `figures/<animal_id>/<compartment>/<date>/<animal_id>_<compartment>_<date>_<dendrite>_matrix_similarity_heatmap.svg` - one per dendrite, grouped into animal/compartment/date folders like the day figures
- `figures/matrix_similarity_distribution.svg` - basal/apical coefficient distributions for spine-spine state-pair comparisons when those compartments are present
- `figures/state_coverage_heatmap.svg`
- `figures/demo_validation_scatter.svg` when running demo mode

## Checkpoint Gallery

The checkpoint gallery is written to `results/dendrites_pipeline/checkpoint_examples/` and gives you one representative image for each major stage of the pipeline.
The gallery is generated for both demo and real runs whenever the relevant data exist, and the basal/apical variants are picked from the observation-level compartment labels so a dendrite can still contribute the correct anatomy even if it appears in multiple experiments.
The spine-spine coefficient distribution figure prefers basal/apical panels when those labels are present; if a dataset only has other compartment labels, it falls back to those labels so the figure still renders.
The per-dendrite spine-spine matrix heatmaps are written under `results/dendrites_pipeline/<branch>/<basis>/figures/<animal_id>/<compartment>/<date>/` instead of the checkpoint folder, so they stay grouped the same way as the day figures and are easier to browse alongside the other figures.

The gallery includes:

- `01_loading_qc_all.svg`
- `01_loading_qc_basal.svg`
- `01_loading_qc_apical.svg`
- `02_spine_regression_qc_all.svg`
- `02_spine_regression_qc_basal.svg`
- `02_spine_regression_qc_apical.svg`
- `03_state_summary_all.svg`
- `03_state_summary_basal.svg`
- `03_state_summary_apical.svg`
- `basal_apical_summary.svg`
- `05_correlation_summary_all.svg`
- `05_correlation_summary_basal.svg`
- `05_correlation_summary_apical.svg`
- `06_matrix_similarity_heatmap_basal.svg`
- `06_matrix_similarity_heatmap_apical.svg`
- `06_matrix_similarity_distribution_all.svg` - basal/apical coefficient distributions in one figure when those compartments are present
- `07_mixed_model_contrasts_all_state.svg`
- `07_mixed_model_contrasts_selected_state.svg`
- `manifest.json`, which records the checkpoint, variant, and representative experiment/dendrite for each image

When basal and apical data are present, the gallery writes separate basal/apical examples for the plots that benefit from that split.
The combined overview plots are still kept in `figures/`.

## Review Figures

The review-only plot examples are written to `review_figures/` at the repository root.
They are meant for quick inspection of plot changes without rerunning the full pipeline, so the folder stays separate from the normal checkpoint gallery.
The current review examples focus on the state-summary plots that now export a composed SVG plus per-panel SVG components, and those component files live in matching `*_components/` subfolders under `review_figures/state_summary/`.

The mixed-model CSVs now come from two interaction-aware branches:

- `mixed_model_summary_mean_dendrite_activity.csv`
- `mixed_model_summary_mean_spine_activity_per_dendrite.csv`
- `mixed_model_contrasts.csv`
- `mixed_model_selected_state_summary_mean_dendrite_activity.csv`
- `mixed_model_selected_state_summary_mean_spine_activity_per_dendrite.csv`
- `mixed_model_contrasts_selected_state.csv`

`all_state` uses the full requested-state table, while `selected_state` re-fits the same model on the existing `state_comparison_states` subset. The checkpoint gallery also includes both `07_mixed_model_contrasts_all_state.svg` and `07_mixed_model_contrasts_selected_state.svg`. The report lists the exact fixed-effect terms that were tested, the model equation used for each response, and the shuffle-based robustness checks for the post-fit contrasts.

## Notes

- Same-day experiments can share conversion data and ROI numbering.
- `basal_expids` and `apical_expids` are tags on top of the movie experiments.
- If an experiment is missing, the pipeline skips it and writes an alert instead of failing the full run.
- The config and demo files use plain JSON, so comments are stored as `_comment` fields.
- If `matplotlib` is unavailable, the pipeline still runs and simply skips figure creation.
