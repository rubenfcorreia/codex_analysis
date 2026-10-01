# Soma/Bouton Pipeline

This workflow compares soma activity from `ch2` data against axonal bouton
activity from `ch1` data. The shared preset flow lives in `analysis/shared/`, and the branch-aware ROI split helper is shared through `analysis/shared/roi_split.py`.

See also: [../README.md](../README.md), [../dendrites_pipeline/README.md](../dendrites_pipeline/README.md).

Key points:

- same-date experiments are grouped together as the same field of view/day
- the state categories are shared with the spine/dendrite pipeline
- the workflow keeps its caches under `results/soma_bouton_pipeline/`, and warm reruns reuse the pipeline-local analysis tables and grouped summaries instead of rebuilding them from scratch
- outputs are split into separate result divisions for activity, correlation,
  lag/offset analyses, coincidence, and ROI split comparisons
- the lag scan evaluates bouton-vs-soma correlation within a `\u00b12 s`
  offset window
- the shared coincidence layer measures exact-onset soma-vs-bouton event matches per state and writes both directional views for each pair
- coincidence example figures default to the top 5 pairs per day/state and live under `pooled/all/figures/coincidence_event_examples/` for comparison preset runs, while direct runs keep them under `figures/coincidence_event_examples/`
- coincidence CSVs are written alongside the other outputs, including daily summaries and cohort-specific copies under `csv/cohort/`

The default example config uses:

- `2026-05-13_01_ESRC033` for movie
- `2026-05-13_02_ESRC033` and `2026-05-13_03_ESRC033` for sleep

Outputs are written under `results/soma_bouton_pipeline/` by default.

## Progression and State-Transition Outputs

When `progression_analysis.enabled` is true, the pipeline writes bouton and soma mean dF/F progression outputs under `results/soma_bouton_pipeline/general/progression/`. Each native compartment contains `blank_trial_progression.csv`, `sleep_expid_progression.csv`, `progression_summary.csv`, and a figure with blank-trial-time and sleep-expID progression panels. Blank trials are aligned to onset and summarized in one-second bins by default; sleep sessions are ordered chronologically by expID. Values are averaged per experiment before group means and SEM are calculated.

State-transition CSVs and figures are kept under `results/soma_bouton_pipeline/general/state_transitions/`. The existing `transition_analysis` settings continue to control scopes, window modes, and metrics.
\n## State-transition analysis

When `transition_analysis.enabled` is true, the pipeline detects within-recording state changes and analyzes soma and bouton activity and event frequency around those transitions. The same transition outputs are also used by the comparison-preset runs.

Primary transition figures and comparison tables use `aggregation_level=expday`: repeated transitions are averaged within each entity and experiment-day, then entities are averaged within the experiment-day. Each `expID/day` therefore contributes one paired before/after summary and one aligned trace. The aligned figures show faint experiment-day traces plus the grand mean ± SEM; their annotations report experiment-days, animals, entities, and raw transitions.

Transition outputs are written under the relevant result root:

- `state_transition_events_<scope>_<window_mode>.csv` contains the raw entity-by-transition rows for auditing.
- `state_transition_comparisons_<scope>_<window_mode>.csv` contains the primary experiment-day-balanced summaries and paired statistics.
- `state_transition_pooled_comparisons_<scope>_<window_mode>.csv` contains the event-pooled diagnostic summaries.
- `state_transitions/<scope>/<window_mode>/` contains paired before/after figures and aligned `*_expday_trace.svg` figures.

The primary paired tests are calculated across experiment-days, not across individual transitions or ROIs.


## Paired State Summary Figures

Blank and movie comparison presets generate paired-only state summaries for soma and bouton ROIs. Only entities present in every plotted state are included for:

- Quiet Awake vs NREM
- Quiet Awake vs NREM vs REM

These figures are written under the poster-ready paired_state_summary/ directory and remain single-panel per entity.

## Coincidence Outputs

The soma/bouton pipeline now writes:

- `csv/soma_bouton_coincidence_by_roi.csv` for the pair-level coincidence rows
- `csv/soma_bouton_coincidence_by_day.csv` for day/state summaries
- `csv/cohort/<cohort>/soma_bouton_coincidence_by_day.csv` for cohort-split summaries
- `figures/coincidence_event_examples/<event_detection_method>/<mode>/<day_id>/<state>/` for direct runs, and `pooled/all/figures/coincidence_event_examples/<event_detection_method>/<mode>/<day_id>/<state>/` for comparison preset runs

Coincidence uses the shared exact-onset event-run match that is also used by the spine-coactivity path, so the soma/bouton and spine/dendrite analyses stay aligned on the same definition.

## ROI Split Outputs

The soma/bouton pipeline also writes:

- `csv/roi_split_subject_state.csv` for the pooled per-ROI, per-state rows used to build the split
- `csv/roi_split_membership.csv` for the split assignments, using `more_active` / `less_active` for activity splits and `higher_frequency` / `lower_frequency` for frequency splits
- `csv/roi_split_comparisons.csv` for the split comparison rows
- `csv/roi_split_summary.csv` for the per-window split summaries

The split is global across pooled eligible soma and bouton ROIs, so recordings with only one ROI still contribute. The shared helper in `analysis/shared/roi_split.py` ranks the pooled ROIs by duration-weighted activity or event-frequency scores, uses `more_active` / `less_active` for activity splits and `higher_frequency` / `lower_frequency` for frequency splits, repeats each branch for `overall`, `NREM`, and `REM`, and uses sleep-session rows only for the `NREM` and `REM` bases. The same split membership is then passed into the mixed-model leaves. The matching figures are written under `results/soma_bouton_pipeline/<branch>/<basis>/figures/roi_split/<roi_type>/roi_split_<roi_type>_<split_name>_<basis_name>.svg|png`, where `branch` is one of `activity_split`, `frequency_split`, or `activity_frequency_split` and `basis` is one of `all`, `nrem`, or `rem`.
