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
- coincidence example figures default to the top 5 pairs per day/state and live under `figures/coincidence_event_examples/`
- coincidence CSVs are written alongside the other outputs, including daily summaries and cohort-specific copies under `csv/cohort/`

The default example config uses:

- `2026-05-13_01_ESRC033` for movie
- `2026-05-13_02_ESRC033` and `2026-05-13_03_ESRC033` for sleep

Outputs are written under `results/soma_bouton_pipeline/` by default.


## Coincidence Outputs

The soma/bouton pipeline now writes:

- `csv/soma_bouton_coincidence_by_roi.csv` for the pair-level coincidence rows
- `csv/soma_bouton_coincidence_by_day.csv` for day/state summaries
- `csv/cohort/<cohort>/soma_bouton_coincidence_by_day.csv` for cohort-split summaries
- `figures/coincidence_event_examples/<event_detection_method>/<mode>/<day_id>/<state>/` for the top-ranked coincidence example figures

Coincidence uses the shared exact-onset event-run match that is also used by the spine-coactivity path, so the soma/bouton and spine/dendrite analyses stay aligned on the same definition.

## ROI Split Outputs

The soma/bouton pipeline also writes:

- `csv/roi_split_subject_state.csv` for the pooled per-ROI, per-state rows used to build the split
- `csv/roi_split_membership.csv` for the split assignments, using `more_active` / `less_active` for activity splits and `higher_frequency` / `lower_frequency` for frequency splits
- `csv/roi_split_comparisons.csv` for the split comparison rows
- `csv/roi_split_summary.csv` for the per-window split summaries

The split is global across pooled eligible soma and bouton ROIs, so recordings with only one ROI still contribute. The shared helper in `analysis/shared/roi_split.py` ranks the pooled ROIs by duration-weighted activity or event-frequency scores, uses `more_active` / `less_active` for activity splits and `higher_frequency` / `lower_frequency` for frequency splits, repeats each branch for `overall`, `NREM`, and `REM`, and uses sleep-session rows only for the `NREM` and `REM` bases. The same split membership is then passed into the mixed-model leaves. The matching figures are written under `results/soma_bouton_pipeline/<branch>/<basis>/figures/roi_split/<roi_type>/roi_split_<roi_type>_<split_name>_<basis_name>.svg|png`, where `branch` is one of `activity_split`, `frequency_split`, or `activity_frequency_split` and `basis` is one of `all`, `nrem`, or `rem`.
