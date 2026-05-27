# Sleep State Across Days

Use `sleep_state_across_days.py` when you want to compare sleep metrics across days for each animal and across animals.
It reads directly from the repository, only uses `sleep_score/sleep_state.pickle`, and still works when neuronal activity is missing.

## Script And Config

| File | What it does |
| --- | --- |
| `sleep_state_across_days.py` | Sleep-state-only analysis across days and animals. |
| `sleep_state_across_days_config.json` | Dedicated config with the sleep expID lists plus repo and output settings. |

## What It Does

- resolves `/home/<user_id>/data/Repository/<animal_id>/<exp_id>`
- loads `sleep_score/sleep_state.pickle` only
- never uses `sleep_state_sim.pickle`
- computes state fraction, bout count, total state time, and mean bout duration for active wake, quiet wake, NREM, and REM
- pools same-day expIDs within `(animal_id, date, category)`
- keeps `movie` and `sleep` sessions separate at the fraction-calculation stage, while pooling same-day sleep sessions before sleep-state fractions are computed
- writes per-animal figures and combined across-animal figures
- writes stacked-area summaries that show the state composition across days
- writes fraction-vs-time figures that show 5-minute elapsed-time profiles, plus matching elapsed-time percent views
- writes a Markdown report with the full REM transition summary tables, probability curves, and REM-specific plots
- aligns the combined plots by within-animal day index so the trajectory shapes are easier to compare across animals

## Methods

For the repo-wide explanation of metrics and statistical tests, see [../methods/README.md](../methods/README.md).
That page covers how this workflow fits into the larger analysis stack and how the main pipeline's tests differ from this descriptive sleep-only summary.

## Inputs

The config only needs:

- `user_id`
- `repo_base`
- `movie_expids`
- `sleep_expids`
- `output_dir`

If a sleep bundle is missing, the expID is skipped and recorded in the manifest rather than inferred from other files.

If you do not set `output_dir`, the script writes to `/home/rubencorreia/code/codex_analysis/results/sleep_state_across_days/`.

## Example

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/sleep_state_across_days/sleep_state_across_days.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/sleep_state_across_days/sleep_state_across_days_config.json
```

## Outputs

The run writes:

- exp-level and day-level CSV tables
- REM transition CSV tables for per-exp, per-day, and cumulative probability summaries
- a JSON manifest with processed expIDs, skipped expIDs, and source sleep paths
- a Markdown report at `results/sleep_state_across_days/sleep_state_across_days_report.md`
- poster-ready SVG and PNG figures for the per-animal and combined summaries
- a poster-ready composite in `results/poster_ready/` that combines the overall sleep-state pie, REM day-presence pie, sleep-only stacked area, and movie-plus-sleep fraction summary
- stacked-area SVG and PNG figures for the per-animal and combined composition summaries
- REM latency SVG + PNG figures for per-animal and combined summaries
- REM probability SVG + PNG figures for per-animal and combined summaries
- REM fraction SVG + PNG figures for per-animal and combined summaries
- a REM day-presence pie chart in SVG + PNG
- an overall sleep-state composition pie chart in SVG + PNG
- the stacked-area figures live under `results/sleep_state_across_days/figures/stacked_area/`
- the fraction figures live under `results/sleep_state_across_days/figures/probability_time/` and `results/sleep_state_across_days/figures/probability_time_percent/`, use 5-minute elapsed-time bins, and pool same-day sleep sessions before computing the sleep curves
- the REM cumulative figures live under `results/sleep_state_across_days/figures/rem_summary/`
- the REM fraction figures live under `results/sleep_state_across_days/figures/rem_summary/fraction_time/`
- the state montage figures live under `results/sleep_state_across_days/figures/state_montage/per_exp/<animal>/<exp_id>/` and show eye frame, pupil size, spectrogram, EMG, locomotion, and hypnogram rows
- a review copy of the first montage example is written to `review_figures/state_montage/` at the repository root for quick inspection
- the pie charts live under `results/sleep_state_across_days/figures/rem_summary/overall/` and `results/sleep_state_across_days/figures/composition_summary/`
- the poster-ready composite lives under `results/poster_ready/`

## Notes

- Same-day expIDs are combined before plotting, but movie and sleep categories stay separate.
- The combined figure is meant to answer whether day-to-day evolution looks similar across animals.
- The stacked-area figures make the composition easier to read: each day sums to 100%, the four canonical states are stacked in the same order, and Unclassified only appears when there is residual time outside those states.
- If a canonical state never appears in the source bundles for a panel, the figure annotates that state as missing instead of leaving the gap ambiguous.
- The fraction-vs-time figures use 5-minute elapsed-time bins for the minute view and normalized 0-100% axes for the percent view, with movie and sleep expIDs computed separately and same-day sleep sessions pooled before sleep-state fraction curves are built.
- The REM report combines multiple sleep recordings from the same animal/date before calculating first-REM latency and REM probability curves.
- The REM figures include first-REM latency plus cumulative REM probability from recording start and from first active wake.
- The REM fraction figures show the mean REM occupancy across 5-minute elapsed-time bins.
- The REM day-presence pie chart shows how many pooled sleep days contained REM versus did not.
- The pie chart summarizes the total canonical sleep-state time across all processed recordings.
- The workflow does not depend on neuronal activity files, so experiments without those files can still be included when the sleep bundle exists.
