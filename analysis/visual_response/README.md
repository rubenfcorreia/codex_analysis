# Movie Dendrites and Rapid Retinotopy

This folder contains the visual-response launcher for the movie dendrite cohorts and the rapid-retinotopy soma cohorts.

See also: [../README.md](../README.md), [../dendrites_pipeline/README.md](../dendrites_pipeline/README.md), [../../docs/visual_response/README.md](../../docs/visual_response/README.md).

## Launcher

- `movie_visual_response.py`

## What It Does

- treats `basal_expids` and `apical_expids` as movie experiments
- treats `soma_expids` as optional rapid-retinotopy experiments
- prefers `cut_with_intertrials/` for the movie visual-response metrics
- pools repeated expIDs within each basal/apical compartment
- uses the native time axes already stored in the aligned data instead of a fixed pre/post plotting window
- uses an explicit `soma_group_map` only when soma-enabled runs are requested
- writes poster-ready PNG and SVG figures under `results/visual_response/figures/<group>/`
- adds one movie-only significance-count figure that shows the number of basal and apical dendrites that are significant vs intertrial, blank, both, or neither
- emits separate basal and apical movie figures for each group, split into one onset figure and one boxplot figure per movie category, with one averaged trace per dendrite in each onset panel
- renders the soma rapid-retinotopy outputs as separate onset, boxplot, and 2D response-map figures; the retinotopy panel becomes a 2D response map when the trial table contains repeated x/y stimulus positions, and the onset figure shows the 1 s grating window from the trial table

## Poster Composite

- `poster_ready_visual_response.py` builds a 360 mm x 180 mm, SVG-only poster composite under `results/poster_ready/`
- running `movie_visual_response.py` also emits the same poster-ready SVG sidecar into `results/poster_ready/` when soma-enabled runs are requested
- the poster figure assembles three rows:
  - soma mean image, rapid-retinotopy map, grating onset, grating boxplot
  - apical mean image, blank onset, blank boxplot, movies onset, movies boxplot
  - basal mean image, blank onset, blank boxplot, movies onset, movies boxplot
- the soma mean-image panel falls back to a single labeled ROI when no conversion library is available
- ROI overlays reuse the spine-day label placement and contour styling

## Example

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/visual_response/movie_visual_response.py --config /home/rubencorreia/code/codex_analysis/analysis/visual_response/movie_visual_response_config.json
```

## Notes

- The script expects the soma pairing to be explicit in `soma_group_map` only when soma-enabled runs are used; movie-only basal/apical runs do not need somas.
- The example config does not include pre/post plotting windows; the launcher uses the native time axes stored in the data and only keeps internal fallback windows for raw soma traces when needed.
- If a session is missing `cut_with_intertrials/`, the launcher prints an alert and skips that session for the visual-response metrics.
- The output manifest is written to `results/visual_response/manifest.json`.
