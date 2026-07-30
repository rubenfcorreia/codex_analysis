# Visual Response Methods

This page documents the archived visual-response launcher in `analysis/deprecated/visual_response/`.

See also: [../../README.md](../../README.md), [../../analysis/README.md](../../analysis/README.md), [../../analysis/deprecated/visual_response/README.md](../../analysis/deprecated/visual_response/README.md).

## Data Split

- `basal_expids` and `apical_expids` are movie experiments.
- `soma_expids` are rapid-retinotopy experiments.
- The script expects an explicit `soma_group_map` so the soma groups can be associated with the intended movie dendrite cohorts without guessing. Processed soma data is preferred from `/home/<user>/data/Repository/...`; if it is not present yet, the launcher falls back to the raw rapid-ret source under `/data/Remote_Repository/...`.

## Alignment

- Trial onset comes from the `time` column in the trial table.
- Movie dendrites are grouped by the movie categories `blank`, `zebra`, `movies`, and `gratings`.
- Soma sessions are summarized with separate onset, boxplot, and retinotopy-map figures, using the processed repository first and rapid-ret fallback only when the processed soma bundle is absent. The soma onset panel uses the trial-table stimulus duration directly, which is 1 s for the ESRC028 rapid-ret data. For ESRC028-style retinotopy tables, the map figure becomes a 2D x/y response map averaged across repeated stimulus positions.
- The plots use the time vectors already stored in the aligned data; the example config does not need to specify a fixed pre/post plotting window.
- `cut_with_intertrials/` is required for the movie visual-response metrics. If that bundle is missing, the launcher prints an alert and skips the session rather than silently falling back to a different cut bundle.

## Poster Composite

- `poster_ready_visual_response.py` writes the poster composite to `results/poster_ready/` as SVG only
- `movie_visual_response.py` also renders the poster-ready SVG into `results/poster_ready/` when a soma-enabled visual-response run is executed
- the canvas is 360 mm x 180 mm and uses the spine-day ROI label placement and contour styling
- the figure is a three-row landscape composite:
  - soma mean image, rapid-retinotopy map, grating onset, grating boxplot
  - apical mean image, blank onset, blank boxplot, movies onset, movies boxplot
  - basal mean image, blank onset, blank boxplot, movies onset, movies boxplot
- the soma mean-image panel falls back to a single labeled ROI when no conversion library is available

## Outputs

- Figures are exported as PNG and SVG siblings.
- The output directory is `results/visual_response/`.
