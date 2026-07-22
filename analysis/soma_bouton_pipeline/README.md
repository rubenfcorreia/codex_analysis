# Soma/Bouton Pipeline

This workflow compares soma activity from `ch2` data against axonal bouton
activity from `ch1` data. The shared preset flow lives in `analysis/shared/`.

See also: [../README.md](../README.md), [../dendrites_pipeline/README.md](../dendrites_pipeline/README.md).

Key points:

- same-date experiments are grouped together as the same field of view/day
- the state categories are shared with the spine/dendrite pipeline
- the workflow keeps its caches under `results/soma_bouton_pipeline/`, and warm reruns reuse the pipeline-local analysis tables and grouped summaries instead of rebuilding them from scratch
- outputs are split into separate result divisions for activity, correlation,
  and lag/offset analyses
- the lag scan evaluates bouton-vs-soma correlation within a `\u00b12 s`
  offset window

The default example config uses:

- `2026-05-13_01_ESRC033` for movie
- `2026-05-13_02_ESRC033` and `2026-05-13_03_ESRC033` for sleep

Outputs are written under `results/soma_bouton_pipeline/` by default.

