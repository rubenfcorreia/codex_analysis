# Soma/Bouton Pipeline

New sibling workflow for comparing soma `ch2` activity against bouton `ch1`
activity across movie and sleep states.

Primary files:

- `soma_bouton_pipeline/README.md`
- `soma_bouton_pipeline/USAGE.md`
- `soma_bouton_pipeline/RESULTS.md`
- `soma_bouton_pipeline/soma_bouton_pipeline_config.json`


Progression analysis is enabled with `progression_analysis.enabled` in the config. It writes mean bouton and soma dF/F blank-trial and continuous elapsed-sleep summaries under `results/soma_bouton_pipeline/general/progression/`; same-day sleep expIDs are concatenated as one animal-day replicate, with `sleep_time_step_s` controlling interpolation. State-transition outputs are under `results/soma_bouton_pipeline/general/state_transitions/`.

## Synthetic demos

Use the separate reproducible demo platform documented in [analysis/demo_pipeline/README.md](demo_pipeline/README.md).

## Plot profiles

The CLI supports --plot-profile full (default), analysis_only, and poster_only.


## Incremental reruns

The pipelines use layered caches. Source and processed artifacts are reusable by experiment/day, and family caches reuse unchanged scopes when their source signatures and analysis settings match. A warm rerun therefore recalculates only new or invalidated experiment scopes for the scoped families. Cache summaries report requested, reused, and recomputed scopes.

Use these profiles for common workflows:

- --plot-profile analysis_only computes numerical results and skips family-level figure rendering.
- --plots-only or --figures-only regenerates figures from compatible saved analysis caches without rerunning numerical families.
- Change analysis settings normally when you want invalidation; do not use --rebuild unless all cached stages should be discarded.

State aggregation and mixed-model fitting remain cohort-wide statistical stages. They are intentionally refit when the experiment cohort or relevant configuration changes, even when per-experiment inputs are reused.
