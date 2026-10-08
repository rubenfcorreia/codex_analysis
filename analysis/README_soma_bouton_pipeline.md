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
