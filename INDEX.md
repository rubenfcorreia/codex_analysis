# Repository navigation index

## Start here

- [End-to-end data-processing and analysis workflow](docs/DATA_PROCESSING_AND_ANALYSIS_WORKFLOW.md)
- [Root README](README.md)
- [Runnable analysis scripts](analysis/README.md)

## Choose by task

| Goal | Start with |
| --- | --- |
| Understand the complete data flow | [Workflow guide](docs/DATA_PROCESSING_AND_ANALYSIS_WORKFLOW.md) |
| Run or modify dendrite/spine analysis | [Dendrite/spine guide](docs/dendrites_pipeline/README.md) |
| Run or modify soma/bouton analysis | [Soma/bouton README](analysis/soma_bouton_pipeline/README.md) |
| Run the staged cache graph | [Cache-builder guide](analysis/cache_builders/README.md) |
| Inspect methods/statistics | [Methods guide](docs/methods/README.md) |
| Analyze sleep-state composition | [Sleep-state guide](docs/sleep_state_across_days/README.md) |
| Run exploratory PCA | [PCA README](analysis/pca_pipeline/README.md) |
| Build synthetic test data | [Demo guide](analysis/demo_pipeline/README.md) |
| Inspect visual-response behavior | [Visual-response guide](docs/visual_response/README.md) |

## Workflow source tree

### Dendrite/spine

- Launcher: analysis/dendrites_pipeline/dendrites_pipeline.py
- Families: analysis/dendrites_pipeline/analysis_families/
- Figures: analysis/dendrites_pipeline/figures/
- Posters: analysis/dendrites_pipeline/posters/
- Demo builder: analysis/dendrites_pipeline/demo/sleep_demo_builder.py

### Soma/bouton

- Wrapper: analysis/soma_bouton_pipeline/run_soma_bouton_pipeline.py
- Driver: analysis/soma_bouton_pipeline/soma_bouton_pipeline.py
- Families: analysis/soma_bouton_pipeline/analysis_families/
- Usage: analysis/soma_bouton_pipeline/USAGE.md

### Shared infrastructure

- Staged builders: analysis/cache_builders/
- Shared helpers: analysis/shared/
- Shared plotting: analysis/shared/plots/
- Demo platform: analysis/demo_pipeline/

### Independent workflows

- Sleep-state: analysis/sleep_state_across_days/
- PCA: analysis/pca_pipeline/
- Deprecated visual response: analysis/deprecated/visual_response/

## Current configurations

- Dendrite/spine: analysis/dendrites_pipeline/pipeline_config.json, plot_config.json, experiments.json
  - Analysis families: analysis/dendrites_pipeline/analysis_families.json
- Soma/bouton: analysis/soma_bouton_pipeline/pipeline_config.json, plot_config.json, experiments.json
  - Analysis families: analysis/soma_bouton_pipeline/analysis_families.json
- Sleep-state: analysis/sleep_state_across_days/sleep_state_across_days_config.json
- PCA: analysis/pca_pipeline/pca_pipeline_config.json

## Deprecated archive

analysis/deprecated/ and docs/deprecated/ preserve historical workflows,
configs, and documentation. Use them for provenance or comparison, not for new
runs.

## Tests

tests/ contains smoke tests and regression tests for cache stages, state
transitions, visual response, statistical methods, demos, and output layout.
