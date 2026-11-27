# Sleep Dendrite/Spine Pipeline

This repository contains several related neural and behavioural analysis
workflows. The end-to-end workflow guide is the canonical map of inputs,
configuration, processing stages, analysis families, caches, and outputs.

## Start here

- [Data-processing and analysis workflow](docs/DATA_PROCESSING_AND_ANALYSIS_WORKFLOW.md) — complete workflow map and troubleshooting order.
- [Analysis scripts](analysis/README.md) — runnable launchers organized by workflow.
- [Repository index](INDEX.md) — compact source and documentation navigation.

## Workflows

| Workflow | Launcher | Detailed documentation |
| --- | --- | --- |
| Dendrite/spine | analysis/dendrites_pipeline/dendrites_pipeline.py | [Dendrite/spine guide](docs/dendrites_pipeline/README.md) |
| Soma/bouton | analysis/soma_bouton_pipeline/run_soma_bouton_pipeline.py | [Soma/bouton README](analysis/soma_bouton_pipeline/README.md) · [Usage](analysis/soma_bouton_pipeline/USAGE.md) |
| Staged cache graph | python -m analysis.cache_builders.run_pipeline | [Cache-builder guide](analysis/cache_builders/README.md) |
| Sleep-state across days | analysis/sleep_state_across_days/sleep_state_across_days.py | [Sleep-state guide](docs/sleep_state_across_days/README.md) |
| Exploratory PCA | python -m analysis.pca_pipeline | [PCA README](analysis/pca_pipeline/README.md) |
| Synthetic demo | analysis/demo_pipeline/ | [Demo guide](analysis/demo_pipeline/README.md) |

## Methods and specialized topics

- [Analysis methods](docs/methods/README.md) — preprocessing, metrics, statistics, correlations, coactivity, mixed models, and transitions.
- [Visual response](docs/visual_response/README.md) — visual/blank trial metrics and movie-style response analysis.
- [Deprecated main pipeline archive](docs/deprecated/main_pipeline/README.md) — historical reference only.

## Current configuration files

| File | Purpose |
| --- | --- |
| analysis/dendrites_pipeline/pipeline_config.json | Dendrite/spine processing, cache, analysis, and plotting configuration. |
| analysis/dendrites_pipeline/analysis_families.json | Dendrite/spine analysis-family enablement, caching, result, plot, and scope settings. |
| analysis/dendrites_pipeline/plot_config.json | Dendrite/spine plot-only controls. |
| analysis/dendrites_pipeline/experiments.json | Dendrite/spine experiment membership and basal/apical assignments. |
| analysis/soma_bouton_pipeline/pipeline_config.json | Soma/bouton processing, cache, analysis, and plotting configuration. |
| analysis/soma_bouton_pipeline/analysis_families.json | Soma/bouton analysis-family enablement, caching, result, plot, and scope settings. |
| analysis/soma_bouton_pipeline/plot_config.json | Soma/bouton plot-only controls. |
| analysis/soma_bouton_pipeline/experiments.json | Soma/bouton experiment membership. |
| analysis/sleep_state_across_days/sleep_state_across_days_config.json | Sleep-state experiment lists and output paths. |
| analysis/pca_pipeline/pca_pipeline_config.json | PCA experiment discovery, windowing, filtering, and output settings. |

Generated results are written below results/; source code is under analysis/;
explanatory documentation is under docs/; validation code is under tests/.
