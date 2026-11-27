# Staged cache workflow

For the complete data flow, see the [end-to-end data-processing and analysis workflow](../../docs/DATA_PROCESSING_AND_ANALYSIS_WORKFLOW.md).

The staged runners use the following dependency order:

```text
source -> processed -> initial_analysis -> classification -> analysis -> results
```

Run a complete analysis pass with:

```bash
/home/rubencorreia/miniconda3/envs/sci/bin/python3 -m analysis.cache_builders.run_pipeline \
  --pipeline dendrites \
  --config analysis/dendrites_pipeline/pipeline_config.json
```

Use `soma_bouton` and `analysis/soma_bouton_pipeline/pipeline_config.json` for the soma/bouton pipeline. Experiment membership is edited only in the sibling `experiments.json` file.

Valid caches are reused automatically. To force a stage rebuild, add its name to `cache.rebuild_stages`; to force selected source experiments, add their IDs to `cache.rebuild_expids`. Downstream stages are rebuilt when an upstream signature changes.

Family selection is controlled by the `analysis` section of the pipeline config:
omit `analysis.families` to use the enabled entries in `analysis_families.json`,
provide a list to run only those families, or use `[]` to build the four
pre-analysis cache stages and stop before `for_analysis`. This is useful when
you want to inspect or export the staged source/processed/initial/classification
artifacts without launching family-dependent analyses.

Regenerate figures from an existing results cache with:

```bash
/home/rubencorreia/miniconda3/envs/sci/bin/python3 -m analysis.cache_builders.run_pipeline_stage \
  --pipeline dendrites \
  --stage results \
  --config analysis/dendrites_pipeline/pipeline_config.json
```

The initial-analysis stage writes general plots, including preferred-state, visual-response, activity/frequency, and overlap summaries. Missing visual-response measurements remain explicitly classified as `unknown`.
