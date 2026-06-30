# Cache-Enabled Soma/Bouton Runner

Use this entrypoint when you want cached raw data and cached results:

```bash
python -m analysis.soma_bouton_pipeline_cache --config analysis/soma_bouton_pipeline/soma_bouton_pipeline_config.json
```

It stores:

- per-experiment cached source snapshots and computed rows
- a summary cache for fast replay
- a `plots_only` mode that reloads saved results and redraws figures

The output layout mirrors the non-cached soma/bouton workflow, with extra
`cache/` artifacts alongside `csv/`, `figures/`, and `summary/`.

