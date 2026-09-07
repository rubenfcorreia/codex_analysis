# Exploratory PCA Pipeline

This standalone workflow produces descriptive, same-day PCA views of soma and
bouton activity. It reuses the existing experiment and state loaders and writes
to `results/pca_pipeline/`.

Run it with:

```bash
python -m analysis.pca_pipeline --config analysis/pca_pipeline/pca_pipeline_config.json
```

The first version uses continuous soma/bouton recordings, one-second windows,
with a one-thread numerical backend limit, and requires scikit-learn,
per-ROI z-scoring, and preserves animal/day/session, state, locomotion, pupil,
and visual-condition metadata in `csv/pca_scores.csv`. Spine support is kept
disabled until an explicit same-day spine source mapping is supplied.
