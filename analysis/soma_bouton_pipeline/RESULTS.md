# Result Layout

The soma/bouton workflow keeps results divided by analysis family:

```text
results/soma_bouton_pipeline/
  csv/
    experiments.csv
    state_activity_by_experiment.csv
    state_activity_by_day.csv
    bouton_soma_correlation_by_roi.csv
    bouton_soma_correlation_by_day.csv
    bouton_soma_lag_scan_by_roi.csv
    bouton_soma_lag_summary_by_day.csv
  figures/
    state_activity/
    correlation/
    lag/
  cache/
  summary/
```

The day-level grouping uses the same `animal + date` logic as the rest of the
analysis code so experiments recorded on the same day stay together.

