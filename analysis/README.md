# Analysis Scripts

These folders group the runnable analysis launchers by workflow.
The runnable code now lives inside these folders, which keeps the root directory much cleaner.

## Folders

| Folder | Purpose |
| --- | --- |
| `dendrites_pipeline/` | Main dendrite/spine analysis, day figures, demo builder, and poster launchers, now split into subfolders. Common cache/state/family utilities live in `analysis/shared/`. |
| `sleep_state_across_days/` | Sleep-state-only across-days launcher. |
| `visual_response/` | Movie dendrite and rapid-retinotopy visual-response launcher. |

## Notes

- Use the folder launchers when you want the organized view.
- The shared analysis helpers that are used by multiple workflows live in `analysis/shared/` so the imports stay local without duplicating infrastructure.
- `analysis/dendrites_pipeline/` keeps the main driver at the root and the analysis-family modules in subfolders, while the generic cache/state helpers are shared.
- `analysis/deprecated/main_pipeline/` preserves the retired main pipeline as a historical archive.
- `analysis/visual_response/` uses movie-style stimulus-vs-blank metrics and prefers `cut_with_intertrials/` for the visual-response summaries.
