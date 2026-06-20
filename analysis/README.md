# Analysis Scripts

These folders group the runnable analysis launchers by workflow.
The runnable code now lives inside these folders, which keeps the root directory much cleaner.

## Folders

| Folder | Purpose |
| --- | --- |
| `main_pipeline/` | Main dendrite/spine analysis, day figures, demo builder, and poster launchers, now split into subfolders. |
| `sleep_state_across_days/` | Sleep-state-only across-days launcher. |
| `visual_response/` | Movie dendrite and rapid-retinotopy visual-response launcher. |

## Notes

- Use the folder launchers when you want the organized view.
- The shared analysis helpers live inside the workflow folders so the imports stay local.
- `analysis/main_pipeline/` keeps the main driver at the root and the analysis-family modules in subfolders.
- `analysis/visual_response/` uses movie-style stimulus-vs-blank metrics and prefers `cut_with_intertrials/` for the visual-response summaries.
