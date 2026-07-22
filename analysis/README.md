# Analysis Scripts

These folders group the runnable analysis launchers by workflow.
The runnable code now lives inside these folders, which keeps the root directory much cleaner.

See also: [root README](../README.md), [dendrites_pipeline](dendrites_pipeline/README.md), [sleep_state_across_days](sleep_state_across_days/README.md), [soma_bouton_pipeline](soma_bouton_pipeline/README.md), [deprecated/main_pipeline](deprecated/main_pipeline/README.md), [deprecated/visual_response](deprecated/visual_response/README.md).

## Folders

| Folder | Purpose |
| --- | --- |
| [dendrites_pipeline/README.md](dendrites_pipeline/README.md) | Main dendrite/spine analysis, day figures, demo builder, and poster launchers, now split into subfolders. Common cache/state/family utilities live in `analysis/shared/`. |
| [sleep_state_across_days/README.md](sleep_state_across_days/README.md) | Sleep-state-only across-days launcher. |
| [deprecated/visual_response/README.md](deprecated/visual_response/README.md) | Archived movie dendrite and rapid-retinotopy visual-response launcher. |

## Notes

- Use the folder launchers when you want the organized view.
- The shared analysis helpers that are used by multiple workflows live in `analysis/shared/` so the imports stay local without duplicating infrastructure. Comparison-preset batching and the deferred poster/readback step also live there now (`analysis/shared/comparison_preset_flow.py`).
- Each workflow keeps its own cache tree and output tree; the shared helpers reduce repeated preprocessing and regrouping inside a workflow without sharing results across pipelines.
- `analysis/dendrites_pipeline/` keeps the main driver at the root and the analysis-family modules in subfolders, while the generic cache/state helpers are shared.
- `analysis/deprecated/main_pipeline/` preserves the retired main pipeline as a historical archive.
- `analysis/deprecated/visual_response/` keeps the archived movie-style stimulus-vs-blank launcher and prefers `cut_with_intertrials/` for the visual-response summaries.
