# Zebra Movie Helpers

These scripts manage the movie-only helper workflow around the main dendrite/spine pipeline.
They handle zebra clip caching, the Gabor library, and the movie-only post-process.

## Script Map

| Script | What it does |
| --- | --- |
| `sleep_zebra_movies_roi_wrapper.py` | ROI-aware wrapper that refreshes the movie cache and writes per-dendrite/per-spine figures. |
| `sleep_zebra_movies_master.py` | Convenience runner that launches the wrapper and then the movie-only Gabor post-process. |
| `sleep_zebra_gabor_postprocess.py` | Movie-only Gabor sidecar post-process that runs after the main pipeline. |
| `sleep_zebra_movies_assets.py` | Shared helper for clip encoding and Gabor asset cache management. |
| `sleep_zebra_gabor_detail.py` | Shared helper for low-level Gabor detail generation. |

## Zebra Movies Wrapper

Use `sleep_zebra_movies_roi_wrapper.py` when you want one script to:

- discover whether each experiment is dendrite/spine, dendrite/axon, soma-only, or mixed
- resolve the zebra movie clips from `all_movie_clips_bv_sets` and cache encoded movies under `/home/rubencorreia/data/zebra_movies/encoded_movies`
- record the WavEn Gabor defaults in `/home/rubencorreia/data/zebra_movies/gabor_library_manifest.json`, and optionally materialize the dense `.npy` library when explicitly requested
- refresh or reuse `results/zebra_movies/sleep_dendrite_spine_cache.npz`
- save the dendrite summary figures under `results/zebra_movies/figures/<animal_ID>/<basal-or-apical>/<date>/` as matching `.png` and `.svg` files with poster-ready, normal poster-panel sizing
- skip figure generation cleanly for soma-only sessions that do not have a dendrite/spine hierarchy

## Methods

For the repo-wide methods page that explains the shared analysis metrics and tests, see [../methods/README.md](../methods/README.md).
The zebra-movie helpers are support workflows rather than inferential analyses, so the methods page only describes their role at a high level.

## Gabor Post-Process

Use `sleep_zebra_gabor_postprocess.py` when you want the movie-only Gabor summaries as a separate sidecar step after the main dendrite/spine pipeline:

- reads the existing `results/zebra_movies/sleep_dendrite_spine_cache.npz`
- processes movie expIDs only
- writes `results/zebra_movies/gabor/manifest.json`
- writes `results/zebra_movies/gabor/gabor_summary.json`
- writes one detail JSON per movie expID under `results/zebra_movies/gabor/experiments/<animal>/<date>/`
- leaves the main pipeline cache and the day figures unchanged

Example:

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/zebra_movies/sleep_zebra_gabor_postprocess.py \
  --cache-path /home/rubencorreia/code/codex_analysis/results/zebra_movies/sleep_dendrite_spine_cache.npz
```

## Zebra Movie Master

Use `sleep_zebra_movies_master.py` when you want one command to launch the ROI wrapper and then the movie-only Gabor post-process on the same cache.
The master runner forwards any wrapper arguments you pass, then runs the sidecar Gabor step unless you tell it to skip one of them.

Example:

```bash
python3 /home/rubencorreia/code/codex_analysis/analysis/zebra_movies/sleep_zebra_movies_master.py \
  --config /home/rubencorreia/code/codex_analysis/analysis/main_pipeline/sleep_dendrite_spine_example_config.json
```

## Shared Helpers

`sleep_zebra_movies_assets.py` and `sleep_zebra_gabor_detail.py` are support modules rather than end-user entry points.
They are imported by the wrapper and the Gabor post-process, so you usually do not run them directly.

## Notes

- The zebra movie layout resolves from `/data/Remote_Repository/bv_resources/all_movie_clips_bv_sets`.
- The movie-only Gabor analysis is a separate post-process and does not feed the day figures or the main analysis cache.
- The movie helper scripts write their own manifests so the cache and figure outputs stay easy to audit.
