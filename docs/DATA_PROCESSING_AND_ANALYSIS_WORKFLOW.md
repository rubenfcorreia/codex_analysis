# Data-processing and analysis workflow

This is the current operational map of the repository: where raw data are
read, how the staged caches are built, which scripts are launchers, how the
configuration is divided between files, and where tables, manifests, caches,
and figures are written.

The central dependency graph is:

```text
raw experiment folders
        |
        v
source -> processed -> initial_analysis -> classification -> for_analysis -> results
                                                               |
                                                               v
                                                        CSV / JSON / SVG / PNG
```

The `for_analysis` stage runs the scientific analysis and writes/reuses the
analysis tables/results caches. The `results` stage is principally a
plot/readback stage.

## Workflow map

The map below shows the complete path from experiment files to final results.
Its labels use the same section numbers as the full document. Where one
workflow box covers more than one section, all relevant section numbers are
shown. The solid path is the staged dendrite/spine or soma/bouton workflow;
the independent workflows branch from the same raw repository but do not
consume the main staged caches.

```text
2. RAW INPUTS
   ├─ recordings and Suite2p data
   ├─ wheel and pupil recordings
   ├─ sleep_score/sleep_state.pickle
   └─ cut neural trial bundles
              │
              v
3. EXPERIMENT MEMBERSHIP
   ├─ dendrite/spine experiments
   ├─ soma/bouton experiments
   ├─ sleep-state experiments
   └─ PCA experiments
              │
              v
4. CONFIGURATION
   ├─ pipeline_config.json
   ├─ plot_config.json
   ├─ channel, event, and state settings
   ├─ cache/output paths
   └─ shuffle and statistical settings
              │
              ├──────────────────────────────────────────────┐
              │                                              │
              v                                              v
5. MAIN STAGED CACHE WORKFLOW                         8. INDEPENDENT WORKFLOWS
   │                                                   │
   ├─ 5.1 SOURCE CACHE                                 ├─ Sleep-state across days
   │     Load traces, ROIs, metadata, states,          │    ├─ Read sleep_state.pickle
   │     wheel, pupil, and trial data                   │    ├─ Calculate fractions and bouts
   │                                                    │    └─ Write tables and figures
   ├─ 5.2 PROCESSED CACHE                              │
   │     Normalize ROI records, IDs,                  ├─ Exploratory PCA
   │     compartments, and day units                  │    ├─ Load soma/bouton contexts
   │                                                    │    ├─ Create windows and filter data
   ├─ 5.3 INITIAL ANALYSIS                             │    ├─ Z-score ROIs and fit PCA
   │     Calculate state activity, frequency,          │    └─ Write PCA outputs
   │     preferred states, and ties                    │
   │                                                    └─ Synthetic demo
   ├─ 5.4 CLASSIFICATION                                    ├─ Generate synthetic data
   │     Visual-response, preferred-state,                  ├─ Run real pipeline drivers
   │     activity, and frequency labels                     └─ Validate outputs
   │
   └─ 5.5 FOR_ANALYSIS ARTIFACTS
         Analysis tables, result caches,
         shuffle caches, manifests, and diagnostics
              │
              ├──────────────────────────────────────────────┐
              │                                              │
              v                                              v
6. DENDRITE/SPINE PIPELINE                           7. SOMA/BOUTON PIPELINE
   │                                                   │
   ├─ Shared analysis families                        ├─ Shared analysis families
   │  1. State summaries and comparisons              │  1. State summaries and comparisons
   │  2. Activity and event-frequency summaries       │  2. Activity and event-frequency summaries
   │  3. Visual-response cohorts                      │  3. Visual-response cohorts
   │  4. Correlation analysis                          │  4. Correlation analysis
   │  5. ROI split analysis                            │  5. ROI split analysis
   │  6. Mixed-model analysis                          │  6. Mixed-model analysis
   │  7. State-transition analysis                     │  7. State-transition analysis
   │                                                   │
   ├─ Dendrite/spine-specific                         ├─ Soma/bouton-specific
   │  8. Basal/apical comparisons                      │  8. Bouton-soma lag/offset analysis
   │  9. Spine coactivity                              │  9. Exact-onset coincidence analysis
   │  10. Spine matrix similarity                      │
   │  11. Direct trial-type comparisons                │
   │                                                   │
   └─ Spine-specific residuals and                   └─ Soma/bouton contexts and
      compartment-specific outputs                       pairwise outputs
              │                                              │
              └──────────────────────┬───────────────────────┘
                                     │
                                     v
                              PIPELINE OUTPUTS
                              ├─ CSV analysis tables
                              ├─ JSON manifests and reports
                              ├─ SVG/PNG figures
                              ├─ Poster-ready figures
                              ├─ Cache metadata
                              └─ Timing and diagnostic reports
                                     │
                                     v
9. SCRIPTS AND LAUNCHERS
   ├─ Main pipeline launchers
   ├─ Staged cache runners
   ├─ Specialist figure/export scripts
   ├─ PCA and sleep-state launchers
   └─ Demo builders and validators
                                     │
                                     v
10. SHARED INFRASTRUCTURE
    Supports stages 5–9:
    ├─ Cache and serialization utilities
    ├─ State and transition utilities
    ├─ Statistics and mixed-model helpers
    ├─ ROI split and comparison utilities
    ├─ Plotting modules
    └─ Logging and diagnostics
                                     │
                                     v
11. VALIDATION AND REPRODUCIBILITY
    ├─ Inspect manifests and cache signatures
    ├─ Review warnings and logs
    ├─ Check row counts and expected artifacts
    ├─ Reuse valid stages on warm reruns
    └─ Rebuild only affected stages when inputs change
```

**Clickable map navigation**

- [2. Raw inputs](#2-raw-inputs)
- [3. Experiment membership](#3-experiment-membership)
- [4. Configuration reference](#4-configuration-reference)
- [5. Staged cache workflow](#5-staged-cache-workflow)
  - [5.1 Source cache](#source)
  - [5.2 Processed / normalization](#processed)
  - [5.3 Initial ROI summaries](#initial-analysis)
  - [5.4 Classification](#classification)
  - [5.5 for_analysis artifacts](#5-staged-cache-workflow)
- [6. Dendrite/spine pipeline](#6-dendritespine-pipeline)
- [7. Soma/bouton pipeline](#7-somabouton-pipeline)
- [8. Independent workflows](#8-independent-workflows)
- [9. Scripts and launchers](#9-which-scripts-to-run)
- [10. Shared infrastructure](#10-shared-infrastructure-catalogue)
- [11. Validation and reproducibility](#11-reproducibility-checklist)


Use the numbered stages as the troubleshooting order: first verify raw files
and configuration, then inspect the first cache stage whose manifest or
signature is missing/stale, and only then inspect family-specific outputs.

## 1. Repository map

| Location | Role |
| --- | --- |
| `analysis/dendrites_pipeline/` | Main dendrite/spine workflow. |
| `analysis/soma_bouton_pipeline/` | Soma-versus-bouton workflow. |
| `analysis/sleep_state_across_days/` | Independent sleep-state/across-days workflow. |
| `analysis/pca_pipeline/` | Exploratory same-day PCA workflow. |
| `analysis/cache_builders/` | Staged-cache adapters and incremental builders. |
| `analysis/shared/` | Reusable cache, state, statistics, plotting, ROI, and diagnostic code. |
| `analysis/demo_pipeline/` | Synthetic-data builder and validation platform. |
| `analysis/deprecated/` | Historical code; not the current workflow. |
| `docs/` | Explanatory documentation and methods. |
| `tests/` | Regression and smoke tests. |
| `results/` | Generated outputs. |

## 2. Raw inputs

Further reading: [analysis methods](methods/README.md), [visual-response details](visual_response/README.md), and [sleep-state details](sleep_state_across_days/README.md).


The normal data root is `/home/<user_id>/data/Repository`. Each experiment is
resolved as:

```text
<repo_base>/<animal_id>/<exp_id>/
```

An expID normally has the form `YYYY-MM-DD_NN_ANIMAL`. Same-date sessions are
pooled into one animal-day analysis unit in the main pipelines, while original
expIDs remain as provenance.

| Raw path/type | Organization | Use |
| --- | --- | --- |
| `recordings/` | Pickled time-series bundles and related recording files. | Neural, wheel, pupil, and metadata traces. |
| Suite2p outputs (`ops.npy`, `stat.npy`, `meanImg.npy`, channel data) | NumPy arrays/dictionaries describing imaging planes and ROIs. | ROI geometry/channel context where available. |
| `recordings/wheel.pickle` | Mapping with time and a speed/wheel/motion/velocity-like array. | Locomotion masks and correlation metadata. |
| `recordings/dlcEyeLeft_resampled.pickle`, `dlcEyeRight_resampled.pickle` | Mapping with pupil diameter/area/radius-like arrays. | Pupil metadata. |
| `sleep_score/sleep_state.pickle` | Mapping containing at least `state_epoch` and `state_epoch_t`. Codes are 0 active wake, 1 quiet wake, 2 NREM, 3 REM. | Sleep-state masks, fractions, bouts, and probability profiles. |
| `cut_intertrials/s2p_ch<channel>_dF_cut.pickle` | Trial-by-ROI matrix, cut-time array, and `trial_meta`. | Preferred visual-response input. |
| `cut_with_intertrials/s2p_ch<channel>_dF_cut.pickle` | Same general cut-bundle organization. | Visual-response fallback. |

The pipeline consumes already calculated `dF/F`; it does not replace upstream
imaging conversion, Suite2p processing, or sleep scoring. Missing sleep bundles
are reported and skipped. Missing both visual-response cut bundles is reported
as unavailable rather than silently substituted.

For spines, the specific signal is the residual after robustly fitting paired
dendrite activity: `spine_specific = spine_trace - alpha*dendrite_trace`.

## 3. Experiment membership

Membership is deliberately separate from processing settings.

| File | Contents |
| --- | --- |
| `analysis/dendrites_pipeline/experiments.json` | `movie_expids`, `sleep_expids`, `basal_expids`, `apical_expids`. |
| `analysis/soma_bouton_pipeline/experiments.json` | `movie_expids`, `sleep_expids`. |
| `analysis/sleep_state_across_days/sleep_state_across_days_config.json` | `movie_expids`, `sleep_expids`, plus run paths. |
| `analysis/pca_pipeline/pca_pipeline_config.json` | Explicit lists or automatic discovery from source configs. |

Edit the sibling `experiments.json` for staged dendrite/soma-bouton membership;
edit the sleep/PCA config lists for those independent workflows.

## 4. Configuration reference

Further reading: [cache-builder usage](../analysis/cache_builders/README.md), [dendrite/spine pipeline notes](dendrites_pipeline/README.md), and [soma/bouton usage](../analysis/soma_bouton_pipeline/USAGE.md).


The staged dendrite and soma/bouton configs use these sections:

| Section | Meaning |
| --- | --- |
| `pipeline` | `name`, `user_id`, `repo_base`. |
| `inputs` | `expids_file`, normally the sibling membership JSON. |
| `paths` | `output_root`, `cache_root`, `figure_root`. |
| `source` | Dendrite `channel`, or soma/bouton `soma_channel`/`bouton_channel`. |
| `processing` | Event detector; dendrites also `state_mode` and `movie_trial_types`. |
| `classification` | Visual metric and preferred/activity/frequency split enable flags. |
| `shuffle` | `enabled`, `null_model`, `shuffle_n`, deterministic `seed`. |
| `analysis` | `families_file`; points to the analysis-family selection used after initial analysis/classification. |
| `cache` | `mode`, signature validation, removed-expID archiving, worker count. |
| `plots` | `profile` (`full`, `analysis_only`, `poster_only`), `plots_only`, and `config_file` (normally `plot_config.json`). |
| `initial_analysis` | General plots and absolute/relative preferred-state tie tolerances. |

Advanced driver keys cover state overrides, locomotion thresholds, rebuild
flags, analysis/figure directories, visual-response settings, progression,
transitions, mixed models, correlation nulls, and poster/comparison presets.
Use the driver `--help` and its defaults as the authoritative list.

Cache controls:

- `cache.mode=incremental` reuses valid experiment records;
- `cache.rebuild_stages` forces selected staged layers;
- `cache.rebuild_expids` forces selected source sessions;
- changed upstream signatures invalidate dependent stages;
- `--plots-only` regenerates figures from compatible analysis outputs;
- `--figures-only` regenerates figures from new-layout caches.

To regenerate figures from a staged result cache:

```bash
python3 -m analysis.cache_builders.run_pipeline_stage \
  --pipeline dendrites --stage results \
  --config analysis/dendrites_pipeline/pipeline_config.json
```

For soma/bouton, use `--pipeline soma_bouton` with its `pipeline_config.json`. Plot-only behavior is controlled by `plot_config.json` or the driver `--plots-only` flag; there is no separate required plots-only config file.

## 5. Staged cache workflow

Further reading: [staged cache workflow README](../analysis/cache_builders/README.md).


### Source

`analysis/cache_builders/build_source_cache.py` adapts the existing loaders;
`incremental_source.py` and `incremental_soma_source.py` reuse valid records.
`run_pipeline.py` calls this stage first.

Input: config, membership, raw experiment folders.

Output:

```text
<cache_root>/source/source_cache.npz
<cache_root>/source/manifest.json
```

The `.npz` contains a serialized Python mapping. Dendrite data are organized
as `animals[animal_id] -> dendrites[dendrite_id] -> observations[expid]`, with
nested `spines[spine_id] -> observations[expid]`; `experiments[expid]` stores
source paths, trial metadata, and signatures. Soma/bouton data are organized
as experiment/context records containing mode, animal/day, channels, ROI
traces, and state/context metadata.

The manifest records pipeline, stage, config signature, active expIDs, cache
path, and source signature. Reuse requires these dependencies to match.

### Processed

`analysis/cache_builders/build_processed_cache.py` converts variable source
layouts into a common flat `roi_records` list.

Output:

```text
<cache_root>/processed/processed_cache.npz
<cache_root>/processed/manifest.json
```

Each record contains (as available):

```text
roi_key, expid, animal_id, day_id, compartment,
state_activity, state_frequency, source_signature,
visual_responsive, visual_class
```

`state_activity` and `state_frequency` are nested state-to-number mappings.

### Initial analysis

`analysis/cache_builders/build_initial_analysis.py` summarizes each processed
ROI's numeric activity/frequency by state, computes the maximum and preferred
state(s), and records tie status. The example tie rule is absolute tolerance
`0.0` plus relative tolerance `0.01`.

Output:

```text
<cache_root>/initial_analysis/initial_analysis_cache.npz
<cache_root>/initial_analysis/manifest.json
```

Optional general initial-analysis figures are written below the figure root.

### Classification

`analysis/cache_builders/build_classification_cache.py` writes per-ROI labels
for enabled preferred-state, visual-response, activity-split, and
frequency-split branches.

Output:

```text
<cache_root>/classification/classification_cache.npz
<cache_root>/classification/manifest.json
```

### for_analysis/results

`run_pipeline_stage.py` passes the upstream paths into the real dendrite or
soma/bouton driver. Typical richer artifacts are:

| Artifact | Format and contents |
| --- | --- |
| `source_cache.npz` | Serialized mapping of traces, ROI hierarchy, masks, metadata, and signatures. |
| `*_analysis_tables_cache.npz` | State-filterable family rows and intermediate tables. |
| `*_analysis_results_cache.npz` | Preset-specific summaries/statistics and manifest-compatible results. |
| `*_shuffle_cache_<null_model>.npz` | Null schedules, exact null results, diagnostics, and metadata. |
| `manifest.json`, `cache_key_manifest.json` | JSON paths, signatures, schema/status, cache scope, and reuse state. |
| optional `*.h5` | Large mask-aware shuffled/null-model dataset. |

These caches are nested mappings/arrays for reruns. Use generated CSVs and
JSON manifests for human inspection.

## 6. Dendrite/spine pipeline

Further reading: [dendrite/spine pipeline documentation](dendrites_pipeline/README.md), [methods](methods/README.md), and [visual-response documentation](visual_response/README.md).


Run the user-facing driver:

```bash
python3 analysis/dendrites_pipeline/dendrites_pipeline.py \
  --config analysis/dendrites_pipeline/pipeline_config.json
```

Or the complete staged graph:

```bash
python3 -m analysis.cache_builders.run_pipeline \
  --pipeline dendrites \
  --config analysis/dendrites_pipeline/pipeline_config.json
```

The staged runner also accepts `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL`
and `--quiet`. It writes a detailed log under `logs/<pipeline>/` in the
current working directory and prints only warnings/errors when `--quiet` is
used.

The driver loads/rebuilds caches, pools same-day sessions, creates state masks,
prepares visual-response cohorts, dispatches analysis families, and writes
manifests/tables/figures.

### Processing sequence

1. Load source experiments and source cache.
2. Normalize dendrite/spine IDs and retain expID provenance.
3. Pool same-date sessions into day-level units.
4. Build movie masks from trial metadata, locomotion, and sleep labels.
5. Build sleep masks from `sleep_state.pickle`.
6. Detect events. The default `derivative` detector uses positive first
   derivative, per-trace centering, a `+3 SD` threshold, and a minimum run.
   Frequency is event count divided by analyzed duration.
7. Compute state activity/frequency and spine-specific traces.
8. Compare visual trials with blanks from cut data.
9. Assign `all`, `responsive`, and `nonresponsive` cohorts.
10. Run selected families and write output artifacts.

### Dendrite source modules

| File | Description |
| --- | --- |
| `dendrites_pipeline.py` | Main CLI, loaders, normalization, cache manager, family dispatcher, output writer. |
| `analysis_families/core.py` | Family orchestration/context. |
| `state.py` | State activity/frequency summaries. |
| `basal_apical.py` | Basal/apical comparisons. |
| `direct_trial_type_comparison.py` | Direct movie trial comparisons. |
| `correlation.py` | Wheel, pupil, spine/dendrite correlation families. |
| `matrix_similarity.py` | Spine-spine correlation-matrix similarity. |
| `mixed_model.py` | Mixed models and preflight diagnostics. |
| `spine_coactivity.py` | Spine/coactivity analyses. |
| `figures/sleep_dendrite_spine_day_figures.py` | Day-level figure regeneration utility. |
| `posters/*_figure.py`, `*_common.py` | Poster-ready launchers and shared rendering. |
| `demo/sleep_demo_builder.py` | Synthetic repository builder/validator. |

### Dendrite outputs

Default root: `results/dendrites_pipeline/`.

| Location | Organization |
| --- | --- |
| `csv/` | Portable rows identified by day/animal/expID, ROI/entity, compartment, state, cohort, metric, and inference fields. |
| `figures/general/` | Shared `visual_response/`, `roi_split/`, `state_transitions/`, `day_detail/`, heatmaps, checkpoint, and metric figures. |
| `figures/<preset>/pooled/<scope>/` | Preset-specific state/cohort figures. |
| `figures/.../paired_state_summary/` | Paired Quiet Awake vs NREM and Quiet Awake vs NREM vs REM summaries. |
| `general/progression/` | Progression CSVs and figures when enabled. |
| `general/state_transitions/` | Raw transition rows, expday-balanced comparisons, pooled diagnostics, aligned traces. |
| `summary/manifest.json` | Artifact inventory, selected settings, counts, warnings, and cache summary. |
| `cache/` | Staged and rich pipeline caches. |

Transition primary replicates are `expday`: repeated transitions are averaged
within entity/day, then entities within day. Raw event rows are retained.

## 7. Soma/bouton pipeline

Further reading: [soma/bouton README](../analysis/soma_bouton_pipeline/README.md) and [soma/bouton usage](../analysis/soma_bouton_pipeline/USAGE.md).


Run:

```bash
python3 analysis/soma_bouton_pipeline/run_soma_bouton_pipeline.py \
  --config analysis/soma_bouton_pipeline/pipeline_config.json
```

Or run the staged graph with `--pipeline soma_bouton`.

`run_soma_bouton_pipeline.py` is the wrapper; `soma_bouton_pipeline.py` is the
main implementation. It compares soma activity from `ch2` (example config
channel 1) with bouton activity from `ch1` (example channel 0), groups
same-date sessions as a field-of-view/day, and runs activity, frequency,
correlation, lag, coincidence, ROI split, progression, and transition work.

| Source file | Description |
| --- | --- |
| `soma_bouton_pipeline.py` | Loader, context builder, cache manager, analysis/output driver. |
| `run_soma_bouton_pipeline.py` | User-facing wrapper. |
| `plots.py` | Workflow-specific plots. |
| `analysis_families/core.py` | Family context/orchestration. |
| `analysis_families/state.py` | State summaries. |
| `analysis_families/correlation.py` | Within- and cross-compartment correlations. |
| `analysis_families/lag.py` | Bouton-vs-soma offset scan, default ±2 s. |
| `analysis_families/transitions.py` | Transition family entry points. |

Important output files under `results/soma_bouton_pipeline/` include:

- `csv/soma_bouton_coincidence_by_roi.csv`: pair-level exact-onset rows;
- `csv/soma_bouton_coincidence_by_day.csv`: day/state summaries;
- `csv/roi_split_subject_state.csv`, `roi_split_membership.csv`,
  `roi_split_comparisons.csv`, `roi_split_summary.csv`;
- `csv/*correlation*.csv` and `csv/*lag*.csv`;
- `general/state_transitions/` and `general/progression/`;
- `figures/coincidence_event_examples/` and `figures/general/roi_split/`;
- `summary/manifest.json` and `cache_key_manifest.json`.

Pairwise correlations use seeded circular shifts. Batch and shift-block sizes
bound memory; null distributions remain pair/state/experiment specific.

The soma/bouton launcher supports the same `--log-level` and `--quiet` flags.
Its structured run log is written under `logs/soma_bouton/` when launched from
the repository working directory.

## 8. Independent workflows

Further reading: [sleep-state documentation](sleep_state_across_days/README.md), [PCA README](../analysis/pca_pipeline/README.md), and [demo-platform README](../analysis/demo_pipeline/README.md).


### Sleep-state across days

Run:

```bash
python3 analysis/sleep_state_across_days/sleep_state_across_days.py \
  --config analysis/sleep_state_across_days/sleep_state_across_days_config.json
```

This workflow reads sleep bundles directly rather than the main staged cache.
It converts `state_epoch`/`state_epoch_t` into active wake, quiet wake, NREM,
and REM summaries, then writes CSV/JSON tables and figures. Config parameters
are `user_id`, `repo_base`, `movie_expids`, `sleep_expids`, `output_dir`,
`analysis_output_dir`, and `figure_output_dir`.

`export_thresholded_cinematic_clips.py` is an optional utility, not the core
launcher.

### PCA

Run:

```bash
python3 -m analysis.pca_pipeline \
  --config analysis/pca_pipeline/pca_pipeline_config.json
```

`pca_pipeline.py` builds one-second (example `window_s=1.0`) same-day windows,
requires `min_valid_fraction=0.8`, removes ROIs below `min_roi_std=1e-9`,
z-scores retained ROIs, and fits up to `n_components=25`. It writes scores with
animal/day/session, state, locomotion, pupil, and visual-condition metadata,
plus component/explained-variance outputs. The example uses `thread_limit=1`,
channels `soma_channel=1`/`bouton_channel=0`, `metric="dF"`, selected states,
and `include_spines=false`.

### Synthetic demo

```bash
python3 analysis/dendrites_pipeline/dendrites_pipeline.py \
  --demo --output-dir /tmp/sleep_codex_demo
```

`analysis/dendrites_pipeline/demo/sleep_demo_builder.py` and the modules under
`analysis/demo_pipeline/` build, preview, validate, and run synthetic data.
Demo outputs are not biological results.

## 9. Which scripts to run

Further reading: [analysis launcher index](../analysis/README.md).


### Normal launchers

- `analysis/dendrites_pipeline/dendrites_pipeline.py`
- `analysis/soma_bouton_pipeline/run_soma_bouton_pipeline.py`
- `analysis/sleep_state_across_days/sleep_state_across_days.py`
- `python -m analysis.pca_pipeline` via `analysis/pca_pipeline/__main__.py`
- `analysis/cache_builders/run_pipeline.py` for complete staged runs
- `analysis/cache_builders/run_pipeline_stage.py` for plot/readback runs
- `analysis/dendrites_pipeline/demo/sleep_demo_builder.py` for synthetic data

### Specialist utilities

`figures/sleep_dendrite_spine_day_figures.py` rebuilds day figures;
`posters/*_figure.py` builds poster products; `export_thresholded_cinematic_clips.py`
exports optional clips; and `build_*_cache.py` files run/debug individual
stages.

### Source modules

Files under `analysis/shared/`, pipeline `analysis_families/`, plotting modules,
poster `*_common.py`, and package `__init__.py` files are imported source
modules. They implement calculations, schemas, dispatch, and rendering and
are normally not invoked directly.

### Deprecated modules

`analysis/deprecated/` contains the old main pipeline, old visual-response
workflow, legacy configs, poster code, and historical families. Use the
organized non-deprecated folders for new analyses.

## 10. Shared infrastructure catalogue

Further reading: [methods and statistical conventions](methods/README.md).


| Files/group | Purpose |
| --- | --- |
| `analysis/shared/cache_utils.py`, `analysis_cache.py`, `cache_config.py` | NPZ serialization, cache keys, normalized configs, analysis payloads. |
| `result_manifest.py`, `serialization_diagnostics.py`, `runtime_diagnostics.py`, `pipeline_logging.py`, `run_timing.py` | Manifests, atomic-write records, timings, structured logging, and resource diagnostics. `pipeline_logging.py` creates timestamped per-pipeline log files and captures stage failures with tracebacks. |
| `state_utils.py`, `state_transitions.py`, `branch_tree.py` | Canonical states, transitions, and output-tree selection. |
| `statistics.py`, `analysis_families/mixed_model.py` | Tests, effect sizes, confidence intervals, p-value metadata, model preflight. |
| `roi_split.py`, `comparison_preset_flow.py` | Activity/frequency splits and preset batching. |
| `analysis/shared/plots/` | State, correlation, mixed-model, ROI split, heatmap, visual-response, and poster renderers. |
| `analysis/shared/analysis_families/` | Registered shared families for correlation, coincidence, calcium events, visual response, mixed models, and helpers. |
| `analysis/cache_builders/incremental_*`, `build_shuffle_cache.py` | Incremental source/null-model cache construction. |
| `analysis/demo_pipeline/` | Demo schema, builder, preview, validation, GUI, and runner. |

Tests are validation code, not production launchers. They cover cache reuse,
schemas, state transitions, visual response, mixed models, demo behavior,
figure layout, and methodology repairs.

## 11. Reproducibility checklist

1. Confirm raw experiment folders and `sleep_state.pickle` availability.
2. Edit membership JSON and the relevant config.
3. Set channels, event detector, state selection, shuffle/null settings, and
   output paths.
4. Run the appropriate launcher.
5. Inspect `summary/manifest.json`, warnings, and cache statuses.
6. Use `csv/` for numerical results and `figures/` for visual review.
7. Keep manifests and cache-key metadata with exported results.

Do not count raw ROIs or raw transitions as independent biological replicates
when a result is explicitly aggregated by animal-day or expday.
