# Reproducible demo pipeline

This package builds a deterministic synthetic repository consumable by the
dendrite and soma/bouton pipelines. It is for pipeline checks and teaching,
not biological inference.

## Commands

```bash
python -m analysis.demo_pipeline build --config analysis/demo_pipeline/configs/fast_unified_demo.json --output-dir /tmp/codex_demo
python -m analysis.demo_pipeline validate --output-dir /tmp/codex_demo
python -m analysis.demo_pipeline run --config analysis/demo_pipeline/configs/fast_unified_demo.json --output-dir /tmp/codex_demo
python -m analysis.demo_pipeline.gui
```

The GUI edits the same JSON recipe used by the CLI. Full mode creates four
synthetic animal-days per condition; fast mode is intended for CI and smoke
tests.

## Planted behavior

For every state, basal dendrites use activity and frequency scale 2.0 and
apical dendrites use 0.5. Soma 1 uses scale 3.0 and soma 2 uses scale 1/3.
The bouton topology is:

```text
bouton_1 -> soma_1
bouton_2 -> independent control
bouton_3 -> soma_2
```

The builder writes truth independently from analysis outputs so validation can
detect a broken pipeline rather than reproduce its mistakes.

## Outputs and tmux

Runs write resolved configs, pipeline configs, truth, expected previews,
validation CSV/JSON, and run metadata. The generated repository is placed directly in the selected output folder, isolated from real-data repositories.

The GUI can launch each analysis in an isolated tmux session. Logs are stored
under `logs/`, and session names are recorded in `demo_run_metadata.json`.

## Limitations

The traces are intentionally simple pulses. They make state effects,
event-frequency effects, and correlation topology obvious; they are not a
model of calcium physiology.


## Plot profiles and runtime limits

Real-data pipelines retain `full` plotting by default. Both accept
`--plot-profile full`, `--plot-profile analysis_only`, or
`--plot-profile poster_only`. Fast demos use `analysis_only`, which writes
analysis tables and caches without figure galleries.

Recipes may set `runtime.timeout_s`, `runtime.max_output_bytes`, and
`runtime.cpu_thread_limit`. A timed-out or oversized run preserves its logs and
metadata and returns a nonzero status.

## Configurations

- `full_unified_demo.json`: four-replicate end-to-end demo.
- `fast_unified_demo.json`: one-replicate CI/smoke demo with figures disabled.
- `dendrite_state_effects.json`: dendrite-focused recipe.
- `soma_bouton_correlation_topology.json`: soma/bouton topology-focused recipe.

`demo_config_original.json`, `demo_config_resolved.json`, `demo_truth.json`,
and `demo_run_metadata.json` are provenance outputs, not editing interfaces.

## Validation and recovery

Validation first checks planted truth, then checks observed pipeline outputs
when they exist: state coverage, manifests, and bouton positive/negative
correlation controls. Tmux sessions are named per output directory and
pipeline, refuse collisions, use the active Python environment, persist logs
and exit codes, and can be stopped safely from the GUI. Failed pipelines can
be rerun with `run --only dendrites` or `run --only soma_bouton`.
