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
validation CSV/JSON, and run metadata. The generated repository is placed
below `data/Repository` in the selected output folder.

The GUI can launch each analysis in an isolated tmux session. Logs are stored
under `logs/`, and session names are recorded in `demo_run_metadata.json`.

## Limitations

The traces are intentionally simple pulses. They make state effects,
event-frequency effects, and correlation topology obvious; they are not a
model of calcium physiology.
