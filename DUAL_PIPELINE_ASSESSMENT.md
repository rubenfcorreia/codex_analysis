# Dual-pipeline code and analysis assessment

Date: 2026-10-09

## Scope and evidence

This assessment covers the dendrite/spine and soma/bouton pipelines, their shared analysis/cache/plot modules, configured comparison presets, existing tests, available result artifacts, and available caches. The sci runtime was used for all dependency-dependent checks: /home/rubencorreia/miniconda3/envs/sci/bin/python3.


The dendrite results tree contains complete-looking outputs for the configured presets. The soma results tree contains only a shuffle cache and no complete analysis-results/manifests, so soma output conclusions are limited to code, shared caches, imports, synthetic tests, and shared implementation paths.

## Confirmed fixes implemented

- Shared mixed-model results now serialize fixed_effect_names and an explicit inferential_status.
- Analysis-result cache schema versions were bumped so old malformed mixed-model caches are rejected rather than reused.
- dF/F heatmaps now default to representative activity/frequency examples, with a separate full-day mode retained. Both dendrite and soma/bouton callers use the representative gallery path.
- Shared progression now preserves individual expID traces and avoids double-counting samples at concatenated segment boundaries.
- State-transition trace and paired before/after views are combined into a single two-panel figure.
- Later preset manifests reference already-generated shared general artifacts where applicable.

## Findings

### High priority

1. **Existing dendrite result caches are stale relative to the corrected cache schema.**
   Existing dendrite analysis-result caches report schema 3, while the current dendrite pipeline expects schema 4. They must be rebuilt before reuse.

2. **Several dendrite mixed-model branches are statistically invalid for inference.**
   Existing reports contain singular/boundary diagnostics and singular-matrix failures. Summary means remain useful descriptively, but affected model contrasts must not be interpreted as valid inferential results. The cache now marks these designs explicitly. A future random-effects redesign is a methodological decision, not an automatic code repair.

3. **The previous event-detection provenance mismatch indicates stale outputs.**
   The latest report used derivative, while an older mixed-model cache recorded amplitude. Cache compatibility now includes the relevant metadata; a clean rebuild is required.

### Medium priority

4. **Shared general artifacts are generated once, while later preset manifests are local.**
   This caused valid files under results/dendrites_pipeline/general to appear absent from later preset JSON summaries. The dendrite path now discovers and reports existing shared artifacts.

5. **The existing result-layout auditor is not scalable on the full dendrite tree.**
   Its split audit repeatedly scans all caches for every branch and reads large artifact sets; the run was interrupted after hanging during manifest/cache scanning. This is an audit-tool performance issue, not evidence of pipeline failure.

6. **The default Python environment is incomplete.**
   The system/default interpreter lacks pytest and pandas. The sci environment is the supported validation runtime and passes the full suite.

### Soma/bouton coverage limitation

No complete soma/bouton result run is present under results/soma_bouton_pipeline, so no result-level claim is made about missing soma figures or numerical outputs. The soma pipeline does use the shared mixed-model, heatmap, progression, transition, and cache code, so the shared fixes apply to it.

## Validation

Passed in the sci environment:

- Both pipeline CLI help commands.
- Explicit imports of both pipeline modules.
- Full regression suite: **141 passed, 1 warning**.
- Focused dual-pipeline assessment tests: **3 passed**.
- Python compilation of both pipelines and all shared modules.
- Representative heatmap smoke test: one gallery per compartment.

The remaining test warning is scipy precision loss for nearly identical paired transition values; it does not fail the test but should be treated as a numerical-warning condition in future reporting.

## Required rebuild commands

Dendrite:

    /home/rubencorreia/miniconda3/envs/sci/bin/python3 analysis/dendrites_pipeline/dendrites_pipeline.py --rebuild --source-cache-rebuild --analysis-tables-rebuild --analysis-results-rebuild

Soma/bouton, when raw/source data are available:

    /home/rubencorreia/miniconda3/envs/sci/bin/python3 analysis/soma_bouton_pipeline/soma_bouton_pipeline.py --rebuild

No destructive cleanup of old generated artifacts was performed.
