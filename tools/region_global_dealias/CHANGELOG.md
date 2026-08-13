# Changelog

## 0.2.0 - 2026-08-11

- Add the opt-in RIFT gate-resolution refinement path for branch ambiguity
  inside otherwise connected region-global components.
- Preserve the existing `dealias()`, `dealias_sweep()`, and `bw_dealias()`
  behavior and ABI.
- Add JavaScript, worker-pool, Rust, and native C entry points for RIFT.
- Add optional projected reference fields and per-gate fold, confidence, and
  reason diagnostics.
- Require physical gate geometry for automatic single-sweep refinement and
  conservatively abstain when the independent trigger, vortex fit, and bounded
  fusion cut do not agree.
- Add deterministic real-case regression fixtures and a 20-volume, 80-cut
  validation corpus summary.
- Add a public C header for both the stable and additive native interfaces.

The automatic RIFT path accepted five local corrections totaling 442 gates in
the current corpus and left the tested hurricane and control cuts unchanged.
This is bounded validation evidence, not a claim of universal meteorological
truth.
