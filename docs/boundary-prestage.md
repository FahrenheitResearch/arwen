# Resident boundary preparation

`gpuwm.boundary_prestage.prestage_boundaries(prepare, devices=(),
device_budget_bytes=None)` starts the ordinary mapped preparation in a worker
thread. `prepare` is a zero-argument callable and must host the adapter in this
process. It may fetch the complete forcing window first. A subprocess cannot
carry this handle or its resident arrays.

```python
with prestage_boundaries(prepare, devices=(0,)) as handle:
    if not handle.wait_ready(timeout=7200):
        raise TimeoutError("boundary preparation has not finished")
    # Wait for the independent analysis, then supply its normal JSON declaration.
    prepared = handle.finish_initial(initial_inputs_json)
    run_prepared_forecast(inputs, output_directory=output, prestaged_boundaries=handle)
```

The single-domain runner's `main(argv, prestaged_boundaries=handle)` also accepts
the handle. The forecast inputs must bind the completed cache returned by this
preparation. The root path and content digest are checked before attachment.
Closing the handle requires the forecast and its CUDA work to have finished.

Before readiness, the usual admitted forcing workers compose every target-lattice
frame. Immutable value/tendency tables for intervals 1 onward are retained and
registered as pinned host pages. Frames 0 and 1 are retained for the analysis
update. The analysis-dependent work is named in `handle.receipt`: decode and
initialize the analysis atmosphere/land/soil, bind the configured aerosol donor,
replace the first coupling frame, and compute interval 0's tendency. Every later
interval is the same object constructed before readiness. Normal cache publication
still runs and supplies the forecast's provenance seal.

The caller may provide `{device_id: admitted_bytes}` in `device_budget_bytes`.
This is memory reserved **after** forecast state, physics and tile admission.
Free VRAM alone is not sufficient authorization to consume that memory. Linear
full-domain tables are uploaded when the complete set fits this budget and the
device's currently free memory. Attachment transfers those exact buffers and a
CUDA event between threads; it adds no host synchronization. Tile-window and
multi-device slab routes use the pinned host tables through their existing
window/upload machinery. They do not recompose source data. Nonlinear table
producers also retain the host route.

The producer currently requires a complete single-domain mapped forcing window.
An as-posted producer cannot promise that every boundary is resident; a hierarchy
requires an analysis update for every child; a physical-member output store needs
its complete initial physical state. Those combinations are refused explicitly.
Producer failures reach both `wait_ready` and `finish_initial`.

Receipts contain stage durations, worker admission, host/pinned/device bytes,
readiness time and resident interval-read wall time. A small scheduling smoke
checks exact boundary words and object reuse. It does not replace a real-size
forecast comparison.
