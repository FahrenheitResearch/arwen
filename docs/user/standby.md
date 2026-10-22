# Warm standby

`gpuwm standby --recipe RECIPE --cycle next --outdir RUN` warms the recipe
before its next nominal source cycle. An explicit cycle uses `YYYY-MM-DDTHH`.
`--hours`, `--devices`, `--tiles`, `--poll-s` (default 3), `--geog-root`, and
`--data-dir` control the ordinary forecast shape, polling and staged inputs.
`--hydro off` opts out of the normal hydro decision. Products and publication
remain local; `--publish-root` selects their directory.

The recipe must declare `[fetch].initial_source`. The source route supplies
its objects, endpoint ladder, companion inputs and composition; the boundary
source and any cycle/lead offsets also come from the recipe tables. Adding a
recipe needs no model-specific standby code. `next` means the first nominal
cycle strictly after the current UTC hour, respecting the source's cadence.

Standby checks the installation and memory admission, creates CUDA contexts,
warms kernels, and builds the clock and execution schedule. It runs the normal
static/boundary preparation in process and retains its boundary handle while
waiting for analysis. The initial state, its dependent forecast buffers and
physics construction finish after the analysis arrives. The runner verifies
and reuses the warm execution plan and resident boundary handle.

All declared endpoints are probed concurrently. A complete version-bound GRIB
tail permits immediate transfer, using conditional parallel ranged GETs.
Companions follow the source table; if a required companion is late, its wait
is part of the analysis-to-first-step time. A recipe choosing late companions
cannot meet a short launch target merely by observing its native file early.

`RUN/standby-receipt.json` records stage start/end times and parallelism, poll
start, first observations and Last-Modified headers by endpoint, fetch,
native preparation events, forecast launch and the first completed step.
`object_seen_to_first_step_seconds` uses a monotonic clock from the first
successful object HEAD to the runner's first completed step. The forecast's
own receipt is linked as `forecast_receipt`. Receipt writes run beside the
launch and flush before exit. A warmed launch that fails or returns without
a completed step exits nonzero and records `FAILED`; repeat attempts keep
the old receipt and use a fresh attempt directory.

A real launch requires the initial-source pairing, resident-boundary and kernel
cache implementations installed with this door. Stub-store tests establish
orchestration behavior; they do not establish real forecast launch latency.
