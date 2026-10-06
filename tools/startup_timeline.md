# Startup timeline

Run `python tools/startup_timeline.py --log progress.jsonl --receipt proof.json --receipt report.json --origin 2026-01-01T00:00:00Z --output startup.json --csv startup.csv --markdown startup.md` with the box-available or run-claimed wall timestamp. Each `--log` and `--receipt` is repeatable. The tool opens only supplied files, without importing the engine or a GPU library.

Native progress JSONL provides restore, verification, physics initialization, first-step completion and durable history publication times. The first forecast hour is selected from the root domain's output valid time relative to the recorded model start. An initial analysis frame, a child frame or a model-step crossing does not count as the root forecast hour written.

For additional stages, write JSONL events with `event`, `stage` and either an absolute interval or paired markers:

```json
{"event":"startup_stage","stage":"device_upload","start_unix_s":100,"end_unix_s":104}
{"event":"stage_start","stage":"downloads","utc":"2026-01-01T00:00:00Z","worker":0}
{"event":"stage_end","stage":"downloads","utc":"2026-01-01T00:00:10Z","worker":0}
{"event":"startup_stage","stage":"device_upload","start_unix_s":100,"end_unix_s":104,"host_call_only":true}
```

The stage names are `install_bridges`, `downloads`, `statics_geography`, `decode`, `decode_compose`, `horizontal_interpolation`, `vertical_interpolation`, `preparation`, `seal`, `bundle_verify`, `bundle_restore`, `health_scan`, `model_build`, `kernel_compile`, `device_upload` and `first_step`. An origin event can use `event: box_available` or `event: run_claimed` with `utc`. A separately measured durable hour can use `event: first_forecast_hour_written` with `utc`.

Bracketed `[HH:MM:SSZ] start NAME` and `done NAME (N s)` bootstrap markers need `--date YYYY-MM-DD`; midnight rollover is handled. Receipt `timing_seconds`, forcing `horizontal_seconds` and named vertical interpolation durations are retained with their JSON keys. Missing stages remain unknown. Combined decode and composition is kept as its own stage because it does not measure decoder time alone.

Located intervals are clipped to the supplied origin and first durable forecast hour. Overlapping intervals use their union, both within each stage and for overall coverage. Stage rows may overlap, so they must not be summed. An asynchronous upload with `host_call_only: true` is listed in a separate host-call column. Its API return does not establish device-transfer completion. Receipt durations without wall placement are recorded work, which can be nested or parallel, and never fill a wall-time gap. The native terminal compile estimate is also unplaced because its emission occurs at run end. JSON includes source line numbers or JSON keys, raw intervals, unplaced durations, milestones and missing-evidence warnings.

JSON provides three separately clipped windows: `startup_until_first_step` ends at the first completed root step, `first_hour_window` ends at the first durable root forecast hour, and `after_first_step_until_first_hour` contains the calls during stepping after step 1. Each has its own total, interval union, stage summary and clipped evidence intervals. Recurring model initialization or upload callbacks keep their original stage names but fall into the appropriate window. Terminal checks after f01 never contribute to either startup ranking. Unplaced receipt work is excluded from these window rankings. The top-level `stage_summary` retains the first-hour summary plus unplaced work for compatibility. CSV has a `window` column; Markdown leads with startup-only rows and both process-origin totals.

`forcing_work_breakdown` preserves every numeric native forcing-column label, each forcing record's valid time and index, and its aggregate totals. Its leaf summary ranks recorded work across records without adding aggregate totals again. Repeated forcing indexes or valid times remain separate records. CPU preparation phases containing `upload` in their native names are retained there; they do not become measured GPU uploads.

`health_scan` is the read-only full-state validation phase. Existing store-direct health log durations are retained as unplaced evidence; their text has no wall timestamp. A wrapper around `_store_full_state_health` using explicit startup-stage intervals supplies its wall placement. Model construction and nested health scans retain separate stage names and overlap accounting.

Inputs ending in `.gz` are read as gzip text while retaining the original uncompressed line numbers in references. `--output timeline.json.gz` writes a compressed full report. For large upload-call traces, use `--summary` to omit repeated raw spans from JSON and Markdown. Every window still retains its union, observed interval count, recorded input-byte sum and per-source first/last line range with reference count. These byte sums are recorded host inputs, not a synchronized transfer measurement. Unplaced durations retain grouped work/count/source evidence. Keep the compressed raw trace alongside the compact report for individual call inspection.
