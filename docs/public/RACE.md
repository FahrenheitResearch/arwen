# Race a locally published forecast

`gpuwm race RECIPE --cycle HHZ` runs the installed standby door on this box,
forecasts 48 hours with hourly products, and compares the local publisher's
committed `online_at` times with source posting times. The headline reference is the earliest observed public
posting on NOMADS or S3; S3 `LastModified` and NOMADS directory mtime are both retained.
`HHZ` means today in UTC; use `YYYY-MM-DDTHH` for an explicit date. The `hrrr`
and `hrrr_v4` aliases are rows in the recipe registry. Their 48-hour cycles are
00, 06, 12 and 18 UTC.

```sh
gpuwm race hrrr --cycle 12Z --devices 8 --outdir ./race-runs
```

Each invocation allocates a fresh run folder and writes `RACE.md` and `race.json`.
The directory is printed at launch; `race-runs/latest-run.txt` names it. The
receipts include every hour's comparison, launch and first completed step,
source stage walls, hydro decision, products, box and engine SHA. Positive
margin means our products appeared first. Unknown times remain unknown.

The default web root is the race run's `www/`. Publishing only makes local
files and their manifest readable; it does not start a server or upload them.
`--publish-root DIR` selects another local root. The publisher's five priority
products define `online_at`; its receipt lists them. `--hydro auto` preserves
the forecast's default (no hydro: hydro is opt-in); `--hydro off` is an explicit opt-out.
`--hydro SETTINGS.toml` selects prepared settings, and `--hydro on` explicitly
enables the normal hydro door. These choices work with both standby and go.

The object watcher uses one background listing per source every 30 seconds for the whole
cycle (about four requests per minute), querying NOMADS first. It records both the server timestamp and its own first-seen timestamp;
only the server timestamp enters the margin. NOMADS directory times have
one-minute precision; differences below that scale are not a precise ranking. The default four-hour observation
budget never terminates a running forecast. A failed run or incomplete
publication produces receipts and a nonzero exit status. Interrupting the race
also stops its forecast child.

Until the standby door is installed, race waits for the analysis object then
calls `gpuwm go` with the available pieces. The receipt names that fallback.
`--initial-inputs JSON` supplies a separate analysis manifest on that path.
Without the hourly publisher, there are no claimed online times. A fallback
without a separate analysis cannot report a complete analysis-initialized race.

For a short full-domain door check, use `--hours 1 --tiles auto --devices 1`.
An S3-compatible local HTTP listing can be selected with `--object-endpoint`
for the NOAA side of a test. The forecast still uses its own input source.

## Price a measured box

```sh
gpuwm race hrrr --cycle 2026-10-21T00 --dry \
  --posting-times ./posting-times.json \
  --seconds-per-hour 81.8 --launch-to-first-step 60 --publish-lag 60 \
  --box-label 8x5090 --measurement-note 'Hourly history and products on'
```

The calculation is:

```text
predicted online(N) = earliest observed NOAA analysis f00 posting
                    + launch delay + launch-to-first-completed-step
                    + N * measured seconds-per-forecast-hour + publish lag
margin(N)           = NOAA hourly object LastModified - predicted online(N)
```

`--launch-delay` defaults to zero. Other costs are required; no startup or render
budget is guessed. The formula reserves N full steady hours after the first
completed step, a conservative budget rather than a step-count extrapolation.
The steady measurement must include the intended history and product workload.
A products-off measurement does not establish performance with products on.

`--posting-times` reads the timing study's `rows` with `cycle`, `key`/`file`,
and `s3_last_modified` / `nomads_mtime`. If omitted, race reads that cycle's source listings and saves
its evidence locally. Both analysis f00 and the requested final hour must be
present. Unmeasured intervening hours stay unknown. The registry specifies the
comparison products: hourly `wrfprsfNN`, and the earliest of `wrfnatf48`,
`wrfprsf48`, `wrfsfcf48` at the final hour; the launch object is `wrfnatf00`.
The launch cost must include any wait for required companion GRIBs or indexes.
If only S3 evidence is retained for an old cycle, the prediction uses that
observed clock; it does not claim NOMADS was later.

Instead of three numeric flags, `--receipt FILE` accepts a JSON object (or its
`prediction_inputs` member) with `seconds_per_hour`, `launch_to_first_step`,
`publish_lag` and optional `launch_delay`, all in seconds. Numeric flags override
receipt values. A completed live `race.json` also works: race derives steady
cost from the first and last positive hour-close timestamps and uses the largest
observed publish lag. At least two published hours are needed for that cost.

```sh
gpuwm race hrrr --cycle 2026-10-21T00 --dry \
  --posting-times ./posting-times.json --receipt ./previous-run/race.json
```
