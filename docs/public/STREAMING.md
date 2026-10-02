# Streaming: forecasts that run as their source posts

[Tiling a domain that does not fit on the card](TILES.md) describes the
`[tiles]` table, which cycles a domain through the card a tile at a time.
This page describes when a forecast starts and waits against a source
that is still posting its cycle. Both can be used together: in a domain
tree, a tiled root loads its host store from the preparation head and
takes later boundary intervals as the model reaches each seam.

A source such as GFS, GEFS, HRRR or ICON posts a cycle one forecast hour at a
time, over one to three hours. A forecast that runs as the source posts
starts once the cycle's **first** hours are posted, and waits whenever it
reaches an hour the source has not posted yet. A forecast that waits for the
whole cycle starts only once the **last** hour is posted. Both produce the
same bytes from the same cycle; only the start time differs.

## What your engine does

Two things decide how early a forecast can start.

1. **Chained preparation.** A forecast starts at its preparation's head
   (the start state and the first boundary interval) and steps while the
   later intervals are prepared beside it. This is on by default;
   `GPUWM_CHAINED_PREP=0` turns it off. It holds for one domain and for a
   domain tree from native HRRR, a mapped source or the GFS series,
   prepared on the CPU or the card: the nests, including nests that start
   later and nests that follow a storm (a `gpuwm cyclone-setup`
   configuration), are prepared into the head, and the forecast waits only
   at the root's intervals. A GFS domain or tree, a native `hrrr` single
   domain or tree, and a single domain from `hrrr-prs`, `rap`, `rrfs`,
   `icon-eu` or `gem-gdps` also prepare as their hours post, through
   `gpuwm go` and `gpuwm run-plan` alike; a mapped tree, `gefs` (its member
   is checked over every file first) and ICON's global and D2 meshes (their
   files are normalized whole) are prepared once the whole window is
   fetched.
   In a domain tree, a `[tiles]` root follows the same chain, including
   `mode = "auto"` when the planner places it in host memory. Its start
   state loads from the head, each later boundary interval is checked at
   its seam, and the complete preparation is verified against the head at
   the end. Checkpoints taken before the seal bind the boundary prefix
   already prepared. A single `[tiles]` domain that builds its host store
   directly (`--stream-init store`, or `auto` when the resident state does
   not fit the card) still starts at the seal, because that road reads the
   sealed preparation. The other routes
   [Pipeline stages](PIPELINE-STAGES.md) lists are prepared whole before
   their forecast starts.
2. **Fetching as the source posts.** An engine that fetches each hour as the
   source posts it answers the readiness question (`--readiness` on
   `gpuwm go`, `gpuwm fetch` and `gpuwm run-plan`) and takes
   `as_posted = true` under `[fetch]`, its default. An engine without it
   fetches a cycle once its last hour is posted: `latest` means the newest
   cycle whose last hour is posted, and a named cycle whose last hour is
   missing is refused. New forecast's **Data posting** row says which of the
   two your engine does, and the MCP run tools refuse `as_posted: true` on an
   engine without it rather than start hours later than asked.

Whichever applies, every wait is reported the same way (below).

## Starting at the first hours

On an engine that fetches as the source posts, a run can start once one host
holds every object of its **start needs**:

- the window's first lead (the analysis) and the first boundary interval
  after it;
- the route's lead-free objects and the step-0 objects bound to the first
  lead (AIFS and GEM GDPS statics);
- a same-cycle donor lead the route declares (AI-GFS and AI-GEFS take the
  cycle's GDAS analysis);
- for a domain tree, each nest's own start lead when the nest starts later
  than the forecast;
- for a source that posts its cycle whole, the last lead.

After that the forecast waits for a later hour only when the model reaches
it. For every rolling source measured the model outruns the posting, so a
run launched at its first hours spends most of its time at seams, waiting
minutes each.

When each lead is expected comes from the source table: the earliest posting
seen per cycle hour, plus a per-lead slope, the same rows `latest` already
reads. Such an engine's rows also say how the cycle posts (`shape`), how late
a lead may be before the run stops (`late_after_minutes`) and how often a due
lead is asked about. `gpuwm sources ID --json` prints a source's row.

How each kind of source posts:

| Shape | Sources | What a run waits for |
|---|---|---|
| rolling | hrrr, hrrr-prs, gfs, gefs, rap, rrfs, icon-global, icon-eu, icon-d2, gem-gdps, aifs | the first hours, then each hour at its seam |
| whole cycle | ecmwf-open-data, gdas | the cycle, which appears complete at once |
| donor gated | aigfs, aigefs | the same-cycle GDAS analysis, after which every lead is posted |
| archive | era5 (keyless ARCO copy), 20crv3-cf | nothing: a window before the archive's end starts at once, one past it is refused naming the end |
| brokered | era5 (Copernicus CDS), era5-l137 | the job's file |

A tree adds nothing to its root's shape.

### Settings

```toml
[fetch]
source = "gefs"
cycle = "2026-09-30T12"
hours = 48
as_posted = true           # the default; false waits for the whole cycle
late_after_minutes = 60    # the default is the source row's own budget
```

`as_posted = false` (`--whole-cycle` on `gpuwm go` and `gpuwm fetch`) is a
choice, not a workaround: the run's output is the same either way. New
forecast offers it as **Wait for the whole cycle**, off by default. A run
plan carries the same choice as the `as_posted` run option, and
`late_after_minutes` as a run option or `--late-after-minutes`.

### The readiness question

The readiness question prints one `gpuwm.readiness.v1` JSON document on
stdout and runs nothing: the source, cycle and window, the posting facts,
each start need with its expected time, its late time and the host's answer
(`posted`, `not_posted`, `not_heard`, or none when not asked),
`expected_ready_at`, `expected_final_at`, and `state` (`ready`, `waiting`,
`refused` or `unprobeable`). With `--no-probe` it asks no host and gives the
schedule alone. Its exit code is **0** ready (or a source that cannot be
asked), **75** not ready yet (with `retry_after_seconds`), **2** refused: the
window can never start, and `refusal` says why.

## Waiting at a seam

A forecast that reaches a boundary interval not there yet waits between two
steps, after that instant's frames and checkpoint are written, and says why in
four places:

- **Events** (`events.jsonl`, relayed by `gpuwm go` and emitted by
  `gpuwm run-plan`): `source_wait_started`, `source_wait_progress` every 60 s
  and `source_wait_finished` when a source hour is not posted yet (or its
  host cannot be heard, which is said as such), each with
  the source, cycle, lead, its expected and late times and the model time
  reached; `boundary_wait_started` and `boundary_wait_finished` with
  `cause: "preparation"` when the hour is posted and its interval is still
  being prepared. [run-plan.md](../run-plan.md) lists every field.
- **`progress.json`** (`evidence/progress.json` for a domain tree): a
  `waiting` block with `on` (`source` or `preparation`), `phase`,
  `since_utc`, the lead and its times, and the interval, dropped when the
  wait ends.
- **`run-progress.json`**: status `waiting:source` or `waiting:preparation`
  with a `wait` record. The watchdog times a source wait against the lead's
  late time plus 120 s and a preparation wait against the preparation's
  silence limit, never against the step time, so a long wait is not taken
  for a stalled forecast. Under `gpuwm run-plan` (which `gpuwm go` uses for
  a mapped source), a run whose forecast failed while its preparation still
  runs says `waiting:preparation` until the preparation ends, so a retry can
  reuse it, and then reports the failure.
- **The run's page and My forecasts**: the live line says what the run
  waits for, when that hour was expected and when it counts as late, the
  model time reached and how long it has waited.
  `GET /api/runs/RUN/status` carries the same facts in its `wait` block.

The terminal says it too:

```text
forecast: waiting at 27:00 (valid 2026-10-01T15Z) for gefs f030, not posted yet (scheduled from about 15:53Z)
forecast: gefs f030 arrived after 38 s; stepping
```

## When a source falls behind

On an engine that fetches as the source posts, a lead not posted by its
expected time plus `late_after_minutes` stops the run with **exit 75**. The forecast stops at the seam; the frames already
written stay, and so does a checkpoint written at the seam. The run emits
`source_behind` (the lead, its expected and late times, the last answer,
the model time reached, the frames kept and the checkpoint) and then `failed`
with `error_class: "SourceBehind"`. The message names the lead:

```text
gefs f030 of the 2026-09-30T12 cycle has not posted by 16:53Z, 60 min after its scheduled
time (15:53Z; the source table's late_after_minutes is 60). The forecast stopped at 27:00
(valid 2026-10-01T15Z); the frames through that time are kept.
what to do: launch the same config again once f030 posts (it resumes from the fetched
prefix), or raise [fetch] late_after_minutes for this source.
```

A host that could not be heard at all is said as such, never as the
publisher being late. To resume the forecast from the checkpoint at the
seam, run `gpuwm go CONFIG --prepared-root PREPARED --restart CHECKPOINT`.
Do not retry in a loop: a lead is later than its budget.

Exit codes of `gpuwm go`, `gpuwm run-plan` and an as-posted fetch: **0**
completed, **1** failed, **2** refused before any work, **75** the source
fell behind, **130** interrupted.

## Launching from a schedule

A site that launches runs on a clock launches at the first hours posted, not
the last:

1. **Launch time.** For each product (source, cycle hours, window, domain),
   launch at `expected_ready_at`, the latest expected time among the start
   needs. It comes from the readiness question with `--no-probe`, or from
   the source's row in `gpuwm sources ID --json`. It is not
   `expected_final_at`.
2. **Box start.** Start the machine that long before, less its boot time.
3. **Ask on the box.** Ask the readiness question of the run plan: on 0,
   launch; on 75, launch anyway (the run waits for its start needs up to
   their late time and reports `phase: start` waits) or ask again after
   `retry_after_seconds`; on 2, skip this cycle and report the refusal.
4. **Name the cycle.** Launch with the cycle named, never `latest`: a minute
   early, `latest` is still the previous cycle, and the run would forecast
   from stale data.
5. **Follow the run** through `events.jsonl` (`posting_schedule`,
   `lead_posted`, `lead_ready`, the waits, `source_behind` and the last
   event) and the `waiting` block of `progress.json`.
6. **On exit 75**, the next cycle's launch is the retry, unless the product
   needs this cycle; then launch again once the readiness question shows the
   late lead posted.

On an engine that fetches a cycle only once its last hour is posted, the same
schedule launches at `expected_final_at` instead, and step 3 is the run's own
start check.

## Following an HRRR cycle with `gpuwm stream`

`gpuwm stream PLAN.toml` is the older controller for one case: it follows an
uploading HRRR cycle with a nested domain tree, one sealed hourly leg at a
time. Each leg is a new process that fetches the new hour, extends the root
preparation, rebuilds the nested hierarchy in full and resumes the forecast
from the preceding tree checkpoint. It stays until the native HRRR route and
its trees run as posted through `gpuwm go`; after that a schedule that names
each cycle does its job.

The experiment must contain at least two domains and set
`restart_interval_s = 3600`. The plan also needs the native root-domain JSON
and the WPS, native and stock WRF namelists of the prepared-hierarchy route.

```toml
schema = "gpuwm-stream-plan-v1"

[stream]
work_root = "/forecast/stream-job"
cycle = "latest"
cycle_count = 2
target_lead = 4
poll_seconds = 30
wait_timeout_seconds = 7200

[fetch]
cache_dir = "/forecast/cache"

[prepare]
experiment_config = "/forecast/config/experiment.toml"
domain_spec = "/forecast/config/root-domain.json"
wps_namelist = "/forecast/config/namelist.wps"
namelist_input = "/forecast/config/namelist.input"
stock_wrf_namelist_input = "/forecast/config/namelist.stock.input"
geog_root = "/forecast/WPS_GEOG"
physics_profile = "thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1"
pipeline_workers = 8
prepare_workers = 8
child_workers = 8
preprocess_backend = "cpu"
preprocess_workers = 8

[run]
io_mode = "history"
health_debug = false
gpu_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
allow_shared_gpu = false
```

`physics_profile` was `wsm6-ysu-mm5-noah-no-radiation-v1` in this plan
through 1.8.7. That name does not run "no radiation": it runs Dudhia
shortwave with longwave off, so nothing computes the downward longwave the
land surface reads, and a job that crosses local night runs the 1.7.1
dewpoint-collapse configuration. `thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1`
is the HRRR route's default and runs both radiation streams.

Run it once:

```console
gpuwm stream PLAN.toml
```

- `cycle = "latest"` is resolved when the command starts. Each later cycle
  must be the exact hourly successor; the controller refuses to leapfrog a
  missed cycle. `cycle_count` is bounded, so a completed plan is idempotent.
- `target_lead` defaults to 1. Every hourly lead keeps a complete root
  preparation, prepared hierarchy, forecast run and tree-checkpoint
  generation, and the controller never deletes older generations.
- `gpu_uuid` selects the GPU by UUID. With `allow_shared_gpu = false` the
  controller holds the supervisor UUID lock and refuses other compute
  processes on that GPU.
- The watcher requires `wrfnat`, `wrfprs` and both `.idx` indexes, records
  each one's URL, Content-Length, ETag, Last-Modified and first-observed
  time, and refuses identity drift after the fetch. The fetch manifest binds
  the URLs and transports to byte counts, sizes and SHA-256 digests.
- The initial `disk-capacity.json` prices every source hour, the cache copy,
  the kept forecast generations and a 2 GiB margin; before each fetch,
  `disk-headroom.json` rechecks the next writes, one generation and a
  512 MiB margin. A full-object envelope of 8 GiB per source hour refuses
  an observation above it before download.
- The work root holds `stream-summary.json` (the cross-cycle hash chain),
  `cycles/<cycle>/chain-summary.json`, one
  `cycles/<cycle>/legs/fNNN/chain-link.json` per sealed leg, and
  `active-cycle.json`, the crash-resume marker. A rerun verifies every kept
  link and artifact before it adopts completed work.
- The timeline reference `forcing_set_first_observed_at` is when this
  controller first saw the complete four-object set, not a producer upload
  time; `remote_ready_last_modified_at` is the newest Last-Modified. Each
  timeline row reports the root and hierarchy preparation intervals
  separately.
