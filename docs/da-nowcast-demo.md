# The radar-DA nowcast front door (demo-grade)

**First time here?**  Read
[`da-nowcast-quickstart.md`](da-nowcast-quickstart.md) instead: it covers
install, the GPU and cuSOLVER prerequisites, the interactive launcher,
the continuous daemon, and the failures you are likely to hit.  This page
is the reference for the pipeline itself.

One command takes any WSR-88D site id and a time window and produces a
cycled, ensemble, GPU-LETKF nowcast with a map-styled gallery that
verifies itself as reality arrives:

```
python -m tools.da_nowcast run --site XXXX --window-end latest --out CASE_DIR
```

That is the whole command.  **Verification is automatic**: when the run
finishes it hands the case to a detached rolling verifier that grades
each free-forecast frame as the archive covers its valid time, updates
the gallery in place, and stops with a verdict.  `--no-verify` opts
out; the same verifier can be started or re-run by hand later:

```
python -m tools.da_nowcast watch  --case-dir CASE_DIR   # roll until graded
python -m tools.da_nowcast verify --case-dir CASE_DIR   # one pass, then stop
```

## What it honestly is

A **demo**, productized from the live-fire engineering exercises
(receipts under `evidence/da-demo/`).  It is UNSCORED and outside any
registered campaign.  Defaults are the demo shape: **N=10 members**,
six applied 15-minute cycles, six free forecast legs, a single 3 km
domain sized to the radar's range authority.  Every figure it emits
carries that statement; free-forecast panels are stamped
"PAST LAST OBS" until an observed counterpart exists.

For how this configuration compares to the Warn-on-Forecast System --
members, domain, observations, compute, and why the FSS below does not
sit next to a published WoFS number -- see
[`da-vs-wofs.md`](da-vs-wofs.md).  That page also scores the demo against
the **operational HRRR**, the one external baseline that could be scored:
on one case it wins across the whole 90-minute free forecast, with the
margin peaking near +45 min and spent by +1:30.  The window is part of the
result -- quoting the win without it misstates the page -- and the HRRR
ingests strictly more radar data than this demo does.

Two capabilities live beside this quickstart rather than in it, because
each is optional and each changes what a run costs:

- [`da-nested-forecast.md`](da-nested-forecast.md) -- a fine one-way nest
  that runs inside the parent over the FREE forecast legs, inheriting the
  analysis rather than resampling it.  Reached with the `--nest-*` flags
  on `tools/da_cycle_prepared.py`; the parent is proven bitwise unchanged
  by the nest's presence.
- [`da_jacobi_eigensolver.md`](da_jacobi_eigensolver.md) -- the batched
  Jacobi kernel that factors the LETKF's own matrices.  There is no CLI
  flag for it: at `--solve-device cuda` and 2-64 members it is simply
  what runs, which means cuSOLVER is no longer required for radar DA.
  `gpuwm doctor` reports a "radar-DA eigensolver" line, and each cycle
  report now names the solver that produced it under
  `filter.eigensolver`.

Known limits, stated up front:

- **No skill claim.**  The verification numbers (>=35 dBZ column
  counts in the echo mask; FSS at 30 dBZ / 27 km via
  `gpuwm.verify.field_metrics`) are demo-grade diagnostics, not
  campaign scores.  N=10 is far below what ensemble calibration wants.
- **No velocity dealiasing** -- the obs ladder masks fold-risk
  signatures and counts every rejection (`tools/obs_radar_grid_build.py`
  provenance), which is not the same thing as unwrapping.
- **HRRR background by default** (Drew ruling, 2026-08-06: permanent;
  `--source gfs` is retained for archival reproduction of pre-HRRR
  runs), with the no-radiation WSM6/YSU demo profile.
- **Prepared cases are host-bound** -- preparation runs on the local
  box, by receipted finding; do not point the front door at remote
  prep.
- Storm motion for domain siting is **centroid displacement** between
  two volumes: it mixes advection with growth/decay and is used only
  to bias the domain downstream, never as a forecast.

## What to expect from the pictures

Read this before comparing a forecast panel to a radar panel, because
the honest target is narrower than the imagery suggests:

> **Band placement, orientation and convective mode by T+30..T+60 is
> the winnable target. A cell-for-cell match at T+90 storm scale is
> not winnable at any setting.**

Three separate reasons, none of them a defect:

1. **Predictability.** Individual convective cells have a useful
   lifetime of tens of minutes. No initial condition, ensemble size or
   resolution recovers a particular cell's position an hour and a half
   out; what survives is the system -- the band, its axis, its mode.
2. **Effective resolution.** A finite-difference core resolves roughly
   seven grid lengths, so a 1.5 km run resolves ~10 km features while
   the observed composite carries gate-scale texture. Placed side by
   side at native texture the forecast will always look too smooth,
   even when it is right. The per-member verification strips
   (`09-verification-*.png`) therefore show the observed field twice:
   at gate texture, and box-averaged in linear Z to the model's own
   resolving power. FSS is computed against the RAW observation in
   both cases, so the numbers never flatter the display.
3. **What the analysis can constrain.** Radial velocity alone nudges
   winds; it does not tell the filter where echo is and is not. A
   velocity-only single-radar analysis therefore leaves every member
   carrying the background's own misplaced storms -- the ensemble
   agrees with itself and disagrees with the radar. Assimilating
   reflectivity beside velocity (`--reflectivity-analysis` with
   `--hydrometeors` and an explicit `--positivity-policy`) is what
   moves placement; measured on one 1.5 km HRRR case (ktbw, 2026-08-06)
   it raised mean member FSS30 from 0.559 to 0.700 while member-to-
   member agreement barely moved. The members did not become more
   diverse; they became more correct. That is the intended behaviour,
   and it is also why an ensemble that looks tight is not automatically
   an ensemble that is confident.

## What one run does

1. **survey** -- S3 listing for the site; archive freshness measured
   and enforced (default ceiling 15 min; `--allow-stale` to override);
   two volumes decoded through the rw_nexrad seam; echo census and
   centroid-displacement motion.
2. **domain** -- a range-authority-sized box centered on the echo,
   biased downstream by the measured motion and clamped so the radar
   keeps observing the grid, emitted by the `gpuwm domain` wizard
   (config and WPS namelist are never hand-typed).
3. **prepare** -- authority materialization, GFS fetch, front-door
   manifest, prepared cache -- all hash-chained, all on this box.
4. **forecast** -- the georeference run that gives every observation
   its grid and every verification its wrfout.
5. **obs** -- one `gpuwm-obs.radar-grid.v1` file per cycle.
6. **cycle** -- N-member cycled LETKF (`tools/da_cycle_prepared.py`,
   CUDA solves by default) with free forecast legs past the last
   observation.
7. **render** -- the map-styled gallery (`tools/da_nowcast_render.py`):
   ArWen product-map frame from the vendored basemap assets, one
   reflectivity scale, honesty stamps throughout.

8. **verify** -- the handoff: a detached `watch` process polls the
   archive, builds each free-forecast frame's observed composite as
   its valid time gets covered, re-renders the gallery (counts and FSS
   on every pair, observed above forecast at the same valid time), and
   exits once the last frame is graded or its safety ceiling is hit.

## The defaults, and the measurements behind them

Set 2026-08-05 from runs on three cards.  Each number below is a
measurement with a receipt, not a preference; the code carries the same
citations (`tools/da_nowcast.py`, `DEFAULT_MEMBERS` and
`DEFAULT_MEMORY_BUDGET_MIB`).

### Ensemble size: `--members 10`

Bigger is not better here, and the reason is in the metric.  FSS is
computed on the ensemble **mean** composite, and these forecasts already
under-produce echo -- about 2,000 columns over 35 dBZ against about
2,800 observed.  Averaging pulls peaks down, so the mean field clears
30 dBZ in fewer places and moves *away* from truth as N grows.  Holding
one N=20 analysis fixed and varying only the averaging depth
(`evidence/da-demo/ensemble-size-sweep/skill-decomposition-partial.json`):

| depth | 1 | 2 | 4 | 8 | 10 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| FSS | .7470 | .7451 | .7435 | .7428 | .7425 | .7425 | .7423 |

The measured skill-and-cost ladder, same case and same scorer
throughout, mean FSS over the six free-forecast leads:

| N | FSS (32 GB) | wall (32 GB) | FSS (16 GB) | wall (16 GB) |
|---|---|---|---|---|
| 4 | -- | -- | 0.7331 | 431 s |
| 10 | **0.7408** | **465 s** | **0.7397** | **727 s** |
| 20 | 0.7423 | 847 s | 0.7435 | 1163 s |
| 36 | 0.7396 | 1826 s | -- | -- |

Receipts: `evidence/da-demo/ensemble-size-sweep/n{10,20,36}/` and
`evidence/16gb-frontier/runs/f198n{04,10,20}/`.

- **Up from 10 is not distinguishable from zero.**  With averaging depth
  held at 10 for both, analysis quality alone gives 0.7408 at N=10 and
  0.7426 at N=20: +0.0018.  Per member -- the statistic averaging cannot
  flatter -- +0.0038.  The across-member FSS scatter is 0.0062 at N=10
  and 0.0074 at N=20, larger than either effect, and the per-member
  difference is t = 1.47.  The price is +82% wall clock.
- **N=36 scores below N=10** on the mean field (0.7396 vs 0.7408) at
  3.9x the wall clock.
- **Down from 10 is measurably worse and saves nothing**: N=4 costs
  0.007 FSS at +15 min and 0.008 at +90, and leaves peak VRAM where it
  was.
- **N=36 is also where it starts to fall over.**  The N=36 arm of the
  2026-08-05 local sweep ran 1527 s and then died with
  `CUDA_ERROR_OUT_OF_MEMORY` copying back from the device, on a 32 GB
  card at the default 6144 MiB budget.  The card was shared with other
  work at the time, so this is not "N=36 needs more than 32 GB" -- it
  is "at N=36 the default budget leaves no room for a neighbour", which
  is the failure mode a first run should not meet.

One caveat on the cost column, stated because it is unresolved: the
N=36 timings were taken with cuSOLVER, and this tree now factors the
LETKF's own matrices with a batched Jacobi kernel by default.  The
*skill* half of the table is unaffected (the two solvers agree to about
1e-11); the *cost* half at large N is provisional.

### Memory: `--memory-budget-mib 6144`, and peak does not scale with N

The driver holds one trajectory on the GPU at a time and frees the pool
between members, so the ensemble is a **time** budget, not a memory one.
Measured whole-card peak on a 16,376 MiB RTX 4080, at 4 Hz with stage
markers (`evidence/16gb-frontier/runs/*/vram.json`):

| N | peak VRAM | % of card | chunk points |
|---|---|---|---|
| 4 | 15,946 MiB | 97.4% | 17,316 |
| 10 | 15,888 MiB | 97.0% | 6,912 |
| 20 | 15,796 MiB | 96.5% | 3,444 |

Peak is flat in N and what motion there is runs the *wrong* way.  The
last column is why: `letkf` sizes its chunk as `budget // per_point` and
`per_point` scales with members, so chunk points fall as 1/N while the
chunk's **bytes** stay pinned to `--memory-budget-mib`.  The model is

```
peak = 2 fields x N x nx x ny x nz x 8 B     (143 MB at N=10 here)
     + a chunk workspace pinned to --memory-budget-mib
     + whatever high-water CuPy's pool churn leaves behind
```

Only the middle term is under the operator's control, which is why it is
now reachable from the front door instead of only from the cycle driver.

**The default is measured-to-fit, not measured-optimal.**  15,888 MiB is
97.0% of a 16 GB card with 488 MiB spare, for a few seconds per cycle,
against a median of 2,168 MiB.  Lower it if you want margin.  The
budget ladder that would choose a better default, and the CuPy pool-cap
probe beside it (`tools/ens_sweep/pool_limit_probe.sh`, caps at
16/12/8/6/4 GiB), had **not finished** when these defaults were set, so
the shipped 6144 is unchanged rather than guessed at.

### `--profile card-16gib`

The 16 GB frontier produced a profile, and its content is that nothing
has to change:

```
python -m tools.da_nowcast run --site XXXX --window-end latest \
    --profile card-16gib --out CASE_DIR
```

The shipped demo shape -- N=10, 3 km, 49 levels, 198 km box half-width,
six applied 15-minute cycles, six free legs -- ran to completion on a
16,376 MiB RTX 4080 and returned the 32 GB card's answer: mean FSS
0.7397 over six leads against 0.7403 for the same case on the 5090, and
innovation RMS tracking the 32 GB run within 0.35 m/s cycle by cycle
across all six cycles.  (Not a controlled pair: driving today's wizard
with the shipped 198 km box half-width fits 136x134 on the node against
the 132x132 the 32 GB run recorded, 4.9% more cells at the same dx,
centre, radar and valid times.)  Cycle stage 533.8 s against 506 s on
the 5090; the solve is about 1.55x slower -- 65.3 s against 42.2 s
summed over six cycles -- because it is bandwidth-bound, while the
forecast legs run at near parity (3.4 s against 3.15 s per member-leg)
because at 893k state points they are launch-latency-bound.

So the profile does not shrink the run.  It points the memory preflight
at the card that is actually in the box (`--vram-gib 16`), which the
wizard would otherwise guess, and it names a configuration that has been
run rather than estimated.  An explicit flag always beats the profile.

Ada-specific, recorded while there: on sm_89 NVRTC `--ftz=false` **is**
effective and FP32 subnormals survive, unlike sm_120
(`evidence/16gb-frontier/receipts/ftz_probe_sm89.json`, with its
negative control).  The FP64-emulation countermeasure the 5090 carries
is unnecessary on a 4080.

### Off by default, and why

Both of these are opt-in, and neither is reachable from this front door.

- **The fine free-forecast nest** (`--nest-*` on
  `tools/da_cycle_prepared.py`, default off).  The parent is proven
  bitwise unchanged by the nest's presence, but its cost is only
  *computed* -- `evidence/da-nested-forecast/cost-model.json` says
  `"basis": "computed (gpuwm.core.preflight), not measured"`.  The
  measured A/B was still queued behind a busy card when these defaults
  were set.  The computed prices, for scale: a 60 km half-width 1 km
  nest adds 176 MiB and about 47 s to a control-only free forecast, and
  about 513 s if every member carries one.
- **Concurrent member advance** (`--member-workers`).  Not in this tree
  at all: the lane carrying it was held out of the integration with its
  byte-identity proof still open, and its extracted worker had three
  silent failures on the resume path.  There is nothing to enable.

### Pending measurement

Left at the current default deliberately, to be revisited when the run
that would settle it lands:

| Default | Left at | What is still running |
|---|---|---|
| `--memory-budget-mib` | 6144 | the 3-budget ladder at N=36, and `pool_limit_probe.sh` (CuPy pool caps at 16/12/8/6/4 GiB) |
| nested free forecast | off | the measured unnested / nest-control / nest-ensemble A/B |
| `--members` cost model | -- | N=64, and whether the N=36 solve cost survives the Jacobi kernel |
| launcher solve estimate | linear in N | the superlinear term's size on this tree's solver |

## Watching the grade arrive

The free forecast runs past the last observation, so its grade does
not exist when the run ends.  The verifier is what closes that gap
without a human in the loop.  Its state lives in the run's own
receipt, under `verification`:

| state | meaning |
| --- | --- |
| `pending` | frames known, verifier not started |
| `rolling` | a detached verifier is grading as the archive fills |
| `complete` | every free-forecast frame carries numbers |
| `incomplete` | the verifier stopped with frames still uncovered |
| `disabled` | `--no-verify` |

Beside it sit a verdict line, the watcher's pid, log and command line,
and one entry per frame with its status and its numbers.  The renderer
is the only place counts and FSS are computed; the verifier copies the
rows it publishes, so the receipt and the figures cannot disagree.
Polling that one file is enough to drive a progress display:

```
python - <<'PY'
import json, pathlib
v = json.loads(pathlib.Path("CASE_DIR/nowcast-receipt.json")
                .read_text())["verification"]
print(v["state"], "-", v["verdict"])
PY
```

## Which feed serves the observations

There are two routes to the same decoder, and the observation builder
picks between them with `--source`:

| mode | behaviour |
| --- | --- |
| `auto` (default) | prefer the real-time chunk feed; fall back to the archive when it cannot cover the request, recording why |
| `live` | the chunk feed or a refusal -- never a quiet fallback |
| `archive` | one Archive-II file per finished volume, the route this pipeline opened with |

The archive bucket only gains a volume file when the volume **ends**, so
its newest object is on average half a volume period old and at worst a
whole one.  The chunk feed publishes the same bytes in ~110 LDM records
as they are collected.  Measured on 2026-08-05 from this box, both
routes driven through `gpuwm.obs.radar_source.acquire_volume` at the
same instant for the same site:

| route | volume | feed lag | fetch + decode | end to end |
| --- | --- | --- | --- | --- |
| live, mid-scan | 97 of a still-turning volume | **0.0 s** | 3.6 s | **3.6 s** |
| live, newest complete | the finished volume | 382 s | 4.1 s | 386 s |
| archive | the same finished volume | 386 s | 5.3 s | 391 s |

`feed lag` is the newest object's S3 `LastModified` against the bucket's
own `Date` header, so it measures the feed and not this host's clock;
the receipt records the skew separately.  The live and archive routes'
finished volumes are **byte-identical** -- same sha256 -- because a
chunk-prefix concatenation is the archive file, not a re-encoding of it.

One instant is not an honest number for the archive, whose lag swings
from ~0 right after a volume lands to the whole volume period just
before the next one does.  Polled every 30 s across more than one volume
period on two sites, listing only:

| site | samples | live lag min/median/max | archive lag min/median/max |
| --- | --- | --- | --- |
| one 410 s VCP | 24 | 0 / **2** / 4 s | 18 / **197** / 406 s |
| one 197 s VCP | 24 | 0 / **1.5** / 5 s | 4 / **95** / 186 s |

Receipts: `evidence/da-demo/live-feed/`.  There is **no skill claim**
here.  This measures when an observation becomes readable, not whether
assimilating it earlier forecasts better.

### Partial volumes

A live assembly that stops mid-scan ends on an LDM block boundary and
decodes into the radials that arrived.  That is what puts the lag under
the scan time itself, and it is never implied:

- it must be asked for (`--allow-partial`, off by default);
- it is published as `{SITE}{YYYYMMDD}_{HHMMSS}_{FMT}_P{NNN}`, a name
  the archive key parser refuses, so nothing downstream can mistake it
  for a finished volume;
- the receipt and the NetCDF provenance carry `feed`, `partial_volume`,
  `chunks`, every object key, and the measured `feed_lag_seconds`;
- `--min-chunks` (front-door default 2) refuses an assembly too short to
  hold a radial at all.  Six data chunks make a full 360-degree sweep at
  0.5-degree azimuth spacing, so `--min-chunks 7` asks for one complete
  low tilt.

A gap in the chunk sequence truncates the assembly at the last usable
chunk and says so in `truncation`; a missing sequence-1 chunk or two
objects claiming one sequence number refuse the volume outright.
Nothing is padded, and nothing is skipped quietly.

## Seam for tooling

Everything is CLI + versioned JSON: each stage writes a receipt under
`CASE_DIR/receipts/` (`gpuwm-da.nowcast-stage.v1`, survey
`gpuwm-da.nowcast-survey.v1`), and a run ends with
`CASE_DIR/nowcast-receipt.json` (`gpuwm-da.nowcast.v1`) naming every
output and carrying the `gpuwm-da.nowcast-verification.v1` state
machine, rewritten atomically on every verifier tick so a poller never
reads a half-written file.  A GUI can drive the front door on this
seam without parsing any transcript.

Sites, times, and buckets are **arguments**.  No radar-site name
appears in this machinery, its defaults, or its identifiers, and
`tests/test_da_nowcast.py` checks that mechanically.
