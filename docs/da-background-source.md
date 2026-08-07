# Choosing the background the radar-DA nowcast starts from

EXPERIMENTAL. Nothing here is on a default route, and GFS remains the
default background: an existing invocation that names no source gets
exactly the case and the answer it got before this document existed.

## What the choice is

The cycling driver integrates an ensemble forward from a **prepared
case** -- one deterministic first guess plus the lateral boundary series
that forces it. Which model produced that first guess is the background
choice, and until now it was GFS by construction rather than by
argument: the driver's front door
(`gpuwm.prepared_single_domain_forecast.preflight_prepared_forecast`)
knew three sources and HRRR was not one of them.

It is now four. `--source hrrr` reads the bundle the certified native
HRRR preparation publishes, through the same front door, with the same
hash-bound verification re-derived in every process that touches it.

Two things differ in the first guess, and both matter to radar DA:

| | GFS 0.25 degree | HRRR 3 km CONUS |
|---|---|---|
| grid spacing | ~25 km | 3 km |
| condensate at hour zero | `explicit zero (WRF Vtable.GFS parity)` | QC/QI/QR/QS/QG decoded natively |
| cycle cadence | 6 h | 1 h |
| usable window | 120 h of hourly leads | 18 h, or 48 h at 00/06/12/18Z |
| coverage | global | CONUS Lambert grid, fail-closed |

The condensate row is the one that changes the filter's job. Radial
velocity cannot create hydrometeors, so a GFS-initialised cycle asks the
analysis to build every storm out of wind increments and whatever the
hot-start nudge inserts. An HRRR background arrives with the storms
already in it.

## How to select it

Preparation, one added flag:

```
python -m gpuwm.source_cli --source hrrr \
    --source-root <grib2 dir> \
    --source-manifest <SHA256SUMS> --source-manifest-sha256 <digest> \
    --namelist-input <namelist.input> \
    --wps-namelist <namelist.wps> \
    --domain-spec <target-domain.json> \
    --static-cache <native-static.npz> --static-receipt <receipt.json> \
    --valid-time <YYYY-MM-DD_HH:00:00> --forecast-start-hour <lead> \
    --run-seconds <seconds> --history-interval-seconds <seconds> \
    --output-root <prepared root>
```

`--wps-namelist` is the whole opt-in. With it, the preparation publishes
three portable authorities beside the native bundle it has always
written -- `proof.json`, `source-input-manifest.json` and
`experiment.toml` -- and prints the digests the forecast stage binds
under `portable_bundle` in `public-wrapper-result.json`. Without it, the
output root is byte-for-byte what it has always been.

Cycling, with the digests that receipt printed:

```
python -m tools.da_cycle_prepared \
    --source hrrr \
    --prepared-root <prepared root> \
    --authority-dir <prepared root> \
    --proof-sha256 <portable_bundle.proof_sha256> \
    --source-manifest-sha256 <portable_bundle.source_manifest_sha256> \
    --prepared-content-sha256 <portable_bundle.prepared_content_sha256> \
    --physics-profile <profile> --run-seconds <seconds> \
    --history-interval-seconds <seconds> \
    --obs <volume>.json --grid-wrfout <wrfout> --out <run dir>
```

`--source` is a statement of which case this is, not a switch that
changes how it is read. A prepared root already IS one source's case,
and naming a different one is refused at the front door by the proof
schema.

## Where the ensemble spread comes from

**Perturbations, and only perturbations.** This is unchanged by the
background choice and the receipt says so in those words.

The driver builds one deterministic trajectory (`control`, never
analysed) plus N members. At leg 0 each member gets smooth Gaussian
initial-condition perturbations from `gpuwm.da.perturb` with its own
seed (`--seed` + member index): u and v by default, and with
`--hydrometeors` also theta, a lognormal qv, and every mass/number/volume
moment of each species the scheme advances, scaled multiplicatively so
the drop size distribution survives and nothing can go negative. Spread
is thereafter maintained by RTPS relaxation.

Nothing in that reads the background's provenance, which is why
substituting HRRR changes the central estimate and leaves the spread
machinery untouched. `gpuwm/da/background.py` records the construction
per trajectory rather than asserting it once, so a reader can tell a
control from a member and two members from each other.

### What refuses

`plan_member_backgrounds` runs before any GPU work and refuses three
things:

* **A perturbation that touches nothing.** Every amplitude zero and no
  species listed would hand the filter N bit-identical copies of the
  control -- an honest member count over a fabricated ensemble. Refused
  by name.
* **Two trajectories with identical construction records.** Compared as
  records, so the guarantee outlives any future construction.
* **A construction this driver does not implement**, with the reason:

  * `lagged-cycle` -- each member would need its own prepared case, and
    `tools/da_ensemble_state.py`'s `EnsembleIdentity` binds one
    `prepared_content_sha256` for the whole generation. N lagged members
    cannot share one generation today; building it means N prepared
    cases and a cross-case carry-forward that nobody has specced.
  * `multi-source` -- same blocker. It is the shape a public
    convection-allowing ensemble feed would want; there is no such
    public real-time feed today.

### What is NOT tuned

The shipped amplitudes -- 1.5 m/s wind at a 150 km length scale, 0.5 K
theta, 5% qv, 70% per species at 60 km -- were tuned against a
storm-free GFS first guess. On a background that already contains sharp,
balanced convection, an unbalanced 150 km wind perturbation is a cruder
instrument: it can radiate gravity waves off real storms and displace
structure HRRR got right. Treat the defaults as unproven under HRRR
until a controlled A/B says otherwise, and expect `--hydrometeors` to
matter more, not less.

The spread bar itself has to be re-checked too. A better-centred
background produces smaller innovations, which mechanically reduces the
spread the filter needs to explain them; a bar calibrated on GFS
innovations does not carry over unexamined.

## Coverage

HRRR's grid is CONUS-only and the refusal is fail-closed at two
strengths, both derived from the grid definition rather than a hand-held
box:

* `gpuwm.da.background.refuse_uncovered_area` -- the lat/lon envelope,
  computed over the boundary ring of the 1799 x 1059 native mass grid.
  The message names the true envelope and the way out
  (`use --source gfs ...`).
* `gpuwm.da.background.refuse_uncovered_domain` -- the stricter one. The
  interpolation stencil plus the surface-donor halo need real source
  cells on **every** side, so a domain that clears the envelope can
  still run its halo off the grid. That has been paid for once already,
  after a download rather than before it.

Practically: anything in Alaska, Hawaii, Puerto Rico, Guam or offshore
is HRRR-impossible, and a domain within roughly a halo-width of the
CONUS edge refuses at sizing depending on its size. Neither gate
extrapolates, pads or fills.

## What the receipt records

Every cycling report carries a `background` block
(`gpuwm-da.background-provenance.v1`) with the source and its label, the
cycle cadence and forcing interval, what the first guess carries for
condensate, the prepared case's own content digest, the cycle and lead
window taken from the hash-bound proof (not from the command line), and
one construction record per trajectory. A skill comparison between two
runs can attribute a difference to the background instead of guessing at
it.

## The daemon rides it, and HRRR is the default

**Drew ruling, 2026-08-06: HRRR is the WaH background, default and
permanent. GFS is retained only for archival reproduction of pre-HRRR
runs.** `tools/da_nowcast.py run` and the `tools/da_nowcast_auto.py`
daemon both default `--source hrrr` and say so in their own `--help`.

The wiring (`lane/wah-hrrr-background`):

* the window plan calls `gpuwm.da.background.plan_background_cycle`, so
  cycle cadence, per-lead publication lag and horizon are the
  registry's, not the planner's; `WindowPlan.gfs_cycle` became
  `background_cycle` with a compatibility shim and receipts carry both
  spellings for one deprecation cycle;
* the prepare stage speaks each source's own grammar: GFS keeps
  `--gfs-series` + `gfs-input-manifest.json`, HRRR passes the raw fetch
  directory bound by its `SHA256SUMS` through `gpuwm.source_cli
  --source hrrr` with the wizard-emitted `*.namelist.input` and
  `*.d01-target.json`, and `--wps-namelist` so the portable bundle is
  published; downstream stages bind the bundle's own three pins from
  `public-wrapper-result.json` and use the bundle root as the authority
  directory;
* the lead-0 hint hole is closed: a window ending at HH:30 with 2
  cycles puts init ON an hourly HRRR cycle, the wizard omits the
  `forecast_start_hour` hint at lead 0, and the fetch stage now reads
  the absent key as 0 for both sources instead of raising (issue #74).

The registry is the extension point on purpose: HRRR sits on NOAA's
retirement path behind RRFS, and the next background must arrive as a
`BACKGROUND_SOURCES` entry in `gpuwm/da/background.py` — cadence, lag,
horizon, coverage — not as another branch in the planner or the daemon.
