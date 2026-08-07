# GOES CWP assimilation: the path from pack to analysis

**Status: IMPLEMENTED, UNCALIBRATED, UNSCORED.** The path is complete and
tested end to end — a GOES CWP observation can change a gpuwm analysis,
in both directions. No forecast has been scored against one, and the
observation-error constants are stated assumptions rather than measured
ones. Read "What must happen before a scored campaign" before using this
for anything that gets a number attached to it.

Pairs with `obs-goes-cwp-bridge-design.md` (the `rw_goes` bridge, which
acquires and packs) and `obs-goes-cwp-operator-spec.md` (the one-page
design this implements, with one deliberate divergence recorded below).

## The ceiling: satellite CWP is a SPARSE constraint on an inner nest

Read this before planning a campaign around it, not after.

ABI cloud products are **2 km**. A model cell takes at most one satellite
pixel once the grid is finer than that, so coverage falls as the grid
refines — measured on a live CONUS scan against real domains:

| dx | cells with an observation |
| --- | --- |
| 12 km | 72% |
| 3 km | 86% |
| 1 km | **13.9%** |
| 333 m | **1.77%** |

At LES resolution a 2 km pixel cannot fill a 333 m grid, and 98% of cells
receive nothing. CWP does not become a dense analysis constraint on an
inner nest at any QC setting — **the localisation radius does nearly all
the spatial work**, and the honest expectation is a broad, smooth
condensate adjustment rather than cell-by-cell control. Budget for it as
a large-scale constraint carried into the nest, not as a nest-resolution
observation.

The good news in the same measurement: the QC gate is not what costs you
this. `phase_uniform_fraction = 1.0` rejects nothing at all at 3 km and
finer (zero mixed cells, because a cell holds at most one pixel). The
strict default is free where it matters. Full table under "Measured on
live granules".

---

## The three prominent assumptions

These are not buried. Each is also written into every product and every
receipt the path produces, so a reader who never opens this file still
meets them.

### 1. The observation error is UNCALIBRATED

There is no measured CWP observation-error covariance for this system,
and none this project can honestly borrow. So the five constants have
**no defaults anywhere**: `CwpErrorModel` requires all five, and
`tools/obs_goes_grid_build.py` makes all five required CLI flags. A stage
cannot inherit a confident-looking number nobody earned. The values used
are written into the product's `error_model` attribute alongside
`"calibration": "UNCALIBRATED"`, and travel from there into the cycle
report.

The form is the operator spec's: a constant for a clear-sky zero,
`max(rel * CWP, floor)` per phase class for a cloudy one, with
`rel_ice >= rel_liquid` enforced because the upstream ice coefficient is
flagged PROVISIONAL in every pack.

### 2. The thin/thick DQF inflation: CLOSED, and uncalibrated

**Was** unimplementable from a v1 pack, which carried DQF counts and the
condemn mask but no per-pixel plane. `gpuwm-obs.goes-cwp.v2`
(`lane/goes-bridge` c64185d6d) adds `cod_dqf` / `cps_dqf` / `actp_dqf`,
and the inflation is now implemented.

Measured on the live full-CONUS scan s20262161801170, 3,750,000 pixels:
**27,119 thin (bit 256), 775,251 thick (bit 512), 0 carrying both,
47,162 with an unreadable DQF word.** Thick alone is 20.7% of the sector,
so this was never a rounding correction — a fifth of the scene was being
assimilated at an error its own quality flag disputes.

How it is applied: a pixel is thin/thick if *either* DCOMP product's DQF
word says so (they measured identical on this granule, but that is an
observation about one scan, not a format guarantee). A cell's sigma_o is
multiplied by the **mean** inflation factor over the valid pixels
averaged into it, since the cell's value is their areal mean, and applied
*after* the class floor so the flag can still raise sigma above it.

The factors themselves are UNCALIBRATED like everything else in the error
model, default to 1.0, and requesting >1.0 against a v1 pack is a
**refusal** naming the schema — not a silent no-op, which is the failure
mode the version bump exists to remove.

NaN in a DQF plane means the flag itself was unreadable; those pixels
take a factor of 1.0 and are counted separately. They are never read as a
bitfield: `astype(uint16)` on a NaN returns garbage rather than raising.

What the pack's DQF policy *does* deliver is honoured by construction:
condemned pixels arrive as `NaN`, become no observation, and are counted.
The rule name and condemn mask travel into the product's `dqf_policy`
attribute so a consumer can state how its observations were screened.

### 3. `cloud_top_height_m` is treated as height above MSL

ABI ACHA publishes cloud-top height above the geoid; `TargetGrid.z_w` is
above mean sea level. The two are used as the same datum. A geoid-to-MSL
offset of tens of metres is far below a model layer's depth at cloud-top
altitudes and far above zero, so it is stated rather than absorbed. It
appears in every join receipt as `datum`.

---

## What each stage does

### Reading (`gpuwm/obs/goes_pack.py`)

The `GPWMGOES` container, checked the way `gpuwm/obs/sweeps.py` checks
`GPWMRDR1`: magic, version, declared lengths, JSON metadata, **schema
family**, `status == READY`, payload digest, then per-array dtype, shape,
and bounds — all before a payload byte is interpreted. Both families
share the reader, and `expected_schema` is how a caller fails closed on
which one it is holding: a cloud-top pack decodes perfectly and answers a
different question.

`tests/test_goes_pack_crosslane.py` hands packs to the **Rust** `rw_goes
verify` and requires PASS, and requires both lanes to refuse the same
corrupted payload. That is the only check that settles the container
contract rather than restating one lane's reading of it.

### The cross-grid join (`gpuwm/obs/goes_cwp.py::join_cloud_top`)

The bridge's separate-pack ruling (2026-08-06) says the bridge never
regrids and the join is the consumer's explicit, recorded choice. This is
that consumer.

**Method: `nearest`, by default, in geostationary fixed-grid scan-angle
space.** Both packs carry `x_scan_rad`/`y_scan_rad` and the same
projection, so the resample happens in the coordinate the instrument
samples — no reprojection through lat/lon, no assumption that the grids'
rows line up. Nearest is the default because **cloud-top height is
discontinuous at a cloud edge**: a bilinear blend across that edge
returns a height between "cloud at 12 km" and "no cloud", which no pixel
observed and which would place a CWP observation in clear air.
`bilinear` is available, is recorded when used, and leaves any cell
touching a NaN corner as NaN.

Before joining, three refusals: the `(satellite, sector, scan_start)`
pairing key must match; the geostationary projection blocks must be
identical; and where the cloud-top pack carries a `sibling` block from
`rw_goes cloud-top --pairs-with`, its `content_sha256` must be the CWP
pack's — so the pairing is *proved*, not trusted to a filename.

When no cloud-top pack is supplied the receipt still says so, explicitly,
because a product with every observation at the fallback height must not
read the same as one placed at real cloud tops.

### Superobbing and QC (`gpuwm/obs/goes_cwp.py::grid_cwp`)

Gates in series, every drop counted:

1. **Derivation cross-check.** CWP is re-derived from the pack's own
   `cod`/`cps`/`phase` with the pack's own declared coefficients, and a
   pack whose `cwp` plane does not reproduce is refused. A payload digest
   proves the bytes are the writer's bytes; it does not prove they are
   the numbers the writer said it was computing.
2. **DQF**, honoured upstream as above.
3. **`min_pixels`** — contributing pixels a cell needs (default 1).
4. **`min_valid_fraction`** — of the pixels landing in a cell, the share
   that survived the DQF gate (default 0.5).
5. **`phase_uniform_fraction`** — the share of valid pixels agreeing on a
   phase class (default 1.0, the design note's rule verbatim: "a cell
   half clear, half deep ice is not one observation"). Below 1.0 the cell
   takes the areal mean of every valid pixel and the *dominant* class's
   error model, and both counts are recorded.

Classes are `clear` / `liquid` (1, 2) / `ice` (3, 4) — the same branch
split `rw_sat::cwp` uses to pick a density, so a cell's class is the
class its CWP was derived under.

### Placement — and what it does not claim

CWP is a column integral. It has no height. The LETKF localises in metres
about an observation's gridpoint, so a column observation has to be
*centred* somewhere: at the retrieved cloud top where the join served one,
otherwise `fallback_placement_agl_m` (default 3000 m) above the cell's own
terrain.

`obs_level` is that centre. **It is not a claim about where the water
is**, and the reach around it is the caller's vertical localisation
radius. This is why `--cwp-vertical-loc-m` is a required flag with no
default: a column integral under a 4 km radius is assimilated as a 4 km
tall measurement, and in particular a clear-sky zero then cannot reach
model cloud sitting outside that slab. The adapter measures the radius
against the actual column depth and records the ratio; it does not
enforce a fraction, because what fraction is right is a scoreboard
question.

The mask carries **exactly one observation per column**. Repeating the
value down the column would hand the filter `nz` copies of one
measurement, each with the weight of an independent observation, and the
filter has no way to detect it.

### The observation operator (`gpuwm/da/obsop_cwp.py`)

```
CWP(j,i) = 1000 * sum_k q_cond[k,j,i] * (c1h[k]*mu[j,i] + c2h[k]) * (-dnw[k]) / G   [g m-2]
```

**The measure** is the model's own eta-coordinate column mass, taken from
`gpuwm/verify/cases/moist_bubble.py:125-134` (`_water_mass`) — this
project's only pre-existing column water integral, and the one its
moisture-conservation case is judged against. Preferred over `rho_d * dz`
for two reasons: it is exact in the discretisation (by the `alt`
definition in `gpuwm/core/kernels/diagnostics.cu:11` the two agree to
rounding, but only the mass form telescopes exactly to the column dry
mass), and it never goes stale — `gpuwm/da/perturb.py:1639` documents
`p`/`al`/`alt` as invalid after any state mutation, which is exactly the
condition an operator sees when evaluating a perturbed member.

**The species** are the model's own optical condensate: liquid = `qc`,
ice = `qi + qs`, from `gpuwm/core/rrtmgp.py:1097-1098` and
`gpuwm/core/rrtmg_legacy_prep.py:538-539` — the definition that feeds the
model's own optical depths, which is what an optical retrieval sees. Rain
is excluded, citing `gpuwm/core/rrtmgp.py:1243` (rain never enters the
model's cloud-fraction condensate) and `gpuwm/core/refl.py:13` (which puts
`qr`/`qs`/`qg` on the reflectivity side). Graupel and hail are excluded
for the spec's reason: large precipitating ice contributes little optical
depth per unit mass.

**`G`** is `gpuwm.core.constants.G` = 9.81, the constant the measure it
came from uses. `gpuwm/core/rrtmgp.py:1096` uses 9.80665, so a CWP
compared against RRTMGP's `clwp + ciwp` differs by 0.035% on this alone.

**Phase composition** uses the *observation's* phase, never the model's —
a state-dependent species selection would make H discontinuous in `x` and
give the filter a covariance it has no right to:

| observed | integrated |
| --- | --- |
| liquid / supercooled | `qc` |
| ice / mixed | `qc + qi + qs` |
| clear-sky zero | `qc + qi + qs` |

The clear row is the suppression row. A zero composed against `qc` alone
would leave model ice invisible to the one observation most confident it
should not exist.

#### The one deliberate divergence from the operator spec

The spec's v1 ice rule is `qc + qi`, with `qs` excluded. This
implementation includes `qs` by default, because the model's own optical
definition includes it in both radiation paths, and a spec written before
the operator was built is a weaker authority than the code the operator
must be consistent with. The divergence is recorded in every receipt as
`cwp_composition.diverges_from_spec`, and the spec's variant is one
argument away — `CwpComposition(ice=("qc", "qi"))`, or
`--cwp-ice-species qc,qi` on the driver. Which is right is a scoreboard
question, exactly as the spec says.

### Wiring (`gpuwm/da/radar_assimilation.py`, `tools/da_cycle_prepared.py`)

CWP is a batch in the same LETKF solve as the radar batches, with its own
errors, thinning and localisation — the only arrangement in which a
satellite column integral and a radar gate constrain the same state
without one being re-expressed as the other. `assimilate_radar_grid` now
also accepts `observations=None` for a satellite-only analysis.

Driver flags: `--goes-cwp` (one file per leg), `--cwp-thin-cells`,
`--cwp-err-inflation`, `--cwp-horizontal-loc-m`, `--cwp-vertical-loc-m`
(required), `--cwp-ice-species`.

Refusals at the CLI: `--goes-cwp` without `--hydrometeors` (CWP against a
wind-only state vector is sampling noise); without
`--cwp-vertical-loc-m`; and more satellite files than legs.

The cycle report's `analysis.assimilated` block names what was actually
assimilated per leg, so a leg whose satellite file held no usable
observation does not read the same as a leg that had none to read.

---

## Running it

```
python tools/obs_goes_grid_build.py \
    --cwp-pack scan.cwp.goespack --cloudtop-pack scan.cloudtop.goespack \
    --grid-wrfout wrfout_d01_... --valid-time 2026-08-04T18:01:17Z \
    --out goes_1801.nc \
    --err-clear-g-m2 ... --err-rel-liquid ... --err-floor-liquid-g-m2 ... \
    --err-rel-ice ... --err-floor-ice-g-m2 ...

python tools/da_cycle_prepared.py ... \
    --hydrometeors --positivity-policy clip \
    --obs radar_1800.nc --goes-cwp goes_1801.nc \
    --cwp-vertical-loc-m 12000
```

---

## What must happen before a scored campaign

Nothing below is a nice-to-have. Each is a reason a number produced today
would not mean what it appears to.

1. **Calibrate the observation error.** The five constants are stated
   assumptions. The defensible route is the one the radar lane already
   uses: run the cycle, read the innovation statistics this path reports
   (`innovations[].innovation_rms` against `ensemble_spread_mean` and
   `obs_error_mean`), and set sigma_o so the surplus lands where it
   belongs. Until that is done, `cwp_error_inflation` is tuning a number
   with no anchor.

   **Start with the floor, not the relative term.** Measured on a live
   CONUS scan, the cloudy CWP median is **65 g m-2** (p25 24, p75 172).
   So at any `floor_liquid_g_m2` near 40, `max(rel * CWP, floor)` returns
   the *floor* for more than half of all cloudy observations — the
   relative term never engages, and tuning `rel_liquid` moves nothing
   across most of the scene. The floor is the live parameter at typical
   cloudiness; the relative term only takes over in the upper quartile.
   The same is true of the ice branch against its own floor. Whoever
   calibrates should sweep the floors first and treat `rel_*` as the
   high-CWP tail control it actually is.

   Two more things the same measurement gives you for free: the observed
   distribution is heavily skewed (p99 2098, max 7982 g m-2), so a
   Gaussian sigma_o fits the bulk far better than the tail; and 60% of
   observations are clear-sky zeros, whose error is a single constant and
   therefore the single most leveraged number in the table.
2. **Settle the vertical localisation.** A required flag is not a
   calibrated one. The ratio the adapter records
   (`vertical_reach.fraction_of_median_column_reached`) is the diagnostic;
   what value it should take is unmeasured.
3. **A per-pixel DQF plane upstream**, or the thin/thick inflation stays
   unimplementable and ~8% of retrievals stay gated on
   contamination-risk grounds rather than error-weighted.
4. **The PROVISIONAL ice coefficient.** `rw_sat::cwp` uses a
   spherical-particle form at bulk ice density and says so. Every ice CWP
   observation this path assimilates inherits that. Open question 1 in
   the bridge design note is still open.
5. **Obs-skill on the scoreboard**, with the composition A/B (`qi+qs`
   versus `qi`) and the error constants as arms. Nothing here has been
   judged against MRMS/ASOS.
6. ~~A real-granule run.~~ **DONE, and it inverted the concern.** See
   "Measured on live granules" below.
7. **A cycling run through the driver.** Still open, and the blocker is
   assets rather than compute: the real DA case (the `obs_root` named in
   the last full cycle report) lives on the Linux nodes, and this box
   has no prepared root, no radar-grid obs and no member ensemble. The
   analysis path itself HAS now been run end to end on live packs against
   a real model vertical coordinate (below); what has not run is
   `da_cycle_prepared.py`'s own leg loop. Queue it behind node access.

---

## Measured on live granules (scan s20262161801170, 2026-08-04T18:01Z)

Full CONUS sector, 2 km, 3,750,000 pixels. 81.2% finite CWP: 1,827,048
clear-sky zeros, 382,639 liquid, 834,143 ice/mixed. Cloudy CWP
distribution, g m-2: p5 10, p25 24, **p50 65**, p75 172, p95 769,
p99 2098, max 7982.

That median matters for calibration item 1: a `floor_liquid_g_m2` of 40
would dominate `rel * CWP` for more than half of all cloudy observations.
The floor, not the relative term, is the parameter doing the work at
typical cloudiness.

### QC yield vs grid spacing — the concern was backwards

Against four real domains from the 1974 Super Outbreak nest ladder
(geometry only; the meteorology does not pair with a 2026 scene):

| domain | dx | pixels/cell | cells touched | obs at u=1.00 | survival | relaxing to u=0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| d01 | 12 km | 22.9 | 50,000 | 36,173 | 72.3% | +10,511 (+29.1%) |
| d02 | 3 km | 1.5 | 197,173 | 171,547 | 87.0% | +1,332 (+0.8%) |
| d03 | 1 km | 1.0 | 41,253 | 34,774 | 84.3% | +0 (+0.0%) |
| d04 | 333 m | 1.0 | 6,581 | 6,356 | 96.6% | +0 (+0.0%) |

**`phase_uniform_fraction = 1.0` is free at nest resolution and expensive
only on the outer domain.** At 3 km and finer a model cell receives at
most one 2 km satellite pixel, so class uniformity is satisfied trivially
and there are literally zero mixed cells to reject. The gate bites at
12 km, where a cell holds ~23 pixels. The strict default is therefore
right for a nested campaign, and the knob is an outer-domain question.

**The real constraint at nest resolution is coverage, not QC.** d03 sees
observations in 13.9% of its cells; d04 in 1.77%. That is geometry — a
2 km pixel cannot fill a 333 m grid — and it means CWP is a *sparse*
constraint at inner-nest resolution, with the localisation radius doing
nearly all the spatial work. Worth knowing before anyone expects a dense
satellite analysis on an LES domain.

### The coverage figure, reproduced independently

The 3 km row above was measured on a 1974-lineage domain over the Ohio
valley with v1 packs. A second, unrelated run — a 150x150 grid at 3 km
over the Gulf coast, 40 levels, built from the model's own
`make_vertical_coord`/`make_base_state` — landed **19,492 observations in
22,500 cells, 86.6%**, against the ladder's 86%. Different domain,
different vertical structure, same number.

That matters because coverage is the finding most likely to shape a
campaign plan, and it is a geometric claim (pixel size against grid
spacing) rather than a meteorological one. Two independent measurements
agreeing to half a percent is the evidence that it travels.

The same run reproduced the rest of the picture: cloudy CWP median
91 g m-2, H(x) 198–599 against observations of 0–1026 (same units, same
range), and every increment negative against an over-cloudy background.

### The cross-grid join earns its seat

Nearest served 2,277,975 of 3,750,000 pixels (60.7%), exactly the ACHA
finite fraction — the join loses nothing, it simply has no top where ACHA
published none. At d01, **16,151 of 16,153 cloudy observations (99.99%)
were placed at a retrieved cloud top** rather than the fallback height.
The 10 km to 2 km resample is not a bottleneck.

### The analysis path on live packs

Real v2 packs, the model's own `make_vertical_coord` / `make_base_state`
(nz=30, hybrid_opt=2, p_top 5717.8 Pa, 19,807 m column), a real Lambert
projection at 6 km over the Gulf coast, the real operator and the real
LETKF. 1,407 observations (923 clear, 465 liquid, 19 ice) of 2,304 cells;
269 cells error-inflated at a mean factor of 1.183.

H(x) came out 233–730 g m-2 against observations of 0–753 g m-2 — the
same units and the same range, which is the check this was for. Every
increment was negative (qc to -2.25e-4 kg/kg): the background was
constructed far cloudier than the real scene, and the analysis removed
condensate. Suppression works on real observations.

**This is a magnitude and plumbing check, not a skill statement.** There
is no gpuwm forecast valid at 2026-08-04T18:01Z on this box, so the
background is constructed rather than forecast, and the innovation
(-514 g m-2 against a 33 g m-2 sigma_o) measures that mismatch and
nothing about the system.

### Both lanes read v1 and v2

The v2 bump briefly made the shipped `rw_goes` v2-only, which stranded
every v1 pack whose digest an earlier receipt recorded. Fixed upstream
(rw-goes `acda95029`) rather than worked around here: `verify` now reads
every schema the tool has ever written, while `cwp` / `cloud-top` still
write v2 only. This reader always took both.

The capability flag both sides branch on is `per_pixel_dqf` — a fact
about the version, not a fault, so a v1 pack reports `false` and still
passes. Pinned in `test_both_lanes_read_v1_and_v2`.
