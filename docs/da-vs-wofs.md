# Where the radar-DA demo sits next to Warn-on-Forecast

Three questions were asked of the demo run: **is it ten members**, **how
does that compare to WoFS**, and **could we run higher resolution with
three members instead**.

Short answers: yes, ten plus a never-analysed control.  WoFS runs 36 for
its analysis and forecasts from 18 of them, over a domain five times
larger, assimilating five observation types where we assimilate one.
And no -- three members is the one trade that cannot be made, for a
reason that is arithmetic rather than aesthetic, and the rest of this
page is that reason plus what to do instead.

Everything below about our system is measured on a single run
(`evidence/da-demo/live-fire-3/`, KDMX, 2026-08-05, six 15-minute cycles
04:15-05:30Z and a 90-minute free forecast).  Everything about WoFS is
cited.  Where the two cannot be compared, this says so instead of
comparing them.

---

## 1. Configuration, side by side

| | this demo | WoFS (cb-WoFS, SFE 2025) |
|---|---|---|
| ensemble | **10** + 1 control | **36** analysis members; forecasts from the first 18 |
| horizontal spacing | 3 km | 3 km |
| grid | 132 x 132 x 49 | 300 x 300 x 50 (Skinner et al. 2025 says 51; sources disagree) |
| domain | 396 x 396 km, fixed | 900 x 900 km, relocated daily on the SPC Day 1 outlook; two domains since 2023 |
| domain area | 157,000 km<sup>2</sup> | 810,000 km<sup>2</sup> (**5.2x ours**) |
| cycling | 6 cycles, 15 min, 90 min total | 15 min continuously, typically 12-15 h per event |
| DA algorithm | LETKF, our own, GPU | EnKF inside GSI ("GSI-EnKF") |
| observations | **radial velocity, one radar (KDMX)**; reflectivity and clear-air zeroes implemented, opt-in, unproven on real data | MRMS reflectivity **and radar "zeroes"**, WSR-88D radial velocity, GOES cloud water path, GOES clear-sky radiances, prepbufr conventional, Oklahoma Mesonet |
| analysis variables | **u, v only** by default; the scheme's full moisture and hydrometeor set under `--hydrometeors` | full state; reflectivity assimilated |
| free forecast | 90 min | 6 h hourly, 3 h half-hourly, out to 12-15 h after init |
| background / LBCs | GFS 0.25 deg | HRRRDAS 36-member 1-h forecast; HRRR + GEFS-perturbed boundaries |
| physics spread | none -- single profile, perturbed initial winds | 3 PBL x 2 radiation schemes across members |
| microphysics | WSM6, no longwave, Dudhia SW | NSSL two-moment; RUC LSM |
| compute | **one RTX 5090** | 2,800 cores / 120 nodes (retired Cray); Azure EPYC now, `<$1000` per 12-h run |
| wall clock | 506 s for the whole 3 h exercise | 30 min for one 18-member 6-h forecast on 54 nodes |
| product latency | not a product | forecasts on the viewer 25-30 min after analysis time |

**Every line where we are smaller or simpler.**  Fewer members (10 vs
36).  A fifth of the domain area.  One radar instead of a national mosaic
plus two satellite products plus conventional and mesonet data.  Radial
velocity only -- and reflectivity assimilation is what actually places
and maintains storms, so this is the weaker half of radar DA.  Two
analysis variables instead of the full state.  Ninety minutes of cycling
where WoFS spins up for two hours *before* it launches its first
forecast.  A 90-minute forecast against WoFS's six hours.  No physics
diversity, so our spread comes only from perturbed initial winds.  A
global 25 km background where WoFS starts from a 3 km convection-allowing
DA system.  One case; WoFS's published skill aggregates 95.

The one line where we are not smaller is compute: one consumer GPU
against 2,800 CPU cores.  That comparison is not clean either -- the
2,800-core figure covers 36-member DA cycling *plus* 18-member
forecasting *plus* 20,000 images and 5 TB/day of product, and
post-processing alone is 20% of WoFS's compute.  Ours is model
integration and the filter, with no product pipeline and no history
output at all during cycling.

---

## 2. The skill number, and why it does not go next to a WoFS number

The demo's headline is FSS 0.727-0.766 for the free forecast against
0.236-0.341 for a never-analysed control, over six frames from +15 to
+90 minutes.  Here is everything that number is, stated precisely enough
to be checked:

| property | value | where |
|---|---|---|
| formulation | Roberts & Lean (2008) fractional-coverage FSS, `1 - sum(Pf-Po)^2 / sum(Pf^2+Po^2)` | `gpuwm/verify/field_metrics.py:146` |
| truth field | **smoothed**, same boxcar as the forecast | same function, symmetric in its two arguments |
| threshold | 30 dBZ composite reflectivity | `FSS_BOX_KM`/`FSS_THRESHOLD_DBZ`, `tools/da_nowcast_render.py:59` |
| neighborhood | **27 km square SIDE**, i.e. a 9-cell box, +/-12 km | `half_width = round(27/2/3) = 4`; box is `2*hw+1` cells across |
| edge handling | edge-extended boxcar, no shrinking | `field_metrics.py:58` |
| scored field | **arithmetic mean over the 10 members of each member's column-max dBZ** | `mean_comp`, `da_nowcast_render.py:349` |
| verification region | the full 132 x 132 grid | |
| radar coverage of it | **65.6%**; the other **34.4% is filled with -35 dBZ and counts as observed no-echo** | measured, `kdmx-verify-0545.nc` |
| observed base rate | **22.3% of columns exceed 30 dBZ** | measured, same frame |
| sample | 1 case, 1 radar, 6 frames | |

Four of those lines are why this cannot be set beside a published WoFS
number:

**It is neither of the two things WoFS reports.**  WoFS publishes
*member* FSS (deterministic, one member at a time) and *ensemble* eFSS
(Duc et al. 2013 / Schwartz et al. 2010, a probabilistic score over the
whole ensemble).  Ours is a third thing: a single deterministic map made
by averaging ten members' reflectivity in dBZ, then scored as if it were
a forecast.  No WoFS paper reports that quantity.  Averaging in dBZ also
smooths the field and costs area -- our ensemble mean puts 16.3% of
columns over 30 dBZ against 22.3% observed, a frequency bias near 0.73 --
so it is not a stand-in for a member either.

**No absolute WoFS FSS is published in text.**  In Skinner et al. (2025),
Wang et al. (2022) and Kerr et al. (2023) the FSS and eFSS values appear
only in figures; the prose reports *differences* ("generally less than
0.05", "greater than 0.10").  There is no published number of the form
"WoFS FSS(30 dBZ, 27 km) = X" to put beside 0.727.  Producing one would
mean digitising figures, and this page does not do that.

**FSS formulations are not interchangeable, and the gap is large.**
Roberts et al. (2020) measured it directly: the same forecasts score
about **0.65 against a smoothed truth field and about 0.40 against a
binary one**, at an 80 km neighborhood.  The same paper warns that the
familiar "FSS > 0.5 is useful" rule does not carry across formulations.
Ours is the smoothed-truth kind -- the flattering kind.

**The threshold and the domain both flatter us.**  30 dBZ is the bottom
of Skinner et al.'s sweep, and Kerr et al. (2023, Table 5) put the 90th
percentile of MRMS composite reflectivity at 26.6 dBZ -- so 30 dBZ is a
common event, not a convective core, and our own measurement agrees
(22.3% base rate).  WoFS headlines 40 and 45 dBZ because that is where
the ensemble earns its keep.  Separately, a third of our verification
area is outside the radar and is scored as confidently-observed no-echo;
FSS rewards quiet area, and Roberts et al. (2020) found CAM skill is
"modestly better when computed over the entire CONUS than when limited to
the SFE daily domains ... presumably due to the abundance of easy nulls".

### What can honestly be said

- The DA works.  Going from 0.236 to 0.727 -- roughly tripling -- against
  an identically-configured, identically-scored control is a real
  measurement of what six cycles of velocity assimilation bought on this
  case.  That is an internally-controlled comparison and it does not
  depend on any WoFS number.
- **The ratio does not transfer.**  WoFS is benchmarked against the
  operational HRRR, which is itself radar-initialised.  Our control is a
  cold GFS start that has never seen an observation -- a far weaker
  baseline, so a far larger gain.  WoFS's own gain over HRRR is near zero
  deterministically and about +0.10 for eFSS.
- **The lead time is the easy window.**  90 minutes sits inside the 0-2 h
  band where Skinner et al. find WoFS members and the HRRR statistically
  indistinguishable.  Flat FSS to +90 min is a much weaker claim than
  WoFS's curves, which are scored to 3 and 6 h and decay materially
  (object-based CSI 0.7 to 0.4 over 3 h, Skinner et al. 2018).
- **Six cycles is at or below WoFS's own minimum spin-up.**  WoFS cycles
  for 2 h before its first forecast and typically 12-15 h in total, and
  Guerra et al. (2022) show POD climbing from ~0.65 to ~0.75 with the
  number of cycles a storm has been through.  The fair framing is
  "comparable to WoFS at its weakest cycling depth", not to a spun-up
  WoFS.

One number that will turn up in a search and must not be misread: the
**WoFSCast 0.9** FSS (Flora et al. 2025) is a machine-learning emulator
scored **against WoFS itself**.  It measures fidelity to a training
target, not skill against observed storms.

---

## 3. Why three members is the wrong trade

### The arithmetic

An ensemble Kalman filter does not carry a covariance matrix; it carries
members, and infers the covariance from their scatter.  A sample
covariance built from `N` members has **rank at most `N-1`**, because one
degree of freedom goes into the mean.  The analysis increment is a linear
combination of the member perturbations, so **the filter can only move
the state inside that `N-1`-dimensional subspace.**

With `N = 3` that subspace is **two-dimensional**.  Our analysis updates
`u` and `v` on 853,776 cells -- about 1.7 million numbers -- and a
3-member ensemble is allowed to correct them along two directions.

Localization is what makes any of this work, and it does raise the
effective rank: because each analysis point solves in its own local
patch, the global increment is a *patchwork* of independent `N-1`-
dimensional solutions, not one of them.  With our 12 km horizontal and
3 km vertical localization, the 396 x 396 km domain holds order a
thousand quasi-independent patches, so the effective global rank is order
`1000 x (N-1)`.  That is a genuine rescue and it is why storm-scale EnKF
is possible at all with tens rather than millions of members.

But it does not rescue **the local problem**, and the local problem is
the one that has to be solved.  In this run each analysis point saw up to
**143-173 observations inside its localization volume** (measured,
`cycle-report.json`, `max_local_obs` rising 143 to 173 across the six
cycles).  With `N = 36` the filter fits those ~150 local observations
with 35 degrees of freedom.  With `N = 10` it has 9.  **With `N = 3` it
has 2.**  Two degrees of freedom against 150 observations is not a
poorly-conditioned fit; it is a different activity.

Sampling error in the local covariance falls only as `1/sqrt(N-1)`: about
71% at `N = 3`, 33% at `N = 10`, 17% at `N = 36`.  Everything downstream
-- spurious long-range correlations, spread collapse, filter divergence
-- gets worse in that order.

### What we have actually observed

We have no measured skill-versus-`N` curve, and this page will not invent
one.  What the receipts do contain:

- At `N = 10` the ensemble's observation-space spread ran
  1.166 -> 0.795 -> 0.885 over six cycles: it sagged and then
  **self-recovered** as the storm grew.  That is a filter operating with
  its margin visible.
- In round 2, `N = 10` stabilised near 0.73 where **`N = 2` collapsed**
  (`evidence/da-demo/live-fire-2/`).  That is our own evidence, on our own
  system, that the small-`N` failure is real and is not far below 10.

I could not find a published study establishing a *minimum* member count
for storm-scale ensemble DA, and I am not going to cite one that does not
exist.  What can be cited is the operational reference point: **WoFS uses
36**, at 3 km and in both of its 1 km prototypes (Wang et al. 2022; Kerr
et al. 2023), and it has been through more than a decade of tuning.  Ten
is already an explicitly demo-grade choice on the low side of that.
Three would be below anything anyone runs.

### Why members are the cheap axis to buy and resolution is the expensive one

This is the part that makes the proposed trade backwards.  The two axes
do not cost the same, and they do not cost the same *by a lot*:

| | how cost scales | 3 km -> 1 km at fixed footprint |
|---|---|---|
| **members** | **linear** | -- |
| advance, resolution | cells x steps = `dx^-3` | **27x** |
| solve, resolution at fixed physical localization radius | active points x stencil slots = `dx^-4` | **87x** |

The solve term is the one people miss.  The localization stencil is a
Gaspari-Cohn disc measured *in grid cells*, so holding the radius at a
fixed 12 km while refining the grid grows it as `(radius/dx)^2`.  Counted
exactly out of the shipped code:

| radius/dx | disc slots |
|---|---|
| 4 (our 12 km at 3 km) | **45** |
| 6 | 109 |
| 8 | 193 |
| 12 (12 km at 1 km) | **437** |

Nine times the cells multiplied by 9.7 times the stencil is 87 times the
solve.  Members, by contrast, are strictly linear: the driver runs one
trajectory at a time (`tools/da_cycle_prepared.py:659`, `wire` ->
`execute` -> `teardown` freeing the whole CuPy pool), so `N = 36` costs
`37/11 = 3.4x` the advance of `N = 10` and nothing else changes.

And each member currently **underfills the card**.  At 853,776 cells this
run sustained **17.3 M cell-steps/s**; the 3 km CONUS grounding run at
24.8 M cells sustained **26.1 M cell-steps/s with heavier physics**
(Thompson + RTE-RRTMGP + Kain-Fritsch).  Our domain is small enough to be
launch-bound -- the GPU is about a third less efficient per cell here
than it is when properly loaded.  Because members are serial and only one
trajectory is resident at a time, **VRAM does not scale with `N` at all**
in the current design; `N` shows up in host memory and disk staging.

So the proposed trade spends the expensive axis (27-87x) to buy back the
cheap one (3.4x), and it spends it on the axis that is *not* currently
binding, to relieve one that is.

---

## 4. What to buy instead, and what it costs

All figures below come from a three-term model -- advance, solve,
staging -- calibrated on this run and reproducing its total to **0.6%**
(503 s predicted against 506.0 s measured).  Constants:
`5.79e-8` s per cell-step, `1.0e-9` s per (active point x member x
stencil slot), timestep `5 s` per km of spacing (a shipped rule,
`domain_wizard.py:1371`).  The solve constant cross-checks to within 15%
against an independent bench already in the tree at a completely
different operating point (`letkf.py:606`).

### (a) 36 members at our current 3 km grid -- do this

| | value |
|---|---|
| full 12-leg exercise | **~28 min** (1,705 s), against 506 s at `N = 10` |
| per 15-minute cycle | **~2.8 min** -- 5.4x faster than real time |
| VRAM | unchanged (members are serial) |
| host RAM | ~2.4 GB of resident snapshots |

3.4x the cost for 3.6x the members, and it lands us on WoFS's own
analysis ensemble size.  The one soft spot: the `R x R` eigendecomposition
is invisible at `N = 10` but is the same order as the gather term at
`N = 36`, so the solve could run up to 1.7x the modelled 24.4 s/cycle.
Even then the answer is ~30 min.  **This is the cheapest large improvement
available and it is the one the sweep in section 6 measures first.**

### (b) 1 km at the current 396 km footprint -- do not

| | value |
|---|---|
| cells | 7.7 M (9x) |
| full 12-leg run, `N = 10` | **3.1-4.1 h** |
| per 15-minute cycle | **21-25 min** -- **slower than real time** |
| solve alone, at a 12 km radius | ~593 s/cycle, larger than all the advances combined |

There is a second, quieter problem here: at 437 x 41 stencil slots the
memory budget drives `chunk_points` down to ~748, **below the ~2,048
saturation knee** in the tree's own throughput bench, so the real solve is
likely 25-33% worse than modelled.  Refining the grid without shrinking
the localization radius is the single most expensive thing in this
system.

### (c) 1 km over a storm-scale box -- this is the affordable one

At the 152 x 152 x 49 grid the sweep actually stages (1.13 M cells, 1.33x
our current count, `dt = 5 s`):

| `N` | localization | full 12-leg run | per 15-min cycle |
|---|---|---|---|
| 10 | 12 km (held) | ~36 min | 3.9 min |
| 10 | 6 km (scaled) | ~30 min | 2.8 min |
| 36 | 12 km (held) | ~2.1 h | **13.3 min** |
| 36 | 6 km (scaled) | ~1.7 h | **9.3 min** |

**36 members at 1 km over a storm-scale box is real-time-feasible on one
5090, and the localization radius is the knob that decides it.**

With one honest caveat that could overturn the `N = 36` rows: the 43.1%
active-point fraction above is a property of *this* radar geometry on a
396 km box.  A 152 km box sitting wholly inside one radar's coverage could
approach 100%, which raises the solve by up to 2.3x.  At that ceiling the
12 km-radius row goes to ~20 min per cycle -- **outside** real time --
while the 6 km-radius row stays near 11 min.  Which of those is right is
exactly what arms `B1km-loc6` and `B1km-loc12` of the staged sweep
measure.

### The recommendation

1. **Buy members first.**  `N = 36` at 3 km costs 3.4x and fixes the axis
   that is actually starving the filter.
2. **Buy resolution by shrinking the footprint, not by refining in
   place.**  Cell count is what costs; 1 km over a storm-scale box is
   1.33x our current cells, while 1 km over the current box is 9x.
3. **Scale the localization radius with the grid.**  It is nearly free to
   change and it is worth up to 4x of the solve.  Holding a 12 km radius
   at 1 km spacing is paying `(12/1)^2 = 144` cells of stencil for
   physics that only ever justified `(12/3)^2 = 16`.
4. **Consider analysing at 3 km and downscaling the free forecast.**  The
   filter is healthy at 3 km, and the expensive term is the *analysis*,
   not the forecast; a fine free forecast from a coarse analysis avoids
   the `dx^-4` solve entirely.  This is a design we have not measured, and
   it is not what WoFS-1km does -- that prototype assimilates at 1 km
   (Kerr et al. 2023) -- so it is a proposal, not a plan.

---

## 5. What this is

**A demo-grade system with one verified case.  Not a verified forecast
system.**

Every receipt in `evidence/da-demo/` is stamped LIVE-FIRE ENGINEERING
EXERCISE, NOT CAMPAIGN EVIDENCE, and that stamp is accurate:

- **One case, one radar, six cycles, six scored frames.**  A single draw
  with no error bar.  WoFS's numbers aggregate 95 cases x 8 forecasts with
  bootstrap confidence intervals.  This can show the machinery works.  It
  cannot state a skill level.
- **The baseline is doing nothing.**  FSS 0.72-0.77 against 0.24-0.34 says
  DA beat a never-analysed cold start.  Not persistence, not optical-flow
  nowcasting, not WRFDA, not the HRRR.
- **No dual-run byte comparison was done for any DA run**, so the standing
  no-ECC corruption screen has not been applied to these numbers.  Each is
  one sample on a card with no ECC.
- **VRAM is not measured anywhere in the DA receipts.**  The only
  GPU-memory lines are two device-wide `nvidia-smi` snapshots taken
  *before* the work started.  Every VRAM figure here is extrapolated from
  a different run with different physics on a different machine, and is
  labelled as such.
- **No velocity dealiasing.**  Fold-risk signatures are masked and
  counted, which is not the same as unwrapping.
- **The cycling legs write no `wrfout`**, so none of these timings include
  the per-member history a real product would have to write.
- **Only 2 of the 6 scored frames have their observation volumes hashed
  into the lane.**  The other four are graded in the gallery but their
  provenance is not committed here.

---

## 6. The staged sweep

The cost figures in section 4 are a model.  `evidence/da-demo/sweep/`
stages the experiment that replaces them with measurements, on this same
KDMX case, scored with the same FSS so the results land directly beside
the 0.72-0.77 already published:

| arm | what it settles |
|---|---|
| `A10-3km-N10` | re-measures the published run; the night's own control for card contention |
| `A36-3km-N36` | **the owner's question**: what a WoFS-sized ensemble costs and buys |
| `B3km-storm-N10` | footprint control -- separates "smaller domain" from "finer grid" |
| `B1km-loc6-N10` | 1 km with localization scaled to the grid |
| `B1km-loc12-N10` | 1 km with localization held at 12 km: the price of not scaling it |

Family A varies **exactly one flag** (`--members`) on live-fire-3's own
prepared case.  Family B has to rebuild the case for a new grid, and
therefore changes the verification region as well as the spacing --
which is why `B3km-storm` exists.  `tools/da_sweep_score.py` reproduces
all twelve published FSS values exactly from the committed composites,
so Family A is on the published axis by construction rather than by
assertion.

**The ensemble-size question was settled elsewhere, and this plan's
framing of it was too strong.**  Family A never produced a committed
result -- its N=36 arm died with `CUDA_ERROR_OUT_OF_MEMORY` after
1527 s.  The answer of record comes from the executed ladder,
`tools/ens_sweep/` with receipts in
`evidence/da-demo/ensemble-size-sweep/`, and it is what
`DEFAULT_MEMBERS` cites.

The two designs disagreed about what "confounded" means, and the
disagreement is worth keeping straight because only one of the two
senses is removable by careful argv:

- **Family A's sense -- the design confound.**  Does any flag other
  than `--members` differ between arms?  Its test asserts argv
  equality, and that discipline is correct and worth keeping.
- **The executed sweep's sense -- the metric confound.**  The scored
  field is the mean of N composites, so N is inside the estimator as
  well as inside the system being estimated.  No amount of argv
  hygiene removes that, which is why "varies exactly one flag,
  therefore controlled" was over-broad: it eliminates the first
  confound and is silent about the second.

The second sense was then measured rather than argued.  Holding one
N=20 analysis fixed and varying only the averaging depth gives a
monotonic decline from 0.7470 at depth 1 to 0.7423 at depth 20.  So on
this case the mean-field metric **penalises** large N -- averaging
dilutes peaks below the 30 dBZ threshold, and these forecasts already
under-produce echo -- and the published mean-field curve understates
what members buy rather than flattering it.  The same effect on the
baseline run itself: per-member mean FSS 0.7429 against 0.7403 for the
mean field.

Family A's instrument could not have found this: `tools/da_sweep_score.py`
computes one field per leg and emits no per-member statistic, so had
Family A run, its curve would have been reported as a clean controlled
measurement while silently carrying the depth term.  That is the reason
the ruling goes to the executed sweep and not to the more carefully
specified plan.

What the sweep still will not deliver: an error bar (one draw per arm), a
dual-run corruption screen, or a skill comparison against anything
stronger than a cold-start control.

---

## Sources

- Skinner, Stratman & Kerr, 2025: Comparing Short-Term Thunderstorm
  Forecasts from WoFS and the HRRR. *Wea. Forecasting* **40**, 1839-1858.
  doi:10.1175/WAF-D-24-0238.1
- Skinner et al., 2018: Object-Based Verification of a Prototype WoFS.
  *Wea. Forecasting* **33**, 1225-1250. doi:10.1175/WAF-D-18-0020.1
- Guerra et al., 2022: Quantification of NSSL WoFS Accuracy by Storm Age.
  *Wea. Forecasting* **37**, 1973-1983. doi:10.1175/WAF-D-22-0043.1
- Heinselman et al., 2024: Warn-on-Forecast System: From Vision to
  Reality. *Wea. Forecasting* **39**, 75-95. doi:10.1175/WAF-D-23-0147.1
- Martin et al., 2024: Cb-WoFS: Migrating WoFS to the Cloud. *BAMS*
  **105**, E1962. doi:10.1175/BAMS-D-23-0296.1
- Wang et al., 2022: An Experimental 1-km WoFS. *Mon. Wea. Rev.* **150**,
  3081-3102. doi:10.1175/MWR-D-22-0094.1
- Kerr et al., 2023: Results from a Pseudo-Real-Time Next-Generation 1-km
  WoFS Prototype. *Wea. Forecasting* **38**, 307-319.
  doi:10.1175/WAF-D-22-0080.1
- Roberts et al., 2020: What Does a Convection-Allowing Ensemble of
  Opportunity Buy Us in Forecasting Thunderstorms? *Wea. Forecasting*
  **35**, 2293-2316. doi:10.1175/WAF-D-20-0069.1 -- the FSS-formulation
  calibration used above
- Roberts & Lean, 2008: *Mon. Wea. Rev.* **136**, 78-97 -- the FSS we
  implement
- Flora et al., 2025: WoFSCast. *GRL*. doi:10.1029/2024GL112383
- HWT Spring Forecasting Experiment 2025 Operations Plan, sec. 2c and
  Table 14 (cb-WoFS configuration)

**Dating caveat on the WoFS column.** The most recent published WoFS
specification found is the SFE 2025 operations plan (May 2025), which
lists WRF-ARW v3.9+.  MPAS and JEDI are scheduled for WoFS v2, with a
Concept of Operations targeted September 2026.  No SFE 2026 plan was
found, so the table is *current as of May 2025*, not confirmed for the
2026 season.  The vertical level count genuinely disagrees between
primary sources (50 in the SFE 2025 table and the NSSL configuration
page, 51 in Skinner et al. 2025); it is reported unresolved rather than
guessed.

## Clear-air "zeroes": what ours are, and what caps them

WoFS assimilates radar *zeroes* -- observations that there is no echo at
a location -- and they are a large part of why it suppresses convection
the model invents.  We now build and assimilate them too, opt-in and off
by default (`--clear-air-analysis`).  Two things about ours are worth
stating plainly, because both are ceilings rather than settings.

**A zero here is a measurement, never an absence.**  A model cell is
reported clear only when enough gates were decoded to real numbers inside
it, all of them below the significant-echo floor, and no contributing
radar saw echo there.  A cell the beam never reached -- below the lowest
tilt, beyond range, behind terrain, or simply not scanned -- accumulates
nothing and makes no claim.  This is the distinction that matters: "no
echo was recorded here" covers most of a domain and is not an
observation, while "the radar measured this cell and found nothing" is.
The adapter refuses an observation file that has no clear-air assessment
rather than deriving zeroes from the echo mask, because that derivation
is exactly the mistake that fabricates data at continental scale.

**The decoder used to cap the yield, badly.  It no longer does.**  In the
WSR-88D Message-31 encoding a reflectivity gate word of 0 means "below
threshold" -- the radar looked and detected nothing, which is precisely a
zero -- and a word of 1 means "range folded", an ambiguous second-trip
return that may well be a storm.  The Level-II decoder in the vendored
Rust stack mapped **both** to NaN, and the pack builder NaN-fills any
radial that did not carry the moment at all, so downstream the three were
indistinguishable and NaN could not license a zero without inventing
observations wholesale.

The measured cost on a real KDMX volume (2026-08-05 04:28:56Z): 9,732,174
non-finite gates against 10,815 finite below-floor ones, and 46,216 echo
cells.  At the shipped clear-air settings that yielded **zero**
assimilable clear-air observations -- the result the adversarial review
ship-blocked the feature for.

`wx-radar` now records *why* each gate is not a number, as a `censor`
plane carried beside the moment plane.  `rw_nexrad decode --censor-flags`
transcribes it into a `gpuwm-obs.radar-sweeps.v2` pack and
`superob_volume(..., clear_air_from_censor=True)` builds zeroes from the
below-threshold gates as well as the finite below-floor ones.  The moment
values did not change and neither did the default: a decode without the
flag is byte-identical to what the tool always wrote, proved by
re-decoding the eight committed live-fire-3 volumes and matching their
pack digests exactly.

On those same eight volumes the yield goes from 0 assimilable clear-air
observations per volume (1 across all eight) to **17,943-20,831** -- about
126,000 clear cells before the driver's default thinning and ~19,900
after, against 41,897-64,775 echo cells.  See
`evidence/da-demo/clear-air-yield.json`.

**Range-folded gates never became clear air, and never will.**  Raw 1 is
a return the radar cannot place in range; assimilating it as "no echo"
would erase storms that are really there.  The censored regime tests for
equality with one code rather than for "not an echo", the DA adapter's
`clear_air_source` allow-list contains no regime built from folded gates,
and `test_range_folded_gates_are_never_clear_air_under_any_configuration`
sweeps every permissive parameter setting in both regimes and requires
the yield to stay zero.

**What zeroes can and cannot do for us.**  They suppress; they do not
initiate.  A zero only carries information where the model has
condensate, and there the ensemble has spread, so the filter can remove
spurious echo.  Where the radar sees a storm and every member is clear
the prior variance is zero and no observation can create one --
`gpuwm.da.perturb` perturbs only jointly-active species pairs, by design.
That is the honest split: reflectivity DA in this system removes storms
the model should not have and cannot conjure storms it never made.
