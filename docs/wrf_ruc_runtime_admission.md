# RUC runtime admission — what runs, and what measuring it changed

`sf_surface_physics=3` runs a forecast end to end. This document records the
admission, the enforced option identity, all 25 published restrictions
including the ones the schema has no field for, and the two open RUC leads,
both now settled by measurement rather than by argument.

Measured against pinned WRF v4.6.1 commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Reference hardware: one rented
RTX 5090, CuPy 14.1.1, glibc 2.39. RUC was the last of the three schemes
(MYNN, Noah-MP, RUC) still refused by
`gpuwm.physics_compat.pending_wrf_physics_components`.

## The forecast that admitted it

Not a unit test. `sf_surface_physics=3` on a 16×12×40 grid, dx=dy=3000 m,
dt=6 s, **600 RK3 steps = one hour of forecast**, 168 land columns over 24
water columns, with a 15 mm snow pack, `bldt=0` so the land surface runs on
every step. Run with `launch_noah`, `noahmp_lsm_step` and Noah's SFCDIAGS
refresh all replaced by raising tripwires, so "it ran RUC" is a property of
the process rather than of a configuration value.

| | |
|---|---|
| steps / forecast length | 600 / 3600 s |
| wall clock | 412.1 s; 686.9 ms/step mean, 679.5 median, 667.0/1390.3 min/max |
| per land column per surface call | ~4.1 ms, host FP32 |
| non-finite carriers after 600 steps | none |
| CuPy pool peak | 2.9 MB (the VRAM rail is 29-30 GB) |
| column census | 168 land + 24 water + 0 sea ice = 192, the whole grid |
| soil layers in the wrfout | 9, on `soil_layers_stag` |
| restart resumed at step 300, re-ran 301→320 | bit identical on all 43 carriers and all 6 dycore prognostics |

The physics is not merely finite. TSK starts at 294.7 K over a snow pack under
a 300 K boundary layer with HFX at −43 W m⁻², warms through the hour under
700 W m⁻² of shortwave, and HFX crosses zero near step 175 and reaches
+67.5 W m⁻² by the end of the hour. `TSLB(nzs)` is pinned to TBOT = 288 K on every land column and to
the water skin temperature on every water column, which is the difference
between `:1057` and the `:824-847` water arm and is asserted separately.

## The admission checklist

1. **Dispatch.** `PHYSICS_SLOT_DISPATCH["sf_surface_physics"]` gains
   `3 -> _run_ruc`. Before value dispatch existed, the driver branched on the
   *truthiness* of `sf_surface_physics`, so a RUC request ran Noah and produced
   a complete plausible forecast with no error. That is the failure mode this
   whole row exists to prevent, so the routing gate is a **tripwire on the
   Noah entry point** (`test_a_forecast_runs_with_every_other_lsm_entry_point_booby_trapped`),
   not an equality check on `driver.scheme_dispatch`. The tripwire is then
   pointed at RUC's own seam and shown to fire, because a booby trap nobody
   proved can fire is the same non-evidence as a gate that has never failed.
2. **The seam, not the column.** `module_surface_driver.F:3438-3596` is not a
   bare `CALL LSMRUC`. `gpuwm/core/ruc_runtime.py` is that arm: the
   unconditional sea-ice `ALBBCK = 0.65` override at `:3453-3459` (outside the
   `FRACTIONAL_SEAICE` block), the argument binding with WRF's own naming
   traps, `GSW = SWDOWN*(1-ALBEDO)` because RUC takes **absorbed** shortwave,
   the post-call `CQS`/`CHS` rebuild at `:3580-3585`, and `SFCDIAGS_RUCLSM`.
   Two of those are silent if you get them wrong. `:3506` passes `p_phy` — the
   layer **MID** pressure — into an argument LSMRUC names `p8w`; every
   saturation humidity in RUC divides by it (`qsg = qsn(soilt,tbq)/(p8w*1e-2)`),
   so binding the interface pressure is an O(dp/2) bias that looks right. And
   RUC brings its **own** 2-m diagnostic, so `3` is deliberately absent from
   `LAND_SURFACE_SFCDIAGS_SCHEMES`; borrowing Noah's SFCDIAGS would clamp
   nothing and would build Q2 from QSFC rather than from the QFX-derived
   proxy `module_sf_sfcdiags_ruclsm.F:97-101` says is deliberate.
3. **Allocation.** 12 extra 2-D state arrays, 2 nine-level soil arrays and 4
   published driver locals, **only** for this selector, so a Noah run's
   restart inventory, VRAM budget and health-descriptor count are unchanged
   (`test_a_noah_run_gains_no_ruc_arrays`).
4. **Cold start.** `ruclsminit` runs once at driver construction, where
   `module_physics_init.F` runs it, deriving SH2O/SMFR3D/MAVAIL/ZNT from
   TSLB/SMOIS by the freezing curve. The remaining carriers are left at the
   Registry cold state of zero because LSMRUC's own `ktau==1` block
   (`:481-565`) repairs six of them, and duplicating that repair in the
   allocator would give it two implementations.
5. **Preflight.** `physics_field_names_2d` and `physics_array_shapes` count all
   18 additions. Under-counting VRAM is a correctness bar on this hardware.
6. **Output.** RUC's names are emitted gated on the driver's **resolved
   routing**, not on field existence. `SMFR3D` and `KEEPFR3DFLAG` join
   `_SOIL_LAYER_FIELDS`, which is the bug this lane found in the pre-crash
   work: `_dims_for`'s shape table is keyed on `(nz, ny, nx)`, so a
   nine-level soil field over a forty-level column raised `KeyError: (9,4,6)`
   rather than picking a wrong axis. Descriptions and units for all 14
   Registry-spelled fields are transcribed verbatim from
   `Registry.EM_COMMON`, including the two rows that leave `units` empty; the
   four `ruc_*` driver locals keep their prefix and say in their description
   that they have no Registry counterpart.
7. **Restart.** `LAND_SURFACE_PARAMETER_SOURCES` gains
   `3: ("ruc_params", (vegparm, soilparm, genparm))` — the RUC **sections** of
   the same three files Noah reads, so the asset roles are shared while the
   bundle object is not. `LANDUSE.TBL` is absent because `gpuwm.core.ruc`
   never opens it. Unlike Noah-MP there is no solar geometry to bind: RUC
   takes GSW and GLW as forcing and reads no latitude, no `julian` and no
   `cosz`.
8. **Readiness.** `pending_wrf_physics_components` no longer refuses RUC. It
   refuses two narrower things: a soil geometry that is not nine layers, and
   RUC paired with the MYNN surface layer (two 2-m diagnostics, the second
   silently winning).
9. **Registry.** `land_surface.ruc-lsm` is `implemented: true` at maturity
   `implemented-unverified`.  `registry_sha256` moves
   `c3abb1b23a11a503f64088f70d3c4020f5556c250d1c4dc5ea407955bd5a56df` ->
   `d558eced52c10901fdde42ce7b78133f3767f60d48fdcc399f5b76d67d6ec195`.
   The four knobs also gained `consuming_read` citations, without which a
   SELECTABLE option pinning them is a contradiction the registry's own spec
   check refuses; and `num_soil_layers`' warning, which said only 4 was
   accepted, was corrected in `tools/build_registry.py` rather than in the
   generated JSON, because the builder owns that row and a hand-edit would be
   reverted on its next run.

## The enforced option identity

`gpuwm.config.RUC_OPTION_IDENTITY_EVIDENCE`, refused by
`validate_run_config` **before a run starts**, and re-checked at the seam that
would have had to implement each branch.

| knob | admitted | why that is the only value |
|---|---|---|
| `mosaic_lu` | 0 | dead-proved. `ruc_surface_parameters` is fail-closed on SOILVEGIN's mosaic arms, and LSMRUC's irrigation block (`:984-1009`) is gated on the same `mosaic_lu==1`, so it is unreachable wherever SOILVEGIN is. |
| `mosaic_soil` | 0 | the `soilctop`/`nscat` half of the same refusal. |
| `flag_sm_adj` | 0 | no consumer. `share/module_soil_pre.F:2063` reads it inside `init_soil_ruc`, i.e. in real.exe, to adjust a Noah-derived RUC soil state — gpuwm's RUC soil ingest (`gpuwm/ingest/ruc_soil.py`) is `init_soil_3_real`, which never reads it. |
| `spp_lsm` | 0 | not ported. The ARW/`EM_CORE==1` surface path is active, but `LSMRUC:446-450` additionally needs the stochastic `pattern_spp_lsm`/`field_sf` inputs and their restart contract when this knob is nonzero. |

RUC's namelist surface is only four knobs, which is precisely why the
interesting restrictions are the ones with **no** namelist field.

## Restrictions the schema has no field for

The remaining restrictions are data in
`gpuwm.core.ruc_runtime.RUC_RUNTIME_RESTRICTIONS`; each carries what gpuwm
does and why it differs. The most consequential:

* **The WRF-ARW surface seam is active** — `EM_CORE==1` consumes
  `rainncv`/`snowncv`/`graupelncv` at `:618-652`; the lake bypass at
  `:823-826`, fractional-sea-ice pre/post blend in
  `module_surface_driver.F:3461-3577`, and radiation-cadence GSW carrier are
  also wired. The historical `EM_CORE==0` oracle remains explicit-only.
* **`RUC_soil_ingest_is_wired_but_RUC_is_not_a_registry_profile`** —
  formerly `there_is_no_RUC_SOIL_INGEST`.  The remap exists
  (`gpuwm/ingest/ruc_soil.py`, max_ulp 0) and is wired: the shared
  `preprocess_land_surface_soil` seam routes `sf_surface_physics=3` to it and
  schemes 2/4 to Noah.  The "Still open" paragraph below and its superseding
  section record the history; what remains is registry-profile membership
  (source-absent-state tables) and the host-side column loop.
* **`sr_is_a_BINARY_phase_proxy_off_the_wsm_family`** — this lane's sweep for
  an undisclosed restriction of its own, and the RUC analogue of the `FLAG_QS`
  that MYNN shipped undeclared. Under `mp_physics` in (1, 6, 8, 10, 18) SR is
  the microphysics scheme's own; under anything else gpuwm substitutes the
  **binary** proxy `T(k=1) <= 273.15`, so every precipitation event is
  all-rain or all-snow with no mixed phase. RUC consumes SR as a continuous
  fraction at `:654-660` (`snowrat = rainbl*frzfrac`). `frpcpn=True` itself is
  *not* the divergence — WRF sets it whenever SR is present
  (`module_surface_driver.F:3448-3452`) and ARW always passes it — and the
  other arm of WRF's own gate is worse and worth naming: with SR absent WRF
  sets `SR = 1.`, i.e. all precipitation frozen.
* **`ilnb_is_defined_not_reproduced`** — see below.
* **`the_column_loop_runs_on_the_host`** — `gpuwm/core/ruc_gpu.py` has no
  device `sfctmp` and no device `lsmruc`; the CUDA leaves stop at
  `ruc_snow_soil_step_cuda`, so there is nothing to launch. ~4.1 ms per land
  column per surface call. This is the scheme's scaling blocker and it is
  measured, not estimated.

One restriction is recorded as a **negative** finding, because the equivalent
assumption shipped undeclared in the Noah-MP lane and `sf_urban_physics` is
not even a `RunConfig` field, so no schema check could have caught it:
**RUC in WRF v4.6.1 has no urban coupling at all.** `CASE (RUCLSMSCHEME)`
(`:3438-3596`) contains no reference to `sf_urban_physics`, no urban
`PRESENT()` guard and no urban call, unlike the Noah arm (`:2702`) and the
Noah-MP arm (`:3185`, `noahmp_urban`). There is nothing to omit.

## Lead 1: `udrunoff` is exactly 0 in all 48 fixture cases — correct

Settled from the WRF source and then driven. `:1041` is
`udrunoff = udrunoff + runoff2*dt*1000`. In `soilmoist`, `:5901` zeroes
`runoff2` on every call and the **only** accumulation is `:6071`/`:6073`,
inside `else if(qq.gt.dqm)` for levels `k=2..nzs` — a *sub-surface* level
whose implicit solution overshoots saturation. The top level's saturation
overshoot goes to `runoff` (surface) instead, at `:6048`; WRF even has the
`runoff2` version of that line commented out at `:6043`.

So `udrunoff == 0` on 48 unsaturated fixture columns is **correct, not a port
defect**. Two things measurement corrected about how to prove it, both
counter-intuitive:

* **Rain does not drive `runoff2`.** Swept over `rainbl` in
  (0, 1, 10, 40, 200) mm the value is bit-identical, because `infmax` caps
  infiltration at `:6009` and every excess millimetre is sent to surface
  runoff at `:6012`. Water cannot reach a deep level fast enough to
  oversaturate it. The first attempt at this test raised the rain and failed
  for exactly this reason.
* **The threshold is `smois >= MAXSMC`, not `smois > dqm`.** Measured on
  STAS-RUC loam (`MAXSMC=0.451`, `dqm = MAXSMC - DRYSMC = 0.401`): 0.45 gives
  exactly zero, 0.451 gives nonzero.

The standing gate drives a fully saturated loam column with **no rain at
all**, so it has `sfcrunoff` exactly zero and `udrunoff` nonzero — the exact
inverse of the fixture pattern that opened the lead (`sfcrunoff` nonzero in 14
of 48, `udrunoff` zero in all 48) — and asserts `:1041` exactly, in the port's
own float32 grouping `fl32(fl32(runoff2*dt) * 1000)`. Measured
`runoff2 = 1.2418e-9 m s⁻¹` giving `udrunoff = 1.4901e-5 kg m⁻²`. The
converse control is asserted too: rain on an unsaturated column moves
`sfcrunoff` and leaves `udrunoff` at exactly zero, so neither accumulator is
mirroring the other.

## Lead 2: WRF's uninitialised `ilnb` — gpuwm implements the defined behaviour

`sfctmp` declares `ilnb` as a plain local (`:1385`) and never initialises it;
`snowseaice` takes it `intent(inout)` (`:3896`) and `snowtemp` takes it
`intent(out)` (`:4994`), so it is undefined on entry in both by the Fortran
standard. Both assign it only inside `if(snhei.ge.snth)`
(`:4038`/`:4059`, `:5124`/`:5148`) and both then read it at `if(ilnb.gt.1)`
under the wider `if(snhei.gt.0.)` (`:4410`, `:5716`), selecting the one- or
two-layer `tsnav`.

The unassigned window is exactly `0 < snhei < snth`. There
`deltsn = 0.05e3/rhosn` is five times `snth = 0.01e3/rhosn`
(`:3387-3388`, `:3945-3946`), so `snhei - deltsn < 0` and the two-layer form
weights a **negative** thickness. **A pack thinner than the depth at which WRF
models one layer cannot have two: `ilnb = 1` is the defined answer, not merely
the safe one.**

The runtime passes `ilnb=1, ilnb_chain=False`, so no column inherits another
column's value and the answer is order-independent. The gate is a measurement,
not an assertion: it runs the same physical columns in forward and reversed
memory order and requires the same `tsnav`, then shows WRF's chained form
gives *different* answers in the two orders on the same columns — so the
divergence is real and the order-independence claim is testing something.
`oracle/lsmruc.csv` still pins WRF's chained answer, because that is what WRF
did, which is why `ilnb_chain` defaults to `True` in `gpuwm.core.ruc` and is
turned off only by the runtime.

Published in three places a user actually reads: the `DEFINED_ILNB` docstring,
the `ilnb_is_defined_not_reproduced` restriction row, and the registry warning
list.

## What the forecast found that the unit tests did not

Two things, both of which read like defects and are not, and neither of which
any per-routine oracle gate could have surfaced.

**The snow pack vanished in the first 48 s of model time.** The run started
with SNOW=15 mm / SNOWH=0.08 m and the first wrfout frame, at t=600 s, already
read SNOW=0. `RUC_SMELT` read 0 in the same frame, which does not clear it —
`RUC_SMELT` is an instantaneous *rate* at frame time, not an accumulator, and
`ACSNOM` (which is) is not in the history output. Measured step by step, the
pack **melts** and is fully accounted for: `ACSNOM` reaches 15.0020 mm against
15.0 mm of initial SWE over eight 6 s steps, and `SWE + ACSNOM + SFCEVP/2`
closes to 0.03%. What was wrong was the *initial state*, not the scheme: a
15 mm pack was handed a 303 K skin temperature over a 295–303 K soil column,
and a warm 5 cm soil layer holds about 3.4 MJ m⁻² of excess heat against the
5.0 MJ m⁻² the pack needs, so there is nothing unphysical about the rate. TSK
dropping 303 → 292.5 K on the first step is that inconsistency resolving. A
60 mm pack on the same soil melts 9.5 mm in step one and then stops, once the
interface has cooled and the remaining pack insulates it; and an identical
15 mm pack over a subfreezing column keeps its SWE to within 1e-3 mm and
accumulates exactly zero melt over 12 steps. All three are gated by
`test_the_snow_mass_budget_closes_and_a_cold_pack_persists`.

The consequence for a caller is worth stating plainly: **RUC's snow branch is
only meaningfully exercised when TSLB is consistent with the pack.** Supply a
subfreezing soil column, or the pack is an initialisation transient — which is
also why the frozen configuration, not the warm one, is where the SMFR3D and
KEEPFR3DFLAG gates live.

**RHOSNF reads −1000 kg m⁻³ for the whole forecast.** LSMRUC seeds it at
`:552` in the `ktau==1` block and only ever overwrites it from `rhosnfall`,
the density of *new snowfall*, so an unrained run reports the sentinel forever
— even with a pack present and melting. Pinned by test and published, because
a negative density in a wrfout reads as corruption, and because it means
RHOSNF cannot be used to infer anything about an *existing* pack: it describes
the last snowfall, not the snow on the ground.

**A known output gap, not fixed here.** RUC writes `ACSNOM`, `SFCRUNOFF` and
`UDRUNOFF`, and none of the three reaches the wrfout, because they are generic
`_F2D` names and gpuwm's history writer emits a curated subset for every LSM
rather than the whole surface field dict. `RUC_RUNOFF1`/`RUC_RUNOFF2` (the
rates) and `ACRUNOFF` are emitted. Widening the generic emit list would change
Noah's output too, so it is recorded rather than done: today the `udrunoff`
finding above is reachable through the restart and through
`driver.fields`, not through a history file.

## What running it measured that no routine gate could

**18 of RUC's 43 restart carriers are inert.** Every one round-trips bit for
bit, but its restored value provably does not reach the next step because
LSMRUC recomputes it first: diagnostics assigned unconditionally every call
(`grdflx = -sflx` at `:1096`, `lh`, `sfcexc`, `smstav`, `smstot`, `chklowq`,
`snowc`, and T2/TH2/Q2 which SFCDIAGS_RUCLSM rebuilds after the LSM), table
lookups SOILVEGIN refreshes every call (`emiss`, `lai`, `z0`), and state the
freezing curve re-derives (`sh2o`, `qsg`). The list is
`RUC_MEASURED_INERT_CARRIERS` and it is a gate, not prose — a user integrating
a checkpointed SFCEXC or GRDFLX as model state is reading a receipt of the
last call.

**SMFR3D and KEEPFR3DFLAG are conditional carriers, and this is WRF's own
structure.** `soilprop` rebuilds `soilice` from TSO and SOILMOIS through the
freezing curve (`:2695-2703`) and consults the restored SMFR3D **only** as the
cap `soilice(k)=min(soilice(k),smfrkeep(k))` at `:2704-2707`, inside
`if(keepfr(k).eq.1.)`; and `keepfr` is assigned 1 at `:2744` only inside
`if (soilice(k).gt.0.)`. On a warm column `log(tso/273.15) >= 0`, so
`soilice` is 0, `keepfr` never reaches 1, and SMFR3D is **never read at all**
— measured: 0 of 216 soil cells have `KEEPFR3DFLAG==1` on the warm grid, 16 of
216 on a 265 K one, all at soil level 1. Because the read is a `min()`, the
nudge must also be *downward*: scaling SMFR3D to 0.1× at a `keepfr==1` cell
reaches 21 other carriers, while adding +0.05 at the same cell reaches 3 and
the same downward scaling at a `keepfr==0` cell reaches none. Both halves are
gated.

### The restart falsification threshold, and why it is not a field count

The Noah-MP lane's restart gate was later found fixture-tuned: ">= 5 of 20
fields move" held on its grid and only 2 of 51 moved on a verifier's. How many
fields a perturbation reaches is a property of the grid, the scheme's coupling
and the length of the watch list — none of which the claim depends on. What is
configuration-independent is the claim itself, that the restored carrier feeds
the next step, and its minimal witness is **"at least two OTHER carriers
moved"**. Two, not one: a nudged TSLB reaches GRDFLX *alone* at any size down
to 0.001 K, because `grdflx = -sflx` is assigned from the soil solve
unconditionally, so one mover does not distinguish a coupled column from a
single bookkeeping assignment.

The pre-crash version of this test added **one ULP** to TSLB and the next step
came back bit identical on every watched field. That reads like a broken
restart and is not one. Sweeping all 43 carriers found TSLB live, and then
sweeping nudge *size* found the reason:

| nudge | others reached, unfrozen grid | frozen grid |
|---|---|---|
| TSLB +0.001 … +0.01 K | 1 (`grdflx` only) | 19 |
| TSLB +0.05 K | 8 | 21 |
| TSLB +0.1 K | 10 | 19 |
| TSLB +0.5 K | 12 | 21 |
| SMOIS +1e-5 … +2e-2 | 16–17 (flat) | 20–22 (flat) |

On a warm column the top soil level reaches the surface energy balance through
terms that absorb a 3e-5 relative change in one 12 s float32 step. So the
standing gate leads with **SMOIS**, whose count is flat over three decades of
nudge size and therefore not sitting on a threshold, and follows with TSLB at
its measured floor — and then asserts the floor *is* a floor, by requiring
that 0.005 K still reaches `grdflx` and nothing else. If RUC's soil coupling
ever becomes sensitive enough that it propagates, that assertion fails and the
table above gets corrected instead of quietly going stale.

## Still open

**There is no RUC soil ingest.** The nine-level TSLB/SMOIS a RUC run starts
from must be supplied by the caller; `ruc_cold_start` then derives SH2O,
SMFR3D, MAVAIL and ZNT from them through `ruclsminit`.
`gpuwm/ingest/soil_contract.py` remaps exactly one soil geometry — Noah's four
layers — and refuses RUC by name. That refusal is correct and this lane did
not remove it: RUC's levels are a different discretization with a different
value-location convention (`module_soil_pre.F:init_soil_depth_3` tabulates
LEVEL depths including a zero-depth surface level, not layer centres), so
producing a RUC target would mean inventing a remap with no oracle behind it,
and `flag_sm_adj` — the real.exe knob that would adjust a Noah-derived RUC
soil state — is refused for the same reason. **This is the restriction that
stops a RUC forecast being launched from a GRIB source today, and no amount of
runtime work removes it.** Closing it needs WRF-real nine-level initial
conditions to verify a remap against.

**No WRF trajectory comparison exists.** Every routine is bitwise against its
WRF oracle and the assembled driver is bitwise against `oracle/lsmruc.csv`
except its 26 pinned upstream-residue cells, but no gpuwm/WRF *forecast*
comparison exists, which is what `validation-candidate` would require. Hence
`implemented-unverified`.

**The six-layer geometry is refused.** `init_soil_depth_3` also tabulates a
six-level RUC grid, but every RUC oracle fixture in the tree is nine-level and
the CUDA leaves index a `__constant__ real ruc_soil_layer_depth[9]`. Refused
as a port blocker, not as a schema error: it is a coherent WRF request gpuwm
has no fixture for.

## Can a user select RUC yet?  No — and now for two reasons, not one

Asked and answered on 2026-07-26, after commit `46ce211` landed the soil
remap.  **RUC still gets no registry template.**  The hold that commit
declared is still true, and measuring the scheme at width found a second
blocker that is independent of it and that no amount of ingest work removes.

The measurements below are host-side and needed no GPU, because the thing
being measured is a Python loop.

### 1.  The hold `46ce211` declared, re-checked

That commit's stated reason was not the remap and was never the arithmetic:

> Every initializer -- era5_direct, gfs_direct, nest_init, hrrr_physics --
> calls preprocess_noah_soil and receives a NoahSoilState whose four-layer
> shape, Noah SH2O partition and sea-ice column are Noah's.  A RUC run needs
> its own state object and its own handoff into ruc_cold_start.

Checked against the import graph rather than against the sentence:
**`gpuwm/ingest/ruc_soil.py` has zero importers anywhere under `gpuwm/`.**  It
is imported by `tests/test_ruc_soil_ingest.py` and by the two scripts in
`tools/ruc_soil_ingest_wrf461_oracle/`, and by nothing else; it is not even
re-exported from `gpuwm/ingest/__init__.py`, whose lazy-import table lists
`NoahSoilState` and `preprocess_noah_soil` and no RUC name.  The three
occurrences of the string `ruc_soil` inside `gpuwm/core/ruc_runtime.py` are
all inside error messages telling a caller where the remap lives.

All seven initializers still take Noah's object:
`gpuwm/era5_direct.py:399`, `gpuwm/gfs_direct.py:683`,
`gpuwm/hrrr_hierarchy_direct.py:506`, `gpuwm/mapped_direct.py:442`,
`gpuwm/runtime.py:487`, `gpuwm/ingest/nest_init.py:848` (which also declares
`soil: NoahSoilState | None` on its own result at `:93`) and
`gpuwm/ingest/hrrr_physics.py:201`.  `ruc_cold_start` has exactly one caller,
`gpuwm/core/physics.py:2016`, and it reads the nine-level TSLB/SMOIS out of
`driver.fields` -- i.e. out of whatever the caller already put there.

**This supersedes the "Still open / There is no RUC soil ingest" paragraph
earlier in this document**, which says producing a RUC target "would mean
inventing a remap with no oracle behind it" and that closing it "needs
WRF-real nine-level initial conditions to verify a remap against".  Both
sentences were true when written and are not true now:
`gpuwm/ingest/ruc_soil.py` is `init_soil_depth_3` + `init_soil_3_real` at
max_ulp 0 against `gpuwm/data/ruc/oracle/soil_ingest.csv`, and `46ce211` drove
it over all 861,001 columns of the four-domain case.  What replaced the
missing remap as the blocker is the missing *wiring*, which is a smaller and
much more checkable thing.

### 2.  The cost per land column is FLAT, measured over a 512x range

`gpuwm/core/ruc.py:ruc_land_surface_step` reshapes to `ncolumn` and then runs
`for i in range(ncolumn)` over `np.float32` scalars.  So the prediction is
that per-column cost does not move with the count, and the consequence is
that a nest costs exactly its width.  Measured directly on that function --
no device, best of 2-3 samples per point, one identical land column
replicated:

| land columns | 48 | 96 | 192 | 384 | 768 | 1536 | 3072 | 6144 | 12288 | 24576 |
|---|---|---|---|---|---|---|---|---|---|---|
| warm grassland, ms/column | 1.687 | 1.685 | 1.781 | 1.714 | 1.718 | 1.671 | 1.688 | 1.723 | 1.721 | 1.724 |
| snow-covered, ms/column | 3.230 | 3.286 | 3.224 | 3.260 | 3.197 | 3.298 | 3.292 | 3.327 | 3.257 | 3.244 |

Mean 1.711 ms/column warm and 3.261 snow; **spread 1.066x and 1.041x across a
512-fold change in the column count.**  There is no amortization left at all
-- not the flattening curve Noah-MP's lane measured (17.47 ms/column at 2
columns down to 7.24 at 352), because that curve is a fixed per-call cost
being diluted and this sweep starts past it.

Two things the sweep establishes that the existing single-grid figure could
not.  The cost is **configuration-dependent**: a snow-covered column is 1.9x
a warm one, so a single per-column number is only meaningful with the column
state attached.  And it is *flat in both configurations*, so the flatness is
a property of the loop rather than of the physical branch.  The registry's
published 4.1 ms/column -- measured end to end on a 168-land-column grid with
a 15 mm pack, including both host transfer sets and the write-back -- sits
above the snow host figure by about what the transfers and
`SFCDIAGS_RUCLSM` cost, which is the expected relationship and not a
disagreement.

### What 360,000 columns costs

`configs/real74_4dom.toml` is d01 250x200, d02 500x400, d03 501x501, d04
600x600 = 861,001 columns, `bldt = 0.0` (a land-surface call every step), and
a 43,200 s forecast.  The child timestep chain is 60 / 4 / 3 / 3, so d04 runs
at dt = 1.667 s for 25,920 steps.  Taking every column as land, which is the
worst case and the only one a fail-closed rail may assume:

| | host, warm | host, snow | registry's end-to-end 4.09 |
|---|---|---|---|
| one d04 land-surface call (360,000 columns) | 616 s (10.3 min) | 1,174 s (19.6 min) | 1,472 s (24.5 min) |
| d04 alone, 12 h of forecast | 184.8 days | 352.2 days | 441.7 days |
| all four domains, 12 h of forecast | 239.9 days | 457.2 days | 573.4 days |

That is the land surface only; it excludes the dycore, the microphysics and
the radiation entirely.  **RUC is not usable at production width whatever the
registry says**, and this is the same trap that made Noah-MP expert-only, at
the same order of magnitude (Noah-MP's flat 7.18 ms/column projects d04 to
43 minutes per call).

### A gap this lane found and did not close

`gpuwm/config.py:validate_run_config` already hands the readiness authority
`columns=nx*ny`, and `gpuwm/physics_compat.py:378-403` refuses **Noah-MP**
above its measured 352-column ceiling unless the caller states the budget
they accept.  There is no such arm for `sf_surface_physics == 3`.  Measured:
`pending_wrf_physics_components(sf_surface_physics=3, num_soil_layers=9,
columns=360000, ...)` returns an empty tuple, while the same call at
`sf_surface_physics=4` returns the Noah-MP column-budget blocker.

It is a gap behind a closed door rather than a live fail-open -- no template
and no route override reaches RUC, so the only way to make that call is the
in-process API -- and it is recorded here and gated by
`tests/test_ruc_admission.py::test_the_width_rail_covers_noahmp_and_not_ruc`
rather than fixed, because it is one of the items a template must close and
closing it before the template would be a rail with nothing behind it.

*Update 2026-07-27:* Noah-MP's slab orchestration moved
`NOAHMP_MEASURED_COLUMN_CEILING` to its measured 360,000 columns
(0.202--0.227 s per d04-width call), so the control in that gate now probes
one column past the ceiling instead of production width -- the authority
still fires on width, for widths nothing has measured.

### The checkable list: what a RUC template needs first

Each item is a file and a condition, so it can be checked rather than
discussed.

1. **A RUC soil state object and its handoff.**  `remap_soil_to_ruc_levels`
   returns `RucSoilColumns` (nine LEVEL depths including a zero-depth surface
   level, plus ZS and DZS).  `NoahSoilState` is four layer MIDPOINTS with
   Noah's SH2O partition and Noah's sea-ice column.  They are not the same
   object at a different length.  Something must produce the RUC object from
   a source and hand it to `ruc_cold_start`.
2. **Seven initializers**, listed above, each of which today calls
   `preprocess_noah_soil` unconditionally with no land-surface selector in
   sight.  A RUC run reaching any of them gets four Noah layers.
3. **`gpuwm/ingest/soil_contract.py`.**  `NOAH_LAYER_BOUNDS_M` is one
   hardcoded target and `validate_soil_layer_contract` refuses any other, by
   design and correctly; the mapped-source path needs RUC's target added
   explicitly with its own remap policy.  (This is the condition
   `tests/test_registry_reachability.py::test_the_ruc_blocker_is_still_true`
   re-runs.)
4. **`gpuwm/physics_compat.py`.**  All five entries of
   `_SINGLE_DOMAIN_RUNTIME_SWITCHES` pin `sf_surface_physics: 2` and
   `num_soil_layers: 4`; there is no RUC profile row.  And there is no RUC
   arm in the column-budget rail -- see above.
5. **`gpuwm/namelist_compat.py:764-768`** reports `UNSUPPORTED_PHYSICS_STATE`
   for anything but `sf_surface_physics=2` with `num_soil_layers=4`, so a
   stock namelist selecting RUC is refused at import.
6. **The width answer itself.**  Items 1-5 are wiring and are finite.  The
   flat host cost is not wiring: a RUC template that satisfied all five would
   still be a template for a scheme that cannot finish a nest.  Either a
   device `sfctmp`/`lsmruc` exists (`gpuwm/core/ruc_gpu.py` stops at
   `ruc_snow_soil_step_cuda`; there is nothing to launch), or the template is
   expert-only with a stated column budget the way Noah-MP's is.

Item 6 is the one that decides what a RUC template could ever look like, and
it is why this lane did not add one behind a promise to fix the ingest.

### The gates, and each one's failing form

`tests/test_ruc_admission.py` is host-only: it imports no cupy, so
`tests/conftest.py` does not auto-mark it `gpu` and it runs on a machine with
no card.  Five gates, each shown failing against the exact event it watches
for before being trusted:

| gate | shown to fail when |
|---|---|
| `test_the_ruc_soil_ingest_is_reachable_from_no_runtime_path` | pointed at a tree whose initializers DO import `gpuwm.ingest.ruc_soil` |
| `test_every_initializer_still_hands_the_land_surface_a_noah_soil_state` | one initializer names `remap_soil_to_ruc_levels` |
| `test_no_template_selects_ruc_and_no_route_overrides_the_land_surface` | one template's `land_surface` is edited to `ruc-lsm`; and separately when a route gains `land_surface` in `allowed_component_overrides` |
| `test_the_host_column_cost_is_flat_in_the_column_count` | its own statistic is fed constant work instead of per-column work |
| `test_the_width_rail_covers_noahmp_and_not_ruc` | a RUC column-budget blocker is stood up in the readiness authority |

Each also carries an in-test control, so a passing run does not rest on those
falsification runs having happened once: the import scan must find
`gpuwm.ingest.soil`'s seven importers, the template scan must find Noah's
templates, the flatness statistic must reject constant work, and the width
rail must still refuse Noah-MP at 360,000 columns.

### What changed in the registry

Nothing that makes anything selectable, and no maturity grade moved.  Two of
`land_surface.ruc-lsm`'s warnings said things that stopped being true when
`46ce211` landed:

* the cost warning quoted one grid and called 4.1 ms/column "measured end to
  end", which was true and gave a reader no way to know the cost is flat.  It
  now carries the sweep above, both column configurations, and the d04
  projection.
* the ingest warning opened "There is NO RUC SOIL INGEST" and said producing
  a RUC target "would mean inventing a remap with no oracle behind it".  A
  reader would have concluded that no remap and no oracle exist.  Both do.
  It now says the remap is finished and oracle-matched, that the blocker is
  that nothing imports it, and names the seven initializers.

### Addendum: the same sweep at the driver seam, on the card

The measurements above are the host loop alone.  Re-run end to end on the
user's RTX 5090 by wrapping `gpuwm.core.physics`'s own `ruc_lsm_step` seam and
stepping the real driver, so the figure includes both host transfer sets, the
`CQS`/`CHS` rebuild, `SFCDIAGS_RUCLSM` and the write-back — best of the two
steady-state calls of a three-step run, first call dropped because `ktau == 1`
runs LSMRUC's own initialisation block:

| land columns | 48 | 96 | 192 | 384 | 768 | 1536 |
|---|---|---|---|---|---|---|
| warm grassland, ms/land column | 1.866 | 1.779 | 1.766 | 1.728 | 1.713 | 1.702 |
| snow-covered + frozen soil, ms/land column | 3.433 | 3.355 | 3.335 | 3.315 | 3.300 | 3.270 |

Spread 1.096x and 1.050x, still falling slightly, still with no sign of
levelling into anything cheaper.  **The device work is not measurable next to
the loop**: 1.702 ms/land column end to end against 1.711 host-only warm, and
3.270 against 3.261 snow.  There is no transfer cost to optimise away and no
kernel to make faster, because there is no kernel — this is the same
conclusion Noah-MP's lane reached by a different route (992 ms of solver
against 4.3 ms of transfers on a 160-column call).  CuPy pool peak was
29.01 MiB at the widest point and total board occupancy stayed at 3.2 GiB,
which is the user's desktop; nothing here approaches the 29,500 MiB rail.

**This also corrects the 4.1 ms/column figure this document and the registry
have been publishing.**  That number came from dividing a whole 694 ms RK3
step by its 168 land columns, so it charged the dycore, the microphysics and
the 24-column water arm to the land surface.  Timed at the seam, the same
snow-covered configuration is 3.3-3.4 ms per land column.  The "registry's
end-to-end 4.09" column of the projection table above is therefore an
over-attribution and should be read as the snow row; the conclusion does not
move, because the honest number is still flat:

| | one d04 call (360,000 columns) | d04 alone, 12 h |
|---|---|---|
| warm, seam-timed 1.702 ms/column | 613 s (10.2 min) | 183.8 days |
| snow-covered, seam-timed 3.270 ms/column | 1,177 s (19.6 min) | 353.2 days |

Reported rather than quietly improved: the corrected figure is *smaller* than
the one that was published, and it changes nothing about whether RUC can run a
nest.

Reference hardware for this addendum: RTX 5090, driver API 13030, CUDA runtime
12090, CuPy 14.0.1, numpy 2.2.6, Windows.

---

# Both blockers, closed and measured (2026-07-26, later)

*The document above is the record as it stood this morning.  Everything below
is a separate pass, and it changes two of that record's conclusions rather
than appending to them.*

## Lead

**The soil ingest now reaches a RUC forecast.**  All seven initializers route
through one seam; a `sf_surface_physics=3` request initializes on RUC's nine
LEVEL depths, and Noah and Noah-MP are bitwise unchanged — structurally, not
merely by measurement.

**The column loop is no longer a Python `for i in range(ncolumn)`.**  It was
never a loop over physics: `sfctmp` and every leaf under it are *already*
written over a column axis, so the loop paid `sfctmp`'s fixed per-call cost
once per column.  Batching it, bitwise:

| ms per land column | 48 | 24,576 | spread | vs. before |
|---|---|---|---|---|
| warm grassland, before | 1.687 | 1.724 | 1.07x | — |
| **warm grassland, after** | **0.539** | **0.479** | **1.13x** | **3.6x faster** |
| snow-covered, before | 3.230 | 3.244 | 1.04x | — |
| **snow-covered, after** | **1.322** | **1.254** | **1.06x** | **2.6x faster** |

Host, over the same 512x range as the published sweep.  The full sweep:

| land columns | 48 | 96 | 192 | 384 | 768 | 1536 | 3072 | 6144 | 12288 | 24576 |
|---|---|---|---|---|---|---|---|---|---|---|
| warm grassland | 0.539 | 0.512 | 0.503 | 0.492 | 0.487 | 0.482 | 0.481 | 0.478 | 0.484 | 0.479 |
| snow-covered | 1.322 | 1.309 | 1.276 | 1.268 | 1.270 | 1.259 | 1.257 | 1.258 | 1.256 | 1.254 |

**Is it still flat?  Yes** — and that is the finding rather than a
disappointment.  A 1.13x spread over a 512-fold change in the column count
says the residue is genuine per-element work, not per-call overhead.  It is
also *named*: `cProfile` over the batched call attributes 45% to
`ruc_soil_properties`, 16% to `ruc_soil_moisture_step`, 11% to
`_ruc_phase_partition`, 11% to `ruc_soil_temperature_step` and 5% to
`_ruc_vilka` — 97% of the call inside `ruc_soil_step`, which has had an
oracle-matched CUDA leaf for weeks with nothing able to launch it at width.

**Wall seconds per simulated minute, d04 of the four-domain case** — 360,000
columns, `bldt = 0` so the land surface runs every step, dt = 1.667 s so 36
land-surface calls per simulated minute.  Land surface only; this excludes
the dycore, the microphysics and the radiation entirely.

| | one call | **wall seconds per simulated minute** | i.e. |
|---|---|---|---|
| warm, before (1.711 ms/col) | 616 s | 22,180 s | **6.2 h per simulated minute** |
| **warm, after (0.479)** | **172 s** | **6,207 s** | **1.7 h per simulated minute** |
| snow, before (3.261) | 1,174 s | 42,270 s | **11.7 h per simulated minute** |
| **snow, after (1.254)** | **451 s** | **16,251 s** | **4.5 h per simulated minute** |

**RUC is still not usable at production width.**  1.7 hours of wall clock per
simulated minute, from one domain's land surface alone, is not a forecast.
The conversion moved the number by 3.6x and did not move the conclusion, and
**no registry template was added and no maturity grade moved.**

**Measured ULP: 0** on the host, everywhere it was checked; and **0 on the
card for snow-free columns**, where the device leaves also make the cost stop
being flat -- 0.022 ms/land column at 24,576 against 0.503 on the host, a
**22.9x** speedup and a **4.8 minute** wall-clock cost per simulated minute
for d04.  **Snow-covered columns are NOT bitwise on the device** (up to 10
ULP of `infiltr`) and no device leaf set is admissible for them; that
divergence, and the fact that only driver-level composition could surface it,
is the last section of this document.

## Blocker 1: the ingest is wired

### What was actually wrong

`gpuwm/ingest/ruc_soil.py` was `init_soil_depth_3` + `init_soil_3_real` at
max_ulp 0 against its WRF oracle and had **zero importers anywhere under
`gpuwm/`**.  All seven initializers called `preprocess_noah_soil`
unconditionally.  The failure mode was not a crash — it was a complete,
plausible forecast on a soil discretization RUC does not use.

### The seam

`gpuwm.ingest.ruc_soil.preprocess_land_surface_soil` is the one function all
seven now call, with `sf_surface_physics` passed explicitly.  It is
**fail-closed on the selector**: an unrecognised scheme raises rather than
falling through to Noah, because falling through is precisely the defect
being removed.  The shape check downstream (`_as_soil`) would have caught a
four-layer array handed to a nine-layer allocation, but only as an
unexplained broadcast error, and only for the schemes whose counts differ.

`preprocess_ruc_soil` **calls** `preprocess_noah_soil` for the surface half —
the land/water/sea-ice partition, the lake override, the SST and TSK repair,
the 271.4 K sea-ice TMN, the SNOW/SNOWH reconciliation, all of which WRF runs
identically for every `sf_surface_physics` — and then discards its four-layer
soil column and rebuilds it from the source profiles through
`remap_soil_to_ruc_levels`.  **`gpuwm/ingest/soil.py` is not edited at all.**
That makes "Noah and Noah-MP are bitwise unchanged" a structural property
first and a measurement second; it is measured too, byte for byte over every
`NoahSoilState` array, on three source modes and both schemes.

`RucSoilState` spells `NoahSoilState`'s ten fields identically, so every
consumer downstream of the initializers — `initialize_physics`,
`canonical_noah_surface`, `initialize_landuse` — reads it unchanged.  What
differs is the leading axis, and two extra fields carrying ZS and DZS.
`liquid_moisture` is a copy of `soil_moisture` and is deliberately *not* a
partition: WRF derives a RUC run's SH2O in `ruclsminit`, not in `real.exe`,
and `ruc_cold_start` overwrites every value.  Running Noah's `sh2o_init`
freezing curve would put a Noah table lookup on Noah's geometry into a RUC
field that is about to be recomputed.

Three wired source modes and one refused:

* **ERA5 layers** and **GFS layers** take WRF's `flag_soil_layers` arm.  The
  sample depths are the INTEGER centimetre midpoints WRF's own reader forms
  (`char2int2`, `share/module_optional_input.F:1949-1954`), parsed off the
  field names rather than tabulated — a 0-7 cm layer is sampled at 3 cm, not
  3.5 and not 7.
* **HRRR nodes** take the `flag_soil_levels` arm.
  `gpuwm.ingest.soil.HRRR_SOIL_NODE_DEPTHS_M` **is** `RUC_LEVEL_DEPTHS_M[9]`
  in metres, bit for bit, so this is the one source with an exactly known
  answer — see the finding below.
* **The declarative mapped contract is REFUSED for RUC by name.**
  `validate_soil_layer_contract` compares `target_layers` against
  `NOAH_LAYER_BOUNDS_M` and refuses any other, by design.  Admitting RUC
  there means proving a remap against a WRF-real nine-level initialization;
  relabelling a Noah-bounds remap as nine levels would fabricate a soil
  column.  This is the one item of the six-item list that is **not** closed.

### Three divergences from stock `real.exe`, each recorded in the docstring

| | what gpuwm does | why |
|---|---|---|
| `flag_sm_adj` | not applied | `RUC_OPTION_IDENTITY_EVIDENCE` admits only 0; the two must not drift apart |
| open-water fill | WRF's **no-SST** arm, against an already-repaired TSK | `:2131-2144` writes SST into TSK unchecked and stock `real.exe` then aborts at `module_initialize_real.F:3283` on any SST gap — 10.4% of the water cells of a production coarse-domain frame.  `preprocess_noah_soil` has already resolved TSK to the valid SST where one exists, so the result equals WRF's wherever WRF would not have aborted |
| sea ice | left to the runtime | `LSMRUC:862-895` resets SOILMOIS, SMFR3D, SH2O, KEEPFR3DFLAG and `min(271.4, TSO)` on **every** step, not once, so an ice column written here would be overwritten before it was read.  TMN is carried, because RUC reads it as the bottom boundary at `:1057` |

### The gate, shown failing first

`test_every_initializer_routes_through_the_land_surface_seam` failed on
`gpuwm/era5_direct.py` before the wiring landed.  Separately, the two
`tests/test_ruc_admission.py` gates that asserted the **absence** of wiring
both fired when it landed — which is the notification mechanism they were
written to be.  They are inverted here rather than deleted.

The distinguishing check is not a shape assertion.  A `RucSoilState` built
out of Noah's four-layer column padded to nine slots satisfies every shape
assertion; what it cannot satisfy is that RUC's shallowest level is the
ground surface itself, so on a layer source its value is the anchoring skin
temperature rather than the 5 cm layer mean Noah's shallowest midpoint
carries.  That check has its own falsification, asserting the two are not
accidentally equal on the fixture.

### A finding: WRF's own interpolation is not exact at a coinciding node

HRRR's native nodes are RUC's levels bit for bit, so that source should be
the identity.  It is not.  `_bracket` (`:1958-1968`) takes the FIRST interval
containing an interior target, so a target coinciding with node *k* is taken
from the interval **above** it and `init_soil_3_real:1969-1972` collapses to
`(x*d)/d` — two float32 roundings, not a copy.  **Measured: 20 of 90 land
values move, all by exactly one ULP.**  The gate is the WRF expression
re-derived in the test, bitwise, rather than the identity a reader would
expect: reproducing the identity would be a silent correction to WRF applied
at the ingest.

### Also fixed: a case-token leak in this lane's own file

`gpuwm/ingest/ruc_soil.py:358` carried a metgrid filename including the case
date in a comment **at HEAD**, which failed
`tests/test_runtime.py::test_generic_ingest_and_runtime_have_no_met_em_reader`
before this lane started.  Removed; the measurement it cites is kept without
naming the file.

## Blocker 2: the column loop

### What was actually wrong

`ruc_land_surface_step` was transcribed the way WRF is written — `DO j / DO i`
around a scalar `SFCTMP` call.  But `ruc_surface_temperature_step` and every
leaf under it are already mask-based over a column axis.  A `cProfile` of a
48-column call is **188,322 Python calls** with essentially no arithmetic in
the leaves: the loop was calling a vectorized routine 48 times with
one-element arrays.

So this was not a case for bolting a kernel onto the side.  It is the same
structural finding the Noah-MP lane reported from the other direction — its
seam *rewound and replayed* each column's prefix, and that is why moving 63%
of the work bought only 1.25x.  Here the seam did not need a continuation at
all; it needed the loop deleted.

### What moved

The prologue (`:612-935`), the water and sea-ice arms (`:824-895`) and the
epilogue (`:1024-1116`) evaluate over the whole column axis, and `SFCTMP` is
called **once** for all land columns.  Every expression keeps WRF's float32
grouping.

**`ilnb_chain=True` is the exception and cannot be batched.**  WRF never
initialises `ilnb`, so column *i*'s seed is column *i-1*'s answer — a genuine
sequential dependence.  That path dispatches the **same call** one column at
a time and rejoins the pieces, so there is one implementation of the physics
and two batchings of it, not two implementations.  `oracle/lsmruc.csv` still
reproduces bitwise.

**`conflx` had to be lifted first**, and this is the subtle part.
`LSMRUC` passes `0.5*dz8w(i,1,j)` — half the lowest model layer, a
**per-column** quantity on a terrain-following coordinate — into a *scalar*
`sfctmp` argument, purely because `sfctmp` is called from inside `DO j / DO i`.
Batching the columns without lifting it would have silently applied one
column's layer depth to the whole field.  It is now a column field through
`soiltemp`, `sice`, `soil`, `snowseaice`, `snowtemp`, `snowsoil` and
`sfctmp`, masked with the columns at all six masked leaf calls, and it was
verified bitwise-neutral **on its own**, before the loop was touched.

### The answer does not change

* Four independent six-step trajectories — warm, snow-covered, and both again
  with water and sea-ice columns present — are bitwise against digests
  **re-derived from this tree before the conversion**, not copied from a
  report.  Each feeds the driver's own output back in, so the accumulators
  and the carried soil state are compared, not just one call's fluxes.
* The whole host RUC suite passes: **115 tests**, including
  `oracle/lsmruc.csv`, `oracle/sfctmp.csv` and `oracle/lsmruc_stackfill.csv`.
* `tests/test_ruc_column_batching.py` adds three claims that are
  configuration-independent and need no reference implementation:
  **partition invariance** (one call over N columns equals any disjoint split
  of the same columns), **order independence**, and **chained-versus-batched
  agreement** over six steps on a grid where `ilnb` is never read.

Three divisions that WRF only ever reaches on one arm are now guarded rather
than evaluated-and-discarded, so no NaN is produced: `:657`'s snow fraction,
`:748`'s snow density, and `:915`'s `mavail` — the last of which a water
vegetation category makes 0/0.  The suite passes under
`-W error::RuntimeWarning`.

### The falsification failed to fail, and what replaced it

One ULP of the entry `soilt` reaches **nothing**: VILKA's solve converges the
seed away, the same absorption the Noah-MP lane measured for `BARE_FLUX`'s
`TGB`.  Swept per input on a warm 12-column grid, the floor at which a
perturbation first reaches any returned field:

| input | ULP | fields reached |
|---|---|---|
| `soilmois` | 1 | 8, led by `qvg`/`qsfc`/`sh2o` |
| `t3d` | 1 | 1 — `hfx` alone |
| `soilt` | 2 | 10 |
| `z3d` | 2 | 6 |
| `tso` | 64 | 2 — `grdflx` and `tso` |
| `gsw` | — | absorbed below about 1e-3 W m-2 |

The gate leads with `soilmois` and adds **`z3d`**, because `z3d` is the input
whose *shape* this conversion changed — a batch that took one column's layer
depth for the whole field would be invisible to every other check.  And it
asserts that `soilt` at one ULP really is absorbed, so the `2` cannot be
quietly tidied back to a `1` and the table cannot go stale silently.

## The device leaves — measured on the card, and admissible for SNOW-FREE
## columns only

**Status first.** Snow-free columns are **bitwise, max ULP 0**, and cost
**0.022 ms per land column at 24,576 columns against 0.503 on the host — 22.9x.**
Snow-covered columns are **NOT bitwise** and the device set is **not
admissible** for them; see the divergence below. Reference hardware: the
user's RTX 5090, CuPy, Windows; whole-machine VRAM never exceeded ~3.9 GiB,
which is the desktop, and the CuPy pool peaked at 55.71 MiB — nothing near
the 29,500 MiB rail. No process was killed, stopped or suspended, and the
shared lock was taken and released for every run.

Batching is what made the CUDA leaves *reachable*: before it there was
nothing to launch, because each call carried one column.
`ruc_surface_temperature_step` now takes a `leaves` mapping, and
`gpuwm.core.ruc_gpu.RUC_SFCTMP_DEVICE_LEAVES` supplies the device set for all
four arithmetic leaves (`soil`, `sea_ice`, `snow_soil`, `snow_sea_ice`).

The injection is a mapping rather than an import because `gpuwm/core/ruc.py`
**must not import cupy**: `tests/conftest.py` auto-marks any module that does
as `gpu`, and the whole RUC oracle suite runs on a machine with no card.

`conflx` is now a `const real* __restrict__` in the four kernels that read it
(`ruc_soil_temperature_step`, `ruc_sea_ice_step`, `ruc_snow_sea_ice_step`,
`ruc_snow_temperature_step`), for the same reason it was lifted on the host.

**The transfer boundary is per batch, not per column** — one upload and one
download per leaf per land-surface call.  `sfctmp`'s own recombination
arithmetic (`:1979-2115`) stays on the host; making the whole column
device-resident is the next conversion, not this one.

### What it costs, measured on the card

`RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE` (`soil` + `sea_ice`) against the same
tree's host path, paired in one process, best of 2-3:

| land columns | 48 | 96 | 192 | 384 | 768 | 1536 | 3072 | 6144 | 12288 | 24576 |
|---|---|---|---|---|---|---|---|---|---|---|
| host, ms/column | 0.542 | 0.531 | 0.517 | 0.514 | 0.505 | 0.500 | 0.502 | 0.499 | 0.503 | 0.503 |
| **device, ms/column** | 0.379 | 0.203 | 0.109 | 0.070 | 0.044 | 0.033 | 0.029 | 0.024 | 0.023 | **0.022** |
| speedup | 1.4x | 2.6x | 4.8x | 7.3x | 11.5x | 15.3x | 17.5x | 20.6x | 22.0x | **22.9x** |

**Is it still flat? No — and that is the answer to the question the whole
lane exists for.** The host spread is 1.09x over the 512x range; the device
spread is **17.3x**, falling monotonically. A cost that keeps falling as
columns are added is what work actually reaching the device looks like. The
flat term is gone.

**Wall seconds per simulated minute, d04, snow-free**, at 0.022 ms/column:

| | one 360,000-column call | wall seconds per simulated minute |
|---|---|---|
| host loop, before this lane | 616 s | 22,180 s (6.2 h) |
| host, batched | 181 s | 6,516 s (1.8 h) |
| **device, snow-free** | **8 s** | **288 s (4.8 min)** |

That is a **77x** improvement on the original number for one domain's land
surface. It is still not a free scheme — 4.8 minutes of wall clock per
simulated minute from the land surface alone leaves no budget for the
dycore, the microphysics and the radiation — but it is the first RUC figure
in this document that is the right order of magnitude to argue about.

**None of it applies to a snow-covered forecast yet.**

### The divergence this found, which no per-routine gate could

Composing the leaves at the driver level over a perturbed snow grid shows
that the CUDA and host halves **disagree**, in exactly four fields:

| field | max ULP |
|---|---|
| `infiltr` | 10 |
| `acrunoff` | 4 |
| `sfcrunoff` | 4 |
| `runoff1` | 2 |

Nothing else in the 43-field result moves. Every RUC CUDA leaf is max_ulp 0
against its own WRF fixture — the whole of `tests/test_ruc_gpu.py` passes —
so this is the **"a mirror is not an oracle"** trap again, and the same shape
as the sibling RUC snow lane's 3 ULP of `thdif` amplifying to 53,248 ULP of
an energy residual while both halves matched their fixtures.

Parametrising the gate over **both** device sets is what located it: the
**snow-free subset diverges too**, so the leaf at fault is not `snowsoil`. It
is `soil` running on the mosaic snow-free fraction (`:1770-1822`) of a
snow-covered column — i.e. `ruc_soil_moisture_step_cuda`'s infiltration arm,
which a warm column with no melt water never reaches. All four fields come
out of `SOILMOIST`: `infiltr` is its infiltration capacity, `runoff1` its
surface runoff, and `sfcrunoff`/`acrunoff` are the driver's accumulators of
`runoff1` at `:1038-1044`.

It is recorded as a **measurement, not tolerated as a tolerance**. The gate
is two-sided: it fails if the divergence grows, if it reaches a fifth field,
**and if it is ever fixed** — so the record cannot go quietly stale. **Until
it is explained, no device leaf set is admissible for snow-covered columns**
and `RUC_SFCTMP_DEVICE_LEAVES_SNOW_FREE` is the only set with evidence
behind it.

### One regression this lane caused, and fixed

Lifting `conflx` to a pointer broke
`tests/test_ruc_gpu.py::test_ruc_snow_temperature_cuda_has_no_unpinned_contraction`,
which builds the kernel argument list **by hand** to compile a `--fmad=false`
control. A scalar where a pointer is expected is a garbage pointer, and
`cudaErrorIllegalAddress` poisons the CUDA context for every later test in
the process — so it presented as six failures with one cause. Fixed, and the
same lift needed a second binding inside `ruc_snow_soil_step_cuda`, which
launches `ruc_snow_temperature_step` directly rather than going through its
wrapper. `tests/test_ruc_gpu.py` and `tests/test_ruc_runtime.py` are green
again: **74 passed** across the three device files.

## The checkable list, re-stated against what is now true

The six-item list earlier in this document said what a RUC template needs
first.  Four of the six moved.  **This is not a recommendation to add one.**

| | item | then | now |
|---|---|---|---|
| 1 | a RUC soil state object and its handoff | missing | **CLOSED.**  `RucSoilState` + `preprocess_ruc_soil`, spelling `NoahSoilState`'s ten fields so every downstream consumer reads it unchanged |
| 2 | seven initializers calling `preprocess_noah_soil` with no selector in sight | all seven | **CLOSED.**  All seven call `preprocess_land_surface_soil` with `sf_surface_physics`, and it is fail-closed on the selector.  Gated per file by AST, not by text search |
| 3 | `gpuwm/ingest/soil_contract.py` has one hardcoded target | one target | **STILL OPEN, and deliberately.**  The mapped declarative path refuses RUC by name.  ERA5, GFS and HRRR sources are wired; the mapped-contract source is not, because admitting it needs a WRF-real nine-level initialization to verify against |
| 4 | no RUC profile row in `_SINGLE_DOMAIN_RUNTIME_SWITCHES`; no RUC arm in the column-budget rail | both missing | **UNCHANGED, on purpose.**  `gpuwm/physics_compat.py` was not touched.  A width rail is the lead's decision and the number it would carry is in this document |
| 5 | `gpuwm/namelist_compat.py:764-768` refuses anything but Noah at four layers | refuses RUC | **UNCHANGED.**  Not touched |
| 6 | **the width answer itself** | 1.711 / 3.261 ms per land column, flat | **NO LONGER FLAT, on snow-free columns.**  0.022 ms/land column on the card at 24,576 columns (22.9x over the batched host, 77x over the original loop), device spread 17.3x and still falling -- 4.8 min of wall clock per simulated minute for d04's land surface, against 6.2 h.  Snow-covered columns are NOT device-admissible: up to 10 ULP of `infiltr` |

Item 6 is still the one that decides what a RUC template could look like.
Its answer is no longer "the cost is flat, so a nest costs its width" -- on
snow-free columns the cost falls 17.3x across the sweep and a d04
land-surface call is 8 s rather than 616 s.  It is still no, for two reasons
that are now specific rather than structural: 4.8 minutes of wall clock per
simulated minute is the LAND SURFACE ALONE, before the dycore, the
microphysics and the radiation; and **snow-covered columns have a measured
device divergence**, which for a tornado case over a spring continental
domain is not an edge case.

**Whether RUC becomes selectable is not this lane's decision and nothing here
was written on the assumption that it will.**  No registry template, no
maturity grade, no column budget, no route override.

# The snow divergence, closed — and a second one it made visible
# (2026-07-26, later still)

## Verdict

**The snow divergence is closed and RUC is now device-admissible on
snow-covered columns.**  It was one token.  A snow-covered grid is bitwise
against the host across all 43 returned fields, with the **full** device leaf
set, at 48, 64, 512, 4,096 and 24,576 columns, measured twice in separate
processes.

**And closing it exposed a second divergence that was already live on the
snow-free path this document had called admissible.**  Every bitwise gate
this lane had ever run was 48 or 64 columns wide.  Widened to 512 and 4,096,
the snow-free comparison failed: `hfx` moved 1 ULP on 1 of 512 warm columns
and 2 ULP on 13 of 4,096.  That one is closed too, by the same substitution,
and the widened gate is now part of the record rather than a thing nobody
ran.

**Wall seconds per simulated minute at d04**, measured at d04's own width
(360,000 columns, `dt = 1.667 s`, one call per step), not extrapolated from a
narrower sweep:

| | before this session | now |
|---|---|---|
| snow-free | 288 s (4.8 min) | **16.8 s** |
| snow-covered | *not admissible* | **65.7 s** |

For scale, the original host loop was 22,180 s (6.2 h) per simulated minute.

## What the divergence was

`gpuwm/core/kernels/ruc.cu`, `ruc_soil_moisture_step`:

```c
real series = __fadd_rn(1.0f, __fdiv_rn(powf(acrt, 2.0f), 2.0f));
```

against the host's `gpuwm/core/ruc.py:1533`:

```python
series = f32(series + f32(f32(np.power(f64(acrt), f64(2.0))) / f32(2.0)))
```

`module_sf_ruclsm.F:5983` is `SUM = SUM + (ACRT ** (CVFRZ-JK)) / FLOAT(K)`,
and `:5826` declares **`real :: cvfrz`** — REAL, set to `3.` at `:5936`.  So
`CVFRZ-JK` is REAL and gfortran lowers the whole expression to glibc `powf`.
It does **not** expand to a multiplication, which is what it would do with
Noah-MP's `INTEGER PARAMETER :: CVFRZ`, and the two lanes are right to differ.

Neither float32 pow available here is glibc's.  This file settled that
question a long time ago — see `_RUC_PROVISIONAL_TRANSCENDENTALS` at the head
of `ruc.cu` — and the answer was: evaluate in float64 and round once, on
**both** halves, because that form is the closest of the four candidates
measured against glibc 2.39 (2.55 % of arguments 1 ULP away, against 9.86 %
for the CUDA device libm and 14.4-29.0 % for numpy's float32 loop depending
on release) *and* because it is the only one identical on the host and the
device.  This one site was left behind when that convention was applied.

For an exponent of 2 the float64 form is the exactly-rounded square: a
float32 squared is exact in float64, so the double rounding cannot bite.
Verified on the card — `__double2float_rn(pow((double)x, 2.0))` and
`__fmul_rn(x, x)` differ on **0 of 300,000** arguments, while
`powf(x, 2.0f)` differs from both on **16,493 of 300,000**, always by 1 ULP.

## Why no gate could see it — the fixture annihilates the branch

`gpuwm/data/ruc/oracle/soilmoist.csv` has four cases, and
`tools/ruc_wrf461_oracle/run_soilmoist.F90` sets `soilice = 0.0` before its
`select case`.  Only the fourth case overrides it.  So:

* three cases never satisfy `fdmax > 1.e-2` at `:5972`, and the ACRT
  reduction at `:5975-5985` does not execute at all; and
* the fourth executes it and then multiplies the answer by an `infmax1` of
  **exactly zero**, because its only surface water is a melt term of the
  wrong sign, so `px = max(0, -total*delt)` is zero.

The frozen arm is evaluated exactly once in the entire fixture and its result
is annihilated before it reaches an output.  **An oracle can be blind to what
matters**, and this is the cleanest instance of it this project has produced:
not a missing input class at the edge of a distribution, but a branch the
fixture enters and then multiplies by zero.

## The amplification, predicted on the CPU before the card was touched

`fcr = 1 - exp(-acrt)*SUM` at `:5985` is a cancellation.  On a snow grid
`acrt` lands in **1.026 ... 1.467**, where `exp(-acrt)*SUM` is about 0.89 and
`fcr` about 0.11, so the relative error is multiplied by roughly 30.

Nudging the **host's** `acrt**2` by one ULP and nothing else, then running
the whole driver over the same perturbed snow grid, reproduces:

```
{'acrunoff': 2, 'infiltr': 10, 'runoff1': 2, 'sfcrunoff': 2}
```

against the measured device divergence of
`{'infiltr': 10, 'acrunoff': 4, 'sfcrunoff': 4, 'runoff1': 2}`.  `infiltr`
and `runoff1` match exactly; the accumulators differ because the device's
error is not uniformly +1 ULP across columns.  That was the diagnosis, and it
was made without a GPU.

`soilmois` does **not** move, and that is a consistency check rather than a
puzzle: the top layer is saturated on these columns, so it is clamped to
`dqm` regardless of the flux, and the flux difference goes into `runoff`
through the `saturated_flux - flux` term instead.

## The falsification: one token, both directions

`tests/test_ruc_gpu.py::test_ruc_soil_moisture_cuda_agrees_off_fixture` is
the gate that would have caught it — 1,536 columns carrying frozen soil
**and** liquid water arriving at the surface, the pair the fixture cannot
supply, asserting `max_ulp 0` over all seven outputs and asserting that the
generator actually reaches the arm so it cannot pass for nothing.

Both variants were compiled in one process from the same source, differing in
that single call:

| `ruc_soil_moisture_step` built with | vs the host transcription |
|---|---|
| `powf(acrt, 2.0f)` | `infiltrp` 19, `infmax` 19, `soilmois` 1, `mavail` 1, `runoff` 1 |
| `ruc_powf_rn(acrt, 2.0f)` | **max_ulp 0** |

## Toolchain probes, for the record

The three usual suspects in this project were probed on the same card and
none of them was the cause here, though two are live and worth knowing:

* **`-ftz=true` is on.**  `1e-30f * 1e-30f` returns `+0.0` on the device.
  CuPy appends it and these kernels are written not to depend on subnormals.
* **FMA contraction is on.**  `(1+2^-23)^2 - 1 - 2^-22` returns `0.0`, so
  NVRTC contracted it.  Every arithmetic boundary in these kernels is pinned
  with `__fadd_rn`/`__fmul_rn`/`__fdiv_rn`, which is why that does not
  matter, and `tests/test_ruc_gpu.py` keeps a `--fmad=false` control.
* **ptxas tie-folding** is held off by the two `__constant__` tables already
  documented above.

## The second divergence: `hfx` at width

A 64-column gate cannot see a 1-in-16 argument-dependent difference.  Four
`**` sites still had **numpy's float32 power on the host and the CUDA device
libm's `powf` on the device** — two different non-glibc functions.  Measured
on this card over 300,000 arguments spanning each site's range, they disagree
by 1 ULP on:

| site | disagreement rate |
|---|---|
| `wetcan = (cst/sat) ** cn` | 6.3 % |
| `hfx` factor `(1/patm) ** rovcp` | 7.5 % |
| sea-ice `exner = (p0/patm) ** rovcp` | 2.4 % |

`ruc_snow_soil_canopy_setup` and `ruc_snow_soil_finalize` were already on the
float64-rounded-once form on both halves; `ruc_soil_canopy_setup`,
`ruc_soil_finalize`, `ruc_sea_ice_step` and `ruc_snow_sea_ice_step` were not.
The snow lane's convention had been applied to the snow kernels and never
back-ported to the snow-free ones — which is the same omission that left the
`acrt` square behind, and it is worth saying plainly that **one incomplete
convention migration produced both of this document's divergences.**

Observed before the lift, on the driver, device leaves:

```
 512 warm columns: hfx 1 ULP on  1 column
4096 warm columns: hfx 2 ULP on 13 columns
4096 snow columns: hfx 2 ULP on  9 columns
```

At d04's 360,000 columns that is on the order of a thousand columns per call
carrying a wrong surface sensible heat flux into the PBL — and it would have
been admitted on the strength of a 64-column gate.

## Parity now, at width

Paired in-process against the same tree, host leaves against device leaves,
every one of the 43 returned fields:

| columns | grid | snow-free leaf subset | full leaf set | leaves + device stage |
|---|---|---|---|---|
| 512 | warm | max_ulp 0 | max_ulp 0 | max_ulp 0 |
| 512 | snow | max_ulp 0 | max_ulp 0 | max_ulp 0 |
| 4,096 | warm | max_ulp 0 | max_ulp 0 | max_ulp 0 |
| 4,096 | snow | max_ulp 0 | max_ulp 0 | max_ulp 0 |
| 24,576 | warm | max_ulp 0 | max_ulp 0 | max_ulp 0 |
| 24,576 | snow | max_ulp 0 | max_ulp 0 | max_ulp 0 |

The 24,576-column row was measured twice, in separate processes, with the
same answer — this card has no ECC and near-capacity runs have corrupted
results before, so a single run is not evidence.  Whole-machine VRAM never
exceeded **2,513 MiB**, against the 29,500 MiB rail.

The WRF oracle suite is unchanged and green through all of it (60 tests in
`tests/test_ruc.py`).  That is the point worth keeping: **the fixtures cannot
distinguish the two forms**, which is exactly why the defect survived.  The
gate that catches it is composition at width, not the fixture.

---

# What one call actually costs, and what got faster

## The decomposition, before optimising

One 24,576-column call, warm grid, with the four leaves already on the card
and everything else on the host:

| | ms | share | ms/column |
|---|---|---|---|
| `ruc_snow_preparation` | 317.4 | **87.6 %** | 0.01292 |
| `leaf:soil` (kernels + per-batch transfers) | 17.1 | 4.7 % | 0.00069 |
| `ruc_surface_parameters` | 1.4 | 0.4 % | 0.00006 |
| driver + recombination | 26.4 | 7.3 % | 0.00108 |
| **total** | **362.3** | | **0.01474** |

**The kernels were never the cost.**  The same decomposition on the all-host
path, at 3,072 columns, reads `leaf:soil` 0.4841 ms/column,
`ruc_snow_preparation` 0.0130, `ruc_surface_parameters` 0.0069, driver
0.0017 — and 0.0130 + 0.0069 + 0.0017 = **0.0216**, which is the 0.022
ms/column this document previously recorded as the device figure.  The
"device cost" was almost entirely two host-only per-column Python loops that
no leaf conversion could reach, because neither is a leaf.

## What changed

**1. `ruc_surface_parameters` is vectorised over the horizontal.**
`soilvegin` is a table lookup and five float32 operations; every one of them
keeps WRF's default-real evaluation order and operand pairing, so the answer
is bit-identical.  Checked against the loop it replaces — re-spelled verbatim
from HEAD, so the comparison is not the new code against itself — over
6 x 20,000 random columns in both `rdlai2d` modes.  **0.0060 -> 0.00009
ms/column, 66-76x.**

**2. A stage seam, and the snow-preparation block on the card.**
`RUC_SFCTMP_HOST_STAGES` / `RUC_SFCTMP_DEVICE_STAGES`, with a `stages=`
keyword on `ruc_surface_temperature_step` and `ruc_land_surface_step`.  It is
deliberately **not** a fifth entry in `RUC_SFCTMP_HOST_LEAVES`: a column takes
exactly one of the four leaves, whereas every column goes through the
preparation block, and conflating the two would make the leaf docstring's
`:1767-2195` citation wrong.  The device entry wraps
`ruc_snow_preparation_cuda`, which has been `max_ulp 0` against the
unmodified WRF preparation block across all three snow-cover options for
weeks with nothing able to launch it from the driver.

It is fail-closed on a foreign parameter bundle.  The kernel indexes
`z0tbl`/`lemitbl`/`URBAN` uploaded from the default bundle, so a caller who
supplies a different one is refused rather than silently handed the default
tables — `tests/test_ruc_device_column.py` gates that by constructing an
altered bundle and requiring the raise.

## The decomposition after

Same call, same width, device stages:

| | ms | share | ms/column |
|---|---|---|---|
| `leaf:soil` | 18.1 | 32.3 % | 0.00074 |
| `stage:snow_prep` | 10.9 | 19.4 % | 0.00044 |
| `ruc_surface_parameters` | 1.2 | 2.1 % | 0.00005 |
| driver + recombination | 26.0 | **46.2 %** | 0.00106 |
| **total** | **56.2** | | **0.00229** |

362.3 ms -> 56.2 ms, and what is left is no longer a Python loop: it is the
driver's host-side marshalling and the per-batch transfer boundary.  A
`cProfile` of a 98,304-column call names it — `ruc_land_surface_step` 0.035 s
of `tottime`, `ruc_surface_temperature_step` 0.022, `ndarray.get` (the D2H
side) 0.016, `_float_field` 0.015, `_sfctmp_values` 0.012.  **That is the
signature of a call that should stop returning to the host between stages**,
and making the whole column device-resident is the next conversion — a
different job from converting a leaf, and not one this lane started.

On a snow grid the balance moves further that way: the four leaves and the
stage are 32.6 % of a 24,576-column snow call and the driver + recombination
is **66.7 %** (0.00458 ms/column), because the mosaic recombination weights
43 fields against `snowfrac` on the host.

## Per-column cost across width

Warm grid, best of three, paired in one process:

| land columns | 3,072 | 12,288 | 24,576 | 49,152 | 98,304 |
|---|---|---|---|---|---|
| host leaves | 0.4999 | 0.5324 | 0.5233 | 0.5082 | 0.4975 |
| device leaves | 0.0195 | 0.0229 | 0.0152 | 0.0148 | 0.0146 |
| **device leaves + stages** | 0.0091 | 0.0133 | **0.0025** | **0.0019** | **0.0016** |
| speedup over host | 54.7x | 40.1x | 211.9x | 267.6x | 320.0x |

The two narrow points are noisy — the CuPy pool is still growing there — and
should be read as an upper bound rather than a measurement.  The curve is
still falling at 98,304, which is what work genuinely reaching the device
looks like.

## d04, measured at d04's width

Not extrapolated.  600 x 600 = 360,000 columns, best of four, `dt = 1.667 s`,
one land-surface call per step:

| columns | grid | ms/column | one call | **wall s per simulated minute** | whole-machine VRAM |
|---|---|---|---|---|---|
| 49,152 | warm | 0.00173 | 0.62 s | 22.4 s | 1,689 MiB |
| 98,304 | warm | 0.00137 | 0.50 s | 17.8 s | 1,825 MiB |
| 196,608 | warm | 0.00130 | 0.47 s | 16.8 s | 2,089 MiB |
| **360,000** | **warm** | **0.00130** | **0.47 s** | **16.8 s** | 2,489 MiB |
| 49,152 | snow | 0.00578 | 2.08 s | 74.9 s | 1,697 MiB |
| 98,304 | snow | 0.00521 | 1.87 s | 67.5 s | 1,833 MiB |
| 196,608 | snow | 0.00502 | 1.81 s | 65.1 s | 2,105 MiB |
| **360,000** | **snow** | **0.00507** | **1.83 s** | **65.7 s** | 2,513 MiB |

The snow figure is a **fully** snow-covered d04, where every column runs both
`soil` on the mosaic snow-free fraction and `snowsoil` on the snow column.
An early-April continental outbreak with snow across the northern part of the
domain sits between the two rows, nearer the warm one.

The whole history of this number, for one domain's land surface:

| | one 360,000-column call | wall s per simulated minute |
|---|---|---|
| host loop, before this lane | 616 s | 22,180 s (6.2 h) |
| host, batched | 181 s | 6,516 s (1.8 h) |
| device leaves, snow-free only | 8 s | 288 s (4.8 min) |
| **device leaves + stages, snow-free** | **0.47 s** | **16.8 s** |
| **device leaves + stages, all snow** | **1.83 s** | **65.7 s** |

For scale, the four-domain MYNN stack costs 27 s per simulated minute.  RUC's
land surface on d04 is now **16.8 s** snow-free — the same order, rather than
ten times the whole rest of the model — and it is bitwise on snow, which it
was not before.

## The checkable list, re-stated again

| | item | previous state | now |
|---|---|---|---|
| 1 | RUC soil state object and handoff | CLOSED | unchanged |
| 2 | seven initializers behind one selector | CLOSED | unchanged |
| 3 | `soil_contract.py` mapped source refuses RUC | STILL OPEN, deliberately | unchanged.  Admitting it needs a WRF-real nine-level initialization to verify against |
| 4 | no RUC profile row, no column-budget arm | UNCHANGED, on purpose | **still untouched.**  `gpuwm/physics_compat.py` not edited |
| 5 | `namelist_compat.py:764-768` refuses RUC | UNCHANGED | **still untouched** |
| 6 | the width answer | 0.022 ms/column, snow-free only, snow NOT admissible | **0.0013 ms/column snow-free and 0.0051 on snow, both bitwise at 24,576 columns.  16.8 / 65.7 wall seconds per simulated minute at d04** |

Item 6 no longer carries a correctness asterisk.  What remains against RUC as
a selectable option is items 3, 4 and 5, all of which are decisions rather
than measurements, plus the honest statement that **16.8 s per simulated
minute is still the land surface alone** and the driver's host-side
marshalling is now 46-67 % of it.

**Whether RUC becomes selectable is not this lane's decision and nothing here
was written on the assumption that it will.**  No registry template, no
maturity grade, no column budget, no route override, and `bldt` and the call
frequency were not touched.


# The conversion finished, and the door opened (2026-07-26, later still)

## Verdict, in the unit that was asked for

**Wall seconds per simulated minute at d04**, 600 x 600 = 360,000 columns,
`dt = 1.667 s`, `bldt = 0` so one land-surface call per timestep.  Measured at
that width, not extrapolated, and paired in one process.

"Before" has two honest readings and both belong here, because the difference
between them is the finding of this session:

| | what a forecast RAN | what the device leaves COULD do | after |
|---|---|---|---|
| snow-free | 6,745 s | 17.14 s | **2.36 s** |
| fully snow-covered | 17,663 s | 69.94 s | **3.95 s** |

The middle column is the path the previous session measured and published as
16.8 / 65.7 s; today's re-derivation of the same path gives 17.14 / 69.94, so
that measurement was right about the code and wrong about what ran.  The left
column is what `ruc_lsm_step` actually executed, because **nothing under
`gpuwm/` imported the device leaves**.  Against the path a user was really
paying for, the change is **2,858x warm and 4,472x on snow**; against the
device-leaf path it is **7.27x and 17.69x**.

And **RUC is selectable**, at `implemented-unverified`, through
`wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1` on the `era5` and
`gfs` sources of `tools.prepared_single_domain_forecast`.

## The finding that mattered more than any of the timings

**The device leaves had zero importers under `gpuwm/`.**

`gpuwm/core/ruc_gpu.py` carried four bitwise `sfctmp` leaves and a bitwise
snow-preparation stage.  `gpuwm/core/ruc_runtime.py:ruc_lsm_step` -- the one
seam a forecast goes through -- called `ruc_land_surface_step` with no
`leaves=` and no `stages=`, so **every RUC forecast this project has ever run
took the host Python path**, at 613 s per d04 land-surface call, while the
per-leaf parity gates stayed green and the previous session's 16.8 s figure
described a code path nothing launched.

This is the same failure the RUC soil ingest had and the same one this
document already recorded for it: *finished, verified, and wired to nothing.*
Two independent things had it in the same lane within a week.  A timing gate
cannot catch it, because the host path is also correct and merely slow.  What
catches it is
`tests/test_ruc_runtime.py::test_a_forecast_runs_with_every_host_sfctmp_leaf_booby_trapped`,
which replaces all four host leaves and the host stage with raising tripwires
and runs a snow-covered forecast; then it withholds the device sets from the
same call and requires the tripwire to fire, so the pass is evidence rather
than a gate that has never been able to fail.

## The decomposition, before optimising anything

The previous session stopped at "host marshalling and the transfer boundary,
46% of a warm call and 67% of a snow call."  That is a name, not a
measurement, and this lane's own rule says decompose first.  One
24,576-column call with the leaves and the stage on the card, split by where
the time actually goes:

One 24,576-column **warm** call with the four leaves and the preparation
stage already on the card and the dispatch on the host -- i.e. the best this
tree could do before this session:

| | ms | share | calls |
|---|---|---|---|
| `leaf:soil` kernels | 14.51 | 26.0 % | 1 |
| `stage:snow_prep` kernels | 6.85 | 12.3 % | 1 |
| H2D copy | 3.36 | 6.0 % | **154** |
| **H2D per-field finiteness validation** | **9.88** | **17.7 %** | **150** |
| D2H, `leaf:soil` | 1.83 | 3.3 % | 1 |
| D2H, `stage:snow_prep` | 2.80 | 5.0 % | 1 |
| residue: host dispatch | 16.51 | 29.6 % | |
| **total** | **55.7** | | |

Two things in that table were not what "the transfer boundary" suggested.

**The bytes were never the problem.**  Moving the data is 6.0 % going down
and 8.3 % coming back.  What cost 17.7 % -- more than both copies together --
was `_float_field`'s `bool(cp.all(cp.isfinite(raw)))`, run once per field:
**150 full stream synchronisations for one land-surface call**, each costing
what a synchronisation costs whether it guards four bytes or a megabyte.  The
same shape appeared at the end of `sfctmp`, which read 53 finiteness
reductions field by field; both now stack their reductions and take **one**
host read, which is the same check and the same message.

**The host dispatch was the largest single term**, at 29.6 %, and none of it
is arithmetic anyone would call physics -- it is masking, gathering,
scattering and the mosaic recombination.  That is what the array namespace
moved.

And then the **snow** call, same width, which is where the real answer was:

| | ms | share | calls |
|---|---|---|---|
| **`_ruc_tanh_array` (host Python loop)** | **78.64** | **49.2 %** | 1 |
| residue: host dispatch | 17.39 | 10.9 % | |
| H2D per-field finiteness validation | 15.89 | 9.9 % | **246** |
| `leaf:soil` kernels | 14.40 | 9.0 % | 1 |
| `leaf:snow_soil` kernels | 14.23 | 8.9 % | 1 |
| `stage:snow_prep` kernels | 6.95 | 4.3 % | 1 |
| H2D copy | 5.65 | 3.5 % | 252 |
| D2H, all three | 6.75 | 4.2 % | 3 |
| **total** | **159.9** | | |

**Half of a snow-covered RUC call was one Python `for` loop.**  Not the
transfer boundary, not the kernels: `TANH` at `module_sf_ruclsm.F:2087`,
called once per snow-covered column from inside the dispatch.

That is the **third** time in this lane that the thing which looked like a
floor turned out to be a host loop hiding behind a device figure --
`ruc_snow_preparation` at 87.6 % of "the device cost", `ruc_surface_parameters`
at 32 %, and now this at 49.2 %.  Each time the previous session had already
named the remainder confidently and wrongly.  **Decompose before optimising,
every time**, and decompose the snow case separately from the warm one: they
are not the same call with more work in it, and here they had different
dominant terms.

## What changed

**One transcription, two array namespaces.**  `ruc_surface_temperature_step`
and `ruc_land_surface_step` now take an `arrays=` namespace and rebind `np`
to it for the whole body:

```python
np = arrays if arrays is not None else _NUMPY
```

`arrays=None` is numpy, and then `np` inside the body **is** the module
object it always was -- so the host path is not a configuration of a new
mechanism, it is unchanged code, which is what lets the oracle fixtures keep
meaning what they meant.  `gpuwm.core.ruc_gpu.RUC_DEVICE_ARRAYS` is CuPy, and
with it the masking, the four gathers, the scatters, the mosaic
recombination, LSMRUC's prologue and its epilogue are all kernels over the
same device arrays the leaves wrote.

The alternative was a device mirror of the dispatch, and this document has
already recorded what mirrors cost: a mirror is not an oracle.  The whole
conversion is 158 added lines in `gpuwm/core/ruc.py`, almost all of them
`arrays=` threading and docstring, because the 1,900 lines of transcribed
arithmetic did not have to change at all.

**`np.float32` has two jobs and CuPy can only do one of them.**  The drivers
spell `np.float32(a * b)` to pin a float32 boundary and `dtype=np.float32` to
allocate.  `np.float32(device_array)` raises -- a CuPy array has no
`__float__`.  numpy accepts *any object with a `dtype` attribute* wherever a
dtype is wanted, so `_RucDeviceFloat32` is a valid dtype **and** a callable
cast, and not one of the 314 call sites had to be respelled.  The cast
copies, exactly as `np.float32(host_array)` does; returning the input where
the dtype already matched would alias a caller's array into a scattered
assignment and is a different function.

**`_ruc_tanh_array` was a Python loop inside the dispatch.**
`module_sf_ruclsm.F:2087` evaluates `TANH(snhei / (2.5*min(0.2,znt) *
(rhosn/rhonewsn)))` in the melt-out block -- which is in the DISPATCH, not in
any leaf, so **no leaf conversion could ever have reached it** and every
earlier decomposition charged it to "driver + recombination".  On a fully
snow-covered grid it ran once per column, as a Python function call spelling
out fdlibm's reduction.  `ruc.cu` already had `ruc_tanhf_glibc` as a
`__device__` function for the preparation kernel; it now also has a kernel
wrapper, and the two spellings are gated against each other at `max_ulp 0`
over 48,015 arguments including both branch boundaries.

**The surface slab stays on the card.**  `ruc_lsm_step` used to copy roughly
seventy 2-D fields and five nine-level profiles down to the host and back --
about 165 MB each way at d04 -- for arrays that were already on the device
and went straight back to it.  It now hands `ruc_land_surface_step` the
device arrays directly.

## What is deliberately still on the host, and why

`SFCDIAGS_RUCLSM` (`module_sf_sfcdiags_ruclsm.F`) raises `(1e5/psfc)` to
`R/cp` with `np.power` on a float32 array.  **numpy's float32 power and
CUDA's `powf` are two different non-glibc functions** -- that is the exact
divergence class that put 2 ULP into `hfx` at width earlier in this lane and
cost a session to find.  Moving it to the card is a transcendental-policy
decision (see `_RUC_PROVISIONAL_TRANSCENDENTALS`), not a performance one, and
the requirement that outranks speed is that the answer must not change.  So
exactly the twelve fields it reads come down and the three it writes go back
up: a bounded, named cost instead of the whole slab, and the arithmetic stays
where it is verified.

`ilnb_chain=True` is **refused** on a non-host namespace rather than
accepted-and-ruinous.  It reproduces WRF's uninitialised `ilnb`, which is a
genuine per-column sequential dependence -- column *i*'s seed is column
*i-1*'s answer -- so dispatching it one column at a time to the card would be
slower than the host.  The forecast runtime uses `ilnb_chain=False`, gpuwm's
DEFINED behaviour, and the fixture path that needs the chain still runs on
the host where it belongs.

## Parity

Paired in one process against the same tree, host driver versus
device-resident driver, **all 46 fields of `RucLandSurfaceStep`**:

| land columns | warm | snow |
|---|---|---|
| 512 | max_ulp **0** | max_ulp **0** |
| 4,096 | max_ulp **0** | max_ulp **0** |
| 24,576 | max_ulp **0** | max_ulp **0** |

Each row carries water and sea-ice columns as well as land, so all four
leaves, both mosaic recombinations and both snow-free arms are exercised.  A
six-step trajectory at 512 snow-covered columns is bitwise too, which is what
brings the accumulators (`SFCEVP`, `ACSNOW`, `SFCRUNOFF`, `UDRUNOFF`,
`ACRUNOFF`) and the carried soil state into the comparison rather than only
one call's fluxes.  `ruc_tanhf_glibc` is `max_ulp 0` against `_f32_tanh` over
48,015 arguments spanning both branch boundaries and the subnormal cutoff.

**The falsification failed first, and it was right to.**  The gate perturbs
one ULP of the entry `soilmois` and requires the answer to move.  Its first
spelling nudged column 0 of a grid whose first eight columns are water, and
it FAILED -- correctly: `LSMRUC`'s water arm (`:824-850`) overwrites the soil
profile without ever reading it, so a one-ULP change there is genuinely
unobservable.  The test now derives a land index from the water and sea-ice
counts, asserts the layout before perturbing, **and keeps the water-column
nudge as a second assertion requiring it NOT to move the answer** -- so the
explanation is itself measured rather than asserted.

The whole RUC oracle suite (`gpuwm/data/ruc/oracle`) is unchanged and green
throughout; the host path is the same code it was, because `arrays=None`
binds `np` to the numpy module object itself.

## What one call costs now

Paired in one process at d04's width, 360,000 columns, best of three:

| | one call | wall s per simulated minute | whole-machine VRAM |
|---|---|---|---|
| host loop, warm (projected from a measured-flat 0.49 ms/column) | 187.4 s | 6,745 s | 1,617 MiB |
| host loop, snow (projected) | 490.7 s | 17,663 s | 1,617 MiB |
| device leaves + stage, host dispatch, warm | 476.2 ms | 17.14 s | 2,189 MiB |
| device leaves + stage, host dispatch, snow | 1,943.1 ms | 69.94 s | 2,245 MiB |
| **device-resident, warm** | **65.5 ms** | **2.36 s** | 3,399 MiB |
| **device-resident, snow** | **109.9 ms** | **3.95 s** | 3,585 MiB |

**7.27x warm and 17.69x snow** over the path this session started from, and
the snow figure moved further because the snow path was where the remaining
host loop was: `_ruc_tanh_array`, once per snow-covered column.  The snow/warm
ratio was 4.1x before and is 1.7x now -- a fully snow-covered d04 is no longer
qualitatively more expensive than a warm one.

For scale, the four-domain MYNN stack is 27 wall seconds per simulated
minute.  **RUC's d04 land surface is now 2.36 s snow-free and 3.95 s over a
full snow pack**, i.e. under 15% of the rest of the model in the worst case,
where at the start of this session a forecast was actually paying 6,745 s --
about 112 minutes of wall clock per simulated minute.

The three rows are the same physics: the first is what `ruc_lsm_step` ACTUALLY
ran, the second is what the device leaves could do with the dispatch left on
the host, and the third is the whole column device-resident.  Peak
whole-machine VRAM 3,585 MiB against the 29,500 MiB rail.

# The registry template

## What shipped

`wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1`, at
`implemented-unverified`, listed on the `era5` and `gfs` sources of
`tools.prepared_single_domain_forecast`.

It differs from `wsm6-ysu-mm5-noah-no-radiation-v1` in **exactly one
selector** -- `sf_surface_physics` 2 -> 3, with `num_soil_layers` 4 -> 9
following from it -- so a side-by-side run isolates the land-surface change
from the microphysics, PBL, surface layer, radiation, cumulus and diffusion
settings.  That is the same discipline the MYNN template used and it is what
makes the template useful as an experiment rather than merely available.

The maturity did **not** move.  `implemented-unverified` is the honest grade:
every routine is oracle-matched against the byte-unmodified
`module_sf_ruclsm.F`, the assembled driver is bitwise against
`oracle/lsmruc.csv` except its 26 pinned upstream-residue cells, and the CUDA
column a forecast launches is `max_ulp 0` against that host driver at width --
but **no gpuwm/WRF forecast-trajectory comparison exists**, and that is what
`validation-candidate` requires.

`registry_sha256()` is `sha256(raw[:-1])` and still reproduces: the file is
one canonical compact line with a trailing newline, and the edit was applied
as a JSON transform re-serialised with `sort_keys=True,
separators=(',', ':')`, checked round-trip byte-identical before the edit.

## Why `era5` and `gfs` and not `20crv3`

`gpuwm/mapped_direct.py` hands the soil seam the composition's DECLARATIVE
layer contract, and `gpuwm/ingest/soil_contract.py:validate_soil_layer_contract`
still declares **exactly one target**: Noah's four layers.  RUC's nine are a
different discretization with a different value-location convention rather
than a longer list of the same thing, so `preprocess_ruc_soil` refuses a
contract by name.  Listing the template for the mapped source would be a
reachability claim the ingest cannot honour.  Both halves are gated:
`test_the_mapped_source_is_deliberately_not_offered_ruc` checks that the route
omits it **and** that the refusal is real.

The same reasoning kept RUC out of `SINGLE_DOMAIN_PHYSICS_PROFILES`, which
`tools/hrrr_single_domain_benchmark.py` publishes verbatim: the HRRR path runs
the soil state through `canonical_noah_surface`, which nothing has exercised
on a nine-level `RucSoilState`.

## The column-count rail: a decision, not an omission

`gpuwm/physics_compat.py` has no `sf_surface_physics == 3` arm and **this lane
did not add one.**  That is not the same gap it was.

A width rail exists to refuse a call that *cannot finish* before hours are
spent discovering it.  Noah-MP's was 7.18 ms per land column, flat, which was
43 minutes for one d04 land-surface call -- a run that would not complete.
(*Update 2026-07-27:* the slab orchestration retired that premise -- the same
d04 call is measured at 0.202--0.227 s -- and the rail's ceiling moved to the
measured 360,000 columns, refusing only widths nothing has measured.)
RUC's d04 call is measured at 65.5 ms warm and 109.9 ms over a full
snow pack.  A rail there would be refusing a run that finishes.

What a user needs instead is the number, before the run rather than during
it, and that is published in the template's own warnings and in the option's:
2.36 s and 3.95 s wall seconds per simulated minute at d04, with
the four-domain MYNN stack's 27 s beside it for scale.
`tests/test_ruc_admission.py::test_the_width_rail_covers_noahmp_and_not_ruc`
records both halves and shows the authority firing for the scheme that has
one, so "RUC has no rail" is a measured statement rather than an untested
assumption.

Two further reasons not to reach into that file now, in descending order of
weight: the rail's premise is false by measurement, and `physics_compat.py`
is outside this lane's file set and currently carries another lane's
uncommitted work.  The first is the reason; the second is why it would have
been awkward even if the first had gone the other way.

## The checkable list, closed out

| | item | previous state | now |
|---|---|---|---|
| 1 | RUC soil state object and handoff | CLOSED | unchanged |
| 2 | seven initializers behind one selector | CLOSED | unchanged |
| 3 | `soil_contract.py` mapped source refuses RUC | STILL OPEN, deliberately | **STILL OPEN, and now scoped**: the template is offered for `era5`/`gfs` only and says so.  Admitting the mapped source needs RUC's target added and its remap proved against a WRF-real nine-level initialization |
| 4 | no RUC profile row, no column-budget arm | UNCHANGED, on purpose | **profile row: CLOSED** (`RUC_PROFILE_ID` in `physics_compat.py`, `_SOURCE_PHYSICS_PROFILES` in the runner).  **column-budget arm: DECIDED, not added** -- see above |
| 5 | `namelist_compat.py:764-768` refuses RUC | UNCHANGED | **still untouched.**  A namelist import of a RUC run is a separate lane from selecting one in the registry |
| 6 | the width answer | 0.0013 ms/column snow-free, 0.0051 on snow; 16.8 / 65.7 wall s per simulated minute, on a code path nothing launched | **2.36 s / 3.95 s wall seconds per simulated minute at d04, on the path `ruc_lsm_step` actually takes**, gated by a booby trap so it cannot silently revert |

## Still open, honestly

* No gpuwm/WRF **forecast trajectory** comparison.  That is the whole of the
  distance between `implemented-unverified` and `validation-candidate`, and
  nothing in this session moved it.
* The **mapped/declarative soil contract** still refuses RUC by name (item 3).
* `namelist_compat.py` still refuses a RUC namelist import (item 5).
* `SFCDIAGS_RUCLSM` is still host arithmetic, on purpose, and it is now the
  only host round trip left on the seam.  Closing it is a transcendental
  policy question -- glibc `powf` transcribed identically on both sides --
  and not a performance one.
* `isncovr_opt` 1 and 3 are transcribed and still **not** oracle-verified;
  the pinned object compiles 2.
* The binary `SR` proxy outside the supported microphysics family and the
  doubled `SFCEVP` are unchanged and still published. WRF-ARW's
  `EM_CORE==1` species partition is now the forecast default.
