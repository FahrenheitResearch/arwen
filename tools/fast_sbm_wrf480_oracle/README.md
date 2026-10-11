# Fast SBM Fortran oracle (WRF v4.8.0, mp_physics 30)

This directory builds WRF's own fast spectral bin microphysics (FSBM-2,
`phys/module_mp_fast_sbm.F`) as a standalone Fortran program, runs it on
chosen inputs, and packs what it returns into the fixtures in
`tests/data/fast_sbm_oracle/`. The float64 host port in
`gpuwm/core/fast_sbm/` is gated against those fixtures (design:
`fast-sbm-DESIGN.md`, sections 3.3, 3.7 and 5, in the physics program folder).

Nothing here runs at model time. The fixtures are read only through
`fixtures.py`, which checks every file's SHA-256 before returning anything.

Milestone 1 made the warm fixtures (the tables, the warm sweeps and the warm
column KC-1). Milestone 2 (the cold half, plan `FSBM-MILESTONE-2-PLAN.md` in
the physics program folder) extends the same harness with the cold sweeps and
refusals, a graupel build and the cold column KC-2. Every milestone 1 stream
comes out of the extended harness byte for byte as milestone 1 packed it:
`pack_fixtures.py` refuses otherwise, and the test pins their SHA-256.

## Files

| File | Role |
| --- | --- |
| `build.sh` | Normalizes and pin-checks the two WRF sources, fetches and pin-checks the NCAR table archive, applies the two transforms, compiles every build, runs every runner, and writes the controls. Output: `<work>/out/`. |
| `sources.sha256` | The pins: the two WRF sources (LF content), `SBM_input_33.tgz` and its 18 members. |
| `stubs_wrf.F90` | Stand-ins for the five WRF framework routines and two modules the scheme uses. `wrf_error_fatal` writes its message to stderr and to a refusal record (`FSBM_REFUSAL_FILE`), then `STOP 99`. |
| `transform_access.py` | Deletes exactly line 3862 (` PRIVATE` of `module_mp_fast_sbm`), matched by content and cross-checked by number, and refuses unless its diff is that one line. |
| `transform_graupel.py` | Changes exactly `hail_opt = 1` to `hail_opt = 0` (line 3886 of the pristine source), on the pristine source or on the probe source (whose hash it must be given), and refuses unless its diff is that one line. |
| `instrument_probes.py` | Applies the access transform, then inserts CALL lines to the probe routines at named anchors in `FAST_SBM`, `COAL_BOTT_NEW`, `JERNUCL01_KS`, `ONECOND2` and `ONECOND3`; refuses unless the diff is those CALL lines only and removing them restores the input byte for byte. |
| `fsbm_probe.F90` | The record writer (module `fsbm_io`, always compiled without promotion) and the probe routines (compiled with the build's flags, so their dummies promote exactly as WRF's actuals do). The cold routines write records only in a run that configured a tile (KC-2); in any other run they only count and keep capture values for the sweeps. |
| `run_dump_tables.F90` | Tables as read, derived tables, and the collision kernels at ten pressures. |
| `run_sweeps.F90` | The process sweeps and the refusal cases, warm (milestone 1) and cold (milestone 2). |
| `run_column.F90` | The warm kinematic column KC-1, its one-ULP sensitivity rerun and the cross-replay. |
| `run_kc2.F90` | The cold kinematic column KC-2 (hail and graupel builds), its one-ULP sensitivity rerun and the cross-replays. |
| `pack_fixtures.py` | Checks everything `build.sh` wrote and packs it; refuses on any failed check. |
| `fixtures.py` | The only reader of the packed fixtures. |

## Rebuilding the fixtures

The authority is the reference build host (gfortran 15.2.0); gfortran 13.3.0
reproduces every stream byte for byte (see below). The whole
harness is CPU only and takes about five minutes (milestone 1 alone took one).

1. Make a private scratch directory on a CPU build host and copy into it
   `wrf/phys/module_mp_fast_sbm.F` and `wrf/phys/module_mp_SBM_polar_radar.F`
   from the WRF tree (the science-refs checkout is CRLF; `build.sh` normalizes
   it) and this directory's harness files under `tools/`.
2. Run `bash tools/build.sh wrf work > build.log 2>&1` there. It fetches
   `SBM_input_33.tgz` from NCAR (or takes `--tables <path>`) and refuses unless
   every pin matches. Every compile and run is `nice -n 10`, each run is
   bounded at 600 s.
3. Copy `work/out/` back with the runners' stdout files (`tar -czf - out`,
   about 27 MB): the pack tells ICE_NUCL's bare `STOP 099` from a stub
   whose refusal record failed by ICE_NUCL's own print in the refusal
   case's `.out`, and refuses without it.
4. From the worktree root: `python tools/fast_sbm_wrf480_oracle/pack_fixtures.py <out>`.
   To record a second compiler's reproduction, add
   `--reproduction <dir> --reproduction-label <text>`, where `<dir>` holds that
   build's `raw.sha256` (from `cd out && find . -name '*.raw' | sort | xargs sha256sum`),
   `build_info.json` and `controls.json`.
5. Run `tests/test_fast_sbm_oracle_fixtures.py`.
6. Remove the scratch directory and report its size.

Node etiquette: CPU only, niced, in the private scratch directory, removed at
the end with its size reported. The harness never touches the GPU or the
`gpu-mutex` directory, and it does not need either. A harness file changed
without rebuilding makes the test fail: the manifest records the SHA-256 of
every harness file that produced the fixtures.

## Builds

| Build | Flags (after `-cpp -DBUILD_SBM_FAST=1 -DIWORDSIZE=4 -ffree-form -ffree-line-length-none`) | Role |
| --- | --- | --- |
| O64 | `-O0 -ffp-contract=off -freal-4-real-8` | WRF promoted to float64: the tight gate for the float64 host |
| O32 | `-O0 -ffp-contract=off` | WRF as it runs: the envelope comparison |
| O32-wrf | `-O2 -ftree-vectorize -funroll-loops` | evidence only: its differences are recorded, never gated |

Each precision is built three ways: pristine, access-transformed and probe.
The graupel variant is built two ways, pristine and probe (`gpristine`,
`gprobe`). The controls in the manifest, all asserted by the test: over the
whole KC-1 and KC-2 runs the three hail builds give byte-identical `FAST_SBM`
outputs in each precision, and so do the two graupel builds over KC-2; two
runs of the same build are byte-identical; every sweep run twice is
byte-identical; each build replayed on its own recorded pre-call states in a
fresh process returns its own post-call states byte for byte, so `FAST_SBM`
carries no state between calls that the replay records miss; and the graupel
build's KC-2 differs from the hail build's, so its transform changed the
answer.

## Record format

Every stream is a sequence of records. A record is an int32 type tag followed
by that type's fields, raw little-endian, arrays in Fortran order. The first
record of each type defines its layout; the harness stops (exit 70) if a later
record of the same type differs in any field's name, kind or shape. A field
kind is `f8` (float64), `f4` (float32) or `i4` (int32). O64 streams hold no
`f4` field at all. O32 streams hold `f4` where WRF declares default real or
`REAL(KIND=4)` and `f8` where it declares `REAL(KIND=8)`.

In the manifest each record type lists its fields as `"<name> <kind> <Fortran
dims...>"`; `fixtures.py` returns each array with those dimensions, so
`a[i - 1, j - 1]` is `A(I, J)`.

## Families

| Family | Record types |
| --- | --- |
| `tables_o64`, `tables_o32` | `read` (every table as read from `SBM_input_33`), `derived` (ima, chucm, both mg mass grids, ECOALMASSM, BRKWEIGHT, Prob, Gain_Var_New, NND, ikr_spon_break, the aerosol spectra and grids, DROPRADII) |
| `kernels_o64`, `kernels_o32` | `kernels` x 10: cwll and every cw* family after `Kernals_KS` at 1000, 950, 850, 750, 700, 600, 500, 400, 300 and 250 hPa |
| `sweep_polysvp_*` | `case`: 233-313 K and edge values, water and ice |
| `sweep_jerrate_*` | `case`: drops at 4 temperatures and 4 pressures |
| `sweep_jersupsat_*` | `case`: every branch, with the harness's branch classification and the G2 clamp flag |
| `sweep_jerdfun_*` | `case`: growth and evaporation, small and large, both IDROP settings |
| `sweep_jernewf_*` | `case`: the remap through `JERDFUN_KS` with IND 2 (JERNEWF_KS is private to its own module): growth, evaporation, below bin 1, the 3-point path, the tail remap |
| `sweep_onecond1_*` | `case`: condensation and evaporation |
| `sweep_nucleation_*` | `case`: warm `JERNUCL01_KS`, both activation branches |
| `sweep_coll_xxx_*` | `kernel` (cwll per pressure and step), `case` |
| `sweep_coal_bott_*` | `case`: `COAL_BOTT_NEW`, with its breakup path and final ndiv read back through the probe build |
| `sweep_breakup_*` | `case` (single calls), `sequence` (the ndiv 1, 2, 4, 8 substep sequences) |
| `sweep_spont_*` | `case`, including both early-return paths |
| `sweep_falflux_*` | `case`: uniform and stretched grids, several substeps |
| `refusals` | a container of 12 streams (six cases, two builds) with each case's message and exit status |
| `refusals_cold` | a container of 26 streams (thirteen cold cases, two builds), each with its message, exit status, stop kind, WRF line and whether FAST_SBM can reach the input |
| `kc1_case` | `case` (the definition in float64), `literal_advection_check` |
| `kc1_summary_o64`, `kc1_summary_o32` | `initial`, `step` x 300 (state after every call), `acceptance` (tmin, rain and branch counts) |
| `kc1_replay_o64`, `kc1_replay_o32` | `pre` and `post` at steps 2, 3, 4, 6, 10, 15, 25, 40, 60, 80, 100, 130, 160, 200, 250, 300 |
| `kc1_crossreplay_o64_on_o32` | `post` x 16: the O64 build once on each O32 pre-call state |
| `kc1_sensitivity_o64` | as `kc1_summary_o64`, every initial theta one ULP higher |
| `kc1_probe_o64`, `kc1_probe_o32` | records inside `FAST_SBM` and `COAL_BOTT_NEW` at steps 10, 40, 80, 130, 200, 300 on levels 1, 5, 10, 15, 20, 25, 30, 40 of every column: `entry`, `forcing`, `nucl_before/after`, `cond1_before/after`, `coal_before/after`, `xxx_before/after`, `kernel` (cwll for the first 40 recorded collision calls), `breakup`, `spont_before/after`, `fall_before/after`, `fall_ice_before/after`, `diag`, and `counts` |
| `sweep_freez_*`, `sweep_melt_*` | `grid`, `case`: FREEZ and J_W_MELT on the seven species (liquid, three habits, snow, graupel, hail) |
| `sweep_ice_nucl_*` | `case`: ICE_NUCL through JERNUCL01_KS with the water supersaturation below zero, with ICE_NUCL's own call read back |
| `sweep_jernucl01_ice_*` | `case`: JERNUCL01_KS with ice, with homogeneous freezing and ICE_NUCL read back |
| `sweep_jerrate_ice_*`, `sweep_jertimesc_ice_*`, `sweep_jerdfun_ice_*` | `case`: the ice branches of the growth-rate, relaxation and remap routines |
| `sweep_onecond2_*`, `sweep_onecond3_*` | `case`: every KCOND regime, with the JERSUPSAT_KS call inside read back (its rates, KCOND and branch) |
| `sweep_coll_cold_*` | `grid`, `draws`, `orient` (the old port's 24 cases in three sets: synthetic y = 1.25 x, synthetic y = x, and the real tables), `table_grid`, `table` (the calls COAL_BOTT_NEW makes) |
| `sweep_modkrn_*` | `case`: modkrn_KS CASE 11, the output starting at -1 |
| `sweep_coal_cold_*`, `sweep_coal_cold_graupel_*` | `case`: COAL_BOTT_NEW with ice in the hail and graupel builds, with the cold block's calls, Hallett-Mossop values and riming heat read back; `colleff` is WRF's value only where `colleff_defined` is 1 (modkrn_KS leaves it unset above 0 C in the build's arithmetic and at exactly -40 C, and the record then holds 0) |
| `kc2edge_replay_*`, `kc2edge_probe_*` | the low-edge KC-2 run: `edge_case`, the static fields, `pre` and `post` at steps 2, 4 and 6; with the probe build, the tile's pre-pass arrays at the same steps and the per-column counts |
| `kc2_case` | `case` (the KC-2 definition in float64) |
| `kc2_summary_o64`, `kc2_summary_o32` | `initial`, `step` x 300, `profile` every 30 steps, `acceptance` (per-column counts) |
| `kc2_replay_o64`, `kc2_replay_o32` | `pre` and `post` at steps 2, 12, 30, 60, 120, 200, 300 (the whole domain's theta, qv, th_old, qv_old; the tile's bins) |
| `kc2_crossreplay_o64_on_o32` | `post` x 7: the O64 build once on each O32 pre-call state |
| `kc2_sensitivity_o64` | as `kc2_summary_o64`, every initial theta one ULP higher |
| `kc2_probe_o64`, `kc2_probe_o32` | at steps 10, 60, 150, 300 on levels 5, 12, 18, 23, 27, 31, 36, 42 of the four cold columns: milestone 1's records plus `prepass` (DSUPICE_XYZ, DTFREEZ_XYZ, the cloud base, and T_OLD, QV_OLD and pcgs over tile and halo), `nucl_ice`, `homfrz`, `icenucl_before/after`, `cond2_before/after`, `cond3_before/after`, `supsat`, `coal_ice_before/after`, `coal_cold_before/after`, `kernel_cold` (the seven slices for the first 2 cold calls), `hm`, `rime`, `freez_before/after`, `melt_before/after`, `merge`, `snowbrk`, `exit`, `tile_config` and `counts_tile` |
| `kc2g_replay_*`, `kc2g_summary_*` | KC-2 in the graupel build: `pre` and `post` at steps 2, 150, 300; the per-call and acceptance records |

## Reading the fixtures

```python
from tools.fast_sbm_wrf480_oracle import fixtures as fx

tables = fx.load_fixture("tables_o64")          # {"read.xl": (1, 33), ...}
for step, pre, post in fx.iter_replay("kc1_replay_o64"):
    ...                                         # pre["chem"]: (4, 40, 142)
for rec in fx.iter_probe("kc1_probe_o64", step=40, types=["cond1_before"]):
    ...
for ref in fx.iter_refusals(build="o64"):
    ref.case, ref.exit_status, ref.message, ref.records
```

`load_fixture` stacks each record type's field over its records
(`"<type>.<field>"`, leading record axis). `iter_records` keeps the stream
order, which in the probe families is the order WRF reached the anchors.
Every call raises `FixtureError` on a hash, size or layout mismatch.

## The warm column KC-1

Four columns in one tile, called together, so the cross-column behaviour of
`FAST_SBM` (design findings D1 and D5) is inside the fixture: c1 land at
1000 hPa with a 2 m/s updraft `w1 sin(pi z / H)` for 900 s; c2 the same over
water at 1005 hPa; c3 still air over water at 995 hPa with Marshall-Palmer
rain for 20 mm/h in bins 15-33 above 1.2 km; c4 land at 850 hPa, last in the
tile, with 0.3 m/s for 1800 s. 40 levels, lowest 50 m, each 2.5 percent deeper
(top 3369 m); 12 s steps, 300 steps. Between calls theta, qv and all 142 bins
are advected by first-order upwind; `th_old` and `qv_old` are not. The full
definition is in the manifest under `kc1.definition`.

Two choices differ from the design text, and both are recorded:

* The advection is the flux form with its divergence term restored. The plain
  flux form with a fixed density and a height-dependent w empties the lowest
  level; on c1's theta over the 900 s updraft it reaches 55.8 K
  (`kc1_case`, record `literal_advection_check`).
* c4 carries a moist cap, 99 percent relative humidity above 2.8 km. Without
  it the cloud-base activation branch is never reached (finding D9 below).

Acceptance, asserted in both builds: minimum temperature above 275.15 K in
every column over the whole run, surface rain in c1, c2 and c3, collisional
breakup in c3, both activation branches, the evaporation branch of `ONECOND1`,
and spontaneous breakup. The counts are in the manifest.

## Findings recorded by the oracle

* D9, cloud-base precedence. The cloud-base test at :4387 reads
  `... .and. K_found .eqv. .FALSE. .and. ...`. `.EQV.` binds more loosely than
  `.AND.`, so the right-hand side is always false and `KZ_Cloud_Base` becomes
  the highest level that is not (supersaturated with w above 10 cm/s), not the
  lowest level that is. With the design's sounding the cloud-base branch never
  ran; with the moist cap it runs once, at step 40, level 40, column c4.
* Still air has no diffusional growth. Condensation and evaporation are driven
  only by the change since the previous call (`DIFFU = 0` when `DIV1` equals
  `DIV3`, :4615), so rain falling through c3's 60 percent humidity never
  evaporates.
* Spontaneous breakup acts only on bin 33 (`ikr_spon_break` is 33 on the
  33-bin grid), so whether it does any work depends on whether bin 33 holds
  any positive value, which differs by precision.
* `FALFLUXHUCM_Z` with every fall speed zero ends the process with a bare
  `STOP` (:5360): exit status 0 and no message.

## The cold column KC-2

A 3 x 3 tile (i, j = 2..4) in the corner of a 4 x 4 domain, called together,
so the ice pre-pass (:4373-4512) reads a halo row and column (i = 1, j = 1,
never called and never changed) and takes both its interior and its IDE-1 and
JDE-1 edge branches. Four tile columns carry the cold physics: A (2,2) a deep
mixed-phase updraft from a 300 K surface, B (3,3) a Hallett-Mossop updraft
with supercooled cloud, rain and dense ice between 4.2 and 5.5 km (about -2.4
to -10.9 C, across the -3 to -8 C Hallett-Mossop band), C (4,4) a melting
stratiform column whose snow falls through a 0 C level near 390 m to the
ground, and D (4,2) a hail shaft in a downdraft. The other five tile columns
are the environment. 50 levels to 13.06 km (lowest 134 m, each 2.5 percent
deeper), 12 s steps, 300 steps; prescribed u and v that vary in i, j and k
enter only the pre-pass; advection is KC-1's operator, vertical only. The
full definition is in the manifest under `kc2.definition`.

Acceptance, asserted in both precisions for the hail build (and, where it
applies, the graupel build): temperatures from about 300 K through 0 C to
below -40 C; ICE_NUCL called in every habit band and adding crystals in
bands 2 to 6 (band 1, above -4 C, lies above -5 C, where HELEK1 is zero);
homogeneous freezing; ONECOND1, 2 and 3; KCOND 20 and 21, 30, 31 and 32;
every live cold collision call of the build and none of the others;
Hallett-Mossop splinters; riming heat; FREEZ to plates and to hail; every
snow and dense-ice melt band; D11 and D15 points; BreakUp_Snow never called
though its guard is true everywhere; and surface rain, snow and hail or
graupel. The counts are in the manifest, and the test recomputes each one
from the summary fixture the Fortran wrote.

KC-2's tile starts at i, j = 2, so it never takes the pre-pass's I = 1 and
J = 1 branches (DERIVT_X and DERIVT_Y at :4422-4423 and :4429-4430, and the
DSUPICE_XYZ differences at :4460-4462, :4468-4469, :4483-4485 and
:4491-4492). A short low-edge run records them: the same domain with the
tile moved to i, j = 1..3 and a second Hallett-Mossop column in the corner
(1, 1), because the DSUPICE_XYZ branch (:4441) needs 2 percent ice
supersaturation below -5 C at the point itself, which the environment
columns never reach; 6 calls, with replay records at steps 2, 4 and 6 and
the tile's pre-pass arrays from the probe build (`kc2edge` in the manifest).

## Findings recorded by the cold oracle

* ICE_NUCL's `STOP 099` (:3463) checks the change of a running real(8) sum
  over all bins against 10 per cm3, while nucleation adds at most HELEK1 =
  1.106 per cm3. With nonnegative bins it cannot fire. Finite bins of mixed
  sign reach it through rounding in the sum (refusal case 13, both builds),
  and a non-finite change reaches it too (refusal case 1, a subnormal grid
  mass, the promoted build only; the native build returns). FAST_SBM cannot
  reach it: it passes JERNUCL01_KS no crystals, because FF2R is zeroed at
  every point (:4548-4550) and after every nucleation call (:4667-4674).
* ONECOND2's two negative-substep checks are both refusal cases: :7452 on
  the KCOND 21 path (case 4) and :7535 on the KCOND 20 path (case 12).
* modkrn_KS CASE 11 leaves its output unset at exactly -40 C (INTERPOL_SE
  assigns nothing at its last node), as above 0 C (D6).
* A negative bin with growth is dropped by JERNEWF_KS's remap without a
  refusal; JERDFUN_KS's own negative check can never fire.
* The deldrop check (:8598) is reached only through coll_xxx_lwf's gmin floor
  on very dilute drops over a very long step.
* The snow-breakup guard is true at every point (it sums an empty slice), and
  BreakUp_Snow is never called.
* ONECOND2's sign checks cannot fire; JERSUPSAT_KS's branches 8 and 9 (S1)
  cannot be reached from ONECOND3 with nonnegative spectra; S2 (branch 7) is.
* In the graupel build the drop-hail calls never run; the drop-graupel-to-hail
  calls need 100 g/m3 of liquid.

The text of each, with its evidence, is in the manifest under `findings_cold`.

## What milestone 2 took from the earlier port

Read only (an earlier, uncommitted machine transcription of the scheme and its
sort-out notes); none of its code is here.

* The call shapes and argument layouts of its Fortran bindings for FREEZ,
  J_W_MELT, ICE_NUCL, JERNUCL01_KS with ICEPROCS = 1, coll_xyx/xyy/xyz_lwf and
  modkrn_KS, re-expressed as `run_sweeps.F90` families in the access-transformed
  module, so both precisions run them. The seven-species layout of its phase
  binding (liquid, three habits, snow, graupel, hail) is kept.
* Its 143 live pinned cases, with their grids, on the real tables: FREEZ and
  J_W_MELT at 5 temperatures x 4 time steps; ICE_NUCL at the 9 habit and
  Meyers boundaries x 2 SUP2_OLD; JERNUCL01_KS at 8 temperatures x 3
  supersaturation pairs x cloud base; the 24 live collision cases with a
  non-symmetric kernel, three ways: on their synthetic grid with y = 1.25 x
  (where an orientation error shows), on the synthetic y = x, and on the real
  tables (x = y = xl_mg with the module's chucm and ima); modkrn CASE 11 at 12;
  and the dt = 200 s melt case.
  Its draws came from numpy; here a Park-Miller generator started at the old
  seeds draws them, and every draw is in the records.
* Its record of which ice routines need which FAST_HUCMINIT tables, and its
  generated core as a map of the ice call structure; its note that J_W_MELT
  declares the mass grids INTENT(INOUT) and never writes them (the melt sweep
  checks the grids come back unchanged).
* Not taken: the C binding, the generated C++, the CUDA kernels, its 43-bin
  cases, coll_xxy_lwf and modkrn CASE 1 (dead on this path), and its
  bit-identity results.

## Reproduction and evidence

The same harness under Ubuntu 24.04 with gfortran 13.3.0 produced all 54
raw streams byte-identical to the reference host's gfortran 15.2.0, including the
production-flag evidence build; the milestone 2 harness, with the low-edge run
and the two added refusal cases, produced all 134 of its raw streams
byte-identical too (`reproduction` in the manifest). On KC-1 the
O32-wrf build differs from the O32 build
from step 1, in cloud water, rain water and drop number where those are tiny
(largest absolute differences over the run 6.5e-19 and 5.4e-20 kg/kg and
3.7e-9 per kg); theta, qv and surface rain are identical after one hour. On
KC-2 it does not give the O32 state byte for byte either; its surface
precipitation first differs at step 161 (`evidence_o32_wrf_kc2`). Neither
comparison is gated.

## Licence and attribution

WRF is developed at NCAR, operated by UCAR, and is in the public domain; this
harness compiles the unmodified WRF source (plus the two recorded transforms)
and transcribes nothing into the model. The scheme is the Hebrew University
Cloud Model spectral bin microphysics formulated by Alexander Khain; FSBM-2 in
WRF was coded and is maintained by Jacob Shpund, and WRF asks users to cite
Shpund et al. (2019). The tables are NCAR's `SBM_input_33` distribution.
