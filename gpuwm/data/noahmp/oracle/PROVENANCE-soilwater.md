# Noah-MP soil-water oracle -- provenance

Leaves: `CANWATER`, `INFIL`, `SRT`, `SSTEP` and their driver `SOILWATER`.
Plan steps 13 and 15 of `docs/mynn_noahmp_ruc_completion_plan.md`.

## Source identity

| Item | Value |
|---|---|
| WRF tree | `<wrf-4.6.1-checkout>` (WSL) |
| commit | `d66e442fccc04111067e29274c9f9eaccc3cef28` |
| `phys/module_sf_noahmplsm.F` sha256 | `bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282` |
| compiler | GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| accessibility lift | `tools/noahmp_wrf461_oracle/visibility_patch_leaves.py` (50 symbols, padded `public  ::` spelling) |
| object-code proof | 85 of 85 module procedures byte-identical, `.data`/`.data.rel.local`/`.rodata` identical |

WRF line ranges: `CANWATER` 6265-6394, `SOILWATER` 7234-7556, `INFIL`
7616-7712, `SRT` 7716-7846, `SSTEP` 7850-7973.  `WDFCND1` 9153-9188 and
`WDFCND2` 9192-9232 were already pinned by the leaf harness and are imported,
not re-transcribed.

## Pinned option identity

```
dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
```

`run_soilwater.F90` calls `noahmp_options` with those values and then asserts
`OPT_RUN==3`, `OPT_INF==1`, `OPT_TDRN==0`, `OPT_IRR==0`, `OPT_INFDV==0`,
`OPT_CROP==0` and `DVEG==4` against the module's own variables, so the identity
is checked rather than assumed.

### Asserted off, not ported

| Killed by | Routines / blocks |
|---|---|
| `opt_run=3` | `GROUNDWATER`, `SHALLOWWATERTABLE`, `ZWTEQ`, `COMPUTE_VIC_SURFRUNOFF`, `COMPUTE_XAJ_SURFRUNOFF`, `DYNAMIC_VIC`; `SOILWATER`'s OPT_RUN 1/2/4/5/6/7/8 blocks (7350-7419, 7442-7462, 7531-7541); `SRT`'s OPT_RUN 1/2/4/5 drainage forms (7801-7818); `SSTEP`'s water-table block (7923-7947) |
| `opt_inf=1` | `SRT`'s `WDFCND2` loop (7781-7787).  `WDFCND2` itself stays live through `INFIL` at 7703 |
| `opt_tdrn=0` | `TILE_DRAIN`, `TILE_HOOGHOUDT` (7521-7529) |
| `opt_irr=0` | the five irrigation routines; none is called from this group |

## Fixture

| File | sha256 | rows |
|---|---|---|
| `noahmp-soilwater.csv` | `17cb1be26852fc025d5b3e3fe904a2b16cc3e3699233ace43d0292805266e1a0` | 4540 |
| `noahmp-soilwater-libm.csv` | `68b4c7d89ad9e69e6b9bf167cf0f1cae9003d5b1c5fefd69dfb506b1c7c6bead` | 783 |

Cases: `canwater` 21, `infil` 10, `srt` 8, `sstep` 10, `soilwater` 13.

Built by `bash tools/noahmp_wrf461_oracle/build_soilwater.sh <wrf-tree> <work>`
and validated by `validate_soilwater_oracle.py` before it is allowed out of the
build directory.

## Two findings that shaped the build

### 1. libmvec -- why this fixture is built at `-O0`

At WRF's own `FCOPTIM = -O2 -ftree-vectorize -funroll-loops`, gfortran 13.3.0
vectorises `SOILWATER`'s frozen-fraction loop (7333-7337) and emits a call to
glibc's **libmvec** `_ZGVbN4v_expf` in place of scalar `expf`.  `nm -u` on the
compiled module shows that is the *only* libmvec reference anywhere in
`module_sf_noahmplsm.o`, and `objdump -dr` places the single relocation inside
`__module_sf_noahmplsm_MOD_soilwater`.

libmvec's 4-wide `expf` is a different function from scalar `expf`: it carries a
4-ULP accuracy contract rather than glibc's scalar implementation, so no port
that calls `expf` can reproduce it.  Measured against the `-O0` fixture the
divergence is **2 rows, 1 ULP each**, both in `slw_frozen`:

```
soilwater,slw_frozen,output,qdrain,0   -O0 348BC044   -O2 348BC043   1 ulp
soilwater,slw_frozen,output,fcrmax,0   -O0 3D1A89FA   -O2 3D1A89F9   1 ulp
```

The `-O0` build emits scalar `expf`, which is the routine's own arithmetic, so
that is what the fixture pins and what the port is held to at `max_ulp 0`.  The
`wrf` optlevel remains available in `build_soilwater.sh` as a **recorded
divergence**, not a gate.  Nothing was widened to make it match.

### 2. INTENT(OUT) scalars that live paths leave unassigned

Two of them, both real and both pinned rather than assumed:

* `SOILWATER`'s `RUNSUB` is declared `INTENT(OUT)` at 7280, but under
  `OPT_RUN==3` the only statement that touches it is
  `RUNSUB = RUNSUB - XS/DT` at 7549, which **reads it before it is ever
  assigned**.  gfortran passes scalar dummies by reference, so it behaves as
  `INOUT`.  `WATER` always enters with `RUNSUB = 0.0` (6109), so the forecast
  path is well defined.  The fixture drives it non-zero in four cases
  (`slw_moderate_rain` 7.5, `slw_runsub_alias` -3.25, `slw_inert_probe` 7.5,
  `slw_watmin_fixup` 2.0) so the aliasing is measured; `slw_watmin_fixup` is
  the only case where `XS` is non-zero, and there `RUNSUB` goes 2.0 -> 1.99055.

* `INFIL` declares `PDDUM` and `RUNSRF` `INTENT(OUT)` (7638-7639) but assigns
  them only inside `IF (QINSUR > 0.0)` at 7655.  `infil_qinsur_zero` enters
  with -999.0 / -499.5 and gets them back unchanged, which is the pinned
  behaviour.  `SOILWATER` zeroes both at 7318-7319 before its first call.

The second of these bit the harness itself: at `-O2`, with `INFIL`'s explicit
interface in scope, gfortran removed the harness's own `ppddum = 0.0` store as
dead before an `INTENT(OUT)` call, and the previous case's stack value stood.
`run_soilwater.F90` now reads both variables into the CSV before the call,
which forces the stores to be live.  `-finit-real=snan` did **not** catch this,
because the compiler eliminated the `snan` initialisation on the same grounds;
the leak was found by noticing that two cases with different `QINSUR` emitted
the same probe value.

## Negative controls

| Control | Result |
|---|---|
| `-O0` vs `-O0 -ffp-contract=off` | byte-identical CSV |
| `-O0` vs `-O0 -finit-real=snan -finit-integer=-2147483647 -finit-logical=false` | byte-identical CSV |
| `-O0` vs WRF `FCOPTIM` | 2 rows differ, 1 ULP each; cause identified above |
| `visibility_patch_leaves.py --self-test` | passes (renamed symbol, edited body, dropped line and width change all rejected) |
| object-code equivalence, pristine vs patched | PASS over 85 procedures |

## Inertness, measured rather than claimed

Four cases exist only to make the dead-argument claim a row in the CSV.  Each
perturbs every argument the pinned identity does not consume and must reproduce
its baseline bit for bit on every output that is not a declared pass-through:

| Probe | Baseline | Arguments perturbed | Outputs held bit-identical |
|---|---|---|---|
| `canw_inert_probe` | `canw_unfrozen_evap` | VEGTYP, TG | 13 |
| `srt_inert_probe` | `srt_wet_baseline` | DT, ILOC, JLOC, SH2O, ZWT, SICEMAX, FCRMAX, SMCWTD | 21 |
| `sstep_inert_probe` | `sstep_baseline` | ZSOIL, ZWT, SMC, ILOC, JLOC, DZSNSO(-2:0), SMCWTD, QDRAIN, DEEPRECH | 28 |
| `slw_inert_probe` | `slw_moderate_rain` | ILOC, JLOC, VEGTYP, DX, TDFRACMP, ZWT, SMCWTD, DEEPRECH, QTLDRN | 21 |

`ILOC`/`JLOC` are compile-time constants in `case_canwater`, so `canw_inert_probe`
perturbs only the two it can.

## Parameter sets

The base set varies every per-layer parameter with layer, so a port that
indexes the wrong layer cannot reproduce the fixture.  `TIMEAN` and `FSATMX`
carry non-zero values although both are dead under `opt_run=3`, so their
inertness is measured rather than vacuous.

Two variants exist and each is used for exactly one reason:

* a coarse, highly conductive set (WRF SOILPARM's sand row) for
  `slw_heavy_rain_niter6` and `slw_coarse_niter3`.  With the base set `INFIL`'s
  `INFMAX` is capped by `DD*VAL/DT`, an order of magnitude below
  `DZSNSO(1)*SMCMAX(1)/DT` for every admissible `QINSUR`, so `SOILWATER`'s
  `NITER = NITER*2` at 7443 is unreachable under the base set alone.
* `FRZX = 6.5e-3` for `infil_dice_at_limit` / `infil_dice_just_above_limit`.
  At `DICE` one ULP above the `> 1.0E-2` test the base `FRZX = 0.1519` gives
  `ACRT = 45.6`, so `EXP(-ACRT)*SUM` underflows and `FCR` is exactly 1.0 on
  both sides -- the boundary is real but numerically inert there.  The
  physical-FRZX cases `infil_frozen` (DICE 0.105) and `infil_heavy_ice`
  (DICE 0.409) exercise the same block without any parameter change.

## The NITER probe

`SOILWATER`'s iteration count is a local, so branch coverage of
`PDDUM*DT > DZSNSO(1)*SMCMAX(1)` cannot be read off the outputs.  Each
`soilwater` case therefore carries a `probe` stage that re-derives it from the
inputs alone: the RSAT clamp (7324-7329) and `SICEMAX` (7341-7347) are MIN/MAX
over the inputs, and `PDDUM` then comes from **the module's own `INFIL`** -- no
physics is duplicated in the harness.

## Mutation study

`tools/noahmp_wrf461_oracle/mutation_study_soilwater.py` generates 94 mutants
against `gpuwm/core/noahmp_soilwater.py` -- one per argument each routine
consumes and per `noahmp_parameters` component it reads (43), plus one per
non-zero `_f(<literal>)` site perturbed by a relative 1e-3 (51) -- and runs each
through `tests/test_noahmp_soilwater.py`.

**94 of 94 killed, no survivors.**

Three of the five cases added in the second round exist only because of it, and
each closes a hole the first fixture genuinely had:

* `canw_fwet_floor_ice_melt` and `canw_fwet_floor_liq_freeze`.  The `1.0E-06`
  floors at 6360/6362 are unobservable in any ordinary case: the zeroing tests
  at 6346/6355 use the *same* constant, so whenever the floor is selected the
  numerator is either exactly 0 or strictly larger than the floor and
  `MIN(FWET,1.0)` erases the difference.  Putting CANLIQ/CANICE exactly one ULP
  above the zeroing threshold with `LSAI = 0` makes FWET land at 1.0000001,
  one ULP clear of the MIN -- and simultaneously puts the value exactly on the
  melt/freeze tests at 6372/6379, so `TV` collapses to `TFRZ` if and only if
  all four constants are what WRF says they are.
* `sstep_epore_floor_upward_only`.  Every other saturating case runs both
  cascades, and the downward pass at 7963-7969 re-clamps with its own copy of
  the `1.0E-4` floor, so the upward loop's copy at 7952 never reaches an
  output.  Holding layer 1 under its own EPORE keeps the downward pass off.

One mutant was found to be *defective* rather than surviving: overwriting
`sstep`'s `dzsnso` argument after the anchor is a no-op, because `dz` has
already been extracted from it.  The mutant now targets `dz`, and is killed.

Two sensitivities are recorded because they are real and bounded, not because
they fail:

* `WATMIN` (7551) is killed at the study's relative 1e-3, but an absolute
  perturbation below ~1e-6 is invisible.  In `slw_watmin_fixup` the bottom
  layer arrives at `MLIQ = -17.0`, so `XS = WATMIN - MLIQ` is ~17.01 and
  `WATMIN`'s low bits are lost to the FP32 exponent before the addition.  That
  is a property of the routine, not of the fixture.
* `_A = 4.0` (7317) is discriminated by exactly one case.  `FCR` is
  identically 0 wherever `FICE = 0` and identically 1 wherever `FICE = 1`, for
  any `A`, so only a partially frozen column can see it; `slw_frozen` is that
  column.

## Device parity

`gpuwm/core/kernels/noahmp_soilwater.cu` was gated on the rented RTX 5090
(sm_120, CUDA 12.8, nvcc V12.8.93, CuPy 14.1.1, driver runtime 12090) against
this same fixture, not against the CPU transcription -- so a shared mistake in
the two ports cannot pass.  `tests/test_noahmp_soilwater_cuda.py`:
**8 passed**, `max_ulp 0` on all five kernels plus both libm sweeps, with no
tolerance applied anywhere.  Peak device memory 17 MiB.

Two device-only defects the fixture caught, both of which would have been
invisible to a tolerance-based gate:

* `K_0P667` was transcribed as `0x3F2AAAAB`, which is 2/3, not `0.667`
  (`0x3F2AC083`).  The error is 5e-4 relative -- far inside any sane
  "close enough" band, and it moved `FWET` and therefore `TV` on nine cases.
* `glibc_powf(+0, y)` returned NaN, because the transcription's domain guard
  rejected a zero base.  glibc returns `+0`, and CANWATER reaches it on every
  dry-canopy case.  The guard now handles exactly that sub-case and still
  refuses the rest of the domain rather than returning a value it cannot
  vouch for.

The kernel does **not** reproduce gfortran's `-O2` vectorisation of the
frozen-fraction loop; it calls the scalar `glibc_expf` transcription once per
layer, which is what the `-O0` fixture pins.

`test_the_device_gate_can_fail` perturbs `K_A` from `4.0` to `4.001` -- the same
relative 1e-3 the CPU mutation study uses, because a one-ULP nudge is quantised
away by `FCRMAX` -- and requires the comparison to reject it.
