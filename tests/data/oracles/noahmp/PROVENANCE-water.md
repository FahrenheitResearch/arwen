# Noah-MP WATER oracle -- provenance

Leaf: `WATER`, the whole-column water assembly.  Plan step 16 of
`docs/mynn_noahmp_ruc_completion_plan.md`.

`WATER` composes leaves that are already pinned bitwise: `CANWATER`,
`SOILWATER`, `INFIL`, `SRT`, `SSTEP` (`gpuwm/core/noahmp_soilwater.py`), the
snow chain `SNOWFALL`/`COMPACT`/`COMBINE`/`DIVIDE`/`COMBO`/`SNOWH2O`/
`SNOWWATER` (`gpuwm/core/noahmp_snow.py`) and `WDFCND1`/`WDFCND2`/`ROSR12`
(`gpuwm/core/noahmp_leaves.py`).  All of them are imported; nothing is
re-transcribed, and `gpuwm/core/noahmp_water.py` evaluates no transcendental
of its own.

## Source identity

| Item | Value |
|---|---|
| WRF tree | `<wrf-4.6.1-checkout>` (WSL) |
| commit | `d66e442fccc04111067e29274c9f9eaccc3cef28` |
| `phys/module_sf_noahmplsm.F` sha256 | `bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282` |
| compiler | GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| build flags | `-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O0` |
| accessibility lift | `tools/noahmp_wrf461_oracle/visibility_patch_leaves.py` (50 symbols, padded `public  ::` spelling) |
| object-code proof | 85 of 85 module procedures byte-identical, `.data`/`.data.rel.local`/`.rodata` identical |

WRF line range: `WATER` 5954-6261.  Its executable body is 5954+153 .. 6259;
every line number in the harness, the port and this file refers to that file at
that sha256.

## Pinned option identity

```
dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
soiltstep=0.0
```

`run_water.F90` calls `noahmp_options` with the first twenty and then asserts
`OPT_RUN==3`, `OPT_INF==1`, `OPT_TDRN==0`, `OPT_IRR==0`, `OPT_INFDV==0`,
`OPT_CROP==0` and `DVEG==4` against the module's own variables, so the identity
is checked rather than assumed.

`soiltstep` is new to this lane.  It is not an option flag: `soil_update_steps`
and `calculate_soil` are bare module variables that `module_sf_noahmpdrv.F`
(drivers/wrf, 648-676) sets from it.  With `soiltstep = 0.0`,
`soil_update_steps = max(nint(0/DT), 1) = 1` and, since `mod(itimestep,1) == 0`
always, `calculate_soil = .true.` on every step.  The harness sets both and
asserts both.  Under that value `DT_soil == DT` (6185) and the three divisions
at 6204-6206 are the identity, which the port writes as such -- `x / 1.0` is
`x` for every finite binary32 `x`, so this is exact and not an approximation.

### Asserted off, not ported

| Killed by | Routines / blocks |
|---|---|
| `opt_run=3` | `GROUNDWATER` (6225-6231) and `SHALLOWWATERTABLE` (6242-6250), and through them `ZWTEQ`, `COMPUTE_VIC_SURFRUNOFF`, `COMPUTE_XAJ_SURFRUNOFF` and `DYNAMIC_VIC`.  `RUNSUB = RUNSUB + QDRAIN` (6233-6236) is the only live baseflow form. |
| `opt_irr=0` | `FLOOD_IRRIGATION` (6188-6193) and `MICRO_IRRIGATION` (6196-6202).  See below -- the kill runs through the caller, not through a gate in WATER. |
| `opt_tdrn=0` | the tile drain inside `SOILWATER`; `QTLDRN` is zeroed at 6112 and scaled by `DT_soil` at 6254, so it is identically `0.0`. |
| `WRF_HYDRO` undefined | `sfcheadrt`/`WATBLED` do not exist, so the `QINSUR` term at 6174 is absent and `SOILWATER`'s `WATBLED` argument is not passed. |

**The irrigation kill is indirect and worth stating precisely, because WATER's
own gates do not mention `OPT_IRR`.**  6188 and 6196 read `IRAMTFI` and
`IRAMTMI`.  Under `opt_irr=0`, `TRIGGER_IRRIGATION` takes its
`(OPT_IRR .GT. 3) .OR. (OPT_IRR .LT. 1)` branch at 9291, sets
`IRR_ACTIVE = .false.` and zeroes `IRAMTSI`/`IRAMTMI`/`IRAMTFI` at 9344-9346;
`NOAHMP_SFLX` additionally zeroes all three at 919-923 whenever
`IRRFRA < parameters%IRR_FRAC`, and the Registry default `IRFRACT = 0` makes
that unconditional.  WATER therefore cannot be reached with either amount above
zero.  The fixture measures the consequence rather than asserting it: both
inert probes set `CROPLU = .true.` and perturb `IRRFRA`, `MIFAC`, `FIFAC`,
`IRFIRATE` and `IRMIRATE`, and every output stays bit-identical.

## Fixture

| File | sha256 | rows |
|---|---|---|
| `noahmp-water.csv` | `07fcd12e4e45571e3d8735907e322a35fa86eaf038fccd019723986ec2af534d` | 11448 |
| `noahmp-water-libm.csv` | `a7960673073979528b41d855a55c0ce75d6d7d18908dd73fdcc6506963796913` | 1040 |

36 cases.  Built by
`bash tools/noahmp_wrf461_oracle/build_water.sh <wrf-tree> <work>` and validated
by `validate_water_oracle.py` before it is allowed out of the build directory.

## Three findings that shaped the build

### 1. `nm -u` is now a build stage, not a comment

The soil-water lane found that at WRF's own
`FCOPTIM = -O2 -ftree-vectorize -funroll-loops`, gfortran 13.3.0 vectorises
`SOILWATER`'s frozen-fraction loop (7333-7337) into glibc's **libmvec**
`_ZGVbN4v_expf`, a different function from scalar `expf` with a 4-ULP contract
rather than glibc's scalar implementation.  WATER calls `SOILWATER`, so the
trap applies here directly.

Stage 5 of `build_water.sh` runs `nm -u` on the compiled module and the harness
and **exits non-zero if any `_ZGV*` symbol is present** in a build that claims
scalar libm.  That is the check which would have caught the trap the first
time, and it makes the `-O0` choice falsifiable instead of inherited.  At
`optlevel=wrf` the same stage prints the symbol instead of failing:

```
[5] libmvec references at WRF FCOPTIM (recorded, not gated):
      _ZGVbN4v_expf
```

so the fixture build and the divergence build both *state* what libm they got.

### 2. At WRF FCOPTIM this fixture happens not to diverge -- which changes nothing

Measured, `-O2 -ftree-vectorize -funroll-loops` reproduces the `-O0` fixture
**byte for byte, all 11448 rows**.  That is not a licence to build at `-O2`.
`_ZGVbN4v_expf` *is* linked into that build; it simply agrees with scalar
`expf` on the `-A*(1-FICE)` arguments these 36 columns produce.  The soil-water
fixture, over a different set of columns, disagrees on two of them by 1 ULP
(`PROVENANCE-soilwater.md`).  The divergence is input-dependent, so the correct
statement is "the libmvec exposure exists and this case set does not reach it",
and the fixture stays at `-O0` where the routine's own scalar arithmetic runs.

### 3. A third INTENT(OUT) hazard: `QIN` and `QDIS` are never written at all

The soil-water lane pinned two INTENT(OUT) arguments that WRF leaves unassigned
on *some* paths.  WATER has one that is unassigned on *every* path under this
identity.  `QIN` (6052) and `QDIS` (6053) are `INTENT(OUT)`, and the only
statement that assigns either is inside the `OPT_RUN==1` `GROUNDWATER` call at
6226-6229.  gfortran passes scalar dummies by reference, so the caller's value
stands unchanged.

That is not a latent bug in WRF: `NOAHMP_SFLX` declares both as uninitialised
locals at 671-672, passes them to WATER, and never reads either afterwards.
Nothing downstream consumes them.

The port therefore does not return them.  What makes that omission checkable
rather than convenient is that the fixture drives both to distinct non-zero
values on every case, `validate_water_oracle.py` requires entry == exit on all
36, and `wat_out_entry_probe`/`wat_snow_out_entry_probe` enter with *different*
values again and still come back unchanged.  If a future option identity ever
made `GROUNDWATER` live, `test_qin_qdis_are_never_written` fails before the
omission becomes a silent wrong answer.

## Negative controls

| Control | Result |
|---|---|
| `-O0` vs `-O0 -ffp-contract=off` | byte-identical CSV |
| `-O0` vs `-O0 -finit-real=snan -finit-integer=-2147483647 -finit-logical=false` | byte-identical CSV |
| `-O0` vs WRF `FCOPTIM` | byte-identical CSV; `_ZGVbN4v_expf` present but not reached differently (see finding 2) |
| `nm -u` on the `-O0` build | no `_ZGV*` symbol; `expf`, `powf`, `logf`, `log10f` are the scalar glibc entries |
| `validate_water_oracle.py` on three corrupted copies of the fixture | rejects all three: a moved pass-through column, a decimal that no longer round-trips to its bit pattern, and a non-zero `QTLDRN` (`test_the_validator_can_fail`) |
| `visibility_patch_leaves.py --self-test` | passes (renamed symbol, edited body, dropped line and width change all rejected) |
| object-code equivalence, pristine vs patched | PASS over 85 procedures; `.data`, `.data.rel.local` and `.rodata` identical |

The `snan` control is load-bearing here rather than decorative.  WATER's
`QIN`/`QDIS` are read-never-written scalars, `SNOWWATER`'s `DIVIDE` leaves the
slots above `ISNOW` conditionally assigned, and `COMBINE` leaves
`PONDING1`/`PONDING2` untouched on some paths.  A byte-identical CSV under
`-finit-real=snan` is the statement that no emitted value depends on any of
them.

## Inertness, measured rather than claimed

Two probe cases exist only to make the dead-argument claim a row in the CSV.
Each perturbs every argument the pinned identity does not consume and must
reproduce its baseline bit for bit on every output.

| Probe | Baseline | Arguments perturbed | Outputs held bit-identical |
|---|---|---|---|
| `wat_inert_probe` | `wat_bare_rain` | 31 names: UU, VV, QPRECC, QPRECL, FP, RAIN, SNOW, LATHEAV, LATHEAG, IRRFRA, SMCEQ, WA, WT, RECH, MIFAC, FIFAC, IRAMTFI, IRAMTMI, IRFIRATE, IRMIRATE, CROPLU, VEGTYP, TG, ILOC, JLOC, DX, TDFRACMP, ZWT, SMCWTD, DEEPRECH, ZSNSO | all |
| `wat_lake_inert_probe` | `wat_lake_filling` | the same 31, on the `IST == 2` branch where a different set of statements is live | all |

Beyond the probes, `validate_water_oracle.py` requires every one of those
arguments to come back bit-identical on **all 36 cases**, not only in the
probes, which is a stronger statement than a single paired comparison.

Three of those entries need their reason stated rather than assumed:

* **`ZSNSO` is write-only.**  `SNOWWATER` rebuilds every slot at 6522-6533 from
  `ISNOW`, `DZSNSO` and `ZSOIL` and reads no entry value.  The probes drive all
  seven slots to unrelated values.  The port's mutation study reports the
  matching argument-drop mutant as an *expected* survivor for the same reason.
* **`IRAMTFI`/`IRAMTMI` cannot be positive** under `opt_irr=0`, so the only
  non-zero value the fixture is allowed to give them is a negative one.  The
  probes use `-0.5` and `-0.25`.  That is not a forecast state and is not
  claimed to be one: it measures that the 6188/6196 gates are `.GT. 0.0` and
  not `.NE. 0.0`, which is the only part of them a fixture can reach at all.
  `validate_water_oracle.py` records this in `NEGATIVE_ONLY` so the choice is
  visible rather than buried.
* **`QTLDRN`'s entry value is dead**, not inert: 6112 overwrites it.  The
  probes drive it to 4.25 and it comes back `0.0` like every other case.

## Branch coverage

Twenty-two branches, each with a predicate over the *input* and *probe* columns
only -- never over an output, so coverage cannot be satisfied by coincidence.
Every one is taken by at least one case and not taken by at least one case.
The full list is printed by the validator; the ones that needed a case built
specifically for them are:

* `MIN(QVAP, SNEQV/DT)` picking the pack (6128).  For a layered pack `SNEQV/DT`
  is orders of magnitude above any `QVAP` a surface energy balance produces, so
  the physically reachable shape of that branch is a *trace* pack:
  `wat_snow_sublimation_masslimited` uses `SNEQV = 0.01` at `DT = 1800`.
* `SNOWFALL` creating the first layer (6570-6577), which is the only statement
  in the whole assembly that reads `SFCTMP`.  `wat_new_snow_layer` crosses
  `SNOWH >= 0.025` inside the call.
* `COMPACT`'s melt term (7052-7056), the only reader of `FICEOLD` and the only
  consumer of `IMELT`.  It is inert unless `FICEOLD` exceeds the current
  `SNICE/(SNICE+SNLIQ)`; every other pack in this fixture is drier than it was,
  so the `MAX` picks zero.  `wat_melting_pack_ficeold` is wetter.
* `SNOFLOW` non-zero (6259).  Needs `SNEQV > 5000` inside `SNOWWATER`, so
  `wat_glacier_snoflow` carries a 5400 mm pack, and `wat_glacier_short_step`
  repeats it at `DT = 60` so the `SNOFLOW*DT` factor is discriminated from
  `SNOFLOW`.
* `WSLAKE >= WSLMAX` (6211).  The test is `>=`, so `wat_lake_at_wslmax` sits
  exactly on 5000.0.
* the accumulator aliasing (6178-6180 with 6204-6206).  `wat_acc_nonzero`
  enters with `ACC_QINSUR`, `ACC_QSEVA` and `ACC_ETRANI` all non-zero, which is
  the only way to tell a port that adds to them from one that overwrites them.
* `NROOT < NSOIL` (6169).  `wat_nroot_full` and `wat_nroot_short` carry the
  *same* non-zero `BTRANI(4)` at `NROOT = 4` and `NROOT = 3`, so the loop bound
  is observable rather than masked by a zero.

## The probe stage

Four of WATER's branches are decided by locals that never reach an output:
`QSNSUB`/`QSNFRO`'s `SNEQV > 0.0` gates, the `MIN` inside `QSNSUB`, and
`SNOWWATER`'s glacier `SNOFLOW`.  The `probe` stage makes each assertable from
the inputs.

The first three are re-derived by `MIN` and subtraction over the emitted input
columns; `validate_water_oracle.py` re-derives them a second time and requires
agreement, so the probe cannot drift from the entry state it describes.
`SNOFLOW`, `QSNBOT` and `PONDING1`/`PONDING2` come from running **the module's
own `SNOWWATER`** on a copy of the entry state -- no physics is duplicated in
the harness.

The probe stage is never the gate.  The port is held to the `output` stage, so
a probe that agreed with a wrong port could not hide anything; what it buys is
that a compensating pair of errors either side of `SNOWWATER` is caught, which
the composed output alone would not do.

`run_case` reads `ponding1`/`ponding2` into the CSV *before* the probe
`SNOWWATER` call.  That is load-bearing, not diagnostic: `COMBINE` declares both
`INTENT(OUT)` and leaves them untouched on some paths, so with the explicit
interface in scope the compiler may treat the zeroing stores as dead.  The same
hazard bit the soil-water harness (`PROVENANCE-soilwater.md`, finding 2).

## Mutation study

`tools/noahmp_wrf461_oracle/mutation_study_water.py` generates 88 mutants
against `gpuwm/core/noahmp_water.py` and runs each through
`tests/test_noahmp_water.py`:

* **argument mutants** (34) -- one per argument `water()` consumes, per
  `SnowColumn` field it forwards, and per `WaterParameters` component it reads.
* **structure mutants** (33) -- WATER's own statements, each broken in the one
  way a careless transcription would break it: a dropped term, a flipped
  comparison, a reordered sum, a loop bound taken from the wrong variable, the
  `DT_soil` scaling hoisted out of the soil branch, `QSNSUB`/`QSNFRO` swapped in
  the `SNOWWATER` call.  A composition needs these and a leaf does not, because
  a driver's whole job is the plumbing between calls.
* **constant mutants** (3) -- every non-zero `_f(<literal>)` site, perturbed by
  a relative 1e-3.  There are only three because WATER carries almost no
  constants of its own: `1000.0`, `0.001` and `WSLMAX`.

**86 of 88 killed.  Two survivors, both argued unreachable and recorded in the
study's own `EXPECTED_SURVIVORS` table so that a survivor which is *not* on
that list fails the run:**

* `arg/col.zsnso` -- `ZSNSO` is write-only, as above.
* `struct/6127 gate becomes >=` -- `QVAP` is `MAX(FGEV/LATHEAG, 0.0)` at 982
  and is therefore never negative, so with `SNEQV == 0` the mutated gate
  computes `MIN(QVAP, 0.0/DT) = 0.0`, which is exactly what 6126 already
  assigned.  The two forms agree on every state `NOAHMP_SFLX` can produce.
  The 6133 gate is **not** equivalent under the same change -- `QSNFRO` would
  take `QDEW` and `QSDEW` would lose it -- and the corresponding mutant is
  killed, which is what stops this from being a general excuse.

Two cases in the fixture exist only because of this study: `wat_new_snow_layer`
and `wat_melting_pack_ficeold` closed `arg/sfctmp`, `arg/ficeold` and
`arg/imelt`, which survived the first round.

## Cross-check against the byte-unmodified whole-column oracle

`noahmp-water.csv` is built from a visibility-patched copy of the module.  The
patch is proven inert at the object-code level over all 85 procedures, but
"proven inert" and "measured inert on the values this port is held to" are
different claims.  `noahmp-sflx.csv` is the second one: `run_sflx.F90` compiles
the **pristine, unpatched** module, and its `output` rows are every
`INTENT(OUT)` and `INTENT(INOUT)` argument of `NOAHMP_SFLX`, which includes most
of WATER's.

WATER's *inputs* are not reconstructible from that file -- `FCEV`, `QVAP`,
`SNOWHIN`, `IMELT` and `BDFALL` are ENERGY's and PRECIP_HEAT's internal results
and `NOAHMP_SFLX` never emits them -- so this is not a replay.  What is
reconstructible is the set of identities WATER's own statements impose among
the columns the whole-column fixture *does* emit, and those cover every
statement of WATER that touches a scalar flux:

| WRF statement | identity | result |
|---|---|---|
| 984 `EDIR = QVAP - QDEW` | `QVAP = MAX(EDIR,0)`, `QDEW = -MIN(EDIR,0)` -- exactly one of the pair is non-zero at 982-983, so `EDIR` recovers both without knowing which latent-heat pathway ENERGY chose | holds on 4/4 |
| 6126-6128 | `QSNSUB` within `[0, QVAP]` under the `SNEQV > 0` gate | holds on 4/4 |
| 6130, 6147-6148, 6167, 6179 | `ACC_QSEVA = (QVAP - QSNSUB) * 0.001`, or `0.0` on the FROZEN_GROUND branch, with the fork taken from the emitted `TG` and then checked | bit-identical on 4/4 |
| 6132-6136 | `QSDEW = QDEW - QSNFRO`, `QSNFRO` either `0` or `QDEW` | holds on 4/4 |
| 6159-6164, 6178 | `ACC_QINSUR` from `PONDING`, `PONDING1`, `PONDING2`, `QSNBOT`, `QSDEW`, `QRAIN` and `ISNOW` | bit-identical on 4/4 |
| 6392 | `ECAN = QEVAC + QSUBC - QDEWC - QFROC`, left-associative | bit-identical on 4/4 |

`tests/test_noahmp_water_sflx_crosscheck.py` evaluates all six in float32 in the
port's own association order and byte-pins `noahmp-sflx.csv` first, so a
regenerated whole-column fixture cannot make the cross-check vacuous.  The four
sflx columns include no lake and no glacier, so 6209-6212 and 6259 are held only
by `tests/test_noahmp_water.py`; that limit is stated in the test's docstring.

## Device parity

`gpuwm/core/kernels/noahmp_water.cu` was gated on the rented RTX 5090 (sm_120,
CUDA 12.8, nvcc V12.8.93, CuPy 14.1.1, driver 12080 / runtime 12090) against
this same fixture, not against the CPU transcription -- so a shared mistake in
the two ports cannot pass.  `tests/test_noahmp_water_cuda.py`: **8 passed**,
`max_ulp 0` on all 36 cases across the whole column -- ISNOW, SNOWH, SNEQV,
SNICE, SNLIQ, STC, ZSNSO, DZSNSO, SH2O, SICE, SMC and every scalar flux --
plus the `probe` stage and both libm sweeps, with no tolerance applied
anywhere.  Peak device allocation 28160 bytes.

CuPy's `RawModule` compiles from a string with no include path, so
`noahmp_water.cu` has to be self-contained and therefore carries **copies** of
`noahmp_soilwater.cu` and `noahmp_snow.cu`, delimited by
`>>> BEGIN imported section` / `<<< END imported section` markers.  The copies
are produced by a documented transform:

* take `noahmp_soilwater.cu` from `#define NSOIL 4` to the start of its
  host-facing kernels, and `noahmp_snow.cu` from `#define NSNOW 3` to the start
  of its entry points;
* drop from the snow copy the four pieces the soil copy already provides -- the
  `__exp2f_data` tables, the rounding-pinned `FADD`/`DADD` macros, `f_max`/
  `f_min`, and `glibc_expf`;
* drop the per-leaf host layout macros from both;
* move the snow lane's private constant table into its own namespace,
  `C_F32` -> `C_SN_F32` and `K_*` -> `SN_K_*`.

Nothing else is touched, so every arithmetic site keeps the exact form its own
lane's device gate already accepted.
`test_imported_sections_match_their_sources` re-derives both copies from their
source files by that transform and requires **byte equality**, and
`test_the_drift_check_can_fail` proves that comparison can fail.  Both run
without a GPU, so a change in either lane that this file has not picked up
fails on any machine rather than silently forking a transcription that is
already gated at `max_ulp 0`.

`test_the_device_gate_can_fail` perturbs WATER's own `0.001` -- the mm/s to m/s
conversion at 6159-6170, which is the one constant this file adds on top of the
two imported sections -- by the same relative 1e-3 the CPU mutation study uses,
and requires `ACC_QINSUR` to move on at least one case.
