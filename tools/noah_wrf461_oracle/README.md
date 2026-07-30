# Noah LSM oracle, WRF v4.6.1

The first WRF number ever produced for `sf_surface_physics=2`.  Noah is the
land surface in every registry template and every production config in this
tree, and until this harness existed the only WRF-derived evidence it carried
was its three parameter tables.  Its CUDA kernel had been checked against
`gpuwm.verify.npref.np_noah_column`, a float64 mirror of the same
transcription, which is the arrangement that let a misread line in YSU agree
with itself for months.

## What it builds

```
bash tools/noah_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`WRF_SOURCE_ROOT` must be at `d66e442fccc04111067e29274c9f9eaccc3cef28` with
all five compiled sources clean; `build.sh` refuses otherwise.

Four fixtures land in `gpuwm/data/noah/oracle/` (see `PROVENANCE.md` there),
one per combination of the four driver switches `gpuwm.core.noah.launch_noah`
also exposes.

## The driver, not SFLX

gpuwm's port is one fused device function -- `noah_column` in
`gpuwm/core/kernels/noah.cu` -- whose scope is exactly `lsm`'s
(`phys/module_sf_noahdrv.F:38`): per-column input preparation, then `SFLX`,
then the post-`SFLX` state and flux updates and the driver diagnostics.  An
oracle at `SFLX`'s argument list would have left gpuwm computing `SFLX`'s
inputs from its own transcription of the prep and then measuring itself
against a reference handed those same numbers.  Calling `lsm` means every
value in the fixture is a driver input the port also takes or a driver output
the port also produces.

## Stubs

Smaller than the RUC and Noah-MP harnesses, and smaller in a way that matters:

* `share/module_model_constants.F` is **compiled from the pinned tree**, not
  stubbed.  It carries `CP`, `R_D`, `XLF`, `XLV`, `RHOWATER`, `STBOLT` and
  `KARMAN`, which are the constants the whole reference is measured in.
  `tools/ruc_wrf461_oracle/stub_wrf.F90` restates them by hand as independent
  literals (`r_d = 287.0`, `cp = 1004.5`) where WRF derives `cp = 7.*r_d/2.`;
  those agree here, but only by luck, and a stub is free to stop agreeing.
* `frame/module_wrf_error.F` is compiled from the pinned tree too -- it builds
  standalone once `DM_PARALLEL` is undefined, and it defines
  `wrf_error_fatal`.
* `stub_wrf.F90` is therefore service only: `wrf_abort`, `wrf_debug`, the
  single-rank `wrf_dm_bcast_*` shims `SOIL_VEG_GEN_PARM` calls, and
  abort-only link targets for the urban/BEP/GFDL packages this harness does
  not build.  Nothing in it computes a physical quantity.  `lsm` is called
  with `sf_urban_physics = 0`, so no fixture row can reach an aborting stub.

## -Dwrfmodel is load-bearing

WRF's `arch/postamble:26` passes `-Dwrfmodel`.  Without it,
`module_sf_noahdrv.F:1787`'s guard preprocesses `LSMINIT` and
`SOIL_VEG_GEN_PARM` away entirely and the object exports only `lsm` -- a
harness that missed it would have had to hand-fill the parameter tables, i.e.
to invent them.  `build.sh` checks all three symbols are exported before it
writes a fixture.

## libmvec

Measured on gfortran 13.3.0 / glibc 2.39: no `_ZGV*` symbol appears in
`module_sf_noahlsm.o` or `module_sf_noahdrv.o` at `-O0`, at WRF's own `-O2
-ftree-vectorize`, or at `-Ofast`.  Every transcendental in the Noah column
sits in a loop the vectoriser gives up on -- soil-layer loops carrying
conditionals, `FRH2O`'s `goto`-based Newton loop, `SNOPAC`'s branch tree.  The
oracle is still built at `-O0` and `build.sh` still fails on a `_ZGV*` symbol.

A guard that has never fired proves nothing, so
`libmvec_positive_control.F90` is the same `expf` in a loop the vectoriser
*can* take; `build.sh` fails if that does **not** emit `_ZGVbN4v_expf`.  It
does (see `libmvec-report.txt`).

`-Ofast` does one other interesting thing: `cbrtf` appears in
`module_sf_noahlsm.o`, which is `-ffast-math` rewriting a cube-root-shaped
`**` into `cbrt`.  The reference is the `-O0` object, which calls `powf`.

## Fixture coverage

42 columns, chosen to reach branches rather than to look meteorological:

* warm unfrozen land (NOPAC); dry soil near wilting; saturated soil under
  heavy rain; bare soil (`shdfac = 0`); a closed canopy at maximum
  interception;
* a sub-freezing snowpack, a melting one, a deep one at complete cover, and
  an aged one driving the `SNOTIME` albedo decay (SNOPAC's four arms);
* a fully frozen soil profile, which is what puts `FRH2O`'s Newton iteration
  in every layer;
* `SFCTMP` exactly `273.15` and one float32 step above it -- the pair
  discriminates `FFROZP`, and the second row lands in `NOPAC`'s freezing-rain
  (`FRZGRA`) arm rather than simply skipping the snow physics;
* `T1` exactly `273.14`, exactly `273.0` with `SWDOWN` exactly `10.0`, and one
  float32 step past both: the three `.GT.` tests that select the ice/water
  `Q2SAT` blend and the `DQSDT2` damping;
* `RAINBL` at `+0.0`, `-0.0` and the smallest positive subnormal; `SNOW`
  subnormal (so `SNEQV = SNOW*0.001` underflows and the `SNEQV /= 0 .and.
  SNOWHK == 0` guard decides `SNOWHK`); `QV3D` subnormal, `+0.0` and `-0.0`;
  `VEGFRA` at `+0.0` and `-0.0`; `CANWAT` subnormal; `CHS` subnormal.  Nine
  probes on sign compares and underflow, because CuPy appends `-ftz=true`
  unconditionally and this project has already lost three days to exactly
  that;
* open water (`xland=2`), sea ice (`xice=0.6`), and land ice (`ivgtyp=isice`),
  which is the one row gpuwm skips by documented restriction rather than
  computes;
* soil type 14 at a land point with `xice=0`, which WRF resets to 7; the urban
  category; sand, clay, organic and bedrock; soil moisture above porosity on
  entry;
* a stable nocturnal column with no shortwave and weak exchange, and a
  strong-demand daytime column;
* soil types 3 and 4, and only those two, because `TDFCND`
  (`module_sf_noahlsm.F:4173`) takes the `opt_thcnd == 2` arm for no other
  soil type.  They were added after the fact: without them the `opt_thcnd=2`
  fixture came out **byte-identical** to the `opt_thcnd=1` one, so it could
  not have discriminated a port that ignored the switch.  `build.sh` now fails
  if any variant fixture is byte-identical to the base one.

## Measuring the port

```
python tools/noah_wrf461_oracle/validate_noah_oracle.py [FIXTURE_DIR]
```

Prints per-field ULP against the CUDA kernel; asserts nothing.
`tests/test_noah_wrf461_parity.py` is the gate.
