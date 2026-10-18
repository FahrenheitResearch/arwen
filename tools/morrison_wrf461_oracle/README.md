# Morrison WRF v4.6.1 oracle

This harness calls the public `MP_MORR_TWO_MOMENT` wrapper in the pinned,
byte-unmodified WRF source at
`d66e442fccc04111067e29274c9f9eaccc3cef28`. It also compiles WRF's own radar
module, model constants, and `ccpp_kind_types.F` with `-DRWORDSIZE=4`.
`stub_wrf_error.F90` supplies only WRF's logging/error service; no Morrison
source block is extracted or reimplemented.

From WSL:

```bash
build_dir=$(mktemp -d)
tools/morrison_wrf461_oracle/build.sh \
  "$WRF_TREE" "$build_dir"
```

The linked reference is compiled at `-O0`. The build fails if its undefined
symbols contain `_ZGV`, and also fails if a vectorisable `EXP` positive control
compiled at WRF's `-O2 -ftree-vectorize -funroll-loops` flags does *not* contain
one. The latter prevents a silent, ineffective libmvec guard.

The runner writes 28 columns: 14 atmospheric states under both
`morr_rimed_ice=0` (graupel) and `=1` (hail), each with 32 levels. Inputs span
warm, mixed, glaciated, dry, evaporating, melting, riming, autoconversion,
sedimentation, cumulus-seeding, mass-threshold, ultracold, and phase-cleanup
branches. They include both signed zeros, positive and negative subnormals,
the normal/subnormal boundary, and every hydrometeor at zero, near zero, and
active mass. Every reference output in the CSV was returned by WRF.

To measure gpuwm's Morrison (`gpuwm/core/kernels/morrison.cu`, the literal
transcription of this module) with the repository's sole FP32 total-order
ULP implementation:

```bash
python tools/morrison_wrf461_oracle/validate_morrison_oracle.py --row37-off
python tools/morrison_wrf461_oracle/validate_morrison_oracle.py
```

The first builds the unit with `MORR_ROW37_OFF`, WRF's arithmetic alone:
every one of the 10,948 returned words is bitwise.  The second is the
production build, which keeps the owner-ruled row-37 corrections (the
deposition-nucleation vapour cap and the wide re-evaluation of a
non-finite freezing rate); on the fixture they fire only in case 13 under
both modes, and `tests/test_morrison_wrf461_parity.py` pins exactly that.

## Stress columns beyond the fixture

`run_morrison_columns.F90` drives the same public wrapper over many
columns read from a binary file; `stress_columns.py` writes seeded random
columns for it and grades gpuwm against what WRF returned:

```bash
# beside build.sh's objects (gfortran -O0, same flags):
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -I .     tools/morrison_wrf461_oracle/run_morrison_columns.F90
gfortran -o run_cols stub_wrf_error.o ccpp_kind_types.o     module_model_constants.o module_mp_radar.o     module_mp_morr_two_moment_O0.o run_morrison_columns.o
python tools/morrison_wrf461_oracle/stress_columns.py make stress
for i in 0 1 2 3 4 5; do ./run_cols stress/stress_$i.bin stress/wrf_$i.bin; done
python tools/morrison_wrf461_oracle/stress_columns.py compare     stress/stress_0.bin stress/wrf_0.bin --row37-off
```

The constants the unit reads come from WRF's own MORR_TWO_MOMENT_INIT and
radar_init (`tools/morrison_constants`), dumped from the same -O0 build.
