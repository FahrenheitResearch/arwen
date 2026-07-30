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

To measure the current CUDA port with the repository's sole FP32 total-order
ULP implementation:

```powershell
python tools/morrison_wrf461_oracle/validate_morrison_oracle.py
python tools/morrison_wrf461_oracle/validate_morrison_oracle.py `
  --fmad-false-diagnostic
```

The second command is diagnostic only. Disabling contraction globally is not a
shipped parity fix.
