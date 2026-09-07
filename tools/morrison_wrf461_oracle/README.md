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


The 2026-09-04 platform check used unchanged production source from integration
commit `7af9e74f7366e2eab36bc30e9662e3a20f9e6615`. On an RTX 3080 (sm_86),
WSL Ubuntu 24.04, Python 3.12.3, CuPy 14.2.0, CUDA runtime 12.9, NVRTC 12.8
and driver API 13.3, the complete 19-field residual is pinned as
`MEASURED_SM86_LINUX_MAX_ULP` in `tests/test_morrison_wrf461_parity.py`.
The same card under native Windows, Python 3.13, CuPy 14.0.1 and CUDA
runtime/NVRTC 13.0 reproduces the existing `MEASURED_SM89_LINUX_MAX_ULP`
signature exactly. Architecture alone does not identify the signature.

A fresh GNU Fortran 13.3.0 / glibc 2.39 build of the four byte-pinned WRF
modules reproduced both reference CSVs exactly after line-ending normalization:

- `morrison-levels.csv`: `fd3da3055881ebe8756d9274d9c276478cf6de02799c3904df0dfd822d74e78b`
- `morrison-surface.csv`: `fa45f06b0aef1ab604b1df7c0832bf3a466fe4ca55f14a10e28c66990ace06e7`

The Linux production measurement contains 3,563 differing values out of
10,948. A no-FMAD diagnostic still differs in 3,403 values. These controls
rule out reference drift and show that disabling contraction does not establish
WRF agreement. The added signature records an observed compiler/platform
residual; the strict bitwise acceptance xfail remains. A separate GPU control
changes the latent-heat intercept from 3.1484e6 to 3.1494e6 in an in-memory
compilation and requires the complete signature guard to reject that change.
