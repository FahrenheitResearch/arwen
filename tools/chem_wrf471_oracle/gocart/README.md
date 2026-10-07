# GOCART column oracles, WRF-Chem v4.7.1

`build.sh WRF_SRC BUILD_DIR [FIXTURE_ROOT]` compiles the byte-unmodified WRF
v4.7.1 GOCART process modules (dust, AFWA dust, sea salt, settling, aerosol
dry deposition, aging and PM sums, sulfur chemistry) at `-O0` against
`stub_gocart.F90`, then compiles and runs every `run_gocart_<x>.F90` here, each
writing `FIXTURE_ROOT/<x>/<case>/` through `oracle_io.F90`.

* `SOURCES.sha256` pins every compiled WRF file (commit
  `f52c197ed39d12e087d02c50f412d90d418f6186`); a mismatch stops the build.
* `stub_gocart.F90` replaces only what the WRF Registry generates at build
  time: `module_state_description` index and switch constants for the
  `gocart_simple` package (`Registry/registry.chem:4022`), a
  `grid_config_rec_type` with the fields these files read at their Registry
  defaults, a `calc_zenith` that aborts if reached, and the three
  `wrf_debug`/`wrf_message`/`wrf_error_fatal` service hooks.  Nothing in it
  computes a physical quantity.
* `share/module_model_constants.F` is compiled from the pinned tree, so the
  constants (`g`, `mwdry`, `karman`) are WRF's own.
* Every -O0 object is checked for libmvec `_ZGV*` symbols; the positive control
  must produce one or the check is vacuous.  The libm words the reference calls
  are listed in `libmvec-report.txt` (measured on node-1, gfortran 15.2.0,
  glibc 2.43): `acosf cosf sinf` (sulfur chemistry's `szangle`), `logf powf`
  (AFWA), `exp log log10 pow` in REAL*8, and libgcc's `__powidf2` for integer
  powers of REAL*8 values.

The optics oracle (`build_optics.sh`) is separate because
`module_optical_averaging.F` does not compile against this stub.

The Python side is `gpuwm.verify.chem_oracle` (`load`, `load_case`,
`ulp_table`); fixtures live in `tests/data/oracles/chem/gocart/<x>/`.
