# WRF v4.6.1 single-layer urban column oracle

Source: the original WRF release tag `v4.6.1`, peeled commit
`d66e442fccc04111067e29274c9f9eaccc3cef28` in
<https://github.com/wrf-model/WRF/tree/v4.6.1>.
`module_sf_urban.F` SHA-256 is
`433cc81d97f0096ead284cdcc984b4985f95e1bfb94c6e20d316818997e4e55e`.
All five compiled or read source files are pinned in
`tools/urban_wrf461_oracle/SOURCES.sha256`.

Built 2026-10-04 UTC with GNU Fortran 15.2.0 and glibc 2.43 at `-O0`.
The original source files were not edited. The existing Fortran argument
driver `tools/urban_wrf471_oracle/ucm_column_oracle.F90` calls the original
`urban_param_init`, `read_param` and `urban` routines. Service stubs do no
physics. `libm-report.txt` records the scalar libm calls and the positive
control that detects vector libm.

Rebuild:

```sh
bash tools/urban_wrf461_oracle/fetch_sources.sh SOURCE_DIR
nice -n 10 bash tools/urban_wrf461_oracle/build.sh SOURCE_DIR BUILD_DIR OUT_DIR
```

There are 3,240 rows covering three NLCD and eleven LCZ classes, nine forcing
scenarios and six carried steps under twelve switch combinations. Each row
records the forcing, state at entry, all nineteen outputs, and all forty-four
state words after the call. Fixture hashes are in `oracle-sha256sums.txt`.

Nine variants, 2,322 rows, are byte-identical to the existing WRF v4.7.1
fixtures. The three retention variants differ: `ucm-nlcd-imp2` has 45 changed
rows, `ucm-lcz-imp2ahalh` has 165 and `ucm-nlcd-griri` has 45. These are the
three `IMP_SCHEME=2` evaporation-depth unit corrections in WRF v4.7.1. The
engine keeps the corrected retention equations. The city table defaults use
`IMP_SCHEME=1`, so the city path matches the original WRF v4.6.1 equations.

Green-roof dew rows use the `-finit-real=zero` reference, as the existing
driver documents: WRF reads `ETR` without initialization when potential
evaporation is nonpositive. The `-finit-real=snan` control confirms the read;
counts are in `libm-report.txt`. The port initializes that local to zero.

Distributed drag and gridded morphology are not exercised. The normal
default morphology is exercised. First-level geometry rejection is covered
by the existing SLUCM tests.
