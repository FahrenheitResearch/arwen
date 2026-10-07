# GOCART inventory-emission oracle fixture

WRF v4.7.1 (`f52c197ed39d12e087d02c50f412d90d418f6186`, public domain)
`chem/emissions_driver.F` lines 1597-1619, the `emiss_opt == 6` block of the
GOCART_SIMPLE emission case, lifted by exact line range (file sha256
`92dcab10ba512c9758dd7ba3c4c9df6ac578a7ecf1acb3db5756c69fb34e2851`, checked
before extraction) into `tools/chem_wrf471_oracle/gocart/inventory_harness.F90`
and driven by `inventory_driver.F90`, built at `-O0` by

    bash tools/chem_wrf471_oracle/gocart/build_inventory.sh <wrf-src> <build> <fixtures>

on Linux (GNU Fortran 15.2.0, glibc 2.43), 2026-09-30.  Four cases:
`kemit` 1, 3, 9 (the Registry default) and 12 (the whole column) with
`dtstep` 36, 7.5, 60 and 18 s; each a 64 x 1 patch of 12-level columns cycling through
zero, trace, urban and point-source emission magnitudes for all six
`gocart_ecptec` species (`registry.chem:4058`).  `chem_in`/`chem_out` are the
whole `chem` array before and after; `oracle-sha256sums.txt` pins every file.
