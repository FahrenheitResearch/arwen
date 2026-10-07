# GOCART emission oracle fixture: dust

GOCART dust (dust_opt=1): gocart_dust_driver and source_du, chem/module_gocart_dust.F. Four variants of a 38 x 2 patch of 3-level columns (every soil category 1-19 including water, wet, dry and zero soil moisture, wind under and over threshold, a thin lowest layer that switches the wind to u_phy/v_phy, class-by-class zero erodibility), three consecutive steps each: 12 cases.

Built by `bash tools/chem_wrf471_oracle/gocart/build.sh <wrf-src> <build> <fixtures>`
from the byte-unmodified WRF v4.7.1 sources pinned in
`tools/chem_wrf471_oracle/gocart/SOURCES.sha256` (commit
`f52c197ed39d12e087d02c50f412d90d418f6186`), compiled at -O0 against
`stub_gocart.F90` with the libmvec guard and its positive control, by GNU
Fortran 15.2.0 with glibc 2.43 on Linux, 2026-09-30.  The driver is
`tools/chem_wrf471_oracle/gocart/run_gocart_dust.F90`.  Each case records the
driver's input arrays, switches and step, and its outputs; W-level inputs
carry three levels and the replay pads the unused top one.
`oracle-sha256sums.txt` pins every file; `measured-ulp.json` is the
per-case ULP table the parity test must reproduce.  The fixture was built
twice, by the implementer lane and independently from the pinned sources,
and the bytes agree.
