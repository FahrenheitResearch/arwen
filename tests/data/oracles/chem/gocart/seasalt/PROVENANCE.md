# GOCART emission oracle fixture: seasalt

GOCART sea salt (seas_opt=1): gocart_seasalt_driver and source_ss, chem/module_gocart_seasalt.F. Open ocean at 10 m winds 0, 3, 10 and 25 m s-1, an elevated lake (no emission), land, thin lowest layers, dt 30 and 60 s, three consecutive steps: 6 cases.

Built by `bash tools/chem_wrf471_oracle/gocart/build.sh <wrf-src> <build> <fixtures>`
from the byte-unmodified WRF v4.7.1 sources pinned in
`tools/chem_wrf471_oracle/gocart/SOURCES.sha256` (commit
`f52c197ed39d12e087d02c50f412d90d418f6186`), compiled at -O0 against
`stub_gocart.F90` with the libmvec guard and its positive control, by GNU
Fortran 15.2.0 with glibc 2.43 on Linux, 2026-09-30.  The driver is
`tools/chem_wrf471_oracle/gocart/run_gocart_seasalt.F90`.  Each case records the
driver's input arrays, switches and step, and its outputs; W-level inputs
carry three levels and the replay pads the unused top one.
`oracle-sha256sums.txt` pins every file; `measured-ulp.json` is the
per-case ULP table the parity test must reproduce.  The fixture was built
twice, by the implementer lane and independently from the pinned sources,
and the bytes agree.
