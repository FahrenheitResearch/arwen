# GOCART emission oracle fixture: afwa

AFWA dust (dust_opt=3): gocart_dust_afwa_driver and source_dust, chem/module_gocart_dust_afwa.F. A covering set of 14 switch variants, not the 192-way factorial (a 101 MB fixture): every dust_smois x sf_surface_physics pair (2, 3, 7 and another scheme), then dust_veg 1 and 2, dust_dsr 1, dust_soils 1 and non-default alpha/gamma/smtune/ustune each moved alone from a volumetric Noah base, then all of them together. The base variant runs three consecutive steps, the others one: 16 cases on the same 38 x 2 patch, with negative DRI and NGA sentinels for the fallbacks.

Built by `bash tools/chem_wrf471_oracle/gocart/build.sh <wrf-src> <build> <fixtures>`
from the byte-unmodified WRF v4.7.1 sources pinned in
`tools/chem_wrf471_oracle/gocart/SOURCES.sha256` (commit
`f52c197ed39d12e087d02c50f412d90d418f6186`), compiled at -O0 against
`stub_gocart.F90` with the libmvec guard and its positive control, by GNU
Fortran 15.2.0 with glibc 2.43 on Linux, 2026-09-30.  The driver is
`tools/chem_wrf471_oracle/gocart/run_gocart_afwa.F90`.  Each case records the
driver's input arrays, switches and step, and its outputs; W-level inputs
carry three levels and the replay pads the unused top one.
`oracle-sha256sums.txt` pins every file; `measured-ulp.json` is the
per-case ULP table the parity test must reproduce.  The fixture was built
twice, by the implementer lane and independently from the pinned sources,
and the bytes agree.
