These portable tests grade inputs and expected SHA-256 values from fresh
scalar Fortran column oracles. No raw output stream is kept here.

Generation command on G1:
`python tools/sfclay_noftz_check/focused_fixtures.py` after `prepare.sh`.

WRF source: v4.6.1 d66e442fccc04111067e29274c9f9eaccc3cef28.
Classic MM5: only module_sf_sfclay.F:767 is corrected from PSIH to PSIH10,
as specified by tools/sfclay_classic_wrf461_oracle/wrf-ck-correction.patch.
GSD MYNN: NOAA-EMC/HRRR v4.1.21 module_bl_mynn.F, SHA-256
42178a22693ad14b1a3adcf3e44be3047f051abf80f3a64c0ce9ed9ac2ec6d8a,
with the qtke line squared. Compiler: gfortran 13.3.0, glibc 2.39.

MYNN surface inputs contain QV1 words 1, 2, 0xfff, 0x7fffff and two normal
controls, over land and water. The PBL fixture covers very dry convection
at a 20 s step. Classic MM5 includes tiny negative previous MOL. Revised
MM5 includes tiny humidity, fluxes and negative previous MOL. Eta includes
tiny moisture, signed winds and persistent surface state.
