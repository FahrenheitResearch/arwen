# GSL drag suite (gwd_opt = 3) against WRF v4.6.1

Column oracle for WOOF's `launch_gwdo_gsl` (`gpuwm/core/terrain_drag.py`,
`gwdo_gsl_column` in `gpuwm/core/kernels/terrain_drag.cu`) against WRF
v4.6.1's own, byte-unmodified `phys/module_bl_gwdo_gsl.F`.

On a Linux box with gfortran (the fixture of record: gfortran 13.3.0, glibc
2.39):

    bash tools/gsl_drag_wrf461_oracle/fetch_sources.sh WRF461
    bash tools/gsl_drag_wrf461_oracle/build.sh WRF461 BUILD
    python tools/gsl_drag_wrf461_oracle/pack.py BUILD tests/data/oracles/gsl_drag

`build.sh` refuses any WRF bytes other than `SOURCES.sha256` pins and writes
three trees: `fixtures` (the oracle of record, -O0, scalar libm),
`fixtures-wrfflags` (the module at WRF's own gfortran flags) and
`fixtures-cr-atan2` (a correctly rounded atan2f linked in).  The last two are
for attribution only.

On a GPU (strict arithmetic, then default):

    GPUWM_WRF_EXACT=1 python tools/gsl_drag_wrf461_oracle/compare.py strict.json
    python tools/gsl_drag_wrf461_oracle/compare.py default.json
    python -m pytest tests/test_gsl_drag_wrf461_column_oracle.py

`compare.py FIXTURE OUT.json` also takes a raw build tree, for example
`BUILD/fixtures-wrfflags`, to measure the port against the attribution
builds.
