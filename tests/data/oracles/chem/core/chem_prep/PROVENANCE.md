# WRF v4.7.1 chem_prep oracle

WRF commit f52c197ed39d12e087d02c50f412d90d418f6186, public domain.
GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0, glibc 2.43-2ubuntu2.4.
Flags are defined in tools/chem_wrf471_oracle/build.sh. Exact source
SHA-256 values are in oracle-sha256sums.txt.

Build: bash tools/chem_wrf471_oracle/build.sh source build run_chem_prep run_vertmx
The owned source overlay contains the pinned chem sources and the separately
fetched share/module_model_constants.F at the same WRF commit. The script
copy had CRLF converted to LF for bash; extracted WRF routines kept their bytes.
No -O0 object imports a _ZGV symbol. The -Ofast positive control does.
Every WRF source, harness file and generated wrapper hash: oracle-sha256sums.txt.
fixtures.sha256 pins every binary and MANIFEST.txt in this directory.

Families 1-4: tropical, winter, 60 kPa surface/high terrain, near saturation.
Each runs nz=49 and 59. fnm/fnp use the pinned WRF practical eta ladder
(module_initialize_real.F:7654-7663), linearly resampled to each extent,
and the WRF coefficient formulas at :3732-3745. These are column fixtures,
not an operational prepared domain. Inputs retain WRF's padding. Outputs
retain padding for audit; parity compares only physical fields, because WRF
never fills z/rh at kde. PROVENANCE does not treat the -999 sentinel as physics.
