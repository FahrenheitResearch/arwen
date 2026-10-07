# WRF v4.7.1 Wesely column oracle
GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0
Copyright (C) 2025 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

ldd (Ubuntu GLIBC 2.43-2ubuntu2.4) 2.43
Flags: -O0 -fno-fast-math -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4
Command: nice -n 15 bash tools/chem_wrf471_oracle/wesely/build.sh WRF_SOURCE_ROOT BUILD_DIR (WRF_SOURCE_ROOT = the pinned reference tree, SMOKE-AQ-2026-09-30/wrf-src layout)
cc6539387999e2d2e56c3038ae8cf1d865813658b77c822e4002d6477b438894  chem/module_dep_simple.F
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
2ed7dc6e90e0fe442ffee84512b4998d31c8ec3400d3a7ab6078404065f784a6  frame/module_wrf_error.F
No NETCDF; non-MOZART path only. No GPU used.
Scalar objects have no _ZGV symbol. Positive control:
                 U expf
                 U _ZGVbN4v_expf
