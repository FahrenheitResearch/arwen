# RTE-RRTMGP MERRA aerosol oracle

Oracle: public earth-system-radiation/rte-rrtmgp commit
fa107a16120051c4124305c6b3d4c87059119f58. Table source: public
rrtmgp-data v1.9, commit eff0433faf9cbac3ad14fbf608bef0c26ebc4c79.
Licences: BSD-3-Clause. Unmodified licence texts are in the harness source folder.
SOURCES.sha256 pins every fetched file; HARNESS.sha256 pins the compiled driver
and the unchanged shared oracle_io.F90; TABLES.sha256 pins both NetCDF tables.
The shared writer uses little-endian f4/i4, Fortran array order, unchanged API.

Command:
`bash tools/chem_wrf471_oracle/rrtmgp_aerosol/build.sh build tables`
Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
Runtime: Ubuntu GLIBC 2.43-2ubuntu2.4. NetCDF-Fortran 4.6.2.
Flags: -O0 -cpp -ffree-form -ffree-line-length-none
-fallow-argument-mismatch -ffp-contract=off -fcheck=bounds.
sp adds -DRTE_USE_SP. dp leaves the upstream default wp=double unchanged.
WRF preprocessor defines are not used because these are RTE modules.
The source files are compiled byte-unmodified. No source extracts or stubs.
The libmvec positive control finds
_ZGVbN4v_expf and _ZGVdN8v_expf. No -O0 object has a _ZGV symbol.

sp and dp each contain four cases. Each types case has 25 columns and eight
layers: all seven types, every dust and salt bin, both mineral types on every
lower bin edge, RH 0.1/0.5/0.8/0.95/0.99/1.0/0.0/0.5, mass zero in the top
layer, and 1e-12 kg/m2 in layer seven. RH=1 is above the table top
0.9900000095367432. Each mixture case has 25 columns and eight layers, including
dust storm and marine columns, sulfate, and zero mass at the top.
The harness also prints the source's invalid type, size and RH messages.
Inputs are explicitly rounded to float32 before both builds. dp output is
rounded to float32 by the unchanged fixture writer. This isolates wp effects.

packed-sw.npz and packed-lw.npz are portable test inputs produced on the host
by `prepare_tables.py`, through gpuwm.netcdf_bridge.Dataset. They are not
Fortran output and are not Python NetCDF decodes. Runtime loading uses the
same Rust reader. oracle-sha256sums.txt pins all binary inputs, outputs and
MANIFEST.txt files. The test additionally pins that manifest's SHA256.

GPU assertions: `python -m pytest tests/test_chem_rrtmgp_aerosol_parity.py`
on a CUDA card (measured on an RTX 4090, CuPy 14.0.1, NumPy 2.2.6). All nine
tests pass. Maximum CuPy pool size: 505856 bytes.

Every measured float32 port output has max_ulp=0 and n_nonzero=0.
There are 2800 values per output per SW case and 3200 per LW case.
Default-double differences, including very small mass epsilon effects, are
asserted by the CPU precision test.
Mutation check: multiplying tau by 1.01 in the CUDA source causes the
SW types parity test to fail. The source was restored
and the complete suite rerun successfully.
