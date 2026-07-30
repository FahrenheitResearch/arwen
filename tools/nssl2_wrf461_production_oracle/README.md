# WRF v4.6.1 NSSL production-stage oracles

This narrow oracle build compiles the official NCAR WRF v4.6.1
`phys/module_mp_nssl_2mom.F` at commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. The source file is required to
have SHA-256
`5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3`.
`visibility.patch` adds only public declarations for the private `NUCOND` and
`radardd02` routines; it changes no executable statement.

`nucond_production.F90` uses the default option-18 `ipconc=5`, `rcond=2`,
`irenuc=2`, `iqcinit=2`, and `imaxsupopt=4` configuration. Its 64 rows include
simultaneous cloud and rain condensation, clear-air activation, interior
renucleation, complete and partial evaporation, moment bounds, cleanup, and
both ordinary and maximum-supersaturation `QVEXCESS` paths.

`radardd02.F90` uses the same initialized registry layout and exercises rain,
cloud ice, snow, graupel, and hail alone and in mixtures. Number moments and
predicted graupel/hail volume span their native size and density bounds.

Run with GNU Fortran on Linux:

```sh
./build.sh /path/to/WRF-v4.6.1 /new/oracle-build
```
