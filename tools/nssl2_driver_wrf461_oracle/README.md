# NSSL driver-support WRF v4.6.1 oracle

This harness compiles the byte-pinned official WRF v4.6.1
`module_mp_nssl_2mom.F` and changes only the visibility of `calcnfromq`,
`calcnfromcuten`, and `sediment1d`. The Fortran comparator reproduces the
immediately surrounding default driver gather, four KF tendency loads,
precipitation reduction, and final scatter. It executes the complete default
`sediment1d` category sequence, including cloud-droplet mass/number fallout.

It intentionally stops before GS and therefore does not execute GS, NUCOND,
radar, effective radii, finish, restart, preflight, or global selection code.

Run from a fresh build directory:

```bash
./build.sh /path/to/WRF-v4.6.1 /new/oracle-build
```

The build refuses a source whose SHA-256 differs from the pinned official
source, emits both source and fixture hashes, and covers empty/no-op,
initialized and uninitialized moments, all four KF rates, adaptive CFL > 1,
variable-density graupel/hail, cloud-droplet transport, precipitation
reduction, and column water-plus-surface-export conservation.
