# Pinned smoke oracle

Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
C library: Ubuntu GLIBC 2.43-2ubuntu2.4.
Flags: -O0 -cpp -DRWORDSIZE=4, free form, no fast math.
GSL reference kind_phys=4. KIND=8 is measured only, never the gate.
The shared libmvec guard and its positive control passed.
Build: bash tools/chem_wrf471_oracle/build_fire_oracles.sh SOURCE_ROOT BUILD_ROOT
Poison: bash tools/chem_wrf471_oracle/poison_fire_oracles.sh SOURCE_ROOT POISON_ROOT BUILD_ROOT
Source bytes are checked before compilation. Shared harness files are unmodified.

WRF v4.7.1: f52c197ed39d12e087d02c50f412d90d418f6186, public domain.
GSL ccpp-physics: 3e6660c6df54e95a0871e990c2294dd397ae3860, Apache-2.0.
https://github.com/wrf-model/WRF/blob/f52c197ed39d12e087d02c50f412d90d418f6186/chem/module_wetdep_ls.F
https://github.com/ufs-community/ccpp-physics/tree/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust

Source SHA256s (the lane's complete pin list; not every family compiles every file):

```
74aec657b3058fd8d9c9e54ea987106a7044cbab4dc8ca2c867fcf5b086ac098  chem/module_wetdep_ls.F
30d3e0bd345fa9a1ecf3098e4665aa6df81d36ef9994c4580f2cb16c5fda461a  gsl-ccpp-physics/physics/smoke_dust/module_add_emiss_burn.F90
e111a72dd34008eeb365f59c3a7c0f607d7340b7acc359c81f5dab1f05b39125  gsl-ccpp-physics/physics/smoke_dust/rrfs_smoke_wrapper.F90
27c5dc0503decb631f4e057241f35032b9a84b2f84616d8718dcee3c987ec577  gsl-ccpp-physics/physics/smoke_dust/rrfs_smoke_config.F90
```

Every published binary/manifest and license is pinned by oracle-sha256sums.txt.
Full measured output ULP tables, KIND=8 differences, compiler receipts and
mutation failures are in ../VALIDATION/. CPU outputs all measure max_ulp=0.
Device parity was not run. See tools/chem_wrf471_oracle/FIRE_INTEGRATION.md
for source lines, defined PBL edges, density choice and integration limits.
