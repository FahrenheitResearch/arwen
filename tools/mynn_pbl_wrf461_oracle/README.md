# WRF 4.6.1 MYNN PBL level-2 oracle

This harness compiles the pinned, unmodified WRF sources
`phys/module_bl_mynn_common.F` and `phys/module_bl_mynn.F`, then calls
`mym_level2` for four eight-level profiles. The 28 output rows exercise stable,
convective, exact-neutral, and moist/cloud-modified buoyancy gradients.

It also calls `get_pblh` and `scale_aware` for four ten-level columns spanning
convective land, stable land, marine, and low-TKE cold-pool conditions. The
resulting `pblh-scale.csv` contains the complete column inputs and repeated
column output identity for independent reproduction.

Finally, `mixlength.csv` calls the default `bl_mynn_mixlength=1` path for
stable, convective, high-shear, and active-EDMF columns. It records all column
inputs plus WRF's interface `el` and `qkw` outputs.

Run:

```bash
./build.sh /path/to/WRF-4.6.1 /absolute/build/directory
```

The generated `pbl-level2.csv`, source hashes, compiler identity, and validator
result form the first numerical oracle for the coupled MYNN PBL port. This
oracle alone does not admit `bl_pbl_physics=5`.

`run_driver.F90` exercises the assembled driver for two calls over five
columns. Its `snow_anvil` column sets `FLAG_QS=.TRUE.`, carries nonzero snow
with no cloud ice above the inversion, and records `sqs3d` alongside every
input and output. This directly pins the wrapper/driver snow plumbing against
the unmodified WRF v4.6.1 source rather than inferring it from the ArWen port.
The source tree is tag `v4.6.1` at
`d66e442fccc04111067e29274c9f9eaccc3cef28`; GNU Fortran 13.3.0 produced
the 300-row fixture. SHA-256:

- `module_bl_mynn_common.F`:
  `71cc8a9a77280fff950b89223cc65d6ad62b1844f48fb9cf8f85e17e7313ae17`
- `module_bl_mynn.F`:
  `7fde5fc9d9760c106a400416feb127e5b18f6ba6f03b8aaf0ac8f2c1af2766e2`
- `run_driver.F90`:
  `8668f0c74168c34730bffabe5656693f209ebad5dc3a1a9dd1f3f92e32c18a03`
- `driver.csv`:
  `f608d456eaa21878491985c3ed4d5bde70896b452f3eb9b71076d6466fdbf955`
