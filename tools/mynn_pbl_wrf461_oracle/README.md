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
