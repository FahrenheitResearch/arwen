# BEP-lane oracle fixtures (sf_urban_physics = 2, YSU/MYJ coupling)

Written by `tools/urban_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR`
(the shared urban harness) from byte-unmodified WRF v4.7.1 sources pinned in
`tools/urban_wrf471_oracle/SOURCES.sha256` (the lane's extra sources,
`phys/ccpp_kind_types.F` and `phys/physics_mmm/bl_ysu.F90` at MMM-physics
`20240626-MPASv8.2`, are named in `sources-bep.list`), on an x86-64 host:
`compiler.txt`, `libmvec-report.txt` (no `_ZGV*` in any -O0 object; the
positive control does emit one) and `oracle-sha256sums.txt` are that run's
receipts.  Each directory is `build/fixtures/<driver>/<case>/` copied as is;
`gpuwm.verify.urban_oracle.load_case` reads it.

| directory | driver | contents |
| --- | --- | --- |
| `bep_nlcd/steps`, `bep_lcz/steps` | `run_bep_{nlcd,lcz}.F90` (body `bep_driver.inc`, columns `bep_cases.inc`) | `SUBROUTINE BEP` for 4 steps over 21 / 36 columns (3 URBPARM.TBL classes / 11 URBPARM_LCZ.TBL classes, six regimes: day, night, low sun, gridded morphology + height histogram, near calm, dawn; frc 0, 0.02 .. 1); every input, every state word in and out per step, all outputs, and the post-`urban_param_init` table words (`tbl_*`) the port's packer is fed |
| `bep_couple_nlcd/steps`, `bep_couple_lcz/steps` | `run_bep_couple_{nlcd,lcz}.F90` (body `couple_driver.inc`) | zeroing + BEP + the coupling block for Noah (`module_sf_noahdrv.F:1679-1776`) and Noah-MP (`module_sf_noahmpdrv.F:3363-3372, 3689-3776`), 2 steps, each LSM chain carrying its own PBL arrays; and `module_surface_driver.F:3028-3032` (`sfcdiag_*`) |
| `ysu_bep/columns` | `run_ysu_bep.F90` | `bl_ysu_run` with `flag_bep = .true.`, ctopo = 1 (WRF's driver) and ctopo absent, on the 24 columns of the shipped v4.6.1 YSU fixture with BEP-shaped forcing |
| `ysu_bep_fix/columns` | `run_ysu_bep.F90` built by `build_ysu_bep_fix.sh` | the same 24 columns and both arms, with `bl_ysu.F90:1313` changed in one token (the `frc_urb1d(i)*` factor dropped), the rural-drag divergence gpuwm carries (`tests/test_ysu_bep_rural_drag.py`); the script's stock build reproduces `ysu_bep/columns` byte for byte, and the two differ only in `ctopo_utnp`/`ctopo_vtnp`; built on a second x86-64 host 2026-09-30, gfortran 15.2.0, glibc 2.43 |
| `myjurb/columns` | `run_myjurb.F90` | `MYJURB` (idiff 0, flag_bep true), 18 columns x 2 carried steps, HT 0 and 350 m |

The coupling and override blocks are INCLUDEd verbatim, never re-typed
(`tests/test_urban_bep_couple_wrf471_parity.py` pins their sha256):

```
sed -n 1679,1776p phys/module_sf_noahdrv.F    > couple_noahdrv_1679_1776.inc
{ sed -n 3363,3372p phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F;
  sed -n 3689,3776p phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F; } \
                                              > couple_noahmpdrv_3363_3372_3689_3776.inc
sed -n 3028,3032p phys/module_surface_driver.F > sfcdiag_surface_driver_3028_3032.inc
```

The same fixtures were first produced by the lane's own build step before
the shared harness landed; every one of the 1,120 files is byte-identical
between the two builds.
