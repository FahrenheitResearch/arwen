# SPP column oracle, WRF v4.6.1

Grades WOOF's stochastically perturbed parameter (SPP) consumers against
unmodified WRF v4.6.1 Fortran, word for word, on many columns at once.

## What WRF v4.6.1 actually perturbs

Read from the pinned v4.6.1 source on the oracle box:

| switch | WRF v4.6.1 consumer | live in stock WRF v4.6.1? |
| --- | --- | --- |
| `spp_pbl` | `module_bl_mynn.F`: diffusivity (:3135-3140), cloud PDF saturation deficit (:3842), mass-flux entrainment (:6195); `module_sf_mynn.F`: roughness ZNTstoch and the thermal roughness (:665, :715, :1269, :1458, :1491) | yes |
| `spp_conv` | `module_cu_gf_deep.F` closure ensemble `rand_clos` (:2634-2676) | **no**: `module_cumulus_driver.F:752-754` sets `pattern_spp_conv = 0` and `spp_conv = 0` locally, and GFDRV passes `rand_clos = 0` with `nranflag = 0` to `CUP_gf` (`module_cu_gf_wrfdrv.F:262`, `:589-590`) after computing the clipped pattern it then drops (:310-318) |
| `spp_lsm` | `module_sf_ruclsm.F` keeps `rstochcol` and `field_sf` arguments | **no**: the hydraulic operator was removed after V3.9.1; every RUC output is identical with `spp_lsm` 0 and 1 (measured, see below) |

WOOF applies the closure pattern where `CUP_gf` would have applied it, after
GFDRV's own clip, and restores the V3.9.1 RUC hydraulic operator
(`gpuwm/core/ruc_spp.py`).  Those two are WOOF's choices; their references
are `CUP_gf` called directly and the V3.9.1 operator overlaid on current RUC
(`tools/ruc_spp_wrf_oracle`), not a stock WRF v4.6.1 run, which ignores both
switches.

The pattern generator itself (`gpuwm/ensemble/stochastic.py`) is not a
column operation and is out of scope here.  It draws from counter-based
Philox and transforms with cuFFT; WRF draws from the compiler's
`RANDOM_NUMBER` stream and FFTPACK, so no bitwise reference exists for it.

## The columns

* MYNN, `spp_pbl = 1`, WRF's own `mynn_bl_driver`: 72 columns = six column
  families (convective day, stable night with dew, marine stratocumulus,
  shallow cumulus, a sub-freezing cold pool with fog, ice and snow, high
  wind) x three surface elevations (sea level; 0.85 ps, about 1.4 km; 0.70 ps,
  about 3 km, on a deep stretched grid whose top passes 8 km so the
  diffusivity taper is live) x four pattern shapes (zero; WRF's default
  amplitude, 0.3 sin; an edge table with -1, -0.9, -2/3, -0.3, -0, +0, 0.3,
  2/3, 1; alternating +-0.9).  Eight 20 s steps, mixing lengths 1 and 2.
  The program is generated from `tools/mynn_pbl_wrf461_oracle/run_driver_families.F90`.
* GF closure SPP, `CUP_gf` after GFDRV's column preparation: the 216 GF
  fixture columns (18 cases x 6 dx x 2 shallow arms), each with its own
  four-member closure pattern drawn from a table of 23 values (-1.7, -1,
  one float32 step inside -1, ..., both zeros, ..., one step inside 1, 1,
  one step above 1, 1.5, 2.5), clipped by GFDRV's own statements; two
  pattern sets.  Generated from `tools/gf_wrf461_oracle/run_cup_gf.F90`.

## Rerun

On the oracle box (gfortran 13.3.0, glibc 2.39), CPU only:

```
python tools/spp_wrf461_column_oracle/build.py --wrf-source <WRF-4.6.1>/phys --output <dir>
python tools/spp_wrf461_column_oracle/pack.py --oracle <dir>          # writes tests/data/oracles/spp/columns
```

On a GPU, once per arithmetic mode:

```
python tools/spp_wrf461_column_oracle/compare.py --output default.json --cpu
GPUWM_WRF_EXACT=1 python tools/spp_wrf461_column_oracle/compare.py --output strict.json
python -m pytest tests/test_spp_wrf461_column_oracle.py tests/test_spp_wrf461_column_oracle_gpu.py
```

## Files

| file | role |
| --- | --- |
| `spp_column_sources.py` | generates the two Fortran drivers by exact-once anchor replacement |
| `build.py` | checks the v4.6.1 byte pins, compiles -O0 -ffp-contract=off, refuses libmvec calls, runs, writes `receipt.json` |
| `spp_column_fixtures.py` | packs CSV captures into float32 `.npz` fixtures and reads them back |
| `pack.py` | writes the fixtures, after checking the GF capture's prepared columns equal the GF fixture's word for word |
| `spp_column_grade.py` | the word-for-word grade, per field, per step and per regime |
| `compare.py` | the box run that writes one JSON receipt per mode |
