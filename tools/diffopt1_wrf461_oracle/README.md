# diff_opt=1 column oracle against compiled WRF v4.6.1

This oracle compares WOOF's GPU coordinate-surface mixing (`diff_opt = 1`
with `km_opt = 2` or `4`) with WRF v4.6.1's own Fortran, word for word.

## What is compared

Both sides get the same input words: winds, geopotential, theta and WRF's
`t_init`, dry inverse density, pressure, six water species, two number
concentrations, prognostic TKE, map factors, the vertical coordinate and
`cf1..cf3`.

- **The WRF side.** `pipeline.F90` calls unmodified WRF v4.6.1 routines in
  the order a serial run calls them on RK step 1: `phy_prep`,
  `compute_diff_metrics`, `set_physical_bc3d` on the metrics,
  `cal_deform_and_div`, `calculate_km_kh` (which calls `smag2d_km`, or
  `calculate_N2` and `tke_km`), `phy_bc`, then the `diff_opt1` blocks of
  `module_em.F`: `horizontal_diffusion` for u, v and w with `xkmh`,
  `horizontal_diffusion_3dmp` for theta with `xkhh` and `t_init`, and
  `rk_scalar_tend`'s `horizontal_diffusion` for every moist species, every
  number scalar and TKE with `xkhh`. Under `bl_pbl_physics = 0` it also
  calls WRF's constant-`kvdif` vertical routines (`kvdif = 0`, the only value
  WOOF accepts with `km_opt` 2 to 4). WRF's own `set_physical_bc2d/3d` fills
  every halo.
- **The WOOF side.** `oracle.py compare` runs the forecast's production code:
  `launch_wrf_smag2d_km` / `launch_wrf_tke_km`, then
  `_compute_wrf_smag_tendencies` (which routes `diff_opt = 1` to
  `_compute_coordinate_tendencies`) on the specs `_smag2d_specs` builds. The
  `t_init` reference is captured by `initialize_coordinate_reference`, as a
  forecast's first step does.
- **The outputs.** Every word, signed zeros included. Consumed by the model:
  `xkmh`, `xkhh` and the tendencies of u, v, w, theta, qv, qc, qr, qi, qs, qg,
  ni, nr and TKE. Also compared: D11, D22, D12 and, for `km_opt = 2`, `xkmv`,
  `xkhv` and BN2 (WRF writes them; nothing reads them under `diff_opt = 1`).

## Configurations (arms)

| arm | km_opt | mix_isotropic | isfflx | bl_pbl_physics |
|---|---|---|---|---|
| km4 | 4 | 0 | 1 | 1 |
| km4_pbl0 | 4 | 0 | 1 | 0 |
| km2 | 2 | 0 | 1 | 1 |
| km2_isotropic | 2 | 1 | 1 | 1 |
| km2_seed_pbl0 | 2 | 0 | 0 (TKE seed 1e-6) | 0 |

## The cases

`oracle.py cases` takes the 27 windows of the km_opt=4 column oracle
(14 x 12 columns, every level: 12 km CONUS WRF v4.6.1 history at
2019-11-26 13 UTC, a 3 km convective window, and nine named edge
transformations) and adds the two inputs `diff_opt = 1` needs: prognostic
TKE (a boundary-layer profile plus columns at exactly 0, 1e-12, the 1e-6 seed,
1e-40 and -0) and `t_init`. 27 cases, 4,536 columns, 5 arms.

`oracle.py seal` crops six of them to 8 x 7 columns (336 columns, one per
regime) and keeps the unmodified-WRF and seam-control words in
`tests/data/diffopt1_wrf461`.

## Commands

On W1, with `T` set to a tree and `source /work/pverify/base/env.sh`:

```sh
O=tools/diffopt1_wrf461_oracle
python $O/build.py /work/pverify/wrf-build/src/WRF-4.6.1 BUILD
python $O/build.py /work/pverify/wrf-build/src/WRF-4.6.1 BUILD-STOCK --flags stock
python $O/build.py /work/pverify/wrf-build/src/WRF-4.6.1 BUILD-CTL --control periodic-top-slope
python $O/oracle.py cases --source KM4_CASES --out CORPUS
python $O/oracle.py wrf --library BUILD/oracle.so --cases CORPUS --prefix wrf
python $O/oracle.py wrf --library BUILD-STOCK/oracle.so --cases CORPUS --prefix wrfstock
python $O/oracle.py wrf --library BUILD-CTL/oracle.so --cases CORPUS --prefix wrfctl
python $O/oracle.py wrf-vs-wrf --cases CORPUS --a wrf --b wrfstock
# GPU, through the mutex.  Strict build: GPUWM_WRF_EXACT=1.
GPUWM_WRF_EXACT=1 python $O/oracle.py compare --cases CORPUS --reference wrf --receipt R.json
python $O/oracle.py seal --cases CORPUS --library BUILD/oracle.so --control BUILD-CTL/oracle.so \
    --out tests/data/diffopt1_wrf461
python -m pytest -q tests/test_diffopt1_wrf461_column_oracle.py
```

`build.py` checks the four WRF source files against their v4.6.1 SHA-256 pins
and compiles at `-O0 -ffp-contract=off -fno-tree-vectorize -fcheck=bounds`.
`--flags stock` uses WRF's own GNU flags (`-O2 -ftree-vectorize
-funroll-loops`); it exists only to measure whether WRF's words depend on its
own optimisation level.

## The WRF defect this oracle documents

`compute_diff_metrics` computes the terrain slopes `zx`/`zy` for
`k = 1..kte` in the interior, but its periodic-boundary branches compute the
seam faces only for `k = 1..ktf = kde-1`. In a periodic domain the model-top
w-level slope at the seam stays 0 while every other column has its true
slope. WOOF computes the true slope there. `--control periodic-top-slope`
widens exactly those eight loops; it is an attribution control, never the
reference. It moves words only in periodic cases.

## WRF's own optimisation level

With WRF's stock flags, gfortran vectorises loops that call `powf`, `expf` and
`logf` and routes them to the vector math library, whose words differ from the
scalar library's. The km_opt=4 arms are identical at both optimisation levels;
the km_opt=2 arms are not (BN2, `xkmv`/`xkhv`, and with `mix_isotropic = 1`
`xkhh` and the tendencies). The `-O0` scalar build is the reference.
