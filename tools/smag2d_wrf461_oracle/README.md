# km_opt=4 / diff_opt=2 column oracle against compiled WRF v4.6.1

This oracle compares WOOF's GPU 2-D Smagorinsky mixing (`km_opt = 4`,
`diff_opt = 2`) with WRF v4.6.1's own Fortran, word for word.

## What is compared

Both sides get the same input words: winds, geopotential, theta, dry
inverse density, pressure, six water species, two number concentrations,
map factors, the vertical coordinate and `cf1..cf3`.

- **The WRF side.** `pipeline.F90` calls the unmodified WRF v4.6.1 routines
  in the order `module_first_rk_step_part1/part2.F` calls them, for one serial
  tile covering the domain: `phy_prep`, `compute_diff_metrics`,
  `set_physical_bc3d` on the metrics, `cal_deform_and_div`, `calculate_km_kh`
  (which calls `smag2d_km`), `phy_bc`, then `horizontal_diffusion_2`.
  WRF's own `set_physical_bc2d/3d` fills every halo, so the halos hold what a
  serial WRF run holds.
- **The WOOF side.** `oracle.py compare` runs the forecast's own production
  code: `launch_wrf_smag2d_km` and `_compute_wrf_smag_tendencies`, on the
  specs `_smag2d_specs` builds.
- **The outputs.** Every word of these is compared: D11, D22, D12, D13 and
  D23, xkmh, xkmv and xkhh, and the horizontal-diffusion tendencies of u, v,
  w, theta, qv, qc, qr, qi, qs, qg, ni and nr. Signed zeros count.
- **D13 and D23.** The strict w operator evaluates these inline.
  `tensor_probe.cu` launches the same device functions (`wrf_defor13`,
  `wrf_defor23`) and holds no tensor arithmetic of its own.

## The cases

There are 27 cases and 4,536 columns. Each window is 14 x 12 columns with
every model level kept.

| Group | Source | Regimes |
|---|---|---|
| 12 real CONUS windows | 12 km WRF v4.6.1 run, 2019-11-26 13 UTC, `diff_opt = 2`, `km_opt = 4` | High terrain with snow and mountain waves; dry high terrain; stable night plains; marine heating; Pacific snow storm; Great Lakes snow; Gulf cloud; open Atlantic; the steepest, snowiest, iciest and strongest-w windows in the domain |
| 1 warm-season window | 3 km, 2024-05-21, Iowa | Convective environment |
| 9 edge cases | Transformations of the stable-night window | All-zero flow; signed-zero flow; near-zero flow (subnormal strain); map factors from 0.25 to 4; 1500 m synthetic terrain (slope limiter); winds times 25 (strain above `def_limit`, K at the `10*mlen` cap); a 28 m/s synthetic updraft core with heavy hydrometeors; a north-south mirror |

Lateral boundaries are mostly WRF's real-data `specified`. Two real cases and
two edge cases also run `periodic`. Two cases also run `open`.

Inverse density comes from WRF's moist-theta equation of state. Theta is
WRF's `t_2 = theta - 300 K`, first rounded through `fl(fl(300 + T) - 300)`,
which is exact. So WOOF's `thp + thb - 300` with `thb = 300` gives the same
word in every arithmetic mode.

## Commands

On W1, with `T` set to a tree and `source /work/pverify/base/env.sh`:

```sh
python tools/smag2d_wrf461_oracle/build.py /work/pverify/wrf-build/src/WRF-4.6.1 BUILD
python tools/smag2d_wrf461_oracle/build.py /work/pverify/wrf-build/src/WRF-4.6.1 BUILD-CTL --control periodic-top-slope
python tools/smag2d_wrf461_oracle/oracle.py cases --reader tools/rustwx/target/release/rw_netcdf \
    --conus WRFOUT_2019-11-26_13 --convective WRFOUT_2024-05-21 --out CASES
python tools/smag2d_wrf461_oracle/oracle.py wrf --library BUILD/oracle.so --cases CASES
python tools/smag2d_wrf461_oracle/oracle.py wrf --library BUILD-CTL/oracle.so --cases CASES --prefix wrfctl
python tools/smag2d_wrf461_oracle/oracle.py seal --cases CASES --out tests/data/smag2d_wrf461
# GPU, through the mutex; the strict build is GPUWM_WRF_EXACT=1
GPUWM_WRF_EXACT=1 python tools/smag2d_wrf461_oracle/oracle.py compare \
    --cases tests/data/smag2d_wrf461 --reference wrf --receipt receipt.json
python -m pytest -q tests/test_smag2d_wrf461_column_oracle.py
```

`build.py` checks the four WRF source files against their v4.6.1 SHA-256
pins. It compiles with gfortran at `-O0 -ffp-contract=off -fno-tree-vectorize
-fcheck=bounds`.

## The WRF defect this oracle documents

WRF v4.6.1's `compute_diff_metrics` has two loop ranges that disagree:

- Its interior loops compute the terrain slopes `zx` and `zy` for
  `k = 1..kte`.
- Its periodic-boundary branches compute the seam face (`ids`/`ide`,
  `jds`/`jde`) only for `k = 1..ktf = kde-1`.

So in a periodic domain, the model-top w-level slope at the seam is never
written. It stays 0 while every other column has the true slope. This
changes D11, D22, D12, K and the tendencies at the top one or two mass
levels of periodic domains.

WOOF computes the true slope there and does not copy the defect.

The `--control periodic-top-slope` build widens exactly those eight loops.
It is an attribution control, never the reference. Against it, WOOF
matches every word of every case. Non-periodic WRF zeroes the boundary slope
at every level, so specified and open domains are unaffected.
