# BEP column oracle, WRF v4.6.1

Grades WOOF's BEP port (`sf_urban_physics = 2`: `gpuwm/core/kernels/urban_bep.cu`
and the coupling block in `urban_bep_couple.cu`) against WRF v4.6.1's own
Fortran, word for word.

```
bash tools/urban_bep_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
bash tools/urban_bep_wrf461_oracle/build_crlibm_arm.sh WRF_SOURCE_ROOT CORE_MATH_REPO BUILD_DIR_CR
GPUWM_WRF_EXACT=1 python tools/urban_bep_wrf461_oracle/compare.py BUILD_DIR_CR/fixtures   # strict
python tools/urban_bep_wrf461_oracle/compare.py BUILD_DIR_CR/fixtures                     # default
```

`WRF_SOURCE_ROOT` is wrf-model/WRF tag v4.6.1 (commit `d66e442f`); every file
read is checked against `SOURCES.sha256`. `CORE_MATH_REPO` is a clone of
https://gitlab.inria.fr/core-math/core-math.git (used only on the oracle host).

## What runs

* `run_bep461_{nlcd,lcz}`: `SUBROUTINE BEP` from the byte-unmodified
  `phys/module_sf_bep.F`, 4 steps, 100 (URBPARM.TBL) and 180 (URBPARM_LCZ.TBL)
  columns. dt 36 s / 12 s, solar declination +0.3752 / -0.3920 rad.
* `run_bep461_couple_{nlcd,lcz}`: the zeroing, BEP, then the Noah
  (`module_sf_noahdrv.F:1679-1776`) and Noah-MP
  (`noahmpdrv.F:3364-3373, 3690-3777`) coupling blocks and the surface
  diagnostic override (`module_surface_driver.F:3027-3031`), 2 steps. The
  blocks are cut from v4.6.1 by line range; `build.sh` checks they are
  byte-identical to the v4.7.1 cuts `tools/urban_wrf471_oracle` includes.

The columns (`bep461_cases.inc`) cross every urban class with sixteen
regimes (convective day, stable night, low sun windy, gridded mid-rise,
near calm noon, dawn, high terrain noon at 785 hPa, high basin night at
720 hPa, snow-season cold with an inversion, coastal gale, hot desert,
towers to 75 m, exactly zero wind, sun on the horizon, overcast, strong
night inversion) and eight morphology variants (class defaults, histogram +
hgt/lp/lb, towers to the 15th bin, histogram only, hgt/lp/lb only, the two
`param()` fall-backs, a rejected negative bin), plus four edge columns
(urban fraction 0, 1e-3, 0.02, 1).

## Two WRF arms

gfortran lowers REAL `acos/asin/atan/tan` to the C library's
`acosf/asinf/atanf/tanf`. On Ubuntu 24.04 (glibc 2.39) those are not
correctly rounded (about 7 % of random `acosf` inputs are 1 ULP off);
WOOF's own float32 routines are correctly rounded, and so was the C library
of the host that built the v4.7.1 oracles (glibc 2.43).

* `build.sh` alone: the host C library (the stock WRF build on that host).
* `build_crlibm_arm.sh`: the same objects linked with CORE-MATH's
  correctly rounded `acosf/asinf/atanf/tanf`. `compiler.txt` lists which math
  symbols each driver defines (T) or imports (U).

On this column set the two arms differ only on the regime-9 columns, where
`acosf(0.24f)` is 1 ULP off on glibc 2.39.

## Grading

`compare.py` runs four arms and compares every output word as a float32 bit
pattern: BEP from each step's recorded inputs, BEP free running from the
step-1 state, the coupling block end to end (`urban_bep.after_lsm`) for both
LSMs, and the surface-diagnostic override. It reports words compared, words
whose bits differ, max ULP and max |diff| per field.

WRF defect not copied: the coupling block divides the rural stress by the
level-1 wind speed, so on an exactly calm column WRF writes -inf drag and a
NaN `ust`. WOOF's calm guard leaves the directionless rural stress out and
takes `ust`'s zero-wind limit; `compare.py` counts those WRF words separately
(`calm_wrf_nonfinite_words`) and requires WOOF's to be finite.

Fixtures for both arms are committed as
`tests/data/oracles/urban/bep461/wrf461-bep-fixtures.tar.xz`; the gate is
`tests/test_urban_bep_wrf461_column_oracle.py`.
