# gwd_opt = 1 (KIM orographic drag) WRF v4.6.1 column oracle

`run_gwdo_kim.F90` calls SUBROUTINE gwdo of the pinned, byte-unmodified
`phys/module_bl_gwdo.F`, which calls `bl_gwdo_run` of
`phys/physics_mmm/bl_gwdo.F90` (WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`), with WRF's own
`module_model_constants` parameters, exactly as `module_pbl_driver.F:2136-2151`
passes them. It does no drag arithmetic of its own.

Both WRF files are byte-identical to WRF v4.7.1's (`SOURCES.sha256` here and in
`tools/terrain_drag_wrf471_oracle`), and so is the driver's call. What differs
from the v4.7.1 oracle is the platform: this one runs on the reference platform
of the 2026-10-07 physics verification (box W1: Ubuntu 24.04, gfortran 13.3.0,
glibc 2.39), the v4.7.1 one ran on gfortran 15.2 / glibc 2.43.

## Build and run (Linux, gfortran, CPU only)

```bash
tools/gwdo_kim_wrf461_oracle/build.sh "$WRF_TREE" "$BUILD_DIR"
```

- `make_columns.py` writes the inputs: 148 columns in 12 regimes (convective,
  stable night, near-neutral windy, blocking, high terrain, ocean, cold snow,
  tropical moist, jet shear, critical level, sector boundary, 16 edge cases)
  on two WRF-like eta grids (50 and 90 levels, p_top 50 hPa), six
  (dx, dt) cases: 1 km/6 s, 3 km/18 s, 9 km/54 s, 12 km/72 s, 30 km/180 s,
  50 km/300 s.
- The reference of record is compiled `-O0 -ffp-contract=off`: scalar
  `atan2f`/`powf`, no FMA. build.sh fails if an object pulls a libmvec `_ZGV`
  symbol or contains an FMA instruction.
- The same sources are also built with WRF's stock GNU physics flags
  (`-O2 -ftree-vectorize -funroll-loops`); on this fixture that build writes
  the same bytes as the -O0 one (build.sh reports it in `libmvec-report.txt`).
- It also builds `libsector_sweep.so` (the CPU half of `sector_sweep.py`).

## Grade the GPU kernel (under the GPU mutex)

```bash
GPUWM_WRF_EXACT=1 python tools/gwdo_kim_wrf461_oracle/compare_gpu.py "$BUILD_DIR"   # strict
python tools/gwdo_kim_wrf461_oracle/compare_gpu.py "$BUILD_DIR"                      # default
GPUWM_WRF_EXACT=1 python tools/gwdo_kim_wrf461_oracle/compare_gpu.py "$BUILD_DIR" --reference O2vec
GPUWM_WRF_EXACT=1 python tools/gwdo_kim_wrf461_oracle/sector_sweep.py "$BUILD_DIR"
GPUWM_WRF_EXACT=1 python tools/gwdo_kim_wrf461_oracle/branch_mutations.py
GPUWM_WRF_EXACT=1 python -m pytest tests/test_gwdo_kim_wrf461_parity.py
```

- `compare_gpu.py` runs the production launcher
  (`gpuwm.core.terrain_drag.launch_gwdo`) on every grid and case and prints,
  per output field, the words that are not bit-identical, the max FP32 ULP and
  the max absolute difference, plus how many columns WRF actually dragged per
  regime. `--write-fixture DIR` writes the `.npz` pairs the pytest gate reads
  from `tests/data/oracles/gwdo_kim/`.
- `sector_sweep.py` holds the kernel's `td_atanf` to this host's `atanf` on
  every float32 and its `td_atan2f` wind sector to this host's `atan2f` on
  44 million sector-boundary pairs, 1 million special pairs and 268 million
  random pairs. `--write-sample` stores the boundary sample and the pairs where
  this host's atan2f is not the correctly rounded angle
  (`tests/data/oracles/gwdo_kim/sector-sample.npz`).
- `branch_mutations.py` changes one gwdo_column statement at a time (flow
  blocking, wave breaking, the dtfac limiter, every clamp, every drag-off
  test) and counts the WRF words the oracle then misses: a control the oracle
  catches is a branch the columns reach.
