# Grell 3D cumulus oracle, WRF v4.6.1 (`cu_physics = 5`)

Drives the byte-unmodified `phys/module_cu_g3.F` (sha256
`68089efd157753e8d050c19d4e65f711838147381639014f5f0be40576a337e9`) the way
`phys/module_cumulus_driver.F` does for a single-tile, single-patch,
non-periodic ARW run: `G3DRV` over a whole domain, then `conv_grell_spread3d`
on the same arrays.

```
bash tools/g3_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR> [tests/data/oracles/g3]
```

## What it proves about itself

* **One file, no dependencies.** `module_cu_g3.F` uses no other module; the
  object's undefined symbols are libm and libgfortran only (`undefined-O0.txt`).
* **The cpp defines match WRF's.** `-DEM_CORE=1` is required: G3DRV's
  high-resolution forcing average sits inside `#if ( EM_CORE == 1 )`, and a
  build that forgets it compiles the block out silently.
* **Three builds, one answer.** The reference (`-O0`), the same source at the
  scalar WRF tree's FCOPTIM (`-O2 -fno-fast-math -ffp-contract=off
  -fno-tree-vectorize`), and the preprocessed `module_cu_g3.f90` the scalar
  WRF build actually compiled (when `WRF_BUILD_F90` names it) must write the
  same words on every configuration, or the build fails.
* **One uninitialised read, pinned and fenced.** With `ishallow = 0`, G3DRV
  writes `K22_SHALLOW`/`KBCON_SHALLOW`/`KTOP_SHALLOW` from locals that
  `CUP_enss_3d` only assigns inside the shallow block (`:487-490` against
  `:1223`). Measured: the `-O0` and `-O2` builds write different stack words
  there. The reference pins every uninitialised local to zero (G3DRV's own
  "no shallow cloud" value, which the port implements as a declared
  divergence), and `ub_control.py` runs a build that poisons every
  uninitialised local and fails unless the ONLY words that move are those
  three fields, in the band, under `ishallow = 0`.

## The fixture

`gen_g3_cases.py` builds seven 16 x 16 x 40 domains (configurations `a`-`g`):
both shallow arms, `cugd_avedx` 1 and 3, closure choices 0, 5, 7 and 13, day
and night caps, with and without prognostic ice, dx from 3 to 27 km. Column
families: deep land and ocean convection, shallow-only, capped, dry, high
terrain, moist-neutral maritime and ensemble edge cases. The graded band is
the 8 x 8 interior WRF writes (448 columns); every band column sits on the
3 x 3 (5 x 5 under `cugd_avedx = 3`) stencil G3DRV reads its forcing ensemble
from, so the stencil and `conv_grell_spread3d`'s spreading and smoothing are
inside the check. Packed by `pack_g3_fixture.py` into
`tests/data/oracles/g3/g3-<cfg>.npz` (inputs `in_*`, outputs `out_*`, raw
float32 words). `g3_fixture_io.py` is the one reader of the binary layout.


Continuation coverage: all closure selectors 1 through 16 join the original
seven configurations, plus maxens=4/maxens2=2/maxens3=17/ensdim=136 and
maxiens=2/ensdim=288. G3DRV itself sets iens=1, as WRF does; the larger
ensdim participates in its closure statistics and untouched outer slots.
The graded band now contains 1600 columns. The dimension minima come from
WRF's literal references to cap member 3, closure member 16 and edtc member
2 (member 3 in the high-resolution arm). A short array is refused by its
concrete out-of-bounds reference.

`diag_oracle.py WRF_ROOT BUILD_DIR FIXTURE_DIR` separately compiles the
byte-unmodified dyn_em/module_convtrans_prep.F. It checks fixed and adaptive
12-step windows over 64 columns, including the first window, multiple reset
boundaries, water/ice cloud fraction and the accumulators. O0 and scalar O2
outputs must match before fixtures are packed. The kernel's counter is held
per domain rather than shared across domains by a procedure-local SAVE.

Periodic coverage adds x, y and combined boundary configurations (2112
written columns across 28 domains). The oracle fills the tendency halo with
periodic words before calling the unchanged public routine. The kernel reads
those same words; the driver forwards each domain's open boundary flags.
Both arithmetic modes use the shared direct-NVRTC cache with contraction and
FTZ disabled, and require 0 ULP on columns and diagnostic windows.

`tests/test_g3_halo.py` exercises uneven x/y logical partitions through the
existing seam transport on one card. A stale-halo control changes retained
outputs; the repaired transport must restore every output word. Separate
physical GPU peer links are outside that test.
