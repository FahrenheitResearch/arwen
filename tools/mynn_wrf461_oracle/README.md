# WRF v4.6.1 MYNN surface-layer oracle

This harness compiles the unmodified NCAR WRF v4.6.1
`phys/module_sf_mynn.F` and calls its public `SFCLAY1D_mynn` entry point. The
source must be commit `d66e442fccc04111067e29274c9f9eaccc3cef28` with SHA-256
`86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3`.

`run_surface_layer.F90` writes the original six-column fixture
`surface-layer.csv`. That file is byte-pinned by
`gpuwm/core/noahmp_mynn_contract.py` (8,600 bytes, SHA-256
`049caf3b…72ff9b04`) and neither it nor its harness may be regenerated. The
six deterministic columns cover stable, unstable, and neutral land;
snow-covered land; and stable/unstable water.

`run_surface_layer_wide.F90` is a separate, additive harness that drives two
entry points of the same pinned source and writes two more CSVs:

* `surface-layer-wide.csv` -- `SFCLAY1D_mynn` over ten columns, advanced
  across two successive timesteps carrying UST/MOL/QSFC/ZNT/USTM, plus an
  `ISFFLX=0` pass. It covers what the narrow fixture missed: REGIME=1
  (BR>0.2, both unclipped and clipped), the `itimestep>1` ZOL-from-MOL first
  guess, a land column entering with QSFC<=0 (which separates WRF's `:532`
  pre-update predicate from its `:573` post-update one), the water ZNT that
  `charnock_1955` rewrites in place and that must persist to the next step,
  the `ZA<=7` and `7<ZA<13` 10 m diagnostic branches, and the TH2 bracketing
  fallback.
* `surface-layer-wrapper.csv` -- the `SFCLAY_mynn` wrapper at `itimestep`
  1 and 2, entered from WRF's `module_physics_init.F` cold start
  (`UST=1e-4`), so its `:329-337` UST/MOL/QSFC/qstar seeding block is
  observable in the recorded outputs. `wstar`/`qstar` are wrapper locals and
  are therefore not reported for this entry point.

`validate_wide_oracle.py` imports the gpuwm CPU reference under `PYTHONPATH`,
asserts parity per output column with max_abs/max_rel/max_ulp receipts, and
fails closed if any of the branches above goes dead. These are
official-source oracles, not replacement implementations and not evidence
that MYNN 5/5 is executable in gpuwm.

Run on Linux with GNU Fortran:

```sh
./build.sh /path/to/WRF-v4.6.1 /new/build-directory
```

The build refuses an existing output directory, validates the source commit
and bytes before compiling, enables Fortran bounds/floating-point traps, runs
both validators, and writes compiler and SHA-256 receipts beside the three
CSVs. Re-running must leave `surface-layer.csv` byte-identical; if it does
not, the pinned contract has drifted and the build is not trustworthy.

`run_surface_layer_water.F90` is a third additive harness, for the over-water
`ISFTCFLX` branches. It drives `SFCLAY1D_mynn` and, separately, seven of the
module's own roughness subroutines at their public entry points, writing:

* `surface-layer-water.csv` -- twelve columns x `ISFTCFLX` 0/1/2/3 x two
  timesteps. Nine columns are water and span u* from 0.01 to 9 m/s;
  `control_land` and `control_snow_land` must not move with `ISFTCFLX` at all;
  `xland_exactly_1p5` sits where `:625` (`.GE. 0`), `:1065`/`:1073`
  (`.GT.`/`.LT.`) and `garratt_1992`'s own `landsea-1.5 .GT. 0` disagree about
  whether the column is water.
* `surface-layer-water-leaf.csv` -- `charnock_1955`, `edson_etal_2013`,
  `davis_etal_2008`, `Taylor_Yelland_2001`, `fairall_etal_2003`,
  `fairall_etal_2014` and `garratt_1992` (at landsea 2.0, 1.5 and 1.0) over 32
  samples chosen to bind every clamp and internal arm they have. This is the
  only oracle `edson_etal_2013` and `fairall_etal_2014` can have: the
  compiled-in `COARE_OPT=3.0` at `:85` makes them unreachable through
  `SFCLAY1D_mynn`, and editing that parameter would stop the module being the
  unmodified one.

`validate_water_oracle.py` asserts the gpuwm CPU reference against both files
-- `max_ulp 0` on every leaf row, the measured three-platform union imported
from `tests/test_mynn_surface_water.py` on the columns -- and fails closed if
any branch stops being bound by a case.

Run on Linux with GNU Fortran:

```sh
./build_water.sh /path/to/WRF-v4.6.1 /new/build-directory
```

`build_water.sh` validates the same pinned commit and source hash as
`build.sh`, records `nm -u module_sf_mynn.o` and refuses to continue if the
object binds libmvec (`_ZGV*`), because the `max_ulp 0` leaf claim is against
scalar glibc `expf`/`logf`/`powf`.
