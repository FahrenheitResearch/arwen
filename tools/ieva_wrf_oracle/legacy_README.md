# Earlier WRF implicit vertical-advection reference

The `wrf_legacy` variant follows the implicit routines distributed in
NOAA-EMC/HRRR at commit `40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`, under
`sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em`. `legacy_build.py` checks the SHA-256
of `module_advect_em.F`, `module_big_step_utilities_em.F`, `module_em.F`
and `solve_em.F` before building. The WRF notice is reproduced in the
root `NOTICE` and `licenses/LICENSE-WRF-public-domain.txt`.

The native reference extracts the original `TRIDIAG2D`, `WW_SPLIT` and
five implicit-advection routines. It expands their `HYBRID_COORD=1`
mass macros as written in the source. Numerical expressions otherwise
remain the source's. A minimal configuration type supplies only the
consumed fields; logging is inert. This is an operator oracle, not a
complete native WRF integration.

Two native binaries are built. One preserves the original routines. The
other applies the already declared A179 lower w-boundary correction:
coupled u/v tendencies are uncoupled before constructing the surface w
increment. The original earlier upper boundary already divides the whole
increment by gravity, and its FP32 grouping is retained. The correction
must leave every output except the w tendency byte-identical. Both w
answers are retained in the fixture, and the CPU test checks that changed
interior columns have exactly the active lower-boundary mask.

The earlier variant differs materially from WRF 4.7.1:

- Its splitter uses `alpha_max=1.0` and a directional horizontal-flux
  estimate including the mass-point map factor. Modern WRF uses 1.1 and
  the magnitude of the centered horizontal wind.
- Its theta and momentum solves use current stage mass in both the
  coefficients and old-field terms. Modern WRF uses estimated new mass
  and old mass respectively.
- Its scalar splitter reads post-acoustic current winds, as
  `solve_em.F:2404` passes `u_2` and `v_2`. Modern WRF reads time-n winds.
- Its scalar solve uses current post-acoustic mass in both places.

Both variants activate only on the final RK3 stage. Dynamics
`module_em.F:473` reduces the full timestep by the remaining stage count,
and scalar `module_em.F:1217` reconstructs the full timestep for the
split. On the final stage both equal the full timestep. The call sequence
is explicit u/v advection then their implicit solves, explicit w and
theta advection, theta solve, `rhs_ph` on explicit flux, geopotential solve,
w solve, pressure gradient and buoyancy. Scalar advection and its limiter
see explicit flux before the implicit scalar solve.

The retained fixture is 9 by 8 by 50. `legacy-coordinate.json` carries the
51 eta values of the runbook grid, hybrid option 2, transition eta 0.2 and
source model top 5000 Pa. Its hybrid coefficients and surface weights are
the engine grid builder's. All horizontal data are synthetic, with
nonuniform map factors and terrain and both signs of implicit flux.
The native routines consume these exact input words. No full-forecast or
observation-skill claim follows from this comparison.

Rebuild with gfortran and NumPy, with `TMPDIR` inside owned scratch:

```sh
python tools/ieva_wrf_oracle/legacy_build.py SOURCE BUILD
python tools/ieva_wrf_oracle/synth.py FIXTURE 284 50 \
  tools/ieva_wrf_oracle/legacy-coordinate.json
python tools/ieva_wrf_oracle/legacy_pack.py BUILD FIXTURE \
  tests/data/ieva_wrf_legacy.npz
```

Run GPU replay under the applicable ownership protocol:

```sh
python -m pytest -q tests/test_zadvect_implicit.py tests/test_zadvect_legacy.py
python tools/ieva_wrf_oracle/legacy_measure.py RECEIPT.json
```

The test gates nonzero words exactly and reports signed zero separately,
matching the existing WRF oracle convention. The measurement records both
counts, maximum ULP distance and SHA-256 of every compared field. The
forced outer w ring is excluded because its terrain slope follows the
engine's existing one-sided boundary convention. Full across-card byte
identity is a separate tiled-dycore gate and includes signed zero.
