# WRF v4.7.1 vertical diffusion and TKE oracle

The harness calls byte-preserved WRF subroutine bodies with WRF C-grid
dimensions, staggering and three outside halo cells. It compiles WRF's actual
`module_model_constants.F`. The configuration type and C ABI are service code;
they do not replace a physical calculation.

```
python tools/wrf_diffusion_oracle/vertical_build.py WRF_SOURCE BUILD_DIR
python tools/wrf_diffusion_oracle/vertical_capture.py WRFINPUT RUST_NETCDF DEFORMATION_SO VERTICAL_SO FIXTURE_DIR --evolved WRFOUT_19Z
python tools/wrf_diffusion_oracle/vertical_compare.py FIXTURE_DIR RESULTS_JSON
python -m pytest -q tests/test_vertical_diffusion_wrf471_parity.py
```

The source builder checks the pinned WRF v4.7.1 source hashes. The source
receipt records the exact routine body hashes and original line ranges,
compiler flags, constants, wrapper and library. The fixture receipt records
the source data and Rust reader hashes. Fixture files and every measured GPU
output array have SHA-256 pins. Tests reproduce the full word arrays and the
exact word-difference statistics. A numerical tolerance does not hide a
difference.

Four vertical leaves are isolated using identical WRF coefficient inputs:
`vertical_diffusion_u_2`, `vertical_diffusion_v_2`,
`vertical_diffusion_w_2` and `vertical_diffusion_s`. The scalar leaf covers
water vapor, theta and doubled TKE self-diffusion. All fourteen input cases
retain every output and boundary word. Seven start from an actual WRF
initialization-state crop. Seven supplement those thermodynamic fields with
actual one-hour WRF C-grid winds. Those seven are explicit flow operator
probes and do not claim to be complete evolved-state snapshots.

The complete-output `vertical_driver.py` packet also calls the composite
`vertical_diffusion_2` driver for `km_opt=2` and `km_opt=4`, each with all three
`isfflx` values. All fourteen input cases give 84 driver cases. Every active
tendency and mutable input array is measured, including inactive slots and
all nine NBA arrays. The original periodic 48-case driver outputs remain in
the leaf packet for continuity.

Open-boundary composite surface forcing has a caller distinction: WRF's raw
driver writes prescribed surface fluxes on outer rows, whereas the engine
mixing package clears outer dry and moisture rows after the vertical launch.
The periodic composite cases avoid conflating that caller policy with a
surface-flux formula. Open rows remain covered by the isolated leaves and TKE
source regressions. Legacy constant-K routines are covered by the constant-K
oracle tools. There is no matching engine path for WRF's implicit vertical
driver in this scope.

`tke_shear`, `tke_buoyancy`, `tke_dissip` and `tke_rhs` are called directly.
The first three exports are the cumulative tendency after each WRF call;
`tke_rhs` is independently called from zero. The GPU exports those cumulative
states from the production budget increments, so reconstruction rounding is
included in the reported differences. The source matrix contains all three
surface switch values, zero TKE and coherent positive TKE profiles.

Two engine defects found by the compiled WRF oracle have exact boundary
regressions: the RHS source
previously advanced the excluded outer mass rows, and vertical TKE
self-diffusion previously entered those rows. The RHS now excludes those
rows. Vertical self-diffusion is masked before adding its tendency and
recording the budget term. Both fixes are default-on.

For `km_opt=2/3`, the composite W tendency retains a declared coefficient
divergence. WRF's driver supplies horizontal `xkmh`; the engine supplies
vertical `xkmv` to the vertical `tau33` stress. The launcher docstring in
`gpuwm/core/dycore.py` declares this choice. The coefficients are equal for
`km_opt=4`. The leaf oracle uses identical input coefficients; the composite
oracle preserves WRF's own driver choice and records the resulting difference.

Three tools-only controls attribute numerical differences: contraction off,
compiled WRF metric words with contraction off, and WRF metric words plus
the scalar glibc power used by the TKE dissipation expression. Their receipts
are marked diagnostic. They do not change the production fixture or GPU pin.
The production source includes on-demand geopotential metric arithmetic,
fused multiply-add contraction, a multiply/square-root dissipation power and
CUDA subnormal flushing. The receipts retain the full measured differences
and their exact input witnesses.
