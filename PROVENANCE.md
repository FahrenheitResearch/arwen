# gpuwm oracle provenance and deviation register

The transcription authority for every WRF-derived mechanism is WRF v4.6.1
(upstream <https://github.com/wrf-model/WRF.git>, tag `v4.6.1`, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`). A verified checkout of that
tree is the source-reading and line-citation authority; its machine-local
staging path is deliberately not part of the contract. The optional real74
case bundle is selected with `GPUWM_REAL74_REFERENCE_BUNDLE` in
`gpuwm/verify/cases/real74_d01.py`. The WRF source tree is **not**, by itself,
the binding numerical oracle. Scheme-data provenance and scheme-local
deviations live next to their data (`gpuwm/data/*/PROVENANCE.md`); this
file is the repo-level register for cross-module numerical authority,
arithmetic policy, deliberate deviations, and fail-loud scope cuts.

## Native GFS direct-WRF adapter evidence (2026-07-20)

- **Source**: NOAA/NCEP GFS `pgrb2.0p25`, cycle 2026-07-20 00Z, f000 and
  f003, fetched from the NOMADS `filter_gfs_0p25.pl` service for
  20..55 N and 110..55 W.  Input SHA-256 values are
  `24bf5b5b302962ed66da502e67a91b62120714a76ee4689c648c91539fb688e9`
  (f000) and
  `9ada4e2f8e086773417236781b2211d9a6007f4350dc7ffb8eed0104d50b0f40`
  (f003).  The complete input-manifest SHA-256 is
  `8d4f66f25f181f86611581d057115a1f70730f245f8aa8c777210431ae7eb548`.
- **Mapping authority**: WRF v4.6.1 `WPS/ungrib/Variable_Tables/Vtable.GFS`
  for GFS field semantics, plus `share/module_soil_pre.F` for Noah soil
  initialization.  The adapter uses GFS RH as WPS supplies it, declares
  initial hydrometeors zero (the stock GFS Vtable has none), copies the four
  exact GFS/Noah soil slabs without ERA5-style interpolation, and retains
  bitmap missingness through land-aware horizontal selection.
- **Decoder gate**: `tools/grib1_bridge/src/bin/gfs_grib2_bridge.rs` uses raw
  discipline/category/parameter codes, requires all five atmosphere fields
  at each of 21 pressure levels from 1000 through 100 hPa, and validates both
  fixed surfaces for every soil layer.  The certified slice permits only GRIB2
  simple packing (DRT 5.0), binds process identifier 81 at f000 and 96 for
  forecasts, and rejects wrong cycle/time/grid, duplicates, gaps, nonuniform
  cadence, stale or malformed bitmaps, nonfinite required values, and
  out-of-contract ranges before atomic publication.  Source files are decoded
  once and the proof binds source, inventory, decoded-product, implementation,
  git commit, and git tree hashes.
- **Horizontal-policy scope**: atmospheric fields use the native interpolation
  contract, while masked surface and soil fields use land-aware nearest donors
  and a globally proven nearest-water donor for model lake cells.  This is not
  a claim of numerical equivalence to WPS METGRID's four- and sixteen-point
  masked interpolation.  The unchanged-WRF gate below proves file structure,
  acceptance, and a stable smoke step for this exact slice, not WPS bitwise or
  numerical parity.
- **Native runtime**: clean git commit
  `528e9d795da1090b239ff29367f463c52866e422` (tree
  `752ecac00c6964c8573e4d89935e6bfb394765ec`) completed decode,
  initialization, prepared-cache publication, and direct export in
  21.559 seconds internally (22.57 seconds process wall time).
- **Stock-WRF gate**: native initialization/direct export produced
  `wrfinput_d01` SHA-256
  `064b216010c0ed43ad06ac197b82cd3dbfe884890251e0ebac30a4405fe387f3`
  and `wrfbdy_d01` SHA-256
  `c74c5b9636f0e5a5cb54dbd9366d86a221ecbeb9a203fbbd6a3c0d39adc923f8`.
  The unchanged stock WRF v4.6.1 executable SHA-256
  `f0fb585bf37b72fbdcece562047934cb8386db3958f153d6e4e6876e5fd997ac`
  accepted both, advanced to 2026-07-20 00:00:05, printed success, and wrote
  a finite history file.  Acceptance evidence SHA-256:
  `b8fca75fc263b23699b17ccadaf4a122199d92f460c83fbae3a8def702c05cc3`.

## Binding numerical-oracle manifest

The binding numerical oracle is the following pinned instrumented build,
not an arbitrary executable produced from the WRF v4.6.1 tree:

- **Source/build recipe**: the source tree above, copied read-only into the
  disposable WSL build tree and built by
  `.superpowers/sdd/codex/wrf-build-spike-report.md`.  The environment is
  WSL `Ubuntu-24.04` (recorded as Ubuntu 24.04.4 LTS), GNU Fortran/GCC
  13.3.0, netCDF-Fortran 4.5.4 (netCDF-C 4.9.2), WRF's serial/STUBMPI
  execution (no MPI processes), no OpenMP (the generated OMP flags are
  commented out), one tile, and the `em_real` target.  Configure used the
  displayed GNU serial option 32 plus basic-nesting option 1 with
  `NETCDF=/usr` and `WRFIO_NCD_LARGE_FILE_SUPPORT=1`; generated
  `configure.wrf` supplied `-fallow-argument-mismatch` and
  `-fallow-invalid-boz`.  There were **no hand edits** to `configure.wrf`.
  The recipe's mandatory prerequisite normalized CRLF endings in the
  disposable copy (4,838 text files), never in the read-only bundle.
- **Effective oracle case**: `tools/wrf_instrumented/namelist.input.n1p5`,
  the d01+d02 one-outer-step case with Morrison (`mp_physics=10`), YSU
  (`bl_pbl_physics=1`), and revised-MM5 surface layer
  (`sf_sfclay_physics=91`) as the matched-physics scheme selection.  The
  case itself explicitly sets `use_theta_m = 0` in `&dynamics` and
  `nwp_diagnostics = 0` in `&time_control`.  The first setting is
  load-bearing: WRF's default is `use_theta_m=1`
  (`Registry.EM_COMMON:2860`), under which the dumped `t` table family is the
  moist-theta `thm` state (`Registry.EM_COMMON:211`), differing from gpuwm's
  dry-theta consumers by approximately the `(1 + 1.61*qv)` field factor.  The
  second
  setting keeps the case inside D2's output-due diagnostic scope.
  `tools/wrf_instrumented/instrument-med-force.patch` is the exact
  instrumentation applied after the pristine source-file SHA check; it
  captures the output boundary tables, the bracketing `PROG` samples,
  the solve-side `DBDY` clock records, and (p5n15 extension) the 25
  complete pre-coupling `INPT` input arrays per domain that make an
  independent candidate producer possible.  The p5n15 review audited
  every hunk as insertion-only/write-out-only: zero changes to any
  computed value, loop bound, or call ordering in the integration.
- **Reference dump status**: the binding pin lives in ONE place,
  `tools/wrf_instrumented/n1p5-dump.sha256`, currently
  `e4a930daac3f7295eef9e6d9e31a885d6e776a713d128c8ccb8b0a115805e060`
  (844,428,012 bytes; schema records: legacy tables + 4 `PROG`
  prognostic-diagnostic samples + 50 full pre-coupling `INPT` input
  arrays + 4 `DBDY` boundary-clock samples).  Produced by the p5n15
  worker lane and BYTE-REPRODUCED by the controller's independent WSL
  regeneration from pristine sources + the committed patch (identical
  SHA-256); the p5n15 full-scope review additionally proved the patch
  insertion-only/write-out-only hunk-by-hunk.  Superseded lineage,
  retained for provenance only: (1)
  `049ece3d96bb912d094304e2616b2e7021f9ea0ed815da280d95eaead360ebd3` —
  ran `nwp_diagnostics=1` and omitted `use_theta_m` (Registry default 1),
  so its `t` tables are moist-theta state, not binding for gpuwm's
  dry-theta consumers; (2)
  `441da3cba30fd1fb958488ba55c7e3aa0e5b11854a92666a3ba73f020f8022cc` —
  corrected case but diagnostics-only schema (12 scalar `PROG` samples
  per domain), which cannot drive an independent nest-force candidate
  producer; any producer built on it would be tautological against the
  dumped output tables.  The instrumented executables (controller
  oracle build tree, outside this repository) have SHA-256
  `f0fb585bf37b72fbdcece562047934cb8386db3958f153d6e4e6876e5fd997ac`
  (`wrf.exe`) and
  `4b47e80e55009144410b5530e0bbbcd1a5426b427716301a078a39af7b6f4622`
  (`real.exe`), rebuilt from pristine + patch by the controller's
  verification run (the retired diagnostics-era `wrf.exe` was
  `e4ac817bd0dc1bb5dde85c76bd24f7a9674613c014e2452a94ecfde255bb9932`).

After the controller re-pins the regenerated reference dump,
numerical-fidelity comparisons are against dumps made by **this build** and
case and apply only the tolerances registered in the gate ledger
(`gpuwm/verify/nest_gates.py`; N1.5 currently registers the FP32-relative
`1.0e-6` table band).  A “bitwise” statement is always scoped to its named
comparison—for example a kernel/mirror seam, restart identity, or a
controlled A/B refactor/feature-inertness comparison.  It is never a claim
against arbitrary WRF builds.  In particular, no gpuwm-versus-WRF output
file byte-identity claim is made; file-format, metadata, and I/O-library
bytes are outside the numerical oracle.

## Arithmetic contract

- **Compilation**: the COMMON CuPy `RawModule` loader
  (`gpuwm/core/kernels/__init__.py`) passes only `-std=c++17`; it never
  enables `--use_fast_math` and leaves FMA contraction at NVRTC's default
  `--fmad=true`, so contraction is permitted.  CPU-mirror comparisons for
  those COMMON-loader kernels therefore sit behind registered tolerance
  seams unless explicit `__*_rn` intrinsics pin operation boundaries.  Only
  the dedicated nest/coupler loader in `gpuwm/core/nest_interp.py` passes
  `-fmad=false`, so SINT/TR4, boundary interpolation, terrain adjustment,
  and dormant feedback arithmetic do not contract multiply/add pairs.
<!-- BEGIN GENERATED ftz-statement: provenance-compile-policy (tools/ftz_receipt/render_statement.py) -->
- **Rounding and subnormals**: ordinary FP32/FP64 operations and conversions
  use round-to-nearest-even.  FP32 subnormal handling is not one policy: it
  differs between the compile routes this codebase uses, so it is recorded
  per route, as measured on NVIDIA GeForce RTX 5090 (compute capability
  12.0, driver 13.3 (13030), NVRTC 13.0,
  CuPy 14.0.1) by `tools/ftz_receipt/probe.py` over
  6 arithmetic mechanisms:
  - `R1` loader RawModule (`gpuwm/core/kernels/__init__.py:20`,
    gpuwm.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`]
  - `R1-ftztrue` loader RawModule + explicit --ftz=true (control)
    (`gpuwm/core/kernels/__init__.py:20 + control flag`, cupy.RawModule),
    effective NVRTC options `-std=c++17` `--ftz=true` `-ftz=true`:
    `flush-to-zero` on 6 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1_ftztrue.sass`]
  - `R2` RawModule with the shortwave option tuple
    (`gpuwm/core/rrtmg_sw.py:2890`, cupy.RawModule), effective NVRTC options
    `-std=c++17` `--ftz=false` `-ftz=true`: `flush-to-zero` on 6 of 6
    mechanisms [disassembly `tools/ftz_receipt/receipt/sass/r2.sass`]
  - `R3` direct NVRTC + cuda.function.Module (`gpuwm/core/rrtmg_lw.py:3723`,
    cupy.cuda.compiler.compile_using_nvrtc), effective NVRTC options
    `-std=c++17` `--ftz=false` `-arch=compute_120`: `ieee-agreement` on 6 of
    6 mechanisms [disassembly `tools/ftz_receipt/receipt/sass/r3.sass`]
  - `R4` CuPy-generated ReductionKernel (`gpuwm/core/mynn_pbl_gpu.py:290`,
    cupy.ReductionKernel), effective NVRTC options `--std=c++17`
    `-ftz=true`: `flush-to-zero` on 6 of 6 mechanisms [disassembly of 6
    objects, `tools/ftz_receipt/receipt/sass/r4_0.sass` and siblings]
  - `R5` inline PTX without .ftz (`gpuwm/core/kernels/__init__.py:20`,
    gpuwm.core.kernels.load_module), effective NVRTC options `-std=c++17`
    `-ftz=true` (this route rides `R1`'s compile): `ieee-agreement` on 4 of
    6 mechanisms; `not-applicable` on 2 of 6 mechanisms [disassembly
    `tools/ftz_receipt/receipt/sass/r1.sass`, the object `R1` compiled]
  The option tuple each row names is the tuple NVRTC received, captured by
  wrapping the compiler entry point rather than re-deriving it: CuPy appends
  `-ftz=true` to whatever the caller passed, at
  `cupy.cuda.compiler` line 585 (`options += ('-ftz=true',)`), after the
  caller's options, and NVRTC honours the last occurrence.
  The inventory records 5 distinct caller-supplied option tuples across the 14
  compile sites in the shipped package, each listed here with a site that
  supplies it:
  - no caller options -- `gpuwm/core/mynn_pbl_gpu.py:283`
    (cp.ReductionKernel), and 6 other site(s)
  - `-std=c++17` -- `gpuwm/core/kernels/__init__.py:20` (cp.RawModule), and
    2 other site(s)
  - `-std=c++17` `--ftz=false` a target flag built from the device
    architecture at run time -- `gpuwm/core/rrtmg_lw.py:3723`
    (_cc.compile_using_nvrtc), and 1 other site(s)
  - `-std=c++17` `--ftz=false` -- `gpuwm/core/rrtmg_sw.py:2890`
    (cp.RawModule)
  - `-std=c++17` `-fmad=false` -- `gpuwm/core/nest_interp.py:259`
    (cp.RawModule)
  `R5` and `R1` are kernels inside ONE compiled object -- same device, same
  flags, one compile -- and they did not measure alike, so on this device
  the outcome follows the instruction the compiler emitted rather than the
  hardware alone.
  Evidence: the device bit table `tools/ftz_receipt/receipt/bitpatterns.csv` (both
  passes byte-identical: true), the objects the routes themselves compiled,
  `tools/ftz_receipt/receipt/cubin/`, and their disassembly,
  `tools/ftz_receipt/receipt/sass/` (Cuda compilation tools, release 13.0,
  V13.0.39), all recorded in `tools/ftz_receipt/receipt/receipt.json`.
<!-- END GENERATED ftz-statement: provenance-compile-policy -->
- **Operation boundaries**: explicit parity-critical kernels use
  `__fadd_rn`/`__fmul_rn`/related intrinsics where an operation boundary
  itself is part of the seam.  NVRTC's documented option defaults are
  `--ftz=false`, `--prec-div=true`, and `--prec-sqrt=true` (see the NVIDIA
  [NVRTC compile-option reference](https://docs.nvidia.com/cuda/nvrtc/index.html#supported-compile-options));
  what the compiler defaults to and what a given route ends up asking for
  are separate questions, and the measured answer to the second is the
  generated block above.
- **Reduction order**: reductions that mirror serial WRF loops are serial
  within each CUDA work item and retain the WRF iteration order.  One
  checkable example is the two `n=0..49` wet-snow/wet-graupel sums in
  `gpuwm/core/kernels/refl.cu`, which accumulate the WRF 50-bin terms in
  order rather than using a parallel reduction tree.
- **Transcendentals**: device `exp`/`log`/`pow` results are **not** claimed
  bit-equal to gfortran's host math library.  Every consumer that matters
  to a fidelity claim terminates at a registered tolerance seam.  D2's
  density-offset pin is the local pattern:
  `tests/test_refl.py::test_prepared_pressure_seam_has_predicted_dry_dbz_offset`
  uses an absolute `5e-8` tolerance rather than a libm bit-identity claim.
- **Classic Thompson coefficient tables**: table-dependent native CUDA
  kernels consume the validated `ClassicTableSet.to_device` result.  In
  particular, rain-cloud accretion requires canonical `t_Efrw(100,100)`, and
  cloud-ice-to-snow autoconversion requires canonical `tps_iaus(64,55)` and
  `tni_iaus(64,55)`.  All are complete FP64, Fortran-contiguous WRF arrays.
  The production path has no WRF runtime dependency; WRF v4.6.1 is used only
  to generate and hash-pin the canonical external table asset and direct
  column fixtures.  The focused `warm-accrete` and `ice-auto` gates sparsify
  only their known-active table entries after verifying them against that
  asset, so they do not define alternate production tables.
- **Classic Thompson rain-evaporation ordering**: the ordinary
  Srivastava-Coen process updates temperature and vapor before sedimentation,
  while WRF retains the pre-evaporation volumetric rain mass and number for
  that same step's fallout.  The staged CUDA composition records the original
  FP32 density into an explicit scratch field and supplies it to the admitted
  rain-sedimentation kernel; fallout velocities still use the updated
  environmental density.  This is a numerical-ordering seam, not an external
  WRF runtime dependency.

## Deliberate runtime deviations from WRF v4.6.1

### D1 — retired real74 compatibility-mode microphysics/h_diabatic cadence

- **WRF-native cadence**: the moist-physics prep/driver/finish block runs
  once per model step with the model-step `dtm` (solve_em.F:3604-3616
  description; prep call :3661-3672, driver :3689+, finish :4082-4094),
  and the retained heating `h_diabatic = mpten/dtm`
  (module_big_step_utilities_em.F:5745) is consumed by the NEXT model
  step's RK tendencies (rk_addtend_dry, module_em.F:1076-1080) — a
  one-model-step lag.
- **Thermodynamic scope**: gpuwm's `h_diabatic` transcription is the
  non-moist-theta (`use_theta_m=0`) branch in
  `gpuwm/core/microphysics.py`.  The moist-theta branch changes the stored
  thermodynamic variable and the heating conversion
  (module_big_step_utilities_em.F:5505/:5735); it is not implemented.
  `gpuwm/namelist_import.py` therefore rejects `use_theta_m=1` loudly,
  including WRF's Registry default of 1 when the key is omitted.
- **Retired gpuwm compatibility mode**: before the native-dt ratification,
  the real74 compatibility integrator
  (`gpuwm/verify/cases/real74_d01.py` `phase3_integration_config`,
  :161-163) replaced `dt = 60 s` with `DYNAMICS_SUBSTEPS = 8` uniform
  internal dynamics steps of `dt = 7.5 s` per 60 s WRF model-clock
  interval (`clock_dt = 60`; inner loop :1044-1048).  In that mode,
  `dycore.step` ran the full microphysics bracket once per INTERNAL step
  (gpuwm/core/dycore.py, post-RK3 microphysics slot), so the heating was
  captured as `mpten/7.5`, the clamp was `mp_tend_lim*7.5` per call, and
  the lag was one 7.5 s internal step — not one 60 s clock step.
- **Why this is self-consistent (and not the W0AVG defect class)**: no
  WRF-clock-referenced constant enters the mechanism.  The capture dt and
  the apply window are the SAME internal step everywhere: each internal
  step's RK loop integrates the previous internal step's rate, removes
  exactly what it added (small_step_finish removal,
  module_small_step_em.F:416-426 mirrored at
  `dycore._finish_small_steps`), and applies the fresh microphysics
  increment once.  Conservation and the no-double-count invariant hold
  per internal step by the same algebra as WRF's per-model-step form.
- **Current enforcement**: the frozen real74 profile pins
  `DYNAMICS_SUBSTEPS == 1`; `phase3_integration_config` rejects any other
  value.  The experiment schema exposes no dynamics-substeps knob and no
  other `gpuwm` package code reads that frozen-profile constant.  At the pin,
  `cfg.dt = clock_dt = 60`, so the internal step is the model step and the
  capture/apply/lag cadence is WRF-native.  The constant and no-reader
  invariant are pinned in `tests/test_real74_d01.py`.
- **Unsupported configuration**: any path with `DYNAMICS_SUBSTEPS != 1`
  is rejected.  The historical behavior was conservation-consistent but
  not WRF-equivalent.

### D2 — REFL_10CM is computed on output-due microphysics steps only

- **WRF-native seam**: `refl_10cm` is computed INSIDE the microphysics
  driver call on history steps (`diagflag .and. do_radar_ref == 1`,
  phys/module_microphysics_driver.F `refl_10cm` argument; Morrison wrapper
  phys/module_mp_morr_two_moment.F:911-918).  The wrapper passes the
  scheme's post-call `t1d` and moments with the unchanged prepared `p1d`
  (:730, :780, :913-914), before `moist_physics_finish_em` applies its
  theta-increment clamp (module_big_step_utilities_em.F:5706-5707) or a
  later EOS refresh/boundary overwrite.
- **`nwp_diagnostics` scope**: WRF overrides the history cadence and sets
  `diag_flag = .true.` on every step when `nwp_diagnostics == 1`
  (dyn_em/solve_em.F:369).  gpuwm's output-due seam matches the WRF cadence
  only under `nwp_diagnostics == 0`; `gpuwm/namelist_import.py` rejects any
  nonzero value loudly instead of silently dropping it.
  *Amendment (STEP17, 2026-07-29):* `nwp_diagnostics = 1` now imports and
  enables the UP_HELI_MAX member of WRF's nwp_output family
  (`gpuwm/core/uh_diag.py`, oracle-pinned to dyn_em/module_diffusion_em.F
  `cal_helicity`).  The REFL_10CM seam above is UNCHANGED and remains
  correct under either value: WRF's forced every-step `diag_flag` only
  computes instantaneous diagnostics WRF then discards between frames --
  the emitted history-time REFL_10CM is the same field either way.  The
  unimplemented family members (WSPD10MAX, W_UP_MAX/W_DN_MAX, W_MEAN,
  GRPL_MAX, HAIL_MAX*) stay absent from wrfouts.
- **gpuwm arrangement**: the real74 output calendar marks only the final
  model step before each scheduled frame.  `dycore.step` threads that flag
  through `microphysics.apply`; the Morrison adapter computes from its
  post-call temperature/moments plus the pre-writeback prepared pressure,
  and the physics driver holds the FP32 `(nz,ny,nx)` result until
  `_write_case_output` consumes it exactly once.  A scheduled frame with no
  stash raises loudly.  The Kessler fallback uses the same timing.
- **Retired output-time seam**: evaluating after the dycore's
  post-microphysics EOS refresh changed the dry-branch density by
  `rho_new/rho_WRF = theta1*(1+(Rv/Rd)*qv1) /
  (theta0*(1+(Rv/Rd)*qv0))` at unchanged specific volume, hence changed
  dry reflectivity by exactly `10*log10(rho_new/rho_WRF)` before the floor.
  Representative latent-heating changes are 0.03-0.06 dB, already far
  above the mirror tolerance; near 273.15 K the reconstructed temperature
  could also flip the strict melting scan
  (module_mp_morr_two_moment.F:4586-4599) and its Blahak replacement.  That
  density-ratio analysis is why output-time evaluation was retired.
- **Scope (a), prognostic trajectory — inert**: gpuwm launches the
  reflectivity kernel only on output-due steps and does not maintain an
  always-current field between frames.  `REFL_10CM` has no reader back into
  dynamics, physics tendencies/state updates, nesting/coupling, or the
  restart-resume path; the only consumer API call outside its defining
  module is runtime output assembly.  The repository-wide no-reader pin is
  `tests/test_refl.py::test_refl_stash_has_no_trajectory_or_restart_reader`.
- **Scope (b), scientific products — real differences**: two omissions are
  intentional and observable: (1) the cold-start frame omits `REFL_10CM`
  because no microphysics call precedes it; and (2) resume does not rewrite
  the restart-boundary frame, so that boundary also has no rebuilt
  reflectivity product.  The stash is transient driver-rebuilt state in
  `gpuwm/io/restart.py`; normal loop order consumes output before a
  same-step restart write, and the next scheduled due step after resume
  rebuilds it.  Serializing an already-consumed/stale product is not part of
  trajectory identity, but the missing product frames remain product-level
  differences.
- **Scope (c), bytewise output — not claimed**: gpuwm does not claim that a
  `wrfout` carrying this diagnostic is byte-identical to a WRF output file.
  Comparisons are field-level under the registered seam/tolerance, not
  NetCDF byte comparisons.
- **Input-domain adjudication (review F2)**: WRF activates each species only
  by mass (`q > 1e-9`) and then immediately forms the slope from its number
  moment (module_mp_morr_two_moment.F:4544-4584); it has no zero/negative/
  non-finite number guard, so such inputs do not have a defined
  meteorological result.  The production scheme clips number moments
  nonnegative and reconstructs active moments from bounded slopes
  (:1528-1635).  gpuwm documents and tests this post-Morrison precondition;
  for an invalid active pair its kernel explicitly emits non-finite output
  instead of allowing CUDA `fmaxf` to disguise invalid slope arithmetic as
  the meteorological -35 dBZ clear-air floor.
- **Quadrature adjudication (review F3)**: the wet-particle path remains the
  exact WRF 50-log-bin weighted sum (module_mp_radar.F:83-90/:124-146;
  module_mp_morr_two_moment.F:4626-4667).  It is a compatibility rule, not
  a claim of uniformly converged composite-Simpson integration; no refined
  accuracy mode replaces it on the WRF-compatible path.

### D3 — experiment-schema fail-loud rejections (Phase 5, lane L1)

#### D3a — fail-loud scope rejections

- **WRF-native behavior**: WRF accepts these namelist settings and runs
  (or silently ignores them); gpuwm's experiment loader
  (`gpuwm/experiment.py`, `gpuwm/namelist_import.py`) rejects them loudly
  at load.  Each rejection message names the mechanism and the Phase that
  owns it, where one exists:
  - **Nonzero `spec_exp` on a nested child** is rejected rather than
    silently ignored.  WRF's nested `lbc_fcx_gcx` branch has no sponge
    term (dyn_em/module_bc_em.F:1297-1341; `spongeweight` only in the
    specified branch :1320, with the nested branch :1325-1337 carrying
    the sponge lines commented out).  gpuwm's reuse of the Phase-4 Davies
    `_weights` with `spec_exp = 0` is therefore a stated transliteration of
    that branch instead of a default-value coincidence.
  - **Moving nests** (`num_moves`/`move_*`/`vortex_*`/corral keys),
    **vertical nesting** (per-domain `e_vert`/`eta_levels`/`p_top`
    differences and `vert_refine_method != 0`; WRF only calls
    `init_domain_vert_nesting` when a nest refines the vertical grid, guarded
    by `nest%e_vert /= parent%e_vert` at
    share/mediation_integrate.F:663 with the call at :665),
    **adaptive time step** (`use_adaptive_time_step`), and
    **`input_from_hires`** terrain are rejected design-scope options that the
    bundle namelist never exercises.
  - Only **`interp_method_type = 2`** (SINT, the WRF default per
    Registry.EM_COMMON:2301) is accepted.  Bilinear/nearest/quadratic and
    `nest_interp_coord = 1` isobaric re-interpolation are rejected as a scope
    cut, not a behavior change; the bundle namelist runs the defaults.  The
    SINT kernels themselves are lane L3.
  - **`feedback != 0` and `smooth_option != 0`** are rejected this phase
    (one-way nesting; the dormant feedback machinery has separate Phase-5b
    activation gates).
  - **`use_theta_m = 1`** is rejected because gpuwm implements the
    non-moist-theta thermodynamic/heating branch only (D1).  Omission is
    also rejected because WRF's Registry default is 1; supported effective
    namelists must say `use_theta_m = 0` explicitly.
  - **`nwp_diagnostics != 0`** is rejected because it forces WRF's
    `diag_flag` every step and falls outside D2's output-due diagnostic
    cadence.  *Amendment (STEP17, 2026-07-29): retired -- the knob now
    imports, gating only the UP_HELI_MAX running-max diagnostic; D2's
    output-due REFL_10CM cadence is unaffected (see the amended
    `nwp_diagnostics` scope entry above).*
- **Inactive when**: none of the rejected options is requested.  Future
  implementation of any rejected mechanism is a roadmap item requiring its
  own validation and scope amendment; it is not a vanish condition for this
  entry.

#### D3b — exact-rational derivation and chained-FP32 dt

- Child `dx` and the clock interval are never hand-typed authorities.  The
  loader derives `dx_child = dx_parent/parent_grid_ratio` and
  `dt_child = dt_parent/parent_time_step_ratio` as exact rationals
  (share/set_timekeeping.F:366-368; Registry.EM_COMMON:2245-2246), then
  cross-checks supplied decimals.  The bundle's truncated `333.333333` d04
  `dx` passes; a hand-typed “500 m” is a hard error.  WRF itself trusts the
  namelist `dx` decimals.
- The WRF REAL time-step chain is now **implemented bit-exactly**, not a
  future intent: `gpuwm/core/clock.py` consumes the schema's chained-FP32
  values and validates the root construction and every child division by
  bit pattern.  `tests/test_clock.py::test_chained_fp32_dt_consumed_bit_pins`
  pins d01..d04 to `0x42700000`, `0x41700000`, `0x40A00000`, and
  `0x3FD55555`; `test_chained_dt_validation_is_bit_exact` proves a one-ULP
  neighbor is rejected.  D4 records the runtime tick representation and
  the remaining `dtbc` deviation.

#### D3c — physics substitutions and comparison scope

- The importer maps mp 55 (ISHMAEL) to Morrison 2-moment, bl_pbl 11
  (Shin-Hong) to YSU, and ra_lw/ra_sw 4 (RRTMG) to RTE+RRTMGP, emitting a
  structured `SubstitutionReport`; every other unimplemented scheme id is a
  hard error.  These are **model-form changes with no error bound**.
- Numerical fidelity gates in the N-ladder use the matched-physics
  instrumented oracle in the manifest above (Morrison+YSU, including the
  N1.5 registered tolerance).  Comparisons against the original ISHMAEL +
  Shin-Hong + RRTMG reference frames are statistical comparisons under the
  gate ledger, not numerical-equivalence claims.
- The flagship publication framing is the **effective namelist**.  Its
  reproducibility artifacts publish both the original WRF namelists and the
  effective gpuwm namelist/config after the explicit, structured
  substitutions and scope selections; neither may be presented as though
  it were the other.

### D4 — integer-tick clock: WRF-recurrent dtbc / running seconds and the exact-calendar scope (Phase 5, lane L2)

- **WRF-native mechanism**: WRF's boundary-tendency clock is a REAL
  accumulator — `grid%dtbc = grid%dtbc + grid%dt` inside solve_em
  (dyn_em/solve_em.F:371-372), executed BEFORE the relax/spec boundary
  routines consume it (:932-952), reset to zero at every nested force
  (share/mediation_force_domain.F:203-206) and, on d01, at every
  external-boundary interval seam (`IF (currentTime .EQ.
  grid%this_bdy_time) grid%dtbc = 0.`, share/mediation_integrate.F:1522).
  Running seconds (`curr_secs`) are DERIVED per step from the rational
  clock interval as REAL via `dt_whole + dt_num / REAL(dt_den)`
  (dyn_em/adapt_timestep_em.F `real_time`; interval built at
  dyn_em/solve_em.F:330-333, consumed by first-RK physics :800-815,
  widened to REAL*8 only at :4752-4755).
- **gpuwm mechanism** (`gpuwm/core/clock.py`): dtbc is now
  implemented-identical to WRF.  `DomainClock.dtbc_fp32` resets to
  positive FP32 zero on the existing force/LBC-seam calendar, then
  `prepare_step()` evaluates
  `np.float32(dtbc_fp32 + spec.dt_fp32)` immediately before every solve;
  `dtbc_launch_fp32` exposes that post-increment accumulator.  The reset
  at every parent interval prevents cross-interval drift, so this is
  bit-faithful for every ratio chain, including the former 1:7
  counterexample.  The existing d04 pins remain 0x3FD55555,
  0x40555555, 0x40A00000.  Exact integer ticks remain the sole
  sync/alarm/restart authority.  Running seconds remain tick-derived:
  `DomainClock.elapsed_seconds_fp32` mirrors WRF's REAL expression
  exactly (`FP32(whole) + FP32(num)/FP32(den)` from the tick
  decomposition, two FP32 roundings), while `elapsed_seconds` (one FP64
  division) is the internal interval-selection/restart authority.
  Scaling num/den by a common factor is guaranteed to leave the FP32 quotient
  unchanged when both integers satisfy `|n| <= 2^24` (and hence are exactly
  representable); that sufficient condition is not the definition of FP32
  integer representability.  `clock.py` defensively asserts this conservative
  bound for `whole`, `num`, and `den` at the decomposition; the
  current `num < den <= tick_den` bounds make the condition trivial for the
  registered clocks, and `tests/test_clock.py` pins the boundary/fail-loud
  behavior.  (The former "dtbc deviation scope" bullet is retired: the
  WRF-recurrent accumulator above is bit-faithful for every ratio chain,
  including the 1:7 counterexample it documented — those bits are now
  reproduced, not deviated from, and remain test-pinned.)
- **curr_secs**: NOT a deviation on the WRF-compatible path —
  `elapsed_seconds_fp32` reproduces WRF's two-rounding expression
  bit-for-bit (one ULP ABOVE the single-rounded FP64→FP32 cast at times
  like d04's 5/3 s: 0x3FD55556 vs 0x3FD55555; equal at all
  whole-second and dyadic instants).  Consumers that instead take the
  FP64 `elapsed_seconds` (interval selection, restart bookkeeping) are
  deliberately exact; kernel-facing WRF-parity consumers must use the
  FP32 mirror.
- **Root external-boundary dtbc — deviation CLOSED (Davies clock bind,
  2026-07-28; supersedes the F20 adjudication of 2026-07-17)**: the
  ROOT's external Davies launches now consume the WRF-recurrent
  `dtbc_launch_fp32` accumulator exactly like nest launches.  The tree
  build binds the root boundary clock (`gpuwm/core/model.py`
  `build_experiment` calls `bind_lateral_boundary_clock(root.state,
  root.clock)`; the N5S restored-model builder binds its manual root
  the same way), so root relaxation, the held moist/scalar targets, and
  the final specified-ring overwrite take WRF's post-increment dtbc
  (reset on boundary read, share/mediation_integrate.F:1515-1522;
  increment before the solve, dyn_em/solve_em.F:371-372: dt..T_bdy per
  interval) in place of the retired elapsed-based calculation
  (`state.elapsed_seconds - interval.start_seconds`, one dt BEHIND WRF
  in the relax zone).  The bound final ring additionally owns the OLD
  record at dtbc=T_bdy on the last pre-seam step (solve_em.F:4531-4639;
  the previous half-open lookup installed the new record at dtbc=0 —
  equal-valued, not FP32-bit-identical), closing the once-per-interval
  seam micro-mismatch.  At the campaign cadence (dt=60 s, T_bdy=21600 s)
  every FP32 dtbc partial sum is exactly representable, so the fix is a
  pure phase correction with zero roundoff residue.  CONSEQUENCES:
  (1) the frozen Phase-3/4 reference bytes (out/real74-t7-final,
  out/real74-t7-final-r2, out/cardinal) encode the retired semantics
  and are historical evidence only — the N-series invariance ratchets
  regenerate against the seam-closure anchor epoch
  (out/real74-t7-final-r3) in the batched end-of-Wave-1 regeneration;
  (2) restart headers carry the `root_external_lbc_clock` semantic
  identity ("wrf-postincrement-v1" bound / "legacy-elapsed-v0" unbound;
  a header without the key is a pre-bind file), and restores refuse a
  semantic mismatch, so old unbound checkpoints fail closed;
  (3) legacy direct paths that attach external boundaries without a
  DomainClock (era5_direct/gfs_direct) REMAIN elapsed-based by scope
  decision — their checkpoints self-identify as legacy and cannot mix
  with the bound production trajectory.  Child launches were already
  bit-faithful to WRF per the pins above and are unchanged.
- **Exact-calendar scope restriction**: every cadence
  (radt/cudt/bldt/history/restart/lbc_interval) must be an exact whole
  number of the owning domain's steps — noncommensurate alarms are a
  load-time error (`_cadence_ticks`, resolve_clock).  WRF is more
  permissive: its `Is_alarm_tstep` crossing-alarm predicate
  (frame/module_domain.F:2503-2516, `PrevRingTime + RingInterval <=
  CurrTime + TimeStep`) rings an alarm on the step that CROSSES the
  ring time, so WRF accepts e.g. a 100 s history interval on a 60 s
  clock (frames at 120, 180, 300, ... s).  gpuwm rejects such configs
  loudly (fail-loud scope cut; the bundle's calendars are all
  commensurate).  Note that for integer refinement a parent-commensurate
  cadence is automatically child-commensurate
  (`k*dt_parent = k*r*dt_child`), so the restriction binds only on
  genuinely noncommensurate namelist values.
- **Terminal feedback placement (schedule table)**: WRF does not
  suppress the final period's feedback; the per-kid loop guard
  (frame/module_integrate.F:439-445) skips it there and the last-io
  tail RE-ISSUES it (`med_nest_feedback` at :523 inside :507-526)
  whenever the head grid or any ancestor-chain grid is at stop time.
  gpuwm's flat table carries both families at WRF's exact call
  positions (identical op sequence on linear chains; reordered on
  branching trees), so the dormant table is already complete for the
  Phase-5b two-way activation.  The ratified architecture's F8 wording
  ("final period omits its FEEDBACK ops") predates this correction and
  is amended by the controller (p5t9 review round, 2026-07-16).
- **Vanishes when**: dtbc is no longer a deviation; the recurrence is
  implemented-identical for all ratios.  Calendar scope — if a future
  case needs WRF crossing
  alarms, `Is_alarm_tstep` semantics would be implemented behind the
  same DomainClock predicates (a plan amendment, not a silent change).

### D5 — SINT geometry precomputed FP64-on-host, stored FP32 (Phase 5, lane L3)

- **WRF-native behavior**: `SINT`/`SINTB` construct the XIG/XJG offset
  coefficient tables in REAL, on the fly, on every call
  (share/sint.F:13-14, :31, :46-57), alongside the donor-cell index
  arithmetic of the wrappers (share/interp_fcn.F:975-985, :2562-2575).
- **gpuwm**: nests are static, so the geometry — donor index maps plus
  the XIG/XJG tables — is precomputed ONCE at registration
  (`gpuwm/core/nest_interp.py` `sint_offsets`/`register_nest`), built in
  FP64 on host and stored FP32 on device; the FP64 verification mirrors
  (`gpuwm/verify/npref.py`) consume the SAME FP32-rounded tables so the
  N1 fp32_floor oracle stays discriminating (architecture §D, F6
  amendment; ratified deviations list).  ONLY geometry is precomputed:
  the DONOR/TR4 flux and min/max-limiter arithmetic is field-dependent
  and is evaluated per field at force time, never baked into weights.
- **Bound**: donor index maps are integer arithmetic (no precision
  content).  For rr ∈ {1, 2, 3, 4} — every ratio the bundle chain
  (4/3/3) exercises — the FP64-built/FP32-stored XIG/XJG values
  coincide bitwise with WRF's per-op REAL construction, machine-proven
  against a REAL-build emulation of sint.F:49-57 in
  `tests/test_nest_interp.py::test_sint_offsets_vs_wrf_real_build_emulation`
  (the stored values themselves are additionally pinned in
  `test_sint_offset_tables_fp32_pins`).  At rr = 5 the two
  constructions diverge by exactly 1 ULP at ip = 3: WRF per-op
  `fl32(0.4f - 0.6f)` = -0.20000002 (0xBE4CCCCE) versus gpuwm
  `fl32(fl64(-0.2))` = -0.2 (0xBE4CCCCD); every other (rr <= 5, ip)
  pair coincides.  The counterexample is recorded because the ratified
  schedule-pin family includes a (1,5,3) ratio chain, making rr = 5 a
  plausible future fixture ratio.  The general numeric consequence is
  owned by milestone N1.5 (the instrumented-WRF bdy-table oracle
  compares the full force pipeline at the pre-registered FP32-relative
  1e-6 band).
- **Vanishes when**: moving nests (a design non-goal) would force
  per-call geometry; or N1.5 evidence shows a discrepancy, which
  reopens this entry by controller adjudication.

### D6 — adjust_tempqv evaluates in FP64 on device, stores FP32 (Phase 5, lane L3)

- **WRF-native behavior**: `adjust_tempqv`
  (dyn_em/nest_init_utils.F:812-890) evaluates in REAL: the post-blend
  theta/qv correction chain `tc = (th+300)*(p/1e5)**(2/7) - 273.15`
  (:853/:855/:881), Magnus saturation (:857/:882), RH conservation and
  qv reconstruction (:859, :883-884).
- **gpuwm**: the `nest_adjust_tempqv` kernel
  (`gpuwm/core/kernels/nest.cu`) transliterates the identical algorithm
  and constants but computes each point in FP64, storing the REAL/FP32
  result — an implementation-forced deviation.  The tc chain cancels
  ~275 K of magnitude and the Magnus exponential amplifies the residue:
  a REAL-internal kernel is irreducibly hundreds of ULPs from ANY FP64
  mirror in qv, which makes the pre-registered N1 gate
  (`adjust_tempqv_vs_fp64_mirror`, fp32_floor at
  `FP32_FLOOR_MAX_ULPS = 8`) unpassable.  The alternatives are strictly
  worse under gates-first discipline (lane-review adjudication A3):
  degrading the mirror to FP32 internals destroys the oracle's
  discriminating power, and loosening a pre-registered threshold
  requires the N1.5 attribution run in hand (nest_gates ledger).
- **Bound**: a one-shot child-initialization operator (called once per
  nest at build, mediation_integrate.F:749 position).  Outside the
  blend rows `mub == save_mub`, hence `p_new == p_old`, `dth = 0`, and
  theta is bit-unchanged with qv reconstructed through the same tc
  (identity up to FP64 roundoff) — the deviation is CONFINED to the
  terrain-blend rows, where it is O(1e-5) relative in t/qv versus WRF's
  REAL evaluation; monitored by the N1 HGT/MUB static oracles, the N3
  blend-zone T2/TSK bias diagnostic, and the N3 statistical gates.
- **Vanishes when**: never by default; the controller may revisit with
  N1.5 evidence in hand (matched-physics instrumented WRF), which would
  arbitrate REAL-faithful internals against the gate design.

### D7 — Noah sea-ice runtime thermodynamics are not implemented (Phase 5, lane L4)

- **WRF initialization semantics, verified in the bundled v4.6.1 source**:
  for `fractional_seaice=0`, `adjust_for_seaice_post` uses an XICE threshold
  of 0.5 (`share/module_soil_pre.F:250-260`), snaps qualifying ice to XICE=1
  (`:260-262`), sets water-point TMN=271.4 K (`:263-264`), constructs four
  equispaced TSLB levels through a 3 m ice column from TSK to TMN
  (`:289-295`), and initializes SMOIS=1/SH2O=0 (`:297-300`); sub-threshold
  XICE is zeroed (`:301-302`).  gpuwm now mirrors all four cheap init
  behaviors in `gpuwm/ingest/soil.py`.  On the first Noah call WRF then sets
  sea-ice SH2O=1 and returns (`phys/module_sf_noahdrv.F:1067-1076`), which
  `gpuwm/core/kernels/noah.cu:938-941` already mirrors.
- **Registered runtime deviation**: WRF separately calls `seaice_noah` from
  the surface driver (`phys/module_surface_driver.F:2917-2929`; interface and
  updated fields in `phys/module_sf_noah_seaice_drv.F:14-23`).  That scheme
  evolves the sea-ice surface/ice-column energy state and its TSK, GRDFLX,
  HFX/QFX, snow, albedo, emissivity, and roughness coupling.  gpuwm has no
  sea-ice scheme in this phase.  Its Noah land kernel correctly recognizes
  an ice column and returns, but no later scheme advances that column; TSK
  therefore remains frozen at its initialized SKINTEMP over ice, while the
  surface-layer flux path continues to use that frozen skin.
- **Observed scope**: CPU reproduction of the staged May 3 1999 ERA5 SEAICE
  interpolation on the configured CONUS grid found 11 water cells with
  positive ice and 6 cells at/above the 0.5 Noah threshold, in the Gulf of
  St. Lawrence.  This is a documented small-signal deviation for that
  May-1999 case.  It does **not** affect the frozen real74 profile: its GRIB
  inventory contains neither SEAICE nor SOILGEO, so XICE remains exact zero
  and every sea-ice-only init/runtime branch is inactive.
- **Vanishes when**: a WRF-v4.6.1-equivalent Noah sea-ice driver and column
  thermodynamics are implemented and verified.  Adding a different generic
  ice model does not silently close this WRF-parity entry.

### Registered deviation — non-mutating force coupling (Phase 5 external review #4)

- **WRF-native transaction and arithmetic**: every `med_force_domain`
  call couples the parent in place, couples the child in place, performs
  the transfer/boundary interpolation, uncouples the child, uncouples the
  parent, clears `first_force`, and resets child dtbc, in that order
  (`share/mediation_force_domain.F:111-206`).  The forward coupling
  factors and reciprocal uncoupling factors are formed in separate REAL
  loops (`dyn_em/couple_or_uncouple_em.F:117-182` and `:193-261`), then
  applied to the full-patch prognostics in separate multiply passes
  (`:270-348`).  Consequently the FP32 round trip is not algebraically
  inert: `fl(fl(x*m)*fl(1/m)) != x` for a nontrivial subset of values.
  Both participating domains' full-patch prognostics can therefore be
  perturbed at every force call even when feedback is zero.
- **Bundle-chain exposure over 12 hours**: the 60 s root step and
  `parent_time_step_ratio = 1,4,3,3` are the bundled namelist values
  (`namelists/namelist.input:2,32-33,53,57`).  WRF's grouped force loop
  calls once per parent step for each sibling edge
  (`frame/module_integrate.F:409-423`), giving these per-domain in-place
  couple/uncouple round trips:

  | domain | as child | as parent | total round trips |
  |---|---:|---:|---:|
  | d01 | 0 | 720 | 720 |
  | d02 | 720 | 2,880 | 720 + 2,880 = 3,600 |
  | d03 | 2,880 | 8,640 | 2,880 + 8,640 = 11,520 |
  | d04 | 8,640 | 0 | 8,640 |

- **gpuwm registered improvement**: `force()` couples one full parent field
  at a time into the shared write-before-read arena scratch, pairs it with a
  full child field in a dead-RK scratch borrow, and feeds the full-extent
  `bdy_interp1` interface.  It never mutates either domain for coupling; only
  the rolling child boundary value/tendency frames persist.  The WRF
  round-trip perturbations are absent by construction.  The parent-read-only
  rule applies to `force()` only; Phase 5b's explicit `feedback_commit()`/
  `feedback_finalize()` transaction is the registered parent-mutation
  boundary.
- **Consequences for verification**: (a) a true nested WRF d01 is **not**
  bit-identical to its no-nest run, while gpuwm's d01 is; (b) the N3/N4/N5
  bitwise ratchets certify gpuwm's own stronger invariance and protect this
  deviation, not WRF identity; and (c) GPU-vs-WRF fidelity is adjudicated
  by the matched-physics shadow gates at ensemble-envelope tolerances,
  which absorb this deviation class.

### Registered N1.5 harness seam — dumped-WRF theta restoration (Phase 5, lane p5n15)

- **Scope**: the N1.5 candidate producer ONLY
  (`tools/wrf_instrumented/produce_nest_force_candidate.py`).  No
  production code path is affected.
- **Mechanism**: WRF couples the stored dry `t_2` directly
  (`dyn_em/couple_or_uncouple_em.F:283-286`); gpuwm's state carries
  `thb/thp`, and its production coupling converts natively via
  `fl(fl(t_init+thp)-300)` (`gpuwm/core/kernels/lbc_state.cu:575-582`).
  The producer must restore the dumped WRF `t_2` INTO the gpuwm
  representation first — `thp := fl(fl(t_2+300)-t_init)` — and that
  inverse map is lossy: the composition collapses to
  `fl(fl(t_2+300)-300)`, up to half a theta-scale FP32 ULP
  (2^-16 K at theta ~ 300 K).  Amplified through hybrid-mass coupling,
  the field-nonlinear SINT limiter, and the near-cancelling
  `(SINT(parent)-nfld)/cdt` tendency division, this appears as
  4-8e-2 coupled-unit tendency errors confined to `state.t.*.tendency`.
- **Attribution evidence** (p5n15 numerics shadow, empirical): child
  `nfld` is bit-exact on all four theta sides (0/441,000 mismatches);
  substituting direct `t_2` coupling for the parent donor makes ALL
  882,000 theta VALUE and TENDENCY elements bit-identical; the
  production N3+ recurrence never executes the inverse map (rolling
  boundary targets are rebuilt from gpuwm's native live state,
  `gpuwm/core/nest.py:178-196`).
- **Registered treatment** (F18 amendment): theta tendency tables are
  compared at the FP32 successor scale
  `E = fl32(VALUE + fl32(cdt*TENDENCY))` — exactly the expression WRF's
  boundary consumer applies — under the UNCHANGED 1e-6 band (measured
  2.06e-7..2.59e-7); all other tables keep the raw registered
  comparator.  The producer additionally hard-fails if gpuwm's own
  stored-table recurrence deviates from its raw SINT successor
  (measured <= 4.4e-10).
- **Adjacent findings closed in the same investigation**: (1) the
  west/south u/v tendency misses were a REAL production defect — gpuwm
  one-sided both physical faces where WRF applies the four-term
  duplicated-halo weight at low faces and repairs only the high faces
  (`couple_or_uncouple_em.F:140-166`) — FIXED at p5n15 head 80eebb4
  (kernel + mirror), after which all sixteen u/v tables are
  bit-identical to instrumented WRF; not a deviation.  (2) WRF's
  intermediate-grid `t/qv/ph` re-coupling
  (`force_domain_em_part2.F:285-300`) was investigated and REFUTED as a
  contributor: it is dead code in this case (`vert_refine_method=0`
  guard), and direct coupling through gpuwm's SINT path reproduces
  every WRF theta table bit-for-bit.
- **Vanishes when**: the N1.5 instrument gains a lossless
  representation bridge (e.g. WRF-side dumps of gpuwm-native
  `thb/thp`), or N1.5 is superseded by a stronger oracle.

## Seam-closure ratchet epoch (Davies clock bind, 2026-07-28)

The batched end-of-Wave-1 ratchet regeneration executed with the Davies
clock bind (branch `fix/davies-clock-bind`, fix commit `439bc083`):

- **Fresh N0 probe receipt** (`out/rungs/n0-alloc-probe-r3.json`,
  `gpuwm check configs/real74_4dom.toml --alloc --reserve-gib 2`): all
  three legs TRUE — pool used peak 15,459,084,288 B <= alloc estimate
  21,421,799,373 B <= budget 30,352,080,896 B.  Supersedes the stale
  `n0-alloc-probe-r2.json` calibration (2026-07-16, pre-arena-sharing;
  its margin against the grown Wave-1 estimator had inverted to
  -1,202,489 B per the ring-lane ledger).
- **d01 anchor epoch `out/real74-t7-final-r3`** (replaces the retired
  `-r2` role; `-r2`/`real74-t7-final`/`cardinal` bytes encode pre-bind
  semantics and remain historical evidence only): d01-only 75-min run
  at the fix tip, dual-run byte-identical
  (`real74-t7-final-r3/anchor-r3-dualrun.json`); 13:00 frame sha256
  `b4ee56f6…12616e00` (old r2: `7f588501…`), 12:00 frame
  `42d0dae9…eba7bc78`.  Triple identity re-proven at the bound epoch
  (`out/rungs/N3-runA/d01-triple-identity.json`): anchor == nested
  straight d01 == d01-only ancestor control, byte-for-byte.
- **N3 regenerated** (`out/rungs/N3-runA`, report PASSED 12/12;
  ratchet `out/rungs/N3/manifest.json`, experiment fingerprint
  `13d3d884…` [old `8a1c9117…` — input catalog gained the committed
  SOILHGT assets and the Wave-1 output inventory grew], d02 frames
  12:00 `6a656c5e…` … 13:15 `fe1e97e5…`): straight + ancestor-control
  arms dual-run byte-identical across two full rung executions
  (`N3-runA/n3-dualrun.json`, 10/10 frames); restart split
  bit-identity PASSED (v5 + `root_external_lbc_clock` identity
  round-trip at production scale); values moved vs the 2026-07-18
  epoch because the trajectory legitimately changed under ring-MP +
  SEAM A + init-surface + diff6 + Thompson + Davies bind (e.g. MSLP
  corr 0.9936 -> 0.9996, T850 RMSE 0.0741 -> 0.0942 K); the
  d02_refl_10cm_structure verdict is lane-adjudicated
  (`out/rungs/N3-verdicts-r3.json`, evidence
  `N3-runA/refl_structure_comparison.png`, amplitude offset 58.1
  dBZ/7357 cells vs prior ratified 57.2/7177) pending controller
  ratification.
- **N4 regenerated** (`out/rungs/N4-runA`, report PASSED 9/9; ratchet
  `out/rungs/N4/manifest.json`, fingerprint `8ee7107c…` [old
  `2df55493…`]): d01-vs-anchor and d02-vs-regenerated-N3 bitwise
  ratchets TRUE; straight arm dual-run byte-identical (14/14 frames,
  `N4-runA/n4-dualrun.json`); d03 statistics moved toward the
  reference (MSLP corr 0.9526 -> 0.9989); both structural verdicts
  lane-adjudicated (`out/rungs/N4-verdicts-r3.json`, evidence
  `N4-runA/d03_refl_structure_matched_comparison.png`, d03 interior
  w_max 1.83 m/s) pending controller ratification.
- **N5/N5B/N6**: no green recorded evidence exists at the old epoch on
  this box (no N5-report; N5B partial members m01-m03; N6 never run);
  their producers consume the regenerated N3/N4 manifests at next
  execution.
- **N5S — one green pre-bind report EXISTS and is RETIRED on paper**
  (correction 2026-07-28, adversarial-review finding; the original
  epoch record wrongly claimed no green N5S evidence existed): the
  main repo retains `out/n5s-shadow-report.json` (metric
  `N5S_matched_physics_wrf_shadow`, `passed: true`, generated
  2026-07-18T18:52:12Z, evaluator_commit `55e79f23`) together with its
  gpu evidence chain `out/n5s-gpu/` (`N5S-gpu-run.json`,
  `n5s-gpu-evidence.json`, `n5s-run-artifact.json`,
  `n5s-preregistration.json`, `restored_input_sha256.txt`).  That
  report is wholly VACUOUS — `all_envelopes_degenerate: true`, its
  pass auto-accepted under the f28 degenerate-envelope adjudication —
  and was never consumed by any N5-report; `_load_gate_report`
  re-scores with a `score_binding_sha256` refusal, so stale
  consumption fails closed.  Like the `-r2` anchor and the r2 N0
  receipt, it predates the Davies bind and encodes the retired
  unbound root-clock semantics (and, via piece 3, a shadow the
  regenerated N5S runner would no longer reproduce): it is SUPERSEDED
  as evidence, retained in place unmodified as historical record
  only, and must not be cited by any bind-era consumer.  N5S
  regeneration rides its next execution against the regenerated N3/N4
  manifests, with the now-bound restored-model builder.
- Every ratchet value above is derived from reference bytes produced
  by the fix-tip runs, never hand-tuned.

### Davies bind A/B measurement (2026-07-28, `out/rungs/davies-ab/`)

Root-domain-only production pair, 2 h at 15-min cadence, base
`a8b14b4c` (pre-bind) vs fix tip `439bc083`, identical t=0 bytes
(cross-checked); BOTH sides dual-run byte-identical
(`ab-dualrun.json`), zone-resolved deltas in `ab-zone-report.json`:

- **Specified ring**: T/MU/QVAPOR byte-zero at every lead — the final
  ring overwrite was already time-aligned away from interval seams,
  exactly as the dossier narrowed the seam.  (The staggered-U ring
  statistic in the JSON is projection-contaminated by the first
  interior face and is not a ring deviation.)
- **Relax rows (2-4)**: immediate quasi-steady offset — the closed
  one-step phase lag: mean |dT| 1.9e-3 -> 3.1e-3 K, mean |dMU| ~0.19 Pa
  (the recorded F20 one-step magnitude), mean |dU| 4e-3 -> 7e-3 m/s;
  local maxima 0.08-0.32 K / 0.29-0.74 m/s / 0.9-4.7 Pa in
  frontal/jet gradients.
- **Interior**: the structured phase correction seeds growing local
  differences while means stay small: |dT| max 0.19 K at +15 min ->
  ~2.6 K local at +105 min (mean 3.5e-3 K at +120 min); |dMU| max to
  ~36 Pa local (mean 0.31 Pa); |dQVAPOR| max ~1e-3 kg/kg local.
- **Cost**: byte-inert on timing — 0.335 vs 0.335 wall s per sim-min
  (both sides, dual-run).
## EXPERIMENTAL ensemble / perturbation / DA-cycle route (v1.2)

Registered here because a consumer can reach this machinery from the repo
without ever seeing a finished manifest, and everything below is a scope
cut rather than a mechanism transcribed from an oracle. **Nothing in this
section is on a certified forecast path**, no product it draws has been
calibrated against a verification archive, and every manifest, receipt and
PNG it writes carries an `experimental` stamp.

- **Entry points, and only these**: `tools/ensemble_forecast.py` (`run`,
  `cycle`, `bench`, `status`), `gpuwm enprod`, and
  `tools/da_synthetic_cycle.py`. `gpuwm run` reaches none of it.
- **Authority**: no WRF mechanism is transcribed. The perturbation
  construction is a prescribed isotropic Gaussian random field, the
  probability-matched mean follows Ebert (2001) with the every-*M*-th
  pooled selection described in Flora et al. (2018) fn. 1, and the
  neighborhood-maximum ensemble probability is the standard
  max-in-neighborhood-per-member-then-ensemble-fraction order.
- **Scientific non-goals of `gpuwm.da.perturb`**, stated before a run and
  repeated in every returned provenance dict: no mass balance (`mu'`
  untouched, no hydrostatic re-balance); no wind balance (the `u`/`v`
  increments are neither non-divergent nor in geostrophic/gradient balance
  with the temperature increment); one shared, unperturbed lateral
  boundary file for all members, so ensemble spread decays toward the rim
  by construction; lateral rim taper only, no vertical taper; no surface,
  soil, or physics-parameter perturbation and no `w`, `mu'` or
  hydrometeor perturbation; no flow dependence.
- **Registered artifact — vertical FFT periodicity.** The draw is periodic
  on every axis. Horizontally the rim taper hides it. Vertically nothing
  does: on a periodic column level 0 and level `nz-1` are ONE interval
  apart, so their correlation is approximately `exp(-1/(2 Lv^2))` --
  0.983 at the admitted quarter-column cap (`nz=24`, `Lv=6`), not the
  `exp(-2) ~ 0.135` earlier documentation claimed by confusing the seam
  with the maximum circular separation. The cap bounds the HALF-COLUMN
  correlation (0.270 at that setting, itself above `exp(-2)` because
  periodic images contribute covariance) and does not control the seam.
  Every draw reports its exact figures in
  `provenance["fields"][i]["vertical_wrap"]`, computed from the applied
  filter.
- **Registered scope cut — horizontal scale admission.** A prescribed
  length scale is admitted only up to `span/(2*pi)`, because the
  documented radial-spectrum peak `k = 1/L` is resolved only when
  `1/L >= 2*pi/span`. The previous quarter-span limit admitted
  configurations whose peak fell below the domain's lowest nonzero
  wavenumber (measured `peak * L = 2.356` at the old cap on 32-, 64- and
  128-point domains).
- **Determinism scope**: "byte-identical given the seed" holds on ONE
  software and device stack. The white noise is drawn on the host with
  NumPy Philox and hash-stamped, so `noise_sha256` is machine-independent;
  the filtered field is not, because cuFFT and pocketfft round
  differently, and the provenance records which FFT backend ran.
- **Ensemble-product NaN policy**: `mask` (default) excludes non-finite
  member values from every reduction at that point, shrinks the
  denominator with them, leaves a point with no finite member blank, and
  records the resulting coverage in the provenance and on the panel;
  `refuse` fails the product naming the members. The NMEP voting roster is
  fixed by the point's own data so the denominator does not move with the
  neighborhood radius and the advertised monotonicity is exact.
- **Ensemble-product roster override**: `--accept-status` beyond
  `DONE,complete` triggers a frame-inventory check of every admitted
  member against the manifest before anything renders, and stamps the
  override on every panel.

## Attribution defects in this repository's own history (2026-07-26)

This register exists so a provenance claim is never re-derived from memory.
Attribution is a provenance claim, so known defects in it are recorded here
rather than only in a lane's tracking document, which may be retired.

Every commit produced under agent delegation on this project must end with the
trailer
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
One commit does not:

| Commit | Subject | Defect |
|---|---|---|
| `ab60dc8` | `feat(noahmp): route sf_surface_physics=4 to a real runner and admit it` | no trailer at all; `git log -1 --format='%(trailers:only=true)' ab60dc8` prints empty |

**Not repaired, deliberately.**  The omission was found after the branch
(`codex/mynn-noahmp-ruc-cuda`) had advanced past that commit and other lanes
had built on it, so `git commit --amend`/rebase would have rewritten history
shared with concurrent work.  The lane flagged it instead of rewriting shared
history.  Recording it here is the repair: the commit's authorship is not in
doubt, the trailer is simply absent, and this row is what prevents a later
reader from treating the absence as evidence of an unattributed change.

Verification, on any checkout of the branch:

    git log --format='%h %s%n  trailers:[%(trailers:only=true,valueonly=true)]' -12

The narrative context of that commit -- Noah-MP's runtime admission -- is in
`docs/wrf_noahmp_runtime_admission.md`, and the same fact is duplicated in
`docs/mynn_noahmp_ruc_completion_plan.md` under "Provenance and process
record".
