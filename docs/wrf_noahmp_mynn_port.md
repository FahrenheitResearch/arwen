# WRF v4.6.1 Noah-MP 4 + MYNN 5/5 port contract

## Claim boundary

This lane adds Noah-MP (`sf_surface_physics=4`) and the coupled MYNN surface
layer/PBL (`sf_sfclay_physics=5`, `bl_pbl_physics=5`) as selectable
alternatives. It does not replace or weaken the existing classic Noah 2,
MM5/revised-MM5 91/1, or YSU 1 paths. Thompson 8, Morrison 10, WSM6 6, and
NSSL-2 18 remain independent microphysics choices.

Landing a selector, a Python function with a familiar name, or a stable short
run is not implementation evidence. Until every admission gate below passes,
the importer and runtime must reject 4 and 5/5 with one complete explanation
and must never substitute Noah/YSU/MM5.

## Numerical authority

The atmospheric authority is NCAR WRF v4.6.1 at commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Its Noah-MP submodule is pinned at
commit `848f54ad3d28c4303151fe5ad83724e232694422`.

The executable contract in `gpuwm/core/noahmp_mynn_contract.py` checks the
exact byte sizes and SHA-256 digests of:

- `Registry/Registry.EM_COMMON` and `Registry/registry.noahmp`;
- `phys/module_sf_mynn.F`, `module_bl_mynn.F`,
  `module_bl_mynn_common.F`, and `module_bl_mynn_wrapper.F`;
- the WRF Noah-MP driver, core LSM, glacier, and groundwater sources; and
- `MPTABLE.TBL`, `SOILPARM.TBL`, and `GENPARM.TBL`.

Oracle results from a different source revision or different parameter-table
bytes are a different numerical target. They cannot be relabeled as this
contract.

## First admitted option identity

The first Noah-MP lane is deliberately the exact WRF Registry default, not an
open-ended collection of Noah-MP modes:

| Option | Value | Option | Value |
|---|---:|---|---:|
| `dveg` | 4 | `opt_crs` | 1 |
| `opt_btr` | 1 | `opt_run` | 3 |
| `opt_sfc` | 1 | `opt_frz` | 1 |
| `opt_inf` | 1 | `opt_rad` | 3 |
| `opt_alb` | 2 | `opt_snf` | 1 |
| `opt_tbot` | 2 | `opt_stc` | 1 |
| `opt_gla` | 1 | `opt_rsf` | 1 |
| `opt_soil` | 1 | `opt_pedo` | 1 |
| `opt_crop` | 0 | `opt_irr` | 0 |
| `opt_irrm` | 0 | `opt_infdv` | 0 |
| `opt_tdrn` | 0 | `soiltstep` | 0.0 |
| `noahmp_acc_dt` | 0.0 | `noahmp_output` | 1 |

Other science switches can be added later only as named, independently tested
option identities.

The canonical Noah-MP parameter Git blobs are now packaged in
`gpuwm/data/noahmp`. The loader checks byte count and SHA-256 before parsing
all ten MPTABLE namelist groups (433 assignments), both 19-category STAS and
STAS-RUC soil tables, and all GENPARM values. CRLF-expanded or otherwise
modified copies fail before a solver can use them. This establishes executable
parameter identity; the Noah-MP energy/water column solver is still open.

The first MYNN lane is likewise the exact WRF Registry default:

| Option | Value | Option | Value |
|---|---:|---|---:|
| `bl_mynn_tkeadvect` | false | `bl_mynn_cloudpdf` | 2 |
| `bl_mynn_mixlength` | 1 | `bl_mynn_edmf` | 1 |
| `bl_mynn_edmf_mom` | 1 | `bl_mynn_edmf_tke` | 0 |
| `bl_mynn_mixscalars` | 0 | `bl_mynn_output` | 0 |
| `bl_mynn_cloudmix` | 1 | `bl_mynn_mixqt` | 0 |
| `icloud_bl` | 1 | `bl_mynn_closure` | 2.6 |

`mfshconv` was previously listed here and has been removed: it is not a MYNN
knob. WRF reads it only under the QNSE case in
`module_pbl_driver.F:1515-1521`, where it routes to `module_bl_mfshconvpbl`.
MYNN never reads it, so asserting it as part of this identity would bind the
contract to something the scheme does not consume.

The namelist knobs above are not the whole identity. Four compile-time module
parameters in `phys/module_bl_mynn.F` also fix it, and they have no namelist
override:

| Parameter | Value | Consequence |
|---|---:|---|
| `bl_mynn_topdown` (`:328`) | 0 | `topdown_cloudrad` is dead; `TKEprodTD` is identically zero |
| `bl_mynn_edmf_dd` (`:330`) | 0 | `DDMF_JPL` is dead; every `sd_aw*` is identically zero |
| `dheat_opt` (`:333`) | 1 | dissipative heating is **active** and enters the `thl` tendency |
| `bl_mynn_stfunc` (`:340`) | 1 | `pmz`/`phh` come from `phim`/`phih`, not the Kansas forms |

The first two remove ~490 lines from the port surface. The second two are
required work that a namelist-only reading of the identity would miss.

MYNN is admitted as a coupled 5/5 suite. Enabling only its surface layer or
only its PBL does not form a validated configuration.

## State contract

WRF allocates the MYNN scalar `qke_adv` (twice TKE) plus `qke`, `tke_pbl`,
`sh3d`, `sm3d`, `tsq`, `qsq`, `cov`, and `el_pbl`. Default EDMF also requires
the plume-top and mass-flux diagnostics, and `icloud_bl=1` requires the MYNN
PBL-cloud fields. Optional three-dimensional diagnostics remain off under the
first option identity but still need an explicit future-output contract.

WRF's `noahmpscheme` package allocates 224 named fields spanning layered snow,
soil/canopy stores, vegetation and carbon pools, groundwater/runoff, energy
and water fluxes, diagnostics, and accumulators. The complete ordered field
tuple is embedded and hash-bound in the executable contract. Classic Noah's
four-layer arrays are not a sufficient Noah-MP state implementation.

For both suites, all trajectory-affecting state must participate in:

- initial-condition construction and lateral/nest policies where applicable;
- GPU allocation and health/finite checks;
- exact restart serialization plus identity validation;
- wrfout diagnostics with units, staggering, and accumulation semantics; and
- parent/child initialization and feedback policy.

## Implementation and admission gates

1. **Contract gate:** pinned commits, bytes, defaults, selectors, and Registry
   state inventories are executable tests. Incomplete selectors stay blocked.
2. **Official-source oracle gate:** build standalone WRF v4.6.1 harnesses for
   `SFCLAY1D_mynn`, MYNN PBL columns, and representative Noah-MP land columns.
   Preserve exact inputs, outputs, compiler identity, and source hashes.
3. **CPU algorithm gate:** gpuwm CPU references reproduce stable, unstable,
   neutral, cloudy, snow/ice, wet/dry soil, vegetation, and groundwater oracle
   cases without hidden clipping or state resets.
4. **CUDA kernel gate:** device-resident kernels match the CPU reference and
   WRF oracle for tendencies, fluxes, diagnostics, conservation, and option
   branching. Host round trips inside the forecast loop are rejected.
5. **Coupling gate:** surface exchange, Noah-MP energy/water fluxes, MYNN TKE,
   diffusion, EDMF, PBL cloud, radiation, and microphysics state are ordered
   and coupled consistently with the pinned WRF driver.
6. **Persistence gate:** cold start, restart/resume, wrfout, lateral boundary,
   nest initialization, and feedback tests cover every trajectory field.
7. **Forecast gate:** matched WRF/gpuwm single-column, single-domain, and real
   nested forecasts pass numerical-health and preregistered parity metrics.
8. **Production gate:** only after all prior gates pass may config and namelist
   maps advertise Noah-MP 4 and MYNN 5/5 as executable.

## Landed progress: first Noah-MP official-source oracle harness

`tools/noahmp_wrf461_oracle/` builds a standalone harness against the pinned
WRF v4.6.1 tree. `share/module_model_constants.F`, `phys/module_sf_gecros.F`,
`phys/module_sf_noahmplsm.F`, `phys/module_sf_noahmp_glacier.F`,
`phys/module_sf_noahmp_groundwater.F` and `phys/module_sf_noahmpdrv.F` all
compile **unmodified**; their SHA-256 digests are the ones already pinned in
`gpuwm/core/noahmp_mynn_contract.py`.

`stub_wrf.F90` supplies services only: `wrf_message`, `wrf_error_fatal`,
`wrf_debug`, an opaque `module_domain::domain` handle (`GROUNDWATER_INIT`
declares it `TARGET` but never dereferences a component outside the
compiled-out `EM_CORE`/`DM_PARALLEL` halo block), the `IRI_SCHEME` namelist
switch, and `EXTERNAL` link targets for `urban`, `bep`, `bep_bem` and
`cal_mon_day` that abort if ever reached. No stub returns a physical quantity.

Parameter identity is executable, not asserted: `build.sh` copies
`gpuwm/data/noahmp/{MPTABLE,SOILPARM,GENPARM}.TBL` into the build directory,
re-verifies their SHA-256 there, and WRF's own `NOAHMP_TABLES` readers parse
those exact bytes in the same order as `NOAHMP_INIT`. The oracle and
`gpuwm/core/noahmp.py` consume the same file bytes.

Two fixtures are pinned in `gpuwm/data/noahmp/oracle/` (full provenance in its
`README.md`):

- `noahmp-parameters.csv` -- `read_mp_*` plus `TRANSFER_MP_PARAMETERS` over
  seven land identities x 181 transferred fields, including the `URBAN_FLAG`
  soil/`CSOIL` override and its knock-on effect on `FRZX`.
- `noahmp-sflx.csv` -- `NOAHMP_SFLX` under the option identity above over four
  regimes: sunlit snow-free canopy, nocturnal raining canopy, a two-layer
  snowpack over `OPT_FRZ=1` partly frozen soil, and bare ground with
  sub-layer snow melting to ponding. WRF's own shortwave and energy residuals
  close to better than 0.01 W/m2 in every column.

### What the harness does NOT admit

- **No gpuwm parity number exists.** There is no Noah-MP column solver in
  gpuwm yet, so gate 3 and everything after it remain fully open. The
  validators check structure, conservation and branch coverage only.
- **No per-leaf oracles are reachable.** Every leaf routine of
  `MODULE_SF_NOAHMPLSM` (`THERMOPROP`, `CSNOW`, `TDFCND`, `PHASECHANGE`,
  `FRH2O`, `ESAT`, `TSNOSOI`, `HRT`, `SOILWATER`, `WDFCND1/2`, ...) carries an
  explicit `private ::` statement at `module_sf_noahmplsm.F` lines 26-84;
  `NOAHMP_SFLX` and `noahmp_options` are the module's only public procedures.
  `NOAHMP_GLACIER_ROUTINES` is structured identically (lines 71-99). Isolating
  a leaf would require editing the pinned source, so the harness drives the
  whole column instead. A future leaf-level CPU port must be compared against
  whole-column fixtures, or against a separately justified exposure mechanism.
- **Only `ICE = 0`, `IST = 1` land columns.** No glacier column, no lake
  column, no `NOAHMP_GLACIER` call.
- **Only single first steps.** `soiltstep = 0` (`soil_update_steps = 1`) and
  every `ACC_*` accumulator enters at zero; multi-step soil sub-cycling is
  untested.
- **Crops, irrigation, tile drainage, groundwater/lateral flow, the urban
  canopy coupling and the WRF-level `noahmplsm` driver are all untouched**,
  as are every non-default option value and the USGS land-use table.
- Under this identity WRF leaves `parameters%IRR_*`, the crop block, `SLAREA`,
  `EPS`, and `FRZX` for `soiltype = 14` formally undefined; those fields are
  deliberately absent from the parameter fixture.

## RW-WPS/node usability acceptance

The integrated RW-WPS surface must make the expanded physics stack inspectable
and reproducible rather than relying on remembered environment variables or
hand edits:

- a capability command reports executable, guarded-experimental, porting, and
  unsupported physics separately, including the reason for every blocked
  selector;
- namelist import preserves the user's exact physics request or emits one
  readable fail-closed report; it never rewrites a scheme to a nearby option;
- a fresh-node bootstrap installs pinned Python/Rust/CUDA-facing dependencies,
  verifies bundled decoder/table manifests, and leaves a machine-readable
  receipt;
- CLI flags support explicit source, physics profile, domain hierarchy,
  preprocessing backend, output root, restart, and R2 destination without
  editing generated files; and
- `--dry-run` prints the exact command and resolved paths without creating or
  overwriting files.

These are release requirements, not documentation wishes. A node deployment
that depends on an undocumented shell history does not pass.
