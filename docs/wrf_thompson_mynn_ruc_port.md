# WRF Thompson + MYNN + RUC physics port

## Scope and claim boundary

The source authority is the official WRF repository tag `v4.6.1`, resolved to
commit `d66e442fccc04111067e29274c9f9eaccc3cef28`.  The target namelist is:

- `mp_physics=8`: Thompson microphysics;
- `ra_lw_physics=4`, `ra_sw_physics=4`: WRF RRTMG selection, intentionally
  mapped by gpuwm to its existing RTE+RRTMGP implementation with an explicit
  compatibility receipt; this is not algorithmic or bitwise RRTMG identity;
- `sf_sfclay_physics=5`, `bl_pbl_physics=5`: the coupled MYNN surface layer
  and prognostic-TKE PBL;
- `sf_surface_physics=3`, `num_soil_layers=9`: nine-layer RUC LSM;
- `radt=5`, `bldt=0`, `cu_physics=0`, surface fluxes and cloud-radiation
  coupling on, urban and mosaic modes off.

Scheme numbers remain fail-closed until every required state, kernel, coupling,
restart/output, and validation gate for that component lands.  A numerically
nearby existing gpuwm scheme is never an implicit substitute.

## Source-anchored port map

### Thompson (`mp_physics=8`)

WRF's Registry package is
`Registry/Registry.EM_COMMON:3024`: six mass species (`qv`, `qc`, `qr`, `qi`,
`qs`, `qg`), two number scalars (`qni`, `qnr`), and effective radii
`re_cloud`, `re_ice`, and `re_snow`.  The scalar definitions are at
`Registry/Registry.EM_COMMON:523-534`; effective-radius state is at
`:497-499`.  These fields participate in input/restart/history and boundary
interpolation according to the Registry flags, so a column-only kernel is not
an end-to-end port.

The implementation authority is `phys/module_mp_thompson.F`.  Its header cites
Thompson et al. (2008, *Monthly Weather Review*, 136, 5095-5115) and explains
that WRF v3.1+ predicts rain number while option 8 retains prescribed cloud
droplet number.  `thompson_init` begins at line 424 and `mp_gt_driver` at line
1070.  WRF initializes option 8 through `module_physics_init.F:4515-4520` and
calls the operational driver at `module_microphysics_driver.F:1209-1277` with
the six mass species, `NI`, `NR`, theta/Exner/pressure/vertical velocity/layer
depth/timestep, accumulated and step precipitation, frozen fraction, optional
reflectivity, and three effective radii.

The expensive immutable tables are part of the scheme, not optional test data.
`module_mp_thompson.F:899-1060` constructs the lookup families;
`:4091-4310`, `:4313-4591`, and `:4600-4791` implement rain-graupel,
rain-snow, and freezing tables with the WRF namelist policies declared at
`Registry.EM_COMMON:2403-2405`.  gpuwm will package deterministic table bytes,
hash them into restart/setup identity, and validate generated/read values
against the WRF build used by the oracle.

Integration must follow WRF's post-RK moist-physics path, visible in
`dyn_em/solve_em.F:3660-3720`, including the existing gpuwm latent-heating and
mass-coupling contract.  Acceptance requires:

1. official-WRF column oracle fixtures covering warm rain, mixed phase, ice,
   snow, graupel, evaporation/sublimation, freezing/melting, and sedimentation;
2. CPU transcription parity before CUDA implementation;
3. CUDA versus CPU parity with conservation, non-negativity, precipitation,
   `SR`, effective-radius, and reflectivity checks;
4. restart split-run identity, output schema, external LBC and one-way nest
   initialization/forcing tests;
5. short real-data GPU stability and WRF comparison receipts.

### MYNN surface layer + PBL (`sf_sfclay_physics=5`, `bl_pbl_physics=5`)

This is one coupled port.  WRF registers the surface layer at
`Registry.EM_COMMON:3133` and the PBL at `:3168`.  The PBL package adds advected
twice-TKE `qke_adv` and state `qke`, `tke_pbl`, `sh3d`, `sm3d`, `tsq`, `qsq`,
`cov`, and `el_pbl`; definitions and I/O flags are at
`Registry.EM_COMMON:1118-1129` and `:1231-1232`.

`module_surface_driver.F:2359-2403` shows the MYNN surface call contract:
near-surface wind, temperature, moisture, pressure, density and layer depth;
surface pressure/temperature/roughness/land mask/snow; exchange coefficients,
fluxes, Monin-Obukhov diagnostics, 2-m and 10-m diagnostics, and `ch`.  Its setup
entry is `module_physics_init.F:3216-3219`.

The PBL driver gate and call are `module_pbl_driver.F:1629-1721`.  The call
consumes surface `ust/ch/hfx/qfx/rmol`, full thermodynamic/wind columns,
radiative heating, `qke/qke_adv`, variance state and configured EDMF/cloud/mix
options, and produces momentum/temperature/moisture tendencies, exchange
coefficients, PBL height, mixing length and optional cloud/EDMF diagnostics.
The setup explicitly admits MYNN surface option 5 (and selected legacy surface
layers) at `module_physics_init.F:3837-3850`.

WRF v4.6.1 defaults that must be explicit in gpuwm's trajectory identity are at
`Registry.EM_COMMON:2473-2484`: TKE advection false, cloud PDF 2, mix length 1,
EDMF 1, EDMF momentum 1, EDMF TKE 0, scalar mixing 0, extra output 0, cloud
species mixing 1, total-water mixing 0, boundary-layer cloud coupling 1, and
closure 2.6.  The first executable target will preserve those defaults and
will not claim a reduced "MYNN-like" operator.

Acceptance requires surface-layer standalone columns, coupled surface/PBL
columns, stable/convective/cloud-topped regimes, prognostic-TKE conservation
and bounds, CUDA parity, restart/LBC/nest treatment of `qke_adv`, and a short
real-data GPU comparison.  `bldt=0` means the coupled stack executes every
model step under gpuwm's existing physics calendar.

### RUC LSM (`sf_surface_physics=3`, `num_soil_layers=9`)

WRF documents six or nine RUC layers in `run/README.namelist:937-943` and
registers option 3 plus persistent state in `Registry.EM_COMMON:3147`:
`smfr3d`, `keepfr3dflag`, `soilt1`, `rhosnf`, `snowfallac`, `precipfr`, and
`acrunoff`.  Their definitions/I/O flags are at `Registry.EM_COMMON:866`,
`:1003-1007`, and `:1975-1976`, in addition to the shared soil temperature,
total/liquid moisture, canopy, snow and surface fields.

The implementation authority is `phys/module_sf_ruclsm.F`; `lsmruc` begins at
line 79, cites Smirnova et al. (1997, 2000), and carries soil, frozen-soil,
snow, canopy, radiation, precipitation phase, land-use/soil tables, flux and
runoff state.  `ruclsminit` begins at line 6991.  WRF setup calls it at
`module_physics_init.F:3505-3518`; the complete surface-driver call is
`module_surface_driver.F:3438-3528`.

The port therefore requires nine-level geometry and all persistent snow/frozen
soil state, not merely expanding Noah's four arrays.  Ingest contracts must
define HRRR native, ERA5, and WPS/real.exe sources, interpolation/extrapolation,
land/water/ice masks, units and physical bounds.  Mosaic land-use/soil and
urban physics remain fail-closed because the requested namelist disables them.
Acceptance requires initialization parity, warm/cold soil and snow columns,
water/ice branches, water/energy budget checks, MYNN surface coupling, CUDA
parity, restart/output, nesting and short real-data receipts.

### Radiation and remaining flags

WRF documents 4 as RRTMG at `run/README.namelist:574-587`; gpuwm option 4 is
its existing modern RTE+RRTMGP solver.  The importer records the intentional
mapping and emits compatibility token
`wrf-rrtmg-4-4-to-rte-rrtmgp-v1`.  Restart identity already names the actual
RTE+RRTMGP algorithms/assets, so neither output nor a resumed run can call it
legacy WRF RRTMG.  `radt=5` and cloud coupling are supported; cloud coupling
off is rejected on this adapter because its current implementation is always
coupled.

`cu_physics=0`, `bldt=0`, `isfflx=1`, `sf_urban_physics=0`,
`sf_surface_mosaic=0`, `mosaic_lu=0`, and `mosaic_soil=0` are direct supported
settings.  Any nonzero urban/mosaic request or disabled surface flux request
fails instead of being dropped.

## Staged implementation and evidence

1. **Configuration and fail-closed plumbing.** Add explicit compatibility
   identity, model-relevant WRF option schema, complete three-component blocker
   receipts, and tests.  Existing trajectories must remain identical.
2. **Thompson end to end.** Freeze WRF oracle/table assets; add prognostic and
   diagnostic state, CPU transcription, CUDA kernels, physics-driver coupling,
   radiative radii, reflectivity/precipitation, restart/output/nesting, then
   column and short-case GPU gates.  Only then admit `mp_physics=8`.
3. **MYNN surface + PBL.** Implement and validate the coupled surface exchange,
   prognostic TKE, diffusion/EDMF and cloud coupling; add restart/output/LBC and
   nesting.  Only then admit both option 5 selectors.
4. **RUC nine-layer LSM.** Implement initialization contracts and table assets,
   nine-layer/snow/frozen-soil state and CUDA coupling, then restart/output and
   real-data validation.  Only then admit option 3 with nine layers.
5. **Integrated suite.** Run deterministic CPU tests, CUDA poison/parity gates,
   restart split runs, nested short cases, end-to-end timing/VRAM receipts and
   WRF comparison.  Scientific claims will identify the deliberate RRTMGP
   radiation difference and will not generalize beyond exercised regimes.

Every milestone is a small commit plus a git bundle and hashed evidence copied
off the nonpersistent rental node.

## Landed progress

- Milestone 1 (`58e89ec`) established explicit namelist compatibility and a
  fail-closed three-component receipt.  The requested RRTMG 4/4 selectors map
  only through the named RTE+RRTMGP compatibility token.
- Milestone 2 (`42ee613`) landed the complete classic Thompson state,
  transport, nesting, radiation-radius and I/O contract plus a strict reader
  and official-WRF table generator.  Thirty finite arrays totaling
  379,839,912 payload bytes are held in off-repository evidence with canonical
  hashes; the unused/uninitialized WRF `tnr_rev` allocation is excluded.
- The current milestone content-addresses those coefficient bytes and adds
  deterministic warm, mixed-phase and ice `mp_gt_driver` columns generated by
  unmodified WRF.  These fixtures now pin full before/after state,
  precipitation, effective radii and reflectivity for the CPU/CUDA port.
- The first state-changing CUDA slice isolates WRF's warm saturation
  adjustment with a fourth direct-driver `condense` column.  It changes only
  temperature, vapor and cloud water, and remains a diagnostic milestone;
  option 8 stays fail-closed until the complete process and sedimentation
  network passes the full-column gates.
- Native CUDA two-moment rain sedimentation now has its own liquid-saturated
  `rain-sed` oracle, including WRF's separate mass/number fall speeds,
  timestep splitting, surface rain and closed column-water budget.  The
  remaining cloud/ice/snow/graupel fallout and the full process network are
  still required before option 8 can execute.
- Native CUDA cloud-ice sedimentation now has a separate ice-saturated
  `ice-sed` oracle, exercising differential ice mass/number fall speeds,
  frozen precipitation and an 80-level generic-depth no-op gate.
- Native CUDA cloud-water sedimentation now has a separate liquid-saturated
  `cloud-sed` oracle below WRF's autoconversion threshold.  It exercises the
  fixed cloud-number fall-speed law, conservative vertical redistribution,
  and an 80-level generic-depth no-op gate.
- Native CUDA one-moment snow sedimentation now has a separate ice-saturated
  `snow-sed` oracle.  It exercises WRF's Field et al. moment conversion,
  timestep splitting, frozen precipitation, closed water budget, and an
  80-level generic-depth no-op gate.  Snow mass is bitwise identical to the
  direct WRF column.
- Native CUDA classic fixed-density graupel sedimentation now has a separate
  ice-saturated `graupel-sed` oracle.  It exercises WRF's diagnosed intercept,
  mass fall speed, timestep splitting, graupel precipitation, water budget,
  and an 80-level generic-depth no-op gate.
- Native CUDA Berry-Reinhardt warm-cloud autoconversion now has a direct
  liquid-saturated `warm-auto` oracle.  It composes with the admitted rain
  fallout kernel and matches WRF cloud water bitwise; rain mass/number remain
  within `1.01e-6` relative and surface rain within `1.83e-6` relative.  The
  mixed-phase process network remains open.
- Native CUDA Seifert rain self-collection now has a direct liquid-saturated
  `rain-self` oracle at 500-micron median-volume diameter.  Composed with
  admitted rain fallout, CUDA rain mass and surface rain are bitwise identical
  to WRF; rain number stays within `3.86e-7` relative.
- Native CUDA rain-cloud accretion now has a direct liquid-saturated
  `warm-accrete` oracle below the autoconversion threshold.  It fuses
  accretion and Seifert self-collection so both rates use the same incoming
  rain state, then composes with admitted rain fallout.  Cloud water and
  surface rain are bitwise identical to WRF; rain mass stays within
  `1.03e-7` relative and rain number within `3.37e-7` relative.  This is the
  first admitted process that consumes the canonical external FP64
  `t_Efrw(100,100)` collision-efficiency table.  The measured closed-column
  fallout differs from the separately rounded WRF state sum by exactly
  `2.9802322387695312e-8` kg m-2.
- Native CUDA cloud-ice-to-snow autoconversion now has a direct ice-saturated
  `ice-auto` oracle with one 200-micron mass-weighted ice layer.  It consumes
  canonical external FP64 `tps_iaus(64,55)` and `tni_iaus(64,55)` tables and
  composes with the admitted ice and snow fallout kernels.  CUDA differs from
  WRF by at most `1.82e-12` in ice mixing ratio, `8.89e-15` in snow mixing
  ratio and `0.0078125 kg-1` in ice number; all four frozen precipitation
  diagnostics remain within `2.53e-7` relative.  The composition gate also
  exposed and fixed per-step `RAINNCV`/`SNOWNCV` overwrite: later species now
  explicitly accumulate onto earlier fallout while standalone launchers keep
  overwrite semantics.
- Native CUDA ordinary rain evaporation now has a direct 99.5%-RH
  `rain-evap` oracle with a bounded 45-micron rain distribution.  The
  Srivastava-Coen mass and number sinks, vapor source, and evaporative cooling
  compose with admitted rain fallout while retaining WRF's pre-evaporation
  volumetric rain density.  Temperature and vapor are bitwise identical to
  WRF; rain mass and number stay within `2.46e-6` relative and surface rain
  within `8.31e-7` relative.  This gate exposed and fixed the cross-process
  density-ordering seam instead of masking it with a wider surface tolerance.
- Native CUDA ordinary snow sublimation now has a direct cold, 99.5%-ice-RH
  `snow-subl` oracle.  The Srivastava-Coen ice-saturation prefactor and Field
  snow moments drive a snow sink, vapor source, and sublimative cooling before
  admitted snow fallout.  Temperature, vapor, and all four surface
  precipitation fields are bitwise identical to WRF; final snow stays within
  `7.71e-8` relative (`1.46e-11` absolute).  WRF recomputes snow volumetric
  density after the thermodynamic update, so this path needs no rain-style
  density scratch.
- Native CUDA ordinary classic-graupel sublimation now has a matching cold,
  99.5%-ice-RH `graupel-subl` oracle.  It reconstructs WRF's rounded diagnosed
  number, exponential intercept, ventilation moment, and sphere-capacitance
  sublimation before admitted graupel fallout.  Temperature and vapor are
  bitwise identical to WRF; final graupel stays within `1.10e-7` relative
  (`1.46e-11` absolute), and all four surface fields stay within `1.18e-7`
  relative.  In classic option 8, graupel number is diagnostic rather than an
  exposed prognostic field and is re-diagnosed from updated mass before
  sedimentation.

The Thompson process/sedimentation kernel, direct GPU comparison and coupled
trajectory gates remain open; `mp_physics=8` therefore still fails closed.
