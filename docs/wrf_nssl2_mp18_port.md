# WRF v4.6.1 NSSL two-moment CUDA port

## Exact target

The numerical authority is NCAR WRF v4.6.1 commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`.  The target is the unified native
`mp_physics=18` default introduced in WRF v4.6: two moments, hail, predicted
CCN, predicted graupel and hail particle volume/density, and no optional sixth
moments.  It is not a clone of deprecated `mp_physics=17`; that spelling maps
to option 18 with predicted CCN disabled.

Primary source anchors:

- `doc/README.NSSLmp:16-126` describes the scheme, compatibility modes, and
  v4.6 changes.
- `Registry/Registry.EM_COMMON:2410-2425,3033,3049-3056` defines selectors,
  defaults, and allocated packages.
- `share/module_check_a_mundo.F:3377-3465` resolves the unified mode.
- `phys/module_physics_init.F:4615-4652` packages namelist values and selects
  two- versus three-moment initialization.
- `phys/module_microphysics_driver.F:1938-2018` maps WRF state to
  `nssl_2mom_driver`.
- `phys/module_mp_nssl_2mom.F:1168-2219,2224-3410` contains initialization and
  the public driver; the remaining module contains the process kernels.

The v4.6.1 Registry default `nssl_rho_qhl=900 kg m-3` is authoritative.  The
module declaration and `README.NSSLmp` still say 800, but
`module_physics_init.F` forwards the Registry value into `nssl_2mom_init`,
which overwrites the module declaration.

## State, restart, and coupling

The native default transports mass `qv,qc,qr,qi,qs,qg,qh`; number
`qndrop,qnr,qni,qns,qng,qnh`; predicted CCN `qnn`; and volume moments
`qvolg,qvolh`.  All are trajectory state and therefore belong in restart,
boundary/nesting, halo, and checksum identities.  Optional three-moment mode
adds sixth moments `qzr,qzg,qzh`; it is outside the first admission target.

Radiation consumes `re_cloud,re_ice,re_snow`.  NSSL computes these every
microphysics step in `calc_eff_radius` and clamps them in the public driver.
Radar reflectivity is mandatory for option 18 (`refl_10cm`), and category
precipitation includes rain, snow, graupel, and hail accumulated and per-step
fields.  Hail maximum diagnostics are also part of the driver interface.

The process network includes nucleation, explicit condensation, deposition,
evaporation/sublimation, collection/coalescence, variable-density riming,
shedding, ice multiplication, aggregation, freezing/melting, category
conversion, and adaptive multi-moment sedimentation.  WRF's source recommends
WENO-5 positive-definite transport (`moist_adv_opt=4`, `scalar_adv_opt=4`) to
preserve mass/number relationships at cloud edges.

## Admission gates

The runtime remains fail-closed until all of these land:

1. Exact option resolution, setup constants, full prognostic allocation,
   initialization, restart, nesting, and transport.
2. CUDA ports of every active default process and adaptive sedimentation,
   tested first as isolated slices against compiled official WRF source.
3. Category precipitation, effective radii, reflectivity, and hail diagnostics.
4. Official-WRF deterministic process, full-column, restart-split, and nested
   parity gates with explicit FP32 error budgets.
5. Coupled radiation/dynamics integration, water/number/volume invariants,
   long-column stability, and GPU performance evidence.

Until the gates are complete, `mp_physics=18` yields one detailed compatibility
blocker and cannot silently fall back to a neighboring scheme.

## Admitted CUDA slices

The staged implementation currently admits independently callable,
official-source-oracled pieces while the global selector remains fail-closed:

- radiation effective radii for cloud water, cloud ice, and snow;
- mass-only initialization of all default mass/number/CCN/volume moments; and
- trajectory-changing rain self-collection/breakup, including native mean-size
  bounds, efficiency branches, process shutoff, and timestep depletion limit.

The self-collection kernel mutates `qnr` in Registry #/kg units and preserves
`qr`.  It is a real forecast-state process slice, but is not yet a substitute
for the complete warm-rain network or the full option-18 dispatcher.

- trajectory-changing Ziegler warm-cloud autoconversion, coupling cloud/rain
  mass and droplet/rain number with the native 7.51-micron threshold,
  depletion guards, and post-process two-moment bounds.  Its 48-cell oracle
  is a direct `nssl_2mom_gs` run with no initial rain, so accretion and rain
  self-collection are identically excluded;
- trajectory-changing rain collection of cloud droplets, including the
  initiation-radius gate, both collector-radius branches, independent
  mass/number depletion limits, and post-process moment bounds;
- warm-rain evaporation into vapor with latent cooling, the exact native
  0.002-K saturation lookup, Wisner ventilation, proportional rain-number
  loss, the 10-percent process cap, and evaporation-only gate; and
- clear-air warm-cloud activation with the native two-pass 0.4-percent
  saturation adjustment, Twomey droplet population, predicted-CCN coupling,
  mean-mass bounds, and cloud-free cleanup;
- existing-cloud warm-water adjustment with native analytic full/partial
  evaporation, predicted-CCN restoration, adaptive RK2 condensation through
  the 0.5-percent cloud-interior-renucleation boundary, and final droplet
  bounds; and
- default `irenuc=2` cloud-interior renucleation above that boundary, coupling
  adaptive condensation to Twomey/Cohard-Pinty activation, predicted-CCN
  depletion, vertical boundary gates, the half-condensed-mass cap, and final
  droplet bounds;
- cold-phase two-moment snow aggregation/self-collection, preserving snow
  mass while applying the native temperature-efficiency ramp, snow-size
  diagnosis, ten-percent process depletion limit, and final number/volume
  bound; and
- cloud-ice growth by vapor deposition with latent heating, exact native
  column-ice size/capacitance and ventilation, the two-pass ice-saturation
  bound, and default `iscni=4` conversion of half the positive deposition to
  snow once the diagnosed ice maximum dimension reaches 100 microns; and
- adaptive two-moment rain sedimentation, including native q/N/Z fall speeds,
  one- and multi-substep CFL paths, hybrid size sorting, surface export, and
  precipitation accumulation.

These are admission slices, not a composed substitute for `nssl_2mom_gs`.
The maximum-supersaturation QVEXCESS branch, rain condensation, cloud-ice
sublimation and deposition onto existing snow/graupel/hail, the remaining
frozen-category processes and conversions, their shared aggregate limiters,
remaining sedimentation categories, and full diagnostic/integration gates are
still required before the selector can be unlocked.
