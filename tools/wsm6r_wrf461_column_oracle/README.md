# WSM6R column port

Column conformance only. No matched whole-forecast WRF trajectory claim.

WRF source is public domain. This code uses WOOF's own floating point math
routines through the registered header and direct NVRTC loader routes.
No additional math source is vendored.

## Source review

Reference commit d66e442fccc04111067e29274c9f9eaccc3cef28, WRF 4.6.1.
Read phys/module_mp_wsm6.F alongside phys/module_mp_wsm6r.F and compare
phys/physics_mmm/mp_wsm6.F90, the actual WSM6 calculation body.
The source diffs and parameter CSV are delivery receipts. They were
reviewed during continuation of the inherited transcription.

The shared WSM6 host module, constants module, six-species state layout,
kernel loader, diagnostics and oracle column layout are reused. WSM6R
uses a new CUDA translation unit because WRF reorganizes the full process
sequence and changes sedimentation. The existing frozen WSM6 kernel and
its compilation source remain unchanged.

| WRF WSM6R lines | Revised calculation |
|---|---|
| 5-51 | Module-local parameters instead of WSM6 caller constants and hail initialization. The parameter inventory records every declared value. |
| 85-171 | Wrapper converts theta to temperature and back; output is RAINNC and RAINNCV only. |
| 224-244 | Initialization per call, initial negative-mass clipping, nint(dt/120) minor loops. |
| 251-292 | Apply and limit each process in sequence: fallk, fallkc, rainsc, warmr, accret1, accret2, accret3, pconadd. WSM6 accumulates rates before its combined donor limiting. |
| 300-327 | Saturation and humidity evaluation is recalculated after process updates. |
| 332-451 | Derived power-law coefficients. The 60-word frozen initialization table contains every coefficient read by the device; the WRF dump and enum order test bind the words. |
| 513-678 | Explicit upstream rain, snow and graupel flux with column substeps; snow melting then graupel melting after each substep. No semi-Lagrangian PLM remap. |
| 683-739 | Explicit substepped cloud-ice flux with its revised power laws. |
| 744-853 | Surface fallout, ice melting, homogeneous freezing, heterogeneous freezing and rain freezing, with updates after each. |
| 857-952 | Cloud autoconversion, rain accretion and rain evaporation or condensation, each donor-limited and applied immediately. |
| 956-1165 | Rain/ice, ice/rain, snow/ice, graupel/ice, snow/cloud and graupel/cloud accretion with sequential updates. |
| 1170-1389 | Rain/snow, snow/rain, graupel/rain and graupel/snow collection, then snow and graupel melting. |
| 1394-1676 | Ice/snow/graupel deposition, ice generation, snow and graupel conversion, snow and graupel evaporation. Saturation is reevaluated between terms. |
| 1682-1724 | Final cloud condensation or evaporation and latent heating. |
| 1729-1753 | Binary64 logistic smoothif switches replace many discrete process thresholds. |
| 1786-1810 | Local heat capacity, latent heat and condensation denominator evaluation. |

Driver: phys/module_microphysics_driver.F:2356-2376 supplies theta, six
masses, density, Exner, pressure, layer thickness, dt and precipitation.
It supplies no hail_opt. module_physics_init.F:1004-1023 omits WSM6R from
the microphysics radius list. WSM6R calls its own initializer internally.

WSM6R does not use WSM6/WDM6's sedimentation remap, so the remap water-loss
divergence does not apply to this arm. Source: fallk and fallkc above.

## Mechanical gate

Build the unmodified WRF source on a CPU box:

    bash tools/wsm6r_wrf461_column_oracle/build.sh "$WRF_SOURCE_ROOT" "$L/oracle-build"

The build pins the WRF commit and refuses changed oracle sources.
The reference uses gfortran 13.3.0 -O0 and FP32. The optimized WRF arm is
also recorded. 66 columns, 45 levels, 30/60/150/240/300 second time steps,
and 28 total calls cover convection, stratiform, cold, warm, terrain and
threshold/negative/low-density/thin-layer edge cases.

Through the GPU card queue, in fresh processes:

    GPUWM_WRF_EXACT=1 python tools/wsm6r_wrf461_column_oracle/run_gpu.py check-fixture tests/data/oracles/wsm6r --out strict.json
    python tools/wsm6r_wrf461_column_oracle/run_gpu.py check-fixture tests/data/oracles/wsm6r --out default.json
    GPUWM_WRF_EXACT=1 python tools/wsm6r_wrf461_column_oracle/run_gpu.py plant tests/data/oracles/wsm6r --out plant.json
    GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_wsm6r_wrf461_column_exact.py

The gate requires exact finite words and exact nonfinite locations at
every field and every step, with no excluded column. Raw field hashes
are retained. Final fields also report word counts and ULP values.
The mutation flips the low mantissa bit of pgmlt_b, a revised graupel
melting term, only after requiring the clean baseline to pass.

Declared divergences:
* RAINNCV accumulates every minor loop per call instead of reporting the
  last loop. RAINNC itself remains exactly graded. The per-call increment
  check uses at most eight ULP of the accumulated rain total.
* At dt150 the thin-layer edge makes WRF itself nonfinite at two levels.
  NaN sign and payload may differ. Raw hashes and payload counts remain
  visible; canonical NaN hashes grade the location without excusing a
  finite mismatch.

WSM6R writes no native reflectivity or radii. The engine diagnoses WSM6
reflectivity and WSM6 effective radii on the revised state for its output
and RTE+RRTMGP coupling. Legacy RRTMG keeps WRF's own-radius behavior.
Mixed-scheme nest edges touching 106 remain refused with the missing
ratification named. Same-scheme 106 edges are admitted.

Public forecast witness: configs/wsm6r_column_forecast.toml and its
namelist pair. It uses public pressure-level forcing and published static
fields, 50x50x49 cells at 3 km, 720 seconds, and three history frames.
See the delivery REPORT.md for commands, receipts and coupled-run verdict.
