# RUC hydraulic SPP native fixtures

These are whole `soil` and `snowsoil` outputs from current WRF v4.6.1 with
the exact historical WRF V3.9.1 hydraulic operator restored. They are not
outputs of an unmodified current WRF stochastic scheme. Current v4.6.1 and
v4.7.1 retain the SPP arguments but omit the hydraulic operator.

`manifest.json` pins both official module hashes, the generated overlay,
compiler, parameter tables and every CSV. Only these historical operations
are inserted before `end subroutine soilprop`:

```fortran
fieldcol_sf(k)=hydro(k)*rstochcol(k)
hydro(k)=hydro(k)*(1+rstochcol(k))
```

They run for every soil level when `spp_lsm == 1`. The separately removed
historical optics code is not restored. The current deterministic soil and
snow routines otherwise remain unmodified.

Rebuild on a CPU node using downloaded official source files:

```
python tools/ruc_spp_wrf_oracle/build.py module_sf_ruclsm-v4.6.1.F module_sf_ruclsm-V3.9.1.F oracle
python tools/ruc_spp_wrf_oracle/validate.py oracle --output native-validation.json
```

The sources are `wrf-model/WRF` GitHub paths
`v4.6.1/phys/module_sf_ruclsm.F` and
`V3.9.1/phys/module_sf_ruclsm.F`. The hydraulic operator is at historical
lines 6358-6365; current `soilprop` ends at line 6301. The driver fixtures
derive from the existing `tools/ruc_wrf461_oracle/run_soil.F90` and
`run_snowsoil.F90`, with explicit pattern input and conductivity diagnostic
columns added by the build script.

Four warm/frozen and four snow regimes each run with disabled SPP, enabled
zero, -0.9, -0.3, +0.3, +0.9 and depth-varying patterns. Every prognostic
field, conductivity diagnostic and snow output must be bitwise. Only the
four existing warm cosine-dependent fluxes (`edir1`, `eeta`, `qfx`, `evapl`)
retain their existing two-ULP CPU/native tolerance. That same tight limit
also passed the CUDA fixtures on RTX 4090. The old warm CUDA gate has wider
field limits; these fixtures do not use those wider limits.

The consumer tests also require column-permutation identity, zero-pattern
identity, finite physical states and a real moisture response to every
nonzero pattern. They are implementation qualification, not a demonstration
of forecast skill or calibration.
