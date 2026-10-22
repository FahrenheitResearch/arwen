# Chemistry aerosol radiation feedback

Set `aer_ra_feedback = 1` in `[shared]` with an active chemistry set that
names `optics.gocart`, and select option 4 radiation with either
`ra_rrtmg_variant = "rrtmg_legacy"` or `"rte-rrtmgp"`. The requested feedback
uses the transported particulate fields at the radiation clock, including
the first call. It does not create aerosol initial or boundary conditions.
A clean initial state remains clean until emissions or inflow supply mass.

The legacy path uses the pinned WRF-Chem volume-Mie implementation and its
four-wavelength conversion to the 14 RRTMG solar bands, plus the 16 thermal
bands. The modern path uses the pinned RRTMGP MERRA aerosol Mie tables in its
native bands. They are separate spectral parameterizations and need not
give identical fluxes. Both cover every model level, leave above-model
buffer layers aerosol-free, and retain aerosols in clear-sky fluxes. Cloud
subcolumn masks apply only to cloud. Solar delta scaling follows mixing of
gas, cloud and aerosol optical moments.

The optics row supplies composition, size, humidity treatment and mass
conversion. The bulk `smoke` tracer represents fire PM2.5 as hydrophilic
organic particulate mass. It does not diagnose soot or brown-carbon
composition. Use resolved GOCART carbon rows when that composition matters.
The surrogate's `mass_multiplier = 1` preserves particulate mass, unlike
the carbon-to-organic-matter conversion used by organic carbon tracers.

`aer_opt` and a prescribed smoke manifest describe alternative aerosol
inputs. They cannot be combined with prognostic feedback because doing so
would replace or count the same aerosol twice. `aer_op_opt = 1` selects the
ported WRF-Chem volume mixing rule. Unsupported radiation families refuse
at configuration validation rather than dropping the requested optics.

History contains `EXTCOF55` (km^-1) and `AOD5502D` (dimensionless column
integral); the renderer's existing `aod_550` product reads `AOD5502D`.
Surface dimming and temperature response depend on aerosol placement,
composition, cloud and boundary forcing. Covering GPU tests establish
wiring and numerical invariants. Historical-event qualification requires
a populated aerosol analysis, a feedback on/off forecast pair, rendered
surface flux/temperature fields and observation checks. See the lane's
run report for the event evidence and any remaining gaps.

The WRF-Chem tables and arithmetic originate in public-domain WRF v4.7.1
(`LICENSE.txt`, WRF distribution). RTE-RRTMGP optical-property conventions
and MERRA tables carry the BSD-3-Clause notice already shipped in
`licenses/LICENSE-RTE-RRTMGP-BSD-3-Clause.txt`.
