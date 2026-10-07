# Bulk smoke from the coupled fire

Set `fire_smoke = true` alongside an active `ifire = 2` fire domain. This
selects `chem_sets = "sfire_smoke"` before preparation, memory pricing and
cache identity. A nesting tree carries the same row on its parent and
children; domains without a fire grid have zero local source. Initial
background and external inflow are zero unless a separately declared
input supplies them. No satellite emissions are required.

The source follows WRF 4.7.1 SFIRE: each completed step's consumed fraction
`BURNT_AREA_DT` times `FGIP`, multiplied by `fire_tracer_smoke` (default
0.02). `FGIP` is the native effective wet fuel loading. The heat routine's
dry-burned mass is a different quantity and is not substituted. The source
is already a completed-step amount, so there is no second timestep factor,
hourly persistence rule or diurnal multiplier.

The native increment is in grams of bulk smoke per kilogram of air. The
existing tracer arena stores micrograms per kilogram of dry air, using
`native_increment * 1e6 * rho_air / rho_dry`. It emits once before RK tracer
transport; that same step transports the new smoke. It has no extra
Freitas plume calculation: the coupled atmosphere responds to the fire's
heat, while `fire_smk_scheme = 0` releases smoke into the first layer and
`fire_smk_scheme = 1` uses the corrected conserved Gaussian distribution
with `fire_smk_peak`, `fire_smk_ext` and `fire_tg_ub`.

`FIRE_SMOKE`, `SFIRE_SMOKE_SFC` and `SFIRE_SMOKE_COLUMN` are bulk smoke,
with units ug/kg dry air, ug/m3 and mg/m2 respectively. They are separate
from the satellite PM2.5 row and are not part of `PM2_5_DRY`, an AQI
calculation, or a validated particulate-size or chemical-composition
prediction. The Rust renderer draws `SFIRE_SMOKE_SFC` and
`SFIRE_SMOKE_COLUMN` as `smoke_near_surface` and `smoke_column`; a run
that also carries the transported AQ smoke row draws their total.
`SFIRE_SMOKE_FUEL_SOURCE`, `SFIRE_SMOKE_INJECTED` and
`SFIRE_SMOKE_EMITTED` record expected fuel-derived mass, added mass using
density and thickness, and accumulated areal emission. The run receipt
also carries the shared tracer ledger; closure accounts for every change
but does not independently prove conservation.

The native registry name `fire_smoke` is also written in
`g_smoke/kg_air`. A supplied native `wrfinput` volume with that name is
converted into the same dry-air row at cold start. The native and AQ
output names declare different units; conversion round trips can differ
by a few floating-point ULPs. An input volume with smoke disabled is
refused so its state cannot be silently discarded.

The optional `chem_sets = "sfire_smoke_mixed"` profile adds the existing
vertical mixing process to the same row. It requires a PBL scheme that
publishes `exch_h`. The passive profile runs the original ideal case with
PBL disabled. Bulk smoke has no selected dry deposition, wet removal,
optics or microphysics feedback.

The constructor echoes the canonical set and shortcut consistently.
Clearing both `fire_smoke` and the SFIRE smoke set disables the feature.
Native `&dynamics tracer_opt = 3` imports the same bulk profile; supported
`&fire` smoke controls retain their native values.

An existing chemistry-off prepared cache must be explicitly augmented or
prepared again. The augmentation initializes only the endogenous bulk
row and its declared ledger/source arrays to zero, preserves meteorology
and input source identities, and records original and transformed cache
digests. It does not reinterpret a restart or admit arbitrary mismatched
input configurations. A checkpoint keeps the bulk row, source clock and
accumulated source/ledger arrays, so continuation does not replay emission.

The source-only `sfire_smoke` profile also runs on atmospheric tiles. One
domain owner updates the mass ledger with the resident reduction order;
tile totals are not summed into a different result. Firebrands retain
their separate single owner. Ring and shadow stores preserve source clocks
and the native fire state through a fresh disk restore.

History diagnostics are rebuilt outputs. A restore recomputes them in
bounded tile buffers from the saved tracer and atmospheric operands before
publishing a frame. The mixed profile requires resident execution because
the tiled accounting owner does not account for post-step vertical mixing.
