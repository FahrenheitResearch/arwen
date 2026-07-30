# RW-WPS namelist compatibility

RW-WPS accepts the user's standard `namelist.wps` and `namelist.input` as
authorities. It does not silently replace a scheme, shorten a domain array, or
force the historical 49-mass-level profile.

The machine-readable preflight is:

```text
gpuwm-wrf-init --namelist-support-report \
  --wps-namelist namelist.wps \
  --namelist-input namelist.input \
  --source-top-pressure-pa 5000
```

Exit status is zero only when the stock-WRF initialized-state contract passes;
an unsupported or unclassified setting prints the same JSON schema with
`"verdict": "FAIL"` and exits with configuration status 78. The report schema
is `rw-wps.namelist-support.v1`.

## What the report separates

- `preprocessing_relevant`: time, projection, ordered domain topology,
  horizontal geometry, explicit eta/p-top, external boundary, and similar
  settings that change native preparation.
- `physics_state_relevant`: scheme choices that alter the variables or lower
  boundary state which must exist in `wrfinput_dNN`.
- `runtime_output_only`: settings such as history cadence and physics call
  cadence which CPU WRF consumes after initialization.
- `legacy_stage_only`: WPS `ungrib`/`metgrid` controls retained in the report
  for visibility but replaced by native source decoding.

An unknown key is not guessed to be harmless. It receives
`UNCLASSIFIED_NAMELIST_SETTING` with an action describing the missing rule.

## Current initialized-state contract

The stock-WRF and gpuwm-runtime verdicts are intentionally distinct. RW-WPS
can inventory and prepare state for a CPU-WRF physics package even when the
gpuwm forecast runtime does not implement that package.

The following microphysics package inventories come directly from WRF v4.6.1
`Registry/Registry.EM_COMMON` package declarations and field I/O flags:

| `mp_physics` | Package | `wrfinput` package members |
|---:|---|---|
| 6 | WSM6 | `QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP` |
| 8 | Thompson | WSM6 mass species plus `QNICE QNRAIN` |
| 10 | Morrison two-moment | WSM6 mass species plus `QNICE QNSNOW QNRAIN QNGRAUPEL` |

All are WRF `real` fields: NetCDF float32 on
`Time,bottom_top,south_north,west_east`. Water vapor comes from the source;
source-absent hydrometeors and number moments use real.exe's zero
initialization policy. Restart/runtime-only effective radii and Morrison
convective tendencies are listed separately and are not falsely claimed as
required `wrfinput` variables.

The direct NetCDF writer resolves this inventory per domain. For Thompson and
Morrison it adds the corresponding number-moment variables to each
`wrfinput_dNN` and their value/tendency arrays on all four sides of root
`wrfbdy_d01`; the resolved contract digest and per-domain scheme are sealed in
the export manifest. Structural writer tests exercise 35, 49, and 80 mass
levels. An unchanged-stock-WRF launch remains a separate live evidence gate.

The accepted companion state is presently YSU (`bl_pbl_physics=1`), classic
MM5 surface layer (`sf_sfclay_physics=91`), four-layer Noah
(`sf_surface_physics=2`, `num_soil_layers=4`), no urban state, static one-way
nests, and Lambert conformal, Mercator, or polar stereographic geometry.
Other choices fail with the exact missing initialized-state adapter rather
than being changed.

## Domain and vertical behavior

Domain columns remain in d01...dNN order. WRF's short-array convention is
honored by repeating the final declared value; extra values beyond `max_dom`
are rejected rather than truncated. The parser supports the compiled
1-through-21 domain range and has an explicit six-domain regression gate.
`map_proj='lambert'`, `'mercator'`, and `'polar'` pass. Following WPS
`module_llxy` semantics, Mercator may omit `truelat2` and `stand_lon`, and
polar stereographic may omit `truelat2`; those parameters do not enter the
respective projection math. The Mercator, polar stereographic, and
southern-hemisphere Lambert acceptances carry oracle-verified plus
smoke-run-verified maturity, not matched-run verification. The report
verifies finite projection parameters and positive spacing, exact WPS/WRF
dimension and topology agreement, per-domain spacing implied by each parent
ratio, WRF's `(e_we-1)`/`(e_sn-1)` ratio divisibility, child containment
within its parent, one specified root plus nested children, and
`spec_bdy_width=spec_zone+relax_zone`. Latitude/longitude geometry, invalid
placement, and inconsistent boundary declarations fail with dedicated issue
codes before source data is touched.

Vertical acceptance is structural. `e_vert` must equal
`len(eta_levels)`, eta must decrease exactly from 1 to 0, all domains must
share the coordinate for the current one-way initializer, and p-top must lie
within source coverage. Tests cover 35, 49, and 80 mass levels. A source whose
atmosphere stops below the requested model top is rejected; RW-WPS does not
extrapolate it.

The support report is a preflight/state-inventory gate. A source/domain/scheme
combination is not called end-to-end certified until its native export is
opened by unchanged stock WRF and the resulting run has its separate evidence
receipt.
