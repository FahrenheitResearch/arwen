# Public fire fuel and terrain inputs

`gpuwm fire-static EXPERIMENT.toml --domain GRID_ID --request sources.json --output fire.npz --receipt fire-inputs.json`
resolves the atmospheric projection and fire refinement in Rust. Bind the resulting
artifact with `fire_static` in that same domain. The bundle loader refuses a different
grid location, projection, shape, spacing, source hash or modified field payload.

The source rows live in `gpuwm/authorities/sfire-static-routes.v1.json`. Region rows
in `sfire-regions.v1.json` select CONUS, Hawaii or Alaska fuel and canopy coverage by
geographic bounds. Explicit source IDs can select either FBFM13 or FBFM40. An explicit
`region` is still checked against its geographic footprint. Unsupported regions have
no invented fuel default; bind a public fuel raster with `path` and `sha256`.

A source request is JSON:

```json
{"sources": {"fuel_year": 2024, "fuel": "landfire-fbfm40"}}
```

Use the release at or before the intended experiment date. The receipt records both
the requested year and actual available release. Post-event releases are not a
historical reconstruction. Fuel models retain their categories under nearest-pixel
sampling. Terrain uses four-point interpolation and the existing WRF REAL gradient
calculation. Independent public requests run concurrently; raster sampling uses Rayon.

Public LANDFIRE acquisition includes all four canopy fields by default. The shared
`sfire-input-fields.v1.json` table defines source scaling and admitted SI units:
cover fraction, height and base height in metres, bulk density in kg/m3. NoData is
masked before scaling. All four layers are required together. Missing land canopy
refuses preparation; captured open water may have zero canopy. Nonforest pixels
already carry explicit zero values. Canopy fields survive the sealed bundle, coupled
state, restart identity and history. They are admitted structural inputs; this change
does not add a crown-fire spread or combustion parameterization.

Legacy requests binding only terrain and fuel remain valid. To bind canopy rasters,
add `canopy` with keys `CANOPY_COVER`, `CANOPY_HEIGHT`, `CANOPY_BASE_HEIGHT` and
`CANOPY_BULK_DENSITY`, each holding a bound raster spec. A spec can include
`nodata_override` and `scale_factor`. Hashes bind the original encoded TIFF bytes.

CONUS and Hawaii select USGS 3DEP terrain. Alaska and other regions select Copernicus
DEM GLO-30 only after explicit research opt-in. Name its exact licence in the request:

```json
{"sources": {"accept_licenses": ["Copernicus-DEM-EULA-free-open"]}}
```

The route records the licence URL and required attribution. Copernicus is a surface
model, including vegetation and structures, with EGM2008 heights; it is not a bare-earth
3DEP equivalent. No vertical-datum conversion is claimed. Land holes are never filled
with sea level. The acquisition receipt retains absent tile IDs and the derive audit.

Render `fire_fuel`, `fire_terrain` and `fire_perimeter` through `gpuwm.rustwx` using
`rw_wrfbatch` from the coupled history. Terrain limits adapt to the actual domain
range. Fuel colours preserve category intervals without palette blending. The numeric
legend labels interval boundaries around the original fuel codes.
The zero level-set perimeter is overlaid on each layer.
