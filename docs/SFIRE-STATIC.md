Refined SFIRE static preparation runs in Rust. Python acquires opaque public files and calls the native ABI. Fuel uses nearest source pixels; terrain uses WPS four-point REAL interpolation and centered fine-grid derivatives. The artifact carries NFUEL_CAT, ZSF, DZDXF, DZDYF, FXLAT and FXLONG, plus projection, refinement, spacing and source/field hashes.

Prepare from a resolved experiment:

```sh
gpuwm fire-static experiment.toml --domain 3 --request sources.json \
  --output fire-static.npz --receipt fire-static-receipt.json
```

Rust derives the root anchor and every child placement from the experiment's projection, dimensions, parent IDs, start indices and refinement ratios. The selected domain supplies sr_x and sr_y. Omit --domain when exactly one domain declares ifire = 2. Relative source and captured-perimeter paths resolve against sources.json. Without --request, the public terrain and fuel routes are acquired into the static cache; the experiment start year selects the latest available fuel release at or before that year.

A source request can name packaged routes:

```json
{
  "schema": "gpuwm-sfire-static-v1",
  "sources": {
    "terrain": "usgs-3dep-wcs",
    "fuel": "landfire-fbfm40",
    "fuel_year": 2024
  }
}
```

The route table is gpuwm/authorities/sfire-static-routes.v1.json. LANDFIRE FBFM13 and FBFM40 use the native 30 m EPSG:5070 lattice. Raw category codes remain in NFUEL_CAT; the coupled model applies its fuel crosswalk. Missing terrain becomes zero only where fuel identifies open water, code 98. Beyond LANDFIRE's coastal water margin the fuel raster holds NoData (-9999); those cells become open water (98) only where the terrain source has no value or lies at or below sea level, so a coastal fire domain can reach offshore. Missing fuel over land and terrain gaps on nonwater cells refuse with the affected cell.

For already captured inputs, replace routes with terrain and fuel objects containing path, sha256 and optional expected_bytes. Rust verifies and decodes those raster objects. A JSON request may instead declare an experiment path and grid_id, or provide an explicit grid_spec, sr_x and sr_y; these grid authorities are mutually exclusive. The standalone gpuwm-sfire-static console script accepts the same arguments. The rw-wps fire-static command delegates its JSON request to that native-backed entry point.

An optional observed_perimeter object adds LFN_HIST and HISTORICAL_TIGN:

```json
{
  "geojson": "captured-perimeters.geojson",
  "source_contract": "source-contract.json",
  "observed_utc_epoch_ms": 1784160751719,
  "model_elapsed_seconds": 751.719,
  "assumed_burn_age_seconds": 900.0
}
```

The captured contract binds the GeoJSON SHA-256, EPSG:4326 coordinates, UTC unix_ms time format and timestamp property. Rust preserves polygon holes and separate islands, projects all geometry and creates the signed-distance field. Negative values mark burned interiors. Burn age is an explicit uniform assumption because a perimeter observation does not measure ignition time within the burned area.

Set ifire = 2, the matching sr_x/sr_y and fire_static = "fire-static.npz" in the selected domain. The runtime loads the bundle through Rust. Load verifies ZIP/NPY structure, CRCs, every field hash, source hashes, shape, spacing and categories. When the runtime supplies its stage grid, every caller-supplied projection parameter and dimension must agree exactly. Derived reference coordinates may differ by at most 0.001 atmospheric cell. Fine coordinate arrays must describe that declared location. A same-sized bundle from another location is refused.

The Rust renderer exposes fire_perimeter, fire_ros and fire_heat_flux. Each product uses its own fine-grid coordinates, ZSF shaded terrain and labeled latitude/longitude graticules. The zero LFN contour marks the perimeter on all three products. ROS_FRONT uses m/s and FGRNHFX uses W/m2. FXLAT/FXLONG require explicit FIRE_COORDINATE_MODE = "geographic"; presence or degree-like variable units alone cannot distinguish geographic data from incorrectly labeled native ideal metric coordinates. Each PNG has a georef.json sidecar describing its actual projection and viewport. Product folders retain case/domain/product/day ordering.
