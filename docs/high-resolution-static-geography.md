# High-resolution static geography

RW-WPS currently uses the standard 30-arc-second WPS geography tree for its
production static fields.  That is roughly 0.8 km east-west in Ohio, so it is
not a genuinely sub-kilometre description even when the model grid is 500 m
or 333 m.  The opt-in code in `gpuwm.static.highres` is the first narrow,
provenance-bound path beyond that baseline.  It reads local GeoTIFF subsets
directly and does not invoke `geogrid.exe`.

This is a pilot, not a claim of global or production support.  The normal
static builder and public RW-WPS command remain unchanged.

## Pilot data pack

The first pack covers an 18 km square around Caesar Creek, Ohio, inside the
real74 d04 footprint.  Every downloaded byte is bound by path, byte count,
SHA-256, source URL, nominal resolution, reference year, and licence in the
pilot manifest.

| Field | Source | Native resolution | Terms | Pilot treatment |
| --- | --- | ---: | --- | --- |
| Bare-earth terrain | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services) | 1/3 arc-second, about 10 m | US public domain | Area-average to the WRF spherical-Lambert cells, then one WPS smooth/desmooth pass |
| Land cover and inland water | [Annual NLCD Collection 1.2](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-land-cover-annual-nlcd-collection-12-land-cover) | 30 m | US public domain | Area fractions, explicit NLCD-to-WRF-MODIS-21 crosswalk; inland open water becomes lake category 21 |
| Sand, silt, clay | [SoilGrids v2](https://docs.isric.org/globaldata/soilgrids/wcs.html) | 250 m | [CC BY 4.0](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html) | Thickness-weighted 0--30 cm and 30--100 cm medians, normalized and classified with the USDA texture triangle |

The inland-water rule is deliberately scoped to this Ohio window.  It must
not be used at a coast, where WRF ocean category 17 and lake category 21 need
a coastline-aware split.

Annual NLCD begins in 1985.  The April 1974 case therefore uses the earliest
available map, which is still an explicit **11-year anachronism**.  It is an
experiment in spatial detail, not a historically exact 1974 surface.  A
modern map must never be presented as a silent historical replacement.

## Required Noah fields

The pilot replaces terrain, land-use fractions/index/mask, and top/bottom
soil fractions/categories.  It retains the hash-bound 30-arc-second WPS
monthly green fraction, LAI, albedo, snow albedo, and deep-soil temperature.
Where the new water mask exposes land that was water in the old mask, the
climatologies use a deterministic nearest old-land donor and the receipt
records the number of affected cells.  Newly identified water gets WRF water
fills and soil category 14.  `TMN` is recomputed from the merged surface.

SoilGrids does not provide a direct WRF soil category.  The pilot records the
raw sand+silt+clay total, normalizes the three components, and applies the
USDA texture rules.  Missing target land cells use the existing WPS soil
fractions only as an explicit, counted fallback.

## Scientific and operational gates

The implementation:

- verifies each source SHA-256 before decoding it;
- uses WPS's spherical Earth and mass-point registration for Lambert grids;
- performs continuous area averaging and categorical area-fraction
  aggregation rather than nearest-neighbour sampling at model-cell centres;
- reads no network data and refuses incomplete terrain or land-cover
  coverage;
- emits comparison arrays, plots, timings, source/fallback audits, and a
  receipt that states exactly what was and was not certified.

The pilot does **not** certify full real74 d04 coverage, a forecast
improvement, stock-WRF parity, coastal handling, historical surface fidelity,
or a public high-resolution-pack CLI.

## Caesar Creek result

The hash-bound pilot completed at both target spacings.  Timings below are one
Windows workstation run and are not a performance certification.

| Grid | 30s baseline build | High-res overrides | Terrain RMSE / max difference | Changed water mask | Changed land-use class | Changed top-soil class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 x 36 at 500 m | 0.188 s | 0.788 s | 3.591 / 21.832 m | 34 / 1,296 | 900 / 1,296 | 671 / 1,296 |
| 54 x 54 at 333.333 m | 0.206 s | 0.472 s | 4.816 / 36.058 m | 89 / 2,916 | 2,031 / 2,916 | 1,520 / 2,916 |

At 500 m, the new mask exposed 13 land cells and masked 21 old land cells;
the SoilGrids subset covered every target land cell.  At 333 m, those counts
were 38 and 51, with three land cells per soil layer using the declared WPS
fallback.  The large class-change counts are expected when comparing a 30 m
NLCD crosswalk against the older 30-arc-second MODIS/USGS products, but they
are not by themselves evidence that the new categories are more accurate.

## Global fallback design

The intended open-data fallback is [Copernicus DEM GLO-30](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html)
for 30 m terrain, [ESA WorldCover 2021 v200](https://worldcover2021.esa.int/)
for 10 m land cover, and SoilGrids 250 m for soil texture.  Copernicus GLO-30
is available under its Full, Free and Open licence; WorldCover and SoilGrids
are CC BY 4.0.  This fallback is design-only: it has not been downloaded,
wired into RW-WPS, or certified.  WorldCover 2021 also has the same, usually
larger, historical-anachronism problem.

Production work still required includes tiled/cache-aware downloading,
coastline and lake separation, full-domain halo coverage, selection policy,
public CLI/schema integration, attribution packaging, global fixtures, and
trajectory/stock-WRF gates.

## Reproducing the bounded pilot

Install the optional raster dependencies and run:

```text
python -m pip install -e ".[geog]"
python tools/run_highres_geog_pilot.py \
  --manifest /path/to/pilot-manifest.json \
  --geog-root /path/to/WPS_GEOG \
  --output /new/output/directory
```

The command refuses to overwrite an existing result directory.  Its receipt
binds the source rasters, the WPS baseline index files, code commit, generated
arrays, plots, timings, and quantitative comparisons.
