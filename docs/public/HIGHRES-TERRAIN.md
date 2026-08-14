# High-resolution terrain, worldwide

The default static geography everywhere is `topo_gmted2010_30s` — GMTED2010
at 30 arc-seconds, roughly 900 m. That is fine for a 3 km domain and much
too coarse for a 100 m one, where a whole ridge can fall inside a single
source cell.

`[static.highres]` replaces it with real high-resolution sources. Until now
that only worked in the United States. It now works internationally for
**terrain**.

## What you get, and where

| | Source | Resolution | Where it is published |
|---|---|---|---|
| Terrain (default abroad) | Copernicus DEM GLO-30 | ~30 m | 90 S to 84 N, all longitudes |
| Terrain (on request) | SRTM 1 arc-second v3 | ~30 m | 56 S to 60 N, all longitudes |
| Terrain (default in the US) | USGS 3DEP | ~10 m | conterminous United States |
| Land cover | Annual NLCD | 30 m | **conterminous United States only** |
| Soil texture | SoilGrids v2 | 250 m | global, but only wired into the US path today |

Neither Copernicus DEM nor SRTM needs an account, a token or an API key.
They are fetched by plain anonymous HTTPS. This is deliberate: the program
already asks for one set of credentials (ERA5 through CDS) and that single
requirement is its largest source of user friction. A second one would be a
worse product.

## The honest limitation: land cover is United States only

There is no global land-cover source wired, so **outside the United States a
high-resolution run replaces terrain and nothing else**. Land use, soil,
green fraction, albedo, LAI and deep-soil temperature all remain the
30-arc-second baseline.

You are told this three ways and never left to infer it:

- the console line says `APPLIED (terrain only, copernicus-dem-glo30; ...)`
  followed by an explicit sentence naming what stayed at 30 arc-seconds;
- the receipt carries `"mode": "terrain"`, a `scope_statement` in plain
  English, and a `fields_retained_30s` list;
- `fields = "auto"` resolves to `"terrain"` abroad, and asking for
  `fields = "all"` abroad refuses by name rather than quietly degrading.

Terrain alone is still worth having. It is the field that sets where air is
lifted, where cold pools drain and where a 100 m nest's vertical coordinate
sits — and it is the field the 900 m baseline damages most.

### Why not ESA WorldCover?

ESA WorldCover is the obvious global candidate: 10 m, CC-BY-4.0, anonymous
on AWS Open Data, 60 S to 84 N, and a legend small enough to crosswalk
(tree cover, shrubland, grassland, cropland, built-up, bare/sparse, snow and
ice, permanent water bodies, herbaceous wetland, mangroves, moss and lichen).

The blocker is not the crosswalk. It is class 80, *permanent water bodies*,
which — exactly like NLCD's class 11 — does not distinguish a lake from the
sea. The existing United States path maps open water to WRF's inland lake
category and therefore refuses any coastal domain outright. Wiring
WorldCover on the same rule would ship a "global" land-cover path that
refuses at every coastline, which in practice means most of Europe. The
prerequisite is a coastline-aware water rule, not another raster.

## Configuration

```toml
[static.highres]
enabled    = true
cache_root = "D:/gpuwm-cache/highres"

# Optional. Defaults shown.
terrain_source = "auto"   # auto | copernicus-dem-glo30 | srtm-gl1 | usgs-3dep-13as
fields         = "auto"   # auto | all | terrain
on_refuse      = "error"  # error | fallback-30s
```

`terrain_source = "auto"` uses 3DEP inside the conterminous United States
and Copernicus DEM GLO-30 everywhere else. Naming a source pins it, and a
domain that leaves that source's published coverage refuses with the source
id, the footprint and how far past the edge it went.

`fields = "auto"` selects `"all"` inside the United States envelope and
`"terrain"` outside it. `fields = "terrain"` is also valid inside the United
States — that is how the two terrain sources are cross-validated against
each other on the same domain.

## Choosing between Copernicus DEM and SRTM

Prefer Copernicus DEM unless you have a specific reason not to. It is newer
(TanDEM-X, 2010–2015), actively maintained, void-filled, and it reaches
84 N. SRTM stopped acquiring in 2000 and stops at **60 N and 56 S**, which
excludes Canada, Scandinavia, Alaska and most of Russia; ask for it there
and you get a refusal naming the cut-off.

They are not interchangeable in the vertical either. Copernicus DEM heights
are metres above the **EGM2008** geoid; SRTM uses **EGM96**; USGS 3DEP is
orthometric on **NAVD88**. The three differ regionally, so every run records
which source and which datum it used instead of treating the numbers as one
quantity.

Copernicus DEM and SRTM are **surface** models: they see forest canopy and
buildings. USGS 3DEP 1/3 arc-second is **bare earth**. That difference is
larger than the datum difference and it is measured below.

## What changing `terrain_source` does to your terrain

Measured, not asserted — one 50 x 50 km domain at 500 m over the Colorado
Front Range (39.55 N 105.55 W, 2.2 km of relief, spanning treeline) built
three times through the same code path. Full method, pre-registered
predictions and resolution limits are in the evidence gallery under
`2026-08-14-intl-highres-terrain`.

### Copernicus GLO-30 against USGS 3DEP

**Copernicus reads about 3.5 m higher, and that is the forest.**

| | median (Copernicus − 3DEP) |
|---|---|
| below 3000 m (forested) | **+3.34 m** |
| 3000–3500 m (dense subalpine forest) | **+5.43 m** |
| above 3500 m (alpine, bare rock and tundra) | **−0.07 m** |

The offset tracks vegetation and vanishes where vegetation does. Copernicus
is a surface model and 3DEP is bare earth; above treeline, where there is no
canopy for the two to disagree about, they agree to 7 cm. The remaining
datum term (NAVD88 vs EGM2008) is what is left up there: below the sources'
own accuracy in this footprint, though it will not be everywhere.

**Practical consequence:** switching a forested US domain from 3DEP to
Copernicus raises its terrain by a few metres. Switching an alpine or
desert domain barely moves it.

### The shape is the same

Once that single offset is removed:

| | |
|---|---|
| cells within 20 m | **9999 of 10 000** |
| cells past 10 m | 166 of 10 000 |
| median terrain slope, 3DEP vs Copernicus | 0.13703 vs 0.13671 m/m — **0.09 % apart** |
| ridge-vs-valley agreement (Laplacian sign) | **99.06 %** |
| residual spread after the offset | 4.2 m RMS |

That last figure is the two datasets' own vertical accuracy, not this code's
error: 3DEP at ~1–2 m RMSE and GLO-30 at 4 m LE90 combine to ~3.1 m, and
both degrade in steep terrain. It is the same size above treeline as below,
so it is not canopy variation either.

### Against the 900 m baseline, which is what you are replacing

| | 3DEP | Copernicus |
|---|---|---|
| RMS difference from the baseline | 27.5 m | 26.1 m |
| cells past 50 m | 749 | 659 |
| median slope, relative to baseline | **1.18×** | **1.18×** |
| ridge-vs-valley agreement with baseline | 86.5 % | 86.5 % |

The 900 m baseline is 18 % too flat, puts this domain's highest cell 73 m
too low and its lowest cell 56 m too high, and disagrees about where the
ridges are once every seven cells.

**The comparison that matters:** the two high-resolution sources agree with
*each other* 99.1 % of the time on ridge-and-valley placement, and with the
baseline only 86.5 %. Copernicus reproduces the US gold standard, which is
the evidence for trusting it where no gold standard exists.

You can reproduce any of this:

```
python tools/terrain_source_crossvalidation.py \
    --cache-root <cache> --out report.json
python tools/terrain_source_crossvalidation.py --self-test-only   # offline
```

It costs about 540 MB of cache, most of it one 3DEP tile.

## What the gates protect

- **Coverage is per source.** Each dataset declares its own envelope and the
  footprint is checked against the source actually selected, so a refusal
  says which dataset does not reach where — not merely that something was
  out of bounds.
- **Unpublished tiles.** Neither global product publishes all-water tiles; a
  404 is the product saying "this square is sea". That is one source making
  a claim about another source's land mask, so it is checked: an absent tile
  is filled with sea level only where the domain's own baseline mask already
  says water, and otherwise the run refuses naming the tile and the number
  of land cells underneath it. A footprint where *every* tile is absent is
  open ocean and refuses.
- **Antimeridian.** A domain straddling 180° yields a bounding box that
  claims the whole planet. It refuses rather than enumerating 64,800 tiles.
- **Coast safety.** The coastal refusal exists because the land-cover
  crosswalk's inland-water rule is not coast-safe. Terrain-only runs no
  land-use rule at all — `LANDMASK`, `LU_INDEX` and `LANDUSEF` pass through
  from the baseline untouched — so the ocean/lake distinction is never made
  and the gate does not apply. It is skipped there deliberately, not by
  oversight, and still enforced for `fields = "all"`.
- **Zero cells replaced is a refusal.** An enabled feature that changed
  nothing must never read afterwards as a feature that ran.

## Attribution

Runs that use these sources must carry their attribution. The exact strings
live on each source in `gpuwm/static/highres_fetch.py` and are copied into
every receipt.

- **Copernicus DEM GLO-30** — produced using Copernicus WorldDEM-30 © DLR
  e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided
  under COPERNICUS by the European Union and ESA; all rights reserved.
- **SRTMGL1 v3** — NASA JPL 2013, doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003;
  distributed by OpenTopography, doi:10.5069/G9445JDF.
- **USGS 3DEP** — public domain.
- **Annual NLCD** — public domain (MRLC).
- **SoilGrids v2** — CC-BY-4.0 (ISRIC).
