# hydro-world: fabric contract and the global fabric builder

`rw_hydro_world` builds a NextGen-shaped hydrofabric for any lon/lat box on Earth and writes it in the
HYDRO-WORLD contract (`arwen.hydro-world.fabric.v1`, spec `docs/dev/HYDRO-WORLD-CONTRACT.md`), the exact
GeoPackage the engine's `rw_hydrofabric prepare` / `gpuwm hydro prepare` reads.

The worldwide default (`global-copernicus-glo90`) is fully open: the river network and catchments are
derived from the Copernicus GLO-90 DEM itself (Priority-Flood fill, D8, flats, accumulation, streams,
reaches, catchments), lakes and reservoirs come from HydroLAKES. HydroSHEDS v1 (HydroRIVERS + HydroBASINS)
is RESEARCH / CROSS-CHECK ONLY under the 2026-10-18T11:50Z ruling: its restricted redistribution,
end-user EULA/records and revocable licence exclude it from shipment and default fetching. A private
standalone cross-check build refuses it unless the caller passes `--accept-licence 'WWF HydroSHEDS v1 License Agreement'`.

Everything the builder uses is a table row: `data/sources.v1.json` (inputs, licences, method, parameters)
and `data/recipes.v1.json` (channel geometry, level-pool constants, each with provenance). A new region or
dataset is a new row, not new code.

## Build a fabric

```
# on the build host: CARGO_TARGET_DIR and TMPDIR under your scratch
cargo build --release --offline -j 24
python3 hydro_world_build.py --bbox=4,46,12,52 --out rhine.gpkg \
    --data-root "$HYDRO_WORLD_DATA" --exe $CARGO_TARGET_DIR/release/rw_hydro_world \
    --engine /path/to/rw_hydrofabric --grid-dx 1000
```

The orchestrator (Python, orchestration only) asks `rw_hydro_world tiles` which DEM tiles the box needs,
fetches missing ones with a SOURCE.md row each (a 404 tile is open sea and goes to ABSENT.txt), runs
`rw_hydro_world build`, and optionally the engine reader with and without a parent grid in the fabric's own
CRS. Outputs: `FABRIC.gpkg`, `FABRIC.gpkg.terrain.json` (per-catchment elevation, slope, aspect, centroid
for hydro-world-params), `FABRIC.receipt.json` (every stage with time and parallelism, counts, the
read-back equality and the contract check).

Other commands: `check`, `roundtrip`, `schema`, `sources`, `admit`, `compare` (topology against a NOAA
EPSG:5070 fabric at its gauges), `render` (PNG of catchments, rivers by order, lakes, outlets).

## Measured (build host, 24 threads, 2026-10-18)

| Box | Cells | Reaches | Outlets | Lakes | Levels | Total | Check |
|---|---|---|---|---|---|---|---|
| Rhine 4-12E 46-52N (GLO-90) | 69.0 M | 18,917 | 259 | 159 | 219 | 3.25 s | 0 errors, lossless |
| CONUS check 101-99W 30.5-31.5N (GLO-90) | 2.9 M | 1,050 | 45 | 1 | 57 | 0.54 s | 0 errors, lossless |
| Rhine (HydroSHEDS research only) | basins | 2,943 | 122 | 150 | 107 | 1.5 s | 0 errors, lossless |

The only contract gap is `model_attributes` (soil/vegetation/CFE parameters, owned by hydro-world-params);
pass `--params` with their packet to write it. Receipts: `receipts/hydro-world-fabric/`.

## Tests

`cargo test --release --offline -j 24` (unit + contract + builder tests; the builder test needs the
GLO-90 tiles and HydroLAKES under `HYDRO_WORLD_DATA`, the fetched hydro-world data root; a
missing input fails, never skips). Add `RW_HYDROFABRIC=/path/to/rw_hydrofabric -- --include-ignored` for
the engine equivalence test.

## Licence

Apache-2.0. No third-party source code; the system SQLite is linked. Data terms: Copernicus WorldDEM-90
licence (attribution and no-liability sentence), HydroLAKES CC BY 4.0, both recorded in every fabric's
`fabric_metadata.data_terms`. Fabrics derived from HydroSHEDS v1 are never committed or bundled.

## DEM hydrography on the GPU (lane hydro-world-dem-hydrography, 2026-10-18)

The shipped worldwide fabric derives its own river network from the Copernicus DEM (coordinator ruling
2026-10-18T11:50Z: HydroSHEDS v1 is research and cross-check only). The per-cell passes now run on the GPU:

| Pass | GPU method (`src/cuda/hydrography.cu`, driven by `src/gpu.rs`) |
|---|---|
| depression fill | each cell becomes max(DEM, min over its 8 neighbours), repeated on 32 x 32 shared-memory tiles until stable; only tiles whose neighbourhood changed run again. Box-edge and sea cells are the outlets. |
| D8 | the same f32 drop and the same f64 row distances as the CPU, strict comparison in the same neighbour order |
| flats | Barnes, Lehman and Mulla (2014): flat labels and both distances (to lower terrain, from higher terrain) by the same tile relaxation, then the same mask and direction rule |
| accumulation | integer cell areas; one thread walks down from each source cell and only the last arrival at a confluence continues |

The GPU result is bit-identical to the CPU passes in `src/flow.rs`: the filled surface (every f32 bit), every D8 direction
and every accumulated area, on the Rhine box (69.1 M cells), the CONUS check box at GLO-90 (2.9 M) and at GLO-30 (25.9 M).
`rw_hydro_world hydrography-compare` runs both and compares every value; the Rhine fabric built on the GPU and on the CPU has
identical tables. The kernels are compiled by NVRTC at start into a CUBIN for the device (defaults: no fast math, no flush
to zero), through the in-tree `cudarc 0.19.8` pin with the driver loaded at run time, so the binary needs no CUDA toolkit.

`--device auto` (default) uses the GPU when a driver, a device and NVRTC load, otherwise all CPU cores, and says which in the
receipt (`counts.device`). `--device cuda` fails without a GPU; `--device cpu` forces the CPU.

New table rows:
- `global-copernicus-glo30` (1 arc-second, id namespace 3), and the table's `auto` list: `--row auto` takes GLO-30 when its
  grid over the box has at most 80 M cells, GLO-90 otherwise.
- a `dams` input (GDW v1.0 barriers) on both Copernicus rows: every dam in the box is snapped to the derived network (same
  area-ratio rule as lakes), linked to its fabric lake through the HydroLAKES id, and written to `FABRIC.gpkg.dams.json`
  (GDW id, name, flowpath, lake nexus, height, length, capacity, year, main use). The contract tables do not change; level-pool
  constants still come from the recipe.

`compare` also reports the drainage read at the gauge position (`drainage_at_gauge_rel_diff`), because the reach-interval
metric scores a coarse network as exact whenever the reference area falls inside one long reach.

The builder accepts `--table` for `--sources` and reads the data root from `GPUWM_HYDRO_WORLD_TILES` when `--data-root` is
absent, which is how the default-on door's provider row (`gpuwm/data/hydro/fabric-providers.v1.json`) calls it.

### Measured (an RTX 5070 Ti through the hydro pool, 24 threads; builder revision 99b4387a556)

| Box | Row | Cells | Reaches | Outlets | Levels | Lakes / dams | Build | Check |
|---|---|---|---|---|---|---|---|---|
| Rhine 4-12E 46-52N | GLO-90 (`--row auto`) | 69.1 M | 18,917 | 259 | 219 | 159 / 385 (345 on the network) | 2.40 s GPU, 3.32 s CPU | 0 errors, lossless |
| CONUS check 101-99W 30.5-31.5N | GLO-90 | 2.9 M | 1,050 | 45 | 57 | 1 / 0 | 0.56 s | 0 errors, lossless |
| CONUS check | GLO-30 (`--row auto`) | 25.9 M | 1,071 | 50 | 57 | 1 / 0 | 1.13 s | 0 errors, lossless |

Per pass on the Rhine box (ms, `hydrography-compare` on the RTX 5070 Ti host): fill 546 CPU vs 122 GPU; D8 + flats 584 vs 50; accumulation 452 vs
158 (65 in the build run); host-device copies 240. The largest remaining stages are CPU (24 threads): catchment pointer jumping
(0.30 s), GeoPackage write (0.31 s, the SQLite insert is one thread by design) and lake/tile reads. The `7 read dams` receipt row says
24 threads but is a 26 ms single-thread scan of the 41,145-row GDW table (label corrected in the code after this run).

Rhine drainage at the gauges (DEM-derived, gauges snapped with no use of the published area; the area-checked snap picks the
same reaches), against the BfG published catchment areas:

| Gauge | DEM-derived | Published (BfG Undine) | Difference |
|---|---|---|---|
| Kaub | 103,591 km2 | 103,729 km2 | -0.13 % |
| Koeln | 144,395 km2 | 144,232 km2 | +0.11 % |
| Rees | 159,316 km2 | 159,300 km2 | +0.01 % |

With `model_attributes` from `rw_hydro_params derive` the Rhine fabric checks 0 errors and 0 gaps, and `gpuwm hydro prepare`
reads it (18,917 catchments, 219 levels, 21 payloads with a 1 km grid in the fabric's own LAEA).

CONUS check box against the NOAA NHDPlus-derived fabric at the 9 gauges whose whole basin is in the box (5 % tolerance):

| Build | Matched | Drainage, reach interval | Drainage at the gauge | Sources (magnitude) | Main stem | Outlets |
|---|---|---|---|---|---|---|
| GLO-90 (DEM) | 9/9 | 7/9 (max 8.6 %) | 5/9 | 8/9 | 4/9 | 45 |
| GLO-30 (DEM) | 9/9 | 8/9 (max 5.4 %) | 5/9 | 8/9 | 4/9 | 50 |
| HydroSHEDS v1 (research only) | 9/9 | 9/9 (max 1.4 %) | 5/9 (max 477 %) | 3/9 | 6/9 | 18 |

The misses are small headwater basins (20, 113, 275 km2) on the flat Edwards Plateau, where a DEM-only divide departs from the
NHDPlus divide; HydroSHEDS passes the interval metric there only because one long reach covers the whole range.

Receipts: `receipts/hydro-world-dem-hydrography/` (SUMMARY.json plus every build, pass comparison, network comparison, gauge
match, parameter and prepare receipt, each with its command, per-stage time and parallelism). Proof scripts: `proof/`.

### Attribution for every source used

- Copernicus DEM GLO-90 and GLO-30: produced using Copernicus WorldDEM-90 / WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus
  Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The
  organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the
  Copernicus WorldDEM-90 / WorldDEM-30.
- HydroLAKES: Contains HydroLAKES v1.0 data (Messager, M.L., Lehner, B., Grill, G., Nedeva, I., Schmitt, O. (2016), Nature
  Communications 13603, doi:10.1038/ncomms13603), CC BY 4.0, www.hydrosheds.org.
- GDW: Contains Global Dam Watch database v1.0 (Lehner et al. 2024, Scientific Data, doi:10.1038/s41597-024-03752-9; data
  doi:10.6084/m9.figshare.25988293), CC BY 4.0. Includes GRanD v1.3, used only through GDW.
- SoilGrids (model_attributes): Contains SoilGrids 2.0 data, ISRIC World Soil Information, CC BY 4.0 (Poggio et al. 2021,
  SOIL 7, 217-240).
- ESA WorldCover (model_attributes): (c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021)
  processed by ESA WorldCover consortium, CC BY 4.0.
- PEGELONLINE (gauge positions and observed mean flow for snapping): Pegeldaten: Wasserstrassen- und Schifffahrtsverwaltung
  des Bundes (WSV), via PEGELONLINE, dl-de/zero-2-0.
- Verification references only, never inputs to a built fabric: published catchment areas from the BfG Undine gauge pages
  (Bundesanstalt fuer Gewaesserkunde, undine.bafg.de); the NOAA NextGen hydrofabric v2.0.1 subset in the t-route repository
  (Apache-2.0 repository, NHDPlus-derived, held as private terms by the hydro lead); HydroSHEDS v1 (WWF HydroSHEDS v1 License
  Agreement, research cross-check only, its fabric stays in scratch).

## Original CONUS source terms

The `conus-nhdplus-v21` river-network and catchment entries use EPA/USGS
NHDPlus V2.1. The [original dataset metadata](https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Documentation/Metadata/NHDPlusV2_metadata.htm)
states `Access_Constraints: None.` and `Use_Constraints: None.` Credit the
U.S. Environmental Protection Agency and U.S. Geological Survey. The metadata
read on 2026-10-18 has SHA-256
`c4a6c790321f87f69666b0ed28669acc5eb73c0771a9a0ee5dc9a93a4ccdbb22`
(335,058 bytes, UTF-16). These two input terms are verified in the source table.

This evidence covers the original NHDPlus dataset, including its topology and
catchment geometry. It does not grant terms to separate NOAA parameter or
reference-fabric products. The four ancillary selections are now also verified
from their publishers: [USGS 3DEP seamless 1-arc-second DEM](https://pubs.usgs.gov/tm/11b9/tm11B9.pdf)
(public domain; CONUS orthometric metres in NAVD88, matching this row),
[SoilGrids 2.0](https://isric.org/explore/soilgrids),
[ESA WorldCover 2021 v200](https://esa-worldcover.org/en/data-access) and
[HydroLAKES pour points v1.0](https://www.hydrosheds.org/products/hydrolakes)
(the latter three CC BY 4.0). Their source-page byte hashes, read date and short
licence evidence are in the table. Preserve ISRIC/Poggio et al. (2021), ESA's
WorldCover/Sentinel attribution and Messager et al. (2016) credit. The CONUS
recipe is therefore a mix of federal data and attribution-licensed inputs;
CC BY 4.0 inputs are not described as public domain.

The native `admit` command now passes **source terms only** for this row. It does
not establish valid geometry, complete domain coverage, correct parameters,
initialized stores or a connected native phase. The CONUS build specification
is still absent, so `build` refuses before writing a fabric. Its provider stays
pending and has no admitted coupled realization. No new guard is introduced.

Only the HydroLAKES archive URL is an exact verified acquisition object here.
NHDPlus region packages, 3DEP tile editions, SoilGrids property/depth/statistic
VRTs and their referenced rasters, and WorldCover covering tiles still need exact
object selection and byte pins. Their `url` fields remain null; publisher
landing pages or bucket roots are not substituted for data objects. No raster,
lake archive or source dataset was downloaded for this source-terms update.
HydroLAKES IDs are co-registered to HydroSHEDS, not NHDPlus ComIDs; a same-network
lake crosswalk must be proved before using its attributes in the CONUS fabric.
The shared world-fabric writer/checker remains authoritative; this table change
does not add a competing writer, silently repair geometry or invent parameters.
