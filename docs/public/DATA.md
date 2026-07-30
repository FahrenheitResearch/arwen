# Getting data

ArWen initializes from three public sources -- ERA5, GFS, and HRRR --
through one front door, `gpuwm fetch`, plus a user-staged static
geography tree. This page covers each route end to end, the manifest
handoff into the preprocessor, and honest disk numbers (all measured).

Two routes to keep straight from the start:

- **ERA5** feeds the config-driven GPU forecast loop
  (`gpuwm check` / `run` / `resume`) through the native GRIB1 decode
  path.
- **GFS and HRRR** are real GRIB2 downloads that feed the `rw-wps`
  native initialization front door, which emits
  `wrfinput_d0N`/`wrfbdy_d01` for unchanged stock WRF
  ([WRF-INTEROP.md](WRF-INTEROP.md)) and archives for
  [downscaling](DOWNSCALE.md). They do not feed the `[case_data]` GPU
  run path today; the domain wizard prints exactly which route your
  source gets.

Everything downstream of `fetch` is fail-closed: the Rust GRIB bridges
validate envelopes, inventories, grids, and hashes before decoding, and
refuse rather than approximate.

## GFS (0.25-degree, NOMADS)

```bash
gpuwm fetch --source gfs --cycle latest --hours 24 \
  --area 25,-110,45,-85 --out data/gfs-latest
```

- Downloads NOMADS `filter_gfs_0p25.pl` subsets: your spatial window
  plus exactly the 124 records per forecast hour the `gfs_grib2_bridge`
  requires (21 pressure levels x 5 variables, 11 surface, 8 soil).
  Every file passes a 124-message envelope walk before it counts.
- `--cycle latest` resolves the newest cycle whose *final* requested
  hour exists (anonymous S3 HEAD probes, newest first) -- a
  mid-publication cycle can never win.
- Emits `gfs-series.tsv` (relative paths; the directory is relocatable)
  and `fetch-manifest.json` with per-file SHA-256.
- **Disk:** about 3.3 KB per square degree per forecast hour. Measured:
  ~83 KB/h at 5x5 degrees, ~3 MB/h at 30x30, ~16 MB/h at 60x80.
- **Reach:** NOMADS retention is about 10 days. Older GFS cycles are
  not fetchable today (the raw S3 archive stores north-to-south grids
  the fail-closed bridge rejects by design).

**Antimeridian.** A `--area` longitude pair spanning more than 180
degrees is read as the complementary box crossing 180E:
`--area 45,170,60,-170` is the 20-degree Pacific box over the
dateline, never the 340-degree box that excludes it. `--point` /
`--radius-km` boxes wrap across the seam the same way, and GFS
antimeridian crops decode onto a continuous longitude axis. Boxes
genuinely wider than 180 degrees must be requested as the full band or
split.

**Area margin.** The front door must prove every model lake's nearest
source-water donor lies inside the crop, and interior-continental lakes
can sit far from GFS-resolved water. `gpuwm domain` suggests a fetch
area with the required margin already built in (interpolation-stencil
halo plus a 15-degree lake-donor allowance); if you author your own
area, allow 15 degrees beyond the outer domain.

### The manifest handoff (do not hand-author)

The GFS front door requires an input manifest binding every file's
SHA-256 -- including the bridge executable's own hash. One command
authors it from the fetch:

```bash
gpuwm fetch --source gfs --author-front-door-manifest \
  --out data/gfs-latest \
  --bridge tools/grib1_bridge/target/release/gfs_grib2_bridge \
  --wps-namelist configs/myarea.namelist.wps \
  --experiment-config configs/myarea.toml
```

This writes `gfs-input-manifest.json` (schema
`gpuwm-gfs-direct-input-manifest-v1`), prints its SHA-256, and prints
the complete ready-to-paste `rw-wps --source gfs ...` command with
`--bridge`, `--source-manifest`, and `--source-manifest-sha256` filled
in. Measured front-door cost after the handoff: 44.7 s for a
two-domain (d01 218x176 @ 12 km + d02 432x352 @ 3 km) static build,
decode, and initialization on the Rust CPU backend.

## HRRR (3 km CONUS, NOMADS or AWS Open Data)

```bash
gpuwm fetch --source hrrr --cycle 2026-07-28T00 --hours 18 \
  --point 35.2,-97.4 --radius-km 400 --out data/hrrr-2026072800
```

- Byte-range downloads via the published `.idx` indexes: the native
  hybrid-level `wrfnat` atmosphere subset (561 records/hour) plus the
  soil records of `wrfprs` (18 records/hour) -- the exact
  `hrrr_grib2_bridge` inventory. Existing files must pass the same
  record-count and manifest-digest bar as fresh downloads; failures
  are quarantined aside, never deleted.
- Two hosts serve byte-identical files and indexes: the NOMADS
  production mirror (roughly the newest 48 h, where each hour
  publishes first) and the full `noaa-hrrr-bdp-pds` S3 archive.
  `--transport auto` (default) probes NOMADS for the requested window
  and falls back to S3; `nomads`/`s3` force a host. The manifest
  records each file's transport, and a directory fetched over one host
  resumes over the other. (NOMADS' grib-filter scripts cover only the
  2-D `wrfsfc` file, so subsetting stays `.idx` byte ranges on both
  hosts.)
- `--wait-for` is the live-cycle mode: hours are downloaded in order
  as they publish (polling at most every 30 s), so `rw-wps` can start
  on the early hours before the cycle finishes publishing. On timeout
  (default 90 min, `--wait-timeout-minutes`) the manifest still
  records the complete fetched prefix and re-running the same command
  resumes.
- HRRR files are CONUS-wide; `.idx` subsetting selects records, not
  areas, so `--area`/`--point` act as a coverage check. `--cycle
  latest` requires both the final `wrfnat` and `wrfprs` objects to
  exist (with `--wait-for` it instead picks the newest cycle whose f00
  pair has begun publishing).
- The fetch prints the complete front-door handoff line
  (`--source-manifest SHA256SUMS --source-manifest-sha256 <digest>`).
- **Disk:** roughly 0.4 GB per forecast hour; ~8 GB for f00..f18.

## ERA5 (reanalysis, Copernicus CDS)

ERA5 requires a personal (free) Copernicus CDS account and API key, so
nothing is downloaded for you:

```bash
# 1. Emit the exact request + instructions
gpuwm fetch --source era5 --cycle 1999-05-03T00 --hours 24 \
  --area 30,-105,42,-90 --out data/era5-request

# 2. Retrieve with your own cdsapi key (instructions are printed),
#    concatenate the two GRIB parts, then:

# 3. Validate before anything expensive
gpuwm fetch --source era5 --validate era5-combined.grib
```

The emitted `era5-cds-request.json` is the precise two-part `cdsapi`
request: pressure-level z/t/u/v/r at all 37 levels, and the sixteen
single-level fields (sp, msl, 10u/10v, 2t/2d, lsm, skt, sst, ci, sd,
stl1-4, swvl1-4) plus invariant geopotential, at your times and area
(CDS order [N,W,S,E]).

`--validate` checks a retrieved set in seconds without decoding data
values: strict GRIB1 envelopes, the five pressure-level parameters at
every valid time with identical level ladders, the required surface
set at every time, soil accepted at native CDS or CDO-normalized level
types, and invariant orography presence. A wrong retrieval fails here,
not at initialization.

- **Disk:** tens of MB per valid time for a regional box at
  0.25 degrees.
- `--cycle latest` is rejected for ERA5 (reanalysis latency is weeks);
  ERA5 is the route for historical cases.

## Fetch semantics common to all sources (worth knowing)

- **Resumable by design:** re-running the same command verifies and
  skips complete files; extending `--hours` fetches only the new ones
  (per-hour files are byte-identical for the same source/cycle/area).
- **Refuses a changed request:** a different area, cycle, or source
  into the same `--out` refuses with the exact per-field difference.
  `--force-refetch` moves the old files aside (nothing is deleted) and
  re-downloads.
- **Refuses an unaccounted directory:** a nonempty `--out` without a
  readable `fetch-manifest.json` (a fetch interrupted before its
  manifest, or a corrupted one) also refuses -- the existing files
  cannot be tied to any recorded request, so they are never resumed.
  Fetch elsewhere or pass `--force-refetch` (quarantines, re-downloads).
- Every fetch writes `fetch-manifest.json`
  (`gpuwm-fetch-manifest-v1`): source, cycle, area, and per-file
  name/role/bytes/SHA-256. Resume re-verifies existing files against
  the manifest's recorded SHA-256 as well as the per-file record bars.

## Static geography (WPS_GEOG)

Terrain, land use, and soil come from the standard NCAR WPS
geographical dataset. Nothing downloads it for you, and this is the
one large one-time download:

1. Download the "highest resolution mandatory fields" WPS_GEOG archive
   from NCAR (~29 GB unpacked).
2. Stage it so these nine directories sit under one root:
   `topo_gmted2010_30s`, `modis_landuse_20class_30s_with_lakes`,
   `soiltype_top_30s`, `soiltype_bot_30s`, `greenfrac_fpar_modis`,
   `lai_modis_10m`, `albedo_modis`, `maxsnowalb_modis`,
   `soiltemp_1deg`.
3. Point `--geog-root` at that root, or set `GPUWM_CASE_DATA_ROOT` so
   it resolves as `${GPUWM_CASE_DATA_ROOT}/WPS_GEOG`.

Any area the global tree covers works; `gpuwm check` verifies the
exact tiles your footprint intersects (presence and hashes) before
anything expensive runs. Static fields arrive at 30 arcsec regardless
of nest spacing; no VAR_SSO/orographic-drag, urban-fraction, or
lake-depth datasets are produced in this release.

### `GPUWM_CASE_DATA_ROOT` layout

The root is the directory that *contains* your data, not a dataset
itself:

```
$GPUWM_CASE_DATA_ROOT/
  WPS_GEOG/            <- the nine geog dataset directories
  my-case-data/        <- whatever your config's [case_data] names
```

`gpuwm doctor` checks this layout and says exactly what is missing.

## Disk budget summary

| item | size | when |
|---|---|---|
| WPS_GEOG static tree | ~29 GB | once |
| GFS subsets | ~3.3 KB/deg2/h (e.g. ~3 MB/h at 30x30 deg) | per case |
| HRRR subsets | ~0.4 GB/h (~8 GB for f00..f18) | per case |
| ERA5 retrieval | tens of MB per valid time (regional box) | per case |
| Output frames (`wrfout`) | grid-dependent; ~198 MB/frame on a 250x200x49 domain; 20.1 GB for the four-domain 6 h reference run | per run |
| Restart checkpoints | ~5.7 GB per four-domain checkpoint on the reference case | per `restart_interval_s` |

Output fields and cadence are config-controlled; size a run's disk
before launching it, the way `gpuwm check` sizes its VRAM.
