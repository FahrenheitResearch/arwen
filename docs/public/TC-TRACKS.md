# Tropical cyclone tracks from completed output

`gpuwm tc-tracks` consumes completed fields. It writes `tracks.json`, `tracks.csv`,
`manifest.json` and a timing/provenance `receipt.json`. It does not run a forecast.

```bash
export GPUWM_EARTH2_PYTHON=/path/to/earth2studio-environment/bin/python
gpuwm tc-tracks --input earth2-fcn3-20240924T12.nc \
  --best-track best-track-AL092024.txt --out tracks/ --render
gpuwm tc-tracks --manifest members.json --backend vitart --workers 4 --out tracks/
GPUWM_DETECTNODES=/path/to/DetectNodes GPUWM_STITCHNODES=/path/to/StitchNodes \
  gpuwm tc-tracks --manifest members.json --backend tempestextremes --workers 24 --out tracks-te/
```

The default backend is `wuduan`. Earth2Studio is imported only by a subprocess of
`GPUWM_EARTH2_PYTHON`. Both GPU backends use one completed valid time per call and
reset the path buffer between members. Their 300 km search distance and two-step
history are recorded with each member's actual cadence. `--workers` bounds native
field-extraction processes and TempestExtremes workers. Detection divides the budget
between members and disjoint valid-time groups, then stitches each full time series once. Extraction divides
available CPU threads among its processes. GPU members run sequentially on the
assigned GPU; use the site's resource allocator before invoking the command.

The interpreter needs Earth2Studio's cyclone dependencies, including CuPy,
cuCIM, SciPy and scikit-image. Importing the tracker class alone does not check
these optional dependencies. For a read-only Earth2Studio environment, install
compatible dependency wheels into a separate directory and name it with
`GPUWM_EARTH2_EXTRA_SITE`. Only the tracker subprocess prepends it to `PYTHONPATH`.
Keep the supplied interpreter and environment
unchanged. Match the CuPy/cuCIM CUDA major version to the installed runtime.

The native helper is `rw_tc_fields`; the renderer is `rw_wrfbatch`. Bundled bridges
are discovered normally. A source build can name them with `GPUWM_RW_TC_FIELDS`
and `GPUWM_RW_WRFBATCH`. No Python atmospheric-array decoder is used by this door.

## Manifest

```json
{
  "schema": "gpuwm.tc-tracks-manifest.v1",
  "members": [{
    "member_id": "ai-000", "family": "ai", "source": "earth2-fcn3",
    "init": "2024-09-24T12:00:00Z", "weight": 1.0,
    "ancestry": {"cycle": "2024-09-24T12:00:00Z"},
    "vocabulary": "earth2studio", "file": "forecast.nc"
  }]
}
```

Families are `ai`, `arwen`, `hex` or `analysis`. Paths are relative to the manifest.
Instead of `file`, supply `frames: [{"path":"frame.nc","valid":"2024-09-24T18:00:00Z"}]`
and an explicit `init`. Each frame can override `vocabulary` and supply a `select`
object of dimension names to integer indices. A file's independent initialization,
lead-time and ensemble axes expand into separate members. Ensemble identity is
retained in the member suffix and each frame's selection. Explicit weights are
split across expanded members. Ancestry is carried through as arbitrary JSON.

Omitted weights default to `1/N` **within each family**. All weights in a family
must be explicit or all omitted. Family totals are never combined into a single
probability distribution. Tracks retain their member's weight, including tracks
with missing fixes. Multiple candidates from one member are not independent members.

`--input` is a one-path shortcut for an Earth2Studio NetCDF, an ERA5-name Zarr, one ArWen GRIB2 valid time, or
an ArWen run directory. A run without an exported Zarr invokes `gpuwm ml-export`
with the required pressure-level fields and earth-relative winds. Use a manifest
for explicit source/family/ancestry, GRIB2 frame series, or converted hex output.

## Fields and regional coverage

`gpuwm/authorities/tc-track-fields.v1.json` is the field authority. New model names
that use a vocabulary need no reader change. A new export vocabulary is a new row
with variable names or GRIB selectors, pressure-coordinate selections, input units,
scale and offset. The reader branches only on storage format.

| Vocabulary | Wind | Vertical fields |
| --- | --- | --- |
| `earth2studio` | `u10m/v10m/u850/v850`, m/s earth-relative | `z200/z300/z500/z850`, geopotential m2/s2; `t200/t250/t300/t400/t500`, K |
| `era5-zarr` | `u10/v10` and `u/v(level)`, earth-relative | `z(level)` in m2/s2 and `t(level)` in K |
| `arwen-grib2` | UGRD/VGRD, export with `--winds earth` | HGT in gpm multiplied by 9.80665; TMP in K |

All use mean sea-level pressure in Pa. Grid-relative GRIB winds are rejected.
WuDuan needs surface pressure/wind and 850 hPa wind. Vitart additionally needs its
warm-core temperature and height levels. TempestExtremes uses the pressure, surface
wind and 300/500 hPa height contour tests declared in the same authority.

The native reader interpolates onto the covered 0.25-degree lattice. GPU trackers
require global 721x1440 arrays. Complete global inputs use the upstream detector on its native lattice.
For a crop the driver embeds only real cells and
carries an explicit mask. Filled cells have zero vorticity and wind and infinite
pressure during detection. A one-cell derivative halo is excluded. Returned
centres outside real support are rejected. A fix within 300 km of a coverage edge
gets `edge`. Receipts count real, filled and rejected cells. This route cannot
recover a storm outside the crop.

TempestExtremes uses the actual crop with `--regional`, never a global fill. It
flags fixes within the configured maximum contour/search footprint of an edge.
Its executables are optional and external. The lane build used upstream commit
`901cea1e4d6d061d505298ee67d6427a68b583cf`, BSD-2-Clause (CMake support files use
Boost Software License 1.0). DetectNodes runs one process per member, with many
members assigned to the bounded worker pool.

## Output contract

The contract shared with `cyclone-setup --storm auto` is exactly
`gpuwm.tc-tracks.v1`:

```json
{
  "schema": "gpuwm.tc-tracks.v1",
  "tracker": {"backend":"wuduan","config":{},"config_sha256":"...","code_revision":"..."},
  "members": [{"member_id":"ai-000","family":"ai","source":"earth2-fcn3",
    "init":"2024-09-24T12:00:00Z","weight":1.0,"ancestry":{},
    "inputs":[{"path":"forecast.nc","sha256":"..."}]}],
  "tracks": [{"track_id":"ai-000-000","member_id":"ai-000","weight":1.0,
    "fixes":[{"valid":"2024-09-24T12:00:00Z","lead_hours":0,
      "lat":20.0,"lon":-85.0,"mslp_pa":99000.0,"max_wind10_ms":30.0,"flags":[]}]}]
}
```

Optional track identity fields `atcf_id`, `storm_id`, `name`, `storm_name` and
`aliases` let `cyclone-setup` select a labelled system explicitly. Trackers do
not invent observed identities for unlabelled detections.

Longitude is in -180..180. Every sampled valid time remains present. Missing fixes
keep valid/lead time and have null latitude, longitude, pressure and wind, with
`flags:["missing"]`. Finite fixes can have `flags:["edge"]`. Tracks are never
silently shortened to successful detections. CSV has exactly these columns:

```text
member_id,track_id,weight,valid,lead_hours,lat,lon,mslp_pa,max_wind10_ms,flags
```

CSV null quantities are empty cells and flags are JSON arrays. JSON also carries
optional `best_track` and `scores`; provenance metadata has no CSV equivalent.

## Verification and pictures

`--best-track` accepts NHC ATCF BEST b-decks or HURDAT2. Format columns and
coordinate scales are declared in the authority. ATCF special-fix minutes and
HURDAT2 landfall markers are retained. Use `--storm-id AL092024` for a full
multi-storm HURDAT2 file. Public sources are the [NHC archive](https://ftp.nhc.noaa.gov/atcf/archive/2024/)
and [HURDAT2](https://www.nhc.noaa.gov/data/#hurdat).

Each member is scored at common valid times. The nearest candidate is explicitly
associated for centre distance in km and central-pressure error in hPa, within
the authority's explicit `verification.association_max_distance_km` (500 km).
This verification choice is recorded in scores; it is not a detector threshold.
An unrelated distant cyclone cannot stand in for a missing target. Misses
remain counted, and all raw candidates remain in the output. This association
can select different track IDs and is recorded in scores. Model instantaneous
maximum 10 m wind and NHC maximum 1-minute sustained wind are different quantities.

With a best track the Rust renderer also finds ocean-to-land coastline crossings
between adjacent finite fixes. The receipt reports distance and time error against
HURDAT2 landfall fixes, along with the input time bracket. Linear interpolation
cannot establish timing finer than the sampling cadence. A missing crossing is
null. ATCF files without landfall markers cannot supply that comparison.

`--render` draws coloured member tracks, weights in the legend and NHC in black
through `rw_wrfbatch --tc-track-plume tracks.json OUT.png`. Missing fixes break the
lines. Python dispatch uses `gpuwm.rustwx.renderer_env()` for basemap assets. No
matplotlib or cartopy map is involved. The same option writes an intensity chart using
matplotlib, with the two wind definitions labelled separately. Matplotlib is
needed only for that optional intensity product.

Native float staging and temporary NetCDF/Zarr are deleted on success or failure;
`deleted-files.json` records their paths and sizes. Small requests, logs, output
tracks and receipts remain for review.
