# Native source adapters for stock WRF

`rw-wps` (also installed as `gpuwm-wrf-init`) is the public native
preprocessing interface.  It is designed
to replace the WPS + `real.exe` runtime path, not wrap or invoke it:

```text
Rust decoder -> canonical source frame -> GPU initialization ->
prepared cache -> wrfinput_dNN + wrfbdy_d01 -> unchanged stock wrf.exe
```

The decoder boundary is the canonical `rusty-weather` model registry and its
GRIB implementation.  gpuwm owns the source-specific science adapter and WRF
export.  A model being listed or decodable is **not** proof that it contains a
complete three-dimensional atmosphere, surface, and soil state.

## Interface

```bash
rw-wps --list-sources
rw-wps --show-source hrrr
rw-wps --source hrrr \
  --source-root /data/hrrr-f00-f12 \
  --source-sha256s /data/hrrr-f00-f12/SHA256SUMS \
  --source-sha256s-sha256 EXPECTED_MANIFEST_HASH \
  --geog-root /data/WPS_GEOG \
  --domain-spec /case/hrrr-target-domain.json \
  --namelist-input /case/namelist.input \
  --valid-time 2026-07-18_00:00:00 \
  --output-root /output/run \
  --pipeline-workers 8
```

That single command builds the domain-specific native static cache, decodes
and initializes the downloaded HRRR series, and atomically publishes
`wrf-native-input/wrfinput_d01` and `wrf-native-input/wrfbdy_d01`.  It invokes
neither WPS nor `real.exe`.  A previously sealed static cache can instead be
reused with `--static-cache` plus `--static-receipt`; pass the same
`--domain-spec` for non-legacy geometry.
The final PASS line reports separate static-build, native-prepare, direct-
export, and total wall times in milliseconds.

For the certified nested HRRR path, prepare the root once and pass it back to
the same public command with `--root-preparation`, both standard WPS/WRF
namelists, and `--child-workers`. This schedules d02..dNN initialization for
`max_dom=1..21` and publishes the complete hierarchy without invoking WPS or
`real.exe`; see
`docs/native_wrf_hierarchy_export.md` for the exact command and sealed scope.

HRRR, GFS, and ERA5 are certified runnable adapters in v1. The named 20CRv3
route and strict declarative `mapped` engine are runnable, but they are not
globally certified: stock-WRF evidence is keyed to exact retained source,
mapping, and composition authorities, and newly authored contracts remain
validated-not-certified. The separate `20crv3-cf` registry entry exposes
synthetic-gated NetCDF-CF coordinate discovery, canonical normalization, and
memory-bounded ensemble streaming; it is not yet a public WRF runner and does
not inherit the exact GRIB2 route's evidence. HRRR
consumes consecutive downloaded f00 through f12 native/surface GRIB2 files,
uses the parallel native preparation path, and writes a 12-record WRF
boundary file.
The expected names are `hrrr.tHHz.wrfnatfFF.grib2` and
`hrrr.tHHz.soilfFF.grib2`; `HH` is derived from the exact-hour
`--valid-time`, and `FF` spans 00 through 12.  The frozen products were
accepted by unchanged WRF v4.6.1 and advanced successfully.

The certified ERA5 slice consumes one combined, uniformly spaced GRIB1 time
series through the prebuilt all-Rust decoder.  Its geometry namelist, native
static cache, source-orography artifact, experiment configuration, GRIB,
Vtable, and decoder are all covered by one input manifest.  For example:

```bash
gpuwm-wrf-init --source era5 \
  --grib /data/era5-series.grb \
  --vtable /data/Vtable.ERA5_CDO \
  --bridge /opt/gpuwm/grib1_bridge \
  --wps-namelist /case/namelist.wps \
  --static-input /case/native-static.npz \
  --source-orography /case/source-orography.nc \
  --experiment-config /case/experiment.toml \
  --source-sha256s /case/input-manifest.json \
  --source-sha256s-sha256 EXPECTED_MANIFEST_HASH \
  --output-root /output/run
```

The accepted 1974 proof mapped three 6-hourly forcing states for a
250x200x49, 12-km Lambert domain and produced 12 hours of boundary forcing in
17.346 seconds.  Unchanged WRF v4.6.1 accepted both files, completed a
5-second model step, and produced a finite history file.  The runtime invoked
neither WPS nor `real.exe`.

The certified GFS slice consumes a manifest-bound, uniformly spaced
`pgrb2.0p25` series.  The dedicated Rust GRIB2 bridge selects fields only by
discipline/category/parameter and fixed-surface coordinates, requires all 21
pressure levels from 1000 through 100 hPa, and validates both fixed surfaces
of the exact 0-10, 10-40, 40-100, and 100-200 cm Noah soil slabs.  It never
routes WRF initialization through the visualization-oriented quantized
`rusty-weather` store.  For example:

```bash
gpuwm-wrf-init --source gfs \
  --gfs-series /data/gfs-series.tsv \
  --cycle 2026-07-20_00:00:00 \
  --bridge /opt/gpuwm/gfs_grib2_bridge \
  --wps-namelist /case/namelist.wps \
  --static-input /case/native-static.npz \
  --static-receipt /case/native-static-receipt.json \
  --experiment-config /case/experiment.toml \
  --source-sha256s /case/input-manifest.json \
  --source-sha256s-sha256 EXPECTED_MANIFEST_HASH \
  --output-root /output/run
```

The real 2026-07-20 00Z proof used f000/f003 GFS subsets on a 250x200x49,
12-km Lambert domain.  The public command completed native decode,
initialization, cache publication, and direct export in 21.559 seconds
internally and 22.57 seconds process wall time.  Its `wrfinput_d01` and
`wrfbdy_d01` SHA-256 values are respectively
`064b216010c0ed43ad06ac197b82cd3dbfe884890251e0ebac30a4405fe387f3`
and `c74c5b9636f0e5a5cb54dbd9366d86a221ecbeb9a203fbbd6a3c0d39adc923f8`.
Unchanged WRF v4.6.1 accepted both, advanced five seconds, printed
`wrf: SUCCESS COMPLETE WRF`, and emitted a finite history file.  The decoder
contract for this proof is DRT 5.0 simple packing, GFS process identifier 81
for f000, and identifier 96 for forecast records.

Masked surface and soil interpolation on this native path is intentionally
land-aware, with nearest valid donors and a globally proven nearest-water
donor for every target lake cell.  It is not numerically identical to WPS
METGRID's four- and sixteen-point masked interpolation.  The stock-WRF result
proves structural compatibility and stable execution for the certified slice;
it does not by itself prove WPS numerical parity or forecast skill.

The named `20crv3` route consumes one paired every-member pressure/surface
GRIB2 series. A create-only manifest binds member identity to exact
`memNNN_YYYYMMDDHH_{pl,sfc}.grb2` filenames, and the installed wheel supplies
byte-verified mapping, composition, and provenance authorities that cannot be
replaced at the command line. It uses the shared native hierarchy preparation
path with the packaged mapping's honest `max_dom=4` ceiling. This route is
runnable and distribution-complete, but remains uncertified until its retained
input/domain envelope passes unchanged stock WRF.

The certified root geometry slice is one specified Lambert domain, 49 mass levels,
five boundary cells (`spec_zone=1`, `relax_zone=4`), equal positive `dx`/`dy`,
and complete coverage by the CONUS HRRR source grid including interpolation
halos.  Dynamic horizontal geometry is stock-WRF verified for Oklahoma and
Ohio 192x160 at 3 km and Oklahoma 1000x1000 at 1 km.  Other projections,
vertical counts, invalid or moving nests, Alaska
HRRR, gapped/non-hourly forcing, and other physics suites fail closed.

The `mapped` entry is the declarative format-level route. Its strict
GRIB1/GRIB2/NetCDF consumer, mixed-product composition, canonical-frame
packing, native initialization, and atomic hierarchy export are wired through
the same public command. A genuine GFS GRIB2 d01-through-d04 hierarchy and genuine
ERA5 GRIB1, GFS GRIB2, and ERA5 NetCDF single domains passed unchanged stock
WRF v4.6.1. For example, the GFS hierarchy route is:

```bash
rw-wps --source mapped --source-format grib2 \
  --mapping share/configs/rw-wps-gfs-pressure-grib2.mapping.json \
  --composition share/configs/rw-wps-gfs-terrain.composition.json \
  --input /data/gfs-f000.grib2 --input /data/gfs-f003.grib2 \
  --supplement gfs_valid_time_terrain=/data/gfs-f000.grib2 \
  --supplement gfs_valid_time_terrain=/data/gfs-f003.grib2 \
  --provenance gfs_valid_time_terrain_provenance=/case/terrain-provenance.md \
  --wps-namelist share/configs/gfs_wrf_hierarchy_proof.namelist.wps \
  --geog-root /data/WPS_GEOG \
  --experiment-config share/configs/gfs_wrf_hierarchy_proof.toml \
  --source-sha256s /case/input-manifest.json \
  --source-sha256s-sha256 EXPECTED_MANIFEST_HASH \
  --hierarchy-workers 1 --output-root /output/run --dry-run
```

The installed bundle supplies the generic GRIB2 inventory/dump decoders. The
input manifest must bind those exact executables along with every primary,
supplement, provenance, mapping, and composition byte. Remove `--dry-run` only
after inspecting the exact internal argv. This runner does not treat arbitrary
undeclared data as a complete WRF state: unsupported selectors, soil packing,
cadence, domain topology, vertical coordinate, or source coverage fail closed.
New GRIB1/GRIB2/NetCDF products can use the source-family-independent explicit
descriptor and create-only manifest authoring path instead of a new hardcoded
adapter. The resulting status is `VALIDATED_NOT_STOCK_WRF_CERTIFIED`; only the
exact retained mapping/domain envelopes inherit their stock-WRF evidence. See
`docs/native-mapped-source-authoring.md` and
`docs/native-mapped-source-status.md` for the contracts and current limits.

All other sources fail before reading or writing data.  Each becomes runnable
only after it has explicit field, level, time/cadence, accumulation, wind
rotation, soil, hydrometeor, and missing-state policies. It becomes certified
only after the exact source/domain envelope passes the same stock-WRF
acceptance gate.

## Inventory and readiness

The inventory binds the exact 23-model `rusty-weather` enum plus ERA5 and the
format-level declarative `mapped` route. HRRR,
GFS, and RRFS-A have full upstream `rusty-weather` ingest today; that status is
tracked separately from direct-WRF readiness.  GFS direct initialization uses
the raw hardened GRIB2 decoder because the `.rws` store omits required WRF
surface/soil state and quantizes volumes.  ERA5 uses gpuwm's native Rust
GRIB1 decode, GPU interpolation, prepared-cache path, and direct-WRF exporter;
its certified geometry/physics slice is recorded explicitly in the registry.

Surface analyses and postprocessed products such as RTMA, URMA, and NBM must
be explicitly composed with a complete atmosphere source.  HREF/REFS/SREF
statistics cannot substitute for a dynamically balanced ensemble member.
Ensemble families require selection of an actual member.  The tool never
silently invents missing vertical state.

## Canonical source frame

`gpuwm.source_frame` defines the adapter boundary.  It records:

- projection, earth shape, scan order, and earth/grid-relative wind semantics;
- pressure, hybrid/model, height, and soil-depth axes;
- canonical field name, units, staggering, shape, dtype, and data reference;
- reference/valid time and interval/accumulation reset semantics; and
- explicit initialization policy for every optional-but-trajectory-relevant
  field absent from a source.

The schema is intentionally array-location agnostic: references may identify
GPU-resident buffers or hash-bound cache arrays.  Finite/range and digest
checks are performed when an adapter materializes those references.
