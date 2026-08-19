# RW-WPS examples

These examples use placeholders deliberately. Source payloads, WPS geography,
namelists, manifests, and stock WRF are not redistributed by this repository.
Run the compatibility report before replacing `--dry-run` with a real run.

## Inspect a case without a GPU

```bash
rw-wps --show-source gfs
rw-wps --namelist-support-report \
  --wps-namelist /case/namelist.wps \
  --namelist-input /case/namelist.input \
  --source-top-pressure-pa 10000 \
  > /case/namelist-support.json
```

## GFS on the Rust CPU backend

First fetch the series and author the exact input manifest the front
door verifies (never hand-author it; the manifest binds every file's
sha256 including the bridge executable's own hash):

```bash
gpuwm fetch --source gfs --cycle latest --hours 24 \
  --area 12,-130,58,-65 --out /case/gfs-data

gpuwm fetch --source gfs --author-front-door-manifest \
  --out /case/gfs-data \
  --wps-namelist /case/namelist.wps \
  --experiment-config /case/experiment.toml
# prints: /case/gfs-data/gfs-input-manifest.json + MANIFEST_SHA256
# and the complete rw-wps command below with both filled in, --bridge
# included -- omitting --bridge above resolves the built
# gfs_grib2_bridge this install has and binds THAT one.
```

Then run the front door.  Paste the command the step above printed; it
already names the bridge the manifest bound.  Written out, `--bridge`
is the built `gfs_grib2_bridge` executable (a wheel install gets it
from `gpuwm setup` into `~/.gpuwm/bridges`; a clone can `cargo build
--release --locked --offline` in `tools/grib1_bridge`; `gpuwm doctor`
lists where bridges are found, or set `GPUWM_GFS_GRIB2_BRIDGE` instead
of passing the flag):

```bash
rw-wps --source gfs \
  --gfs-series /case/gfs-data/gfs-series.tsv \
  --cycle 2026-07-20_00:00:00 \
  --bridge /clone/tools/grib1_bridge/target/release/gfs_grib2_bridge \
  --wps-namelist /case/namelist.wps \
  --geog-root /data/WPS_GEOG \
  --experiment-config /case/experiment.toml \
  --source-manifest /case/gfs-data/gfs-input-manifest.json \
  --source-manifest-sha256 MANIFEST_SHA256 \
  --preprocess-backend cpu --preprocess-workers 16 \
  --hierarchy-workers 8 \
  --output-root /case/output --dry-run
```

The TSV is an ordered `HOUR<TAB>PATH` inventory beginning at f000. The exact
manifest must cover every file and the selected decoder. The named GFS route
requires the complete supported pressure and Noah soil inventory at every
time.

## Named 20CRv3 member preparation

First author the create-only filename-member manifest and record the emitted
SHA-256. The run then resolves its exact mapping, composition, and provenance
from immutable wheel payloads:

```bash
rw-wps --source 20crv3 \
  --source-root /data/20crv3/member072 \
  --author-input-manifest /case/20crv3-member072.manifest.json \
  --author-only

rw-wps --source 20crv3 \
  --source-manifest /case/20crv3-member072.manifest.json \
  --source-manifest-sha256 MANIFEST_SHA256 \
  --wps-namelist /case/namelist.wps \
  --geog-root /data/WPS_GEOG \
  --experiment-config /case/experiment.toml \
  --preprocess-backend cpu --preprocess-workers 16 \
  --hierarchy-workers 8 \
  --output-root /case/output-20crv3-member072
```

A bare run composes in `gpuwm_mapped_engine` and reads the archive's
raw record inventory in process, so it names no subprocess decoder.
`--grib2-inventory`/`--grib2-dump` (or `--mapped-engine python`) select
the documented Python-engine workaround, which decodes through those
two executables instead.

The packaged mapping declares `max_dom=4`. Successful preparation is not a
stock-WRF certificate; preserve the member manifest, packaged-authority and
decoder identities, proof JSON, and subsequent unchanged-WRF evidence.

## Author a new GRIB2 mapping

```bash
rw-wps --source mapped --source-format grib2 \
  --descriptor /case/product.descriptor.json \
  --vtable /case/Vtable.PRODUCT \
  --author-mapping /case/product.mapping.json \
  --composition /case/product.composition.json \
  --input /data/product-f000.grib2 \
  --input /data/product-f003.grib2 \
  --supplement terrain=/data/terrain-f000.grib2 \
  --supplement terrain=/data/terrain-f003.grib2 \
  --provenance terrain_provenance=/case/terrain-source.md \
  --author-input-manifest /case/input-manifest.json \
  --author-only
```

The command refuses existing outputs. The resulting status is validated, not
stock-WRF certified. Preserve the mapping authoring receipt beside the
descriptor, Vtable, composition, and input manifest.

## Evidence to retain

For any acceptance run keep the exact command argv, source manifest and
digest, both namelists, resolved experiment, static catalog/receipt, decoder
and CPU-library identities, proof JSON, export manifest, WRF input hashes,
stock-WRF executable hash, log, exit code, and history-file finite-state
audit. A screenshot or `SUCCESS COMPLETE WRF` line alone is not sufficient.
