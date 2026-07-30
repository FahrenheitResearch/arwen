# RW-WPS native stock-WRF initialization distributions 0.1.1

The Linux and Windows x86-64 bundles install the dedicated `rw-wps` Python distribution with
HRRR, ERA5, GFS, 20CRv3, and declarative mapped adapters, and supplies five
prebuilt Rust GRIB bridges. The wheel retains the internal `gpuwm.*` module
namespace for source compatibility but omits the forecast driver, supervisor,
dycore/physics executors, verification suite, and forecast-only data tables.
The adapters emit `wrfinput_d01..dNN` and root-only `wrfbdy_d01` directly.
Neither WPS nor `real.exe` is run.
An unchanged stock WRF 4.6.1 executable is an optional downstream acceptance
consumer, not part of the bundle.

The bundle is fail-closed and case-neutral.  It contains code, launchers,
example configurations, and decoder executables.  Meteorological GRIBs,
geography/static caches, input manifests, WPS/WRF namelists, and `wrf.exe`
remain external case inputs and are SHA-bound by each adapter.

## Supported runtime

- Linux x86-64 with Bash: CPU or CUDA setup/export, including the HRRR shell
  pipeline.
- Windows x86-64 with PowerShell 5.1 or newer: CPU-only public CLI and pure
  Python/Rust routes. The HRRR shell pipeline is not supported on Windows.
- Python 3.11 or newer.
- NumPy 1.26 or newer and netCDF4 1.6 or newer.
- On Linux, an NVIDIA GPU with a CUDA 12.x runtime, compatible driver, and
  `cupy-cuda12x` 13.0 or newer for the CUDA backend. ERA5, GFS, and mapped
  native setup/export can instead use the bundled Rust/NumPy CPU backend
  without importing CuPy. Descriptor/mapping/manifest authoring and dry-run
  planning also require no GPU.
- On Linux, standard POSIX programs used by the HRRR shell lane: `find`, `sha256sum`,
  `sort`, `date`, and `bash`.

The installer performs no network access and does not install dependencies.
Point `GPUWM_PYTHON` at a Python environment that already satisfies the list.
The exact resolved interpreter is recorded and must also be available when the
launcher runs.

## Install and verify

Extract the Linux versioned archive into a new directory, then run:

```bash
GPUWM_PYTHON=/absolute/path/to/python ./install.sh
./bin/gpuwm-wrf-init --list-sources
./bin/rw-wps --show-source mapped
```

On a CPU-only host, use
`GPUWM_PYTHON=/absolute/path/to/python ./install.sh --skip-gpu`. This skips
only the CuPy/device probe. Wheel RECORDs, native decoder identities and ABIs,
CPU preprocessing library, helper inventory, and all artifact hashes are
still verified. Verification also executes one FP32 bilinear transform with
one and three Rust workers, requires byte-identical output, and records the
output hash in `native-wrf-runtime-receipt.json`. The resulting launcher
permits actual ERA5, GFS, and mapped WRF-file generation only when the command
selects `--preprocess-backend cpu` or `auto`; an explicit CUDA run still
performs the full GPU check.

`install.sh` first verifies every artifact file against `SHA256SUMS`, installs
the one `rw-wps` wheel into a temporary directory with
`pip --no-index --no-deps`, and
runs the wheel/bridge/dependency/CUDA self-check.  Generic GRIB2 bridges must
also expose the exact tabular ABI consumed by the installed mapped decoder;
matching only the executable name or usage text is insufficient.  Only a
successful temporary install is renamed to `runtime`.  The launcher repeats
the artifact and runtime checks on every invocation; it does not trust an
editable prior PASS receipt.

For the Windows CPU archive, point `GPUWM_PYTHON` at a prepared venv and use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -SkipGpu
.\bin\rw-wps.cmd --list-sources
.\bin\rw-wps.cmd --show-source mapped
```

The PowerShell installer has the same create-only target, hash, wheel RECORD,
PE bridge identity, DLL ABI, and native arithmetic gates. ZIP extraction and
checksum paths reject traversal, non-regular/reparse entries, and
case-insensitive aliases. Windows is CPU-only; select `cpu` or `auto` for
actual preprocessing and do not claim the named HRRR Bash driver there.

Set `GPUWM_INSTALL_ROOT` for a non-default installation directory.  If
`GPUWM_PYTHON` is supplied at launch it must resolve to the interpreter used at
installation.  Existing targets and outputs are never overwritten.

Release maintainers can assemble every input artifact from one clean checkout
with `tools/build_rw_wps_release.py`, then run a new-environment, no-CUDA
archive gate with `tools/smoke_rw_wps_cpu_install.py`. See `docs/install.md`
for the exact commands and the explicit remaining package-boundary caveat.

## Public scientific controls

The source-specific routes currently produce one specified Lambert d01; the
mapped route also supports a parent-first one-way hierarchy up to its sealed
mapping's declared `max_dom`. They use WSM6, classic/old-MM5 surface layer
(option 91), Noah, YSU, and no cumulus. The
shipped examples use the accepted 49-mass-level hybrid coordinate with a
100-hPa model top, but the initializer and exporter are not hard-coded to that
grid.  A case may supply any finite, strictly decreasing explicit eta grid
from exactly 1.0 to 0.0 when `nz`, WRF `e_vert`, the static receipt, and every
serialized coordinate shape agree.  `hybrid_opt=2` remains required, and the
requested model top may not extend above the source product's atmosphere.
Unsupported physics or vertical substitutions are not performed silently.

### HRRR

Input is the ordered hourly f00 through f12 pair inventory
`wrfnatfHH.grib2` plus `soilfHH.grib2` from one exact HRRR cycle.  The public
domain JSON can change the validated CONUS Lambert center, positive equal
`dx`/`dy`, `nx`, `ny`, and positive integer timestep in seconds.  The boundary width remains
five points (`spec_zone=1`, `relax_zone=4`), and the target JSON `nz` must match
the supplied WRF namelist's explicit `e_vert - 1` grid.  Run duration is 1
through 43,200 seconds, pipeline workers
are 1 through 64, and preparation workers are 1 through 32.  The supplied WRF
namelist is the physics and explicit-vertical proof contract; the target-domain
JSON is the authoritative horizontal output geometry.

Use either a hash-bound prebuilt native static cache and receipt or
`--geog-root` plus the domain JSON to build the domain-specific cache natively.
The requested `--valid-time` must be an exact hourly cycle and must match the
hour encoded in all 26 source filenames.

### ERA5

Input is a uniform combined GRIB1 time series beginning exactly at the
experiment start and covering the requested run, plus an ERA5 Vtable and
source-orography field.  One Lambert WPS namelist and one experiment TOML must
match exactly for d01 in projection, grid dimensions, spacing, and reference
point; additional unused WPS domains are permitted.
The static NPZ requires a separate geometry-and-SHA receipt, preventing a
same-shape cache from another domain.  Cadence must be uniform in whole hours.

### GFS

Input is an ordered `pgrb2.0p25` f000 series from an exact 00, 06, 12, or 18
UTC cycle.  Cadence must be uniformly one or three hours and the series may not
exceed f384.  Every time requires the complete certified 1000--100-hPa
atmosphere and four exact Noah soil layers.  The WPS namelist, experiment TOML,
static NPZ, and static receipt follow the same exact geometry and frozen
vertical contract as ERA5.

### 20CRv3 exact member

Input is one paired pressure/surface every-member GRIB2 series whose
`memNNN_YYYYMMDDHH_{pl,sfc}.grb2` filenames and create-only SHA-256 manifest
own member identity. The named route loads byte-verified mapping, composition,
and provenance authorities from the installed wheel; callers cannot replace
them with command-line paths. The shared Lambert hierarchy remains bounded by
the packaged mapping's `max_dom=4` and is runnable but not stock-WRF certified.

### Declarative mapped GRIB/NetCDF

Input is a strict `rw-wps.mapping.v1` document, a matching
`gpuwm-mapped-composition-v2` join, ordered primary files, repeatable
`ROLE=PATH` supplement bindings, provenance bindings, and one exact input
manifest covering all of those bytes plus the selected decoder executables.
GRIB1 uses the bundled generic GRIB1 bridge; GRIB2 uses the bundled generic
inventory and selective-dump executables; NetCDF requires no external decoder.
For an installed bundle, the input manifest's decoder entries must resolve to
those exact files under `libexec/bridges` and bind their installed byte counts
and SHA-256 values.  A manifest retained from a source checkout or another
distribution is rejected even when its decoder happens to be functionally
equivalent. `--author-input-manifest PATH --author-only` now discovers those
installed decoder paths, proves their executable/tabular ABI, and writes the
exact create-only manifest. `--descriptor`, `--vtable`, and
`--author-mapping` can first compile a source-family-independent explicit
mapping. See `docs/native-mapped-source-authoring.md`.
The WPS namelist and experiment TOML are the joint Lambert hierarchy,
explicit-eta, and physics authority. One-way children share the root eta grid;
vertical refinement, moving nests, and two-way feedback fail closed.

The retained real gates cover genuine ERA5 GRIB1, GFS GRIB2, and ERA5 NetCDF
single domains plus a genuine GFS GRIB2 d01-through-d04 hierarchy. They do not certify
every future mapping or source product: each new mapping remains responsible
for exact field selectors, units, time semantics, declarative soil geometry,
remapping and source-land/ocean missing-data policy, donor coverage, and
provenance.

## Launcher examples

The bundle automatically selects its sealed source-specific Rust GRIB bridge.
HRRR, ERA5, GFS, and 20CRv3 additionally accept
`--preprocess-backend cuda|cpu|auto` for the
horizontal and WRF-real transforms.  The CPU route uses the bundle's sealed
Rust library; select its deterministic thread count with
`--preprocess-workers N`. For HRRR this is a total native transform-thread
budget across all simultaneously active preprocessing jobs. The separate
`--prepare-workers` setting controls f01+ job concurrency, while
`--pipeline-workers` controls decoder/hour concurrency. Preparation receipts
record the requested native budget, each deterministic per-job allocation,
the peak active native allocation, and the separate decoder selection. Use
`--dry-run` first to validate the public route and print the exact internal
argv without reading case data.

`--author-only`, `--dry-run`, compatibility reports, and explicit CPU/auto
runs skip the launcher's CUDA-device check. With `--preprocess-backend cpu`,
interpolation, WRF-real transforms, native setup state, cache writing, and
stock-WRF export remain on the Rust/NumPy path. `auto` selects CUDA only when
a device and the certified CUDA 12.x runtime family are available; otherwise
it falls back to CPU. This implementation path is not a claim that every
source/domain combination has a clean-machine CPU stock-WRF gate.

```bash
./bin/gpuwm-wrf-init --source hrrr \
  --source-root /case/hrrr --source-sha256s /case/SHA256SUMS \
  --source-sha256s-sha256 HEX --static-cache /case/static.npz \
  --static-receipt /case/static-receipt.json \
  --domain-spec share/configs/hrrr_target_oklahoma_192x160_3km.json \
  --namelist-input /case/namelist.input \
  --valid-time 2026-07-20_00:00:00 --run-seconds 43200 \
  --pipeline-workers 8 --prepare-workers 2 \
  --preprocess-backend cpu --preprocess-workers 8 \
  --output-root /case/out --dry-run

# Reuse that sealed root for a standard-namelist d01..dNN hierarchy.
# max_dom is read from both namelists and must be in 1..21.
./bin/gpuwm-wrf-init --source hrrr \
  --root-preparation /case/root-preparation \
  --domain-spec /case/hrrr-root-domain.json \
  --wps-namelist /case/namelist.wps \
  --namelist-input /case/namelist.native.input \
  --stock-wrf-namelist-input /case/namelist.stock.input \
  --geog-root /data/WPS_GEOG \
  --source-sha256s /case/SHA256SUMS \
  --source-sha256s-sha256 HEX \
  --valid-time 2026-07-20_00:00:00 \
  --child-workers 8 --output-root /case/hierarchy --dry-run

./bin/gpuwm-wrf-init --source era5 --grib /case/era5.grb \
  --vtable /case/Vtable.ERA5 --wps-namelist /case/namelist.wps \
  --static-input /case/static.npz \
  --static-receipt /case/static-receipt.json \
  --source-orography /case/orography.nc --source-orography-variable SOILHGT \
  --experiment-config share/configs/era5_wrf_direct_proof.toml \
  --source-sha256s /case/input-manifest.json \
  --source-sha256s-sha256 HEX --output-root /case/out --dry-run

./bin/gpuwm-wrf-init --source gfs --gfs-series /case/gfs-series.tsv \
  --cycle 2026-07-20_00:00:00 --wps-namelist /case/namelist.wps \
  --static-input /case/static.npz \
  --static-receipt /case/static-receipt.json \
  --experiment-config share/configs/gfs_wrf_direct_proof.toml \
  --source-sha256s /case/input-manifest.json \
  --source-sha256s-sha256 HEX --preprocess-backend cpu \
  --preprocess-workers 8 --output-root /case/out --dry-run

./bin/rw-wps --source 20crv3 \
  --source-manifest /case/member072.manifest.json \
  --source-manifest-sha256 HEX \
  --wps-namelist /case/namelist.wps --geog-root /data/WPS_GEOG \
  --experiment-config /case/experiment.toml \
  --preprocess-backend cpu --preprocess-workers 8 \
  --hierarchy-workers 4 --output-root /case/20crv3-out --dry-run

./bin/rw-wps --source mapped --source-format grib2 \
  --mapping share/configs/rw-wps-gfs-pressure-grib2.mapping.json \
  --composition share/configs/rw-wps-gfs-terrain.composition.json \
  --input /case/gfs-f000.grib2 --input /case/gfs-f003.grib2 \
  --supplement gfs_valid_time_terrain=/case/gfs-f000.grib2 \
  --supplement gfs_valid_time_terrain=/case/gfs-f003.grib2 \
  --provenance gfs_valid_time_terrain_provenance=/case/terrain.md \
  --wps-namelist share/configs/gfs_wrf_hierarchy_proof.namelist.wps \
  --geog-root /data/WPS_GEOG \
  --experiment-config share/configs/gfs_wrf_hierarchy_proof.toml \
  --source-sha256s /case/input-manifest.json \
  --source-sha256s-sha256 HEX --hierarchy-workers 1 \
  --output-root /case/hierarchy --dry-run
```

Remove `--dry-run` only after the printed contract is correct.  Each real route
creates a new output tree with machine-readable input, implementation,
geometry, timing, and export receipts.  Preserve those receipts beside the
resulting WRF files.
