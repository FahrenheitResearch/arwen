# RW-WPS CLI reference

`rw-wps` and `gpuwm-wrf-init` call the same entry point. Exit status `0`
means the requested validation or run completed. Status `64` is invalid CLI
usage, `70` is an adapter launch failure, and `78` is an unsupported or
invalid scientific configuration.

This page covers the native preprocessor CLI. The `gpuwm domain` wizard is
documented in docs/public/FIRST-LIGHT.md: it works worldwide, auto-selecting
Lambert conformal, Mercator, or polar stereographic from the point latitude
(`--projection` overrides), with presets `12`, `12-3`, `12-3-1`,
`12-3-1-0.5`, or `auto`, and a custom form `--root-dx KM --chain
R1,R2,...` for any other root spacing and integer refinement chain.

## Inventory and validation

| Command | Result |
|---|---|
| `rw-wps --version` | Installed RW-WPS version |
| `rw-wps --list-sources` | Complete source registry and canonical-field contract as JSON |
| `rw-wps --show-source MODEL` | One named source declaration as JSON |
| `rw-wps --show-support-matrix` | Fail-closed release matrix as JSON |
| `rw-wps --namelist-support-report ...` | WPS/WRF geometry, vertical, boundary, and physics-state report |
| `rw-wps ... --dry-run` | Route validation and exact internal argv; no contract authoring or data processing |

`--namelist-support-report` requires `--wps-namelist` and
`--namelist-input`. Add `--source-top-pressure-pa` when the source top is
known; requesting a higher model top then fails before preprocessing.

## `gpuwm adapt`

`gpuwm adapt` is the create-only front door for an arbitrary GRIB2 source
whose capabilities can be verified without a named adapter.

```text
gpuwm adapt --vtable VTABLE --skeleton DESCRIPTOR

gpuwm adapt --vtable VTABLE --descriptor DESCRIPTOR
  --input GRIB2 [--input GRIB2 ...] --output-dir DIR
  [--grib2-inventory EXE --grib2-dump EXE]
```

Skeleton mode writes a review-required `rw-wps.descriptor.v1` scaffold.
Authoring mode compiles its Vtable selectors, inventories the actual files,
checks exact fields/levels, target unit/axis/staggering bindings, complete
declared soil policy, one regular-lat/lon GDT 0 grid, uniform cadence, and
source-top coverage of the descriptor's model top. Any failure names the
missing capability and writes no adapter.

Success writes an SHA-256-bound mapping/composition/provenance authority
triple plus `adapter.inputs.json`. Its exact status is
`runnable_mapping_not_stock_wrf_certified`; unchanged-stock-WRF evidence is a
separate exact-authority gate. See
`arbitrary-verified-adapters.md` for the schema, battery, refusal examples,
output contract, and runnable-versus-certified boundary.

The battery proves the emitted files implement your descriptor and that your
GRIB files satisfy it. Units, absolute geolocation, cell registration, level
sufficiency, intended time semantics, land-mask polarity, and soil depth
labels are trusted from your declaration. `adapt-validation-contract.md`
gives the two-column contract for every input dimension and a self-check you
can run for each trusted row.

## Named source routes

| `--source` | Required source control | Notes |
|---|---|---|
| `hrrr` | f00..f12 native/surface pairs, SHA manifest, valid time | Named certified CONUS slice; hierarchy reuses a sealed root preparation |
| `gfs` | ordered `pgrb2.0p25` series, cycle, SHA manifest | One- or three-hour uniform series; 1000..100 hPa and four Noah slabs |
| `era5` | combined GRIB1, Vtable, orography, SHA manifest | Uniform series beginning at experiment start |
| `20crv3` | exact filename-member manifest; paired pressure/surface GRIB2 | Packaged immutable authorities; Lambert `max_dom=4`; not stock-WRF certified |
| `mapped` | mapping, composition, ordered inputs, role bindings, SHA manifest | Generic GRIB1, GRIB2, or NetCDF; new mappings are validated, not certified |

All executing routes require an output path that does not already exist.
Static fields come from `--geog-root` or a route-specific sealed cache and
receipt. WPS and experiment/WRF namelists remain explicit authorities.

HRRR, ERA5, GFS, 20CRv3, and mapped routes accept:

```text
--preprocess-backend cuda|cpu|auto
--preprocess-workers N
--cpu-preprocess-bridge PATH
```

An explicit CPU bridge is valid only with the CPU backend. More than one
hierarchy worker also requires the CPU backend; current CUDA hierarchy setup
is deterministic with one worker. During HRRR root preparation,
`--preprocess-workers N` is one total native CPU transform-thread budget, not
`N` threads for each active hour. `--prepare-workers` controls the independent
f01+ mapping/initialization job slots; the controller deterministically
partitions the native budget across those slots and records every effective
job allocation plus the peak allocation. `--pipeline-workers` independently
controls 1 through 64 decoder/hour jobs and is never charged to or multiplied
into the native transform budget.

## Native HRRR subset download

The sealed runtime includes `download_hrrr_native_subset.py` for retrieving
only the atmosphere and soil messages required by the HRRR bridge. For a full
12-hour forcing authority, bounded range and product concurrency can overlap:

```bash
python tools/download_hrrr_native_subset.py \
  --cycle 2026-07-18_00:00:00 \
  --forecast-hours 0,1,2,3,4,5,6,7,8,9,10,11,12 \
  --output-root /case/hrrr-f00-f12 \
  --file-workers 8 \
  --workers 4
```

`--workers` is the range-request limit within one GRIB product;
`--file-workers` is the number of independent atmosphere/soil products in
flight. Both preserve deterministic receipt order. The product of the two
limits may not exceed 64, preventing an apparently small setting from
creating an unbounded connection fan-out. Publication remains create-only,
and each assembled subset is checked for its exact byte count and complete
GRIB framing before it enters the final SHA-256 manifest.

## Named 20CRv3 route

`rw-wps --source 20crv3` authors or consumes the custom exact-member manifest
used by the NOAA-CIRES-DOE every-member GRIB2 profile. Mapping, composition,
and provenance are immutable wheel payloads and cannot be replaced through
named-route arguments. Author a manifest with `--source-root`,
`--author-input-manifest`, and `--author-only`; then run with that manifest
and its SHA-256. Member identity remains filename-plus-manifest authoritative
because the accepted archive profile does not encode it in the GRIB2 PDT.
The mapping still declares `max_dom=4`, and this runnable route is not yet a
live unchanged-stock-WRF certificate.

## Declarative authoring

The mapped route can compile a `rw-wps.descriptor.v1` plus a WPS Vtable into
an executable `rw-wps.mapping.v1`, then author the exact input manifest:

```text
--descriptor DESCRIPTOR --vtable VTABLE --author-mapping MAPPING
--author-input-manifest MANIFEST --author-only
```

Outputs are create-only. GRIB selectors, decoder bytes, inputs, supplements,
provenance files, sizes, and SHA-256 values are bound. NetCDF descriptors do
not use a Vtable. See `native-mapped-source-authoring.md` for the complete
schema and soil/remapping contract.

## Output transaction

A successful real run publishes an atomic output directory containing
`wrfinput_d01..dNN`, root-only `wrfbdy_d01`, and machine-readable receipts.
Children do not receive external boundary files. A failed run must not leave
a partially published final directory, and existing outputs are never
silently overwritten.
