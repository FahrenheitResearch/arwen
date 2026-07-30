# Native NOAA 20CRv3 ensemble source-adapter specification

Status: the exact private-sample GRIB2 canonical-decode/native-preparation
route is implemented, while native GPU/stock-WRF execution certification
remains pending. A complementary metadata-driven NetCDF-CF ensemble streamer
is implemented and synthetic-gated, but is not yet a public WRF runner.

## Implemented private-sample slice (2026-07-21)

The implemented named adapter accepts paired files named
`memNNN_YYYYMMDDHH_{pl,sfc}.grb2`. The first real gate used member 072: 26
files, 13 pressure/surface pairs, and a uniform three-hour cadence from
1932-03-21 00 UTC through 1932-03-22 12 UTC. The complete 32,335,416-byte
inventory is pathname/size/SHA-256 bound before decode.

The supplied GRIB2 PDT does not contain ensemble membership. The adapter
therefore treats the `memNNN` filename plus the sealed manifest as the member
authority, rejects mixed member labels, and carries `072` into every canonical
frame. It also binds the observed NOAA/NCEP table and process identity
(center/subcenter/master/local 7/2/2/1, generating process 4/195), PDT 0,
0- or 3-hour lead semantics, 0.5-degree GDT 0 geometry, and the exact 115-field
pressure plus 19-field surface inventories.

All 13 real times pass canonical materialization on a 151 x 101 x 23 source
grid. Direct/derived fields include Z, T, Q, U, V, pressure, surface pressure,
terrain, skin and 2-m state, 10-m wind, land and sea-ice fraction, and the exact
four Noah soil layers (0-.1, .1-.4, .4-1, and 1-2 m). Hydrometeors and vertical
velocity retain explicit cold-start-zero policies. Snow water equivalent is
absent; supplied snow depth is deliberately not injected until its bitmap
semantics are separately bound.

`python -m gpuwm.twentycrv3_wrf` now passes these member-bound frames directly
to the shared native hierarchy initializer and WRF exporter. It does not
translate the custom manifest into a generic manifest or erase the member
identity. The exact real series decoded in 4.157 seconds on the development
Windows host, including the redundant fail-closed metadata inventory pass.

This establishes real-data decode and preprocessing readiness, not forecast
certification. A bounded native 25 km -> 12.5 km, 49-level GPU smoke and an
unchanged stock-WRF startup remain required. The supplied three-hour sample
does not certify a separate private hourly/80-member archive layout.

## Complementary NetCDF-CF ensemble primitives

`gpuwm.twentycrv3` preserves the separate metadata-driven implementation for
NetCDF-CF ensemble products. It discovers coordinates and member count from
the files, validates hourly or three-hourly coverage without manufacturing
missing times, normalizes dimension order and units into canonical source
frames, plans memory-bounded member batches, and publishes deterministic
hash-addressed generations atomically. Synthetic fixtures cover 2- and
5-member files, permuted dimensions, ascending/descending axes, 1- and 3-hour
cadence, arbitrary target vertical counts including 80, corrupt members,
bounded retention, and worker-count-independent output bytes.

This NetCDF-CF surface is intentionally registered separately as
`20crv3-cf` with aliases such as `20crv3-netcdf`. It remains non-runnable from
the public WRF CLI until a real corpus is bound to a complete field mapping
and the resulting initialization passes the native/stock-WRF gates. It does
not inherit the exact filename-bound GRIB2 route's certification evidence.

## Purpose

Add 20CRv3 as another producer of gpuwm's canonical source-frame contract, so
the same native horizontal/vertical initialization and direct-WRF exporters
can process one member or a large ensemble. The model core must not contain a
20CR-specific branch.

## Discovery instead of assumptions

The adapter discovers and records from file metadata:

- member coordinate and labels;
- valid-time coordinate, calendar, units, and ordering;
- latitude/longitude coordinates, orientation, periodicity, and staggering;
- pressure/vertical coordinate, units, and ordering;
- variable names, dimensions, missing-value encodings, and units; and
- analysis/forecast/derived product identity.

Member count and cadence are not constants. Public 20CRv3 analysis products
are commonly encountered as 80 members at three-hour intervals, while a
private or derived corpus may be hourly. The adapter validates the actual time
coordinate and manifest. It accepts a requested hourly or three-hourly series
only when the files contain that series. It does not manufacture hourly fields
from three-hourly data unless a future, explicitly selected and validated time
interpolation policy is added.

Uniform cadence is inferred from consecutive decoded times and written to the
manifest. Duplicate, reversed, missing, or irregular times fail with the exact
gap listed. Requested boundary coverage must start at the initialization time
and extend through the forecast endpoint.

## Canonical field contract

Each member/time frame must provide or explicitly derive, with units checked:

- three-dimensional pressure-level temperature, water vapor, geopotential or
  height, and earth-relative horizontal wind;
- surface pressure, near-surface temperature/humidity/wind, terrain, land/sea
  mask, sea-surface or skin temperature, snow, and sea ice as available; and
- soil temperature/moisture state required by the selected land model.

Pressure levels must be finite, unique, monotone after normalization, and
cover the requested model top. The adapter refuses unsupported upper-air
extrapolation. Surface or land fields absent from a product are reported as
named gaps. A gap may be filled only by a separately identified static or
climatology policy whose provenance is included in the output identity.

Physics initialization is a downstream contract, not an excuse to invent
source fields. Each selected scheme declares required prognostic fields and a
validated cold-start policy. For example, MYNN TKE, Thompson number/ice state,
or a nine-layer RUC land state cannot silently reuse WSM6/YSU/Noah defaults.

## Ensemble streaming and batching

Members are independent jobs sharing immutable geometry, static geography,
horizontal weights, and target vertical terms. The controller accepts an
ordered member selection and a memory budget, then chooses a bounded batch
size. It never loads all 80 members merely because they exist.

For each member, the pipeline is:

1. decode only the requested fields/times;
2. normalize into the canonical source-frame layout;
3. map and initialize the requested domain tree;
4. write/hash prepared-cache or WRF files atomically; and
5. release member arrays before admitting the next batch.

Completion order may differ under parallel execution, but filenames,
manifests, and final inventory are sorted by canonical member ID. Worker counts
must not change bytes. Failed members remain isolated and cannot produce a
global PASS manifest.

## Manifest and integrity

The run manifest binds:

- every input pathname/object identity, byte count, and SHA-256;
- discovered coordinate metadata and normalized units;
- selected member IDs and valid times;
- detected cadence and any explicit gap/fill policy;
- source-product identity and analysis/forecast semantics;
- target domain-tree, vertical, physics, and static identities;
- decoder and gpuwm commit/source hashes; and
- every output byte count and SHA-256.

Writes use a new staging directory and are published only after all declared
files hash and validate. Existing valid output is never destroyed by a failed
same-name rewrite.

## Test gates

Synthetic fixtures must cover:

- 2 and 5 members with permuted dimension order;
- hourly and three-hourly time axes discovered from metadata;
- missing, duplicate, and irregular times;
- ascending and descending latitude/pressure coordinates;
- several unlike target vertical counts, including but not special-casing 80;
- source-top coverage rejection;
- missing variables, bad units, fill values, and member-specific corruption;
- deterministic 1/4/8-worker output; and
- bounded-memory streaming that proves released members are not retained.

Real certification then requires a collaborator-provided filename plus a
header/metadata sample, one member/time decode comparison, a multi-time member
run, and an unchanged stock-WRF or gpuwm startup gate. Until those pass, the
implemented adapter/preparation route must remain explicitly uncertified; it
must not inherit another mapped source's stock-WRF evidence.
