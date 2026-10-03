# Simulated radar resource contract

`gpuwm simulated-radar --describe` probes the installed executable. A worker
requires `supports.resource_estimate` and `supports.named_input_refusals`
before relying on this contract. An older executable can have the volume ABI
without these additions; its capability report identifies that difference.

`gpuwm simulated-radar --estimate [FULL_SCENE.nc ...] --config radar.toml`
calls `rw_simradar --estimate REQUEST.json` and returns
`simulated-radar.resources/v1`. It reads configuration and optional NetCDF
headers, without sampling weather arrays, acquiring an output lock, or writing
radar files. Empty `history_paths` produces a geometry-only estimate. Native
column transports first pass through the common canonical adapter if their
atmospheric shapes are to enter this estimate.

## Admission and named failures

- Each upper limit names what it prevents. `range_km` is at most 460: that is
  the WSR-88D's longest unambiguous range (its long-PRT surveillance cut), and
  a simulated S-band volume reaching farther would hold echo where that radar's
  own data range-folds. `azimuth_step_deg` is at most 720: the ray count is
  `360 / azimuth_step_deg` rounded, zero above 720. `volume_duration_s` is at
  most 3600: a custom ladder turns at `360 x elevations / volume_duration_s`
  deg/s and the simulator holds that rate at no less than 0.1 deg/s.
  `gate_spacing_m` has no universal ceiling; with `level2` it is at most
  32767 m, the largest gate spacing Message 31 stores.
- Gate spacing must be a positive whole metre because the shared writer
  geometry stores integer metres. Tiny fractional positive values fail before
  conversion to integer counts.
- The conservative radial count `ceil(360 / azimuth_step_deg)` must fit 65535,
  and `ceil(1000 * range_km / gate_spacing_m)` must fit 16384 gates. These are
  writer geometry limits, not service pricing limits. Nonfinite, overflowing
  and out-of-range dimensions fail before allocation.
- Level II permits at most 32 physical scan cuts. Other formats permit the
  configured ladder up to 255 elevations. A named VCP can repeat an elevation;
  its physical cuts, including repeats, are the work unit.
- The native process estimates polar working rows, retained moments, writer
  copies and one PPI working set. It checks that estimate against current
  available host memory, including the process's Linux memory cgroup and its
  ancestors. On Windows it reads `GlobalMemoryStatusEx`. If the estimate does
  not fit, the error names estimated bytes, available bytes and the scan
  parameters that reduce the allocation. If memory availability cannot be
  determined, the allocation is refused explicitly.
- This check runs before history hashing, before atmospheric decoding, and
  before each site. The atmosphere estimate adds `nx * ny * nz * 128` bytes
  per retained scene with checked arithmetic. A scan using adjacent histories
  budgets both scenes before decoding the first one.

No site-count cap is imposed. Sites execute serially; additional sites add
work and storage rather than multiplying simultaneous scan memory. Memory
admission is a conservative estimate checked at the time of execution, not a
reservation against other processes. The estimate is for scanning and its
writer/PPI working set; it does not reserve accumulated output storage or the
complete compressed GIF history buffer.

## Inputs for a worker quote

The native estimate returns the actual rounded rays per cut, complete gates
per ray, physical cuts, polar bins per site-volume, and the nine-point beam
quadrature count. Their product is an upper bound on pulse samples per
site-volume. It also returns scan working bytes, internal moment count,
formats, requested fields, timing mode and the per-file atmospheric shapes.
`memory_admitted_now` applies only to memory, and `input_fields_validated`
remains false: a quote does not admit a display-only tape as a physical scene.

For explicit sites, `site_count` is the length of the list. For `auto`, it is
null until coverage is resolved from the domain. The catalog size is returned
as `catalog_site_count_upper_bound`; it is a bound on the automatic selection,
not a cap on custom sites. A quote must use actual coverage or state that upper
bound. It must not substitute an arbitrary number of sites.

For all domains, count selected sites and committed history times, including
the initial output and nested domains. Multiply the per-site-volume work by
those counts. Preserve scan timing, available neighboring records, fields,
formats, CPU allocation, shape and input bytes as quote inputs. Include radar
file writing, PPI rendering, cumulative GIF updates, storage and upload costs
separately where those are measured. A worker can apply its declared job
budget and deadline to this estimate; the native engine does not infer an
unmeasured dollar price or silently reduce resolution to meet it.

## Measured reference

The complete three-site path over one 1799 x 1059 x 50 analytic history at
3 km spacing took **20.436 seconds on four CPU threads**, with peak RSS
**10,196,888 KiB**. Configuration was VCP 212, 230 km range, 250 m gates,
1 degree requested azimuth spacing, reflectivity and velocity, Level II and
CfRadial 1. The timing includes source hashing, history reading, beam
simulation, radar files, PNGs and GIFs. It excludes forecast work and upload.

That measurement belongs to executable SHA-256
`821ba1e703c63a640c07e26f0936f08cf1aca60e9f87461a26a9e40bdc1475f7`.
It is a reference observation, not a universal seconds-per-sample rate. It
does not measure dual-pol throughput, every microphysics scheme, repeated
long-history GIF encoding, other CPU allocations or other storage systems.
The native resource response carries the same measured configuration and
scope for consumers that do not read this document.

## Output size bound

`output.output_bytes_per_site_volume_upper_bound` bounds the bytes one site's
volume leaves on disk: every requested format at its widest gate word (2 bytes
for Level II, 4 for the others) with 1% for compression framing, 1 KiB per ray
and 4 MiB per file; each PPI image as an uncompressed RGBA PNG; and each PPI's
frame in its GIF loop at the 12-bit LZW worst case, counted twice because a
loop is rewritten whole beside the one it replaces. A forecast's disk admission
multiplies it by the listed sites and the committed history frames of every
grid. `scene_shapes` (`[nx, ny, nz]` per grid) lets a forecast door price
memory for grids whose history does not exist yet.
