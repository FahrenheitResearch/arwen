# Catchment to parent-grid area weights

The existing production preparation emits catchment-row CSR directly from
divide polygons and an explicit projected metre grid. Forcing consumes the
three fields below from `arwen.nextgen.hydrofabric.v1`; no separate geometry
adapter or external regrid implementation is needed here.

| Field | Little-endian raw layout | Meaning |
|---|---|---|
| `weights_offsets` | u64[C+1] | Edge range for each ordered catchment, starting at zero and ending at K |
| `weights_indices` | u32[K] | Zero-based flat parent cell index |
| `weights` | f64[K] | Intersection area divided by the sum of covered intersection areas in that catchment |

All payloads are one-dimensional contiguous arrays; `files[NAME]` in the
manifest pins relative path, dtype, shape, byte count and SHA256. Current
preparation requires `nx*ny <= u32::MAX`; signed exchange aliases impose an
additional i32 index bound. It does not emit u64 cell indices. The forcing
kernel should retain these f64 weights rather than use the Noah exchange f32
aliases.

## Roster, grid and reduction order

CSR row i is exactly `network.json.catchment_ids[i]`, a preserved full string
ID. `auxiliary["network.json"].sha256` pins that indexed roster. Numeric
`node_ids`, node topology order and routing output IDs are separate axes.
`catchment_to_node` is an explicit receiver mapping, not permission to reorder
catchment rows. Pin the selecting manifest SHA as well as its payload hashes.
`weights.json` is diagnostic metadata; its digest is not in this manifest's
auxiliary map, so it must not replace the verified network roster or grid.

`manifest.grid` supplies nx, ny, x0, y0, dx, dy, crs and row_order. The flat
field is C[ny,nx], x-fast: `flat = row*nx + column`. x0 and y0 are the western
and SOUTHERN cell edges, including when row_order is `north_to_south`.
For southern-origin row y, the storage row is y for `south_to_north`, or
`ny-1-y` for `north_to_south`. Positive dx/dy are metre spacings.

Within each catchment, Rust visits southern-origin rows south-to-north and
columns west-to-east, retaining positive intersections only. This order is
deterministic and is NOT ascending flat index on a north-first grid. Rayon
parallelizes catchments; indexed collection preserves the catchment roster and
the serial edge order inside each row. A forcing reduction that needs the
same arithmetic order must retain this sequence; a parallel GPU reduction
must qualify its own rounding against its referee. The preparer supplies
geometry and order, not that numerical equivalence claim.

`overlap_m2` f64[K], `catchment_area_m2` f64[C], and `coverage_fraction`
f64[C] accompany the CSR. Weights sum to one within floating-point rounding
over COVERED area; `coverage_fraction` records covered/full polygon area.
Partial coverage is retained; zero overlap errors. A normalized row does not
establish full polygon coverage, a disjoint source partition, or a particular
physical rate/depth interpretation. Destination-cell transpose weights divide
by full cell area and have a different contract in STARTUP.md.

## Preparation and resident loading

At forecast setup obtain the actual `physics.hydro_parent_geometry`, call
`gpuwm.hydrofabric.parent_grid_contract(parent_geometry)`, and supply both
`contract["grid"]` and the full contract as `grid` and
`parent_grid_contract` in the GeoPackage preparation configuration. Existing
`rw_hydrofabric prepare CONFIG.json` / `gpuwm.hydrofabric.prepare(config)`
performs all projection/intersection work in Rust and emits
`OUTPUT/<request-sha256-first-20>/manifest.json`. Rust rejects inconsistent
grid metadata; the manifest and resident restart identity retain the executed
parent identity. Geometry from another grid does not establish this binding.

The registered prepare-phase factory is
`gpuwm.hydrofabric:prepare_resource`, with `options.manifest`. Alternatively
`load_cuda(manifest_path, device_id=...)` returns a PreparedHydrofabric with
`arrays["weights_offsets"]`, `arrays["weights_indices"]`, `arrays["weights"]`,
`catchment_ids`, `manifest["grid"]` and `restart_identity`. The production Rust
reader checks hashes and copies into native pinned allocations; startup H2D
preserves the declared dtype. No Python numeric parse, per-step CPU work or
fabric advance is involved. Driver/speed assemble the numerical owners, and
forcing owns CUDA area aggregation and external forcing regrid.

## Existing private case(b) artifact

Prepared manifest:
`<prepared-root>/lowercolorado-north/6a8a6af4d40cf0bb6438/manifest.json`

SHA256 `919dba63435f7d9b6670cfb84c9f5284b59600b9db69efc47dac7e522212f886`.
The verified `network.json` SHA256 is
`708fc7301006f2ad1280aee507d2e4c6a91b492bc1828e1992fef3b6d0d64b89`.
It has 6971 catchment rows and 180005 intersections on EPSG:5070,
nx=737, ny=582, north_to_south, dx=dy=1000,
x0=-730254.9996999999, y0=612815.0003.

| Relative payload | Shape | SHA256 |
|---|---|---|
| weights_offsets.bin | [6972] | cd0b279f1b24a39b801e4a597343acccd8c9012cd56df82f57d84a20f0b33994 |
| weights_indices.bin | [180005] | 259af6e3c29fc248231f7f6b776c7e2106a52db15379b966eb776431ef0c19eb |
| weights.bin | [180005] | c49b7da09a23b3eaf78e2862e349a3f674d29dacb35b2add14fa6a024b033430 |

This is an actual polygon-intersection artifact on the explicit routing grid,
not an executed ArWen-parent overlap or a licensed public demonstration.
Source data terms remain unresolved/private. The actual forecast parent,
corresponding polygons and coverage policy must be selected before preparing
the coupled-parent artifact. Geographic AORC provider sampling/area authority
is separate; this projected metre-grid contract does not manufacture it.
