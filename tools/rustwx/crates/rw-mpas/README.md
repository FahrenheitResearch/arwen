# rw-mpas

MPAS mesh generation, initial conditions, and the history-to-wrfout render
bridge, in Rust. Three binaries:

| binary | what it does |
| --- | --- |
| `rw_mpas_mesh` | makes an MPAS grid file from a resolution request |
| `rw_mpas_init` | builds an initial-conditions file from a WPS intermediate file |
| `rw_mpas_convert` | gathers an MPAS history frame onto a structured render window |

## `rw_mpas_mesh`

Before this existed, "arbitrary resolution" meant arbitrary among the meshes
somebody else had published. Two were registered — a uniform 120 km mesh at
40,962 cells and a variable 25 km/92 km mesh at 163,842 — so a 10 GiB card fell
onto the uniform one, because the next published mesh up needs 26.4 GiB.

    rw_mpas_mesh --spec examples/conus-box-15km.json --vram-gib 10 \
                 --fit-spacing yes --out mesh.grid.nc --receipt mesh.json

### The request is data

A spec is a background spacing plus a list of refinement regions. Adding a
place to refine — any box, any cap, any polygon, anywhere, at any spacing, at
any ratio — is a row in the `regions` array and touches no code.

```json
{
  "background_km": 120.0,
  "regions": [
    { "shape": { "kind": "cap", "center_deg": [39.0, -98.0], "radius_km": 1200 },
      "spacing_km": 20.0, "transition_km": 900.0 },
    { "shape": { "kind": "lat_lon_box", "lat_deg": [30, 45], "lon_deg": [-105, -90] },
      "spacing_km": 12.0, "transition_cells": 30 }
  ]
}
```

Shapes: `cap`, `lat_lon_box`, `polygon`. Ramp: `transition_km` or
`transition_cells`. Where regions overlap, the finer one wins.

The ramp is a `tanh` centred **on the region boundary**, so a region has to be
several ramp widths across before its middle reaches its own requested spacing.
The receipt reports the DELIVERED spacing against the REQUESTED spacing cell by
cell for exactly this reason; a region's nominal `spacing_km` is not the answer
to "did I get what I asked for".

### Sizing

`--vram-gib` sizes the mesh from the measured device footprint model

    footprint_MiB = 9798 + cells x 86,630 B

fitted to three points measured on ONE card and ONE build (RTX 5090, float32,
nVertLevels = 55, full physics): 38,857 cells peaked at 13,165 MiB (model
13,008, -1.2 %), 40,962 at 13,182 (13,190, +0.1 %) and 163,842 at 23,334
(23,334, 0.0 %). That gives 5,350 cells at 10 GiB, 79,717 at 16 GiB and
278,030 at 32 GiB.

The fixed term dominates a small card because most of it is the CUDA
local-memory backing store, which is sized from the card's resident-thread
capacity rather than from the mesh. It is therefore a property of the CARD, not
of this model: `gpuwm/data/mpas/mesh-sizing.json` carries it per card and
refuses an unmeasured one by name rather than borrowing this card's number.

The model this replaced -- `5018 + cells x 140,916 B` -- was wrong by 22 % in
the direction that hurts: it sized a 38,857-cell mesh as a 10 GiB fit when that
mesh peaks at 13,165 MiB. Its anchors came from registry notes describing runs
on other hardware.

A request that does not fit is **refused with both numbers** rather than quietly
delivered coarser. `--fit-spacing yes` rescales every spacing by ONE factor,
keeping every ratio between them, until it fits.

### Quality

`--tolerance` is the contract: stop when the MEAN of `delta/h` falls below it,
where `delta` is how far a generator sits from its own density-weighted Voronoi
centroid and `h` is the local spacing. The contract is on the mean and not on
the max by measurement — over 400 sweeps of a uniform relaxation the mean falls
monotonically from 8.6e-2 to 2.2e-4 while the max wanders between 3e-3 and
1.5e-2 with no trend, because the max is set by a handful of cells next to the
twelve pentagons and it steps every time a Delaunay edge flips. The max is
measured and reported on every run.

Reaching `--sweeps` without meeting `--tolerance` is a refusal. So is a stalled
relaxation and so is a limit cycle, each with the number that made it refuse.

### The emit gate

Nothing is written until the mesh passes. Every check names the concrete
breakage it prevents, and four of them exist because nothing downstream looks at
them: the MPAS port's own validator silently accepts a swapped `cellsOnEdge`, a
swapped `verticesOnEdge`, a wrong `angleEdge` value and a wrong `weightsOnEdge`
value. Those four produce a mesh that validates clean everywhere else and
integrates wrong.

The gate covers: the closed-sphere Euler characteristic; the total coordination
defect of 12; `nVertices = 2n-4` and `nEdges = 3n-6`; mutual neighbours and
edge/vertex reciprocity; a connected cell graph; the three area sums closing on
`4*pi`; the kite partition; the polygon and kite decompositions of every cell
agreeing to within the arithmetic's own noise (an ABSOLUTE area in machine
epsilons, because the two are analytically identical and what separates them in
`f64` is a constant, not a fraction of the cell); primal/dual orthogonality;
the edge orientation lock; every ring
winding counter-clockwise seen from outside; positive finite metrics; the
Thuburn antisymmetry of the TRiSK weights over every ordered stencil pair; and
zero nonzero weight padding.

### The regional cull

`--cull-parent` cuts a limited-area mesh out of an existing global grid or
static file instead of generating one:

    rw_mpas_mesh --cull-parent x4.163842.grid.nc --region window.json \
                 --out region.grid.nc --graph region.graph.info --receipt cull.json

`--region` is a single Shape row — the same `cap` / `lat_lon_box` / `polygon`
rows a resolution spec's regions use — so a new region is a JSON row, never a
code path. The output byte-matches the native MPAS-Limited-Area v2.2 cull of
the same region on the same parent: the three `bdyMask` variables (0 interior,
rings 1..7 outward), parent-subset element ordering, contiguous `indexTo*ID`,
true-0 sentinels on outermost-ring connectivity, `on_a_sphere`/`sphere_radius`
only, the parent's classic format, and the METIS `graph.info`. Graded
byte-identical against the pinned native culls of both published parents
(grid and static), Windows and Linux builds producing the same bytes.

One documented divergence: the native tool reindexes stored 0 through a numpy
`map[field-1]` wrap that reads the LAST global element's map entry; this
culler maps stored 0 to 0 always, and the receipt's `native_wrap_divergence`
records whether a parent would ever expose the difference (neither published
parent does).

### The unit-sphere gotcha

An MPAS grid file carries `sphere_radius = 1.0`. `areaCell`, `dcEdge`,
`dvEdge`, `kiteAreasOnVertex` and `nominalMinDc` are all UNIT-SPHERE quantities
despite their `units` attributes reading `"m"` and `"m^2"`. Multiply lengths by
6,371,229 m. Computing a spacing without that normalisation prints 0.0.

### What this does NOT produce

A **grid** file. Running the mesh also needs a matching **static** file —
terrain, land use and soil on the same cells, `nVertLevels = 55`,
`nSoilLevels = 4`, and a `nominalMinDc` that is FP32-bit-exact against the
declared nominal spacing. That is built against a terrain archive and is not
part of this binary. The boundary is stamped into every file it writes as the
`rw_mesh_boundary` global attribute.

## Tests

    cargo build --release --locked --offline -p rw-mpas
    cargo test  --release --locked --offline -p rw-mpas

`tests/goldens/` holds the two published meshes in a compact container. The
field emit is graded against them field by field; the spherical Delaunay is
graded by feeding their own cell centres back in and requiring the published
neighbour ring exactly; and `tests/mesh_pipeline.rs` runs the whole generator
and reads its output back through `netcrust`, a reader that shares no code with
the writer.

## Surface update file (MPAS-A v8.4.2 case 8)

The existing init binary also writes the immutable surface stream directly:

```sh
rw_mpas_init --mode surface-update \
  --sfc-prefix /case/SFC --static /case/static.nc --out /case/surface.nc \
  --start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS \
  --fg-interval 21600 --surface-interval 21600 --receipt /case/surface.json
```

The two intervals are explicit seconds and must match. The Gregorian time loop
includes the stop time when it is on the interval, reading `PREFIX:YYYY-MM-DD_HH`
and writing `xtime`, `Time`, `sst`, and `xice` at each time. Formats 3, 4 and 5
are decoded in Rust; case 8 supports lat/lon, Gaussian (format 5), and polar
stereographic grids. A missing LANDSEA slab follows the native unmasked path.
SST/SKINTEMP and SEAICE are consumed in file order, then cold oceans freeze and
sea ice is limited to [0, 1]. Atmospheric, soil and vertical-capsule arguments
are unnecessary for this mode. `--mode surface-update --abi` reports its ABI;
the original case-7 ABI and invocation remain valid.

The output is staged beside its final path and renamed after every record is
written. Invalid intervals, a missing landmask, corrupt intermediate framing,
and missing later inputs preserve an existing output. A static with maximum
landmask zero refuses, matching native's grid-versus-static check.

The standalone oracle reuses the full pristine MPAS surface module with its
native WPS reader, llxy, interpolation, queue and bitarray modules. Framework
stubs replace only pools, the clock, logging and stream output. Build it through
the lane's build-slot wrapper, then set `MPAS_SURFACE_ORACLE` to its executable
and `MPAS_SURFACE_TEST_OUT` to an owned scratch directory before running:

```sh
cargo +1.94.0 test --release --offline --locked -j4 -p rw-mpas \
  --test surface_update -- --include-ignored --skip strict_native_projection_refusals
```

These binary and module tests are implementation evidence. Real native decks,
whole-model surface substitution, and G0-G2 are separately required for landing.

The ignored native error-path test also reaches the surface module's zero-mask
and both file-open refusals. To reach the second-open failure, its framework
I/O adapter deletes the specified fixture after the first scan when
`MPAS_SURFACE_ORACLE_DROP_AFTER_SCAN` is set. The test sets this only for that
case; numerical replays leave it unset, and the pinned MPAS sources are unedited.
`MPAS_SURFACE_ENGINE` can select a copied optimized binary for an artifact replay
on a host that did not build the tests.

Native case 8 initializes only lat/lon, Gaussian and polar mappings. Active
Mercator/Lambert fields therefore reach its explicit `LATLON_TO_IJ` fatal error
before interpolation (`mpas_init_atm_llxy.F:601-604`); Rust refuses those same
fields. Unrelated slabs with those projections remain readable and ignored.
The ignored `strict_native_projection_refusals` test witnesses this behavior
with f0's complete strict executable at one and two ranks. Set
`MPAS_SURFACE_NATIVE`, `MPAS_SURFACE_NATIVE_STATIC` and
`MPAS_SURFACE_NATIVE_GRAPH_PREFIX` to that executable, a published full init
file and its decomposition prefix, using the pinned MPICH runtime from f0's
`toolchain.txt`.

The ignored `surface_published` test runs the Rust binary and strict native
executable on real inputs, compares every `Time`/`sst`/`xice` binary32 word and
all 64 bytes per `xtime`, and removes outputs after writing `real-replay.json`.
Set `MPAS_SURFACE_DECKS_JSON` to a JSON array of explicit deck objects with
`label`, `static_path`, `sfc_prefix`, `graph_prefix`, `start`, `stop`, `interval`
and optional `published_native` paths. It requires `MPAS_SURFACE_NATIVE` and
`MPAS_SURFACE_TEST_OUT`; `MPAS_SURFACE_ENGINE` can select the copied engine.
Set `MPAS_SURFACE_ORACLE` as well to compare every real destination cell and
record against the strict standalone complete-module build. Its clock adapter
allocates the requested record sequence, including complete five-day updates.
Its receipt records that the hex front door and whole forecast remain separate
proofs.

`tests/surface_oracle/prepare_ungrib.py` prepares control files and input links
for f0's pinned native WPS decoder. The reference decoder performs all field
processing; Python only selects the existing surface Vtable definitions and
writes its namelist. This bounds scratch output when replaying complete update
sequences from published GRIB inputs.
