# Prepared graph handoff to diffusive routing

Licence: original reader/audit code uses the root LICENSE, read as “Apache
License”, “Version 2.0, January 2004”. The retained t-route source notice is
for pin `12a8eae0cdfed437143c590659fa7077605a5e70`. This handoff adds no upstream
solver code, dependency, numerical Stage, mesh adapter or initialization.
Dataset permission remains separate; these case(b) artifacts are private.

## Actual prepared artifacts

Paths below are existing fixture locations for review, never installed defaults.
Payloads live beside their manifest; use its descriptors and pinned authority.

| Graph | Nodes | Edges | Forward / reverse levels | Identity |
|---|---:|---:|---:|---|
| Original LowerColorado GPKG | 6971 | 6970 | 323 / 323 | Preserved flowpath strings; numeric suffix transport IDs |
| Native MC/lake replay, collapsed | 6742 | 6741 | 272 / 272 | Captured native MC/lake IDs, sorted i64; 6723 MC nodes and 19 lakes |

Original graph manifest:
`<prepared-root>/lowercolorado-north/6a8a6af4d40cf0bb6438/manifest.json`

* Manifest SHA256: `919dba63435f7d9b6670cfb84c9f5284b59600b9db69efc47dac7e522212f886`
* `network.json` SHA256: `708fc7301006f2ad1280aee507d2e4c6a91b492bc1828e1992fef3b6d0d64b89`
* Axis follows the original SQLite flowpath scan; original strings are retained.
  Upstream incoming-list order is source scan order, explicitly **unverified
  against native arithmetic reduction order**.

Native MC/lake graph manifest:
`<prepared-root>/native-routing-07/44e7c5ec67199d0d036f/manifest.json`

* Manifest SHA256: `ef6c0eb437ea5d16728bb17b418cd523c680ab4ea9ae80f48f9e9b3732e8a50e`
* `network.json` SHA256: `454c3059c8409842711339ca18f900ec5c855d247fb01e7cd5564e483b13a97e`
* Upstream incoming-list order is the captured native reverse-network order.
  Both artifacts retain the original 6971 catchments on a separate axis.
* `output_reach_ids` can remap a collapsed lake ID to its captured outlet
  flowpath for publication. It is not the topology axis.
* This is a replay artifact, not proof of a standalone domain initializer or
  a native diffusive grouped-reach mapping.

## Existing fields and index conventions

Manifest schema: `arwen.nextgen.hydrofabric.v1`. Each `files[name]` contains
relative `path`, little-endian `dtype`, `shape`, `bytes`, and `sha256`.
The Rust audit emits these original descriptors unchanged.

| Name | Dtype / shape | Meaning |
|---|---|---|
| `node_ids` | `<i8 [N]` | Stable IDs on this graph's node axis, with the provenance above |
| `downstream` | `<i4 [N]` | Zero-based receiving node, or -1 exterior; both graphs have one exterior |
| `upstream_offsets` | `<i4 [N+1]` | CSR offsets, starts 0, ends E |
| `upstream_indices` | `<i4 [E]` | Zero-based incoming nodes, retaining the declared reduction order |
| `ordered_nodes` | `<i4 [N]` | Headwater-to-outlet dependency roster |
| `level_offsets` | `<i4 [Lf+1]` | Ranges into `ordered_nodes` |
| `reverse_ordered_nodes` | `<i4 [N]` | Outlet-to-headwater dependency roster |
| `reverse_level_offsets` | `<i4 [Lr+1]` | Ranges into `reverse_ordered_nodes` |

For a forward range, every upstream dependency occurs in an earlier range.
For a reverse range, every downstream dependency occurs in an earlier range.
Each roster covers every node once, ranges are nonempty, indices within a range
ascend, and offsets start 0/end N. Reverse levels are computed independently
from downstream dependencies; reversing the forward array is insufficient.
All these fields are vectors, whose linear storage is identical in C/F order.
They do not establish any 2D Fortran layout or native one-based solver index.

The existing `routing_parameters` is **C-order f32[N,8] MC geometry**. It is
not native f64 F-order diffusive geometry. Numeric suffix IDs from the normal
GPKG and native MC/lake IDs must not be called native diffusive solver reach IDs.

## Read APIs and exact audit

Production Rust: `manifest::PreparedManifest::open(path)`, then
`read_field_into(name, &mut destination)` and `read_auxiliary("network.json")`.
These check descriptor shape/bytes, path confinement, length and SHA256.
Existing production ABI1 `arwen_hydrofabric_manifest_read_into` supports the
driver's own pinned transport. Python parses verified metadata only.

For a separate owner-defined F-order packet, existing production
`gpuwm.hydrofabric.RawPayloadReader().read_into(path, expected_sha256,
destination_address, nbytes)` preserves the exact bytes. The numerical owner
must declare dtype, shape and order, allocate the matching F-contiguous pinned
and device buffers, and pin the packet identity. The general
`gpuwm.hydrofabric.load_cuda` currently allocates C-order arrays; it does not
provide a diffusive F-order mesh loader. No new reader is required.

Read-only Rust audit command, with all Cargo/cache/target/tmp under lane scratch:

```sh
cargo build --release --offline --locked --example network_contract -j8
RAYON_NUM_THREADS=24 "$CARGO_TARGET_DIR/release/examples/network_contract" MANIFEST.json
```

It SHA-checks all eight raw graph fields and network.json, compares every
decoded graph vector to that sealed JSON, checks unique string/numeric IDs,
CSR reciprocity, and both complete canonical level sequences. It reports
per-stage CPU timings, exact descriptors and source stamp. It does not write
the prepared graph or parse numeric payloads in Python. This is graph integrity
and dependency validation, not a new native diffusive physics replay.

## Missing solver inputs

Neither artifact supplies diffusive per-reach computational node counts
`ncomp`, `frnw`, ordered mainstem roster, the solver-reach-to-flowpath/lake
mapping, f64 F-order `[mx,nr]` mesh/geometry/boundary/initial fields, or resident
adaptive-step state. A fabric flowpath count is not `ncomp`. The diffusive
owner's native one-based `frnw`/mainstem conventions and zero-based prepared
level convention require an explicit, source-owned mapping before binding.

Provide the authoritative grouped-reach roster, computational-node membership
and geometry/boundary policy before extending this handoff to a solver packet.
No grouping, cell insertion, F-order conversion, native ID crosswalk or
physical initialization is inferred from either prepared graph.
