# Prepared fabric binding and minimal source integration

Licence: actual root LICENSE at `99133b887eff9386d96c2a4f01df33cdacca4824`
reads “Apache License”, “Version 2.0, January 2004”. This document adds no
upstream implementation or dependency. Existing source/data notices remain
separate. No UCAR code, input, fixture or ABI is involved.

## Existing driver resource API

Use the implemented generic assembler's preparation row:

```json
{
  "name": "fabric",
  "factory": "gpuwm.hydrofabric:prepare_resource",
  "phase": "prepare",
  "options": {"manifest": "SELECTED_PREPARED_MANIFEST.json"}
}
```

The factory accepts keyword-only `settings, options, state, physics,
device_id, mode, components`. It calls
`gpuwm.hydrofabric.load_cuda(manifest_path, *, device_id, array_module=None)`.
Its `PreparedHydrofabric` result exposes `.arrays`, `.restart_identity`,
`.manifest`, string `.node_ids` and `.catchment_ids`, and setup-only tuple
`.level_offsets`/`.reverse_level_offsets`. Array versions of both level rosters
and offsets are in `.arrays`. This immutable geometry resource has no independent
`advance`; the `prepare` phase registers its inventory/identity without a launch.
Numerical components retain their own `.advance(dt_seconds, *, stream=None)` or
explicit owner launch method. Do not create a second routing advance for fabric.

Every payload retains its manifest dtype/shape and C-order allocation; current
IDs are i64, graph indices/offsets i32, original overlap u64/u32/f64 and Noah
aliases i64/i32/f32. Do not cast one consumer's CSR into another in Python.
`restart_identity` records source/request/payload/network hashes and declared
ordering/numerics, plus parent authority when supplied. The selecting driver
must additionally pin the manifest authority. All Rust reads, pinned allocation
and native H2D copies happen once during preparation. No per-step host work.

For caller-owned pinned/F-order transport use the already registered production
`RawPayloadReader().read_into(path, expected_sha256, address, nbytes)`;
Python passes metadata/opaque addresses. Rust checks complete length and digest
before copying. Numerical owners supply dtype/order/shape and destination
lifetime. This preserves bytes, not an inferred diffusive mesh. Rust users can
open `manifest::PreparedManifest`, call `read_field_into` and verified
`read_auxiliary("network.json")`; additive ABI1 also exports
`arwen_hydrofabric_manifest_read_into`. No replay crate/native runtime dependency.

## Actual case inventory

Case(b) original GeoPackage graph, not an initialized routing realization:

`<prepared-root>/lowercolorado-north/6a8a6af4d40cf0bb6438/manifest.json`

SHA256 `919dba63435f7d9b6670cfb84c9f5284b59600b9db69efc47dac7e522212f886`;
6971 nodes/catchments, 6970 edges, 323 forward and reverse levels. Stable source
strings survive; numeric suffixes are local transport IDs. Source incoming-list
reduction order is explicitly unverified against native arithmetic order.

Case(b) native MC/lake replay resource:

`<prepared-root>/native-routing-07/44e7c5ec67199d0d036f/manifest.json`

SHA256 `ef6c0eb437ea5d16728bb17b418cd523c680ab4ea9ae80f48f9e9b3732e8a50e`;
6742 internal nodes (6723 MC + 19 lakes), 6741 edges, 272 forward and reverse
levels, separate original 6971-catchment axis. It preserves native collapsed
connections/reduction order/MC parameters. Publication IDs and MC above-bed
depth IDs have separate rosters. This replay resource does not establish a user
initializer, diffusive grouped reaches, or event initial state. Both case(b)
fabric resources retain unresolved source data terms and remain private.

Both case(b) manifests already contain `catchment_area_m2` (f64, 6971 rows)
and `catchment_to_node` (i32, 6971 rows). The areas are polygon areas computed
from the original EPSG:5070 divides, independent of normalized forcing weights.
The replay preparer copies those area bytes unchanged and maps the original
divide receivers through the captured waterbody collapse. The separate
catchment roster remains in verified `network.json`; the replay receiver index
addresses its internal collapsed node roster, including reservoirs, rather than
the output reach-ID roster. Do not describe these existing arrays as absent.

The preparation audit checks their seals, shapes, positive finite areas,
receiver bounds and equality to the network's receiver words:

```sh
# CARGO_HOME, CARGO_TARGET_DIR and TMPDIR must be inside your own scratch.
cargo build --release --offline --locked -j8 --example audit_catchment_routing
"$CARGO_TARGET_DIR/release/examples/audit_catchment_routing" FABRIC_MANIFEST.json CFE_INITIAL_MANIFEST.json
```

The optional second argument compares the exact string roster from an actual
`cfe.initial.pack/v1` manifest. It checks only producer metadata, without loading
or qualifying producer parameter/state payloads. A valid geometry inventory is
reported independently of producer membership/order; this diagnostic never
admits physical transfer. No numeric suffix join or partial-roster replication
is used. The retained example CFE initializer names cat-67/cat-27/cat-52 at
2015-12-01, which is a different roster from either case(b) fabric.

The remaining case(b) dependency is a CFE initializer and explicit area
convention for the selected real catchments, its receiver binding and native
delivery phase, plus a disjoint source/replacement budget. Polygon area is not
automatically the native CFE `areasqkm` parameter, and summing overlapping
polygons does not establish a disjoint drainage area. Geographic provider
weights and event/public-data qualification also remain separate. The audit
performs no CUDA work, state initialization or per-step processing.

Case(c) available original government source subset is a different schema:
`<public-source-root>/russian-river-subset/network.json`,
SHA256 `ac3c274d1dfb3408bc62a448e8a82ed0a4806b548a86464b113dc75b84bcacba`.
Its `arwen.source-network-subset.v1` JSON preserves 1445 nodes/1475 explicit
arcs and source attributes, not GPU resident buffers or an HYFeatures crosswalk.
The retained polygon subset in delivery/code/public-geometry/russian-river has
1433 polygons; 12 source IDs are missing. Do not pass either audit schema to
`load_cuda` or fill missing polygons. The selected v2.2 data's noncommercial
terms remain a separate private-referee hold. No production case(c) prepared
manifest, geographic AORC overlap, native physical initializer or executed
parent geometry is supplied by these partial source artifacts.

## Minimal integration into lead 93c523

Lead `93c523ee290c7af2bc3bc9edeb63ee45d0cf43db` already has production raw reader
and roster verification equivalents. The additional logical prerequisites are:

1. `b5473ebfc46d376ce06a77979c2d93fe2f1b93d7`: manifest module, public header,
   schema and read example; add `pub mod manifest`.
2. `183d9229030d3b8d57000151aedc39d0b072b0c8`: optional preprocessing registrar.
3. `dae3b50d0887992cb992aba573eb28194ea74b7a`: cell-overlap module plus additive
   Rust/Python/CLI entry points and registrar switch.

Whole-commit application is not a minimal closure: old lib/Python hunk context
and documentation reference other source-tool/documentation commits absent from
lead. A checked nine-file source-only patch recontexts those additive module
and entry-point lines, preserving implementation exactly. It excludes the
already integrated bridge/roster changes, unrelated source tools, optional
NetCDF features and documentation prerequisites. Patch SHA256:
`dfb7879e3297a571cf8f147768fc3adec863d2a5feb73ba5129190ac7415e9f6`.

The exact patch, base/file hashes, commands and results are delivered under
`code/loader-integration/`. Lead can inspect/apply the patch and create a forward
commit on its own branch; no writes to the lead tree were performed here.
The separate parent-grid change `7400845` is already owned/imported by driver;
it is not silently included or removed by this patch.

Focused actual validation of the isolated base plus patch:

```sh
# Run from tools/nextgen_hydrofabric; all three directories are lane scratch.
cargo build --release --offline --locked -j8 --bin rw_hydrofabric --example read_manifest
RAYON_NUM_THREADS=24 "$CARGO_TARGET_DIR/release/examples/read_manifest" ACTUAL_MANIFEST.json
# Source-only Python registrar invokes the freshly built Rust CLI:
python3 -B VALIDATION_DIRECTORY/validate.py
```

The retained validation script checks the exact generic factory keywords and
`set_defaults(func=callable)`, executes `prepare --cell-overlap` on the actual
737x582 north-first grid, and compares all ten resulting fields against the
existing reference using Rust verified reads and native byte comparison.
The separate read example compares manifest reads against the existing raw
Rust ABI on both real graph manifests. These are source-integration/transport
checks, not new native physics or coupled/GPU/render evidence. The isolated
archive build reports `unrecorded-source`; exact base, patch and source file
hashes identify it instead of relabelling it as a full repository commit.

Catchment-cell overlap remains geometry, not a driver resource or physical
source policy. Preserve max coverage 2 and explicit overlapping polygons.
Actual parent authority, initialized physical producers/parameters/stores,
drain receivers, disjoint water accounting, reservoir controls and diffusive
native grouping/mesh/boundary/adaptive state remain owner inputs. See
[STARTUP.md](STARTUP.md) and [DIFFUSIVE-NETWORK.md](DIFFUSIVE-NETWORK.md).
