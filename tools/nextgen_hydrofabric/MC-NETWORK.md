# Prepared network for MC and reservoir hooks

Licence: original reader/audit code, root LICENSE read as “Apache License”,
“Version 2.0, January 2004”. Retained t-route notice is pinned at
`12a8eae0cdfed437143c590659fa7077605a5e70`. No additional upstream code or
dependency is introduced. Case(b) source-specific data terms remain unresolved;
this captured-input resource is private replay material.

The numerical contract belongs to the MC owner:
`the channel-reach lane's CONTRACT.md (an internal lane note, not in the public tree)`.
This document binds existing prepared fields to that contract. The driver owns
assembly, MC owns ordered numerical launches, and reservoirs owns its subset
component. No additional parser, runtime, packer or numerical adapter is needed.

## Existing resource

Manifest:
`<prepared-root>/native-routing-07/44e7c5ec67199d0d036f/manifest.json`

SHA256: `ef6c0eb437ea5d16728bb17b418cd523c680ab4ea9ae80f48f9e9b3732e8a50e`.
`network.json`: `454c3059c8409842711339ca18f900ec5c855d247fb01e7cd5564e483b13a97e`.
Common topology has 6742 nodes (6723 MC, 19 reservoirs), 6741 edges and
272 dependency levels. Original catchments retain their separate 6971-row axis.
IDs are native collapsed IDs sorted as i64. Incoming CSR lists preserve native
compute reduction order; they are never sorted by ID or topological rank.

| Payload | Native prepared layout | Consumer |
|---|---|---|
| `node_ids` | i64[N] | Internal common topology IDs |
| `upstream_offsets`, `upstream_indices` | i32[N+1], i32[E] | Exact incoming reduction order |
| `ordered_nodes`, `level_offsets` | i32[N], i32[L+1] | Full topology including reservoirs |
| `mc_ordered_nodes` | i32[6723] | MC launch roster, excludes reservoirs |
| `routing_parameters` | C-f32[N,8] | dx,bw,tw,twcc,n,ncc,cs,s0 |
| `routing_lateral` | f32[N] | Captured one-hour m3/s interval, twelve 300 s updates |
| `routing_state` | C-f32[N,3] | Captured initial q,velocity,depth |
| `reservoir_ids` | i64[19] | Native reservoir row axis |
| `reservoir_node_indices` | i32[19] | Reservoir row to common topology node |
| `reservoir_level_*_rows` | i32[selection length] | Reservoir row selections at topology levels |

`routing.mc_level_offsets` is host metadata with L+1 entries into
`mc_ordered_nodes`. Empty MC ranges at reservoir-only levels are retained.
`routing.reservoir_levels` names the 17 selection payloads and their levels;
every reservoir row is selected exactly once. `routing.validated_reservoir_mapping`
seals the producer's map validation. Use these metadata entries, not the full
`ordered_nodes` array as an all-MC roster. `routing.assume_short_ts=true` retains
the actual native upstream-time convention.

MC parameter source columns are captured f64 and are explicitly converted to
f32 at Rust setup, matching the native MC call boundary. Reservoir rows have
zero MC parameters/lateral and are never launched as MC. Their common routing
state starts at native release, zero velocity and zero placeholder depth;
the reservoir owner's height/store/controls are separate native-width carriers.
After reservoir hooks, routing depth on those rows is water elevation and must
not be labelled channel depth. `output_reach_ids` is a distinct publication
crosswalk to native outlet labels; it does not replace internal IDs.

## Bind once at setup

`gpuwm.hydrofabric:prepare_resource` with `options.manifest`, `phase=prepare`
uses `load_cuda(manifest_path, *, device_id)` and the registered production
Rust reader. It returns `.arrays`, `.manifest`, `.restart_identity` and host
`.level_offsets` / `.reverse_level_offsets` tuples from verified network JSON.
Host offsets are Python metadata integers; device numeric payloads retain the
explicit i32/i64/f32 descriptors. Python never decodes numeric raw files.
The existing raw C ABI can instead fill caller-owned pinned memory.

MC factory binds `options.network_component`, `.lateral_component` and
`.reservoir_component` to these prepared resources. It uses the fabric's
`routing.initial_state_array` by default; `.initial_state_component` can
override that binding. A declared state must have a prepared array and payload
hash. MC owns `ReservoirLevel(... rows=selection, node_indices=map, ...)`
and `bind_level_hooks`; the reservoir resource is prepare-only and must not
advance independently. Preserve selection/map hashes and all shared arrays in
the restart inventory. Physical lateral producers and source partitions are
separate contracts; this native fixture has no new physical coupling policy.

## Focused native input replay

Build `cargo build --release --offline --locked --example mc_contract -j8`
with target/cache/tmp under lane scratch. Run
`RAYON_NUM_THREADS=24 "$CARGO_TARGET_DIR/release/examples/mc_contract" MANIFEST`.
The read-only Rust example checks payload hashes/types/shapes, source manifest
authority, exact incoming IDs against captured `compute-network` lists, MC and
reservoir selections at every topology level, and every parameter/state/lateral
f32 word against captured native columns. It retains f64-to-f32 boundary
conversions; no tolerance, omitted dry/negative value or inferred source ID.
The earlier `network_contract` audit independently covers complete forward and
reverse dependency levels. Neither check advances a numerical solver.

This replay resource does not establish an independent GeoPackage reservoir
collapse, native-free user initializer, physical runoff/event, permitted public
demo, grouped diffusive reach mesh, or whole-hydro speed. The original 6971-node
GeoPackage graph explicitly has source-scan incoming order unverified against
native arithmetic. It cannot substitute for this captured compute graph.
