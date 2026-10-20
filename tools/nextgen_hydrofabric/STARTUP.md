# Static domain preparation and driver binding

The production static exporter is `nextgen_hydrofabric::bundle::prepare(&Path)`
or `rw_hydrofabric prepare CONFIG.json`. It reads an actual NextGen GeoPackage;
neither a native executable nor a captured state manifest is an input. The
Python orchestration entry is `gpuwm.hydrofabric.prepare(config_path)` and the
registered root door is `gpuwm hydro prepare CONFIG.json`.

Required configuration keys are `geopackage`, `output`, and `data_license`.
Optional `tables` maps semantic layer roles to actual table names. `divides`,
`flowpaths`, and `nexus` are required. `gauge` or `huc`/`huc_column` selects a
subset; the default includes the supplied domain. Record the actual dataset
terms separately from the Apache software licence. A nonempty terms string is
provenance, not a licence clearance decision.

The returned JSON is also written to
`OUTPUT/<first-20-hex-of-request-sha256>/manifest.json`. Its schema is
`arwen.nextgen.hydrofabric.v1`. The request digest covers the configuration bytes
and source GeoPackage digest. Payloads live beside the manifest. Each
`files[NAME]` descriptor supplies relative `path`, LE `dtype`, `shape`, `bytes`,
and `sha256`. `auxiliary["network.json"].sha256` binds the string rosters and
level metadata. The selecting driver must also pin the manifest authority.

| Export | Source and availability | Resident layout |
|---|---|---|
| Reach graph | Actual flowpath/nexus connectivity | `downstream` i32[N], `upstream_offsets` i32[N+1], `upstream_indices` i32[E]; -1 is an exterior downstream boundary |
| Dependency levels | Derived from that graph | `ordered_nodes` and `reverse_ordered_nodes` i32[N]; matching `level_offsets` and `reverse_level_offsets` i32[L+1] |
| Catchment receiver | Actual divides/connectivity | `catchment_to_node` i32[C], in the separately preserved catchment roster order |
| Reach IDs | Full strings always in network.json; numeric payload only when every suffix parses uniquely | Optional `node_ids` i64[N]; a numeric suffix is a local transport ID, never a crosswalk to another dataset |
| MC parameters | Optional flowpath_attributes table, with all eight actual numeric columns for every selected reach | `routing_parameters` f32[N,8], columns `length_m,BtmWdth,TopWdth,TopWdthCC,n,nCC,ChSlp,So`, matching MC `dx,bw,tw,twcc,n,ncc,cs,s0` |
| Parent overlap | Actual divide polygons plus explicit projected grid | `weights_offsets` u64[C+1], `weights_indices` u32[K], `weights` and `overlap_m2` f64[K], `catchment_area_m2` and `coverage_fraction` f64[C] |
| Noah exchange aliases | Same overlap, prepared in Rust | `exchange_offsets` i64[C+1], `exchange_indices` i32[K], `exchange_weights` f32[K] |
| Channel incidence | Actual flowpath geometry plus the same explicit grid | `channel_node` and `stream_order` i32[ny,nx]; not an off-channel drainage or DEM reconstruction |

See FORCING-WEIGHTS.md (an internal lane note, not in the public tree) for the catchment-row CSR,
covered-area normalization, exact edge order, roster/grid hashes, resident
loader and actual case(b) payload paths. The explicit prepared routing grid
still needs the executed-parent authority before coupled use.

The parameter table's absence leaves `routing_parameters` absent; an incomplete
present table errors rather than inventing coefficients. Node and catchment
orders are distinct. Default upstream reduction order follows the source scan
and is explicitly unverified against native arithmetic order. An explicit
`upstream_order` must be the actual referee roster to support that replay claim.

The forward level ranges in `ordered_nodes` run from headwaters toward outlets:
every upstream node is in an earlier range than its receiver. The reverse
ranges in `reverse_ordered_nodes` run from outlets toward headwaters: each
downstream receiver precedes its upstream nodes. Offsets start at zero and end
at N; both rosters cover each node exactly once. Nodes within a range ascend
in the preserved node-axis indices. Reverse levels are computed independently,
not by reversing the forward roster. These are i32 zero-based graph indices,
not a native diffusive solver's ordered mainstem roster, cell counts or frnw
flags. The diffusive owner must establish that solver-specific mapping from
actual geometry and boundary inputs before binding its kernels.

See DIFFUSIVE-NETWORK.md (an internal lane note, not in the public tree) for the actual pinned case(b)
graphs, both level sequences, Rust audit command and missing solver fields.

For the real coupled parent, obtain `parent_grid_contract` from
`physics.hydro_parent_geometry` during setup, then put both `contract["grid"]`
and the full `contract` into the configuration as `grid` and
`parent_grid_contract`. The existing Rust static bridge derives coordinates;
Python handles metadata only. Rust checks the exact grid match and retains the
parent authority in the manifest. This seam needs the driver's actual prepared
forecast and corresponding polygons; a captured routing-grid footprint does
not supply the executed parent.

## Existing resource and read APIs

### Catchment intersections on a terrain grid

`rw_hydrofabric prepare-cell-overlap CONFIG.json`,
`gpuwm.hydrofabric.prepare_cell_overlap(config_path)`, or
`gpuwm hydro prepare --cell-overlap CONFIG.json` transposes existing polygon
intersections into destination-cell CSR. Configuration supplies
`fabric_manifest`, `fabric_sha256`, `destination_manifest`,
`destination_sha256`, and `output`. Both manifests and all source payloads are
SHA-checked. The terrain-grid/v1 destination and source grid must have identical
projected geometry, including the southern edge and north-first row order.
No DEM or executed-parent geometry is inferred.

The separate result schema is `arwen.nextgen.catchment-cell-overlap.v1`.
It pins both manifests, network JSON, complete ordered catchment IDs, data
terms, destination grid and every raw file's dtype/shape/bytes/SHA. It is not a
resident routing manifest or a registered numerical component. Existing
production `read_raw_into` can read its payloads into consumer-owned pinned
allocations; Python reads metadata only.

Destination C-order cell rows have `cell_catchment_offsets` i64[cells+1],
`cell_catchment_indices` i32[intersections] and
`cell_catchment_overlap_m2` f64[intersections], with ascending catchment indices
within rows. `cell_catchment_weights` f64 and `_f32` f32 are intersection area
divided by the FULL destination cell area, with no normalization over covered
area. If an actual producer declares each source depth spatially uniform,
the weighted sum gives full-cell mean depth and preserves the SUM of covered
source-polygon volumes. That arithmetic statement does not establish a
disjoint physical source partition: overlapping source polygons contribute
multiple times. Raw intersections are preserved, never clamped or normalized
to disguise overlap.

`cell_covered_area_m2`, `cell_coverage_fraction`, `catchment_covered_area_m2`
and `catchment_coverage_fraction` are f64. `cell_positive_overlap` i32 marks
positive summed intersection, NOT geometric union, DEM validity, a physical
active mask or a terrain-to-reach drainage map. Actual producer units,
epoch/cadence, overlap policy, initialization, surface/recharge budget and
drain receiving map remain numerical-owner inputs. This helper does not bind
the Noah parent factory, manufacture fluxes or choose those policies.

### Existing hydrofabric resource binding

The driver can consume the normal hydrofabric result with its existing
prepare-phase row:

```json
{
  "name": "fabric",
  "factory": "gpuwm.hydrofabric:prepare_resource",
  "phase": "prepare",
  "options": {"manifest": "ACTUAL_PREPARED_MANIFEST"}
}
```

This is one resource row, not a complete runoff/routing realization.
`prepare_resource` calls `load_cuda(manifest, device_id=...)`, returning
`PreparedHydrofabric.arrays`, `.manifest`, `.node_ids`, `.catchment_ids`,
`.level_offsets`, `.reverse_level_offsets`, and `.restart_identity`.
The existing MC factory is `gpuwm.hydro.troute_mc:prepare`; its
`network_component` can select `fabric`. It also needs the real producer's
`lateral_component` providing reach-aligned `routing_lateral` f32[N] in m3/s,
and `time_origin`. A fabric declaring `routing.initial_state_array` supplies
the default f32[N,3] (Q, velocity, depth) state. An explicit
`initial_state_component` overrides it. Only a fabric with no declaration
falls back to zeros, recorded as `cold_start` in restart identity. A missing
declared array or payload hash is an error. Static geometry alone does not
select an event's initial condition.

Driver-owned Rust binding can instead call
`manifest::PreparedManifest::open(path)`, inspect `.document.files`, and use
`read_field_into(name, &mut destination)` and
`read_auxiliary("network.json")`. The existing ABI1 header also exposes
`arwen_hydrofabric_manifest_read_into`. For other raw layouts, production
`RawPayloadReader.read_into(path, expected_sha256, address, nbytes)` reads and
checks bytes in Rust before a native copy into caller-owned pinned memory.
These APIs are already implemented; no second reader or factory is required.
The packaged raw bridge has no replay/native-executable dependency. See README
for registration, platform qualification, and destination lifetime details.

## Terrain receiving map is a separate physical input

### Products consume an explicit terrain and depth roster

The products owner accepts `nx`, `ny`, `latitude`, `longitude`, `elevation_m`,
`downstream_cell`, `channel_node`, and `stream_order`. Supply the same C-order
cell layout for every raster. The existing terrain-grid/v1 descriptors provide
coordinates, elevation and flat zero-based D8 downstream indices (`-1` at
pit/inactive/boundary); a matching hydrofabric manifest supplies rasterized
channel incidence and stream order. Compare the full projected geometry and
row orientation before combining these sources. Keep the terrain source terms
and the fabric source terms separately; a public-domain DEM does not license
the fabric or its derived node mapping.

The normal fabric's `channel_node` indexes its preserved flowpath roster. A
prepared collapsed routing replay's raster indexes its own `node_ids`, after
the explicit native waterbody membership mapping. Its `raw_node_to_routing_node`
records that transformation. These are distinct rosters, even on the same grid.
The raster/inverse CSR describes channel incidence, not an off-channel drain
receiving policy or a terrain initialization.

For HAND depth, products requires the actual above-bed channel-depth roster.
Reservoir water-surface elevations cannot fill missing above-bed depths. If
that roster excludes lakes or other IDs, the products/ensemble owner's explicit
ID join must mark those channel cells missing and stop their downstream paths
there. Keep that render-boundary mask separate from the terrain solver's D8
field and from the full routing graph; never change either source in place.
Use the owner-supplied i64 gather for differently ordered buffers. Internal
`node_ids`, publication `output_reach_ids`, and channel-depth IDs are not
interchangeable simply because some vectors have equal length.

Raw sources use explicit `<f4`, `<f8`, `<i4`, or `<i8`; NetCDF sources identify
path/variable and any leading-axis record. Python supplies metadata only;
Rust reads/gathers the payload. Dynamic terrain-depth maps need the real depth,
coordinates, grid and clock but no fabricated channel stage. Point-only flow
plots need the actual ordered output IDs and Q; missing terrain is not a reason
to invent a terrain/fabric crosswalk. Neither view establishes a physical
composed forecast by itself.

The clean-room terrain factories are `gpuwm.nextgen.terrain:prepare` and
`:prepare_lateral`. The latter consumes `routing_cell_offsets` i64[N+1] and
`routing_cell_indices` i32[M], with ascending flat C-order cell indices per
node. Preserve every node row, including reservoirs, and bind the node roster,
grid, receiving policy and source-budget identities. Every positive physical
drain must have a receiver or an explicitly budgeted external boundary.

The captured routing replay bundle's arrays with these names invert its
channel-incidence raster. Their names and types do not establish coverage of
off-channel terrain drains. Parent/catchment overlap weights also do not supply
this map. A physical terrain map needs actual drainage and receiving controls;
neither nearest-channel assignment nor catchment-average replication is an
established policy. No such map is currently emitted from the available source
geometry alone.

The producer consumes direct f64 interval surface/groundwater volumes in m3,
reset by terrain at the first update of each routing period. It accumulates
surface then groundwater and divides once by the actual period to publish
reach-aligned f32 m3/s. Do not derive an interval by subtracting cumulative
totals. The driver orders parent exchange, terrain, interval producer, any
owner-supplied CUDA composition of disjoint sources, then routing/reservoirs.
Source water sent through terrain must be excluded from direct runoff routing.
Physical reservoir receivers require the MC owner's optional
`reservoir_lateral_inflow` semantics (default false): transfer once through
current inflow with separate level-pool lateral zero. Older receiver semantics
cannot establish that transfer. These are producer/receiver requirements, not
new hydrofabric initialization or a qualified physical realization.

## Initialization still owned by the producers

| Missing input | Owner / required decision |
|---|---|
| Actual executed parent authority and matching overlap | Coupled-driver supplies the selected forecast ParentGeometry; hydrofabric prepares weights from the matching real polygons |
| Live parent Noah fields, flux units and receiver order | Driver, Noah coupling and forcing bind the existing GPU producer |
| CFE/LGAR/Noah/SFT physical parameters and initial reservoirs/soil state | Respective module/config/native-reference owners supply model-specific admitted inputs; preserved `model_attributes` JSON is not a typed initializer |
| Reservoir parameters, state, controls and collapsed topology | Reservoir/config owners supply actual tables and policy; hydrofabric then prepares the agreed mapping. Merely reading a `lakes` layer does not produce validated MC/reservoir hook selections |
| Terrain DEM, soil/aquifer/drainage parameters and initial head | Terrain and actual data providers; channel incidence/overlap cannot substitute |
| Disjoint runoff/terrain water accounting | Runoff, terrain and driver agree the physical receiving policy before binding transfers |
| Cross-dataset ID mapping | Authoritative producer supplies namespace/version/cardinality; matched original COMIDs alone do not map HYFeatures IDs |

No complete user-initialized domain is claimed merely because the exporter and
resident transport exist. Native-routing replay bundles remain replay resources.
The separate `arwen.source-netcdf-subset.v1` output is a lossless source audit,
not this resident schema or a model initializer. Fulldom/geogrid/rt% requests
belong to the withdrawn WRF-Hydro brief and are not a NextGen input contract.
