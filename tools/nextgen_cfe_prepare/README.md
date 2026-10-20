# CFE parameter preparation

`nextgen-cfe-prepare REQUEST.json` constructs the CUDA CFE layout from sealed,
explicit parameter rows in catchment order. It reads no native executable or
capture packet directly. Rust reads/verifies/initializes columns and writes/hashes buffers
with Rayon; ordered packing is startup work. Stepping remains entirely CUDA.

Build from this directory with `cargo build --release --locked --offline`.
Set `CARGO_TARGET_DIR`, `CARGO_HOME` and `TMPDIR` to the lane data-disk scratch.
The request schema is:

```json
{
  "schema": "arwen.cfe.prepare/v1",
  "time_origin": "2023-01-01T00:00:00Z",
  "columns": [
    {"id": "catchment-id", "config": {"path": "/absolute/parameters.ini", "sha256": "file-sha256"}}
  ],
  "output": "/absolute/new-output-directory"
}
```

Each INI uses the pinned CFE parameter names, with `forcing_file=BMI`.
`soil_storage` and `gw_storage` are explicit fractions (e.g. `0.667[m/m]`,
`0.5[m/m]`); percent strings reproduce a known upstream parser defect and are
rejected before unit annotations are removed. This includes `0.667[%]` and
`0.5[%]`, which must never be interpreted as fractions of 0.667 and 0.5.
Required soil, groundwater, partitioning and routing parameters are
supplied explicitly. Native documented optional defaults are retained. No
calibration or absent hydrofabric attributes are inferred. All unused state is
defined zero, matching the referee's declared allocation fix. Soil depth slots
for inactive root-zone coupling use the same padded buffer ABI as replay.

The output `manifest.json` has schema `cfe.initial.pack/v1`, LE f64/i32 buffers,
config seals, ordered catchment IDs, time origin, source pin and stage timings.
`gpuwm.hydro.cfe:prepare` consumes it through the shared Rust packet reader:

```json
{
  "name": "cfe",
  "factory": "gpuwm.hydro.cfe:prepare",
  "launch_method": "launch",
  "phase": "after",
  "options": {
    "manifest": "/absolute/prepared/manifest.json",
    "precipitation_component": "runoff_source",
    "precipitation_field": "water.qinsur",
    "precipitation_units": "m/s",
    "pet_component": "pet",
    "initial_step": 0,
    "time_step_fraction": 1.0,
    "identity": {"catchment_ids": ["catchment-id"], "time_origin": "2023-01-01T00:00:00Z"}
  }
}
```

The source component and field are table bindings. Canonical `RAINRATE` uses
`mm/s`; `mm/h` supports native BMI packets; `m/s` supports explicit water-flux
bindings. Native mm/h and m/s semantics require the pinned 3600-second period.
The m/s binding retains the measured native UDUNITS 2.2.28 coefficient
`3600000.0000000005`. Canonical mm/s normally multiplies by the prepared step
duration. An external mm/s source can explicitly select
`"precipitation_conversion": "ngen-udunits-2.2.28"` to reproduce the native
hourly BMI coefficient `3600.0000000000005`. This named compatibility choice
requires the 3600-second period and is recorded in checkpoint identity; it does
not alter the forcing provider or the physical duration. The native mm/h
binding performs no additional scaling. Each conversion precedes native
`/1000` and time-step-fraction multiplication inside the CFE kernel.
PET is a resident f64 `potential_et_m_s` vector. No numerical conversion or
readback occurs in Python. Conversion, BMI clock and ponded depth share the
single CFE kernel. The initialized 48-hour native Noah/PET/CFE composition at
driver `43f8` with PET v1 has an explicit strict proof: the replay owner's
independent five CFE/PET histories compare with zero of 720 stored words
different; the driver's broader six-field comparison has zero of 864 different.
This is historical evidence at those pins. Driver `15b52` with PET v2 has a
separate default-fast native-entry CFE BMI replay with zero of 426 words
different, while its full 48-hour trajectory still differs in 88 of 864 words.
PET here is a rate in m/s, matching
`water_potential_evaporation_flux` and CFE's internal
`et_struct.potential_et_m_per_s`. CFE BMI `POTENTIAL_ET` is a depth in m/step
and cannot replace that rate binding. These are the three shipped example
catchments, not a calibrated event or whole-network acceptance claim.

Initialization equations: NOAA-OWP/cfe
`f9182dfb3d81c407c66c5b782114173893c30cf2`, Apache-2.0. Complete upstream
LICENSE and NOAA TERMS are retained in
`gpuwm/core/kernels/NOTICE-CFE.txt` at the repository root.

## Physical parameter handoff

The hydrofabric/config provider supplies one sealed INI per ordered catchment
in the existing request above. Catchment geometry, connectivity and forcing
alone cannot construct this packet. Values must already use the pinned CFE
native units: INI `[unit]` annotations are stripped, not converted. Aliases in
each table cell are listed in lookup order; supply one spelling per parameter.

| Required parameter | Accepted key(s) |
|---|---|
| Resident forcing | `forcing_file=BMI` |
| Soil depth | `soil_params.D`, `soil_params.depth` |
| Soil exponent | `soil_params.bb`, `soil_params.b` |
| Saturated conductivity | `soil_params.satdk` |
| Saturated capillary head | `soil_params.satpsi` |
| Drainage slope | `soil_params.slop`, `soil_params.slope` |
| Saturated soil water content | `soil_params.smcmax`, `soil_params.maxsmc` |
| Wilting soil water content | `soil_params.wltsmc` |
| Field-capacity parameter | `soil_params.alpha_fc`, `alpha_fc` |
| Initial soil storage fraction | `soil_storage` |
| Lateral soil reservoir coefficient | `K_lf` |
| Maximum groundwater storage | `max_gw_storage` |
| Initial groundwater storage fraction | `gw_storage` |
| Groundwater reservoir coefficient/exponent | `Cgw`, `expon` (both required) |
| Subsurface Nash coefficient | `K_nash_subsurface`, `K_nash` |
| Infiltration partition | `surface_water_partitioning_scheme`, `surface_partitioning_scheme`: `Schaake`/`1` or `Xinanjiang`/`2` |

Additional inputs depend on the selected formulation:

| Selection | Required inputs |
|---|---|
| `Schaake` | No additional required parameter unless SFT is enabled |
| `Xinanjiang` | `a_Xinanjiang_inflection_point_parameter`, `b_Xinanjiang_shape_parameter`, `x_Xinanjiang_shape_parameter`, `urban_decimal_fraction` |
| `surface_runoff_scheme=GIUH`/`1` (default) | Nonempty `giuh_ordinates` vector |
| `surface_runoff_scheme=NASH_CASCADE`/`2` | Positive `N_nash_surface`, matching `nash_storage_surface` vector, `K_nash_surface` |
| `is_sft_coupled=true` with Schaake | `ice_content_threshold`; also requires an agreed resident ice-input binding |
| `is_aet_rootzone=true` | Increasing `soil_layer_depths` ending exactly at soil depth, zero-based `max_rootzone_layer` within that vector; also requires an agreed resident moisture-profile binding |

Initializer defaults are `soil_params.refkdt`/`refkdt=3`,
`soil_params.expon=1`, `soil_params.expon_secondary=1`,
`N_nash_subsurface=2`, and both coupling flags false. An explicit
`nash_storage_subsurface` vector with at least two entries determines its own
count; otherwise initial subsurface Nash storage is zero at the declared count.
Surface Nash defaults are `nsubsteps_nash_surface=10`,
`Kinf_nash_surface=0.001`, and `retention_depth_nash_surface=0.001`.
The pinned native surface Nash initialization uses supplied storage only for
counts greater than two; smaller accepted counts initialize to zero.
These preserve native initialization behavior, rather than infer calibration.
The packet period is 3600 seconds. The current prepared factory binds rainfall
and PET only; enabling SFT or root-zone flags alone does not connect another
component's resident ice or moisture arrays.

The provider must identify parameter provenance, permitted data terms,
catchment roster/order and initial-state time. An event's soil/GW storage and
routing state require an explicit initialization policy. The three shipped
example catchments use a declared common-parameter cold start; copying that
packet over a different network does not establish physical event inputs.
The config/driver also owns the choice of runoff source and its disjoint
partition with terrain routing. This handoff adds no provider, schema or
per-step host processing. Optional formulation inputs listed here describe the
initializer contract; they do not extend the measured replay coverage above.

### Spin-up state is more than the INI fractions

The world-init producer at `16012b0158aef5d683aa636639172f85b7f5cacc`
writes `cfe_soil_storage` and `cfe_gw_storage` fractions, as well as
`cfe_nash_storage_subsurface`, `cfe_giuh_queue` and `cfe_flux_perc_carry`.
Its `cfe-initial.json` handoff at that revision contains only the two fractions. Joining that
JSON into the INIs initializes reservoir amounts, but does not restore the
complete spun-up CFE state. The native INI initialization can initialize Nash
stores through an explicit `nash_storage_subsurface` join, but starts the GIUH
queue and carried `flux_perc_m` at zero. The raw-state extension below supplies
these without changing the native fractional path.
The next CFE call reads carried percolation before assigning its new value.

Fractions also need not reconstruct the original f64 reservoir word. The
private `audit_initial_handoff.rs` tool checks the original seals, reads native
CFE packets in Rust, and tests the producer's `storage / storage_max` followed
by this initializer's multiplication order. On native example updates 35 and
36, three of 24 reservoir words differ by one ULP across four inspected
boundaries. Update 35 exit and 36 entry repeat the same stores: there are two
differences among 18 words at distinct boundaries. All differences are soil
storage; groundwater reconstructs exactly on these inputs. Every boundary
also has nonzero Nash, queue and percolation state. This is a component
handoff witness, not execution of world-init or a physical event.

The next joint producer/consumer handoff must bind raw f64 reservoir amounts,
true per-catchment ragged counts, Nash stores, GIUH queues and carried
percolation to the sealed catchment IDs, fabric, parameters and valid epoch.
The producer's queue omits the native extra sentinel; its count and the
sentinel initialization must be explicit. A physical initialization must name
its diagnostic/budget baseline, including held Nash and queue water. Exact
restart additionally preserves every carrier and clock through the existing
resident checkpoint inventory. A fractions-only packet is not that restart.
The producer still owns that ID/fabric/parameter/epoch-bound handoff. The
consumer extension does not add a second world-init writer or claim that the
current producer has supplied the physical packet.

### Raw water-state initialization

The same sealed per-catchment INI may supply all five keys below. They are
extensions of this Rust preparer, not keys accepted by native CFE. When supplied,
they replace the corresponding initialized water state by direct f64 parsing;
there is no opt-in runtime flag or fraction reconstruction.

| Key | Value |
|---|---|
| `initial_soil_storage_m` | Raw soil reservoir amount, metres |
| `initial_gw_storage_m` | Raw groundwater reservoir amount, metres |
| `initial_nash_subsurface_m` | Comma-separated raw Nash stores, metres |
| `initial_giuh_queue_m` | Comma-separated raw queue stores, metres; excludes the extra native sentinel |
| `initial_flux_perc_m` | Carried prior percolation flux, native metres per step |

Use round-trip f64 decimal values (17 significant digits suffice). Ordinary
parameter keys, including `soil_storage` and `gw_storage`, remain required to
construct the native parameter slots. The raw amounts replace live stores,
not those retained parameter fractions. Nash count must match the prepared
`N_nash_subsurface`; queue count must match `giuh_ordinates`. Rust appends the
native extra +0 sentinel. Partial state is refused because it would discard
carried water; nonfinite/negative water, reservoir overcapacity, mismatched
counts and overflowing total inventory are refused with a named error.

This extension covers GIUH without SFT or root-zone coupling. Surface Nash,
frozen-water and root-zone formulations require additional state absent from
these five keys; use the complete resident checkpoint for those formulations.
Without any raw keys the ten legacy output buffers remain byte-identical.

`initial_water_inventory` records ordered per-catchment soil/GW, held Nash and
queue water, their total, true counts, the sentinel and percolation carry.
Percolation is a prior flux, not an additional water stock. Native fresh
diagnostics and the clock start at zero. Their `volstart` remains soil plus
groundwater because the native global budget counts lateral output before
Nash; initial Nash and surface offsets are therefore recorded separately.
Those offsets must be included by a complete physical budget; the metadata
does not itself provide a new runtime budget diagnostic or an exact restart.
Exact continuation uses the existing full carrier/clock checkpoint path.

The CLI receipt test in `src/tests.rs` uses four retained sealed native
boundaries (BMI35/36 entry/exit), checks all 132 selected raw state words,
including both one-ULP fraction witnesses, and compares all ten legacy typed
buffers against the previous CLI. Run it with `--include-ignored` and explicit
`CFE_RAW_FIXTURES`, `CFE_RAW_EXE`, `CFE_RAW_BEFORE_EXE` and scratch `TMPDIR`;
its ignore marker names these external receipt prerequisites. A separate
unit test covers 16 malformed/incomplete/unsupported configurations.

Build the audit with `rustc --edition=2021 -O`, linking the existing pinned
`nextgen_replay` and `serde_json` rlibs, then pass one or more packed native
fixture directories (each has `manifest.json`, `compare.json`, `input/` and
`expected/`). Build the same file with `--test` for its checksum-negative
control; set `CFE_AUDIT_FIXTURE` to an original fixture and `TMPDIR` to lane
scratch. It makes no GPU calls, runs no model steps and writes no data into
the original fixture. Keep the executable and all build outputs in scratch.

## Composition with reach routing

The row above uses the existing driver factory signature
`prepare(*, settings, options, state, physics, device_id, mode, components)`.
The default `launch_method="advance"` also works: CFE aliases `advance` to
`launch(dt_seconds, *, stream=None)`. Prepare the precipitation and PET
providers first, then CFE; select process numerics before any kernel compiles.
The factory retains the producer vectors and seals the state inventory before
stepping. Restore every producer and CFE carrier in place on the consumer
stream, preserving aliases and the original identity.

The Rust initializer currently prepares a 3600-second CFE period, and CFE
rejects any other launch duration. The measured routing-only Lower Colorado
case uses 300-second steps. Binding both as ordinary stages with the same
300-second interval is therefore invalid. The resident driver's prepared
`after_schedule` can run CFE and its transfer once per 3600-second macrostep,
then the router twelve times at 300 seconds, then history once per macrostep.
That schedule is a launch contract; a connected native producer/receiver trace
must separately establish the correct first and subsequent publication phase.

CFE's `fields[10]` (`flux_Qout_m`) is runoff depth in metres per CFE step.
`gpuwm.hydro.runoff:prepare` binds that resident vector to explicit area and
network providers and supplies MC's `lateral_component` in m3/s:

```json
{
  "name": "runoff_to_route",
  "factory": "gpuwm.hydro.runoff:prepare",
  "options": {
    "source_component": "cfe",
    "depth_field": "flux_Qout_m",
    "source_catchments_path": ["preparation", "catchments"],
    "network_component": "fabric",
    "area_component": "physical_cfe_areas",
    "area_array": "areasqkm",
    "area_units": "km2"
  }
}
```

The area provider exposes `catchment_ids`, `arrays` and `restart_identity`;
the fabric is the existing prepared network. Source, area and fabric IDs must
agree in order, and their extents must match. Choose `m2` or `km2` explicitly;
there is no inference that polygon area equals native CFE `areasqkm`.
An optional `source_partition` metadata object is retained in identity. It
records the caller's policy, not a proof of disjoint water sources.

GPU setup creates a deterministic reverse map. Each launch gathers ascending
catchment indices and divides each catchment's volume by the producer period
before aggregation, retaining native `(Q_OUT*(areasqkm*1000000))/3600` order.
The outputs are contiguous f32 `routing_lateral` for MC and f64
`routing_discharge_m3_s`/`routing_volume_m3` for diagnostics. No atomics, host
numerics, readbacks or allocations occur in a transfer step. A duration change
is refused rather than silently reusing the prepared conversion. Invalid depth,
area, CSR or overflow leaves sticky diagnostics for driver health admission.
All persistent arrays, policy, provider identities and units join checkpoints.

This transfer does not establish a physical runoff/routing composition. Actual
same-network parameters/stores, a connected native clock/publication witness
and a disjoint replacement budget are still required. Those physical inputs
are absent from the routing-only capture; the three example catchments cannot
stand in for the network's catchment parameter and initial-state rows. Mapping
an overlapping polygon area sum does not establish conservative source closure.

MC owns `reach_output()` with ordered contiguous same-card int64 `reach_id`
and native-real `discharge_m3_s`; optional velocity, depth and nudging fields
follow that same order. Prepare `gpuwm.hydro.output:prepare_history` after MC,
with its `source` option naming MC, so the original driver inventory contains
history arrays and their resident cursor. CFE does not claim `reach_output()`.
Reservoir resources advance through the router's level hooks, not an additional
independent stage. These are existing owner interfaces, not adapter files or
Python array processing.

## Sealed hourly publication and serialized transfer continuation

The consumer `publication_fixture.rs` extracts, without recomputing expected
Q, native CFE depth and original binary nexus discharge from the existing
referee capture. Build it with `rustc --edition=2021 -O`, with both output and
`TMPDIR` under lane scratch, then supply framework root, nexus root, sealed
roster TSV and a new fixture directory. The orchestration seals all source
artifacts against the referee index before extraction and seals the packed
files and manifest. The roster supplies explicit native km2 areas and ordered
catchment/nexus IDs; it is not a new hydrofabric or a source of global areas.

`publication_cuda.py STAGED_ROOT` consumes those sealed files through the
shared Rust `RawCuda` bridge. The stage includes `source-identity.json` (source
hashes, artifact hashes and candidate revision), the sealed `fixture/`, native
index, `nextgen-replay`, `libnextgen_replay.so`, and the real Rust output library
selected by `GPUWM_NEXTGEN_OUTPUT_LIBRARY`. Set `PYTHONPATH=STAGED_ROOT/source`,
process numerics and all cache/output paths before invoking it under the GPU
pool/common lock. No Python numerical unpacking, packing or comparison occurs.

At b357326da41445c7f3dc14f6beed133676432051, the existing 48-hour, three-catchment
native capture produced zero differing words out of 144 f64 discharge values
in both fast and strict on the 5070 Ti. A checkpoint after update 24 went
through the actual Rust writer and reader (CDF5 through netCDF-C); all nine
carriers, pointer addresses, identity and clocks survived restore in place.
The resumed 24-hour discharge history and complete final inventory matched
continuous replay, zero differing words out of 100 per mode (28 carrier words
and 72 discharge words). This closes the former in-memory-only transfer test.
It does not establish 300-second receiver sampling, physical initial stores,
source partition closure, or the default-on forecast's total cost. Those
remain separate physical preparation and connected-clock witnesses.

The HYDRO-WORLD contract at 3c8f1ecda3d834a0993d198cd75d4063f76057ae preserves
the existing prepared-fabric runtime interface. Native and global parameter
providers retain their own ordered IDs and explicit area convention; this
transfer does not rename polygon area as native CFE area. Global fabric,
parameter derivation and reanalysis initialization remain their owners' work.

## Resident CFE producer and transfer composition

`composition_cuda.py STAGED_ROOT` binds the actual CFE and runoff factories
to the existing native BMI entry packet at update 35. The captured forcing,
PET, clock and ponded-depth carriers retain their original types and units.
An explicit identity path selects the audited framework catchment IDs from
the captured packet. Areas and distinct nexus IDs come from the original
hourly publication fixture. Rust selects the original three expected nexus
words by byte offset; it does not recompute discharge.

The real resident driver advances CFE followed by transfer on the same
non-default stream, with two ordinary launches and 128 threads per block.
The transfer depth aliases CFE's live `flux_Qout_m`; pointer checks reject an
intermediate replacement. All health checks pass, all 25 inventoried carriers
(3,272 logical bytes, including the alias) retain their pointers, and the
driver clock reaches one 3,600-second update. Setup and readbacks are outside
the launch window. Payload I/O and comparison use Rust; Python orchestrates.

At a6584b10c5003a564c9fe6c52c7bef428e2f040f, the 5070 Ti produced zero differing
words out of 429 in both fast and strict: 426 CFE carrier words and three
original native nexus discharge words. CFE uses 254 registers and transfer
43; both report zero local and shared bytes. Endpoint CuPy used/reserved
memory is 15,360/18,944 bytes under a 64 MiB cap. The host launch-and-sync
diagnostics are 0.038193/0.047558 ms; these are not whole-step cost or speed
claims. Each short pool/common-exclusive lease starts and ends with no
compute processes. The initial resource-reporting attempt passed comparison
but failed to read the low-level Function attributes; a forward reporting
fix and repeated fast run supplied the complete receipt.

The staged identity seals every source and input. Receipts retain transfer
compiler image hashes and actual options. CFE's spliced source, header bytes
and compiler implementation/options are sealed; its compiled image is not
separately retained. This is a connected software proof with captured native
upstream inputs, not a live parent, 300-second routing sampler or physical
event. Same-network initialization/parameters/area convention, first and
next hourly publication phase, disjoint source-volume closure, per-member
physical continuation and the forecast's total hydro cost remain open.

Combined fresh-driver continuation

`composition_restart_cuda.py` reuses `prepare_composition` from the native
consumer. It prepares uninterrupted and fresh member-0 drivers and an
independent member-1 driver through the original member capture scope. A real
Rust CDF5 checkpoint after the first update restores into the fresh driver;
its next update matches uninterrupted execution. Member 1 refuses member 0's
checkpoint and remains unchanged. All 25 carriers are compared at each of
three boundaries (75 fields, 1,362 words), including the strided CFE depth
alias, empty carriers and both component clocks. Cross-driver allocation
extents are disjoint; all original pointers and the depth alias survive.

At 82de0c71aa93dbd5785ba5b33ac4a38b3f9fd43c, fast and strict each have zero
differing words out of 1,362 on the 5070 Ti. A Rust negative control comparing
the saved boundary to the unadvanced peer finds 105/454 differing words in
each mode (expected comparator exit 1); the checkpoint is not a trivial copy
of initial state. Full driver identities differ only by member ID. There
are two ordinary model launches per update, six total in this check. CFE
uses 254 registers and transfer 43, with zero local/shared bytes. Endpoint
CuPy used/reserved is 84,480/86,016 bytes under a 64 MiB allocator cap.

Snapshot plus Rust write takes 18.060737/17.254618 ms (fast/strict); fresh
read/restore takes 0.530572/0.789993 ms. These are host boundary diagnostics,
not canonical step-cost or speed claims. Both short pool/common-exclusive
leases have empty compute-app lists before and after. The packet seals
2,450 source/input/artifact hashes, including the actual Rust libraries.
The Rust output library was built offline/locked with 16 Cargo jobs in
14 seconds. Source/notices, compiler and dependency pins accompany the
receipts; no Fortran or withdrawn WRF-Hydro source/fixture is used.

The second update holds captured precipitation and PET fixed. Its verdict
is explicitly `tool-self-test`, not a second native update or a physical
ensemble proof. The standing native one-update 0/429 result is unchanged.
A live parent, same-network physical parameters/initial state/areas,
first-and-next publication phase, disjoint source-volume closure, physical
member/event products and full coupled hydro cost remain unqualified.

Consecutive native updates through a fresh checkpoint

`composition_native_restart_cuda.py` narrows that fixed-forcing limitation.
The existing Rust capture packer selects consecutive native BMI updates 35
and 36 from the sealed shipped Noah/PET/CFE capture. After update 35, the
actual Rust CDF5 checkpoint restores into a fresh member-0 driver. Only the
next captured precipitation and PET vectors are copied into the uninterrupted
and fresh drivers; their internal state, ponded depth and clocks continue.
Native expected discharge is selected from the original hourly nexus file
by Rust byte-copy, without recomputing it. Python handles metadata and
orchestration; Rust compares payloads and CUDA performs model arithmetic.

Fast and strict on the 5070 Ti each compare zero differing words out of
1,769: 454 at the restored boundary, 454 after the next update, two sets of
429 native CFE/nexus words at update 36, and three original nexus words at
update 35. A separate driver deliberately retaining update-35 forcing
differs in 96/429 words against update 36 in each mode. The changed native
inputs therefore exercise a meaningful continuation, including output
differences beyond the forcing vectors. All pointers and the strided depth
alias survive; all three drivers report healthy component status.

There are ten ordinary model launches in this check, two per update, with
128 threads per block. CFE/transfer use 254/43 registers and zero local/shared
bytes. Endpoint CuPy used/reserved is 43,008/46,592 bytes under a 64 MiB cap.
Snapshot/write takes 11.130237/12.843153 ms and fresh read/restore takes
0.550464/0.776593 ms (fast/strict). These are host diagnostics, not canonical
step costs. Short hydro pool/common-exclusive leases have empty compute-app
lists before and after; no CUDA graphs are used. The borrowed private output
ELF is SHA-pinned and exercised by the actual CDF5 roundtrip; this receipt
does not claim a fresh output-library build.

The three captured example catchments and original distinct nexuses are
native component evidence, not a physical routed event. The separate
synthetic connected routing witness does not establish native 300-second
receiver phase. Physical parameters/stores/areas, disjoint source-volume
closure, live parent/member products, event skill and total coupled hydro
cost remain open. Earlier member-isolation and fixed-forcing proofs stand.
