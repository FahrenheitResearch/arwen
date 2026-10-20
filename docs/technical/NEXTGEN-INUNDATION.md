# Resident terrain inundation diagnostic

`gpuwm.nextgen.inundation.InundationDepth` is original Apache-2.0 code. It
observes the clean-room terrain component's surface water depth in metres; it
does not implement routing equations or use a UCAR reference program.

Prepare it once on the terrain card and parent stream:

```python
diagnostic = InundationDepth(
    surface_depth=terrain.arrays["surface_depth"],
    active=terrain.arrays["active"],
    source_health=terrain.arrays["health"],
    source_identity=terrain.restart_identity,
    period_seconds=period_seconds,
    stream=parent_stream,
)
```

Inputs are stable same-card C-contiguous arrays: float32 `(ny,nx)` depth,
int32 `(ny,nx)` active mask, and uint32 `(1,)` source health. The terrain
component owns and checkpoints them. Prepare the diagnostic before driver
checkpoint inventory is sealed. Its `.arrays` contains float32 `(ny,nx)`
`depth_m`, `maximum_depth_m`, and uint32 `(1,)` `health`; these are all its
cross-step carriers. `.restart_identity` records the source identity, numeric
mode, grid ordering, cadence, units and boundary limitation.

The coupled driver's published assembly ABI can select
`factory="gpuwm.nextgen.inundation:prepare"`, `phase="after"`, with
`options={"terrain_component": "the-prior-terrain-stage-name"}`. This explicit
table link binds that component's named arrays and prepared period on the
parent card/current preparation stream. Order this row immediately after its
terrain stage. The realization/config owner chooses the stage names and rows.

Construction enqueues an observation of the initial terrain state. Place
`advance(dt_seconds, stream=parent_stream)` after terrain advancement at the
prepared cadence. `step` is the same callable. Each call launches one grid,
256 threads per block and one thread per cell, with no allocation, field
readback, file I/O, event or synchronization. Both setup and stepping must
follow the terrain's stream ordering. `maximum_depth_m` is the maximum over
the initial observation and explicit diagnostic calls at the prepared cadence.
Intermediate terrain substeps are not sampled automatically, so peaks between
observations can be missed. Label products as a sampled maximum and name the
sampling cadence; this is not a continuous event maximum.

An active valid depth is copied without conversion. Inactive cells have
canonical float32 NaN
(`0x7fc00000`) in both outputs. Negative or nonfinite active depth produces
NaN and persistent health bit 2; an invalid mask produces NaN and bit 4; any
source health produces NaN everywhere and bit 1. Any nonzero diagnostic health
invalidates the result even if subsequent samples become finite. Inspect
health only at an output/checkpoint boundary.

The products owner's `terrain_depth_m` raw frame consumes `depth_m`. Export
through Rust at an output boundary, preserving C ordering and metres; use the
shared `tools/nextgen_replay/raw_cuda.py` bridge for replay artifacts. Production
history snapshots remain device-resident until their export boundary. Maps
must go through `rw_wrfbatch` via `gpuwm.rustwx` and carry the terrain source
and boundary limitations: sealed exterior, one-way stage drains, and no
receiving-channel backwater flooding. This diagnostic cannot add missing
channel-to-terrain inundation physics.

## Executed diagnostic check

`tools/nextgen_replay/examples/inundation_fixture.rs` writes small sealed raw
inputs and independent exact expected words. `check_inundation.py` executes
the CUDA diagnostic, exports through Rust pinned staging, and invokes the
shared exact comparator. It checks copied depth, initial-state maximum,
inactive cells and source/depth/mask failures. Fixtures are explicitly
`tool-self-test`, not native or observed routing evidence.

```sh
# From tools/nextgen_replay; scratch must be on the data disk.
cargo run --release --offline --locked --example inundation_fixture -- "$scratch/new-fixture"
GPUWM_NUMERICS=fast RAYON_NUM_THREADS=8 python check_inundation.py \
  --fixture "$scratch/new-fixture/fixture.json" \
  --library "$CARGO_TARGET_DIR/release/libnextgen_replay.so" \
  --comparator "$CARGO_TARGET_DIR/release/nextgen-replay" \
  --out "$scratch/new-check" --revision ACTUAL_CANDIDATE_COMMIT
```

The measured candidate `7c69b0b689097b813718b80c257ae65c91e70c9e`
on an RTX 4090 in the shared pool, fast numerics, matched **0 / 46 differing
stored words** across nine fields. The separate synthetic 1024×1024 diagnostic
timing was 2.669834 ms for 64 direct calls, **0.041716 ms/call**, with two
batch-boundary synchronizations and no per-call events/sync. A separately
instrumented single call was 0.023552 ms, including submission gaps. Setup's
initial observation was outside timing; no CUDA graphs were used. These are
diagnostic timings, not the Lower Colorado terrain or whole-hydro speed line.
Later changes only add the assembly `advance` alias and numeric-mode identity;
the assembly factory binds the same constructor to the driver's actual prior
component mapping. The measured CUDA kernel, launch arguments, grid and step
are unchanged. A real assembled physical run remains pending.

Observed-event USGS/NWM comparisons and an inspected physical inundation image
remain required. They depend on the referee's event/fabric selection, the
explicit fabric-to-NWM crosswalk, and actual terrain DEM/parameters/forcing.

## Actual-grid engineering replay

`examples/terrain_depth_fixture.rs INPUT.json NEW_DIRECTORY` validates sealed
initial/final depths, active mask, owner health receipt and public DEM manifest.
Rust independently computes masked final depth and the two-observation maximum
across cells with Rayon. It refuses invalid active depths or a failed source
receipt. This preparer does not implement terrain routing or use a native
hydrology reference. Source health is reconstructed from the sealed owner's
receipt; it is explicitly distinguished from a live terrain health carrier.

`check_engineering_depth.py` consumes that fixture through Rust pinned staging,
executes the published `prepare` factory using the prior captured component,
observes initial and final depth, and checks the output words with the shared
comparator. It also times repeated final-state observations. Its full command,
fixture/provenance seals, restart identity, and timings are written to the new
output directory. No new physical terrain advances are inferred from those
repeated observations.

Measured candidate `52ce4b1e9a0` used the terrain owner's imposed-water
experiment on the public USGS 3DEP rectangle, **737×582**, 1 km EPSG:5070,
north-first, and sealed terrain candidate `a415ecaa206` initial/final frames
(final after 2400 elapsed seconds). The executed factory/advance check matched
**0 / 857,872 differing stored words**: 428,934 depth values, 428,934 maxima,
and four health bytes. Fast RTX4090, shared pool; 64 direct observations took
2.496353 ms, **0.039006 ms/call**, with 1,676 blocks of 256 threads. Setup was
25.861890 ms. Rust Rayon8 expectation preparation was 6.731990 ms and exact
comparison 8.291753 ms. No per-call events/sync or CUDA graphs were used.

This is real-grid diagnostic evidence with imposed inputs. No actual flood
forcing, basin/channel mapping, observed skill, or whole-hydro speed is claimed.
The current referee event is Russian River, January 5–17 2023, superseding
July Winooski and fitting the NWM v3 retrospective coverage. Actual model
histories and an explicit gauge/fabric/reach mapping remain required. Source
dataset terms must be checked independently of the software licence: the
current v2.2 Lynker bucket notice is CC BY-NC-SA4.0, and restricted fabric data
must not be included in a shipped demo/default download. The public DEM and
original engineering depth frames used in this check do not include that fabric.

The initial and elapsed-2400s engineering frames were subsequently rendered by
the products lane's depth-only path through `gpuwm.rustwx` and `rw_wrfbatch`.
Both maps were inspected: the uniform initial water becomes terrain-dependent
drainage/depression patterns, with no visible blank or flipped field. Captions
state imposed water, sealed edges and no channel backwater; both use a 0–0.06 m
scale. Raw renderer exports matched **0 / 857,868 differing float32 words**
against the sealed initial/final depth frames. Rendering took 546.917089 ms
with 24 Rayon workers, outside simulation stepping. The receipt preserves the
renderer binary fingerprint and post-render working-tree source snapshots;
the products lane's depth-only changes were not yet committed at rendering.
This is an inspected engineering picture, not the pending physical event run.

## Live terrain and diagnostic contract check

`tools/nextgen_replay/check_live_terrain.py` executes both published factories
on the sealed engineering initial state and public DEM. It binds the actual
terrain health array and advances terrain plus diagnostic on one stream for
eight 300-second periods, with preallocated device history snapshots. Rust
verifies the last history frame against the independent terrain-owner final
export and computes maxima over initial water plus all eight observations.
The preparer's optional `history={frames,depth:{path,sha256}}` input describes
packed float32 C-order `(frames,ny,nx)` bytes, validated before use. It rejects
bad sizes, an unequal final frame, and invalid active depths.

Candidate `1a29b359b209acbba8acdd64b8b63f08e168df77`, terrain
`2e8f1893780dce8473baf6dcc7b91e313d894e1d`, matched **0 / 857,876 differing
stored words** across final depth, nine-observation maximum, diagnostic health
and live terrain health. Both sets of resident pointers stayed fixed. Wrapper
and CUDA source hashes matched their committed worktrees. On the shared RTX
4090, fast numerics, the eight-period batch took 79.489488 ms, **9.936186 ms
per period** including one diagnostic and one device snapshot per period.
No terrain-state warmup is discarded: timing includes the first advance from
the sealed initial state and any remaining first-launch overhead.
Each period enqueues 60 terrain grids of 3,352×128 threads, one diagnostic grid
of 1,676×256 threads and one history-copy grid with 256 threads per block.
There are no per-period allocations, events, synchronizations or field reads,
and no CUDA graphs. Setup was 236.432660 ms, Rust Rayon8 history expectation
preparation 21.061175 ms, and exact comparison 10.798175 ms.

This extends the captured-frame check with live factory/stream/health/history
execution. It remains the imposed-water engineering setup: observed-event
skill, full driver/history assembly and whole-hydro speed are still pending.
Its final depth is byte-identical to the frame in the inspected engineering
map above, so that map also depicts this live run's final depth. The renderer
source and binary fingerprints subsequently matched the products owner's
committed `bb097bd9eb3576b1ebf2e68d79b1bcaaaefe378f` worktree.

## Observed-event reference available; terrain history pending

The verification owner's Russian River NWM v3 versus USGS comparison for
January 5–17 2023 is available and its three hydrographs were inspected.
Each gauge has 289 exact hourly pairs, with no interpolation; the 864
intervening observation samples per gauge do not enter the scores.

| USGS gauge | NWM NSE | NWM KGE (2009) | Global peak offset |
|---|---:|---:|---:|
| 11462500, Hopland | -0.214984 | 0.597035 | +102 h |
| 11464000, Healdsburg | 0.053922 | 0.607973 | +7 h |
| 11467000, Guerneville | 0.059691 | 0.577822 | -9 h |

Positive offset means NWM's global peak is later on the common support.
Hopland's maxima select different event pulses; +102 h is not a measured
travel time or phase lag. Visually, several NWM pulses are higher and sharper
than observed, and the Guerneville observed crest is broader. These are
archive-versus-observation results from the verification owner, not new
terrain scores or an identified cause of model error. The owner's comparison
receipt names source revision `170e3a9770503cca2532796fde099b35ef25eb17`.

Actual terrain event histories remain absent. Physical input/state providers,
their units and cadence, gauge-to-terrain/reach mapping, initial storage and
reservoir treatment must be supplied by their owners before this lane can
use the existing exact-UTC scorer on terrain output. Public NHDPlus COMIDs
used to select the archive series do not establish that terrain mapping.
No new archive fetch, scoring implementation or guessed coupling ABI is
needed here. The inspected analysis charts and sealed owner receipts are
retained separately from the imposed-water engineering maps.

## Nonzero terrain transfer carriers

`examples/terrain_transfer_fixture.rs` in the shared replay crate generates
three closed-form, one-cell cases. `check_terrain_transfers.py` binds the
published terrain factory with caller-owned float64 cumulative volume arrays
seeded to 0.125 m³. Each case advances once, then an immediate same-stream CUDA
consumer copies all three volumes to preallocated receiving buffers before
any host synchronization. Rust checks both producer and receiver exports.

All cases have dx=2 m, dt=period=1 s, soil depth=1 m, Manning=1, zero external
sources, stage=0 and active=1. Surface drainage starts with h=1, w=0, z=0,
yield=1, K=0 and outlet width=1: the drain rate is 1/4 m/s, giving h=3/4 and
surface volume=1.125 m³. Groundwater drainage starts with h=0, w=1/2, z=1,
yield=1, K=1 and width=1: the rate is 1/16 m/s, giving w=7/16 and ground
volume=0.375 m³. Exfiltration starts with h=0, w=3/4, z=0, yield=1/2, K=0
and width=0: spill=1/4 m gives h=1/4, w=1/2 and exfiltration volume=1.125 m³.
The other volumes retain 0.125 m³. These arithmetic choices are binary exact;
no native math-library witness or tolerance is used.

Verification candidate `c85ffe58417856f38ce9efbb3a43f00fd1c2a696`, terrain
`4d9fd293aaa15e893533bc6ed05dc3d9d432e47a`, passes **0/36 differing words in
default fast and 0/36 in explicit strict**. Each verdict covers h/w, all three
cumulative volumes, their immediate consumer copies and four health bytes
per case. The factory preserves the supplied cumulative arrays; terrain,
scratch and receiving-buffer pointers remain fixed. Measured staged sources
match the named committed source snapshots, even though the owner worktree
subsequently advanced.

Shared RTX4090 completed three-case batches were 0.090164 ms fast and 0.069486
ms strict; setup was 92.404671/166.939592 ms. Each batch has six terrain and
three consumer launches, each one block of 128 threads, with no warmup
advances, per-advance allocations/events/synchronizations/readbacks or graphs.
Rust comparison took 2.188785/2.125691 ms with eight Rayon workers, offline.
Fixture generation took 2.971737 ms in one Rust process; build took 0.42 s.
These tiny carrier checks establish neither a physical event nor routing-step
speed acceptance.

The same fixture preparer accepts a second argument `intervals` to generate
schema `nextgen.terrain-transfer.fixture/v2`. Verification candidate
`f93cb5d3af5a618c413135243f95af494a328ada` with committed terrain
`48d1ca4f2066b9a10ae37b779db5218a0003a888` passes **0/145 differing words in
default fast and 0/145 in explicit strict**. This exercises two consecutive
advances per case, both direct float64 interval outputs, five immediate volume
consumer copies and the published `prepare_lateral` factory. Three-row CSR
`offsets=[0,0,1,1], indices=[0]` retains the two empty rows; the isolated lateral
contribution remains unbound to channels or reservoirs.

Here the surface case starts at h=3/4 with source=1/4 m/s, so both advances
drain exactly 1 m³ and finish at h=3/4. Its cumulative buffers start at 2^60 m³;
adding 1 m³ cannot change that total, while the direct interval and lateral
must each remain 1. Interval buffers start at 32 m³ to detect a missing reset.
Groundwater's second advance leaves w=399/1024 m, cumulative ground
volume=145/256 m³ and interval=49/256 m³. Exfiltration's second advance keeps
the first spill's cumulative total and produces zero interval drainage.

A separate sealed three-cell producer witness uses surface volumes
`[2^-53,2^-53,2^-53]` and ground volumes `[1+2^-24,0,0]`, with ascending CSR
indices. Ordered interleaved float64 addition rounds the output to float32
1; grouping all surface terms first would round to the next float32. This
checks the specified summation order using nonnegative volumes.

The shared RTX4090 batches took 0.343192 ms fast and 0.370923 ms strict,
including 12 terrain launches, seven lateral launches, six consumer launches
(all one block of 128 threads) and 60 preallocated device history copies.
Setup took 266.333090/202.915959 ms; eight-worker Rust comparisons took
0.992458/1.889759 ms, offline. Fixture generation took 4.414026 ms in one Rust
process; build took 0.49 s. No warmup advances, per-advance CUDA allocations,
events, host synchronization or field readbacks, and no graphs. All pointers
are stable and measured source hashes match the committed snapshots. This
extends the analytical transfer contract only; the physical-event and whole
hydro-step requirements remain pending.
