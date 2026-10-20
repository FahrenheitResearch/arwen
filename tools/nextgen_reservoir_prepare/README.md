# Rust native reservoir startup preparation

NOAA-OWP/t-route `12a8eae0cdfed437143c590659fa7077605a5e70`:
“Apache License”, “Version 2.0, January 2004”. The actual LICENSE and constructor
source were read before adaptation; retained notices are under `licenses/`.
No withdrawn UCAR source, fixtures or code are used.

`nextgen-reservoir-prepare NATIVE_MANIFEST NEW_OUTPUT TIME_ORIGIN` verifies
sealed native observation columns, packs float32 level-pool SoA parameters,
and initializes fixed GPU carriers in Rust. It validates the native execution
pin and successful run. Native `levelpool.pyx` supplies dam length10m;
`levelpool_structs.c` converts parameters to float before cold initialization
`orifice + (maximum-orifice)*ifd`, when h0<-900000000. No reservoir time step
runs on CPU. Independent column verification and payload writes use Rayon.

The captured type/parameter row IDs must match and be unique. This initial
packer accepts actual type1 rows only: persistence/RFC require their actual
native series/control captures rather than invented baselines. Output is a
manifest **fragment**, not a fabricated hydrofabric/network. The hydrofabric
owner merges its `files` payloads and `reservoirs` declaration with the actual
collapsed native network; all paths must remain relative to the final manifest.
Reservoir IDs remain in captured row order. Factory
`gpuwm.hydro.reservoirs:prepare` consumes the prepared component/manifest;
realization phase must be `prepare`. MC owns dependency-level subset advances.
Driver resource support is present in owner commit `c949c7d7152`.

Build with CARGO_HOME, CARGO_TARGET_DIR and TMPDIR under the lane scratch. Input
native captures and derived case payloads remain private research material;
this code contains no case data. Do not bundle NC hydrofabric or derived data
in a wheel/public demo. The shipped demo uses the separately permitted fabric
recipe, owned by the hydrofabric lane.

## Native-free world/CONUS lake preparation

`nextgen-reservoir-prepare --fabric FABRIC_MANIFEST INITIAL_STATE_JSON NEW_OUTPUT`
binds `rw_hydrofabric prepare`'s sealed `attributes.json.lakes` to the same CUDA
ABI. This is the reservoir adapter for HYDRO-WORLD CONTRACT v1, item 3. No
native installation/capture is required. The original captured-input command
above remains available with its original semantics.

Lake rows require `hl_link` (unique positive i64), `id` (nexus), `toid` (outlet
flowpath), and finite numeric `LkArea` (km²), `WeirE`, `LkMxE`, `WeirC`, `WeirL`,
`Dam_Length`, `OrificeE`, `OrificeC`, `OrificeA`, `ifd`. Elevations and lengths
are metres; orifice area is m². This mode retains the **supplied dam length**;
the captured constructor's 10 m convention is not a world-fabric default.
Incomplete rows, invalid geometry and a declared `reservoir_type` other than 1
refuse before creating output. DA rows need their series/control preparer;
they must not be silently replaced by level pool or dropped.

The initialization owner supplies this packet (no example physical values are
implied):

```json
{
  "schema": "arwen.nextgen.reservoir-initial.v1",
  "time_origin": "2023-01-05T00:00:00Z",
  "vertical_datum": "producer-named elevation datum",
  "source": {
    "method": "analysis",
    "sha256": "64 hexadecimal digits identifying the admitted source",
    "fabric_manifest_sha256": "SHA256 of FABRIC_MANIFEST bytes",
    "time_origin": "2023-01-05T00:00:00Z",
    "vertical_datum": "producer-named elevation datum"
  },
  "rows": [{"id": 123, "height_m": 100.0, "release_m3_s": 1.0}]
}
```

Methods are `analysis`, `retrospective`, or `replay-spinup`. All IDs must match
the lakes exactly, with no duplicates, extras or missing stores. Releases retain
the parsed float64 value in their float64 ABI array; heights and geometric
parameters use the existing float32 level-pool ABI. Heights must
be at or above their orifice elevation and releases nonnegative. Datum/time
agreement and source hash are producer attestations: this adapter does not
acquire water state, establish dataset permission, convert storage to height,
or prove datum conversion. `ifd` is checked but never used as a silent cold
initializer. The initialized relative clock and previous inflow start at zero,
as in the existing constructor; this is a fresh-run packet, not an arbitrary
mid-run checkpoint. Full continuation uses the existing checkpoint path.

Rows are sorted by `hl_link`; `reservoir_ids.i64` records that axis.
`lake_outlets` preserves nexus/flowpath bindings. Output has 21 hashed ABI
payloads, ordered IDs and `reservoirs.initialization` provenance. The fabric
owner copies/renames the fragment payload paths into its sealed manifest and
collapses the network using this axis; this tool does not produce a complete
network or claim upstream inflows are connected. Include fragment/source and
initialization identities in the assembled manifest's restart identity.
Data permissions remain those of the source producer. No private research
capture is read by this mode, and none is needed for a permissive builder.

The actual Rust tests write synthetic ABI fixtures only, entirely inside
`HYDRO_TEST_ROOT` (required, owned data-disk scratch):

```sh
RAYON_NUM_THREADS=24 HYDRO_TEST_ROOT="$LANE_SCRATCH/test-data" \
  cargo test --release --offline --locked -j24 \
  --manifest-path tools/nextgen_reservoir_prepare/Cargo.toml
```

Set `CARGO_HOME`, `CARGO_TARGET_DIR` and `TMPDIR` to owned scratch as above.
