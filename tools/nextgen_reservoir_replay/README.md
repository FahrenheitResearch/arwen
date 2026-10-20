# NextGen reservoir native replay

Source: NOAA-OWP/t-route `12a8eae0cdfed437143c590659fa7077605a5e70`.
Read Apache-2.0 LICENSE and NOAA TERMS before porting. Full retained licence
and attribution are in `licenses/`. All native Fortran below is an offline
referee tool, never a product dependency or installed runtime.

`prepare.rs` extracts the unchanged LEVELPOOL_PHYSICS routine from that pin,
and the shipped `lp_reservoir3`/`test_lp3_run` Pocono lake 4185105 parameters
and upstream routing-network inflows. It creates a native driver that calls
the routine once per 300-second inflow, preserving the native binding's
current/current inflow arguments. The driver captures all 14 numeric inputs
and both outputs immediately at the function boundary. Rust rearranges bytes
only; it does not implement reservoir physics.

```sh
rustc --edition=2021 -O tools/nextgen_reservoir_replay/prepare.rs -o "$SCRATCH/prepare"
"$SCRATCH/prepare" "$PINNED_SOURCE" "$SCRATCH/capture"
# Run in $SCRATCH/capture; all generated files stay on the data disk.
gfortran -O2 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations \
  native_levelpool.f90 capture.f90 -o native-capture
./native-capture
"$SCRATCH/prepare" pack "$SCRATCH/capture/native-records.f32" "$SCRATCH/capture"
```

The native test case is an initial exact-issue check, not the required
LowerColorado/three-gauge event replay or short coupled-run acceptance.
Those fixtures are owned by the fleet referee. The independent persistence/RFC
call replay and production carrier-commit check now use real captured native
inputs; their pinned recipe is in
`reservoir_da` (the native reference tree, not in the public tree).

`gpuwm/core/kernels/nextgen_reservoirs.cuh` publishes device-callable
`ngen_levelpool`, `ngen_persistence` and `ngen_rfc`. Python `Reservoirs` binds
Rust-decoded resident views once and launches levelpool then assimilation on
the parent stream. No CUDA graphs, copies, synchronization or allocation per
step. Strict proof prepends the existing general `glibc_flt32.cuh` witness and
uses no contraction, FTZ=false and precise division/square root; fast default.

Levelpool parameters: 9-by-N float32 in `PARAMETERS` order. Assimilation
controls: 8-by-N float64 in `CONTROLS` order. Persistence carriers: 3-by-N
float64 (`persisted_release`, `update_time`, `persistence_time`), int32 index.
RFC uses the same update-time/index carriers. Clock is a float64 device
vector. Observations/forecasts are CSR float64 values/times and int64 offsets;
Rust verifies bounds and times before upload. All arrays exposed via `.arrays`
are checkpointed, including clocks, indices and intermediate levelpool views.
Actual native call captures establish double assimilation arithmetic and
float32 assignment at the native carrier boundary. The production kernel
preserves those assignment roundings in its float64 restart storage in both
numerical modes. Strict independent-call replays compare all returned fields
and committed carriers exactly; valid-observation persistence trajectories
and full-physics continuation remain separate qualifications.

RFC input filenames are immutable Rust table entries; the kernel writes an
int32 source ID. Unavailable forecast indices set device fault=1 rather than
reading beyond a forecast slice; inspect faults at output boundaries. Native
forecast expiry, negative-flow fallback and height-update ordering are kept.

The realization factory `gpuwm.hydro.reservoirs:prepare` selects exactly one
`options.network_component` or `options.fabric_manifest`. The Rust-prepared
source publishes `manifest.reservoirs`: `arrays` maps every canonical name in
`Reservoirs` to its payload key, plus `source_pin`, `parameter_columns`,
`control_columns`, `validated_csr`, `time_origin`, `reservoir_order_sha256`.
The factory seals every mapped payload hash and the source identity, including
controls, types, offsets and observation times. Register the reservoir row as
a **prepare-only resource**; MC/diffusive routing owns the per-level updates.
Advancing both the row and its embedded network hook would double-step it.
Driver assembly support for resource rows is owned by the coupled-driver lane.

In native hybrid MC/diffusive routing, MC advances the reservoir at its own
routing cadence. Its release enters the diffusive tributary boundary series
(`qtrib_g`, float64, Fortran order) by float32-to-float64 promotion. Diffusive
routing interpolates that series at its native adaptive clock and sums
tributaries in native connectivity order. It does not call levelpool again
at each adaptive substep. A captured reservoir-to-diffusive edge and its initial
boundary value are still required before claiming that coupling is verified.

Full-network replay must also preserve the actual bound native
`assume_short_ts` argument. It changes the upstream snapshot used by MC and
therefore the inflow delivered to reservoirs. The hydrofabric manifest carries
this routing option; reservoir initialization alone cannot establish it.
Compare complete native and GPU histories with the same captured mode and
explicit lake-to-outlet output IDs before interpreting a reservoir mismatch.

`run_hook.py` checks the prepared factory and the MC owner's `ReservoirLevel`
against the same real native 287-inflow trajectory: GPU boundary feed, native
current/current gather, selected reservoir row, scatter and output publication.
It imports `gpuwm.hydro.troute_mc` from the named MC revision in the receipt;
stage that owner's files separately when checking this unmerged lane. Its
one-reservoir/no-MC-reach harness is a composition check, not full-network or
persistence/RFC acceptance. Use explicit `GPUWM_NUMERICS=strict`. The harness
also checks that every reservoir carrier enters the MC checkpoint inventory.

The integrated short LowerColorado root-CLI proof uses 12 captured-lateral
steps of 300 seconds, 6,723 MC reaches, 7 level pools and 12 RFC reservoirs.
Strict output matches all seven native NetCDF variables and their ordered
typed headers. Fast is the default and is not bitwise to that native output.
This routing-only proof does not qualify runoff, terrain, active USGS/USACE
persistence, the diffusive reservoir edge, or the whole-hydro speed target.
