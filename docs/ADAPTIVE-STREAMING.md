# Adaptive steps on a streamed domain

The shared experiment executor uses the same adaptive controller for resident
and streamed domains. Before each streamed sweep, every reused tile receives
the domain's current `dt`, acoustic substep count, radiation/cumulus due
decisions and physics cadence intervals. The existing `DomainClock` supplies
the epoch and elapsed time. These derived operands do not replace the
configured timestep in restart identity.

The radiation observer reads the live store's producer records. A retained
resident state or a preparation row slab is not the current streamed forecast.
Full stored geography supplies the acoustic controller's static map-factor
maximum when there is no resident domain state.

Tile planning reserves the halo needed at the configured maximum adaptive
timestep, using the same WRF acoustic-count calculation as the controller and
the existing dependency-radius formula. Live tree planning resolves the actual
static map factors before allocating tile buffers, ring arenas and boundary
windows. A cold config-only estimate records that its unit map-factor estimate
must be refined on live geometry. The live count and timestep remain unchanged;
only the allocated halo envelope is larger when required.

Each completed sweep contributes one CFL diagnostic row. All three RK stages
of all tiles fold their owned mass columns into that row, using the original
WRF CFL arithmetic. Halos are excluded from both the maximum and histogram.
The accumulator is the already priced per-grid diagnostic ring, with no new
device array. Stream events order its clear, tile reductions and publication;
an aborted sweep does not commit a new diagnostic step.

CUDA graphs bind the live scalar operands, CFL row and owned window. A changed
timestep or acoustic count invalidates old captures after pending work drains.
Default sweep-scoped graphs expire at the next sweep, including repeated
`sweep(1)` calls, so neither absolute-time arguments nor old graph workspaces
accumulate across the run.

The permanent controls are `tests/test_adaptive_stream_control.py` and
`tests/test_adaptive_stream_gpu.py`. They compare actual resident and tiled
dycore state and every CFL histogram word through fixed and changing steps,
with graph capture required as well as disabled. The real-input acceptance
also exercises radiation, cumulus, two domains and a streamed checkpoint
resume through `runtime.run_experiment`'s experiment-tree path.
