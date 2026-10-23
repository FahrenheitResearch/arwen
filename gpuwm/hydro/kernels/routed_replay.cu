// Original ArWen kernel: the whole sealed routing spin-up in ONE launch.
// Composed after troute_mc.cu, nextgen_reservoirs.cuh and runoff_replay.cu
// (routed_init.py), so every element runs the same device function as the
// per-step launch sequence: replay_lateral_node, troute_mc_node_with, the
// reservoir gather/level-pool/scatter rows. Breakage it removes: the host
// launched 25,920 routing steps one kernel at a time on one CPU thread
// (17.5 s on a 5070 Ti host, 29.5 s on a 4090 host) before forecast step 1.
//
// Dataflow, not level barriers. A level-synchronous replay is 82 dependent
// stages per routing step on the measured basin (19.8 s on the 5070 Ti even
// as one launch). Here each routing node is a task owned by one warp's lane 0
// (a worker); a task runs step t of its node once every upstream node has
// published step t and every downstream reader has finished step t-slots+1,
// so headwaters run ahead of the outlet and steps overlap down the network.
// Each node's step-t state lives in ring slot t % slots: step t reads
// "previous" from slot t-1 and the upstream "current" from slot t, exactly
// the values the per-step snapshot and level order gave it, and the same
// arithmetic produces the same bits.
//
// Progress: workers walk their tasks in (step, topological position) order and
// every dependency is an earlier (step, position) task, so the earliest
// unfinished task can always run; the launch is cooperative, so every worker
// is resident.

// Published state of other workers: read from L2, past this SM's L1.
struct McLoadCg { __device__ __forceinline__ float operator()(const float *p) const { return __ldcg(p); } };

__device__ __forceinline__ void routed_replay_wait(const int *done, int node, int step) {
    while (*(const volatile int *)(done + node) < step) { }
}

// Level-pool row of ngen_reservoir_levelpool, then the type-1 (level pool)
// result of ngen_reservoir_assimilate. The replay refuses any other type
// before launch (routed_init.py), exactly as the sequence did.
__device__ __forceinline__ void routed_replay_levelpool_row(int r, int n,
    const float *parameters, const float *previous_inflow, const float *inflow,
    const float *lateral, float period, double period_double,
    float *height, float *initial_height, float *lp_release, float *lp_height,
    double *clock, double *release, double *assimilated, int *dynamic_type,
    int *active_source) {
    float h = height[r], q;
    initial_height[r] = h;
    ngen_levelpool(previous_inflow[r], inflow[r], lateral[r], period, h, q,
        parameters[r], parameters[n+r], parameters[2*n+r], parameters[3*n+r],
        parameters[4*n+r], parameters[5*n+r], parameters[6*n+r],
        parameters[7*n+r], parameters[8*n+r]);
    lp_height[r] = h;
    lp_release[r] = q;
    NgenAssimilationResult answer = {double(lp_release[r]), double(lp_height[r]), -9999., 1, 0};
    height[r] = float(answer.height);
    release[r] = answer.release;
    assimilated[r] = answer.value;
    dynamic_type[r] = answer.type;
    active_source[r] = answer.source;
    clock[r] += period_double;
}

// order[p]: the node of task p, in the router's step order (each MC level,
// each reservoir hook after its level), then the nodes no stage advances.
// kind[p]: -2 MC reach, -1 never advanced (lateral inflow only), r >= 0 the
// reservoir row whose hook advances that node.
extern "C" __global__ void routed_replay(
    const double *depth, const double *area, const long long *lateral_offsets,
    const int *lateral_indices, float *routing_lateral, int *lateral_status,
    int hours, double scale,
    int n, int substeps, float dt, int short_ts,
    const int *offsets, const int *upstream, const int *down_offsets, const int *downstream,
    const float *parameters, const float *lateral, const float *boundary_previous,
    const float *boundary_current, float *previous, float *state, float *diagnostics,
    float *ring, int slots, int *done, const int *order, const int *kind, int tasks,
    const int *node_indices, int reservoir_count, const float *reservoir_parameters,
    float *previous_inflow, float *inflow, float *reservoir_lateral, float period,
    double period_double, float *height, float *initial_height, float *lp_release,
    float *lp_height, double *clock, double *release, double *assimilated,
    int *dynamic_type, int *active_source) {
    if ((threadIdx.x & 31) != 0) return;
    int worker = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int workers = (gridDim.x * blockDim.x) >> 5;
    int total = hours * substeps;
    McLoadCg load;
    for (int t = 0; t < total; ++t) {
        int hour = t / substeps;
        bool fresh = t % substeps == 0;
        float *now = ring + (long long)(t % slots) * 3 * n;
        const float *before = ring + (long long)((t + slots - 1) % slots) * 3 * n;
        for (int p = worker; p < tasks; p += workers) {
            int i = order[p], task = kind[p];
            if (fresh)
                replay_lateral_node(i, depth, area, lateral_offsets, lateral_indices,
                    routing_lateral, lateral_status, hours, hour, scale);
            if (task == -1) continue;
            for (int k = offsets[i]; k < offsets[i+1]; ++k) routed_replay_wait(done, upstream[k], t);
            for (int k = down_offsets[i]; k < down_offsets[i+1]; ++k)
                routed_replay_wait(done, downstream[k], t - slots + 1);
            __threadfence();
            if (task == -2) {
                troute_mc_node_with(load, i, dt, short_ts, offsets, upstream, parameters,
                    lateral, boundary_previous, boundary_current, before, now, diagnostics);
            } else {
                troute_mc_reservoir_gather_routed_row(load, task, short_ts, node_indices,
                    offsets, upstream, boundary_previous, boundary_current, before, now,
                    lateral, previous_inflow, inflow, reservoir_lateral);
                routed_replay_levelpool_row(task, reservoir_count, reservoir_parameters,
                    previous_inflow, inflow, reservoir_lateral, period, period_double,
                    height, initial_height, lp_release, lp_height, clock, release,
                    assimilated, dynamic_type, active_source);
                troute_mc_reservoir_scatter_row(task, node_indices, release, height, now);
            }
            __threadfence();
            *(volatile int *)(done + i) = t;
        }
    }
    // Final carriers, each worker its own nodes: state after the last step,
    // previous = the last step's snapshot (state after the step before).
    const float *last = ring + (long long)((total - 1) % slots) * 3 * n;
    const float *prior = ring + (long long)((total - 2 + slots) % slots) * 3 * n;
    for (int p = worker; p < tasks; p += workers) {
        int i = order[p];
        for (int k = 0; k < 3; ++k) {
            if (kind[p] == -1) {
                previous[3*i+k] = state[3*i+k];
            } else {
                state[3*i+k] = last[3*i+k];
                previous[3*i+k] = prior[3*i+k];
            }
        }
    }
}

// The replay's own refusal and health checks, in this module rather than as
// CuPy reductions and fills: those compiled one at a time at hydro attach on
// a cold cache (a CUB any/all pair is 1-2 s on one core).
extern "C" __global__ void routed_replay_types(const int *types, int count, int *flags) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        if (types[i] != 1) atomicOr(flags, 1);
}
extern "C" __global__ void routed_replay_fill(double *values, int count, double value) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        values[i] = value;
}
// flags: 1 nonfinite channel state, 2 lateral inflow status set, 4 nonfinite
// reservoir height, 8 reservoir fault.
extern "C" __global__ void routed_replay_health(const float *state, int state_count,
    const int *status, int status_count, const float *height, const int *fault,
    int reservoir_count, int *flags) {
    int count = max(state_count, max(status_count, reservoir_count));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        if (i < state_count && !isfinite(state[i])) atomicOr(flags, 1);
        if (i < status_count && status[i] != 0) atomicOr(flags, 2);
        if (i < reservoir_count && !isfinite(height[i])) atomicOr(flags, 4);
        if (i < reservoir_count && fault[i] != 0) atomicOr(flags, 8);
    }
}
