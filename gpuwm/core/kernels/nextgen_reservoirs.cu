// NOAA-OWP/t-route pin and Apache-2.0 attribution: nextgen_reservoirs.cuh.
// Parameter columns: area_km2,weir,max_h,wc,width,dam,orifice,oc,opening.
extern "C" __global__ void ngen_reservoir_levelpool(
    int active_count, const int *rows, int n, const float *parameters, const float *previous_inflow,
    const float *inflow, const float *lateral, float period,
    float *height, float *initial_height, float *lp_release, float *lp_height) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= active_count) return;
    int r = rows ? rows[lane] : lane;
    float h = height[r], q;
    initial_height[r] = h;
    ngen_levelpool(previous_inflow[r], inflow[r], lateral[r], period, h, q,
        parameters[r], parameters[n+r], parameters[2*n+r], parameters[3*n+r],
        parameters[4*n+r], parameters[5*n+r], parameters[6*n+r],
        parameters[7*n+r], parameters[8*n+r]);
    lp_height[r] = h;
    lp_release[r] = q;
}

// Persistent state/controls are column-major SoA, declared in reservoirs.py.
// Every numerical operation and observation/forecast search stays on device.
extern "C" __global__ void ngen_reservoir_assimilate(
    int active_count, const int *rows, int n, const float *parameters, const float *inflow,
    const float *initial_height, const float *lp_release, const float *lp_height,
    const int *types, const long long *offsets, const double *series,
    const double *times, const double *controls, double period,
    double *clock, double *carriers, int *indices, float *height,
    double *release, double *assimilated, int *dynamic_type,
    int *active_source, int *fault) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= active_count) return;
    int r = rows ? rows[lane] : lane;
    int type = types[r], count = int(offsets[r+1] - offsets[r]);
    const double *values = series + offsets[r];
    const double *observed_times = times + offsets[r];
    NgenAssimilationResult answer = {double(lp_release[r]), double(lp_height[r]), -9999., 1, 0};
    double area = parameters[r], max_h = parameters[2*n+r], orifice = parameters[6*n+r];
    if (type == 2 || type == 3) {
        NgenPersistenceState state = {carriers[r], carriers[n+r], carriers[2*n+r], indices[r]};
        answer = ngen_persistence(values, observed_times, count, clock[r], state,
            lp_release[r], inflow[r], period, area, max_h, orifice,
            initial_height[r], controls[r], controls[n+r], controls[2*n+r]);
        answer.type = type;
        // Actual mc_reach.pyx stores each Python DA return into float32
        // carrier arrays. Keep those native boundary roundings even though
        // our restart storage uses f64 (observed native DA call fixtures).
        carriers[r] = double(float(state.persisted));
        carriers[n+r] = double(float(state.update_time));
        carriers[2*n+r] = double(float(state.persistence_time));
        indices[r] = state.index;
    } else if (type == 4 || type == 5) {
        NgenRFCState state = {carriers[n+r], indices[r]};
        bool enabled = controls[3*n+r] != 0.;
        int total = int(controls[6*n+r]);
        int next = state.index + int(clock[r] >= state.update_time && state.index < total);
        // Prevent an invalid upstream horizon from reading outside the sealed
        // forecast slice. Fault is checked only at output/checkpoint boundaries.
        if (enabled && clock[r] <= controls[5*n+r] && (next < 0 || next >= count)) {
            fault[r] = 1;
            return;
        }
        answer = ngen_rfc(enabled, values, total, period, clock[r], state,
            controls[4*n+r], controls[5*n+r], type, inflow[r], initial_height[r],
            lp_release[r], lp_height[r], area * 1e6, max_h, int(controls[7*n+r]));
        carriers[n+r] = double(float(state.update_time));
        indices[r] = state.index;
    }
    height[r] = float(answer.height); // native Cython assimilate_elevation is f32
    release[r] = answer.release;
    assimilated[r] = answer.value;
    dynamic_type[r] = answer.type;
    active_source[r] = answer.source;
    clock[r] += period;
}

// Independent captures for native LEVELPOOL_PHYSICS replay. Input columns are
// previous,inflow,lateral,period,height,area,weir,max_h,wc,width,dam,orifice,oc,opening.
extern "C" __global__ void ngen_reservoir_replay(int n, const float *input, float *output) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float h = input[4*n+r], q;
    ngen_levelpool(input[r], input[n+r], input[2*n+r], input[3*n+r], h, q,
        input[5*n+r], input[6*n+r], input[7*n+r], input[8*n+r], input[9*n+r],
        input[10*n+r], input[11*n+r], input[12*n+r], input[13*n+r]);
    output[r] = h;
    output[n+r] = q;
}
