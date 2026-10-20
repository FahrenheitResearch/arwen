// Adapted from NOAA-OWP/t-route, Apache-2.0, pin
// 12a8eae0cdfed437143c590659fa7077605a5e70:
// src/kernel/reservoir/Level_Pool/module_levelpool.F and
// src/troute-routing/troute/routing/fast_reach/reservoir_{hybrid,RFC}_da.py.
// See licenses/LICENSE-t-route.txt and licenses/NOTICE-t-route-reservoirs.txt.
#ifndef GPUWM_NEXTGEN_RESERVOIRS_CUH
#define GPUWM_NEXTGEN_RESERVOIRS_CUH

__device__ __forceinline__ double ngen_reservoir_nan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ __forceinline__ float ngen_reservoir_power(float depth) {
#ifdef GPUWM_NEXTGEN_RESERVOIR_STRICT
    return gfk_pow(depth, 1.5f);
#else
    return powf(depth, 1.5f);
#endif
}

__device__ __forceinline__ float ngen_reservoir_release(
    float initial_h, float stage_h, float weir, float maximum, float wc,
    float width, float dam, float orifice, float oc, float opening) {
    float depth = stage_h - weir;
    if (depth > maximum - weir) depth = maximum - weir;
    float outlet = (oc * opening) * sqrtf((2.f * 9.81f) * (stage_h - orifice));
    float spill = (wc * width) * ngen_reservoir_power(depth);
    if (initial_h > maximum)
        return (outlet + spill) + (wc * (width * dam)) * ngen_reservoir_power(initial_h - maximum);
    if (depth > 0.f) return outlet + spill;
    if (stage_h > orifice) return outlet;
    return 0.f;
}

// Native argument order, omitting diagnostic-only lake number. Area is km2,
// elevations and lengths m, opening m2, inflows and release m3/s, period s.
// No inflow/history mutation: the network caller owns its dependency ordering.
__device__ __forceinline__ void ngen_levelpool(
    float previous_inflow, float inflow, float lateral, float period,
    float &height, float &release, float area, float weir, float maximum,
    float wc, float width, float dam, float orifice, float oc, float opening) {
    float second_inflow = previous_inflow + ((inflow + lateral - previous_inflow) * 0.33f);
    float third_inflow = previous_inflow + ((inflow + lateral - previous_inflow) * 0.67f);
    float surface = area * 1.0e6f;
    float q = ngen_reservoir_release(height, height, weir, maximum, wc, width, dam, orifice, oc, opening);
    float k1 = surface > 0.f ? ((previous_inflow - q) / surface) * period : 0.f;
    q = ngen_reservoir_release(height, height + k1 / 3.f, weir, maximum, wc, width, dam, orifice, oc, opening);
    float k2 = surface > 0.f ? ((second_inflow - q) / surface) * period : 0.f;
    q = ngen_reservoir_release(height, height + 0.667f * k2, weir, maximum, wc, width, dam, orifice, oc, opening);
    float k3 = surface > 0.f ? ((third_inflow - q) / surface) * period : 0.f;
    height = height + (k1 / 4.f + 0.75f * k3);
    release = ngen_reservoir_release(height, height, weir, maximum, wc, width, dam, orifice, oc, opening);
}

struct NgenPersistenceState {
    double persisted, update_time, persistence_time;
    int index;
};
struct NgenAssimilationResult {
    double release, height, value;
    int type, source;
};

// Timeseries are CSR slices with strictly increasing times, pre-QC'd NaNs.
// Preserve upstream's argmin tie/empty-eligible behaviour and storage order.
__device__ __forceinline__ NgenAssimilationResult ngen_persistence(
    const double *observed, const double *times, int count, double now,
    NgenPersistenceState &state, double levelpool_release, double inflow,
    double period, double area_km2, double maximum, double orifice,
    double initial_h, double lookback_hours, double update_interval,
    double persistence_interval) {
    double persisted = state.persisted;
    if (now >= state.update_time) {
        double search_time = state.update_time;
        int nearest = 0;
        double best = __longlong_as_double(0x7ff0000000000000LL);
        for (int j = 0; j < count; ++j) {
            double diff = state.update_time - times[j];
            if (diff >= 0. && diff < best) { best = diff; nearest = j; }
        }
        double observation = ngen_reservoir_nan(), time = 0.;
        for (int j = nearest; j >= 0 && count > 0; --j) {
            if (!isnan(observed[j])) { observation = observed[j]; time = times[j]; break; }
        }
        if (!isnan(observation)) state.update_time += update_interval;
        if (!isnan(observation) && search_time - time <= lookback_hours * 60. * 60.) {
            persisted = observation;
            state.index = 1;
            state.persistence_time += persistence_interval;
        } else if (now >= state.persistence_time) {
            state.index += 1;
            state.persistence_time += persistence_interval;
        }
    } else if (now >= state.persistence_time) {
        int previous_index = state.index;
        state.index += 1;
        state.persistence_time += persistence_interval;
        if (previous_index > 11) { persisted = levelpool_release; state.index = 0; }
    }
    double out = persisted;
    if (isnan(out)) { out = levelpool_release; state.index = 0; }
    state.persisted = persisted;
    double surface = area_km2 * 1e6;
    double initial_storage = (initial_h - orifice) * surface;
    double max_storage = (maximum - orifice) * surface;
    double assessed = out;
    if (assessed < 0.) out = 0.;
    double projected = initial_storage + (inflow - assessed) * period;
    bool exceeded = projected > max_storage;
    // The native optional minimum_storage is zero at this BMI call. Its
    // `projected < minimum && projected > 0` branch therefore cannot execute.
    if (projected <= 0.) out = inflow;
    if (out < 0.) out = 0.;
    if (exceeded && out < levelpool_release) out = levelpool_release;
    double change = (inflow - out) * period;
    return {out, initial_h + change / surface, ngen_reservoir_nan(), 0, 0};
}

struct NgenRFCState { double update_time; int index; };

// area_m2 is explicitly converted by upstream model_reservoir.py at its call.
// source is the immutable Rust-decoded filename-table ID (0 means empty).
__device__ __forceinline__ NgenAssimilationResult ngen_rfc(
    bool enabled, const double *series, int total_counts, double period,
    double now, NgenRFCState &state, double series_period, double persist_until,
    int type, double inflow, double initial_h, double lp_q, double lp_h,
    double area_m2, double maximum, int source_id) {
    NgenAssimilationResult result = {type == 4 ? lp_q : inflow, lp_h, -9999., 1, 0};
    if (enabled && now <= persist_until) {
        if (now >= state.update_time && state.index < total_counts) {
            state.update_time += series_period;
            state.index += 1;
        }
        double value = series[state.index];
        result = {type == 4 ? value : inflow + value, 0., value, type, source_id};
        result.height = initial_h + ((inflow - result.release) / area_m2) * period;
        if (result.height < 0.) result.height = 0.;
        else if (result.height > maximum) result.height = maximum;
        if (result.release < 0.) {
            int previous = state.index;
            while (result.release < 0. && previous > 1) result.release = series[--previous];
            if (result.release < 0.) result = {type == 4 ? lp_q : inflow, lp_h, -9999., 1, 0};
        }
    }
    return result;
}
#endif
