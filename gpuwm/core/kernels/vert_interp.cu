// WRF real-data vertical interpolation kernels.
//
// vertical_interpolate_logp: retained linear-in-log(p) helper (one thread
// per target cell; float64 mirror gpuwm.verify.npref.
// np_vertical_interpolate_logp).  This is NOT WRF real's default scheme;
// the production ingest path uses wrf_real_vertical_interpolate below.
// Source pressure is strictly descending in memory.  Interior interpolation
// is linear in log(p); below-temperature is module_initialize_real.F's
// t_extrap_type=2 standard-atmosphere potential-temperature extrapolation.

extern "C" __global__
void vertical_interpolate_logp(const real* __restrict__ field,
                               const real* __restrict__ source_p,
                               const real* __restrict__ target_p,
                               real* __restrict__ output,
                               int nsource, int ntarget, int ncolumn,
                               int below_temperature)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ntarget * ncolumn;
    if (tid >= total) return;
    int kt = tid / ncolumn;
    int c = tid - kt * ncolumn;
    real pt = target_p[(size_t)kt * ncolumn + c];
    real pbottom = source_p[c];
    real ptop = source_p[(size_t)(nsource - 1) * ncolumn + c];

    if (pt > pbottom) {
        real q = field[c];
        if (below_temperature) {
            real t1 = q * powf(__fdiv_rn(pbottom, P0), RCP);
            real pavg = 0.5f * (pt + pbottom);
            real dhdp = 11880.516f * 0.1902632f
                      * powf(__fdiv_rn(pavg, 100.0f), 0.1902632f - 1.0f);
            real dt = dhdp * (__fdiv_rn((pt - pbottom), 100.0f)) * 0.0065f;
            q = (t1 + dt) * powf(P0 / pt, RCP);
        }
        output[tid] = q;
        return;
    }
    if (pt < ptop) {
        // The host launcher rejects this WRF-fatal condition before launch.
        output[tid] = 0.0f / 0.0f;
        return;
    }

    for (int k = 0; k < nsource - 1; ++k) {
        real pa = source_p[(size_t)k * ncolumn + c];
        real pb = source_p[(size_t)(k + 1) * ncolumn + c];
        if (pa >= pt && pt >= pb) {
            real qa = field[(size_t)k * ncolumn + c];
            real qb = field[(size_t)(k + 1) * ncolumn + c];
            real weight = (logf(pt) - logf(pa)) / (logf(pb) - logf(pa));
            output[tid] = qa + weight * (qb - qa);
            return;
        }
    }
    output[tid] = 0.0f / 0.0f;  // validated monotonic input makes unreachable.
}

// WRF real's default vertical interpolation (one thread per column).
// Float64 authority mirror: gpuwm.verify.npref.np_wrf_real_vert_interp,
// a transcription of module_initialize_real.F:vert_interp/lagrange_setup/
// lagrange_interp (v4.6.1) at the reference run's Registry defaults:
// use_surface=T, use_levels_below_ground=T, lagrange_order=2 (vboundb=4
// linear band, averaged overlapping quadratic pairs above), plus the
// force_sfc_in_vinterp and zap_close_levels column-assembly removals.
// interp_in_logp selects interp_type=2 (LOG p) versus 1 (plain p);
// extrap_temperature selects the t_extrap_type=2 CRC standard-atmosphere
// below-ground branch versus extrap_type=2 constant.
//
// field/source_p are (nsource, column) bottom-up isobaric levels WITHOUT
// the surface; sfc_field/sfc_p carry the surface pseudo-level.  The host
// launcher validates monotonicity, the surface bracket, and that no target
// lies above the source top (WRF-fatal), so the in-kernel NaN writes are
// unreachable guards.
//
// WRF_VI_MAX_LEVELS sizes the two per-thread column arrays below: the
// source levels plus the surface pseudo-level.  64 is the default binary
// every source up to 63 levels runs on; gpuwm/ingest/vert.py compiles the
// deeper tiers (WRF_VERT_INTERP_LEVEL_TIERS) by defining it ahead of this
// file and hands a column deeper than the top tier to the CPU bridge.  The
// bound is an allocation size only: no expression reads it and every loop
// runs to the column's own count, so a tier cannot change a result.

#ifndef WRF_VI_MAX_LEVELS
#define WRF_VI_MAX_LEVELS 64
#endif

// Match the bridge's host float64-to-float32 field conversion, including tiny values.
extern "C" __global__
void wrf_vertical_field_float32(const double* __restrict__ input,
                                 float* __restrict__ output, int total,
                                 int ny, int nx, long long sk,
                                 long long sj, long long si)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int k = tid / (ny * nx);
    int j = (tid / nx) % ny;
    int i = tid % nx;
    output[tid] = gfk_d2f_rn(input[(long long)k * sk + (long long)j * sj + (long long)i * si]);
}

// Same glibc tables and polynomial; fused library-internal double ops.
// Exhaustively equal to host logf on every positive normal float on both cards.
__device__ float wrf_vi_pressure_log(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u)
        // Production arguments are pressures, normal by construction.
        return gfk_log(x);
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = __fma_rn(z, GFK_LOGF_INVC[i], -1.0);
    double y0 = __fma_rn((double)k, GFK_LOGF_LN2, GFK_LOGF_LOGC[i]);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_LOGF_A1, r, GFK_LOGF_A2);
    y = __fma_rn(GFK_LOGF_A0, r2, y);
    y = __fma_rn(y, r2, __dadd_rn(y0, r));
    return __double2float_rn(y);
}

// Field products can be tiny. Intrinsics acquire the loader's flush flag;
// explicit RN PTX keeps their bits and prevents contraction on both cards.
__device__ static float wrf_vi_mul(float a, float b)
{
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ static float wrf_vi_div(float a, float b)
{
    float result = __fdiv_rn(a, b);
    unsigned int aa = __float_as_uint(a) & 0x7fffffffu;
    unsigned int ab = __float_as_uint(b) & 0x7fffffffu;
    unsigned int ar = __float_as_uint(result) & 0x7fffffffu;
    // Keep the fast intrinsic where inputs and output survive its flag.
    if ((aa != 0u && aa < 0x00800000u) ||
        (ab != 0u && ab < 0x00800000u) || (ar == 0u && aa != 0u))
        asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__device__ static float wrf_vi_add(float a, float b)
{
    float result;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

template<int order>
__device__ __forceinline__ static real wrf_vi_lagrange(const real* x, const real* y,
                                                     real target_x)
{
    // WRF lagrange_interp: full Lagrange polynomial through order+1 points.
    real px = 0.0f;
    #pragma unroll
    for (int term = 0; term <= order; ++term) {
        real numer = 1.0f;
        real denom = 1.0f;
        #pragma unroll
        for (int k = 0; k <= order; ++k) {
            if (k == term) continue;
            numer = wrf_vi_mul(numer, __fsub_rn(target_x, x[k]));
            denom = wrf_vi_mul(denom, __fsub_rn(x[term], x[k]));
        }
        if ((__float_as_uint(denom) & 0x7fffffffu) != 0u)
            px = wrf_vi_add(px, wrf_vi_div(wrf_vi_mul(y[term], numer), denom));
    }
    return px;
}

extern "C" __global__
void wrf_real_vertical_interpolate(const real* __restrict__ field,
                                   const real* __restrict__ sfc_field,
                                   const real* __restrict__ source_p,
                                   const real* __restrict__ sfc_p,
                                   const real* __restrict__ target_p,
                                   real* __restrict__ output,
                                   int nsource, int ntarget, int ncolumn,
                                   int interp_in_logp, int extrap_temperature,
                                   int force_sfc, real zap_close_levels,
                                   int vboundb, real rust_p0, real rust_rcp, real hpa,
                                   real crc_product, real crc_exponent)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncolumn) return;

    real ox[WRF_VI_MAX_LEVELS];
    real oy[WRF_VI_MAX_LEVELS];
    real psfc = sfc_p[c];

    // First source level strictly above the surface (WRF ko_above_sfc).
    int m_above = -1;
    for (int m = 0; m < nsource; ++m) {
        if (source_p[(size_t)m * ncolumn + c] < psfc) { m_above = m; break; }
    }
    if (m_above < 0) {
        for (int kt = 0; kt < ntarget; ++kt)
            output[(size_t)kt * ncolumn + c] = __int_as_float(0x7fc00000);
        return;
    }

    int count = 0;
    if (m_above > 0) {
        // Surface sits inside the column: below-ground levels first, a
        // single close-level check on the deepest one against the surface.
        for (int m = 0; m < m_above; ++m) {
            ox[count] = source_p[(size_t)m * ncolumn + c];
            oy[count] = field[(size_t)m * ncolumn + c];
            ++count;
        }
        if (__fsub_rn(ox[count - 1], psfc) < zap_close_levels) --count;
        ox[count] = psfc;
        oy[count] = sfc_field[c];
        ++count;
        int knext = m_above;
        if (force_sfc > 0) {
            real pforce = target_p[(size_t)(force_sfc - 1) * ncolumn + c];
            for (int m = m_above; m < nsource; ++m) {
                if (source_p[(size_t)m * ncolumn + c] <= pforce) {
                    knext = m;
                    break;
                }
            }
        }
        int kst = knext;
        if (__fsub_rn(ox[count - 1], source_p[(size_t)knext * ncolumn + c])
                < zap_close_levels)
            kst = knext + 1;
        for (int m = kst; m < nsource; ++m) {
            ox[count] = source_p[(size_t)m * ncolumn + c];
            oy[count] = field[(size_t)m * ncolumn + c];
            ++count;
        }
    } else {
        // Surface is the lowest level; iterative close-level check that
        // never removes the topmost input level.
        ox[0] = psfc;
        oy[0] = sfc_field[c];
        count = 1;
        int knext = 0;
        if (force_sfc > 0) {
            real pforce = target_p[(size_t)(force_sfc - 1) * ncolumn + c];
            for (int m = 0; m < nsource; ++m) {
                if (source_p[(size_t)m * ncolumn + c] <= pforce) {
                    knext = m;
                    break;
                }
            }
        }
        for (int m = knext; m < nsource; ++m) {
            real pm = source_p[(size_t)m * ncolumn + c];
            if (__fsub_rn(ox[count - 1], pm) < zap_close_levels && m < nsource - 1)
                continue;
            ox[count] = pm;
            oy[count] = field[(size_t)m * ncolumn + c];
            ++count;
        }
    }

    // Once assembled, pressure coordinates can hold their own logarithms.
    // Only the bottom pressure is still needed by the extrapolation branch.
    real bottom_pressure = ox[0];
    real* x = ox;
    for (int m = 0; m < count; ++m)
        // Production log arguments are pressures, normal by construction.
        x[m] = interp_in_logp ? wrf_vi_pressure_log(ox[m]) : ox[m];

    int previous_window = 0;
    real previous_target = __int_as_float(0x7f800000);
    for (int kt = 0; kt < ntarget; ++kt) {
        real pt = target_p[(size_t)kt * ncolumn + c];
        real xt = interp_in_logp ? wrf_vi_pressure_log(pt) : pt;
        int found = -1;
        // Descending targets cannot return to an earlier bracket. Reset
        // for an increasing target so arbitrary target order stays valid.
        int first_window = xt <= previous_target ? previous_window : 0;
        previous_target = xt;
        for (int loop = first_window; loop < count - 1; ++loop) {
            real a = __fsub_rn(xt, x[loop]);
            real b = __fsub_rn(xt, x[loop + 1]);
            if (__fmul_rn(a, b) <= 0.0f) { found = loop; break; }
        }
        previous_window = found < 0 ? 0 : found;
        real result;
        if (found < 0) {
            if (pt > bottom_pressure) {
                if (extrap_temperature) {
                    // lagrange_setup t_extrap_type=2 CRC branch.
                    real t1 = wrf_vi_mul(oy[0], gfk_pow(__fdiv_rn(bottom_pressure, rust_p0), rust_rcp));
                    real pavg = __fmul_rn(0.5f, __fadd_rn(pt, bottom_pressure));
                    real dhdp = __fmul_rn(crc_product,
                        gfk_pow(__fdiv_rn(pavg, hpa), crc_exponent));
                    real dt = __fmul_rn(__fmul_rn(dhdp,
                        __fdiv_rn(__fsub_rn(pt, bottom_pressure), hpa)), 0.0065f);
                    result = wrf_vi_mul(wrf_vi_add(t1, dt),
                        gfk_pow(__fdiv_rn(rust_p0, pt), rust_rcp));
                } else {
                    result = oy[0];
                }
            } else {
                result = __int_as_float(0x7fc00000);  // launcher rejects targets above top.
            }
        // Rust uses usize. Unsigned arithmetic also preserves its wrap at -1
        // and its linear band for every other admitted negative int32 option.
        } else if ((unsigned int)(kt + 1) >= 1u + (unsigned int)vboundb) {
            bool fits_upper = found + 2 <= count - 1;
            bool fits_lower = found - 1 >= 0;
            if (fits_upper && fits_lower) {
                result = wrf_vi_mul(0.5f, wrf_vi_add(
                    wrf_vi_lagrange<2>(&x[found], &oy[found], xt),
                    wrf_vi_lagrange<2>(&x[found - 1], &oy[found - 1], xt)));
            } else if (fits_upper) {
                result = wrf_vi_lagrange<2>(&x[found], &oy[found], xt);
            } else if (fits_lower) {
                result = wrf_vi_lagrange<2>(&x[found - 1], &oy[found - 1], xt);
            } else {
                result = __int_as_float(0x7fc00000);  // all_dim >= 3 makes this unreachable.
            }
        } else {
            result = wrf_vi_lagrange<1>(&x[found], &oy[found], xt);
        }
        output[(size_t)kt * ncolumn + c] = result;
    }
}
