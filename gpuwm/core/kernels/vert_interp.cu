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
            real t1 = q * powf(pbottom / P0, RCP);
            real pavg = 0.5f * (pt + pbottom);
            real dhdp = 11880.516f * 0.1902632f
                      * powf(pavg / 100.0f, 0.1902632f - 1.0f);
            real dt = dhdp * ((pt - pbottom) / 100.0f) * 0.0065f;
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

#define WRF_VI_MAX_LEVELS 64

__device__ static real wrf_vi_lagrange(const real* x, const real* y,
                                       int order, real target_x)
{
    // WRF lagrange_interp: full Lagrange polynomial through order+1 points.
    real px = 0.0f;
    for (int term = 0; term <= order; ++term) {
        real numer = 1.0f;
        real denom = 1.0f;
        for (int k = 0; k <= order; ++k) {
            if (k == term) continue;
            numer *= target_x - x[k];
            denom *= x[term] - x[k];
        }
        if (denom != 0.0f) px += y[term] * numer / denom;
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
                                   int vboundb)
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
            output[(size_t)kt * ncolumn + c] = 0.0f / 0.0f;
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
        if (ox[count - 1] - psfc < zap_close_levels) --count;
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
        if (ox[count - 1] - source_p[(size_t)knext * ncolumn + c]
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
            if (ox[count - 1] - pm < zap_close_levels && m < nsource - 1)
                continue;
            ox[count] = pm;
            oy[count] = field[(size_t)m * ncolumn + c];
            ++count;
        }
    }

    real x[WRF_VI_MAX_LEVELS];
    for (int m = 0; m < count; ++m)
        x[m] = interp_in_logp ? logf(ox[m]) : ox[m];

    for (int kt = 0; kt < ntarget; ++kt) {
        real pt = target_p[(size_t)kt * ncolumn + c];
        real xt = interp_in_logp ? logf(pt) : pt;
        int found = -1;
        for (int loop = 0; loop < count - 1; ++loop) {
            real a = xt - x[loop];
            real b = xt - x[loop + 1];
            if (a * b <= 0.0f) { found = loop; break; }
        }
        real result;
        if (found < 0) {
            if (pt > ox[0]) {
                if (extrap_temperature) {
                    // lagrange_setup t_extrap_type=2 CRC branch.
                    real t1 = oy[0] * powf(ox[0] / P0, RCP);
                    real pavg = 0.5f * (pt + ox[0]);
                    real dhdp = 11880.516f * 0.1902632f
                              * powf(pavg / 100.0f, 0.1902632f - 1.0f);
                    real dt = dhdp * ((pt - ox[0]) / 100.0f) * 0.0065f;
                    result = (t1 + dt) * powf(P0 / pt, RCP);
                } else {
                    result = oy[0];
                }
            } else {
                result = 0.0f / 0.0f;  // launcher rejects targets above top.
            }
        } else if (kt + 1 >= 1 + vboundb) {
            bool fits_upper = found + 2 <= count - 1;
            bool fits_lower = found - 1 >= 0;
            if (fits_upper && fits_lower) {
                result = 0.5f * (wrf_vi_lagrange(&x[found], &oy[found], 2, xt)
                                 + wrf_vi_lagrange(&x[found - 1],
                                                   &oy[found - 1], 2, xt));
            } else if (fits_upper) {
                result = wrf_vi_lagrange(&x[found], &oy[found], 2, xt);
            } else if (fits_lower) {
                result = wrf_vi_lagrange(&x[found - 1], &oy[found - 1], 2, xt);
            } else {
                result = 0.0f / 0.0f;  // all_dim >= 3 makes this unreachable.
            }
        } else {
            result = wrf_vi_lagrange(&x[found], &oy[found], 1, xt);
        }
        output[(size_t)kt * ncolumn + c] = result;
    }
}
