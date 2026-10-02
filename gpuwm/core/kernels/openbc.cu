// gpuwm/core/kernels/openbc.cu
//
// Open (gravity-wave radiative) lateral boundary conditions and the
// w_damping=1 vertical-velocity limiter (Phase 2 Task 9).
//
// open_u_radiative / open_v_radiative ADD the Klemp-Wilhelmson radiative
// term to the coupled slow tendency at the two boundary-normal velocity
// faces, transcribed from WRF v4.6.1 dyn_em/module_advect_em.F
// advect_u/advect_v (the open_xs/open_xe//open_ys/open_ye blocks,
// ``tendency = tendency + ...`` at module_advect_em.F:1252/1267 and
// 2721/2736; called with u_old = u, the RK stage estimate, per
// module_em.F rk_tendency):
//
//   west :  ub = MIN(ru - cb*(c1h*mut + c2h), 0)     [outbound only]
//   east :  ub = MAX(ru + cb*(c1h*mut + c2h), 0)
//   tend += -rdx * ub * (one-sided du at the boundary)
//
// with ru = (c1h*mut + c2h)*u/msfuy coupled by the boundary CELL's column mass
// (WRF's muu at the domain edge under the zero-gradient mu ghost copy).
// The radiation speed cb comes from the launcher (dycore.OPEN_CB): WRF's
// cb = 25 m/s per share/module_model_constants.F:47, adjudicated over the
// plan's original 30 (the published KW78 value) -- the local WRF source is
// authoritative.  The advection kernels' open-BC loop bounds
// (advection.cu, Task 11 prerequisite) exclude the boundary-normal
// advection at these faces -- the radiative term stands in for it -- while
// whatever WRF retains there (e.g. u's vertical advection when only x is
// open) is already accumulated in the tendency this term now adds to;
// Task 9's REPLACE semantics dropped those retained terms, a linear-order
// error under base-state shear with boundary-normal flow.
//
// w_damp is WRF's vertical-velocity limiter, transcribed from WRF 4.7.1
// dyn_em/module_big_step_utilities_em.F w_damp (w_damping = 1, map
// factors 1): on interior w levels, where the vertical Courant number
//   vert_cfl = |ww/(c1f*mut + c2f) * rdnw[k] * dt|
// exceeds w_damp_on, the coupled w tendency gets
//   -SIGN(1, w) * w_alpha * (vert_cfl - w_crit_cfl) * (c1f*mut + c2f).
// WRF sets w_damp_on = w_crit_cfl when zadvect_implicit > 0 and w_beta
// (1.0) otherwise, and measures the excess from the namelist w_crit_cfl
// either way.  Both arrive as arguments (dycore.w_damp_onset, A165).

// WRF share/module_model_constants.F:88.
#define W_DAMP_ALPHA 0.3f

extern "C" __global__
void open_u_radiative(real* __restrict__ ru_t,          // (nz, ny, nx+1)
                      const real* __restrict__ u,       // (nz, ny, nx+1) t*
                      const real* __restrict__ mup,     // (ny, nx)
                      const real* __restrict__ mub2d,   // (ny, nx)
                      const real* __restrict__ c1h,     // (nz,)
                      const real* __restrict__ c2h,     // (nz,)
                      const real* __restrict__ msfu,    // (ny, nx+1)
                      real rdx, real cb,
                      int nz, int ny, int nx, int has_msf)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;      // one per (k, j)
    if (t >= nz * ny) return;
    int k = t / ny, j = t - k * ny;
    int nxf = nx + 1;

    size_t cw = (size_t)j * nx;                          // west cell (j, 0)
    real mw = c1h[k] * (mub2d[cw] + mup[cw]) + c2h[k];
    size_t f0 = I3S(k, j, 0, ny, nxf);
    size_t f1 = I3S(k, j, 1, ny, nxf);
    real ub = fminf(mw * u[f0] - cb * mw, 0.0f);
    if (has_msf)
        ub = fminf(__fdiv_rn(mw * u[f0], msfu[(size_t)j * nxf]) - cb * mw, 0.0f);
    ru_t[f0] += -rdx * ub * (u[f1] - u[f0]);

    size_t ce = (size_t)j * nx + (nx - 1);               // east cell
    real me = c1h[k] * (mub2d[ce] + mup[ce]) + c2h[k];
    size_t fe = I3S(k, j, nx, ny, nxf);
    size_t fi = I3S(k, j, nx - 1, ny, nxf);
    ub = fmaxf(me * u[fe] + cb * me, 0.0f);
    if (has_msf)
        ub = fmaxf(__fdiv_rn(me * u[fe], msfu[(size_t)j * nxf + nx]) + cb * me, 0.0f);
    ru_t[fe] += -rdx * ub * (u[fe] - u[fi]);
}

extern "C" __global__
void open_v_radiative(real* __restrict__ rv_t,          // (nz, ny+1, nx)
                      const real* __restrict__ v,       // (nz, ny+1, nx) t*
                      const real* __restrict__ mup,     // (ny, nx)
                      const real* __restrict__ mub2d,   // (ny, nx)
                      const real* __restrict__ c1h,     // (nz,)
                      const real* __restrict__ c2h,     // (nz,)
                      const real* __restrict__ msfv,    // (ny+1, nx)
                      real rdy, real cb,
                      int nz, int ny, int nx, int has_msf)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;      // one per (k, i)
    if (t >= nz * nx) return;
    int k = t / nx, i = t - k * nx;
    int nyf = ny + 1;

    size_t cs = (size_t)i;                               // south cell (0, i)
    real ms = c1h[k] * (mub2d[cs] + mup[cs]) + c2h[k];
    size_t f0 = I3S(k, 0, i, nyf, nx);
    size_t f1 = I3S(k, 1, i, nyf, nx);
    real vb = fminf(ms * v[f0] - cb * ms, 0.0f);
    if (has_msf)
        vb = fminf(__fdiv_rn(ms * v[f0], msfv[i]) - cb * ms, 0.0f);
    rv_t[f0] += -rdy * vb * (v[f1] - v[f0]);

    size_t cn = (size_t)(ny - 1) * nx + i;               // north cell
    real mn = c1h[k] * (mub2d[cn] + mup[cn]) + c2h[k];
    size_t fn = I3S(k, ny, i, nyf, nx);
    size_t fi = I3S(k, ny - 1, i, nyf, nx);
    vb = fmaxf(mn * v[fn] + cb * mn, 0.0f);
    if (has_msf)
        vb = fmaxf(__fdiv_rn(mn * v[fn], msfv[(size_t)ny * nx + i]) + cb * mn, 0.0f);
    rv_t[fn] += -rdy * vb * (v[fn] - v[fi]);
}

extern "C" __global__
void w_damp(real* __restrict__ rw_t,                    // (nz+1, ny, nx)
            const real* __restrict__ ww,                // (nz+1, ny, nx) Omega
            const real* __restrict__ w,                 // (nz+1, ny, nx) t*
            const real* __restrict__ mup,               // (ny, nx)
            const real* __restrict__ mub2d,             // (ny, nx)
            const real* __restrict__ c1f,               // (nz+1,)
            const real* __restrict__ c2f,               // (nz+1,)
            const real* __restrict__ rdnw,              // (nz,)
            real dt, real w_damp_on, real w_crit_cfl,
            int nz, int ny, int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;      // interior w levels
    int plane = ny * nx;
    if (t >= (nz - 1) * plane) return;
    int k = t / plane + 1;                              // k = 1 .. nz-1
    int c = t - (k - 1) * plane;

    // Preserve WRF's REAL operator boundaries. An FMA here can move the
    // CFL across the strict 1/2 onset and turn the limiter on or off.
    real m = __fadd_rn(__fmul_rn(c1f[k], __fadd_rn(mub2d[c], mup[c])), c2f[k]);
    size_t ix = (size_t)k * plane + c;
    real vert_cfl = fabsf(__fmul_rn(__fmul_rn(__fdiv_rn(ww[ix], m), rdnw[k]), dt));
    if (vert_cfl > w_damp_on) {
        real damp = __fmul_rn(copysignf(1.0f, w[ix]), W_DAMP_ALPHA);
        damp = __fmul_rn(damp, __fsub_rn(vert_cfl, w_crit_cfl));
        rw_t[ix] = __fsub_rn(rw_t[ix], __fmul_rn(damp, m));
    }
}


// ---------------------------------------------------------------------
// w_cfl_stat -- MEASUREMENT ONLY, off unless GPUWM_WRF_CFL_PROBE is set.
//
// Reduces the SAME vert_cfl the w_damp kernel above computes and discards
//   vert_cfl = |ww/(c1f*mut + c2f) * rdnw[k] * dt|
// over the SAME index range (k = 1 .. nz-1, WRF's k = 2 .. kde-1), so the
// number it reports is WRF's max_vert_cfl (module_big_step_utilities_em.F
// :2646) and not health.cu's geometric |w|/dz, which sweeps every level
// including the thin surface layer and is a different quantity on a
// different scale.  That distinction is the whole reason this exists:
// target_cfl grades the eta-coordinate form.
//
// out[0] = max vert_cfl as float bits (non-negative floats order as uint)
// out[1] = cells ABOVE w_damp_on (strict, matching w_damp above)
// out[2] = cells visited
// out[3] = max horiz_cfl, WRF's max(|u*rdx*msfux*dt|, |v*rdy*msfvy*dt|)
// out[4 + b] = vert_cfl HISTOGRAM, b = 0..CFL_HIST_BINS-1
//
// Why the histogram, when out[0] already has the maximum: out[0] is an
// ORDER STATISTIC.  One cell decides it, so a single rogue column sets
// the timestep for the whole domain, and a controller reading it cannot
// tell "the flow got faster everywhere" from "one grid point is having a
// moment".  That ambiguity is what upstream's +5% per-step growth clamp
// exists to survive (adapt_timestep_em.F:174) -- it is a rate limiter
// standing in for a distribution nobody could afford to measure.
//
// A histogram makes the distribution free.  Bin b counts cells with
// vert_cfl in [b/CFL_HIST_SCALE, (b+1)/CFL_HIST_SCALE), with the top bin
// absorbing everything at or above CFL_HIST_BINS/CFL_HIST_SCALE = 2.0 --
// and NaN too, since every comparison against it is false.  Suffix-sum
// the bins on the host and any upper quantile reads straight off.
//
// This costs one shared-memory atomicAdd per thread and CFL_HIST_BINS
// global ones per BLOCK.  The frame stays 0 B, which is the property
// that keeps this kernel free of the local-memory reservation law.
#define CFL_HIST_BINS  32
#define CFL_HIST_SCALE 16.0f            // bin width 1/16; range [0, 2)
__device__ __forceinline__
void w_cfl_stat_impl(const real* __restrict__ ww,            // (nz+1, ny, nx)
                const real* __restrict__ mup,           // (ny, nx)
                const real* __restrict__ mub2d,         // (ny, nx)
                const real* __restrict__ c1f,           // (nz+1,)
                const real* __restrict__ c2f,           // (nz+1,)
                const real* __restrict__ rdnw,          // (nz,)
                const real* __restrict__ u,             // (nz, ny, nx+1)
                const real* __restrict__ v,             // (nz, ny+1, nx)
                const real* __restrict__ msfu,          // (ny, nx+1)
                const real* __restrict__ msfv,          // (ny+1, nx)
                unsigned int* __restrict__ out,         // (4 + CFL_HIST_BINS,)
                real dt, real rdx, real rdy, real w_damp_on,
                int nz, int ny, int nx, int i0, int i1, int j0, int j1)
{
    __shared__ unsigned int block_max;
    __shared__ unsigned int block_hmax;
    __shared__ unsigned int block_hits;
    __shared__ unsigned int block_hist[CFL_HIST_BINS];
    if (threadIdx.x == 0) { block_max = 0u; block_hmax = 0u; block_hits = 0u; }
    // Strided so the clear does not assume blockDim.x >= CFL_HIST_BINS.
    for (int b = threadIdx.x; b < CFL_HIST_BINS; b += blockDim.x)
        block_hist[b] = 0u;
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = ny * nx;
    int width = i1 - i0;
    int owned_plane = width * (j1 - j0);
    unsigned int bits = 0u;
    unsigned int hbits = 0u;
    unsigned int hit = 0u;
    int bin = -1;                        // -1 = this thread has no cell
    if (t < (nz - 1) * owned_plane) {
        int k = t / owned_plane + 1;                    // k = 1 .. nz-1
        int cell = t - (k - 1) * owned_plane;
        int j = j0 + cell / width, i = i0 + cell % width;
        int c = j * nx + i;
        real m = __fadd_rn(__fmul_rn(c1f[k], __fadd_rn(mub2d[c], mup[c])), c2f[k]);
        size_t ix = (size_t)k * plane + c;
        real vert_cfl = fabsf(__fmul_rn(__fmul_rn(__fdiv_rn(ww[ix], m), rdnw[k]), dt));
        bits = __float_as_uint(vert_cfl);
        hit = (vert_cfl > w_damp_on) ? 1u : 0u;
        // Written as "< top ? floor : last" rather than a min(), so that a
        // NaN lands in the overflow bin instead of wherever a comparison
        // with an undefined float cast happens to send it: every compare
        // against NaN is false, so NaN takes the else arm by construction.
        bin = (vert_cfl < CFL_HIST_BINS / CFL_HIST_SCALE)
            ? (int)(vert_cfl * CFL_HIST_SCALE) : (CFL_HIST_BINS - 1);

        // WRF's max_horiz_cfl, same loop and same index range
        // (module_big_step_utilities_em.F:2662):
        //   horiz_cfl = max(|u*rdx*msfux*dt|, |v*rdy*msfvy*dt|)
        // BOTH components and BOTH map factors.  health.cu reads u only
        // and no map factor, which is why it cannot stand in for this.
        size_t iu = (size_t)k * ny * (nx + 1) + (size_t)j * (nx + 1) + i;
        size_t iv = (size_t)k * (ny + 1) * nx + (size_t)j * nx + i;
        real cu = fabsf(u[iu] * rdx * msfu[(size_t)j * (nx + 1) + i] * dt);
        real cv = fabsf(v[iv] * rdy * msfv[(size_t)j * nx + i] * dt);
        hbits = __float_as_uint(fmaxf(cu, cv));
    }
    atomicMax(&block_max, bits);
    atomicMax(&block_hmax, hbits);
    atomicAdd(&block_hits, hit);
    if (bin >= 0) atomicAdd(&block_hist[bin], 1u);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(&out[0], block_max);
        atomicAdd(&out[1], block_hits);
        atomicAdd(&out[2], (unsigned int)min(blockDim.x,
                                             max(0, (nz - 1) * owned_plane
                                                    - blockIdx.x * (int)blockDim.x)));
        atomicMax(&out[3], block_hmax);
    }
    // Outside the thread-0 branch: CFL_HIST_BINS global atomics per block,
    // spread over the block's threads rather than serialised on one of
    // them.  The empty-bin test matters -- most bins are zero on a healthy
    // step, and skipping them keeps this to a handful of real atomics.
    for (int b = threadIdx.x; b < CFL_HIST_BINS; b += blockDim.x)
        if (block_hist[b]) atomicAdd(&out[4 + b], block_hist[b]);
}


extern "C" __global__
void w_cfl_stat(const real* __restrict__ ww,            // (nz+1, ny, nx)
                const real* __restrict__ mup,           // (ny, nx)
                const real* __restrict__ mub2d,         // (ny, nx)
                const real* __restrict__ c1f,           // (nz+1,)
                const real* __restrict__ c2f,           // (nz+1,)
                const real* __restrict__ rdnw,          // (nz,)
                const real* __restrict__ u,             // (nz, ny, nx+1)
                const real* __restrict__ v,             // (nz, ny+1, nx)
                const real* __restrict__ msfu,          // (ny, nx+1)
                const real* __restrict__ msfv,          // (ny+1, nx)
                unsigned int* __restrict__ out,         // (4 + CFL_HIST_BINS,)
                real dt, real rdx, real rdy, real w_damp_on,
                int nz, int ny, int nx)
{
    w_cfl_stat_impl(ww, mup, mub2d, c1f, c2f, rdnw, u, v, msfu, msfv, out, dt, rdx, rdy, w_damp_on, nz, ny, nx, 0, nx, 0, ny);
}

extern "C" __global__
void w_cfl_stat_window(const real* __restrict__ ww,            // (nz+1, ny, nx)
                const real* __restrict__ mup,           // (ny, nx)
                const real* __restrict__ mub2d,         // (ny, nx)
                const real* __restrict__ c1f,           // (nz+1,)
                const real* __restrict__ c2f,           // (nz+1,)
                const real* __restrict__ rdnw,          // (nz,)
                const real* __restrict__ u,             // (nz, ny, nx+1)
                const real* __restrict__ v,             // (nz, ny+1, nx)
                const real* __restrict__ msfu,          // (ny, nx+1)
                const real* __restrict__ msfv,          // (ny+1, nx)
                unsigned int* __restrict__ out,         // (4 + CFL_HIST_BINS,)
                real dt, real rdx, real rdy, real w_damp_on,
                int nz, int ny, int nx, int i0, int i1, int j0, int j1)
{
    w_cfl_stat_impl(ww, mup, mub2d, c1f, c2f, rdnw, u, v, msfu, msfv, out, dt, rdx, rdy, w_damp_on, nz, ny, nx, i0, i1, j0, j1);
}
