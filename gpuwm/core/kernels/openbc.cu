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
// with ru = (c1h*mut + c2h)*u coupled by the boundary CELL's column mass
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
// w_damp is WRF's vertical-velocity limiter, transcribed from
// dyn_em/module_big_step_utilities_em.F w_damp (w_damping = 1, non-IEVA,
// map factors 1): on interior w levels, where the vertical Courant number
//   vert_cfl = |ww/(c1f*mut + c2f) * rdnw[k] * dt|
// exceeds w_beta, the coupled w tendency gets
//   -SIGN(1, w) * w_alpha * (vert_cfl - w_crit_cfl) * (c1f*mut + c2f).

// WRF share/module_model_constants.F:88-89; w_crit_cfl is the Registry
// namelist default (Registry.EM_COMMON).
#define W_DAMP_ALPHA 0.3f
#define W_DAMP_BETA  1.0f
#define W_CRIT_CFL   1.0f

extern "C" __global__
void open_u_radiative(real* __restrict__ ru_t,          // (nz, ny, nx+1)
                      const real* __restrict__ u,       // (nz, ny, nx+1) t*
                      const real* __restrict__ mup,     // (ny, nx)
                      const real* __restrict__ mub2d,   // (ny, nx)
                      const real* __restrict__ c1h,     // (nz,)
                      const real* __restrict__ c2h,     // (nz,)
                      real rdx, real cb,
                      int nz, int ny, int nx)
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
    ru_t[f0] += -rdx * ub * (u[f1] - u[f0]);

    size_t ce = (size_t)j * nx + (nx - 1);               // east cell
    real me = c1h[k] * (mub2d[ce] + mup[ce]) + c2h[k];
    size_t fe = I3S(k, j, nx, ny, nxf);
    size_t fi = I3S(k, j, nx - 1, ny, nxf);
    ub = fmaxf(me * u[fe] + cb * me, 0.0f);
    ru_t[fe] += -rdx * ub * (u[fe] - u[fi]);
}

extern "C" __global__
void open_v_radiative(real* __restrict__ rv_t,          // (nz, ny+1, nx)
                      const real* __restrict__ v,       // (nz, ny+1, nx) t*
                      const real* __restrict__ mup,     // (ny, nx)
                      const real* __restrict__ mub2d,   // (ny, nx)
                      const real* __restrict__ c1h,     // (nz,)
                      const real* __restrict__ c2h,     // (nz,)
                      real rdy, real cb,
                      int nz, int ny, int nx)
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
    rv_t[f0] += -rdy * vb * (v[f1] - v[f0]);

    size_t cn = (size_t)(ny - 1) * nx + i;               // north cell
    real mn = c1h[k] * (mub2d[cn] + mup[cn]) + c2h[k];
    size_t fn = I3S(k, ny, i, nyf, nx);
    size_t fi = I3S(k, ny - 1, i, nyf, nx);
    vb = fmaxf(mn * v[fn] + cb * mn, 0.0f);
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
            real dt, int nz, int ny, int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;      // interior w levels
    int plane = ny * nx;
    if (t >= (nz - 1) * plane) return;
    int k = t / plane + 1;                              // k = 1 .. nz-1
    int c = t - (k - 1) * plane;

    real m = c1f[k] * (mub2d[c] + mup[c]) + c2f[k];
    size_t ix = (size_t)k * plane + c;
    real vert_cfl = fabsf(ww[ix] / m * rdnw[k] * dt);
    if (vert_cfl > W_DAMP_BETA) {
        rw_t[ix] -= copysignf(1.0f, w[ix]) * W_DAMP_ALPHA
                    * (vert_cfl - W_CRIT_CFL) * m;
    }
}
