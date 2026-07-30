// gpuwm/core/kernels/diffusion.cu
//
// Constant-K 2nd-order diffusion on coordinate surfaces (WRF diff_opt=1
// style) and the Rayleigh damping layer (WRF damp_opt=3 analogue).
//
// add_diff2 ADDS  kh*(d2f/dx2 + d2f/dy2) + kv*d/dz(df/dz)  into tend for one
// field.  Horizontal second derivatives are taken along the (periodic)
// coordinate surfaces with uniform dx/dy; the vertical derivative is taken
// in physical space using per-level base-state dz supplied as two arrays:
//   rdzf[m] = 1 / (z_center[m+1] - z_center[m])   -- (nlev-1,) face spacings
//   rdzc[k] = 1 / (cell thickness at level k)     -- (nlev,)
// Half-level fields (u, v, theta') get zero-flux top/bottom boundaries; for
// w-staggered fields (wstag = 1) the BC-pinned boundary levels k = 0 and
// k = nlev-1 receive no tendency and rdzc's boundary entries are unused.
//
// Storage is (nlev, nys, nxs) with the periodic core (ny, nx); all stencil
// reads wrap over the core, so the redundant staggered column (nxs = nx+1)
// or row (nys = ny+1) receives a tendency identical to column/row 0 -- the
// same convention as advection.cu.

extern "C" __global__
void add_diff2(const real* __restrict__ f,        // (nlev, nys, nxs)
               real* __restrict__ tend,           // (nlev, nys, nxs) +=
               real kh, real kv,
               real dx_inv2, real dy_inv2,
               const real* __restrict__ rdzf,     // (nlev-1,)
               const real* __restrict__ rdzc,     // (nlev,)
               int nlev, int ny, int nys, int nx, int nxs, int wstag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nxs || j >= nys || k >= nlev) return;
    if (wstag && (k == 0 || k == nlev - 1)) return;  // BC-pinned w levels

    int ic = PERIODIC(i, nx);
    int jc = PERIODIC(j, ny);
    real fc = f[I3S(k, jc, ic, nys, nxs)];
    real lap = kh * (dx_inv2 * (f[I3S(k, jc, PERIODIC(i + 1, nx), nys, nxs)]
                                - 2.0f * fc
                                + f[I3S(k, jc, PERIODIC(i - 1, nx), nys, nxs)])
                   + dy_inv2 * (f[I3S(k, PERIODIC(j + 1, ny), ic, nys, nxs)]
                                - 2.0f * fc
                                + f[I3S(k, PERIODIC(j - 1, ny), ic, nys, nxs)]));
    real fup = (k < nlev - 1)
        ? (f[I3S(k + 1, jc, ic, nys, nxs)] - fc) * rdzf[k] : 0.0f;
    real fdn = (k > 0)
        ? (fc - f[I3S(k - 1, jc, ic, nys, nxs)]) * rdzf[k - 1] : 0.0f;
    lap += kv * (fup - fdn) * rdzc[k];
    tend[I3S(k, j, i, nys, nxs)] += lap;
}

// In-place per-level implicit Rayleigh relaxation toward zero:
// f[k, j, i] *= rdamp[k], with rdamp[k] = 1/(1 + dt*tau_k) precomputed on
// host (tau = 0 => rdamp = 1 below the damping layer).  plane = nys*nxs.
extern "C" __global__
void rayleigh_damp(real* __restrict__ f,
                   const real* __restrict__ rdamp,  // (nlev,)
                   int nlev, int plane)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nlev * plane) return;
    f[idx] *= rdamp[idx / plane];
}
