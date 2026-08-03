// gpuwm/core/kernels/smag2d.cu
//
// The first four entry points retain gpuwm's Phase-2 coordinate-surface /
// flat-metric Smagorinsky oracle API.  They are no longer the production
// km_opt=4 route.  The WRF v4.6.1 metric/stress implementation begins at
// ``WrfSmagGrid`` below.
//
// smag2d_km fills the eddy viscosities at mass points from the horizontal
// deformation of the C-grid winds:
//   D11 = 2*du/dx, D22 = 2*dv/dy at mass points; D12 = du/dy + dv/dx at
//   the cell corners (vorticity points); def2 = 0.25*(D11-D22)^2 +
//   (4-corner average of D12)^2; K_m = min((c_s*mlen)^2*sqrt(def2),
//   10*mlen) with mlen = sqrt(dx*dy); K_h = K_m/prandtl (prandtl = 1/3,
//   WRF share/module_model_constants.F -- scalars get 3x the momentum K).
//
// The four smag_hd_* kernels ADD the mass-coupled variable-K tendency
//   d/dx(mk * df/dx) + d/dy(mk * df/dy),  mk = (c1[k]*mut + c2[k]) * K
// into tend for one field, with K and the coupled mass averaged from mass
// points to the flux faces exactly as the WRF branches ('u', 'v', 'w',
// 'm') -- including the WRF v4.6.1 quirk that the 'v' branch's normal (y)
// fluxes carry NO (c1*mut+c2) factor (see smag_hd_v; every other flux in
// the routine is mass-coupled).  Storage/periodicity conventions match
// diffusion.cu: stencil reads wrap over the periodic core, the redundant
// staggered column/row gets a tendency identical to column/row 0, and the
// BC-pinned boundary w levels receive no tendency.
//
// Open lateral boundaries: smag_hd_u / smag_hd_v take an open_x / open_y
// flag that redirects the ONE cross-boundary read WRF performs honestly.
// WRF's 'u' branch computes the boundary-normal face i_end = ide-1 with
// the TRUE boundary datum field(i+1) = u(ide)
// (module_big_step_utilities_em.F:2786-2787 bounds, 2819 stencil); the
// periodic wrap would read the OPPOSITE (west) boundary instead, so with
// open_x the face nx-1 reads the stored boundary column nx (analogously
// v's face ny-1 reads row ny under open_y; 'v' bounds 2834-2837, stencil
// 2861).  The remaining open-BC trimming -- WRF's excluded width-1 strip
// (ids+1 / ide-1|2 per stagger) -- is applied host-side afterwards
// (gpuwm.core.dycore._zero_open_strips).

#define CHM(c1k, c2k, jj, ii) ((c1k) * mut[(size_t)(jj) * nx + (ii)] + (c2k))

// D12 at the SW corner (jc, ic) of mass cell (jc, ic): u rows jc-1/jc at
// face ic, v columns ic-1/ic at face jc.  __fmul_rn keeps the two products
// out of FMA contraction: deformation-free flow (solid-body rotation) has
// rdy*du == -rdx*dv and must cancel EXACTLY, but a fused rdy*du + rdx*dv
// keeps the first product unrounded and leaves an O(eps) residual.
__device__ __forceinline__
real d12_corner(const real* __restrict__ u, const real* __restrict__ v,
                real rdx, real rdy, int k, int jc, int ic, int ny, int nx)
{
    return __fmul_rn(rdy, u[I3S(k, jc, ic, ny, nx + 1)]
                          - u[I3S(k, PERIODIC(jc - 1, ny), ic, ny, nx + 1)])
         + __fmul_rn(rdx, v[I3S(k, jc, ic, ny + 1, nx)]
                          - v[I3S(k, jc, PERIODIC(ic - 1, nx), ny + 1, nx)]);
}

extern "C" __global__
void smag2d_km(const real* __restrict__ u,     // (nz, ny, nx+1)
               const real* __restrict__ v,     // (nz, ny+1, nx)
               real* __restrict__ xkmh,        // (nz, ny, nx) out: momentum K
               real* __restrict__ xkhh,        // (nz, ny, nx) out: scalar K
               real rdx, real rdy, real mlen, real c_s, real prandtl,
               int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int ip1 = PERIODIC(i + 1, nx);
    int jp1 = PERIODIC(j + 1, ny);
    real d11 = 2.0f * rdx * (u[I3S(k, j, ip1, ny, nx + 1)]
                             - u[I3S(k, j, i, ny, nx + 1)]);
    real d22 = 2.0f * rdy * (v[I3S(k, jp1, i, ny + 1, nx)]
                             - v[I3S(k, j, i, ny + 1, nx)]);
    real s12 = 0.25f * (d12_corner(u, v, rdx, rdy, k, j,   i,   ny, nx)
                        + d12_corner(u, v, rdx, rdy, k, j,   ip1, ny, nx)
                        + d12_corner(u, v, rdx, rdy, k, jp1, i,   ny, nx)
                        + d12_corner(u, v, rdx, rdy, k, jp1, ip1, ny, nx));
    real def2 = 0.25f * (d11 - d22) * (d11 - d22) + s12 * s12;
    real km = c_s * c_s * mlen * mlen * sqrtf(def2);
    km = fminf(km, 10.0f * mlen);
    xkmh[IDX3(k, j, i)] = km;
    xkhh[IDX3(k, j, i)] = km / prandtl;
}

// ---------------------------------------------------------------------------
// WRF horizontal_diffusion, 'm' branch: scalars at mass points (nlev = nz).
// ---------------------------------------------------------------------------
extern "C" __global__
void smag_hd_s(const real* __restrict__ f,     // (nz, ny, nx)
               const real* __restrict__ xk,    // (nz, ny, nx) at mass points
               const real* __restrict__ mut,   // (ny, nx) total dry mass
               const real* __restrict__ c1,    // (nz,)
               const real* __restrict__ c2,    // (nz,)
               real rdx, real rdy,
               real* __restrict__ tend,        // (nz, ny, nx) +=
               int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int ip1 = PERIODIC(i + 1, nx), im1 = PERIODIC(i - 1, nx);
    int jp1 = PERIODIC(j + 1, ny), jm1 = PERIODIC(j - 1, ny);
    real c1k = c1[k], c2k = c2[k];
    real fc = f[IDX3(k, j, i)];
    real mkrdxm = 0.5f * (xk[IDX3(k, j, i)] + xk[IDX3(k, j, im1)])
                * 0.5f * (CHM(c1k, c2k, j, i) + CHM(c1k, c2k, j, im1)) * rdx;
    real mkrdxp = 0.5f * (xk[IDX3(k, j, ip1)] + xk[IDX3(k, j, i)])
                * 0.5f * (CHM(c1k, c2k, j, ip1) + CHM(c1k, c2k, j, i)) * rdx;
    real mkrdym = 0.5f * (xk[IDX3(k, j, i)] + xk[IDX3(k, jm1, i)])
                * 0.5f * (CHM(c1k, c2k, j, i) + CHM(c1k, c2k, jm1, i)) * rdy;
    real mkrdyp = 0.5f * (xk[IDX3(k, jp1, i)] + xk[IDX3(k, j, i)])
                * 0.5f * (CHM(c1k, c2k, jp1, i) + CHM(c1k, c2k, j, i)) * rdy;
    tend[IDX3(k, j, i)] +=
          rdx * (mkrdxp * (f[IDX3(k, j, ip1)] - fc)
                 - mkrdxm * (fc - f[IDX3(k, j, im1)]))
        + rdy * (mkrdyp * (f[IDX3(k, jp1, i)] - fc)
                 - mkrdym * (fc - f[IDX3(k, jm1, i)]));
}

// ---------------------------------------------------------------------------
// WRF horizontal_diffusion, 'u' branch: u points (nlev = nz, nxs = nx+1).
// x fluxes live at mass centers, y fluxes at corners (4-point K/mass avg).
// ---------------------------------------------------------------------------
extern "C" __global__
void smag_hd_u(const real* __restrict__ u,     // (nz, ny, nx+1)
               const real* __restrict__ xk,    // (nz, ny, nx)
               const real* __restrict__ mut,   // (ny, nx)
               const real* __restrict__ c1,    // (nz,)
               const real* __restrict__ c2,    // (nz,)
               real rdx, real rdy,
               real* __restrict__ tend,        // (nz, ny, nx+1) +=
               int nz, int ny, int nx, int open_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;

    int ic = PERIODIC(i, nx);                  // u-face core column
    int im1 = PERIODIC(ic - 1, nx);            // mass column left of face
    int ip1 = PERIODIC(ic + 1, nx);
    // Open east boundary: WRF computes face ide-1 with the true boundary
    // datum field(i+1) = u(ide) (see header); column nx stores it -- the
    // periodic wrap would read the west boundary instead.
    if (open_x && ic == nx - 1) ip1 = nx;
    int jp1 = PERIODIC(j + 1, ny), jm1 = PERIODIC(j - 1, ny);
    real c1k = c1[k], c2k = c2[k];
    real uc = u[I3S(k, j, ic, ny, nx + 1)];
    real mkrdxm = CHM(c1k, c2k, j, im1) * xk[IDX3(k, j, im1)] * rdx;
    real mkrdxp = CHM(c1k, c2k, j, ic) * xk[IDX3(k, j, ic)] * rdx;
    real mkrdym = 0.25f * (CHM(c1k, c2k, j, ic) + CHM(c1k, c2k, jm1, ic)
                           + CHM(c1k, c2k, jm1, im1) + CHM(c1k, c2k, j, im1))
                * 0.25f * (xk[IDX3(k, j, ic)] + xk[IDX3(k, jm1, ic)]
                           + xk[IDX3(k, jm1, im1)] + xk[IDX3(k, j, im1)])
                * rdy;
    real mkrdyp = 0.25f * (CHM(c1k, c2k, jp1, ic) + CHM(c1k, c2k, j, ic)
                           + CHM(c1k, c2k, j, im1) + CHM(c1k, c2k, jp1, im1))
                * 0.25f * (xk[IDX3(k, jp1, ic)] + xk[IDX3(k, j, ic)]
                           + xk[IDX3(k, j, im1)] + xk[IDX3(k, jp1, im1)])
                * rdy;
    tend[I3S(k, j, i, ny, nx + 1)] +=
          rdx * (mkrdxp * (u[I3S(k, j, ip1, ny, nx + 1)] - uc)
                 - mkrdxm * (uc - u[I3S(k, j, im1, ny, nx + 1)]))
        + rdy * (mkrdyp * (u[I3S(k, jp1, ic, ny, nx + 1)] - uc)
                 - mkrdym * (uc - u[I3S(k, jm1, ic, ny, nx + 1)]));
}

// ---------------------------------------------------------------------------
// WRF horizontal_diffusion, 'v' branch: v points (nlev = nz, nys = ny+1).
// ---------------------------------------------------------------------------
extern "C" __global__
void smag_hd_v(const real* __restrict__ v,     // (nz, ny+1, nx)
               const real* __restrict__ xk,    // (nz, ny, nx)
               const real* __restrict__ mut,   // (ny, nx)
               const real* __restrict__ c1,    // (nz,)
               const real* __restrict__ c2,    // (nz,)
               real rdx, real rdy,
               real* __restrict__ tend,        // (nz, ny+1, nx) +=
               int nz, int ny, int nx, int open_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;

    int jc = PERIODIC(j, ny);                  // v-face core row
    int jm1 = PERIODIC(jc - 1, ny);            // mass row south of face
    int jp1 = PERIODIC(jc + 1, ny);
    // Open north boundary: WRF computes face jde-1 with the true boundary
    // datum field(j+1) = v(jde) (see header); row ny stores it -- the
    // periodic wrap would read the south boundary instead.
    if (open_y && jc == ny - 1) jp1 = ny;
    int ip1 = PERIODIC(i + 1, nx), im1 = PERIODIC(i - 1, nx);
    real c1k = c1[k], c2k = c2[k];
    real vc = v[I3S(k, jc, i, ny + 1, nx)];
    // WRF v4.6.1 quirk, transcribed exactly: the 'v' branch's normal (y)
    // fluxes are NOT mass-coupled (module_big_step_utilities_em.F:2854-2855,
    //   mkrdym=(msfty(i,j-1)/msftx(i,j-1))*xkmhd(i,k,j-1)*rdy
    //   mkrdyp=(msfty(i,j)/msftx(i,j))*xkmhd(i,k,j)*rdy
    // -- no (c1(k)*MUT+c2(k)) factor), unlike the 'u' branch's normal (x)
    // fluxes at lines 2801-2802 and every other flux in the routine.
    real mkrdym = xk[IDX3(k, jm1, i)] * rdy;
    real mkrdyp = xk[IDX3(k, jc, i)] * rdy;
    real mkrdxm = 0.25f * (CHM(c1k, c2k, jc, i) + CHM(c1k, c2k, jc, im1)
                           + CHM(c1k, c2k, jm1, im1) + CHM(c1k, c2k, jm1, i))
                * 0.25f * (xk[IDX3(k, jc, i)] + xk[IDX3(k, jc, im1)]
                           + xk[IDX3(k, jm1, im1)] + xk[IDX3(k, jm1, i)])
                * rdx;
    real mkrdxp = 0.25f * (CHM(c1k, c2k, jc, ip1) + CHM(c1k, c2k, jc, i)
                           + CHM(c1k, c2k, jm1, i) + CHM(c1k, c2k, jm1, ip1))
                * 0.25f * (xk[IDX3(k, jc, ip1)] + xk[IDX3(k, jc, i)]
                           + xk[IDX3(k, jm1, i)] + xk[IDX3(k, jm1, ip1)])
                * rdx;
    tend[I3S(k, j, i, ny + 1, nx)] +=
          rdx * (mkrdxp * (v[I3S(k, jc, ip1, ny + 1, nx)] - vc)
                 - mkrdxm * (vc - v[I3S(k, jc, im1, ny + 1, nx)]))
        + rdy * (mkrdyp * (v[I3S(k, jp1, i, ny + 1, nx)] - vc)
                 - mkrdym * (vc - v[I3S(k, jm1, i, ny + 1, nx)]));
}

// ---------------------------------------------------------------------------
// WRF horizontal_diffusion, 'w' branch: w points (nlev = nz+1); K averaged
// from the half levels straddling each w level; c1/c2 are c1f/c2f.  The
// BC-pinned boundary levels k = 0 and k = nz receive no tendency.
// ---------------------------------------------------------------------------
extern "C" __global__
void smag_hd_w(const real* __restrict__ w,     // (nz+1, ny, nx)
               const real* __restrict__ xk,    // (nz, ny, nx)
               const real* __restrict__ mut,   // (ny, nx)
               const real* __restrict__ c1,    // (nz+1,)
               const real* __restrict__ c2,    // (nz+1,)
               real rdx, real rdy,
               real* __restrict__ tend,        // (nz+1, ny, nx) +=
               int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz + 1) return;
    if (k == 0 || k == nz) return;             // BC-pinned w levels

    int ip1 = PERIODIC(i + 1, nx), im1 = PERIODIC(i - 1, nx);
    int jp1 = PERIODIC(j + 1, ny), jm1 = PERIODIC(j - 1, ny);
    real c1k = c1[k], c2k = c2[k];
    real wc = w[IDX3(k, j, i)];
    real mkrdxm = 0.5f * (CHM(c1k, c2k, j, i) + CHM(c1k, c2k, j, im1))
                * 0.25f * (xk[IDX3(k, j, i)] + xk[IDX3(k, j, im1)]
                           + xk[IDX3(k - 1, j, i)] + xk[IDX3(k - 1, j, im1)])
                * rdx;
    real mkrdxp = 0.5f * (CHM(c1k, c2k, j, ip1) + CHM(c1k, c2k, j, i))
                * 0.25f * (xk[IDX3(k, j, ip1)] + xk[IDX3(k, j, i)]
                           + xk[IDX3(k - 1, j, ip1)] + xk[IDX3(k - 1, j, i)])
                * rdx;
    real mkrdym = 0.5f * (CHM(c1k, c2k, j, i) + CHM(c1k, c2k, jm1, i))
                * 0.25f * (xk[IDX3(k, j, i)] + xk[IDX3(k, jm1, i)]
                           + xk[IDX3(k - 1, j, i)] + xk[IDX3(k - 1, jm1, i)])
                * rdy;
    real mkrdyp = 0.5f * (CHM(c1k, c2k, jp1, i) + CHM(c1k, c2k, j, i))
                * 0.25f * (xk[IDX3(k, jp1, i)] + xk[IDX3(k, j, i)]
                           + xk[IDX3(k - 1, jp1, i)] + xk[IDX3(k - 1, j, i)])
                * rdy;
    tend[IDX3(k, j, i)] +=
          rdx * (mkrdxp * (w[IDX3(k, j, ip1)] - wc)
                 - mkrdxm * (wc - w[IDX3(k, j, im1)]))
        + rdy * (mkrdyp * (w[IDX3(k, jp1, i)] - wc)
                 - mkrdym * (wc - w[IDX3(k, jm1, i)]));
}

// ---------------------------------------------------------------------------
// Production WRF v4.6.1 diff_opt=2 / km_opt=4 path.
//
// The kernels above are retained as the Phase-2 flat-coordinate oracle API.
// Production uses the kernels below.  They derive compute_diff_metrics on
// demand; D11/D22/D12 briefly borrow dead carrying-buffer prefixes, avoiding
// dedicated domain-volume allocations.  The path retains WRF's metric-aware
// deformation, local mixing length/slope limiter, tensor momentum stress,
// and terrain-following scalar fluxes
// (module_diffusion_em.F:17-1190,1934-2044,3118-3999,6882-7130).
// ---------------------------------------------------------------------------

struct WrfSmagGrid {
    const real* u;
    const real* v;
    const real* w;
    const real* php;
    const real* phb;
    const real* alt;
    const real* qv;
    const real* msft;
    const real* msfu;
    const real* msfv;
    const real* fnm;
    const real* fnp;
    const real* dn;
    const real* dnw;
    real rdx, rdy, dx, dy;
    real cf1, cf2, cf3;
    int nz, ny, nx;
    int phb3d, boundary_x, boundary_y, moist;
};

__device__ __forceinline__
WrfSmagGrid wrf_grid(const real* u, const real* v, const real* w,
                     const real* php, const real* phb, const real* alt,
                     const real* qv,
                     const real* msft, const real* msfu, const real* msfv,
                     const real* fnm, const real* fnp,
                     const real* dn, const real* dnw,
                     real rdx, real rdy, real dx, real dy,
                     real cf1, real cf2, real cf3,
                     int nz, int ny, int nx, int phb3d,
                     int boundary_x, int boundary_y, int moist)
{
    WrfSmagGrid q;
    q.u = u; q.v = v; q.w = w; q.php = php; q.phb = phb; q.alt = alt;
    q.qv = qv;
    q.msft = msft; q.msfu = msfu; q.msfv = msfv;
    q.fnm = fnm; q.fnp = fnp; q.dn = dn; q.dnw = dnw;
    q.rdx = rdx; q.rdy = rdy; q.dx = dx; q.dy = dy;
    q.cf1 = cf1; q.cf2 = cf2; q.cf3 = cf3;
    q.nz = nz; q.ny = ny; q.nx = nx; q.phb3d = phb3d;
    q.boundary_x = boundary_x; q.boundary_y = boundary_y; q.moist = moist;
    return q;
}

__device__ __forceinline__ int wrf_ix(const WrfSmagGrid& q, int i)
{
    if (!q.boundary_x) return PERIODIC(i, q.nx);
    return i < 0 ? 0 : (i >= q.nx ? q.nx - 1 : i);
}

__device__ __forceinline__ int wrf_iy(const WrfSmagGrid& q, int j)
{
    if (!q.boundary_y) return PERIODIC(j, q.ny);
    return j < 0 ? 0 : (j >= q.ny ? q.ny - 1 : j);
}

__device__ __forceinline__ int wrf_iu(const WrfSmagGrid& q, int i)
{
    if (!q.boundary_x) return PERIODIC(i, q.nx);
    return i < 0 ? 0 : (i > q.nx ? q.nx : i);
}

__device__ __forceinline__ int wrf_jv(const WrfSmagGrid& q, int j)
{
    if (!q.boundary_y) return PERIODIC(j, q.ny);
    return j < 0 ? 0 : (j > q.ny ? q.ny : j);
}

__device__ __forceinline__
real wrf_phi(const WrfSmagGrid& q, int kw, int j, int i)
{
    int ii = wrf_ix(q, i), jj = wrf_iy(q, j);
    kw = kw < 0 ? 0 : (kw > q.nz ? q.nz : kw);
    real base = q.phb3d ? q.phb[I3(kw, jj, ii, q.ny, q.nx)] : q.phb[kw];
    return base + q.php[I3(kw, jj, ii, q.ny, q.nx)];
}

__device__ __forceinline__
real wrf_rdzw(const WrfSmagGrid& q, int k, int j, int i)
{
    k = k < 0 ? 0 : (k >= q.nz ? q.nz - 1 : k);
    return G / (wrf_phi(q, k + 1, j, i) - wrf_phi(q, k, j, i));
}

__device__ __forceinline__
real wrf_rdz(const WrfSmagGrid& q, int kw, int j, int i)
{
    if (kw <= 0)
        return 2.0f * G / (wrf_phi(q, 1, j, i) - wrf_phi(q, 0, j, i));
    if (kw >= q.nz)
        return 2.0f * G / (wrf_phi(q, q.nz, j, i)
                         - wrf_phi(q, q.nz - 1, j, i));
    return 2.0f * G / (wrf_phi(q, kw + 1, j, i)
                     - wrf_phi(q, kw - 1, j, i));
}

__device__ __forceinline__
real wrf_zx(const WrfSmagGrid& q, int kw, int j, int iface)
{
    return q.rdx * (wrf_phi(q, kw, j, iface)
                  - wrf_phi(q, kw, j, iface - 1)) / G;
}

__device__ __forceinline__
real wrf_zy(const WrfSmagGrid& q, int kw, int jface, int i)
{
    return q.rdy * (wrf_phi(q, kw, jface, i)
                  - wrf_phi(q, kw, jface - 1, i)) / G;
}

__device__ __forceinline__
real wrf_rho(const WrfSmagGrid& q, int k, int j, int i)
{
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    size_t h = I3(k, jj, ii, q.ny, q.nx);
    // module_big_step_utilities_em.F:4856.  ALT is dry specific volume;
    // WRF's diffusion density includes vapor loading.  The dry branch must
    // not read qv so it stays bitwise identical to the former 1/ALT path.
    if (q.moist) return (1.0f + q.qv[h]) / q.alt[h];
    return 1.0f / q.alt[h];
}

__device__ __forceinline__
real wrf_uhat(const WrfSmagGrid& q, int k, int j, int iface)
{
    int jj = wrf_iy(q, j), ii = wrf_iu(q, iface);
    return q.u[I3S(k, jj, ii, q.ny, q.nx + 1)]
         / q.msfu[(size_t)jj * (q.nx + 1) + ii];
}

__device__ __forceinline__
real wrf_vhat(const WrfSmagGrid& q, int k, int jface, int i)
{
    int jj = wrf_jv(q, jface), ii = wrf_ix(q, i);
    return q.v[I3S(k, jj, ii, q.ny + 1, q.nx)]
         / q.msfv[(size_t)jj * q.nx + ii];
}

__device__ __forceinline__
real wrf_what(const WrfSmagGrid& q, int kw, int j, int i)
{
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    return q.w[I3(kw, jj, ii, q.ny, q.nx)]
         / q.msft[(size_t)jj * q.nx + ii];
}

__device__ __forceinline__
real wrf_full_weights(const WrfSmagGrid& q, int kw,
                      real p0, real p1, real p2,
                      real plast, real pprev,
                      real pcur, real pbelow)
{
    if (kw <= 0) return q.cf1 * p0 + q.cf2 * p1 + q.cf3 * p2;
    if (kw >= q.nz) {
        real cft2 = -0.5f * q.dnw[q.nz - 1] / q.dn[q.nz - 1];
        return (1.0f - cft2) * plast + cft2 * pprev;
    }
    return q.fnm[kw] * pcur + q.fnp[kw] * pbelow;
}

__device__ __forceinline__
real wrf_u_w_xcenter(const WrfSmagGrid& q, int kw, int j, int i)
{
    real p0 = wrf_uhat(q, 0, j, i) + wrf_uhat(q, 0, j, i + 1);
    real p1 = wrf_uhat(q, 1, j, i) + wrf_uhat(q, 1, j, i + 1);
    real p2 = wrf_uhat(q, 2, j, i) + wrf_uhat(q, 2, j, i + 1);
    real pl = wrf_uhat(q, q.nz - 1, j, i) + wrf_uhat(q, q.nz - 1, j, i + 1);
    real pp = wrf_uhat(q, q.nz - 2, j, i) + wrf_uhat(q, q.nz - 2, j, i + 1);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_uhat(q, kc, j, i) + wrf_uhat(q, kc, j, i + 1);
    real pb = wrf_uhat(q, kb, j, i) + wrf_uhat(q, kb, j, i + 1);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_v_w_ycenter(const WrfSmagGrid& q, int kw, int j, int i)
{
    real p0 = wrf_vhat(q, 0, j, i) + wrf_vhat(q, 0, j + 1, i);
    real p1 = wrf_vhat(q, 1, j, i) + wrf_vhat(q, 1, j + 1, i);
    real p2 = wrf_vhat(q, 2, j, i) + wrf_vhat(q, 2, j + 1, i);
    real pl = wrf_vhat(q, q.nz - 1, j, i) + wrf_vhat(q, q.nz - 1, j + 1, i);
    real pp = wrf_vhat(q, q.nz - 2, j, i) + wrf_vhat(q, q.nz - 2, j + 1, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_vhat(q, kc, j, i) + wrf_vhat(q, kc, j + 1, i);
    real pb = wrf_vhat(q, kb, j, i) + wrf_vhat(q, kb, j + 1, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_u_w_corner(const WrfSmagGrid& q, int kw, int j, int i)
{
    real p0 = wrf_uhat(q, 0, j - 1, i) + wrf_uhat(q, 0, j, i);
    real p1 = wrf_uhat(q, 1, j - 1, i) + wrf_uhat(q, 1, j, i);
    real p2 = wrf_uhat(q, 2, j - 1, i) + wrf_uhat(q, 2, j, i);
    real pl = wrf_uhat(q, q.nz - 1, j - 1, i) + wrf_uhat(q, q.nz - 1, j, i);
    real pp = wrf_uhat(q, q.nz - 2, j - 1, i) + wrf_uhat(q, q.nz - 2, j, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_uhat(q, kc, j - 1, i) + wrf_uhat(q, kc, j, i);
    real pb = wrf_uhat(q, kb, j - 1, i) + wrf_uhat(q, kb, j, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_v_w_corner(const WrfSmagGrid& q, int kw, int j, int i)
{
    real p0 = wrf_vhat(q, 0, j, i - 1) + wrf_vhat(q, 0, j, i);
    real p1 = wrf_vhat(q, 1, j, i - 1) + wrf_vhat(q, 1, j, i);
    real p2 = wrf_vhat(q, 2, j, i - 1) + wrf_vhat(q, 2, j, i);
    real pl = wrf_vhat(q, q.nz - 1, j, i - 1) + wrf_vhat(q, q.nz - 1, j, i);
    real pp = wrf_vhat(q, q.nz - 2, j, i - 1) + wrf_vhat(q, q.nz - 2, j, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_vhat(q, kc, j, i - 1) + wrf_vhat(q, kc, j, i);
    real pb = wrf_vhat(q, kb, j, i - 1) + wrf_vhat(q, kb, j, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_defor11(const WrfSmagGrid& q, int k, int j, int i)
{
    real tmpzx = 0.25f * (wrf_zx(q, k, j, i) + wrf_zx(q, k, j, i + 1)
                            + wrf_zx(q, k + 1, j, i)
                            + wrf_zx(q, k + 1, j, i + 1));
    real slope = (wrf_u_w_xcenter(q, k + 1, j, i)
                - wrf_u_w_xcenter(q, k, j, i))
               * tmpzx * wrf_rdzw(q, k, j, i);
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    real mm = q.msft[(size_t)jj * q.nx + ii];
    return 2.0f * mm * mm
         * (q.rdx * (wrf_uhat(q, k, j, i + 1) - wrf_uhat(q, k, j, i))
            - slope);
}

__device__ __forceinline__
real wrf_defor22(const WrfSmagGrid& q, int k, int j, int i)
{
    real tmpzy = 0.25f * (wrf_zy(q, k, j, i) + wrf_zy(q, k, j + 1, i)
                            + wrf_zy(q, k + 1, j, i)
                            + wrf_zy(q, k + 1, j + 1, i));
    real slope = (wrf_v_w_ycenter(q, k + 1, j, i)
                - wrf_v_w_ycenter(q, k, j, i))
               * tmpzy * wrf_rdzw(q, k, j, i);
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    real mm = q.msft[(size_t)jj * q.nx + ii];
    return 2.0f * mm * mm
         * (q.rdy * (wrf_vhat(q, k, j + 1, i) - wrf_vhat(q, k, j, i))
            - slope);
}

__device__ __forceinline__
real wrf_defor12(const WrfSmagGrid& q, int k, int j, int i)
{
    int iu = wrf_iu(q, i), jm = wrf_iy(q, j - 1), jc = wrf_iy(q, j);
    int iv = wrf_ix(q, i), im = wrf_ix(q, i - 1), jv = wrf_jv(q, j);
    real mm = 0.25f * (q.msfu[(size_t)jm * (q.nx + 1) + iu]
                       + q.msfu[(size_t)jc * (q.nx + 1) + iu])
            * (q.msfv[(size_t)jv * q.nx + im]
               + q.msfv[(size_t)jv * q.nx + iv]);
    real rr = wrf_rdzw(q, k, j, i) + wrf_rdzw(q, k, j, i - 1)
            + wrf_rdzw(q, k, j - 1, i - 1) + wrf_rdzw(q, k, j - 1, i);
    real tmpzy = 0.25f * (wrf_zy(q, k, j, i - 1) + wrf_zy(q, k, j, i)
                            + wrf_zy(q, k + 1, j, i - 1)
                            + wrf_zy(q, k + 1, j, i));
    real uslope = (wrf_u_w_corner(q, k + 1, j, i)
                 - wrf_u_w_corner(q, k, j, i)) * 0.25f * tmpzy * rr;
    real tmpzx = 0.25f * (wrf_zx(q, k, j - 1, i) + wrf_zx(q, k, j, i)
                            + wrf_zx(q, k + 1, j - 1, i)
                            + wrf_zx(q, k + 1, j, i));
    real vslope = (wrf_v_w_corner(q, k + 1, j, i)
                 - wrf_v_w_corner(q, k, j, i)) * 0.25f * tmpzx * rr;
    return mm * (q.rdy * (wrf_uhat(q, k, j, i)
                         - wrf_uhat(q, k, j - 1, i)) - uslope
               + q.rdx * (wrf_vhat(q, k, j, i)
                         - wrf_vhat(q, k, j, i - 1)) - vslope);
}

__device__ __forceinline__
real wrf_k(const WrfSmagGrid& q, const real* xk, int k, int j, int i)
{
    return xk[I3(k, wrf_iy(q, j), wrf_ix(q, i), q.ny, q.nx)];
}

extern "C" __global__
void wrf_smag_deform(const real* u, const real* v, const real* w,
                     const real* php, const real* phb, const real* alt,
                     const real* qv,
                     const real* msft, const real* msfu, const real* msfv,
                     const real* fnm, const real* fnp,
                     const real* dn, const real* dnw,
                     real rdx, real rdy, real dx, real dy,
                     real cf1, real cf2, real cf3,
                     int moist,
                     real* d11, real* d22, real* d12,
                     int nz, int ny, int nx, int phb3d,
                     int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WrfSmagGrid q = wrf_grid(u, v, w, php, phb, alt, qv, msft, msfu, msfv,
                             fnm, fnp, dn, dnw, rdx, rdy, dx, dy,
                             cf1, cf2, cf3, nz, ny, nx, phb3d,
                             boundary_x, boundary_y, moist);
    d11[IDX3(k, j, i)] = wrf_defor11(q, k, j, i);
    d22[IDX3(k, j, i)] = wrf_defor22(q, k, j, i);
    d12[IDX3(k, j, i)] = wrf_defor12(q, k, j, i);
}

__device__ __forceinline__
real wrf_d(const WrfSmagGrid& q, const real* d, int k, int j, int i)
{
    return d[I3(k, wrf_iy(q, j), wrf_ix(q, i), q.ny, q.nx)];
}

extern "C" __global__
void wrf_smag2d_km(const real* u, const real* v, const real* w,
                   const real* php, const real* phb, const real* alt,
                   const real* qv,
                   const real* msft, const real* msfu, const real* msfv,
                   const real* fnm, const real* fnp,
                   const real* dn, const real* dnw,
                   real rdx, real rdy, real dx, real dy,
                   real cf1, real cf2, real cf3, int moist,
                   real c_s, real prandtl,
                   const real* d11a, const real* d22a, const real* d12a,
                   real* xkmh, real* xkhh,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WrfSmagGrid q = wrf_grid(u, v, w, php, phb, alt, qv, msft, msfu, msfv,
                             fnm, fnp, dn, dnw, rdx, rdy, dx, dy,
                             cf1, cf2, cf3, nz, ny, nx, phb3d,
                             boundary_x, boundary_y, moist);
    real d11 = wrf_d(q, d11a, k, j, i);
    real d22 = wrf_d(q, d22a, k, j, i);
    real d12 = 0.25f * (wrf_d(q, d12a, k, j, i)
                         + wrf_d(q, d12a, k, j + 1, i)
                         + wrf_d(q, d12a, k, j, i + 1)
                         + wrf_d(q, d12a, k, j + 1, i + 1));
    real strain = sqrtf(0.25f * (d11 - d22) * (d11 - d22) + d12 * d12);
    real map = msft[(size_t)j * nx + i];
    real dxm = dx / map, dym = dy / map;
    real mlen = sqrtf(dxm * dym);
    real km = fminf(c_s * c_s * mlen * mlen * strain, 10.0f * mlen);
    real sx = 0.25f * (fabsf(wrf_zx(q, k, j, i))
                        + fabsf(wrf_zx(q, k, j, i + 1))
                        + fabsf(wrf_zx(q, k + 1, j, i))
                        + fabsf(wrf_zx(q, k + 1, j, i + 1)))
            * wrf_rdzw(q, k, j, i) * dxm;
    real sy = 0.25f * (fabsf(wrf_zy(q, k, j, i))
                        + fabsf(wrf_zy(q, k, j + 1, i))
                        + fabsf(wrf_zy(q, k + 1, j, i))
                        + fabsf(wrf_zy(q, k + 1, j + 1, i)))
            * wrf_rdzw(q, k, j, i) * dym;
    real alpha = fmaxf(sqrtf(sx * sx + sy * sy), 1.0f);
    real def_limit = fmaxf(10.0f / mlen, 1.0e-3f);
    km /= strain > def_limit ? alpha * alpha : alpha;
    xkmh[IDX3(k, j, i)] = km;
    xkhh[IDX3(k, j, i)] = km / prandtl;
}

extern "C" __global__
void wrf_smag_km_bc(real* xkmh, real* xkhh, int nz, int ny, int nx,
                    int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    // smag2d_km writes only ids+1:ide-2 / jds+1:jde-2 at a physical
    // boundary.  phy_bc(set_physical_bc3d(..., 't')) fills OUTSIDE ghost
    // cells from the active boundary; it does not copy the first interior K
    // into that boundary.  xkmh/xkhh are cold-zeroed WRF state arrays (and
    // real74's damp_opt=3 never writes the omitted row), so the logical
    // outer active row remains exactly zero for the life of the domain.
    if ((boundary_x && (i == 0 || i == nx - 1))
            || (boundary_y && (j == 0 || j == ny - 1))) {
        xkmh[IDX3(k, j, i)] = 0.0f;
        xkhh[IDX3(k, j, i)] = 0.0f;
    }
}

__device__ __forceinline__
real wrf_tau11(const WrfSmagGrid& q, const real* km, const real* d11,
               int k, int j, int i)
{
    return -wrf_rho(q, k, j, i) * wrf_k(q, km, k, j, i)
           * wrf_d(q, d11, k, j, i);
}

__device__ __forceinline__
real wrf_tau22(const WrfSmagGrid& q, const real* km, const real* d22,
               int k, int j, int i)
{
    return -wrf_rho(q, k, j, i) * wrf_k(q, km, k, j, i)
           * wrf_d(q, d22, k, j, i);
}

__device__ __forceinline__
real wrf_tau12(const WrfSmagGrid& q, const real* km, const real* d12,
               int k, int j, int i)
{
    real rhoavg = 0.25f * (wrf_rho(q, k, j, i)
                            + wrf_rho(q, k, j, i - 1)
                            + wrf_rho(q, k, j - 1, i - 1)
                            + wrf_rho(q, k, j - 1, i));
    real kavg = 0.25f * (wrf_k(q, km, k, j, i)
                          + wrf_k(q, km, k, j, i - 1)
                          + wrf_k(q, km, k, j - 1, i - 1)
                          + wrf_k(q, km, k, j - 1, i));
    return -rhoavg * kavg * wrf_d(q, d12, k, j, i);
}

__device__ __forceinline__
real wrf_tau11_uavg(const WrfSmagGrid& q, const real* km, const real* d11,
                    int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_tau11(q, km, d11, kw, j, i - 1)
                                 + wrf_tau11(q, km, d11, kw, j, i))
                  + q.fnp[kw] * (wrf_tau11(q, km, d11, kw - 1, j, i - 1)
                                 + wrf_tau11(q, km, d11, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_tau12_uavg(const WrfSmagGrid& q, const real* km, const real* d12,
                    int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_tau12(q, km, d12, kw, j + 1, i)
                                 + wrf_tau12(q, km, d12, kw, j, i))
                  + q.fnp[kw] * (wrf_tau12(q, km, d12, kw - 1, j + 1, i)
                                 + wrf_tau12(q, km, d12, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_tau12_vavg(const WrfSmagGrid& q, const real* km, const real* d12,
                    int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_tau12(q, km, d12, kw, j, i + 1)
                                 + wrf_tau12(q, km, d12, kw, j, i))
                  + q.fnp[kw] * (wrf_tau12(q, km, d12, kw - 1, j, i + 1)
                                 + wrf_tau12(q, km, d12, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_tau22_vavg(const WrfSmagGrid& q, const real* km, const real* d22,
                    int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_tau22(q, km, d22, kw, j - 1, i)
                                 + wrf_tau22(q, km, d22, kw, j, i))
                  + q.fnp[kw] * (wrf_tau22(q, km, d22, kw - 1, j - 1, i)
                                 + wrf_tau22(q, km, d22, kw - 1, j, i)));
}

#define WRF_SMAG_GRID_ARGS                                                     \
    const real* u, const real* v, const real* w,                              \
    const real* php, const real* phb, const real* alt, const real* qv,        \
    const real* msft, const real* msfu, const real* msfv,                     \
    const real* fnm, const real* fnp, const real* dn, const real* dnw,         \
    real rdx, real rdy, real dx, real dy, real cf1, real cf2, real cf3,        \
    int moist

#define WRF_SMAG_MAKE_GRID                                                     \
    WrfSmagGrid q = wrf_grid(u, v, w, php, phb, alt, qv, msft, msfu, msfv,    \
                             fnm, fnp, dn, dnw, rdx, rdy, dx, dy,              \
                             cf1, cf2, cf3, nz, ny, nx, phb3d,                 \
                             boundary_x, boundary_y, moist)

extern "C" __global__
void wrf_smag_hd_u(WRF_SMAG_GRID_ARGS, const real* km,
                   const real* d11, const real* d12, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    int ic = wrf_ix(q, i), im = wrf_ix(q, i - 1);
    int iu = wrf_iu(q, i), jj = wrf_iy(q, j);
    real msf = msfu[(size_t)jj * (nx + 1) + iu];
    real tmpdz = 0.5f * (1.0f / wrf_rdzw(q, k, j, ic)
                          + 1.0f / wrf_rdzw(q, k, j, im));
    real zx_u = 0.5f * (wrf_zx(q, k, j, i) + wrf_zx(q, k + 1, j, i));
    real zy_u = 0.125f * (
        wrf_zy(q, k, j, im) + wrf_zy(q, k, j, ic)
      + wrf_zy(q, k, j + 1, im) + wrf_zy(q, k, j + 1, ic)
      + wrf_zy(q, k + 1, j, im) + wrf_zy(q, k + 1, j, ic)
      + wrf_zy(q, k + 1, j + 1, im) + wrf_zy(q, k + 1, j + 1, ic));
    real divh = msf * rdx * (wrf_tau11(q, km, d11, k, j, ic)
                              - wrf_tau11(q, km, d11, k, j, im))
              + msf * rdy * (wrf_tau12(q, km, d12, k, j + 1, i)
                              - wrf_tau12(q, km, d12, k, j, i));
    real divz = msf * zx_u * (wrf_tau11_uavg(q, km, d11, k + 1, j, i)
                              - wrf_tau11_uavg(q, km, d11, k, j, i)) / tmpdz
              + msf * zy_u * (wrf_tau12_uavg(q, km, d12, k + 1, j, i)
                              - wrf_tau12_uavg(q, km, d12, k, j, i)) / tmpdz;
    tend[I3S(k, j, i, ny, nx + 1)] += G * tmpdz / dnw[k] * (divh - divz);
}

extern "C" __global__
void wrf_smag_hd_v(WRF_SMAG_GRID_ARGS, const real* km,
                   const real* d22, const real* d12, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    int jc = wrf_iy(q, j), jm = wrf_iy(q, j - 1);
    int jv = wrf_jv(q, j), ii = wrf_ix(q, i);
    real msf = msfv[(size_t)jv * nx + ii];
    real tmpdz = 0.5f * (1.0f / wrf_rdzw(q, k, jc, i)
                          + 1.0f / wrf_rdzw(q, k, jm, i));
    real zx_v = 0.125f * (
        wrf_zx(q, k, jm, i) + wrf_zx(q, k, jm, i + 1)
      + wrf_zx(q, k, jc, i) + wrf_zx(q, k, jc, i + 1)
      + wrf_zx(q, k + 1, jm, i) + wrf_zx(q, k + 1, jm, i + 1)
      + wrf_zx(q, k + 1, jc, i) + wrf_zx(q, k + 1, jc, i + 1));
    real zy_v = 0.5f * (wrf_zy(q, k, j, i) + wrf_zy(q, k + 1, j, i));
    real divh = msf * rdy * (wrf_tau22(q, km, d22, k, jc, i)
                              - wrf_tau22(q, km, d22, k, jm, i))
              + msf * rdx * (wrf_tau12(q, km, d12, k, j, i + 1)
                              - wrf_tau12(q, km, d12, k, j, i));
    real divz = msf * zx_v * (wrf_tau12_vavg(q, km, d12, k + 1, j, i)
                              - wrf_tau12_vavg(q, km, d12, k, j, i)) / tmpdz
              + msf * zy_v * (wrf_tau22_vavg(q, km, d22, k + 1, j, i)
                              - wrf_tau22_vavg(q, km, d22, k, j, i)) / tmpdz;
    tend[I3S(k, j, i, ny + 1, nx)] += G * tmpdz / dnw[k] * (divh - divz);
}

__device__ __forceinline__
real wrf_w_xavg(const WrfSmagGrid& q, int k, int j, int i)
{
    return 0.25f * (wrf_what(q, k, j, i) + wrf_what(q, k + 1, j, i)
                     + wrf_what(q, k, j, i - 1)
                     + wrf_what(q, k + 1, j, i - 1));
}

__device__ __forceinline__
real wrf_w_yavg(const WrfSmagGrid& q, int k, int j, int i)
{
    return 0.25f * (wrf_what(q, k, j, i) + wrf_what(q, k + 1, j, i)
                     + wrf_what(q, k, j - 1, i)
                     + wrf_what(q, k + 1, j - 1, i));
}

__device__ __forceinline__
real wrf_defor13(const WrfSmagGrid& q, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    int jj = wrf_iy(q, j), iu = wrf_iu(q, i);
    real msf = q.msfu[(size_t)jj * (q.nx + 1) + iu];
    real rz = 0.5f * (wrf_rdz(q, kw, j, i) + wrf_rdz(q, kw, j, i - 1));
    real slope = (wrf_w_xavg(q, kw, j, i) - wrf_w_xavg(q, kw - 1, j, i))
               * wrf_zx(q, kw, j, i) * rz;
    real dwdx = msf * msf * (q.rdx * (wrf_what(q, kw, j, i)
                                      - wrf_what(q, kw, j, i - 1)) - slope);
    int iface = wrf_iu(q, i);
    real dudz = (q.u[I3S(kw, jj, iface, q.ny, q.nx + 1)]
                 - q.u[I3S(kw - 1, jj, iface, q.ny, q.nx + 1)]) * rz;
    return dwdx + dudz;
}

__device__ __forceinline__
real wrf_defor23(const WrfSmagGrid& q, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    int jv = wrf_jv(q, j), ii = wrf_ix(q, i);
    real msf = q.msfv[(size_t)jv * q.nx + ii];
    real rz = 0.5f * (wrf_rdz(q, kw, j, i) + wrf_rdz(q, kw, j - 1, i));
    real slope = (wrf_w_yavg(q, kw, j, i) - wrf_w_yavg(q, kw - 1, j, i))
               * wrf_zy(q, kw, j, i) * rz;
    real dwdy = msf * msf * (q.rdy * (wrf_what(q, kw, j, i)
                                      - wrf_what(q, kw, j - 1, i)) - slope);
    real dvdz = (q.v[I3S(kw, jv, ii, q.ny + 1, q.nx)]
                 - q.v[I3S(kw - 1, jv, ii, q.ny + 1, q.nx)]) * rz;
    return dwdy + dvdz;
}

__device__ __forceinline__
real wrf_tau13(const WrfSmagGrid& q, const real* km, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    real rhoavg = 0.5f * (q.fnm[kw] * (wrf_rho(q, kw, j, i - 1)
                                        + wrf_rho(q, kw, j, i))
                           + q.fnp[kw] * (wrf_rho(q, kw - 1, j, i - 1)
                                        + wrf_rho(q, kw - 1, j, i)));
    real kavg = 0.5f * (q.fnm[kw] * (wrf_k(q, km, kw, j, i - 1)
                                      + wrf_k(q, km, kw, j, i))
                         + q.fnp[kw] * (wrf_k(q, km, kw - 1, j, i - 1)
                                      + wrf_k(q, km, kw - 1, j, i)));
    return -rhoavg * kavg * wrf_defor13(q, kw, j, i);
}

__device__ __forceinline__
real wrf_tau23(const WrfSmagGrid& q, const real* km, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    real rhoavg = 0.5f * (q.fnm[kw] * (wrf_rho(q, kw, j - 1, i)
                                        + wrf_rho(q, kw, j, i))
                           + q.fnp[kw] * (wrf_rho(q, kw - 1, j - 1, i)
                                        + wrf_rho(q, kw - 1, j, i)));
    real kavg = 0.5f * (q.fnm[kw] * (wrf_k(q, km, kw, j - 1, i)
                                      + wrf_k(q, km, kw, j, i))
                         + q.fnp[kw] * (wrf_k(q, km, kw - 1, j - 1, i)
                                      + wrf_k(q, km, kw - 1, j, i)));
    return -rhoavg * kavg * wrf_defor23(q, kw, j, i);
}

// WRF v4.6.1 vertical_diffusion_2, selected by diff_opt=2 when the PBL is
// off (dyn_em/module_first_rk_step_part2.F:1008-1074).  For km_opt=4
// smag2d_km makes xkmv=xkmh and xkhv=0
// (dyn_em/module_diffusion_em.F:2018-2023), so the
// interior vertical operator has momentum stresses only.  Scalar interior
// fluxes are identically zero; their explicit surface fluxes are below.
extern "C" __global__
void wrf_smag_vd_u(WRF_SMAG_GRID_ARGS, const real* km, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    real lower = wrf_tau13(q, km, k, j, i);
    real upper = wrf_tau13(q, km, k + 1, j, i);
    tend[I3S(k, j, i, ny, nx + 1)] += G * (upper - lower) / dnw[k];
}

extern "C" __global__
void wrf_smag_vd_v(WRF_SMAG_GRID_ARGS, const real* km, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    real lower = wrf_tau23(q, km, k, j, i);
    real upper = wrf_tau23(q, km, k + 1, j, i);
    tend[I3S(k, j, i, ny + 1, nx)] += G * (upper - lower) / dnw[k];
}

__device__ __forceinline__
real wrf_tau33(const WrfSmagGrid& q, const real* km, int k, int j, int i)
{
    if (k < 0 || k >= q.nz) return 0.0f;
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    real defor33 = 2.0f * (
        q.w[I3(k + 1, jj, ii, q.ny, q.nx)]
        - q.w[I3(k, jj, ii, q.ny, q.nx)])
        * wrf_rdzw(q, k, j, i);
    return -wrf_rho(q, k, j, i) * wrf_k(q, km, k, j, i) * defor33;
}

extern "C" __global__
void wrf_smag_vd_w(WRF_SMAG_GRID_ARGS, const real* km, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kw = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || kw <= 0 || kw >= nz) return;
    WRF_SMAG_MAKE_GRID;
    tend[I3(kw, j, i, ny, nx)] += G * (
        wrf_tau33(q, km, kw, j, i)
        - wrf_tau33(q, km, kw - 1, j, i)) / dn[kw];
}

// vertical_diffusion_2's isfflx=1 wall-stress branch
// (dyn_em/module_diffusion_em.F:4182-4250).  USTM is the friction velocity
// without the convective-wind correction, exactly the grid%ustm field WRF
// passes.
extern "C" __global__
void wrf_smag_surface_u(WRF_SMAG_GRID_ARGS, const real* ustm, int active,
                        real* tend, int nz, int ny, int nx, int phb3d,
                        int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (!active || i >= nx + 1 || j >= ny) return;
    WRF_SMAG_MAKE_GRID;
    int ic = wrf_ix(q, i), im = wrf_ix(q, i - 1), jj = wrf_iy(q, j);
    real vv = 0.25f * (
        q.v[I3S(0, wrf_jv(q, j), ic, ny + 1, nx)]
        + q.v[I3S(0, wrf_jv(q, j + 1), ic, ny + 1, nx)]
        + q.v[I3S(0, wrf_jv(q, j), im, ny + 1, nx)]
        + q.v[I3S(0, wrf_jv(q, j + 1), im, ny + 1, nx)]);
    real uu = q.u[I3S(0, jj, wrf_iu(q, i), ny, nx + 1)];
    real speed = sqrtf(uu * uu + vv * vv) + 1.0e-15f;
    real ustar = 0.5f * (
        ustm[(size_t)jj * nx + ic] + ustm[(size_t)jj * nx + im]);
    real stress = ustar * ustar * uu / speed;
    real rhoavg = 0.5f * (
        wrf_rho(q, 0, j, ic) + wrf_rho(q, 0, j, im));
    tend[I3S(0, j, i, ny, nx + 1)] += G * stress * rhoavg / dnw[0];
}

extern "C" __global__
void wrf_smag_surface_v(WRF_SMAG_GRID_ARGS, const real* ustm, int active,
                        real* tend, int nz, int ny, int nx, int phb3d,
                        int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (!active || i >= nx || j >= ny + 1) return;
    WRF_SMAG_MAKE_GRID;
    int jc = wrf_iy(q, j), jm = wrf_iy(q, j - 1), ii = wrf_ix(q, i);
    real uu = 0.25f * (
        q.u[I3S(0, jc, wrf_iu(q, i), ny, nx + 1)]
        + q.u[I3S(0, jc, wrf_iu(q, i + 1), ny, nx + 1)]
        + q.u[I3S(0, jm, wrf_iu(q, i), ny, nx + 1)]
        + q.u[I3S(0, jm, wrf_iu(q, i + 1), ny, nx + 1)]);
    real vv = q.v[I3S(0, wrf_jv(q, j), ii, ny + 1, nx)];
    real speed = sqrtf(vv * vv + uu * uu) + 1.0e-15f;
    real ustar = 0.5f * (
        ustm[(size_t)jc * nx + ii] + ustm[(size_t)jm * nx + ii]);
    real stress = ustar * ustar * vv / speed;
    real rhoavg = 0.5f * (
        wrf_rho(q, 0, jc, i) + wrf_rho(q, 0, jm, i));
    tend[I3S(0, j, i, ny + 1, nx)] += G * stress * rhoavg / dnw[0];
}

// vertical_diffusion_2's isfflx=1 heat and isfflx=1/2 water-vapour fluxes
// (dyn_em/module_diffusion_em.F:4288-4310,4384-4400).  ArWen's admitted
// identity fixes use_theta_m=0, so the theta-m cross term is intentionally
// absent.  ``apply_heat`` gates the hflux CASE(1) arm and ``apply_moist``
// the qflux CASE(1,2) arm; isfflx=0/2 heat is the constant-flux kernel
// below (hflux CASE(0,2)), which is mutually exclusive with apply_heat.
extern "C" __global__
void wrf_smag_surface_scalars(
        WRF_SMAG_GRID_ARGS, const real* hfx, const real* qfx,
        int apply_heat, int apply_moist,
        real* rth, real* rqv, int nz, int ny, int nx, int phb3d,
        int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((!apply_heat && !apply_moist) || i >= nx || j >= ny) return;
    WRF_SMAG_MAKE_GRID;
    size_t h = (size_t)j * nx + i;
    real vapor = moist ? qv[I3(0, j, i, ny, nx)] : 0.0f;
    real cpm = CP * (1.0f + 0.8f * vapor);
    if (apply_heat)
        rth[I3(0, j, i, ny, nx)] -= G * hfx[h] / cpm / dnw[0];
    if (apply_moist && moist)
        rqv[I3(0, j, i, ny, nx)] -= G * qfx[h] / dnw[0];
}

__device__ __forceinline__
real wrf_tau13_mavg(const WrfSmagGrid& q, const real* km, int k, int j, int i)
{
    return 0.25f * (wrf_tau13(q, km, k, j, i)
                     + wrf_tau13(q, km, k, j, i + 1)
                     + wrf_tau13(q, km, k + 1, j, i)
                     + wrf_tau13(q, km, k + 1, j, i + 1));
}

__device__ __forceinline__
real wrf_tau23_mavg(const WrfSmagGrid& q, const real* km, int k, int j, int i)
{
    return 0.25f * (wrf_tau23(q, km, k, j, i)
                     + wrf_tau23(q, km, k, j + 1, i)
                     + wrf_tau23(q, km, k + 1, j, i)
                     + wrf_tau23(q, km, k + 1, j + 1, i));
}

extern "C" __global__
void wrf_smag_hd_w(WRF_SMAG_GRID_ARGS, const real* km, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int kw = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || kw >= nz + 1 || kw == 0 || kw == nz) return;
    WRF_SMAG_MAKE_GRID;
    int ii = wrf_ix(q, i), jj = wrf_iy(q, j);
    real map = msft[(size_t)jj * nx + ii];
    real zx_w = 0.5f * (wrf_zx(q, kw, j, i) + wrf_zx(q, kw, j, i + 1));
    real zy_w = 0.5f * (wrf_zy(q, kw, j, i) + wrf_zy(q, kw, j + 1, i));
    real rz = wrf_rdz(q, kw, j, i);
    real divh = map * rdx * (wrf_tau13(q, km, kw, j, i + 1)
                              - wrf_tau13(q, km, kw, j, i))
              + map * rdy * (wrf_tau23(q, km, kw, j + 1, i)
                              - wrf_tau23(q, km, kw, j, i));
    real divz = map * rz * (zx_w * (wrf_tau13_mavg(q, km, kw, j, i)
                                     - wrf_tau13_mavg(q, km, kw - 1, j, i))
                           + zy_w * (wrf_tau23_mavg(q, km, kw, j, i)
                                     - wrf_tau23_mavg(q, km, kw - 1, j, i)));
    tend[I3(kw, j, i, ny, nx)] += G / (dn[kw] * rz) * (divh - divz);
}

struct WrfScalarField {
    const real* f;
    const real* thb;
    int full_theta;
    int thb3d;
};

// share/module_model_constants.F:37.  Keep the production source
// self-contained as well as compatible with gpuwm's generated preamble.
static constexpr real WRF_T0 = 300.0f;

__device__ __forceinline__
real wrf_scalar(const WrfSmagGrid& q, const WrfScalarField& s,
                int k, int j, int i)
{
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    real value = s.f[I3(k, jj, ii, q.ny, q.nx)];
    if (s.full_theta) {
        size_t h = s.thb3d ? I3(k, jj, ii, q.ny, q.nx) : (size_t)k;
        // WRF horizontal_diffusion_2 receives grid%t_2 = theta - T0.
        // gpuwm stores thp = theta - thb, so reconstruct the WRF field at
        // the point of use without allocating a full-domain temporary.
        value = __fsub_rn(__fadd_rn(s.thb[h], value), WRF_T0);
    }
    return value;
}

__device__ __forceinline__
real wrf_scalar_w_xface(const WrfSmagGrid& q, const WrfScalarField& s,
                        int kw, int j, int i)
{
    real p0 = wrf_scalar(q, s, 0, j, i - 1) + wrf_scalar(q, s, 0, j, i);
    real p1 = wrf_scalar(q, s, 1, j, i - 1) + wrf_scalar(q, s, 1, j, i);
    real p2 = wrf_scalar(q, s, 2, j, i - 1) + wrf_scalar(q, s, 2, j, i);
    real pl = wrf_scalar(q, s, q.nz - 1, j, i - 1)
            + wrf_scalar(q, s, q.nz - 1, j, i);
    real pp = wrf_scalar(q, s, q.nz - 2, j, i - 1)
            + wrf_scalar(q, s, q.nz - 2, j, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_scalar(q, s, kc, j, i - 1) + wrf_scalar(q, s, kc, j, i);
    real pb = wrf_scalar(q, s, kb, j, i - 1) + wrf_scalar(q, s, kb, j, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_scalar_w_yface(const WrfSmagGrid& q, const WrfScalarField& s,
                        int kw, int j, int i)
{
    real p0 = wrf_scalar(q, s, 0, j - 1, i) + wrf_scalar(q, s, 0, j, i);
    real p1 = wrf_scalar(q, s, 1, j - 1, i) + wrf_scalar(q, s, 1, j, i);
    real p2 = wrf_scalar(q, s, 2, j - 1, i) + wrf_scalar(q, s, 2, j, i);
    real pl = wrf_scalar(q, s, q.nz - 1, j - 1, i)
            + wrf_scalar(q, s, q.nz - 1, j, i);
    real pp = wrf_scalar(q, s, q.nz - 2, j - 1, i)
            + wrf_scalar(q, s, q.nz - 2, j, i);
    int kc = kw >= q.nz ? q.nz - 1 : kw;
    int kb = kc > 0 ? kc - 1 : 0;
    real pc = wrf_scalar(q, s, kc, j - 1, i) + wrf_scalar(q, s, kc, j, i);
    real pb = wrf_scalar(q, s, kb, j - 1, i) + wrf_scalar(q, s, kb, j, i);
    return 0.5f * wrf_full_weights(q, kw, p0, p1, p2, pl, pp, pc, pb);
}

__device__ __forceinline__
real wrf_h1(const WrfSmagGrid& q, const WrfScalarField& s, const real* kh,
            int k, int j, int i)
{
    real rhoavg = 0.5f * (wrf_rho(q, k, j, i - 1)
                           + wrf_rho(q, k, j, i));
    real kavg = 0.5f * (wrf_k(q, kh, k, j, i - 1)
                         + wrf_k(q, kh, k, j, i));
    real tmpzx = 0.5f * (wrf_zx(q, k, j, i) + wrf_zx(q, k + 1, j, i));
    real rdzu = 2.0f / (1.0f / wrf_rdzw(q, k, j, i)
                         + 1.0f / wrf_rdzw(q, k, j, i - 1));
    int jj = wrf_iy(q, j), iu = wrf_iu(q, i);
    real map = q.msfu[(size_t)jj * (q.nx + 1) + iu];
    real grad = q.rdx * (wrf_scalar(q, s, k, j, i)
                         - wrf_scalar(q, s, k, j, i - 1))
              - tmpzx * (wrf_scalar_w_xface(q, s, k + 1, j, i)
                         - wrf_scalar_w_xface(q, s, k, j, i)) * rdzu;
    return -map * rhoavg * kavg * grad;
}

__device__ __forceinline__
real wrf_h2(const WrfSmagGrid& q, const WrfScalarField& s, const real* kh,
            int k, int j, int i)
{
    real rhoavg = 0.5f * (wrf_rho(q, k, j - 1, i)
                           + wrf_rho(q, k, j, i));
    real kavg = 0.5f * (wrf_k(q, kh, k, j - 1, i)
                         + wrf_k(q, kh, k, j, i));
    real tmpzy = 0.5f * (wrf_zy(q, k, j, i) + wrf_zy(q, k + 1, j, i));
    real rdzv = 2.0f / (1.0f / wrf_rdzw(q, k, j, i)
                         + 1.0f / wrf_rdzw(q, k, j - 1, i));
    int jv = wrf_jv(q, j), ii = wrf_ix(q, i);
    real map = q.msfv[(size_t)jv * q.nx + ii];
    real grad = q.rdy * (wrf_scalar(q, s, k, j, i)
                         - wrf_scalar(q, s, k, j - 1, i))
              - tmpzy * (wrf_scalar_w_yface(q, s, k + 1, j, i)
                         - wrf_scalar_w_yface(q, s, k, j, i)) * rdzv;
    return -map * rhoavg * kavg * grad;
}

__device__ __forceinline__
real wrf_fx(const WrfSmagGrid& q, const real* fx, int k, int j, int i)
{
    int jj = wrf_iy(q, j), ii = wrf_iu(q, i);
    return fx[I3S(k, jj, ii, q.ny, q.nx + 1)];
}

__device__ __forceinline__
real wrf_fy(const WrfSmagGrid& q, const real* fy, int k, int j, int i)
{
    int jj = wrf_jv(q, j), ii = wrf_ix(q, i);
    return fy[I3S(k, jj, ii, q.ny + 1, q.nx)];
}

__device__ __forceinline__
real wrf_h1_mavg(const WrfSmagGrid& q, const real* fx,
                 int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_fx(q, fx, kw, j, i + 1)
                                 + wrf_fx(q, fx, kw, j, i))
                  + q.fnp[kw] * (wrf_fx(q, fx, kw - 1, j, i + 1)
                                 + wrf_fx(q, fx, kw - 1, j, i)));
}

__device__ __forceinline__
real wrf_h2_mavg(const WrfSmagGrid& q, const real* fy,
                 int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    return 0.5f * (q.fnm[kw] * (wrf_fy(q, fy, kw, j + 1, i)
                                 + wrf_fy(q, fy, kw, j, i))
                  + q.fnp[kw] * (wrf_fy(q, fy, kw - 1, j + 1, i)
                                 + wrf_fy(q, fy, kw - 1, j, i)));
}

extern "C" __global__
void wrf_smag_flux_s(WRF_SMAG_GRID_ARGS, const real* f, const real* kh,
                     const real* thb, int full_theta, int thb3d,
                     real* fx, real* fy, int nz, int ny, int nx, int phb3d,
                     int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    WrfScalarField s = {f, thb, full_theta, thb3d};
    if (i < nx + 1 && j < ny)
        fx[I3S(k, j, i, ny, nx + 1)] = wrf_h1(q, s, kh, k, j, i);
    if (i < nx && j < ny + 1)
        fy[I3S(k, j, i, ny + 1, nx)] = wrf_h2(q, s, kh, k, j, i);
}

extern "C" __global__
void wrf_smag_hd_s(WRF_SMAG_GRID_ARGS, const real* fx, const real* fy,
                   real* tend, int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    int ii = wrf_ix(q, i), jj = wrf_iy(q, j);
    real map = msft[(size_t)jj * nx + ii];
    real zx_m = 0.25f * (wrf_zx(q, k, j, i) + wrf_zx(q, k, j, i + 1)
                          + wrf_zx(q, k + 1, j, i)
                          + wrf_zx(q, k + 1, j, i + 1));
    real zy_m = 0.25f * (wrf_zy(q, k, j, i) + wrf_zy(q, k, j + 1, i)
                          + wrf_zy(q, k + 1, j, i)
                          + wrf_zy(q, k + 1, j + 1, i));
    real rz = wrf_rdzw(q, k, j, i);
    real divh = map * rdx * (wrf_fx(q, fx, k, j, i + 1)
                              - wrf_fx(q, fx, k, j, i))
              + map * rdy * (wrf_fy(q, fy, k, j + 1, i)
                              - wrf_fy(q, fy, k, j, i));
    real divz = map * zx_m * (wrf_h1_mavg(q, fx, k + 1, j, i)
                              - wrf_h1_mavg(q, fx, k, j, i)) * rz
              + map * zy_m * (wrf_h2_mavg(q, fy, k + 1, j, i)
                              - wrf_h2_mavg(q, fy, k, j, i)) * rz;
    tend[IDX3(k, j, i)] += G / (dnw[k] * rz) * (divh - divz);
}

// ---------------------------------------------------------------------------
// WRF v4.6.1 diff_opt=2 / km_opt=3 (3-D Smagorinsky) closure.
//
// Transcribed from module_diffusion_em.F: calculate_N2 (:1485-1713), smag_km
// (:1777-1929), vertical_diffusion_s (:4789-4907), and the isfflx=0/2
// prescribed-flux branches of vertical_diffusion_2 (:4155-4250 vflux,
// :4286-4330 hflux).  The deformation tensors reuse the production
// cal_deform_and_div device functions above (wrf_defor13/23, defor33
// inline); D12 reuses the mass-point four-corner average.  km_opt=3 keeps
// WRF's constant Prandtl treatment (pr = prandtl = 1/3, Kh = 3 Km in BOTH
// directions -- unlike km_opt=2's stability-dependent vertical Pr) and has
// no 10*mlen cap and no terrain-slope alpha reduction (both are smag2d_km /
// km_opt=4 features).  mix_full_fields=.true. semantics throughout (the
// admitted gpuwm identity); use_theta_m=0 for the surface heat flux.
// ---------------------------------------------------------------------------

// Temperature at a mass point, WRF phy_prep convention t = theta*(p/p0)^rcp.
__device__ __forceinline__
real wrf_n2_temp(real theta_full, real p_full)
{
    return theta_full * powf(p_full / P0, RCP);
}

// Saturation mixing ratio (calculate_N2 :1630-1637): es in Pa from the
// Tetens form (SVP1 is kPa, hence the 1000 factor), qvs = EP2*es/(p-es).
__device__ __forceinline__
real wrf_n2_qvs(real t, real p)
{
    real tc = t - SVPT0;
    real es = 1000.0f * SVP1 * expf(SVP2 * tc / (t - SVP3));
    return EP2 * es / (p - es);
}

struct WrfN2Column {
    const real* thp;   // theta - thb (gpuwm perturbation storage)
    const real* thb;   // base theta, 1-D (nz,) or 3-D (nz,ny,nx)
    int thb3d;
    const real* p;     // full pressure at mass points
    const real* qc;    // may be null (has_qc = 0)
    const real* qi;    // may be null (has_qi = 0)
    int has_qc, has_qi;
};

__device__ __forceinline__
real wrf_n2_theta(const WrfSmagGrid& q, const WrfN2Column& n,
                  int k, int j, int i)
{
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    size_t h = I3(k, jj, ii, q.ny, q.nx);
    real base = n.thb3d ? n.thb[h] : n.thb[k];
    return base + n.thp[h];
}

// qtot = qv + qc + qi (calculate_N2's tmp1: only P_QV/P_QC/P_QI members).
__device__ __forceinline__
real wrf_n2_qtot(const WrfSmagGrid& q, const WrfN2Column& n,
                 int k, int j, int i)
{
    if (!q.moist) return 0.0f;
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    size_t h = I3(k, jj, ii, q.ny, q.nx);
    real tot = q.qv[h];
    if (n.has_qc) tot += n.qc[h];
    if (n.has_qi) tot += n.qi[h];
    return tot;
}

__device__ __forceinline__
real wrf_n2_qv(const WrfSmagGrid& q, int k, int j, int i)
{
    if (!q.moist) return 0.0f;
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    return q.qv[I3(k, jj, ii, q.ny, q.nx)];
}

// WRF calculate_N2, all three level branches (interior kts+1..ktf-1,
// MARTA surface form at kts, copy at ktf).  Each thread owns one mass
// point; the k = nz-1 thread evaluates the nz-2 formula (BN2(ktf) =
// BN2(ktf-1), :1705-1710).
extern "C" __global__
void wrf_calc_n2(WRF_SMAG_GRID_ARGS,
                 const real* thp, const real* thb, int thb3d,
                 const real* p, const real* qc, const real* qi,
                 int has_qc, int has_qi,
                 real* bn2,
                 int nz, int ny, int nx, int phb3d,
                 int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    WrfN2Column n = {thp, thb, thb3d, p, qc, qi, has_qc, has_qi};

    int ke = (k == nz - 1 && nz > 1) ? nz - 2 : k;  // BN2(ktf)=BN2(ktf-1)
    size_t hp = I3(ke, j, i, ny, nx);
    real th_c = wrf_n2_theta(q, n, ke, j, i);
    real p_c = p[hp];
    real t_c = wrf_n2_temp(th_c, p_c);
    real qv_c = wrf_n2_qv(q, ke, j, i);
    real qc_c = (moist && has_qc) ? qc[hp] : 0.0f;
    bool saturated = moist
        && (qv_c >= wrf_n2_qvs(t_c, p_c) || qc_c >= 1.0e-5f);

    real value;
    if (ke > 0) {
        // Interior centered form, tmpdz = 1/rdz(k) + 1/rdz(k+1).
        real tmpdz = 1.0f / wrf_rdz(q, ke, j, i)
                   + 1.0f / wrf_rdz(q, ke + 1, j, i);
        real th_p = wrf_n2_theta(q, n, ke + 1, j, i);
        real th_m = wrf_n2_theta(q, n, ke - 1, j, i);
        real qtot_p = wrf_n2_qtot(q, n, ke + 1, j, i);
        real qtot_m = wrf_n2_qtot(q, n, ke - 1, j, i);
        if (saturated) {
            size_t hpp = I3(ke + 1, j, i, ny, nx);
            size_t hpm = I3(ke - 1, j, i, ny, nx);
            real t_p = wrf_n2_temp(th_p, p[hpp]);
            real t_m = wrf_n2_temp(th_m, p[hpm]);
            real xlvqv = XLV * qv_c;
            real coefa = (1.0f + xlvqv / RD / t_c)
                       / (1.0f + XLV * xlvqv / CP / RV / t_c / t_c)
                       / th_c;
            real thetaep1 = th_p
                * (1.0f + XLV * wrf_n2_qvs(t_p, p[hpp]) / CP / t_p);
            real thetaem1 = th_m
                * (1.0f + XLV * wrf_n2_qvs(t_m, p[hpm]) / CP / t_m);
            value = G * (coefa * (thetaep1 - thetaem1) / tmpdz
                         - (qtot_p - qtot_m) / tmpdz);
        } else {
            real qv_p = wrf_n2_qv(q, ke + 1, j, i);
            real qv_m = wrf_n2_qv(q, ke - 1, j, i);
            value = G * ((th_p - th_m) / th_c / tmpdz
                         + 1.61f * (qv_p - qv_m) / tmpdz
                         - (qtot_p - qtot_m) / tmpdz);
        }
    } else {
        // Surface level kts.
        real th_p = wrf_n2_theta(q, n, 1, j, i);
        real qtot_p = wrf_n2_qtot(q, n, 1, j, i);
        // tmp1sfc: cf-extrapolated qv+qc+qi.
        real qtot_sfc = q.cf1 * wrf_n2_qtot(q, n, 0, j, i)
                      + q.cf2 * wrf_n2_qtot(q, n, 1, j, i)
                      + q.cf3 * wrf_n2_qtot(q, n, 2, j, i);
        if (saturated) {
            real tmpdz = 1.0f / wrf_rdz(q, 1, j, i)
                       + 0.5f / wrf_rdzw(q, 0, j, i);
            size_t h1 = I3(1, j, i, ny, nx);
            size_t h2 = I3(2, j, i, ny, nx);
            real t_1 = wrf_n2_temp(th_p, p[h1]);
            real th_2 = wrf_n2_theta(q, n, 2, j, i);
            real t_2 = wrf_n2_temp(th_2, p[h2]);
            // phy_prep surface extrapolation for p8w/t8w (z-linear,
            // module_big_step_utilities_em.F:4916-4923).
            real z0 = wrf_phi(q, 0, j, i) / G;
            real z1 = 0.5f * (wrf_phi(q, 0, j, i)
                              + wrf_phi(q, 1, j, i)) / G;
            real z2 = 0.5f * (wrf_phi(q, 1, j, i)
                              + wrf_phi(q, 2, j, i)) / G;
            real w1 = (z0 - z2) / (z1 - z2);
            real w2 = 1.0f - w1;
            real p8w0 = w1 * p_c + w2 * p[h1];
            real t8w0 = w1 * t_c + w2 * wrf_n2_temp(
                wrf_n2_theta(q, n, 1, j, i), p[h1]);
            real thetasfc = t8w0 / powf(p8w0 / P0, RCP);
            real qvs0 = wrf_n2_qvs(t_c, p_c);
            real qvs1 = wrf_n2_qvs(t_1, p[h1]);
            real qvs2 = wrf_n2_qvs(t_2, p[h2]);
            real qvsfc = q.cf1 * qvs0 + q.cf2 * qvs1
                       + q.cf3 * qvs2;
            real xlvqv = XLV * qv_c;
            real coefa = (1.0f + xlvqv / RD / t_c)
                       / (1.0f + XLV * xlvqv / CP / RV / t_c / t_c)
                       / th_c;
            real thetaep1 = th_p * (1.0f + XLV * qvs1 / CP / t_1);
            real thetaesfc = thetasfc
                * (1.0f + XLV * qvsfc / CP / t8w0);
            value = G * (coefa * (thetaep1 - thetaesfc) / tmpdz
                         - (qtot_p - qtot_sfc) / tmpdz);
        } else {
            // MARTA/WCS surface form (:1690-1698).
            real qvsfc = q.cf1 * wrf_n2_qv(q, 0, j, i)
                       + q.cf2 * wrf_n2_qv(q, 1, j, i)
                       + q.cf3 * wrf_n2_qv(q, 2, j, i);
            real tmpdz = 1.0f / wrf_rdzw(q, 0, j, i);
            value = G * ((th_p - th_c) / th_c / tmpdz
                         + 1.61f * (wrf_n2_qv(q, 1, j, i) - qvsfc) / tmpdz
                         - (qtot_p - qtot_sfc) / tmpdz);
        }
    }
    bn2[IDX3(k, j, i)] = value;
}

// WRF smag_km (:1777-1929): the full 3-D deformation invariant with the
// off-diagonal tensors averaged to mass points BEFORE squaring, the
// buoyancy reduction sqrt(max(0, D^2 - N^2/pr)), and the two mix_isotropic
// mixing-length branches with WRF's exact floors and mix_upper_bound caps.
extern "C" __global__
void wrf_smag3d_km(WRF_SMAG_GRID_ARGS,
                   real c_s, real prandtl, real dt, real mix_upper_bound,
                   int isotropic,
                   const real* d11a, const real* d22a, const real* d12a,
                   const real* bn2,
                   real* xkmh, real* xkhh, real* xkmv, real* xkhv,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    real d11 = wrf_d(q, d11a, k, j, i);
    real d22 = wrf_d(q, d22a, k, j, i);
    real d33 = 2.0f * (q.w[I3(k + 1, j, i, ny, nx)]
                       - q.w[I3(k, j, i, ny, nx)])
             * wrf_rdzw(q, k, j, i);
    real d12 = 0.25f * (wrf_d(q, d12a, k, j, i)
                         + wrf_d(q, d12a, k, j + 1, i)
                         + wrf_d(q, d12a, k, j, i + 1)
                         + wrf_d(q, d12a, k, j + 1, i + 1));
    real d13 = 0.25f * (wrf_defor13(q, k + 1, j, i)
                         + wrf_defor13(q, k, j, i)
                         + wrf_defor13(q, k + 1, j, i + 1)
                         + wrf_defor13(q, k, j, i + 1));
    real d23 = 0.25f * (wrf_defor23(q, k + 1, j, i)
                         + wrf_defor23(q, k, j, i)
                         + wrf_defor23(q, k + 1, j + 1, i)
                         + wrf_defor23(q, k, j + 1, i));
    real def2 = 0.5f * (d11 * d11 + d22 * d22 + d33 * d33)
              + d12 * d12 + d13 * d13 + d23 * d23;
    real tmp = sqrtf(fmaxf(0.0f, def2 - bn2[IDX3(k, j, i)] / prandtl));

    real map = msft[(size_t)wrf_iy(q, j) * nx + wrf_ix(q, i)];
    real dxm = dx / map, dym = dy / map;
    real rdzw_c = wrf_rdzw(q, k, j, i);
    real kmh, kmv, khh, khv;
    if (isotropic == 0) {
        real mlen_h2 = dxm * dym;
        real mlen_v = 1.0f / rdzw_c;
        real mlen_v2 = mlen_v * mlen_v;
        kmh = fmaxf(c_s * c_s * mlen_h2 * tmp, 1.0e-6f * mlen_h2);
        kmh = fminf(kmh, mix_upper_bound * mlen_h2 / dt);
        kmv = fmaxf(c_s * c_s * mlen_v2 * tmp, 1.0e-6f * mlen_v2);
        kmv = fminf(kmv, mix_upper_bound * mlen_v2 / dt);
        khh = fminf(kmh / prandtl, mix_upper_bound * mlen_h2 / dt);
        khv = fminf(kmv / prandtl, mix_upper_bound * mlen_v2 / dt);
    } else {
        real deltas = powf(dxm * dym / rdzw_c, 0.33333333f);
        real deltas2 = deltas * deltas;
        kmh = fmaxf(c_s * c_s * deltas2 * tmp, 1.0e-6f * deltas2);
        kmh = fminf(kmh, mix_upper_bound * dxm * dym / dt);
        kmv = fminf(kmh, mix_upper_bound / rdzw_c / rdzw_c / dt);
        khh = fminf(kmh / prandtl, mix_upper_bound * dxm * dym / dt);
        khv = fminf(kmv / prandtl,
                    mix_upper_bound / rdzw_c / rdzw_c / dt);
    }
    xkmh[IDX3(k, j, i)] = kmh;
    xkmv[IDX3(k, j, i)] = kmv;
    xkhh[IDX3(k, j, i)] = khh;
    xkhv[IDX3(k, j, i)] = khv;
}

// WRF vertical_diffusion_s (:4789-4907), doing_tke=.false.: interior
// down-gradient vertical scalar flux with density- and K-averages on w
// levels, H3 = 0 at the surface and top interfaces (the explicit surface
// fluxes are separate kernels).  ``full_theta`` reconstructs WRF's
// t_2-like field from gpuwm's thp = theta - thb storage exactly as the
// horizontal wrf_smag_flux_s does; the T0 offset cancels in the k
// difference and is omitted.
__device__ __forceinline__
real wrf_vd_s_h3(const WrfSmagGrid& q, const real* var,
                 const real* thb, int full_theta, int thb3d,
                 const real* khv, int kw, int j, int i)
{
    if (kw <= 0 || kw >= q.nz) return 0.0f;
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    size_t hc = I3(kw, jj, ii, q.ny, q.nx);
    size_t hm = I3(kw - 1, jj, ii, q.ny, q.nx);
    real xkx = q.fnm[kw] * khv[hc] + q.fnp[kw] * khv[hm];
    xkx *= q.fnm[kw] * wrf_rho(q, kw, j, i)
         + q.fnp[kw] * wrf_rho(q, kw - 1, j, i);
    real dvar = var[hc] - var[hm];
    if (full_theta) {
        real bc = thb3d ? thb[hc] : thb[kw];
        real bm = thb3d ? thb[hm] : thb[kw - 1];
        dvar += bc - bm;
    }
    return -xkx * dvar * wrf_rdz(q, kw, j, i);
}

extern "C" __global__
void wrf_smag_vd_s(WRF_SMAG_GRID_ARGS,
                   const real* var, const real* thb,
                   int full_theta, int thb3d,
                   const real* khv, real* tend,
                   int nz, int ny, int nx, int phb3d,
                   int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    real lower = wrf_vd_s_h3(q, var, thb, full_theta, thb3d, khv, k, j, i);
    real upper = wrf_vd_s_h3(q, var, thb, full_theta, thb3d, khv,
                             k + 1, j, i);
    tend[IDX3(k, j, i)] += G * (upper - lower) / dnw[k];
}

// vertical_diffusion_2 vflux CASE(0) (:4155-4200): prescribed constant
// drag coefficient (namelist tke_drag_coefficient), tao = cd0*|V|*u.
// WRF epsilon = 1.E-15 (share/module_model_constants.F:10).
extern "C" __global__
void wrf_smag_surface_u_cd0(WRF_SMAG_GRID_ARGS, real cd0,
                            real* tend, int nz, int ny, int nx, int phb3d,
                            int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx + 1 || j >= ny) return;
    WRF_SMAG_MAKE_GRID;
    int ic = wrf_ix(q, i), im = wrf_ix(q, i - 1), jj = wrf_iy(q, j);
    real vv = 0.25f * (
        q.v[I3S(0, wrf_jv(q, j), ic, ny + 1, nx)]
        + q.v[I3S(0, wrf_jv(q, j + 1), ic, ny + 1, nx)]
        + q.v[I3S(0, wrf_jv(q, j), im, ny + 1, nx)]
        + q.v[I3S(0, wrf_jv(q, j + 1), im, ny + 1, nx)]);
    real uu = q.u[I3S(0, jj, wrf_iu(q, i), ny, nx + 1)];
    real speed = sqrtf(uu * uu + vv * vv) + 1.0e-15f;
    real stress = cd0 * speed * uu;
    real rhoavg = 0.5f * (wrf_rho(q, 0, j, ic) + wrf_rho(q, 0, j, im));
    tend[I3S(0, j, i, ny, nx + 1)] += G * stress * rhoavg / dnw[0];
}

extern "C" __global__
void wrf_smag_surface_v_cd0(WRF_SMAG_GRID_ARGS, real cd0,
                            real* tend, int nz, int ny, int nx, int phb3d,
                            int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny + 1) return;
    WRF_SMAG_MAKE_GRID;
    int jc = wrf_iy(q, j), jm = wrf_iy(q, j - 1), ii = wrf_ix(q, i);
    real uu = 0.25f * (
        q.u[I3S(0, jc, wrf_iu(q, i), ny, nx + 1)]
        + q.u[I3S(0, jc, wrf_iu(q, i + 1), ny, nx + 1)]
        + q.u[I3S(0, jm, wrf_iu(q, i), ny, nx + 1)]
        + q.u[I3S(0, jm, wrf_iu(q, i + 1), ny, nx + 1)]);
    real vv = q.v[I3S(0, wrf_jv(q, j), ii, ny + 1, nx)];
    real speed = sqrtf(vv * vv + uu * uu) + 1.0e-15f;
    real stress = cd0 * speed * vv;
    real rhoavg = 0.5f * (wrf_rho(q, 0, jc, i) + wrf_rho(q, 0, jm, i));
    tend[I3S(0, j, i, ny + 1, nx)] += G * stress * rhoavg / dnw[0];
}

// vertical_diffusion_2 hflux CASE(0,2) (:4286-4305): prescribed constant
// kinematic surface heat flux (namelist tke_heat_flux, K m s-1), applied
// to the coupled theta tendency through the use_theta_m=0 branch
// rt_tendf -= g*heat_flux*rho(kts)/dnw(kts).  WRF's hfx(i,j) refresh at
// :4293 is "provided for output only" and is not reproduced here (gpuwm
// holds hfx on the physics side; documented divergence, output-only).
extern "C" __global__
void wrf_smag_surface_heat_const(WRF_SMAG_GRID_ARGS, real heat_flux,
                                 real* rth, int nz, int ny, int nx,
                                 int phb3d, int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    WRF_SMAG_MAKE_GRID;
    rth[I3(0, j, i, ny, nx)] -=
        G * heat_flux * wrf_rho(q, 0, j, i) / dnw[0];
}

// ---------------------------------------------------------------------------
// WRF v4.6.1 diff_opt=2 / km_opt=2 (1.5-order prognostic TKE) closure.
//
// Transcribed from module_diffusion_em.F: tke_km (:2049-2336), calc_l_scale
// (:2341-2406), tke_rhs = tke_shear + tke_buoyancy + tke_dissip + the
// positivity limiter (:6099-6229, :6529-6877, :6234-6379, :6384-6524).
// Constant notes: prandtl = 1/3 enters only the HORIZONTAL heat coefficient
// (xkhh = xkmh/prandtl, uncapped in the anisotropic branch -- WRF's
// documented limiter asymmetry); the vertical inverse Prandtl number is the
// Deardorff stability form 1 + 2*l/deltas.  tke_shear SQUARES the
// off-diagonal deformations THEN averages (the opposite order from
// smag_km).  The Deardorff wall coefficient 3.9 applies at BOTH k=kts and
// k=ktf exactly as the source does (":For LES with fine grid, no need for
// this wall effect!" -- code applies it anyway).  epsilon in the CASE(1,2)
// drag speed is module_model_constants' 1.e-15.
// ---------------------------------------------------------------------------

// calc_l_scale (:2341-2406): the BN2-limited isotropic length.
__device__ __forceinline__
real wrf_l_scale(real tke_v, real bn2_v, real deltas)
{
    real l = deltas;
    if (bn2_v > 1.0e-6f) {
        real tmp = sqrtf(fmaxf(tke_v, 1.0e-6f));
        l = 0.76f * tmp / sqrtf(bn2_v);
        l = fminf(l, deltas);
        l = fmaxf(l, 0.001f * deltas);
    }
    return l;
}

// theta at a mass point from gpuwm's perturbation storage.
__device__ __forceinline__
real wrf_theta_full(const WrfSmagGrid& q, const real* thp, const real* thb,
                    int thb3d, int k, int j, int i)
{
    int jj = wrf_iy(q, j), ii = wrf_ix(q, i);
    size_t h = I3(k, jj, ii, q.ny, q.nx);
    return (thb3d ? thb[h] : thb[k]) + thp[h];
}

// tke_km (:2049-2260): the four exchange coefficients from prognostic TKE.
// dthrdn is the centered d(theta)/dz with the one-sided surface/top forms
// (:2185-2205); thetasfc/thetatop use phy_prep's p8w/t8w extrapolations
// (module_big_step_utilities_em.F:4916-4930, log-p at the top).
extern "C" __global__
void wrf_tke_km(WRF_SMAG_GRID_ARGS,
                const real* thp, const real* thb, int thb3d,
                const real* p, const real* tke, const real* bn2,
                real c_k, real prandtl, real dt, real mix_upper_bound,
                int isotropic, real tke_seed,
                real* xkmh, real* xkhh, real* xkmv, real* xkhv,
                int nz, int ny, int nx, int phb3d,
                int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    size_t idx = IDX3(k, j, i);
    real map = msft[(size_t)j * nx + i];
    real dxm = dx / map, dym = dy / map;
    real rdzw_c = wrf_rdzw(q, k, j, i);
    real tmp = sqrtf(fmaxf(tke[idx], tke_seed));
    real kmh, kmv, khh, khv;
    if (isotropic == 0) {
        // dthrdn at this level.
        real theta_c = wrf_theta_full(q, thp, thb, thb3d, k, j, i);
        real dthrdn;
        if (k == 0) {
            real tmpdz = 1.0f / wrf_rdzw(q, 1, j, i)
                       + 1.0f / wrf_rdzw(q, 0, j, i);
            real z0 = wrf_phi(q, 0, j, i) / G;
            real z1 = 0.5f * (wrf_phi(q, 0, j, i)
                              + wrf_phi(q, 1, j, i)) / G;
            real z2 = 0.5f * (wrf_phi(q, 1, j, i)
                              + wrf_phi(q, 2, j, i)) / G;
            real w1 = (z0 - z2) / (z1 - z2);
            real w2 = 1.0f - w1;
            real t0 = wrf_theta_full(q, thp, thb, thb3d, 0, j, i)
                    * powf(p[IDX3(0, j, i)] / P0, RCP);
            real t1 = wrf_theta_full(q, thp, thb, thb3d, 1, j, i)
                    * powf(p[IDX3(1, j, i)] / P0, RCP);
            real p8w0 = w1 * p[IDX3(0, j, i)] + w2 * p[IDX3(1, j, i)];
            real t8w0 = w1 * t0 + w2 * t1;
            real thetasfc = t8w0 / powf(p8w0 / P0, RCP);
            real th1 = wrf_theta_full(q, thp, thb, thb3d, 1, j, i);
            dthrdn = (th1 - thetasfc) / tmpdz;
        } else if (k == nz - 1) {
            real tmpdz = 1.0f / wrf_rdz(q, nz - 1, j, i)
                       + 0.5f / wrf_rdzw(q, nz - 1, j, i);
            real z0 = wrf_phi(q, nz, j, i) / G;
            real z1 = 0.5f * (wrf_phi(q, nz - 1, j, i)
                              + wrf_phi(q, nz, j, i)) / G;
            real z2 = 0.5f * (wrf_phi(q, nz - 2, j, i)
                              + wrf_phi(q, nz - 1, j, i)) / G;
            real w1 = (z0 - z2) / (z1 - z2);
            real w2 = 1.0f - w1;
            real tm1 = wrf_theta_full(q, thp, thb, thb3d, nz - 1, j, i)
                     * powf(p[IDX3(nz - 1, j, i)] / P0, RCP);
            real tm2 = wrf_theta_full(q, thp, thb, thb3d, nz - 2, j, i)
                     * powf(p[IDX3(nz - 2, j, i)] / P0, RCP);
            real p8wt = expf(w1 * logf(p[IDX3(nz - 1, j, i)])
                             + w2 * logf(p[IDX3(nz - 2, j, i)]));
            real t8wt = w1 * tm1 + w2 * tm2;
            real thetatop = t8wt / powf(p8wt / P0, RCP);
            real thm1 = wrf_theta_full(q, thp, thb, thb3d, nz - 2, j, i);
            dthrdn = (thetatop - thm1) / tmpdz;
        } else {
            real tmpdz = 1.0f / wrf_rdz(q, k + 1, j, i)
                       + 1.0f / wrf_rdz(q, k, j, i);
            real thp1 = wrf_theta_full(q, thp, thb, thb3d, k + 1, j, i);
            real thm1 = wrf_theta_full(q, thp, thb, thb3d, k - 1, j, i);
            dthrdn = (thp1 - thm1) / tmpdz;
        }
        real mlen_h = sqrtf(dxm * dym);
        real deltas = 1.0f / rdzw_c;
        real mlen_v = deltas;
        if (dthrdn > 0.0f) {
            real mlen_s = 0.76f * tmp
                        / sqrtf(fabsf(G / theta_c * dthrdn));
            mlen_v = fminf(mlen_v, mlen_s);
        }
        kmh = fmaxf(c_k * tmp * mlen_h, 1.0e-6f * mlen_h * mlen_h);
        kmh = fminf(kmh, mix_upper_bound * mlen_h * mlen_h / dt);
        kmv = fmaxf(c_k * tmp * mlen_v, 1.0e-6f * deltas * deltas);
        kmv = fminf(kmv, mix_upper_bound * deltas * deltas / dt);
        khh = kmh * (1.0f / prandtl);            // uncapped (WRF asymmetry)
        khv = kmv * (1.0f + 2.0f * mlen_v / deltas);
    } else {
        real deltas = powf(dxm * dym / rdzw_c, 0.33333333f);
        real l = wrf_l_scale(tke[idx], bn2[idx], deltas);
        kmh = c_k * tmp * l;
        kmh = fminf(mix_upper_bound * dxm * dym / dt, kmh);
        kmv = c_k * tmp * l;
        kmv = fminf(mix_upper_bound / rdzw_c / rdzw_c / dt, kmv);
        real pr_inv = 1.0f + 2.0f * l / deltas;
        khh = fminf(mix_upper_bound * dxm * dym / dt, kmh * pr_inv);
        khv = fminf(mix_upper_bound / rdzw_c / rdzw_c / dt, kmv * pr_inv);
    }
    xkmh[idx] = kmh;
    xkmv[idx] = kmv;
    xkhh[idx] = khh;
    xkhv[idx] = khv;
}

// tke_rhs: shear + buoyancy + dissipation, then the positivity limiter
// (:6221-6227).  Everything is a coupled tendency (c1*mu+c2 factors).
// The k=kts shear terms ADD the MARTA surface-drag contributions
// (:6767-6877, uflux/vflux SELECT CASE) and the k=kts buoyancy takes the
// averaged surface-flux form (:6340-6365).  ust here is WRF's grid%ustm
// (module_first_rk_step_part2.F:914).
extern "C" __global__
void wrf_tke_rhs(WRF_SMAG_GRID_ARGS,
                 const real* thp, const real* thb, int thb3d,
                 const real* tke, const real* bn2,
                 const real* d11a, const real* d22a, const real* d12a,
                 const real* kmh_a, const real* kmv_a, const real* khv_a,
                 const real* mut, const real* c1, const real* c2,
                 const real* ustm, const real* hfx,
                 int use_ustm, int use_hfx,
                 real c_k, real dt, real cd0, real heat_flux_const,
                 int isfflx,
                 real* tend,
                 real* b_shear, real* b_buoy, real* b_diss, real* b_lim,
                 int budget,
                 int nz, int ny, int nx, int phb3d,
                 int boundary_x, int boundary_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    WRF_SMAG_MAKE_GRID;
    size_t idx = IDX3(k, j, i);
    real chm = c1[k] * mut[(size_t)j * nx + i] + c2[k];
    real kmh = kmh_a[idx], kmv = kmv_a[idx];
    real t = 0.0f;

    // --- tke_shear: six squared-deformation production terms.
    real d11 = wrf_d(q, d11a, k, j, i);
    real d22 = wrf_d(q, d22a, k, j, i);
    t += 0.5f * chm * kmh * (d11 * d11);
    t += 0.5f * chm * kmh * (d22 * d22);
    real rdzw_c = wrf_rdzw(q, k, j, i);
    real d33 = 2.0f * (q.w[I3(k + 1, j, i, ny, nx)]
                       - q.w[I3(k, j, i, ny, nx)]) * rdzw_c;
    t += 0.5f * chm * kmv * (d33 * d33);
    real s12a = wrf_d(q, d12a, k, j, i);
    real s12b = wrf_d(q, d12a, k, j + 1, i);
    real s12c = wrf_d(q, d12a, k, j, i + 1);
    real s12d = wrf_d(q, d12a, k, j + 1, i + 1);
    t += chm * kmh * 0.25f * (s12a * s12a + s12b * s12b
                              + s12c * s12c + s12d * s12d);
    real s13a = wrf_defor13(q, k + 1, j, i);
    real s13b = wrf_defor13(q, k, j, i);
    real s13c = wrf_defor13(q, k + 1, j, i + 1);
    real s13d = wrf_defor13(q, k, j, i + 1);
    t += chm * kmv * 0.25f * (s13a * s13a + s13b * s13b
                              + s13c * s13c + s13d * s13d);
    real s23a = wrf_defor23(q, k + 1, j, i);
    real s23b = wrf_defor23(q, k, j, i);
    real s23c = wrf_defor23(q, k + 1, j + 1, i);
    real s23d = wrf_defor23(q, k, j + 1, i);
    t += chm * kmv * 0.25f * (s23a * s23a + s23b * s23b
                              + s23c * s23c + s23d * s23d);

    if (k == 0) {
        // MARTA surface drag additions (u_2/v_2 raw winds, ust at (i,j)).
        real usum = q.u[I3S(0, j, wrf_iu(q, i), ny, nx + 1)]
                  + q.u[I3S(0, j, wrf_iu(q, i + 1), ny, nx + 1)];
        real vsum = q.v[I3S(0, wrf_jv(q, j), i, ny + 1, nx)]
                  + q.v[I3S(0, wrf_jv(q, j + 1), i, ny + 1, nx)];
        real absU = 0.5f * sqrtf(usum * usum + vsum * vsum);
        real Cd;
        if (isfflx == 0) {
            Cd = cd0;
        } else {
            absU += 1.0e-15f;
            real us = use_ustm ? ustm[(size_t)j * nx + i] : 0.0f;
            Cd = (us * us) / (absU * absU);
        }
        real d13sum = 0.5f * (wrf_defor13(q, 1, j, i)
                              + wrf_defor13(q, 1, j, i + 1));
        t += chm * (0.5f * usum * Cd * absU * d13sum);
        real d23sum = 0.5f * (wrf_defor23(q, 1, j, i)
                              + wrf_defor23(q, 1, j + 1, i));
        t += chm * (0.5f * vsum * Cd * absU * d23sum);
    }

    // Budget bookkeeping: each term is the running total's increment, so
    // the four exports sum EXACTLY to the tendency this kernel deposits.
    real s_shear = t;

    // --- tke_buoyancy.
    real khv = khv_a[idx];
    if (k >= 1) {
        t -= chm * khv * bn2[idx];
    } else {
        real heat_flux;
        if (isfflx == 0 || isfflx == 2) {
            heat_flux = heat_flux_const;
        } else {
            real vapor = moist ? qv[I3(0, j, i, ny, nx)] : 0.0f;
            real cpm = CP * (1.0f + 0.8f * vapor);
            real hf = use_hfx ? hfx[(size_t)j * nx + i] : 0.0f;
            heat_flux = (hf / cpm) / wrf_rho(q, 0, j, i);
        }
        real theta_c = wrf_theta_full(q, thp, thb, thb3d, 0, j, i);
        t -= chm * ((khv * bn2[idx])
                    - (G / theta_c) * heat_flux) * 0.5f;
    }

    real s_buoy = t - s_shear;

    // --- tke_dissip (l_scale from calc_l_scale, computed inline).
    real map = msft[(size_t)j * nx + i];
    real deltas = powf((dx / map) * (dy / map) / rdzw_c, 0.33333333f);
    real l = wrf_l_scale(tke[idx], bn2[idx], deltas);
    real ce1 = (c_k / 0.10f) * 0.19f;
    real ce2 = fmaxf(0.0f, 0.93f - ce1);
    real coefc = (k == 0 || k == nz - 1) ? 3.9f
               : (ce1 + ce2 * l / deltas);
    real tketmp = fmaxf(tke[idx], 1.0e-6f);
    t -= chm * coefc * tketmp * sqrtf(tketmp) / l;

    real s_diss = t - s_shear - s_buoy;

    // --- positivity limiter (:6221-6227).
    real t_limited = fmaxf(t, -chm * fmaxf(0.0f, tke[idx]) / dt);
    real s_lim = t_limited - t;
    t = t_limited;
    tend[idx] += t;
    if (budget) {
        b_shear[idx] = s_shear;
        b_buoy[idx] = s_buoy;
        b_diss[idx] = s_diss;
        b_lim[idx] = s_lim;
    }
}

#undef WRF_SMAG_MAKE_GRID
#undef WRF_SMAG_GRID_ARGS
