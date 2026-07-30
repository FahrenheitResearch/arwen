// gpuwm/core/kernels/advection.cu
//
// Flux-form advection on the Arakawa C-grid (WRF-ARW schemes):
//   horizontal: 5th-order upwind-biased fluxes (WRF flux5),
//   vertical:   3rd-order upwind-biased fluxes (WRF flux3), with a
//               2nd-order centered fallback one face in from each boundary
//               and zero flux through the domain top/bottom faces.
//
// All four kernels ADD  -dF_x/dx - dF_y/dy - dF_eta/deta  into tend_out.
// ru (nz,ny,nx+1), rv (nz,ny+1,nx), rw (nz+1,ny,nx) are coupled mass
// fluxes (mu*u etc.) precomputed by the caller.  rw is the eta-directed
// advecting flux, Omega-signed: negative for physically upward motion,
// since eta decreases with height.  Vertical fluxes therefore upwind on
// -vel -- WRF module_advect_em's flux3(..., -vel) convention -- which is
// why flux3's dissipation term carries the opposite sign to flux5's.
// x and y are periodic: stencil reads of the advected field wrap with
// period nx (ny), so the redundant staggered columns u[...,nx] / v[:,ny,:]
// are never read and their tendencies are computed identically to column 0.
//
// Face-flux index convention (matches gpuwm/verify/npref.py mirrors):
// the u-face f lies between cells f-1 and f, so
//   F(f) = vel*(37*(q[f]+q[f-1]) - 8*(q[f+1]+q[f-2]) + (q[f+2]+q[f-3]))/60
//        - |vel|*(10*(q[f]-q[f-1]) - 5*(q[f+1]-q[f-2]) + (q[f+2]-q[f-3]))/60
// and a cell's divergence uses F(right) - F(left).
//
// Open lateral boundaries (Task 11 prerequisite; WRF v4.6.1
// module_advect_em.F transcription): with open_x/open_y each kernel takes
// a dedicated path that (a) degrades the 5th-order stencil to 2nd order
// one face in from the boundary and WRF's horizontal flux3 two faces in
// (the degrade_xs/xe/ys/ye blocks), (b) applies WRF's loop-bound
// exclusions -- no advection normal to an open boundary at the boundary
// cells/faces, no cross-boundary stencil wrap anywhere -- and (c) adds
// WRF's non-cb open advective terms at the boundary cells of
// scalar/w/tangential-velocity fields ("the computations that don't
// require cb", with field_old == field per the rk_tendency call).  The
// boundary-normal velocity faces get NO horizontal advection here; the
// additive cb radiative term (openbc.cu) stands in for it.  Their
// vertical advection is retained with the boundary cell's Omega (WRF's
// zero-gradient rom ghost) for u unless open_y is also set (the Fortran's
// commented-out open_x bounds + active open_ys/ye bounds), and dropped
// for v whenever open_y is set -- both asymmetries transcribed verbatim.
// The periodic path (open_x = open_y = 0) is the original code,
// bitwise unchanged (tests/data/advection_periodic_regression.npz).
//
// Specified boundaries (Phase 4 transport FIX-D; WRF's `specified`
// logical): with spec != 0 (dycore sets it from cfg.specified, always
// together with open_x/open_y) the non-cb open advective terms do NOT
// fire -- WRF gates them on open_xs/xe/ys/ye only and simply excludes the
// boundary cells from the tendency loops under specified -- and the
// boundary-adjacent 2nd-order u/v fluxes take WRF's "specified uses
// upstream normal wind at boundaries" substitution (advect_u F:690-723,
// advect_v F:1978-2013).  Vertical faces one in from the eta boundaries
// and advect_w's horizontal advecting velocities carry WRF's
// stretched-grid fnm/fnp weights (see zface_half; identical to the old
// 0.5/0.5 averages on uniform grids, bitwise).
//
// Map-scale factors (Phase 3 Task 3 + Phase 4 transport FIX-A; ARW tech
// note eqns 2.23-2.26, WRF module_advect_em.F mrdx/mrdy): every kernel
// takes the 2-D map factor at its TENDENCY points (msfu for u, msfv for v,
// msft for scalars and w) and, when has_msf != 0, weights the horizontal
// flux divergence by it — WRF's mrdx = msf*rdx / mrdy = msf*rdy, identical
// for both axes at a given point, in BOTH the periodic and the
// open/specified paths (WRF applies mrdx/mrdy unconditionally: scalar
// F:3534/3644, u F:633/740, v F:2050/2676, w F:5096/5548; real74 runs
// specified + Lambert msf through the open path every step).  The non-cb
// open advective terms follow WRF exactly: u's y-term and v's x-term are
// msf-weighted (mrdy at F:1296, mrdx at F:2766), the scalar and w terms
// carry plain rdx/rdy (F:4119-4177, F:5695-5806).  Vertical divergence
// stays unweighted: exact for u (eqn 2.23), w (2.25) and the my-divided
// scalar form (2.26); for v the Fortran's (msfvy/msfvx) vertical ratio is
// identically 1 with the isotropic single msfv gpuwm carries
// (Lambert/Mercator/polar).  has_msf == 0 keeps the original expressions
// verbatim so the msf==1 path stays BITWISE Phase 2 (regression-pinned by
// tests/data/phase2_step_regression.npz) regardless of FMA contraction
// choices.

__device__ __forceinline__
real flux5(real qm3, real qm2, real qm1, real q0, real qp1, real qp2, real vel)
{
    return (vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            - fabsf(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))) / 60.0f;
}

__device__ __forceinline__
real flux3(real qm2, real qm1, real q0, real qp1, real vel)
{
    return (vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            + fabsf(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))) / 12.0f;
}

// Horizontal 3rd-order face flux (WRF flux3 with the flux5 upwinding
// sign; the vertical flux3 above carries the opposite, Omega-signed
// dissipation) -- used two faces in from an open boundary.
__device__ __forceinline__
real flux3h(real qm2, real qm1, real q0, real qp1, real vel)
{
    return (vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            - fabsf(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))) / 12.0f;
}

__device__ __forceinline__
real flux2(real qm1, real q0, real vel)
{
    return 0.5f * vel * (q0 + qm1);
}

// x-face flux (face f = 0..n) of a CELL-type field q (row width nxs) with
// an OPEN x boundary pair: 0 at the never-consumed boundary faces, WRF's
// degraded orders near them, full 5th order in the interior (no wrap).
__device__ __forceinline__
real xface_cell_open(const real* q, real vel, int k, int j, int f,
                     int n, int ny, int nxs)
{
    if (f <= 0 || f >= n) return 0.0f;
    int d = min(f, n - f);
    if (d == 1)
        return flux2(q[I3S(k, j, f - 1, ny, nxs)],
                     q[I3S(k, j, f, ny, nxs)], vel);
    if (d == 2)
        return flux3h(q[I3S(k, j, f - 2, ny, nxs)],
                      q[I3S(k, j, f - 1, ny, nxs)],
                      q[I3S(k, j, f, ny, nxs)],
                      q[I3S(k, j, f + 1, ny, nxs)], vel);
    return flux5(q[I3S(k, j, f - 3, ny, nxs)],
                 q[I3S(k, j, f - 2, ny, nxs)],
                 q[I3S(k, j, f - 1, ny, nxs)],
                 q[I3S(k, j, f, ny, nxs)],
                 q[I3S(k, j, f + 1, ny, nxs)],
                 q[I3S(k, j, f + 2, ny, nxs)], vel);
}

// Periodic counterpart (full 5th order, wrapped reads).
__device__ __forceinline__
real xface_cell_per(const real* q, real vel, int k, int j, int f,
                    int n, int ny, int nxs)
{
    return flux5(q[I3S(k, j, PERIODIC(f - 3, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f - 2, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f - 1, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f + 1, n), ny, nxs)],
                 q[I3S(k, j, PERIODIC(f + 2, n), ny, nxs)], vel);
}

// y-face analogues (face g = 0..n over the nys rows of q).
__device__ __forceinline__
real yface_cell_open(const real* q, real vel, int k, int g, int i,
                     int n, int nys, int nxs)
{
    if (g <= 0 || g >= n) return 0.0f;
    int d = min(g, n - g);
    if (d == 1)
        return flux2(q[I3S(k, g - 1, i, nys, nxs)],
                     q[I3S(k, g, i, nys, nxs)], vel);
    if (d == 2)
        return flux3h(q[I3S(k, g - 2, i, nys, nxs)],
                      q[I3S(k, g - 1, i, nys, nxs)],
                      q[I3S(k, g, i, nys, nxs)],
                      q[I3S(k, g + 1, i, nys, nxs)], vel);
    return flux5(q[I3S(k, g - 3, i, nys, nxs)],
                 q[I3S(k, g - 2, i, nys, nxs)],
                 q[I3S(k, g - 1, i, nys, nxs)],
                 q[I3S(k, g, i, nys, nxs)],
                 q[I3S(k, g + 1, i, nys, nxs)],
                 q[I3S(k, g + 2, i, nys, nxs)], vel);
}

__device__ __forceinline__
real yface_cell_per(const real* q, real vel, int k, int g, int i,
                    int n, int nys, int nxs)
{
    return flux5(q[I3S(k, PERIODIC(g - 3, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g - 2, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g - 1, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g + 1, n), i, nys, nxs)],
                 q[I3S(k, PERIODIC(g + 2, n), i, nys, nxs)], vel);
}

// Vertical face flux for a half-level (mass/u/v-point) field q at w-level
// kf (0..nz): zero at the boundaries, WRF's stretched-grid fnm/fnp-weighted
// 2nd-order face value one face in (module_advect_em.F vert_order 3:
// vflux = rom*(fzm(k)*f(k) + fzp(k)*f(k-1)) at k=kts+1 and k=ktf, scalars
// :4322/:4327, u :1486/:1490, v mirrors; reduces bitwise to the 0.5/0.5
// average on a uniform grid), 3rd order in the interior.  q is indexed
// with row width nxs (nx or nx+1).
__device__ __forceinline__
real zface_half(const real* q, real vel, int kf, int j, int i,
                int nz, int ny, int nxs,
                const real* fnm, const real* fnp)
{
    if (kf == 0 || kf == nz) return 0.0f;
    if (kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * q[I3S(kf,     j, i, ny, nxs)]
                      + fnp[kf] * q[I3S(kf - 1, j, i, ny, nxs)]);
    return flux3(q[I3S(kf - 2, j, i, ny, nxs)],
                 q[I3S(kf - 1, j, i, ny, nxs)],
                 q[I3S(kf,     j, i, ny, nxs)],
                 q[I3S(kf + 1, j, i, ny, nxs)], vel);
}

// ---------------------------------------------------------------------------
// Scalar q at mass points (nz, ny, nx).
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_scalar(const real* __restrict__ q,
                     const real* __restrict__ ru,       // (nz, ny, nx+1)
                     const real* __restrict__ rv,       // (nz, ny+1, nx)
                     const real* __restrict__ rw,       // (nz+1, ny, nx)
                     real* __restrict__ tend_out,       // (nz, ny, nx) +=
                     const real* __restrict__ rdnw,     // (nz,)
                     const real* __restrict__ fnm,      // (nz,) face weights
                     const real* __restrict__ fnp,      // (nz,)
                     const real* __restrict__ msf,      // (ny, nx) mass-pt
                     real dx_inv, real dy_inv,
                     int nz, int ny, int nx,
                     int open_x, int open_y, int has_msf, int spec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    if ((open_x || open_y) && !has_msf) {              // WRF advect_scalar
        real t = 0.0f;                                 // open path, msf==1:
                                                       // ORIGINAL Task-11
                                                       // body kept verbatim
                                                       // (bitwise-pinned)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ru0 = ru[I3(k, j, 0, ny, nx + 1)];
                real ru1 = ru[I3(k, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ru0 + ru1), 0.0f);
                t += -dx_inv * (ub * (q[IDX3(k, j, 1)] - q[IDX3(k, j, 0)])
                                + q[IDX3(k, j, 0)] * (ru1 - ru0));
            } else {
                real ru0 = ru[I3(k, j, nx - 1, ny, nx + 1)];
                real ru1 = ru[I3(k, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ru0 + ru1), 0.0f);
                t += -dx_inv * (ub * (q[IDX3(k, j, nx - 1)]
                                      - q[IDX3(k, j, nx - 2)])
                                + q[IDX3(k, j, nx - 1)] * (ru1 - ru0));
            }
        } else {
            real fx0, fx1;
            real v0 = ru[I3(k, j, i, ny, nx + 1)];
            real v1 = ru[I3(k, j, i + 1, ny, nx + 1)];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            t += -(fx1 - fx0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rv0 = rv[I3(k, 0, i, ny + 1, nx)];
                real rv1 = rv[I3(k, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rv0 + rv1), 0.0f);
                t += -dy_inv * (vb * (q[IDX3(k, 1, i)] - q[IDX3(k, 0, i)])
                                + q[IDX3(k, 0, i)] * (rv1 - rv0));
            } else {
                real rv0 = rv[I3(k, ny - 1, i, ny + 1, nx)];
                real rv1 = rv[I3(k, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rv0 + rv1), 0.0f);
                t += -dy_inv * (vb * (q[IDX3(k, ny - 1, i)]
                                      - q[IDX3(k, ny - 2, i)])
                                + q[IDX3(k, ny - 1, i)] * (rv1 - rv0));
            }
        } else {
            real fy0, fy1;
            real v0 = rv[I3(k, j, i, ny + 1, nx)];
            real v1 = rv[I3(k, j + 1, i, ny + 1, nx)];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            t += -(fy1 - fy0) * dy_inv;
        }
        real fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[I3(kf, j, i, ny, nx)], kf, j, i,
                                nz, ny, nx, fnm, fnp);
        }
        tend_out[IDX3(k, j, i)] += t - (fzo[1] - fzo[0]) * rdnw[k];
        return;
    }

    if (open_x || open_y) {                            // WRF advect_scalar
        real th = 0.0f;                                // open path with msf
        real tb = 0.0f;                                // (FIX-A): weighted
                                                       // flux divergences /
                                                       // plain non-cb terms
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: no x tendency at boundary cells
                // (F:4037-4038); non-cb terms are open-only (F:4115).
            } else if (i == 0) {                       // (plain rdx, WRF
                real ru0 = ru[I3(k, j, 0, ny, nx + 1)];        // F:4119)
                real ru1 = ru[I3(k, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ru0 + ru1), 0.0f);
                tb += -dx_inv * (ub * (q[IDX3(k, j, 1)] - q[IDX3(k, j, 0)])
                                 + q[IDX3(k, j, 0)] * (ru1 - ru0));
            } else {
                real ru0 = ru[I3(k, j, nx - 1, ny, nx + 1)];
                real ru1 = ru[I3(k, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ru0 + ru1), 0.0f);
                tb += -dx_inv * (ub * (q[IDX3(k, j, nx - 1)]
                                       - q[IDX3(k, j, nx - 2)])
                                 + q[IDX3(k, j, nx - 1)] * (ru1 - ru0));
            }
        } else {
            real fx0, fx1;
            real v0 = ru[I3(k, j, i, ny, nx + 1)];
            real v1 = ru[I3(k, j, i + 1, ny, nx + 1)];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            th += -(fx1 - fx0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rv0 = rv[I3(k, 0, i, ny + 1, nx)];
                real rv1 = rv[I3(k, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rv0 + rv1), 0.0f);
                tb += -dy_inv * (vb * (q[IDX3(k, 1, i)] - q[IDX3(k, 0, i)])
                                 + q[IDX3(k, 0, i)] * (rv1 - rv0));
            } else {
                real rv0 = rv[I3(k, ny - 1, i, ny + 1, nx)];
                real rv1 = rv[I3(k, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rv0 + rv1), 0.0f);
                tb += -dy_inv * (vb * (q[IDX3(k, ny - 1, i)]
                                       - q[IDX3(k, ny - 2, i)])
                                 + q[IDX3(k, ny - 1, i)] * (rv1 - rv0));
            }
        } else {
            real fy0, fy1;
            real v0 = rv[I3(k, j, i, ny + 1, nx)];
            real v1 = rv[I3(k, j + 1, i, ny + 1, nx)];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            th += -(fy1 - fy0) * dy_inv;
        }
        real fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[I3(kf, j, i, ny, nx)], kf, j, i,
                                nz, ny, nx, fnm, fnp);
        }
        real t = msf[(size_t)j * nx + i] * th + tb;    // WRF mrdx/mrdy
        tend_out[IDX3(k, j, i)] += t - (fzo[1] - fzo[0]) * rdnw[k];
        return;
    }

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // u-face index 0..nx
        fx[s] = flux5(q[IDX3(k, j, PERIODIC(f - 3, nx))],
                      q[IDX3(k, j, PERIODIC(f - 2, nx))],
                      q[IDX3(k, j, PERIODIC(f - 1, nx))],
                      q[IDX3(k, j, PERIODIC(f,     nx))],
                      q[IDX3(k, j, PERIODIC(f + 1, nx))],
                      q[IDX3(k, j, PERIODIC(f + 2, nx))],
                      ru[I3(k, j, f, ny, nx + 1)]);
        int g = j + s;                                 // v-face index 0..ny
        fy[s] = flux5(q[IDX3(k, PERIODIC(g - 3, ny), i)],
                      q[IDX3(k, PERIODIC(g - 2, ny), i)],
                      q[IDX3(k, PERIODIC(g - 1, ny), i)],
                      q[IDX3(k, PERIODIC(g,     ny), i)],
                      q[IDX3(k, PERIODIC(g + 1, ny), i)],
                      q[IDX3(k, PERIODIC(g + 2, ny), i)],
                      rv[I3(k, g, i, ny + 1, nx)]);
        int kf = k + s;                                // w-face index 0..nz
        real vel = rw[I3(kf, j, i, ny, nx)];
        fz[s] = zface_half(q, vel, kf, j, i, nz, ny, nx, fnm, fnp);
    }
    if (has_msf) {                       // WRF mrdx/mrdy = msft*rdx/rdy
        tend_out[IDX3(k, j, i)] += msf[(size_t)j * nx + i]
                                   * (-(fx[1] - fx[0]) * dx_inv
                                      - (fy[1] - fy[0]) * dy_inv)
                                   - (fz[1] - fz[0]) * rdnw[k];
    } else {                             // original: bitwise Phase 2
        tend_out[IDX3(k, j, i)] += -(fx[1] - fx[0]) * dx_inv
                                   - (fy[1] - fy[0]) * dy_inv
                                   - (fz[1] - fz[0]) * rdnw[k];
    }
}

// ---------------------------------------------------------------------------
// u momentum at u-points (nz, ny, nx+1).  x-faces of the u control volume
// sit at mass centers m (advecting flux 0.5*(ru[m]+ru[m+1])); y- and z-faces
// sit at corner points (rv, rw averaged across the two mass columns
// straddling the u-point).
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_u(const real* __restrict__ u,            // (nz, ny, nx+1)
                const real* __restrict__ ru,           // (nz, ny, nx+1)
                const real* __restrict__ rv,           // (nz, ny+1, nx)
                const real* __restrict__ rw,           // (nz+1, ny, nx)
                real* __restrict__ tend_out,           // (nz, ny, nx+1) +=
                const real* __restrict__ rdnw,         // (nz,)
                const real* __restrict__ fnm,          // (nz,) face weights
                const real* __restrict__ fnp,          // (nz,)
                const real* __restrict__ msf,          // (ny, nx+1) u-pt
                real dx_inv, real dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // u-point 0..nx
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx + 1 || j >= ny || k >= nz) return;

    if (open_x || open_y) {                            // WRF advect_u
        int nxs = nx + 1;                              // open-BC path
        bool xedge = open_x && (i == 0 || i == nx);    // boundary-normal face
        int cl = open_x ? max(i - 1, 0) : PERIODIC(i - 1, nx);
        int cr = open_x ? min(i, nx - 1) : PERIODIC(i, nx);
        int iu = open_x ? i : cr;                      // u read column: the
        real th = 0.0f;                                // periodic duplicate
                                                       // face nx reads col 0

        if (!xedge) {                                  // x advection
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? cl : cr;            // mass-center face
                real velx = 0.5f * (ru[I3(k, j, m, ny, nxs)]
                                    + ru[I3(k, j, m + 1, ny, nxs)]);
                if (open_x) {
                    int d = min(m, nx - 1 - m);
                    if (d == 0) {
                        // WRF "specified uses upstream normal wind at
                        // boundaries" (advect_u F:690-723): fqx(ids+1)
                        // takes ub = u(ids+1) when u(ids+1) < 0, fqx(ide)
                        // takes ub = u(ide-1) when u(ide-1) > 0.
                        real qa = u[I3(k, j, m, ny, nxs)];
                        real qb = u[I3(k, j, m + 1, ny, nxs)];
                        if (spec) {
                            if (m == 0)      { if (qb < 0.0f) qa = qb; }
                            else             { if (qa > 0.0f) qb = qa; }
                        }
                        fx2[s] = flux2(qa, qb, velx);
                    }
                    else if (d == 1)
                        fx2[s] = flux3h(u[I3(k, j, m - 1, ny, nxs)],
                                        u[I3(k, j, m, ny, nxs)],
                                        u[I3(k, j, m + 1, ny, nxs)],
                                        u[I3(k, j, m + 2, ny, nxs)], velx);
                    else
                        fx2[s] = flux5(u[I3(k, j, m - 2, ny, nxs)],
                                       u[I3(k, j, m - 1, ny, nxs)],
                                       u[I3(k, j, m, ny, nxs)],
                                       u[I3(k, j, m + 1, ny, nxs)],
                                       u[I3(k, j, m + 2, ny, nxs)],
                                       u[I3(k, j, m + 3, ny, nxs)], velx);
                } else {
                    fx2[s] = flux5(u[I3(k, j, PERIODIC(m - 2, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m - 1, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 1, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 2, nx), ny, nxs)],
                                   u[I3(k, j, PERIODIC(m + 3, nx), ny, nxs)],
                                   velx);
                }
            }
            th += -(fx2[1] - fx2[0]) * dx_inv;
        }

        if (!xedge) {                                  // y advection
            if (open_y && (j == 0 || j == ny - 1)) {   // non-cb open y-term
                if (spec) {
                    // WRF specified: u rows jds/jde-1 get no y tendency
                    // (degrade bounds j_start=jds+1, j_end=jde-2); the
                    // non-cb y-terms are open-only (F:1292/:1314).
                } else if (j == 0) {                   // (msf-weighted with
                    real vw = 0.5f * (rv[I3(k, 0, cl, ny + 1, nx)]   // th:
                                      + rv[I3(k, 0, cr, ny + 1, nx)]);
                    real vb = fminf(vw, 0.0f);         // WRF mrdy, F:1296)
                    real dsum = (rv[I3(k, 1, cl, ny + 1, nx)]
                                 - rv[I3(k, 0, cl, ny + 1, nx)])
                              + (rv[I3(k, 1, cr, ny + 1, nx)]
                                 - rv[I3(k, 0, cr, ny + 1, nx)]);
                    th += -dy_inv * (vb * (u[I3(k, 1, iu, ny, nxs)]
                                           - u[I3(k, 0, iu, ny, nxs)])
                                     + 0.5f * u[I3(k, 0, iu, ny, nxs)] * dsum);
                } else {
                    real vw = 0.5f * (rv[I3(k, ny, cl, ny + 1, nx)]
                                      + rv[I3(k, ny, cr, ny + 1, nx)]);
                    real vb = fmaxf(vw, 0.0f);
                    real dsum = (rv[I3(k, ny, cl, ny + 1, nx)]
                                 - rv[I3(k, ny - 1, cl, ny + 1, nx)])
                              + (rv[I3(k, ny, cr, ny + 1, nx)]
                                 - rv[I3(k, ny - 1, cr, ny + 1, nx)]);
                    th += -dy_inv * (vb * (u[I3(k, ny - 1, iu, ny, nxs)]
                                           - u[I3(k, ny - 2, iu, ny, nxs)])
                                     + 0.5f * u[I3(k, ny - 1, iu, ny, nxs)]
                                       * dsum);
                }
            } else {
                real fy2[2];
                for (int s = 0; s < 2; ++s) {
                    int g = j + s;                     // corner row 0..ny
                    real vely = 0.5f * (rv[I3(k, g, cl, ny + 1, nx)]
                                        + rv[I3(k, g, cr, ny + 1, nx)]);
                    fy2[s] = open_y
                        ? yface_cell_open(u, vely, k, g, iu, ny, ny, nxs)
                        : yface_cell_per(u, vely, k, g, iu, ny, ny, nxs);
                }
                th += -(fy2[1] - fy2[0]) * dy_inv;
            }
        }

        // WRF advect_u weights every horizontal term (flux divergences AND
        // the non-cb open y-term) by msfux at the u point (mrdx F:740,
        // mrdy F:633/1296); the vertical term below stays unweighted.
        real t = has_msf ? msf[(size_t)j * (nx + 1) + i] * th : th;

        // z advection: retained at the boundary-normal faces with the
        // boundary cell's Omega unless open_y (Fortran advect_u vertical
        // bounds: open_x exclusions commented out, open_ys/ye + specified
        // active).
        if (!(xedge && open_y)) {
            real fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                real velz;
                if (open_x && i == 0)
                    velz = rw[I3(kf, j, 0, ny, nx)];
                else if (open_x && i == nx)
                    velz = rw[I3(kf, j, nx - 1, ny, nx)];
                else
                    velz = 0.5f * (rw[I3(kf, j, cl, ny, nx)]
                                   + rw[I3(kf, j, cr, ny, nx)]);
                fz2[s] = zface_half(u, velz, kf, j, iu, nz, ny, nxs, fnm, fnp);
            }
            t += -(fz2[1] - fz2[0]) * rdnw[k];
        }
        tend_out[I3(k, j, i, ny, nxs)] += t;
        return;
    }

    int ic   = PERIODIC(i, nx);                        // right mass column
    int im1c = PERIODIC(i - 1, nx);                    // left mass column

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int m = (s == 0) ? im1c : ic;                  // mass-center face
        real velx = 0.5f * (ru[I3(k, j, m,     ny, nx + 1)]
                          + ru[I3(k, j, m + 1, ny, nx + 1)]);
        fx[s] = flux5(u[I3(k, j, PERIODIC(m - 2, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m - 1, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m,     nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 1, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 2, nx), ny, nx + 1)],
                      u[I3(k, j, PERIODIC(m + 3, nx), ny, nx + 1)],
                      velx);
        int g = j + s;                                 // corner row 0..ny
        real vely = 0.5f * (rv[I3(k, g, im1c, ny + 1, nx)]
                          + rv[I3(k, g, ic,   ny + 1, nx)]);
        fy[s] = flux5(u[I3(k, PERIODIC(g - 3, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g - 2, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g - 1, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g,     ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g + 1, ny), ic, ny, nx + 1)],
                      u[I3(k, PERIODIC(g + 2, ny), ic, ny, nx + 1)],
                      vely);
        int kf = k + s;
        real velz = 0.5f * (rw[I3(kf, j, im1c, ny, nx)]
                          + rw[I3(kf, j, ic,   ny, nx)]);
        fz[s] = zface_half(u, velz, kf, j, ic, nz, ny, nx + 1, fnm, fnp);
    }
    if (has_msf) {                       // WRF advect_u: msfux at the u point
        tend_out[I3(k, j, i, ny, nx + 1)] += msf[(size_t)j * (nx + 1) + i]
                                             * (-(fx[1] - fx[0]) * dx_inv
                                                - (fy[1] - fy[0]) * dy_inv)
                                             - (fz[1] - fz[0]) * rdnw[k];
    } else {                             // original: bitwise Phase 2
        tend_out[I3(k, j, i, ny, nx + 1)] += -(fx[1] - fx[0]) * dx_inv
                                             - (fy[1] - fy[0]) * dy_inv
                                             - (fz[1] - fz[0]) * rdnw[k];
    }
}

// ---------------------------------------------------------------------------
// v momentum at v-points (nz, ny+1, nx).  Mirror image of flux_div_u.
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_v(const real* __restrict__ v,            // (nz, ny+1, nx)
                const real* __restrict__ ru,           // (nz, ny, nx+1)
                const real* __restrict__ rv,           // (nz, ny+1, nx)
                const real* __restrict__ rw,           // (nz+1, ny, nx)
                real* __restrict__ tend_out,           // (nz, ny+1, nx) +=
                const real* __restrict__ rdnw,         // (nz,)
                const real* __restrict__ fnm,          // (nz,) face weights
                const real* __restrict__ fnp,          // (nz,)
                const real* __restrict__ msf,          // (ny+1, nx) v-pt
                real dx_inv, real dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;     // v-point 0..ny
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny + 1 || k >= nz) return;

    if (open_x || open_y) {                            // WRF advect_v
        int nys = ny + 1;                              // open-BC path
        bool yedge = open_y && (j == 0 || j == ny);    // boundary-normal face
        int rs = open_y ? max(j - 1, 0) : PERIODIC(j - 1, ny);
        int rn = open_y ? min(j, ny - 1) : PERIODIC(j, ny);
        int jv = open_y ? j : rn;                      // v read row: the
        real th = 0.0f;                                // periodic duplicate
                                                       // face ny reads row 0

        if (!yedge) {                                  // y advection
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? rs : rn;            // mass-center face
                real vely = 0.5f * (rv[I3(k, m, i, nys, nx)]
                                    + rv[I3(k, m + 1, i, nys, nx)]);
                if (open_y) {
                    int d = min(m, ny - 1 - m);
                    if (d == 0) {
                        // WRF "specified uses upstream normal wind at
                        // boundaries" (advect_v F:1978-2013): fqy(jds+1)
                        // takes vb = v(jds+1) when v(jds+1) < 0, fqy(jde)
                        // takes vb = v(jde-1) when v(jde-1) > 0.
                        real qa = v[I3(k, m, i, nys, nx)];
                        real qb = v[I3(k, m + 1, i, nys, nx)];
                        if (spec) {
                            if (m == 0)      { if (qb < 0.0f) qa = qb; }
                            else             { if (qa > 0.0f) qb = qa; }
                        }
                        fy2[s] = flux2(qa, qb, vely);
                    }
                    else if (d == 1)
                        fy2[s] = flux3h(v[I3(k, m - 1, i, nys, nx)],
                                        v[I3(k, m, i, nys, nx)],
                                        v[I3(k, m + 1, i, nys, nx)],
                                        v[I3(k, m + 2, i, nys, nx)], vely);
                    else
                        fy2[s] = flux5(v[I3(k, m - 2, i, nys, nx)],
                                       v[I3(k, m - 1, i, nys, nx)],
                                       v[I3(k, m, i, nys, nx)],
                                       v[I3(k, m + 1, i, nys, nx)],
                                       v[I3(k, m + 2, i, nys, nx)],
                                       v[I3(k, m + 3, i, nys, nx)], vely);
                } else {
                    fy2[s] = flux5(v[I3(k, PERIODIC(m - 2, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m - 1, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m + 1, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m + 2, ny), i, nys, nx)],
                                   v[I3(k, PERIODIC(m + 3, ny), i, nys, nx)],
                                   vely);
                }
            }
            th += -(fy2[1] - fy2[0]) * dy_inv;
        }

        if (!yedge) {                                  // x advection
            if (open_x && (i == 0 || i == nx - 1)) {   // non-cb open x-term:
                // corner-averaged ru AT the boundary face (WRF advect_v
                // open_xs/xe, module_advect_em.F:2763/2785).
                if (spec) {
                    // WRF specified: v columns ids/ide-1 get no x tendency
                    // (F:2660-2661); the non-cb x-terms are open-only
                    // (F:2763/:2785).
                } else if (i == 0) {
                    real uw0 = 0.5f * (ru[I3(k, rs, 0, ny, nx + 1)]
                                       + ru[I3(k, rn, 0, ny, nx + 1)]);
                    real uw1 = 0.5f * (ru[I3(k, rs, 1, ny, nx + 1)]
                                       + ru[I3(k, rn, 1, ny, nx + 1)]);
                    real ub = fminf(uw0, 0.0f);
                    th += -dx_inv * (ub * (v[I3(k, jv, 1, nys, nx)]
                                           - v[I3(k, jv, 0, nys, nx)])
                                     + v[I3(k, jv, 0, nys, nx)] * (uw1 - uw0));
                } else {
                    real uw0 = 0.5f * (ru[I3(k, rs, nx - 1, ny, nx + 1)]
                                       + ru[I3(k, rn, nx - 1, ny, nx + 1)]);
                    real uw1 = 0.5f * (ru[I3(k, rs, nx, ny, nx + 1)]
                                       + ru[I3(k, rn, nx, ny, nx + 1)]);
                    real ub = fmaxf(uw1, 0.0f);
                    th += -dx_inv * (ub * (v[I3(k, jv, nx - 1, nys, nx)]
                                           - v[I3(k, jv, nx - 2, nys, nx)])
                                     + v[I3(k, jv, nx - 1, nys, nx)]
                                       * (uw1 - uw0));
                }
            } else {
                real fx2[2];
                for (int s = 0; s < 2; ++s) {
                    int f = i + s;                     // corner column 0..nx
                    real velx = 0.5f * (ru[I3(k, rs, f, ny, nx + 1)]
                                        + ru[I3(k, rn, f, ny, nx + 1)]);
                    fx2[s] = open_x
                        ? xface_cell_open(v, velx, k, jv, f, nx, nys, nx)
                        : xface_cell_per(v, velx, k, jv, f, nx, nys, nx);
                }
                th += -(fx2[1] - fx2[0]) * dx_inv;
            }
        }

        // WRF advect_v weights every horizontal term (flux divergences AND
        // the non-cb open x-term) by msfvy at the v point (mrdy F:2050,
        // mrdx F:2676/2766); the vertical term below stays unweighted.
        real t = has_msf ? msf[(size_t)j * nx + i] * th : th;

        // z advection: excluded at the boundary-normal faces whenever
        // open_y or specified (Fortran advect_v vertical open_ys/ye
        // j-bounds).
        if (!yedge) {
            real fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                real velz = 0.5f * (rw[I3(kf, rs, i, ny, nx)]
                                    + rw[I3(kf, rn, i, ny, nx)]);
                fz2[s] = zface_half(v, velz, kf, jv, i, nz, nys, nx, fnm, fnp);
            }
            t += -(fz2[1] - fz2[0]) * rdnw[k];
        }
        tend_out[I3(k, j, i, nys, nx)] += t;
        return;
    }

    int jc   = PERIODIC(j, ny);                        // north mass row
    int jm1c = PERIODIC(j - 1, ny);                    // south mass row

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // corner column 0..nx
        real velx = 0.5f * (ru[I3(k, jm1c, f, ny, nx + 1)]
                          + ru[I3(k, jc,   f, ny, nx + 1)]);
        fx[s] = flux5(v[I3(k, jc, PERIODIC(f - 3, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f - 2, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f - 1, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f,     nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f + 1, nx), ny + 1, nx)],
                      v[I3(k, jc, PERIODIC(f + 2, nx), ny + 1, nx)],
                      velx);
        int m = (s == 0) ? jm1c : jc;                  // mass-center face
        real vely = 0.5f * (rv[I3(k, m,     i, ny + 1, nx)]
                          + rv[I3(k, m + 1, i, ny + 1, nx)]);
        fy[s] = flux5(v[I3(k, PERIODIC(m - 2, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m - 1, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m,     ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m + 1, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m + 2, ny), i, ny + 1, nx)],
                      v[I3(k, PERIODIC(m + 3, ny), i, ny + 1, nx)],
                      vely);
        int kf = k + s;
        real velz = 0.5f * (rw[I3(kf, jm1c, i, ny, nx)]
                          + rw[I3(kf, jc,   i, ny, nx)]);
        fz[s] = zface_half(v, velz, kf, jc, i, nz, ny + 1, nx, fnm, fnp);
    }
    if (has_msf) {                       // WRF advect_v: msfvy at the v point
        tend_out[I3(k, j, i, ny + 1, nx)] += msf[(size_t)j * nx + i]
                                             * (-(fx[1] - fx[0]) * dx_inv
                                                - (fy[1] - fy[0]) * dy_inv)
                                             - (fz[1] - fz[0]) * rdnw[k];
    } else {                             // original: bitwise Phase 2
        tend_out[I3(k, j, i, ny + 1, nx)] += -(fx[1] - fx[0]) * dx_inv
                                             - (fy[1] - fy[0]) * dy_inv
                                             - (fz[1] - fz[0]) * rdnw[k];
    }
}

// ---------------------------------------------------------------------------
// w momentum at w-points (nz+1, ny, nx).  Horizontal advecting fluxes are
// ru/rv averaged across the two half levels straddling the w-level; the
// vertical flux of w lives at mass (half) levels m with advecting flux
// 0.5*(rw[m]+rw[m+1]).  Tendencies are computed only for interior w-levels
// k = 1..nz-1 (boundary w is set by the rigid-lid/flat-bottom BCs).
// ---------------------------------------------------------------------------
extern "C" __global__
void flux_div_w(const real* __restrict__ w,            // (nz+1, ny, nx)
                const real* __restrict__ ru,           // (nz, ny, nx+1)
                const real* __restrict__ rv,           // (nz, ny+1, nx)
                const real* __restrict__ rw,           // (nz+1, ny, nx)
                real* __restrict__ tend_out,           // (nz+1, ny, nx) +=
                const real* __restrict__ rdn,          // (nz,)
                const real* __restrict__ fnm,          // (nz,) w-level weights
                const real* __restrict__ fnp,          // (nz,)
                const real* __restrict__ msf,          // (ny, nx) mass-pt
                real dx_inv, real dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;     // w-level 0..nz
    if (i >= nx || j >= ny || k > nz) return;
    if (k == 0 || k == nz) return;                     // BC levels: no tendency

    if ((open_x || open_y) && !has_msf) {              // WRF advect_w
        real t = 0.0f;                                 // ORIGINAL Task-11
                                                       // body kept verbatim
                                                       // (bitwise-pinned)
        // horizontal advecting fluxes interpolated to the w level with
        // WRF's fnm/fnp weights (advect_w F:5004/:4531)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ruw0 = fnm[k] * ru[I3(k, j, 0, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 0, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ruw0 + ruw1), 0.0f);
                t += -dx_inv * (ub * (w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)])
                                + w[IDX3(k, j, 0)] * (ruw1 - ruw0));
            } else {
                real ruw0 = fnm[k] * ru[I3(k, j, nx - 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx - 1, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, nx, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ruw0 + ruw1), 0.0f);
                t += -dx_inv * (ub * (w[IDX3(k, j, nx - 1)]
                                      - w[IDX3(k, j, nx - 2)])
                                + w[IDX3(k, j, nx - 1)] * (ruw1 - ruw0));
            }
        } else {
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                real velx = fnm[k] * ru[I3(k, j, f, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            t += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rvw0 = fnm[k] * rv[I3(k, 0, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 0, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rvw0 + rvw1), 0.0f);
                t += -dy_inv * (vb * (w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)])
                                + w[IDX3(k, 0, i)] * (rvw1 - rvw0));
            } else {
                real rvw0 = fnm[k] * rv[I3(k, ny - 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny - 1, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, ny, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rvw0 + rvw1), 0.0f);
                t += -dy_inv * (vb * (w[IDX3(k, ny - 1, i)]
                                      - w[IDX3(k, ny - 2, i)])
                                + w[IDX3(k, ny - 1, i)] * (rvw1 - rvw0));
            }
        } else {
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                real vely = fnm[k] * rv[I3(k, g, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            t += -(fy2[1] - fy2[0]) * dy_inv;
        }
        real fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;                         // mass-level face
            real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
            if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
            else
                fz2[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                               w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)],
                               velz);
        }
        tend_out[IDX3(k, j, i)] += t - (fz2[1] - fz2[0]) * rdn[k];
        return;
    }

    if (open_x || open_y) {                            // WRF advect_w with
        real th = 0.0f;                                // msf (FIX-A):
                                                       // weighted flux
        real tb = 0.0f;                                // divergences / plain
                                                       // non-cb open terms
        // horizontal advecting fluxes interpolated to the w level with
        // WRF's fnm/fnp weights (advect_w F:5004/:4531)
        if (open_x && (i == 0 || i == nx - 1)) {       // non-cb open term
            if (spec) {
                // WRF specified: the boundary cells get NO advective
                // tendency along the specified axis (bounds ids+1..ide-2,
                // F:4037-4038 scalar / F:5570-5571 w); the non-cb terms
                // are gated on open_xs/xe only (F:4115/:5695).
            } else if (i == 0) {
                real ruw0 = fnm[k] * ru[I3(k, j, 0, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 0, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, 1, ny, nx + 1)];
                real ub = fminf(0.5f * (ruw0 + ruw1), 0.0f);
                tb += -dx_inv * (ub * (w[IDX3(k, j, 1)] - w[IDX3(k, j, 0)])
                                 + w[IDX3(k, j, 0)] * (ruw1 - ruw0));
            } else {
                real ruw0 = fnm[k] * ru[I3(k, j, nx - 1, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx - 1, ny, nx + 1)];
                real ruw1 = fnm[k] * ru[I3(k, j, nx, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, nx, ny, nx + 1)];
                real ub = fmaxf(0.5f * (ruw0 + ruw1), 0.0f);
                tb += -dx_inv * (ub * (w[IDX3(k, j, nx - 1)]
                                       - w[IDX3(k, j, nx - 2)])
                                 + w[IDX3(k, j, nx - 1)] * (ruw1 - ruw0));
            }
        } else {
            real fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                real velx = fnm[k] * ru[I3(k, j, f, ny, nx + 1)]
                          + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            th += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
                // WRF specified: no y tendency at boundary cells
                // (F:4056-4057 scalar / F:5607-5608 w); non-cb terms are
                // open-only (F:4147/:5775).
            } else if (j == 0) {
                real rvw0 = fnm[k] * rv[I3(k, 0, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 0, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, 1, i, ny + 1, nx)];
                real vb = fminf(0.5f * (rvw0 + rvw1), 0.0f);
                tb += -dy_inv * (vb * (w[IDX3(k, 1, i)] - w[IDX3(k, 0, i)])
                                 + w[IDX3(k, 0, i)] * (rvw1 - rvw0));
            } else {
                real rvw0 = fnm[k] * rv[I3(k, ny - 1, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny - 1, i, ny + 1, nx)];
                real rvw1 = fnm[k] * rv[I3(k, ny, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, ny, i, ny + 1, nx)];
                real vb = fmaxf(0.5f * (rvw0 + rvw1), 0.0f);
                tb += -dy_inv * (vb * (w[IDX3(k, ny - 1, i)]
                                       - w[IDX3(k, ny - 2, i)])
                                 + w[IDX3(k, ny - 1, i)] * (rvw1 - rvw0));
            }
        } else {
            real fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                real vely = fnm[k] * rv[I3(k, g, i, ny + 1, nx)]
                          + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            th += -(fy2[1] - fy2[0]) * dy_inv;
        }
        real fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;                         // mass-level face
            real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
            if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
            else
                fz2[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                               w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)],
                               velz);
        }
        // WRF advect_w mrdx/mrdy = msftx*rdx/rdy (F:5096); the non-cb open
        // terms carry plain rdx/rdy (F:5695).
        real t = msf[(size_t)j * nx + i] * th + tb;
        tend_out[IDX3(k, j, i)] += t - (fz2[1] - fz2[0]) * rdn[k];
        return;
    }

    real fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;                                 // u-face column 0..nx
        real velx = fnm[k] * ru[I3(k,     j, f, ny, nx + 1)]
                  + fnp[k] * ru[I3(k - 1, j, f, ny, nx + 1)];
        fx[s] = flux5(w[IDX3(k, j, PERIODIC(f - 3, nx))],
                      w[IDX3(k, j, PERIODIC(f - 2, nx))],
                      w[IDX3(k, j, PERIODIC(f - 1, nx))],
                      w[IDX3(k, j, PERIODIC(f,     nx))],
                      w[IDX3(k, j, PERIODIC(f + 1, nx))],
                      w[IDX3(k, j, PERIODIC(f + 2, nx))],
                      velx);
        int g = j + s;                                 // v-face row 0..ny
        real vely = fnm[k] * rv[I3(k,     g, i, ny + 1, nx)]
                  + fnp[k] * rv[I3(k - 1, g, i, ny + 1, nx)];
        fy[s] = flux5(w[IDX3(k, PERIODIC(g - 3, ny), i)],
                      w[IDX3(k, PERIODIC(g - 2, ny), i)],
                      w[IDX3(k, PERIODIC(g - 1, ny), i)],
                      w[IDX3(k, PERIODIC(g,     ny), i)],
                      w[IDX3(k, PERIODIC(g + 1, ny), i)],
                      w[IDX3(k, PERIODIC(g + 2, ny), i)],
                      vely);
        int m = k - 1 + s;                             // mass-level face 0..nz-1
        real velz = 0.5f * (rw[IDX3(m, j, i)] + rw[IDX3(m + 1, j, i)]);
        if (m == 0 || m == nz - 1)
            fz[s] = flux2(w[IDX3(m, j, i)], w[IDX3(m + 1, j, i)], velz);
        else
            fz[s] = flux3(w[IDX3(m - 1, j, i)], w[IDX3(m, j, i)],
                          w[IDX3(m + 1, j, i)], w[IDX3(m + 2, j, i)], velz);
    }
    if (has_msf) {                       // WRF advect_w: msftx at mass point
        tend_out[IDX3(k, j, i)] += msf[(size_t)j * nx + i]
                                   * (-(fx[1] - fx[0]) * dx_inv
                                      - (fy[1] - fy[0]) * dy_inv)
                                   - (fz[1] - fz[0]) * rdn[k];
    } else {                             // original: bitwise Phase 2
        tend_out[IDX3(k, j, i)] += -(fx[1] - fx[0]) * dx_inv
                                   - (fy[1] - fy[0]) * dy_inv
                                   - (fz[1] - fz[0]) * rdn[k];
    }
}
