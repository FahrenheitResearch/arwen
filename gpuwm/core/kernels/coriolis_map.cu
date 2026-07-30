// gpuwm/core/kernels/coriolis_map.cu
//
// Coriolis + curvature large-step tendencies (Phase 3 Task 3), transcribed
// from WRF v4.6.1 dyn_em/module_big_step_utilities_em.F SUBROUTINE coriolis
// and SUBROUTINE curvature (the rk_tendency slow-tendency slot; ARW tech
// note sec. 2.3-2.4, the F_U/F_V/F_W terms of eqns 2.23-2.25).
//
// Reductions relative to the Fortran, all documented:
//   * isotropic map factors: gpuwm carries one msf per staggering
//     (msfu == msfux == msfuy etc., exact for Lambert/Mercator/polar
//     stereographic), so the Fortran ratios msfux/msfuy, msfvy/msfvx and
//     msftx/msfty are identically 1 and are omitted; the msf GRADIENTS in
//     vxgm are kept in full.
//   * the polar / map_proj == 6 'tan(xlat)' branch of curvature is not
//     ported (Lambert phase); the "normal code" vxgm branch is.
//   * open/specified x/y boundaries exclude the boundary-normal u/v faces
//     owned by module_bc, matching WRF's loop bounds.  Tangential velocity
//     and w tendencies remain active at those boundaries.
//
// The sina/cosa rotation terms are transcribed IN FULL: WRF applies them
// unconditionally (module_em.F:761-769 passes grid sina/cosa — geo_em
// SINALPHA/COSALPHA, Registry.EM_COMMON:1405-1406 — for every projection),
// and on a Lambert grid away from stand_lon they are NOT identity:
// |e*sina| reaches ~28% of mean f on the real74 d01 domain.  The three
// rotation-aware coriolis terms (coriolis, module_big_step_utilities_em.F):
//   u (:3726-3729):  - <e>_x * <cosa>_x * <rw>
//   v (:3800-3803):  + <e>_y * <sina>_y * <rw>   (msfvy/msfvx == 1)
//   w (:3839-3844):  + e * (cosa*<ru> - sina*<rv>)  (msftx/msfty == 1)
// curvature carries no sina/cosa.  Unrotated grids pass sina = 0 /
// cosa = 1 (WRF's identity, :3703-3704) and reproduce the rotation-free
// algebra exactly.
//
// Inputs: ru/rv are the RK stage's coupled momenta from dycore.stage_fluxes
// ((c1h*<mu>_face + c2h)*u/msfu etc. — WRF couple_momentum); u/v/w the
// uncoupled stage winds; rw = (c1f*mut + c2f)*w/msft (WRF couple_momentum's
// 'w' coupling) is formed inline from mut = mub2d + mup.  f/e/sina/cosa at
// mass points; fzm/fzp are WRF's half->full interpolation weights
// (rk_tendency passes fnm/fnp).  One thread per (k, j, i) on the union of
// the u grid (j < ny), v grid (i < nx), and the interior w levels (k >= 1);
// the periodic duplicate faces (u face nx, v row ny) are computed with
// wrapped mass-point reads exactly like face 0 / row 0.
//
// Float64 mirror: gpuwm/verify/npref.py np_coriolis_curvature.

// Coupled w flux rw = (c1f*mut + c2f)*w/msft at full level kf of cell c.
static __device__ __forceinline__
real rw_at(const real* __restrict__ w, const real* __restrict__ mut,
           const real* __restrict__ msft,
           const real* __restrict__ c1f, const real* __restrict__ c2f,
           int kf, size_t c, size_t st)
{
    return (c1f[kf] * mut[c] + c2f[kf]) * w[(size_t)kf * st + c] / msft[c];
}

// WRF curvature vxgm at mass point (j, i): v cross grad m.  All reads are
// in-range on the staggered arrays (no wrap needed).
static __device__ __forceinline__
real vxgm_at(const real* __restrict__ u, const real* __restrict__ v,
             const real* __restrict__ msfu, const real* __restrict__ msfv,
             real rdx, real rdy, int k, int j, int i, int ny, int nx)
{
    int nxf = nx + 1;
    real ub = 0.5f * (u[I3S(k, j, i, ny, nxf)] + u[I3S(k, j, i + 1, ny, nxf)]);
    real vb = 0.5f * (v[I3S(k, j, i, ny + 1, nx)]
                      + v[I3S(k, j + 1, i, ny + 1, nx)]);
    return ub * (msfv[(size_t)(j + 1) * nx + i] - msfv[(size_t)j * nx + i])
               * rdy
         - vb * (msfu[(size_t)j * nxf + i + 1] - msfu[(size_t)j * nxf + i])
               * rdx;
}

extern "C" __global__
void coriolis_curvature(const real* __restrict__ ru,   // (nz, ny, nx+1)
                        const real* __restrict__ rv,   // (nz, ny+1, nx)
                        const real* __restrict__ u,    // (nz, ny, nx+1)
                        const real* __restrict__ v,    // (nz, ny+1, nx)
                        const real* __restrict__ w,    // (nz+1, ny, nx)
                        const real* __restrict__ mut,  // (ny, nx) total mass
                        const real* __restrict__ msft, // (ny, nx)
                        const real* __restrict__ msfu, // (ny, nx+1)
                        const real* __restrict__ msfv, // (ny+1, nx)
                        const real* __restrict__ f,    // (ny, nx)
                        const real* __restrict__ e,    // (ny, nx)
                        const real* __restrict__ sina, // (ny, nx)
                        const real* __restrict__ cosa, // (ny, nx)
                        const real* __restrict__ c1f,  // (nz+1,)
                        const real* __restrict__ c2f,  // (nz+1,)
                        const real* __restrict__ fzm,  // (nz,) == fnm
                        const real* __restrict__ fzp,  // (nz,) == fnp
                        real rdx, real rdy,
                        real* __restrict__ ru_t,       // (nz, ny, nx+1) +=
                        real* __restrict__ rv_t,       // (nz, ny+1, nx) +=
                        real* __restrict__ rw_t,       // (nz+1, ny, nx) +=
                        int boundary_x, int boundary_y,
                        int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    size_t st = (size_t)ny * nx;

    // WRF open/specified bounds leave boundary-normal velocity faces to
    // module_bc.  Tangential velocity and w tendencies remain active.
    if (j < ny && (!boundary_x || (i > 0 && i < nx))) {
        int ic = i % nx, imc = (i - 1 + nx) % nx;
        size_t cA = (size_t)j * nx + ic, cB = (size_t)j * nx + imc;
        // 4-point averages of rv (rows j, j+1 of the two flanking columns)
        // and of rw (full levels k, k+1) — WRF coriolis/curvature stencils.
        real rv4 = 0.25f * (rv[I3S(k, j,     imc, nyf, nx)]
                          + rv[I3S(k, j,     ic,  nyf, nx)]
                          + rv[I3S(k, j + 1, imc, nyf, nx)]
                          + rv[I3S(k, j + 1, ic,  nyf, nx)]);
        real rw4 = 0.25f * (rw_at(w, mut, msft, c1f, c2f, k,     cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k,     cA, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cA, st));
        // coriolis: +(msfux/msfuy == 1)*<f>_x*<rv> - <e>_x*<cosa>_x*<rw>
        // (WRF :3726-3729; the cosa average is grouped so cosa == 1
        // multiplies by exactly 1.0f — rotation-free grids bitwise.)
        real t = 0.5f * (f[cA] + f[cB]) * rv4
               - 0.5f * (e[cA] + e[cB])
                 * (0.5f * (cosa[cA] + cosa[cB])) * rw4;
        // curvature: +<vxgm>_x*<rv> - u*<rw>/a
        real vx = 0.5f * (vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, j, ic, ny, nx)
                        + vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, j, imc, ny, nx));
        t += vx * rv4
           - u[I3S(k, j, i, ny, nxf)] * RERADIUS * rw4;
        ru_t[I3S(k, j, i, ny, nxf)] += t;
    }

    if (i < nx && (!boundary_y || (j > 0 && j < ny))) {
        int jc = j % ny, jmc = (j - 1 + ny) % ny;
        size_t cA = (size_t)jc * nx + i, cB = (size_t)jmc * nx + i;
        real ru4 = 0.25f * (ru[I3S(k, jc,  i,     ny, nxf)]
                          + ru[I3S(k, jc,  i + 1, ny, nxf)]
                          + ru[I3S(k, jmc, i,     ny, nxf)]
                          + ru[I3S(k, jmc, i + 1, ny, nxf)]);
        real rw4 = 0.25f * (rw_at(w, mut, msft, c1f, c2f, k,     cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k,     cA, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cA, st));
        // coriolis: -(msfvy/msfvx == 1)*<f>_y*<ru>  (WRF :3800-3803)
        real t = -0.5f * (f[cA] + f[cB]) * ru4;
        // curvature: -<vxgm>_y*<ru> - (msfvy/msfvx == 1)*v*<rw>/a
        real vx = 0.5f * (vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, jc, i, ny, nx)
                        + vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, jmc, i, ny, nx));
        t += -vx * ru4
             - v[I3S(k, j, i, nyf, nx)] * RERADIUS * rw4;
        // coriolis rotation: +(msfvy/msfvx == 1)*<e>_y*<sina>_y*<rw>
        // (WRF :3800-3803).  Kept as a trailing accumulation so a zero
        // sina average adds an exact zero and the f/curvature FMA
        // contraction above is byte-identical to the pre-rotation kernel
        // — rotation-free grids stay bitwise.
        t += 0.5f * (e[cA] + e[cB])
             * (0.5f * (sina[cA] + sina[cB])) * rw4;
        rv_t[I3S(k, j, i, nyf, nx)] += t;
    }

    if (i < nx && j < ny && k >= 1) {            // interior w level k
        size_t c = (size_t)j * nx + i;
        // half->full interpolation (fzm/fzp) of the coupled and uncoupled
        // horizontal momenta straddling w level k.
        real ruf = 0.5f * (fzm[k] * (ru[I3S(k, j, i,     ny, nxf)]
                                   + ru[I3S(k, j, i + 1, ny, nxf)])
                         + fzp[k] * (ru[I3S(k - 1, j, i,     ny, nxf)]
                                   + ru[I3S(k - 1, j, i + 1, ny, nxf)]));
        real uf = 0.5f * (fzm[k] * (u[I3S(k, j, i,     ny, nxf)]
                                  + u[I3S(k, j, i + 1, ny, nxf)])
                        + fzp[k] * (u[I3S(k - 1, j, i,     ny, nxf)]
                                  + u[I3S(k - 1, j, i + 1, ny, nxf)]));
        real rvf = 0.5f * (fzm[k] * (rv[I3S(k, j,     i, nyf, nx)]
                                   + rv[I3S(k, j + 1, i, nyf, nx)])
                         + fzp[k] * (rv[I3S(k - 1, j,     i, nyf, nx)]
                                   + rv[I3S(k - 1, j + 1, i, nyf, nx)]));
        real vf = 0.5f * (fzm[k] * (v[I3S(k, j,     i, nyf, nx)]
                                  + v[I3S(k, j + 1, i, nyf, nx)])
                        + fzp[k] * (v[I3S(k - 1, j,     i, nyf, nx)]
                                  + v[I3S(k - 1, j + 1, i, nyf, nx)]));
        // coriolis: +e*(cosa*<ru> - (msftx/msfty == 1)*sina*<rv>)
        //           (WRF :3839-3844)
        // curvature: +(<ru><u> + (msftx/msfty == 1)*<rv><v>)/a
        rw_t[(size_t)k * st + c] += e[c] * (cosa[c] * ruf - sina[c] * rvf)
                                  + RERADIUS * (ruf * uf + rvf * vf);
    }
}
