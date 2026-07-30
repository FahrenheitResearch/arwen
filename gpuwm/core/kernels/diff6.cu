// gpuwm/core/kernels/diff6.cu
//
// WRF 6th-order horizontal numerical diffusion, transcribed from v4.6.1
// dyn_em/module_big_step_utilities_em.F SUBROUTINE sixth_order_diffusion
// (Knievel; references Xue MWR 2000, Durran 1999 sec. 2.4.3), specialized
// to gpuwm's storage: map factors 1 on the tendency, periodic x/y.  The
// Fortran's non-periodic loop trimming is applied host-side AFTER this
// kernel (gpuwm.core.dycore._zero_open_strips, width 3), and the
// outermost boundary-normal staggered face (WRF's u ide-3 / v jde-3,
// which WRF computes: its dflux_p1 reads field(i+3) = field(ide), the
// true boundary datum, which the wrapped FX/FY stencil below would
// corrupt with the OPPOSITE boundary's value) is replaced by the honest
// recomputation in kernels/diff6_seam.cu (dycore._launch_diff6_seam).
// This file must stay byte-stable outside comments: the seam fix is
// pinned bit-neutral off the seam faces against a capture of this
// kernel's 4d2ce99 binary (tests/test_diff6_boundary_face.py).
//
// diff6 ADDS the coupled tendency for ONE field into tend:
//
//   dflux_p0 = 10*(f(i)-f(i-1)) - 5*(f(i+1)-f(i-2)) + (f(i+2)-f(i-3))
//   dflux_p1 = the same one face to the right           (Xue eq. 3)
//   monotonic (mono == 2): dflux zeroed when dflux*(local gradient) <= 0
//                                                       (Xue eq. 10 variant)
//   tendency += coef * (mu_p1*dflux_p1 - mu_p0*dflux_p0)   per direction,
//
// with coef = diff_6th_factor * 0.015625 / (2*dt) -- the factor/2^6
// normalization: integrated over the full dt this removes `factor` of a
// 2-D 2dx checkerboard's amplitude per step (factor/2 per direction).  No
// grid spacing enters; the damping rate is resolution-independent.
//
// mu_p0/p1 are the hybrid face masses c1[k]*<MUT> + c2[k] averaged to the
// flux location exactly as the Fortran's mu_avg_p0/p1:
//   variant 1 (u):    x-fluxes at mass centers (MUT of the donor cell),
//                     y-fluxes at corners (4-point MUT average);
//   variant 2 (v):    the transpose of variant 1;
//   variant 0 (mass/w points): both flux sets at the cell faces (2-point).
// c1/c2 carry nlev entries: c1h/c2h for half-level fields, c1f/c2f for w
// (wstag = 1), whose BC-pinned boundary levels k = 0 and nlev-1 get no
// tendency (the Fortran's kts+1..ktf loop for 'w').
//
// diff_6th_slopeopt (WRF sixth_order_diffusion slopeopt branch, Fortran
// :6487-6501 x / :6569-6583 y): with slopeopt >= 1 each face flux is
// scaled by slopedamp = MAX(1 - dzmax/dzthr, 0), dzthr =
// diff_6th_thresh*9.81*dx (the routine's literal 9.81), dzmax the
// BASE-state (phb, full-level array read at the field's own level index
// k) face geopotential jump scaled by the face msf (msfux for x slopes,
// msfvy for y); the u/v variants take the max over their two adjacent
// mass faces exactly as the Fortran.  slopeopt = 0 keeps the untapered
// arithmetic bitwise (slopedamp stays the exact constant 1.0f and never
// reads phb/msfu/msfv, so flat and legacy calls pass null-equivalent
// dummies).
//
// Storage is (nlev, nys, nxs) with the periodic core (ny, nx); stencil
// reads wrap over the core, so the redundant staggered column (nxs = nx+1)
// or row (nys = ny+1) receives a tendency identical to column/row 0 -- the
// same convention as advection.cu / diffusion.cu.

extern "C" __global__
void diff6(const real* __restrict__ f,      // (nlev, nys, nxs)
           real* __restrict__ tend,         // (nlev, nys, nxs) +=
           const real* __restrict__ mut,    // (ny, nx) total dry mass
           const real* __restrict__ c1,     // (nlev,) hybrid c1
           const real* __restrict__ c2,     // (nlev,) hybrid c2
           const real* __restrict__ phb,    // (>=nlev, ny, nx) base geopot.
           const real* __restrict__ msfu,   // (ny, nx+1) u-face msf
           const real* __restrict__ msfv,   // (ny+1, nx) v-face msf
           real coef, int mono, int slopeopt,
           real dzthr_x, real dzthr_y,
           int nlev, int ny, int nys, int nx, int nxs,
           int variant, int wstag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nxs || j >= nys || k >= nlev) return;
    if (wstag && (k == 0 || k == nlev - 1)) return;  // BC-pinned w levels

    int ic = PERIODIC(i, nx);
    int jc = PERIODIC(j, ny);

#define FX(m) f[I3S(k, jc, PERIODIC(ic + (m), nx), nys, nxs)]
#define FY(m) f[I3S(k, PERIODIC(jc + (m), ny), ic, nys, nxs)]
#define MU(jj, ii) mut[(size_t)PERIODIC(jj, ny) * nx + PERIODIC(ii, nx)]
#define PHB(jj, ii) phb[I3(k, PERIODIC(jj, ny), PERIODIC(ii, nx), ny, nx)]
#define MSFX(jj, ii) msfu[(size_t)PERIODIC(jj, ny) * (nx + 1) \
                          + PERIODIC(ii, nx)]
#define MSFY(jj, ii) msfv[(size_t)PERIODIC(jj, ny) * nx + PERIODIC(ii, nx)]

    // ---- diffusion in x (Fortran "Diffusion in x (i index)") ----
    real dflux_x_p0 = 10.0f * (FX(0) - FX(-1)) - 5.0f * (FX(1) - FX(-2))
                      + (FX(2) - FX(-3));
    real dflux_x_p1 = 10.0f * (FX(1) - FX(0)) - 5.0f * (FX(2) - FX(-1))
                      + (FX(3) - FX(-2));
    if (mono == 2) {                       // prohibit up-gradient diffusion
        if (dflux_x_p0 * (FX(0) - FX(-1)) <= 0.0f) dflux_x_p0 = 0.0f;
        if (dflux_x_p1 * (FX(1) - FX(0)) <= 0.0f) dflux_x_p1 = 0.0f;
    }
    real mu_x_p0, mu_x_p1;
    if (variant == 1) {                    // u: fluxes at mass centers
        mu_x_p0 = MU(jc, ic - 1);
        mu_x_p1 = MU(jc, ic);
    } else if (variant == 2) {             // v: fluxes at corners
        mu_x_p0 = 0.25f * (MU(jc - 1, ic - 1) + MU(jc - 1, ic)
                           + MU(jc, ic - 1) + MU(jc, ic));
        mu_x_p1 = 0.25f * (MU(jc - 1, ic) + MU(jc - 1, ic + 1)
                           + MU(jc, ic) + MU(jc, ic + 1));
    } else {                               // mass/w points: fluxes at faces
        mu_x_p0 = 0.5f * (MU(jc, ic - 1) + MU(jc, ic));
        mu_x_p1 = 0.5f * (MU(jc, ic) + MU(jc, ic + 1));
    }
    real sdx_p0 = 1.0f, sdx_p1 = 1.0f;     // slopeopt taper (Fortran
    if (slopeopt >= 1) {                   // :6487-6501)
        real a0 = fabsf(PHB(jc, ic) - PHB(jc, ic - 1)) * MSFX(jc, ic);
        real a1 = fabsf(PHB(jc, ic + 1) - PHB(jc, ic)) * MSFX(jc, ic + 1);
        real dz0 = a0, dz1 = a1;
        if (variant == 1) {                // u: max over the two mass faces
            dz0 = fmaxf(a0, fabsf(PHB(jc, ic - 1) - PHB(jc, ic - 2))
                            * MSFX(jc, ic - 1));
            dz1 = fmaxf(a1, a0);
        } else if (variant == 2) {         // v: max over rows j and j-1
            dz0 = fmaxf(a0, fabsf(PHB(jc - 1, ic) - PHB(jc - 1, ic - 1))
                            * MSFX(jc - 1, ic));
            dz1 = fmaxf(a1, fabsf(PHB(jc - 1, ic + 1) - PHB(jc - 1, ic))
                            * MSFX(jc - 1, ic + 1));
        }
        sdx_p0 = fmaxf(1.0f - dz0 / dzthr_x, 0.0f);
        sdx_p1 = fmaxf(1.0f - dz1 / dzthr_x, 0.0f);
    }
    real tendency_x = coef
        * (sdx_p1 * (c1[k] * mu_x_p1 + c2[k]) * dflux_x_p1
           - sdx_p0 * (c1[k] * mu_x_p0 + c2[k]) * dflux_x_p0);

    // ---- diffusion in y (Fortran "Diffusion in y (j index)") ----
    real dflux_y_p0 = 10.0f * (FY(0) - FY(-1)) - 5.0f * (FY(1) - FY(-2))
                      + (FY(2) - FY(-3));
    real dflux_y_p1 = 10.0f * (FY(1) - FY(0)) - 5.0f * (FY(2) - FY(-1))
                      + (FY(3) - FY(-2));
    if (mono == 2) {
        if (dflux_y_p0 * (FY(0) - FY(-1)) <= 0.0f) dflux_y_p0 = 0.0f;
        if (dflux_y_p1 * (FY(1) - FY(0)) <= 0.0f) dflux_y_p1 = 0.0f;
    }
    real mu_y_p0, mu_y_p1;
    if (variant == 1) {                    // u: fluxes at corners
        mu_y_p0 = 0.25f * (MU(jc - 1, ic - 1) + MU(jc - 1, ic)
                           + MU(jc, ic - 1) + MU(jc, ic));
        mu_y_p1 = 0.25f * (MU(jc, ic - 1) + MU(jc, ic)
                           + MU(jc + 1, ic - 1) + MU(jc + 1, ic));
    } else if (variant == 2) {             // v: fluxes at mass centers
        mu_y_p0 = MU(jc - 1, ic);
        mu_y_p1 = MU(jc, ic);
    } else {                               // mass/w points: fluxes at faces
        mu_y_p0 = 0.5f * (MU(jc - 1, ic) + MU(jc, ic));
        mu_y_p1 = 0.5f * (MU(jc, ic) + MU(jc + 1, ic));
    }
    real sdy_p0 = 1.0f, sdy_p1 = 1.0f;     // slopeopt taper (Fortran
    if (slopeopt >= 1) {                   // :6569-6583)
        real b0 = fabsf(PHB(jc, ic) - PHB(jc - 1, ic)) * MSFY(jc, ic);
        real b1 = fabsf(PHB(jc + 1, ic) - PHB(jc, ic)) * MSFY(jc + 1, ic);
        real dz0 = b0, dz1 = b1;
        if (variant == 1) {                // u: max over columns i and i-1
            dz0 = fmaxf(b0, fabsf(PHB(jc, ic - 1) - PHB(jc - 1, ic - 1))
                            * MSFY(jc, ic - 1));
            dz1 = fmaxf(b1, fabsf(PHB(jc + 1, ic - 1) - PHB(jc, ic - 1))
                            * MSFY(jc + 1, ic - 1));
        } else if (variant == 2) {         // v: max over the two mass faces
            dz0 = fmaxf(b0, fabsf(PHB(jc - 1, ic) - PHB(jc - 2, ic))
                            * MSFY(jc - 1, ic));
            dz1 = fmaxf(b1, b0);
        }
        sdy_p0 = fmaxf(1.0f - dz0 / dzthr_y, 0.0f);
        sdy_p1 = fmaxf(1.0f - dz1 / dzthr_y, 0.0f);
    }
    real tendency_y = coef
        * (sdy_p1 * (c1[k] * mu_y_p1 + c2[k]) * dflux_y_p1
           - sdy_p0 * (c1[k] * mu_y_p0 + c2[k]) * dflux_y_p0);

    tend[I3S(k, j, i, nys, nxs)] += tendency_x + tendency_y;

#undef FX
#undef FY
#undef MU
#undef PHB
#undef MSFX
#undef MSFY
}
