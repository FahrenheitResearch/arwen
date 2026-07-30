// gpuwm/core/kernels/diff6_seam.cu
//
// WRF 6th-order diffusion: the outermost computed boundary-normal
// staggered face on non-periodic domains.  WRF v4.6.1
// sixth_order_diffusion computes u at i = ide-3 (specified/nested and
// open_xe bounds, module_big_step_utilities_em.F:6354-6358/:6348-6350)
// and v at j = jde-3 (:6381-6385/:6378-6380), whose dflux_p1 reads the
// true boundary datum field(ide)/field(jde) (:6465-6467 x /:6547-6549 y)
// -- a read the periodic-wrap stencil in diff6.cu cannot make (it would
// substitute the OPPOSITE boundary's value), so the host masks that face
// out of the main kernel's output and these kernels recompute it with
// honest unwrapped reads: 0-based, u face i = nx-3 and v face j = ny-3.
//
// The arithmetic is the same WRF transcription as diff6.cu (Xue eq. 3
// fluxes, the diff_6th_opt=2 up-gradient zeroing, the diff_6th_slopeopt
// taper, the u/v hybrid face-mass averages); every non-seam face keeps
// the main kernel's untouched binary, so the fix is bit-neutral off the
// seam by construction (pinned in tests/test_diff6_boundary_face.py).
//
// diff6_seam_u writes rows h0..h1 of the fixed face column: WRF's
// j bounds jds+3..jde-4 (0-based 3..ny-4) when y is also non-periodic
// (bndc = 1), the full periodic row range otherwise; cross-axis (y)
// stencil reads wrap exactly like the main kernel iff bndc = 0.
// diff6_seam_v is the transpose.  The host zeroes the whole seam
// column/row first, so the += here lands on zeros and rows/columns
// outside h0..h1 stay zero (the caller's width-3 strip mask re-zeroes
// its own share regardless).

extern "C" __global__
void diff6_seam_u(const real* __restrict__ f,     // (nlev, ny, nx+1)
                  real* __restrict__ tend,        // (nlev, ny, nx+1) +=
                  const real* __restrict__ mut,   // (ny, nx)
                  const real* __restrict__ c1,    // (nlev,)
                  const real* __restrict__ c2,    // (nlev,)
                  const real* __restrict__ phb,   // (>=nlev, ny, nx)
                  const real* __restrict__ msfu,  // (ny, nx+1)
                  const real* __restrict__ msfv,  // (ny+1, nx)
                  real coef, int mono, int slopeopt,
                  real dzthr_x, real dzthr_y,
                  int nlev, int ny, int nx,
                  int h0, int h1, int bndc)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int j = h0 + t;
    if (j > h1 || k >= nlev) return;
    const int i = nx - 3;                        // WRF ide-3
    const int nxs = nx + 1;

// x reads are all direct: i-3 = nx-6 >= 0 and i+3 = nx, the stored true
// east-boundary datum u(ide).  y reads wrap iff the y axis is periodic.
#define JR(m) (bndc ? (j + (m)) : PERIODIC(j + (m), ny))
#define FX(m) f[I3S(k, j, i + (m), ny, nxs)]
#define FY(m) f[I3S(k, JR(m), i, ny, nxs)]
#define MU(jj, ii) mut[(size_t)(jj) * nx + (ii)]
#define CM(jj, ii) (c1[k] * MU(jj, ii) + c2[k])
#define PHB(jj, ii) phb[I3(k, jj, ii, ny, nx)]
#define MSFX(jj, ii) msfu[(size_t)(jj) * (nx + 1) + (ii)]
#define MSFY(jj, ii) msfv[(size_t)(bndc ? (jj) : PERIODIC(jj, ny)) * nx \
                          + (ii)]

    // ---- diffusion in x (Fortran :6461-6533, 'u' branch) ----
    real dflux_x_p0 = 10.0f * (FX(0) - FX(-1)) - 5.0f * (FX(1) - FX(-2))
                      + (FX(2) - FX(-3));
    real dflux_x_p1 = 10.0f * (FX(1) - FX(0)) - 5.0f * (FX(2) - FX(-1))
                      + (FX(3) - FX(-2));         // FX(3) = u(ide)
    if (mono == 2) {
        if (dflux_x_p0 * (FX(0) - FX(-1)) <= 0.0f) dflux_x_p0 = 0.0f;
        if (dflux_x_p1 * (FX(1) - FX(0)) <= 0.0f) dflux_x_p1 = 0.0f;
    }
    real sdx_p0 = 1.0f, sdx_p1 = 1.0f;
    if (slopeopt >= 1) {                          // :6487-6501 'u'
        real a0 = fabsf(PHB(j, i) - PHB(j, i - 1)) * MSFX(j, i);
        real a1 = fabsf(PHB(j, i + 1) - PHB(j, i)) * MSFX(j, i + 1);
        real dz0 = fmaxf(a0, fabsf(PHB(j, i - 1) - PHB(j, i - 2))
                             * MSFX(j, i - 1));
        real dz1 = fmaxf(a1, a0);
        sdx_p0 = fmaxf(1.0f - dz0 / dzthr_x, 0.0f);
        sdx_p1 = fmaxf(1.0f - dz1 / dzthr_x, 0.0f);
    }
    real tendency_x = coef
        * (sdx_p1 * CM(j, i) * dflux_x_p1
           - sdx_p0 * CM(j, i - 1) * dflux_x_p0);

    // ---- diffusion in y (Fortran :6543-6616, 'u' branch) ----
    real dflux_y_p0 = 10.0f * (FY(0) - FY(-1)) - 5.0f * (FY(1) - FY(-2))
                      + (FY(2) - FY(-3));
    real dflux_y_p1 = 10.0f * (FY(1) - FY(0)) - 5.0f * (FY(2) - FY(-1))
                      + (FY(3) - FY(-2));
    if (mono == 2) {
        if (dflux_y_p0 * (FY(0) - FY(-1)) <= 0.0f) dflux_y_p0 = 0.0f;
        if (dflux_y_p1 * (FY(1) - FY(0)) <= 0.0f) dflux_y_p1 = 0.0f;
    }
    real sdy_p0 = 1.0f, sdy_p1 = 1.0f;
    if (slopeopt >= 1) {                          // :6569-6583 'u'
        real b0 = fabsf(PHB(JR(0), i) - PHB(JR(-1), i)) * MSFY(JR(0), i);
        real b1 = fabsf(PHB(JR(1), i) - PHB(JR(0), i)) * MSFY(JR(1), i);
        real dz0 = fmaxf(b0, fabsf(PHB(JR(0), i - 1) - PHB(JR(-1), i - 1))
                             * MSFY(JR(0), i - 1));
        real dz1 = fmaxf(b1, fabsf(PHB(JR(1), i - 1) - PHB(JR(0), i - 1))
                             * MSFY(JR(1), i - 1));
        sdy_p0 = fmaxf(1.0f - dz0 / dzthr_y, 0.0f);
        sdy_p1 = fmaxf(1.0f - dz1 / dzthr_y, 0.0f);
    }
    real mu_y_p0 = 0.25f * (CM(JR(-1), i - 1) + CM(JR(-1), i)
                            + CM(JR(0), i - 1) + CM(JR(0), i));
    real mu_y_p1 = 0.25f * (CM(JR(0), i - 1) + CM(JR(0), i)
                            + CM(JR(1), i - 1) + CM(JR(1), i));
    real tendency_y = coef
        * (sdy_p1 * mu_y_p1 * dflux_y_p1 - sdy_p0 * mu_y_p0 * dflux_y_p0);

    tend[I3S(k, j, i, ny, nxs)] += tendency_x + tendency_y;

#undef JR
#undef FX
#undef FY
#undef MU
#undef CM
#undef PHB
#undef MSFX
#undef MSFY
}

extern "C" __global__
void diff6_seam_v(const real* __restrict__ f,     // (nlev, ny+1, nx)
                  real* __restrict__ tend,        // (nlev, ny+1, nx) +=
                  const real* __restrict__ mut,   // (ny, nx)
                  const real* __restrict__ c1,    // (nlev,)
                  const real* __restrict__ c2,    // (nlev,)
                  const real* __restrict__ phb,   // (>=nlev, ny, nx)
                  const real* __restrict__ msfu,  // (ny, nx+1)
                  const real* __restrict__ msfv,  // (ny+1, nx)
                  real coef, int mono, int slopeopt,
                  real dzthr_x, real dzthr_y,
                  int nlev, int ny, int nx,
                  int h0, int h1, int bndc)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int i = h0 + t;
    if (i > h1 || k >= nlev) return;
    const int j = ny - 3;                        // WRF jde-3
    const int nys = ny + 1;

// y reads are all direct: j-3 = ny-6 >= 0 and j+3 = ny, the stored true
// north-boundary datum v(jde).  x reads wrap iff the x axis is periodic.
#define IC(m) (bndc ? (i + (m)) : PERIODIC(i + (m), nx))
#define FX(m) f[I3S(k, j, IC(m), nys, nx)]
#define FY(m) f[I3S(k, j + (m), i, nys, nx)]
#define MU(jj, ii) mut[(size_t)(jj) * nx + (ii)]
#define CM(jj, ii) (c1[k] * MU(jj, ii) + c2[k])
#define PHB(jj, ii) phb[I3(k, jj, ii, ny, nx)]
#define MSFX(jj, ii) msfu[(size_t)(jj) * (nx + 1) \
                          + (bndc ? (ii) : PERIODIC(ii, nx))]
#define MSFY(jj, ii) msfv[(size_t)(jj) * nx + (ii)]

    // ---- diffusion in x (Fortran :6461-6533, 'v' branch) ----
    real dflux_x_p0 = 10.0f * (FX(0) - FX(-1)) - 5.0f * (FX(1) - FX(-2))
                      + (FX(2) - FX(-3));
    real dflux_x_p1 = 10.0f * (FX(1) - FX(0)) - 5.0f * (FX(2) - FX(-1))
                      + (FX(3) - FX(-2));
    if (mono == 2) {
        if (dflux_x_p0 * (FX(0) - FX(-1)) <= 0.0f) dflux_x_p0 = 0.0f;
        if (dflux_x_p1 * (FX(1) - FX(0)) <= 0.0f) dflux_x_p1 = 0.0f;
    }
    real sdx_p0 = 1.0f, sdx_p1 = 1.0f;
    if (slopeopt >= 1) {                          // :6487-6501 'v'
        real a0 = fabsf(PHB(j, IC(0)) - PHB(j, IC(-1))) * MSFX(j, IC(0));
        real a1 = fabsf(PHB(j, IC(1)) - PHB(j, IC(0))) * MSFX(j, IC(1));
        real dz0 = fmaxf(a0, fabsf(PHB(j - 1, IC(0)) - PHB(j - 1, IC(-1)))
                             * MSFX(j - 1, IC(0)));
        real dz1 = fmaxf(a1, fabsf(PHB(j - 1, IC(1)) - PHB(j - 1, IC(0)))
                             * MSFX(j - 1, IC(1)));
        sdx_p0 = fmaxf(1.0f - dz0 / dzthr_x, 0.0f);
        sdx_p1 = fmaxf(1.0f - dz1 / dzthr_x, 0.0f);
    }
    real mu_x_p0 = 0.25f * (CM(j - 1, IC(-1)) + CM(j - 1, IC(0))
                            + CM(j, IC(-1)) + CM(j, IC(0)));
    real mu_x_p1 = 0.25f * (CM(j - 1, IC(0)) + CM(j - 1, IC(1))
                            + CM(j, IC(0)) + CM(j, IC(1)));
    real tendency_x = coef
        * (sdx_p1 * mu_x_p1 * dflux_x_p1 - sdx_p0 * mu_x_p0 * dflux_x_p0);

    // ---- diffusion in y (Fortran :6543-6616, 'v' branch) ----
    real dflux_y_p0 = 10.0f * (FY(0) - FY(-1)) - 5.0f * (FY(1) - FY(-2))
                      + (FY(2) - FY(-3));
    real dflux_y_p1 = 10.0f * (FY(1) - FY(0)) - 5.0f * (FY(2) - FY(-1))
                      + (FY(3) - FY(-2));         // FY(3) = v(jde)
    if (mono == 2) {
        if (dflux_y_p0 * (FY(0) - FY(-1)) <= 0.0f) dflux_y_p0 = 0.0f;
        if (dflux_y_p1 * (FY(1) - FY(0)) <= 0.0f) dflux_y_p1 = 0.0f;
    }
    real sdy_p0 = 1.0f, sdy_p1 = 1.0f;
    if (slopeopt >= 1) {                          // :6569-6583 'v'
        // j = ny-3, so the Fortran's phb(i,k,j+1) is mass row ny-2: in
        // range, read directly as the source does.
        real b0 = fabsf(PHB(j, i) - PHB(j - 1, i)) * MSFY(j, i);
        real b1 = fabsf(PHB(j + 1, i) - PHB(j, i)) * MSFY(j + 1, i);
        real dz0 = fmaxf(b0, fabsf(PHB(j - 1, i) - PHB(j - 2, i))
                             * MSFY(j - 1, i));
        real dz1 = fmaxf(b1, b0);
        sdy_p0 = fmaxf(1.0f - dz0 / dzthr_y, 0.0f);
        sdy_p1 = fmaxf(1.0f - dz1 / dzthr_y, 0.0f);
    }
    real tendency_y = coef
        * (sdy_p1 * CM(j, i) * dflux_y_p1 - sdy_p0 * CM(j - 1, i)
           * dflux_y_p0);

    tend[I3S(k, j, i, nys, nx)] += tendency_x + tendency_y;

#undef IC
#undef FX
#undef FY
#undef MU
#undef CM
#undef PHB
#undef MSFX
#undef MSFY
}
