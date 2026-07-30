// gpuwm/core/kernels/diagnostics.cu
//
// Equation-of-state diagnostics (WRF v4.6.1 calc_p_rho_phi analogue,
// dyn_em/module_big_step_utilities_em.F:1025-1052 NONHYDROSTATIC branch,
// keyed on hypsometric_opt; general hybrid/terrain form).
// One thread per (i, j) column; loops over k recomputing, from the
// prognostic perturbation fields:
//   alt  total dry specific volume.
//        hypso = 1 (frozen gpuwm default): from the geopotential and the
//        hybrid column-mass increment:
//          alt[k] = -((ph[k+1] - ph[k]) * rdnw[k]) / (c1h[k]*mu + c2h[k])
//        (rdnw < 0 => alt > 0; c1h = 1, c2h = 0 reduces bitwise to the
//        Phase 1 form -dph*rdnw/mu).
//        hypso = 2 (WRF Registry default, F:1042-1051 verbatim): the
//        log-pressure hypsometric form on the reference dry pressures
//          pfu = c3f[k+1]*mu + c4f[k+1] + p_top
//          pfd = c3f[k  ]*mu + c4f[k  ] + p_top
//          phm = c3h[k  ]*mu + c4h[k  ] + p_top
//          al[k]  = dph/phm/log(pfd/pfu) - alb[k]
//          alt[k] = al[k] + alb[k]
//        with mu = mub + mu' the total dry column mass (WRF MUTS).
//   p    full pressure via the ideal-gas EOS on the moist potential
//        temperature theta_m = theta*(1 + Rv/Rd*qv) (ARW ch. 2; theta_m
//        reduces to theta bitwise when moist = 0 or qv = 0):
//          p = P0 * ((RD * th_m) / (P0 * alt))^GAMMA
//        (for hypso = 2 alt = al + alb, exactly WRF's moist_nonhydro
//        temp = r_d*t*qvf/(p0*(al+alb)), F:1061-1066, use_theta_m = 0).
//   al   perturbation specific volume alt - alb.
//
// Base-state profiles are 1-D columns (base3d = 0, flat terrain) or full
// per-column 3-D fields (base3d = 1, terrain); mub is always the (ny, nx)
// dry-mass field (a broadcast scalar when flat).  qv is read only when
// moist = 1 (dry launches pass a dummy pointer).  c3h/c4h/c3f/c4f and
// p_top are read only when hypso = 2 (opt-1 launches may pass zeros).

extern "C" __global__
void calc_p_alpha(const real* __restrict__ thp,   // (nz,   ny, nx) theta'
                  const real* __restrict__ php,   // (nz+1, ny, nx) phi'
                  const real* __restrict__ mup,   // (ny, nx)       mu'
                  const real* __restrict__ thb,   // (nz[,ny,nx])   base theta
                  const real* __restrict__ phb,   // (nz+1[,ny,nx]) base geopot.
                  const real* __restrict__ alb,   // (nz[,ny,nx])   base alpha
                  const real* __restrict__ rdnw,  // (nz,)  1/dnw (< 0)
                  const real* __restrict__ c1h,   // (nz,)  dB/deta
                  const real* __restrict__ c2h,   // (nz,)  (1-c1h)(p0-pt)
                  const real* __restrict__ c3h,   // (nz,)  B(eta) half levels
                  const real* __restrict__ c4h,   // (nz,)  (eta-B)(p0-pt)
                  const real* __restrict__ c3f,   // (nz+1,) full-level c3
                  const real* __restrict__ c4f,   // (nz+1,) full-level c4
                  const real* __restrict__ mub,   // (ny, nx) base column mass
                  const real* __restrict__ qv,    // (nz, ny, nx) vapor (moist)
                  real p_top, int hypso,
                  int moist, int base3d, int nz, int ny, int nx,
                  real* __restrict__ p,           // (nz, ny, nx) full pressure
                  real* __restrict__ al,          // (nz, ny, nx) alpha'
                  real* __restrict__ alt)         // (nz, ny, nx) total alpha
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    // Base-profile indexing: level stride and column offset collapse to
    // (1, 0) for flat 1-D columns.
    size_t kstr = base3d ? (size_t)ny * nx : 1;
    size_t coff = base3d ? (size_t)j * nx + i : 0;

    real mu = mub[(size_t)j * nx + i] + mup[(size_t)j * nx + i];
    for (int k = 0; k < nz; ++k) {
        real th  = thb[k * kstr + coff] + thp[IDX3(k, j, i)];
        if (moist) th *= 1.0f + RVOVRD * qv[IDX3(k, j, i)];
        real dph = (phb[(k + 1) * kstr + coff] + php[IDX3(k + 1, j, i)])
                 - (phb[k * kstr + coff]       + php[IDX3(k,     j, i)]);
        real a, ap;
        if (hypso == 2) {
            real pfu = c3f[k + 1] * mu + c4f[k + 1] + p_top;
            real pfd = c3f[k]     * mu + c4f[k]     + p_top;
            real phm = c3h[k]     * mu + c4h[k]     + p_top;
            ap = dph / phm / logf(pfd / pfu) - alb[k * kstr + coff];
            a  = ap + alb[k * kstr + coff];
        } else {
            a  = -dph * rdnw[k] / (c1h[k] * mu + c2h[k]);
            ap = a - alb[k * kstr + coff];
        }
        alt[IDX3(k, j, i)] = a;
        al[IDX3(k, j, i)]  = ap;
        p[IDX3(k, j, i)]   = P0 * powf((RD * th) / (P0 * a), GAMMA);
    }
}
