// Fused large-step pressure-gradient, buoyancy, and geopotential RHS terms.
//
// These kernels replace eager CuPy expression trees.  Each CuPy binary
// operator formerly wrote an FP32 temporary, so a compiler-contracted FMA
// would change the rounding tree.  The rn_* helpers spell out the same
// round-to-nearest FP32 boundary while retaining the repository-wide NVRTC
// flags (default contraction policy, no fast math).

static __device__ __forceinline__ real rn_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real rn_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real rn_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ real rn_div(real a, real b)
{
    real r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__
real base_value(const real* __restrict__ field, int k, size_t c,
                size_t st, int base3d)
{
    return field[base3d ? (size_t)k * st + c : (size_t)k];
}

static __device__ __forceinline__
real mut_value(const real* __restrict__ mub2d,
               const real* __restrict__ mup, size_t c)
{
    return rn_add(mub2d[c], mup[c]);
}

static __device__ __forceinline__
real pp_value(const real* __restrict__ p, const real* __restrict__ pb,
              int k, size_t c, size_t st, int base3d)
{
    return rn_sub(p[(size_t)k * st + c],
                  base_value(pb, k, c, st, base3d));
}

static __device__ __forceinline__
real dpn_value(const real* __restrict__ p, const real* __restrict__ pb,
               int kf, size_t c, size_t st, int nz, int base3d,
               const real* __restrict__ fnm,
               const real* __restrict__ fnp,
               real cf1, real cf2, real cf3,
               int top_lid, real cfn, real cfn1)
{
    if (kf == nz) {
        if (!top_lid) return 0.0f;
        real top = rn_mul(cfn, pp_value(p, pb, nz - 1, c, st, base3d));
        real below = rn_mul(cfn1, pp_value(p, pb, nz - 2, c, st, base3d));
        return rn_add(top, below);
    }
    if (kf == 0) {
        real t0 = rn_mul(cf1, pp_value(p, pb, 0, c, st, base3d));
        real t1 = rn_mul(cf2, pp_value(p, pb, 1, c, st, base3d));
        real t2 = rn_mul(cf3, pp_value(p, pb, 2, c, st, base3d));
        return rn_add(rn_add(t0, t1), t2);
    }
    real hi = rn_mul(fnm[kf], pp_value(p, pb, kf, c, st, base3d));
    real lo = rn_mul(fnp[kf], pp_value(p, pb, kf - 1, c, st, base3d));
    return rn_add(hi, lo);
}

static __device__ __forceinline__
real php_half_value(const real* __restrict__ php,
                    const real* __restrict__ phb,
                    int k, size_t c, size_t st, int base3d)
{
    real perturbation = rn_add(php[(size_t)k * st + c],
                               php[(size_t)(k + 1) * st + c]);
    if (!base3d)
        return rn_mul(0.5f, perturbation);
    real base = rn_add(phb[(size_t)k * st + c],
                       phb[(size_t)(k + 1) * st + c]);
    return rn_mul(0.5f, rn_add(base, perturbation));
}

static __device__
real pgf_face(size_t c_a, size_t c_b, int k, size_t st, int nz,
              real rd, real half_rd,
              const real* __restrict__ p, const real* __restrict__ pb,
              const real* __restrict__ al, const real* __restrict__ alt,
              const real* __restrict__ php, const real* __restrict__ phb,
              const real* __restrict__ mup, const real* __restrict__ mub2d,
              const real* __restrict__ c1h, const real* __restrict__ c2h,
              const real* __restrict__ rdnw,
              const real* __restrict__ fnm, const real* __restrict__ fnp,
              real cf1, real cf2, real cf3, int base3d,
              int top_lid, real cfn, real cfn1)
{
    size_t a = (size_t)k * st + c_a;
    size_t b = (size_t)k * st + c_b;

    // muf/dmu retain the old (sum -> multiply by 0.5) temporary boundary.
    real muf = rn_mul(0.5f, rn_add(mut_value(mub2d, mup, c_a),
                                   mut_value(mub2d, mup, c_b)));
    real dmu = rn_mul(0.5f, rn_add(mup[c_a], mup[c_b]));
    real layer_mass = rn_add(rn_mul(c1h[k], muf), c2h[k]);

    real dph_hi = rn_sub(php[a + st], php[b + st]);
    real dph_lo = rn_sub(php[a], php[b]);
    real bracket = rn_add(dph_hi, dph_lo);

    real alt_sum = rn_add(alt[a], alt[b]);
    real dpp = rn_sub(pp_value(p, pb, k, c_a, st, base3d),
                      pp_value(p, pb, k, c_b, st, base3d));
    bracket = rn_add(bracket, rn_mul(alt_sum, dpp));

    real al_sum = rn_add(al[a], al[b]);
    real dpb = rn_sub(base_value(pb, k, c_a, st, base3d),
                      base_value(pb, k, c_b, st, base3d));
    bracket = rn_add(bracket, rn_mul(al_sum, dpb));
    real left = rn_mul(rn_mul(half_rd, layer_mass), bracket);

    real dphp = rn_sub(php_half_value(php, phb, k, c_a, st, base3d),
                        php_half_value(php, phb, k, c_b, st, base3d));
    real dpf_hi = rn_mul(
        0.5f,
        rn_add(dpn_value(p, pb, k + 1, c_a, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1),
               dpn_value(p, pb, k + 1, c_b, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1)));
    real dpf_lo = rn_mul(
        0.5f,
        rn_add(dpn_value(p, pb, k, c_a, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1),
               dpn_value(p, pb, k, c_b, st, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3,
                         top_lid, cfn, cfn1)));
    real vertical = rn_mul(rdnw[k], rn_sub(dpf_hi, dpf_lo));
    vertical = rn_sub(vertical, rn_mul(c1h[k], dmu));
    real right = rn_mul(rn_mul(rd, dphp), vertical);
    return rn_add(left, right);
}

extern "C" __global__
void slow_pgf(real* __restrict__ ru_t, real* __restrict__ rv_t,
              const real* __restrict__ p, const real* __restrict__ pb,
              const real* __restrict__ al, const real* __restrict__ alt,
              const real* __restrict__ php, const real* __restrict__ phb,
              const real* __restrict__ mup, const real* __restrict__ mub2d,
              const real* __restrict__ c1h, const real* __restrict__ c2h,
              const real* __restrict__ rdnw,
              const real* __restrict__ fnm, const real* __restrict__ fnp,
              real cf1, real cf2, real cf3,
              int top_lid, real cfn, real cfn1,
              const real* __restrict__ cqu,
              const real* __restrict__ cqv, int use_cq,
              real rdx, real rdy, real half_rdx, real half_rdy,
              int boundary_x, int boundary_y, int base3d,
              int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int rem = tid - k * nyf * nxf;
    int j = rem / nxf;
    int i = rem - j * nxf;
    size_t st = (size_t)ny * nx;

    if (j < ny && ((!boundary_x && i <= nx)
                   || (boundary_x && i > 0 && i < nx))) {
        int ia = i % nx, ib = (i - 1 + nx) % nx;
        size_t ca = (size_t)j * nx + ia, cb = (size_t)j * nx + ib;
        real term = pgf_face(ca, cb, k, st, nz, rdx, half_rdx,
                             p, pb, al, alt, php, phb, mup, mub2d,
                             c1h, c2h, rdnw, fnm, fnp,
                             cf1, cf2, cf3, base3d,
                             top_lid, cfn, cfn1);
        size_t out = I3S(k, j, i, ny, nxf);
        if (use_cq) term = rn_mul(cqu[out], term);
        ru_t[out] = rn_sub(ru_t[out], term);
    }
    if (i < nx && ((!boundary_y && j <= ny)
                   || (boundary_y && j > 0 && j < ny))) {
        int ja = j % ny, jb = (j - 1 + ny) % ny;
        size_t ca = (size_t)ja * nx + i, cb = (size_t)jb * nx + i;
        real term = pgf_face(ca, cb, k, st, nz, rdy, half_rdy,
                             p, pb, al, alt, php, phb, mup, mub2d,
                             c1h, c2h, rdnw, fnm, fnp,
                             cf1, cf2, cf3, base3d,
                             top_lid, cfn, cfn1);
        size_t out = I3S(k, j, i, nyf, nx);
        if (use_cq) term = rn_mul(cqv[out], term);
        rv_t[out] = rn_sub(rv_t[out], term);
    }
}

static __device__ __forceinline__
real q_total(const real* __restrict__ qv, const real* __restrict__ qc,
             const real* __restrict__ qr, const real* __restrict__ qi,
             const real* __restrict__ qs, const real* __restrict__ qg,
             const real* __restrict__ qh,
             size_t ix, int moist_mode)
{
    real q = rn_add(qv[ix], qc[ix]);
    q = rn_add(q, qr[ix]);
    if (moist_mode >= 2) {
        q = rn_add(q, qi[ix]);
        q = rn_add(q, qs[ix]);
        q = rn_add(q, qg[ix]);
    }
    if (moist_mode == 3) q = rn_add(q, qh[ix]);
    return q;
}

extern "C" __global__
void slow_buoyancy(real* __restrict__ rw_t,
                   const real* __restrict__ p,
                   const real* __restrict__ pb,
                   const real* __restrict__ mup,
                   const real* __restrict__ mub2d,
                   const real* __restrict__ qv,
                   const real* __restrict__ qc,
                   const real* __restrict__ qr,
                   const real* __restrict__ qi,
                   const real* __restrict__ qs,
                   const real* __restrict__ qg,
                   const real* __restrict__ qh,
                   const real* __restrict__ rdn,
                   const real* __restrict__ rdnw,
                   const real* __restrict__ c1f,
                   const real* __restrict__ c2f,
                   const real* __restrict__ msft,
                   int moist_mode, int has_msf, int base3d,
                   int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    size_t ix = (size_t)k * st + c;
    size_t im = ix - st;
    if (k == nz) {
        real cqw = 0.0f;
        if (moist_mode) {
            size_t top = (size_t)(nz - 1) * st + c;
            size_t below = top - st;
            cqw = rn_mul(0.5f,
                         rn_add(q_total(qv, qc, qr, qi, qs, qg, qh,
                                        top, moist_mode),
                                q_total(qv, qc, qr, qi, qs, qg, qh,
                                        below, moist_mode)));
        }
        real cq1 = rn_div(1.0f, rn_add(1.0f, cqw));
        real pp_top = pp_value(p, pb, nz - 1, c, st, base3d);
        real pressure = rn_mul(
            rn_mul(rn_mul(cq1, 2.0f), rdnw[nz - 1]),
            rn_sub(0.0f, pp_top));
        real term = rn_sub(pressure, rn_mul(c1f[nz], mup[c]));
        if (moist_mode) {
            real loading = rn_mul(cqw, cq1);
            real base_mass = rn_add(rn_mul(c1f[nz], mub2d[c]), c2f[nz]);
            term = rn_sub(term, rn_mul(loading, base_mass));
        }
        real buoy = rn_mul(G, term);
        if (has_msf) buoy = rn_div(buoy, msft[c]);
        rw_t[ix] = rn_add(rw_t[ix], buoy);
        return;
    }
    real dp = rn_sub(pp_value(p, pb, k, c, st, base3d),
                     pp_value(p, pb, k - 1, c, st, base3d));
    real term;
    if (moist_mode) {
        real cqw = rn_mul(0.5f,
                          rn_add(q_total(qv, qc, qr, qi, qs, qg, qh,
                                         ix, moist_mode),
                                 q_total(qv, qc, qr, qi, qs, qg, qh,
                                         im, moist_mode)));
        real cq1 = rn_div(1.0f, rn_add(1.0f, cqw));
        term = rn_mul(rn_mul(cq1, rdn[k]), dp);
        term = rn_sub(term, rn_mul(c1f[k], mup[c]));
        real loading = rn_mul(cqw, cq1);
        real base_mass = rn_add(rn_mul(c1f[k], mub2d[c]), c2f[k]);
        term = rn_sub(term, rn_mul(loading, base_mass));
    } else {
        term = rn_mul(rdn[k], dp);
        term = rn_sub(term, rn_mul(c1f[k], mup[c]));
    }
    real buoy = rn_mul(G, term);
    if (has_msf) buoy = rn_div(buoy, msft[c]);
    rw_t[ix] = rn_add(rw_t[ix], buoy);
}

static __device__ __forceinline__
real ph_total(const real* __restrict__ php, const real* __restrict__ phb,
              int k, size_t c, size_t st, int base3d)
{
    return rn_add(base_value(phb, k, c, st, base3d),
                  php[(size_t)k * st + c]);
}

static __device__ __forceinline__
real avg_mut(const real* __restrict__ mup,
             const real* __restrict__ mub2d, size_t ca, size_t cb)
{
    return rn_mul(0.5f, rn_add(mut_value(mub2d, mup, ca),
                               mut_value(mub2d, mup, cb)));
}

static __device__
real fcx_value(int face, int j, int k, size_t st, int ny, int nx,
               const real* __restrict__ u,
               const real* __restrict__ mup,
               const real* __restrict__ mub2d,
               const real* __restrict__ c1f,
               const real* __restrict__ c2f,
               const real* __restrict__ msfu, int has_msf,
               int top, real cfn, real cfn1)
{
    int fidx = face % nx;
    int ia = fidx, ib = (fidx - 1 + nx) % nx;
    size_t ca = (size_t)j * nx + ia, cb = (size_t)j * nx + ib;
    real mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    size_t f = I3S(k - 1, j, fidx, ny, nx + 1);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        size_t fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(size_t)j * (nx + 1) + fidx]);
    return value;
}

static __device__
real fcy_value(int row, int i, int k, size_t st, int ny, int nx,
               const real* __restrict__ v,
               const real* __restrict__ mup,
               const real* __restrict__ mub2d,
               const real* __restrict__ c1f,
               const real* __restrict__ c2f,
               const real* __restrict__ msfv, int has_msf,
               int top, real cfn, real cfn1)
{
    int ridx = row % ny;
    int ja = ridx, jb = (ridx - 1 + ny) % ny;
    size_t ca = (size_t)ja * nx + i, cb = (size_t)jb * nx + i;
    real mass = rn_add(rn_mul(c1f[k], avg_mut(mup, mub2d, ca, cb)), c2f[k]);
    size_t f = I3S(k - 1, ridx, i, ny + 1, nx);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        size_t fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(size_t)ridx * nx + i]);
    return value;
}

// Compatibility rhs_ph entry points accept already-averaged face masses.
// Keep these separate from fcx/fcy_value so the production fused kernel's
// reconstructed-mass hot path and generated arithmetic remain unchanged.
static __device__
real supplied_fcx_value(int face, int j, int k, int ny, int nx,
                        const real* __restrict__ u,
                        const real* __restrict__ mux,
                        const real* __restrict__ c1f,
                        const real* __restrict__ c2f,
                        const real* __restrict__ msfu, int has_msf,
                        int top, real cfn, real cfn1)
{
    int fidx = face % nx;
    real mass = rn_add(rn_mul(c1f[k], mux[(size_t)j * (nx + 1) + fidx]),
                       c2f[k]);
    size_t f = I3S(k - 1, j, fidx, ny, nx + 1);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, j, fidx, ny, nx + 1);
        wind = rn_add(rn_mul(cfn, u[f]), rn_mul(cfn1, u[fm]));
    } else {
        size_t fp = I3S(k, j, fidx, ny, nx + 1);
        wind = rn_add(u[fp], u[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfu[(size_t)j * (nx + 1) + fidx]);
    return value;
}

static __device__
real supplied_fcy_value(int row, int i, int k, int ny, int nx,
                        const real* __restrict__ v,
                        const real* __restrict__ muy,
                        const real* __restrict__ c1f,
                        const real* __restrict__ c2f,
                        const real* __restrict__ msfv, int has_msf,
                        int top, real cfn, real cfn1)
{
    int ridx = row % ny;
    real mass = rn_add(rn_mul(c1f[k], muy[(size_t)ridx * nx + i]), c2f[k]);
    size_t f = I3S(k - 1, ridx, i, ny + 1, nx);
    real wind;
    if (top) {
        size_t fm = I3S(k - 2, ridx, i, ny + 1, nx);
        wind = rn_add(rn_mul(cfn, v[f]), rn_mul(cfn1, v[fm]));
    } else {
        size_t fp = I3S(k, ridx, i, ny + 1, nx);
        wind = rn_add(v[fp], v[f]);
    }
    real value = rn_mul(mass, wind);
    if (has_msf)
        value = rn_mul(value, msfv[(size_t)ridx * nx + i]);
    return value;
}

static __device__ __forceinline__
real x_difference(int i, int offset, int j, int k, size_t st, int nx,
                  const real* __restrict__ php,
                  const real* __restrict__ phb, int base3d)
{
    int ip = (i + offset + nx) % nx;
    int im = (i - offset + nx) % nx;
    return rn_sub(ph_total(php, phb, k, (size_t)j * nx + ip, st, base3d),
                  ph_total(php, phb, k, (size_t)j * nx + im, st, base3d));
}

static __device__ __forceinline__
real y_difference(int j, int offset, int i, int k, size_t st, int ny, int nx,
                  const real* __restrict__ php,
                  const real* __restrict__ phb, int base3d)
{
    int jp = (j + offset + ny) % ny;
    int jm = (j - offset + ny) % ny;
    return rn_sub(ph_total(php, phb, k, (size_t)jp * nx + i, st, base3d),
                  ph_total(php, phb, k, (size_t)jm * nx + i, st, base3d));
}

static __device__ __forceinline__
real centered_x(int i, int j, int k, size_t st, int nx,
                const real* __restrict__ php,
                const real* __restrict__ phb, int base3d,
                real w1, real w2, real w3, real denom)
{
    real t1 = rn_mul(w1, x_difference(i, 1, j, k, st, nx, php, phb, base3d));
    real t2 = rn_mul(w2, x_difference(i, 2, j, k, st, nx, php, phb, base3d));
    real t3 = rn_mul(w3, x_difference(i, 3, j, k, st, nx, php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}

static __device__ __forceinline__
real centered_y(int j, int i, int k, size_t st, int ny, int nx,
                const real* __restrict__ php,
                const real* __restrict__ phb, int base3d,
                real w1, real w2, real w3, real denom)
{
    real t1 = rn_mul(w1, y_difference(j, 1, i, k, st, ny, nx,
                                      php, phb, base3d));
    real t2 = rn_mul(w2, y_difference(j, 2, i, k, st, ny, nx,
                                      php, phb, base3d));
    real t3 = rn_mul(w3, y_difference(j, 3, i, k, st, ny, nx,
                                      php, phb, base3d));
    return rn_div(rn_add(rn_add(t1, t2), t3), denom);
}

// Invalid horizontal-advection configurations historically failed only
// after rhs_ph had applied these vertical terms.  This cold-path kernel
// preserves that partial-state contract without branching the fused kernel.
extern "C" __global__
void slow_geopotential_vertical(real* __restrict__ rph_t,
                                const real* __restrict__ ww,
                                const real* __restrict__ w,
                                const real* __restrict__ php,
                                const real* __restrict__ phb,
                                const real* __restrict__ mup,
                                const real* __restrict__ mub2d,
                                const real* __restrict__ rdnw,
                                const real* __restrict__ fnm,
                                const real* __restrict__ fnp,
                                const real* __restrict__ c1f,
                                const real* __restrict__ c2f,
                                const real* __restrict__ msft,
                                int has_msf, int base3d,
                                int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = top ? 0.0f : rph_t[ix];

    if (!top) {
        real ph_km = ph_total(php, phb, k - 1, c, st, base3d);
        real ph_k = ph_total(php, phb, k, c, st, base3d);
        real ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
        real wd_lo = rn_mul(rn_mul(rn_mul(0.5f,
                                         rn_add(ww[ix], ww[ix - st])),
                                  rdnw[k - 1]),
                            rn_sub(ph_k, ph_km));
        real wd_hi = rn_mul(rn_mul(rn_mul(0.5f,
                                         rn_add(ww[ix + st], ww[ix])),
                                  rdnw[k]),
                            rn_sub(ph_kp, ph_k));
        real omega = rn_add(rn_mul(fnm[k], wd_hi),
                            rn_mul(fnp[k], wd_lo));
        tendency = rn_sub(tendency, omega);
    }

    real mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
    real gw = rn_mul(mass, rn_mul(G, w[ix]));
    if (has_msf) gw = rn_div(gw, msft[c]);
    rph_t[ix] = rn_add(tendency, gw);
}

extern "C" __global__
void slow_geopotential(real* __restrict__ rph_t,
                       const real* __restrict__ ww,
                       const real* __restrict__ w,
                       const real* __restrict__ u,
                       const real* __restrict__ v,
                       const real* __restrict__ php,
                       const real* __restrict__ phb,
                       const real* __restrict__ mup,
                       const real* __restrict__ mub2d,
                       const real* __restrict__ rdnw,
                       const real* __restrict__ fnm,
                       const real* __restrict__ fnp,
                       const real* __restrict__ c1f,
                       const real* __restrict__ c2f,
                       real cfn, real cfn1,
                       const real* __restrict__ msft,
                       const real* __restrict__ msfu,
                       const real* __restrict__ msfv,
                       real quarter_rdx, real quarter_rdy,
                       int has_msf, int boundary_x, int boundary_y,
                       int specified, int order, int add_vertical,
                       int base3d, int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = (top && add_vertical) ? 0.0f : rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(2.0f, quarter_rdx);
        quarter_rdy = rn_mul(2.0f, quarter_rdy);
    }

    if (add_vertical) {
        if (!top) {
            real ph_km = ph_total(php, phb, k - 1, c, st, base3d);
            real ph_k = ph_total(php, phb, k, c, st, base3d);
            real ph_kp = ph_total(php, phb, k + 1, c, st, base3d);
            real wd_lo = rn_mul(rn_mul(rn_mul(0.5f,
                                             rn_add(ww[ix], ww[ix - st])),
                                      rdnw[k - 1]),
                                rn_sub(ph_k, ph_km));
            real wd_hi = rn_mul(rn_mul(rn_mul(0.5f,
                                             rn_add(ww[ix + st], ww[ix])),
                                      rdnw[k]),
                                rn_sub(ph_kp, ph_k));
            real omega = rn_add(rn_mul(fnm[k], wd_hi),
                                rn_mul(fnp[k], wd_lo));
            tendency = rn_sub(tendency, omega);
        }

        real mass = rn_add(rn_mul(c1f[k], mut_value(mub2d, mup, c)), c2f[k]);
        real gw = rn_mul(mass, rn_mul(G, w[ix]));
        if (has_msf) gw = rn_div(gw, msft[c]);
        tendency = rn_add(tendency, gw);
    }

    if (order == 2) {
        real dphx;
        if (boundary_x && i == 0) {
            dphx = 0.0f;
        } else {
            int im = (i - 1 + nx) % nx;
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)j * nx + im, st, base3d));
        }
        real fx = rn_mul(fcx_value(i, j, k, st, ny, nx, u, mup, mub2d,
                                   c1f, c2f, msfu, 0,
                                   top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(size_t)j * (nx + 1) + i]);
        real fxp;
        if (!(boundary_x && i == nx - 1)) {
            int ip = (i + 1) % nx;
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(fcx_value(i + 1, j, k, st, ny, nx, u, mup, mub2d,
                                   c1f, c2f, msfu, 0,
                                   top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(size_t)j * (nx + 1)
                                       + ((i + 1) % nx)]);
        } else {
            fxp = 0.0f;
        }
        real hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        tendency = rn_sub(tendency, hx);

        real dphy;
        if (boundary_y && j == 0) {
            dphy = 0.0f;
        } else {
            int jm = (j - 1 + ny) % ny;
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)jm * nx + i, st, base3d));
        }
        real fy = rn_mul(fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                                   c1f, c2f, msfv, 0,
                                   top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(size_t)j * nx + i]);
        real fyp;
        if (!(boundary_y && j == ny - 1)) {
            int jp = (j + 1) % ny;
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d,
                                   c1f, c2f, msfv, 0,
                                   top, cfn, cfn1), dph);
            if (has_msf)
                fyp = rn_mul(fyp, msfv[(size_t)((j + 1) % ny) * nx + i]);
        } else {
            fyp = 0.0f;
        }
        real hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        tendency = rn_sub(tendency, hy);
    } else {
        real hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)j * nx + im, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(fcx_value(i, j, k, st, ny, nx, u, mup,
                                          mub2d, c1f, c2f, msfu, has_msf,
                                          top, cfn, cfn1), d0);
                real f1 = rn_mul(fcx_value(i + 1, j, k, st, ny, nx, u, mup,
                                          mub2d, c1f, c2f, msfu, has_msf,
                                          top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = 0.0f;
            }
        } else {
            real csx = rn_add(fcx_value(i, j, k, st, ny, nx, u, mup, mub2d,
                                        c1f, c2f, msfu, has_msf,
                                        top, cfn, cfn1),
                              fcx_value(i + 1, j, k, st, ny, nx, u, mup,
                                        mub2d, c1f, c2f, msfu, has_msf,
                                        top, cfn, cfn1));
            real diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }

        real hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)jm * nx + i, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(fcy_value(j, i, k, st, ny, nx, v, mup,
                                          mub2d, c1f, c2f, msfv, has_msf,
                                          top, cfn, cfn1), d0);
                real f1 = rn_mul(fcy_value(j + 1, i, k, st, ny, nx, v, mup,
                                          mub2d, c1f, c2f, msfv, has_msf,
                                          top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                real csy = rn_add(
                    fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                              c1f, c2f, msfv, has_msf,
                              top, cfn, cfn1),
                    fcy_value(j + 1, i, k, st, ny, nx, v, mup, mub2d,
                              c1f, c2f, msfv, has_msf,
                              top, cfn, cfn1));
                real diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, 8.0f, -1.0f, 0.0f, 12.0f);
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = 0.0f;
            }
        } else {
            real csy = rn_add(fcy_value(j, i, k, st, ny, nx, v, mup, mub2d,
                                        c1f, c2f, msfv, has_msf,
                                        top, cfn, cfn1),
                              fcy_value(j + 1, i, k, st, ny, nx, v, mup,
                                        mub2d, c1f, c2f, msfv, has_msf,
                                        top, cfn, cfn1));
            real diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;
}

extern "C" __global__
void slow_geopotential_faces(real* __restrict__ rph_t,
                             const real* __restrict__ u,
                             const real* __restrict__ v,
                             const real* __restrict__ php,
                             const real* __restrict__ phb,
                             const real* __restrict__ mux,
                             const real* __restrict__ muy,
                             const real* __restrict__ c1f,
                             const real* __restrict__ c2f,
                             real cfn, real cfn1,
                             const real* __restrict__ msft,
                             const real* __restrict__ msfu,
                             const real* __restrict__ msfv,
                             real quarter_rdx, real quarter_rdy,
                             int has_msf, int boundary_x, int boundary_y,
                             int specified, int order, int base3d,
                             int nz, int ny, int nx)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)ny * nx;
    if (tid >= (size_t)nz * st) return;
    int k = (int)(tid / st) + 1;
    size_t c = tid - (size_t)(k - 1) * st;
    int j = (int)(c / nx), i = (int)(c - (size_t)j * nx);
    size_t ix = (size_t)k * st + c;
    int top = (k == nz);
    real tendency = rph_t[ix];
    if (top) {
        quarter_rdx = rn_mul(2.0f, quarter_rdx);
        quarter_rdy = rn_mul(2.0f, quarter_rdy);
    }

    if (order == 2) {
        real dphx;
        if (boundary_x && i == 0) {
            dphx = 0.0f;
        } else {
            int im = (i - 1 + nx) % nx;
            dphx = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)j * nx + im, st, base3d));
        }
        real fx = rn_mul(supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dphx);
        if (has_msf)
            fx = rn_mul(fx, msfu[(size_t)j * (nx + 1) + i]);
        real fxp;
        if (!(boundary_x && i == nx - 1)) {
            int ip = (i + 1) % nx;
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)j * nx + ip, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fxp = rn_mul(supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                            c1f, c2f, msfu, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fxp = rn_mul(fxp, msfu[(size_t)j * (nx + 1)
                                       + ((i + 1) % nx)]);
        } else {
            fxp = 0.0f;
        }
        real hx = rn_mul(quarter_rdx, rn_add(fxp, fx));
        if (has_msf) hx = rn_div(hx, msft[c]);
        tendency = rn_sub(tendency, hx);

        real dphy;
        if (boundary_y && j == 0) {
            dphy = 0.0f;
        } else {
            int jm = (j - 1 + ny) % ny;
            dphy = rn_sub(ph_total(php, phb, k, c, st, base3d),
                          ph_total(php, phb, k,
                                   (size_t)jm * nx + i, st, base3d));
        }
        real fy = rn_mul(supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dphy);
        if (has_msf)
            fy = rn_mul(fy, msfv[(size_t)j * nx + i]);
        real fyp;
        if (!(boundary_y && j == ny - 1)) {
            int jp = (j + 1) % ny;
            real dph = rn_sub(ph_total(php, phb, k,
                                       (size_t)jp * nx + i, st, base3d),
                              ph_total(php, phb, k, c, st, base3d));
            fyp = rn_mul(supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                            c1f, c2f, msfv, 0,
                                            top, cfn, cfn1), dph);
            if (has_msf)
                fyp = rn_mul(fyp, msfv[(size_t)((j + 1) % ny) * nx + i]);
        } else {
            fyp = 0.0f;
        }
        real hy = rn_mul(quarter_rdy, rn_add(fyp, fy));
        if (has_msf) hy = rn_div(hy, msft[c]);
        tendency = rn_sub(tendency, hy);
    } else {
        real hx;
        if (specified && (i < 3 || i >= nx - 3)) {
            if (i == 1 || i == nx - 2) {
                int im = (i - 1 + nx) % nx, ip = (i + 1) % nx;
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)j * nx + im, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)j * nx + ip, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(supplied_fcx_value(
                                     i, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d0);
                real f1 = rn_mul(supplied_fcx_value(
                                     i + 1, j, k, ny, nx, u, mux,
                                     c1f, c2f, msfu, has_msf,
                                     top, cfn, cfn1), d1);
                hx = rn_mul(quarter_rdx, rn_add(f1, f0));
            } else {
                hx = 0.0f;
            }
        } else {
            real csx = rn_add(
                supplied_fcx_value(i, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1),
                supplied_fcx_value(i + 1, j, k, ny, nx, u, mux,
                                   c1f, c2f, msfu, has_msf,
                                   top, cfn, cfn1));
            real diff = centered_x(i, j, k, st, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hx = rn_mul(rn_mul(quarter_rdx, csx), diff);
        }

        real hy;
        if (specified && (j < 3 || j >= ny - 3)) {
            if (j == 1 || j == ny - 2) {
                int jm = (j - 1 + ny) % ny, jp = (j + 1) % ny;
                real d0 = rn_sub(ph_total(php, phb, k, c, st, base3d),
                                 ph_total(php, phb, k,
                                          (size_t)jm * nx + i, st, base3d));
                real d1 = rn_sub(ph_total(php, phb, k,
                                          (size_t)jp * nx + i, st, base3d),
                                 ph_total(php, phb, k, c, st, base3d));
                real f0 = rn_mul(supplied_fcy_value(
                                     j, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d0);
                real f1 = rn_mul(supplied_fcy_value(
                                     j + 1, i, k, ny, nx, v, muy,
                                     c1f, c2f, msfv, has_msf,
                                     top, cfn, cfn1), d1);
                hy = rn_mul(quarter_rdy, rn_add(f1, f0));
            } else if (j == 2 || j == ny - 3) {
                real csy = rn_add(
                    supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1),
                    supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                       c1f, c2f, msfv, has_msf,
                                       top, cfn, cfn1));
                real diff = centered_y(j, i, k, st, ny, nx, php, phb,
                                       base3d, 8.0f, -1.0f, 0.0f, 12.0f);
                hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
            } else {
                hy = 0.0f;
            }
        } else {
            real csy = rn_add(
                supplied_fcy_value(j, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1),
                supplied_fcy_value(j + 1, i, k, ny, nx, v, muy,
                                   c1f, c2f, msfv, has_msf,
                                   top, cfn, cfn1));
            real diff = centered_y(j, i, k, st, ny, nx, php, phb, base3d,
                                   45.0f, -9.0f, 1.0f, 60.0f);
            hy = rn_mul(rn_mul(quarter_rdy, csy), diff);
        }
        if (has_msf) {
            hx = rn_div(hx, msft[c]);
            hy = rn_div(hy, msft[c]);
        }
        tendency = rn_sub(tendency, rn_add(hx, hy));
    }
    rph_t[ix] = tendency;
}

// ---- RK small-step preparation/finalization -------------------------------
//
// The former CuPy implementation materialized every binary operator below as
// a separate FP32 array.  rn_* preserves each of those store/load rounding
// boundaries explicitly, so batching the pointwise chains changes neither
// expression order nor contraction while removing the temporary traffic.
//
// THE FACE-MASS NEIGHBOUR IS BOUNDARY-DEPENDENT.  Both uv kernels below build
// WRF's muu/muus and muv/muvs inline -- the column mass carried to a u/v face
// as the mean of the two mass cells that share it (module_small_step_em.F
// small_step_prep :225-228 and small_step_finish :96-98 consume exactly those
// arrays).  WRF builds them in calc_mu_uv
// (module_big_step_utilities_em.F:59-115 x / :118-174 y) and that routine
// chooses the off-domain neighbour from ``config_flags%periodic_x`` /
// ``periodic_y``:
//
//   west face  i = ids:  im = its   normally, its-1 under periodic_x
//   east face  i = ide:  im = ite-1 normally, ite   under periodic_x
//
// The periodic branch reads the halo cell, which the period fill has loaded
// with the opposite edge -- the ``i % nx`` / ``(i-1+nx) % nx`` below.  The
// NON-periodic branch names the boundary cell twice, so the boundary face
// carries that cell's own mass: a zero-gradient ghost, which is min/max
// clamping of the same two indices.  ``advance_mu_th`` (kernels/acoustic.cu)
// already spells the identical choice for its four face masses.
//
// Until this flag arrived both kernels wrapped unconditionally, so on a
// specified, nested or open-lateral domain the west boundary face took its
// mass from the EAST-most mass column and vice versa -- a real coupling of
// the two physical boundaries, at both ends of every acoustic rung of every
// RK stage.  These were the last two sites where a wrapped index SURVIVED on
// a non-periodic domain: ``advance_uv`` (kernels/acoustic.cu) also spells
// ``i % nx``, but under specified/nested its spec_zone guard never reaches
// the boundary face, and under open boundaries acoustic.py overwrites both
// face sets from their saved pre-substep values immediately after the
// launch.  MEASURED after the fix, 96x64x24 dry, one step, 25 Pa on one
// west mass column, over all 76 device arrays the state owns: the influence
// stops at column 10 on an open domain and at column 0 on a specified one,
// while the periodic control still reaches the far edge in 33 arrays.
// Before the fix the same probe moved the far edge in 34 open-domain arrays
// and in the two stage mass fluxes of a specified one.
//
// The two boundary-normal face columns are the whole footprint of the
// change: interior faces 1 <= i <= nx-1 index i-1 and i on both branches, so
// a periodic run is bit-identical (proved by tilestream/test_gate.py, which
// is periodic throughout, and by tests/test_small_step_lateral_wrap.py's
// periodic rows).  A specified domain moves by exactly one float32 ULP at
// one step, in 22 of 148992 ``ru`` faces and 26 of 149760 ``rv`` faces --
// all of them at i = 0 / i = nx and j = 0 / j = ny, free interior exactly
// zero; the persisted state first differs at step 3 and by step 8 is
// 1.1e-3 m/s in u and 1.9e-2 Pa in mu.

extern "C" __global__
void small_step_init_uv(real* __restrict__ u_pp,
                        real* __restrict__ v_pp,
                        const real* __restrict__ u0,
                        const real* __restrict__ v0,
                        const real* __restrict__ u,
                        const real* __restrict__ v,
                        const real* __restrict__ mup0,
                        const real* __restrict__ mup,
                        const real* __restrict__ mub2d,
                        const real* __restrict__ c1h,
                        const real* __restrict__ c2h,
                        const real* __restrict__ msfu,
                        const real* __restrict__ msfv,
                        int has_msf, int boundary_x, int boundary_y,
                        int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;

    if (j < ny) {
        int ia = boundary_x ? min(i, nx - 1) : i % nx;
        int ib = boundary_x ? max(i - 1, 0)  : (i - 1 + nx) % nx;
        size_t ca = (size_t)j * nx + ia;
        size_t cb = (size_t)j * nx + ib;
        real mt_a = rn_add(mub2d[ca], mup0[ca]);
        real mt_b = rn_add(mub2d[cb], mup0[cb]);
        real ms_a = rn_add(mub2d[ca], mup[ca]);
        real ms_b = rn_add(mub2d[cb], mup[cb]);
        real mtf = rn_mul(0.5f, rn_add(mt_a, mt_b));
        real msf = rn_mul(0.5f, rn_add(ms_a, ms_b));
        real ct = rn_add(rn_mul(c1h[k], mtf), c2h[k]);
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        size_t ix = I3S(k, j, i, ny, nxf);
        real value = rn_sub(rn_mul(ct, u0[ix]), rn_mul(cs, u[ix]));
        if (has_msf) value = rn_div(value, msfu[(size_t)j * nxf + i]);
        u_pp[ix] = value;
    }
    if (i < nx) {
        int ja = boundary_y ? min(j, ny - 1) : j % ny;
        int jb = boundary_y ? max(j - 1, 0)  : (j - 1 + ny) % ny;
        size_t ca = (size_t)ja * nx + i;
        size_t cb = (size_t)jb * nx + i;
        real mt_a = rn_add(mub2d[ca], mup0[ca]);
        real mt_b = rn_add(mub2d[cb], mup0[cb]);
        real ms_a = rn_add(mub2d[ca], mup[ca]);
        real ms_b = rn_add(mub2d[cb], mup[cb]);
        real mtf = rn_mul(0.5f, rn_add(mt_a, mt_b));
        real msf = rn_mul(0.5f, rn_add(ms_a, ms_b));
        real ct = rn_add(rn_mul(c1h[k], mtf), c2h[k]);
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        size_t ix = I3S(k, j, i, nyf, nx);
        real value = rn_sub(rn_mul(ct, v0[ix]), rn_mul(cs, v[ix]));
        if (has_msf) value = rn_div(value, msfv[(size_t)j * nx + i]);
        v_pp[ix] = value;
    }
}

extern "C" __global__
void small_step_init_column(real* __restrict__ w_pp,
                            real* __restrict__ th_pp,
                            real* __restrict__ ph_pp,
                            real* __restrict__ mu_pp,
                            real* __restrict__ al_pp,
                            real* __restrict__ p_pp,
                            real* __restrict__ p_pp_old,
                            const real* __restrict__ w0,
                            const real* __restrict__ w,
                            const real* __restrict__ thp0,
                            const real* __restrict__ thp,
                            const real* __restrict__ php0,
                            const real* __restrict__ php,
                            const real* __restrict__ p,
                            const real* __restrict__ alt,
                            const real* __restrict__ mup0,
                            const real* __restrict__ mup,
                            const real* __restrict__ mub2d,
                            const real* __restrict__ thb,
                            const real* __restrict__ c1h,
                            const real* __restrict__ c2h,
                            const real* __restrict__ c1f,
                            const real* __restrict__ c2f,
                            const real* __restrict__ rdnw,
                            const real* __restrict__ msft,
                            int has_msf, int base3d,
                            int nz, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    real mut = rn_add(mub2d[c], mup0[c]);
    real mus = rn_add(mub2d[c], mup[c]);
    real mupp = rn_sub(mup0[c], mup[c]);
    mu_pp[c] = mupp;

    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        real cft = rn_add(rn_mul(c1f[k], mut), c2f[k]);
        real cfs = rn_add(rn_mul(c1f[k], mus), c2f[k]);
        real value = rn_sub(rn_mul(cft, w0[f]), rn_mul(cfs, w[f]));
        if (has_msf) value = rn_div(value, msft[c]);
        w_pp[f] = value;
        ph_pp[f] = rn_sub(php0[f], php[f]);
    }

    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        size_t b = (size_t)k * bstr + boff;
        real cht = rn_add(rn_mul(c1h[k], mut), c2h[k]);
        real chs = rn_add(rn_mul(c1h[k], mus), c2h[k]);
        real tht = rn_add(thb[b], thp0[h]);
        real ths = rn_add(thb[b], thp[h]);
        real thpp = rn_sub(rn_mul(cht, tht), rn_mul(chs, ths));
        th_pp[h] = thpp;

        real c2a = rn_div(rn_mul(GAMMA, p[h]), alt[h]);
        real al0 = rn_mul(c1h[k], mupp);
        real al1 = rn_mul(alt[h], al0);
        real dph = rn_sub(ph_pp[h + st], ph_pp[h]);
        real al2 = rn_mul(rdnw[k], dph);
        real al = rn_div(-rn_add(al1, al2), cht);
        al_pp[h] = al;

        real p0 = rn_mul(c1h[k], mupp);
        real p1 = rn_mul(p0, ths);
        real p2 = rn_sub(thpp, p1);
        real p3 = rn_mul(alt[h], p2);
        real p4 = rn_mul(cht, ths);
        real p5 = rn_div(p3, p4);
        real p6 = rn_sub(p5, al);
        real pnew = rn_mul(c2a, p6);
        p_pp[h] = pnew;
        p_pp_old[h] = pnew;
    }
}

extern "C" __global__
void small_step_finish_uv(real* __restrict__ u,
                          real* __restrict__ v,
                          const real* __restrict__ u_pp,
                          const real* __restrict__ v_pp,
                          const real* __restrict__ mup,
                          const real* __restrict__ mu_pp,
                          const real* __restrict__ mub2d,
                          const real* __restrict__ c1h,
                          const real* __restrict__ c2h,
                          const real* __restrict__ msfu,
                          const real* __restrict__ msfv,
                          int has_msf, int boundary_x, int boundary_y,
                          int nz, int ny, int nx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;

    if (j < ny) {
        int ia = boundary_x ? min(i, nx - 1) : i % nx;
        int ib = boundary_x ? max(i - 1, 0)  : (i - 1 + nx) % nx;
        size_t ca = (size_t)j * nx + ia;
        size_t cb = (size_t)j * nx + ib;
        real msa = rn_add(mub2d[ca], mup[ca]);
        real msb = rn_add(mub2d[cb], mup[cb]);
        real mna = rn_add(msa, mu_pp[ca]);
        real mnb = rn_add(msb, mu_pp[cb]);
        real msf = rn_mul(0.5f, rn_add(msa, msb));
        real mnf = rn_mul(0.5f, rn_add(mna, mnb));
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        real cn = rn_add(rn_mul(c1h[k], mnf), c2h[k]);
        size_t ix = I3S(k, j, i, ny, nxf);
        real pp = u_pp[ix];
        if (has_msf) pp = rn_mul(pp, msfu[(size_t)j * nxf + i]);
        u[ix] = rn_div(rn_add(rn_mul(cs, u[ix]), pp), cn);
    }
    if (i < nx) {
        int ja = boundary_y ? min(j, ny - 1) : j % ny;
        int jb = boundary_y ? max(j - 1, 0)  : (j - 1 + ny) % ny;
        size_t ca = (size_t)ja * nx + i;
        size_t cb = (size_t)jb * nx + i;
        real msa = rn_add(mub2d[ca], mup[ca]);
        real msb = rn_add(mub2d[cb], mup[cb]);
        real mna = rn_add(msa, mu_pp[ca]);
        real mnb = rn_add(msb, mu_pp[cb]);
        real msf = rn_mul(0.5f, rn_add(msa, msb));
        real mnf = rn_mul(0.5f, rn_add(mna, mnb));
        real cs = rn_add(rn_mul(c1h[k], msf), c2h[k]);
        real cn = rn_add(rn_mul(c1h[k], mnf), c2h[k]);
        size_t ix = I3S(k, j, i, nyf, nx);
        real pp = v_pp[ix];
        if (has_msf) pp = rn_mul(pp, msfv[(size_t)j * nx + i]);
        v[ix] = rn_div(rn_add(rn_mul(cs, v[ix]), pp), cn);
    }
}

extern "C" __global__
void small_step_finish_column(real* __restrict__ w,
                              real* __restrict__ thp,
                              real* __restrict__ php,
                              real* __restrict__ mup,
                              const real* __restrict__ w_pp,
                              const real* __restrict__ th_pp,
                              const real* __restrict__ ph_pp,
                              const real* __restrict__ mu_pp,
                              const real* __restrict__ mub2d,
                              const real* __restrict__ thb,
                              const real* __restrict__ c1h,
                              const real* __restrict__ c2h,
                              const real* __restrict__ c1f,
                              const real* __restrict__ c2f,
                              const real* __restrict__ msft,
                              const real* __restrict__ h_diabatic,
                              real hdiab_dt, int remove_hdiab,
                              int has_msf, int base3d,
                              int nz, int ny, int nx)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ny * nx) return;
    size_t st = (size_t)ny * nx;
    size_t bstr = base3d ? st : 1;
    size_t boff = base3d ? (size_t)c : 0;
    real mus = rn_add(mub2d[c], mup[c]);
    real mun = rn_add(mus, mu_pp[c]);

    for (int k = 0; k <= nz; ++k) {
        size_t f = (size_t)k * st + c;
        real pp = w_pp[f];
        if (has_msf) pp = rn_mul(pp, msft[c]);
        real cs = rn_add(rn_mul(c1f[k], mus), c2f[k]);
        real cn = rn_add(rn_mul(c1f[k], mun), c2f[k]);
        w[f] = rn_div(rn_add(rn_mul(cs, w[f]), pp), cn);
        php[f] = rn_add(php[f], ph_pp[f]);
    }

    for (int k = 0; k < nz; ++k) {
        size_t h = (size_t)k * st + c;
        size_t b = (size_t)k * bstr + boff;
        real cs = rn_add(rn_mul(c1h[k], mus), c2h[k]);
        real theta = rn_add(thb[b], thp[h]);
        real numerator = rn_add(rn_mul(cs, theta), th_pp[h]);
        if (remove_hdiab) {
            real removal = rn_mul(rn_mul(hdiab_dt, cs), h_diabatic[h]);
            numerator = rn_sub(numerator, removal);
        }
        real cn = rn_add(rn_mul(c1h[k], mun), c2h[k]);
        thp[h] = rn_sub(rn_div(numerator, cn), thb[b]);
    }
    mup[c] = rn_add(mup[c], mu_pp[c]);
}
