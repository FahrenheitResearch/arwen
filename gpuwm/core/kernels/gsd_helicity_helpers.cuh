// GSD helicity device helpers copied verbatim from uh_diag.cu at fe99a1680d.
// Keep the stock translation unit and its frozen compile source unchanged.
// Only device helpers belong in this header; global entry points stay in .cu.
extern "C" {

// ---- pointwise compute_diff_metrics pieces ---------------------------------

// z at a w level: z_at_w = (ph + phb) / g   (module_diffusion_em.F:6951)
__device__ __forceinline__ real uh_z_at_w(
    const real* ph, const real* phb, int phb3d,
    int kw, int jy, int ix, int ny, int nx) {
  const real phv = ph[I3(kw, jy, ix, ny, nx)];
  const real phbv = phb3d ? phb[I3(kw, jy, ix, ny, nx)] : phb[kw];
  return __fdiv_rn(__fadd_rn(phv, phbv), G);
}

// rdzw at a mass level: 1.0 / (z_at_w(k+1) - z_at_w(k))   (:6957)
__device__ __forceinline__ real uh_rdzw(
    const real* ph, const real* phb, int phb3d,
    int km, int jy, int ix, int ny, int nx) {
  const real zu = uh_z_at_w(ph, phb, phb3d, km + 1, jy, ix, ny, nx);
  const real zl = uh_z_at_w(ph, phb, phb3d, km, jy, ix, ny, nx);
  return __fdiv_rn(1.0f, __fsub_rn(zu, zl));
}

// zx at a u point and w level (:6988 + :6996, two rounded passes summed;
// zero on the domain west/east faces for the non-periodic case, :7024-7035).
__device__ __forceinline__ real uh_zx(
    const real* ph, const real* phb, int phb3d, real rdx,
    int kw, int jy, int ixs, int ny, int nx) {
  if (ixs == 0 || ixs == nx) return 0.0f;
  const real phb_i = phb3d ? phb[I3(kw, jy, ixs, ny, nx)] : phb[kw];
  const real phb_im1 = phb3d ? phb[I3(kw, jy, ixs - 1, ny, nx)] : phb[kw];
  const real pass1 = __fdiv_rn(
      __fmul_rn(rdx, __fsub_rn(phb_i, phb_im1)), G);
  const real pass2 = __fdiv_rn(
      __fmul_rn(rdx, __fsub_rn(ph[I3(kw, jy, ixs, ny, nx)],
                               ph[I3(kw, jy, ixs - 1, ny, nx)])), G);
  return __fadd_rn(pass1, pass2);
}

// zy at a v point and w level (:7004 + :7012; zero on south/north faces).
__device__ __forceinline__ real uh_zy(
    const real* ph, const real* phb, int phb3d, real rdy,
    int kw, int jys, int ix, int ny, int nx) {
  if (jys == 0 || jys == ny) return 0.0f;
  const real phb_j = phb3d ? phb[I3(kw, jys, ix, ny, nx)] : phb[kw];
  const real phb_jm1 = phb3d ? phb[I3(kw, jys - 1, ix, ny, nx)] : phb[kw];
  const real pass1 = __fdiv_rn(
      __fmul_rn(rdy, __fsub_rn(phb_j, phb_jm1)), G);
  const real pass2 = __fdiv_rn(
      __fmul_rn(rdy, __fsub_rn(ph[I3(kw, jys, ix, ny, nx)],
                               ph[I3(kw, jys - 1, ix, ny, nx)])), G);
  return __fadd_rn(pass1, pass2);
}

// ---- cal_helicity pieces ----------------------------------------------------

// v-hat = v / msfvy at a v point and mass level (:7260).
__device__ __forceinline__ real uh_hat_v(
    const real* v, const real* msfv,
    int km, int jys, int ix, int ny, int nx) {
  return __fdiv_rn(v[I3S(km, jys, ix, ny + 1, nx)], msfv[jys * nx + ix]);
}

// u-hat = u / msfux at a u point and mass level (:7344).
__device__ __forceinline__ real uh_hat_u(
    const real* u, const real* msfu,
    int km, int jy, int ixs, int ny, int nx) {
  return __fdiv_rn(u[I3S(km, jy, ixs, ny, nx + 1)], msfu[jy * (nx + 1) + ixs]);
}

// v-hat averaged to (corner, w level) kw: interior fnm/fnp (:7270-7272),
// bottom cf1..cf3 (:7281-7287), top cft1/cft2 (:7288-7290).
__device__ __forceinline__ real uh_hatavg_v(
    const real* v, const real* msfv,
    const real* fnm, const real* fnp,
    real cf1, real cf2, real cf3, real cft1, real cft2,
    int kw, int cy, int cx, int nz, int ny, int nx) {
  if (kw == 0) {
    // 0.5*(cf1*hat(i-1,1)+cf2*hat(i-1,2)+cf3*hat(i-1,3)
    //      +cf1*hat(i,1)+cf2*hat(i,2)+cf3*hat(i,3))
    real acc = __fmul_rn(cf1, uh_hat_v(v, msfv, 0, cy, cx - 1, ny, nx));
    acc = __fadd_rn(acc, __fmul_rn(
        cf2, uh_hat_v(v, msfv, 1, cy, cx - 1, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf3, uh_hat_v(v, msfv, 2, cy, cx - 1, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf1, uh_hat_v(v, msfv, 0, cy, cx, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf2, uh_hat_v(v, msfv, 1, cy, cx, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf3, uh_hat_v(v, msfv, 2, cy, cx, ny, nx)));
    return __fmul_rn(0.5f, acc);
  }
  if (kw == nz) {
    // 0.5*(cft1*(hat(i,ktes1)+hat(i-1,ktes1)) + cft2*(hat(i,ktes2)+hat(i-1,ktes2)))
    const real t1 = __fmul_rn(cft1, __fadd_rn(
        uh_hat_v(v, msfv, nz - 1, cy, cx, ny, nx),
        uh_hat_v(v, msfv, nz - 1, cy, cx - 1, ny, nx)));
    const real t2 = __fmul_rn(cft2, __fadd_rn(
        uh_hat_v(v, msfv, nz - 2, cy, cx, ny, nx),
        uh_hat_v(v, msfv, nz - 2, cy, cx - 1, ny, nx)));
    return __fmul_rn(0.5f, __fadd_rn(t1, t2));
  }
  // 0.5*(fnm(k)*(hat(i-1,k)+hat(i,k)) + fnp(k)*(hat(i-1,k-1)+hat(i,k-1)))
  const real t1 = __fmul_rn(fnm[kw], __fadd_rn(
      uh_hat_v(v, msfv, kw, cy, cx - 1, ny, nx),
      uh_hat_v(v, msfv, kw, cy, cx, ny, nx)));
  const real t2 = __fmul_rn(fnp[kw], __fadd_rn(
      uh_hat_v(v, msfv, kw - 1, cy, cx - 1, ny, nx),
      uh_hat_v(v, msfv, kw - 1, cy, cx, ny, nx)));
  return __fmul_rn(0.5f, __fadd_rn(t1, t2));
}

// u-hat averaged to (corner, w level) kw: interior (:7354-7356), bottom
// (:7365-7371), top (:7372-7374).  The pair order is (j-1, j).
__device__ __forceinline__ real uh_hatavg_u(
    const real* u, const real* msfu,
    const real* fnm, const real* fnp,
    real cf1, real cf2, real cf3, real cft1, real cft2,
    int kw, int cy, int cx, int nz, int ny, int nx) {
  if (kw == 0) {
    real acc = __fmul_rn(cf1, uh_hat_u(u, msfu, 0, cy - 1, cx, ny, nx));
    acc = __fadd_rn(acc, __fmul_rn(
        cf2, uh_hat_u(u, msfu, 1, cy - 1, cx, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf3, uh_hat_u(u, msfu, 2, cy - 1, cx, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf1, uh_hat_u(u, msfu, 0, cy, cx, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf2, uh_hat_u(u, msfu, 1, cy, cx, ny, nx)));
    acc = __fadd_rn(acc, __fmul_rn(
        cf3, uh_hat_u(u, msfu, 2, cy, cx, ny, nx)));
    return __fmul_rn(0.5f, acc);
  }
  if (kw == nz) {
    const real t1 = __fmul_rn(cft1, __fadd_rn(
        uh_hat_u(u, msfu, nz - 1, cy - 1, cx, ny, nx),
        uh_hat_u(u, msfu, nz - 1, cy, cx, ny, nx)));
    const real t2 = __fmul_rn(cft2, __fadd_rn(
        uh_hat_u(u, msfu, nz - 2, cy - 1, cx, ny, nx),
        uh_hat_u(u, msfu, nz - 2, cy, cx, ny, nx)));
    return __fmul_rn(0.5f, __fadd_rn(t1, t2));
  }
  const real t1 = __fmul_rn(fnm[kw], __fadd_rn(
      uh_hat_u(u, msfu, kw, cy - 1, cx, ny, nx),
      uh_hat_u(u, msfu, kw, cy, cx, ny, nx)));
  const real t2 = __fmul_rn(fnp[kw], __fadd_rn(
      uh_hat_u(u, msfu, kw - 1, cy - 1, cx, ny, nx),
      uh_hat_u(u, msfu, kw - 1, cy, cx, ny, nx)));
  return __fmul_rn(0.5f, __fadd_rn(t1, t2));
}

// Vertical relative vorticity on the corner column at a mass level:
// rvort = mm*(rdx*(hat_v(i)-hat_v(i-1)) - tmp1_v)
//       - mm*(rdy*(hat_u(j)-hat_u(j-1)) - tmp1_u)      (:7325-7326, :7403-7405)
// with tmp1 = (hatavg(k+1)-hatavg(k)) * 0.25*tmpz*(4-point rdzw sum)
// (:7302-7313 for x -- rdzw order NE,SE,SW,NW; :7380-7391 for y -- rdzw
// order NE,NW,SW,SE; the two orders differ in the authority and are kept).
__device__ __forceinline__ real uh_rvort(
    const real* u, const real* v, const real* ph, const real* phb, int phb3d,
    const real* msfu, const real* msfv,
    const real* fnm, const real* fnp,
    real cf1, real cf2, real cf3, real cft1, real cft2,
    real rdx, real rdy, real mm,
    int km, int cy, int cx, int nz, int ny, int nx) {
  // dv/dx part.
  real tmpzx = __fmul_rn(0.25f, __fadd_rn(__fadd_rn(__fadd_rn(
      uh_zx(ph, phb, phb3d, rdx, km, cy - 1, cx, ny, nx),
      uh_zx(ph, phb, phb3d, rdx, km, cy, cx, ny, nx)),
      uh_zx(ph, phb, phb3d, rdx, km + 1, cy - 1, cx, ny, nx)),
      uh_zx(ph, phb, phb3d, rdx, km + 1, cy, cx, ny, nx)));
  real rdzw4 = __fadd_rn(__fadd_rn(__fadd_rn(
      uh_rdzw(ph, phb, phb3d, km, cy, cx, ny, nx),
      uh_rdzw(ph, phb, phb3d, km, cy - 1, cx, ny, nx)),
      uh_rdzw(ph, phb, phb3d, km, cy - 1, cx - 1, ny, nx)),
      uh_rdzw(ph, phb, phb3d, km, cy, cx - 1, ny, nx));
  real dhat = __fsub_rn(
      uh_hatavg_v(v, msfv, fnm, fnp, cf1, cf2, cf3, cft1, cft2,
                  km + 1, cy, cx, nz, ny, nx),
      uh_hatavg_v(v, msfv, fnm, fnp, cf1, cf2, cf3, cft1, cft2,
                  km, cy, cx, nz, ny, nx));
  const real tmp1v = __fmul_rn(__fmul_rn(__fmul_rn(dhat, 0.25f), tmpzx),
                               rdzw4);
  const real dvdx_hat = __fsub_rn(
      uh_hat_v(v, msfv, km, cy, cx, ny, nx),
      uh_hat_v(v, msfv, km, cy, cx - 1, ny, nx));
  real rv = __fmul_rn(mm, __fsub_rn(__fmul_rn(rdx, dvdx_hat), tmp1v));

  // du/dy part (subtracted).
  real tmpzy = __fmul_rn(0.25f, __fadd_rn(__fadd_rn(__fadd_rn(
      uh_zy(ph, phb, phb3d, rdy, km, cy, cx - 1, ny, nx),
      uh_zy(ph, phb, phb3d, rdy, km, cy, cx, ny, nx)),
      uh_zy(ph, phb, phb3d, rdy, km + 1, cy, cx - 1, ny, nx)),
      uh_zy(ph, phb, phb3d, rdy, km + 1, cy, cx, ny, nx)));
  rdzw4 = __fadd_rn(__fadd_rn(__fadd_rn(
      uh_rdzw(ph, phb, phb3d, km, cy, cx, ny, nx),
      uh_rdzw(ph, phb, phb3d, km, cy, cx - 1, ny, nx)),
      uh_rdzw(ph, phb, phb3d, km, cy - 1, cx - 1, ny, nx)),
      uh_rdzw(ph, phb, phb3d, km, cy - 1, cx, ny, nx));
  dhat = __fsub_rn(
      uh_hatavg_u(u, msfu, fnm, fnp, cf1, cf2, cf3, cft1, cft2,
                  km + 1, cy, cx, nz, ny, nx),
      uh_hatavg_u(u, msfu, fnm, fnp, cf1, cf2, cf3, cft1, cft2,
                  km, cy, cx, nz, ny, nx));
  const real tmp1u = __fmul_rn(__fmul_rn(__fmul_rn(dhat, 0.25f), tmpzy),
                               rdzw4);
  const real dudy_hat = __fsub_rn(
      uh_hat_u(u, msfu, km, cy, cx, ny, nx),
      uh_hat_u(u, msfu, km, cy - 1, cx, ny, nx));
  return __fsub_rn(
      rv, __fmul_rn(mm, __fsub_rn(__fmul_rn(rdy, dudy_hat), tmp1u)));
}

// 8-point w average onto the corner column at a mass level (:7458-7462).
__device__ __forceinline__ real uh_wavg(
    const real* w, int km, int cy, int cx, int ny, int nx) {
  real acc = w[I3(km, cy, cx, ny, nx)];
  acc = __fadd_rn(acc, w[I3(km, cy, cx - 1, ny, nx)]);
  acc = __fadd_rn(acc, w[I3(km, cy - 1, cx, ny, nx)]);
  acc = __fadd_rn(acc, w[I3(km, cy - 1, cx - 1, ny, nx)]);
  acc = __fadd_rn(acc, w[I3(km + 1, cy, cx, ny, nx)]);
  acc = __fadd_rn(acc, w[I3(km + 1, cy, cx - 1, ny, nx)]);
  acc = __fadd_rn(acc, w[I3(km + 1, cy - 1, cx, ny, nx)]);
  acc = __fadd_rn(acc, w[I3(km + 1, cy - 1, cx - 1, ny, nx)]);
  return __fmul_rn(0.125f, acc);
}

// 4-point corner AGL height at a w level (:7487-7497): terms in the
// authority's order (i,j), (i-1,j), (i,j-1), (i-1,j-1), each
// ((ph+phb)/g - ht) rounded per operation.
__device__ __forceinline__ real uh_agl(
    const real* ph, const real* phb, int phb3d, const real* ht,
    int kw, int cy, int cx, int ny, int nx) {
  real acc = __fsub_rn(uh_z_at_w(ph, phb, phb3d, kw, cy, cx, ny, nx),
                       ht[cy * nx + cx]);
  acc = __fadd_rn(acc, __fsub_rn(
      uh_z_at_w(ph, phb, phb3d, kw, cy, cx - 1, ny, nx),
      ht[cy * nx + cx - 1]));
  acc = __fadd_rn(acc, __fsub_rn(
      uh_z_at_w(ph, phb, phb3d, kw, cy - 1, cx, ny, nx),
      ht[(cy - 1) * nx + cx]));
  acc = __fadd_rn(acc, __fsub_rn(
      uh_z_at_w(ph, phb, phb3d, kw, cy - 1, cx - 1, ny, nx),
      ht[(cy - 1) * nx + cx - 1]));
  return __fmul_rn(0.25f, acc);
}

}  // extern "C"
