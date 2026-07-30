// gpuwm/core/kernels/uh_diag.cu
//
// WRF v4.6.1 updraft-helicity diagnostic (UP_HELI_MAX), transcribed from
// SUBROUTINE cal_helicity, dyn_em/module_diffusion_em.F:7132-7579 of the
// pinned tree d66e442fccc04111067e29274c9f9eaccc3cef28 (blob sha256
// a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54), plus
// the metric prep it consumes (compute_diff_metrics, :6882-7130) evaluated
// pointwise on demand instead of into zx/zy/rdzw scratch -- same
// expressions, same operands, same rounding.
//
// Bitwise discipline (the noahmp kernels' rule 1): every FP32 operation is
// an explicit rounding intrinsic (__fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn),
// which pins round-to-nearest-even AND makes nvcc's contraction pass a
// no-op, so -fmad=true cannot fuse a site gfortran -O0 did not fuse.  The
// two subroutines are pure arithmetic (build receipt: the -O0 oracle object
// has no libm/libmvec undefined symbols), so no transcribed libm is needed.
//
// Geometry: the specified/nested (non-periodic) single-tile bounds of the
// WRF call (its=ids, ite=ide-1, k_end=kpe; solve_em.F:299-300,
// module_first_rk_step_part2.F:533-556):
//   i_start=ids+1, i_end=ide-1 -> 0-based corner range [1, nx-1];
//   smoother [MAX(ids+1,its), MIN(ide-2,ite)] -> [1, nx-2] (same in j).
// WRF's boundary copy also writes up_heli_max(ide,j)/up_heli_max(i,jde),
// one slot beyond the mass grid into halo memory that never reaches
// output; gpuwm's (ny,nx) field has no such slot and the write is dropped.
//
// One defined-behaviour divergence, unreachable in practice: at the top
// mass layer (k = ktf) WRF reads wavg(ktf+1)/rvort(ktf+1), which its loops
// never wrote (uninitialised automatic arrays), guarded only by the
// zu <= 5000 m test -- reachable solely when the model top is below
// 5000 m AGL.  gpuwm substitutes 0.0f for both (which zeroes the column
// through the wavg > 0 test) instead of reading undefined memory.
//
// Layout notes: gpuwm arrays are C-contiguous (k, j, i) with k outermost;
// u is (nz, ny, nx+1), v (nz, ny+1, nx), w/ph (nz+1, ny, nx), phb either
// (nz+1, ny, nx) (terrain) or a flat (nz+1,) column broadcast via phb3d=0.
// dn/dnw/fnm/fnp are the (nz,) coordinate columns with the WRF k=1
// endpoint zero at index 0 (gpuwm/core/grid.py:41-64); WRF's 1-based
// dn(k)/fnm(k) is dn[k-1]/fnm[k-1] here.

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

// Per-column integration (:7474-7511): uh and use_column over the corner
// range [1, nx-1] x [1, ny-1]; cells outside carry uh = 0 exactly like
// WRF's never-written allocation rows.
__global__ void uh_columns(
    const real* __restrict__ u, const real* __restrict__ v,
    const real* __restrict__ w, const real* __restrict__ ph,
    const real* __restrict__ phb, const int phb3d,
    const real* __restrict__ msfu, const real* __restrict__ msfv,
    const real* __restrict__ ht,
    const real* __restrict__ dn, const real* __restrict__ dnw,
    const real* __restrict__ fnm, const real* __restrict__ fnp,
    const real cf1, const real cf2, const real cf3,
    const real rdx, const real rdy,
    real* __restrict__ uh, real* __restrict__ usecol,
    const int nz, const int ny, const int nx) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= nx * ny) return;
  const int cy = col / nx;
  const int cx = col - cy * nx;
  if (cx == 0 || cy == 0) {          // outside the corner loop range
    uh[col] = 0.0f;
    usecol[col] = 1.0f;
    return;
  }

  // cft1/cft2 (:7211-7215): ktes1 = kte-1 -> dnw/dn 1-based index nz,
  // 0-based nz-1.
  const real cft2 = -__fdiv_rn(__fmul_rn(0.5f, dnw[nz - 1]), dn[nz - 1]);
  const real cft1 = __fsub_rn(1.0f, cft2);

  // mm (:7251): 0.25*(msfux(i,j-1)+msfux(i,j))*(msfvy(i-1,j)+msfvy(i,j)).
  const real mm = __fmul_rn(
      __fmul_rn(0.25f, __fadd_rn(msfu[(cy - 1) * (nx + 1) + cx],
                                 msfu[cy * (nx + 1) + cx])),
      __fadd_rn(msfv[cy * nx + cx - 1], msfv[cy * nx + cx]));

  real uh_acc = 0.0f;
  real use = 1.0f;
  real zl = uh_agl(ph, phb, phb3d, ht, 0, cy, cx, ny, nx);
  real have_k = 0.0f;     // whether wa_k/rv_k carry layer km's values
  real wa_k = 0.0f, rv_k = 0.0f;
  for (int km = 0; km < nz; ++km) {
    const real zu = uh_agl(ph, phb, phb3d, ht, km + 1, cy, cx, ny, nx);
    if (zl >= 2000.0f && zu <= 5000.0f) {
      if (have_k == 0.0f) {
        wa_k = uh_wavg(w, km, cy, cx, ny, nx);
        rv_k = uh_rvort(u, v, ph, phb, phb3d, msfu, msfv, fnm, fnp,
                        cf1, cf2, cf3, cft1, cft2, rdx, rdy, mm,
                        km, cy, cx, nz, ny, nx);
      }
      // WRF's wavg/rvort at ktf+1 are never written; substitute 0.0f
      // (defined stand-in, reachable only with a sub-5000 m model top).
      real wa_k1 = 0.0f, rv_k1 = 0.0f;
      if (km + 1 < nz) {
        wa_k1 = uh_wavg(w, km + 1, cy, cx, ny, nx);
        rv_k1 = uh_rvort(u, v, ph, phb, phb3d, msfu, msfv, fnm, fnp,
                         cf1, cf2, cf3, cft1, cft2, rdx, rdy, mm,
                         km + 1, cy, cx, nz, ny, nx);
      }
      if (wa_k > 0.0f && wa_k1 > 0.0f) {
        // uh = uh + ((wavg(k)*rvort(k) + wavg(k+1)*rvort(k+1))*0.5)*(zu-zl)
        const real pair = __fmul_rn(
            __fadd_rn(__fmul_rn(wa_k, rv_k), __fmul_rn(wa_k1, rv_k1)),
            0.5f);
        uh_acc = __fadd_rn(uh_acc, __fmul_rn(pair, __fsub_rn(zu, zl)));
      } else {
        use = 0.0f;
        uh_acc = 0.0f;
      }
      wa_k = wa_k1;
      rv_k = rv_k1;
      have_k = 1.0f;
    } else {
      have_k = 0.0f;
    }
    zl = zu;
  }
  uh[col] = uh_acc;
  usecol[col] = use;
}

// Smoother + running max (:7515-7533): interior [1, nx-2] x [1, ny-2],
// 9-point weights 0.25/0.125/0.0625 in the authority's term order, strict
// > against the carried maximum, gated on the column's use_column flag.
__global__ void uh_smooth_max(
    const real* __restrict__ uh, const real* __restrict__ usecol,
    real* __restrict__ up_heli_max,
    const int ny, const int nx) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= nx * ny) return;
  const int cy = col / nx;
  const int cx = col - cy * nx;
  if (cx < 1 || cx > nx - 2 || cy < 1 || cy > ny - 2) return;

  const real t1 = __fmul_rn(0.25f, uh[cy * nx + cx]);
  real edge = __fadd_rn(uh[cy * nx + cx + 1], uh[cy * nx + cx - 1]);
  edge = __fadd_rn(edge, uh[(cy + 1) * nx + cx]);
  edge = __fadd_rn(edge, uh[(cy - 1) * nx + cx]);
  real corner = __fadd_rn(uh[(cy + 1) * nx + cx + 1],
                          uh[(cy - 1) * nx + cx + 1]);
  corner = __fadd_rn(corner, uh[(cy + 1) * nx + cx - 1]);
  corner = __fadd_rn(corner, uh[(cy - 1) * nx + cx - 1]);
  const real uh_smth = __fadd_rn(
      __fadd_rn(t1, __fmul_rn(0.125f, edge)), __fmul_rn(0.0625f, corner));

  if (usecol[cy * nx + cx] != 0.0f) {
    if (uh_smth > up_heli_max[cy * nx + cx]) {
      up_heli_max[cy * nx + cx] = uh_smth;
    }
  }
}

}  // extern "C"
