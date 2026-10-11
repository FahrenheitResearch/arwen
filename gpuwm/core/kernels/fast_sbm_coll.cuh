// WRF mp_physics = 30: collision kernels at pressure, the Bott flux
// collision routines, modkrn_KS, collisional and spontaneous breakup.
// module_mp_SBM_Collision (:447-1691), Kernals_KS (:6238-6558) and
// module_mp_SBM_BreakUp (:96-444).  See fast_sbm_common.cuh for the
// transcription rules.  Arrays here are 1-based: a[1..n].

#ifndef WOOF_FAST_SBM_COLL_CUH
#define WOOF_FAST_SBM_COLL_CUH

// ---------------------------------------------------------------------------
// Kernals_KS (:6238-6541) evaluated on demand.  WRF fills 40 (33,33) module
// arrays per collision call; only cwll, cwls/cwsl, cwlg/cwgl, cwlh/cwhl and
// cwss are ever read (the crystal families are dead: FF2R is zero before
// every collision, :4671-4672).  Each value is computed with the same
// operations, in the same precision, as Kernals_KS stores it, so reading
// one here gives the bits WRF reads from its array.
// ---------------------------------------------------------------------------
template <class R> FHD R fsbm_dlnr() {
  // dlnr = dlog(2.0d0)/(3.0d0*scal) with REAL scal = 1.0 (:6256-6257);
  // dlog(2.0d0) is folded at compile time, correctly rounded.
  R scal = RL(1.0);
  return (R)(0.6931471805599453 / (3.0 * (double)scal));
}

template <class R> FHD R ckern_z(R p_z, R p_1, R p_2, R p_3, R c1, R c2, R c3) {
  R v = (R)0;  // every p_z falls in one band
  if (p_z >= p_1) v = c1;
  if (p_z <= p_3) v = c3;
  if (p_z < p_1 && p_z >= p_2) v = c2 + (c1 - c2) * (p_z - p_2) / (p_1 - p_2);
  if (p_z < p_2 && p_z > p_3) v = c3 + (c2 - c3) * (p_z - p_3) / (p_2 - p_3);
  return v;
}

// cwll(i,j) after Kernals_KS (:6263-6278).
template <class R> struct CkLL {
  const FsbmTab<R>* t;
  R p_z, dtc;
  FHD double operator()(int i, int j) const {
    R dlnr = fsbm_dlnr<R>();
    R c1 = TK(t->ywll, i, j, 0), c2 = TK(t->ywll, i, j, 1), c3 = TK(t->ywll, i, j, 2);
    R z = ckern_z<R>(p_z, RL(1.0e6), RL(0.75e6), RL(0.50e6), c1, c2, c3);
    double cw = (double)(z * dtc * dlnr);
    return T2(t->ecoalmassm, i, j, FSBM_NKR) * cw;
  }
};

// The ice-related families (:6280-6513): tables in slots 0 = 300, 1 = 500,
// 2 = 750 hPa.  transpose gives the cwil/cwgl/cwhl/cwsl transposes
// (:6515-6537).
template <class R> struct CkIce {
  const R* tab;
  R p_z, dtc;
  int transpose;
  FHD double operator()(int i0, int j0) const {
    int i = transpose ? j0 : i0;
    int j = transpose ? i0 : j0;
    R dlnr = fsbm_dlnr<R>();
    R dtimelnr = dtc * dlnr;
    R p_1 = RL(0.75e6), p_2 = RL(0.50e6), p_3 = RL(0.3e6);
    R y300 = TK(tab, i, j, 0), y500 = TK(tab, i, j, 1), y750 = TK(tab, i, j, 2);
    R v = (R)0;
    if (p_z >= p_1) v = y750 * dtimelnr;
    if (p_z <= p_3) v = y300 * dtimelnr;
    if (p_z < p_1 && p_z >= p_2) {
      R pdm = (p_z - p_2) / (p_1 - p_2);
      v = (y500 + (y750 - y500) * pdm) * dtimelnr;
    }
    if (p_z < p_2 && p_z > p_3) {
      R pdm = (p_z - p_3) / (p_2 - p_3);
      v = (y300 + (y500 - y300) * pdm) * dtimelnr;
    }
    return (double)v;
  }
};

// ---------------------------------------------------------------------------
// The flux-method collision routines (:463-1236).  gmin and the integration
// limits follow WRF, including its "return when the first active bin is
// nkr-1" quirk.  K is a kernel provider: K(i,j) is ckxy(i,j).
// A wrf_error_fatal returns its line.
// ---------------------------------------------------------------------------
FHD int fsbm_fl_check(double* fl, int k, int kp, int line) {
  if (fl[k] > 1.0 && fl[k] <= 1.0001) fl[k] = 1.0;
  if (fl[kp] > 1.0 && fl[kp] <= 1.0001) fl[kp] = 1.0;
  if (fl[k] > 1.0001 || fl[kp] > 1.0001 || fl[k] < 0.0 || fl[kp] < 0.0) return line;
  return FSBM_OK;
}

// D20: the finite breakup tables and collision floors can change mass.
// Project a nonnegative spectrum back to its incoming mass, keeping its shape.
// A negative breakup iterate stays untouched so the driver's retry still acts.
FHD double fsbm_gmass(const double* g,int n) {
 double total=0; for(int k=1;k<=n;++k) total+=g[k]; return total;
}
FHD void fsbm_gconserve(double* g,int n,double target) {
 double after=0; int largest=1;
 for(int k=1;k<=n;++k) { if(g[k]<0) return; after+=g[k]; if(g[k]>g[largest]) largest=k; }
 if(after<=0 || target<0) return;
 double factor=target/after;
 for(int k=1;k<=n;++k) g[k]*=factor;
 // Restore the tiny summation remainder to an occupied bin.
 double remainder=target-fsbm_gmass(g,n);
 if(g[largest]+remainder>=0) g[largest]+=remainder;
}

// coll_xyy_lwf (:463-650)
template <class K>
FHD_NOINLINE int coll_xyy_lwf(double* gx, double* gy, double* flx, double* fly, const K& ckxy,
                              const double* x, const double* y, const double* c, const int* ima,
                              double prdkrn, int nkr, int indc) {
  double gmin = 1.0e-60;
  int jx0 = 1, jx1 = 1, iy0 = 1, iy1 = 1;
  for (int j = 1; j <= nkr - 1; ++j) { jx0 = j; if (gx[j] > gmin) break; }
  if (jx0 == nkr - 1) return FSBM_OK;
  for (int j = nkr - 1; j >= jx0; --j) { jx1 = j; if (gx[j] > gmin) break; }
  for (int i = 1; i <= nkr - 1; ++i) { iy0 = i; if (gy[i] > gmin) break; }
  if (iy0 == nkr - 1) return FSBM_OK;
  for (int i = nkr - 1; i >= iy0; --i) { iy1 = i; if (gy[i] > gmin) break; }
  for (int i = iy0; i <= iy1; ++i) {
    if (gy[i] <= gmin) continue;
    int jmin = i;
    if (jmin == nkr - 1) return FSBM_OK;
    if (i < jx0) jmin = jx0 - indc;
    for (int j = jmin + indc; j <= jx1; ++j) {
      if (gx[j] <= gmin) continue;
      int k = T2(ima, i, j, nkr), kp = k + 1;
      double ckxy_ji = ckxy(j, i);
      double x01 = ckxy_ji * gy[i] * gx[j] * prdkrn;
      double x02 = fmin2(x01, gy[i] * x[j]);
      double x03 = fmin2(x02, gx[j] * y[i]);
      double gsi = x03 / x[j];
      double gsj = x03 / y[i];
      double gsk = gsi + gsj;
      if (gsk <= gmin) continue;
      double gsi_w = gsi * fly[i];
      double gsj_w = gsj * flx[j];
      double gsk_w = gsi_w + gsj_w;
      gsk_w = fmin2(gsk_w, gsk);
      gy[i] = gy[i] - gsi;
      gy[i] = fmax2(gy[i], 0.0);
      gx[j] = gx[j] - gsj;
      gx[j] = fmax2(gx[j], 0.0);
      double gk = gy[k] + gsk;
      if (gk <= gmin) continue;
      double gk_w = gy[k] * fly[k] + gsk_w;
      gk_w = fmin2(gk_w, gk);
      double fl_gk = gk_w / gk;
      double fl_gsk = gsk_w / gsk;
      double x1 = fx_log(gy[kp] / gk + 1.0e-15);
      double flux = gsk / x1 * (fx_exp(0.5 * x1) - fx_exp(x1 * (0.5 - T2(c, i, j, nkr))));
      flux = fmin2(flux, gsk);
      flux = fmin2(flux, gk);
      if (kp > FSBM_KP_FLUX_MAX) flux = 0.5 * flux;
      double flux_w = flux * fl_gsk;
      flux_w = fmin2(flux_w, gsk_w);
      flux_w = fmin2(flux_w, gk_w);
      gy[k] = gk - flux;
      gy[k] = fmax2(gy[k], gmin);
      double gy_k_w = gk * fl_gk - flux_w;
      gy_k_w = fmin2(gy_k_w, gy[k]);
      gy_k_w = fmax2(gy_k_w, 0.0);
      fly[k] = gy_k_w / gy[k];
      double gy_kp_old = gy[kp];
      gy[kp] = gy[kp] + flux;
      gy[kp] = fmax2(gy[kp], gmin);
      double gy_kp_w = gy_kp_old * fly[kp] + flux_w;
      gy_kp_w = fmin2(gy_kp_w, gy[kp]);
      fly[kp] = gy_kp_w / gy[kp];
      int e = fsbm_fl_check(fly, k, kp, 633);
      if (e) return e;
    }
  }
  return FSBM_OK;
}

// coll_xxx_lwf (:652-827)
template <class K>
FHD_NOINLINE int coll_xxx_lwf(double* g, double* fl, const K& ckxx, const double* x,
                              const double* c, const int* ima, double prdkrn, int nkr) {
  double gmin = 1.0e-16 * 1.0e3;  // g_lim*1.0d3 (:458, :668)
  int ix0 = 1, ix1 = 1;
  for (int i = 1; i <= nkr - 1; ++i) { ix0 = i; if (g[i] > gmin) break; }
  if (ix0 == nkr - 1) return FSBM_OK;
  for (int i = nkr - 1; i >= 1; --i) { ix1 = i; if (g[i] > gmin) break; }
  for (int i = ix0; i <= ix1; ++i) {
    if (g[i] <= gmin) continue;
    for (int j = i; j <= ix1; ++j) {
      if (g[j] <= gmin) continue;
      int k = T2(ima, i, j, nkr), kp = k + 1;
      double x01 = ckxx(i, j) * g[i] * g[j] * prdkrn;
      double x02 = fmin2(x01, g[i] * x[j]);
      double x03;
      if (j != k) x03 = fmin2(x02, g[j] * x[i]);
      else x03 = x02;
      double gsi = x03 / x[j];
      double gsj = x03 / x[i];
      double gsk = gsi + gsj;
      if (gsk <= gmin) continue;
      double gsi_w = gsi * fl[i];
      double gsj_w = gsj * fl[j];
      double gsk_w = gsi_w + gsj_w;
      gsk_w = fmin2(gsk_w, gsk);
      g[i] = g[i] - gsi;
      g[i] = fmax2(g[i], 0.0);
      g[j] = g[j] - gsj;
      if (j != k) g[j] = fmax2(g[j], 0.0);
      double gk = g[k] + gsk;
      if (g[j] < 0.0 && gk <= gmin) {
        g[j] = 0.0;
        g[k] = g[k] + gsi;
        continue;
      }
      if (gk <= gmin) continue;
      double gk_w = g[k] * fl[k] + gsk_w;
      gk_w = fmin2(gk_w, gk);
      double fl_gk = gk_w / gk;
      double fl_gsk = gsk_w / gsk;
      (void)fl_gk;
      double x1 = fx_log(g[kp] / gk + 1.0e-15);
      double flux = gsk / x1 * (fx_exp(0.5 * x1) - fx_exp(x1 * (0.5 - T2(c, i, j, nkr))));
      flux = fmin2(flux, gsk);
      flux = fmin2(flux, gk);
      if (kp > FSBM_KP_FLUX_MAX) flux = 0.5 * flux;
      double flux_w = flux * fl_gsk;
      flux_w = fmin2(flux_w, gsk_w);
      flux_w = fmin2(flux_w, gk_w);
      g[k] = gk - flux;
      g[k] = fmax2(g[k], gmin);
      double g_k_w = gk_w - flux_w;
      g_k_w = fmin2(g_k_w, g[k]);
      g_k_w = fmax2(g_k_w, 0.0);
      fl[k] = g_k_w / g[k];
      double g_kp_old = g[kp];
      g[kp] = g[kp] + flux;
      g[kp] = fmax2(g[kp], gmin);
      double g_kp_w = g_kp_old * fl[kp] + flux_w;
      g_kp_w = fmin2(g_kp_w, g[kp]);
      fl[kp] = g_kp_w / g[kp];
      int e = fsbm_fl_check(fl, k, kp, 812);
      if (e) return e;
    }
  }
  return FSBM_OK;
}

// coll_xyx_lwf (:829-1030)
template <class K>
FHD_NOINLINE int coll_xyx_lwf(double* gx, double* gy, double* flx, double* fly, const K& ckxy,
                              const double* x, const double* y, const double* c, const int* ima,
                              double prdkrn, int nkr, int indc, double* dm_rime) {
  double gmin = 1.0e-16 * 1.0e3;
  int jx0 = 1, jx1 = 1, iy0 = 1, iy1 = 1;
  for (int j = 1; j <= nkr - 1; ++j) { jx0 = j; if (gx[j] > gmin) break; }
  if (jx0 == nkr - 1) return FSBM_OK;
  for (int j = nkr - 1; j >= jx0; --j) { jx1 = j; if (gx[j] > gmin) break; }
  for (int i = 1; i <= nkr - 1; ++i) { iy0 = i; if (gy[i] > gmin) break; }
  if (iy0 == nkr - 1) return FSBM_OK;
  for (int i = nkr - 1; i >= iy0; --i) { iy1 = i; if (gy[i] > gmin) break; }
  for (int i = 1; i <= nkr; ++i) dm_rime[i] = 0.0;
  for (int i = iy0; i <= iy1; ++i) {
    if (gy[i] <= gmin) continue;
    int jmin = i;
    if (jmin == nkr - 1) return FSBM_OK;
    if (i < jx0) jmin = jx0 - indc;
    for (int j = jmin + indc; j <= jx1; ++j) {
      if (gx[j] <= gmin) continue;
      int k = T2(ima, i, j, nkr), kp = k + 1;
      double x01 = ckxy(j, i) * gy[i] * gx[j] * prdkrn;
      double x02 = fmin2(x01, gy[i] * x[j]);
      double x03;
      if (j != k) x03 = fmin2(x02, gx[j] * y[i]);
      else x03 = x02;
      double gsi = x03 / x[j];
      double gsj = x03 / y[i];
      double gsk = gsi + gsj;
      if (gsk <= gmin) continue;
      double gsi_w = gsi * fly[i];
      double gsj_w = gsj * flx[j];
      double gsk_w = gsi_w + gsj_w;
      gsk_w = fmin2(gsk_w, gsk);
      gy[i] = gy[i] - gsi;
      gy[i] = fmax2(gy[i], 0.0);
      gx[j] = gx[j] - gsj;
      if (j != k) gx[j] = fmax2(gx[j], 0.0);
      double gk = gx[k] + gsk;
      if (gk <= gmin) continue;
      double gk_w = gx[k] * flx[k] + gsk_w;
      gk_w = fmin2(gk_w, gk);
      double fl_gk = gk_w / gk;
      double fl_gsk = gsk_w / gsk;
      (void)fl_gk;
      double x1 = fx_log(gx[kp] / gk + 1.0e-15);
      double flux = gsk / x1 * (fx_exp(0.5 * x1) - fx_exp(x1 * (0.5 - T2(c, i, j, nkr))));
      flux = fmin2(flux, gsk);
      flux = fmin2(flux, gk);
      if (kp > FSBM_KP_FLUX_MAX) flux = 0.5 * flux;
      double flux_w = flux * fl_gsk;
      flux_w = fmin2(flux_w, gsk_w);
      flux_w = fmin2(flux_w, gk_w);
      double frac_split = flux / gsk;
      if (frac_split < 0.0) frac_split = 0.0;
      if (frac_split > 1.0) frac_split = 1.0;
      dm_rime[k] = dm_rime[k] + gsi * (1.0 - frac_split);
      dm_rime[kp] = dm_rime[kp] + gsi * frac_split;
      gx[k] = gk - flux;
      gx[k] = fmax2(gx[k], gmin);
      double gx_k_w = gk_w - flux_w;
      gx_k_w = fmin2(gx_k_w, gx[k]);
      gx_k_w = fmax2(gx_k_w, 0.0);
      flx[k] = gx_k_w / gx[k];
      double gx_kp_old = gx[kp];
      gx[kp] = gx[kp] + flux;
      gx[kp] = fmax2(gx[kp], gmin);
      double gx_kp_w = gx_kp_old * flx[kp] + flux_w;
      gx_kp_w = fmin2(gx_kp_w, gx[kp]);
      flx[kp] = gx_kp_w / gx[kp];
      int e = fsbm_fl_check(flx, k, kp, 1014);
      if (e) return e;
    }
  }
  return FSBM_OK;
}

// coll_xyz_lwf (:1032-1236)
template <class K>
FHD_NOINLINE int coll_xyz_lwf(double* gx, double* gy, double* gz, double* flx, double* fly,
                              double* flz, const K& ckxy, const double* x, const double* y,
                              const double* c, const int* ima, double prdkrn, int nkr,
                              int indc) {
  double gmin = 1.0e-60;
  int jx0 = 1, jx1 = 1, iy0 = 1, iy1 = 1;
  for (int j = 1; j <= nkr - 1; ++j) { jx0 = j; if (gx[j] > gmin) break; }
  if (jx0 == nkr - 1) return FSBM_OK;
  for (int j = nkr - 1; j >= jx0; --j) { jx1 = j; if (gx[j] > gmin) break; }
  for (int i = 1; i <= nkr - 1; ++i) { iy0 = i; if (gy[i] > gmin) break; }
  if (iy0 == nkr - 1) return FSBM_OK;
  for (int i = nkr - 1; i >= iy0; --i) { iy1 = i; if (gy[i] > gmin) break; }
  for (int i = iy0; i <= iy1; ++i) {
    if (gy[i] <= gmin) continue;
    int jmin = i;
    if (jmin == nkr - 1) return FSBM_OK;
    if (i < jx0) jmin = jx0 - indc;
    for (int j = jmin + indc; j <= jx1; ++j) {
      if (gx[j] <= gmin) continue;
      int k = T2(ima, i, j, nkr), kp = k + 1;
      double ckxy_ji = ckxy(j, i);
      double x01 = ckxy_ji * gy[i] * gx[j] * prdkrn;
      double x02 = fmin2(x01, gy[i] * x[j]);
      double x03 = fmin2(x02, gx[j] * y[i]);
      double gsi = x03 / x[j];
      double gsj = x03 / y[i];
      double gsk = gsi + gsj;
      if (gsk <= gmin) continue;
      double gsi_w = gsi * fly[i];
      double gsj_w = gsj * flx[j];
      double gsk_w = gsi_w + gsj_w;
      gsk_w = fmin2(gsk_w, gsk);
      gy[i] = gy[i] - gsi;
      gy[i] = fmax2(gy[i], 0.0);
      gx[j] = gx[j] - gsj;
      gx[j] = fmax2(gx[j], 0.0);
      double gk = gz[k] + gsk;
      if (gk <= gmin) continue;
      double gk_w = gz[k] * flz[k] + gsk_w;
      gk_w = fmin2(gk_w, gk);
      double fl_gk = gk_w / gk;
      double fl_gsk = gsk_w / gsk;
      double x1 = fx_log(gz[kp] / gk + 1.0e-15);
      double flux = gsk / x1 * (fx_exp(0.5 * x1) - fx_exp(x1 * (0.5 - T2(c, i, j, nkr))));
      flux = fmin2(flux, gsk);
      flux = fmin2(flux, gk);
      if (kp > FSBM_KP_FLUX_MAX) flux = 0.5 * flux;
      double flux_w = flux * fl_gsk;
      flux_w = fmin2(flux_w, gsk_w);
      flux_w = fmin2(flux_w, gk_w);
      gz[k] = gk - flux;
      gz[k] = fmax2(gz[k], gmin);
      double gz_k_w = gk * fl_gk - flux_w;
      gz_k_w = fmin2(gz_k_w, gz[k]);
      gz_k_w = fmax2(gz_k_w, 0.0);
      flz[k] = gz_k_w / gz[k];
      double gz_kp_old = gz[kp];
      gz[kp] = gz[kp] + flux;
      gz[kp] = fmax2(gz[kp], gmin);
      double gz_kp_w = gz_kp_old * flz[kp] + flux_w;
      gz_kp_w = fmin2(gz_kp_w, gz[kp]);
      flz[kp] = gz_kp_w / gz[kp];
      int e = fsbm_fl_check(flz, k, kp, 1221);
      if (e) return e;
    }
  }
  return FSBM_OK;
}

// ---------------------------------------------------------------------------
// modkrn_KS CASE(11) (:1481-1498) through INTERPOL_SE (:1362-1393).
// The SE table values 10.0**(-x) are compile-time constants in gfortran
// (MPFR, correctly rounded in the build's REAL kind); they are written here
// as those correctly rounded values.  Above 0 C WRF returns before setting
// the output (D6) and, at exactly -40 C, INTERPOL_SE assigns nothing at its
// last node (an uninitialised read).  Both are defined here: 1.0 above 0 C
// (the value only feeds an unused diagnostic there) and the last node's
// value at -40 C, the limit of the interpolation.  Declared divergences.
// ---------------------------------------------------------------------------
template <class R> FHD double modkrn_case11(double tt) {
  R t_tab[7] = {RL(0.0), RL(-0.813), RL(-5.26), RL(-10.13), RL(-14.63), RL(-20.02), RL(-40.0)};
  R se_tab[7];
  // 10**(-0.693), 10**(-0.72), 10**(-0.877), 10**(-1.050), 10**(-1.212),
  // 10**(-1.401), 10**(-2.082): correctly rounded in each REAL kind.
  if (fsbm_is_double<R>::value) {
    se_tab[0] = (R)0x1.9f44f8c59b4a1p-3; se_tab[1] = (R)0x1.863d04d5f8cbfp-3;
    se_tab[2] = (R)0x1.0fd9b2d3df69dp-3; se_tab[3] = (R)0x1.6d0e6f32e6ea2p-4;
    se_tab[4] = (R)0x1.f6cb38bf12933p-5; se_tab[5] = (R)0x1.45611af0b3404p-5;
    se_tab[6] = (R)0x1.0f4cd294a37f5p-7;
  } else {
    se_tab[0] = (R)0x1.9f44f8p-3f; se_tab[1] = (R)0x1.863d04p-3f;
    se_tab[2] = (R)0x1.0fd9b4p-3f; se_tab[3] = (R)0x1.6d0e72p-4f;
    se_tab[4] = (R)0x1.f6cb38p-5f; se_tab[5] = (R)0x1.45611ap-5f;
    se_tab[6] = (R)0x1.0f4cd2p-7f;
  }
  double tc = tt - (double)RL(273.15);
  if (tc > 0.0) return 1.0;  // D6, defined
  double factor_t;
  if (tc > (double)t_tab[0]) {
    factor_t = (double)se_tab[0];
  } else if (tc < (double)t_tab[6]) {
    factor_t = (double)se_tab[6];
  } else {
    factor_t = (double)se_tab[6];  // tc == -40 exactly: the limit
    for (int i = 1; i < 7; ++i) {
      if (tc > (double)t_tab[i]) {
        int j = i - 1;
        factor_t = (double)se_tab[j] + (double)((se_tab[i] - se_tab[j]) / (t_tab[i] - t_tab[j])) *
                                           (tc - (double)t_tab[j]);
        break;
      }
    }
  }
  if (tc < -40.0) factor_t = 0.0;
  if (factor_t > (double)RL(1.0)) factor_t = 1.0;
  if (tc > 0.0) factor_t = 1.0;
  return factor_t;
}

// ---------------------------------------------------------------------------
// coll_breakup_KS (:1505-1634): Bleck's first-order breakup over the top
// JBREAK bins, on 1-based gt_mg/xt_mg of length jmax.
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE int coll_breakup_ks(double* gt_mg, const double* xt_mg, int jmax, double dt,
                                 int jbreak, const R* pkij, const R* qkj, int nkrinput,
                                 int nkr) {
  int ie = jbreak, je = jbreak, ke = jbreak;
  int nkrdiff = nkrinput - nkr;
  double gt[FSBM_JMAX + 1], xt[FSBM_JMAX + 2], ft[FSBM_JMAX + 1], fa[FSBM_JMAX + 1];
  double dg[FSBM_JMAX + 1], df[FSBM_JMAX + 1], dbreak[FSBM_JBREAK + 1], amweight[FSBM_JBREAK + 1];
  for (int j = 1; j <= jmax; ++j) {
    xt[j] = xt_mg[j];
    gt[j] = gt_mg[j];
    ft[j] = gt[j] / xt[j] / xt[j];
  }
  int jdiff = jmax - jbreak;
  for (int k = 1; k <= jmax; ++k) fa[k] = (double)RL(0.0);
  for (int k = 1; k <= ke - nkrdiff; ++k) fa[k] = ft[k + jdiff + nkrdiff];
  xt[jmax + 1] = xt[jmax] * 2.0;
  for (int k = 1; k <= jbreak; ++k) { amweight[k] = 0.0; dbreak[k] = 0.0; }
  for (int k = 1; k <= ke - nkrdiff; ++k) {
    double gain = 0.0;
    for (int i = 1; i <= ie - nkrdiff; ++i)
      for (int j = 1; j <= i; ++j)
        gain = gain + fa[i] * fa[j] * (double)pkij[(k - 1) + (i - 1) * FSBM_JBREAK +
                                                   (j - 1) * FSBM_JBREAK * FSBM_JBREAK];
    double aloss = 0.0;
    for (int j = 1; j <= je - nkrdiff; ++j)
      aloss = aloss + fa[j] * (double)qkj[(k - 1) + (j - 1) * FSBM_JBREAK];
    int j = jmax - jbreak + k + nkrdiff;
    amweight[k] = (double)RL(2.0) /
                  (fx_pow(xt[j + 1], (double)RL(2.0)) - fx_pow(xt[j], (double)RL(2.0)));
    dbreak[k] = amweight[k] * (gain - fa[k] * aloss);
    if (dbreak[k] != dbreak[k]) return 1592;
  }
  for (int j = 1; j <= jmax; ++j) df[j] = 0.0;
  for (int j = 1; j <= ke - nkrdiff; ++j) df[j + jdiff] = dbreak[j];
  for (int j = 1; j <= jmax; ++j) dg[j] = df[j] * xt[j] * xt[j];
  for (int j = 1; j <= jmax; ++j) gt[j] = gt[j] + dg[j] * dt;
  for (int j = 1; j <= jmax; ++j) gt_mg[j] = gt[j];
  return FSBM_OK;
}

// ---------------------------------------------------------------------------
// Spont_Rain_BreakUp (:205-300) on a double FF1R (1-based).  DTwrf is REAL.
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE void spont_rain_breakup(R dtwrf, double* ff1r, const double* xl, const double* prob,
                                     const double* gain_var_new, int nkr, int ikr_spon_break) {
  // COL is DOUBLE PRECISION PARAMETER = 0.23105 (a REAL literal, :108)
  const double col = (double)RL(0.23105);
  R dtime_spon_break = dtwrf;
  // DROPRADII is recomputed at :223-226 and never read: not repeated.
  double s = 0.0;
  for (int kr = 1; kr <= nkr; ++kr) s = s + ff1r[kr];
  if (s <= nkr * 1.0e-30) return;
  int imax = nkr;
  for (int i = nkr; i >= 1; --i) {
    imax = i;
    if (ff1r[i] > 0.0) break;
  }
  if (imax < ikr_spon_break) return;
  double psi1[FSBM_NKR + 1], drops_break[FSBM_NKR + 1], dropconc_bf[FSBM_NKR + 1],
      dropconc_af[FSBM_NKR + 1];
  for (int kr = 1; kr <= nkr; ++kr) {
    psi1[kr] = ff1r[kr];
    drops_break[kr] = 0.0;
    dropconc_bf[kr] = 0.0;
  }
  for (int kr = 1; kr <= imax; ++kr) {
    double dm = 3.0 * col * xl[kr];
    dropconc_bf[kr] = dropconc_bf[kr] + dm * psi1[kr];
  }
  for (int kr = 1; kr <= nkr; ++kr) dropconc_af[kr] = dropconc_bf[kr];
  for (int kr = imax; kr >= ikr_spon_break; --kr) {
    double tmp_1 = prob[kr - 1] * (double)dtime_spon_break;
    double tmp_2 = fx_exp(-tmp_1);
    double tmp_3 = dropconc_bf[kr];
    dropconc_af[kr] = tmp_2 * tmp_3;
    drops_break[kr] = dropconc_bf[kr] - dropconc_af[kr];
  }
  for (int j = ikr_spon_break; j <= imax; ++j)
    for (int i = 1; i <= j - 1; ++i)
      dropconc_af[i] = dropconc_af[i] + drops_break[j] * T2(gain_var_new, j, i, FSBM_NKR) * xl[i];
  for (int kr = 1; kr <= imax; ++kr) {
    double dm = 3.0 * col * xl[kr];
    psi1[kr] = dropconc_af[kr] / dm;
  }
  for (int kr = 1; kr <= nkr; ++kr) ff1r[kr] = psi1[kr];
}

#endif  // WOOF_FAST_SBM_COLL_CUH
