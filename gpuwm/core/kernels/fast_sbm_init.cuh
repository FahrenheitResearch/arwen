// WRF mp_physics = 30: the derived tables of FAST_HUCMINIT (:5384-6236):
// the Bott grid (courant_bott_KS), DROPRADII, the aerosol spectra
// (LogNormal_modes_Aerosol), the drop coalescence efficiencies
// (BREAKINIT_KS and its ecoal* family), the spontaneous-breakup tables
// (Spontanous_Init) and the mass grids in mg.
//
// Host only.  These tables depend on the NCAR files alone (not on dt), so
// they are computed once, by this code under the platform libm (what the
// Fortran oracle calls), checked against WRF's own dump, and shipped as
// the table pack the device reads (gpuwm/data/fast_sbm/).

#ifndef WOOF_FAST_SBM_INIT_CUH
#define WOOF_FAST_SBM_INIT_CUH
#if !FSBM_DEVICE

// courant_bott_KS (:1636-1689).  chucm, ima (33,33) zero on entry.
template <class R> void courant_bott_ks(const R* xl, double* chucm, int* ima, double scal) {
  const int nkr = FSBM_NKR;
  double xl_mg[FSBM_NKR + 1];
  for (int i = 1; i <= nkr; ++i) xl_mg[i] = (double)xl[i - 1] * 1.0e3;
  double dlnr = 0.6931471805599453 / (3.0 * scal);
  for (int i = 1; i <= nkr; ++i) {
    for (int j = i; j <= nkr; ++j) {
      double x0 = xl_mg[i] + xl_mg[j];
      for (int k = j; k <= nkr; ++k) {
        int kk = k;
        if (k == 1) continue;
        if (xl_mg[k] >= x0 && xl_mg[k - 1] < x0) {
          T2(chucm, i, j, nkr) = ::log(x0 / xl_mg[k - 1]) / (3.0 * dlnr);
          if (T2(chucm, i, j, nkr) > 1.0 - 1.0e-08) {
            T2(chucm, i, j, nkr) = 0.0;
            kk = kk + 1;
          }
          T2(ima, i, j, nkr) = (nkr - 1 < kk - 1) ? nkr - 1 : kk - 1;
          break;
        }
      }
      T2(chucm, j, i, nkr) = T2(chucm, i, j, nkr);
      T2(ima, j, i, nkr) = T2(ima, i, j, nkr);
    }
  }
}

// LogNormal_modes_Aerosol (:3735-3838).  itype 1 maritime, 2 continental.
template <class R>
void lognormal_modes_aerosol(double* fccnr_out, R* xccn, R* rccn, const R* xl, R col, R ro_solute,
                             R scale_fa, int itype) {
  const int nk = FSBM_NKR_AERO;
  const double rccn_max = 0.4e-4, rccn_min_3ln = 0.00048e-4, pi = 3.14159265;
  double deg01 = 1.0 / 3.0;
  double x0drop = (double)xl[1];
  double x0ccn = x0drop / (double)powi_gcc<R>(RL(2.0), nk);
  double roccn[FSBM_NKR_AERO + 1];
  for (int kr = nk; kr >= 1; --kr) {
    roccn[kr] = (double)ro_solute;
    double x0 = x0ccn * powi_gcc<double>(2.0, kr);
    double r0 = ::pow(3.0 * x0 / 4.0 / 3.141593 / roccn[kr], deg01);
    xccn[kr - 1] = (R)x0;
    rccn[kr - 1] = (R)r0;
  }
  double ccncon1, radius_mean1, sig1, ccncon2, radius_mean2, sig2, ccncon3, radius_mean3, sig3;
  if (itype == 1) {
    ccncon1 = (double)RL(340.000); radius_mean1 = 0.00500e-04; sig1 = (double)RL(1.60000);
    ccncon2 = (double)RL(60.0000); radius_mean2 = 0.03500e-04; sig2 = (double)RL(2.00000);
    ccncon3 = (double)RL(3.10000); radius_mean3 = 0.31000e-04; sig3 = (double)RL(2.70000);
  } else {
    ccncon1 = (double)RL(1000.000); radius_mean1 = 0.00800e-04; sig1 = (double)RL(1.60000);
    ccncon2 = (double)RL(800.0000); radius_mean2 = 0.03400e-04; sig2 = (double)RL(2.10000);
    ccncon3 = (double)RL(0.72000); radius_mean3 = 0.46000e-04; sig3 = (double)RL(2.20000);
  }
  double tmp[FSBM_NKR_AERO + 1];
  for (int kr = 1; kr <= nk; ++kr) tmp[kr] = (double)RL(0.0);
  double arg11 = ccncon1 / (::sqrt(2.0 * pi) * ::log(sig1));
  double arg21 = ccncon2 / (::sqrt(2.0 * pi) * ::log(sig2));
  double arg31 = ccncon3 / (::sqrt(2.0 * pi) * ::log(sig3));
  // log(2.0)/3.0: a default-REAL constant expression, folded at compile time
  R l23 = fsbm_is_double<R>::value ? (R)(0.6931471805599453 / 3.0) : (R)(0.6931472f / 3.0f);
  for (int kr = nk; kr >= 1; --kr) {
    double rc = (double)rccn[kr - 1];
    if (rc > rccn_min_3ln && rc < rccn_max) {
      double arg12 = ::pow(::log(rc / radius_mean1), (double)RL(2.0));
      double arg13 = 2.0 * (::pow(::log(sig1), (double)RL(2.0)));
      double n1 = arg11 * ::exp(-arg12 / arg13) * (double)l23;
      double arg22 = ::pow(::log(rc / radius_mean2), (double)RL(2.0));
      double arg23 = 2.0 * (::pow(::log(sig2), (double)RL(2.0)));
      double n2 = n1 + arg21 * ::exp(-arg22 / arg23) * (double)l23;
      double arg32 = ::pow(::log(rc / radius_mean3), (double)RL(2.0));
      double arg33 = 2.0 * (::pow(::log(sig3), (double)RL(2.0)));
      double n3 = n2 + arg31 * ::exp(-arg32 / arg33) * (double)l23;
      tmp[kr] = n3 / (double)col;
    }
  }
  for (int kr = 1; kr <= nk; ++kr) fccnr_out[kr - 1] = (double)scale_fa * tmp[kr];
}

// vTBeard (:9026-9048) over the radii it searches.
inline double vt_beard(double diam, const double* radii, const double* vr1) {
  double aa = diam / 2.0;
  double v = 0.0;
  if (aa <= radii[1]) v = vr1[1];
  if (aa > radii[FSBM_NKR]) v = vr1[FSBM_NKR];
  for (int kr = 1; kr <= FSBM_NKR - 1; ++kr)
    if (aa > radii[kr] && aa <= radii[kr + 1]) v = vr1[kr + 1];
  return v;
}

inline void collenergy(double& dgr, double& dkl, double& cke, double& st, double& sc,
                       const double* radii, const double* vr1) {
  const double pi = 3.1415927, rho = 1.0, sigma = 72.8, epsf = 1.0e-30, fpmin = 1.0e-30;
  double ak10 = rho * pi / 12.0;
  dgr = fmax2(dgr, epsf);
  dkl = fmax2(dkl, epsf);
  double dgka2 = (dgr * dgr) + (dkl * dkl);
  double dgka3 = (powi_gcc<double>(dgr, 3)) + (powi_gcc<double>(dkl, 3));
  if (dgr != dkl) {
    double v1 = vt_beard(dgr, radii, vr1);
    double v2 = vt_beard(dkl, radii, vr1);
    double dv = (v1 - v2);
    if (dv < fpmin) dv = fpmin;
    dv = dv * dv;
    if (dv < fpmin) dv = fpmin;
    double dgkb3 = (powi_gcc<double>(dgr, 3)) * (powi_gcc<double>(dkl, 3));
    cke = ak10 * dv * dgkb3 / dgka3;
  } else {
    cke = 0.0;
  }
  st = pi * sigma * dgka2;
  sc = pi * sigma * ::pow(dgka3, 2.0 / 3.0);
}

inline double ecoal_low_list(double dgr, double dkl, const double* radii, const double* vr1) {
  const double sigma = 72.8, aka = 0.778, akb = 2.61e-4, epsi = 1.0e-20;
  double cke, st, sc;
  collenergy(dgr, dkl, cke, st, sc, radii, vr1);
  double dstsc = st - sc;
  double et = cke + dstsc;
  double ecl;
  if (et < 50.0) {
    double qq0 = 1.0 + (dkl / dgr);
    double qq1 = aka / (qq0 * qq0);
    double qq2 = akb * sigma * (et * et) / (sc + epsi);
    ecl = qq1 * ::exp(-qq2);
  } else {
    ecl = 0.0;
  }
  return ecl;
}

inline double ecoal_ochs(double d_l, double d_s, const double* radii, const double* vr1) {
  const double pi = 3.1415927, sigma = 72.8, fpmin = 1.0e-30;
  double r_s = 0.5 * d_s, r_l = 0.5 * d_l;
  double p = r_s / r_l;
  double vtl = vt_beard(d_l, radii, vr1);
  double vts = vt_beard(d_s, radii, vr1);
  double dv = fabs_(vtl - vts);
  if (dv < fpmin) dv = fpmin;
  double weber = r_s * (dv * dv) / sigma;
  double pa1 = 1.0 + p;
  double pa2 = 1.0 + p * p;
  double pa3 = 1.0 + powi_gcc<double>(p, 3);
  double g = ::sqrt(8.0) / (6.0 * pi) * powi_gcc<double>(p, 4) * pa1 / (pa2 * pa3);
  double x = ::pow(weber, 0.5) * g;
  return 0.767 - 10.14 * x;
}

inline double ecoal_diam(double deta, double dksi, const double* radii, const double* vr1) {
  const double pi = 3.1415927, one = 1.0, eps = 1.0e-30;
  double dgr = fmax2(deta, dksi), dkl = fmin2(deta, dksi);
  double rgr = 0.5 * dgr, rkl = 0.5 * dkl;
  double q = 0.5 * (rkl + rgr);
  double qmin = 250.0e-4, qmax = 500.0e-4;
  double e;
  if (dkl < 100.0e-4) {
    e = 1.0;
  } else if (q < qmin) {
    e = ecoal_ochs(dgr, dkl, radii, vr1);
  } else if (q >= qmin && q < qmax) {
    double x = (q - qmin) / (qmax - qmin);
    double sin1 = ::sin(pi / 2.0 * x);
    double cos1 = ::cos(pi / 2.0 * x);
    double e1 = ecoal_ochs(dgr, dkl, radii, vr1);
    double e2 = ecoal_low_list(dgr, dkl, radii, vr1);
    e = cos1 * cos1 * e1 + sin1 * sin1 * e2;
  } else if (q >= qmax) {
    e = ecoal_low_list(dgr, dkl, radii, vr1);
  } else {
    e = 0.999;
  }
  return fmax2(fmin2(one, e), eps);
}

// ECOALMASSM as BREAKINIT_KS fills it (:8761-8772).  own_bin: D10.
template <class R>
void ecoalmassm_table(const R* xl, const R* dropradii, const R* vr1, int own_bin, double* out) {
  const double pi = 3.1415927, rho = 1.0;
  double akpi = 6.0 / pi;
  double radii[FSBM_NKR + 1], v[FSBM_NKR + 1], xd[FSBM_NKR + 1];
  for (int k = 1; k <= FSBM_NKR; ++k) {
    xd[k] = (double)xl[k - 1];
    v[k] = (double)vr1[k - 1];
    if (!own_bin) radii[k] = (double)dropradii[k - 1];
    else radii[k] = ::pow(akpi * xd[k] / rho, 1.0 / 3.0) / 2.0;
  }
  for (int j = 1; j <= FSBM_NKR; ++j)
    for (int i = 1; i <= FSBM_NKR; ++i) {
      double deta = ::pow(akpi * xd[i] / rho, 1.0 / 3.0);
      double dksi = ::pow(akpi * xd[j] / rho, 1.0 / 3.0);
      T2(out, i, j, FSBM_NKR) = ecoal_diam(deta, dksi, radii, v);
    }
  for (int j = 25; j <= 31; ++j) T2(out, FSBM_NKR, j, FSBM_NKR) = 0.1e-29;
}

// Spontanous_Init (:120-189)
template <class R>
void spontanous_init(const R* xl, const R* dropradii, double* prob, double* gain_var_new, int& ikr) {
  const int nkr = FSBM_NKR;
  double xl_dp[FSBM_NKR + 1], r_dp[FSBM_NKR + 1];
  static double gain_var[(FSBM_NKR + 1) * (FSBM_NKR + 1)];
  for (int k = 1; k <= nkr; ++k) {
    xl_dp[k] = (double)xl[k - 1];
    r_dp[k] = (double)dropradii[k - 1];
  }
  ikr = 1;
  for (int kr = 1; kr <= nkr; ++kr) {
    ikr = kr;
    if (dropradii[kr - 1] >= RL(0.3)) break;
  }
  for (int kr = 1; kr <= nkr; ++kr) prob[kr - 1] = 2.94e-7 * ::exp(34.0 * (double)dropradii[kr - 1]);
  for (int k = 0; k < (FSBM_NKR + 1) * (FSBM_NKR + 1); ++k) gain_var[k] = 0.0;
  for (int k = 0; k < nkr * nkr; ++k) gain_var_new[k] = 0.0;
  for (int j = ikr; j <= nkr; ++j)
    for (int i = 1; i <= j - 1; ++i)
      gain_var[j + i * (FSBM_NKR + 1)] =
          (145.37 / xl_dp[i]) * (r_dp[i] / r_dp[j]) * ::exp(-7.0 * r_dp[i] / r_dp[j]);
  double q_m = (double)RL(0.0);
  for (int i = 1; i <= ikr - 1; ++i) q_m = q_m + gain_var[ikr + i * (FSBM_NKR + 1)] * (xl_dp[i] * xl_dp[i]);
  double ratio_new = q_m / xl_dp[ikr];
  for (int j = ikr; j <= nkr; ++j)
    for (int i = 1; i <= j - 1; ++i)
      T2(gain_var_new, j, i, nkr) = gain_var[j + i * (FSBM_NKR + 1)] / ratio_new;
}

// The derived tables, in WRF's kinds.
template <class R> struct FsbmDerivedHost {
  double xl_mg[FSBM_NKR], xs_mg[FSBM_NKR], xg_mg[FSBM_NKR], xh_mg[FSBM_NKR];
  int ima[FSBM_NKR * FSBM_NKR];
  double chucm[FSBM_NKR * FSBM_NKR];
  double ecoal_wrf[FSBM_NKR * FSBM_NKR], ecoal_own[FSBM_NKR * FSBM_NKR];
  double prob[FSBM_NKR], gain_var_new[FSBM_NKR * FSBM_NKR];
  R dropradii[FSBM_NKR];
  double fccnr_con[FSBM_NKR_AERO], fccnr_mar[FSBM_NKR_AERO];
  R xccn[FSBM_NKR_AERO], rccn[FSBM_NKR_AERO];
  int ikr_spon_break;
};

template <class R>
void fsbm_derive(const R* xl, const R* xs, const R* xg, const R* xh, const R* vr1,
                 FsbmDerivedHost<R>& d) {
  for (int i = 0; i < FSBM_NKR; ++i) {
    d.xl_mg[i] = (double)(xl[i] * RL(1.e3));
    d.xs_mg[i] = (double)(xs[i] * RL(1.e3));
    d.xg_mg[i] = (double)(xg[i] * RL(1.e3));
    d.xh_mg[i] = (double)(xh[i] * RL(1.e3));
  }
  for (int k = 0; k < FSBM_NKR * FSBM_NKR; ++k) {
    d.chucm[k] = 0.0;
    d.ima[k] = 0;
  }
  courant_bott_ks<R>(xl, d.chucm, d.ima, 1.0);
  R deg01 = RL(1.) / RL(3.);
  for (int kr = 0; kr < FSBM_NKR; ++kr)
    d.dropradii[kr] = fx_pow(RL(3.0) * xl[kr] / RL(4.0) / RL(3.141593) / RL(1.0), deg01);
  for (int k = 0; k < FSBM_NKR_AERO; ++k) {
    d.fccnr_con[k] = 0.0;
    d.fccnr_mar[k] = 0.0;
  }
  lognormal_modes_aerosol<R>(d.fccnr_mar, d.xccn, d.rccn, xl, FSBM_COL, RL(1.79), RL(1.0), 1);
  lognormal_modes_aerosol<R>(d.fccnr_con, d.xccn, d.rccn, xl, FSBM_COL, RL(1.79), RL(1.0), 2);
  ecoalmassm_table<R>(xl, d.dropradii, vr1, 0, d.ecoal_wrf);
  ecoalmassm_table<R>(xl, d.dropradii, vr1, 1, d.ecoal_own);
  spontanous_init<R>(xl, d.dropradii, d.prob, d.gain_var_new, d.ikr_spon_break);
}

#endif  // !FSBM_DEVICE
#endif  // WOOF_FAST_SBM_INIT_CUH
