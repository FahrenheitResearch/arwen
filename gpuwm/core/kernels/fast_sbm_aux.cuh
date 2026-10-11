// WRF mp_physics = 30: module_mp_SBM_Auxiliary (:1694-3010), the
// diffusional growth core: POLYSVP, JERRATE_KS, JERTIMESC_KS,
// JERSUPSAT_KS, JERDFUN_KS, JERNEWF_KS, JERDFUN_NEW_KS.  Relaxation_Time
// (:2836-3008) is imported by FAST_SBM and never called: not ported.
// Arrays are 1-based; (NKR,ID) arrays are a[kr + (ice-1)*NKR] with a[0]
// unused, i.e. pointer + 1 offset (see the A2 macro).

#ifndef WOOF_FAST_SBM_AUX_CUH
#define WOOF_FAST_SBM_AUX_CUH

#define A2(a, kr, ice) ((a)[(kr) + ((ice) - 1) * FSBM_NKR])

// POLYSVP (:1716-1742): saturation vapour pressure, dyn/cm2.  TT is default
// REAL and the constants are real(4) PARAMETERs.
template <class R> FHD double polysvp(R tt, int itype) {
  R aa1 = RL(2.53E12), bb1 = RL(5.42E3), aa2 = RL(3.41E13), bb2 = RL(6.13E3);
  if (itype == 0) {
    R es1n = aa1 * fx_exp(-bb1 / tt);
    return (double)es1n;
  }
  R es2n = aa2 * fx_exp(-bb2 / tt);
  return (double)es2n;
}

// JERRATE_KS (:1744-1921).  xls, vxl (NKR,ID) double; riec, ro1bl (NKR,ID) R;
// b11 (NKR,ID) R out; fl1 (NKR) R.  All 1-based with the A2 layout.
template <class R>
FHD_NOINLINE void jerrate_ks(const double* xls, double tp, double pp, const double* vxl,
                             const R* riec, const R* ro1bl, R* b11_my, int id, int in,
                             const R* fl1) {
  const double rv_my = 461.5e4, cf_my = 2.4e3, d_myin = 0.211;
  const double ventpl_max = 5.0;
  int nskin[FSBM_NKR + 1];
  R ventplm[FSBM_NKR + 1];
  R al1_my[3], esat1[3];
  double pzero = 1.013e6, tzero = 273.15;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    if ((in == 2 && fl1[kr] == (R)0) || in == 6 || (in == 3 && tp < (double)RL(273.15)))
      nskin[kr] = 2;
    else
      nskin[kr] = 1;
  }
  al1_my[1] = (R)2.5e10;
  al1_my[2] = (R)2.834e10;
  R deg01 = RL(1.0) / RL(3.0);
  R deg03 = RL(1.0) / RL(3.0);
  double cnst = 12.566372;
  double d_my = d_myin * (pzero / pp) * fx_pow(tp / tzero, 1.94);
  double coeff_viscous = 0.13;
  double shmidt_number = coeff_viscous / d_my;
  double a = 2.0 * fx_pow(3.0 / 4.0 / 3.141593, (double)deg01);
  double b = a / coeff_viscous;
  double rvt = rv_my * tp;
  R tpreal = (R)tp;
  esat1[1] = (R)polysvp<R>(tpreal, 0);
  esat1[2] = (R)polysvp<R>(tpreal, 1);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) ventplm[kr] = (R)0.0;
  double shmidt_number03 = fx_pow(shmidt_number, (double)deg03);
  for (int ice = 1; ice <= id; ++ice) {
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      double xls_kr_ice = A2(xls, kr, ice);
      double ro1bl_kr_ice = (double)A2(ro1bl, kr, ice);
      double vxl_kr_ice = A2(vxl, kr, ice);
      double reinolds_number = b * vxl_kr_ice * fx_pow(xls_kr_ice / ro1bl_kr_ice, (double)deg03);
      double reshm = fx_sqrt(reinolds_number) * shmidt_number03;
      double ventpl;
      if (reinolds_number < 2.5) {
        ventpl = 1.0 + 0.108 * reshm * reshm;
        ventplm[kr] = (R)ventpl;
      } else {
        ventpl = 0.78 + 0.308 * reshm;
        ventplm[kr] = (R)ventpl;
      }
    }
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      double ventpl = (double)ventplm[kr];
      if (ventpl > ventpl_max) {
        ventpl = ventpl_max;
        ventplm[kr] = (R)ventpl;
      }
      double constl = cnst * (double)A2(riec, kr, ice);
      R fd1 = (R)(rvt / d_my / (double)esat1[nskin[kr]]);
      R fk1 = (R)(((double)al1_my[nskin[kr]] / rvt - 1.0) * (double)al1_my[nskin[kr]] / cf_my / tp);
      R xl_my1 = (R)(ventpl * constl);
      double detl = (double)(fk1 + fd1);
      A2(b11_my, kr, ice) = (R)((double)xl_my1 / detl);
    }
  }
}

// JERTIMESC_KS (:1925-1964).  fi1, b11 (NKR,ID) R; x1 (NKR,ID) double;
// sfn11 (ID) R, 1-based.
template <class R>
FHD void jertimesc_ks(const R* fi1, const double* x1, R* sfn11, const R* b11_my, R cf, int id,
                      R col) {
  for (int ice = 1; ice <= id; ++ice) {
    R sfn11s = (R)0.0;
    sfn11[ice] = cf * sfn11s;
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      R fk = A2(fi1, kr, ice);
      R delm = (R)(A2(x1, kr, ice) * 3.0 * (double)col);
      R fun = fk * delm;
      R b11 = A2(b11_my, kr, ice);
      sfn11s = sfn11s + fun * b11;
    }
    sfn11[ice] = cf * sfn11s;
  }
}

// JERSUPSAT_KS (:1966-2330).  The time-integrated supersaturations.  isym2
// is 1-based (3).  supsat_rederived (S1, S2) replaces WRF's small-ratio
// expansions and the branch-7 GAMA clamp with the rederived forms.
FHD double fsbm_expm1_series(double x) {
  return x + x * x / 2.0 + x * x * x / 6.0 + x * x * x * x / 24.0 + x * x * x * x * x / 120.0;
}

template <class R>
FHD_NOINLINE int jersupsat_ks(double del1, double del2, double& del1n, double& del2n, double& rw,
                              double& pw, double& ri, double& pi, R dt, double& del1int,
                              double& del2int, R dyn1, R dyn2, int& isym1, int* isym2,
                              int& isym3, int& isym4, int& isym5, int rederived) {
  const double rw_pw_min = 1.0e-10, rw_pw_ri_pi_min = 1.0e-10, ratio_icew_min = 1.0e-4;
  int isymice = isym2[1] + isym2[2] + isym2[3] + isym3 + isym4 + isym5;
  int irw = 1, ipw = 1, iri = 1, ipi = 1;
  (void)ipw;
  (void)ipi;
  double mx = fmax2(fmax2(rw, pw), fmax2(ri, pi));
  if (mx <= rw_pw_ri_pi_min) {
    rw = (double)RL(0.0); irw = 0;
    pw = (double)RL(0.0); ipw = 0;
    ri = (double)RL(0.0); iri = 0;
    pi = (double)RL(0.0); ipi = 0;
    isym1 = 0;
    isymice = 0;
  } else {
    if (fmax2(rw, pw) > rw_pw_min) {
      if (rw < rw_pw_min * rw_pw_min) { rw = 1.0e-20; irw = 0; }
      if (pw < rw_pw_min * rw_pw_min) { pw = 1.0e-20; ipw = 0; }
      if (fmax2(pi / pw, ri / rw) <= ratio_icew_min) {
        ri = (double)RL(0.0); iri = 0;
        pi = (double)RL(0.0); ipi = 0;
        isymice = 0;
      }
      if (fmin2(pi / pw, ri / rw) > 1.0 / ratio_icew_min) {
        rw = (double)RL(0.0); irw = 0;
        pw = (double)RL(0.0); ipw = 0;
        isym1 = 0;
      }
    } else {
      rw = (double)RL(0.0); irw = 0;
      pw = (double)RL(0.0); ipw = 0;
      isym1 = 0;
    }
  }
  if (isymice == 0) {
    isym2[1] = isym2[2] = isym2[3] = 0;
    isym3 = 0;
    isym4 = 0;
    isym5 = 0;
  }
  if (irw == 0 && iri == 0) {
    del1n = del1 + (double)(dyn1 * dt);
    del2n = del2 + (double)(dyn2 * dt);
    del1int = del1 * (double)dt + (double)(dyn1 * dt * dt) / 2.0;
    del2int = del2 * (double)dt + (double)(dyn2 * dt * dt) / 2.0;
    return FSBM_OK;
  }
  const double DT = (double)dt, DYN1 = (double)dyn1, DYN2 = (double)dyn2;
  if (iri == 0) {
    // only water
    double expr = fx_exp(-rw * DT);
    if (fabs_(rw * DT) > (double)RL(1.0E-6)) {
      del1n = del1 * expr + (DYN1 / rw) * (1.0 - expr);
      del2n = pw * del1 * expr / rw - pw * DYN1 * DT / rw - pw * DYN1 * expr / (rw * rw) +
              (double)(dyn2 * dt) + del2 - pw * del1 / rw + pw * DYN1 / (rw * rw);
      del1int = -del1 * expr / rw + DYN1 * DT / rw + DYN1 * expr / (rw * rw) + del1 / rw -
                DYN1 / (rw * rw);
      del2int = pw * del1 * expr / (-rw * rw) - pw * DYN1 * DT * DT / (2.0 * rw) +
                pw * DYN1 * expr / (rw * rw * rw) + (double)(dyn2 * dt * dt) / 2.0 + del2 * DT -
                pw * del1 * DT / rw + pw * DYN1 * DT / (rw * rw) + pw * del1 / (rw * rw) -
                pw * DYN1 / (rw * rw * rw);
      return FSBM_OK;
    }
    expr = fsbm_expm1_series(-rw * DT);
    del1n = del1 + del1 * expr + (DYN1 / rw) * (0.0 - expr);
    del2n = pw * del1 * expr / rw - pw * DYN1 * DT / rw - pw * DYN1 * expr / (rw * rw) +
            (double)(dyn2 * dt) + del2;
    del1int = -del1 * expr / rw + DYN1 * DT / rw + DYN1 * expr / (rw * rw);
    del2int = pw * del1 * expr / (-rw * rw) - pw * DYN1 * DT * DT / (2.0 * rw) +
              pw * DYN1 * expr / (rw * rw * rw) + (double)(dyn2 * dt * dt) / 2.0 + del2 * DT -
              pw * del1 * DT / rw + pw * DYN1 * DT / (rw * rw);
    return FSBM_OK;
  }
  if (irw == 0) {
    // only ice
    double expp = fx_exp(-pi * DT);
    if (fabs_(pi * DT) > (double)RL(1.0E-6)) {
      del2n = del2 * expp + (DYN2 / pi) * (1.0 - expp);
      del2int = -del2 * expp / pi + DYN2 * DT / pi + DYN2 * expp / (pi * pi) + del2 / pi -
                DYN2 / (pi * pi);
      del1n = +ri * del2 * expp / pi - ri * DYN2 * DT / pi - ri * DYN2 * expp / (pi * pi) +
              (double)(dyn1 * dt) + del1 - ri * del2 / pi + ri * DYN2 / (pi * pi);
      del1int = -ri * del2 * expp / (pi * pi) - ri * DYN2 * DT * DT / (2.0 * pi) +
                ri * DYN2 * expp / (pi * pi * pi) + (double)(dyn1 * dt * dt) / 2.0 + del1 * DT -
                ri * del2 * DT / pi + ri * DYN2 * DT / (pi * pi) + ri * del2 / (pi * pi) -
                ri * DYN2 / (pi * pi * pi);
      return FSBM_OK;
    }
    expp = fsbm_expm1_series(-pi * DT);
    del2n = del2 + del2 * expp - expp * DYN2 / pi;
    del2int = -del2 * expp / pi + DYN2 * DT / pi + DYN2 * expp / (pi * pi);
    del1n = +ri * del2 * expp / pi - ri * DYN2 * DT / pi - ri * DYN2 * expp / (pi * pi) +
            (double)(dyn1 * dt) + del1;
    del1int = -ri * del2 * expp / (pi * pi) - ri * DYN2 * DT * DT / (2.0 * pi) +
              ri * DYN2 * expp / (pi * pi * pi) + (double)(dyn1 * dt * dt) / 2.0 + del1 * DT -
              ri * del2 * DT / pi + ri * DYN2 * DT / (pi * pi);
    return FSBM_OK;
  }
  if (irw == 1 && iri == 1) {
    double a = (rw - pi) * (rw - pi) + (double)RL(4.0E0) * pw * ri;
    if (a < 0.0) return 2166;
    double alfa = fx_sqrt((rw - pi) * (rw - pi) + 4.0 * pw * ri);
    double beta = 0.5 * (alfa + rw + pi);
    double gama = 0.5 * (alfa - rw - pi);
    double g31 = pi * DYN1 - ri * DYN2;
    double g32 = -pw * DYN1 + rw * DYN2;
    double g2 = rw * pi - ri * pw;
    if (g2 < 1.0e-20) g2 = 1.0004e-11 * 1.0003e-11 - 1.0002e-11 * (double)RL(1.0001e-11);
    double expb = fx_exp(-beta * DT);
    double expg = fx_exp(gama * DT);
    if (fabs_(gama * DT) > (double)RL(1.0E-6)) {
      double c11 = (beta * del1 - rw * del1 - ri * del2 - beta * g31 / g2 + DYN1) / alfa;
      double c21 = (gama * del1 + rw * del1 + ri * del2 - gama * g31 / g2 - DYN1) / alfa;
      double c12 = (beta * del2 - pw * del1 - pi * del2 - beta * g32 / g2 + DYN2) / alfa;
      double c22 = (gama * del2 + pw * del1 + pi * del2 - gama * g32 / g2 - DYN2) / alfa;
      del1n = c11 * expg + c21 * expb + g31 / g2;
      del1int = c11 * expg / gama - c21 * expb / beta + (c21 / beta - c11 / gama) + g31 * DT / g2;
      del2n = c12 * expg + c22 * expb + g32 / g2;
      del2int = c12 * expg / gama - c22 * expb / beta + (c22 / beta - c12 / gama) + g32 * DT / g2;
      return FSBM_OK;
    }
    if (fabs_(ri / rw) > (double)RL(1.0E-12)) {
      if (fabs_(rw / ri) > (double)RL(1.0E-12)) {
        // branch 7
        alfa = fx_sqrt((rw - pi) * (rw - pi) + 4.0 * pw * ri);
        beta = 0.5 * (alfa + rw + pi);
        gama = 0.5 * (alfa - rw - pi);
        if (!rederived) {
          if (gama < (double)RL(0.5) * 2.0e-10) gama = 0.5 * (2.002e-10 - 2.001e-10);  // S2
        }
        expg = fsbm_expm1_series(gama * DT);
        expb = fx_exp(-beta * DT);
        double c11 = (beta * del1 - rw * del1 - ri * del2 + DYN1) / alfa;
        double c21 = (gama * del1 + rw * del1 + ri * del2 - gama * g31 / g2 - DYN1) / alfa;
        double c12 = (beta * del2 - pw * del1 - pi * del2 + DYN2) / alfa;
        double c22 = (gama * del2 + pw * del1 + pi * del2 - gama * g32 / g2 - DYN2) / alfa;
        // EXPG/GAMA: the rederived form keeps GAMA's sign and takes the limit
        // DT at GAMA = 0, where WRF's clamp kept it away from zero.
        double expg_over_gama = (gama != 0.0) ? expg / gama : DT;
        double a1del1n = c11;
        double a2del1n = c11 * expg;
        double a3del1n = c21 * expb;
        double a4del1n = g31 / g2 * (gama / alfa + (gama / alfa - 1.0) * expg);
        del1n = a1del1n + a2del1n + a3del1n + a4del1n;
        double a1del1int = rederived ? c11 * expg_over_gama : c11 * expg / gama;
        double a2del1int = -c21 * expb / beta;
        double a3del1int = c21 / beta;
        double a4del1int = g31 / g2 * DT * (gama / alfa);
        del1int = a1del1int + a2del1int + a3del1int + a4del1int;
        double a1del2n = c12;
        double a2del2n = c12 * expg;
        double a3del2n = c22 * expb;
        double a4del2n = g32 / g2 * (gama / alfa + (gama / alfa - 1.0) * (gama * DT + gama * gama * DT * DT / 2.0));
        del2n = a1del2n + a2del2n + a3del2n + a4del2n;
        double a1del2int = rederived ? c12 * expg_over_gama : c12 * expg / gama;
        double a2del2int = -c22 * expb / beta;
        double a3del2int = c22 / beta;
        double a4del2int = g32 / g2 * DT * (gama / alfa);
        double a5del2int = g32 / g2 * (gama / alfa - 1.0) * (gama * DT * DT / 2.0);
        del2int = a1del2int + a2del2int + a3del2int + a4del2int + a5del2int;
      } else {
        // branch 8: |RW/RI| <= 1e-12
        double x = -2.0 * rw * pi + rw * rw + 4.0 * pw * ri;
        if (!rederived) {
          alfa = pi * (1 + (x / pi) / 2.0 - (x / pi) * (x / pi) / 8.0);
          beta = pi + (x / pi) / 4.0 - (x / pi) * (x / pi) / 16.0 + rw / 2.0;
          gama = (x / pi) / 4.0 - (x / pi) * (x / pi) / 16.0 - rw / 2.0;
        } else {
          // S1: sqrt(PI**2 + X) = PI (1 + e/2 - e**2/8), e = X/PI**2
          double e = x / pi / pi;
          alfa = pi * (1 + e / 2.0 - e * e / 8.0);
          beta = pi + (x / pi) / 4.0 - (x / pi) * (x / pi) / (16.0 * pi) + rw / 2.0;
          gama = (x / pi) / 4.0 - (x / pi) * (x / pi) / (16.0 * pi) - rw / 2.0;
        }
        expg = fsbm_expm1_series(gama * DT);
        expb = fx_exp(-beta * DT);
        double c11 = (beta * del1 - rw * del1 - ri * del2 + DYN1) / alfa;
        double c21 = (gama * del1 + rw * del1 + ri * del2 - gama * g31 / g2 - DYN1) / alfa;
        double c12 = (beta * del2 - pw * del1 - pi * del2 + DYN2) / alfa;
        double c22 = (gama * del2 + pw * del1 + pi * del2 - gama * g32 / g2 - DYN2) / alfa;
        del1n = c11 + c11 * expg + c21 * expb + g31 / g2 * (gama / alfa + (gama / alfa - 1) * expg);
        del1int = c11 * expg / gama - c21 * expb / beta + (c21 / beta) + g31 / g2 * DT * (gama / alfa);
        del2n = c12 + c12 * expg + c22 * expb +
                g32 / g2 * (gama / alfa + (gama / alfa - 1.0) * (gama * DT + gama * gama * DT * DT / 2.0));
        del2int = c12 * expg / gama - c22 * expb / beta + (c22 / beta) + g32 / g2 * DT * (gama / alfa) +
                  g32 / g2 * (gama / alfa - 1.0) * (gama * DT * DT / 2.0);
      }
    } else {
      // branch 9: |RI/RW| <= 1e-12
      double x = -2.0 * rw * pi + pi * pi + 4.0 * pw * ri;
      if (!rederived) {
        alfa = rw * (1.0 + (x / rw) / 2.0 - (x / rw) * (x / rw) / 8.0);
        beta = rw + (x / rw) / 4.0 - (x / rw) * (x / rw) / 16.0 + pi / 2.0;
        gama = (x / rw) / 4.0 - (x / rw) * (x / rw) / 16.0 - pi / 2.0;
      } else {
        double e = x / rw / rw;
        alfa = rw * (1.0 + e / 2.0 - e * e / 8.0);
        beta = rw + (x / rw) / 4.0 - (x / rw) * (x / rw) / (16.0 * rw) + pi / 2.0;
        gama = (x / rw) / 4.0 - (x / rw) * (x / rw) / (16.0 * rw) - pi / 2.0;
      }
      expg = fsbm_expm1_series(gama * DT);
      expb = fx_exp(-beta * DT);
      double c11 = (beta * del1 - rw * del1 - ri * del2 + DYN1) / alfa;
      double c21 = (gama * del1 + rw * del1 + ri * del2 - gama * g31 / g2 - DYN1) / alfa;
      double c12 = (beta * del2 - pw * del1 - pi * del2 + DYN2) / alfa;
      double c22 = (gama * del2 + pw * del1 + pi * del2 - gama * g32 / g2 - DYN2) / alfa;
      del1n = c11 + c11 * expg + c21 * expb + g31 / g2 * (gama / alfa + (gama / alfa - 1.0) * expg);
      del1int = c11 * expg / gama - c21 * expb / beta + (c21 / beta) + g31 / g2 * DT * (gama / alfa);
      del2n = c12 + c12 * expg + c22 * expb +
              g32 / g2 * (gama / alfa + (gama / alfa - 1.0) * (gama * DT + gama * gama * DT * DT / 2.0));
      del2int = c12 * expg / gama - c22 * expb / beta + c22 / beta + g32 / g2 * DT * (gama / alfa) +
                g32 / g2 * (gama / alfa - 1.0) * (gama * DT * DT / 2.0);
    }
  }
  return FSBM_OK;
}

// JERNEWF_KS (:2444-2757).  rr, fi, psi, rn double, 1-based, length nrx.
template <class R>
FHD_NOINLINE int jernewf_ks(int nrx, const double* rr, double* fi, double* psi, double* rn,
                            int i3point, int& idrop, R col_r) {
  const double coeff_remaping = 0.0066667;
  const int krdrop_remaping_min = 6, krdrop_remaping_max = 12;
  double col = (double)col_r;
  double cdrop[FSBM_NKR + 1], delta_cdrop[FSBM_NKR + 1], rrs[FSBM_NKR + 2], psinew[FSBM_NKR + 2];
  int nrxp = nrx + 1;
  int nrx1 = FSBM_NKR;
  for (int i = 1; i <= nrx; ++i) {
    if (rn[i] < 0.0) {
      rn[i] = 1.0e-50;
      fi[i] = 0.0;
    }
  }
  for (int k = 1; k <= nrx; ++k) rrs[k] = rr[k];
  int i3point_condevap = i3point;
  if (rn[1] < rrs[1]) {
    i3point_condevap = 0;
    idrop = 0;
    nrx1 = nrx;
  }
  if (idrop == 0) i3point_condevap = 0;
  for (int k = 1; k <= nrx; ++k) {
    psi[k] = 0.0;
    cdrop[k] = 0.0;
    delta_cdrop[k] = 0.0;
    psinew[k] = 0.0;
  }
  rrs[nrxp] = rrs[nrx] * 1024.0;
  psinew[nrxp] = 0.0;
  int isign_growth = 0;
  for (int k = 1; k <= nrx; ++k)
    if (rn[k] != rr[k]) { isign_growth = 1; break; }
  if (isign_growth == 1) {
    for (int k = 1; k <= nrx1; ++k) {
      if (fi[k] > (double)RL(0.0)) {
        if (fabs_(rn[k] - rr[k]) < 1.0e-16) {
          psinew[k] = fi[k] * rr[k];
          continue;
        }
        int i = 1;
        while (!(rrs[i] <= rn[k] && rrs[i + 1] >= rn[k]) && i < nrx1) i = i + 1;
        if (rn[k] < rrs[1]) {
          double rntmp = rn[k], rrtmp = 0.0, rrp = rrs[1];
          double gmat2 = (rntmp - rrtmp) / (rrp - rrtmp);
          psinew[1] = psinew[1] + fi[k] * rr[k] * gmat2;
        } else {
          double rntmp = rn[k], rrtmp = rrs[i], rrp = rrs[i + 1];
          double gmat2 = (rntmp - rrtmp) / (rrp - rrtmp);
          double gmat = (rrp - rntmp) / (rrp - rrtmp);
          psinew[i] = psinew[i] + fi[k] * rr[k] * gmat;
          psinew[i + 1] = psinew[i + 1] + fi[k] * rr[k] * gmat2;
        }
      }
    }
    for (int kr = 1; kr <= nrx1; ++kr) psi[kr] = psinew[kr];
    for (int kr = nrx1 + 1; kr <= nrx; ++kr) psi[kr] = fi[kr];
    if (i3point_condevap == 1) {
      for (int k = 1; k <= nrx1 - 1; ++k) {
        if (fi[k] > (double)RL(0.0)) {
          if (fabs_(rn[k] - rr[k]) < 1.0e-16) {
            psi[k] = fi[k] * rr[k];
            continue;
          }
          if (rrs[2] < rn[k]) {
            int i = 2;
            while (!(rrs[i] <= rn[k] && rrs[i + 1] >= rn[k]) && i < nrx1 - 1) i = i + 1;
            double rntmp = rn[k], rrtmp = rrs[i], rrp = rrs[i + 1], rrm = rrs[i - 1];
            double rntmp2 = rn[k + 1], rrtmp2 = rrs[i + 1], rrp2 = rrs[i + 2], rrm2 = rrs[i];
            double gn1 = (rrp - rntmp) * (rrtmp - rntmp) / (rrp - rrm) / (rrtmp - rrm);
            double gn1p = (rrp2 - rntmp2) * (rrtmp2 - rntmp2) / (rrp2 - rrm2) / (rrtmp2 - rrm2);
            double gn2 = (rrp - rntmp) * (rntmp - rrm) / (rrp - rrtmp) / (rrtmp - rrm);
            double gmat = (rrp - rntmp) / (rrp - rrtmp);
            double gn3 = (rrtmp - rntmp) * (rrm - rntmp) / (rrp - rrm) / (rrp - rrtmp);
            double gmat2 = (rntmp - rrtmp) / (rrp - rrtmp);
            double psi_im = psi[i - 1] + gn1 * fi[k] * rr[k];
            double psi_i = psi[i] + gn1p * fi[k + 1] * rr[k + 1] + (gn2 - gmat) * fi[k] * rr[k];
            double psi_ip = psi[i + 1] + (gn3 - gmat2) * fi[k] * rr[k];
            if (psi_im > 0.0) {
              if (psi_ip > 0.0) {
                if (i > 2) {
                  if ((psi_im > psi[i - 2] && psi_im < psi_i && psi[i - 2] < psi[i]) ||
                      psi[i - 2] >= psi[i]) {
                    psi[i - 1] = psi_im;
                    psi[i] = psi[i] + fi[k] * rr[k] * (gn2 - gmat);
                    psi[i + 1] = psi_ip;
                  }
                }
              } else {
                break;
              }
            } else {
              break;
            }
          }
        }
      }
    }
    for (int k = 1; k <= nrx1; ++k) psi[k] = psi[k] / rr[k];
    for (int k = nrx1 + 1; k <= nrx; ++k) psi[k] = fi[k];
    if (idrop == 1) {
      for (int k = krdrop_remaping_min; k <= krdrop_remaping_max; ++k)
        cdrop[k] = 3.0 * col * psi[k] * rr[k];
      int kmax = krdrop_remaping_min;
      for (int k = krdrop_remaping_max; k >= krdrop_remaping_min; --k) {
        kmax = k;
        if (psi[k] > 0.0) break;
      }
      for (int k = kmax - 1; k >= krdrop_remaping_min; --k) {
        if (cdrop[k] > 0.0) {
          delta_cdrop[k] = cdrop[k + 1] / cdrop[k];
          if (delta_cdrop[k] < coeff_remaping) {
            cdrop[k] = cdrop[k] + cdrop[k + 1];
            cdrop[k + 1] = 0.0;
          }
        }
      }
      for (int k = krdrop_remaping_min; k <= kmax; ++k) psi[k] = cdrop[k] / (3.0 * col * rr[k]);
    }
  } else {
    for (int k = 1; k <= nrx; ++k) psi[k] = fi[k];
  }
  for (int kr = 1; kr <= nrx; ++kr)
    if (psi[kr] < (double)RL(0.0)) return 2750;
  return FSBM_OK;
}

// JERDFUN_KS (:2334-2442).  xi, xin double 1-based; b21, fi2, psi2 R 1-based.
template <class R>
FHD_NOINLINE int jerdfun_ks(const double* xi, double* xin, const R* b21_my, const R* fi2, R* psi2,
                            R del2n, int isym2, int ind, int itype, int& idrop, R col) {
  double c = 2.0 / 3.0;
  double degree1 = 1.0 / 3.0;
  double degree2 = c;
  double degree3 = 3.0 / 2.0;
  double psi2r[FSBM_NKR + 1], fi2r[FSBM_NKR + 1], xir[FSBM_NKR + 1], xinr[FSBM_NKR + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    psi2r[kr] = (double)fi2[kr];
    fi2r[kr] = (double)fi2[kr];
  }
  if (isym2 == 1) {
    if (ind == 1 && itype == 1) {
      for (int kr = 1; kr <= FSBM_NKR; ++kr) {
        double d = fx_pow(xi[kr], degree1);
        double ratexi = c * (double)del2n * (double)b21_my[kr] / d;
        double b = fx_pow(xi[kr], degree2);
        double a = b + ratexi;
        if (a < 0.0) xin[kr] = 1.0e-50;
        else xin[kr] = fx_pow(a, degree3);
      }
    } else {
      for (int kr = 1; kr <= FSBM_NKR; ++kr) {
        double ratexi = (double)(del2n * b21_my[kr]);
        xin[kr] = xi[kr] + ratexi;
      }
    }
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      xir[kr] = xi[kr];
      xinr[kr] = xin[kr];
      fi2r[kr] = (double)fi2[kr];
    }
    int e = jernewf_ks<R>(FSBM_NKR, xir, fi2r, psi2r, xinr, 1, idrop, col);
    if (e) return e;
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      if (psi2r[kr] < 0.0) return 2428;
      psi2[kr] = (R)psi2r[kr];
    }
  }
  return FSBM_OK;
}

// JERDFUN_NEW_KS (:2759-2834)
template <class R>
FHD_NOINLINE int jerdfun_new_ks(const double* xi, const double* xin, const R* b21_my, const R* fi2,
                                R* psi2, int& idrop, R col) {
  double c = 2.0 / 3.0;
  double degree1 = c / 2.0;
  double degree2 = c;
  double degree3 = 3.0 / 2.0;
  double xir[FSBM_NKR + 1], fi2r[FSBM_NKR + 1], psi2r[FSBM_NKR + 1], xinr[FSBM_NKR + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    xir[kr] = xi[kr];
    fi2r[kr] = (double)fi2[kr];
    psi2r[kr] = (double)psi2[kr];
    xinr[kr] = xin[kr];
  }
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    double d = fx_pow(xir[kr], degree1);
    double ratexi = c * (double)b21_my[kr] / d;
    double b = fx_pow(xir[kr], degree2);
    double a = b + ratexi;
    if (a < 0.0) xinr[kr] = 1.0e-50;
    else xinr[kr] = fx_pow(a, degree3);
  }
  int e = jernewf_ks<R>(FSBM_NKR, xir, fi2r, psi2r, xinr, 1, idrop, col);
  if (e) return e;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) psi2[kr] = (R)psi2r[kr];
  return FSBM_OK;
}

#endif  // WOOF_FAST_SBM_AUX_CUH
