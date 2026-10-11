// WRF mp_physics = 30: module_mp_SBM_Nucleation (:3013-3840): drop
// activation (WATER_NUCLEATION, Cloud_Base_Super, supmax_COEFF), Meyers
// ice nucleation (ICE_NUCL) and their driver JERNUCL01_KS.
// LogNormal_modes_Aerosol is an init routine (fast_sbm_init.cuh).
// Identifiers are lowercase: the device preamble defines G, CP, RD, ... as
// macros.

#ifndef WOOF_FAST_SBM_NUCL_CUH
#define WOOF_FAST_SBM_NUCL_CUH

// supmax_COEFF (:3572-3733).  mwaero, ro_solute are default REAL.
template <class R>
FHD void supmax_coeff(double& akoe, double& bkoe, double& c3, double pp, double tt, R ro_solute,
                      int ions, R mwaero) {
  const double rv_my = 461.5e4, cp_my = 1005.0e4, g_my = 9.8e2, rd_my = 287.0e4, pi_c = 3.141593;
  double tpc = tt - 273.15;
  double pzero = 1.01325e6;
  double tzero = 273.15;
  double alw1 = -6.143419998e-2 * fx_pow(tpc, 3.0) + 1.58927 * fx_pow(tpc, 2.0) - 2.36418e3 * tpc +
                2.50079e6;
  alw1 = alw1 * 10.0e3;
  double sw;
  if (tpc < -5.5) {
    sw = 5.285e-11 * fx_pow(tpc, 6.0) + 6.283e-9 * fx_pow(tpc, 5.0) + 2.933e-7 * fx_pow(tpc, 4.0) +
         6.511e-6 * fx_pow(tpc, 3.0) + 6.818e-5 * fx_pow(tpc, 2.0) + 1.15e-4 * tpc + 7.593e-2;
  } else {
    sw = -1.55e-4 * tpc + 7.566165e-2;
  }
  sw = sw * 10.0e2;
  double ro_w;
  if (tpc < 0.0) {
    ro_w = -7.497e-9 * fx_pow(tpc, 6.0) - 3.6449e-7 * fx_pow(tpc, 5.0) -
           6.9987e-6 * fx_pow(tpc, 4.0) + 1.518e-4 * fx_pow(tpc, 3.0) -
           8.486e-3 * fx_pow(tpc, 2.0) + 6.69e-2 * tpc + 9.9986e2;
  } else {
    ro_w = (-3.932952e-10 * fx_pow(tpc, 5.0) + 1.497562e-7 * fx_pow(tpc, 4.0) -
            5.544846e-5 * fx_pow(tpc, 3.0) - 7.92221e-3 * fx_pow(tpc, 2.0) + 1.8224944e1 * tpc +
            9.998396e2) /
           (1.0 + 1.8159725e-2 * tpc);
  }
  ro_w = ro_w * 1.0e-3;
  double hc = 7.1128e-5 * tpc + 2.380696e-2;
  hc = hc * 10.0e4;
  double ew = 6.38780966e-9 * fx_pow(tpc, 6.0) + 2.03886313e-6 * fx_pow(tpc, 5.0) +
              3.02246994e-4 * fx_pow(tpc, 4.0) + 2.65027242e-2 * fx_pow(tpc, 3.0) +
              1.43053301 * fx_pow(tpc, 2.0) + 4.43986062e1 * tpc + 6.1117675e2;
  ew = ew * 10.0;
  akoe = 2.0 * sw / (rv_my * ro_w * (tpc + tzero));
  bkoe = (double)((R)ions * RL(4.3) / mwaero);
  bkoe = bkoe * (4.0 / 3.0) * pi_c * (double)ro_solute;
  double ro_v = ew / (rv_my * (tpc + tzero));
  double dv = 0.211 * (pzero / pp) * fx_pow((tpc + tzero) / tzero, 1.94);
  double ro_a = pzero / ((tpc + tzero) * rd_my);
  double fl = (ro_w * fx_pow(alw1, 2.0)) / (hc * rv_my * fx_pow(tpc + tzero, 2.0));
  double fr = ro_w * rv_my * (tpc + tzero) / (ew * dv);
  double f = fl + fr;
  double qv = ro_v / ro_a;
  double a1 = (g_my * alw1 / (cp_my * rv_my * fx_pow(tpc + tzero, 2.0))) - (g_my / (rd_my * (tpc + tzero)));
  double a2 = (1.0 / qv) + (fx_pow(alw1, 2.0)) / (cp_my * rv_my * fx_pow(tpc + tzero, 2.0));
  double c1 = 1.058;
  c3 = c1 * fx_pow(f * a1 / 3.0, 0.75) * fx_sqrt(3.0 * ro_a / (4.0 * pi_c * ro_w * a2));
}

// Cloud_Base_Super (:3479-3570).  fccnr, rccn double, 1-based (nkr).
template <class R>
FHD int cloud_base_super(const double* fccnr, const double* rccn, double tt, double pp, double wbase,
                         int nkr, double& rcriti, R ro_solute, int ions, R mwaero, double col) {
  double akoe, bkoe, c3;
  supmax_coeff<R>(akoe, bkoe, c3, pp, tt, ro_solute, ions, mwaero);
  double pr = c3 * fx_pow(wbase, 0.75);
  double pl[FSBM_NKR_AERO + 1];
  for (int n = 1; n <= nkr; ++n) pl[n] = (double)RL(0.0);
  int nn = -1;
  for (int nr = 2; nr <= nkr; ++nr) {
    double supmax = fx_sqrt((4.0 * fx_pow(akoe, 3.0)) / (27.0 * fx_pow(rccn[nr], 3.0) * bkoe));
    double ccnconact = 0.0;
    for (int kr = nr; kr <= nkr; ++kr) ccnconact = ccnconact + col * fccnr[kr];
    pl[nr] = supmax * (fx_sqrt(ccnconact));
    if (pl[nr] <= pr) {
      nn = nr;
      break;
    }
  }
  if (nn == -1) return 3548;
  double dl1 = fabs_(pl[nn - 1] - pr);
  double dl2 = fabs_(pl[nn] - pr);
  rcriti = rccn[nn - 1] * fx_exp(col * dl1 / (dl1 + dl2));
  return FSBM_OK;
}

// WATER_NUCLEATION (:3194-3330).  nkr is NKR_aerosol (43) as JERNUCL01_KS
// passes it.  psi1, xl, dropconcn have the drop grid (33); fccnr, rccn the
// aerosol grid.  The RMASSL_NUCL it forms is discarded by WRF (D3) and is
// not formed here.
template <class R>
FHD_NOINLINE int water_nucleation(double col, int nkr, double* psi1, double* fccnr, const double* xl,
                                  double tt, double qq, double ror, double sup1, double* dropconcn,
                                  double pp, int is_this_cloudbase, double win, R ro_solute,
                                  const double* rccn, int ions, R mwaero) {
  (void)qq;
  (void)ror;
  double ccnconc[FSBM_NKR_AERO + 1];
  for (int k = 1; k <= FSBM_NKR; ++k) dropconcn[k] = 0.0;
  double deg01 = 1.0 / 3.0;
  int imax = nkr;
  for (int i = imax; i >= 1; --i) {
    if (fccnr[i] > 0.0) {
      imax = i;
      break;
    }
  }
  int ncriti = 0;
  while (imax >= ncriti) {
    for (int k = 1; k <= nkr; ++k) ccnconc[k] = (double)RL(0.0);
    double akoe = 3.3e-05 / tt;
    double bkoe = (double)((R)ions * RL(4.3) / mwaero);
    bkoe = bkoe * (4.0 / 3.0) * 3.141593 * (double)ro_solute;
    double rcriti;
    if (is_this_cloudbase == 1) {
      int e = cloud_base_super<R>(fccnr, rccn, tt, pp, win, nkr, rcriti, ro_solute, ions, mwaero, col);
      if (e) return e;
    } else {
      rcriti = (akoe / 3.0) * fx_pow(4.0 / bkoe / sup1 / sup1, deg01);
    }
    if (rcriti >= rccn[imax]) break;
    ncriti = imax;
    while (rcriti <= rccn[ncriti] && ncriti > 1) ncriti = ncriti - 1;
    double rccn_minimum = rccn[1] / 10000.0;
    if (ncriti == imax - 1) {
      if (ncriti > 1) {
        double dln1 = fx_log(rcriti) - fx_log(rccn[imax - 1]);
        double dln2 = col - dln1;
        ccnconc[imax] = dln2 * fccnr[imax];
        fccnr[imax] = fccnr[imax] * dln1 / col;
      } else {
        double dln1 = fx_log(rcriti) - fx_log(rccn_minimum);
        double dln2 = fx_log(rccn[1]) - fx_log(rcriti);
        ccnconc[imax] = dln2 * fccnr[imax];
        fccnr[imax] = fccnr[imax] * dln1 / (dln1 + dln2);
      }
    } else {
      ccnconc[imax] = col * fccnr[imax];
      fccnr[imax] = 0.0;
    }
    if (imax <= nkr - 7) {
      dropconcn[1] = dropconcn[1] + ccnconc[imax];
    } else {
      int b = 8 - (nkr - imax);
      dropconcn[b] = dropconcn[b] + ccnconc[imax];
    }
    imax = imax - 1;
  }
  double s = 0.0;
  for (int k = 1; k <= FSBM_NKR; ++k) s = s + dropconcn[k];
  if (s > (double)RL(0.0)) {
    for (int kr = 1; kr <= 8; ++kr) {
      double dx = 3.0 * col * xl[kr];
      psi1[kr] = psi1[kr] + dropconcn[kr] / dx;
    }
  }
  return FSBM_OK;
}

// ICE_NUCL (:3340-3475), Meyers et al. (1992).  psi2, xi (NKR,ICEMAX)
// double in the A2 layout.
FHD_NOINLINE int ice_nucl(double* psi2, const double* xi, double sup2, double tt, double dsupice_xyz,
                          double& sup2_old, double col) {
  const double a1 = -0.639, b1 = 0.1296, a2 = -2.8, b2 = 0.262;
  const double temp1 = -5.0, temp2 = -2.0, temp3 = -20.0;
  const double delsupice_max = 59.0;
  const double c1 = 1.0e-3, c2 = 0.0;
  double conci_bfnucl[FSBM_ICEMAX + 1], conci_afnucl[FSBM_ICEMAX + 1];
  for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
    conci_bfnucl[ice] = 0.0;
    for (int kr = 1; kr <= FSBM_NKR; ++kr)
      conci_bfnucl[ice] = conci_bfnucl[ice] + 3.0 * col * A2(psi2, kr, ice) * A2(xi, kr, ice);
  }
  double tpc = tt - 273.15;
  int itype;
  if ((tpc > -4.0) || (tpc <= -8.1 && tpc > -12.7) || (tpc <= -17.8 && tpc > -22.4)) {
    itype = 2;
  } else if ((tpc <= -4.0 && tpc > -8.1) || (tpc <= -22.4)) {
    itype = 1;
  } else {
    itype = 3;
  }
  int ice = itype;
  double helek1, helek2;
  if (tpc < temp1) {
    double del2n = 100.0 * sup2;
    double del2nn = del2n;
    if (del2n > delsupice_max) del2nn = delsupice_max;
    helek1 = c1 * fx_exp(a1 + b1 * del2nn);
  } else {
    helek1 = 0.0;
  }
  if (tpc < temp2) {
    double tpcc = tpc;
    if (tpcc < temp3) tpcc = temp3;
    helek2 = c2 * fx_exp(a2 - b2 * tpcc);
  } else {
    helek2 = 0.0;
  }
  double ff1bn = helek1 + helek2;
  double fact = 1.0;
  double dsup2n = (sup2 - sup2_old + dsupice_xyz) * 100.0;
  sup2_old = sup2;
  if (dsup2n > delsupice_max) dsup2n = delsupice_max;
  double deltacd = ff1bn * b1 * dsup2n;
  if (deltacd >= ff1bn) deltacd = ff1bn;
  if (deltacd > 0.0) {
    double deltaf = deltacd * fact;
    if (conci_bfnucl[ice] <= helek1) {
      for (int kr = 1; kr <= 1; ++kr) {  // KR = 1, NRGI-1
        double dx = 3.0 * A2(xi, kr, ice) * col;
        double addf = deltaf / dx;
        A2(psi2, kr, ice) = A2(psi2, kr, ice) + addf;
      }
    }
  }
  for (int ic = 1; ic <= FSBM_ICEMAX; ++ic) {
    conci_afnucl[ic] = 0.0;
    for (int kr = 1; kr <= FSBM_NKR; ++kr)
      conci_afnucl[ic] = conci_afnucl[ic] + 3.0 * col * A2(psi2, kr, ic) * A2(xi, kr, ic);
    double delconci = fabs_(conci_afnucl[ic] - conci_bfnucl[ic]);
    if (delconci > 10.0) return 3463;  // STOP 099
  }
  return FSBM_OK;
}

// Per-call nucleation tallies for the budgets.
struct FsbmNuclTally {
  double drop_mass;  // activated drop mass, g/g (formed as the ice branch forms its mass)
  double homfrz_mass;
  double ice_mass;
};

// JERNUCL01_KS (:3037-3192) with ICEPROCS = 1.  psi1_r (33), psi2_r (A2,
// 33x3), fccnr_r (43) are R, 1-based.  ror, pp, col, sup2_old, dsupice, win
// are the REAL actuals FAST_SBM passes.  sink: D3 corrected.
template <class R>
FHD_NOINLINE int jernucl01_ks(R* psi1_r, R* psi2_r, R* fccnr_r, const FsbmTab<R>& T, double& tt,
                              double& qq, R ror_r, R pp_r, double& sup1, double& sup2, R col_r,
                              R sup2_old_r, R dsupice_xyz_r, R win_r, int is_this_cloudbase,
                              int sink, FsbmNuclTally* tally) {
  const double t_nucl_drop_min = -80.0, t_nucl_ice_min = -37.0;
  const double al2 = 2834.0, al1 = 2500.0;
  const R ro_solute = RL(1.79), mwaero = RL(115.0);
  const int ions = 3;
  double psi1[FSBM_NKR + 1], psi2[FSBM_NKR * FSBM_ICEMAX + 1], fccnr[FSBM_NKR_AERO + 1];
  double xl[FSBM_NKR + 1], xi[FSBM_NKR * FSBM_ICEMAX + 1], rccn[FSBM_NKR_AERO + 1];
  double dropconcn[FSBM_NKR + 1], fi2[FSBM_NKR * FSBM_ICEMAX + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    psi1[kr] = (double)psi1_r[kr];
    xl[kr] = (double)T.xl[kr - 1];
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      A2(psi2, kr, ice) = (double)A2(psi2_r, kr, ice);
      A2(xi, kr, ice) = (double)T2(T.xi, kr, ice, FSBM_NKR);
    }
  }
  for (int kr = 1; kr <= FSBM_NKR_AERO; ++kr) {
    fccnr[kr] = (double)fccnr_r[kr];
    rccn[kr] = (double)T.rccn[kr - 1];
  }
  double ror = (double)ror_r, pp = (double)pp_r, col = (double)col_r;
  double sup2_old = (double)sup2_old_r, dsupice_xyz = (double)dsupice_xyz_r, win = (double)win_r;
  double col3 = 3.0 * col;
  double rori = 1.0 / ror;
  double tpn = tt, qpn = qq;
  double tpc = tt - 273.15;
  if (sup1 > 0.0 && tpc > t_nucl_drop_min) {
    double s = 0.0;
    for (int kr = 1; kr <= FSBM_NKR_AERO; ++kr) s = s + fccnr[kr];
    if (s > (double)RL(0.0)) {
      for (int kr = 1; kr <= FSBM_NKR; ++kr) dropconcn[kr] = 0.0;
      double before = 0.0;
      if (sink || tally) {
        for (int kr = 1; kr <= FSBM_NKR; ++kr) before = before + psi1[kr] * xl[kr] * xl[kr];
        before = before * col3 * rori;
        if (before < 0.0) before = 0.0;
      }
      int e = water_nucleation<R>(col, FSBM_NKR_AERO, psi1, fccnr, xl, tt, qq, ror, sup1, dropconcn,
                                  pp, is_this_cloudbase, win, ro_solute, rccn, ions, mwaero);
      if (e) return e;
      if (sink || tally) {
        // D3 corrected: the activated drop mass leaves the vapour and heats
        // the air, the sequence the ice branch applies (:3129-3179) with AL1.
        double after = 0.0;
        for (int kr = 1; kr <= FSBM_NKR; ++kr) after = after + psi1[kr] * xl[kr] * xl[kr];
        after = after * col3 * rori;
        if (after < 0.0) after = 0.0;
        double delmassl = after - before;
        if (tally) tally->drop_mass += delmassl;
        if (sink && delmassl != 0.0) {
          qpn = qq - delmassl;
          qq = qpn;
          tpn = tt + al1 * delmassl;
          tt = tpn;
          R tpnreal = (R)tpn;
          double es1n = polysvp<R>(tpnreal, 0);
          double es2n = polysvp<R>(tpnreal, 1);
          double ew1n = qpn * pp / (0.622 + 0.378 * qpn);
          sup1 = ew1n / es1n - 1.0;
          sup2 = ew1n / es2n - 1.0;
        }
      }
    }
    if (tpc <= -38.0) {
      double sum_ice = 0.0;
      for (int kr = 1; kr <= FSBM_NKR; ++kr) {
        A2(psi2, kr, 2) = A2(psi2, kr, 2) + psi1[kr];
        sum_ice = sum_ice + col3 * xl[kr] * xl[kr] * psi1[kr];
        psi1[kr] = 0.0;
      }
      double arg_1 = 334.0 * sum_ice * rori;
      tt = tt + arg_1;
      if (tally) tally->homfrz_mass += sum_ice * rori;
    }
  }
  double del2n = 100.0 * sup2;
  tpc = tt - 273.15;
  if (tpc < 0.0 && tpc >= t_nucl_ice_min && del2n > 0.0) {
    for (int k = 1; k <= FSBM_NKR * FSBM_ICEMAX; ++k) fi2[k] = psi2[k];
    int e = ice_nucl(psi2, xi, sup2, tt, dsupice_xyz, sup2_old, col);
    if (e) return e;
    double rmassiaa = 0.0, rmassibb = 0.0;
    for (int k = 1; k <= FSBM_NKR; ++k)
      for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
        double xk = A2(xi, k, ice);
        rmassibb = rmassibb + A2(fi2, k, ice) * xk * xk;
      }
    rmassibb = rmassibb * col3 * rori;
    if (rmassibb < 0.0) rmassibb = 0.0;
    for (int k = 1; k <= FSBM_NKR; ++k)
      for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
        double xk = A2(xi, k, ice);
        rmassiaa = rmassiaa + A2(psi2, k, ice) * xk * xk;
      }
    rmassiaa = rmassiaa * col3 * rori;
    if (rmassiaa < 0.0) rmassiaa = 0.0;
    double delmassice_nucl = rmassiaa - rmassibb;
    qpn = qq - delmassice_nucl;
    qq = qpn;
    tpn = tt + al2 * delmassice_nucl;
    tt = tpn;
    R tpnreal = (R)tpn;
    double es1n = polysvp<R>(tpnreal, 0);
    double es2n = polysvp<R>(tpnreal, 1);
    double ew1n = qpn * pp / (0.622 + 0.378 * qpn);
    sup1 = ew1n / es1n - 1.0;
    sup2 = ew1n / es2n - 1.0;
    if (tally) tally->ice_mass += delmassice_nucl;
  }
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    psi1_r[kr] = (R)psi1[kr];
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(psi2_r, kr, ice) = (R)A2(psi2, kr, ice);
  }
  for (int kr = 1; kr <= FSBM_NKR_AERO; ++kr) fccnr_r[kr] = (R)fccnr[kr];
  return FSBM_OK;
}

#endif  // WOOF_FAST_SBM_NUCL_CUH
