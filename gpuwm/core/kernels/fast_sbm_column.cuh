// WRF mp_physics = 30: the column parts of FAST_SBM: entry and exit unit
// conversions (:4214-4257, :5276-5319), the first-call aerosol column
// (:4283-4311), the cloud-base and ice-forcing pre-pass (:4372-4512),
// sedimentation through FALFLUXHUCM_Z (:4890-4962, :5324-5382), the bulk
// diagnostics (:4964-5052) and surface precipitation (:5056-5121).
//
// A column is addressed through strides so the same code serves WRF's
// memory layout (the host oracle driver) and the engine's (the device).

#ifndef WOOF_FAST_SBM_COLUMN_CUH
#define WOOF_FAST_SBM_COLUMN_CUH

// Entry conversion of one point (:4219-4251): advected mass mixing ratios
// to the scheme's distributions, aerosol #/kg to #/cm3 per ln r.
template <class R>
FHD void fsbm_entry_point(const FsbmTab<R>& T, int hail, R* bins, long sn, R rho_phy) {
  const R col = FSBM_COL;
  R rhocgs = rho_phy * RL(0.001);
  for (int krr = 1; krr <= FSBM_NKR; ++krr) {
    R& c = bins[(FSBM_OFF_DROP + krr - 1) * sn];
    R x = T.xl[krr - 1];
    c = c * rhocgs / col / x / x / RL(3.0);
  }
  for (int krr = 1; krr <= FSBM_NKR; ++krr) {
    R& c = bins[(FSBM_OFF_SNOW + krr - 1) * sn];
    R x = T.xs[krr - 1];
    c = c * rhocgs / col / x / x / RL(3.0);
  }
  for (int krr = 1; krr <= FSBM_NKR_AERO; ++krr) {
    R& c = bins[(FSBM_OFF_AERO + krr - 1) * sn];
    c = c * rhocgs / RL(1000.0);
  }
  for (int krr = 1; krr <= FSBM_NKR; ++krr) {
    R& c = bins[(FSBM_OFF_DENSE + krr - 1) * sn];
    R x = hail ? T.xh[krr - 1] : T.xg[krr - 1];
    c = c * rhocgs / col / x / x / RL(3.0);
  }
}

// First-call aerosol of one point (:4288-4306): continental over land
// (XLAND == 1), maritime elsewhere, scaled by density relative to level 1.
template <class R>
FHD void fsbm_aerosol_init_point(const FsbmTab<R>& T, R* bins, long sn, R rhocgs, R rhocgs_k1,
                                 R xland) {
  double rhoair_max = (double)rhocgs_k1;
  for (int krr = 1; krr <= FSBM_NKR_AERO; ++krr) {
    double f = (xland == (R)1) ? T.fccnr_con[krr - 1] : T.fccnr_mar[krr - 1];
    bins[(FSBM_OFF_AERO + krr - 1) * sn] = (R)((f / rhoair_max) * (double)rhocgs);
  }
}

// Cloud-base level of one column (:4373-4390); 0 for none.  t_new, qv,
// pcgs, zcgs and w (nz+1 full levels) by stride sk.  first_level is D9.
template <class R>
FHD int fsbm_cloud_base(const R* t_new, const R* qv, const R* pcgs, const R* zcgs, const R* w,
                        long sk, int nz, int first_level) {
  const R aa1_my = FSBM_AA1_MY, bb1_my = FSBM_BB1_MY;
  int kz = 0;
  bool k_found = false;
  for (int k = 1; k <= nz; ++k) {
    double es1n = (double)(aa1_my * fx_exp(-bb1_my / t_new[(k - 1) * sk]));
    R q = qv[(k - 1) * sk];
    double ew1n = (double)(q * pcgs[(k - 1) * sk] / (RL(0.622) + RL(0.378) * q));
    double sup_water = ew1n / es1n - (double)RL(1.0);
    R w_stag_my;
    if (k < nz) w_stag_my = RL(50.) * (w[(k - 1) * sk] + w[k * sk]);
    else w_stag_my = (R)100 * w[(k - 1) * sk];
    bool a = sup_water > 0.0, b = (double)w_stag_my > (double)RL(0.1) * 1.0e2;
    bool c = k > 2, d = (double)zcgs[(k - 1) * sk] < (double)RL(3.0) * 1.0e5;
    if (first_level) {
      if (a && b && !k_found && c && d) {
        kz = k;
        k_found = true;
      }
    } else {
      // (a .and. b .and. K_found) .eqv. (.FALSE. .and. ...): D9
      if (!(a && b && k_found)) {
        kz = k;
        k_found = true;
      }
    }
  }
  return kz;
}

// DSUPICE_XYZ of one point (:4441-4508).  The neighbour fields come from a
// caller-supplied accessor N(di, dj) returning {t_old, qv_old, pcgs, u, v};
// i_lo/i_hi/j_lo/j_hi say whether I == 1, I == IDE-1, J == 1, J == JDE-1.
// supice/dzz: the column arrays of the first pass (1-based).  w_stag: the
// staggered w this point multiplies (D15 decided by the caller).
template <class R> struct FsbmNbr {
  R t_old, qv_old, pcgs, u, v;
};

template <class R, class N>
FHD R fsbm_dsupice_point(const N& nb, int k, int nz, bool i_lo, bool i_hi, bool j_lo, bool j_hi,
                         const R* supice, const R* dzz, R w_stag, R dxhucm, R dyhucm, R dtcond,
                         int wind_own_point) {
  const R aa2_my = FSBM_AA2_MY, bb2_my = FSBM_BB2_MY;
  FsbmNbr<R> c = nb(0, 0);
  if (!(supice[k] >= RL(0.02) && c.t_old < RL(268.15))) return RL(0.0);
  R es2nplsx, ew1nplsx, del2inplsx;
  if (!i_hi) {
    FsbmNbr<R> p = nb(1, 0);
    es2nplsx = aa2_my * fx_exp(-bb2_my / p.t_old);
    ew1nplsx = p.qv_old * p.pcgs / (RL(0.622) + RL(0.378) * p.qv_old);
  } else {
    es2nplsx = aa2_my * fx_exp(-bb2_my / c.t_old);
    ew1nplsx = c.qv_old * c.pcgs / (RL(0.622) + RL(0.378) * c.qv_old);
  }
  if (es2nplsx == (R)0) del2inplsx = RL(0.5);
  else del2inplsx = ew1nplsx / es2nplsx - RL(1.);
  if (del2inplsx > RL(0.5)) del2inplsx = RL(.5);
  double es2n, ew1n;
  if (!i_lo) {
    FsbmNbr<R> m = nb(-1, 0);
    es2n = (double)(aa2_my * fx_exp(-bb2_my / m.t_old));
    ew1n = (double)(m.qv_old * m.pcgs / (RL(0.622) + RL(0.378) * m.qv_old));
  } else {
    es2n = (double)(aa2_my * fx_exp(-bb2_my / c.t_old));
    ew1n = (double)(c.qv_old * c.pcgs / (RL(0.622) + RL(0.378) * c.qv_old));
  }
  double del2in = ew1n / es2n - (double)RL(1.);
  if (del2in > (double)RL(0.5)) del2in = (double)RL(.5);
  R derivs_x;
  if (!i_lo && !i_hi) derivs_x = (R)(((double)del2inplsx - del2in) / (double)(RL(2.) * dxhucm));
  else derivs_x = (R)(((double)del2inplsx - del2in) / (double)dxhucm);
  R es2nplsy, ew1nplsy, del2inplsy;
  if (!j_hi) {
    FsbmNbr<R> p = nb(0, 1);
    es2nplsy = aa2_my * fx_exp(-bb2_my / p.t_old);
    ew1nplsy = p.qv_old * p.pcgs / (RL(0.622) + RL(0.378) * p.qv_old);
  } else {
    es2nplsy = aa2_my * fx_exp(-bb2_my / c.t_old);
    ew1nplsy = c.qv_old * c.pcgs / (RL(0.622) + RL(0.378) * c.qv_old);
  }
  del2inplsy = ew1nplsy / es2nplsy - RL(1.);
  if (del2inplsy > RL(0.5)) del2inplsy = RL(.5);
  if (!j_lo) {
    FsbmNbr<R> m = nb(0, -1);
    es2n = (double)(aa2_my * fx_exp(-bb2_my / m.t_old));
    ew1n = (double)(m.qv_old * m.pcgs / (RL(0.622) + RL(0.378) * m.qv_old));
  } else {
    es2n = (double)(aa2_my * fx_exp(-bb2_my / c.t_old));
    ew1n = (double)(c.qv_old * c.pcgs / (RL(0.622) + RL(0.378) * c.qv_old));
  }
  del2in = ew1n / es2n - (double)RL(1.);
  if (del2in > (double)RL(0.5)) del2in = (double)RL(.5);
  R derivs_y;
  if (!j_lo && !j_hi) derivs_y = (R)(((double)del2inplsy - del2in) / (double)(RL(2.) * dyhucm));
  else derivs_y = (R)(((double)del2inplsy - del2in) / (double)dyhucm);
  R derivs_z;
  if (k == 1) derivs_z = (supice[k + 1] - supice[k]) / dzz[k];
  else if (k == nz) derivs_z = (supice[k] - supice[k - 1]) / dzz[k];
  else derivs_z = (supice[k + 1] - supice[k - 1]) / dzz[k];
  R ux, vx;
  if (!wind_own_point && !i_hi && !j_hi) {
    FsbmNbr<R> a = nb(1, 0), b = nb(0, 1), d = nb(1, 1);
    ux = RL(25.) * (c.u + a.u + b.u + d.u);
    vx = RL(25.) * (c.v + a.v + b.v + d.v);
  } else {
    ux = c.u * RL(100.);
    vx = c.v * RL(100.);
  }
  return (ux * derivs_x + vx * derivs_y + w_stag * derivs_z) * dtcond;
}

// SUPICE and DZZ of one column (:4392-4402), 1-based outputs.
template <class R>
FHD void fsbm_supice_column(const R* t_old, const R* qv_old, const R* pcgs, const R* zcgs, long sk,
                            int nz, R* supice, R* dzz) {
  const R aa2_my = FSBM_AA2_MY, bb2_my = FSBM_BB2_MY;
  for (int k = 1; k <= nz; ++k) {
    if (k == nz) dzz[k] = zcgs[(k - 1) * sk] - zcgs[(k - 2) * sk];
    else if (k == 1) dzz[k] = zcgs[k * sk] - zcgs[(k - 1) * sk];
    else dzz[k] = zcgs[k * sk] - zcgs[(k - 2) * sk];
    double es2n = (double)(aa2_my * fx_exp(-bb2_my / t_old[(k - 1) * sk]));
    R q = qv_old[(k - 1) * sk];
    double ew1n = (double)(q * pcgs[(k - 1) * sk] / (RL(0.622) + RL(0.378) * q));
    supice[k] = (R)(ew1n / es2n - (double)RL(1.));
    if (supice[k] > RL(0.5)) supice[k] = RL(.5);
  }
}

// FALFLUXHUCM_Z (:5324-5382) for one species of one column, in place on
// the distribution over air density ffx (stride sk, sn), fall speeds
// vr(kr) * sqrt(1e6/pcgs(k)) formed on demand.  outflow (1-based, may be
// null) accumulates the bottom-face flux rho v f dtfall of every substep
// (the D2 correction's surface flux).
template <class R>
FHD int falflux_column(R* ffx, long sk, long sn, const R* vr, const R* pcgs, long skp, const R* rho,
                       long skr, const R* zcgs, long skz, R dt, int nz, double* outflow,
                       const R* dz8w, long skdz, int cell_depth) {
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    R* f = ffx + (kr - 1) * sn;
    int ifall = 0;
    for (int k = 1; k <= nz; ++k)
      if (f[(k - 1) * sk] >= RL(1.E-20)) ifall = 1;
    if (ifall != 1) continue;
    R tfall = RL(1.E10);
    for (int k = 1; k <= nz; ++k) {
      R factor_p = (R)fx_sqrt(1.0e6 / (double)pcgs[(k - 1) * skp]);
      R vfall = vr[kr - 1] * factor_p;
      tfall = fmin2(tfall, zcgs[(k - 1) * skz] / (vfall + RL(1.E-20)));
    }
    if (tfall >= RL(1.E10)) return 5360;  // WRF: bare STOP
    int nsub = (int)(RL(2.0) * dt / tfall) + 1;
    R dtfall = dt / (R)nsub;
    for (int n = 1; n <= nsub; ++n) {
      // DWFLUX(K) uses the substep's starting values of levels K and K+1,
      // and level K+1 is updated after level K, so one upward sweep in
      // place gives WRF's two-pass result.
      if (outflow) {
        R fp1 = (R)fx_sqrt(1.0e6 / (double)pcgs[0]);
        outflow[kr] += (double)(rho[0] * (vr[kr - 1] * fp1) * f[0]) * (double)dtfall;
      }
      R f_k = f[0];
      R v_k = vr[kr - 1] * (R)fx_sqrt(1.0e6 / (double)pcgs[0]);
      for (int k = 1; k <= nz; ++k) {
        R rho_k = rho[(k - 1) * skr];
        R dwflux;
        if (k < nz) {
          R f_kp = f[k * sk];
          R v_kp = vr[kr - 1] * (R)fx_sqrt(1.0e6 / (double)pcgs[k * skp]);
          R rho_kp = rho[k * skr];
          dwflux = -(rho_k * v_k * f_k - rho_kp * v_kp * f_kp) /
                   (rho_k * (cell_depth ? dz8w[(k-1)*skdz]*RL(100.) : (zcgs[k * skz] - zcgs[(k - 1) * skz])));
          f[(k - 1) * sk] = f_k + dwflux * dtfall;
          f_k = f_kp;
          v_k = v_kp;
        } else {
          dwflux = -(rho_k * v_k * f_k) / (rho_k * (cell_depth ? dz8w[(k-1)*skdz]*RL(100.) : (zcgs[(k - 1) * skz] - zcgs[(k - 2) * skz])));
          f[(k - 1) * sk] = f_k + dwflux * dtfall;
        }
      }
    }
  }
  return FSBM_OK;
}

// Everything FAST_SBM does to one column after the point loop: the fall of
// the three species, the bulk diagnostics, the surface precipitation, the
// th_old/qv_old update and the exit conversion.  chem: the column's bins in
// scheme units (stride sk between levels, sn between bins); field arrays
// by stride sk; pcgs and zcgs by stride skz.  vr_last_k1: the pressure
// factor of the tile's last point at level 1 (D1, WRF), or this column's.
template <class R> struct FsbmColumnOut {
  R* qc; R* qr; R* qi; R* qs; R* qg; R* qnc; R* qnr; R* qni; R* qns; R* qng; R* qna;
};

template <class R>
FHD_NOINLINE int fsbm_column(const FsbmTab<R>& T, const FsbmFlags& F, R* chem, long sk, long sn,
                             const R* rho_phy, const R* pcgs, const R* zcgs, long skz, R dt, int nz,
                             R factor_p_rainnc, FsbmColumnOut<R> out, R& rainnc, R& rainncv,
                             R& snownc, R& snowncv, R& graupelnc, R& graupelncv, double* budget, const R* dz8w, long skdz, const R* rhoc, long skr) {
  const R col = FSBM_COL;
  double out1[FSBM_NKR + 1], out3[FSBM_NKR + 1], out6[FSBM_NKR + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) out1[kr] = out3[kr] = out6[kr] = 0.0;
  // chem -> ffx = chem/rhocgs, in place (:4904)
  for (int k = 1; k <= nz; ++k) {
    R rhocgs = rho_phy[(k - 1) * sk] * RL(0.001);
    for (int n = 0; n < 99; ++n) {
      R& c = chem[(k - 1) * sk + n * sn];
      c = c / rhocgs;
    }
  }
  // Reuse the prepared column density. No fixed vertical stack capacity.
  int e = falflux_column<R>(chem + FSBM_OFF_DROP * sn, sk, sn, T.vr1, pcgs, skz, rhoc, skr, zcgs, skz,
                            dt, nz, out1, dz8w, skdz, F.sedimentation_cell_depth);
  if (e) return e;
  e = falflux_column<R>(chem + FSBM_OFF_SNOW * sn, sk, sn, T.vr3, pcgs, skz, rhoc, skr, zcgs, skz, dt,
                        nz, out3, dz8w, skdz, F.sedimentation_cell_depth);
  if (e) return e;
  e = falflux_column<R>(chem + FSBM_OFF_DENSE * sn, sk, sn, F.hail ? T.vr5 : T.vr4, pcgs, skz, rhoc, skr,
                        zcgs, skz, dt, nz, out6, dz8w, skdz, F.sedimentation_cell_depth);
  if (e) return e;
  for (int k = 1; k <= nz; ++k) {
    R rhocgs = rhoc[(k - 1)*skr];
    for (int n = 0; n < 99; ++n) {
      R& c = chem[(k - 1) * sk + n * sn];
      c = c * rhocgs;
    }
  }
  // bulk diagnostics (:4971-5052)
  for (int k = 1; k <= nz; ++k) {
    long o = (long)(k - 1) * sk;
    R rhocgs = rhoc[(k - 1)*skr];
    R qc = RL(0.0), qr = RL(0.0), qi = RL(0.0), qs = RL(0.0), qg = RL(0.0);
    R qnc = RL(0.0), qnr = RL(0.0), qni = RL(0.0), qns = RL(0.0), qng = RL(0.0), qna = RL(0.0);
    for (int krr = 1; krr <= FSBM_NKR; ++krr) {
      R c = chem[o + (FSBM_OFF_DROP + krr - 1) * sn];
      R x = T.xl[krr - 1];
      if (krr < FSBM_KRDROP) {
        qc = qc + (RL(1.) / rhocgs) * col * c * x * x * (R)3;
        qnc = qnc + col * c * x * RL(3.0) / rhocgs * RL(1000.0);
      } else {
        qr = qr + (RL(1.) / rhocgs) * col * c * x * x * RL(3.0);
        qnr = qnr + col * c * x * (R)3 / rhocgs * RL(1000.0);
      }
    }
    for (int krr = 1; krr <= FSBM_NKR; ++krr) {
      R c = chem[o + (FSBM_OFF_SNOW + krr - 1) * sn];
      R x = T.xs[krr - 1];
      if (krr <= FSBM_KRICE) {
        qi = qi + (RL(1.0) / rhocgs) * col * c * x * x * RL(3.0);
        qni = qni + col * c * x * RL(3.0) / rhocgs * RL(1000.0);
      } else {
        qs = qs + (RL(1.0) / rhocgs) * col * c * x * x * RL(3.0);
        qns = qns + col * c * x * RL(3.0) / rhocgs * RL(1000.0);
      }
    }
    for (int krr = 1; krr <= FSBM_NKR; ++krr) {
      R c = chem[o + (FSBM_OFF_DENSE + krr - 1) * sn];
      R x = F.hail ? T.xh[krr - 1] : T.xg[krr - 1];
      qg = qg + (RL(1.0) / rhocgs) * col * c * x * x * RL(3.0);
      qng = qng + col * c * x * RL(3.0) / rhocgs * RL(1000.0);
    }
    for (int krr = 1; krr <= FSBM_NKR_AERO; ++krr) {
      R c = chem[o + (FSBM_OFF_AERO + krr - 1) * sn];
      qna = qna + col * c / rhocgs * RL(1000.0);
    }
    out.qc[o] = qc; out.qr[o] = qr; out.qi[o] = qi; out.qs[o] = qs; out.qg[o] = qg;
    out.qnc[o] = qnc; out.qnr[o] = qnr; out.qni[o] = qni; out.qns[o] = qns; out.qng[o] = qng;
    out.qna[o] = qna;
  }
  // surface precipitation (:5056-5121)
  rainncv = RL(0.0);
  snowncv = RL(0.0);
  graupelncv = RL(0.0);
  const R* x6 = F.hail ? T.xh : T.xg;
  const R* v6 = F.hail ? T.vr5 : T.vr4;
  for (int krr = 1; krr <= FSBM_NKR; ++krr) {
    R x = T.xl[krr - 1];
    R ro = T.ro1bl[krr - 1];
    R term;
    if (!F.rainnc_bottom_outflow) {
      R deltaw = T.vr1[krr - 1] * factor_p_rainnc;
      term = RL(10.0) * (RL(3.) / ro) * col * dt * deltaw * chem[(FSBM_OFF_DROP + krr - 1) * sn] * x * x;
    } else {
      term = (R)((double)(RL(10.0) * (RL(3.) / ro) * col) * out1[krr] * (double)x * (double)x);
    }
    rainnc = rainnc + term;
    rainncv = rainncv + term;
  }
  for (int krr = 1; krr <= FSBM_NKR; ++krr) {
    R x = T.xs[krr - 1];
    R ro = T.ro1bl[krr - 1];
    R term, term2;
    if (!F.rainnc_bottom_outflow) {
      R deltaw = T.vr3[krr - 1] * factor_p_rainnc;
      R c = chem[(FSBM_OFF_SNOW + krr - 1) * sn];
      term = RL(10.0) * (RL(3.) / ro) * col * dt * deltaw * c * x * x;
      term2 = (R)10 * (RL(3.) / ro) * col * dt * deltaw * c * x * x;
    } else {
      term = term2 = (R)((double)(RL(10.0) * (RL(3.) / ro) * col) * out3[krr] * (double)x * (double)x);
    }
    rainnc = rainnc + term;
    rainncv = rainncv + term;
    snownc = snownc + term2;
    snowncv = snowncv + term2;
  }
  for (int krr = 1; krr <= FSBM_NKR; ++krr) {
    R x = x6[krr - 1];
    R ro = T.ro1bl[krr - 1];
    R term, term2;
    if (!F.rainnc_bottom_outflow) {
      R deltaw = v6[krr - 1] * factor_p_rainnc;
      R c = chem[(FSBM_OFF_DENSE + krr - 1) * sn];
      term = RL(10.0) * (RL(3.) / ro) * col * dt * deltaw * c * x * x;
      term2 = (R)10 * (RL(3.) / ro) * col * dt * deltaw * c * x * x;
    } else {
      term = term2 = (R)((double)(RL(10.0) * (RL(3.) / ro) * col) * out6[krr] * (double)x * (double)x);
    }
    rainnc = rainnc + term;
    rainncv = rainncv + term;
    graupelnc = graupelnc + term2;
    graupelncv = graupelncv + term2;
  }
  // exit conversion (:5276-5319), with D4 and D11 clips
  for (int k = 1; k <= nz; ++k) {
    long o = (long)(k - 1) * sk;
    R rhocgs = rho_phy[o] * RL(0.001);
    R qc = out.qc[o], qr = out.qr[o], qs = out.qs[o], qi = out.qi[o], qg = out.qg[o];
    double clipped = 0.0;
    for (int krr = 1; krr <= FSBM_NKR; ++krr) {
      R& c = chem[o + (FSBM_OFF_DROP + krr - 1) * sn];
      R x = T.xl[krr - 1];
      c = c / rhocgs * col * x * x * RL(3.0);
      if (qc + qr < RL(1.e-13)) { clipped += (double)c; c = RL(0.0); }
    }
    R snow_key = F.snow_clip_qi_plus_qs ? (qi + qs) : qs;
    for (int krr = 1; krr <= FSBM_NKR; ++krr) {
      R& c = chem[o + (FSBM_OFF_SNOW + krr - 1) * sn];
      R x = T.xs[krr - 1];
      c = c / rhocgs * col * x * x * RL(3.0);
      if (snow_key < RL(1.e-13)) { clipped += (double)c; c = RL(0.0); }
    }
    for (int krr = 1; krr <= FSBM_NKR_AERO; ++krr) {
      R& c = chem[o + (FSBM_OFF_AERO + krr - 1) * sn];
      c = c / rhocgs * RL(1000.0);
    }
    for (int krr = 1; krr <= FSBM_NKR; ++krr) {
      R& c = chem[o + (FSBM_OFF_DENSE + krr - 1) * sn];
      R x = F.hail ? T.xh[krr - 1] : T.xg[krr - 1];
      c = c / rhocgs * col * x * x * RL(3.0);
      if (qg < RL(1.e-13)) { clipped += (double)c; c = RL(0.0); }
    }
    if (budget) budget[0] += clipped;
  }
  return FSBM_OK;
}

#endif  // WOOF_FAST_SBM_COLUMN_CUH
