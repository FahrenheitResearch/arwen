// WRF mp_physics = 30: the module_mp_fast_sbm routines FAST_SBM calls
// (FREEZ, J_W_MELT, ONECOND1/2/3, COAL_BOTT_NEW, FALFLUXHUCM_Z) and
// FAST_SBM itself (:4001-5322) split into its point part (:4514-4885) and
// its column part (:4890-5320), plus the cloud-base and ice-forcing
// pre-pass (:4372-4512).  1-based arrays; (NKR,ICEMAX) arrays in the A2
// layout of fast_sbm_aux.cuh.

#ifndef WOOF_FAST_SBM_CORE_CUH
#define WOOF_FAST_SBM_CORE_CUH


// FAST_SBM's DATA constants (:4086-4099)
#define FSBM_AA1_MY RL(2.53E12)
#define FSBM_BB1_MY RL(5.42E3)
#define FSBM_AA2_MY RL(3.41E13)
#define FSBM_BB2_MY RL(6.13E3)
#define FSBM_COL RL(0.23105)

// Process tallies of one point, for budgets and the oracle's acceptance
// counts.  All masses g/g.
struct FsbmPointDiag {
  FsbmNuclTally nucl;
  int n_cond1, n_cond2, n_cond3, n_coal, n_freez, n_melt, n_homfrz_point, n_spont;
  double freez_mass, melt_mass, homfrz_point_mass, rime_mass;
};

// ---------------------------------------------------------------------------
// FREEZ (:6560-6636), stochastic immersion freezing (Bigg 1953).  All REAL.
// ff2 in the A2 layout.  fusion: 333 (WRF) or 334 (D13).
// ---------------------------------------------------------------------------
template <class R>
FHD void freez(R* ff1, const R* xl, R* ff2, R* ff5, R& tin, R dt, R ro, R col, R afreezmy,
               R bfreezmy, R bfreezmax, int krfreez, R fusion, double* frozen_mass) {
  R ttin = tin;
  R del_t = ttin - RL(273.15);
  int ice_type = 2;
  R f1_max = RL(0.);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) f1_max = fmax2(f1_max, ff1[kr]);
  if (del_t < (R)0 && f1_max != (R)0) {
    R sum_ice = RL(0.);
    R af = afreezmy;
    R cfreez = (bfreezmax - bfreezmy) / xl[FSBM_NKR];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      R arg_m = xl[kr];
      R bf = bfreezmy + cfreez * arg_m;
      R pf_1 = af * fx_exp(-bf * del_t);
      R pf = arg_m * pf_1;
      R ykk = fx_exp(-pf * dt);
      R df1 = ff1[kr] * (RL(1.) - ykk);
      R yk2 = df1;
      ff1[kr] = ff1[kr] * ykk;
      if (kr <= krfreez) A2(ff2, kr, ice_type) = A2(ff2, kr, ice_type) + yk2;
      else ff5[kr] = ff5[kr] + yk2;
      sum_ice = sum_ice + yk2 * RL(3.) * xl[kr] * xl[kr] * col;
    }
    R arg_1 = fusion * sum_ice / ro;
    R tt_after = ttin + arg_1;
    tin = tt_after;
    if (frozen_mass) *frozen_mass += (double)(sum_ice / ro);
  }
}

// ---------------------------------------------------------------------------
// J_W_MELT (:6638-6746), melting along a constant time scale.  D12:
// exponential keeps exp(-rate dt) of a bin where WRF removes rate*dt of it.
// ---------------------------------------------------------------------------
template <class R>
FHD void fsbm_melt_take(R& arg_m, R& f, R rate, R dt, int expo) {
  if (!expo) {
    arg_m = arg_m + f * (rate * dt);
    f = f - f * (rate * dt);
  } else {
    R ykk = fx_exp(-(rate * dt));
    R d = f * (RL(1.) - ykk);
    arg_m = arg_m + d;
    f = f * ykk;
  }
}

template <class R>
FHD void j_w_melt(R* ff1, const R* xl, R* ff2, R* ff3, R* ff4, R* ff5, R& tin, R dt, R ro, R col,
                  R fusion, int expo, double* melted_mass) {
  R del_t = tin - RL(273.15);
  R f2_max = RL(0.), f3_max = RL(0.), f4_max = RL(0.), f5_max = RL(0.), ff_max = RL(0.);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    f3_max = fmax2(f3_max, ff3[kr]);
    f4_max = fmax2(f4_max, ff4[kr]);
    f5_max = fmax2(f5_max, ff5[kr]);
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) f2_max = fmax2(f2_max, A2(ff2, kr, ice));
    ff_max = fmax2(fmax2(f2_max, f3_max), fmax2(f4_max, f5_max));
  }
  R sum_ice = RL(0.);
  if (del_t >= (R)0 && ff_max != (R)0) {
    const R rate_mid = RL(0.5) / RL(50.), rate_big = RL(0.683) / RL(120.);
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      R arg_m = RL(0.0);
      for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
        R& f = A2(ff2, kr, ice);
        if (ice == 1) {
          if (kr <= 10) {
            arg_m = arg_m + f;
            f = RL(0.0);
          } else if (kr > 10 && kr < 18) {
            fsbm_melt_take<R>(arg_m, f, rate_mid, dt, expo);
          } else {
            fsbm_melt_take<R>(arg_m, f, rate_big, dt, expo);
          }
        }
        if (ice == 2 || ice == 3) {
          if (kr <= 12) {
            arg_m = arg_m + f;
            f = RL(0.);
          } else if (kr > 12 && kr < 20) {
            fsbm_melt_take<R>(arg_m, f, rate_mid, dt, expo);
          } else {
            fsbm_melt_take<R>(arg_m, f, rate_big, dt, expo);
          }
        }
      }
      if (kr <= 14) {
        arg_m = arg_m + ff3[kr];
        ff3[kr] = RL(0.0);
      } else if (kr > 14 && kr < 22) {
        fsbm_melt_take<R>(arg_m, ff3[kr], rate_mid, dt, expo);
      } else {
        fsbm_melt_take<R>(arg_m, ff3[kr], rate_big, dt, expo);
      }
      if (kr <= 13) {
        arg_m = arg_m + ff4[kr] + ff5[kr];
        ff4[kr] = RL(0.);
        ff5[kr] = RL(0.);
      } else {
        R rate = (kr > 13 && kr < 23) ? rate_mid : rate_big;
        if (!expo) {
          arg_m = arg_m + (ff4[kr] + ff5[kr]) * (rate * dt);
          ff4[kr] = ff4[kr] - ff4[kr] * (rate * dt);
          ff5[kr] = ff5[kr] - ff5[kr] * (rate * dt);
        } else {
          R ykk = fx_exp(-(rate * dt));
          arg_m = arg_m + (ff4[kr] + ff5[kr]) * (RL(1.) - ykk);
          ff4[kr] = ff4[kr] * ykk;
          ff5[kr] = ff5[kr] * ykk;
        }
      }
      ff1[kr] = ff1[kr] + arg_m;
      sum_ice = sum_ice + arg_m * RL(3.) * xl[kr] * xl[kr] * col;
    }
    R arg_1 = fusion * sum_ice / ro;
    tin = tin - arg_1;
    if (melted_mass) *melted_mass += (double)(sum_ice / ro);
  }
}

// ---------------------------------------------------------------------------
// Small helpers for the ONECOND routines.
// ---------------------------------------------------------------------------
template <class R> FHD double fsbm_oper2(double ar1) {
  return (double)RL(0.622) / ((double)RL(0.622) + (double)RL(0.378) * ar1) / ar1;
}

template <class R> struct FsbmSpecies {
  // One species' 1-based copies inside ONECOND2/3.
  double rd[FSBM_NKR + 1], rnd[FSBM_NKR + 1], vrd[FSBM_NKR + 1];
  R b1[FSBM_NKR + 1], fi[FSBM_NKR + 1], fl[FSBM_NKR + 1], cap[FSBM_NKR + 1], ro[FSBM_NKR + 1];
};

// ---------------------------------------------------------------------------
// ONECOND1 (:6748-7143): warm diffusional growth and evaporation.
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE int onecond1(double& tt, double& qq, double pp, R ror, const R* vr1, double& del1n,
                          double& del2n, R* ff1, R* psi1, const FsbmTab<R>& T, R bb1_my, R bb2_my,
                          R col, R dtcond, int& isym1, int* isym2, int& isym3, int& isym4,
                          int& isym5) {
  const R al1 = RL(2500.), al2 = RL(2834.), gam = RL(1.E-4), epsdel = RL(0.1E-03);
  R b11_my[FSBM_NKR + 1], ff1_old[FSBM_NKR + 1], supintw[FSBM_NKR + 1], fi1[FSBM_NKR + 1];
  R fl1[FSBM_NKR + 1], r1[FSBM_NKR + 1], rlec[FSBM_NKR + 1], ro1bl[FSBM_NKR + 1];
  R sfndummy[4];
  double r1d[FSBM_NKR + 1], r1nd[FSBM_NKR + 1], vr1_d[FSBM_NKR + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    r1[kr] = T.xl[kr - 1];
    rlec[kr] = T.rlec[kr - 1];
    ro1bl[kr] = T.ro1bl[kr - 1];
  }
  sfndummy[1] = sfndummy[2] = sfndummy[3] = RL(0.0);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) b11_my[kr] = RL(0.0);
  double col3 = (double)(RL(3.0) * col);
  R dt = dtcond, dtt = dtcond;
  R cwhucm = RL(0.), xrad = RL(0.);
  R b6 = cwhucm * gam - xrad;
  R b8l = RL(1.) / ror;
  double rori = (double)(RL(1.) / ror);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    ff1_old[kr] = ff1[kr];
    supintw[kr] = RL(0.0);
  }
  double tpn = tt, qpn = qq;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) fi1[kr] = ff1[kr];
  R timenew = RL(0.);
  double told = tpn, qold = qpn;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    r1d[kr] = (double)r1[kr];
    r1nd[kr] = r1d[kr];
  }
  R sfnl = RL(0.0), sfn11 = RL(0.0);
  int idrop = 0;
  double dal1 = (double)al1;
  for (;;) {
    R timerev = dt - timenew;
    R del1 = (R)del1n, del2 = (R)del2n, del1s = (R)del1n, del2s = (R)del2n;
    double tps = tpn, qps = qpn;
    if (isym1 == 1) {
      for (int kr = 1; kr <= FSBM_NKR; ++kr) {
        fl1[kr] = RL(0.0);
        vr1_d[kr] = (double)vr1[kr];
      }
      jerrate_ks<R>(r1d, tps, pp, vr1_d, rlec, ro1bl, b11_my, 1, 1, fl1);
      sfndummy[1] = sfn11;
      jertimesc_ks<R>(fi1, r1d, sfndummy, b11_my, b8l, 1, col);
      sfn11 = sfndummy[1];
    }
    R sfn12 = RL(0.0);
    sfnl = sfn11 + sfn12;
    R sfni = RL(0.);
    R b5l = (R)((double)bb1_my / tps / tps);
    R b5i = (R)((double)bb2_my / tps / tps);
    R b7l = b5l * b6, b7i = b5i * b6;
    R dopl = RL(1.) + del1s, dopi = RL(1.) + del2s;
    R rw = (R)((fsbm_oper2<R>(qps) + (double)(b5l * al1)) * (double)dopl * (double)sfnl);
    R ri = (R)((fsbm_oper2<R>(qps) + (double)(b5l * al2)) * (double)dopl * (double)sfni);
    R qw = b7l * dopl;
    R pw = (R)((fsbm_oper2<R>(qps) + (double)(b5i * al1)) * (double)dopi * (double)sfnl);
    R pi = (R)((fsbm_oper2<R>(qps) + (double)(b5i * al2)) * (double)dopi * (double)sfni);
    R qi = b7i * dopi;
    (void)qw;
    (void)qi;
    if (rw != rw || pw != pw) return 6941;
    int kcond = 10;
    if (del1n >= 0.0) kcond = 11;
    R dtnewl = dt;
    dtnewl = fmin2(dtnewl, timerev);
    timenew = timenew + dtnewl;
    dtt = dtnewl;
    if (dtt < (R)0.0) return kcond == 11 ? 6954 : 6998;
    double del1_d = del1, del2_d = del2, rw_d = rw, pw_d = pw, ri_d = ri, pi_d = pi;
    double d1n_d = 0.0, d2n_d = 0.0;
    int e = jersupsat_ks<R>(del1_d, del2_d, del1n, del2n, rw_d, pw_d, ri_d, pi_d, dtt, d1n_d, d2n_d,
                            RL(0.0), RL(0.0), isym1, isym2, isym3, isym4, isym5, 0);
    if (e) return e;
    del1 = (R)del1_d;
    del2 = (R)del2_d;
    rw = (R)rw_d;
    pw = (R)pw_d;
    ri = (R)ri_d;
    pi = (R)pi_d;
    R d1n = (R)d1n_d;
    if (isym1 == 1) {
      idrop = isym1;
      e = jerdfun_ks<R>(r1d, r1nd, b11_my, fi1, psi1, d1n, isym1, 1, 1, idrop, col);
      if (e) return e;
    }
    if (kcond == 11) {
      if ((del1 > (R)0 && del1n < 0) && fabs_(del1n) > (double)epsdel) return 6984;
    } else {
      if ((del1 < (R)0 && del1n > 0) && fabs_(del1n) > (double)epsdel) return 7028;
    }
    double rmasslbb = (double)RL(0.), rmasslaa = (double)RL(0.);
    for (int k = 1; k <= FSBM_NKR; ++k) {
      double fk = (double)fi1[k], rk = (double)r1[k];
      rmasslbb = rmasslbb + fk * rk * rk;
    }
    rmasslbb = rmasslbb * col3 * rori;
    if (rmasslbb <= (double)RL(0.)) rmasslbb = (double)RL(0.);
    for (int k = 1; k <= FSBM_NKR; ++k) {
      double fk = (double)psi1[k], rk = (double)r1[k];
      rmasslaa = rmasslaa + fk * rk * rk;
    }
    rmasslaa = rmasslaa * col3 * rori;
    if (rmasslaa <= (double)RL(0.)) rmasslaa = (double)RL(0.);
    double delmassl1 = rmasslaa - rmasslbb;
    qpn = qps - delmassl1;
    tpn = tps + dal1 * delmassl1;
    if (fabs_(dal1 * delmassl1) > (double)RL(3.0)) return 7070;
    if (isym1 == 1)
      for (int kr = 1; kr <= FSBM_NKR; ++kr) supintw[kr] = supintw[kr] + b11_my[kr] * d1n;
    if (!(timenew < dt)) break;
  }
  if (isym1 == 1) {
    int e = jerdfun_new_ks<R>(r1d, r1nd, supintw, ff1_old, psi1, idrop, col);
    if (e) return e;
  }
  double rmasslaa = (double)RL(0.0), rmasslbb = (double)RL(0.0);
  for (int k = 1; k <= FSBM_NKR; ++k) {
    double fk = (double)ff1_old[k], rk = (double)r1[k];
    rmasslbb = rmasslbb + fk * rk * rk;
  }
  rmasslbb = rmasslbb * col3 * rori;
  if (rmasslbb < (double)RL(0.0)) rmasslbb = (double)RL(0.0);
  for (int k = 1; k <= FSBM_NKR; ++k) {
    double fk = (double)psi1[k], rk = (double)r1[k];
    rmasslaa = rmasslaa + fk * rk * rk;
  }
  rmasslaa = rmasslaa * col3 * rori;
  if (rmasslaa < (double)RL(0.0)) rmasslaa = (double)RL(0.0);
  double delmassl1 = rmasslaa - rmasslbb;
  qpn = qold - delmassl1;
  tpn = told + dal1 * delmassl1;
  if (fabs_(dal1 * delmassl1) > (double)RL(5.0)) return 7132;
  tt = tpn;
  qq = qpn;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) ff1[kr] = psi1[kr];
  return FSBM_OK;
}

// The ice species' growth rates and relaxation sums, shared by ONECOND2/3:
// JERRATE_KS and JERTIMESC_KS for crystals (ID 3), snow, graupel and hail.
template <class R>
FHD void fsbm_ice_rates(double tps, double pp, const int* isym2, int isym3, int isym4, int isym5,
                        double* r2d, const double* vr2_d, const R* riec, const R* ro2bl, R* b21_my,
                        const R* fi2, R* sfni1, FsbmSpecies<R>& s3, FsbmSpecies<R>& s4,
                        FsbmSpecies<R>& s5, R& sfn31, R& sfn41, R& sfn51, R b8i, R col) {
  R fl1[FSBM_NKR + 1], sfndummy[4];
  sfndummy[1] = sfndummy[2] = sfndummy[3] = RL(0.0);
  if (isym2[1] + isym2[2] + isym2[3] > 0) {
    for (int kr = 1; kr <= FSBM_NKR; ++kr) fl1[kr] = RL(0.0);
    jerrate_ks<R>(r2d, tps, pp, vr2_d, riec, ro2bl, b21_my, 3, 2, fl1);
    jertimesc_ks<R>(fi2, r2d, sfni1, b21_my, b8i, FSBM_ICEMAX, col);
  }
  if (isym3 == 1) {
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s3.fl[kr] = RL(0.0);
    jerrate_ks<R>(s3.rd, tps, pp, s3.vrd, s3.cap, s3.ro, s3.b1, 1, 3, s3.fl);
    sfndummy[1] = sfn31;
    jertimesc_ks<R>(s3.fi, s3.rd, sfndummy, s3.b1, b8i, 1, col);
    sfn31 = sfndummy[1];
  }
  if (isym4 == 1) {
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s4.fl[kr] = RL(0.0);
    jerrate_ks<R>(s4.rd, tps, pp, s4.vrd, s4.cap, s4.ro, s4.b1, 1, 2, s4.fl);
    sfndummy[1] = sfn41;
    jertimesc_ks<R>(s4.fi, s4.rd, sfndummy, s4.b1, b8i, 1, col);
    sfn41 = sfndummy[1];
  }
  if (isym5 == 1) {
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s5.fl[kr] = RL(0.0);
    jerrate_ks<R>(s5.rd, tps, pp, s5.vrd, s5.cap, s5.ro, s5.b1, 1, 2, s5.fl);
    sfndummy[1] = sfn51;
    jertimesc_ks<R>(s5.fi, s5.rd, sfndummy, s5.b1, b8i, 1, col);
    sfn51 = sfndummy[1];
  }
}

// The ice species' remaps (JERDFUN_KS with the ice supersaturation integral).
template <class R>
FHD int fsbm_ice_remaps(const int* isym2, int isym3, int isym4, int isym5, double* r2d,
                        double* r2nd, const R* b21_my, const R* fi2, R* psi2, FsbmSpecies<R>& s3,
                        R* psi3, FsbmSpecies<R>& s4, R* psi4, FsbmSpecies<R>& s5, R* psi5, R d2n,
                        R col) {
  int idrop, e;
  if (isym2[1] + isym2[2] + isym2[3] > 0) {
    idrop = 0;
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      if (isym2[ice] == 1) {
        e = jerdfun_ks<R>(r2d + (ice - 1) * FSBM_NKR, r2nd + (ice - 1) * FSBM_NKR,
                          b21_my + (ice - 1) * FSBM_NKR, fi2 + (ice - 1) * FSBM_NKR,
                          psi2 + (ice - 1) * FSBM_NKR, d2n, isym2[ice], FSBM_ICEMAX, ice, idrop, col);
        if (e) return e;
      }
    }
  }
  if (isym3 == 1) {
    idrop = 0;
    e = jerdfun_ks<R>(s3.rd, s3.rnd, s3.b1, s3.fi, psi3, d2n, isym3, 1, 3, idrop, col);
    if (e) return e;
  }
  if (isym4 == 1) {
    idrop = 0;
    e = jerdfun_ks<R>(s4.rd, s4.rnd, s4.b1, s4.fi, psi4, d2n, isym4, 1, 4, idrop, col);
    if (e) return e;
  }
  if (isym5 == 1) {
    idrop = 0;
    e = jerdfun_ks<R>(s5.rd, s5.rnd, s5.b1, s5.fi, psi5, d2n, isym5, 1, 5, idrop, col);
    if (e) return e;
  }
  return FSBM_OK;
}

template <class R>
FHD void fsbm_species_init(FsbmSpecies<R>& s, const R* x, const R* vr, const R* cap, const R* ro) {
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    s.rd[kr] = (double)x[kr - 1];
    s.rnd[kr] = s.rd[kr];
    s.vrd[kr] = (double)vr[kr];
    s.cap[kr] = cap[kr - 1];
    s.ro[kr] = ro[kr - 1];
    s.b1[kr] = RL(0.0);
  }
}

// Ice mass sums of ONECOND2/3: sum over crystals, snow, graupel, hail of
// f x x, in WRF's order (crystals inner, then snow, graupel, hail).
template <class R>
FHD double fsbm_ice_mass_sum(const R* f2, const double* r2d, const R* f3, const double* r3,
                             const R* f4, const double* r4, const R* f5, const double* r5,
                             bool with_drops, const R* f1, const double* r1, double& liq) {
  double ice = 0.0;
  liq = 0.0;
  for (int k = 1; k <= FSBM_NKR; ++k) {
    if (with_drops) {
      double fk = (double)f1[k], rk = r1[k];
      liq = liq + fk * rk * rk;
    }
    for (int ic = 1; ic <= FSBM_ICEMAX; ++ic) {
      double fk = (double)A2(f2, k, ic), rk = A2(r2d, k, ic);
      ice = ice + fk * rk * rk;
    }
    double f3k = (double)f3[k], f4k = (double)f4[k], f5k = (double)f5[k];
    double r3k = r3[k], r4k = r4[k], r5k = r5[k];
    double a3 = f3k * r3k * r3k, a4 = f4k * r4k * r4k, a5 = f5k * r5k * r5k;
    ice = ice + a3;
    ice = ice + a4;
    ice = ice + a5;
  }
  return ice;
}

// ---------------------------------------------------------------------------
// ONECOND2 (:7145-7714): ice only.  ff2/psi2 A2 layout.  vr2 A2 (33x3).
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE int onecond2(double& tt, double& qq, double pp, R ror, const R* vr2, const R* vr3,
                          const R* vr4, const R* vr5, double& del1n, double& del2n, R* ff2, R* psi2,
                          R* ff3, R* psi3, R* ff4, R* psi4, R* ff5, R* psi5, const FsbmTab<R>& T,
                          R bb1_my, R bb2_my, R col, R dtcond, int& isym1, int* isym2, int& isym3,
                          int& isym4, int& isym5) {
  const R al1 = RL(2500.), al2 = RL(2834.), gam = RL(1.E-4), epsdel = RL(0.1E-03);
  FsbmSpecies<R> s3, s4, s5;
  fsbm_species_init<R>(s3, T.xs, vr3, T.rsec, T.ro3bl);
  fsbm_species_init<R>(s4, T.xg, vr4, T.rgec, T.ro4bl);
  fsbm_species_init<R>(s5, T.xh, vr5, T.rhec, T.ro5bl);
  double r2d[FSBM_NKR * FSBM_ICEMAX + 1], r2nd[FSBM_NKR * FSBM_ICEMAX + 1], vr2_d[FSBM_NKR * FSBM_ICEMAX + 1];
  R b21_my[FSBM_NKR * FSBM_ICEMAX + 1], fi2[FSBM_NKR * FSBM_ICEMAX + 1];
  R riec[FSBM_NKR * FSBM_ICEMAX + 1], ro2bl[FSBM_NKR * FSBM_ICEMAX + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr)
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      A2(r2d, kr, ice) = (double)T2(T.xi, kr, ice, FSBM_NKR);
      A2(r2nd, kr, ice) = A2(r2d, kr, ice);
      A2(riec, kr, ice) = T2(T.riec, kr, ice, FSBM_NKR);
      A2(ro2bl, kr, ice) = T2(T.ro2bl, kr, ice, FSBM_NKR);
      A2(b21_my, kr, ice) = RL(0.0);
      A2(vr2_d, kr, ice) = (double)A2(vr2, kr, ice);
    }
  R sfni1[FSBM_ICEMAX + 1];
  sfni1[1] = sfni1[2] = sfni1[3] = RL(0.0);
  R sfn31 = RL(0.0), sfn41 = RL(0.0), sfn51 = RL(0.0);
  double col3 = (double)(RL(3.0) * col);
  R dt = dtcond, dtt = dtcond;
  R cwhucm = RL(0.), xrad = RL(0.);
  R b6 = cwhucm * gam - xrad;
  R b8i = RL(1.) / ror;
  double rori = (double)(RL(1.) / ror);
  double tpn = tt, qpn = qq;
  R timenew = RL(0.);
  double dal2 = (double)al2;
  for (;;) {
    R timerev = dt - timenew;
    R del1 = (R)del1n, del2 = (R)del2n, del2s = (R)del2n, del1s = (R)del1n;
    double tps = tpn, qps = qpn;
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      s3.fi[kr] = psi3[kr];
      s4.fi[kr] = psi4[kr];
      s5.fi[kr] = psi5[kr];
      for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(fi2, kr, ice) = A2(psi2, kr, ice);
    }
    fsbm_ice_rates<R>(tps, pp, isym2, isym3, isym4, isym5, r2d, vr2_d, riec, ro2bl, b21_my, fi2, sfni1,
                      s3, s4, s5, sfn31, sfn41, sfn51, b8i, col);
    R sfnii1 = sfni1[1] + sfni1[2] + sfni1[3];
    R sfn21 = sfnii1 + sfn31 + sfn41 + sfn51;
    R sfnl = RL(0.0);
    R sfn22 = RL(0.0);
    R sfni = sfn21 + sfn22;
    R b5l = (R)((double)bb1_my / tps / tps);
    R b5i = (R)((double)bb2_my / tps / tps);
    R b7l = b5l * b6, b7i = b5i * b6;
    R dopl = RL(1.) + del1s, dopi = RL(1.) + del2s;
    R operq = (R)fsbm_oper2<R>(qps);
    R rw = (operq + b5l * al1) * dopl * sfnl;
    R qw = b7l * dopl;
    R pw = (operq + b5i * al1) * dopi * sfnl;
    R ri = (operq + b5l * al2) * dopl * sfni;
    R pi = (operq + b5i * al2) * dopi * sfni;
    R qi = b7i * dopi;
    (void)qw;
    (void)qi;
    int kcond = 20;
    if (del2n > 0.0) kcond = 21;
    if (rw != rw || pw != pw) return 7441;
    R dtnewl = dt;
    dtnewl = fmin2(dtnewl, timerev);
    timenew = timenew + dtnewl;
    dtt = dtnewl;
    if (dtt < (R)0.0) return kcond == 21 ? 7452 : 7535;
    double del1_d = del1, del2_d = del2, rw_d = rw, pw_d = pw, ri_d = ri, pi_d = pi;
    double d1n_d = 0.0, d2n_d = 0.0;
    int e = jersupsat_ks<R>(del1_d, del2_d, del1n, del2n, rw_d, pw_d, ri_d, pi_d, dtt, d1n_d, d2n_d,
                            RL(0.0), RL(0.0), isym1, isym2, isym3, isym4, isym5, 0);
    if (e) return e;
    del1 = (R)del1_d;
    del2 = (R)del2_d;
    R d2n = (R)d2n_d;
    e = fsbm_ice_remaps<R>(isym2, isym3, isym4, isym5, r2d, r2nd, b21_my, fi2, psi2, s3, psi3, s4,
                           psi4, s5, psi5, d2n, col);
    if (e) return e;
    if (kcond == 21) {
      if ((del2 > (R)0 && del2n < 0) && fabs_(del2n) > (double)epsdel) return 7524;
    } else {
      if ((del2 < (R)0 && del2n > 0) && fabs_(del2n) > (double)epsdel) return 7610;
    }
    double liq;
    double rmassibb = fsbm_ice_mass_sum<R>(fi2, r2d, s3.fi, s3.rd, s4.fi, s4.rd, s5.fi, s5.rd, false,
                                           (const R*)0, (const double*)0, liq);
    rmassibb = rmassibb * col3 * rori;
    if (rmassibb < (double)RL(0.0)) rmassibb = (double)RL(0.0);
    double rmassiaa = fsbm_ice_mass_sum<R>(psi2, r2d, psi3, s3.rd, psi4, s4.rd, psi5, s5.rd, false,
                                           (const R*)0, (const double*)0, liq);
    rmassiaa = rmassiaa * col3 * rori;
    if (rmassiaa < (double)RL(0.0)) rmassiaa = (double)RL(0.0);
    double delmassi1 = rmassiaa - rmassibb;
    qpn = qps - delmassi1;
    tpn = tps + dal2 * delmassi1;
    if (fabs_(dal2 * delmassi1) > (double)RL(5.0)) return 7694;
    if (!(timenew < dt)) break;
  }
  tt = tpn;
  qq = qpn;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(ff2, kr, ice) = A2(psi2, kr, ice);
    ff3[kr] = psi3[kr];
    ff4[kr] = psi4[kr];
    ff5[kr] = psi5[kr];
  }
  return FSBM_OK;
}

// ---------------------------------------------------------------------------
// ONECOND3 (:7716-8261): mixed phase.
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE int onecond3(double& tt, double& qq, double pp, R ror, const R* vr1, const R* vr2,
                          const R* vr3, const R* vr4, const R* vr5, double& del1n, double& del2n,
                          R* ff1, R* psi1, R* ff2, R* psi2, R* ff3, R* psi3, R* ff4, R* psi4, R* ff5,
                          R* psi5, const FsbmTab<R>& T, R bb1_my, R bb2_my, R col, R dtcond,
                          int& isym1, int* isym2, int& isym3, int& isym4, int& isym5, int rederived) {
  const R al1 = RL(2500.), al2 = RL(2834.);
  FsbmSpecies<R> s3, s4, s5;
  fsbm_species_init<R>(s3, T.xs, vr3, T.rsec, T.ro3bl);
  fsbm_species_init<R>(s4, T.xg, vr4, T.rgec, T.ro4bl);
  fsbm_species_init<R>(s5, T.xh, vr5, T.rhec, T.ro5bl);
  double r1d[FSBM_NKR + 1], r1nd[FSBM_NKR + 1], vr1_d[FSBM_NKR + 1];
  R b11_my[FSBM_NKR + 1], fi1[FSBM_NKR + 1], fl1[FSBM_NKR + 1], r1[FSBM_NKR + 1], rlec[FSBM_NKR + 1],
      ro1bl[FSBM_NKR + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    r1[kr] = T.xl[kr - 1];
    rlec[kr] = T.rlec[kr - 1];
    ro1bl[kr] = T.ro1bl[kr - 1];
    b11_my[kr] = RL(0.0);
    r1d[kr] = (double)r1[kr];
    r1nd[kr] = r1d[kr];
    vr1_d[kr] = (double)vr1[kr];
  }
  double r2d[FSBM_NKR * FSBM_ICEMAX + 1], r2nd[FSBM_NKR * FSBM_ICEMAX + 1], vr2_d[FSBM_NKR * FSBM_ICEMAX + 1];
  R b21_my[FSBM_NKR * FSBM_ICEMAX + 1], fi2[FSBM_NKR * FSBM_ICEMAX + 1];
  R riec[FSBM_NKR * FSBM_ICEMAX + 1], ro2bl[FSBM_NKR * FSBM_ICEMAX + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr)
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      A2(r2d, kr, ice) = (double)T2(T.xi, kr, ice, FSBM_NKR);
      A2(r2nd, kr, ice) = A2(r2d, kr, ice);
      A2(riec, kr, ice) = T2(T.riec, kr, ice, FSBM_NKR);
      A2(ro2bl, kr, ice) = T2(T.ro2bl, kr, ice, FSBM_NKR);
      A2(b21_my, kr, ice) = RL(0.0);
      A2(vr2_d, kr, ice) = (double)A2(vr2, kr, ice);
    }
  R sfn11 = RL(0.0);
  R sfni1[FSBM_ICEMAX + 1];
  sfni1[1] = sfni1[2] = sfni1[3] = RL(0.0);
  R sfn31 = RL(0.0), sfn41 = RL(0.0), sfn51 = RL(0.0);
  R timenew = RL(0.0);
  R dt = dtcond, dtt = dtcond;
  R b6 = RL(0.);
  R b8l = RL(1.) / ror, b8i = RL(1.) / ror;
  double rori = 1.0 / (double)ror;
  double col3 = 3.0 * (double)col;
  double tpn = tt, qpn = qq;
  double dal1 = (double)al1, dal2 = (double)al2;
  R sfndummy[4];
  sfndummy[1] = sfndummy[2] = sfndummy[3] = RL(0.0);
  for (;;) {
    if ((tpn - (double)RL(273.15)) >= (double)RL(-0.187)) break;  // GO TO 17
    R timerev = dt - timenew;
    R del1 = (R)del1n, del2 = (R)del2n, del1s = (R)del1n, del2s = (R)del2n;
    double tps = tpn, qps = qpn;
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      fi1[kr] = psi1[kr];
      s3.fi[kr] = psi3[kr];
      s4.fi[kr] = psi4[kr];
      s5.fi[kr] = psi5[kr];
      for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(fi2, kr, ice) = A2(psi2, kr, ice);
    }
    if (isym1 == 1) {
      for (int kr = 1; kr <= FSBM_NKR; ++kr) fl1[kr] = RL(0.0);
      jerrate_ks<R>(r1d, tps, pp, vr1_d, rlec, ro1bl, b11_my, 1, 1, fl1);
      sfndummy[1] = sfn11;
      jertimesc_ks<R>(fi1, r1d, sfndummy, b11_my, b8l, 1, col);
      sfn11 = sfndummy[1];
    }
    fsbm_ice_rates<R>(tps, pp, isym2, isym3, isym4, isym5, r2d, vr2_d, riec, ro2bl, b21_my, fi2, sfni1,
                      s3, s4, s5, sfn31, sfn41, sfn51, b8i, col);
    R sfnii1 = sfni1[1] + sfni1[2] + sfni1[3];
    R sfn21 = sfnii1 + sfn31 + sfn41 + sfn51;
    R sfn12 = RL(0.0);
    R sfnl = sfn11 + sfn12;
    R sfn22 = RL(0.0);
    R sfni = sfn21 + sfn22;
    R b5l = (R)((double)bb1_my / tps / tps);
    R b5i = (R)((double)bb2_my / tps / tps);
    R b7l = b5l * b6, b7i = b5i * b6;
    R dopl = RL(1.) + del1s, dopi = RL(1.) + del2s;
    R rw = (R)((fsbm_oper2<R>(qps) + (double)(b5l * al1)) * (double)dopl * (double)sfnl);
    R qw = b7l * dopl;
    R pw = (R)((fsbm_oper2<R>(qps) + (double)(b5i * al1)) * (double)dopi * (double)sfnl);
    R ri = (R)((fsbm_oper2<R>(qps) + (double)(b5l * al2)) * (double)dopl * (double)sfni);
    R pi = (R)((fsbm_oper2<R>(qps) + (double)(b5i * al2)) * (double)dopi * (double)sfni);
    R qi = b7i * dopi;
    (void)qw;
    (void)qi;
    if (rw != rw || pw != pw) return 7993;
    int kcond = 50;
    if (del1n < (double)RL(0.0) && del2n < (double)RL(0.0)) kcond = 30;
    if (del1n > (double)RL(0.0) && del2n > (double)RL(0.0)) kcond = 31;
    if (del1n < (double)RL(0.0) && del2n > (double)RL(0.0)) kcond = 32;
    R dtnewl = dt;
    dtnewl = fmin2(dtnewl, timerev);
    timenew = timenew + dtnewl;
    dtt = dtnewl;
    if (kcond == 50) break;  // anti-Bergeron: no diffusional growth (GO TO 17)
    if (dtt < (R)0.0) return 8049;
    double del1_d = del1, del2_d = del2, rw_d = rw, pw_d = pw, ri_d = ri, pi_d = pi;
    double d1n_d = 0.0, d2n_d = 0.0;
    int e = jersupsat_ks<R>(del1_d, del2_d, del1n, del2n, rw_d, pw_d, ri_d, pi_d, dtt, d1n_d, d2n_d,
                            RL(0.0), RL(0.0), isym1, isym2, isym3, isym4, isym5, rederived);
    if (e) return e;
    R d1n = (R)d1n_d, d2n = (R)d2n_d;
    if (isym1 == 1) {
      int idrop = isym1;
      for (int kr = 1; kr <= FSBM_NKR; ++kr) fl1[kr] = RL(0.0);
      e = jerdfun_ks<R>(r1d, r1nd, b11_my, fi1, psi1, d1n, isym1, 1, 1, idrop, col);
      if (e) return e;
    }
    e = fsbm_ice_remaps<R>(isym2, isym3, isym4, isym5, r2d, r2nd, b21_my, fi2, psi2, s3, psi3, s4,
                           psi4, s5, psi5, d2n, col);
    if (e) return e;
    double rmasslbb, rmasslaa;
    double rmassibb = fsbm_ice_mass_sum<R>(fi2, r2d, s3.fi, s3.rd, s4.fi, s4.rd, s5.fi, s5.rd, true,
                                           fi1, r1d, rmasslbb);
    rmassibb = rmassibb * col3 * rori;
    if (rmassibb < (double)RL(0.0)) rmassibb = (double)RL(0.0);
    rmasslbb = rmasslbb * col3 * rori;
    if (rmasslbb < (double)RL(0.0)) rmasslbb = (double)RL(0.0);
    for (int k = 1; k <= FSBM_NKR; ++k)
      for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(fi2, k, ice) = A2(psi2, k, ice);
    double rmassiaa = fsbm_ice_mass_sum<R>(psi2, r2d, psi3, s3.rd, psi4, s4.rd, psi5, s5.rd, true,
                                           psi1, r1d, rmasslaa);
    rmassiaa = rmassiaa * col3 * rori;
    if (rmassiaa <= (double)RL(0.0)) rmassiaa = (double)RL(0.0);
    rmasslaa = rmasslaa * col3 * rori;
    if (rmasslaa < (double)RL(0.0)) rmasslaa = (double)RL(0.0);
    double delmassl1 = rmasslaa - rmasslbb;
    double delmassi1 = rmassiaa - rmassibb;
    double deltaq1 = delmassl1 + delmassi1;
    qpn = qps - deltaq1;
    tpn = tps + dal1 * delmassl1 + dal2 * delmassi1;
    if (fabs_(dal1 * delmassl1 + dal2 * delmassi1) > (double)RL(5.0)) return 8239;
    if (!(timenew < dt)) break;
  }
  tt = tpn;
  qq = qpn;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    ff1[kr] = psi1[kr];
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(ff2, kr, ice) = A2(psi2, kr, ice);
    ff3[kr] = psi3[kr];
    ff4[kr] = psi4[kr];
    ff5[kr] = psi5[kr];
  }
  return FSBM_OK;
}

// ---------------------------------------------------------------------------
// COAL_BOTT_NEW (:8263-8642).  ff2 A2 layout.  fliqfr/frimfr are zero on
// every FAST_SBM call (:4770-4773) and are not carried.
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE int coal_bott_new(R* ff1r, R* ff2r, R* ff3r, R* ff4r, R* ff5r, double& tt, double& qq,
                               double pp, R rho, R dt_coll, R tcrit, const FsbmTab<R>& T,
                               const FsbmFlags& F, double& del1in, double& del2in, R& colleff,
                               double* rime_mass) {
  const R col = FSBM_COL;
  const R alcr = RL(0.5), alcr_g = RL(100.0);
  const double g_lim = 1.0e-19 * 1.0e3;
  const double prdkrn1 = 1.0;
  double g1[FSBM_NKR + 1], g2[FSBM_NKR * FSBM_ICEMAX + 1], g3[FSBM_NKR + 1], g4[FSBM_NKR + 1],
      g5[FSBM_NKR + 1];
  double fl1[FSBM_NKR + 1], rf1[FSBM_NKR + 1], rf3[FSBM_NKR + 1], rf4[FSBM_NKR + 1], rf5[FSBM_NKR + 1];
  double xl_mg[FSBM_NKR + 1], xs_mg[FSBM_NKR + 1], xg_mg[FSBM_NKR + 1], xh_mg[FSBM_NKR + 1];
  double dm_rime[FSBM_NKR + 1];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    xl_mg[kr] = T.xl_mg[kr - 1];
    xs_mg[kr] = T.xs_mg[kr - 1];
    xg_mg[kr] = T.xg_mg[kr - 1];
    xh_mg[kr] = T.xh_mg[kr - 1];
  }
  int icol_drop_brk = 0, icol_drop = 0, icol_snow = 0, icol_graupel = 0, icol_hail = 0;
  R t_new = (R)tt;
  R pp_r = (R)pp;
  double prdkrn = modkrn_case11<R>(tt);
  colleff = (R)prdkrn;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    R xl = T.xl[kr - 1], xs = T.xs[kr - 1], xg = T.xg[kr - 1], xh = T.xh[kr - 1];
    g1[kr] = (double)(ff1r[kr] * RL(3.) * xl * xl * RL(1.E3));
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      R xi = T2(T.xi, kr, ice, FSBM_NKR);
      A2(g2, kr, ice) = (ice == 1) ? (double)(A2(ff2r, kr, ice) * (R)3 * xi * xi * RL(1.e3))
                                   : (double)(A2(ff2r, kr, ice) * RL(3.) * xi * xi * RL(1.e3));
    }
    g3[kr] = (double)(ff3r[kr] * RL(3.) * xs * xs * RL(1.e3));
    g4[kr] = (double)(ff4r[kr] * RL(3.) * xg * xg * RL(1.e3));
    g5[kr] = (double)(ff5r[kr] * RL(3.) * xh * xh * RL(1.e3));
    if (kr > FSBM_KRMIN_BREAKUP && g1[kr] > g_lim) icol_drop_brk = 1;
    if (g1[kr] > g_lim) icol_drop = 1;
    if (g3[kr] > g_lim) icol_snow = 1;
    if (g4[kr] > g_lim) icol_graupel = 1;
    if (g5[kr] > g_lim) icol_hail = 1;
  }
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    fl1[kr] = (double)RL(1.0);
    rf1[kr] = (double)RL(1.0);
    rf3[kr] = 0.0;  // FRIMFR_S, zeroed by FAST_SBM before the call
    rf4[kr] = (double)RL(0.0);
    rf5[kr] = (double)RL(0.0);
  }
  R cont_init_drop, cont_init_ice;
  {
    double s1 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s1 = s1 + g1[kr];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s3 = s3 + g3[kr];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s4 = s4 + g4[kr];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s5 = s5 + g5[kr];
    cont_init_drop = (R)s1;
    cont_init_ice = (R)(s3 + s4 + s5);
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      double s2 = 0.0;
      for (int kr = 1; kr <= FSBM_NKR; ++kr) s2 = s2 + A2(g2, kr, ice);
      cont_init_ice = (R)((double)cont_init_ice + s2);
    }
  }
  cont_init_drop = col * cont_init_drop * RL(1.e-3);
  cont_init_ice = col * cont_init_ice * RL(1.e-3);
  R alwc = cont_init_drop * RL(1.e6);
  int e;
  CkLL<R> cwll = {&T, pp_r, dt_coll};
  if (icol_drop == 1) {
    for (int kr = 1; kr <= FSBM_NKR; ++kr) fl1[kr] = (double)RL(1.0);
    double drop_mass_before=F.collision_mass_conserved ? fsbm_gmass(g1,FSBM_NKR) : 0;
    e = coll_xxx_lwf(g1, fl1, cwll, xl_mg, T.chucm, T.ima, 1.0, FSBM_NKR);
    if (e) return e;
    if(F.collision_mass_conserved) fsbm_gconserve(g1,FSBM_NKR,drop_mass_before);
    if (icol_drop_brk == 1) {
      int ndiv = 1;
      double gdumb[FSBM_JMAX + 1], xl_dumb[FSBM_JMAX + 1];
      double break_drop_bef = 0.0;
      for (;;) {  // label 10
        bool restart = false, skip = false;
        for (int it = 1; it <= ndiv; ++it) {
          double dtbreakup = (double)(dt_coll / (R)ndiv);
          if (it == 1) {
            for (int kr = 1; kr <= FSBM_JMAX; ++kr) {
              gdumb[kr] = g1[kr] * 1.0e-3;
              xl_dumb[kr] = xl_mg[kr] * 1.0e-3;
            }
            break_drop_bef = 0.0;
            for (int kr = 1; kr <= FSBM_JMAX; ++kr) break_drop_bef = break_drop_bef + g1[kr] * 1.0e-3;
          }
          double breakup_mass_before=F.collision_mass_conserved ? fsbm_gmass(gdumb,FSBM_JMAX) : 0;
          e = coll_breakup_ks<R>(gdumb, xl_dumb, FSBM_JMAX, dtbreakup, FSBM_JBREAK, T.pkij, T.qkj,
                                 FSBM_NKR, FSBM_NKR);
          if (e) return e;
          if(F.collision_mass_conserved) fsbm_gconserve(gdumb,FSBM_JMAX,breakup_mass_before);
        }
        for (int kr = 1; kr <= FSBM_NKR; ++kr) {
          R xl = T.xl[kr - 1];
          ff1r[kr] = (R)((1.0e3 * gdumb[kr]) / (double)(RL(3.0) * xl * xl * RL(1.E3)));
          if (ff1r[kr] < (R)0.0) {
            if (ndiv < 8) {
              ndiv = 2 * ndiv;
              restart = true;
            } else {
              skip = true;  // GO TO 11
            }
            break;
          }
          if (ff1r[kr] != ff1r[kr]) return 8414;
        }
        if (skip) break;
        if (restart) continue;
        double break_drop_aft = 0.0;
        for (int kr = 1; kr <= FSBM_JMAX; ++kr) break_drop_aft = break_drop_aft + gdumb[kr];
        double break_drop_per = break_drop_aft / break_drop_bef;
        if (break_drop_per > 1.001) {
          ndiv = ndiv * 2;
          if (ndiv > F.breakup_retry_cap) return 8423;  // D7: WRF doubles without limit
          continue;
        }
        for (int kr = 1; kr <= FSBM_JMAX; ++kr) g1[kr] = gdumb[kr] * 1.0e3;
        break;
      }
    }
  }
  // label 11: the cold collision block
  if (tt <= (double)RL(273.15)) {
    R conc_old = RL(0.0), conc_new = RL(0.0);
    if (icol_drop == 1) {
      if (icol_snow == 1) {
        CkIce<R> cwls = {T.ywls, pp_r, dt_coll, 0}, cwsl = {T.ywls, pp_r, dt_coll, 1};
        for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf5[kr] = 0.0; rf4[kr] = 0.0; }
        if (F.hail) e = coll_xyz_lwf(g1, g3, g5, rf1, rf3, rf5, cwls, xl_mg, xs_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 0);
        else e = coll_xyz_lwf(g1, g3, g4, rf1, rf3, rf4, cwls, xl_mg, xs_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 0);
        if (e) return e;
        for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf5[kr] = 0.0; rf4[kr] = 0.0; }
        if (alwc < alcr) {
          e = coll_xyx_lwf(g3, g1, rf3, rf1, cwsl, xs_mg, xl_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1, dm_rime);
        } else {
          if (F.hail) e = coll_xyz_lwf(g3, g1, g5, rf3, rf1, rf5, cwsl, xs_mg, xl_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1);
          else e = coll_xyz_lwf(g3, g1, g4, rf3, rf1, rf4, cwsl, xs_mg, xl_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1);
        }
        if (e) return e;
      }
      if (icol_graupel == 1) {
        CkIce<R> cwlg = {T.ywlg, pp_r, dt_coll, 0}, cwgl = {T.ywlg, pp_r, dt_coll, 1};
        if (alwc < alcr_g) {
          for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf4[kr] = 0.0; }
          e = coll_xyy_lwf(g1, g4, rf1, rf4, cwlg, xl_mg, xg_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 0);
          if (e) return e;
          conc_old = RL(0.0);
          conc_new = RL(0.0);
          for (int kr = FSBM_KR_ICEMPL; kr <= FSBM_NKR; ++kr)
            conc_old = (R)((double)conc_old + (double)col * g1[kr] / xl_mg[kr]);
          for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf4[kr] = 0.0; }
          e = coll_xyx_lwf(g4, g1, rf4, rf1, cwgl, xg_mg, xl_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1, dm_rime);
          if (e) return e;
        } else {
          for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf5[kr] = 0.0; rf4[kr] = 0.0; }
          e = coll_xyz_lwf(g1, g4, g5, rf1, rf4, rf5, cwlg, xl_mg, xg_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1);
          if (e) return e;
          conc_old = RL(0.0);
          conc_new = RL(0.0);
          for (int kr = FSBM_KR_ICEMPL; kr <= FSBM_NKR; ++kr)
            conc_old = (R)((double)conc_old + (double)col * g1[kr] / xl_mg[kr]);
          for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf5[kr] = 0.0; rf4[kr] = 0.0; }
          e = coll_xyz_lwf(g4, g1, g5, rf4, rf1, rf5, cwgl, xg_mg, xl_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1);
          if (e) return e;
        }
      }
      if (icol_hail == 1) {
        CkIce<R> cwlh = {T.ywlh, pp_r, dt_coll, 0}, cwhl = {T.ywlh, pp_r, dt_coll, 1};
        for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf5[kr] = 0.0; }
        e = coll_xyy_lwf(g1, g5, rf1, rf5, cwlh, xl_mg, xh_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 0);
        if (e) return e;
        conc_old = RL(0.0);
        conc_new = RL(0.0);
        for (int kr = FSBM_KR_ICEMPL; kr <= FSBM_NKR; ++kr)
          conc_old = (R)((double)conc_old + (double)col * g1[kr] / xl_mg[kr]);
        for (int kr = 1; kr <= FSBM_NKR; ++kr) { rf1[kr] = 1.0; rf5[kr] = 0.0; }
        e = coll_xyx_lwf(g5, g1, rf5, rf1, cwhl, xh_mg, xl_mg, T.chucm, T.ima, prdkrn1, FSBM_NKR, 1, dm_rime);
        if (e) return e;
      }
      if (icol_graupel == 1 || icol_hail == 1) {
        if (tt >= (double)RL(265.15) && tt <= (double)tcrit) {
          for (int kr = FSBM_KR_ICEMPL; kr <= FSBM_NKR; ++kr)
            conc_new = (R)((double)conc_new + (double)col * g1[kr] / xl_mg[kr]);
          R dconc = conc_old - conc_new;
          R conc_icempl = RL(0.0);
          if (tt <= (double)RL(268.15))
            conc_icempl = (R)((double)(dconc * RL(4.e-3)) * ((double)RL(265.15) - tt) /
                              (double)(RL(265.15) - RL(268.15)));
          if (tt > (double)RL(268.15))
            conc_icempl = (R)((double)(dconc * RL(4.e-3)) * ((double)tcrit - tt) /
                              (double)(tcrit - RL(268.15)));
          double splinter_mass=(double)conc_icempl * xs_mg[1] / (double)col;
          if(F.collision_mass_conserved && splinter_mass>0) {
            double* parent=F.hail ? g5 : g4;
            double available=fsbm_gmass(parent,FSBM_NKR);
            splinter_mass=fmin2(splinter_mass,available);
            fsbm_gconserve(parent,FSBM_NKR,available-splinter_mass);
          }
          g3[1] = g3[1] + splinter_mass;
        }
      }
    }
    if (icol_snow == 1) {
      CkIce<R> cwss = {T.ywss, pp_r, dt_coll, 0};
      double snow_mass_before=F.collision_mass_conserved ? fsbm_gmass(g3,FSBM_NKR) : 0;
      e = coll_xxx_lwf(g3, rf3, cwss, xs_mg, T.chucm, T.ima, prdkrn, FSBM_NKR);
      if (e) return e;
      if(F.collision_mass_conserved) fsbm_gconserve(g3,FSBM_NKR,snow_mass_before);
    }
  }
  R cont_fin_drop = RL(0.), cont_fin_ice = RL(0.);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    cont_fin_drop = (R)((double)cont_fin_drop + g1[kr]);
    cont_fin_ice = (R)((double)cont_fin_ice + g3[kr] + g4[kr] + g5[kr]);
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) cont_fin_ice = (R)((double)cont_fin_ice + A2(g2, kr, ice));
  }
  cont_fin_drop = col * cont_fin_drop * RL(1.e-3);
  cont_fin_ice = col * cont_fin_ice * RL(1.e-3);
  R deldrop = cont_init_drop - cont_fin_drop;
  if (t_new <= RL(273.15)) {
    if (deldrop > RL(0.0)) {
      R rime_heat = F.fusion_heat_334 ? RL(334.) : RL(320.);  // D13
      t_new = t_new + rime_heat * deldrop / rho;
      if (rime_mass) *rime_mass += (double)(deldrop / rho);
      R es1n = (R)polysvp<R>(t_new, 0);
      R es2n = (R)polysvp<R>(t_new, 1);
      R ew1n = (R)(qq * pp / ((double)RL(0.622) + (double)RL(0.378) * qq));
      del1in = (double)(ew1n / es1n - RL(1.0));
      del2in = (double)(ew1n / es2n - RL(1.0));
    } else {
      if (fabs_(deldrop) > cont_init_drop * RL(0.05)) return 8599;
    }
  }
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    R xl = T.xl[kr - 1], xs = T.xs[kr - 1], xg = T.xg[kr - 1], xh = T.xh[kr - 1];
    ff1r[kr] = (R)(g1[kr] / (double)(RL(3.) * xl * xl * RL(1.E3)));
    if (ff1r[kr] != ff1r[kr] || ff1r[kr] < (R)0.0) return 8610;
    ff3r[kr] = (R)(g3[kr] / (double)(RL(3.) * xs * xs * RL(1.e3)));
    if (ff3r[kr] != ff3r[kr] || ff3r[kr] < (R)0.0) return 8614;
    if (!F.hail) {
      ff4r[kr] = (R)(g4[kr] / (double)(RL(3.) * xg * xg * RL(1.e3)));
      if (ff4r[kr] != ff4r[kr] || ff4r[kr] < (R)0.0) return 8619;
    } else {
      ff5r[kr] = (R)(g5[kr] / (double)(RL(3.) * xh * xh * RL(1.e3)));
      if (ff5r[kr] != ff5r[kr] || ff5r[kr] < (R)0.0) return 8624;
    }
  }
  if (fabs_(tt - (double)t_new) > (double)RL(5.0)) return 8636;
  tt = (double)t_new;
  return FSBM_OK;
}

// ---------------------------------------------------------------------------
// The point part of FAST_SBM (:4514-4885).  bins: the 142 values of one
// point in the scheme's units (after the entry conversion), stride sn.
// Mass observer used only by the host water-budget probe.
template <class R>
FHD double fsbm_point_water(const FsbmTab<R>& T, R rho, double q,
 const R* f1,const R* f2,const R* f3,const R* f4,const R* f5) {
 double m=0;
 for(int k=1;k<=FSBM_NKR;++k) {
  double xl=T.xl[k-1],xs=T.xs[k-1],xg=T.xg[k-1],xh=T.xh[k-1];
  m+=double(f1[k])*xl*xl+double(f3[k])*xs*xs+double(f4[k])*xg*xg+double(f5[k])*xh*xh;
  for(int ice=1;ice<=3;++ice) { double xi=T2(T.xi,k,ice,33); m+=double(A2(f2,k,ice))*xi*xi; }
 }
 return q+3.0*double(FSBM_COL)*m/double(rho);
}

// Updates t_new and qv; returns a WRF line on a fatal check.
// ---------------------------------------------------------------------------
template <class R>
FHD_NOINLINE int fsbm_point(const FsbmTab<R>& T, const FsbmFlags& F, R* bins, long sn, R t_old,
                            R& t_new, R qv_old, R& qv, R pcgs, R rhocgs, R w_stag_my,
                            int is_kz_cloud_base, R dsupice, R dt, R& dtcond,
                            FsbmPointDiag* dg, double* water_budget=nullptr) {
  const R col = FSBM_COL;
  const R aa1_my = FSBM_AA1_MY, bb1_my = FSBM_BB1_MY, aa2_my = FSBM_AA2_MY, bb2_my = FSBM_BB2_MY;
  const R afreezmy = RL(0.3333E-04), bfreezmy = RL(0.6600E00), bfreezmax = RL(0.6600E00);
  const R tcrit = RL(2.7015E02);
  const int ncond = 3;
  R ff1r[FSBM_NKR + 1], ff3r[FSBM_NKR + 1], ff4r[FSBM_NKR + 1], ff5r[FSBM_NKR + 1];
  R ff2r[FSBM_NKR * FSBM_ICEMAX + 1], ff2in[FSBM_NKR * FSBM_ICEMAX + 1];
  R ff1in[FSBM_NKR + 1], ff3in[FSBM_NKR + 1], ff4in[FSBM_NKR + 1], ff5in[FSBM_NKR + 1];
  R fccn[FSBM_NKR_AERO + 1];
  R vr1_z[FSBM_NKR + 1], vr3_z[FSBM_NKR + 1], vr4_z[FSBM_NKR + 1], vr5_z[FSBM_NKR + 1];
  R vr2_z[FSBM_NKR * FSBM_ICEMAX + 1];
  R factor_p = (R)fx_sqrt(1.0e6 / (double)pcgs);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice)
      A2(vr2_z, kr, ice) = T2(T.vr2, kr, ice, FSBM_NKR) * factor_p;
    vr1_z[kr] = T.vr1[kr - 1] * factor_p;
    vr3_z[kr] = T.vr3[kr - 1] * factor_p;
    vr4_z[kr] = T.vr4[kr - 1] * factor_p;
    vr5_z[kr] = T.vr5[kr - 1] * factor_p;
  }
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    ff1r[kr] = bins[(FSBM_OFF_DROP + kr - 1) * sn];
    if (ff1r[kr] < (R)0.0) ff1r[kr] = RL(0.0);
  }
  for (int kr = 1; kr <= FSBM_NKR_AERO; ++kr) {
    fccn[kr] = bins[(FSBM_OFF_AERO + kr - 1) * sn];
    if (fccn[kr] < (R)0.0) fccn[kr] = RL(0.0);
  }
  for (int k = 1; k <= FSBM_NKR * FSBM_ICEMAX; ++k) ff2r[k] = RL(0.0);
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    ff3r[kr] = bins[(FSBM_OFF_SNOW + kr - 1) * sn];
    if (ff3r[kr] < (R)0.0) ff3r[kr] = RL(0.0);
  }
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    R v = bins[(FSBM_OFF_DENSE + kr - 1) * sn];
    if (v < (R)0.0) v = RL(0.0);
    if (F.hail) { ff5r[kr] = v; ff4r[kr] = RL(0.0); }
    else { ff4r[kr] = v; ff5r[kr] = RL(0.0); }
  }
  int e;
  double observed=0;
#define POINT_WATER(q) fsbm_point_water<R>(T,rhocgs,(q),ff1r,ff2r,ff3r,ff4r,ff5r)
  if (t_old > RL(193.15)) {
    double tt = (double)t_old;
    double qq = (double)qv_old;
    if (qq <= (double)RL(0.0)) qq = 1.0e-10;
    double pp = (double)pcgs;
    double tta = (double)t_new;
    double qqa = (double)qv;
    if (qqa <= 0) qqa = 1.0e-10;
    double es1n = (double)aa1_my * fx_exp(-(double)bb1_my / tt);
    double es2n = (double)aa2_my * fx_exp(-(double)bb2_my / tt);
    double ew1n = qq * pp / ((double)RL(0.622) + (double)RL(0.378) * qq);
    double div1 = ew1n / es1n;
    double del1in = ew1n / es1n - (double)RL(1.);
    double div2 = ew1n / es2n;
    double del2in = ew1n / es2n - (double)RL(1.);
    es1n = (double)aa1_my * fx_exp(-(double)bb1_my / tta);
    es2n = (double)aa2_my * fx_exp(-(double)bb2_my / tta);
    ew1n = qqa * pp / ((double)RL(0.622) + (double)RL(0.378) * qqa);
    double div3 = ew1n / es1n;
    double del1ad = ew1n / es1n - (double)RL(1.);
    double div4 = ew1n / es2n;
    double del2ad = ew1n / es2n - (double)RL(1.);
    R sup2_old = (R)del2in;
    R s1 = RL(0.), s3 = RL(0.), s4 = RL(0.), s5 = RL(0.);
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s1 = s1 + ff1r[kr];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s3 = s3 + ff3r[kr];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s4 = s4 + ff4r[kr];
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s5 = s5 + ff5r[kr];
    if (del1ad > (double)RL(0.0) || del2ad > (double)RL(0.0) || (s1 + s3 + s4 + s5) > RL(1.0e-20)) {
      double del_t = (tta - tt) / ncond;
      double del_q = (qqa - qq) / ncond;
      int diffu = 1;
      if (div1 == div3) diffu = 0;
      if (div2 == div4) diffu = 0;
      R dtnew = RL(0.0);
      for (int ikl = 1; ikl <= ncond; ++ikl) {
        dtcond = fmin2(dt - dtnew, dtcond);
        dtnew = dtnew + dtcond;
        if (diffu != 0) {
          tt = tt + del_t;
          qq = qq + del_q;
          es1n = (double)aa1_my * fx_exp(-(double)bb1_my / tt);
          es2n = (double)aa2_my * fx_exp(-(double)bb2_my / tt);
          ew1n = qq * pp / ((double)RL(0.622) + (double)RL(0.378) * qq);
          div1 = ew1n / es1n;
          del1in = ew1n / es1n - (double)RL(1.0);
          div2 = ew1n / es2n;
          del2in = ew1n / es2n - (double)RL(1.0);
          if (div1 > div2 && tt <= 265) diffu = 0;
          if (diffu == 1) {
            if (del1in > (double)RL(0.0) || del2in > (double)RL(0.0)) {
              for (int kr = 1; kr <= FSBM_NKR; ++kr) {
                ff1in[kr] = ff1r[kr];
                for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) A2(ff2in, kr, ice) = A2(ff2r, kr, ice);
              }
              int is_this_cloudbase = 0;
              R sx = RL(0.);
              for (int kr = 1; kr <= FSBM_NKR; ++kr) sx = sx + ff1in[kr] * T.xl[kr - 1];
              if (is_kz_cloud_base && col * sx < RL(5.0)) is_this_cloudbase = 1;
              if(water_budget) observed=POINT_WATER(qq);
              e = jernucl01_ks<R>(ff1in, ff2in, fccn, T, tt, qq, rhocgs, pcgs, del1in, del2in, col,
                                  sup2_old, dsupice, w_stag_my, is_this_cloudbase,
                                  F.nucleation_water_sink, dg ? &dg->nucl : (FsbmNuclTally*)0);
              if (e) return e;
              for (int kr = 1; kr <= FSBM_NKR; ++kr) {
                ff1r[kr] = ff1in[kr];
                for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
                  ff3r[kr] = ff3r[kr] + A2(ff2in, kr, ice);
                  A2(ff2in, kr, ice) = RL(0.0);
                  A2(ff2r, kr, ice) = RL(0.0);
                }
              }
              if(water_budget) water_budget[0]+=POINT_WATER(qq)-observed;
            }
            R fmax1 = RL(0.), fmax3 = RL(0.), fmax4 = RL(0.), fmax5 = RL(0.);
            R fmax2v[FSBM_ICEMAX + 1];
            fmax2v[1] = fmax2v[2] = fmax2v[3] = RL(0.);
            for (int kr = 1; kr <= FSBM_NKR; ++kr) {
              ff1in[kr] = ff1r[kr];
              fmax1 = fmax2(ff1r[kr], fmax1);
              ff3in[kr] = ff3r[kr];
              fmax3 = fmax2(ff3r[kr], fmax3);
              ff4in[kr] = ff4r[kr];
              fmax4 = fmax2(ff4r[kr], fmax4);
              ff5in[kr] = ff5r[kr];
              fmax5 = fmax2(ff5r[kr], fmax5);
              for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
                A2(ff2in, kr, ice) = A2(ff2r, kr, ice);
                fmax2v[ice] = fmax2(A2(ff2r, kr, ice), fmax2v[ice]);
              }
            }
            int isym1 = 0, isym3 = 0, isym4 = 0, isym5 = 0;
            int isym2[FSBM_ICEMAX + 1] = {0, 0, 0, 0};
            if (fmax1 > (R)0) isym1 = 1;
            if (fmax2v[1] > RL(1.E-10)) isym2[1] = 1;
            if (fmax2v[2] > RL(1.E-10)) isym2[2] = 1;
            if (fmax2v[3] > RL(1.E-10)) isym2[3] = 1;
            if (fmax3 > RL(1.E-10)) isym3 = 1;
            if (fmax4 > RL(1.E-10)) isym4 = 1;
            if (fmax5 > RL(1.E-10)) isym5 = 1;
            int s2 = isym2[1] + isym2[2] + isym2[3];
            double tc = tt - (double)RL(273.15);
            if (isym1 == 1 && (tc > (double)RL(-0.187) || (s2 == 0 && isym3 == 0 && isym4 == 0 && isym5 == 0))) {
              if(water_budget) observed=POINT_WATER(qq);
              e = onecond1<R>(tt, qq, pp, rhocgs, vr1_z, del1in, del2in, ff1r, ff1in, T, bb1_my, bb2_my,
                              col, dtcond, isym1, isym2, isym3, isym4, isym5);
              if (e) return e;
              if(water_budget) water_budget[1]+=POINT_WATER(qq)-observed;
              if (dg) dg->n_cond1++;
            } else if (isym1 == 0 && tc < (double)RL(-0.187) &&
                       (s2 > 1 || isym3 == 1 || isym4 == 1 || isym5 == 1)) {
              if(water_budget) observed=POINT_WATER(qq);
              e = onecond2<R>(tt, qq, pp, rhocgs, vr2_z, vr3_z, vr4_z, vr5_z, del1in, del2in, ff2r, ff2in,
                              ff3r, ff3in, ff4r, ff4in, ff5r, ff5in, T, bb1_my, bb2_my, col, dtcond, isym1,
                              isym2, isym3, isym4, isym5);
              if (e) return e;
              if(water_budget) water_budget[1]+=POINT_WATER(qq)-observed;
              if (dg) dg->n_cond2++;
            } else if (isym1 == 1 && tc < (double)RL(-0.187) &&
                       (s2 > 1 || isym3 == 1 || isym4 == 1 || isym5 == 1)) {
              if (tt > (double)RL(233.15)) {
                if(water_budget) observed=POINT_WATER(qq);
              e = onecond3<R>(tt, qq, pp, rhocgs, vr1_z, vr2_z, vr3_z, vr4_z, vr5_z, del1in, del2in,
                                ff1r, ff1in, ff2r, ff2in, ff3r, ff3in, ff4r, ff4in, ff5r, ff5in, T, bb1_my,
                                bb2_my, col, dtcond, isym1, isym2, isym3, isym4, isym5,
                                F.supsat_rederived);
                if (e) return e;
                if(water_budget) water_budget[1]+=POINT_WATER(qq)-observed;
              if (dg) dg->n_cond3++;
              }
            }
          }
        }
      }
      // collision-coalescence (NCOLL = 1)
      if (tt >= (double)RL(233.15)) {
        R colleff;
        if(water_budget) observed=POINT_WATER(qq);
        e = coal_bott_new<R>(ff1r, ff2r, ff3r, ff4r, ff5r, tt, qq, pp, rhocgs, dt, tcrit, T, F, del1in,
                             del2in, colleff, dg ? &dg->rime_mass : (double*)0);
        if (e) return e;
        if(water_budget) water_budget[2]+=POINT_WATER(qq)-observed;
        if (dg) dg->n_coal++;
      }
      t_new = (R)tt;
      qv = (R)qq;
    }
  }
  if(water_budget) observed=POINT_WATER(double(qv));
  // D14 corrected: every drop at or below -38 C freezes, in any saturation
  if (F.homfrz_any_saturation && (double)t_new - 273.15 <= -38.0) {
    double sum_ice = 0.0, col3 = 3.0 * (double)col, rori = 1.0 / (double)rhocgs;
    for (int kr = 1; kr <= FSBM_NKR; ++kr) {
      double xl = (double)T.xl[kr - 1];
      if (ff1r[kr] > (R)0) {
        A2(ff2r, kr, 2) = A2(ff2r, kr, 2) + ff1r[kr];
        sum_ice = sum_ice + col3 * xl * xl * (double)ff1r[kr];
        ff1r[kr] = RL(0.0);
      }
    }
    if (sum_ice > 0.0) {
      t_new = (R)((double)t_new + 334.0 * sum_ice * rori);
      if (dg) { dg->n_homfrz_point++; dg->homfrz_point_mass += sum_ice * rori; }
    }
  }
  R fusion = F.fusion_heat_334 ? RL(334.) : RL(333.);
  if (t_new < RL(273.15)) {
    freez<R>(ff1r, T.xl - 1, ff2r, ff5r, t_new, dt, rhocgs, col, afreezmy, bfreezmy, bfreezmax,
             FSBM_KRFREEZ, fusion, dg ? &dg->freez_mass : (double*)0);
    if (dg) dg->n_freez++;
  }
  if (t_new > RL(273.15)) {
    j_w_melt<R>(ff1r, T.xl - 1, ff2r, ff3r, ff4r, ff5r, t_new, dt, rhocgs, col, fusion,
                F.melt_exponential, dg ? &dg->melt_mass : (double*)0);
    if (dg) dg->n_melt++;
  }
  if(water_budget) water_budget[3]+=POINT_WATER(double(qv))-observed;
  for (int kr = 1; kr <= FSBM_NKR; ++kr) {
    for (int ice = 1; ice <= FSBM_ICEMAX; ++ice) {
      ff3r[kr] = ff3r[kr] + A2(ff2r, kr, ice);
      A2(ff2r, kr, ice) = RL(0.0);
    }
    if (F.hail) { ff5r[kr] = ff5r[kr] + ff4r[kr]; ff4r[kr] = RL(0.0); }
    else { ff4r[kr] = ff4r[kr] + ff5r[kr]; ff5r[kr] = RL(0.0); }
  }
  {
    R s = RL(0.);
    for (int kr = 1; kr <= FSBM_NKR; ++kr) s = s + ff1r[kr];
    if ((double)s > 43.0 * 1.0e-30) {
      double ff1r_d[FSBM_NKR + 1], xl_d[FSBM_NKR + 1];
      for (int kr = 1; kr <= FSBM_NKR; ++kr) {
        ff1r_d[kr] = (double)ff1r[kr];
        xl_d[kr] = (double)T.xl[kr - 1];
      }
      if(water_budget) observed=POINT_WATER(double(qv));
      spont_rain_breakup<R>(dt, ff1r_d, xl_d, T.prob, T.gain_var_new, FSBM_NKR, T.ikr_spon_break);
      for (int kr = 1; kr <= FSBM_NKR; ++kr) ff1r[kr] = (R)ff1r_d[kr];
      if(water_budget) water_budget[4]+=POINT_WATER(double(qv))-observed;
      if (dg) dg->n_spont++;
    }
  }
  // Snow breakup (:4836-4845) never runs on the 33-bin grid: KR_SNOW_MAX = 35.
  for (int kr = 1; kr <= FSBM_NKR; ++kr) bins[(FSBM_OFF_DROP + kr - 1) * sn] = ff1r[kr];
  for (int kr = 1; kr <= FSBM_NKR_AERO; ++kr) bins[(FSBM_OFF_AERO + kr - 1) * sn] = fccn[kr];
  for (int kr = 1; kr <= FSBM_NKR; ++kr) bins[(FSBM_OFF_SNOW + kr - 1) * sn] = ff3r[kr];
  for (int kr = 1; kr <= FSBM_NKR; ++kr)
    bins[(FSBM_OFF_DENSE + kr - 1) * sn] = F.hail ? ff5r[kr] : ff4r[kr];
  #undef POINT_WATER
  return FSBM_OK;
}

#endif  // WOOF_FAST_SBM_CORE_CUH
