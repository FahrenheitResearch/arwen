#ifndef WOOF_FAST_SBM_DRIVER_CUH
#define WOOF_FAST_SBM_DRIVER_CUH

// Shared host/device FAST_SBM tile driver. Scratch is supplied by the caller.
// The host follows WRF tile order; corrected CUDA columns use frozen halo inputs.

// The arrays of one FAST_SBM call, WRF memory layout (ims:ime, kms:kme,
// jms:jme[, n_chem]); 2-D fields (ims:ime, jms:jme).
template <class R> struct State {
  R *w, *u, *v, *th_old, *chem, *dz8w, *rho, *p, *pii, *th, *qv, *qv_old;
  R *qc, *qr, *qi, *qs, *qg, *qnc, *qnr, *qni, *qns, *qng, *qna;
  R *xland, *rainnc, *rainncv, *snownc, *snowncv, *graupelnc, *graupelncv;
};

template <class R> struct FsbmScratch {
  R *pcgs, *rhocgs, *zcgs, *t_new, *t_old, *qvo, *dsupice, *supice, *dzz;
  int* kz;
};

// Physical rho*dz water ledger for verification; never changes the state.
template <class R>
FHD double fsbm_water(const FsbmTab<R>& T, const FsbmFlags& F, const int* dims,
                      const State<R>& S, bool distributions) {
  long ni=dims[7]-dims[6]+1, nk=dims[11]-dims[10]+1, nj=dims[9]-dims[8]+1;
  long sn=ni*nk*nj;
  double total=0;
  for(int j=dims[14];j<=dims[15];++j)
    for(int k=dims[16];k<=dims[17];++k)
      for(int i=dims[12];i<=dims[13];++i) {
        long o=i-dims[6]+(k-dims[10])*ni+(j-dims[8])*ni*nk;
        double q=S.qv[o];
        for(int b=0;b<99;++b) {
          double x=b<33?T.xl[b]:(b<66?T.xs[b-33]:(F.hail?T.xh[b-66]:T.xg[b-66]));
          double v=S.chem[o+(b+1)*sn];
          q+=distributions ? 3.0*double(FSBM_COL)*x*x*v/(double(S.rho[o])*0.001) : v;
        }
        total+=double(S.rho[o])*double(S.dz8w[o])*q;
      }
  return total;
}
template <class R>
FHD double fsbm_precip(const int* dims,const State<R>& S) {
  long ni=dims[7]-dims[6]+1;
  double total=0;
  for(int j=dims[14];j<=dims[15];++j)
    for(int i=dims[12];i<=dims[13];++i)
      total+=S.rainnc[i-dims[6]+(j-dims[8])*ni];
  return total;
}

// Per-call tallies for the budget tests: [0] exit clip mass, [1] activated
// drop mass, [2] homogeneous freezing mass (JERNUCL01), [3] ice nucleation
// mass, [4] FREEZ mass, [5] melt mass, [6] riming mass, [7] D14 point
// freezing mass, [8..15] counts: cond1, cond2, cond3, coal, freez, melt,
// spont, homfrz-point, [16] the D9 cloud-base level of the first column.
template <class R>
FHD_NOINLINE int fsbm_tile(const FsbmTab<R>& T, const FsbmFlags& F, int itimestep, R dt, R dx, R dy, const int* dims,
                  const State<R>& S, const FsbmScratch<R>& X, double* tally, int mode = 0, double* ledger = nullptr) {
  if (ledger) { ledger[0]=fsbm_water(T,F,dims,S,false); ledger[1]=fsbm_precip(dims,S); }
  const int ids = dims[0], ide = dims[1], jds = dims[2], jde = dims[3];
  const int ims = dims[6], ime = dims[7], jms = dims[8], jme = dims[9], kms = dims[10], kme = dims[11];
  const int its = dims[12], ite = dims[13], jts = dims[14], jte = dims[15], kts = dims[16], kte = dims[17];
  (void)ids; (void)jds;
  const long ni = ime - ims + 1, nk = kme - kms + 1, nj = jme - jms + 1;
  const long sk = ni, sn = ni * nk * nj;
  auto ix = [&](int i, int k, int j) -> long { return (long)(i - ims) + (long)(k - kms) * ni + (long)(j - jms) * ni * nk; };
  auto ix2 = [&](int i, int j) -> long { return (long)(i - ims) + (long)(j - jms) * ni; };
  // chem_new index 1 is unused: the bins are p_ff1i01 = 2 .. p_ff8i43 = 143
  R* bins0 = S.chem + sn;  // n = 2
  const int ncond = 3;
  R dtcond = dt / (R)ncond;
  R *pcgs = X.pcgs, *rhocgs = X.rhocgs, *zcgs = X.zcgs;
  R *t_new = X.t_new, *t_old = X.t_old, *qvo = X.qvo;
  if (mode != 1) {
  // entry conversion (:4214-4257)
  for (int j = jts; j <= jte; ++j)
    for (int i = its; i <= ite; ++i)
      for (int k = kts; k <= kte; ++k) fsbm_entry_point<R>(T, F.hail, bins0 + ix(i, k, j), sn, S.rho[ix(i, k, j)]);
  int i_start = its - 1 > 1 ? its - 1 : 1, j_start = jts - 1 > 1 ? jts - 1 : 1;
  int i_end = ide - 1 < ite + 1 ? ide - 1 : ite + 1, j_end = jde - 1 < jte + 1 ? jde - 1 : jte + 1;
  for (int j = j_start; j <= j_end; ++j)
    for (int i = i_start; i <= i_end; ++i) {
      R z_full = RL(0.);
      for (int k = kts; k <= kte; ++k) {
        long o = ix(i, k, j);
        pcgs[o] = S.p[o] * RL(10.);
        rhocgs[o] = S.rho[o] * RL(0.001);
        zcgs[o] = z_full + RL(0.5) * S.dz8w[o] * (R)100;
        z_full = z_full + S.dz8w[o] * RL(100.);
      }
    }
  if (itimestep == 1)
    for (int j = jts; j <= jte; ++j)
      for (int i = its; i <= ite; ++i)
        for (int k = kts; k <= kte; ++k)
          fsbm_aerosol_init_point<R>(T, bins0 + ix(i, k, j), sn, rhocgs[ix(i, k, j)], rhocgs[ix(i, 1, j)],
                                     S.xland[ix2(i, j)]);
  // D18: the first call's reset of th_old/qv_old changes this call's T_OLD
  // and QV_OLD only (the halo is the neighbours' pre-call state).
  for (int j = j_start; j <= j_end; ++j)
    for (int k = kts; k <= kte; ++k)
      for (int i = i_start; i <= i_end; ++i) {
        long o = ix(i, k, j);
        R tho = itimestep == 1 ? S.th[o] : S.th_old[o];
        qvo[o] = itimestep == 1 ? S.qv[o] : S.qv_old[o];
        t_new[o] = S.th[o] * S.pii[o];
        t_old[o] = tho * S.pii[o];
      }
  }
  if (mode == 2) return FSBM_OK;
  R dxhucm = RL(100.) * dx, dyhucm = RL(100.) * dy;
  // pre-pass (:4372-4512)
  int* kz = X.kz;
  R* dsupice = X.dsupice;
  R w_stag_carry = RL(0.0);
  bool w_stag_set = false;
  int nz = kte - kts + 1;
  R *supice = X.supice, *dzz = X.dzz;
  for (int j = jts; j <= jte; ++j)
    for (int i = its; i <= ite; ++i) {
      long o1 = ix(i, kts, j);
      kz[(i - its) + (j - jts) * (ite - its + 1)] =
          fsbm_cloud_base<R>(&t_new[o1], &S.qv[o1], &pcgs[o1], &zcgs[o1], &S.w[o1], sk, nz,
                             F.cloud_base_first_level);
      fsbm_supice_column<R>(&t_old[o1], &qvo[o1], &pcgs[o1], &zcgs[o1], sk, nz, supice, dzz);
      for (int k = kts; k <= kte; ++k) {
        long o = ix(i, k, j);
        R w_own = (k < kte) ? RL(50.) * (S.w[o] + S.w[o + sk]) : (R)100 * S.w[o];
        if (t_old[o] >= RL(238.15) && t_old[o] < RL(274.15)) {
          w_stag_carry = w_own;
          w_stag_set = true;
        }
        R w_use = F.dsupice_w_own_point ? w_own : (w_stag_set ? w_stag_carry : w_own);
        auto nb = [&](int di, int dj) {
          long q = ix(i + di, k, j + dj);
          FsbmNbr<R> r = {t_old[q], qvo[q], pcgs[q], S.u[q], S.v[q]};
          return r;
        };
        dsupice[o] = fsbm_dsupice_point<R>(nb, k, nz, i == 1, i == ide - 1, j == 1, j == jde - 1,
                                           supice, dzz, w_use, dxhucm, dyhucm, dtcond,
                                           F.dsupice_wind_own_point);
      }
    }
  if (tally) tally[16] = kz[0];
  // the point loop (:4514-4888), WRF order j, k, i
  R factor_last = RL(1.0);
  FsbmPointDiag dg;
  dg = FsbmPointDiag{};
  for (int j = jts; j <= jte; ++j)
    for (int k = kts; k <= kte; ++k)
      for (int i = its; i <= ite; ++i) {
        long o = ix(i, k, j);
        if (!F.dtcond_carry) dtcond = dt / (R)ncond;
        R w_stag_my = (k < kte) ? RL(50.) * (S.w[o] + S.w[o + sk]) : (R)100 * S.w[o];
        int is_kz = kz[(i - its) + (j - jts) * (ite - its + 1)] == k;
        double water_budget[5]={};
        int e = fsbm_point<R>(T, F, bins0 + o, sn, t_old[o], t_new[o], qvo[o], S.qv[o], pcgs[o], rhocgs[o],
                              w_stag_my, is_kz, dsupice[o], dt, dtcond, &dg, ledger ? water_budget : nullptr);
        if(ledger) for(int b=0;b<5;++b) ledger[5+b]+=water_budget[b]*double(S.rho[o])*double(S.dz8w[o]);
        if (e) return e;
        S.th[o] = t_new[o] / S.pii[o];
        if (k == 1) factor_last = (R)fx_sqrt(1.0e6 / (double)pcgs[o]);
      }
  if (ledger) ledger[2]=fsbm_water(T,F,dims,S,true);
  // the column part (:4890-5320)
  for (int j = jts; j <= jte; ++j)
    for (int i = its; i <= ite; ++i) {
      long o1 = ix(i, kts, j);
      R fp = F.rainnc_own_column ? (R)fx_sqrt(1.0e6 / (double)pcgs[o1]) : factor_last;
      FsbmColumnOut<R> out = {S.qc + o1, S.qr + o1, S.qi + o1, S.qs + o1, S.qg + o1, S.qnc + o1,
                              S.qnr + o1, S.qni + o1, S.qns + o1, S.qng + o1, S.qna + o1};
      double clip = 0.0;
      long o2 = ix2(i, j);
      int e = fsbm_column<R>(T, F, bins0 + o1, sk, sn, S.rho + o1, &pcgs[o1], &zcgs[o1], sk, dt, nz, fp,
                             out, S.rainnc[o2], S.rainncv[o2], S.snownc[o2], S.snowncv[o2],
                             S.graupelnc[o2], S.graupelncv[o2], &clip, S.dz8w+o1, sk, rhocgs+o1, sk);
      if (e) return e;
      for (int k = kts; k <= kte; ++k) {
        long o = ix(i, k, j);
        S.th_old[o] = S.th[o];
        S.qv_old[o] = S.qv[o];
      }
      if (tally) tally[0] += clip;
    }
  if (ledger) { ledger[3]=fsbm_water(T,F,dims,S,false); ledger[4]=fsbm_precip(dims,S); }
  if (tally) {
    tally[1] += dg.nucl.drop_mass; tally[2] += dg.nucl.homfrz_mass; tally[3] += dg.nucl.ice_mass;
    tally[4] += dg.freez_mass; tally[5] += dg.melt_mass; tally[6] += dg.rime_mass;
    tally[7] += dg.homfrz_point_mass;
    tally[8] += dg.n_cond1; tally[9] += dg.n_cond2; tally[10] += dg.n_cond3; tally[11] += dg.n_coal;
    tally[12] += dg.n_freez; tally[13] += dg.n_melt; tally[14] += dg.n_spont; tally[15] += dg.n_homfrz_point;
  }
  return FSBM_OK;
}

#endif
