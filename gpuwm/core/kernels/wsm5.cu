// WRF v4.6.1 WSM5 single-moment five-class microphysics (mp_physics = 4).
//
// A word-for-word port of WRF's own call chain for one column, all of it in
// phys/module_mp_wsm5.F:
//   wsm5            the wrapper module_microphysics_driver.F calls: t =
//                   th*pii in, th = t/pii out, reflectivity and effective
//                   radii from th*pii
//   wsm52D          the scheme, with slope_wsm5, slope_rain, slope_snow and
//                   nislfv_rain_plm
//   effectRad_wsm5  the radiation effective radii
// and, as this file's second kernel (wsm5_refl10cm_column, at the end),
// refl10cm_wsm5 with module_mp_radar.F's melting-particle routines.
//
// TWO DECLARED DIVERGENCES FROM WRF, both default-on and both encoded by the
// oracle's defined arm (tools/wsm5_wrf461_column_oracle/patch_arms.py):
//   1. The sedimentation remap's top search (wsm5_nislfv below) tests the
//      final arrival interface za(km+1), takes the cell above it when no
//      interface is high enough, and never leaves kt below kb.  WRF's
//      search stops at za(km); when it falls through it keeps a stale kt,
//      decrements it below kb, and the target cell keeps qn = 0: the
//      condensate in it is deleted (lane 073's water-loss defect, shared
//      with WSM6/WDM6).  The transport-only control of lane 073 loses
//      0.2 kg m-2 of 5.0 at 50 levels in WRF and closes here.
//   2. psevp, the evaporation of melting snow ([HL A35] [RH83 A27]), takes
//      WSM6's own line.  WRF's WSM5 forms it as psdep*work1(2)/work1(1),
//      but psdep is set only where supcol > 0 and this branch runs where
//      supcol < 0, so stock WSM5 never evaporates melting snow.
//
// BITWISE CONTRACT.  Graded word for word against the WRF build trees' own
// objects and the defined arm (tests/test_wsm5.py, tools/
// wsm5_wrf461_column_oracle).  The reference is WRF's own -O2 serial build
// with -ffp-contract=off -fno-tree-vectorize, so:
//   * Every expression keeps Fortran's left-to-right association, with the
//     parenthesization of WRF's inlined statement functions as written.
//   * EXP, LOG and REAL**REAL are WOOF's own float32 math routines gfk_exp,
//     gfk_log and gfk_pow; `x**2` is x*x.  slope_wsm5 forms rslopeb as
//     exp(log(r)*b) while slope_rain/slope_snow (the remap's arrival speed)
//     form it as r**b: two different words, kept apart.
//   * Fortran MAX/MIN are w5max/w5min below (`b > a ? b : a`).
//   * A float division by a literal is __fdiv_rn.
//   * The unit compiles with --fmad=false --ftz=false (kernels/__init__.py
//     load_module_unflushed): no contraction, subnormals kept.
//   * The derived constants are the float32 words wsm5init computes,
//     frozen in gpuwm/core/wsm6_constants.py WSM5_INIT_F32 and passed in
//     `coef`.
//   * An integer power in the binary64 radar code is gfortran -O2's inline
//     chain: x**3 = x*(x*x), x**6 = x3*x3.
//
// LOOP FUSION.  wsm52D sweeps the column once per process family.  After
// sedimentation every family reads and writes level k only, so this port
// runs them as one k loop; the order of operations at each level is WRF's.
// Cloud-ice sedimentation is moved ahead of the snow-melting sweep: it reads
// qi, xni, den and delz only, none of which melting touches, and melting
// reads nothing ice sedimentation writes.
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.

#ifndef WSM5_KMAX
#define WSM5_KMAX 64
#endif

// gfortran's MAX/MIN on two operands, argument order kept.
__device__ __forceinline__ float w5max(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float w5min(float a, float b) { return (b < a) ? b : a; }

// wsm5init's derived words, in WSM5_INIT_F32 order.
enum Wsm5Coef {
    C5_PI, C5_XLV1, C5_QC0, C5_QCK1, C5_PIDNC, C5_PIDN0R, C5_PIDN0S,
    C5_PVTR, C5_PVTS, C5_PACRR, C5_PACRC,
    C5_PRECR1, C5_PRECR2, C5_PRECS1, C5_PRECS2, C5_ROQIMAX,
    C5_RSLOPERMAX, C5_RSLOPESMAX, C5_RSLOPERBMAX, C5_RSLOPESBMAX,
    C5_RSLOPER2MAX, C5_RSLOPES2MAX, C5_RSLOPER3MAX, C5_RSLOPES3MAX,
    WSM5_NCOEF
};

// module_model_constants.F values the microphysics driver passes, and
// module_mp_wsm5.F's own parameters.
#define V5_CPD   1004.5f        // cp = 7.*r_d/2.
#define V5_CPV   1846.4f        // cpv = 4.*r_v
#define V5_RV    461.6f
#define V5_T0C   273.15f        // svpt0
#define V5_QMIN  1.0e-15f       // epsilon
#define V5_XLS   2.85e6f
#define V5_XLV0  2.5e6f
#define V5_XLF0  3.5e5f
#define V5_DEN0  1.28f          // rhoair0
#define V5_DENR  1000.0f        // rhowater
#define V5_CLIQ  4190.0f
#define V5_CICE  2106.0f
#define V5_PSAT  610.78f
#define V5_DTCLDCR 120.0f
#define V5_N0R   8.0e6f
#define V5_BVTR  0.8f
#define V5_XNCR  3.0e8f
#define V5_BVTS  0.41f
#define V5_DICON 11.9f
#define V5_DIMAX 500.0e-6f
#define V5_N0S   2.0e6f
#define V5_ALPHA 0.12f
#define V5_PFRZ1 100.0f
#define V5_PFRZ2 0.66f
#define V5_QCRMIN 1.0e-9f
#define V5_N0SMAX_OVER_N0S 5.0e4f   // n0smax/n0s folded by gfortran: 1.e11/2.e6

// n0s factor of slope_wsm5, slope_snow, wsm52D and effectRad_wsm5.
__device__ __forceinline__ float wsm5_n0sfac(float t) {
    const float supcol = V5_T0C - t;
    return w5max(w5min(gfk_exp(V5_ALPHA * supcol), V5_N0SMAX_OVER_N0S), 1.0f);
}

// denfac = sqrt(den0 * (1/den)): VREC, *den0, VSQRT (wsm52D :536-542).
__device__ __forceinline__ float wsm5_denfac(float den) {
    return sqrtf((1.0f / den) * V5_DEN0);
}

// ---- the inlined fpvs of wsm52D (:547-605, :1183-1201) ---------------------
struct Wsm5Fpvs { float ttp, xa, xb, xai, xbi, ep2; };
__device__ __forceinline__ Wsm5Fpvs wsm5_fpvs_consts() {
    Wsm5Fpvs f;
    f.ttp = V5_T0C + 0.01f;
    const float dldt = V5_CPV - V5_CLIQ;
    f.xa = -__fdiv_rn(dldt, V5_RV);
    f.xb = f.xa + __fdiv_rn(V5_XLV0, V5_RV * f.ttp);
    const float dldti = V5_CPV - V5_CICE;
    f.xai = -__fdiv_rn(dldti, V5_RV);
    f.xbi = f.xai + __fdiv_rn(V5_XLS, V5_RV * f.ttp);
    f.ep2 = __fdiv_rn(287.0f, V5_RV);       // ep_2 = r_d/r_v
    return f;
}
__device__ __forceinline__ float wsm5_qsat(const Wsm5Fpvs &f, float t, float p,
                                           bool ice) {
    const float tr = f.ttp / t;
    const float logtr = gfk_log(tr);
    const bool cold = ice && t < f.ttp;
    float q = V5_PSAT * gfk_exp(logtr * (cold ? f.xai : f.xa)
                                + (cold ? f.xbi : f.xb) * (1.0f - tr));
    q = w5min(q, 0.99f * p);
    q = (f.ep2 * q) / (p - q);
    return w5max(q, V5_QMIN);
}

// ---- slope_wsm5 (exp(log()) form) and slope_rain/slope_snow (powf form) ----
struct Wsm5Slope { float r, rb, r2, r3, vt; };

__device__ __forceinline__ Wsm5Slope wsm5_slope_rain(const float *c, float q,
                                                     float den, float denfac,
                                                     bool powform) {
    Wsm5Slope s;
    if (q <= V5_QCRMIN) {
        s.r = c[C5_RSLOPERMAX]; s.rb = c[C5_RSLOPERBMAX];
        s.r2 = c[C5_RSLOPER2MAX]; s.r3 = c[C5_RSLOPER3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf(c[C5_PIDN0R] / (q * den)));
        s.rb = powform ? gfk_pow(s.r, V5_BVTR) : gfk_exp(gfk_log(s.r) * V5_BVTR);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C5_PVTR] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

__device__ __forceinline__ Wsm5Slope wsm5_slope_snow(const float *c, float q,
                                                     float den, float denfac,
                                                     float t, bool powform) {
    Wsm5Slope s;
    const float n0sfac = wsm5_n0sfac(t);
    if (q <= V5_QCRMIN) {
        s.r = c[C5_RSLOPESMAX]; s.rb = c[C5_RSLOPESBMAX];
        s.r2 = c[C5_RSLOPES2MAX]; s.r3 = c[C5_RSLOPES3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf((c[C5_PIDN0S] * n0sfac) / (q * den)));
        s.rb = powform ? gfk_pow(s.r, V5_BVTS) : gfk_exp(gfk_log(s.r) * V5_BVTS);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C5_PVTS] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// ---- nislfv_rain_plm (:1417-1659) -------------------------------------------
// mode 0: rain (slope_rain arrival speed), 1: snow (slope_snow), 2: cloud
// ice, iter = 0.  rq enters as den*q and leaves as the remapped den*q.
// `wwl` is the departure speed and is not written, as in WRF.
__device__ void wsm5_nislfv(int km, const float *c, const float *den,
                            const float *dz, const float *tk,
                            const float *wwl, float *rq, float *precip_out,
                            float dt, int iter, int mode, int stride)
{
    float ww[WSM5_KMAX], qn[WSM5_KMAX];
    float wi[WSM5_KMAX + 1], zi[WSM5_KMAX + 1], za[WSM5_KMAX + 1];
    float dza[WSM5_KMAX + 1], qa[WSM5_KMAX + 1];
    float qmi[WSM5_KMAX + 1], qpi[WSM5_KMAX + 1];

    *precip_out = 0.0f;
    float allold = 0.0f;
    for (int k = 0; k < km; ++k) allold = allold + rq[k];
    if (allold <= 0.0f) return;

    zi[0] = 0.0f;
    for (int k = 0; k < km; ++k) zi[k + 1] = zi[k] + dz[k * stride];
    for (int k = 0; k < km; ++k) ww[k] = wwl[k];

    int n = 1;
    for (;;) {
        // WRF's 2nd-order interface interpolation is overwritten at every
        // index by its 3rd-order one; only the latter is formed here.
        const float fa1 = 9.0f / 16.0f, fa2 = 1.0f / 16.0f;
        wi[0] = ww[0];
        wi[1] = 0.5f * (ww[1] + ww[0]);
        for (int k = 2; k <= km - 2; ++k)
            wi[k] = fa1 * (ww[k] + ww[k - 1]) - fa2 * (ww[k + 1] + ww[k - 2]);
        wi[km - 1] = 0.5f * (ww[km - 1] + ww[km - 2]);
        wi[km] = ww[km - 1];
        for (int k = 1; k < km; ++k)
            if (ww[k] == 0.0f) wi[k] = ww[k - 1];
        const float con1 = 0.05f;
        for (int k = km - 1; k >= 0; --k) {
            const float dzk = dz[k * stride];
            const float decfl = ((wi[k + 1] - wi[k]) * dt) / dzk;
            if (decfl > con1) wi[k] = wi[k + 1] - (con1 * dzk) / dt;
        }
        for (int k = 0; k <= km; ++k) za[k] = zi[k] - wi[k] * dt;
        for (int k = 0; k < km; ++k) dza[k] = za[k + 1] - za[k];
        dza[km] = zi[km] - za[km];
        for (int k = 0; k < km; ++k) qa[k] = (rq[k] * dz[k * stride]) / dza[k];
        qa[km] = 0.0f;
        if (n > iter) break;
        // Arrival terminal speed through slope_rain/slope_snow (r**b), then
        // the mean of departure and arrival (iter = 1: one pass).
        for (int k = 0; k < km; ++k) {
            const float denk = den[k * stride];
            const float denfac = wsm5_denfac(denk);
            const float qr = qa[k] / denk;
            const float wa = (mode == 0)
                ? wsm5_slope_rain(c, qr, denk, denfac, true).vt
                : wsm5_slope_snow(c, qr, denk, denfac, tk[k], true).vt;
            ww[k] = 0.5f * (wwl[k] + wa);
        }
        n = n + 1;
    }

    for (int k = 1; k < km; ++k) {
        const float dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
        const float dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
        if (dip * dim <= 0.0f) {
            qmi[k] = qa[k];
            qpi[k] = qa[k];
        } else {
            qpi[k] = qa[k] + (0.5f * (dip + dim)) * dza[k];
            qmi[k] = 2.0f * qa[k] - qpi[k];
            if (qpi[k] < 0.0f || qmi[k] < 0.0f) {
                qpi[k] = qa[k];
                qmi[k] = qa[k];
            }
        }
    }
    qpi[0] = qa[0];
    qmi[0] = qa[0];
    qmi[km] = qa[km];
    qpi[km] = qa[km];

    for (int k = 0; k < km; ++k) qn[k] = 0.0f;
    int kb = 0, kt = 0;
    for (int k = 0; k < km; ++k) {
        kb = (kb - 1 > 0) ? kb - 1 : 0;
        kt = (kt - 1 > 0) ? kt - 1 : 0;
        if (zi[k] >= za[km]) break;
        for (int kk = kb; kk < km; ++kk)
            if (zi[k] <= za[kk + 1]) { kb = kk; break; }
        // DECLARED DIVERGENCE 1 (see the head of this file): the search
        // reaches the final arrival interface za[km]; with none high enough
        // the cell above it (qa = 0) closes the target cell; kt >= kb.
        int ktfound = km + 1;
        for (int kk = kt; kk <= km; ++kk)
            if (zi[k + 1] <= za[kk]) { ktfound = kk; break; }
        kt = (ktfound - 1 > kb) ? ktfound - 1 : kb;
        if (kt == kb) {
            const float tl = (zi[k] - za[kb]) / dza[kb];
            const float th = (zi[k + 1] - za[kb]) / dza[kb];
            const float tl2 = tl * tl;
            const float th2 = th * th;
            const float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            const float qqh = qqd * th2 + qmi[kb] * th;
            const float qql = qqd * tl2 + qmi[kb] * tl;
            qn[k] = (qqh - qql) / (th - tl);
        } else {
            const float tl = (zi[k] - za[kb]) / dza[kb];
            const float tl2 = tl * tl;
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            const float qql = qqd * tl2 + qmi[kb] * tl;
            const float dql = qa[kb] - qql;
            float zsum = (1.0f - tl) * dza[kb];
            float qsum = dql * dza[kb];
            if (kt - kb > 1) {
                for (int m = kb + 1; m <= kt - 1; ++m) {
                    zsum = zsum + dza[m];
                    qsum = qsum + qa[m] * dza[m];
                }
            }
            const float th = (zi[k + 1] - za[kt]) / dza[kt];
            const float th2 = th * th;
            qqd = 0.5f * (qpi[kt] - qmi[kt]);
            const float dqh = qqd * th2 + qmi[kt] * th;
            zsum = zsum + th * dza[kt];
            qsum = qsum + dqh * dza[kt];
            qn[k] = qsum / zsum;
        }
    }

    float precip = 0.0f;
    for (int k = 0; k < km; ++k) {
        if (za[k] < 0.0f && za[k + 1] < 0.0f) {
            precip = precip + qa[k] * dza[k];
            continue;
        } else if (za[k] < 0.0f && za[k + 1] >= 0.0f) {
#ifdef WSM5_PLM_SURFACE
            // Correct mode, finding WB5 (physics.mode = "correct"; the WSM6
            // defect of kernels/wsm6.cu, shared by module_mp_wsm5.F's copy of
            // nislfv_rain_plm): book the below-ground part of the straddling
            // source cell with the same piecewise-linear profile (qmi/qpi)
            // the remap above integrates for the part that stays aloft, so
            // column water + surface precipitation is conserved.  The
            // Fortran books it at the flat cell mean (the #else line, exact
            // mode).
            const float th = (0.0f - za[k]) / dza[k];
            precip = precip + dza[k] * (qmi[k] * th
                     + 0.5f * (qpi[k] - qmi[k]) * th * th);
#else
            precip = precip + qa[k] * (0.0f - za[k]);
#endif
            break;
        }
        break;
    }
    for (int k = 0; k < km; ++k) rq[k] = qn[k];
    *precip_out = precip;
}

// venfac as the melting sweep inlines it (:703-708).
__device__ __forceinline__ float wsm5_venfac_melt(float t, float den, float p) {
    const float visc = ((1.496e-6f * (t * sqrtf(t))) / (t + 120.0f)) / den;
    const float diff = (8.794e-5f * gfk_exp(gfk_log(t) * 1.81f)) / p;
    return (gfk_exp(gfk_log(visc / diff) * 0.3333333f) / sqrtf(visc))
         * sqrtf(sqrtf(V5_DEN0 / den));
}

// venfac as the process sweep inlines it (:870-874).
__device__ __forceinline__ float wsm5_venfac_proc(float t, float den, float p) {
    const float a = (1.496e-6f * (t * sqrtf(t))) * p;
    const float b = ((t + 120.0f) * den) * (8.794e-5f * gfk_exp(gfk_log(t) * 1.81f));
    return (gfk_exp(0.3333333f * gfk_log(a / b)) * sqrtf(sqrtf(V5_DEN0 / den)))
         / sqrtf((1.496e-6f * (t * sqrtf(t))) / ((t + 120.0f) * den));
}

// diffac as the process sweep inlines it (:860-868), latent heat `xl`.
__device__ __forceinline__ float wsm5_diffac(float xl, float t, float den,
                                             float p, float qs) {
    return ((((den * xl) * xl) * (t + 120.0f)) * den)
         / (((1.414e3f * (1.496e-6f * (t * sqrtf(t)))) * den) * ((V5_RV * t) * t))
         + p / (qs * (8.794e-5f * gfk_exp(gfk_log(t) * 1.81f)));
}

__device__ void wsm5_column_impl(
    float *theta, float *qv, float *qc, float *qi, float *qr, float *qs,
    const float *den, const float *p, const float *pii, const float *dz,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int stride, int col)
{
    float c[WSM5_NCOEF];
    for (int n = 0; n < WSM5_NCOEF; ++n) c[n] = coef[n];
    const float pi = c[C5_PI];
    const Wsm5Fpvs fp = wsm5_fpvs_consts();

    float t[WSM5_KMAX], q[WSM5_KMAX], qcl[WSM5_KMAX], qil[WSM5_KMAX];
    float qrl[WSM5_KMAX], qsl[WSM5_KMAX];
    float cpm[WSM5_KMAX], xl[WSM5_KMAX], xni[WSM5_KMAX];
    float v1[WSM5_KMAX], v2[WSM5_KMAX], rq1[WSM5_KMAX], rq2[WSM5_KMAX];

    const float *denc = den + col;
    const float *dzc = dz + col;
    const float *pc = p + col;

    // module_mp_wsm5.F:160-168, then wsm52D :476-501.
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        t[k] = theta[id] * pii[id];
        q[k] = qv[id];
        qcl[k] = w5max(qc[id], 0.0f);
        qrl[k] = w5max(qr[id], 0.0f);
        qil[k] = w5max(qi[id], 0.0f);
        qsl[k] = w5max(qs[id], 0.0f);
        const float qq = w5max(q[k], V5_QMIN);
        cpm[k] = V5_CPD * (1.0f - qq) + qq * V5_CPV;
        xl[k] = V5_XLV0 - c[C5_XLV1] * (t[k] - V5_T0C);
    }

    float rainncv_c = 0.0f, snowncv_c = 0.0f, sr_c = 0.0f;
    float rain_c = rainnc[col], snow_c = snownc[col];

    // :517-519
    int loops = (int)roundf(__fdiv_rn(delt, V5_DTCLDCR));
    if (loops < 1) loops = 1;
    float dtcld = delt / (float)loops;
    if (delt <= V5_DTCLDCR) dtcld = delt;

    for (int loop = 0; loop < loops; ++loop) {
        // :637-667 ice number, departure speeds of rain and snow.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float denfac = wsm5_denfac(denk);
            float temp = denk * w5max(qil[k], V5_QMIN);
            temp = sqrtf(sqrtf((temp * temp) * temp));
            xni[k] = w5min(w5max(5.38e7f * temp, 1.0e3f), 1.0e6f);
            v1[k] = wsm5_slope_rain(c, qrl[k], denk, denfac, false).vt;
            v2[k] = wsm5_slope_snow(c, qsl[k], denk, denfac, t[k], false).vt;
            rq1[k] = denk * qrl[k];
            rq2[k] = denk * qsl[k];
            if (qrl[k] <= 0.0f) v1[k] = 0.0f;
            if (qsl[k] <= 0.0f) v2[k] = 0.0f;
        }
        // :668-683
        float delqrs1, delqrs2;
        wsm5_nislfv(nz, c, denc, dzc, t, v1, rq1, &delqrs1, dtcld, 1, 0, stride);
        wsm5_nislfv(nz, c, denc, dzc, t, v2, rq2, &delqrs2, dtcld, 1, 1, stride);
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            qrl[k] = w5max(rq1[k] / denk, 0.0f);
            qsl[k] = w5max(rq2[k] / denk, 0.0f);
        }
        const float dz0 = dzc[0];
        const float fall1 = (delqrs1 / dz0) / dtcld;
        const float fall2 = (delqrs2 / dz0) / dtcld;

        // :726-754 cloud-ice fall speed and sedimentation (iter = 0).
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            if (qil[k] <= 0.0f) {
                v1[k] = 0.0f;
            } else {
                const float xmi = (denk * qil[k]) / xni[k];
                const float diameter = w5max(w5min(V5_DICON * sqrtf(xmi), V5_DIMAX),
                                             1.0e-25f);
                v1[k] = 1.49e4f * gfk_exp(gfk_log(diameter) * 1.31f);
            }
            rq1[k] = denk * qil[k];
        }
        float delqi;
        wsm5_nislfv(nz, c, denc, dzc, t, v1, rq1, &delqi, dtcld, 0, 2, stride);
        for (int k = 0; k < nz; ++k)
            qil[k] = w5max(rq1[k] / denc[k * stride], 0.0f);
        const float fallc = (delqi / dz0) / dtcld;

        // :759-780 surface accumulation.
        {
            const float fallsum = (fall1 + fall2) + fallc;
            const float fallsum_qsi = fall2 + fallc;
            if (fallsum > 0.0f) {
                rainncv_c = (__fdiv_rn(fallsum * dz0, V5_DENR) * dtcld) * 1000.0f
                          + rainncv_c;
                rain_c = (__fdiv_rn(fallsum * dz0, V5_DENR) * dtcld) * 1000.0f
                       + rain_c;
            }
            if (fallsum_qsi > 0.0f) {
                snowncv_c = (__fdiv_rn(fallsum_qsi * dz0, V5_DENR) * dtcld) * 1000.0f
                          + snowncv_c;
                snow_c = (__fdiv_rn(fallsum_qsi * dz0, V5_DENR) * dtcld) * 1000.0f
                       + snow_c;
            }
            if (fallsum > 0.0f) sr_c = snowncv_c / (rainncv_c + 1.0e-12f);
        }

        const float rdtcld = 1.0f / dtcld;
        // Everything below reads and writes level k only.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float pk = pc[k * stride];
            const float denfac = wsm5_denfac(denk);
            float tk = t[k], qk = q[k];
            float qck = qcl[k], qik = qil[k], qrk = qrl[k], qsk = qsl[k];
            const float cpmk = cpm[k], xlk = xl[k];

            // :559-605 saturation at the start of the minor step (t and q
            // at this level are unchanged since then).
            const float qs1 = wsm5_qsat(fp, tk, pk, false);
            const float rh1 = w5max(qk / qs1, V5_QMIN);
            const float qs2 = wsm5_qsat(fp, tk, pk, true);
            const float rh2 = w5max(qk / qs2, V5_QMIN);

            // :690-722 slopes after sedimentation, then snow melting.
            Wsm5Slope sr_ = wsm5_slope_rain(c, qrk, denk, denfac, false);
            Wsm5Slope ss_ = wsm5_slope_snow(c, qsk, denk, denfac, tk, false);
            {
                const float supcol = V5_T0C - tk;
                const float n0sfac = w5max(w5min(gfk_exp(V5_ALPHA * supcol),
                                                 V5_N0SMAX_OVER_N0S), 1.0f);
                if (tk > V5_T0C && qsk > 0.0f) {
                    const float xlf = V5_XLF0;
                    const float work2 = wsm5_venfac_melt(tk, denk, pk);
                    const float coeres = ss_.r2 * sqrtf(ss_.r * ss_.rb);
                    float psmlt = (((((__fdiv_rn((1.414e3f * (((1.496e-6f * (tk * sqrtf(tk)))
                                    / (tk + 120.0f)) / denk)) * denk, xlf)
                                  * (V5_T0C - tk)) * pi) / 2.0f) * n0sfac)
                                * (c[C5_PRECS1] * ss_.r2 + (c[C5_PRECS2] * work2) * coeres))
                                / denk;
                    psmlt = w5min(w5max(psmlt * dtcld, -qsk), 0.0f);
                    qsk = qsk + psmlt;
                    qrk = qrk - psmlt;
                    tk = tk + (xlf / cpmk) * psmlt;
                }
            }

            // :786-838 instantaneous melting/freezing (rain slope from :690).
            {
                const float supcol = V5_T0C - tk;
                float xlf = V5_XLS - xlk;
                if (supcol < 0.0f) xlf = V5_XLF0;
                if (supcol < 0.0f && qik > 0.0f) {
                    qck = qck + qik;
                    tk = tk - (xlf / cpmk) * qik;
                    qik = 0.0f;
                }
                if (supcol > 40.0f && qck > 0.0f) {
                    qik = qik + qck;
                    tk = tk + (xlf / cpmk) * qck;
                    qck = 0.0f;
                }
                if (supcol > 0.0f && qck > 0.0f) {
                    const float supcolt = w5min(supcol, 50.0f);
                    const float pfrzdtc = w5min(
                        ((__fdiv_rn(__fdiv_rn((V5_PFRZ1 * (gfk_exp(V5_PFRZ2 * supcolt) - 1.0f))
                                              * denk, V5_DENR), V5_XNCR) * qck) * qck) * dtcld,
                        qck);
                    qik = qik + pfrzdtc;
                    tk = tk + (xlf / cpmk) * pfrzdtc;
                    qck = qck - pfrzdtc;
                }
                if (supcol > 0.0f && qrk > 0.0f) {
                    const float supcolt = w5min(supcol, 50.0f);
                    float temp = sr_.r;
                    temp = (((((temp * temp) * temp) * temp) * temp) * temp) * temp;
                    const float pfrzdtr = w5min(
                        ((((((((20.0f * (pi * pi)) * V5_PFRZ1) * V5_N0R) * V5_DENR) / denk)
                           * (gfk_exp(V5_PFRZ2 * supcolt) - 1.0f)) * temp) * dtcld),
                        qrk);
                    qsk = qsk + pfrzdtr;
                    tk = tk + (xlf / cpmk) * pfrzdtr;
                    qrk = qrk - pfrzdtr;
                }
            }

            // :843-876 slopes for the process rates, conduction/diffusion and
            // ventilation terms.
            sr_ = wsm5_slope_rain(c, qrk, denk, denfac, false);
            ss_ = wsm5_slope_snow(c, qsk, denk, denfac, tk, false);
            const float work1_1 = wsm5_diffac(xlk, tk, denk, pk, qs1);
            const float work1_2 = wsm5_diffac(V5_XLS, tk, denk, pk, qs2);
            const float work2 = wsm5_venfac_proc(tk, denk, pk);

            float prevp = 0.f, psdep = 0.f, praut = 0.f, psaut = 0.f, pracw = 0.f;
            float psaci = 0.f, psacw = 0.f, pigen = 0.f, pidep = 0.f, psevp = 0.f;

            // :886-922 warm rain.
            {
                const float supsat = w5max(qk, V5_QMIN) - qs1;
                const float satdt = supsat / dtcld;
                if (qck > c[C5_QC0]) {
                    praut = c[C5_QCK1] * gfk_exp(gfk_log(qck) * __fdiv_rn(7.0f, 3.0f));
                    praut = w5min(praut, qck / dtcld);
                }
                if (qrk > V5_QCRMIN && qck > V5_QMIN) {
                    pracw = w5min((((c[C5_PACRR] * sr_.r3) * sr_.rb) * qck) * denfac,
                                  qck / dtcld);
                }
                if (qrk > 0.0f) {
                    const float coeres = sr_.r2 * sqrtf(sr_.r * sr_.rb);
                    prevp = ((rh1 - 1.0f) * (c[C5_PRECR1] * sr_.r2
                                             + (c[C5_PRECR2] * work2) * coeres))
                          / work1_1;
                    if (prevp < 0.0f) {
                        prevp = w5max(prevp, -(qrk / dtcld));
                        prevp = w5max(prevp, satdt / 2.0f);
                    } else {
                        prevp = w5min(prevp, satdt / 2.0f);
                    }
                }
            }

            // :937-1059 cold rain.
            const float supcol = V5_T0C - tk;
            const float n0sfac = w5max(w5min(gfk_exp(V5_ALPHA * supcol),
                                             V5_N0SMAX_OVER_N0S), 1.0f);
            const float supsat = w5max(qk, V5_QMIN) - qs2;
            const float satdt = supsat / dtcld;
            int ifsat = 0;
            {
                float temp = denk * w5max(qik, V5_QMIN);
                temp = sqrtf(sqrtf((temp * temp) * temp));
                xni[k] = w5min(w5max(5.38e7f * temp, 1.0e3f), 1.0e6f);
            }
            const float xnik = xni[k];
            const float eacrs = gfk_exp(0.07f * (-supcol));
            if (supcol > 0.0f) {
                if (qsk > V5_QCRMIN && qik > V5_QMIN) {
                    const float xmi = (denk * qik) / xnik;
                    const float diameter = w5min(V5_DICON * sqrtf(xmi), V5_DIMAX);
                    const float vt2i = 1.49e4f * gfk_pow(diameter, 1.31f);
                    const float vt2s = (c[C5_PVTS] * ss_.rb) * denfac;
                    const float acrfac = (2.0f * ss_.r3 + (2.0f * diameter) * ss_.r2)
                                       + (diameter * diameter) * ss_.r;
                    psaci = ((((((pi * qik) * eacrs) * V5_N0S) * n0sfac)
                              * fabsf(vt2s - vt2i)) * acrfac) / 4.0f;
                }
            }
            if (qsk > V5_QCRMIN && qck > V5_QMIN) {
                psacw = w5min(((((c[C5_PACRC] * n0sfac) * ss_.r3) * ss_.rb) * qck) * denfac,
                              qck * rdtcld);
            }
            if (supcol > 0.0f) {
                if (qik > 0.0f && ifsat != 1) {
                    const float xmi = (denk * qik) / xnik;
                    const float diameter = V5_DICON * sqrtf(xmi);
                    pidep = (((4.0f * diameter) * xnik) * (rh2 - 1.0f)) / work1_2;
                    const float supice = satdt - prevp;
                    if (pidep < 0.0f) {
                        pidep = w5max(w5max(pidep, satdt * 0.5f), supice);
                        pidep = w5max(pidep, -(qik * rdtcld));
                    } else {
                        pidep = w5min(w5min(pidep, satdt * 0.5f), supice);
                    }
                    if (fabsf(prevp + pidep) >= fabsf(satdt)) ifsat = 1;
                }
                if (qsk > 0.0f && ifsat != 1) {
                    const float coeres = ss_.r2 * sqrtf(ss_.r * ss_.rb);
                    psdep = (((rh2 - 1.0f) * n0sfac)
                             * (c[C5_PRECS1] * ss_.r2 + (c[C5_PRECS2] * work2) * coeres))
                          / work1_2;
                    const float supice = (satdt - prevp) - pidep;
                    if (psdep < 0.0f) {
                        psdep = w5max(psdep, -(qsk * rdtcld));
                        psdep = w5max(w5max(psdep, satdt * 0.5f), supice);
                    } else {
                        psdep = w5min(w5min(psdep, satdt * 0.5f), supice);
                    }
                    if (fabsf((prevp + pidep) + psdep) >= fabsf(satdt)) ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {
                    const float supice = ((satdt - prevp) - pidep) - psdep;
                    const float xni0 = 1.0e3f * gfk_exp(0.1f * supcol);
                    const float roqi0 = 4.92e-11f * gfk_exp(gfk_log(xni0) * 1.33f);
                    pigen = w5max(0.0f, (roqi0 / denk - w5max(qik, 0.0f)) * rdtcld);
                    pigen = w5min(w5min(pigen, satdt), supice);
                }
                if (qik > 0.0f) {
                    const float qimax = c[C5_ROQIMAX] / denk;
                    psaut = w5max(0.0f, (qik - qimax) * rdtcld);
                }
            }
            if (supcol < 0.0f) {
                // DECLARED DIVERGENCE 2 (see the head of this file): WSM6's
                // own evaporation of melting snow.
                if (qsk > 0.0f && rh1 < 1.0f) {
                    const float coeres = ss_.r2 * sqrtf(ss_.r * ss_.rb);
                    psevp = (((rh1 - 1.0f) * n0sfac)
                             * (c[C5_PRECS1] * ss_.r2 + (c[C5_PRECS2] * work2) * coeres))
                          / work1_1;
                }
                psevp = w5min(w5max(psevp, -(qsk * rdtcld)), 0.0f);
            }

            // :1066-1178 mass conservation of the generation terms, then the
            // update and its latent heating.
            {
                float value, source, factor;
                if (tk <= V5_T0C) {
                    value = w5max(V5_QMIN, qck);
                    source = (praut + pracw + psacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        psacw = psacw * factor;
                    }
                    value = w5max(V5_QMIN, qik);
                    source = (psaut + psaci - pigen - pidep) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        psaut = psaut * factor;
                        psaci = psaci * factor;
                        pigen = pigen * factor;
                        pidep = pidep * factor;
                    }
                    value = w5max(V5_QMIN, qrk);
                    source = (-praut - pracw - prevp) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        prevp = prevp * factor;
                    }
                    value = w5max(V5_QMIN, qsk);
                    source = (-psdep - psaut - psaci - psacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        psdep = psdep * factor;
                        psaut = psaut * factor;
                        psaci = psaci * factor;
                        psacw = psacw * factor;
                    }
                    const float work2u = -(prevp + psdep + pigen + pidep);
                    qk = qk + work2u * dtcld;
                    qck = w5max(qck - (praut + pracw + psacw) * dtcld, 0.0f);
                    qrk = w5max(qrk + (praut + pracw + prevp) * dtcld, 0.0f);
                    qik = w5max(qik - (psaut + psaci - pigen - pidep) * dtcld, 0.0f);
                    qsk = w5max(qsk + (psdep + psaut + psaci + psacw) * dtcld, 0.0f);
                    const float xlf = V5_XLS - xlk;
                    const float xlwork2 = -(V5_XLS * (psdep + pidep + pigen))
                                        - xlk * prevp - xlf * psacw;
                    tk = tk - (xlwork2 / cpmk) * dtcld;
                } else {
                    value = w5max(V5_QMIN, qck);
                    source = (praut + pracw + psacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        psacw = psacw * factor;
                    }
                    value = w5max(V5_QMIN, qrk);
                    source = (-praut - pracw - prevp - psacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        prevp = prevp * factor;
                        psacw = psacw * factor;
                    }
                    value = w5max(V5_QCRMIN, qsk);
                    source = (-psevp) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        psevp = psevp * factor;
                    }
                    const float work2u = -(prevp + psevp);
                    qk = qk + work2u * dtcld;
                    qck = w5max(qck - (praut + pracw + psacw) * dtcld, 0.0f);
                    qrk = w5max(qrk + (praut + pracw + prevp + psacw) * dtcld, 0.0f);
                    qsk = w5max(qsk + psevp * dtcld, 0.0f);
                    const float xlwork2 = -(xlk * (prevp + psevp));
                    tk = tk - (xlwork2 / cpmk) * dtcld;
                }
            }

            // :1183-1223 condensation/evaporation of cloud water.
            {
                const float qs1n = wsm5_qsat(fp, tk, pk, false);
                const float work1 = (w5max(qk, V5_QMIN) - qs1n)
                    / (1.0f + (((xlk * xlk) / (V5_RV * cpmk)) * qs1n) / (tk * tk));
                float pcond = w5min(w5max(work1 / dtcld, 0.0f), w5max(qk, 0.0f) / dtcld);
                if (qck > 0.0f && work1 < 0.0f)
                    pcond = w5max(work1, -qck) / dtcld;
                qk = qk - pcond * dtcld;
                qck = w5max(qck + pcond * dtcld, 0.0f);
                tk = tk + ((pcond * xlk) / cpmk) * dtcld;
            }
            // :1229-1234 padding for small values.
            if (qck <= V5_QMIN) qck = 0.0f;
            if (qik <= V5_QMIN) qik = 0.0f;

            t[k] = tk; q[k] = qk;
            qcl[k] = qck; qil[k] = qik; qrl[k] = qrk; qsl[k] = qsk;
        }
    }

    rainnc[col] = rain_c; rainncv[col] = rainncv_c;
    snownc[col] = snow_c; snowncv[col] = snowncv_c;
    sr[col] = sr_c;

    // module_mp_wsm5.F:187-241: th = t/pii, then effectRad_wsm5 on th*pii.
    // Radii are formed in metres as WRF stores them; gpuwm's radiation
    // contract is microns, the one multiply WRF's RRTMG applies when it
    // reads them (re*1.e6).
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        const float th = t[k] / pii[id];
        theta[id] = th;
        qv[id] = q[k]; qc[id] = qcl[k]; qi[id] = qil[k];
        qr[id] = qrl[k]; qs[id] = qsl[k];
        const float tt = th * pii[id];
        const float rho = denc[k * stride];
        float re_qc = 2.49e-6f, re_qi = 4.99e-6f, re_qs = 9.99e-6f;
        const float rqc = w5max(1.0e-12f, qcl[k] * rho);
        const float rqi = w5max(1.0e-12f, qil[k] * rho);
        float temp = rho * w5max(qil[k], V5_QMIN);
        temp = sqrtf(sqrtf((temp * temp) * temp));
        const float ni = w5min(w5max(5.38e7f * temp, 1.0e3f), 1.0e6f);
        const float rni = w5max(1.0e-6f, ni * rho);
        const float rqs = w5max(1.0e-12f, qsl[k] * rho);
        if (rqc > 1.0e-12f) {
            const float lamdac = gfk_pow((c[C5_PIDNC] * 3.0e8f) / rqc,
                                         __fdiv_rn(1.0f, 3.0f));
            re_qc = w5max(2.51e-6f, w5min(1.5f * (1.0f / lamdac), 50.0e-6f));
        }
        if (rqi > 1.0e-12f && rni > 1.0e-6f) {
            const float diai = 11.9f * sqrtf(rqi / ni);
            re_qi = w5max(10.01e-6f, w5min((0.75f * 0.163f) * diai, 125.0e-6f));
        }
        if (rqs > 1.0e-12f) {
            const float nsf = wsm5_n0sfac(tt);
            const float lamdas = sqrtf(sqrtf((c[C5_PIDN0S] * nsf) / rqs));
            re_qs = w5max(25.0e-6f, w5min(0.5f * (1.0f / lamdas), 999.0e-6f));
        }
        re_qc = w5max(2.49e-6f, w5min(re_qc, 50.0e-6f));
        re_qi = w5max(4.99e-6f, w5min(re_qi, 125.0e-6f));
        re_qs = w5max(9.99e-6f, w5min(re_qs, 999.0e-6f));
        effc[id] = re_qc * 1.0e6f;
        effi[id] = re_qi * 1.0e6f;
        effs[id] = re_qs * 1.0e6f;
    }
}

extern "C" __global__ void wsm5_column(
    float *theta, float *qv, float *qc, float *qi, float *qr, float *qs,
    const float *den, const float *p, const float *pii, const float *dz,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WSM5_KMAX) return;
    wsm5_column_impl(theta, qv, qc, qi, qr, qs, den, p, pii, dz,
                     rainnc, rainncv, snownc, snowncv, sr, effc, effi, effs,
                     coef, delt, nz, stride, col);
}

// ===========================================================================
// REFL_10CM for WSM5: phys/module_mp_wsm5.F refl10cm_wsm5 (:1669-1803) with
// module_mp_radar.F's rayleigh_soak_wetgraupel / get_m_mix_nested /
// get_m_mix / m_complex_maxwellgarnett for melting snow, and the wrapper's
// floor MAX(-35., dBZ) (:213).
//
// PRECISION is WRF's mix, kept line by line: temperature, density, the
// species densities, the snow intercept and the dry-snow prefactor are
// float32 (gfk_exp for the snow intercept's EXP); slopes, moments and the
// melting integral are float64, rounded to float32 where WRF assigns them to
// a REAL array.  Binary64 EXP, LOG and REAL**REAL are WOOF's own binary64
// words (glibc_flt64.cuh: glibc_exp, glibc_log, glibc_pow).  LOG10, ATAN2
// and HYPOT, and so the complex LOG/SQRT/ABS of the melting term, are
// CUDA's binary64 functions: WRF rounds every such value to float32 before
// it is stored, so a stored word moves only when a binary64 result sits
// within an ULP of a float32 rounding boundary.  gfortran's complex
// arithmetic: products (ac-bd, ad+bc), quotients Smith's range-reduced form
// (-fcx-fortran-rules); integer powers gfortran -O2's inline chains.
//
// The radar constants are module_mp_radar's own words from radar_init after
// wsm5init (gpuwm/core/wsm6_constants.py WSM5_RADAR_*), not recomputations:
// xcrg(4) = WGAMMA(7.) is 720.00006 in float32, and pi5 is a float32
// product held in binary64.
// ===========================================================================

enum Wsm5RadarF32 {
    R5_PI, R5_XAM_R, R5_XAM_S,
    R5_XCRE1, R5_XCRE4, R5_XCRG3, R5_XCRG4,
    R5_XCSE1, R5_XCSE4, R5_XCSG3, R5_XCSG4,
    R5_XBM_S, R5_XMU_S, R5_XOCMS, R5_XOBMS,
    WSM5_NRADAR_F32
};
#define WSM5_NRBINS 50
enum Wsm5RadarF64 {
    D5_K_W, D5_PI5, D5_LAMDA4, D5_MW_RE, D5_MW_IM, D5_MI_RE, D5_MI_IM,
    D5_XXDS, D5_XDTS = D5_XXDS + WSM5_NRBINS,
    D5_SIMPSON = D5_XDTS + WSM5_NRBINS,
    WSM5_NRADAR_F64 = D5_SIMPSON + WSM5_NRBINS + 1
};

struct W5Cplx { double re, im; };
__device__ __forceinline__ W5Cplx w5c(double re, double im) {
    W5Cplx z; z.re = re; z.im = im; return z;
}
__device__ __forceinline__ W5Cplx w5c_add(W5Cplx a, W5Cplx b) {
    return w5c(a.re + b.re, a.im + b.im);
}
__device__ __forceinline__ W5Cplx w5c_sub(W5Cplx a, W5Cplx b) {
    return w5c(a.re - b.re, a.im - b.im);
}
__device__ __forceinline__ W5Cplx w5c_mul(W5Cplx a, W5Cplx b) {
    return w5c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
// Smith's division as GCC lowers it for Fortran (range reduced, no NaN
// recovery).
__device__ __forceinline__ W5Cplx w5c_div(W5Cplx a, W5Cplx b) {
    double tr, ti, div;
    if (fabs(b.re) < fabs(b.im)) {
        const double ratio = b.re / b.im;
        div = (b.re * ratio) + b.im;
        tr = (a.re * ratio) + a.im;
        ti = (a.im * ratio) - a.re;
    } else {
        const double ratio = b.im / b.re;
        div = (b.im * ratio) + b.re;
        tr = (a.im * ratio) + a.re;
        ti = a.im - (a.re * ratio);
    }
    return w5c(tr / div, ti / div);
}
__device__ __forceinline__ W5Cplx w5c_rmul(double x, W5Cplx b) {
    return w5c_mul(w5c(x, 0.0), b);
}
__device__ __forceinline__ double w5c_abs(W5Cplx a) { return hypot(a.re, a.im); }
__device__ __forceinline__ W5Cplx w5c_log(W5Cplx a) {
    return w5c(glibc_log(hypot(a.re, a.im)), atan2(a.im, a.re));
}
__device__ __forceinline__ W5Cplx w5c_sqrt(W5Cplx a) {
    if (a.re == 0.0 && a.im == 0.0) return w5c(0.0, a.im);
    const double d = hypot(a.re, a.im);
    double r, s;
    if (a.re > 0.0) {
        r = sqrt(0.5 * (d + a.re));
        s = 0.5 * (a.im / r);
    } else {
        s = sqrt(0.5 * (d - a.re));
        r = fabs(0.5 * (a.im / s));
    }
    return w5c(r, copysign(s, a.im));
}
__device__ __forceinline__ double w5dmax(double a, double b) { return (b > a) ? b : a; }
__device__ __forceinline__ double w5dmin(double a, double b) { return (b < a) ? b : a; }

// m_complex_maxwellgarnett with inclusion = 'spheroidal'.
__device__ W5Cplx wsm5_maxwellgarnett(double vol1, double vol2, double vol3,
                                      W5Cplx m1, W5Cplx m2, W5Cplx m3, int *error)
{
    *error = 0;
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1.0e-6) {
        *error = 1;
        return w5c(-999.99, -999.99);
    }
    const W5Cplx m1t = w5c_mul(m1, m1);
    const W5Cplx m2t = w5c_mul(m2, m2);
    const W5Cplx m3t = w5c_mul(m3, m3);
    const W5Cplx one = w5c(1.0, 0.0);
    const W5Cplx beta2 = w5c_mul(
        w5c_div(w5c_rmul(2.0, m1t), w5c_sub(m2t, m1t)),
        w5c_sub(w5c_mul(w5c_div(m2t, w5c_sub(m2t, m1t)), w5c_log(w5c_div(m2t, m1t))), one));
    const W5Cplx beta3 = w5c_mul(
        w5c_div(w5c_rmul(2.0, m1t), w5c_sub(m3t, m1t)),
        w5c_sub(w5c_mul(w5c_div(m3t, w5c_sub(m3t, m1t)), w5c_log(w5c_div(m3t, m1t))), one));
    const double v1 = 1.0 - vol2 - vol3;
    const W5Cplx num = w5c_add(w5c_add(w5c_rmul(v1, m1t),
                                       w5c_mul(w5c_rmul(vol2, beta2), m2t)),
                               w5c_mul(w5c_rmul(vol3, beta3), m3t));
    const W5Cplx dnm = w5c_add(w5c_add(w5c(v1, 0.0), w5c_rmul(vol2, beta2)),
                               w5c_rmul(vol3, beta3));
    return w5c_sqrt(w5c_div(num, dnm));
}

// rayleigh_soak_wetgraupel with the strings radar_init sets for snow:
// mixing rule maxwellgarnett, host air, matrix water, inclusion spheroidal,
// host matrix icewater, host inclusion spheroidal.
__device__ double wsm5_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, W5Cplx m_w, W5Cplx m_i, double pi5, double lamda4)
{
    const double pix = 3.1415926535897932384626434;
    const W5Cplx m_air = w5c(1.0, 0.0);
    const double fm = w5dmax(w5dmin(fmelt, 1.0), 0.0);
    double mra = w5dmax(w5dmin(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    const double x_w = x_g * fm;
    const double d_g = a_geo * glibc_pow(x_g, b_geo);
    if (!(d_g >= 1.0e-12)) return 0.0;
    double vg = (pix / 6.0) * (d_g * (d_g * d_g));
    const double rhog = w5dmax(w5dmin(x_g / vg, 900.0), 10.0);
    vg = x_g / rhog;
    const double grenz = 1.0 - rhog / 1000.0;
    double volg;
    if (mra <= grenz) {
        volg = vg * (1.0 - mra * fm);
    } else {
        const double fmgrenz = (900.0 - rhog)
                             / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        if (fm <= fmgrenz) volg = (1.0 - mra * fm) * vg;
        else volg = (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    const double d_large = glibc_pow((6.0 / pix) * volg,
                                     (double)__fdiv_rn(1.0f, 3.0f));
    const double volice = (x_g - x_w) / (volg * 900.0);
    const double volwater = x_w / (1000.0 * volg);
    const double volair = 1.0 - volice - volwater;
    // get_m_mix_nested: host air, matrix water, host matrix icewater.
    int err1, err2;
    const double vol1 = volice / w5dmax(volice + volwater, 1.0e-10);
    const double vol2 = 1.0 - vol1;
    const W5Cplx mtmp = wsm5_maxwellgarnett(vol2, 0.0, vol1, m_w, m_air, m_i, &err1);
    const W5Cplx m_core = wsm5_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_air,
                                              w5c_rmul(2.0, m_air), &err2);
    if (err1 + err2 != 0) return 0.0;
    const W5Cplx mc2 = w5c_mul(m_core, m_core);
    const double a = w5c_abs(w5c_div(w5c_sub(mc2, w5c(1.0, 0.0)),
                                     w5c_add(mc2, w5c(2.0, 0.0))));
    const double d3 = d_large * (d_large * d_large);
    return (((a * a) * pi5) * (d3 * d3)) / lamda4;
}

extern "C" __global__ void wsm5_refl10cm_column(
    const float *qv, const float *qr, const float *qs,
    const float *t, const float *p, const float *__restrict__ rf,
    const double *__restrict__ rd, float *refl, int nz, int ny, int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = ny * nx;
    if (col >= stride || nz > WSM5_KMAX) return;
    float temp[WSM5_KMAX], rs[WSM5_KMAX];
    double ilamr[WSM5_KMAX], ilams[WSM5_KMAX], n0_s[WSM5_KMAX];
    float ze_snow[WSM5_KMAX];
    unsigned char lq[WSM5_KMAX];   // bit 0 rain, 1 snow
    const float pi = rf[R5_PI];
    const double n0_r = (double)V5_N0R;

    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        temp[k] = t[id];
        const float temp_c = w5min(-0.001f, temp[k] - 273.15f);
        const float qvk = w5max(1.0e-10f, qv[id]);
        const float pres = p[id];
        const float rho = (0.622f * pres) / ((287.0f * temp[k]) * (qvk + 0.622f));
        lq[k] = 0;
        if (qr[id] > 1.0e-9f) {
            const float rr = qr[id] * rho;
            const double lamr = glibc_pow(
                (double)(rf[R5_XAM_R] * rf[R5_XCRG3]) * n0_r / (double)rr,
                (double)(1.0f / rf[R5_XCRE1]));
            ilamr[k] = 1.0 / lamr;
            lq[k] |= 1;
        }
        if (qs[id] > 1.0e-9f) {
            rs[k] = qs[id] * rho;
            n0_s[k] = (double)w5min(1.0e11f, V5_N0S * gfk_exp(-V5_ALPHA * temp_c));
            const double lams = glibc_pow(
                (double)(rf[R5_XAM_S] * rf[R5_XCSG3]) * n0_s[k] / (double)rs[k],
                (double)(1.0f / rf[R5_XCSE1]));
            ilams[k] = 1.0 / lams;
            lq[k] |= 2;
        } else {
            rs[k] = 1.0e-12f;
        }
    }

    bool melti = false;
    int k_0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (temp[k] > 273.15f && (lq[k] & 1) && (lq[k + 1] & 2)) {
            k_0 = (k + 1 > k_0) ? k + 1 : k_0;
            melti = true;
            break;
        }
    }

    // Rayleigh term of dry snow (the prefactor float32 as WRF writes it,
    // promoted where it meets the float64 intercept).
    const float six_over_pi = 6.0f / pi;
    const float pref_s = __fdiv_rn(0.176f, 0.93f) * six_over_pi * six_over_pi
                       * __fdiv_rn(rf[R5_XAM_S], 900.0f) * __fdiv_rn(rf[R5_XAM_S], 900.0f);
    for (int k = 0; k < nz; ++k) {
        ze_snow[k] = 1.0e-22f;
        if (lq[k] & 2)
            ze_snow[k] = (float)((double)pref_s * n0_s[k] * (double)rf[R5_XCSG4]
                                 * glibc_pow(ilams[k], (double)rf[R5_XCSE4]));
    }

    if (melti && k_0 >= 1) {
        const W5Cplx m_w = w5c(rd[D5_MW_RE], rd[D5_MW_IM]);
        const W5Cplx m_i = w5c(rd[D5_MI_RE], rd[D5_MI_IM]);
        const double pi5 = rd[D5_PI5], lamda4 = rd[D5_LAMDA4], k_w = rd[D5_K_W];
        for (int k = k_0 - 1; k >= 0; --k) {
            if ((lq[k] & 2) && (lq[k_0] & 2)) {
                const double fmelt_s = w5dmax(0.005, w5dmin(1.0 - (double)(rs[k] / rs[k_0]),
                                                            0.99));
                double eta = 0.0;
                const double lams = 1.0 / ilams[k];
                for (int n = 0; n < WSM5_NRBINS; ++n) {
                    const double xx = rd[D5_XXDS + n];
                    const double x = (double)rf[R5_XAM_S]
                                   * glibc_pow(xx, (double)rf[R5_XBM_S]);
                    const double cback = wsm5_rayleigh_soak_wetgraupel(
                        x, (double)rf[R5_XOCMS], (double)rf[R5_XOBMS], fmelt_s, 0.9,
                        m_w, m_i, pi5, lamda4);
                    const double f_d = n0_s[k] * glibc_pow(xx, (double)rf[R5_XMU_S])
                                     * glibc_exp(-lams * xx);
                    eta = eta + f_d * cback * rd[D5_SIMPSON + n] * rd[D5_XDTS + n];
                }
                ze_snow[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
        }
    }

    for (int k = nz - 1; k >= 0; --k) {
        float ze_rain = 1.0e-22f;
        if (lq[k] & 1)
            ze_rain = (float)(n0_r * (double)rf[R5_XCRG4]
                              * glibc_pow(ilamr[k], (double)rf[R5_XCRE4]));
        const float zsum = ze_rain + ze_snow[k];
        const float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));
        refl[col + k * stride] = w5max(-35.0f, dbz);
    }
}
