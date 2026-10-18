// WRF v4.6.1 WSM6 single-moment six-class microphysics (mp_physics = 6).
//
// A word-for-word port of WRF's own call chain for one column:
//   phys/module_mp_wsm6.F          subroutine wsm6 (the wrapper the
//                                  microphysics driver calls: t = th*pii in,
//                                  th = t/pii out, effective radii from
//                                  th*pii)
//   phys/physics_mmm/mp_wsm6.F90   mp_wsm6_run, slope_wsm6/_rain/_snow/
//                                  _graup, nislfv_rain_plm, nislfv_rain_plm6
//   phys/physics_mmm/mp_wsm6_effectRad.F90  mp_wsm6_effectRad_run
// and, as this file's second kernel (wsm6_refl10cm_column, at the end),
// refl10cm_wsm6 with mp_radar.F90's melting-particle routines.
//
// BITWISE CONTRACT.  Graded word for word against those routines compiled
// by gfortran (tools/wsm6_wrf461_column_oracle, tests/
// test_wsm6_wrf461_column_exact.py).  What that takes, and why each line
// below is spelled the way it is:
//   * Every expression keeps Fortran's left-to-right association; a
//     statement function (diffus, viscos, xka, diffac, venfac, conden,
//     cpmcal, xlcal) is a value, exactly as gfortran evaluates it.
//   * EXP, LOG and REAL**REAL are WOOF's own float32 routines gfk_exp,
//     gfk_log and gfk_pow (glibc_flt32.cuh), which return the words the
//     oracle's gfortran gets from its libm; CUDA's expf/logf/powf are
//     different functions.  `x**2` is x*x, as gfortran expands an integer
//     power.
//   * Fortran MAX/MIN are gfortran's MAX_EXPR chains: MAX(a,b) is
//     `b > a ? b : a` and MIN(a,b) is `b < a ? b : a` (wmax/wmin below).
//     They differ from fmaxf/fminf on a signed zero, which WRF keeps.
//   * A float division by a literal is __fdiv_rn: NVRTC turns `x / C` into
//     a multiply by the rounded reciprocal under -ftz=true (A146).
//   * The unit compiles with --fmad=false (kernels/__init__.py
//     _NO_FMAD_MODULES): gfortran on x86-64 contracts nothing.
//   * The derived constants are the float32 words mp_wsm6_init computes
//     (its rgmma Gamma product included), frozen in
//     gpuwm/core/wsm6_constants.py WRF_INIT_F32 and passed in `coef`.
//
// LOOP FUSION.  mp_wsm6_run sweeps the column once per process family.
// Between sedimentation and the end of a minor step every family reads and
// writes level k only, so this port runs them as one k loop; the order of
// operations at each level is WRF's.  Cloud-ice sedimentation is moved
// ahead of the snow/graupel melting sweep: it reads qi, xni, den and delz
// only, none of which melting touches, and melting reads nothing ice
// sedimentation writes.
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.

#ifndef WSM6_KMAX
#define WSM6_KMAX 64
#endif

// gfortran's MAX_EXPR/MIN_EXPR on two operands, argument order kept.
__device__ __forceinline__ float wmax(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float wmin(float a, float b) { return (b < a) ? b : a; }

// mp_wsm6_init's derived words, in WRF_INIT_F32 order.
enum Wsm6Coef {
    C_PI, C_XLV1, C_QC0, C_QCK1, C_PIDNC, C_PIDN0R, C_PIDN0S, C_PIDN0G,
    C_PVTR, C_PVTS, C_PVTG, C_PACRR, C_PACRC, C_PACRG,
    C_PRECR1, C_PRECR2, C_PRECS1, C_PRECS2, C_PRECG1, C_PRECG2,
    C_G6PBR, C_ROQIMAX, C_N0G, C_BVTG,
    C_RSLOPERMAX, C_RSLOPESMAX, C_RSLOPEGMAX,
    C_RSLOPERBMAX, C_RSLOPESBMAX, C_RSLOPEGBMAX,
    C_RSLOPER2MAX, C_RSLOPES2MAX, C_RSLOPEG2MAX,
    C_RSLOPER3MAX, C_RSLOPES3MAX, C_RSLOPEG3MAX,
    WSM6_NCOEF
};

// module_model_constants.F / mp_wsm6.F90 parameters, as WRF passes them.
#define W_CPD   1004.5f        // cp = 7.*r_d/2.
#define W_CPV   1846.4f        // cpv = 4.*r_v
#define W_RV    461.6f
#define W_T0C   273.15f        // svpt0
#define W_QMIN  1.0e-15f       // epsilon
#define W_XLS   2.85e6f
#define W_XLV0  2.5e6f
#define W_XLF0  3.5e5f
#define W_DEN0  1.28f          // rhoair0
#define W_DENR  1000.0f        // rhowater
#define W_CLIQ  4190.0f
#define W_CICE  2106.0f
#define W_PSAT  610.78f
#define W_DTCLDCR 120.0f
#define W_N0R   8.0e6f
#define W_AVTR  841.9f
#define W_BVTR  0.8f
#define W_XNCR  3.0e8f
#define W_BVTS  0.41f
#define W_DICON 11.9f
#define W_DIMAX 500.0e-6f
#define W_PFRZ1 100.0f
#define W_PFRZ2 0.66f
#define W_QCRMIN 1.0e-9f
#define W_DENS  100.0f
#define W_QS0   6.0e-4f
#define W_N0S   2.0e6f
#define W_ALPHA 0.12f
#define W_N0SMAX_OVER_N0S 5.0e4f   // n0smax/n0s, folded by gfortran: 1.e11/2.e6

// ---- statement functions of mp_wsm6_run --------------------------------
__device__ __forceinline__ float st_diffus(float x, float y) {
    return (8.794e-5f * gfk_exp(gfk_log(x) * 1.81f)) / y;
}
__device__ __forceinline__ float st_viscos(float x, float y) {
    return ((1.496e-6f * (x * sqrtf(x))) / (x + 120.0f)) / y;
}
__device__ __forceinline__ float st_xka(float x, float y) {
    return (1.414e3f * st_viscos(x, y)) * y;
}
__device__ __forceinline__ float st_diffac(float a, float b, float c, float d,
                                           float e) {
    return ((d * a) * a) / (((st_xka(c, d) * W_RV) * c) * c)
         + 1.0f / (e * st_diffus(c, b));
}
__device__ __forceinline__ float st_venfac(float a, float b, float c) {
    return (gfk_exp(gfk_log(st_viscos(b, c) / st_diffus(b, a)) * 0.3333333f)
            / sqrtf(st_viscos(b, c))) * sqrtf(sqrtf(W_DEN0 / c));
}
__device__ __forceinline__ float st_conden(float a, float b, float c, float d,
                                           float e) {
    return (wmax(b, W_QMIN) - c)
         / (1.0f + (((d * d) / (W_RV * e)) * c) / (a * a));
}

// fpvs inlined as mp_wsm6_run does (:519-547), then the saturation ratio.
struct Wsm6Fpvs { float ttp, xa, xb, xai, xbi; };
__device__ __forceinline__ Wsm6Fpvs wsm6_fpvs_consts() {
    Wsm6Fpvs f;
    f.ttp = W_T0C + 0.01f;
    const float dldt = W_CPV - W_CLIQ;
    f.xa = __fdiv_rn(-dldt, W_RV);
    f.xb = f.xa + __fdiv_rn(W_XLV0, W_RV * f.ttp);
    const float dldti = W_CPV - W_CICE;
    f.xai = __fdiv_rn(-dldti, W_RV);
    f.xbi = f.xai + __fdiv_rn(W_XLS, W_RV * f.ttp);
    return f;
}
__device__ __forceinline__ float wsm6_qsat(const Wsm6Fpvs &f, float t, float p,
                                           bool ice) {
    const float tr = f.ttp / t;
    float q;
    if (ice && t < f.ttp)
        q = (W_PSAT * gfk_exp(gfk_log(tr) * f.xai)) * gfk_exp(f.xbi * (1.0f - tr));
    else
        q = (W_PSAT * gfk_exp(gfk_log(tr) * f.xa)) * gfk_exp(f.xb * (1.0f - tr));
    q = wmin(q, 0.99f * p);
    q = (__fdiv_rn(287.0f, W_RV) * q) / (p - q);     // ep2 = r_d/r_v
    return wmax(q, W_QMIN);
}

// ---- slope_rain / slope_snow / slope_graup at one level -------------------
struct Wsm6Slope { float r, rb, r2, r3, vt; };

__device__ __forceinline__ Wsm6Slope wsm6_slope_rain(const float *c, float q,
                                                     float den, float denfac) {
    Wsm6Slope s;
    if (q <= W_QCRMIN) {
        s.r = c[C_RSLOPERMAX]; s.rb = c[C_RSLOPERBMAX];
        s.r2 = c[C_RSLOPER2MAX]; s.r3 = c[C_RSLOPER3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf(c[C_PIDN0R] / (q * den)));
        s.rb = gfk_pow(s.r, W_BVTR);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C_PVTR] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

__device__ __forceinline__ Wsm6Slope wsm6_slope_snow(const float *c, float q,
                                                     float den, float denfac,
                                                     float t) {
    Wsm6Slope s;
    const float supcol = W_T0C - t;
    const float n0sfac = wmax(wmin(gfk_exp(W_ALPHA * supcol),
                                   W_N0SMAX_OVER_N0S), 1.0f);
    if (q <= W_QCRMIN) {
        s.r = c[C_RSLOPESMAX]; s.rb = c[C_RSLOPESBMAX];
        s.r2 = c[C_RSLOPES2MAX]; s.r3 = c[C_RSLOPES3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf((c[C_PIDN0S] * n0sfac) / (q * den)));
        s.rb = gfk_pow(s.r, W_BVTS);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C_PVTS] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

__device__ __forceinline__ Wsm6Slope wsm6_slope_graup(const float *c, float q,
                                                      float den, float denfac) {
    Wsm6Slope s;
    if (q <= W_QCRMIN) {
        s.r = c[C_RSLOPEGMAX]; s.rb = c[C_RSLOPEGBMAX];
        s.r2 = c[C_RSLOPEG2MAX]; s.r3 = c[C_RSLOPEG3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf(c[C_PIDN0G] / (q * den)));
        s.rb = gfk_pow(s.r, c[C_BVTG]);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C_PVTG] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// denfac = sqrt(den0 * (1/den)): vrec, *den0, vsqrt (mp_wsm6.F90:502-514).
__device__ __forceinline__ float wsm6_denfac(float den) {
    return sqrtf((1.0f / den) * W_DEN0);
}

// ---- nislfv_rain_plm (rq2 == nullptr) / nislfv_rain_plm6 -------------------
// mode 0: rain, slope_rain arrival speed; mode 1: snow + graupel,
// slope_snow/slope_graup mass-weighted; mode 2: cloud ice, iter = 0.
// rq/rq2 enter as den*q and leave as the remapped den*q.  `wwl` is the
// departure speed and is not written, as in WRF.
__device__ void wsm6_nislfv(int km, const float *c, const float *den,
                            const float *dz, const float *tk,
                            const float *wwl, float *rq, float *rq2,
                            float *precip1, float *precip2, float dt,
                            int iter, int mode, int stride)
{
    float ww[WSM6_KMAX], qn[WSM6_KMAX];
    float wi[WSM6_KMAX + 1], zi[WSM6_KMAX + 1], za[WSM6_KMAX + 1];
    float dza[WSM6_KMAX + 1], qa[WSM6_KMAX + 1], qa2[WSM6_KMAX + 1];
    float qmi[WSM6_KMAX + 1], qpi[WSM6_KMAX + 1];
    const bool two = (rq2 != nullptr);

    *precip1 = 0.0f;
    if (two) *precip2 = 0.0f;
    float allold = 0.0f;
    for (int k = 0; k < km; ++k) {
        allold = allold + rq[k];
        if (two) allold = allold + rq2[k];
    }
    if (allold <= 0.0f) return;

    zi[0] = 0.0f;
    for (int k = 0; k < km; ++k) zi[k + 1] = zi[k] + dz[k * stride];
    for (int k = 0; k < km; ++k) ww[k] = wwl[k];

    int n = 1;
    for (;;) {
        // The 2nd-order interface interpolation WRF computes first is
        // overwritten at every index by the 3rd-order one; only the latter
        // is formed here.
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
        for (int k = 0; k < km; ++k) {
            const float dzk = dz[k * stride];
            qa[k] = (rq[k] * dzk) / dza[k];
            if (two) qa2[k] = (rq2[k] * dzk) / dza[k];
        }
        qa[km] = 0.0f;
        qa2[km] = 0.0f;
        if (n > iter) break;
        // Arrival terminal speed; the mean of departure and arrival
        // (iter = 1: one pass, so no running average with `was`).
        for (int k = 0; k < km; ++k) {
            const float denk = den[k * stride];
            const float denfac = wsm6_denfac(denk);
            const float qr = qa[k] / denk;
            float wa;
            if (mode == 0) {
                wa = wsm6_slope_rain(c, qr, denk, denfac).vt;
            } else {
                const float qr2 = qa2[k] / denk;
                const float was = wsm6_slope_snow(c, qr, denk, denfac, tk[k]).vt;
                const float wa2 = wsm6_slope_graup(c, qr2, denk, denfac).vt;
                const float tmp = wmax(qr + qr2, 1.0e-15f);
                wa = (tmp > 1.0e-15f) ? (was * qr + wa2 * qr2) / tmp : 0.0f;
            }
            ww[k] = 0.5f * (wwl[k] + wa);
        }
        n = n + 1;
    }

    for (int ist = 0; ist < (two ? 2 : 1); ++ist) {
        float *q = (ist == 0) ? qa : qa2;
        for (int k = 1; k < km; ++k) {
            const float dip = (q[k + 1] - q[k]) / (dza[k + 1] + dza[k]);
            const float dim = (q[k] - q[k - 1]) / (dza[k - 1] + dza[k]);
            if (dip * dim <= 0.0f) {
                qmi[k] = q[k];
                qpi[k] = q[k];
            } else {
                qpi[k] = q[k] + (0.5f * (dip + dim)) * dza[k];
                qmi[k] = 2.0f * q[k] - qpi[k];
                if (qpi[k] < 0.0f || qmi[k] < 0.0f) {
                    qpi[k] = q[k];
                    qmi[k] = q[k];
                }
            }
        }
        qpi[0] = q[0];
        qmi[0] = q[0];
        qmi[km] = q[km];
        qpi[km] = q[km];

        for (int k = 0; k < km; ++k) qn[k] = 0.0f;
        int kb = 0, kt = 0;
        for (int k = 0; k < km; ++k) {
            kb = (kb - 1 > 0) ? kb - 1 : 0;
            kt = (kt - 1 > 0) ? kt - 1 : 0;
            if (zi[k] >= za[km]) break;
            for (int kk = kb; kk < km; ++kk)
                if (zi[k] <= za[kk + 1]) { kb = kk; break; }
            for (int kk = kt; kk < km; ++kk)
                if (zi[k + 1] <= za[kk]) { kt = kk; break; }
            kt = kt - 1;
            if (kt == kb) {
                const float tl = (zi[k] - za[kb]) / dza[kb];
                const float th = (zi[k + 1] - za[kb]) / dza[kb];
                const float tl2 = tl * tl;
                const float th2 = th * th;
                const float qqd = 0.5f * (qpi[kb] - qmi[kb]);
                const float qqh = qqd * th2 + qmi[kb] * th;
                const float qql = qqd * tl2 + qmi[kb] * tl;
                qn[k] = (qqh - qql) / (th - tl);
            } else if (kt > kb) {
                const float tl = (zi[k] - za[kb]) / dza[kb];
                const float tl2 = tl * tl;
                float qqd = 0.5f * (qpi[kb] - qmi[kb]);
                const float qql = qqd * tl2 + qmi[kb] * tl;
                const float dql = q[kb] - qql;
                float zsum = (1.0f - tl) * dza[kb];
                float qsum = dql * dza[kb];
                if (kt - kb > 1) {
                    for (int m = kb + 1; m <= kt - 1; ++m) {
                        zsum = zsum + dza[m];
                        qsum = qsum + q[m] * dza[m];
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
                precip = precip + q[k] * dza[k];
                continue;
            } else if (za[k] < 0.0f && za[k + 1] >= 0.0f) {
                precip = precip + q[k] * (0.0f - za[k]);
                break;
            }
            break;
        }
        float *out = (ist == 0) ? rq : rq2;
        for (int k = 0; k < km; ++k) out[k] = qn[k];
        if (ist == 0) *precip1 = precip; else *precip2 = precip;
    }
}

__device__ void wsm6_column_impl(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *sr, float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int stride, int col)
{
    float c[WSM6_NCOEF];
    for (int n = 0; n < WSM6_NCOEF; ++n) c[n] = coef[n];
    const float pi = c[C_PI];
    const Wsm6Fpvs fp = wsm6_fpvs_consts();

    float t[WSM6_KMAX], q[WSM6_KMAX], qcl[WSM6_KMAX], qil[WSM6_KMAX];
    float qrl[WSM6_KMAX], qsl[WSM6_KMAX], qgl[WSM6_KMAX];
    float cpm[WSM6_KMAX], xl[WSM6_KMAX], xni[WSM6_KMAX];
    float v1[WSM6_KMAX], v2[WSM6_KMAX];
    float rq1[WSM6_KMAX], rq2[WSM6_KMAX], rq3[WSM6_KMAX];

    const float *denc = den + col;
    const float *dzc = dz + col;
    const float *pc = p + col;

    // module_mp_wsm6.F:126-134, then mp_wsm6_run :440-459.
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        t[k] = theta[id] * pii[id];
        q[k] = qv[id];
        qcl[k] = wmax(qc[id], 0.0f);
        qrl[k] = wmax(qr[id], 0.0f);
        qil[k] = wmax(qi[id], 0.0f);
        qsl[k] = wmax(qs[id], 0.0f);
        qgl[k] = wmax(qg[id], 0.0f);
        const float qq = wmax(q[k], W_QMIN);
        cpm[k] = W_CPD * (1.0f - qq) + qq * W_CPV;
        xl[k] = W_XLV0 - c[C_XLV1] * (t[k] - W_T0C);
    }

    float rainncv_c = 0.0f, snowncv_c = 0.0f, graupelncv_c = 0.0f, sr_c = 0.0f;
    float rain_c = rainnc[col], snow_c = snownc[col], graupel_c = graupelnc[col];

    float loopsf = roundf(__fdiv_rn(delt, W_DTCLDCR));
    int loops = (int)loopsf;
    if (loops < 1) loops = 1;
    float dtcld = delt / (float)loops;
    if (delt <= W_DTCLDCR) dtcld = delt;

    for (int loop = 0; loop < loops; ++loop) {
        // :597-603 ice number, :609-634 departure speeds.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float denfac = wsm6_denfac(denk);
            float temp = denk * wmax(qil[k], W_QMIN);
            temp = sqrtf(sqrtf((temp * temp) * temp));
            xni[k] = wmin(wmax(5.38e7f * temp, 1.0e3f), 1.0e6f);
            const Wsm6Slope sr_ = wsm6_slope_rain(c, qrl[k], denk, denfac);
            const Wsm6Slope ss_ = wsm6_slope_snow(c, qsl[k], denk, denfac, t[k]);
            const Wsm6Slope sg_ = wsm6_slope_graup(c, qgl[k], denk, denfac);
            float workr = sr_.vt;
            const float qsum = wmax(qsl[k] + qgl[k], 1.0e-15f);
            v2[k] = (qsum > 1.0e-15f)
                  ? (ss_.vt * qsl[k] + sg_.vt * qgl[k]) / qsum : 0.0f;
            rq1[k] = denk * qrl[k];
            rq2[k] = denk * qsl[k];
            rq3[k] = denk * qgl[k];
            if (qrl[k] <= 0.0f) workr = 0.0f;
            v1[k] = workr;
        }
        float delqrs1, delqrs2, delqrs3;
        wsm6_nislfv(nz, c, denc, dzc, t, v1, rq1, nullptr, &delqrs1, nullptr,
                    dtcld, 1, 0, stride);
        wsm6_nislfv(nz, c, denc, dzc, t, v2, rq2, rq3, &delqrs2, &delqrs3,
                    dtcld, 1, 1, stride);
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            qrl[k] = wmax(rq1[k] / denk, 0.0f);
            qsl[k] = wmax(rq2[k] / denk, 0.0f);
            qgl[k] = wmax(rq3[k] / denk, 0.0f);
        }
        const float dz0 = dzc[0];
        const float fall1 = (delqrs1 / dz0) / dtcld;
        const float fall2 = (delqrs2 / dz0) / dtcld;
        const float fall3 = (delqrs3 / dz0) / dtcld;

        // :705-735 cloud-ice fall speed and sedimentation (iter = 0).
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            if (qil[k] <= 0.0f) {
                v1[k] = 0.0f;
            } else {
                const float xmi = (denk * qil[k]) / xni[k];
                const float diameter = wmax(wmin(W_DICON * sqrtf(xmi), W_DIMAX),
                                            1.0e-25f);
                v1[k] = 1.49e4f * gfk_exp(gfk_log(diameter) * 1.31f);
            }
            rq1[k] = denk * qil[k];
        }
        float delqi;
        wsm6_nislfv(nz, c, denc, dzc, t, v1, rq1, nullptr, &delqi, nullptr,
                    dtcld, 0, 2, stride);
        for (int k = 0; k < nz; ++k)
            qil[k] = wmax(rq1[k] / denc[k * stride], 0.0f);
        const float fallc = (delqi / dz0) / dtcld;

        // :740-771 surface accumulation.
        {
            const float fallsum = ((fall1 + fall2) + fall3) + fallc;
            const float fallsum_qsi = fall2 + fallc;
            const float fallsum_qg = fall3;
            if (fallsum > 0.0f) {
                rainncv_c = (__fdiv_rn(fallsum * dz0, W_DENR) * dtcld) * 1000.0f
                          + rainncv_c;
                rain_c = (__fdiv_rn(fallsum * dz0, W_DENR) * dtcld) * 1000.0f
                       + rain_c;
            }
            if (fallsum_qsi > 0.0f) {
                snowncv_c = (__fdiv_rn(fallsum_qsi * dz0, W_DENR) * dtcld) * 1000.0f
                          + snowncv_c;
                snow_c = (__fdiv_rn(fallsum_qsi * dz0, W_DENR) * dtcld) * 1000.0f
                       + snow_c;
            }
            if (fallsum_qg > 0.0f) {
                graupelncv_c = (__fdiv_rn(fallsum_qg * dz0, W_DENR) * dtcld) * 1000.0f
                             + graupelncv_c;
                graupel_c = (__fdiv_rn(fallsum_qg * dz0, W_DENR) * dtcld) * 1000.0f
                          + graupel_c;
            }
            if (fallsum > 0.0f)
                sr_c = (snowncv_c + graupelncv_c) / (rainncv_c + 1.0e-12f);
        }

        // Everything below reads and writes level k only.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float pk = pc[k * stride];
            const float denfac = wsm6_denfac(denk);
            float tk = t[k], qk = q[k];
            float qck = qcl[k], qik = qil[k], qrk = qrl[k], qsk = qsl[k], qgk = qgl[k];
            const float cpmk = cpm[k], xlk = xl[k];

            // :529-547 saturation at the start of the minor step (t and q at
            // this level are unchanged since then: sedimentation moves
            // neither).
            const float qsat1 = wsm6_qsat(fp, tk, pk, false);
            const float rh1 = wmax(qk / qsat1, W_QMIN);
            const float qsat2 = wsm6_qsat(fp, tk, pk, true);
            const float rh2 = wmax(qk / qsat2, W_QMIN);

            // :661-703 slopes after sedimentation, then melting.
            Wsm6Slope sr_ = wsm6_slope_rain(c, qrk, denk, denfac);
            Wsm6Slope ss_ = wsm6_slope_snow(c, qsk, denk, denfac, tk);
            Wsm6Slope sg_ = wsm6_slope_graup(c, qgk, denk, denfac);
            float supcol = W_T0C - tk;
            float n0sfac = wmax(wmin(gfk_exp(W_ALPHA * supcol), W_N0SMAX_OVER_N0S),
                                1.0f);
            if (tk > W_T0C) {
                const float xlf = W_XLF0;
                const float work2 = st_venfac(pk, tk, denk);
                if (qsk > 0.0f) {
                    const float coeres = ss_.r2 * sqrtf(ss_.r * ss_.rb);
                    float psmlt = __fdiv_rn(st_xka(tk, denk), xlf) * (W_T0C - tk) * pi
                                / 2.0f * n0sfac
                                * (c[C_PRECS1] * ss_.r2 + c[C_PRECS2] * work2 * coeres)
                                / denk;
                    psmlt = wmin(wmax(psmlt * dtcld, -qsk), 0.0f);
                    qsk = qsk + psmlt;
                    qrk = qrk - psmlt;
                    tk = tk + (xlf / cpmk) * psmlt;
                }
                if (qgk > 0.0f) {
                    const float coeres = sg_.r2 * sqrtf(sg_.r * sg_.rb);
                    float pgmlt = __fdiv_rn(st_xka(tk, denk), xlf) * (W_T0C - tk)
                                * (c[C_PRECG1] * sg_.r2 + c[C_PRECG2] * work2 * coeres)
                                / denk;
                    pgmlt = wmin(wmax(pgmlt * dtcld, -qgk), 0.0f);
                    qgk = qgk + pgmlt;
                    qrk = qrk - pgmlt;
                    tk = tk + (xlf / cpmk) * pgmlt;
                }
            }

            // :777-829 instantaneous melting/freezing (slopes from :661).
            {
                const float supc = W_T0C - tk;
                float xlf = W_XLS - xlk;
                if (supc < 0.0f) xlf = W_XLF0;
                if (supc < 0.0f && qik > 0.0f) {
                    qck = qck + qik;
                    tk = tk - (xlf / cpmk) * qik;
                    qik = 0.0f;
                }
                if (supc > 40.0f && qck > 0.0f) {
                    qik = qik + qck;
                    tk = tk + (xlf / cpmk) * qck;
                    qck = 0.0f;
                }
                if (supc > 0.0f && qck > W_QMIN) {
                    const float supcolt = wmin(supc, 50.0f);
                    const float pfrzdtc = wmin(
                        ((__fdiv_rn(__fdiv_rn((W_PFRZ1 * (gfk_exp(W_PFRZ2 * supcolt) - 1.0f))
                                              * denk, W_DENR), W_XNCR) * qck) * qck) * dtcld,
                        qck);
                    qik = qik + pfrzdtc;
                    tk = tk + (xlf / cpmk) * pfrzdtc;
                    qck = qck - pfrzdtc;
                }
                if (supc > 0.0f && qrk > 0.0f) {
                    float temp = sr_.r3;
                    temp = (temp * temp) * sr_.r;
                    const float supcolt = wmin(supc, 50.0f);
                    const float pfrzdtr = wmin(
                        (((((((20.0f * (pi * pi)) * W_PFRZ1) * W_N0R) * W_DENR) / denk)
                          * (gfk_exp(W_PFRZ2 * supcolt) - 1.0f)) * temp) * dtcld,
                        qrk);
                    qgk = qgk + pfrzdtr;
                    tk = tk + (xlf / cpmk) * pfrzdtr;
                    qrk = qrk - pfrzdtr;
                }
            }

            // :835-856 slopes for the process rates, conduction/diffusion
            // and ventilation terms.
            sr_ = wsm6_slope_rain(c, qrk, denk, denfac);
            ss_ = wsm6_slope_snow(c, qsk, denk, denfac, tk);
            sg_ = wsm6_slope_graup(c, qgk, denk, denfac);
            const float work1_1 = st_diffac(xlk, pk, tk, denk, qsat1);
            const float work1_2 = st_diffac(W_XLS, pk, tk, denk, qsat2);
            const float work2 = st_venfac(pk, tk, denk);

            float prevp = 0.f, psdep = 0.f, pgdep = 0.f, praut = 0.f, psaut = 0.f;
            float pgaut = 0.f, pracw = 0.f, praci = 0.f, piacr = 0.f, psaci = 0.f;
            float psacw = 0.f, pracs = 0.f, psacr = 0.f, pgacw = 0.f, paacw = 0.f;
            float pgaci = 0.f, pgacr = 0.f, pgacs = 0.f, pigen = 0.f, pidep = 0.f;
            float pseml = 0.f, pgeml = 0.f, psevp = 0.f, pgevp = 0.f;

            // :866-902 warm rain.
            {
                const float supsat = wmax(qk, W_QMIN) - qsat1;
                const float satdt = supsat / dtcld;
                if (qck > c[C_QC0]) {
                    praut = c[C_QCK1] * gfk_pow(qck, __fdiv_rn(7.0f, 3.0f));
                    praut = wmin(praut, qck / dtcld);
                }
                if (qrk > W_QCRMIN && qck > W_QMIN) {
                    pracw = wmin((((c[C_PACRR] * sr_.r3) * sr_.rb) * qck) * denfac,
                                 qck / dtcld);
                }
                if (qrk > 0.0f) {
                    const float coeres = sr_.r2 * sqrtf(sr_.r * sr_.rb);
                    prevp = ((rh1 - 1.0f) * (c[C_PRECR1] * sr_.r2
                                             + (c[C_PRECR2] * work2) * coeres))
                          / work1_1;
                    if (prevp < 0.0f) {
                        prevp = wmax(prevp, -qrk / dtcld);
                        prevp = wmax(prevp, satdt / 2.0f);
                    } else {
                        prevp = wmin(prevp, satdt / 2.0f);
                    }
                }
            }

            // :917-1195 cold rain.
            supcol = W_T0C - tk;
            n0sfac = wmax(wmin(gfk_exp(W_ALPHA * supcol), W_N0SMAX_OVER_N0S), 1.0f);
            const float supsat = wmax(qk, W_QMIN) - qsat2;
            const float satdt = supsat / dtcld;
            int ifsat = 0;
            {
                float temp = denk * wmax(qik, W_QMIN);
                temp = sqrtf(sqrtf((temp * temp) * temp));
                xni[k] = wmin(wmax(5.38e7f * temp, 1.0e3f), 1.0e6f);
            }
            const float xnik = xni[k];
            const float eacrs = gfk_exp(0.07f * (-supcol));
            const float xmi = (denk * qik) / xnik;
            const float diameter = wmin(W_DICON * sqrtf(xmi), W_DIMAX);
            const float vt2i = 1.49e4f * gfk_pow(diameter, 1.31f);
            const float vt2r = (c[C_PVTR] * sr_.rb) * denfac;
            const float vt2s = (c[C_PVTS] * ss_.rb) * denfac;
            const float vt2g = (c[C_PVTG] * sg_.rb) * denfac;
            const float qsum = wmax(qsk + qgk, 1.0e-15f);
            const float vt2ave = (qsum > 1.0e-15f) ? (vt2s * qsk + vt2g * qgk) / qsum
                                                   : 0.0f;
            if (supcol > 0.0f && qik > W_QMIN) {
                if (qrk > W_QCRMIN) {
                    float acrfac = (2.0f * sr_.r3 + (2.0f * diameter) * sr_.r2)
                                 + (diameter * diameter) * sr_.r;
                    praci = ((((pi * qik) * W_N0R) * fabsf(vt2r - vt2i)) * acrfac) / 4.0f;
                    float m = wmin(wmax(0.0f, qrk / qik), 1.0f);
                    praci = praci * (m * m);
                    praci = wmin(praci, qik / dtcld);
                    piacr = __fdiv_rn((pi * pi) * W_AVTR * W_N0R * W_DENR * xnik
                                      * denfac * c[C_G6PBR] * sr_.r3 * sr_.r3 * sr_.rb,
                                      24.0f) / denk;
                    m = wmin(wmax(0.0f, qik / qrk), 1.0f);
                    piacr = piacr * (m * m);
                    piacr = wmin(piacr, qrk / dtcld);
                }
                if (qsk > W_QCRMIN) {
                    const float acrfac = (2.0f * ss_.r3 + (2.0f * diameter) * ss_.r2)
                                       + (diameter * diameter) * ss_.r;
                    psaci = (((((((pi * qik) * eacrs) * W_N0S) * n0sfac)
                               * fabsf(vt2ave - vt2i)) * acrfac)) / 4.0f;
                    psaci = wmin(psaci, qik / dtcld);
                }
                if (qgk > W_QCRMIN) {
                    const float egi = gfk_exp(0.07f * (-supcol));
                    const float acrfac = (2.0f * sg_.r3 + (2.0f * diameter) * sg_.r2)
                                       + (diameter * diameter) * sg_.r;
                    pgaci = (((((pi * egi) * qik) * c[C_N0G]) * fabsf(vt2ave - vt2i))
                             * acrfac) / 4.0f;
                    pgaci = wmin(pgaci, qik / dtcld);
                }
            }
            if (qsk > W_QCRMIN && qck > W_QMIN) {
                const float m = wmin(wmax(0.0f, qsk / qck), 1.0f);
                psacw = wmin((((((c[C_PACRC] * n0sfac) * ss_.r3) * ss_.rb) * (m * m))
                              * qck) * denfac, qck / dtcld);
            }
            if (qgk > W_QCRMIN && qck > W_QMIN) {
                const float m = wmin(wmax(0.0f, qgk / qck), 1.0f);
                pgacw = wmin(((((c[C_PACRG] * sg_.r3) * sg_.rb) * (m * m)) * qck)
                             * denfac, qck / dtcld);
            }
            if (qsum > 1.0e-15f)
                paacw = (qsk * psacw + qgk * pgacw) / qsum;
            if (qsk > W_QCRMIN && qrk > W_QCRMIN) {
                if (supcol > 0.0f) {
                    const float acrfac = (((5.0f * ss_.r3) * ss_.r3) * sr_.r
                                          + ((2.0f * ss_.r3) * ss_.r2) * sr_.r2)
                                       + ((0.5f * ss_.r2) * ss_.r2) * sr_.r3;
                    pracs = (((((((pi * pi) * W_N0R) * W_N0S) * n0sfac)
                               * fabsf(vt2r - vt2ave)) * (W_DENS / denk)) * acrfac);
                    const float m = wmin(wmax(0.0f, qrk / qsk), 1.0f);
                    pracs = pracs * (m * m);
                    pracs = wmin(pracs, qsk / dtcld);
                }
                const float acrfac = (((5.0f * sr_.r3) * sr_.r3) * ss_.r
                                      + ((2.0f * sr_.r3) * sr_.r2) * ss_.r2)
                                   + ((0.5f * sr_.r2) * sr_.r2) * ss_.r3;
                psacr = (((((((pi * pi) * W_N0R) * W_N0S) * n0sfac)
                           * fabsf(vt2ave - vt2r)) * (W_DENR / denk)) * acrfac);
                const float m = wmin(wmax(0.0f, qsk / qrk), 1.0f);
                psacr = psacr * (m * m);
                psacr = wmin(psacr, qrk / dtcld);
            }
            if (qgk > W_QCRMIN && qrk > W_QCRMIN) {
                const float acrfac = (((5.0f * sr_.r3) * sr_.r3) * sg_.r
                                      + ((2.0f * sr_.r3) * sr_.r2) * sg_.r2)
                                   + ((0.5f * sr_.r2) * sr_.r2) * sg_.r3;
                pgacr = (((((pi * pi) * W_N0R) * c[C_N0G]) * fabsf(vt2ave - vt2r))
                         * (W_DENR / denk)) * acrfac;
                const float m = wmin(wmax(0.0f, qgk / qrk), 1.0f);
                pgacr = pgacr * (m * m);
                pgacr = wmin(pgacr, qrk / dtcld);
            }
            if (qgk > W_QCRMIN && qsk > W_QCRMIN) pgacs = 0.0f;
            if (supcol <= 0.0f) {
                const float xlf = W_XLF0;
                if (qsk > 0.0f)
                    pseml = wmin(wmax(__fdiv_rn((W_CLIQ * supcol) * (paacw + psacr), xlf),
                                      -qsk / dtcld), 0.0f);
                if (qgk > 0.0f)
                    pgeml = wmin(wmax(__fdiv_rn((W_CLIQ * supcol) * (paacw + pgacr), xlf),
                                      -qgk / dtcld), 0.0f);
            }
            if (supcol > 0.0f) {
                if (qik > 0.0f && ifsat != 1) {
                    pidep = (((4.0f * diameter) * xnik) * (rh2 - 1.0f)) / work1_2;
                    const float supice = satdt - prevp;
                    if (pidep < 0.0f) {
                        pidep = wmax(wmax(pidep, satdt / 2.0f), supice);
                        pidep = wmax(pidep, -qik / dtcld);
                    } else {
                        pidep = wmin(wmin(pidep, satdt / 2.0f), supice);
                    }
                    if (fabsf(prevp + pidep) >= fabsf(satdt)) ifsat = 1;
                }
                if (qsk > 0.0f && ifsat != 1) {
                    const float coeres = ss_.r2 * sqrtf(ss_.r * ss_.rb);
                    psdep = (((rh2 - 1.0f) * n0sfac)
                             * (c[C_PRECS1] * ss_.r2 + (c[C_PRECS2] * work2) * coeres))
                          / work1_2;
                    const float supice = (satdt - prevp) - pidep;
                    if (psdep < 0.0f) {
                        psdep = wmax(psdep, -qsk / dtcld);
                        psdep = wmax(wmax(psdep, satdt / 2.0f), supice);
                    } else {
                        psdep = wmin(wmin(psdep, satdt / 2.0f), supice);
                    }
                    if (fabsf((prevp + pidep) + psdep) >= fabsf(satdt)) ifsat = 1;
                }
                if (qgk > 0.0f && ifsat != 1) {
                    const float coeres = sg_.r2 * sqrtf(sg_.r * sg_.rb);
                    pgdep = ((rh2 - 1.0f)
                             * (c[C_PRECG1] * sg_.r2 + (c[C_PRECG2] * work2) * coeres))
                          / work1_2;
                    const float supice = ((satdt - prevp) - pidep) - psdep;
                    if (pgdep < 0.0f) {
                        pgdep = wmax(pgdep, -qgk / dtcld);
                        pgdep = wmax(wmax(pgdep, satdt / 2.0f), supice);
                    } else {
                        pgdep = wmin(wmin(pgdep, satdt / 2.0f), supice);
                    }
                    if (fabsf(((prevp + pidep) + psdep) + pgdep) >= fabsf(satdt))
                        ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {
                    const float supice = (((satdt - prevp) - pidep) - psdep) - pgdep;
                    const float xni0 = 1.0e3f * gfk_exp(0.1f * supcol);
                    const float roqi0 = 4.92e-11f * gfk_pow(xni0, 1.33f);
                    pigen = wmax(0.0f, (roqi0 / denk - wmax(qik, 0.0f)) / dtcld);
                    pigen = wmin(wmin(pigen, satdt), supice);
                }
                if (qik > 0.0f) {
                    const float qimax = c[C_ROQIMAX] / denk;
                    psaut = wmax(0.0f, (qik - qimax) / dtcld);
                }
                if (qsk > 0.0f) {
                    const float alpha2 = 1.0e-3f * gfk_exp(0.09f * (-supcol));
                    pgaut = wmin(wmax(0.0f, alpha2 * (qsk - W_QS0)), qsk / dtcld);
                }
            }
            if (supcol < 0.0f) {
                if (qsk > 0.0f && rh1 < 1.0f) {
                    const float coeres = ss_.r2 * sqrtf(ss_.r * ss_.rb);
                    psevp = (((rh1 - 1.0f) * n0sfac)
                             * (c[C_PRECS1] * ss_.r2 + (c[C_PRECS2] * work2) * coeres))
                          / work1_1;
                    psevp = wmin(wmax(psevp, -qsk / dtcld), 0.0f);
                }
                if (qgk > 0.0f && rh1 < 1.0f) {
                    const float coeres = sg_.r2 * sqrtf(sg_.r * sg_.rb);
                    pgevp = ((rh1 - 1.0f)
                             * (c[C_PRECG1] * sg_.r2 + (c[C_PRECG2] * work2) * coeres))
                          / work1_1;
                    pgevp = wmin(wmax(pgevp, -qgk / dtcld), 0.0f);
                }
            }

            // :1202-1387 mass conservation of the generation terms, then the
            // update and its latent heating.  C's + and - associate left to
            // right as Fortran's do, so each sum is WRF's line as written.
            {
                float delta2 = 0.0f, delta3 = 0.0f;
                if (qrk < 1.0e-4f && qsk < 1.0e-4f) delta2 = 1.0f;
                if (qrk < 1.0e-4f) delta3 = 1.0f;
                float value, source, factor;
                if (tk <= W_T0C) {
                    value = wmax(W_QMIN, qck);
                    source = (praut + pracw + paacw + paacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        paacw = paacw * factor;
                    }
                    value = wmax(W_QMIN, qik);
                    source = (psaut - pigen - pidep + praci + psaci + pgaci) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        psaut = psaut * factor;
                        pigen = pigen * factor;
                        pidep = pidep * factor;
                        praci = praci * factor;
                        psaci = psaci * factor;
                        pgaci = pgaci * factor;
                    }
                    value = wmax(W_QMIN, qrk);
                    source = (-praut - prevp - pracw + piacr + psacr + pgacr) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        prevp = prevp * factor;
                        pracw = pracw * factor;
                        piacr = piacr * factor;
                        psacr = psacr * factor;
                        pgacr = pgacr * factor;
                    }
                    value = wmax(W_QMIN, qsk);
                    source = -((psdep + psaut - pgaut + paacw + piacr * delta3
                                + praci * delta3 - pracs * (1.0f - delta2)
                                + psacr * delta2 + psaci - pgacs) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        psdep = psdep * factor;
                        psaut = psaut * factor;
                        pgaut = pgaut * factor;
                        paacw = paacw * factor;
                        piacr = piacr * factor;
                        praci = praci * factor;
                        psaci = psaci * factor;
                        pracs = pracs * factor;
                        psacr = psacr * factor;
                        pgacs = pgacs * factor;
                    }
                    value = wmax(W_QMIN, qgk);
                    source = -((pgdep + pgaut
                                + piacr * (1.0f - delta3) + praci * (1.0f - delta3)
                                + psacr * (1.0f - delta2) + pracs * (1.0f - delta2)
                                + pgaci + paacw + pgacr + pgacs) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        pgdep = pgdep * factor;
                        pgaut = pgaut * factor;
                        piacr = piacr * factor;
                        praci = praci * factor;
                        psacr = psacr * factor;
                        pracs = pracs * factor;
                        paacw = paacw * factor;
                        pgaci = pgaci * factor;
                        pgacr = pgacr * factor;
                        pgacs = pgacs * factor;
                    }
                    const float work2u = -(prevp + psdep + pgdep + pigen + pidep);
                    qk = qk + work2u * dtcld;
                    qck = wmax(qck - (praut + pracw + paacw + paacw) * dtcld, 0.0f);
                    qrk = wmax(qrk + (praut + pracw + prevp - piacr - pgacr
                                      - psacr) * dtcld, 0.0f);
                    qik = wmax(qik - (psaut + praci + psaci + pgaci - pigen
                                      - pidep) * dtcld, 0.0f);
                    qsk = wmax(qsk + (psdep + psaut + paacw - pgaut + piacr * delta3
                                      + praci * delta3 + psaci - pgacs
                                      - pracs * (1.0f - delta2) + psacr * delta2)
                                     * dtcld, 0.0f);
                    qgk = wmax(qgk + (pgdep + pgaut + piacr * (1.0f - delta3)
                                      + praci * (1.0f - delta3) + psacr * (1.0f - delta2)
                                      + pracs * (1.0f - delta2) + pgaci + paacw
                                      + pgacr + pgacs) * dtcld, 0.0f);
                    const float xlf = W_XLS - xlk;
                    const float xlwork2 = -(W_XLS * (psdep + pgdep + pidep + pigen))
                                        - xlk * prevp
                                        - xlf * (piacr + paacw + paacw + pgacr + psacr);
                    tk = tk - (xlwork2 / cpmk) * dtcld;
                } else {
                    value = wmax(W_QMIN, qck);
                    source = (praut + pracw + paacw + paacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        paacw = paacw * factor;
                    }
                    value = wmax(W_QMIN, qrk);
                    source = (-paacw - praut + pseml + pgeml - pracw - paacw - prevp)
                           * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        prevp = prevp * factor;
                        pracw = pracw * factor;
                        paacw = paacw * factor;
                        pseml = pseml * factor;
                        pgeml = pgeml * factor;
                    }
                    value = wmax(W_QCRMIN, qsk);
                    source = (pgacs - pseml - psevp) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        pgacs = pgacs * factor;
                        psevp = psevp * factor;
                        pseml = pseml * factor;
                    }
                    value = wmax(W_QCRMIN, qgk);
                    source = -((pgacs + pgevp + pgeml) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        pgacs = pgacs * factor;
                        pgevp = pgevp * factor;
                        pgeml = pgeml * factor;
                    }
                    const float work2u = -(prevp + psevp + pgevp);
                    qk = qk + work2u * dtcld;
                    qck = wmax(qck - (praut + pracw + paacw + paacw) * dtcld, 0.0f);
                    qrk = wmax(qrk + (praut + pracw + prevp + paacw + paacw - pseml
                                      - pgeml) * dtcld, 0.0f);
                    qsk = wmax(qsk + (psevp - pgacs + pseml) * dtcld, 0.0f);
                    qgk = wmax(qgk + (pgacs + pgevp + pgeml) * dtcld, 0.0f);
                    const float xlf = W_XLS - xlk;
                    const float xlwork2 = -(xlk * (prevp + psevp + pgevp))
                                        - xlf * (pseml + pgeml);
                    tk = tk - (xlwork2 / cpmk) * dtcld;
                }
            }

            // :1392-1437 condensation/evaporation of cloud water.
            {
                const float qsatn = wsm6_qsat(fp, tk, pk, false);
                const float work1 = st_conden(tk, qk, qsatn, xlk, cpmk);
                float pcond = wmin(wmax(work1 / dtcld, 0.0f), wmax(qk, 0.0f) / dtcld);
                if (qck > 0.0f && work1 < 0.0f)
                    pcond = wmax(work1, -qck) / dtcld;
                qk = qk - pcond * dtcld;
                qck = wmax(qck + pcond * dtcld, 0.0f);
                tk = tk + (((pcond * xlk) / cpmk) * dtcld);
            }
            // :1443-1448 padding for small values.
            if (qck <= W_QMIN) qck = 0.0f;
            if (qik <= W_QMIN) qik = 0.0f;

            t[k] = tk; q[k] = qk;
            qcl[k] = qck; qil[k] = qik; qrl[k] = qrk; qsl[k] = qsk; qgl[k] = qgk;
        }
    }

    rainnc[col] = rain_c; rainncv[col] = rainncv_c;
    snownc[col] = snow_c; snowncv[col] = snowncv_c;
    graupelnc[col] = graupel_c; graupelncv[col] = graupelncv_c;
    sr[col] = sr_c;

    // module_mp_wsm6.F:161-232: th = t/pii, then mp_wsm6_effectRad_run on
    // th*pii.  Radii are formed in metres as WRF stores them; gpuwm's
    // radiation contract is microns, the one multiply WRF's RRTMG applies
    // when it reads them (re*1.e6).
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        const float th = t[k] / pii[id];
        theta[id] = th;
        qv[id] = q[k]; qc[id] = qcl[k]; qi[id] = qil[k];
        qr[id] = qrl[k]; qs[id] = qsl[k]; qg[id] = qgl[k];
        const float tt = th * pii[id];
        const float rho = denc[k * stride];
        float re_qc = 2.49e-6f, re_qi = 4.99e-6f, re_qs = 9.99e-6f;
        const float rqc = wmax(1.0e-12f, qcl[k] * rho);
        const float rqi = wmax(1.0e-12f, qil[k] * rho);
        float temp = rho * wmax(qil[k], W_QMIN);
        temp = sqrtf(sqrtf((temp * temp) * temp));
        const float ni = wmin(wmax(5.38e7f * temp, 1.0e3f), 1.0e6f);
        const float rni = wmax(1.0e-6f, ni * rho);
        const float rqs = wmax(1.0e-12f, qsl[k] * rho);
        if (rqc > 1.0e-12f) {
            const float lamdac = gfk_pow((c[C_PIDNC] * 3.0e8f) / rqc, __fdiv_rn(1.0f, 3.0f));
            re_qc = wmax(2.51e-6f, wmin(1.5f * (1.0f / lamdac), 50.0e-6f));
        }
        if (rqi > 1.0e-12f && rni > 1.0e-6f) {
            const float diai = 11.9f * sqrtf(rqi / ni);
            re_qi = wmax(10.01e-6f, wmin((0.75f * 0.163f) * diai, 125.0e-6f));
        }
        if (rqs > 1.0e-12f) {
            const float supc = W_T0C - tt;
            const float nsf = wmax(wmin(gfk_exp(W_ALPHA * supc), W_N0SMAX_OVER_N0S), 1.0f);
            const float lamdas = sqrtf(sqrtf((c[C_PIDN0S] * nsf) / rqs));
            re_qs = wmax(25.0e-6f, wmin(0.5f * (1.0f / lamdas), 999.0e-6f));
        }
        re_qc = wmax(2.49e-6f, wmin(re_qc, 50.0e-6f));
        re_qi = wmax(4.99e-6f, wmin(re_qi, 125.0e-6f));
        re_qs = wmax(9.99e-6f, wmin(re_qs, 999.0e-6f));
        effc[id] = re_qc * 1.0e6f;
        effi[id] = re_qi * 1.0e6f;
        effs[id] = re_qs * 1.0e6f;
    }
}

extern "C" __global__ void wsm6_column(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *sr, float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WSM6_KMAX) return;
    wsm6_column_impl(theta, qv, qc, qi, qr, qs, qg, den, p, pii, dz,
                     rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
                     sr, effc, effi, effs, coef, delt, nz, stride, col);
}

// ===========================================================================
// REFL_10CM for WSM6: phys/physics_mmm/mp_wsm6.F90 refl10cm_wsm6 (:2275-2444)
// with mp_radar.F90's rayleigh_soak_wetgraupel / get_m_mix_nested /
// get_m_mix / m_complex_maxwellgarnett for the melting particles, and the
// wrapper's floor MAX(-35., dBZ) (module_mp_wsm6.F:246).
//
// PRECISION is WRF's mix, kept line by line: temperature, density, the
// species densities, the snow intercept and the dry-particle prefactors are
// float32 (gfk_exp for the snow intercept's EXP); the slopes, moments and
// the melting integral are float64, rounded to float32 where WRF assigns
// them to a REAL array.  Binary64 EXP, LOG and REAL**REAL are WOOF's own
// glibc_flt64.cuh words (glibc_exp, glibc_log, glibc_pow).  LOG10, ATAN2
// and HYPOT, and so the complex LOG/SQRT/ABS of the melting term, are
// CUDA's binary64 functions: their last bit can differ from the oracle
// host's, and WRF rounds every such value to float32 before it is stored,
// so a stored word moves only when a binary64 result sits within an ULP of
// a float32 rounding boundary.  gfortran's complex arithmetic: products
// (ac-bd, ad+bc), quotients Smith's range-reduced form
// (-fcx-fortran-rules), integer powers above 2 through libgcc's __powidf2
// square-and-multiply (what gfortran -O0 calls).
//
// The radar constants are mp_radar's own words from radar_init after
// mp_wsm6_init (gpuwm/core/wsm6_constants.py WRF_RADAR_*), not
// recomputations: xcrg(4) = WGAMMA(7.) is 720.00006 in float32, and pi5 is
// a float32 product.
// ===========================================================================

enum Wsm6RadarF32 {
    RF_PI, RF_N0G, RF_XAM_R, RF_XAM_S, RF_XAM_G,
    RF_XCRE1, RF_XCRE4, RF_XCRG3, RF_XCRG4,
    RF_XCSE1, RF_XCSE4, RF_XCSG3, RF_XCSG4,
    RF_XCGE1, RF_XCGE4, RF_XCGG3, RF_XCGG4,
    RF_XBM_S, RF_XBM_G, RF_XMU_S, RF_XMU_G,
    RF_XOCMS, RF_XOBMS, RF_XOCMG, RF_XOBMG,
    WSM6_NRADAR_F32
};
#define WSM6_NRBINS 50
enum Wsm6RadarF64 {
    RD_K_W, RD_PI5, RD_LAMDA4, RD_MW_RE, RD_MW_IM, RD_MI_RE, RD_MI_IM,
    RD_XXDS, RD_XDTS = RD_XXDS + WSM6_NRBINS, RD_XXDG = RD_XDTS + WSM6_NRBINS,
    RD_XDTG = RD_XXDG + WSM6_NRBINS, RD_SIMPSON = RD_XDTG + WSM6_NRBINS,
    WSM6_NRADAR_F64 = RD_SIMPSON + WSM6_NRBINS + 1
};

struct WCplx { double re, im; };
__device__ __forceinline__ WCplx wc(double re, double im) {
    WCplx z; z.re = re; z.im = im; return z;
}
__device__ __forceinline__ WCplx wc_add(WCplx a, WCplx b) {
    return wc(a.re + b.re, a.im + b.im);
}
__device__ __forceinline__ WCplx wc_sub(WCplx a, WCplx b) {
    return wc(a.re - b.re, a.im - b.im);
}
__device__ __forceinline__ WCplx wc_mul(WCplx a, WCplx b) {
    return wc(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
// Smith's division as GCC lowers it for Fortran (range reduced, no NaN
// recovery).
__device__ __forceinline__ WCplx wc_div(WCplx a, WCplx b) {
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
    return wc(tr / div, ti / div);
}
// real * complex: the real operand promoted to (x, 0), then the product.
__device__ __forceinline__ WCplx wc_rmul(double x, WCplx b) {
    return wc_mul(wc(x, 0.0), b);
}
__device__ __forceinline__ double wc_abs(WCplx a) { return hypot(a.re, a.im); }
__device__ __forceinline__ WCplx wc_log(WCplx a) {
    return wc(glibc_log(hypot(a.re, a.im)), atan2(a.im, a.re));
}
// Principal square root, the textbook half-angle form: one part from the
// larger of |z| + |Re z|, the other from 2 Re Im = Im z.
__device__ __forceinline__ WCplx wc_sqrt(WCplx a) {
    if (a.re == 0.0 && a.im == 0.0) return wc(0.0, a.im);
    const double d = hypot(a.re, a.im);
    double r, s;
    if (a.re > 0.0) {
        r = sqrt(0.5 * (d + a.re));
        s = 0.5 * (a.im / r);
    } else {
        s = sqrt(0.5 * (d - a.re));
        r = fabs(0.5 * (a.im / s));
    }
    return wc(r, copysign(s, a.im));
}
// x**n for n > 2 as gfortran -O0 evaluates it: square and multiply.
__device__ __forceinline__ double w_powi(double x, int m) {
    unsigned int n = (unsigned int)m;
    double y = (n % 2) ? x : 1.0;
    while (n >>= 1) {
        x = x * x;
        if (n % 2) y = y * x;
    }
    return y;
}
__device__ __forceinline__ double wdmax(double a, double b) { return (b > a) ? b : a; }
__device__ __forceinline__ double wdmin(double a, double b) { return (b < a) ? b : a; }

// m_complex_maxwellgarnett with inclusion = 'spheroidal'.
__device__ WCplx wsm6_maxwellgarnett(double vol1, double vol2, double vol3,
                                     WCplx m1, WCplx m2, WCplx m3, int *error)
{
    *error = 0;
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1.0e-6) {
        *error = 1;
        return wc(-999.99, -999.99);
    }
    const WCplx m1t = wc_mul(m1, m1);
    const WCplx m2t = wc_mul(m2, m2);
    const WCplx m3t = wc_mul(m3, m3);
    const WCplx one = wc(1.0, 0.0);
    const WCplx beta2 = wc_mul(
        wc_div(wc_rmul(2.0, m1t), wc_sub(m2t, m1t)),
        wc_sub(wc_mul(wc_div(m2t, wc_sub(m2t, m1t)), wc_log(wc_div(m2t, m1t))), one));
    const WCplx beta3 = wc_mul(
        wc_div(wc_rmul(2.0, m1t), wc_sub(m3t, m1t)),
        wc_sub(wc_mul(wc_div(m3t, wc_sub(m3t, m1t)), wc_log(wc_div(m3t, m1t))), one));
    const double v1 = 1.0 - vol2 - vol3;
    const WCplx num = wc_add(wc_add(wc_rmul(v1, m1t),
                                    wc_mul(wc_rmul(vol2, beta2), m2t)),
                             wc_mul(wc_rmul(vol3, beta3), m3t));
    const WCplx den = wc_add(wc_add(wc(v1, 0.0), wc_rmul(vol2, beta2)),
                             wc_rmul(vol3, beta3));
    return wc_sqrt(wc_div(num, den));
}

// rayleigh_soak_wetgraupel with the strings WSM6's radar_init sets: mixing
// rule maxwellgarnett, host air, matrix water, inclusion spheroidal, host
// matrix icewater, host inclusion spheroidal.
__device__ double wsm6_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, WCplx m_w, WCplx m_i, double pi5, double lamda4)
{
    const double pix = 3.1415926535897932384626434;
    const WCplx m_air = wc(1.0, 0.0);
    const double fm = wdmax(wdmin(fmelt, 1.0), 0.0);
    double mra = wdmax(wdmin(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    const double x_w = x_g * fm;
    const double d_g = a_geo * glibc_pow(x_g, b_geo);
    if (!(d_g >= 1.0e-12)) return 0.0;
    double vg = (pix / 6.0) * w_powi(d_g, 3);
    const double rhog = wdmax(wdmin(x_g / vg, 900.0), 10.0);
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
    const double vol1 = volice / wdmax(volice + volwater, 1.0e-10);
    const double vol2 = 1.0 - vol1;
    const WCplx mtmp = wsm6_maxwellgarnett(vol2, 0.0, vol1, m_w, m_air, m_i, &err1);
    const WCplx m_core = wsm6_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_air,
                                             wc_rmul(2.0, m_air), &err2);
    if (err1 + err2 != 0) return 0.0;
    const WCplx mc2 = wc_mul(m_core, m_core);
    const double a = wc_abs(wc_div(wc_sub(mc2, wc(1.0, 0.0)),
                                   wc_add(mc2, wc(2.0, 0.0))));
    return (((a * a) * pi5) * w_powi(d_large, 6)) / lamda4;
}

extern "C" __global__ void wsm6_refl10cm_column(
    const float *qv, const float *qr, const float *qs, const float *qg,
    const float *t, const float *p, const float *__restrict__ rf,
    const double *__restrict__ rd, float *refl, int melting,
    int nz, int ny, int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = ny * nx;
    if (col >= stride || nz > WSM6_KMAX) return;
    float temp[WSM6_KMAX], rs[WSM6_KMAX], rg[WSM6_KMAX];
    double ilamr[WSM6_KMAX], ilams[WSM6_KMAX], ilamg[WSM6_KMAX], n0_s[WSM6_KMAX];
    float ze_snow[WSM6_KMAX], ze_graupel[WSM6_KMAX];
    unsigned char lq[WSM6_KMAX];   // bit 0 rain, 1 snow, 2 graupel
    const float pi = rf[RF_PI];
    const double n0_r = (double)W_N0R;
    const double n0_g = (double)rf[RF_N0G];

    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        temp[k] = t[id];
        const float temp_c = wmin(-0.001f, temp[k] - 273.15f);
        const float qvk = wmax(1.0e-10f, qv[id]);
        const float pres = p[id];
        const float rho = (0.622f * pres) / ((287.0f * temp[k]) * (qvk + 0.622f));
        lq[k] = 0;
        if (qr[id] > 1.0e-9f) {
            const float rr = qr[id] * rho;
            const double lamr = glibc_pow(
                (double)(rf[RF_XAM_R] * rf[RF_XCRG3]) * n0_r / (double)rr,
                (double)(1.0f / rf[RF_XCRE1]));
            ilamr[k] = 1.0 / lamr;
            lq[k] |= 1;
        }
        if (qs[id] > 1.0e-9f) {
            rs[k] = qs[id] * rho;
            n0_s[k] = (double)wmin(1.0e11f, W_N0S * gfk_exp(-W_ALPHA * temp_c));
            const double lams = glibc_pow(
                (double)(rf[RF_XAM_S] * rf[RF_XCSG3]) * n0_s[k] / (double)rs[k],
                (double)(1.0f / rf[RF_XCSE1]));
            ilams[k] = 1.0 / lams;
            lq[k] |= 2;
        } else {
            rs[k] = 1.0e-12f;
        }
        if (qg[id] > 1.0e-9f) {
            rg[k] = qg[id] * rho;
            const double lamg = glibc_pow(
                (double)(rf[RF_XAM_G] * rf[RF_XCGG3]) * n0_g / (double)rg[k],
                (double)(1.0f / rf[RF_XCGE1]));
            ilamg[k] = 1.0 / lamg;
            lq[k] |= 4;
        } else {
            rg[k] = 1.0e-12f;
        }
    }

    bool melti = false;
    int k_0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (temp[k] > 273.15f && (lq[k] & 1) && ((lq[k + 1] & 2) || (lq[k + 1] & 4))) {
            k_0 = (k + 1 > k_0) ? k + 1 : k_0;
            melti = true;
            break;
        }
    }

    // Rayleigh terms of dry snow and graupel (the prefactor float32 as WRF
    // writes it, promoted where it meets the float64 intercept).
    const float six_over_pi = 6.0f / pi;
    const float pref_s = __fdiv_rn(0.176f, 0.93f) * six_over_pi * six_over_pi
                       * __fdiv_rn(rf[RF_XAM_S], 900.0f) * __fdiv_rn(rf[RF_XAM_S], 900.0f);
    const float pref_g = __fdiv_rn(0.176f, 0.93f) * six_over_pi * six_over_pi
                       * __fdiv_rn(rf[RF_XAM_G], 900.0f) * __fdiv_rn(rf[RF_XAM_G], 900.0f);
    for (int k = 0; k < nz; ++k) {
        ze_snow[k] = 1.0e-22f;
        ze_graupel[k] = 1.0e-22f;
        if (lq[k] & 2)
            ze_snow[k] = (float)((double)pref_s * n0_s[k] * (double)rf[RF_XCSG4]
                                 * glibc_pow(ilams[k], (double)rf[RF_XCSE4]));
        if (lq[k] & 4)
            ze_graupel[k] = (float)((double)pref_g * n0_g * (double)rf[RF_XCGG4]
                                    * glibc_pow(ilamg[k], (double)rf[RF_XCGE4]));
    }

    if (melting && melti && k_0 >= 1) {
        const WCplx m_w = wc(rd[RD_MW_RE], rd[RD_MW_IM]);
        const WCplx m_i = wc(rd[RD_MI_RE], rd[RD_MI_IM]);
        const double pi5 = rd[RD_PI5], lamda4 = rd[RD_LAMDA4], k_w = rd[RD_K_W];
        for (int k = k_0 - 1; k >= 0; --k) {
            if ((lq[k] & 2) && (lq[k_0] & 2)) {
                const double fmelt_s = wdmax(0.005, wdmin(1.0 - (double)(rs[k] / rs[k_0]),
                                                          0.99));
                double eta = 0.0;
                const double lams = 1.0 / ilams[k];
                for (int n = 0; n < WSM6_NRBINS; ++n) {
                    const double xx = rd[RD_XXDS + n];
                    const double x = (double)rf[RF_XAM_S]
                                   * glibc_pow(xx, (double)rf[RF_XBM_S]);
                    const double cback = wsm6_rayleigh_soak_wetgraupel(
                        x, (double)rf[RF_XOCMS], (double)rf[RF_XOBMS], fmelt_s, 0.9,
                        m_w, m_i, pi5, lamda4);
                    const double f_d = n0_s[k] * glibc_pow(xx, (double)rf[RF_XMU_S])
                                     * glibc_exp(-lams * xx);
                    eta = eta + f_d * cback * rd[RD_SIMPSON + n] * rd[RD_XDTS + n];
                }
                ze_snow[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
            if ((lq[k] & 4) && (lq[k_0] & 4)) {
                const double fmelt_g = wdmax(0.005, wdmin(1.0 - (double)(rg[k] / rg[k_0]),
                                                          0.99));
                double eta = 0.0;
                const double lamg = 1.0 / ilamg[k];
                for (int n = 0; n < WSM6_NRBINS; ++n) {
                    const double xx = rd[RD_XXDG + n];
                    const double x = (double)rf[RF_XAM_G]
                                   * glibc_pow(xx, (double)rf[RF_XBM_G]);
                    const double cback = wsm6_rayleigh_soak_wetgraupel(
                        x, (double)rf[RF_XOCMG], (double)rf[RF_XOBMG], fmelt_g, 0.9,
                        m_w, m_i, pi5, lamda4);
                    const double f_d = n0_g * glibc_pow(xx, (double)rf[RF_XMU_G])
                                     * glibc_exp(-lamg * xx);
                    eta = eta + f_d * cback * rd[RD_SIMPSON + n] * rd[RD_XDTG + n];
                }
                ze_graupel[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
        }
    }

    for (int k = nz - 1; k >= 0; --k) {
        float ze_rain = 1.0e-22f;
        if (lq[k] & 1)
            ze_rain = (float)(n0_r * (double)rf[RF_XCRG4]
                              * glibc_pow(ilamr[k], (double)rf[RF_XCRE4]));
        const float zsum = ze_rain + ze_snow[k] + ze_graupel[k];
        const float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));
        refl[col + k * stride] = wmax(-35.0f, dbz);
    }
}
