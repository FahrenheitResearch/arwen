// WRF v4.6.1 WSM3 simple-ice three-class microphysics (mp_physics = 3).
//
// A word-for-word port of WRF's own call chain for one column,
// phys/module_mp_wsm3.F:
//   wsm3          the wrapper module_microphysics_driver.F:2230 calls
//                 (t = th*pii in, th = t/pii out, effective radii from
//                 th*pii with the phase split read from the scheme's t)
//   wsm32D        the column scheme, its statement functions and the
//                 inlined fpvs
//   slope_wsm3, nislfv_rain_plm, effectRad_wsm3
// WSM3 carries vapour, cloud (``qc``: cloud water at or above 0 C, cloud
// ice below) and precipitation (``qr``: rain at or above 0 C, snow below).
// WRF computes no REFL_10CM for it.
//
// BITWISE CONTRACT.  Graded word for word against those routines compiled
// by gfortran (tools/wsm3_wrf461_column_oracle, tests/
// test_wsm3_wrf461_column_exact.py).  What that takes:
//   * Every expression keeps Fortran's left-to-right association; a
//     statement function (diffus, viscos, xka, diffac, venfac, conden,
//     cpmcal, xlcal) is a value, exactly as gfortran evaluates it.
//   * EXP, LOG and REAL**REAL are WOOF's own float32 routines gfk_exp,
//     gfk_log and gfk_pow (glibc_flt32.cuh).  `x**2` is x*x.  NINT is
//     roundf (half away from zero, as gfortran's lroundf).
//   * Fortran MAX/MIN are gfortran's MAX_EXPR chains: MAX(a,b) is
//     `b > a ? b : a` and MIN(a,b) is `b < a ? b : a` (wmax/wmin below);
//     they keep a signed zero where fmaxf/fminf would not.
//   * A float division by a value WRF passes as a constant is __fdiv_rn:
//     NVRTC turns `x / C` into a multiply by the rounded reciprocal under
//     -ftz=true.
//   * The unit compiles with --fmad=false (kernels/__init__.py
//     _NO_FMAD_MODULES): gfortran on x86-64 contracts nothing.
//   * The derived constants are the float32 words wsm3init computes (its
//     rgmma Gamma product included), frozen in gpuwm/core/wsm3_constants.py
//     WRF_INIT_F32 and passed in `coef`.
//
// DECLARED DIVERGENCES FROM WRF v4.6.1 (each a WRF defect, fixed here,
// default-on; see gpuwm/core/wsm3.py and docs/wsm3_known_deltas.md):
//   D1 cloud ice that falls out of a lowest level at or above 0 C is
//      counted as rain.  WRF adds fallc(i,1) to the surface only when
//      t(i,1) < t0c (module_mp_wsm3.F:604-607), but the semi-Lagrangian
//      CFL limiter moves wi(1) off zero whenever ice falls into a warm
//      lowest level from a cold one above, so ice leaves the column and
//      WRF counts it nowhere: water is lost. Strict verification keeps
//      stock surface accounting; the default cold arm is unchanged.
//   D2 a column whose top level is the melting level with rising air
//      there freezes at the top level itself.  WRF moves the freezing
//      level to kte+1 (:572-574) and reads t(i,kte+1) out of the bounds
//      of its two-dimensional temperature array (undefined behaviour).
//   D3 default remap search includes the final arrival interface and
//      zero-density extension, with kt >= kb, using the shared WSM6/WDM6
//      repair. Strict verification retains the stock search.
//
// LOOP FUSION.  From the second slope call to the end of a minor step
// (:630-875) every WRF loop reads and writes level k only, so this port runs
// them as one k loop; the order of operations at each level is WRF's.
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.

#ifndef WSM3_KMAX
#define WSM3_KMAX 64
#endif

// gfortran's MAX_EXPR/MIN_EXPR on two operands, argument order kept.
__device__ __forceinline__ float wmax(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float wmin(float a, float b) { return (b < a) ? b : a; }

// wsm3init's derived words, in WRF_INIT_F32 order.
enum Wsm3Coef {
    C_PI, C_XLV1, C_QC0, C_QCK1, C_PIDNC, C_PIDN0R, C_PIDN0S,
    C_PVTR, C_PVTS, C_PACRR, C_PACRS,
    C_PRECR1, C_PRECR2, C_PRECS1, C_PRECS2, C_ROQIMAX, C_XMMAX,
    C_RSLOPERMAX, C_RSLOPESMAX, C_RSLOPERBMAX, C_RSLOPESBMAX,
    C_RSLOPER2MAX, C_RSLOPES2MAX, C_RSLOPER3MAX, C_RSLOPES3MAX,
    WSM3_NCOEF
};

// module_model_constants.F / module_mp_wsm3.F parameters, as WRF passes them.
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
#define W_BVTR  0.8f
#define W_BVTS  0.41f
#define W_DICON 11.9f
#define W_DIMAX 500.0e-6f
#define W_N0S   2.0e6f
#define W_ALPHA 0.12f
#define W_QCRMIN 1.0e-9f
#define W_N0SMAX_OVER_N0S 5.0e4f   // n0smax/n0s, folded by gfortran: 1.e11/2.e6

// ---- statement functions of wsm32D (:348-361) ----------------------------
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

// fpvs inlined as wsm32D does (:432-441, :828-837).
struct Wsm3Fpvs { float ttp, xa, xb, xai, xbi; };
__device__ __forceinline__ Wsm3Fpvs wsm3_fpvs_consts() {
    Wsm3Fpvs f;
    f.ttp = W_T0C + 0.01f;
    const float dldt = W_CPV - W_CLIQ;
    f.xa = __fdiv_rn(-dldt, W_RV);
    f.xb = f.xa + __fdiv_rn(W_XLV0, W_RV * f.ttp);
    const float dldti = W_CPV - W_CICE;
    f.xai = __fdiv_rn(-dldti, W_RV);
    f.xbi = f.xai + __fdiv_rn(W_XLS, W_RV * f.ttp);
    return f;
}
// Saturation mixing ratio: over ice below the triple point when `ice`.
__device__ __forceinline__ float wsm3_qsat(const Wsm3Fpvs &f, float t, float p,
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

// ---- slope_wsm3 at one level (:983-1043) ---------------------------------
struct Wsm3Slope { float r, rb, r2, r3, vt; };

__device__ __forceinline__ Wsm3Slope wsm3_slope(const float *c, float q,
                                                float den, float denfac,
                                                float t) {
    Wsm3Slope s;
    float pvt;
    if (t >= 273.15f) {               // slope_wsm3's own PARAMETER t0c
        pvt = c[C_PVTR];
        if (q <= W_QCRMIN) {
            s.r = c[C_RSLOPERMAX]; s.rb = c[C_RSLOPERBMAX];
            s.r2 = c[C_RSLOPER2MAX]; s.r3 = c[C_RSLOPER3MAX];
        } else {
            s.r = 1.0f / sqrtf(sqrtf(c[C_PIDN0R] / (q * den)));
            s.rb = gfk_exp(gfk_log(s.r) * W_BVTR);
            s.r2 = s.r * s.r;
            s.r3 = s.r2 * s.r;
        }
    } else {
        const float supcol = 273.15f - t;
        const float n0sfac = wmax(wmin(gfk_exp(W_ALPHA * supcol),
                                       W_N0SMAX_OVER_N0S), 1.0f);
        pvt = c[C_PVTS];
        if (q <= W_QCRMIN) {
            s.r = c[C_RSLOPESMAX]; s.rb = c[C_RSLOPESBMAX];
            s.r2 = c[C_RSLOPES2MAX]; s.r3 = c[C_RSLOPES3MAX];
        } else {
            s.r = 1.0f / sqrtf(sqrtf((c[C_PIDN0S] * n0sfac) / (q * den)));
            s.rb = gfk_exp(gfk_log(s.r) * W_BVTS);
            s.r2 = s.r * s.r;
            s.r3 = s.r2 * s.r;
        }
    }
    s.vt = (pvt * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// ---- nislfv_rain_plm (:1241-1479) ------------------------------------------
// `rq` enters as den*q and leaves as the remapped den*q.  `wwl` is the
// departure speed and is not written, as in WRF.  iter = 1 recomputes the
// arrival speed with slope_wsm3 once (the precipitation call), iter = 0 is
// pure forward (the cloud-ice call).
//
// The default interface search includes the final arrival interface and
// zero-density extension, as in the delivered WSM6/WDM6 remap repair.
// Strict verification retains the stock search for bitwise WRF comparison.
__device__ void wsm3_nislfv(int km, const float *c, const float *den,
                            const float *dz, const float *tk,
                            const float *denfac, const float *wwl, float *rq,
                            float *precip_out, float dt, int iter, int stride)
{
    float ww[WSM3_KMAX], qn[WSM3_KMAX];
    float wi[WSM3_KMAX + 1], zi[WSM3_KMAX + 1], za[WSM3_KMAX + 1];
    float dza[WSM3_KMAX + 1], qa[WSM3_KMAX + 1];
    float qmi[WSM3_KMAX + 1], qpi[WSM3_KMAX + 1];

    *precip_out = 0.0f;
    float allold = 0.0f;
    for (int k = 0; k < km; ++k) allold = allold + rq[k];
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
        for (int k = 0; k < km; ++k) qa[k] = (rq[k] * dz[k * stride]) / dza[k];
        qa[km] = 0.0f;
        if (n > iter) break;
        // Arrival terminal speed from slope_wsm3 on qa/den; the mean of
        // departure and arrival (iter = 1: one pass, so no running average
        // with `was`).
        for (int k = 0; k < km; ++k) {
            const float denk = den[k * stride];
            const float wa = wsm3_slope(c, qa[k] / denk, denk, denfac[k], tk[k]).vt;
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
#ifdef GPUWM_WRF_EXACT
        // Strict reference arm: retain the unmodified WRF search.
        for (int kk = kt; kk < km; ++kk)
            if (zi[k + 1] <= za[kk]) { kt = kk; break; }
        kt = kt - 1;
#else
        // D3: shared WSM6/WDM6 upper-cell repair, default-on.
        const int kt_start = kt;
        kt = km + 1;
        for (int kk = kt_start; kk <= km; ++kk)
            if (zi[k + 1] <= za[kk]) { kt = kk; break; }
        kt = (kt - 1 > kb) ? kt - 1 : kb;
#endif
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
#ifdef WSM3_PLM_SURFACE
            // Correct mode, finding WB5 (physics.mode = "correct"; the WSM6
            // defect of kernels/wsm6.cu, shared by module_mp_wsm3.F's copy of
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

__device__ void wsm3_column_impl(
    float *theta, float *qv, float *qc, float *qr, const float *w,
    const float *den, const float *p, const float *pii, const float *dz,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int stride, int col)
{
    float c[WSM3_NCOEF];
    for (int n = 0; n < WSM3_NCOEF; ++n) c[n] = coef[n];
    const float pi = c[C_PI];
    const Wsm3Fpvs fp = wsm3_fpvs_consts();

    float t[WSM3_KMAX], q[WSM3_KMAX], qci[WSM3_KMAX], qrs[WSM3_KMAX];
    float cpm[WSM3_KMAX], xl[WSM3_KMAX], denfac[WSM3_KMAX];
    float qsl[WSM3_KMAX], rhl[WSM3_KMAX], xni[WSM3_KMAX];
    float vt[WSM3_KMAX], fall[WSM3_KMAX], rq[WSM3_KMAX];

    const float *denc = den + col;
    const float *dzc = dz + col;
    const float *pc = p + col;
    const float *wc = w + col;

    // wsm3 :136-141 t = th*pii, then wsm32D :369-392 the negative padding,
    // cpm and xl (held for every minor loop).
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        t[k] = theta[id] * pii[id];
        q[k] = qv[id];
        qci[k] = wmax(qc[id], 0.0f);
        qrs[k] = wmax(qr[id], 0.0f);
        const float qq = wmax(q[k], W_QMIN);
        cpm[k] = W_CPD * (1.0f - qq) + qq * W_CPV;
        xl[k] = W_XLV0 - c[C_XLV1] * (t[k] - W_T0C);
    }

    // :397-403
    float rainncv_c = 0.0f, snowncv_c = 0.0f, sr_c = 0.0f;
    float rain_c = rainnc[col], snow_c = snownc[col];

    // :408-410
    float loopsf = roundf(__fdiv_rn(delt, W_DTCLDCR));
    int loops = (int)loopsf;
    if (loops < 1) loops = 1;
    float dtcld = delt / (float)loops;
    if (delt <= W_DTCLDCR) dtcld = delt;

    for (int loop = 0; loop < loops; ++loop) {
        // :421-427 denfac = sqrt(den0 * (1/den)) by vsrec and vssqrt;
        // :442-457 saturation (over ice below the triple point) and rh;
        // :481-486 ice number; :497-498 precipitation fall speed.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            denfac[k] = sqrtf((1.0f / denk) * W_DEN0);
            const float qsk = wsm3_qsat(fp, t[k], pc[k * stride], true);
            qsl[k] = qsk;
            rhl[k] = wmax(q[k] / qsk, W_QMIN);
            xni[k] = wmin(wmax(5.38e7f * gfk_exp(gfk_log(denk * wmax(qci[k], W_QMIN))
                                                 * 0.75f), 1.0e3f), 1.0e6f);
            vt[k] = wsm3_slope(c, qrs[k], denk, denfac[k], t[k]).vt;
            rq[k] = denk * qrs[k];
        }
        // :508-518 precipitation sedimentation.
        float delqrs;
        wsm3_nislfv(nz, c, denc, dzc, t, denfac, vt, rq, &delqrs, dtcld, 1,
                    stride);
        for (int k = 0; k < nz; ++k) {
            qrs[k] = wmax(rq[k] / denc[k * stride], 0.0f);
            fall[k] = (rq[k] * vt[k]) / dzc[k * stride];
        }
        const float dz0 = dzc[0];
        fall[0] = (delqrs / dz0) / dtcld;

        // :522-550 cloud-ice fall speed and sedimentation (iter = 0).
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            if (t[k] < W_T0C && qci[k] > 0.0f) {
                const float xmi = (denk * qci[k]) / xni[k];
                const float diameter = wmax(W_DICON * sqrtf(xmi), 1.0e-25f);
                vt[k] = 1.49e4f * gfk_exp(gfk_log(diameter) * 1.31f);
            } else {
                vt[k] = 0.0f;
            }
            rq[k] = denk * qci[k];
        }
        float delqi;
        wsm3_nislfv(nz, c, denc, dzc, t, denfac, vt, rq, &delqi, dtcld, 0,
                    stride);
        for (int k = 0; k < nz; ++k)
            qci[k] = wmax(rq[k] / denc[k * stride], 0.0f);
        const float fallc = (delqi / dz0) / dtcld;

        // :556-596 freezing/melting at the melting level (1-based k, kk).
        {
            int mstep = 0;
            for (int k = 0; k < nz; ++k)
                if (t[k] >= W_T0C) mstep = k + 1;
            const int kk = mstep;
            int k = mstep;
            if (mstep != 0 && wc[(mstep - 1) * stride] > 0.0f) k = mstep + 1;
            // D2: WRF's k = kte+1 reads t out of bounds; freeze at kte.
            if (k > nz) k = nz;
            if (k * kk >= 1) {
                const int k0 = k - 1, kk0 = kk - 1;
                const float qrsci = qrs[k0] + qci[k0];
                if (qrsci > 0.0f || fall[kk0] > 0.0f) {
                    const float wk = wc[k0 * stride];
                    const float dzk = dzc[k0 * stride];
                    const float frzmlt = wmin(wmax(-((wk * qrsci) / dzk),
                                                   -qrsci / dtcld),
                                              qrsci / dtcld);
                    const float snomlt = wmin(wmax(fall[kk0] / denc[kk0 * stride],
                                                   -qrs[k0] / dtcld),
                                              qrs[k0] / dtcld);
                    if (k == kk) {
                        t[k0] = t[k0] - ((W_XLF0 / cpm[k0]) * (frzmlt + snomlt)) * dtcld;
                    } else {
                        t[k0] = t[k0] - ((W_XLF0 / cpm[k0]) * frzmlt) * dtcld;
                        t[kk0] = t[kk0] - ((W_XLF0 / cpm[kk0]) * snomlt) * dtcld;
                    }
                }
            }
        }

        // :601-625 surface rain and snow.  D1: fallc enters the rain total
        // whatever the lowest level's temperature; the snow arm is WRF's.
        {
            float fallsum = fall[0];
            float fallsum_qsi = 0.0f;
#ifndef GPUWM_WRF_EXACT
            // Default correction D1. Strict verification retains stock WRF.
            fallsum = fallsum + fallc;
#endif
            if ((W_T0C - t[0]) > 0.0f) {
#ifdef GPUWM_WRF_EXACT
                fallsum = fallsum + fallc;
#endif
                fallsum_qsi = fall[0] + fallc;
            }
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
            if (fallsum > 0.0f)
                sr_c = snowncv_c / (rainncv_c + 1.0e-12f);
        }

        // Everything below reads and writes level k only (:630-875).
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float pk = pc[k * stride];
            const float dfk = denfac[k];
            float tk = t[k], qk = q[k], qck = qci[k], qrk = qrs[k];
            const float cpmk = cpm[k], xlk = xl[k];
            const float qsk = qsl[k], rhk = rhl[k];

            // :635-651 slopes after sedimentation and melting; the
            // conduction/diffusion and ventilation terms.
            const Wsm3Slope s = wsm3_slope(c, qrk, denk, dfk, tk);
            const float work1 = (tk >= W_T0C)
                ? st_diffac(xlk, pk, tk, denk, qsk)
                : st_diffac(W_XLS, pk, tk, denk, qsk);
            const float work2 = st_venfac(pk, tk, denk);

            float pres = 0.0f, paut = 0.0f, pacr = 0.0f;
            float pgen = 0.0f, pisd = 0.0f;
            const float supsat = wmax(qk, W_QMIN) - qsk;
            const float satdt = supsat / dtcld;
            if (tk >= W_T0C) {
                // :670-697 warm rain.
                if (qck > c[C_QC0]) {
                    paut = c[C_QCK1] * gfk_exp(gfk_log(qck) * __fdiv_rn(7.0f, 3.0f));
                    paut = wmin(paut, qck / dtcld);
                }
                if (qrk > W_QCRMIN && qck > W_QMIN) {
                    pacr = wmin((((c[C_PACRR] * s.r3) * s.rb) * qck) * dfk,
                                qck / dtcld);
                }
                if (qrk > 0.0f) {
                    const float coeres = s.r2 * sqrtf(s.r * s.rb);
                    pres = ((rhk - 1.0f) * (c[C_PRECR1] * s.r2
                                            + (c[C_PRECR2] * work2) * coeres))
                         / work1;
                    if (pres < 0.0f) {
                        pres = wmax(pres, -qrk / dtcld);
                        pres = wmax(pres, satdt / 2.0f);
                    } else {
                        pres = wmin(pres, satdt / 2.0f);
                    }
                }
            } else {
                // :713-787 cold rain.
                const float supcol = W_T0C - tk;
                const float n0sfac = wmax(wmin(gfk_exp(W_ALPHA * supcol),
                                               W_N0SMAX_OVER_N0S), 1.0f);
                int ifsat = 0;
                const float xnik = wmin(wmax(5.38e7f * gfk_exp(
                    gfk_log(denk * wmax(qck, W_QMIN)) * 0.75f), 1.0e3f), 1.0e6f);
                const float eacrs = gfk_exp(0.07f * (-supcol));
                if (qrk > W_QCRMIN && qck > W_QMIN) {
                    const float xmi = (denk * qck) / xnik;
                    const float diameter = wmin(W_DICON * sqrtf(xmi), W_DIMAX);
                    const float vt2i = 1.49e4f * gfk_pow(diameter, 1.31f);
                    const float vt2s = (c[C_PVTS] * s.rb) * dfk;
                    const float acrfac = (2.0f * s.r3 + (2.0f * diameter) * s.r2)
                                       + (diameter * diameter) * s.r;
                    pacr = wmin(((((((pi * qck) * eacrs) * W_N0S) * n0sfac)
                                  * fabsf(vt2s - vt2i)) * acrfac) / 4.0f,
                                qck / dtcld);
                }
                if (qck > 0.0f) {
                    const float xmi = (denk * qck) / xnik;
                    const float diameter = W_DICON * sqrtf(xmi);
                    pisd = (((4.0f * diameter) * xnik) * (rhk - 1.0f)) / work1;
                    if (pisd < 0.0f) {
                        pisd = wmax(pisd, satdt / 2.0f);
                        pisd = wmax(pisd, -qck / dtcld);
                    } else {
                        pisd = wmin(pisd, satdt / 2.0f);
                    }
                    if (fabsf(pisd) >= fabsf(satdt)) ifsat = 1;
                }
                if (qrk > 0.0f && ifsat != 1) {
                    const float coeres = s.r2 * sqrtf(s.r * s.rb);
                    pres = (((rhk - 1.0f) * n0sfac)
                            * (c[C_PRECS1] * s.r2 + (c[C_PRECS2] * work2) * coeres))
                         / work1;
                    const float supice = satdt - pisd;
                    if (pres < 0.0f) {
                        pres = wmax(pres, -qrk / dtcld);
                        pres = wmax(wmax(pres, satdt / 2.0f), supice);
                    } else {
                        pres = wmin(wmin(pres, satdt / 2.0f), supice);
                    }
                    if (fabsf(pisd + pres) >= fabsf(satdt)) ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {
                    const float supice = (satdt - pisd) - pres;
                    const float xni0 = 1.0e3f * gfk_exp(0.1f * supcol);
                    const float roqi0 = 4.92e-11f * gfk_exp(gfk_log(xni0) * 1.33f);
                    pgen = wmax(0.0f, (roqi0 / denk - wmax(qck, 0.0f)) / dtcld);
                    pgen = wmin(wmin(pgen, satdt), supice);
                }
                if (qck > 0.0f) {
                    const float qimax = c[C_ROQIMAX] / denk;
                    paut = wmax(0.0f, (qck - qimax) / dtcld);
                }
            }

            // :796-826 mass conservation of the generation terms, the update
            // and its latent heating.
            {
                const float qciik = wmax(W_QMIN, qck);
                const float delqci = (((paut + pacr) - pgen) - pisd) * dtcld;
                if (delqci >= qciik) {
                    const float facqci = qciik / delqci;
                    paut = paut * facqci;
                    pacr = pacr * facqci;
                    pgen = pgen * facqci;
                    pisd = pisd * facqci;
                }
                const float qik = wmax(W_QMIN, qk);
                const float delq = ((pres + pgen) + pisd) * dtcld;
                if (delq >= qik) {
                    const float facq = qik / delq;
                    pres = pres * facq;
                    pgen = pgen * facq;
                    pisd = pisd * facq;
                }
                const float w2 = ((-pres) - pgen) - pisd;
                qk = qk + w2 * dtcld;
                qck = wmax(qck - (((paut + pacr) - pgen) - pisd) * dtcld, 0.0f);
                qrk = wmax(qrk + ((paut + pacr) + pres) * dtcld, 0.0f);
                if (tk < W_T0C)
                    tk = tk - ((W_XLS * w2) / cpmk) * dtcld;
                else
                    tk = tk - ((xlk * w2) / cpmk) * dtcld;
            }

            // :838-865 condensation/evaporation of cloud water against
            // liquid saturation at the updated temperature.
            {
                const float qsatn = wsm3_qsat(fp, tk, pk, false);
                const float work1c = st_conden(tk, qk, qsatn, xlk, cpmk);
                float pcon = wmin(wmax(work1c, 0.0f), wmax(qk, 0.0f)) / dtcld;
                if (qck > 0.0f && work1c < 0.0f && tk > W_T0C)
                    pcon = wmax(work1c, -qck) / dtcld;
                qk = qk - pcon * dtcld;
                qck = wmax(qck + pcon * dtcld, 0.0f);
                tk = tk + (((pcon * xlk) / cpmk) * dtcld);
            }
            // :870-875 padding for small values.
            if (qck <= W_QMIN) qck = 0.0f;
            if (qrk <= W_QCRMIN) qrk = 0.0f;

            t[k] = tk; q[k] = qk; qci[k] = qck; qrs[k] = qrk;
        }
    }

    rainnc[col] = rain_c; rainncv[col] = rainncv_c;
    snownc[col] = snow_c; snowncv[col] = snowncv_c;
    sr[col] = sr_c;

    // wsm3 :157-191: th = t/pii, then effectRad_wsm3 on th*pii with the
    // phase split read from the scheme's t.  Radii are formed in metres as
    // WRF stores them; gpuwm's radiation contract is microns, the one
    // multiply WRF's RRTMG applies when it reads them (re*1.e6).
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        const float th = t[k] / pii[id];
        theta[id] = th;
        qv[id] = q[k]; qc[id] = qci[k]; qr[id] = qrs[k];
        const float tt = th * pii[id];
        const float rho = denc[k * stride];
        float qc1, qi1, qs1;
        if (t[k] >= W_T0C) {
            qc1 = qci[k]; qi1 = 0.0f; qs1 = 0.0f;
        } else {
            qc1 = 0.0f; qi1 = qci[k]; qs1 = qrs[k];
        }
        float re_qc = 2.49e-6f, re_qi = 4.99e-6f, re_qs = 9.99e-6f;
        const float rqc = wmax(1.0e-12f, qc1 * rho);
        const float rqi = wmax(1.0e-12f, qi1 * rho);
        float temp = rho * wmax(qi1, W_QMIN);
        temp = sqrtf(sqrtf((temp * temp) * temp));
        const float ni = wmin(wmax(5.38e7f * temp, 1.0e3f), 1.0e6f);
        const float rni = wmax(1.0e-6f, ni * rho);
        const float rqs = wmax(1.0e-12f, qs1 * rho);
        if (rqc > 1.0e-12f) {
            const float lamdac = gfk_pow((c[C_PIDNC] * 3.0e8f) / rqc,
                                         __fdiv_rn(1.0f, 3.0f));
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

extern "C" __global__ void wsm3_column(
    float *theta, float *qv, float *qc, float *qr, const float *w,
    const float *den, const float *p, const float *pii, const float *dz,
    float *rainnc, float *rainncv, float *snownc, float *snowncv, float *sr,
    float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WSM3_KMAX) return;
    wsm3_column_impl(theta, qv, qc, qr, w, den, p, pii, dz,
                     rainnc, rainncv, snownc, snowncv, sr,
                     effc, effi, effs, coef, delt, nz, stride, col);
}
