// WRF v4.6.1 WSM7 single-moment seven-class microphysics (mp_physics = 24):
// WSM6's six classes plus a hail category qh.
//
// A port of WRF's own call chain for one column, phys/module_mp_wsm7.F:
//   wsm7            the wrapper the microphysics driver calls (t = th*pii
//                   in, th = t/pii out, effective radii and REFL_10CM from
//                   th*pii after the call)
//   wsm72D          the scheme body, minor loop by minor loop
//   slope_wsm7/_rain/_snow/_graup/_hail, nislfv_rain_plm, nislfv_rain_plm6
//   effectRad_wsm7
// and, as this file's second kernel (wsm7_refl10cm_column), refl10cm_wsm7
// with module_mp_radar.F's melting-particle routines.
//
// BITWISE CONTRACT.  Graded word for word against those routines compiled by
// gfortran -O0 (tools/wsm7_wrf461_column_oracle, tests/
// test_wsm7_wrf461_column_exact.py).  Each line keeps:
//   * Fortran's left-to-right association; a statement function (diffus,
//     viscos, xka, diffac, venfac, conden) is a value, as gfortran forms it;
//   * EXP, LOG and REAL**REAL as WOOF's own float32 routines gfk_exp,
//     gfk_log and gfk_pow (glibc_flt32.cuh); `x**2` (an integer power) is
//     x*x;
//   * Fortran MAX/MIN as gfortran's MAX_EXPR/MIN_EXPR (wmax/wmin below),
//     which keep a signed zero where fmaxf/fminf would not;
//   * a float division by a constant through __fdiv_rn, so NVRTC cannot turn
//     it into a multiply by a rounded reciprocal (A146);
//   * no FMA contraction and no flush to zero: the unit compiles through
//     gpuwm/core/wsm_family.py with --fmad=false --ftz=false, in the default
//     build as well as the strict one;
//   * wsm7init's float32 words (its rgmma Gamma product included), frozen in
//     gpuwm/data/wsm_family/wsm7-wrf461-words.json and passed in `coef`.
//
// TWO DECLARED DIVERGENCES FROM WRF (default-on; docs/wsm7-mp24.md):
//   D1  the PLM remap search (nislfv_rain_plm/plm6, find_kt) reaches the last
//       arrival interface and never leaves kt below kb.  WRF's search stops
//       one interface short, so a layer whose top lies above every arrival
//       interface gets kt = kb - 1 and its mass is dropped (queue 073: WSM6
//       and WDM6 share the same search).  Marked REMAP-SEARCH below.
//   D2  REFL_10CM includes hail.  refl10cm_wsm7 sums rain, snow and graupel
//       only, so a hail core is invisible in the diagnostic.  Hail adds its
//       dry-ice Rayleigh term with WSM7's own hail PSD (n0h, denh, m = am D^3,
//       mu = 0: the same exponent words radar_init forms for graupel).
//
// FUSION.  wsm72D sweeps the column once per process family.  From the
// melting sweep to the end of a minor step every family reads and writes
// level k only, so those run as one k loop in WRF's order at each level.
// Cloud-ice sedimentation is moved ahead of that loop: it reads qi, xni, den
// and delz only, none of which melting touches, and melting reads nothing ice
// sedimentation writes.  The loop-start saturation (qs, rh) is taken at the
// head of the fused loop, where t and q still hold their loop-start words
// (sedimentation moves neither).
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.

#ifndef WSM7_KMAX
#define WSM7_KMAX 64
#endif

// gfortran's MAX_EXPR/MIN_EXPR on two operands, argument order kept.
__device__ __forceinline__ float wmax(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float wmin(float a, float b) { return (b < a) ? b : a; }

// wsm7init's derived words, in the order of the init_words row of
// gpuwm/data/wsm_family/schemes.json (the words: wsm7-wrf461-words.json).
enum Wsm7Coef {
    C_PI, C_XLV1, C_QC0, C_QCK1, C_PIDNC,
    C_PIDN0R, C_PIDN0S, C_PIDN0G, C_PIDN0H,
    C_PVTR, C_PVTS, C_PVTG, C_PVTH,
    C_PACRR, C_PACRC, C_PACRG, C_PACRH,
    C_PRECR1, C_PRECR2, C_PRECS1, C_PRECS2, C_PRECG1, C_PRECG2,
    C_PRECH1, C_PRECH2, C_PRECH3,
    C_G6PBR, C_ROQIMAX,
    C_RSLOPERMAX, C_RSLOPESMAX, C_RSLOPEGMAX, C_RSLOPEHMAX,
    C_RSLOPERBMAX, C_RSLOPESBMAX, C_RSLOPEGBMAX, C_RSLOPEHBMAX,
    C_RSLOPER2MAX, C_RSLOPES2MAX, C_RSLOPEG2MAX, C_RSLOPEH2MAX,
    C_RSLOPER3MAX, C_RSLOPES3MAX, C_RSLOPEG3MAX, C_RSLOPEH3MAX,
    WSM7_NCOEF
};

// module_model_constants.F as the driver passes it, and module_mp_wsm7.F's
// own PARAMETERs.
#define W_CPD   1004.5f        // cp = 7.*r_d/2.
#define W_CPV   1846.4f        // cpv = 4.*r_v
#define W_RV    461.6f
#define W_G     9.81f
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
#define W_N0G   4.0e6f
#define W_N0H   4.0e4f
#define W_AVTR  841.9f
#define W_BVTR  0.8f
#define W_XNCR  3.0e8f
#define W_BVTS  0.41f
#define W_BVTG  0.8f
#define W_DENG  500.0f
#define W_BVTH  0.8f
#define W_DENH  912.0f
#define W_DICON 11.9f
#define W_DIMAX 500.0e-6f
#define W_PFRZ1 100.0f
#define W_PFRZ2 0.66f
#define W_QCRMIN 1.0e-9f
#define W_EACHS 1.0f
#define W_EACHG 0.5f
#define W_DENS  100.0f
#define W_QS0   6.0e-4f
#define W_N0S   2.0e6f
#define W_ALPHA 0.12f
#define W_N0SMAX_OVER_N0S 5.0e4f   // n0smax/n0s, folded by gfortran: 1.e11/2.e6

// ---- statement functions of wsm72D --------------------------------------
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

// The inline fpvs expansion of wsm72D (:607-616), then the ratio (:617-634).
struct Wsm7Fpvs { float ttp, xa, xb, xai, xbi; };
__device__ __forceinline__ Wsm7Fpvs wsm7_fpvs_consts() {
    Wsm7Fpvs f;
    f.ttp = W_T0C + 0.01f;
    const float dldt = W_CPV - W_CLIQ;
    f.xa = __fdiv_rn(-dldt, W_RV);
    f.xb = f.xa + __fdiv_rn(W_XLV0, W_RV * f.ttp);
    const float dldti = W_CPV - W_CICE;
    f.xai = __fdiv_rn(-dldti, W_RV);
    f.xbi = f.xai + __fdiv_rn(W_XLS, W_RV * f.ttp);
    return f;
}
__device__ __forceinline__ float wsm7_qsat(const Wsm7Fpvs &f, float t, float p,
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

// ---- slope_rain / _snow / _graup / _hail at one level ---------------------
struct Wsm7Slope { float r, rb, r2, r3, vt; };

__device__ __forceinline__ Wsm7Slope wsm7_slope_rain(const float *c, float q,
                                                     float den, float denfac) {
    Wsm7Slope s;
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

__device__ __forceinline__ Wsm7Slope wsm7_slope_snow(const float *c, float q,
                                                     float den, float denfac,
                                                     float t) {
    Wsm7Slope s;
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

__device__ __forceinline__ Wsm7Slope wsm7_slope_graup(const float *c, float q,
                                                      float den, float denfac) {
    Wsm7Slope s;
    if (q <= W_QCRMIN) {
        s.r = c[C_RSLOPEGMAX]; s.rb = c[C_RSLOPEGBMAX];
        s.r2 = c[C_RSLOPEG2MAX]; s.r3 = c[C_RSLOPEG3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf(c[C_PIDN0G] / (q * den)));
        s.rb = gfk_pow(s.r, W_BVTG);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C_PVTG] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

__device__ __forceinline__ Wsm7Slope wsm7_slope_hail(const float *c, float q,
                                                     float den, float denfac) {
    Wsm7Slope s;
    if (q <= W_QCRMIN) {
        s.r = c[C_RSLOPEHMAX]; s.rb = c[C_RSLOPEHBMAX];
        s.r2 = c[C_RSLOPEH2MAX]; s.r3 = c[C_RSLOPEH3MAX];
    } else {
        s.r = 1.0f / sqrtf(sqrtf(c[C_PIDN0H] / (q * den)));
        s.rb = gfk_pow(s.r, W_BVTH);
        s.r2 = s.r * s.r;
        s.r3 = s.r2 * s.r;
    }
    s.vt = (c[C_PVTH] * s.rb) * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// denfac = sqrt(den0 * (1/den)): VREC, *den0, VSQRT (:596-602), with
// frame/libmassv.F's portable vsrec/vssqrt loops.
__device__ __forceinline__ float wsm7_denfac(float den) {
    return sqrtf((1.0f / den) * W_DEN0);
}

// ---- nislfv_rain_plm (rq2 == nullptr) / nislfv_rain_plm6 -------------------
// mode 0: rain (id 1, slope_rain); mode 1: snow + graupel (plm6, slope_snow
// and slope_graup mass-weighted); mode 2: hail (id 2, slope_hail); mode 3:
// cloud ice (iter 0, the departure speed only).  rq/rq2 enter as den*q and
// leave as the remapped den*q.  `wwl` is the departure speed and is not
// written, as in WRF.
__device__ void wsm7_nislfv(int km, const float *c, const float *den,
                            const float *dz, const float *tk,
                            const float *wwl, float *rq, float *rq2,
                            float *precip1, float *precip2, float dt,
                            int iter, int mode, int stride)
{
    float ww[WSM7_KMAX], qn[WSM7_KMAX];
    float wi[WSM7_KMAX + 1], zi[WSM7_KMAX + 1], za[WSM7_KMAX + 1];
    float dza[WSM7_KMAX + 1], qa[WSM7_KMAX + 1], qa2[WSM7_KMAX + 1];
    float qmi[WSM7_KMAX + 1], qpi[WSM7_KMAX + 1];
    const bool two = (rq2 != nullptr);

    *precip1 = 0.0f;
    if (two) *precip2 = 0.0f;
    float allold = 0.0f;
    for (int k = 0; k < km; ++k) {
        if (two) allold = (allold + rq[k]) + rq2[k];
        else allold = allold + rq[k];
    }
    if (allold <= 0.0f) return;

    zi[0] = 0.0f;
    for (int k = 0; k < km; ++k) zi[k + 1] = zi[k] + dz[k * stride];
    for (int k = 0; k < km; ++k) ww[k] = wwl[k];

    int n = 1;
    for (;;) {
        // 3rd-order interface interpolation (:2334-2342).
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
        // Arrival terminal speed, then the mean of departure and arrival
        // (iter = 1: one pass, so no running average with `was`).
        for (int k = 0; k < km; ++k) {
            const float denk = den[k * stride];
            const float denfac = wsm7_denfac(denk);
            const float qr = qa[k] / denk;
            float wa;
            if (mode == 0) {
                wa = wsm7_slope_rain(c, qr, denk, denfac).vt;
            } else if (mode == 2) {
                wa = wsm7_slope_hail(c, qr, denk, denfac).vt;
            } else {
                const float qr2 = qa2[k] / denk;
                const float was = wsm7_slope_snow(c, qr, denk, denfac, tk[k]).vt;
                const float wa2 = wsm7_slope_graup(c, qr2, denk, denfac).vt;
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
            // REMAP-SEARCH BEGIN (declared divergence D1)
            // The target layer's top zi[k+1] is found among ALL arrival
            // interfaces za[0..km] and, past the last one, the column top
            // zi[km] (the empty arrival cell km, qa = 0), so kt >= kb always.
            {
                int ktf = km + 1;
                for (int kk = kt; kk <= km; ++kk)
                    if (zi[k + 1] <= za[kk]) { ktf = kk; break; }
                kt = (kb > ktf - 1) ? kb : ktf - 1;
            }
            // REMAP-SEARCH END
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
#ifdef WSM7_PLM_SURFACE
                // Correct mode, finding WB5 (physics.mode = "correct"; the WSM6
                // defect of kernels/wsm6.cu, shared by module_mp_wsm7.F's copy of
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
                precip = precip + q[k] * (0.0f - za[k]);
#endif
                break;
            }
            break;
        }
        float *out = (ist == 0) ? rq : rq2;
        for (int k = 0; k < km; ++k) out[k] = qn[k];
        if (ist == 0) *precip1 = precip; else *precip2 = precip;
    }
}

// Transport-only remap probe for the D1 budget check (tests/
// test_wsm7_remap_budget.py): one column, departure speeds given, iter 0.
extern "C" __global__ void wsm7_remap_probe(
    const float *dz, const float *den, const float *ww, float *rq,
    float *precip, float dt, int km)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || km > WSM7_KMAX) return;
    float c[WSM7_NCOEF];
    for (int n = 0; n < WSM7_NCOEF; ++n) c[n] = 0.0f;
    float tk[WSM7_KMAX], w[WSM7_KMAX], r[WSM7_KMAX];
    for (int k = 0; k < km; ++k) { tk[k] = W_T0C; w[k] = ww[k]; r[k] = rq[k]; }
    float p1, p2;
    wsm7_nislfv(km, c, den, dz, tk, w, r, nullptr, &p1, &p2, dt, 0, 3, 1);
    for (int k = 0; k < km; ++k) rq[k] = r[k];
    *precip = p1;
}

__device__ void wsm7_column_impl(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, float *qh, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *hailnc, float *hailncv, float *sr,
    float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int stride, int col)
{
    float c[WSM7_NCOEF];
    for (int n = 0; n < WSM7_NCOEF; ++n) c[n] = coef[n];
    const float pi = c[C_PI];
    const Wsm7Fpvs fp = wsm7_fpvs_consts();
    // hvap = xlv0 (:608); the pgwet/phwet saturation at t0c (:1292).
    const float hvap = W_XLV0;

    float t[WSM7_KMAX], q[WSM7_KMAX], qcl[WSM7_KMAX], qil[WSM7_KMAX];
    float qrl[WSM7_KMAX], qsl[WSM7_KMAX], qgl[WSM7_KMAX], qhl[WSM7_KMAX];
    float cpm[WSM7_KMAX], xl[WSM7_KMAX], xni[WSM7_KMAX];
    float v1[WSM7_KMAX], v2[WSM7_KMAX], v3[WSM7_KMAX];
    float rq1[WSM7_KMAX], rq2[WSM7_KMAX], rq3[WSM7_KMAX], rq4[WSM7_KMAX];

    const float *denc = den + col;
    const float *dzc = dz + col;
    const float *pc = p + col;

    // module_mp_wsm7.F:183-193, then wsm72D :531-551.
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        t[k] = theta[id] * pii[id];
        q[k] = qv[id];
        qcl[k] = wmax(qc[id], 0.0f);
        qrl[k] = wmax(qr[id], 0.0f);
        qil[k] = wmax(qi[id], 0.0f);
        qsl[k] = wmax(qs[id], 0.0f);
        qgl[k] = wmax(qg[id], 0.0f);
        qhl[k] = wmax(qh[id], 0.0f);
        const float qq = wmax(q[k], W_QMIN);
        cpm[k] = W_CPD * (1.0f - qq) + qq * W_CPV;
        xl[k] = W_XLV0 - c[C_XLV1] * (t[k] - W_T0C);
    }

    float rainncv_c = 0.0f, snowncv_c = 0.0f, graupelncv_c = 0.0f;
    float hailncv_c = 0.0f, sr_c = 0.0f;
    float tstepsnow = 0.0f, tstepgraup = 0.0f, tstephail = 0.0f;
    float rain_c = rainnc[col], snow_c = snownc[col];
    float graupel_c = graupelnc[col], hail_c = hailnc[col];

    // :578-580
    float loopsf = roundf(__fdiv_rn(delt, W_DTCLDCR));
    int loops = (int)loopsf;
    if (loops < 1) loops = 1;
    float dtcld = delt / (float)loops;
    if (delt <= W_DTCLDCR) dtcld = delt;

    for (int loop = 0; loop < loops; ++loop) {
        // :703-709 ice number, :714-746 departure speeds.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float denfac = wsm7_denfac(denk);
            float temp = denk * wmax(qil[k], W_QMIN);
            temp = sqrtf(sqrtf((temp * temp) * temp));
            xni[k] = wmin(wmax(5.38e7f * temp, 1.0e3f), 1.0e6f);
            float workr = wsm7_slope_rain(c, qrl[k], denk, denfac).vt;
            float workh = wsm7_slope_hail(c, qhl[k], denk, denfac).vt;
            const float vs = wsm7_slope_snow(c, qsl[k], denk, denfac, t[k]).vt;
            const float vg = wsm7_slope_graup(c, qgl[k], denk, denfac).vt;
            const float qsum = wmax(qsl[k] + qgl[k], 1.0e-15f);
            v2[k] = (qsum > 1.0e-15f) ? (vs * qsl[k] + vg * qgl[k]) / qsum : 0.0f;
            rq1[k] = denk * qrl[k];
            rq2[k] = denk * qsl[k];
            rq3[k] = denk * qgl[k];
            rq4[k] = denk * qhl[k];
            if (qrl[k] <= 0.0f) workr = 0.0f;
            if (qhl[k] <= 0.0f) workh = 0.0f;
            v1[k] = workr;
            v3[k] = workh;
        }
        float delqrs1, delqrs2, delqrs3, delqrs4, unused;
        wsm7_nislfv(nz, c, denc, dzc, t, v1, rq1, nullptr, &delqrs1, &unused,
                    dtcld, 1, 0, stride);
        wsm7_nislfv(nz, c, denc, dzc, t, v2, rq2, rq3, &delqrs2, &delqrs3,
                    dtcld, 1, 1, stride);
        wsm7_nislfv(nz, c, denc, dzc, t, v3, rq4, nullptr, &delqrs4, &unused,
                    dtcld, 1, 2, stride);
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            qrl[k] = wmax(rq1[k] / denk, 0.0f);
            qsl[k] = wmax(rq2[k] / denk, 0.0f);
            qgl[k] = wmax(rq3[k] / denk, 0.0f);
            qhl[k] = wmax(rq4[k] / denk, 0.0f);
        }
        const float dz0 = dzc[0];
        const float fall1 = (delqrs1 / dz0) / dtcld;
        const float fall2 = (delqrs2 / dz0) / dtcld;
        const float fall3 = (delqrs3 / dz0) / dtcld;
        const float fall4 = (delqrs4 / dz0) / dtcld;

        // :845-876 cloud-ice fall speed and sedimentation (iter = 0), moved
        // ahead of the melting sweep (see FUSION above).
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
        wsm7_nislfv(nz, c, denc, dzc, t, v1, rq1, nullptr, &delqi, &unused,
                    dtcld, 0, 3, stride);
        for (int k = 0; k < nz; ++k)
            qil[k] = wmax(rq1[k] / denc[k * stride], 0.0f);
        const float fallc = (delqi / dz0) / dtcld;

        // :880-923 surface accumulation.
        {
            const float fallsum = (((fall1 + fall2) + fall3) + fall4) + fallc;
            const float fallsum_qsi = fall2 + fallc;
            const float fallsum_qg = fall3;
            const float fallsum_qh = fall4;
            if (fallsum > 0.0f) {
                rainncv_c = (__fdiv_rn(fallsum * dz0, W_DENR) * dtcld) * 1000.0f
                          + rainncv_c;
                rain_c = (__fdiv_rn(fallsum * dz0, W_DENR) * dtcld) * 1000.0f
                       + rain_c;
            }
            if (fallsum_qsi > 0.0f) {
                tstepsnow = (__fdiv_rn(fallsum_qsi * dz0, W_DENR) * dtcld) * 1000.0f
                          + tstepsnow;
                snowncv_c = (__fdiv_rn(fallsum_qsi * dz0, W_DENR) * dtcld) * 1000.0f
                          + snowncv_c;
                snow_c = (__fdiv_rn(fallsum_qsi * dz0, W_DENR) * dtcld) * 1000.0f
                       + snow_c;
            }
            if (fallsum_qg > 0.0f) {
                tstepgraup = (__fdiv_rn(fallsum_qg * dz0, W_DENR) * dtcld) * 1000.0f
                           + tstepgraup;
                graupelncv_c = (__fdiv_rn(fallsum_qg * dz0, W_DENR) * dtcld) * 1000.0f
                             + graupelncv_c;
                graupel_c = (__fdiv_rn(fallsum_qg * dz0, W_DENR) * dtcld) * 1000.0f
                          + graupel_c;
            }
            if (fallsum_qh > 0.0f) {
                tstephail = (__fdiv_rn(fallsum_qh * dz0, W_DENR) * dtcld) * 1000.0f
                          + tstephail;
                hailncv_c = (__fdiv_rn(fallsum_qh * dz0, W_DENR) * dtcld) * 1000.0f
                          + hailncv_c;
                hail_c = (__fdiv_rn(fallsum_qh * dz0, W_DENR) * dtcld) * 1000.0f
                       + hail_c;
            }
            if (fallsum > 0.0f)
                sr_c = ((tstepsnow + tstepgraup) + tstephail) / (rainncv_c + 1.0e-12f);
        }

        // Everything below reads and writes level k only.
        for (int k = 0; k < nz; ++k) {
            const float denk = denc[k * stride];
            const float pk = pc[k * stride];
            const float denfac = wsm7_denfac(denk);
            float tk = t[k], qk = q[k];
            float qck = qcl[k], qik = qil[k], qrk = qrl[k];
            float qsk = qsl[k], qgk = qgl[k], qhk = qhl[k];
            const float cpmk = cpm[k], xlk = xl[k];

            // :617-636 saturation at the start of the minor step.
            const float qsat1 = wsm7_qsat(fp, tk, pk, false);
            const float rh1 = wmax(qk / qsat1, W_QMIN);
            const float qsat2 = wsm7_qsat(fp, tk, pk, true);
            const float rh2 = wmax(qk / qsat2, W_QMIN);

            // :784-841 slopes after sedimentation, then melting.
            Wsm7Slope sr_ = wsm7_slope_rain(c, qrk, denk, denfac);
            Wsm7Slope ss_ = wsm7_slope_snow(c, qsk, denk, denfac, tk);
            Wsm7Slope sg_ = wsm7_slope_graup(c, qgk, denk, denfac);
            Wsm7Slope sh_ = wsm7_slope_hail(c, qhk, denk, denfac);
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
                if (qhk > 0.0f) {
                    const float coeres = sh_.r2 * sqrtf(sh_.r * sh_.rb);
                    float phmlt = __fdiv_rn(st_xka(tk, denk), xlf) * (W_T0C - tk)
                                * (c[C_PRECH1] * sh_.r2 + c[C_PRECH2] * work2 * coeres)
                                / denk;
                    phmlt = wmin(wmax(phmlt * dtcld, -qhk), 0.0f);
                    qhk = qhk + phmlt;
                    qrk = qrk - phmlt;
                    tk = tk + (xlf / cpmk) * phmlt;
                }
            }

            // :928-975 instantaneous melting/freezing (slopes from :784).
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

            // :988-1002 slopes for the process rates, conduction/diffusion
            // and ventilation terms.
            sr_ = wsm7_slope_rain(c, qrk, denk, denfac);
            ss_ = wsm7_slope_snow(c, qsk, denk, denfac, tk);
            sg_ = wsm7_slope_graup(c, qgk, denk, denfac);
            sh_ = wsm7_slope_hail(c, qhk, denk, denfac);
            const float work1_1 = st_diffac(xlk, pk, tk, denk, qsat1);
            const float work1_2 = st_diffac(W_XLS, pk, tk, denk, qsat2);
            const float work2 = st_venfac(pk, tk, denk);

            float prevp = 0.f, psdep = 0.f, pgdep = 0.f, phdep = 0.f;
            float praut = 0.f, psaut = 0.f, pgaut = 0.f, phaut = 0.f;
            float pracw = 0.f, praci = 0.f, piacr = 0.f, psaci = 0.f;
            float psacw = 0.f, pracs = 0.f, pracg = 0.f, psacr = 0.f;
            float pgacw = 0.f, paacw = 0.f, pgaci = 0.f, pgacr = 0.f;
            float pgacs = 0.f, phacw = 0.f, phaci = 0.f, phacr = 0.f;
            float phacs = 0.f, phacg = 0.f, pigen = 0.f, pidep = 0.f;
            float pseml = 0.f, pgeml = 0.f, pheml = 0.f;
            float psevp = 0.f, pgevp = 0.f, phevp = 0.f;
            // wsm72D initialises these to zero and nothing assigns them a
            // rate; they stay in every sum, where adding +0 is WRF's word.
            const float pvapg = 0.f, pvaph = 0.f, primh = 0.f;

            // :1012-1048 warm rain.
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

            // :1063-1501 cold rain.
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
            const float vt2h = (c[C_PVTH] * sh_.rb) * denfac;
            const float qsum = wmax(qsk + qgk, 1.0e-15f);
            const float vt2ave = (qsum > 1.0e-15f) ? (vt2s * qsk + vt2g * qgk) / qsum
                                                   : 0.0f;
            if (supcol > 0.0f && qik > W_QMIN) {
                if (qrk > W_QCRMIN) {
                    // praci [HL A15], piacr [HL A19]
                    const float acrfac = (2.0f * sr_.r3 + (2.0f * diameter) * sr_.r2)
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
                    // psaci [HDC 10]
                    const float acrfac = (2.0f * ss_.r3 + (2.0f * diameter) * ss_.r2)
                                       + (diameter * diameter) * ss_.r;
                    psaci = (((((((pi * qik) * eacrs) * W_N0S) * n0sfac)
                               * fabsf(vt2ave - vt2i)) * acrfac)) / 4.0f;
                    psaci = wmin(psaci, qik / dtcld);
                }
                if (qgk > W_QCRMIN) {
                    // pgaci [HL A17]
                    const float egi = gfk_exp(0.07f * (-supcol));
                    const float acrfac = (2.0f * sg_.r3 + (2.0f * diameter) * sg_.r2)
                                       + (diameter * diameter) * sg_.r;
                    pgaci = (((((pi * egi) * qik) * W_N0G) * fabsf(vt2ave - vt2i))
                             * acrfac) / 4.0f;
                    pgaci = wmin(pgaci, qik / dtcld);
                }
                if (qhk > W_QCRMIN) {
                    // phaci [BHT]: hail collects cloud ice
                    const float ehi = gfk_exp(0.07f * (-supcol));
                    const float acrfac = (2.0f * sh_.r3 + (2.0f * diameter) * sh_.r2)
                                       + (diameter * diameter) * sh_.r;
                    phaci = (((((pi * ehi) * qik) * W_N0H) * fabsf(vt2h - vt2i))
                             * acrfac) / 4.0f;
                    phaci = wmin(phaci, qik / dtcld);
                }
            }
            if (qsk > W_QCRMIN && qck > W_QMIN) {
                // psacw [HL A7]
                const float m = wmin(wmax(0.0f, qsk / qck), 1.0f);
                psacw = wmin((((((c[C_PACRC] * n0sfac) * ss_.r3) * ss_.rb) * (m * m))
                              * qck) * denfac, qck / dtcld);
            }
            if (qgk > W_QCRMIN && qck > W_QMIN) {
                // pgacw [HL A6]
                const float m = wmin(wmax(0.0f, qgk / qck), 1.0f);
                pgacw = wmin(((((c[C_PACRG] * sg_.r3) * sg_.rb) * (m * m)) * qck)
                             * denfac, qck / dtcld);
            }
            if (qsum > 1.0e-15f)
                paacw = (qsk * psacw + qgk * pgacw) / qsum;
            if (qhk > W_QCRMIN && qck > W_QMIN) {
                // phacw [BHT A08]
                const float m = wmin(wmax(0.0f, qhk / qck), 1.0f);
                phacw = wmin(((((c[C_PACRH] * sh_.r3) * sh_.rb) * (m * m)) * qck)
                             * denfac, qck / dtcld);
            }
            if (qsk > W_QCRMIN && qrk > W_QCRMIN) {
                if (supcol > 0.0f) {
                    // pracs [HL A11]
                    const float acrfac = (((5.0f * ss_.r3) * ss_.r3) * sr_.r
                                          + ((2.0f * ss_.r3) * ss_.r2) * sr_.r2)
                                       + ((0.5f * ss_.r2) * ss_.r2) * sr_.r3;
                    pracs = (((((((pi * pi) * W_N0R) * W_N0S) * n0sfac)
                               * fabsf(vt2r - vt2ave)) * (W_DENS / denk)) * acrfac);
                    const float m = wmin(wmax(0.0f, qrk / qsk), 1.0f);
                    pracs = pracs * (m * m);
                    pracs = wmin(pracs, qsk / dtcld);
                }
                // psacr [HL A10]
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
                if (supcol > 0.0f) {
                    // pracg [BHT A17]: rain collects graupel
                    const float acrfac = (((5.0f * sg_.r3) * sg_.r3) * sr_.r
                                          + ((2.0f * sg_.r3) * sg_.r2) * sr_.r2)
                                       + ((0.5f * sg_.r2) * sg_.r2) * sr_.r3;
                    pracg = ((((((pi * pi) * W_N0R) * W_N0G) * fabsf(vt2r - vt2ave))
                              * (W_DENG / denk)) * acrfac);
                    const float m = wmin(wmax(0.0f, qrk / qgk), 1.0f);
                    pracg = pracg * (m * m);
                    pracg = wmin(pracg, qgk / dtcld);
                }
                // pgacr [HL A12]
                const float acrfac = (((5.0f * sr_.r3) * sr_.r3) * sg_.r
                                      + ((2.0f * sr_.r3) * sr_.r2) * sg_.r2)
                                   + ((0.5f * sr_.r2) * sr_.r2) * sg_.r3;
                pgacr = (((((pi * pi) * W_N0R) * W_N0G) * fabsf(vt2ave - vt2r))
                         * (W_DENR / denk)) * acrfac;
                const float m = wmin(wmax(0.0f, qgk / qrk), 1.0f);
                pgacr = pgacr * (m * m);
                pgacr = wmin(pgacr, qrk / dtcld);
            }
            if (qgk > W_QCRMIN && qsk > W_QCRMIN) pgacs = 0.0f;
            if (qhk > W_QCRMIN && qrk > W_QCRMIN) {
                // phacr [BHT A13]: hail collects rain
                const float acrfac = (((5.0f * sr_.r3) * sr_.r3) * sh_.r
                                      + ((2.0f * sr_.r3) * sr_.r2) * sh_.r2)
                                   + ((0.5f * sr_.r2) * sr_.r2) * sh_.r3;
                phacr = (((((pi * pi) * W_N0R) * W_N0H) * fabsf(vt2h - vt2r))
                         * (W_DENR / denk)) * acrfac;
                const float m = wmin(wmax(0.0f, qhk / qrk), 1.0f);
                phacr = phacr * (m * m);
                phacr = wmin(phacr, qrk / dtcld);
            }
            if (qhk > W_QCRMIN && qsk > W_QCRMIN) {
                // phacs [BHT A14]: hail collects snow
                const float acrfac = (((5.0f * ss_.r3) * ss_.r3) * sh_.r
                                      + ((2.0f * ss_.r3) * ss_.r2) * sh_.r2)
                                   + ((0.5f * ss_.r2) * ss_.r2) * sh_.r3;
                phacs = (((((((pi * pi) * W_EACHS) * W_N0S) * n0sfac) * W_N0H)
                          * fabsf(vt2h - vt2ave)) * (W_DENS / denk)) * acrfac;
                phacs = wmin(phacs, qsk / dtcld);
            }
            if (qhk > W_QCRMIN && qgk > W_QCRMIN) {
                // phacg [BHT A15]: hail collects graupel
                const float acrfac = (((5.0f * sg_.r3) * sg_.r3) * sh_.r
                                      + ((2.0f * sg_.r3) * sg_.r2) * sh_.r2)
                                   + ((0.5f * sg_.r2) * sg_.r2) * sh_.r3;
                phacg = ((((((pi * pi) * W_EACHG) * W_N0G) * W_N0H)
                          * fabsf(vt2h - vt2ave)) * (W_DENG / denk)) * acrfac;
                phacg = wmin(phacg, qgk / dtcld);
            }
            // :1289-1334 wet growth.  pgwet (:1301-1312) is formed by WRF and
            // read by nothing, so it is not formed here.  phwet's only effect
            // is to zero phaci, phacs and phacg when hail's riming falls short
            // of wet growth; with qh at or below qcrmin all three are already
            // +0, so the test (and its saturation and ventilation words) is
            // formed only where it can change a word.
            if (qhk > W_QCRMIN) {
                const float rs0a = (W_PSAT * gfk_exp(gfk_log(__fdiv_rn(fp.ttp, W_T0C)) * fp.xa))
                                 * gfk_exp(fp.xb * (1.0f - __fdiv_rn(fp.ttp, W_T0C)));
                float rs0 = wmin(rs0a, 0.99f * pk);
                rs0 = (__fdiv_rn(287.0f, W_RV) * rs0) / (pk - rs0);
                rs0 = wmax(rs0, W_QMIN);
                const float ghw1 = ((denk * hvap) * st_diffus(tk, pk)) * (rs0 - qk)
                                 - st_xka(tk, denk) * (-supcol);
                const float ghw2 = denk * (W_XLF0 + W_CLIQ * (-supcol));
                const float ghw3 = st_venfac(pk, tk, denk)
                                 * sqrtf(sqrtf(__fdiv_rn(W_G * denk, W_DEN0)));
                const float ghw4 = denk * ((W_XLF0 - W_CLIQ * supcol) + W_CICE * supcol);
                float phaci_w = 0.0f;
                if (phaci > 0.0f) {
                    const float ehi = gfk_exp(0.07f * (-supcol));
                    phaci_w = phaci / ehi;
                }
                float phwet = (ghw1 / ghw2)
                            * ((c[C_PRECH1] * sh_.r2
                                + (c[C_PRECH3] * ghw3) * gfk_pow(sh_.r, 2.75f))
                               + ghw4 * (phaci_w + phacs));
                phwet = wmax(phwet, 0.0f);
                if (phacw + phacr < 0.95f * phwet) {
                    phaci = 0.0f;
                    phacs = 0.0f;
                    phacg = 0.0f;
                }
            }
            if (supcol <= 0.0f) {
                const float xlf = W_XLF0;
                if (qsk > 0.0f)
                    pseml = wmin(wmax(__fdiv_rn((W_CLIQ * supcol) * (paacw + psacr), xlf),
                                      -qsk / dtcld), 0.0f);
                if (qgk > 0.0f)
                    pgeml = wmin(wmax(__fdiv_rn((W_CLIQ * supcol) * (paacw + pgacr), xlf),
                                      -qgk / dtcld), 0.0f);
                if (qhk > 0.0f)
                    pheml = wmin(wmax(__fdiv_rn((W_CLIQ * supcol) * (phacw + phacr), xlf),
                                      -qhk / dtcld), 0.0f);
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
                if (qhk > 0.0f && ifsat != 1) {
                    // phdep [BHT A19]
                    const float coeres = sh_.r2 * sqrtf(sh_.r * sh_.rb);
                    phdep = ((rh2 - 1.0f)
                             * (c[C_PRECH1] * sh_.r2 + (c[C_PRECH2] * work2) * coeres))
                          / work1_2;
                    const float supice = (((satdt - prevp) - pidep) - psdep) - pgdep;
                    if (phdep < 0.0f) {
                        phdep = wmax(phdep, -qhk / dtcld);
                        phdep = wmax(wmax(phdep, satdt / 2.0f), supice);
                    } else {
                        phdep = wmin(wmin(phdep, satdt / 2.0f), supice);
                    }
                    if (fabsf((((prevp + pidep) + psdep) + pgdep) + phdep) >= fabsf(satdt))
                        ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {
                    // pigen: WRF's supice here leaves phdep out (:1436).
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
            // phaut [BHT A18]: graupel to hail, formed at every temperature
            // (:1463, outside the supcol > 0 block), read only below t0c.
            if (qgk > 0.0f) {
                const float alpha2 = 1.0e-3f * gfk_exp(0.09f * (-supcol));
                phaut = wmin(wmax(0.0f, alpha2 * (qgk - W_QS0)), qgk / dtcld);
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
                if (qhk > 0.0f && rh1 < 1.0f) {
                    // phevp [BHT A20]
                    const float coeres = sh_.r2 * sqrtf(sh_.r * sh_.rb);
                    phevp = ((rh1 - 1.0f)
                             * (c[C_PRECH1] * sh_.r2 + (c[C_PRECH2] * work2) * coeres))
                          / work1_1;
                    phevp = wmin(wmax(phevp, -qhk / dtcld), 0.0f);
                }
            }

            // :1506-1763 mass conservation of the generation terms, then the
            // update and its latent heating, each sum as WRF writes it.
            {
                float delta2 = 0.0f, delta3 = 0.0f;
                if (qrk < 1.0e-4f && qsk < 1.0e-4f) delta2 = 1.0f;
                if (qrk < 1.0e-4f) delta3 = 1.0f;
                float value, source, factor;
                if (tk <= W_T0C) {
                    // cloud water
                    value = wmax(W_QMIN, qck);
                    source = ((((praut + pracw) + paacw) + paacw) + phacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        paacw = paacw * factor;
                        phacw = phacw * factor;
                    }
                    // cloud ice
                    value = wmax(W_QMIN, qik);
                    source = ((((((psaut - pigen) - pidep) + praci) + psaci) + pgaci)
                              + phaci) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        psaut = psaut * factor;
                        pigen = pigen * factor;
                        pidep = pidep * factor;
                        praci = praci * factor;
                        psaci = psaci * factor;
                        pgaci = pgaci * factor;
                        phaci = phaci * factor;
                    }
                    // rain
                    value = wmax(W_QMIN, qrk);
                    source = ((((((-praut - prevp) - pracw) + piacr) + psacr) + pgacr)
                              + phacr) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        prevp = prevp * factor;
                        pracw = pracw * factor;
                        piacr = piacr * factor;
                        psacr = psacr * factor;
                        pgacr = pgacr * factor;
                        phacr = phacr * factor;
                    }
                    // snow
                    value = wmax(W_QMIN, qsk);
                    source = -((psdep + psaut + paacw + pvapg + pvaph
                                + psaci - pgaut - pracs * (1.0f - delta2)
                                + piacr * delta3 + praci * delta3
                                + psacr * delta2
                                - pgacs - phacs) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        psdep = psdep * factor;
                        psaut = psaut * factor;
                        pgaut = pgaut * factor;
                        paacw = paacw * factor;
                        psaci = psaci * factor;
                        piacr = piacr * factor;
                        praci = praci * factor;
                        psacr = psacr * factor;
                        pracs = pracs * factor;
                        pgacs = pgacs * factor;
                        phacs = phacs * factor;
                    }
                    // graupel
                    value = wmax(W_QMIN, qgk);
                    source = -((pgdep + pgaut + pgaci + paacw + pgacs
                                + piacr * (1.0f - delta3) + praci * (1.0f - delta3)
                                + psacr * (1.0f - delta2) + pgacr * delta2
                                + pracs * (1.0f - delta2) - pracg * (1.0f - delta2)
                                - phaut - pvapg - phacg + primh) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        pgdep = pgdep * factor;
                        pgaut = pgaut * factor;
                        phaut = phaut * factor;
                        piacr = piacr * factor;
                        praci = praci * factor;
                        pracs = pracs * factor;
                        pracg = pracg * factor;
                        psacr = psacr * factor;
                        paacw = paacw * factor;
                        pgaci = pgaci * factor;
                        pgacr = pgacr * factor;
                        pgacs = pgacs * factor;
                        phacg = phacg * factor;
                    }
                    // hail
                    value = wmax(W_QMIN, qhk);
                    source = -((phdep + phaut
                                + pgacr * (1.0f - delta2) + pracg * (1.0f - delta2)
                                + phacw + phacr + phaci + phacs
                                + phacg - pvaph - primh) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        phdep = phdep * factor;
                        phaut = phaut * factor;
                        pracg = pracg * factor;
                        pgacr = pgacr * factor;
                        phacw = phacw * factor;
                        phaci = phaci * factor;
                        phacr = phacr * factor;
                        phacs = phacs * factor;
                        phacg = phacg * factor;
                    }
                    const float work2u = -(prevp + psdep + pgdep + phdep + pigen + pidep);
                    qk = qk + work2u * dtcld;
                    qck = wmax(qck - (praut + pracw + paacw + paacw + phacw) * dtcld,
                               0.0f);
                    qrk = wmax(qrk + (praut + pracw + prevp - piacr - pgacr - psacr
                                      - phacr) * dtcld, 0.0f);
                    qik = wmax(qik - (psaut + praci + psaci + pgaci + phaci - pigen
                                      - pidep) * dtcld, 0.0f);
                    qsk = wmax(qsk + (psdep + psaut + paacw + pvapg + pvaph - pgaut
                                      + psaci - pgacs - phacs
                                      + piacr * delta3 + praci * delta3
                                      + psacr * delta2
                                      - pracs * (1.0f - delta2)) * dtcld, 0.0f);
                    qgk = wmax(qgk + (pgdep + pgaut
                                      + piacr * (1.0f - delta3) + praci * (1.0f - delta3)
                                      + psacr * (1.0f - delta2) + pgacr * delta2
                                      + pgaci + paacw + pgacs + primh
                                      + pracs * (1.0f - delta2) - pracg * (1.0f - delta2)
                                      - phaut - pvapg - phacg) * dtcld, 0.0f);
                    qhk = wmax(qhk + (phdep + phaut
                                      + pgacr * (1.0f - delta2) + pracg * (1.0f - delta2)
                                      + phacw + phacr + phaci + phacs
                                      + phacg - pvaph - primh) * dtcld, 0.0f);
                    const float xlf = W_XLS - xlk;
                    const float xlwork2 = -(W_XLS * (psdep + pgdep + phdep + pidep + pigen))
                                        - xlk * prevp
                                        - xlf * (piacr + paacw + paacw + phacw + phacr
                                                 + pgacr + psacr);
                    tk = tk - (xlwork2 / cpmk) * dtcld;
                } else {
                    // cloud water
                    value = wmax(W_QMIN, qck);
                    source = ((((praut + pracw) + paacw) + paacw) + phacw) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        pracw = pracw * factor;
                        paacw = paacw * factor;
                        phacw = phacw * factor;
                    }
                    // rain
                    value = wmax(W_QMIN, qrk);
                    source = (pseml + pgeml + pheml - pracw - paacw - paacw - phacw
                              - prevp - praut) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        praut = praut * factor;
                        prevp = prevp * factor;
                        pracw = pracw * factor;
                        paacw = paacw * factor;
                        phacw = phacw * factor;
                        pseml = pseml * factor;
                        pgeml = pgeml * factor;
                        pheml = pheml * factor;
                    }
                    // snow
                    value = wmax(W_QCRMIN, qsk);
                    source = (pgacs + phacs - pseml - psevp) * dtcld;
                    if (source > value) {
                        factor = value / source;
                        pgacs = pgacs * factor;
                        phacs = phacs * factor;
                        psevp = psevp * factor;
                        pseml = pseml * factor;
                    }
                    // graupel
                    value = wmax(W_QCRMIN, qgk);
                    source = -((pgacs + pgevp + pgeml - phacg) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        pgacs = pgacs * factor;
                        pgevp = pgevp * factor;
                        pgeml = pgeml * factor;
                        phacg = phacg * factor;
                    }
                    // hail
                    value = wmax(W_QCRMIN, qhk);
                    source = -((phacs + phacg + phevp + pheml) * dtcld);
                    if (source > value) {
                        factor = value / source;
                        phacs = phacs * factor;
                        phacg = phacg * factor;
                        phevp = phevp * factor;
                        pheml = pheml * factor;
                    }
                    const float work2u = -(prevp + psevp + pgevp + phevp);
                    qk = qk + work2u * dtcld;
                    qck = wmax(qck - (praut + pracw + paacw + paacw + phacw) * dtcld,
                               0.0f);
                    qrk = wmax(qrk + (praut + pracw + prevp + paacw + paacw + phacw
                                      - pseml - pgeml - pheml) * dtcld, 0.0f);
                    qsk = wmax(qsk + (psevp + pseml - pgacs - phacs) * dtcld, 0.0f);
                    qgk = wmax(qgk + (pgacs + pgevp + pgeml - phacg) * dtcld, 0.0f);
                    qhk = wmax(qhk + (phacs + phacg + phevp + pheml) * dtcld, 0.0f);
                    const float xlf = W_XLS - xlk;
                    const float xlwork2 = -(xlk * (prevp + psevp + pgevp + phevp))
                                        - xlf * (pseml + pgeml + pheml);
                    tk = tk - (xlwork2 / cpmk) * dtcld;
                }
            }

            // :1769-1813 condensation/evaporation of cloud water.
            {
                const float qsatn = wsm7_qsat(fp, tk, pk, false);
                const float work1 = st_conden(tk, qk, qsatn, xlk, cpmk);
                float pcond = wmin(wmax(work1 / dtcld, 0.0f), wmax(qk, 0.0f) / dtcld);
                if (qck > 0.0f && work1 < 0.0f)
                    pcond = wmax(work1, -qck) / dtcld;
                qk = qk - pcond * dtcld;
                qck = wmax(qck + pcond * dtcld, 0.0f);
                tk = tk + (((pcond * xlk) / cpmk) * dtcld);
            }
            // :1817-1822 padding for small values.
            if (qck <= W_QMIN) qck = 0.0f;
            if (qik <= W_QMIN) qik = 0.0f;

            t[k] = tk; q[k] = qk;
            qcl[k] = qck; qil[k] = qik; qrl[k] = qrk;
            qsl[k] = qsk; qgl[k] = qgk; qhl[k] = qhk;
        }
    }

    rainnc[col] = rain_c; rainncv[col] = rainncv_c;
    snownc[col] = snow_c; snowncv[col] = snowncv_c;
    graupelnc[col] = graupel_c; graupelncv[col] = graupelncv_c;
    hailnc[col] = hail_c; hailncv[col] = hailncv_c;
    sr[col] = sr_c;

    // module_mp_wsm7.F:214-224 th = t/pii, then :246-269 effectRad_wsm7 on
    // th*pii.  Radii in metres as WRF stores them, then gpuwm's micron
    // contract: the one float32 multiply WRF's RRTMG applies (re*1.e6).
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        const float th = t[k] / pii[id];
        theta[id] = th;
        qv[id] = q[k]; qc[id] = qcl[k]; qi[id] = qil[k];
        qr[id] = qrl[k]; qs[id] = qsl[k]; qg[id] = qgl[k]; qh[id] = qhl[k];
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

extern "C" __global__ void wsm7_column(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, float *qh, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *hailnc, float *hailncv, float *sr,
    float *effc, float *effi, float *effs,
    const float *__restrict__ coef, float delt, int nz, int ny, int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = ny * nx;
    if (col >= stride || nz > WSM7_KMAX) return;
    wsm7_column_impl(theta, qv, qc, qi, qr, qs, qg, qh, den, p, pii, dz,
                     rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
                     hailnc, hailncv, sr, effc, effi, effs, coef, delt, nz,
                     stride, col);
}

// ===========================================================================
// REFL_10CM for WSM7: module_mp_wsm7.F refl10cm_wsm7 (:2788-2960) with
// module_mp_radar.F's rayleigh_soak_wetgraupel / get_m_mix_nested /
// get_m_mix / m_complex_maxwellgarnett for the melting particles, the
// wrapper's floor MAX(-35., dBZ) (:240), and declared divergence D2 (hail).
//
// PRECISION is WRF's mix, line by line: temperature, density, the species
// densities, the snow intercept and the dry-particle prefactors are float32
// (gfk_exp for the snow intercept's EXP); slopes, moments and the melting
// integral are float64, rounded to float32 where WRF assigns a REAL.
// Binary64 EXP, LOG and REAL**REAL are WOOF's own words (g64_exp, g64_log,
// g64_pow in glibc_flt64.cuh).  LOG10, ATAN2 and HYPOT -- so the complex
// LOG/SQRT/ABS of the melting term -- are CUDA's binary64 functions; WRF
// rounds every such value to float32 before it is stored, so a stored word
// can move only where a binary64 result sits within one binary64 ULP of a
// float32 rounding boundary.  gfortran's complex arithmetic: products
// (ac-bd, ad+bc), quotients in Smith's range-reduced form
// (-fcx-fortran-rules), integer powers above 2 through libgcc's __powidf2
// square-and-multiply (what gfortran -O0 calls).
//
// The radar constants are radar_init's own words after wsm7init
// (gpuwm/data/wsm_family/wsm7-wrf461-words.json), not recomputations.
// ===========================================================================

enum Wsm7RadarF32 {
    RF_XAM_R, RF_XAM_S, RF_XAM_G,
    RF_XCRE1, RF_XCRE4, RF_XCRG3, RF_XCRG4,
    RF_XCSE1, RF_XCSE4, RF_XCSG3, RF_XCSG4,
    RF_XCGE1, RF_XCGE4, RF_XCGG3, RF_XCGG4,
    RF_XBM_S, RF_XBM_G, RF_XMU_S, RF_XMU_G,
    RF_XOCMS, RF_XOBMS, RF_XOCMG, RF_XOBMG,
    WSM7_NRADAR_F32
};
#define WSM7_NRBINS 50
enum Wsm7RadarF64 {
    RD_K_W, RD_PI5, RD_LAMDA4, RD_M_W_0_RE, RD_M_W_0_IM, RD_M_I_0_RE, RD_M_I_0_IM,
    RD_XXDS, RD_XDTS = RD_XXDS + WSM7_NRBINS, RD_XXDG = RD_XDTS + WSM7_NRBINS,
    RD_XDTG = RD_XXDG + WSM7_NRBINS, RD_SIMPSON = RD_XDTG + WSM7_NRBINS,
    WSM7_NRADAR_F64 = RD_SIMPSON + WSM7_NRBINS + 1
};

struct W7Cplx { double re, im; };
__device__ __forceinline__ W7Cplx w7c(double re, double im) {
    W7Cplx z; z.re = re; z.im = im; return z;
}
__device__ __forceinline__ W7Cplx w7c_add(W7Cplx a, W7Cplx b) {
    return w7c(a.re + b.re, a.im + b.im);
}
__device__ __forceinline__ W7Cplx w7c_sub(W7Cplx a, W7Cplx b) {
    return w7c(a.re - b.re, a.im - b.im);
}
__device__ __forceinline__ W7Cplx w7c_mul(W7Cplx a, W7Cplx b) {
    return w7c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
// Smith's division as GCC lowers it for Fortran (range reduced, no NaN
// recovery).
__device__ __forceinline__ W7Cplx w7c_div(W7Cplx a, W7Cplx b) {
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
    return w7c(tr / div, ti / div);
}
// real * complex: the real operand promoted to (x, 0), then the product.
__device__ __forceinline__ W7Cplx w7c_rmul(double x, W7Cplx b) {
    return w7c_mul(w7c(x, 0.0), b);
}
__device__ __forceinline__ double w7c_abs(W7Cplx a) { return hypot(a.re, a.im); }
__device__ __forceinline__ W7Cplx w7c_log(W7Cplx a) {
    return w7c(g64_log::log(hypot(a.re, a.im)), atan2(a.im, a.re));
}
// Principal square root, the half-angle form: one part from the larger of
// |z| + |Re z|, the other from 2 Re Im = Im z.
__device__ __forceinline__ W7Cplx w7c_sqrt(W7Cplx a) {
    if (a.re == 0.0 && a.im == 0.0) return w7c(0.0, a.im);
    const double d = hypot(a.re, a.im);
    double r, s;
    if (a.re > 0.0) {
        r = sqrt(0.5 * (d + a.re));
        s = 0.5 * (a.im / r);
    } else {
        s = sqrt(0.5 * (d - a.re));
        r = fabs(0.5 * (a.im / s));
    }
    return w7c(r, copysign(s, a.im));
}
// x**n for n > 2 as gfortran -O0 evaluates it: square and multiply.
__device__ __forceinline__ double w7_powi(double x, int m) {
    unsigned int n = (unsigned int)m;
    double y = (n % 2) ? x : 1.0;
    while (n >>= 1) {
        x = x * x;
        if (n % 2) y = y * x;
    }
    return y;
}
__device__ __forceinline__ double w7dmax(double a, double b) { return (b > a) ? b : a; }
__device__ __forceinline__ double w7dmin(double a, double b) { return (b < a) ? b : a; }

// m_complex_maxwellgarnett with inclusion = 'spheroidal'.
__device__ W7Cplx wsm7_maxwellgarnett(double vol1, double vol2, double vol3,
                                      W7Cplx m1, W7Cplx m2, W7Cplx m3, int *error)
{
    *error = 0;
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1.0e-6) {
        *error = 1;
        return w7c(-999.99, -999.99);
    }
    const W7Cplx m1t = w7c_mul(m1, m1);
    const W7Cplx m2t = w7c_mul(m2, m2);
    const W7Cplx m3t = w7c_mul(m3, m3);
    const W7Cplx one = w7c(1.0, 0.0);
    const W7Cplx beta2 = w7c_mul(
        w7c_div(w7c_rmul(2.0, m1t), w7c_sub(m2t, m1t)),
        w7c_sub(w7c_mul(w7c_div(m2t, w7c_sub(m2t, m1t)), w7c_log(w7c_div(m2t, m1t))), one));
    const W7Cplx beta3 = w7c_mul(
        w7c_div(w7c_rmul(2.0, m1t), w7c_sub(m3t, m1t)),
        w7c_sub(w7c_mul(w7c_div(m3t, w7c_sub(m3t, m1t)), w7c_log(w7c_div(m3t, m1t))), one));
    const double v1 = 1.0 - vol2 - vol3;
    const W7Cplx num = w7c_add(w7c_add(w7c_rmul(v1, m1t),
                                       w7c_mul(w7c_rmul(vol2, beta2), m2t)),
                               w7c_mul(w7c_rmul(vol3, beta3), m3t));
    const W7Cplx den = w7c_add(w7c_add(w7c(v1, 0.0), w7c_rmul(vol2, beta2)),
                               w7c_rmul(vol3, beta3));
    return w7c_sqrt(w7c_div(num, den));
}

// rayleigh_soak_wetgraupel with the strings WSM7's radar_init sets: mixing
// rule maxwellgarnett, host air, matrix water, inclusion spheroidal, host
// matrix icewater, host inclusion spheroidal.
__device__ double wsm7_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, W7Cplx m_w, W7Cplx m_i, double pi5, double lamda4)
{
    const double pix = 3.1415926535897932384626434;
    const W7Cplx m_air = w7c(1.0, 0.0);
    const double fm = w7dmax(w7dmin(fmelt, 1.0), 0.0);
    double mra = w7dmax(w7dmin(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    const double x_w = x_g * fm;
    const double d_g = a_geo * g64_pow::pow(x_g, b_geo);
    if (!(d_g >= 1.0e-12)) return 0.0;
    double vg = (pix / 6.0) * w7_powi(d_g, 3);
    const double rhog = w7dmax(w7dmin(x_g / vg, 900.0), 10.0);
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
    const double d_large = g64_pow::pow((6.0 / pix) * volg,
                                        (double)__fdiv_rn(1.0f, 3.0f));
    const double volice = (x_g - x_w) / (volg * 900.0);
    const double volwater = x_w / (1000.0 * volg);
    const double volair = 1.0 - volice - volwater;
    // get_m_mix_nested: host air, matrix water, host matrix icewater.
    int err1, err2;
    const double vol1 = volice / w7dmax(volice + volwater, 1.0e-10);
    const double vol2 = 1.0 - vol1;
    const W7Cplx mtmp = wsm7_maxwellgarnett(vol2, 0.0, vol1, m_w, m_air, m_i, &err1);
    const W7Cplx m_core = wsm7_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_air,
                                              w7c_rmul(2.0, m_air), &err2);
    if (err1 + err2 != 0) return 0.0;
    const W7Cplx mc2 = w7c_mul(m_core, m_core);
    const double a = w7c_abs(w7c_div(w7c_sub(mc2, w7c(1.0, 0.0)),
                                     w7c_add(mc2, w7c(2.0, 0.0))));
    return (((a * a) * pi5) * w7_powi(d_large, 6)) / lamda4;
}

extern "C" __global__ void wsm7_refl10cm_column(
    const float *qv, const float *qr, const float *qs, const float *qg,
    const float *qh, const float *t, const float *p,
    const float *__restrict__ coef, const float *__restrict__ rf,
    const double *__restrict__ rd, float *refl, int nz, int ny, int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = ny * nx;
    if (col >= stride || nz > WSM7_KMAX) return;
    float temp[WSM7_KMAX], rs[WSM7_KMAX], rg[WSM7_KMAX];
    double ilamr[WSM7_KMAX], ilams[WSM7_KMAX], ilamg[WSM7_KMAX];
    double ilamh[WSM7_KMAX], n0_s[WSM7_KMAX];
    float ze_snow[WSM7_KMAX], ze_graupel[WSM7_KMAX];
    unsigned char lq[WSM7_KMAX];   // bit 0 rain, 1 snow, 2 graupel, 3 hail
    const float pi = coef[C_PI];
    const double n0_r = (double)W_N0R;
    const double n0_g = (double)W_N0G;
    const double n0_h = (double)W_N0H;
    // D2: hail's m(D) prefactor, PI*denh/6 as wsm7init forms graupel's.
    const float xam_h = __fdiv_rn(pi * W_DENH, 6.0f);

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
            const double lamr = g64_pow::pow(
                (double)(rf[RF_XAM_R] * rf[RF_XCRG3]) * n0_r / (double)rr,
                (double)(1.0f / rf[RF_XCRE1]));
            ilamr[k] = 1.0 / lamr;
            lq[k] |= 1;
        }
        if (qs[id] > 1.0e-9f) {
            rs[k] = qs[id] * rho;
            n0_s[k] = (double)wmin(1.0e11f, W_N0S * gfk_exp(-(W_ALPHA * temp_c)));
            const double lams = g64_pow::pow(
                (double)(rf[RF_XAM_S] * rf[RF_XCSG3]) * n0_s[k] / (double)rs[k],
                (double)(1.0f / rf[RF_XCSE1]));
            ilams[k] = 1.0 / lams;
            lq[k] |= 2;
        } else {
            rs[k] = 1.0e-12f;
        }
        if (qg[id] > 1.0e-9f) {
            rg[k] = qg[id] * rho;
            const double lamg = g64_pow::pow(
                (double)(rf[RF_XAM_G] * rf[RF_XCGG3]) * n0_g / (double)rg[k],
                (double)(1.0f / rf[RF_XCGE1]));
            ilamg[k] = 1.0 / lamg;
            lq[k] |= 4;
        } else {
            rg[k] = 1.0e-12f;
        }
        if (qh[id] > 1.0e-9f) {
            // D2: hail, graupel's exponent words (bm = 3, mu = 0).
            const float rh = qh[id] * rho;
            const double lamh = g64_pow::pow(
                (double)(xam_h * rf[RF_XCGG3]) * n0_h / (double)rh,
                (double)(1.0f / rf[RF_XCGE1]));
            ilamh[k] = 1.0 / lamh;
            lq[k] |= 8;
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

    // Rayleigh terms of dry snow, graupel and (D2) hail: the prefactor
    // float32 as WRF writes it, promoted where it meets the float64 intercept.
    const float six_over_pi = 6.0f / pi;
    const float pref_s = __fdiv_rn(0.176f, 0.93f) * six_over_pi * six_over_pi
                       * __fdiv_rn(rf[RF_XAM_S], 900.0f) * __fdiv_rn(rf[RF_XAM_S], 900.0f);
    const float pref_g = __fdiv_rn(0.176f, 0.93f) * six_over_pi * six_over_pi
                       * __fdiv_rn(rf[RF_XAM_G], 900.0f) * __fdiv_rn(rf[RF_XAM_G], 900.0f);
    const float pref_h = __fdiv_rn(0.176f, 0.93f) * six_over_pi * six_over_pi
                       * __fdiv_rn(xam_h, 900.0f) * __fdiv_rn(xam_h, 900.0f);
    for (int k = 0; k < nz; ++k) {
        ze_snow[k] = 1.0e-22f;
        ze_graupel[k] = 1.0e-22f;
        if (lq[k] & 2)
            ze_snow[k] = (float)((double)pref_s * n0_s[k] * (double)rf[RF_XCSG4]
                                 * g64_pow::pow(ilams[k], (double)rf[RF_XCSE4]));
        if (lq[k] & 4)
            ze_graupel[k] = (float)((double)pref_g * n0_g * (double)rf[RF_XCGG4]
                                    * g64_pow::pow(ilamg[k], (double)rf[RF_XCGE4]));
    }

    if (melti && k_0 >= 1) {
        const W7Cplx m_w = w7c(rd[RD_M_W_0_RE], rd[RD_M_W_0_IM]);
        const W7Cplx m_i = w7c(rd[RD_M_I_0_RE], rd[RD_M_I_0_IM]);
        const double pi5 = rd[RD_PI5], lamda4 = rd[RD_LAMDA4], k_w = rd[RD_K_W];
        for (int k = k_0 - 1; k >= 0; --k) {
            if ((lq[k] & 2) && (lq[k_0] & 2)) {
                const double fmelt_s = w7dmax(0.005, w7dmin(1.0 - (double)(rs[k] / rs[k_0]),
                                                            0.99));
                double eta = 0.0;
                const double lams = 1.0 / ilams[k];
                for (int n = 0; n < WSM7_NRBINS; ++n) {
                    const double xx = rd[RD_XXDS + n];
                    const double x = (double)rf[RF_XAM_S]
                                   * g64_pow::pow(xx, (double)rf[RF_XBM_S]);
                    const double cback = wsm7_rayleigh_soak_wetgraupel(
                        x, (double)rf[RF_XOCMS], (double)rf[RF_XOBMS], fmelt_s, 0.9,
                        m_w, m_i, pi5, lamda4);
                    const double f_d = n0_s[k] * g64_pow::pow(xx, (double)rf[RF_XMU_S])
                                     * g64_exp::exp(-lams * xx);
                    eta = eta + f_d * cback * rd[RD_SIMPSON + n] * rd[RD_XDTS + n];
                }
                ze_snow[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
            if ((lq[k] & 4) && (lq[k_0] & 4)) {
                const double fmelt_g = w7dmax(0.005, w7dmin(1.0 - (double)(rg[k] / rg[k_0]),
                                                            0.99));
                double eta = 0.0;
                const double lamg = 1.0 / ilamg[k];
                for (int n = 0; n < WSM7_NRBINS; ++n) {
                    const double xx = rd[RD_XXDG + n];
                    const double x = (double)rf[RF_XAM_G]
                                   * g64_pow::pow(xx, (double)rf[RF_XBM_G]);
                    const double cback = wsm7_rayleigh_soak_wetgraupel(
                        x, (double)rf[RF_XOCMG], (double)rf[RF_XOBMG], fmelt_g, 0.9,
                        m_w, m_i, pi5, lamda4);
                    const double f_d = n0_g * g64_pow::pow(xx, (double)rf[RF_XMU_G])
                                     * g64_exp::exp(-lamg * xx);
                    eta = eta + f_d * cback * rd[RD_SIMPSON + n] * rd[RD_XDTG + n];
                }
                ze_graupel[k] = (float)(lamda4 / (pi5 * k_w) * eta);
            }
        }
    }

    for (int k = nz - 1; k >= 0; --k) {
        // ze_hail is 0 where there is no hail, so (positive sum) + 0 leaves
        // WRF's word untouched in every hail-free cell.
        float ze_rain = 1.0e-22f, ze_hail = 0.0f;
        if (lq[k] & 1)
            ze_rain = (float)(n0_r * (double)rf[RF_XCRG4]
                              * g64_pow::pow(ilamr[k], (double)rf[RF_XCRE4]));
        if (lq[k] & 8)
            ze_hail = (float)((double)pref_h * n0_h * (double)rf[RF_XCGG4]
                              * g64_pow::pow(ilamh[k], (double)rf[RF_XCGE4]));
        const float zsum = ((ze_rain + ze_snow[k]) + ze_graupel[k]) + ze_hail;
        const float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));
        refl[col + k * stride] = wmax(-35.0f, dbz);
    }
}
