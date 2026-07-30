// WRF v4.6.1 WSM6 single-moment six-class microphysics.
//
// Process-rate equations, donor limiting, phase changes, thermodynamic
// feedback, saturation adjustment, fall-speed laws, precipitation category
// accounting, and effective radii are transcribed from:
//   phys/physics_mmm/mp_wsm6.F90
//   phys/physics_mmm/mp_wsm6_effectRad.F90
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.  Sedimentation is WRF's forward semi-Lagrangian,
// monotone piecewise-linear nislfv_rain_plm/plm6 remap.  The implementation
// must still pass end-to-end GPU/WRF trajectory tests before any bitwise or
// whole-model parity claim is made.

#ifdef WSM6_CPU_MIRROR
#include <cmath>
#define __device__
#define __forceinline__ inline
#define __global__
#endif

#ifndef WSM6_KMAX
#define WSM6_KMAX 64
#endif

__device__ __forceinline__ float fmax0(float x) { return fmaxf(x, 0.0f); }
__device__ __forceinline__ float clamp01(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

struct Rimed {
    float n0g, deng, avtg, bvtg, lamdagmax;
    float pidn0g, pvtg, pacrg, precg1, precg2;
    float rslopegmax, rslopegbmax, rslopeg2max, rslopeg3max;
};

__device__ __forceinline__ Rimed rimed_constants(int hail_opt) {
    Rimed r;
    if (hail_opt == 1) {
        r.n0g = 4.0e4f; r.deng = 700.0f; r.avtg = 285.0f;
        r.bvtg = 0.8f; r.lamdagmax = 2.0e4f;
        r.pidn0g = 8.79645943e7f;
        r.pvtg = 8.46323123e2f;
        r.pacrg = 4.19991431e7f;
        r.precg1 = 1.96035382e5f;
        r.precg2 = 2.40250519e6f;
        r.rslopegmax = 5.0e-5f;
        r.rslopegbmax = 3.62389832e-4f;
        r.rslopeg2max = 2.5e-9f;
        r.rslopeg3max = 1.25e-13f;
    } else {
        r.n0g = 4.0e6f; r.deng = 500.0f; r.avtg = 330.0f;
        r.bvtg = 0.8f; r.lamdagmax = 6.0e4f;
        r.pidn0g = 6.28318531e9f;
        r.pvtg = 9.79953090e2f;
        r.pacrg = 4.86305868e9f;
        r.precg1 = 1.96035382e7f;
        r.precg2 = 2.58522814e8f;
        r.rslopegmax = 1.66666667e-5f;
        r.rslopegbmax = 1.50480075e-4f;
        r.rslopeg2max = 2.77777778e-10f;
        r.rslopeg3max = 4.62962963e-15f;
    }
    return r;
}

struct Slope {
    float r, rb, r2, r3, vt;
};

__device__ __forceinline__ Slope rain_slope(float q, float den,
                                             float denfac) {
    Slope s;
    if (q <= 1.0e-9f) {
        s.r = 1.25e-5f; s.rb = 1.19544062e-4f;
        s.r2 = 1.5625e-10f; s.r3 = 1.953125e-15f;
    } else {
        s.r = 1.0f / sqrtf(sqrtf(2.51327412e10f / (q * den)));
        s.rb = powf(s.r, 0.8f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = (q > 0.0f) ? 2.50006820e3f * s.rb * denfac : 0.0f;
    return s;
}

__device__ __forceinline__ Slope snow_slope(float q, float den,
                                             float denfac, float temp,
                                             float *n0sfac_out) {
    Slope s;
    float n0sfac = fmaxf(fminf(expf(0.12f * (273.15f - temp)), 5.0e4f),
                         1.0f);
    *n0sfac_out = n0sfac;
    if (q <= 1.0e-9f) {
        s.r = 1.0e-5f; s.rb = 8.91250938e-3f;
        s.r2 = 1.0e-10f; s.r3 = 1.0e-15f;
    } else {
        s.r = 1.0f / sqrtf(sqrtf(6.28318531e8f * n0sfac / (q * den)));
        s.rb = powf(s.r, 0.41f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = (q > 0.0f) ? 2.00517852e1f * s.rb * denfac : 0.0f;
    return s;
}

__device__ __forceinline__ Slope graupel_slope(float q, float den,
                                                float denfac,
                                                const Rimed &g) {
    Slope s;
    if (q <= 1.0e-9f) {
        s.r = g.rslopegmax; s.rb = g.rslopegbmax;
        s.r2 = g.rslopeg2max; s.r3 = g.rslopeg3max;
    } else {
        s.r = 1.0f / sqrtf(sqrtf(g.pidn0g / (q * den)));
        s.rb = powf(s.r, g.bvtg); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = (q > 0.0f) ? g.pvtg * s.rb * denfac : 0.0f;
    return s;
}

__device__ __forceinline__ float diffus(float t, float p) {
    return 8.794e-5f * powf(t, 1.81f) / p;
}
__device__ __forceinline__ float viscos(float t, float den) {
    return 1.496e-6f * (t * sqrtf(t)) / (t + 120.0f) / den;
}
__device__ __forceinline__ float xka(float t, float den) {
    return 1.414e3f * viscos(t, den) * den;
}
__device__ __forceinline__ float diffac(float latent, float p, float t,
                                        float den, float qsat) {
    return den * latent * latent / (xka(t, den) * 461.6f * t * t)
         + 1.0f / (qsat * diffus(t, p));
}
__device__ __forceinline__ float venfac(float p, float t, float den) {
    return powf(viscos(t, den) / diffus(t, p), 0.3333333f)
         / sqrtf(viscos(t, den)) * sqrtf(sqrtf(1.28f / den));
}
__device__ __forceinline__ float fpvs(float t, bool ice) {
    const float ttp = 273.16f;
    const float xa = -(1846.4f - 4190.0f) / 461.6f;
    const float xb = xa + 2.5e6f / (461.6f * ttp);
    const float xai = -(1846.4f - 2106.0f) / 461.6f;
    const float xbi = xai + 2.85e6f / (461.6f * ttp);
    float tr = ttp / t;
    float aa = (ice && t < ttp) ? xai : xa;
    float bb = (ice && t < ttp) ? xbi : xb;
    return 610.78f * powf(tr, aa) * expf(bb * (1.0f - tr));
}
__device__ __forceinline__ float qsat(float t, float p, bool ice) {
    float es = fminf(fpvs(t, ice), 0.99f * p);
    // WRF passes ep2=rd/rv from the driver.  Keep that ratio instead of the
    // commonly rounded 0.622: the rounding is large enough to move a warm
    // saturation-adjustment oracle by O(1e-3 K) in one call.
    return fmaxf(1.0e-15f, (287.0f / 461.6f) * es / (p - es));
}

// WRF v4.6.1 nislfv_rain_plm/plm6.  Arrays with nz+1 entries represent
// interfaces.  q/partner enter as mixing ratios and are remapped as density
// concentrations, exactly as the Fortran caller does.
__device__ __forceinline__ void plm_arrival(
    const float *qq, const float *qq2, const float *ww, const float *dz,
    const float *den, int nz, float dt, float *zi, float *za, float *dza,
    float *qa, float *qa2)
{
    float wi[WSM6_KMAX + 1];
    zi[0] = 0.0f;
    for (int k = 0; k < nz; ++k) zi[k + 1] = zi[k] + dz[k];

    // WRF computes a second-order interface interpolation first and then
    // overwrites it with the active third-order interpolation.
    wi[0] = ww[0];
    wi[nz] = ww[nz - 1];
    for (int k = 1; k < nz; ++k)
        wi[k] = (ww[k] * dz[k - 1] + ww[k - 1] * dz[k])
              / (dz[k - 1] + dz[k]);
    wi[0] = ww[0];
    if (nz > 1) wi[1] = 0.5f * (ww[1] + ww[0]);
    for (int k = 2; k < nz - 1; ++k)
        wi[k] = (9.0f / 16.0f) * (ww[k] + ww[k - 1])
              - (1.0f / 16.0f) * (ww[k + 1] + ww[k - 2]);
    if (nz > 1) wi[nz - 1] = 0.5f * (ww[nz - 1] + ww[nz - 2]);
    wi[nz] = ww[nz - 1];

    for (int k = 1; k < nz; ++k)
        if (ww[k] == 0.0f) wi[k] = ww[k - 1];
    for (int k = nz - 1; k >= 0; --k) {
        float decfl = (wi[k + 1] - wi[k]) * dt / dz[k];
        if (decfl > 0.05f) wi[k] = wi[k + 1] - 0.05f * dz[k] / dt;
    }
    for (int k = 0; k <= nz; ++k) za[k] = zi[k] - wi[k] * dt;
    for (int k = 0; k < nz; ++k) dza[k] = za[k + 1] - za[k];
    dza[nz] = zi[nz] - za[nz];
    for (int k = 0; k < nz; ++k) {
        qa[k] = qq[k] * dz[k] / dza[k];
        if (qq2) qa2[k] = qq2[k] * dz[k] / dza[k];
    }
    qa[nz] = 0.0f;
    if (qq2) qa2[nz] = 0.0f;
}

__device__ __forceinline__ float plm_regular_remap(
    const float *qa, const float *zi, const float *za, const float *dza,
    int nz, float *qn)
{
    float qmi[WSM6_KMAX + 1], qpi[WSM6_KMAX + 1];
    for (int k = 1; k < nz; ++k) {
        float dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
        float dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
        if (dip * dim <= 0.0f) {
            qmi[k] = qa[k]; qpi[k] = qa[k];
        } else {
            qpi[k] = qa[k] + 0.5f * (dip + dim) * dza[k];
            qmi[k] = 2.0f * qa[k] - qpi[k];
            if (qpi[k] < 0.0f || qmi[k] < 0.0f) {
                qpi[k] = qa[k]; qmi[k] = qa[k];
            }
        }
    }
    qpi[0] = qa[0]; qmi[0] = qa[0];
    qpi[nz] = qa[nz]; qmi[nz] = qa[nz];
    for (int k = 0; k < nz; ++k) qn[k] = 0.0f;

    int kb = 0, kt = 0;
    for (int k = 0; k < nz; ++k) {
        kb = (kb > 0) ? kb - 1 : 0;
        kt = (kt > 0) ? kt - 1 : 0;
        if (zi[k] >= za[nz]) break;
        for (int kk = kb; kk < nz; ++kk) {
            if (zi[k] <= za[kk + 1]) { kb = kk; break; }
        }
        for (int kk = kt; kk < nz; ++kk) {
            if (zi[k + 1] <= za[kk]) { kt = kk; break; }
        }
        kt = (kt > 0) ? kt - 1 : 0;
        if (kt == kb) {
            float tl = (zi[k] - za[kb]) / dza[kb];
            float th = (zi[k + 1] - za[kb]) / dza[kb];
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qqh = qqd * th * th + qmi[kb] * th;
            float qql = qqd * tl * tl + qmi[kb] * tl;
            qn[k] = (qqh - qql) / (th - tl);
        } else if (kt > kb) {
            float tl = (zi[k] - za[kb]) / dza[kb];
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qql = qqd * tl * tl + qmi[kb] * tl;
            float zsum = (1.0f - tl) * dza[kb];
            float qsum = (qa[kb] - qql) * dza[kb];
            for (int m = kb + 1; m < kt; ++m) {
                zsum += dza[m]; qsum += qa[m] * dza[m];
            }
            float th = (zi[k + 1] - za[kt]) / dza[kt];
            float dqh = 0.5f * (qpi[kt] - qmi[kt]) * th * th
                      + qmi[kt] * th;
            zsum += th * dza[kt]; qsum += dqh * dza[kt];
            qn[k] = qsum / zsum;
        }
    }

    float precip = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (za[k] < 0.0f && za[k + 1] < 0.0f) {
            precip += qa[k] * dza[k];
        } else if (za[k] < 0.0f && za[k + 1] >= 0.0f) {
            precip += qa[k] * (0.0f - za[k]);
            break;
        } else {
            break;
        }
    }
    return precip;
}

__device__ float sediment_column(float *q, const float *den,
                                 const float *dz, const float *temp,
                                 int nz, float dt, int species,
                                 const Rimed &g, float *partner,
                                 float *partner_precip_out) {
    float qq[WSM6_KMAX], qq2[WSM6_KMAX], wd[WSM6_KMAX], ww[WSM6_KMAX];
    float wa[WSM6_KMAX], qn[WSM6_KMAX], qn2[WSM6_KMAX];
    float zi[WSM6_KMAX + 1], za[WSM6_KMAX + 1];
    float dza[WSM6_KMAX + 1], qa[WSM6_KMAX + 1], qa2[WSM6_KMAX + 1];
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        float q1 = fmax0(q[k]);
        float q2 = partner ? fmax0(partner[k]) : 0.0f;
        qq[k] = q1 * den[k]; qq2[k] = q2 * den[k];
        float df = sqrtf(1.28f / den[k]);
        if (species == 0) {
            ww[k] = rain_slope(q1, den[k], df).vt;
        } else if (species == 1) {
            float n0s;
            float vs = snow_slope(q1, den[k], df, temp[k], &n0s).vt;
            float vg = graupel_slope(q2, den[k], df, g).vt;
            float total = fmaxf(q1 + q2, 1.0e-15f);
            ww[k] = total > 1.0e-15f ? (vs * q1 + vg * q2) / total : 0.0f;
        } else {
            float xniarg = den[k] * fmaxf(q1, 1.0e-15f);
            float xni = fminf(fmaxf(5.38e7f * powf(xniarg, 0.75f), 1.0e3f),
                              1.0e6f);
            float diameter = fmaxf(fminf(11.9f * sqrtf(den[k] * q1 / xni),
                                         500.0e-6f), 1.0e-25f);
            ww[k] = q1 > 0.0f ? 1.49e4f * powf(diameter, 1.31f) : 0.0f;
        }
        wd[k] = ww[k];
        allold += qq[k] + (partner ? qq2[k] : 0.0f);
    }
    if (partner_precip_out) *partner_precip_out = 0.0f;
    if (allold <= 0.0f) return 0.0f;

    plm_arrival(qq, partner ? qq2 : nullptr, ww, dz, den, nz, dt,
                zi, za, dza, qa, qa2);

    // Rain and the paired snow/graupel categories use WRF iter=1.  Cloud ice
    // uses iter=0, retaining its departure velocity.
    if (species != 2) {
        for (int k = 0; k < nz; ++k) {
            float q1 = qa[k] / den[k];
            float df = sqrtf(1.28f / den[k]);
            if (species == 0) {
                wa[k] = rain_slope(q1, den[k], df).vt;
            } else {
                float q2 = qa2[k] / den[k], n0s;
                float vs = snow_slope(q1, den[k], df, temp[k], &n0s).vt;
                float vg = graupel_slope(q2, den[k], df, g).vt;
                float total = fmaxf(q1 + q2, 1.0e-15f);
                wa[k] = total > 1.0e-15f
                      ? (vs * q1 + vg * q2) / total : 0.0f;
            }
            ww[k] = 0.5f * (wd[k] + wa[k]);
        }
        plm_arrival(qq, partner ? qq2 : nullptr, ww, dz, den, nz, dt,
                    zi, za, dza, qa, qa2);
    }

    float precip = plm_regular_remap(qa, zi, za, dza, nz, qn);
    float precip2 = partner
                  ? plm_regular_remap(qa2, zi, za, dza, nz, qn2) : 0.0f;
    for (int k = 0; k < nz; ++k) {
        q[k] = fmax0(qn[k] / den[k]);
        if (partner) partner[k] = fmax0(qn2[k] / den[k]);
    }
    if (partner_precip_out) *partner_precip_out = precip2;
    return precip;
}

__device__ void wsm6_column_impl(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *sr, float *effc, float *effi, float *effs,
    float delt, int hail_opt, int nz, int stride, int col)
{
    const Rimed gc = rimed_constants(hail_opt);
    float temp[WSM6_KMAX], rho[WSM6_KMAX], delz[WSM6_KMAX];
    float qvv[WSM6_KMAX], qcc[WSM6_KMAX], qii[WSM6_KMAX];
    float qrr[WSM6_KMAX], qss[WSM6_KMAX], qgg[WSM6_KMAX];
    float cpm[WSM6_KMAX], xl[WSM6_KMAX];
    float qsw[WSM6_KMAX], qsi[WSM6_KMAX], rhw[WSM6_KMAX], rhi[WSM6_KMAX];

    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        temp[k] = theta[id] * pii[id];
        rho[k] = den[id]; delz[k] = dz[id];
        qvv[k] = qv[id]; qcc[k] = fmax0(qc[id]); qii[k] = fmax0(qi[id]);
        qrr[k] = fmax0(qr[id]); qss[k] = fmax0(qs[id]); qgg[k] = fmax0(qg[id]);
        float vap = fmaxf(qvv[k], 1.0e-15f);
        cpm[k] = 1004.5f * (1.0f - vap) + vap * 1846.4f;
        xl[k] = 2.5e6f - 2343.6f * (temp[k] - 273.15f);
    }

    rainncv[col] = 0.0f; snowncv[col] = 0.0f; graupelncv[col] = 0.0f;
    sr[col] = 0.0f;
    int loops = (int)floorf(delt / 120.0f + 0.5f);
    if (loops < 1) loops = 1;
    float dtcld = delt / (float)loops;

    for (int loop = 0; loop < loops; ++loop) {
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            qsw[k] = qsat(temp[k], p[id], false);
            qsi[k] = qsat(temp[k], p[id], true);
            rhw[k] = fmaxf(qvv[k] / qsw[k], 1.0e-15f);
            rhi[k] = fmaxf(qvv[k] / qsi[k], 1.0e-15f);
        }

        float pr = sediment_column(qrr, rho, delz, temp, nz, dtcld, 0,
                                    gc, nullptr, nullptr);
        float pg = 0.0f;
        float ps = sediment_column(qss, rho, delz, temp, nz, dtcld, 1,
                                    gc, qgg, &pg);
        // Ice crystals use their own WRF fall-speed law.
        float piice = sediment_column(qii, rho, delz, temp, nz, dtcld, 2,
                                      gc, nullptr, nullptr);

        float total_precip = pr + ps + pg + piice;
        rainncv[col] += total_precip;
        snowncv[col] += ps + piice;
        graupelncv[col] += pg;
        rainnc[col] += total_precip;
        snownc[col] += ps + piice;
        graupelnc[col] += pg;
        if (total_precip > 0.0f)
            sr[col] = (snowncv[col] + graupelncv[col])
                    / (rainncv[col] + 1.0e-12f);

        // Sedimentation-adjacent melting/freezing, then WSM6 process rates.
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            float denfac = sqrtf(1.28f / rho[k]);
            float n0sfac;
            Slope rr = rain_slope(qrr[k], rho[k], denfac);
            Slope ss = snow_slope(qss[k], rho[k], denfac, temp[k], &n0sfac);
            Slope gg = graupel_slope(qgg[k], rho[k], denfac, gc);
            float supcol = 273.15f - temp[k];
            float xlf = 2.85e6f - xl[k];

            if (temp[k] > 273.15f) {
                float vf = venfac(p[id], temp[k], rho[k]);
                if (qss[k] > 0.0f) {
                    float coeres = ss.r2 * sqrtf(ss.r * ss.rb);
                    float melt = xka(temp[k], rho[k]) / 3.5e5f
                        * (273.15f - temp[k]) * 1.57079632679f * n0sfac
                        * (5.2e6f * ss.r2 + 1.86818719e7f * vf * coeres)
                        / rho[k];
                    melt = fminf(fmaxf(melt * dtcld, -qss[k]), 0.0f);
                    qss[k] += melt; qrr[k] -= melt;
                    temp[k] += 3.5e5f / cpm[k] * melt;
                }
                if (qgg[k] > 0.0f) {
                    float coeres = gg.r2 * sqrtf(gg.r * gg.rb);
                    float melt = xka(temp[k], rho[k]) / 3.5e5f
                        * (273.15f - temp[k])
                        * (gc.precg1 * gg.r2 + gc.precg2 * vf * coeres)
                        / rho[k];
                    melt = fminf(fmaxf(melt * dtcld, -qgg[k]), 0.0f);
                    qgg[k] += melt; qrr[k] -= melt;
                    temp[k] += 3.5e5f / cpm[k] * melt;
                }
            }

            supcol = 273.15f - temp[k];
            xlf = (supcol < 0.0f) ? 3.5e5f : 2.85e6f - xl[k];
            if (supcol < 0.0f && qii[k] > 0.0f) {
                qcc[k] += qii[k]; temp[k] -= xlf / cpm[k] * qii[k];
                qii[k] = 0.0f;
            }
            if (supcol > 40.0f && qcc[k] > 0.0f) {
                qii[k] += qcc[k]; temp[k] += xlf / cpm[k] * qcc[k];
                qcc[k] = 0.0f;
            }
            if (supcol > 0.0f && qcc[k] > 1.0e-15f) {
                float sc = fminf(supcol, 50.0f);
                float freeze = fminf(100.0f * (expf(0.66f * sc) - 1.0f)
                    * rho[k] / 1000.0f / 3.0e8f * qcc[k] * qcc[k] * dtcld,
                    qcc[k]);
                qii[k] += freeze; qcc[k] -= freeze;
                temp[k] += xlf / cpm[k] * freeze;
            }
            if (supcol > 0.0f && qrr[k] > 0.0f) {
                float sc = fminf(supcol, 50.0f);
                float r7 = rr.r3 * rr.r3 * rr.r;
                float freeze = fminf(20.0f * 9.86960440f * 100.0f * 8.0e6f
                    * 1000.0f / rho[k] * (expf(0.66f * sc) - 1.0f)
                    * r7 * dtcld, qrr[k]);
                qgg[k] += freeze; qrr[k] -= freeze;
                temp[k] += xlf / cpm[k] * freeze;
            }

            // Refresh slopes after phase changes exactly where WRF calls
            // slope_wsm6 for its process block.
            rr = rain_slope(qrr[k], rho[k], denfac);
            ss = snow_slope(qss[k], rho[k], denfac, temp[k], &n0sfac);
            gg = graupel_slope(qgg[k], rho[k], denfac, gc);
            float workw = diffac(xl[k], p[id], temp[k], rho[k], qsw[k]);
            float worki = diffac(2.85e6f, p[id], temp[k], rho[k], qsi[k]);
            float vf = venfac(p[id], temp[k], rho[k]);

            float prevp=0, psdep=0, pgdep=0, praut=0, psaut=0, pgaut=0;
            float pracw=0, praci=0, piacr=0, psaci=0, psacw=0, pracs=0;
            float psacr=0, pgacw=0, paacw=0, pgaci=0, pgacr=0, pgacs=0;
            float pigen=0, pidep=0, pseml=0, pgeml=0, psevp=0, pgevp=0;

            float supsat = fmaxf(qvv[k], 1.0e-15f) - qsw[k];
            float satdt = supsat / dtcld;
            if (qcc[k] > 5.02654825e-4f) {
                praut = 6.77389540f * powf(qcc[k], 7.0f/3.0f);
                praut = fminf(praut, qcc[k]/dtcld);
            }
            if (qrr[k] > 1.0e-9f && qcc[k] > 1.0e-15f)
                pracw = fminf(2.48133885e10f * rr.r3 * rr.rb * qcc[k]
                              * denfac, qcc[k]/dtcld);
            if (qrr[k] > 0.0f) {
                float coeres = rr.r2 * sqrtf(rr.r * rr.rb);
                prevp = (rhw[k]-1.0f) * (3.92070763e7f*rr.r2
                    + 8.25851867e8f*vf*coeres) / workw;
                prevp = prevp < 0.0f
                      ? fmaxf(fmaxf(prevp, -qrr[k]/dtcld), satdt/2.0f)
                      : fminf(prevp, satdt/2.0f);
            }

            supcol = 273.15f - temp[k];
            n0sfac = fmaxf(fminf(expf(0.12f*supcol), 5.0e4f), 1.0f);
            supsat = fmaxf(qvv[k], 1.0e-15f) - qsi[k];
            satdt = supsat / dtcld;
            int ifsat = 0;
            float xniarg = rho[k]*fmaxf(qii[k],1.0e-15f);
            float xni = fminf(fmaxf(5.38e7f*powf(xniarg,0.75f),1.0e3f),1.0e6f);
            float eacrs = expf(-0.07f*supcol);
            float xmi = rho[k]*qii[k]/xni;
            float diameter = fminf(11.9f*sqrtf(fmaxf(xmi,0.0f)),500.0e-6f);
            float vt2i = 1.49e4f*powf(fmaxf(diameter,0.0f),1.31f);
            float qsum = fmaxf(qss[k]+qgg[k],1.0e-15f);
            float vtave = qsum > 1.0e-15f
                        ? (ss.vt*qss[k]+gg.vt*qgg[k])/qsum : 0.0f;

            if (supcol > 0.0f && qii[k] > 1.0e-15f) {
                if (qrr[k] > 1.0e-9f) {
                    float af=2*rr.r3+2*diameter*rr.r2+diameter*diameter*rr.r;
                    praci=3.14159265f*qii[k]*8.0e6f*fabsf(rr.vt-vt2i)*af/4;
                    praci*=powf(clamp01(qrr[k]/qii[k]),2); praci=fminf(praci,qii[k]/dtcld);
                    piacr=9.86960440f*841.9f*8.0e6f*1000.0f*xni*denfac
                         *495.459567f*rr.r3*rr.r3*rr.rb/(24.0f*rho[k]);
                    piacr*=powf(clamp01(qii[k]/qrr[k]),2); piacr=fminf(piacr,qrr[k]/dtcld);
                }
                if(qss[k]>1.0e-9f){float af=2*ss.r3+2*diameter*ss.r2+diameter*diameter*ss.r;
                    psaci=3.14159265f*qii[k]*eacrs*2.0e6f*n0sfac*fabsf(vtave-vt2i)*af/4;
                    psaci=fminf(psaci,qii[k]/dtcld);}
                if(qgg[k]>1.0e-9f){float egi=expf(-0.07f*supcol);
                    float af=2*gg.r3+2*diameter*gg.r2+diameter*diameter*gg.r;
                    pgaci=3.14159265f*egi*qii[k]*gc.n0g*fabsf(vtave-vt2i)*af/4;
                    pgaci=fminf(pgaci,qii[k]/dtcld);}
            }
            if(qss[k]>1.0e-9f&&qcc[k]>1.0e-15f)
                psacw=fminf(5.54420857e7f*n0sfac*ss.r3*ss.rb
                    *powf(clamp01(qss[k]/qcc[k]),2)*qcc[k]*denfac,qcc[k]/dtcld);
            if(qgg[k]>1.0e-9f&&qcc[k]>1.0e-15f)
                pgacw=fminf(gc.pacrg*gg.r3*gg.rb*powf(clamp01(qgg[k]/qcc[k]),2)
                    *qcc[k]*denfac,qcc[k]/dtcld);
            if(qsum>1.0e-15f) paacw=(qss[k]*psacw+qgg[k]*pgacw)/qsum;

            if(qss[k]>1.0e-9f&&qrr[k]>1.0e-9f){
                if(supcol>0){float af=5*ss.r3*ss.r3*rr.r+2*ss.r3*ss.r2*rr.r2
                    +0.5f*ss.r2*ss.r2*rr.r3;
                    pracs=9.86960440f*8.0e6f*2.0e6f*n0sfac*fabsf(rr.vt-vtave)
                         *(100.0f/rho[k])*af*powf(clamp01(qrr[k]/qss[k]),2);
                    pracs=fminf(pracs,qss[k]/dtcld);}
                float af=5*rr.r3*rr.r3*ss.r+2*rr.r3*rr.r2*ss.r2
                    +0.5f*rr.r2*rr.r2*ss.r3;
                psacr=9.86960440f*8.0e6f*2.0e6f*n0sfac*fabsf(vtave-rr.vt)
                     *(1000.0f/rho[k])*af*powf(clamp01(qss[k]/qrr[k]),2);
                psacr=fminf(psacr,qrr[k]/dtcld);
            }
            if(qgg[k]>1.0e-9f&&qrr[k]>1.0e-9f){float af=5*rr.r3*rr.r3*gg.r
                +2*rr.r3*rr.r2*gg.r2+0.5f*rr.r2*rr.r2*gg.r3;
                pgacr=9.86960440f*8.0e6f*gc.n0g*fabsf(vtave-rr.vt)
                     *(1000.0f/rho[k])*af*powf(clamp01(qgg[k]/qrr[k]),2);
                pgacr=fminf(pgacr,qrr[k]/dtcld);}
            pgacs=0.0f;
            if(supcol<=0){
                if(qss[k]>0)pseml=fminf(fmaxf(4190.0f*supcol*(paacw+psacr)/3.5e5f,
                                               -qss[k]/dtcld),0.0f);
                if(qgg[k]>0)pgeml=fminf(fmaxf(4190.0f*supcol*(paacw+pgacr)/3.5e5f,
                                               -qgg[k]/dtcld),0.0f);
            }
            if(supcol>0){
                if(qii[k]>0&&!ifsat){pidep=4*diameter*xni*(rhi[k]-1)/worki;
                    float supice=satdt-prevp;
                    pidep=pidep<0?fmaxf(fmaxf(fmaxf(pidep,satdt/2),supice),-qii[k]/dtcld)
                                  :fminf(fminf(pidep,satdt/2),supice);
                    if(fabsf(prevp+pidep)>=fabsf(satdt))ifsat=1;}
                if(qss[k]>0&&!ifsat){float co=ss.r2*sqrtf(ss.r*ss.rb);
                    psdep=(rhi[k]-1)*n0sfac*(5.2e6f*ss.r2+1.86818719e7f*vf*co)/worki;
                    float si=satdt-prevp-pidep;
                    psdep=psdep<0?fmaxf(fmaxf(fmaxf(psdep,-qss[k]/dtcld),satdt/2),si)
                                 :fminf(fminf(psdep,satdt/2),si);
                    if(fabsf(prevp+pidep+psdep)>=fabsf(satdt))ifsat=1;}
                if(qgg[k]>0&&!ifsat){float co=gg.r2*sqrtf(gg.r*gg.rb);
                    pgdep=(rhi[k]-1)*(gc.precg1*gg.r2+gc.precg2*vf*co)/worki;
                    float si=satdt-prevp-pidep-psdep;
                    pgdep=pgdep<0?fmaxf(fmaxf(fmaxf(pgdep,-qgg[k]/dtcld),satdt/2),si)
                                 :fminf(fminf(pgdep,satdt/2),si);
                    if(fabsf(prevp+pidep+psdep+pgdep)>=fabsf(satdt))ifsat=1;}
                if(supsat>0&&!ifsat){float xni0=1.0e3f*expf(0.1f*supcol);
                    float roqi0=4.92e-11f*powf(xni0,1.33f);
                    float si=satdt-prevp-pidep-psdep-pgdep;
                    pigen=fminf(fminf(fmaxf(0.0f,(roqi0/rho[k]-fmax0(qii[k]))/dtcld),satdt),si);}
                if(qii[k]>0)psaut=fmaxf(0.0f,(8.125e-5f/rho[k]-qii[k])/-dtcld);
                if(qss[k]>0)pgaut=fminf(fmaxf(0.0f,1.0e-3f*expf(-0.09f*supcol)
                                              *(qss[k]-6.0e-4f)),qss[k]/dtcld);
            }
            if(supcol<0){
                if(qss[k]>0&&rhw[k]<1){float co=ss.r2*sqrtf(ss.r*ss.rb);
                    psevp=(rhw[k]-1)*n0sfac*(5.2e6f*ss.r2+1.86818719e7f*vf*co)/workw;
                    psevp=fminf(fmaxf(psevp,-qss[k]/dtcld),0.0f);}
                if(qgg[k]>0&&rhw[k]<1){float co=gg.r2*sqrtf(gg.r*gg.rb);
                    pgevp=(rhw[k]-1)*(gc.precg1*gg.r2+gc.precg2*vf*co)/workw;
                    pgevp=fminf(fmaxf(pgevp,-qgg[k]/dtcld),0.0f);}
            }

            float d2=(qrr[k]<1e-4f&&qss[k]<1e-4f)?1.0f:0.0f;
            float d3=qrr[k]<1e-4f?1.0f:0.0f;
#define LIMIT(value_, source_, ...) do { float _s=(source_)*dtcld; if(_s>(value_)){float factor=(value_)/_s; __VA_ARGS__;} } while(0)
            if(temp[k]<=273.15f){
                LIMIT(fmaxf(1e-15f,qcc[k]),praut+pracw+2*paacw,
                      praut*=factor;pracw*=factor;paacw*=factor;);
                LIMIT(fmaxf(1e-15f,qii[k]),psaut-pigen-pidep+praci+psaci+pgaci,
                      psaut*=factor;pigen*=factor;pidep*=factor;praci*=factor;psaci*=factor;pgaci*=factor;);
                LIMIT(fmaxf(1e-15f,qrr[k]),-praut-prevp-pracw+piacr+psacr+pgacr,
                      praut*=factor;prevp*=factor;pracw*=factor;piacr*=factor;psacr*=factor;pgacr*=factor;);
                LIMIT(fmaxf(1e-15f,qss[k]),-(psdep+psaut-pgaut+paacw+piacr*d3+praci*d3
                      -pracs*(1-d2)+psacr*d2+psaci-pgacs),
                      psdep*=factor;psaut*=factor;pgaut*=factor;paacw*=factor;piacr*=factor;
                      praci*=factor;psaci*=factor;pracs*=factor;psacr*=factor;pgacs*=factor;);
                LIMIT(fmaxf(1e-15f,qgg[k]),-(pgdep+pgaut+piacr*(1-d3)+praci*(1-d3)
                      +psacr*(1-d2)+pracs*(1-d2)+pgaci+paacw+pgacr+pgacs),
                      pgdep*=factor;pgaut*=factor;piacr*=factor;praci*=factor;psacr*=factor;
                      pracs*=factor;paacw*=factor;pgaci*=factor;pgacr*=factor;pgacs*=factor;);
                qvv[k]+=-(prevp+psdep+pgdep+pigen+pidep)*dtcld;
                qcc[k]=fmax0(qcc[k]-(praut+pracw+2*paacw)*dtcld);
                qrr[k]=fmax0(qrr[k]+(praut+pracw+prevp-piacr-pgacr-psacr)*dtcld);
                qii[k]=fmax0(qii[k]-(psaut+praci+psaci+pgaci-pigen-pidep)*dtcld);
                qss[k]=fmax0(qss[k]+(psdep+psaut+paacw-pgaut+piacr*d3+praci*d3
                     +psaci-pgacs-pracs*(1-d2)+psacr*d2)*dtcld);
                qgg[k]=fmax0(qgg[k]+(pgdep+pgaut+piacr*(1-d3)+praci*(1-d3)
                     +psacr*(1-d2)+pracs*(1-d2)+pgaci+paacw+pgacr+pgacs)*dtcld);
                float latent=-2.85e6f*(psdep+pgdep+pidep+pigen)-xl[k]*prevp
                    -(2.85e6f-xl[k])*(piacr+2*paacw+pgacr+psacr);
                temp[k]-=latent/cpm[k]*dtcld;
            } else {
                LIMIT(fmaxf(1e-15f,qcc[k]),praut+pracw+2*paacw,
                      praut*=factor;pracw*=factor;paacw*=factor;);
                LIMIT(fmaxf(1e-15f,qrr[k]),-2*paacw-praut+pseml+pgeml-pracw-prevp,
                      praut*=factor;prevp*=factor;pracw*=factor;paacw*=factor;pseml*=factor;pgeml*=factor;);
                LIMIT(fmaxf(1e-9f,qss[k]),pgacs-pseml-psevp,
                      pgacs*=factor;psevp*=factor;pseml*=factor;);
                LIMIT(fmaxf(1e-9f,qgg[k]),-(pgacs+pgevp+pgeml),
                      pgacs*=factor;pgevp*=factor;pgeml*=factor;);
                qvv[k]+=-(prevp+psevp+pgevp)*dtcld;
                qcc[k]=fmax0(qcc[k]-(praut+pracw+2*paacw)*dtcld);
                qrr[k]=fmax0(qrr[k]+(praut+pracw+prevp+2*paacw-pseml-pgeml)*dtcld);
                qss[k]=fmax0(qss[k]+(psevp-pgacs+pseml)*dtcld);
                qgg[k]=fmax0(qgg[k]+(pgacs+pgevp+pgeml)*dtcld);
                float latent=-xl[k]*(prevp+psevp+pgevp)-(2.85e6f-xl[k])*(pseml+pgeml);
                temp[k]-=latent/cpm[k]*dtcld;
            }
#undef LIMIT
            float qsat_new=qsat(temp[k],p[id],false);
            float cond=(fmaxf(qvv[k],1e-15f)-qsat_new)
                /(1+xl[k]*xl[k]/(461.6f*cpm[k])*qsat_new/(temp[k]*temp[k]));
            float pcond=fminf(fmaxf(cond/dtcld,0.0f),fmax0(qvv[k])/dtcld);
            if(qcc[k]>0&&cond<0)pcond=fmaxf(cond,-qcc[k])/dtcld;
            qvv[k]-=pcond*dtcld; qcc[k]=fmax0(qcc[k]+pcond*dtcld);
            temp[k]+=pcond*xl[k]/cpm[k]*dtcld;
            if(qcc[k]<=1e-15f)qcc[k]=0; if(qii[k]<=1e-15f)qii[k]=0;
        }
    }

    for(int k=0;k<nz;++k){
        int id=col+k*stride;
        theta[id]=temp[k]/pii[id]; qv[id]=qvv[k]; qc[id]=qcc[k]; qi[id]=qii[k];
        qr[id]=qrr[k]; qs[id]=qss[k]; qg[id]=qgg[k];
        float rqc=fmaxf(1e-12f,qcc[k]*rho[k]);
        float lamc=powf(523.5987756f*3.0e8f/rqc,1.0f/3.0f);
        effc[id]=(rqc>1e-12f)?fmaxf(2.51f,fminf(1.5e6f/lamc,50.0f)):2.49f;
        float itmp=rho[k]*fmaxf(qii[k],1e-15f);
        float ni=fminf(fmaxf(5.38e7f*powf(itmp,0.75f),1e3f),1e6f);
        float dia=11.9f*sqrtf(fmaxf(qii[k]*rho[k]/ni,0.0f));
        effi[id]=(qii[k]*rho[k]>1e-12f)?fmaxf(10.01f,fminf(0.75f*0.163f*dia*1e6f,125.0f)):4.99f;
        float nsfac=fmaxf(fminf(expf(0.12f*(273.15f-temp[k])),5e4f),1.0f);
        float lams=sqrtf(sqrtf(6.28318531e8f*nsfac/fmaxf(qss[k]*rho[k],1e-12f)));
        effs[id]=(qss[k]*rho[k]>1e-12f)?fmaxf(25.0f,fminf(0.5e6f/lams,999.0f)):9.99f;
        effc[id]=fmaxf(2.49f,fminf(effc[id],50.0f));
        effi[id]=fmaxf(4.99f,fminf(effi[id],125.0f));
        effs[id]=fmaxf(9.99f,fminf(effs[id],999.0f));
    }
}

#ifndef WSM6_CPU_MIRROR
extern "C" __global__ void wsm6_column(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *sr, float *effc, float *effi, float *effs,
    float delt, int hail_opt, int nz, int ny, int nx)
{
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=ny*nx;
    if(col>=stride||nz>WSM6_KMAX)return;
    wsm6_column_impl(theta,qv,qc,qi,qr,qs,qg,den,p,pii,dz,
        rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,sr,
        effc,effi,effs,delt,hail_opt,nz,stride,col);
}
#endif
