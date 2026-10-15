// WRF v4.8.0 module_sf_urban.F:4450-4498 and 4504-4618.
// Root-aware soil-water solve. Uses the existing WOOF arithmetic and helpers.
// Include after urban_ucm.cu; this is not yet coupled to a forecast generation.
#ifndef GPUWM_URBAN_GROUND_WATER_480
#define GPUWM_URBAN_GROUND_WATER_480
__device__ void u_srt_480(float* rhstt, float edir, const float* et, const float* smcp,
                      float pcpdrp, const float* zsoil, float dwsat, float dksat,
                      float smcmax, float bexp, float& runoff1, float& runoff2,
                      float dt, float smcwlt, const float* sroot, float* ai, float* bi, float* ci)
{
    const int nsoil = GR_NGR;
    float pddum = pcpdrp;
    runoff1 = 0.0f;
    float par = 2.0e-6f;
    float wdf, wcnd;
    if (pcpdrp != 0.0f) {
        float ddmax[3];
        float smcav = FSUB(smcmax, smcwlt);
        ddmax[0] = FMUL(FSUB(0.0f, zsoil[0]), smcav);
        ddmax[0] = FMUL(ddmax[0], FSUB(1.0f, FDIV(FSUB(smcp[0], smcwlt), smcav)));
        ddmax[1] = FMUL(FSUB(zsoil[0], zsoil[1]), smcav);
        ddmax[1] = FMUL(ddmax[1], FSUB(1.0f, FDIV(FSUB(smcp[1], smcwlt), smcav)));
        ddmax[2] = FMUL(FSUB(zsoil[1], zsoil[2]), smcav);
        ddmax[2] = FMUL(ddmax[2], FSUB(1.0f, FDIV(FSUB(smcp[2], smcwlt), smcav)));
        float dd = FADD(FADD(ddmax[0], ddmax[1]), ddmax[2]);
        float dt1 = FDIV(dt, 86400.0f);
        float kdt = FDIV(FMUL(3.0f, dksat), par);
        float val = FSUB(1.0f, gfk_exp(FSUB(0.0f, FMUL(kdt, dt1))));
        float ddt = FMUL(dd, val);
        float px = FMUL(pcpdrp, dt);
        if (px < 0.0f) px = 0.0f;
        float infmax = FDIV(FMUL(px, FDIV(ddt, FADD(px, ddt))), dt);
        float mxsmc = smcp[0];
        u_wdfcnd(wdf, wcnd, mxsmc, smcmax, bexp, dksat, dwsat);
        infmax = u_max(infmax, wcnd);
        infmax = u_min(infmax, FDIV(px, dt));
        if (pcpdrp > infmax) {
            runoff1 = FSUB(pcpdrp, infmax);
            pddum = infmax;
        }
    }
    u_wdfcnd(wdf, wcnd, smcp[0], smcmax, bexp, dksat, dwsat);
    float ddz = FDIV(1.0f, FSUB(0.0f, FMUL(0.5f, zsoil[1])));
    ai[0] = 0.0f;
    bi[0] = FDIV(FMUL(wdf, ddz), FSUB(0.0f, zsoil[0]));
    ci[0] = FSUB(0.0f, bi[0]);
    float dsmdz = FDIV(FSUB(smcp[0], smcp[1]), FSUB(0.0f, FMUL(0.5f, zsoil[1])));
    rhstt[0] = FDIV(FADD(FADD(FADD(FSUB(FADD(FMUL(wdf, dsmdz), wcnd), pddum), edir), et[0]), sroot[0]),
                    zsoil[0]);
    float ddz2 = 0.0f;
    float wdf2 = 0.0f, wcnd2 = 0.0f, dsmdz2 = 0.0f;
    for (int k = 2; k <= nsoil - 1; ++k) {         // Fortran K
        int i = k - 1;                              // 0-based
        float denom2 = FSUB(zsoil[i - 1], zsoil[i]);
        if (k != nsoil - 1) {
            float mxsmc2 = smcp[i];
            u_wdfcnd(wdf2, wcnd2, mxsmc2, smcmax, bexp, dksat, dwsat);
            float denom = FSUB(zsoil[i - 1], zsoil[i + 1]);
            dsmdz2 = FDIV(FSUB(smcp[i], smcp[i + 1]), FMUL(denom, 0.5f));
            ddz2 = FDIV(2.0f, denom);
            ci[i] = FSUB(0.0f, FDIV(FMUL(wdf2, ddz2), denom2));
        } else {
            u_wdfcnd(wdf2, wcnd2, smcp[nsoil - 2], smcmax, bexp, dksat, dwsat);
            dsmdz2 = 0.0f;
            ci[i] = 0.0f;
        }
        float numer = FADD(FADD(FSUB(FSUB(FMUL(wdf2, dsmdz2), FMUL(wdf, dsmdz)), wcnd), et[i]), sroot[i]);
        rhstt[i] = FDIV(numer, FSUB(0.0f, denom2));
        ai[i] = FSUB(0.0f, FDIV(FMUL(wdf, ddz), denom2));
        bi[i] = FSUB(0.0f, FADD(ai[i], ci[i]));
        if (k == nsoil - 1) runoff2 = 0.0f;
        if (k != nsoil - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

__device__ void u_sstep_480(const float* smcp, float* smc, float cmcp, float& cmc,
                        float* rhstt, float rhsct, float dt, float smcmax,
                        float cmcmax, float& runoff3, const float* zsoil,
                        float* ai, float* bi, float* ci)
{
    const int n = GR_NGR - 1;
    float rhsttin[GR_NGR], ciin[GR_NGR];
    for (int k = 0; k < n; ++k) {
        rhstt[k] = FMUL(rhstt[k], dt);
        ai[k] = FMUL(ai[k], dt);
        bi[k] = FADD(1.0f, FMUL(bi[k], dt));
        ci[k] = FMUL(ci[k], dt);
    }
    for (int k = 0; k < n; ++k) rhsttin[k] = rhstt[k];
    for (int k = 0; k < n; ++k) ciin[k] = ci[k];
    u_rosr12(ci, ai, bi, ciin, rhsttin, rhstt, n);
    float wplus = 0.0f;
    runoff3 = 0.0f;
    float ddz = FSUB(0.0f, zsoil[0]);
    for (int k = 0; k < n; ++k) {
        if (k != 0) ddz = FSUB(zsoil[k - 1], zsoil[k]);
        float stot = FADD(FADD(smcp[k], ci[k]), FDIV(wplus, ddz));
        if (stot > smcmax) {
            if (k == 0) {
                ddz = FSUB(0.0f, zsoil[0]);
            } else {
                ddz = FADD(FSUB(0.0f, zsoil[k]), zsoil[k - 1]);
            }
            wplus = FMUL(FSUB(stot, smcmax), ddz);
        } else {
            wplus = 0.0f;
        }
        smc[k] = u_max(u_min(stot, smcmax), 0.07f);
    }
    runoff3 = wplus;
    cmc = FADD(cmcp, FMUL(dt, rhsct));
    if (cmc < 1.e-20f) cmc = 0.0f;
    cmc = u_min(cmc, cmcmax);
}

__device__ void u_smflx_480(const float* smcp, float* smc, float cmcp, float& cmc,
                        float dt, float prcp1, const float* zsoil, float smcmax,
                        float bexp, float smcwlt, float dksat, float dwsat,
                        float shdfac, float cmcmax, float& runoff1,
                        float& runoff2, float& runoff3, float edir, float ec,
                        const float* et, const float* sroot, float& drip)
{
    float ai[GR_NGR], bi[GR_NGR], ci[GR_NGR], rhstt[GR_NGR];
    for (int k = 0; k < GR_NGR; ++k) { ai[k] = 0.0f; bi[k] = 0.0f; ci[k] = 0.0f; rhstt[k] = 0.0f; }
    float rhsct = FSUB(FDIV(FMUL(FMUL(shdfac, prcp1), 0.001f), 3600.0f), ec);
    drip = 0.0f;
    float trhsct = FMUL(dt, rhsct);
    float excess = FADD(cmcp, trhsct);
    if (excess > cmcmax) drip = FSUB(excess, cmcmax);
    float pcpdrp = FADD(FDIV(FMUL(FMUL(FSUB(1.0f, shdfac), prcp1), 0.001f), 3600.0f),
                        FDIV(drip, dt));
    u_srt_480(rhstt, edir, et, smcp, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp,
          runoff1, runoff2, dt, smcwlt, sroot, ai, bi, ci);
    u_sstep_480(smcp, smc, cmcp, cmc, rhstt, rhsct, dt, smcmax, cmcmax, runoff3,
            zsoil, ai, bi, ci);
}


#endif
