// Batched entry points assembled after the unchanged band bodies.
extern "C" __global__ void rlw_taumol_batched(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const unsigned long long* __restrict__ bandptrs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    const float* const* tabs = (const float* const*)bandptrs[blockIdx.y];
    switch (blockIdx.y) {
    case 0: rlw_gband1(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 1: rlw_gband2(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 2: rlw_gband3(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 3: rlw_gband4(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 4: rlw_gband5(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 5: rlw_gband6(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 6: rlw_gband7(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 7: rlw_gband8(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 8: rlw_gband9(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 9: rlw_gband10(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 10: rlw_gband11(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 11: rlw_gband12(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 12: rlw_gband13(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 13: rlw_gband14(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 14: rlw_gband15(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    case 15: rlw_gband16(ncol, nl, laytrop_v, fs, isv, wx, chi_mls, oneminus, tabs, taug, fracs); break;
    }
}

// Ordered flux accumulation, one thread per (column, level) row and every
// row's thread summing: for each band the block's rows load that band's
// g-points through shared memory (consecutive threads on consecutive
// g-points of a row, so the (column, layer, g-point) slabs are read in
// runs), then each row runs the band's sums in the original order.  The
// statements are those of rlw_rtrn_accum_coalesced with its slab reads
// replaced by the staged copies (tests/test_rrtmg_lw_batched_layout.py).
#define RLW_ACC_ROWS 64
#define RLW_ACC_G 16    // the widest LW band (ng16 .. ng1 are all <= 16)
extern "C" __global__ void rlw_rtrn_accum_rows(
    int ncol, int nl,
    const float* __restrict__ radld_p,     // (ncol, nl, NGPTLW)
    const float* __restrict__ radclrd_p,
    const float* __restrict__ radlu_p,
    const float* __restrict__ radclru_p,
    const unsigned char* __restrict__ iclddn_p,
    const float* __restrict__ radlu_sfc,   // (ncol, NGPTLW)
    const float* __restrict__ radclru_sfc,
    const int* __restrict__ ngs,           // (16) cumulative g-points
    const float* __restrict__ delwave,     // (16)
    float wtdiff,
    float* __restrict__ totuflux,          // (ncol, nl+1)
    float* __restrict__ totdflux,
    float* __restrict__ totuclfl,
    float* __restrict__ totdclfl)
{
    __shared__ float down[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float downc[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float up[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ float upc[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag[RLW_ACC_ROWS][RLW_ACC_G + 1];
    __shared__ unsigned char flag0[RLW_ACC_ROWS][RLW_ACC_G + 1];
    const long long rows = (long long)ncol * (nl + 1);
    const long long base = (long long)blockIdx.x * RLW_ACC_ROWS;
    const int r = threadIdx.x;
    const bool active = base + r < rows;
    const int col = active ? (int)((base + r) / (nl + 1)) : 0;
    const int lev = active ? (int)((base + r) % (nl + 1)) : 0;
    float totu = 0.0f, totd = 0.0f, totuc = 0.0f, totdc = 0.0f;
    for (int iband = 1; iband <= NBNDLW; ++iband) {
        int g_lo = (iband == 1) ? 0 : ngs[iband - 2];
        int g_hi = ngs[iband - 1];
        int ng = g_hi - g_lo;
        __syncthreads();                   // the previous band's tile is consumed
        for (int slot = threadIdx.x; slot < RLW_ACC_ROWS * ng; slot += blockDim.x) {
            int row = slot / ng, gg = slot % ng, g = g_lo + gg;
            long long t = base + row;
            if (t >= rows) continue;
            int c = (int)(t / (nl + 1)), l = (int)(t % (nl + 1));
            if (l < nl) {
                long long i = ((long long)c * nl + l) * NGPTLW + g;
                down[row][gg] = radld_p[i]; downc[row][gg] = radclrd_p[i];
                flag[row][gg] = iclddn_p[i];
            }
            flag0[row][gg] = iclddn_p[(long long)c * nl * NGPTLW + g];
            long long iu = l == 0 ? (long long)c * NGPTLW + g
                                  : ((long long)c * nl + l - 1) * NGPTLW + g;
            up[row][gg] = l == 0 ? radlu_sfc[iu] : radlu_p[iu];
            upc[row][gg] = l == 0 ? radclru_sfc[iu] : radclru_p[iu];
        }
        __syncthreads();
        if (!active) continue;
        float urad = 0.0f, drad = 0.0f, clru = 0.0f, clrd = 0.0f;
        for (int lane = g_lo; lane < g_hi; ++lane) {
            // downward: drad/clrdrad live at levels 0..nl-1
            if (lev <= nl - 1) {
                drad = RLW_AD(drad, down[r][lane - g_lo]);
                if (flag[r][lane - g_lo])
                    clrd = RLW_AD(clrd, downc[r][lane - g_lo]);
                else
                    clrd = drad;
            }
            // upward: urad/clrurad at levels 0 (sfc) and 1..nl
            if (lev == 0) {
                urad = RLW_AD(urad,
                    up[r][lane - g_lo]);
                if (flag0[r][lane - g_lo])
                    clru = RLW_AD(clru,
                        upc[r][lane - g_lo]);
                else
                    clru = urad;
            } else {
                urad = RLW_AD(urad, up[r][lane - g_lo]);
                if (flag0[r][lane - g_lo])
                    clru = RLW_AD(clru, upc[r][lane - g_lo]);
                else
                    clru = urad;
            }
        }
        float dw = delwave[iband - 1];
        totu = RLW_AD(totu, RLW_MU(RLW_MU(urad, wtdiff), dw));
        totd = RLW_AD(totd, RLW_MU(RLW_MU(drad, wtdiff), dw));
        totuc = RLW_AD(totuc, RLW_MU(RLW_MU(clru, wtdiff), dw));
        totdc = RLW_AD(totdc, RLW_MU(RLW_MU(clrd, wtdiff), dw));
    }
    if (!active) return;
    totuflux[(long long)col * (nl + 1) + lev] = totu;
    totdflux[(long long)col * (nl + 1) + lev] = totd;
    totuclfl[(long long)col * (nl + 1) + lev] = totuc;
    totdclfl[(long long)col * (nl + 1) + lev] = totdc;
}
#undef RLW_ACC_ROWS
#undef RLW_ACC_G

extern "C" __global__ void rlw_rtrn_prol_gpoints(
    int ncol, int nl,
    const float* __restrict__ cldfmc,   // (ncol, NGPTLW, nl)
    const float* __restrict__ taucmc,   // (ncol, NGPTLW, nl)
    const float* __restrict__ secdiff,  // (ncol, 16)
    const int* __restrict__ ngb,
    float* __restrict__ odcld,          // (ncol, NGPTLW, nl)
    float* __restrict__ efclfrac,       // (ncol, NGPTLW, nl)
    int* __restrict__ icldlyr)          // (ncol, nl)
{
    long long tid = blockIdx.x;
    if (tid >= (long long)ncol * nl) return;
    int col = (int)(tid / nl);
    int lay = (int)(tid % nl) + 1;
#define MC2(a) a[((long long)col * nl + (lay - 1)) * NGPTLW + (ig - 1)]
    int flag = 0;
    for (int ig = threadIdx.x + 1; ig <= NGPTLW; ig += blockDim.x) {
        if (MC2(cldfmc) == 1.0f) {
            int ib = ngb[ig - 1];
            float od = RLW_MU(secdiff[(long long)col * NBNDLW + (ib - 1)],
                              MC2(taucmc));
            MC2(odcld) = od;
            float transcld = rlw_exp(-od);
            float abscld = RLW_SU(1.0f, transcld);
            MC2(efclfrac) = RLW_MU(abscld, MC2(cldfmc));
            flag = 1;
        } else {
            MC2(odcld) = 0.0f;
            MC2(efclfrac) = 0.0f;
        }
    }
    unsigned int cloudy = __ballot_sync(0xffffffff, flag);
    if (threadIdx.x % 32 == 0 && cloudy)
        atomicOr(&icldlyr[(long long)col * nl + (lay - 1)], 1);
#undef MC2
}
