// WOOF's own legacy float32 operation sequence for GF strict normalization.
// Restored from this engine's history before c0ffc53b5. The default path
// keeps the current correctly rounded routine. No reference data is read.
#ifdef GPUWM_WRF_EXACT
__device__ float gf_wrf_exp(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    double z = DMUL(0x1.71547652b82fep+0 * 32.0, xd);
    return gfk_d2f_rn(gfk_exp2_core(
        z, GFK_EXP2F_SHIFT,
        GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        GFK_EXP2F_P1 / 32.0 / 32.0,
        GFK_EXP2F_P2 / 32.0, 0ULL));
}
__device__ float gf_wrf_gamma_product(float x, float x_eps, int n, float *eps)
{
    double x_full = DADD((double)x, (double)x_eps);
    double ret = x_full;
    for (int i = 1; i < n; i++)
        ret = DMUL(ret, DADD(x_full, (double)i));
    float fret = __double2float_rn(ret);
    *eps = __double2float_rn(DDIV(DSUB(ret, (double)fret), (double)fret));
    return fret;
}
__device__ float gf_wrf_exp2(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(128.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > 0.0f) return __int_as_float(0x7f800000);
        if (x <= -150.0f) return 0.0f;
    }
    double xd = (double)x;
    return gfk_d2f_rn(gfk_exp2_core(
        xd, GFK_EXP2F_SHIFT_SCALED,
        GFK_EXP2F_P0, GFK_EXP2F_P1, GFK_EXP2F_P2, 0ULL));
}
#define EM1_HUGE   __uint_as_float(0x7149F2CAu)
#define EM1_OTHR   __uint_as_float(0x42B17180u)
#define EM1_LN2HI  __uint_as_float(0x3F317180u)
#define EM1_LN2LO  __uint_as_float(0x3717F7D1u)
#define EM1_IVLN2  __uint_as_float(0x3FB8AA3Bu)
#define EM1_Q1     __uint_as_float(0xBD088889u)
#define EM1_Q2     __uint_as_float(0x3AD00D01u)
#define EM1_Q3     __uint_as_float(0xB8A670CDu)
#define EM1_Q4     __uint_as_float(0x36867E54u)
#define EM1_Q5     __uint_as_float(0xB457EDBBu)
#define EM1_TINYM1 __uint_as_float(0xBF800000u)
__device__ float gf_wrf_expm1(float x)
{
    float y, hi, lo, c, t, e, hxs, hfx, r1;
    int k, xsb;
    unsigned int hx = __float_as_uint(x);
    xsb = (int)(hx & 0x80000000u);
    hx &= 0x7fffffffu;
    c = 0.0f;

    if (hx >= 0x4195b844u) {
        if (hx >= 0x42b17218u) {
            if (hx > 0x7f800000u) return FADD(x, x);
            if (hx == 0x7f800000u)
                return (xsb == 0) ? x : -1.0f;
            if (x > EM1_OTHR) return FMUL(EM1_HUGE, EM1_HUGE);
        }
        if (xsb != 0) return EM1_TINYM1;
    }

    if (hx > 0x3eb17218u) {
        if (hx < 0x3F851592u) {
            if (xsb == 0) { hi = FSUB(x, EM1_LN2HI); lo = EM1_LN2LO;  k = 1; }
            else          { hi = FADD(x, EM1_LN2HI); lo = -EM1_LN2LO; k = -1; }
        } else {
            float kf = FADD(FMUL(EM1_IVLN2, x), (xsb == 0) ? 0.5f : -0.5f);
            k  = (int)kf;
            t  = (float)k;
            hi = FSUB(x, FMUL(t, EM1_LN2HI));
            lo = FMUL(t, EM1_LN2LO);
        }
        x = FSUB(hi, lo);
        c = FSUB(FSUB(hi, x), lo);
    } else if (hx < 0x33000000u) {
        t = FADD(EM1_HUGE, x);
        return FSUB(x, FSUB(t, FADD(EM1_HUGE, x)));
    } else {
        k = 0;
    }

    hfx = FMUL(0.5f, x);
    hxs = FMUL(x, hfx);
    r1 = FADD(1.0f, FMUL(hxs, FADD(EM1_Q1, FMUL(hxs, FADD(EM1_Q2,
             FMUL(hxs, FADD(EM1_Q3, FMUL(hxs, FADD(EM1_Q4,
             FMUL(hxs, EM1_Q5))))))))));
    t = FSUB(3.0f, FMUL(r1, hfx));
    e = FMUL(hxs, FDIV(FSUB(r1, t), FSUB(6.0f, FMUL(x, t))));
    if (k == 0) return FSUB(x, FSUB(FMUL(x, e), hxs));
    e = FSUB(FMUL(x, FSUB(e, c)), c);
    e = FSUB(e, hxs);
    if (k == -1) return FSUB(FMUL(0.5f, FSUB(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f) return FMUL(-2.0f, FSUB(e, FADD(x, 0.5f)));
        return FADD(1.0f, FMUL(2.0f, FSUB(x, e)));
    }
    if (k <= -2 || k > 56) {
        y = FSUB(1.0f, FSUB(e, x));
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
        return FSUB(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3f800000u - (0x1000000u >> k));
        y = FSUB(t, FSUB(e, x));
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
    } else {
        t = __uint_as_float((unsigned int)(0x7f - k) << 23);
        y = FSUB(x, FADD(e, t));
        y = FADD(y, 1.0f);
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
    }
    return y;
}
#define GFW_LG_A0  __uint_as_float(0x3D9E233Fu)
#define GFW_LG_A1  __uint_as_float(0x3EA51A66u)
#define GFW_LG_A2  __uint_as_float(0x3D89F001u)
#define GFW_LG_A3  __uint_as_float(0x3CA89915u)
#define GFW_LG_A4  __uint_as_float(0x3BF2027Eu)
#define GFW_LG_A5  __uint_as_float(0x3B3D6EC6u)
#define GFW_LG_A6  __uint_as_float(0x3A9C54A1u)
#define GFW_LG_A7  __uint_as_float(0x3A05B634u)
#define GFW_LG_A8  __uint_as_float(0x39679767u)
#define GFW_LG_A9  __uint_as_float(0x38E28445u)
#define GFW_LG_A10 __uint_as_float(0x37D383A2u)
#define GFW_LG_A11 __uint_as_float(0x383C2C75u)
#define GFW_LG_TC  __uint_as_float(0x3FBB16C3u)
#define GFW_LG_TF  __uint_as_float(0xBDF8CDCDu)
#define GFW_LG_TT  __uint_as_float(0x31E61C52u)


#define GFW_LG_TCM1 __uint_as_float(0x3EEC5B0Cu)
#define GFW_LG_T0  __uint_as_float(0x3EF7B95Eu)
#define GFW_LG_T1  __uint_as_float(0xBE17213Cu)
#define GFW_LG_T2  __uint_as_float(0x3D845A15u)
#define GFW_LG_T3  __uint_as_float(0xBD064D47u)
#define GFW_LG_T4  __uint_as_float(0x3C93373Du)
#define GFW_LG_T5  __uint_as_float(0xBC28FCFEu)
#define GFW_LG_T6  __uint_as_float(0x3BC7E707u)
#define GFW_LG_T7  __uint_as_float(0xBB7177FEu)
#define GFW_LG_T8  __uint_as_float(0x3B141699u)
#define GFW_LG_T9  __uint_as_float(0xBAB7F476u)
#define GFW_LG_T10 __uint_as_float(0x3A66F867u)
#define GFW_LG_T11 __uint_as_float(0xBA0D3085u)
#define GFW_LG_T12 __uint_as_float(0x39A57B6Bu)
#define GFW_LG_T13 __uint_as_float(0xB9A3F927u)
#define GFW_LG_T14 __uint_as_float(0x39AFE9F7u)
#define GFW_LG_U0  __uint_as_float(0xBD9E233Fu)
#define GFW_LG_U1  __uint_as_float(0x3F2200F4u)
#define GFW_LG_U2  __uint_as_float(0x3FBA3AE7u)
#define GFW_LG_U3  __uint_as_float(0x3F7A4BB2u)
#define GFW_LG_U4  __uint_as_float(0x3E6A7578u)
#define GFW_LG_U5  __uint_as_float(0x3C5B3C5Eu)
#define GFW_LG_V1  __uint_as_float(0x401D2EBEu)
#define GFW_LG_V2  __uint_as_float(0x4008392Du)
#define GFW_LG_V3  __uint_as_float(0x3F44EFDFu)
#define GFW_LG_V4  __uint_as_float(0x3DD572AFu)
#define GFW_LG_V5  __uint_as_float(0x3B52D5DBu)
#define GFW_LG_S0  __uint_as_float(0xBD9E233Fu)
#define GFW_LG_S1  __uint_as_float(0x3E5C245Au)
#define GFW_LG_S2  __uint_as_float(0x3EA6CC7Au)
#define GFW_LG_S3  __uint_as_float(0x3E15DCE6u)
#define GFW_LG_S4  __uint_as_float(0x3CDA40E4u)
#define GFW_LG_S5  __uint_as_float(0x3AF135B4u)
#define GFW_LG_S6  __uint_as_float(0x3805FF67u)
#define GFW_LG_R1  __uint_as_float(0x3FB22D3Bu)
#define GFW_LG_R2  __uint_as_float(0x3F38D0C5u)
#define GFW_LG_R3  __uint_as_float(0x3E300F6Eu)
#define GFW_LG_R4  __uint_as_float(0x3C98BF54u)
#define GFW_LG_R5  __uint_as_float(0x3A4BEED6u)
#define GFW_LG_R6  __uint_as_float(0x36F5D7BDu)
#define GFW_LG_W0  __uint_as_float(0x3ED67F1Du)
#define GFW_LG_W1  __uint_as_float(0x3DAAAAABu)
#define GFW_LG_W2  __uint_as_float(0xBB360B61u)
#define GFW_LG_W3  __uint_as_float(0x3A500CFDu)
#define GFW_LG_W4  __uint_as_float(0xBA1C065Cu)
#define GFW_LG_W5  __uint_as_float(0x3A5B3DD2u)
#define GFW_LG_W6  __uint_as_float(0xBAD5C4E8u)

__device__ float gf_wrf_lgamma_pos(float x)
{
    unsigned int hx = __float_as_uint(x);
    int ix = (int)(hx & 0x7fffffffu);
    if ((int)hx < 0 ) return __int_as_float(0x7fc00000);
    if (ix >= 0x7f800000) return FMUL(x, x);
    if (ix == 0) return FDIV(1.0f, fabsf(x));
    if (ix < 0x30800000) return -gfk_log(x);

    float t, y, z, p, p1, p2, p3, q, r, w;
    int i;
    y = 0.0f; i = 0;
    if (ix == 0x3f800000 || ix == 0x40000000) {
        r = 0.0f;
    } else if (ix < 0x40000000) {
        if (ix <= 0x3f666666) {
            r = -gfk_log(x);
            if (ix >= 0x3f3b4a20)      { y = FSUB(1.0f, x); i = 0; }
            else if (ix >= 0x3e6d3308) { y = FSUB(x, GFW_LG_TCM1); i = 1; }
            else                       { y = x; i = 2; }
        } else {
            r = 0.0f;
            if (ix >= 0x3fdda618)      { y = FSUB(2.0f, x); i = 0; }
            else if (ix >= 0x3F9da620) { y = FSUB(x, GFW_LG_TC); i = 1; }
            else                       { y = FSUB(x, 1.0f); i = 2; }
        }
        switch (i) {
        case 0:
            z = FMUL(y, y);
            p1 = FADD(GFW_LG_A0, FMUL(z, FADD(GFW_LG_A2, FMUL(z, FADD(GFW_LG_A4,
                 FMUL(z, FADD(GFW_LG_A6, FMUL(z, FADD(GFW_LG_A8,
                 FMUL(z, GFW_LG_A10))))))))));
            p2 = FMUL(z, FADD(GFW_LG_A1, FMUL(z, FADD(GFW_LG_A3, FMUL(z, FADD(GFW_LG_A5,
                 FMUL(z, FADD(GFW_LG_A7, FMUL(z, FADD(GFW_LG_A9,
                 FMUL(z, GFW_LG_A11)))))))))));
            p = FADD(FMUL(y, p1), p2);
            r = FADD(r, FSUB(p, FMUL(0.5f, y)));
            break;
        case 1:
            z = FMUL(y, y);
            w = FMUL(z, y);
            p1 = FADD(GFW_LG_T0, FMUL(w, FADD(GFW_LG_T3, FMUL(w, FADD(GFW_LG_T6,
                 FMUL(w, FADD(GFW_LG_T9, FMUL(w, GFW_LG_T12))))))));
            p2 = FADD(GFW_LG_T1, FMUL(w, FADD(GFW_LG_T4, FMUL(w, FADD(GFW_LG_T7,
                 FMUL(w, FADD(GFW_LG_T10, FMUL(w, GFW_LG_T13))))))));
            p3 = FADD(GFW_LG_T2, FMUL(w, FADD(GFW_LG_T5, FMUL(w, FADD(GFW_LG_T8,
                 FMUL(w, FADD(GFW_LG_T11, FMUL(w, GFW_LG_T14))))))));
            p = FSUB(FMUL(z, p1), FSUB(GFW_LG_TT, FMUL(w, FADD(p2, FMUL(y, p3)))));
            r = FADD(r, FADD(GFW_LG_TF, p));
            break;
        case 2:
            p1 = FMUL(y, FADD(GFW_LG_U0, FMUL(y, FADD(GFW_LG_U1, FMUL(y, FADD(GFW_LG_U2,
                 FMUL(y, FADD(GFW_LG_U3, FMUL(y, FADD(GFW_LG_U4,
                 FMUL(y, GFW_LG_U5)))))))))));
            p2 = FADD(1.0f, FMUL(y, FADD(GFW_LG_V1, FMUL(y, FADD(GFW_LG_V2,
                 FMUL(y, FADD(GFW_LG_V3, FMUL(y, FADD(GFW_LG_V4,
                 FMUL(y, GFW_LG_V5))))))))));
            r = FADD(r, FADD(FMUL(-0.5f, y), FDIV(p1, p2)));
            break;
        }
    } else if (ix < 0x41000000) {
        i = (int)x;
        y = FSUB(x, (float)i);
        p = FMUL(y, FADD(GFW_LG_S0, FMUL(y, FADD(GFW_LG_S1, FMUL(y, FADD(GFW_LG_S2,
            FMUL(y, FADD(GFW_LG_S3, FMUL(y, FADD(GFW_LG_S4, FMUL(y, FADD(GFW_LG_S5,
            FMUL(y, GFW_LG_S6)))))))))))));
        q = FADD(1.0f, FMUL(y, FADD(GFW_LG_R1, FMUL(y, FADD(GFW_LG_R2, FMUL(y,
            FADD(GFW_LG_R3, FMUL(y, FADD(GFW_LG_R4, FMUL(y, FADD(GFW_LG_R5,
            FMUL(y, GFW_LG_R6))))))))))));
        r = FADD(FMUL(0.5f, y), FDIV(p, q));
        z = 1.0f;
        switch (i) {
        case 7: z = FMUL(z, FADD(y, 6.0f));
        case 6: z = FMUL(z, FADD(y, 5.0f));
        case 5: z = FMUL(z, FADD(y, 4.0f));
        case 4: z = FMUL(z, FADD(y, 3.0f));
        case 3: z = FMUL(z, FADD(y, 2.0f));
                r = FADD(r, gfk_log(z));
                break;
        }
    } else if (ix < 0x4c800000) {
        t = gfk_log(x);
        z = FDIV(1.0f, x);
        y = FMUL(z, z);
        w = FADD(GFW_LG_W0, FMUL(z, FADD(GFW_LG_W1, FMUL(y, FADD(GFW_LG_W2, FMUL(y,
            FADD(GFW_LG_W3, FMUL(y, FADD(GFW_LG_W4, FMUL(y, FADD(GFW_LG_W5,
            FMUL(y, GFW_LG_W6))))))))))));
        r = FADD(FMUL(FSUB(x, 0.5f), FSUB(t, 1.0f)), w);
    } else {
        r = FMUL(x, FSUB(gfk_log(x), 1.0f));
    }
    return r;
}

#define GFW_GAM_C0     __uint_as_float(0x3DAAAAABu)
#define GFW_GAM_C1     __uint_as_float(0xBB360B61u)
#define GFW_GAM_C2     __uint_as_float(0x3A500D01u)
#define GFW_GAM_SQRT12 __uint_as_float(0x3F3504F3u)
#define GFW_GAM_TWOPI  __uint_as_float(0x40C90FDBu)

__device__ float gf_wrf_gammaf_positive(float x, int *exp2_adj)
{
    if (x < 0.5f) {
        *exp2_adj = 0;
        return FDIV(gf_wrf_exp(gf_wrf_lgamma_pos(FADD(x, 1.0f))), x);
    } else if (x <= 1.5f) {
        *exp2_adj = 0;
        return gf_wrf_exp(gf_wrf_lgamma_pos(x));
    } else if (x < 2.5f) {
        *exp2_adj = 0;
        float x_adj = FSUB(x, 1.0f);
        return FMUL(gf_wrf_exp(gf_wrf_lgamma_pos(x_adj)), x_adj);
    } else {
        float eps = 0.0f;
        float x_eps = 0.0f;
        float x_adj = x;
        float prod = 1.0f;
        if (x < 4.0f) {
            float n = ceilf(FSUB(4.0f, x));
            x_adj = FADD(x, n);
            x_eps = FSUB(x, FSUB(x_adj, n));
            prod = gf_wrf_gamma_product(FSUB(x_adj, n), x_eps, (int)n, &eps);
        }
        float exp_adj = -eps;
        float x_adj_int = roundf(x_adj);
        float x_adj_frac = FSUB(x_adj, x_adj_int);
        int x_adj_log2;
        float x_adj_mant = frexpf(x_adj, &x_adj_log2);
        if (x_adj_mant < GFW_GAM_SQRT12) {
            x_adj_log2--;
            x_adj_mant = FMUL(x_adj_mant, 2.0f);
        }
        *exp2_adj = x_adj_log2 * (int)x_adj_int;
        float ret = FDIV(FMUL(FMUL(FMUL(
            gfk_pow(x_adj_mant, x_adj),
            gf_wrf_exp2(FMUL((float)x_adj_log2, x_adj_frac))),
            gf_wrf_exp(-x_adj)),
            FSQRT(FDIV(GFW_GAM_TWOPI, x_adj))),
            prod);
        exp_adj = FADD(exp_adj, FMUL(x_eps, gfk_log(x_adj)));
        float bsum = GFW_GAM_C2;
        float x_adj2 = FMUL(x_adj, x_adj);
        bsum = FADD(FDIV(bsum, x_adj2), GFW_GAM_C1);
        bsum = FADD(FDIV(bsum, x_adj2), GFW_GAM_C0);
        exp_adj = FADD(exp_adj, FDIV(bsum, x_adj));
        return FADD(ret, FMUL(ret, gf_wrf_expm1(exp_adj)));
    }
}

__device__ float gf_wrf_tgamma(float x)
{
    unsigned int hx = __float_as_uint(x);
    if ((hx & 0x80000000u) || (hx & 0x7fffffffu) >= 0x7f800000u
        || (hx & 0x7fffffffu) == 0u)
        return __int_as_float(0x7fc00000);
    if (x >= 36.0f)
        return __int_as_float(0x7f800000);
    int exp2_adj;
    float tret = gf_wrf_gammaf_positive(x, &exp2_adj);
    float ret = scalbnf(tret, exp2_adj);


    return ret;
}
#endif
