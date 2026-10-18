// WOOF's own scalar float32 atan, shared by surface and terrain initialization.
__device__ const unsigned int SFC_CLASSIC_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

__device__ float sfc_classic_atanf(float x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int)hx;
    int id;
#define SFCC_ATAN_HI(i) __uint_as_float(SFC_CLASSIC_ATANF_TAB[(i)])
#define SFCC_ATAN_LO(i) __uint_as_float(SFC_CLASSIC_ATANF_TAB[4 + (i)])
#define SFCC_ATAN_T(i)  __uint_as_float(SFC_CLASSIC_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {                     // |x| >= 2**25
        if (ix > 0x7F800000u) return __fadd_rn(x, x);
        if (signed_hx > 0) return __fadd_rn(SFCC_ATAN_HI(3), SFCC_ATAN_LO(3));
        return __fsub_rn(-SFCC_ATAN_HI(3), SFCC_ATAN_LO(3));
    }
    if (ix < 0x3EE00000u) {                      // |x| < 0.4375
        if (ix < 0x31000000u) return x;          // |x| < 2**-29
        id = -1;
    } else {
        x = __uint_as_float(ix);
        if (ix < 0x3F980000u) {                  // |x| < 1.1875
            if (ix < 0x3F300000u) {              // 7/16 <= |x| < 11/16
                id = 0;
                x = __fdiv_rn(__fsub_rn(__fmul_rn(2.0f, x), 1.0f),
                              __fadd_rn(2.0f, x));
            } else {                             // 11/16 <= |x| < 19/16
                id = 1;
                x = __fdiv_rn(__fsub_rn(x, 1.0f), __fadd_rn(x, 1.0f));
            }
        } else if (ix < 0x401C0000u) {           // |x| < 2.4375
            id = 2;
            x = __fdiv_rn(__fsub_rn(x, 1.5f),
                          __fadd_rn(1.0f, __fmul_rn(1.5f, x)));
        } else {                                 // 2.4375 <= |x| < 2**25
            id = 3;
            x = __fdiv_rn(-1.0f, x);
        }
    }
    float z = __fmul_rn(x, x);
    float w = __fmul_rn(z, z);
    float s1 = __fmul_rn(w, SFCC_ATAN_T(10));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(8), s1));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(6), s1));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(4), s1));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(2), s1));
    s1 = __fmul_rn(z, __fadd_rn(SFCC_ATAN_T(0), s1));
    float s2 = __fmul_rn(w, SFCC_ATAN_T(9));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(7), s2));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(5), s2));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(3), s2));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(1), s2));
    float s = __fadd_rn(s1, s2);
    if (id < 0) return __fsub_rn(x, __fmul_rn(x, s));
    float r = __fsub_rn(SFCC_ATAN_HI(id),
                        __fsub_rn(__fsub_rn(__fmul_rn(x, s), SFCC_ATAN_LO(id)), x));
    return (signed_hx < 0) ? -r : r;
#undef SFCC_ATAN_HI
#undef SFCC_ATAN_LO
#undef SFCC_ATAN_T
}
