/* Pin glibc's FP32 atanf so the CPU and CUDA transcriptions can be tested
 * without WSL, gfortran or a live libm.
 *
 * SFCDIF1 (module_sf_noahmplsm.F:4691, 4698) is Noah-MP's only atanf caller.
 * Its argument is (1 - 16*MOZ)**0.25 with MOZ < 0, so the live domain is
 * [1, inf); the sample below covers that plus every one of the five argument
 * ranges glibc's fdlibm reduction splits on, both signs, and the exact
 * boundary bit patterns between them.
 *
 * Build and run on the oracle host (glibc 2.39, x86-64):
 *
 *     gcc -O2 -ffp-contract=off -o dump_atanf dump_atanf_reference.c -lm
 *     ./dump_atanf > gpuwm/data/noahmp/oracle/glibc-atanf-fp32.csv
 *
 * The exhaustive check that this transcription IS glibc's algorithm -- all
 * 4,278,190,082 non-NaN FP32 inputs, 0 mismatches -- is recorded in
 * gpuwm/core/noahmp_libm.py.  This file only pins a sample of the answers so
 * the port-side tests are self-contained.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float from_bits(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }
static uint32_t to_bits(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }

/* Boundaries of glibc's five argument ranges, and the values either side. */
static const uint32_t EDGES[] = {
    0x00000000u, 0x00000001u, 0x00800000u,
    0x30FFFFFFu, 0x31000000u, 0x31000001u,   /* 2**-29 */
    0x3EDFFFFFu, 0x3EE00000u, 0x3EE00001u,   /* 0.4375 */
    0x3F2FFFFFu, 0x3F300000u, 0x3F300001u,   /* 0.6875 */
    0x3F7FFFFFu, 0x3F800000u, 0x3F800001u,   /* 1 */
    0x3F97FFFFu, 0x3F980000u, 0x3F980001u,   /* 1.1875 */
    0x401BFFFFu, 0x401C0000u, 0x401C0001u,   /* 2.4375 */
    0x4BFFFFFFu, 0x4C000000u, 0x4C000001u,   /* 2**25 */
    0x50800000u, 0x5F000000u, 0x7F7FFFFFu, 0x7F800000u,
};

int main(void)
{
    printf("bits,x,atanf_bits,atanf\n");
    for (unsigned i = 0; i < sizeof(EDGES) / sizeof(EDGES[0]); ++i) {
        for (int sign = 0; sign < 2; ++sign) {
            uint32_t b = EDGES[i] | ((uint32_t)sign << 31);
            float x = from_bits(b);
            printf("%08X,%.9g,%08X,%.9g\n", b, (double)x,
                   to_bits(atanf(x)), (double)atanf(x));
        }
    }
    /* A deterministic LCG sweep of each range, both signs. */
    static const uint32_t LO[] = {0x00000000u, 0x31000000u, 0x3EE00000u,
                                  0x3F300000u, 0x3F980000u, 0x401C0000u,
                                  0x3F800000u};
    static const uint32_t HI[] = {0x31000000u, 0x3EE00000u, 0x3F300000u,
                                  0x3F980000u, 0x401C0000u, 0x4C000000u,
                                  0x42C80000u};
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (unsigned r = 0; r < sizeof(LO) / sizeof(LO[0]); ++r) {
        for (int k = 0; k < 300; ++k) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            uint32_t span = HI[r] - LO[r];
            uint32_t b = LO[r] + (uint32_t)((state >> 33) % span);
            for (int sign = 0; sign < 2; ++sign) {
                uint32_t bs = b | ((uint32_t)sign << 31);
                float x = from_bits(bs);
                printf("%08X,%.9g,%08X,%.9g\n", bs, (double)x,
                       to_bits(atanf(x)), (double)atanf(x));
            }
        }
    }
    return 0;
}
