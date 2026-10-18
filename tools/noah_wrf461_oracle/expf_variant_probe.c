/* expf_variant_probe.c -- which arithmetic does the host C library's expf use?
 *
 *   gcc -O2 -ffp-contract=off -o expf_variant_probe expf_variant_probe.c -lm
 *   ./expf_variant_probe
 *
 * WOOF's gfk_exp (gpuwm/core/kernels/glibc_flt32.cuh) is evaluated here on
 * the host in two forms over all 2^32 float32 inputs and each is compared,
 * bit for bit, with the host's expf:
 *   plain  every multiply and add rounded separately, as the device header
 *          spells it (DMUL/DADD);
 *   fused  the three polynomial multiply-adds of the exp2 core as single
 *          fused operations (fma);
 *   fused2 as fused, and also the argument reduction: the product
 *          InvLn2N*x feeds only an add (kd = z + SHIFT) and a subtract
 *          (r = z - kd), so a compiler contracting multiply-adds turns both
 *          uses into fma and r keeps the product's rounding error.
 * The last is what any FMA-contracting build of the algorithm produces.
 * The table and constants are the header's.  Nothing here is library source.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const uint64_t TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL };
#define P0 0x1.c6af84b912394p-5
#define P1 0x1.ebfce50fac4f3p-3
#define P2 0x1.62e42ff0c52d6p-1
#define SHIFT 0x1.8p+52

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static float woof_exp(float x, int fused)
{
    uint32_t abstop = (f2u(x) >> 20) & 0x7ffu;
    if (abstop >= (f2u(88.0f) >> 20)) {
        if (f2u(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return x + x;
        if (x > u2f(0x42b17218)) return u2f(0x7f800000);
        if (x < -u2f(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    const double c = 0x1.71547652b82fep+0 * 32.0;
    double z = c * xd;
    double kd = fused == 2 ? fma(c, xd, SHIFT) : z + SHIFT;
    uint64_t ki; memcpy(&ki, &kd, 8);
    kd -= SHIFT;
    double r = fused == 2 ? fma(c, xd, -kd) : z - kd;
    uint64_t t = TAB[ki & 31] + (ki << (52 - 5));
    double s; memcpy(&s, &t, 8);
    double p0 = P0 / 32.0 / 32.0 / 32.0, p1 = P1 / 32.0 / 32.0, p2 = P2 / 32.0;
    double y;
    if (fused) {
        double zz = fma(p0, r, p1);
        double r2 = r * r;
        y = fma(p2, r, 1.0);
        y = fma(zz, r2, y);
    } else {
        double zz = p0 * r + p1;
        double r2 = r * r;
        y = p2 * r + 1.0;
        y = zz * r2 + y;
    }
    y = y * s;
    return (float)y;
}

int main(void)
{
    unsigned long long diff[3] = {0, 0, 0};
    uint32_t first[3][4];
    for (uint64_t i = 0; i <= 0xffffffffULL; i++) {
        float x = u2f((uint32_t)i);
        float want = expf(x);
        for (int v = 0; v < 3; v++) {
            float got = woof_exp(x, v);
            if (f2u(got) != f2u(want) && !(isnan(got) && isnan(want))) {
                if (diff[v] < 4) first[v][diff[v]] = (uint32_t)i;
                diff[v]++;
            }
        }
    }
    const char* name[3] = {"plain", "fused", "fused2"};
    for (int v = 0; v < 3; v++) {
        printf("%s: %llu of 4294967296 differ from the host expf", name[v], diff[v]);
        for (unsigned long long k = 0; k < diff[v] && k < 4; k++)
            printf(" %08x", first[v][k]);
        printf("\n");
    }
    return 0;
}
