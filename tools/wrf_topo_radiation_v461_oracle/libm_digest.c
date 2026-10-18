/* Whole-domain digests of the float libm words WRF's slope_rad /
 * topo_shading statements call (host side).
 *
 *   gcc -O2 -fopenmp -fno-builtin -o libm_digest libm_digest.c -lm
 *   ./libm_digest WORD                    digest line, under the libm run
 *   ./libm_digest WORD BUCKET OUT.bin     one bucket's outputs, raw float32
 *
 * For each word: every float32 input in the domain the topo code can reach
 * (NaN inputs skipped), digest = sum over inputs of mix(input bits << 32 |
 * output bits) mod 2^64, overall and per input bucket (the 512 buckets are
 * sign and exponent of the input, u >> 23, so a difference is located to a
 * binade).  libm_digest.py computes the same sums on the device from
 * WOOF's own kernel unit, so equal digests mean the two words agree bit for
 * bit on every input of the bucket.  A bucket dump writes 2^23 outputs,
 * 0xFFFFFFFF where the input is outside the domain.  ``sincosf`` digests
 * the pair gfortran emits for SIN and COS of one argument and reports how
 * many of its outputs differ from sinf/cosf.
 *
 * Three two-argument words index pairs instead of inputs: input u selects
 * (y, x) through the integer hash h32, identically on both sides.
 * ``atan2f_pairs`` draws all 32 bits of each; ``atan2f_grad`` draws
 * magnitudes in [2^-20, 2^11), the range of terrain gradients.  Both skip
 * non-finite arguments and results below about 2^-100 (|y| < |x| 2^-100).
 * ``atan2f_resolve_x1000`` is atan2f(u, 1000) again, for the device word
 * that takes the exact midpoint decision on every input.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t mix(uint64_t h)
{
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return h;
}

static uint32_t h32(uint32_t v)
{
    v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU;
    v ^= v >> 16;
    return v;
}

/* (y, x) of pair u: kind 0 all bits, kind 1 gradient magnitudes. */
static int pair_of(uint32_t u, int kind, float *y, float *x)
{
    uint32_t a = h32(u), b = h32(u ^ 0xA5A5A5A5u);
    if (kind == 1) {
        uint32_t ea = 107u + h32(u ^ 0x1234567u) % 31u;
        uint32_t eb = 107u + h32(u ^ 0x7654321u) % 31u;
        a = (a & 0x807FFFFFu) | (ea << 23);
        b = (b & 0x807FFFFFu) | (eb << 23);
    }
    *y = u2f(a); *x = u2f(b);
    if (!isfinite(*y) || !isfinite(*x)) return 0;
    if (fabs((double)*y) < fabs((double)*x) * 0x1p-100) return 0;
    return 1;
}

/* 1 and *a set if x is in the word's domain; *pair counts sincosf misfits. */
static int word(const char *fn, float x, float *a, long long *pair)
{
    float ax = fabsf(x);
    if (!strcmp(fn, "atan2f_pairs") || !strcmp(fn, "atan2f_grad")) {
        float yy, xx;
        if (!pair_of(f2u(x), fn[7] == 'g', &yy, &xx)) return 0;
        *a = atan2f(yy, xx);
        return 1;
    }
    if (isnan(x)) return 0;
    if (!strcmp(fn, "sinf")) { if (ax > 64) return 0; *a = sinf(x); }
    else if (!strcmp(fn, "cosf")) { if (ax > 64) return 0; *a = cosf(x); }
    else if (!strcmp(fn, "sincosf")) {
        float s, c;
        if (ax > 64) return 0;
        sincosf(x, &s, &c);
        *pair += (f2u(s) != f2u(sinf(x))) + (f2u(c) != f2u(cosf(x)));
        *a = s;
    }
    else if (!strcmp(fn, "tanf")) { if (ax > 64) return 0; *a = tanf(x); }
    else if (!strcmp(fn, "asinf")) { if (ax > 1) return 0; *a = asinf(x); }
    else if (!strcmp(fn, "acosf")) { if (ax > 1) return 0; *a = acosf(x); }
    else if (!strcmp(fn, "atanf")) { *a = atanf(x); }
    else if (!strcmp(fn, "atan2f_x1000")) { *a = atan2f(x, 1000.0f); }
    else if (!strcmp(fn, "atan2f_1000_y")) { *a = atan2f(1000.0f, x); }
    else if (!strcmp(fn, "atan2f_resolve_x1000")) { *a = atan2f(x, 1000.0f); }
    else if (!strcmp(fn, "expf")) { if (ax > 104) return 0; *a = expf(x); }
    else if (!strcmp(fn, "powf_m1.6364")) {
        if (!(x > 0)) return 0;
        *a = powf(x, -1.6364f);
    }
    else if (!strcmp(fn, "powf_0.5")) { if (!(x >= 0)) return 0; *a = powf(x, 0.5f); }
    else { fprintf(stderr, "unknown word %s\n", fn); exit(2); }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: libm_digest WORD [BUCKET OUT.bin]\n");
        return 2;
    }
    const char *fn = argv[1];
    if (argc == 4) {
        long long b = atoll(argv[2]);
        uint32_t *out = malloc(sizeof(uint32_t) << 23);
        long long pair = 0;
        for (long long k = 0; k < (1LL << 23); k++) {
            float a;
            out[k] = word(fn, u2f((uint32_t)((b << 23) + k)), &a, &pair)
                     ? f2u(a) : 0xFFFFFFFFu;
        }
        FILE *f = fopen(argv[3], "wb");
        fwrite(out, sizeof(uint32_t), 1u << 23, f);
        fclose(f);
        return 0;
    }
    static uint64_t bucket[512];
    uint64_t dig = 0;
    long long tot = 0, pair = 0;
#pragma omp parallel for reduction(+:dig, tot, pair) schedule(dynamic)
    for (long long b = 0; b < 512; b++) {
        uint64_t bd = 0;
        for (long long u = b << 23; u < (b + 1) << 23; u++) {
            float a;
            if (!word(fn, u2f((uint32_t)u), &a, &pair)) continue;
            uint64_t h = mix(((uint64_t)u << 32) | f2u(a));
            dig += h;
            bd += h;
            tot++;
        }
        bucket[b] = bd;
    }
    printf("%s %lld %016llx", fn, tot, (unsigned long long)dig);
    if (!strcmp(fn, "sincosf")) printf(" pair_differs=%lld", pair);
    printf(" buckets=");
    for (int b = 0; b < 512; b++)
        printf("%s%llx", b ? "," : "", (unsigned long long)bucket[b]);
    printf("\n");
    return 0;
}
