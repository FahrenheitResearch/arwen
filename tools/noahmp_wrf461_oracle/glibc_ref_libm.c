/* glibc_ref_libm.c -- ground truth for gpuwm/core/noahmp_libm.py.
 *
 * Reads raw little-endian uint32 float32 bit patterns from a file and writes the
 * raw uint32 bit patterns of glibc's answer to another file.  The calls go
 * through the ordinary ABI so glibc's own ifunc resolver picks whichever
 * variant (FMA or SSE2) the host CPU gets -- which is exactly the variant the
 * gfortran-built Noah-MP oracle calls.
 *
 * usage:  glibc_ref_libm <logf|log10f|atanf|expf|powf> <in.bin> <out.bin>
 *         powf consumes two uint32 per result (x then y).
 *
 * build:  gcc -O2 -o glibc_ref_libm glibc_ref_libm.c -lm
 * Do NOT build with -ffast-math / -ffinite-math-only / -O3 -march=native:
 * constant folding or a vectorised libm substitution would stop this measuring
 * scalar glibc.  The `volatile` below blocks folding of the call itself.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static float asfloat(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint32_t asuint(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

#define CHUNK 65536

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <logf|log10f|atanf|expf|powf> <in.bin> <out.bin>\n", argv[0]);
        return 2;
    }
    const char *fn = argv[1];
    int binary_op = (strcmp(fn, "powf") == 0);

    FILE *fi = fopen(argv[2], "rb");
    if (!fi) { perror("open in"); return 3; }
    FILE *fo = fopen(argv[3], "wb");
    if (!fo) { perror("open out"); return 4; }

    static uint32_t in[2 * CHUNK];
    static uint32_t out[CHUNK];
    size_t per = binary_op ? 2u : 1u;
    size_t n;

    while ((n = fread(in, sizeof(uint32_t), per * CHUNK, fi)) > 0) {
        size_t m = n / per;
        for (size_t i = 0; i < m; i++) {
            volatile float x = asfloat(in[per * i]);
            volatile float r;
            if (binary_op) {
                volatile float y = asfloat(in[per * i + 1]);
                r = powf(x, y);
            } else if (strcmp(fn, "logf") == 0) {
                r = logf(x);
            } else if (strcmp(fn, "log10f") == 0) {
                r = log10f(x);
            } else if (strcmp(fn, "atanf") == 0) {
                r = atanf(x);
            } else if (strcmp(fn, "expf") == 0) {
                r = expf(x);
            } else {
                fprintf(stderr, "unknown function %s\n", fn);
                return 5;
            }
            out[i] = asuint(r);
        }
        if (fwrite(out, sizeof(uint32_t), m, fo) != m) { perror("write"); return 6; }
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
