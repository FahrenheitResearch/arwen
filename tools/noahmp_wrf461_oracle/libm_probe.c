/* libm_probe.c -- read float32 bit patterns, print what glibc's expf/powf do.
 *
 * This is the negative-control ground truth for gpuwm/core/noahmp_libm.py.
 * It deliberately calls expf/powf through the normal ABI so that glibc's own
 * ifunc resolver picks whichever variant (FMA or not) the host CPU gets, which
 * is exactly the variant the gfortran-built Noah-MP oracle calls.
 *
 * stdin  : "e <hex32>"        -> expf
 *          "p <hex32> <hex32>" -> powf(x, y)
 * stdout : one hex32 per input line.
 *
 * Build:  gcc -O2 libm_probe.c -o libm_probe -lm
 * Do not build with -ffast-math or -ffinite-math-only: that would let GCC
 * constant-fold or substitute the calls and stop measuring glibc.
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

static float asfloat(uint32_t u)
{
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

static uint32_t asuint(float f)
{
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

int main(void)
{
    char op;
    uint32_t a, b;
    /* volatile stops GCC from hoisting or folding across iterations. */
    volatile float x, y, r;

    while (scanf(" %c", &op) == 1) {
        if (op == 'e') {
            if (scanf(" %x", &a) != 1)
                return 1;
            x = asfloat(a);
            r = expf(x);
            printf("%08x\n", asuint(r));
        } else if (op == 'p') {
            if (scanf(" %x %x", &a, &b) != 2)
                return 1;
            x = asfloat(a);
            y = asfloat(b);
            r = powf(x, y);
            printf("%08x\n", asuint(r));
        } else {
            return 2;
        }
    }
    return 0;
}
