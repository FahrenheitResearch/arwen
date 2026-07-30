/* libm_probe_radiation.c -- ground truth for gpuwm/core/noahmp_libm.py.
 *
 * Owned by the `radiation` lane.  The lane needs logf, which the shared
 * libm_probe.c does not offer; rather than edit a file another lane owns,
 * this probe covers all three entry points the radiation leaves reach.
 *
 * Calls logf/expf/powf through the normal ABI so glibc's own ifunc resolver
 * picks whichever variant (FMA or not) the host CPU gets -- exactly the
 * variant the gfortran-built Noah-MP oracle calls.
 *
 * stdin  : "l <hex32>"           -> logf(x)
 *          "e <hex32>"           -> expf(x)
 *          "p <hex32> <hex32>"   -> powf(x, y)
 * stdout : one hex32 per input line.
 *
 * Build:  gcc -O2 libm_probe_radiation.c -o libm_probe_radiation -lm
 * Never build with -ffast-math / -ffinite-math-only / -fno-math-errno tricks
 * that let GCC fold or substitute the calls: that would stop measuring glibc.
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
    /* volatile stops GCC hoisting or folding across iterations. */
    volatile float x, y, r;

    while (scanf(" %c", &op) == 1) {
        if (op == 'l') {
            if (scanf(" %x", &a) != 1)
                return 1;
            x = asfloat(a);
            r = logf(x);
            printf("%08x\n", asuint(r));
        } else if (op == 'e') {
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
