/* gf_libm_probe.c -- read float32 bit patterns, print what glibc's
 * expf/logf/powf/tgammaf do.
 *
 * This is the negative-control ground truth for the Grell-Freitas float32
 * reference, the same role tools/noahmp_wrf461_oracle/libm_probe.c plays for
 * Noah-MP.  It deliberately calls each function through the normal ABI so
 * that glibc's own ifunc resolver picks whichever variant (FMA or not) the
 * host CPU gets, which is exactly the variant the gfortran-built GF oracle
 * calls.
 *
 * tgammaf is why this file exists at all.  module_cu_gf_deep.F:3854 (and
 * :3882, :3918, :3952) forms
 *
 *     fzu = gamma(alpha + beta) / (gamma(alpha) * gamma(beta))
 *
 * to normalise the beta-function updraft mass-flux shape, and `alpha` is a
 * runtime value -- tunning is clamped to [0.2, 0.9] and alpha =
 * (tunning*(beta-2)+1)/(1-tunning), so alpha ranges over roughly [1.075,
 * 3.7] with beta fixed at 1.3 for "UP".  There is nothing to constant-fold.
 * No prior gpuwm kernel has had to match tgammaf, so the port measures it
 * rather than assuming a CUDA/glibc agreement that has never been checked.
 *
 * stdin  : "e <hex32>"           -> expf(x)
 *          "l <hex32>"           -> logf(x)
 *          "g <hex32>"           -> tgammaf(x)
 *          "p <hex32> <hex32>"   -> powf(x, y)
 * stdout : one hex32 per input line.
 *
 * Build:  gcc -O2 gf_libm_probe.c -o gf_libm_probe -lm
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
        if (op == 'p') {
            if (scanf(" %x %x", &a, &b) != 2)
                return 1;
            x = asfloat(a);
            y = asfloat(b);
            r = powf(x, y);
            printf("%08x\n", asuint(r));
            continue;
        }
        if (scanf(" %x", &a) != 1)
            return 1;
        x = asfloat(a);
        switch (op) {
        case 'e': r = expf(x); break;
        case 'l': r = logf(x); break;
        case 'g': r = tgammaf(x); break;
        default:  return 2;
        }
        printf("%08x\n", asuint(r));
    }
    return 0;
}
