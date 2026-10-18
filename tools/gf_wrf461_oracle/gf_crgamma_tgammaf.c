/* A correctly rounded tgammaf, linked INTO the regime-column oracle in place
 * of the C library's.
 *
 * Why.  WOOF's Grell-Freitas gamma is correctly rounded on purpose and the C
 * library tgammaf that gfortran binds WRF's gamma() intrinsic to is not
 * (docs/gf_gamma_known_delta.md: wrong on 39.44 per cent of [0.25, 36],
 * worst 6 ULP).  That one routine is a DOCUMENTED DELIBERATE DIVERGENCE, and
 * one ULP of the beta-shape normaliser fzu moves the deep mass flux by per
 * cent.  So a bitwise comparison against WRF needs the two programs to agree
 * on gamma, and there are only two ways to get that: feed WRF's fzu back
 * into the kernel (the fzu_override pin the 216-column suites use, which
 * needs a stage capture of fzu for every column), or give WRF's own compiled
 * Fortran the correctly rounded gamma.  This file is the second.  Everything
 * else in the build is the byte-pinned WRF objects and the system libm, so
 * the comparison grades every other line of GFDRV, unpinned, against the
 * kernel's SHIPPED forecast path (fzu computed, override slots 0).
 *
 * The value is libquadmath's 113-bit tgammaq rounded once to float32, the
 * same definition as gf_crgamma_dump.c, which writes the committed fixture
 * gf-crgamma-tgammaf.csv that WOOF's gamma is graded against at max_ulp 0.
 * build_regime.sh checks this symbol against that fixture before using it.
 * The correctness argument holds on [0.25, 36] (the interval the fixture
 * proves); an argument outside it aborts the oracle rather than being
 * silently trusted.
 *
 * It is a strong definition in the executable, so the static linker binds
 * the Fortran objects' undefined tgammaf to it ahead of the shared libm;
 * build_regime.sh proves the binding with nm.
 */
#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long gf_crgamma_calls = 0;

float tgammaf(float x)
{
    if (!(x >= 0.25f && x <= 36.0f)) {
        fprintf(stderr, "gf_crgamma_tgammaf: argument %a outside [0.25, 36]\n",
                (double)x);
        abort();
    }
    gf_crgamma_calls++;
    return (float)tgammaq((__float128)x);   /* one rounding */
}

/* Reported by the oracle driver at exit so a build where the binding did not
 * take (zero calls on a fixture that reaches the PDF) is visible. */
unsigned long gf_crgamma_call_count(void) { return gf_crgamma_calls; }

__attribute__((destructor)) static void gf_crgamma_report(void)
{
    fprintf(stderr, "gf_crgamma_tgammaf: %lu calls\n", gf_crgamma_calls);
}
