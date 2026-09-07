/* Emit the correctly rounded fzu for every (alpha, beta) the committed GF
 * fixture reaches, so the deliberate gamma divergence is a checkable
 * artifact rather than a claim in prose.  See docs/gf_gamma_known_delta.md.
 *
 * fzu is spelled EXACTLY as gfd_get_zu_zd_pdf (gf.cu) and
 * get_zu_zd_pdf_fim (gpuwm/verify/gf_deep_ref.py) spell it --
 *     gamma(alpha+beta) / (gamma(alpha) * gamma(beta))
 * -- with one float32 rounding per operation.  Each gamma is libquadmath's
 * 113-bit tgammaq rounded once to float32, i.e. the correctly rounded value,
 * which is what gfk_tgamma now returns (verified exhaustively over the whole
 * [0.25, 36] reachable interval).
 *
 * usage: gf_crgamma_fzu PAIRSFILE
 *   PAIRSFILE: "alphaword betaword" (lowercase hex float32) per line
 * output: "alphaword,betaword,fzuword\n"
 *
 * build: gcc -O2 -ffp-contract=off -o gf_crgamma_fzu gf_crgamma_fzu.c -lm -lquadmath
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <quadmath.h>

static float w2f(uint32_t w) { float f; memcpy(&f, &w, 4); return f; }
static uint32_t f2w(float f) { uint32_t w; memcpy(&w, &f, 4); return w; }
static float crgamma(float x) { return (float)tgammaq((__float128)x); }

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s PAIRSFILE\n", argv[0]); return 2; }
    FILE *in = fopen(argv[1], "r");
    if (!in) { perror(argv[1]); return 2; }
    uint32_t aw, bw;
    while (fscanf(in, "%x %x", &aw, &bw) == 2) {
        float a = w2f(aw), b = w2f(bw);
        float fzu = crgamma(a + b) / (crgamma(a) * crgamma(b));
        printf("%08x,%08x,%08x\n", aw, bw, f2w(fzu));
    }
    fclose(in);
    return 0;
}
