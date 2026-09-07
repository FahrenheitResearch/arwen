/* Emit the CORRECTLY ROUNDED float32 gamma for the GF gamma fixture.
 *
 * Why this exists.  Through ArWen 2.6.5 gfk_tgamma was a transcription of
 * glibc's LGPL-2.1-or-later e_gammaf_r.c / gamma_productf.c and was graded
 * against gf-libm-tgammaf.csv, a recording of what glibc 2.39 returns.  That
 * code is gone (licence: an Apache-2.0 distribution cannot carry it) and the
 * kernel now computes the correctly rounded value instead.  Grading against
 * "what one binary returned" no longer describes the contract, so this tool
 * writes the oracle that does: the mathematically correct answer.
 *
 * The oracle is libquadmath's 113-bit tgammaq, rounded ONCE to float32.
 * 113 bits is 89 bits past float32's 24, so a wrong rounding would need
 * tgammaq to be off by 2^89 of its own ULP.  It is unrelated to glibc's
 * float32 tgammaf: different algorithm, different precision, different file.
 *
 * usage: gf_crgamma_dump ARGSFILE
 *   ARGSFILE: one lowercase hex float32 word per line
 * output: "xxxxxxxx,yyyyyyyy\n" per line (arg word, correctly rounded word)
 *
 * build: gcc -O2 -o gf_crgamma_dump gf_crgamma_dump.c -lm -lquadmath
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <quadmath.h>

static float word_to_float(uint32_t w) { float f; memcpy(&f, &w, 4); return f; }
static uint32_t float_to_word(float f) { uint32_t w; memcpy(&w, &f, 4); return w; }

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s ARGSFILE\n", argv[0]); return 2; }
    FILE *in = fopen(argv[1], "r");
    if (!in) { perror(argv[1]); return 2; }
    char line[64];
    while (fgets(line, sizeof line, in)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        uint32_t w = (uint32_t)strtoul(line, NULL, 16);
        __float128 q = tgammaq((__float128)word_to_float(w));
        printf("%08x,%08x\n", w, float_to_word((float)q));   /* one rounding */
    }
    fclose(in);
    return 0;
}
