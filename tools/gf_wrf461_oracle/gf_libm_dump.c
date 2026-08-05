/* Dump live-glibc float32 words for the GF libm fixture.
 *
 * usage: gf_libm_dump FN ARGSFILE
 *   FN in {tgammaf, lgammaf, expm1f, exp2f, expf, logf}
 *   ARGSFILE: one lowercase hex float32 word per line
 * output: "xxxxxxxx,yyyyyyyy\n" per line (arg word, result word)
 *
 * Everything goes through volatile so the compiler cannot fold a call;
 * the words printed are what the shared libm.so.6 returned at run time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static float word_to_float(uint32_t w) { float f; memcpy(&f, &w, 4); return f; }
static uint32_t float_to_word(float f) { uint32_t w; memcpy(&w, &f, 4); return w; }

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: %s FN ARGSFILE\n", argv[0]); return 2; }
    const char *fn = argv[1];
    FILE *in = fopen(argv[2], "r");
    if (!in) { perror(argv[2]); return 2; }
    char line[64];
    while (fgets(line, sizeof line, in)) {
        uint32_t w = (uint32_t)strtoul(line, NULL, 16);
        volatile float x = word_to_float(w);
        volatile float y;
        if      (!strcmp(fn, "tgammaf")) y = tgammaf(x);
        else if (!strcmp(fn, "lgammaf")) y = lgammaf(x);
        else if (!strcmp(fn, "expm1f"))  y = expm1f(x);
        else if (!strcmp(fn, "exp2f"))   y = exp2f(x);
        else if (!strcmp(fn, "expf"))    y = expf(x);
        else if (!strcmp(fn, "logf"))    y = logf(x);
        else { fprintf(stderr, "unknown fn %s\n", fn); return 2; }
        printf("%08x,%08x\n", w, float_to_word((float)y));
    }
    fclose(in);
    return 0;
}
