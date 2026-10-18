/* Check the interposed tgammaf (gf_crgamma_tgammaf.c) against the committed
 * correctly-rounded fixture gf-crgamma-tgammaf.csv, word for word, THROUGH
 * the same link the oracle uses: this file declares tgammaf extern and is
 * linked with gf_crgamma_tgammaf.o ahead of -lm, exactly like the Fortran.
 *
 * usage: gf_crgamma_check gf-crgamma-tgammaf.csv
 * prints "checked N, mismatched M"; exit status 1 if M > 0 or N == 0.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float tgammaf(float);

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s FIXTURE.csv\n", argv[0]); return 2; }
    FILE *in = fopen(argv[1], "r");
    if (!in) { perror(argv[1]); return 2; }
    char line[128];
    unsigned long n = 0, bad = 0;
    while (fgets(line, sizeof line, in)) {
        char *comma = strchr(line, ',');
        if (!comma) continue;
        char *end = NULL;
        uint32_t a = (uint32_t)strtoul(line, &end, 16);
        if (end != comma) continue;           /* header or malformed */
        uint32_t want = (uint32_t)strtoul(comma + 1, NULL, 16);
        float x, y;
        memcpy(&x, &a, 4);
        if (!(x >= 0.25f && x <= 36.0f)) continue;
        y = tgammaf(x);
        uint32_t got;
        memcpy(&got, &y, 4);
        n++;
        if (got != want) {
            if (bad < 5)
                fprintf(stderr, "arg %08x: got %08x want %08x\n", a, got, want);
            bad++;
        }
    }
    fclose(in);
    printf("checked %lu, mismatched %lu\n", n, bad);
    return (bad || n == 0) ? 1 : 0;
}
