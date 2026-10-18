/* Find points where WRF v4.6.1 calc_coszen's float32 sum rounds past one
 * (so its min(max(coszen,-1.0),1.0) clamp changes the word).
 *
 *   clamp_search CALENDAR.txt > hits.txt
 *
 * CALENDAR.txt: lines "step julian xtime gmt" (float32 values as %a or
 * decimal).  For each step the search walks float32 latitudes around the
 * subsolar latitude and longitudes around local noon and prints every
 * (step, xlat, xlon, coszen) with coszen > 1, using calc_coszen's
 * statement order, single-rounded float arithmetic (build with -O0
 * -ffp-contract=off) and the C library's sinf/cosf.  The declination uses
 * asinl rounded once (the correctly rounded asin), and a step is skipped
 * when the C library's asinf gives a different word, so a hit does not
 * depend on which asinf the reference runs. */
#include <math.h>
#include <stdio.h>
#include <string.h>

static float nextf(float x, int k) {
    unsigned int u; memcpy(&u, &x, 4);
    if (x >= 0) u += k; else u -= k;
    memcpy(&x, &u, 4); return x;
}

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "r");
    if (!f) return 2;
    const float degrad = 3.1415926535897932384626433f / 180.f;
    const float dpd = 360.f / 365.f;
    int step; float julian, xtime, gmt;
    while (fscanf(f, "%d %a %a %a", &step, &julian, &xtime, &gmt) == 4) {
        volatile float obecl = 23.5f * degrad;
        float sinob = sinf(obecl);
        float sxlong = (julian >= 80.f) ? dpd * (julian - 80.f)
                                        : dpd * (julian + 285.f);
        sxlong = sxlong * degrad;
        volatile float arg = sinob * sinf(sxlong);
        float declin = (float)asinl((long double)arg);
        if (declin != asinf(arg)) continue;
        volatile float da = 6.2831853071795862f * (julian - 1.f);
        da = da / 365.f;
        volatile float two = 2.f * da;
        volatile float eot = 0.000075f + 0.001868f * cosf(da);
        eot = eot - 0.032077f * sinf(da);
        eot = eot - 0.014615f * cosf(two);
        eot = eot - 0.04089f * sinf(two);
        eot = eot * 229.18f;
        volatile float xt24 = fmodf(xtime, 1440.f) + eot;
        float lat0 = declin / degrad;
        float lon0 = 15.f * (12.f - gmt - xt24 / 60.f);
        if (lon0 < -180.f) lon0 += 360.f;
        if (lon0 > 180.f) lon0 -= 360.f;
        int hits = 0;
        for (int i = -400; i <= 400 && hits < 4; i++) {
            float xlat = nextf(lat0, i);
            for (int j = -40; j <= 40 && hits < 4; j++) {
                float xlon = nextf(lon0, j * 64);
                volatile float t1 = xt24 / 60.f;
                volatile float t2 = xlon / 15.f;
                volatile float tloctm = gmt + t1;
                tloctm = tloctm + t2;
                volatile float hrang = 15.f * (tloctm - 12.f);
                hrang = hrang * degrad;
                volatile float xxlat = xlat * degrad;
                volatile float p1 = sinf(xxlat) * sinf(declin);
                volatile float p2 = cosf(xxlat) * cosf(declin);
                p2 = p2 * cosf(hrang);
                volatile float cz = p1 + p2;
                if (cz > 1.f) {
                    printf("%d %a %a %a\n", step, xlat, xlon, cz);
                    hits++;
                }
            }
        }
    }
    return 0;
}
