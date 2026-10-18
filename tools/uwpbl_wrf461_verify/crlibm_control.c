/* Positive control for the crlibm variant's symbol interposition.
 *
 * Linked exactly as run_camuwpbl_cr is (shim + CORE-MATH objects ahead of
 * -lm), a call to cos/acos from this program must return the correctly
 * rounded result, while the same function fetched from libm.so.6 by dlsym
 * must still be the host libm's.  The inputs are arguments where the host
 * libm's cos or acos is known to round differently from the correctly
 * rounded result (tools/uwpbl_wrf471_oracle/libm64/cpu-cos.log and
 * cpu-acos.log, first entries); if the host libm happens to round one of
 * them correctly the control still requires at least one to differ, or it
 * proves nothing.  Exit 0 = interposition active and observable.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

double cos(double);
double acos(double);
double cr_cos(double);
double cr_acos(double);

static double from_bits(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

int main(void)
{
    static volatile uint64_t cos_in[] = {
        0x4003166123b29989ULL, 0x3ff6d029329177caULL, 0x3fec6cf0ef69d330ULL,
        0x3fe4a09df3941e64ULL, 0xbfe1d82006a3b0a8ULL, 0x3fec646e6f2c4618ULL};
    double (*libm_cos)(double), (*libm_acos)(double);
    void *h = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen libm.so.6 failed\n"); return 3; }
    libm_cos = (double (*)(double))dlsym(h, "cos");
    libm_acos = (double (*)(double))dlsym(h, "acos");
    int differ = 0, bad = 0;
    for (unsigned i = 0; i < sizeof cos_in / sizeof cos_in[0]; i++) {
        double x = from_bits(cos_in[i]);
        double a = cos(x), c = cr_cos(x), l = libm_cos(x);
        printf("cos  x=%016llx linked=%016llx cr=%016llx libm=%016llx\n",
               (unsigned long long)bits(x), (unsigned long long)bits(a),
               (unsigned long long)bits(c), (unsigned long long)bits(l));
        if (bits(a) != bits(c)) bad++;
        if (bits(l) != bits(c)) differ++;
    }
    /* acos: scan a fine grid for host-vs-CR disagreements */
    int adiffer = 0;
    for (int i = 0; i < 2000000; i++) {
        double x = -1.0 + 2.0 * (i + 0.5) / 2000000.0;
        double a = acos(x), c = cr_acos(x), l = libm_acos(x);
        if (bits(a) != bits(c)) bad++;
        if (bits(l) != bits(c)) adiffer++;
    }
    printf("cos: %d of 6 known inputs differ host vs CR; acos grid: %d of 2000000 differ host vs CR; interposition failures: %d\n",
           differ, adiffer, bad);
    if (bad) return 1;
    if (differ == 0 && adiffer == 0) return 2;
    return 0;
}
