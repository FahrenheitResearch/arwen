// WOOF's own libm words for the mp=28 units against the host's libm.
//
// Compiles gpuwm/core/kernels/thompson_aerosol_libm.cuh (on glibc_flt32.cuh
// and glibc_flt64.cuh) for the host behind
// tools/thompson_real_column_parity/cuda_host_shim.h and compares every
// returned word with the host C library the gfortran WRF oracle calls:
// expf, logf and log10f over all 2^32 float32 inputs, powf, exp, log, pow
// and log10 on N sampled arguments each (default 2e8), and hypot and log1p
// on N sampled arguments each.  atan2 is reported, not gated on agreement:
// WOOF's word is a different routine from the host's, so the check prints
// how often the two agree and the worst error of each against a 113-bit
// reference, and fails only if WOOF's word is worse than 1.5 ulp.
//
// build (on the oracle host, from the repository root):
//   g++ -O2 -ffp-contract=off -std=c++17 \
//     -Itools/thompson_real_column_parity -Igpuwm/core/kernels \
//     -o libm_check tools/thompson_aerosol_column_oracle/libm_check.cpp \
//     -lquadmath
//   ./libm_check [N]
// Exit status 0 means zero mismatches.
#include "cuda_host_shim.h"
#include "glibc_flt32.cuh"
#include "glibc_flt64.cuh"
#include "thompson_aerosol_libm.cuh"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <quadmath.h>

static inline unsigned fb(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static inline unsigned long long db(double d)
{ unsigned long long u; memcpy(&u, &d, 8); return u; }
static bool same32(float a, float b)
{ return fb(a) == fb(b) || (std::isnan(a) && std::isnan(b)); }
static bool same64(double a, double b)
{ return db(a) == db(b) || (std::isnan(a) && std::isnan(b)); }

int main(int argc, char **argv)
{
    long long N = argc > 1 ? atoll(argv[1]) : 200000000LL;
    long long bad[11] = {0};
    const char *name[11] = {"expf", "logf", "log10f", "powf", "exp", "log",
                            "pow", "log10", "log10-normal-band", "hypot",
                            "log1p"};
    long long atan2_differ = 0;
    double atan2_woof_ulp = 0.0, atan2_host_ulp = 0.0;
    for (unsigned long long u = 0; u < 0x100000000ULL; u++) {
        unsigned uu = (unsigned)u;
        float x;
        memcpy(&x, &uu, 4);
        if (!same32(thompson_aa_expf(x), expf(x))) bad[0]++;
        if (!same32(thompson_aa_logf(x), logf(x))) bad[1]++;
        if (!same32(thompson_aa_log10f(x), log10f(x))) bad[2]++;
    }
    std::mt19937_64 g(20261007);
    std::uniform_real_distribution<double> ue(-745.0, 709.0);
    for (long long i = 0; i < N; i++) {
        unsigned a = (unsigned)g(), b = (unsigned)g();
        float x, y;
        memcpy(&x, &a, 4);
        memcpy(&y, &b, 4);
        if (i & 1) {          // finite-result band: |y| < 64, x >= 0
            y = (float)std::ldexp((double)(b >> 8) / (1 << 24) * 2 - 1, 6);
            x = fabsf(x);
        }
        if (!same32(thompson_aa_powf(x, y), powf(x, y))) bad[3]++;
        unsigned long long c = g(), d = g();
        double X, Y;
        memcpy(&X, &c, 8);
        memcpy(&Y, &d, 8);
        if (i & 1) X = ue(g);
        if (!same64(thompson_aa_exp(X), exp(X))) bad[4]++;
        double Lx = (i & 1) ? fabs(X)
            : std::ldexp(1.0 + (double)(c >> 12) / (1ULL << 52),
                         (int)(d % 200) - 100);
        if (!same64(thompson_aa_log(Lx), log(Lx))) bad[5]++;
        if (!same64(thompson_aa_log10(Lx), log10(Lx))) bad[7]++;
        double Z;
        memcpy(&Z, &c, 8);
        Z = fabs(Z);
        if (!same64(thompson_aa_log10(Z), log10(Z))) bad[8]++;
        double PX = std::ldexp(1.0 + (double)(c >> 12) / (1ULL << 52),
                               (int)(d % 60) - 30);
        double PY = ((double)(d >> 11) / (1ULL << 53) * 2 - 1) * 16.0;
        if (i % 3 == 0) PY = (double)(int)PY;
        if (!same64(thompson_aa_pow(PX, PY), pow(PX, PY))) bad[6]++;
        // hypot and atan2 over seven decades of each argument, both signs;
        // log1p over (-0.5, 1) and over tiny magnitudes of both signs.
        const double HX = ((double)(c >> 11) / (1ULL << 53) * 2 - 1)
            * std::pow(10.0, (double)(d % 7000) / 1000.0 - 3.5);
        const double HY = ((double)(d >> 11) / (1ULL << 53) * 2 - 1)
            * std::pow(10.0, (double)(c % 7000) / 1000.0 - 3.5);
        if (!same64(thompson_aa_hypot(HX, HY), hypot(HX, HY))) bad[9]++;
        const double L1 = (i & 1)
            ? (double)(c >> 11) / (1ULL << 53) * 1.5 - 0.5
            : ((double)(d >> 11) / (1ULL << 53) * 2 - 1)
                * std::pow(10.0, -(double)(c % 16000) / 1000.0);
        if (!same64(thompson_aa_log1p(L1), log1p(L1))) bad[10]++;
        const double aw = thompson_aa_atan2(HY, HX), ah = atan2(HY, HX);
        if (!same64(aw, ah)) atan2_differ++;
        const __float128 aq = atan2q((__float128)HY, (__float128)HX);
        const double ulp = std::nextafter(std::fabs((double)aq), INFINITY)
            - std::fabs((double)aq);
        const double ew = (double)fabsq(((__float128)aw - aq) / ulp);
        const double eh = (double)fabsq(((__float128)ah - aq) / ulp);
        if (ew > atan2_woof_ulp) atan2_woof_ulp = ew;
        if (eh > atan2_host_ulp) atan2_host_ulp = eh;
    }
    long long total = 0;
    for (int k = 0; k < 11; k++) {
        printf("%-18s %s mismatches %lld\n", name[k],
               k < 3 ? "exhaustive 2^32" : "sampled", bad[k]);
        total += bad[k];
    }
    printf("atan2              sampled: differs from the host on %lld of %lld;"
           " worst error WOOF %.3f ulp, host %.3f ulp\n",
           atan2_differ, N, atan2_woof_ulp, atan2_host_ulp);
    printf("sampled N = %lld; total mismatches %lld\n", N, total);
    return (total || atan2_woof_ulp > 1.5) ? 1 : 0;
}
