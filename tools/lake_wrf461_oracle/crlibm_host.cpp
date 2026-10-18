// Host evaluator of WOOF's correctly rounded binary64 functions for
// cr_check.py (same function ids as libm_host.c).
#include "f64cr.cuh"
extern "C" void lake_crlibm_host_eval(long n, const int* fn, const double* a, const double* b, double* out) {
    for (long i = 0; i < n; ++i) {
        switch (fn[i]) {
        case 0: out[i] = wcr_exp(a[i]); break;
        case 1: out[i] = wcr_log(a[i]); break;
        case 2: out[i] = wcr_log10(a[i]); break;
        case 3: out[i] = wcr_sin(a[i]); break;
        case 4: out[i] = wcr_atan(a[i]); break;
        case 5: out[i] = wcr_pow(a[i], b[i]); break;
        default: out[i] = 0.0 / 0.0;
        }
    }
}
extern "C" unsigned long long lake_crlibm_ambiguous() {return wcr_ambiguous_count;}
