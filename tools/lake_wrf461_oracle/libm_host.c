/* Host libm evaluator for libm_trace.py: the library the native WRF lake
   oracle links (gfortran -O0 calls these same entry points). */
#include <math.h>
void libm_host_eval(long n, const int* fn, const double* a, const double* b, double* out) {
    for (long i = 0; i < n; ++i) {
        switch (fn[i]) {
        case 0: out[i] = exp(a[i]); break;
        case 1: out[i] = log(a[i]); break;
        case 2: out[i] = log10(a[i]); break;
        case 3: out[i] = sin(a[i]); break;
        case 4: out[i] = atan(a[i]); break;
        case 5: out[i] = pow(a[i], b[i]); break;
        default: out[i] = NAN;
        }
    }
}
