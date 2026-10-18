// Control oracle only: gives the native WRF lake (lake_fortran_cr.so, linked
// -Bsymbolic) correctly rounded binary64 exp, log, log10, sin, atan and pow
// in place of the host library's, so WRF's own Fortran can be graded
// against WOOF with the library rounding difference removed.  The
// single-precision functions (powf and friends) stay the host library's.
#include <cstdlib>
#include "f64cr.cuh"
extern "C" {
double exp(double x) noexcept {return wcr_exp(x);}
double log(double x) noexcept {return wcr_log(x);}
double log10(double x) noexcept {return wcr_log10(x);}
double sin(double x) noexcept {if (wcr_fabs(x) >= 0x1p+20) std::abort(); return wcr_sin(x);}
double atan(double x) noexcept {return wcr_atan(x);}
double pow(double x, double y) noexcept {return wcr_pow(x, y);}
}
