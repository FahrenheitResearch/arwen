/* The C library's float32 log10f and powf(x, y), evaluated over a
 * contiguous run of float32 bit patterns.  sweep_libm.py calls these
 * through ctypes and compares every word against WOOF's device routines.
 * These are the functions gfortran's ALOG10 and REAL**REAL call in the
 * Dudhia oracle (libm-report.txt lists the imports), resolved in the same
 * process type on the same CPU, so the ifunc selection is the oracle's. */
#include <math.h>
#include <stdint.h>
#include <string.h>

static float from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

void sweep_log10f(uint32_t first, uint64_t n, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i)
        out[i] = log10f(from_bits(first + (uint32_t)i));
}

void sweep_powf(uint32_t first, uint64_t n, float y, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i)
        out[i] = powf(from_bits(first + (uint32_t)i), y);
}
