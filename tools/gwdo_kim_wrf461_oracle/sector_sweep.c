/* CPU half of the gwd_opt = 1 wind-sector sweep (see sector_sweep.py).
 *
 * Built by build.sh as a shared library, -O0 -ffp-contract=off -fopenmp, so
 * every operation below is one IEEE float32 operation and every libm call is
 * this host's scalar atanf/atan2f (build.sh checks the object imports no
 * libmvec _ZGV symbol).
 *
 * cpu_atan2f_sector: for each (ubar, vbar) pair, this host's atan2f and the
 * sector bl_gwdo.F90:338-339 forms from it,
 *
 *     wdir = atan2(ubar(i),vbar(i)) + pi_
 *     idir = mod(nint(fdir*wdir),mdir) + 1          fdir = mdir/(2.0*pi_)
 *
 * with pi_ = 3.141592653 (module_pbl_driver.F's literal) and mdir = 8, in
 * float32 as gfortran evaluates them (NINT rounds half away from zero, as
 * lroundf does).  A non-finite angle gets sector 0.
 *
 * cpu_atanf_range: this host's atanf on the float32 bit patterns
 * start .. start + n - 1.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

void cpu_atan2f_sector(const float *ubar, const float *vbar, float *angle,
                       int32_t *sector, int64_t n)
{
    const float pi_ = 3.141592653f;
    const int mdir = 8;
    const float fdir = (float)mdir / (2.0f * pi_);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        float a = atan2f(ubar[i], vbar[i]);
        angle[i] = a;
        if (!isfinite(a)) { sector[i] = 0; continue; }
        float wdir = a + pi_;
        float prod = fdir * wdir;
        sector[i] = (int32_t)(lroundf(prod) % mdir) + 1;
    }
}

void cpu_atanf_range(uint32_t start, int64_t n, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits = start + (uint32_t)i;
        float x;
        memcpy(&x, &bits, sizeof x);
        out[i] = atanf(x);
    }
}
