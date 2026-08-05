// Host harness: compile gpuwm/core/kernels/gf_deep.cu as plain C++ and run
// it column by column, so the EXACT source that will run on the GPU can be
// graded against the committed oracle fixtures without a GPU.
//
// Fidelity notes.  x86-64 SSE float/double arithmetic is IEEE-754
// round-to-nearest per operation (FLT_EVAL_METHOD == 0); built with
// -ffp-contract=off and without -ffast-math there is no fusing and no
// reassociation, so every __fadd_rn/__fmul_rn/... below maps to the same
// correctly rounded operation the device intrinsic pins.  The one
// difference class left is subnormal handling: the host preserves
// subnormals (MXCSR clear), sm_120 flushes them in FP32 arithmetic -- which
// is exactly what the on-node gate exists to measure.
//
// The CUDA builtins used by the probe kernels (tgammaf/powf negative
// controls) become glibc's own functions here, so those slots are equal by
// construction on the host and are only meaningful on the device.

#include <cstdint>
#include <cstring>
#include <cmath>

// ---- CUDA language shims -------------------------------------------------
#define __device__
#define __global__
#define __constant__ static const
#define __forceinline__ inline
#define __restrict__

struct GfS3 { unsigned x, y, z; };
static GfS3 blockIdx = {0, 0, 0};
static GfS3 blockDim = {1, 1, 1};
static GfS3 threadIdx = {0, 0, 0};

static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }

static inline float __uint_as_float(unsigned int u)
{ float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f)
{ unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i)
{ float f; std::memcpy(&f, &i, 4); return f; }
static inline long long __double_as_longlong(double d)
{ long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l)
{ double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }

// ---- the kernel source, verbatim -----------------------------------------
#include "common.cuh"
#include "gf.cu"

// ---- launchers -----------------------------------------------------------
extern "C" void host_gf_deep_stage(const float *lvin, const float *scin,
                                   const int *iin, float *lev, float *sca,
                                   int *isca, int n, int nz)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int col = 0; col < n; col++) {
        blockIdx.x = (unsigned)col;
        gf_deep_stage(lvin, scin, iin, lev, sca, isca, n, nz);
    }
}

extern "C" void host_gf_libm_unary(const float *x, float *out, int n)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < n; i++) {
        blockIdx.x = (unsigned)i;
        gf_libm_unary_probe(x, out, n);
    }
}

extern "C" void host_gf_libm_pow(const float *x, const float *y, float *out,
                                 int n)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < n; i++) {
        blockIdx.x = (unsigned)i;
        gf_libm_pow_probe(x, y, out, n);
    }
}

extern "C" void host_gf_fzu(const float *a, const float *b, float *out, int n)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < n; i++) {
        blockIdx.x = (unsigned)i;
        gf_fzu_probe(a, b, out, n);
    }
}

extern "C" void host_gf_const_dump(unsigned int *out)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < GF_NCONST; i++) {
        blockIdx.x = (unsigned)i;
        gf_deep_const_dump(out);
    }
}

extern "C" void host_gf_shallow_stage(const float *lvin, const float *scin,
                                      const int *iin, float *lev, float *sca,
                                      int *isca, int k22_wrf_faithful,
                                      int n, int nz)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int col = 0; col < n; col++) {
        blockIdx.x = (unsigned)col;
        gf_shallow_stage(lvin, scin, iin, lev, sca, isca, k22_wrf_faithful,
                         n, nz);
    }
}

extern "C" void host_gf_gfdrv_stage(const float *lvin, const float *scin,
                                    const int *iin, float *lev, float *sca,
                                    int *isca, int k22_wrf_faithful,
                                    int n, int nz)
{
    blockDim.x = 1; threadIdx.x = 0;
    for (int col = 0; col < n; col++) {
        blockIdx.x = (unsigned)col;
        gf_gfdrv_stage(lvin, scin, iin, lev, sca, isca, k22_wrf_faithful,
                       n, nz);
    }
}
