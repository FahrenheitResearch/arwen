// CUDA language shim for compiling gpuwm's Thompson kernel translation units
// as plain C++ on the host, so the exact source string nvrtc receives can be
// run on a CPU with no GPU anywhere.
//
// The pattern is tools/gf_wrf461_oracle/gf_host_harness.cpp's, generalised:
// host_backend.py prepends this header to gpuwm.core.kernels.module_source(),
// appends one generated launcher per extern "C" __global__ kernel, and builds
// the result with -ffp-contract=off and without -ffast-math.
//
// What the host build is and is not.  x86-64 SSE evaluates float and double
// with IEEE-754 round-to-nearest per operation, so every __fadd_rn/__fmul_rn
// pin below is the operation the device intrinsic names, and with contraction
// off no unpinned a*b+c is fused either.  The device differs in three known
// classes that this build does not see: nvrtc's default --fmad=true fuses
// unpinned multiply-adds, CUDA's powf/expf/logf are not glibc's (the host
// calls glibc's, the same libm the gfortran WRF oracle calls), and sm_120
// flushes some FP32 subnormals.  So the host run grades the TRANSCRIPTION
// against WRF; the device toolchain is graded by the GPU gates.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstddef>

#define __device__
#define __global__
#define __host__
#define __constant__ static const
#define __forceinline__ inline
#define __noinline__
#define __restrict__

struct HostUint3 { unsigned x, y, z; };
static HostUint3 threadIdx = {0u, 0u, 0u};
static HostUint3 blockIdx = {0u, 0u, 0u};
static HostUint3 blockDim = {1u, 1u, 1u};
static HostUint3 gridDim = {1u, 1u, 1u};

using std::isfinite;
using std::isnan;
using std::isinf;

// CUDA's integer and floating min/max overloads.
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
static inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
static inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
static inline long long min(long long a, long long b) { return a < b ? a : b; }
static inline long long max(long long a, long long b) { return a > b ? a : b; }
static inline float min(float a, float b) { return fminf(a, b); }
static inline float max(float a, float b) { return fmaxf(a, b); }
static inline double min(double a, double b) { return fmin(a, b); }
static inline double max(double a, double b) { return fmax(a, b); }

// Rounded-operation intrinsics.  With -ffp-contract=off each is one IEEE op.
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline float __fmaf_rn(float a, float b, float c) { return fmaf(a, b, c); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __fma_rn(double a, double b, double c) { return fma(a, b, c); }
static inline float __double2float_rn(double d) { return (float)d; }
// __float2int_rn rounds half to EVEN (the current, default rounding mode).
static inline int __float2int_rn(float x) { return (int)nearbyintf(x); }
static inline int __float2int_rz(float x) { return (int)x; }
static inline int __double2int_rn(double x) { return (int)nearbyint(x); }

static inline float __uint_as_float(unsigned int u)
{ float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f)
{ unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i)
{ float f; std::memcpy(&f, &i, 4); return f; }
static inline int __float_as_int(float f)
{ int i; std::memcpy(&i, &f, 4); return i; }
static inline long long __double_as_longlong(double d)
{ long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l)
{ double d; std::memcpy(&d, &l, 8); return d; }

// Optional host-only rate readback (tools/thompson_real_column_parity/
// instrument_port_rates.py).  The instrumented copy of a kernel calls
// HOST_RATE(slot, idx, value); the pristine source never names it.
#ifdef GPUWM_HOST_RATES
extern "C" {
double *gpuwm_host_rate_buffer = nullptr;
long long gpuwm_host_rate_cells = 0;
int gpuwm_host_rate_slots = 0;
void gpuwm_host_rate_bind(double *buffer, long long cells, int slots)
{ gpuwm_host_rate_buffer = buffer; gpuwm_host_rate_cells = cells;
  gpuwm_host_rate_slots = slots; }
}
static inline void HOST_RATE(int slot, long long idx, double value)
{
    if (gpuwm_host_rate_buffer != nullptr && slot >= 0
            && slot < gpuwm_host_rate_slots && idx >= 0
            && idx < gpuwm_host_rate_cells)
        gpuwm_host_rate_buffer[(long long)slot * gpuwm_host_rate_cells + idx]
            = value;
}
#endif
