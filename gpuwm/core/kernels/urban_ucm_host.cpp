// Compiled CPU twin of the single-layer urban canopy CUDA implementation.
// The physics and its glibc float32 functions are included verbatim below.
// These shims provide CPU equivalents of CUDA's explicit rounding operations.
// Build with -fno-fast-math -ffp-contract=off; no CUDA toolkit is required.

#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#pragma STDC FENV_ACCESS ON

#define __device__
#define __constant__
#define __global__
#define __forceinline__ inline

static inline float __fadd_rn(float a, float b) { volatile float r = a + b; return r; }
static inline float __fsub_rn(float a, float b) { volatile float r = a - b; return r; }
static inline float __fmul_rn(float a, float b) { volatile float r = a * b; return r; }
static inline float __fdiv_rn(float a, float b) { volatile float r = a / b; return r; }
static inline float __fsqrt_rn(float a) { volatile float r = std::sqrt(a); return r; }
static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
static inline double __dadd_rn(double a, double b) { volatile double r = a + b; return r; }
static inline double __dsub_rn(double a, double b) { volatile double r = a - b; return r; }
static inline double __dmul_rn(double a, double b) { volatile double r = a * b; return r; }
static inline double __ddiv_rn(double a, double b) { volatile double r = a / b; return r; }
static inline float __double2float_rn(double a) { volatile float r = static_cast<float>(a); return r; }

template <typename To, typename From> static inline To ucm_host_bits(From from)
{
    static_assert(sizeof(To) == sizeof(From), "bit conversion size");
    To to;
    std::memcpy(&to, &from, sizeof(to));
    return to;
}
static inline float __uint_as_float(unsigned int x) { return ucm_host_bits<float>(x); }
static inline float __int_as_float(int x) { return ucm_host_bits<float>(x); }
static inline unsigned int __float_as_uint(float x) { return ucm_host_bits<unsigned int>(x); }
static inline double __longlong_as_double(long long x) { return ucm_host_bits<double>(x); }
static inline long long __double_as_longlong(double x) { return ucm_host_bits<long long>(x); }

struct UcmHostThread { std::size_t x; };
static thread_local UcmHostThread blockIdx{0}, blockDim{1}, threadIdx{0};
static inline unsigned int atomicOr(unsigned int* word, unsigned int mask)
{
    const unsigned int old = *word;
    *word |= mask;
    return old;
}

#include "glibc_flt32.cuh"
#include "urban_ucm.cu"

class UcmHostRounding
{
    int saved;
public:
    UcmHostRounding() : saved(std::fegetround()) { std::fesetround(FE_TONEAREST); }
    ~UcmHostRounding() { std::fesetround(saved); }
};

// Pointer-plane layouts are exactly the CUDA entry layouts. Each plane is a
// contiguous float32 array; UTYPE and JMONTH input planes are int32 instead.
// Pointer tables carry native addresses represented as unsigned 64-bit words.
extern "C" void ucm_host_columns(
    const unsigned long long* inputs, const unsigned long long* outputs,
    const unsigned long long* state, const float* table, const float* globals,
    const int* switches, int* codes, int n)
{
    UcmHostRounding rounding;
    for (int i = 0; i < n; ++i) {
        threadIdx.x = static_cast<std::size_t>(i);
        ucm_column_test(inputs, outputs, state, table, globals, switches, codes, n);
    }
}

extern "C" void ucm_host_noah_after_lsm(
    const int* urban, const int* utype, const float* fraction,
    const float* u1, const float* v1, const float* hour_angle,
    const unsigned long long* rural, const unsigned long long* fields,
    const unsigned long long* state, const float* table, const float* globals,
    const int* switches, float dt, int month, unsigned int* error, int ny, int nx)
{
    UcmHostRounding rounding;
    for (std::size_t i = 0; i < static_cast<std::size_t>(ny) * nx; ++i) {
        threadIdx.x = i;
        ucm_noah_after_lsm(urban, utype, fraction, u1, v1, hour_angle,
                           rural, fields, state, table, globals, switches,
                           dt, month, error, ny, nx);
    }
}

extern "C" void ucm_host_noahmp_after_lsm(
    const int* urban, const int* utype, const float* fraction,
    const float* u1, const float* v1, const float* temperature,
    const float* qv, const float* p_lower, const float* p_upper,
    const float* dz, const float* hour_angle, const unsigned long long* fields,
    const unsigned long long* state, const float* table, const float* globals,
    const int* switches, float dt, int month, unsigned int* error, int ny, int nx)
{
    UcmHostRounding rounding;
    for (std::size_t i = 0; i < static_cast<std::size_t>(ny) * nx; ++i) {
        threadIdx.x = i;
        ucm_noahmp_after_lsm(urban, utype, fraction, u1, v1, temperature, qv,
                             p_lower, p_upper, dz, hour_angle, fields, state,
                             table, globals, switches, dt, month, error, ny, nx);
    }
}

extern "C" void ucm_host_overrides(
    const int* urban, const float* fraction, const unsigned long long* fields,
    const unsigned long long* state, const float* fveg, const float* t2_veg,
    const float* t2_bare, const float* q2_veg, const float* q2_bare,
    int noahmp, int ny, int nx)
{
    UcmHostRounding rounding;
    for (std::size_t i = 0; i < static_cast<std::size_t>(ny) * nx; ++i) {
        threadIdx.x = i;
        ucm_overrides(urban, fraction, fields, state, fveg, t2_veg, t2_bare,
                       q2_veg, q2_bare, noahmp, ny, nx);
    }
}
