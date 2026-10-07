// CPU-only audit of the CUDA source. This is not a device execution receipt.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#define __device__
#define __global__
using std::min;
using std::max;
using std::isfinite;
struct plume_dim { unsigned int x; };
static plume_dim blockIdx{0},threadIdx{0},blockDim{1};
static inline float __fadd_rn(float a,float b) { volatile float x=a+b; return x; }
static inline float __fsub_rn(float a,float b) { volatile float x=a-b; return x; }
static inline float __fmul_rn(float a,float b) { volatile float x=a*b; return x; }
static inline float __fdiv_rn(float a,float b) { volatile float x=a/b; return x; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline float __fmaf_rn(float a,float b,float c) { return std::fma(a,b,c); }
static inline double __dadd_rn(double a,double b) { volatile double x=a+b; return x; }
static inline double __dsub_rn(double a,double b) { volatile double x=a-b; return x; }
static inline double __dmul_rn(double a,double b) { volatile double x=a*b; return x; }
static inline double __ddiv_rn(double a,double b) { volatile double x=a/b; return x; }
static inline float __double2float_rn(double a) { volatile float x=(float)a; return x; }
static inline float __uint_as_float(uint32_t a) { float x; memcpy(&x,&a,4); return x; }
static inline float __int_as_float(int32_t a) { float x; memcpy(&x,&a,4); return x; }
static inline uint32_t __float_as_uint(float a) { uint32_t x; memcpy(&x,&a,4); return x; }
static inline double __longlong_as_double(int64_t a) { double x; memcpy(&x,&a,8); return x; }
static inline int64_t __double_as_longlong(double a) { int64_t x; memcpy(&x,&a,8); return x; }
