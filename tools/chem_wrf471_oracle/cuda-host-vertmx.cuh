// CPU-only syntax/arithmetic check for the unchanged CUDA bodies.
// This does not establish NVRTC compilation or device parity.
#include <cmath>
#include <cstdint>
#include <cstring>
#define __device__
#define __global__
struct host_dim { int x=0; };
host_dim blockIdx,threadIdx;
host_dim blockDim{1};
inline float __fadd_rn(float a,float b){ volatile float r=a+b;return r; }
inline float __fsub_rn(float a,float b){ volatile float r=a-b;return r; }
inline float __fmul_rn(float a,float b){ volatile float r=a*b;return r; }
inline float __fdiv_rn(float a,float b){ volatile float r=a/b;return r; }
inline double __dadd_rn(double a,double b){ volatile double r=a+b;return r; }
inline double __dsub_rn(double a,double b){ volatile double r=a-b;return r; }
inline double __dmul_rn(double a,double b){ volatile double r=a*b;return r; }
inline double __ddiv_rn(double a,double b){ volatile double r=a/b;return r; }
inline float __double2float_rn(double a){volatile float r=a;return r;}
inline float __fmaf_rn(float a,float b,float c){return std::fma(a,b,c);}
inline float __fsqrt_rn(float a){return std::sqrt(a);}
inline unsigned __float_as_uint(float a){unsigned r;std::memcpy(&r,&a,4);return r;}
inline float __uint_as_float(unsigned a){float r;std::memcpy(&r,&a,4);return r;}
inline float __int_as_float(int a){return __uint_as_float(static_cast<unsigned>(a));}
inline long long __double_as_longlong(double a){long long r;std::memcpy(&r,&a,8);return r;}
inline double __longlong_as_double(long long a){double r;std::memcpy(&r,&a,8);return r;}
extern "C" void host_column(int c){threadIdx.x=c;}
