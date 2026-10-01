#ifndef G64_HOST_SHIM
#define G64_HOST_SHIM
#include <stdint.h>
#include <math.h>
#include <string.h>
#define __device__
#define __forceinline__ inline
static inline double __dadd_rn(double a,double b){return a+b;}
static inline double __dsub_rn(double a,double b){return a-b;}
static inline double __dmul_rn(double a,double b){return a*b;}
static inline double __ddiv_rn(double a,double b){return a/b;}
static inline double __fma_rn(double a,double b,double c){return fma(a,b,c);}
static inline double __dsqrt_rn(double x){return sqrt(x);}
static inline double __longlong_as_double(uint64_t u){double x;memcpy(&x,&u,8);return x;}
static inline int __clzll(uint64_t x){return __builtin_clzll(x);}
#endif
static inline uint64_t __umul64hi(uint64_t a,uint64_t b){return (uint64_t)(((unsigned __int128)a*b)>>64);}
