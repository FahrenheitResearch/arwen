#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#define __device__
inline uint32_t __float_as_uint(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
inline float __uint_as_float(uint32_t u) { float x; std::memcpy(&x,&u,4); return x; }
inline long long __double_as_longlong(double x) { long long u; std::memcpy(&u,&x,8); return u; }
inline double __longlong_as_double(long long u) { double x; std::memcpy(&x,&u,8); return x; }
inline float __double2float_rn(double x) { return (float)x; }
inline long long __double2ll_rn(double x) { return (long long)std::nearbyint(x); }
inline uint64_t __umul64hi(uint64_t x,uint64_t y) { return (uint64_t)(((unsigned __int128)x*y)>>64); }
inline double __dmul_rn(double x,double y) { return x*y; }
inline double __dadd_rn(double x,double y) { return x+y; }
inline double __dsub_rn(double x,double y) { return x-y; }
inline double __ddiv_rn(double x,double y) { return x/y; }
inline float __fmul_rn(float x,float y) { return x*y; }
inline float __fadd_rn(float x,float y) { return x+y; }
inline float __fsub_rn(float x,float y) { return x-y; }
inline float __fdiv_rn(float x,float y) { return x/y; }
inline double __fma_rn(double x,double y,double z) { return std::fma(x,y,z); }
inline float __fmaf_rn(float x,float y,float z) { return std::fma(x,y,z); }
inline double __dsqrt_rn(double x) { return std::sqrt(x); }
