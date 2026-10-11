// Scalar arithmetic for generated Thompson columns. WOOF's existing math
// routines provide the transcendental words in both arithmetic modes.
#ifndef THOMPSON_COLUMN_KMAX
#define THOMPSON_COLUMN_KMAX 64
#endif
template<class A,class B> __device__ auto tc_min(A a,B b)->decltype(a+b) {return a<b?a:b;}
template<class A,class B,class C> __device__ auto tc_min(A a,B b,C c)->decltype(a+b+c) {return tc_min(tc_min(a,b),c);}
template<class A,class B> __device__ auto tc_max(A a,B b)->decltype(a+b) {return a>b?a:b;}
template<class A,class B,class C> __device__ auto tc_max(A a,B b,C c)->decltype(a+b+c) {return tc_max(tc_max(a,b),c);}
__device__ float tc_div(float a,float b) {return __fdiv_rn(a,b);}
__device__ double tc_div(double a,double b) {return __ddiv_rn(a,b);}
__device__ int tc_div(int a,int b) {return a/b;}
template<class A,class B> __device__ auto tc_div(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_div(T(a),T(b));}
__device__ float tc_abs(float a) {return __uint_as_float(__float_as_uint(a)&0x7fffffffU);}
__device__ double tc_abs(double a) {return __longlong_as_double(__double_as_longlong(a)&0x7fffffffffffffffLL);}
__device__ int tc_abs(int a) {return a<0?-a:a;}
__device__ float tc_sign(float a,float b) {return __uint_as_float((__float_as_uint(a)&0x7fffffffU)|(__float_as_uint(b)&0x80000000U));}
__device__ double tc_sign(double a,double b) {return copysign(tc_abs(a),b);}
__device__ float tc_sqrt(float a) {return __fsqrt_rn(a);}
__device__ double tc_sqrt(double a) {return __dsqrt_rn(a);}
__device__ float tc_exp(float a) {return thompson_aa_expf(a);}
__device__ double tc_exp(double a) {return thompson_aa_exp(a);}
__device__ float tc_log(float a) {return thompson_aa_logf(a);}
__device__ double tc_log(double a) {return thompson_aa_log(a);}
__device__ float tc_log10(float a) {return thompson_aa_log10f(a);}
__device__ double tc_log10(double a) {return thompson_aa_log10(a);}
__device__ float tc_pow(float a,float b) {return thompson_aa_powf(a,b);}
__device__ double tc_pow(double a,double b) {return thompson_aa_pow(a,b);}
template<class T> __device__ T tc_pow(T a,int b) {
    bool neg=b<0; unsigned int n=neg?0U-(unsigned int)b:(unsigned int)b;
    T y=(n&1U)?a:T(1);
    while(n>>=1U) {a=a*a;if(n&1U)y=y*a;}
    return neg?tc_div(T(1),y):y;
}
template<class A,class B> __device__ auto tc_pow(A a,B b)->decltype(a+b) {using T=decltype(a+b);return tc_pow(T(a),T(b));}
__device__ double tc_atan(double a) {return atan(a);}
__device__ float tc_atan(float a) {return atanf(a);}
__device__ float tc_sngl(double a) {return (float)a;}
__device__ double tc_dfloat(int a) {return (double)a;}
template<class T> __device__ bool tc_present(T a) {return true;}
__device__ bool tc_any(const bool* a,int lo,int hi) {for(int k=lo;k<=hi;k++)if(a[k-lo])return true;return false;}
