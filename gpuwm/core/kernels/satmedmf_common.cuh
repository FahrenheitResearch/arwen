// Apache-2.0. WOOF vocabulary for the pinned CCPP TKE-EDMF column.
#pragma once
#ifdef GPUWM_SATMEDMF_HOST
#include <cmath>
#include <vector>
#include <stdexcept>
#define __device__
#define __forceinline__ inline
#endif
#ifdef GPUWM_SATMEDMF_HOST
#define SM_OP(name,op) inline double sm_##name(double a,double b){return a op b;} inline float sm_##name(float a,float b){return a op b;}
SM_OP(add,+) SM_OP(sub,-) SM_OP(mul,*) SM_OP(div,/)
#else
__device__ __forceinline__ double sm_add(double a,double b){return __dadd_rn(a,b);}
__device__ __forceinline__ double sm_sub(double a,double b){return __dsub_rn(a,b);}
__device__ __forceinline__ double sm_mul(double a,double b){return __dmul_rn(a,b);}
__device__ __forceinline__ double sm_div(double a,double b){return __ddiv_rn(a,b);}
__device__ __forceinline__ float sm_add(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float sm_sub(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float sm_mul(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float sm_div(float a,float b){return __fdiv_rn(a,b);}
#endif
template<class T> __device__ __forceinline__ T sm_max(T a,T b){return (a>b || a!=a)?a:b;}
template<class T> __device__ __forceinline__ T sm_min(T a,T b){return (a<b || a!=a)?a:b;}
__device__ __forceinline__ double sm_abs(double a){return fabs(a);}
__device__ __forceinline__ float sm_abs(float a){return fabsf(a);}
#ifdef GPUWM_SATMEDMF_HOST
inline double sm_sqrt(double a){return sqrt(a);}
inline float sm_sqrt(float a){return sqrtf(a);}
inline double sm_exp(double a){return exp(a);}
inline float sm_exp(float a){return expf(a);}
inline double sm_log(double a){return log(a);}
inline float sm_log(float a){return logf(a);}
inline double sm_pow(double a,double b){return pow(a,b);}
inline float sm_pow(float a,float b){return powf(a,b);}
#else
__device__ __forceinline__ double sm_sqrt(double a){return __dsqrt_rn(a);}
__device__ __forceinline__ float sm_sqrt(float a){return __fsqrt_rn(a);}
__device__ __forceinline__ double sm_exp(double a){return glibc_exp(a);}
__device__ __forceinline__ float sm_exp(float a){return gfk_exp(a);}
__device__ __forceinline__ double sm_log(double a){return glibc_log(a);}
__device__ __forceinline__ float sm_log(float a){return gfk_log(a);}
__device__ __forceinline__ double sm_pow(double a,double b){return glibc_pow(a,b);}
__device__ __forceinline__ float sm_pow(float a,float b){return gfk_pow(a,b);}
#endif
template<class T> __device__ __forceinline__ T sm_powi(T x,int m){
 unsigned int n=m<0 ? (unsigned)(-m):(unsigned)m;
 T y=(n&1u)?x:(T)1;
 while(n>>=1){x=sm_mul(x,x);if(n&1u)y=sm_mul(y,x);}
 return m<0?sm_div((T)1,y):y;
}
// These CCPP constants use explicit kind_phys literals, unlike WRF default-real parameters.
constexpr double sm_g=gfs_physcons::con_g,sm_rd=gfs_physcons::con_rd;
constexpr double sm_cp=gfs_physcons::con_cp,sm_rv=gfs_physcons::con_rv;
constexpr double sm_hvap=gfs_physcons::con_hvap,sm_hfus=gfs_physcons::con_hfus;
constexpr double sm_eps=gfs_physcons::con_eps,sm_epsm1=gfs_physcons::con_epsm1;
constexpr double sm_fvirt=gfs_physcons::con_fvirt;
__device__ __forceinline__ double sm_fpvs(double t,const double* table){
 double xj=sm_min(sm_max(sm_add(-8999.,sm_mul(50.,t)),1.),7501.);
 int jx=(int)sm_min(xj,7500.);
 return sm_add(table[jx-1],sm_mul(sm_sub(xj,(double)jx),sm_sub(table[jx],table[jx-1])));
}
struct SMWorkspace {
 double* base; const double* table; int offset=0,capacity,highwater=0;
 __device__ SMWorkspace(double* p,const double* t,int n):base(p),table(t),capacity(n){}
 template<class T> __device__ T* alloc(int n){
#ifdef GPUWM_SATMEDMF_HOST
  if(offset+n>capacity)throw std::runtime_error("SATMEDMF workspace exhausted");
#endif
  T* p=(T*)(base+offset);offset+=n;
  if(offset>highwater)highwater=offset;
  return p;
 }
};
struct SMFrame {
 SMWorkspace& ws;int mark;
 __device__ SMFrame(SMWorkspace& w):ws(w),mark(w.offset){}
 __device__ ~SMFrame(){ws.offset=mark;}
};
template<class T> struct SMView {
 T* p; int n1,n2,n3,l1,l2,l3;
 __device__ SMView(T* q,int a,int b=1,int c=1,int x=1,int y=1,int z=1):p(q),n1(a),n2(b),n3(c),l1(x),l2(y),l3(z){}
 __device__ __forceinline__ T& operator()(int i,int j=1,int k=1){
#ifdef GPUWM_SATMEDMF_HOST
  if(i<l1||i>=l1+n1||j<l2||j>=l2+n2||k<l3||k>=l3+n3)throw std::runtime_error("SATMEDMF column index outside declared bounds");
#endif
  return p[(i-l1)+n1*((j-l2)+n2*(k-l3))];
 }
};
