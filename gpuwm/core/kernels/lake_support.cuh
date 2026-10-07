// Numeric and column-array support for the WRF lake transcription.
// WRF public-domain notice: licenses/LICENSE-WRF-public-domain.txt.
#ifndef GPUWM_LAKE_SUPPORT
#define GPUWM_LAKE_SUPPORT
#ifdef __CUDACC__
#define LAKE_HD __device__
#else
#include <cmath>
#define LAKE_HD
using std::sqrt; using std::exp; using std::log; using std::log10;
using std::sin; using std::cos; using std::atan; using std::pow;
using std::round; using std::copysign;
using std::fabs;
#endif

// THE BREAKAGE THE ARENA PREVENTS (A3/A13 2026-10-05): the WRF lake column
// keeps every LakeStorage array in the per-thread local frame, 14,400 B per
// thread.  CUDA backs local memory for the widest frame a context ever
// launches at that size times every resident thread of every SM, and keeps
// it: 3.3 GiB on an RTX 5090 and 3.6 GiB on an RTX PRO 6000, charged once
// per card of every run with sf_lake_physics = 1, which is the largest
// single per-card fixed cost of a HRRR-physics [devices] run and a large
// part of why the full HRRR grid is refused on 4 x 32 GB and 1 x 96 GB.
// On the device the arrays now live in a global arena the launcher sizes
// to the columns of one launch; on the host they stay inline.  Element k of
// a thread's array sits k * stride elements from its first element, with
// consecutive threads in consecutive 8-byte slots, so a warp's accesses
// stay coalesced the way the hardware interleaves local memory.  Only the
// address of each element changes; every arithmetic operation is the same.
template<class T> struct LakePtr {
    T* p;
    int s;
    LAKE_HD T& operator[](int k) const {return p[k*s];}
};
#ifdef __CUDACC__
// One launch block is LAKE_ARENA_BLOCK threads (lake.py launches 32).
#define LAKE_ARENA_BLOCK 32
// Upper bound on the 8-byte slots live at once in one thread: every
// LakeStorage the transcription (lake_wrf.cuh) and the larger of the two
// entry points (lake.cu) declare, one slot per element whatever its
// type, summed as if all were live together (tests/test_lake_contract.py
// recomputes it from the sources).  An arena that small for a future
// transcription traps rather than letting two threads share slots.
#define LAKE_ARENA_SLOTS 1283
__shared__ char* lake_arena_base[LAKE_ARENA_BLOCK];
__shared__ int lake_arena_top[LAKE_ARENA_BLOCK];
__shared__ int lake_arena_threads[LAKE_ARENA_BLOCK];
__device__ inline void lake_arena_begin(char* arena,int threads,int slot) {
    lake_arena_base[threadIdx.x]=arena+8*(long long)slot;
    lake_arena_top[threadIdx.x]=0;
    lake_arena_threads[threadIdx.x]=threads;
}
template<class T> __device__ inline LakePtr<T> lake_arena_take(int count) {
    int top=lake_arena_top[threadIdx.x];
    if(top+count>LAKE_ARENA_SLOTS) __trap();
    lake_arena_top[threadIdx.x]=top+count;
    int threads=lake_arena_threads[threadIdx.x];
    LakePtr<T> ptr;
    ptr.p=reinterpret_cast<T*>(lake_arena_base[threadIdx.x]+8*(long long)top*threads);
    ptr.s=threads*(8/(int)sizeof(T));
    return ptr;
}
__device__ inline void lake_arena_give(int count) {lake_arena_top[threadIdx.x]-=count;}
#endif

template<class T> struct LakeArray {
    LakePtr<T> data;
    int lo1, lo2, lo3, n1, n2, n3;
    LAKE_HD LakeArray(T* p,int a,int b,int c=1,int d=1,int e=1,int f=1)
      : data{p,1},lo1(a),lo2(c),lo3(e),n1(b-a+1),n2(d-c+1),n3(f-e+1) {}
    LAKE_HD LakeArray(LakePtr<T> p,int a,int b,int c=1,int d=1,int e=1,int f=1)
      : data(p),lo1(a),lo2(c),lo3(e),n1(b-a+1),n2(d-c+1),n3(f-e+1) {}
    LAKE_HD T& operator()(int a) {return data[a-lo1];}
    LAKE_HD const T& operator()(int a) const {return data[a-lo1];}
    LAKE_HD T& operator()(int a,int b) {return data[a-lo1+n1*(b-lo2)];}
    LAKE_HD const T& operator()(int a,int b) const {return data[a-lo1+n1*(b-lo2)];}
    LAKE_HD T& operator()(int a,int b,int c) {return data[a-lo1+n1*(b-lo2+n2*(c-lo3))];}
    LAKE_HD const T& operator()(int a,int b,int c) const {return data[a-lo1+n1*(b-lo2+n2*(c-lo3))];}
    LAKE_HD void fill(T value) {for(int j=0;j<n1*n2*n3;++j)data[j]=value;}
};
#ifdef __CUDACC__
// Taken in declaration order and given back in reverse, which is C++'s own
// construction and destruction order for locals and members, so the arena
// is a stack.  Never copied: every routine takes LakeArray by reference.
template<class T,int N> struct LakeStorage: LakeArray<T> {
    __device__ LakeStorage(int a,int b,int c=1,int d=1,int e=1,int f=1)
      : LakeArray<T>(lake_arena_take<T>(N),a,b,c,d,e,f) {}
    __device__ ~LakeStorage() {lake_arena_give(N);}
    LakeStorage(const LakeStorage&) = delete;
    LakeStorage& operator=(const LakeStorage&) = delete;
};
#else
template<class T,int N> struct LakeStorage: LakeArray<T> {
    T values[N];
    LAKE_HD LakeStorage(int a,int b,int c=1,int d=1,int e=1,int f=1)
      : LakeArray<T>(values,a,b,c,d,e,f) {}
    LakeStorage(const LakeStorage&) = delete;
    LakeStorage& operator=(const LakeStorage&) = delete;
};
#endif
template<class A,class B> LAKE_HD auto lake_min(A a,B b)->decltype(a+b) {return a<b?a:b;}
template<class A,class B,class C> LAKE_HD auto lake_min(A a,B b,C c)->decltype(a+b+c) {return lake_min(lake_min(a,b),c);}
template<class A,class B> LAKE_HD auto lake_max(A a,B b)->decltype(a+b) {return a>b?a:b;}
template<class A,class B,class C> LAKE_HD auto lake_max(A a,B b,C c)->decltype(a+b+c) {return lake_max(lake_max(a,b),c);}
LAKE_HD int lake_abs(int x) {return x<0?-x:x;}
// Fortran ABS clears the sign bit even for -0 and signed NaNs. A float
// comparison would retain those signs; fabs.ftz.f32 can erase subnormals.
LAKE_HD float lake_abs(float x) {
#ifdef __CUDACC__
    return __int_as_float(__float_as_int(x) & 0x7fffffff);
#else
    return fabs(x);
#endif
}
LAKE_HD double lake_abs(double x) {
#ifdef __CUDACC__
    return __longlong_as_double(__double_as_longlong(x) & 0x7fffffffffffffffLL);
#else
    return fabs(x);
#endif
}
LAKE_HD int lake_div(int a,int b) {return a/b;}
LAKE_HD float lake_div(float a,float b) {
#ifdef __CUDACC__
    return __fdiv_rn(a,b);
#else
    return a/b;
#endif
}
LAKE_HD double lake_div(double a,double b) {
#ifdef __CUDACC__
    return __ddiv_rn(a,b);
#else
    return a/b;
#endif
}
template<class A,class B> LAKE_HD auto lake_div(A a,B b)->decltype(a+b) {
    using T=decltype(a+b); return lake_div(T(a),T(b));
}
template<class T> LAKE_HD T lake_pow(T a,int b) {
    bool negative=b<0; if(negative)b=-b;
    T result=1;
    while(b) {if(b&1)result=result*a;b>>=1;if(b)a=a*a;}
    return negative?lake_div(T(1),result):result;
}
LAKE_HD float lake_pow(float a,float b) {
#ifdef __CUDACC__
    return gfk_pow(a,b);
#else
    return pow(a,b);
#endif
}
template<class A,class B> LAKE_HD auto lake_pow(A a,B b)->decltype(a+b) {
    using T=decltype(a+b); return pow(T(a),T(b));
}
template<class T> LAKE_HD T lake_sum(const LakeArray<T>& a) {
    T result=0; for(int i=0;i<a.n1*a.n2*a.n3;++i)result=result+a.data[i]; return result;
}
#endif
