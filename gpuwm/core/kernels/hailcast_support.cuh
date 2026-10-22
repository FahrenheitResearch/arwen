// HAILCAST REAL/REAL*8 expression kinds and one-based strided arrays.
template<class T> struct HailArray {
    T* data; int stride;
    __device__ T& operator()(int k) const { return data[(k-1)*stride]; }
};
template<class T,int N> struct HailStorage: HailArray<T> {
    T storage[N];
    __device__ HailStorage() { this->data=storage; this->stride=1; }
};
// A146: IEEE round-to-nearest division in the expression's own precision.  A plain
// a/b by a compile-time constant can be rewritten by the compiler into a multiply by
// the rounded reciprocal, which is not the division HAILCAST's Fortran performs
// (literal-division census, 17 sites through this helper, 2026-10-21).  Integer
// division stays integer.
__device__ inline float lake_div_rn(float a,float b) { return __fdiv_rn(a,b); }
__device__ inline double lake_div_rn(double a,double b) { return __ddiv_rn(a,b); }
__device__ inline int lake_div_rn(int a,int b) { return a/b; }
__device__ inline long long lake_div_rn(long long a,long long b) { return a/b; }
template<class A,class B> __device__ auto lake_div(A a,B b)->decltype(a/b) {
    typedef decltype(a/b) R; return lake_div_rn((R)a,(R)b);
}
template<class A,class B> __device__ auto lake_pow(A a,B b)->decltype(a+b) {
    typedef decltype(a+b) R; return sizeof(R)>4 ? (R)pow((double)a,(double)b) : (R)powf((float)a,(float)b);
}
template<class A> __device__ A lake_abs(A a) { return a<0?-a:a; }
template<class A,class B> __device__ auto lake_min(A a,B b)->decltype(a+b) { return a<b?a:b; }
template<class A,class B> __device__ auto lake_max(A a,B b)->decltype(a+b) { return a>b?a:b; }
__device__ float lake_exp(float a) { return expf(a); }
__device__ double lake_exp(double a) { return exp(a); }
__device__ float lake_log(float a) { return logf(a); }
__device__ double lake_log(double a) { return log(a); }
__device__ float lake_log10(float a) { return log10f(a); }
__device__ double lake_log10(double a) { return log10(a); }
__device__ float lake_sin(float a) { return sinf(a); }
__device__ double lake_sin(double a) { return sin(a); }
