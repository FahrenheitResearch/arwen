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

template<class T> struct LakeArray {
    T* data;
    int lo1, lo2, lo3, n1, n2, n3;
    LAKE_HD LakeArray(T* p,int a,int b,int c=1,int d=1,int e=1,int f=1)
      : data(p),lo1(a),lo2(c),lo3(e),n1(b-a+1),n2(d-c+1),n3(f-e+1) {}
    LAKE_HD T& operator()(int a) {return data[a-lo1];}
    LAKE_HD const T& operator()(int a) const {return data[a-lo1];}
    LAKE_HD T& operator()(int a,int b) {return data[a-lo1+n1*(b-lo2)];}
    LAKE_HD const T& operator()(int a,int b) const {return data[a-lo1+n1*(b-lo2)];}
    LAKE_HD T& operator()(int a,int b,int c) {return data[a-lo1+n1*(b-lo2+n2*(c-lo3))];}
    LAKE_HD const T& operator()(int a,int b,int c) const {return data[a-lo1+n1*(b-lo2+n2*(c-lo3))];}
    LAKE_HD void fill(T value) {for(int j=0;j<n1*n2*n3;++j)data[j]=value;}
};
template<class T,int N> struct LakeStorage: LakeArray<T> {
    T values[N];
    LAKE_HD LakeStorage(int a,int b,int c=1,int d=1,int e=1,int f=1)
      : LakeArray<T>(values,a,b,c,d,e,f) {}
};
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
