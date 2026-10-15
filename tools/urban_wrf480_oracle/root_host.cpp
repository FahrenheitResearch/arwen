// CPU compilation of the same arithmetic helper intended for CUDA.
// The helper is not yet coupled to a forecast generation.
#include <cfenv>
#define __device__
static inline float add(float a, float b) { volatile float c=a+b; return c; }
static inline float sub(float a, float b) { volatile float c=a-b; return c; }
static inline float mul(float a, float b) { volatile float c=a*b; return c; }
static inline float div(float a, float b) { volatile float c=a/b; return c; }
#define FADD add
#define FSUB sub
#define FMUL mul
#define FDIV div
#include "../../gpuwm/core/kernels/urban_root_uptake_480.cuh"
extern "C" void root_host(float* out, int n, const float* smc, float max,
    float dry, float radius, float lai, const float* dz, const float* zsoil,
    float ele, float fvg, float width)
{
    int rounding = std::fegetround();
    std::fesetround(FE_TONEAREST);
    switch(n) {
#define RUN(N) case N: u_root_uptake_480<N>(out,smc,max,dry,radius,lai,dz,zsoil,ele,fvg,width); break
        RUN(1); RUN(2); RUN(3); RUN(4); RUN(8); RUN(16);
#undef RUN
    }
    std::fesetround(rounding);
}
