#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
using Fn=float(*)(float);
static Fn fns[]={::sinf,::cosf,::tanf,::asinf,::acosf,::atanf};
static float frombits(uint32_t u) {float x;std::memcpy(&x,&u,4);return x;}
extern "C" void fill(float* out,uint64_t n) {
  constexpr uint64_t quarter=1ULL<<24;
  #pragma omp parallel for schedule(static)
  for(uint64_t k=0;k<n;k++) {
    if(k<quarter) out[k]=frombits((uint32_t)k*2654435761U);
    else if(k<2*quarter) out[k]=(float)(-10.0+20.0*(double)(k-quarter)/(double)(quarter-1));
    else if(k<3*quarter) out[k]=(float)(-1.0+2.0*(double)(k-2*quarter)/(double)(quarter-1));
    else { uint32_t j=(uint32_t)(k-3*quarter);out[k]=frombits((j&0x7fffffU)|((j>>23)<<31)); }
  }
}
extern "C" void reference(int f,const float* x,uint32_t* out,uint64_t n) {
  #pragma omp parallel for schedule(static)
  for(uint64_t k=0;k<n;k++) {float r=fns[f](x[k]);std::memcpy(out+k,&r,4);}
}
