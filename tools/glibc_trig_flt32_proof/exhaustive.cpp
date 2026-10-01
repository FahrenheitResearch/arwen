#include "host_shim.hpp"
#include "glibc_trig_flt32.cuh"
#include <omp.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfenv>
using Fn=float(*)(float);
const char* names[]={"sinf","cosf","tanf","asinf","acosf","atanf"};
Fn oracle[]={::sinf,::cosf,::tanf,::asinf,::acosf,::atanf};
Fn trans[]={glibc_sinf,glibc_cosf,glibc_tanf,glibc_asinf,glibc_acosf,glibc_atanf};
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IOLBF,0);
  std::fesetround(FE_TONEAREST);
  uint64_t stride=argc>1?strtoull(argv[1],nullptr,0):1;
  for(int f=0;f<6;f++) {
    Dl_info d{}; dladdr((void*)oracle[f],&d);
    printf("%s resolved=%s+0x%llx threads=%d stride=%llu\n",names[f],d.dli_fname,
      (unsigned long long)((uintptr_t)oracle[f]-(uintptr_t)d.dli_fbase),omp_get_max_threads(),(unsigned long long)stride);
    auto t=std::chrono::steady_clock::now();
    uint64_t bad=0, visited=0, input_sum=0, first=UINT64_MAX, count=(0x100000000ULL+stride-1)/stride;
    #pragma omp parallel for reduction(+:bad,visited,input_sum) reduction(min:first) schedule(static)
    for(uint64_t k=0;k<count;k++) {
      uint32_t u=(uint32_t)(k*stride);
      visited++;input_sum+=u;
      float x=__uint_as_float(u);
      uint32_t a=__float_as_uint(oracle[f](x)),b=__float_as_uint(trans[f](x));
      if(a!=b) {bad++; if(u<first) first=u;}
    }
    printf("%s inputs=%llu visited=%llu input_sum=%016llx mismatches=%llu seconds=%.3f",names[f],(unsigned long long)count,(unsigned long long)visited,(unsigned long long)input_sum,(unsigned long long)bad,
      std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());
    if(bad) {
      printf(" first=%08x",(uint32_t)first);
      unsigned shown=0;
      for(uint64_t k=first;k<0x100000000ULL && shown<8;k+=stride) {
        float x=__uint_as_float(k);
        uint32_t a=__float_as_uint(oracle[f](x)),b=__float_as_uint(trans[f](x));
        if(a!=b) {printf(" [%08x:%08x/%08x]",(uint32_t)k,a,b);shown++;}
      }
    }
    puts("");
  }
}
