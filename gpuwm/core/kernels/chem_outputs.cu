// WRF chem/module_gocart_aerosols.F:112-144: every operation rounds to REAL(4).
__device__ float terms(const unsigned long long *ptrs, const int *species,
 const int *starts, const int *opstarts, const int *codes, const float *values,
 int nt, int idx) {
 float total=0.0f;
 for(int t=0;t<nt;++t) {
  float v=((const float*)ptrs[species[starts[t]]])[idx];
  for(int s=starts[t]+1;s<starts[t+1];++s)
   v=__fadd_rn(v,((const float*)ptrs[species[s]])[idx]);
  for(int o=opstarts[t];o<opstarts[t+1];++o)
   v=codes[o]==0 ? __fmul_rn(v,values[o]) : __fdiv_rn(v,values[o]);
  total=__fadd_rn(total,v);
 }
 return total;
}
#define ARGS const unsigned long long *ptrs,const int *species,const int *starts,const int *opstarts,const int *codes,const float *values,int nt,const float *alt,const float *dz,int nz,int plane,int divide,float scale,float *out
#define SUM(i) terms(ptrs,species,starts,opstarts,codes,values,nt,i)
extern "C" __global__ void term_sum_3d(ARGS) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=nz*plane) return;
 float v=SUM(i); out[i]=divide ? __fdiv_rn(v,alt[i]) : v;
}
extern "C" __global__ void term_sum_surface(ARGS) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=plane) return;
 float v=SUM(i); out[i]=divide ? __fdiv_rn(v,alt[i]) : v;
}
// DESIGN section 8: dryrho * dz first, then term sum * mass, levels ascending.
extern "C" __global__ void column_integral(ARGS) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=plane) return;
 float total=0.0f;
 for(int k=0;k<nz;++k) {
  int idx=k*plane+i;
  float v=SUM(idx); if(divide) v=__fdiv_rn(v,alt[idx]);
  float mass=__fmul_rn(__fdiv_rn(1.0f,alt[idx]),dz[idx]);
  total=__fadd_rn(total,__fmul_rn(v,mass));
 }
 out[i]=__fmul_rn(total,scale);
}
