#define main sweep_cpu_main
#include "sweep_cpu.c"
#undef main
extern "C" void g64_batch(int f,uint64_t start,uint64_t n,double *x,double *y,double *out){
 static void *lib=dlopen("libm.so.6",RTLD_NOW);
 static auto ef=(double(*)(double))dlsym(lib,"exp");
 static auto lf=(double(*)(double))dlsym(lib,"log");
 static auto pf=(double(*)(double,double))dlsym(lib,"pow");
 #pragma omp parallel for num_threads(16) schedule(static)
 for(uint64_t j=0;j<n;j++){
  uint64_t i=start+j;
  if(f<3)argument(f,i,x+j,y+j);
  else{x[j]=f==3?-2.1+5.25*unit(mix(i)):-1.+2.*unit(mix(i));y[j]=0.;}
  out[j]=f==0?ef(x[j]):f==1?lf(x[j]):f==2?pf(x[j],y[j]):f==3?cr_cos(x[j]):cr_acos(x[j]);
 }
}
