#include "host_shim.h"
#include "glibc_flt64.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <omp.h>
#include <inttypes.h>
extern "C" double cr_cos(double);
extern "C" double cr_acos(double);
static uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
static uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static double realbits(uint64_t u){double x;memcpy(&x,&u,8);return x;}
static double unit(uint64_t u){return (u>>11)*0x1p-53;}
static void argument(int f,uint64_t i,double *x,double *y){
 uint64_t a=mix(i),b=mix(i^0xd192ed03ULL);
 if(f==2&&i<4096){static const double xs[]={2.,0.5,-2.,-0.5,0x1.fffffffffffffp+1023,0x1p-1022,0x1p-1074,0x1.0000000000001p+0,0x1.fffffffffffffp-1};static const double ys[]={1023.,1024.,1025.,-1022.,-1074.,-1075.,2.,3.,0.5,1.,-1.,0x1p-65,-0x1p-65,0x1p63,-0x1p63};*x=realbits(bits(xs[(i/15)%9])+(i/135)%7-3);*y=ys[i%15];return;}double u=unit(a),v=unit(b);*y=0.;
 static const double powers[]={2./3.,1./3.,3./2.,-3.,-1./3.,2.,3.,0.5};
 static const uint64_t special[]={0,0x8000000000000000ULL,1,0x8000000000000001ULL,0xfffffffffffffULL,0x10000000000000ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL,0x7ff8000000000000ULL,0xfff8000000000000ULL,0x7ff0000000000001ULL,0x7ff8123456789abcULL,0x3ff0000000000000ULL,0xbff0000000000000ULL,0x3ff0000000000001ULL,0x3fefffffffffffffULL,0x7fefffffffffffffULL,0xffefffffffffffffULL,0x4000000000000000ULL,0xbfe0000000000000ULL};
 switch(i%8){
 case 0:case 1:*x=realbits(a);*y=realbits(b);break;
 case 2:case 3:case 4:*x=f==0?100*u-50:f==1?1.e6*(u+0x1p-53):1.e-12+(1.e6-1.e-12)*u;*y=(i%8==4)?8*v-4:powers[(i/8)%8];break;
 case 5:*x=realbits(special[(i/8)%(sizeof(special)/sizeof(*special))]);*y=realbits(special[(i/(8*(sizeof(special)/sizeof(*special))))%(sizeof(special)/sizeof(*special))]);break;
 case 6:*x=f==0?realbits(bits((i&8)?709.782712893384:-745.1332191019411)+(a%8192)-4096):f==1?realbits(bits(1.)+(a%0x2000000000000ULL)-0x1000000000000ULL):-1.e6*u;*y=(int)(b%65)-32;break;
 default:*x=realbits(a&0xfffffffffffffULL);if(f==0&&a>>63)*x=-*x;*y=powers[(i/8)%8];break;
 }
}
struct Example{uint64_t i,x,y,a,b;};
int main(int argc,char **argv){
 const char *name=argc>1?argv[1]:"exp";uint64_t n=argc>2?strtoull(argv[2],0,0):(1ULL<<34);int threads=argc>3?atoi(argv[3]):16;if(threads<1||threads>16)return 2;omp_set_num_threads(threads);
 void *lib=dlopen("libm.so.6",RTLD_NOW);auto expfn=(double(*)(double))dlsym(lib,"exp");auto logfn=(double(*)(double))dlsym(lib,"log");auto powfn=(double(*)(double,double))dlsym(lib,"pow");auto cosfn=(double(*)(double))dlsym(lib,"cos");auto acosfn=(double(*)(double))dlsym(lib,"acos");
 int f=!strcmp(name,"exp")?0:!strcmp(name,"log")?1:!strcmp(name,"pow")?2:!strcmp(name,"acos")?3:!strcmp(name,"cos")?4:!strcmp(name,"composite")?5:!strcmp(name,"dense")?6:!strcmp(name,"cosbits")?7:!strcmp(name,"acosbits")?8:-1;if(f<0)return 2;
 uint64_t mismatches=0,transcription_mismatches=0;Example ex[20];int nex=0;double started=omp_get_wtime();
 #pragma omp parallel for schedule(static) reduction(+:mismatches,transcription_mismatches)
 for(uint64_t i=0;i<n;i++){
  double x,y,a,b;if(f<3){argument(f,i,&x,&y);a=f==0?expfn(x):f==1?logfn(x):powfn(x,y);b=f==0?glibc_exp(x):f==1?glibc_log(x):glibc_pow(x,y);}
  else{
   x=f==4?-2.1+5.25*unit(mix(i)):-1.+2.*unit(mix(i));y=0.;
   if(f==7){x=realbits(mix(i));if(!isfinite(x))x=0.;}
   if(f==8){uint64_t a=mix(i);x=realbits((a&0x8000000000000000ULL)|(a%0x3ff0000000000001ULL));}
   if(f==6){uint64_t k=i/6;switch(i%6){case 0:x=realbits(bits(1.)-k);break;case 1:x=-realbits(bits(1.)-k);break;case 2:x=realbits(k);break;case 3:x=-realbits(k);break;case 4:x=k*0x1p-48;break;default:x=-(double)k*0x1p-48;}}
   if(f==3||f==6||f==8){a=acosfn(x);b=uw_acos(x);transcription_mismatches+=bits(b)!=bits(cr_acos(x));}
   else if(f==4||f==7){a=cosfn(x);b=uw_cos(x);transcription_mismatches+=bits(b)!=bits(cr_cos(x));}
   else {a=cosfn(acosfn(x)/3.);b=uw_cos(uw_acos(x)/3.);transcription_mismatches+=bits(b)!=bits(cr_cos(cr_acos(x)/3.));}
  }
  if(bits(a)!=bits(b)){
   mismatches++;
   #pragma omp critical
   {Example e={i,bits(x),bits(y),bits(a),bits(b)};if(nex<20)ex[nex++]=e;else{int worst=0;for(int j=1;j<20;j++)if(ex[j].i>ex[worst].i)worst=j;if(i<ex[worst].i)ex[worst]=e;}}
  }
 }
 printf("function=%s samples=%" PRIu64 " threads=%d mismatches=%" PRIu64 " transcription_mismatches=%" PRIu64 " seconds=%.6f\n",name,n,threads,mismatches,transcription_mismatches,omp_get_wtime()-started);
 for(int i=0;i<nex;i++)for(int j=i+1;j<nex;j++)if(ex[j].i<ex[i].i){Example e=ex[i];ex[i]=ex[j];ex[j]=e;}
 for(int i=0;i<nex;i++)printf("i=%" PRIu64 " x=0x%016" PRIx64 " y=0x%016" PRIx64 " libm=0x%016" PRIx64 " ours=0x%016" PRIx64 "\n",ex[i].i,ex[i].x,ex[i].y,ex[i].a,ex[i].b);
 return f<3&&mismatches?1:transcription_mismatches?1:0;
}
