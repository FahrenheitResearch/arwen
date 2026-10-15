// Development operation-order referee. Production uses WOOF device math.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>
#include <iostream>
#define __device__
inline float gfk_exp(float x){return std::exp(x);} inline float gfk_log(float x){return std::log(x);}
inline float gfk_pow(float x,float y){return std::pow(x,y);}
inline double plm_exp(double x){return std::exp(x);} inline double plm_log(double x){return std::log(x);}
inline double plm_pow(double x,double y){return std::pow(x,y);}
inline float __fsqrt_rn(float x){return std::sqrt(x);} inline double __dsqrt_rn(double x){return std::sqrt(x);}
#include "goddard_bands.cuh"
int main(int argc,char**argv){
 if(argc!=3)return 2;std::ifstream f(argv[1],std::ios::binary);int n;f.read((char*)&n,4);
 std::vector<float> in(5*n+15),opt(63*n);int flags[3];f.read((char*)in.data(),4*in.size());f.read((char*)flags,12);f.read((char*)opt.data(),4*opt.size());
 std::vector<double> a(in.begin(),in.end()),b(opt.begin(),opt.end()),zero(11*n,0);
 auto a1=[](double*p,int n){return GArray<double,1>{p,{1},{n}};};
 auto a2=[](double*p,int n,int m){return GArray<double,2>{p,{1,1},{n,m}};};
 auto o1=[](float*p,int n){return GArray<float,1>{p,{1},{n}};};
 int at=0;auto p=a1(a.data()+at,n+1);at+=n+1;auto t=a1(a.data()+at,n);at+=n;auto w=a1(a.data()+at,n);at+=n;auto o=a1(a.data()+at,n);at+=n;auto fc=a1(a.data()+at,n);at+=n;
 double mu=a[at++],tb=a[at++],ts=a[at++];auto e=a1(a.data()+at,10);at+=10;double alb=a[at];
 std::vector<float> sw(n+1,0),sd(n+1,0),su(n+1,0),sf(4,0),lw(n+1),ld(n+1),lu(n+1);
 auto zs=a2(zero.data(),n,11),zl=a2(zero.data(),n,10);
 g_swrad(n,mu,p,t,w,o,fc,flags[0],flags[1],a2(b.data(),n,11),a2(b.data()+11*n,n,11),a2(b.data()+22*n,n,11),zs,zs,zs,alb,alb,alb,alb,o1(sw.data(),n+1),o1(sd.data(),n+1),o1(su.data(),n+1),o1(sf.data(),4),flags[2]!=0,1);
 g_lwrad(n,e,tb,ts,flags[0],flags[1],p,t,w,o,fc,a2(b.data()+33*n,n,10),a2(b.data()+43*n,n,10),a2(b.data()+53*n,n,10),zl,zl,zl,o1(lw.data(),n+1),o1(ld.data(),n+1),o1(lu.data(),n+1),1);
 std::ofstream output(argv[2],std::ios::binary);for(auto*v:{&sw,&sd,&su,&sf,&lw,&ld,&lu})output.write((char*)v->data(),v->size()*4);
 int bad=0;for(float x:lw)bad+=!std::isfinite(x);std::cout<<"LW nonfinite="<<bad<<"\n";
}
