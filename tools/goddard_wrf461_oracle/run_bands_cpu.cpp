// Development comparison of generated operation order on the CPU box.
// The production kernels use WOOF math. This runner uses the host scalar math.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
#define __device__
inline float gfk_exp(float x){return std::exp(x);}
inline float gfk_log(float x){return std::log(x);}
inline float gfk_pow(float x,float y){return std::pow(x,y);}
inline double plm_exp(double x){return std::exp(x);}
inline double plm_log(double x){return std::log(x);}
inline double plm_pow(double x,double y){return std::pow(x,y);}
inline float __fsqrt_rn(float x){return std::sqrt(x);}
inline double __dsqrt_rn(double x){return std::sqrt(x);}
#include "generated-bands.cuh"
static std::vector<float> read_f(std::ifstream &in,int n){
 std::vector<float> v(n);in.read(reinterpret_cast<char*>(v.data()),n*4);
 if(!in)throw std::runtime_error("truncated inputs");return v;
}
static std::vector<GReal> column(const std::vector<float>& v,int c){
 std::vector<GReal> out(v.size()/16);
 for(unsigned i=0;i<out.size();++i)out[i]=GReal(v[c+16*i]);return out;
}
static GArray<GReal,1> a1(std::vector<GReal>& v,int n){return {v.data(),{1},{n}};}
static GArray<GReal,2> a2(std::vector<GReal>& v,int n,int b){return {v.data(),{1,1},{n,b}};}
static GArray<float,1> o1(std::vector<float>& v,int n){return {v.data(),{1},{n}};}
int main(int argc,char** argv){
 if(argc!=3)return 2;
 std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
 int32_t header[5]={0x47524435,64,32,4,int(sizeof(GReal)*8)},ih[4];
 in.read(reinterpret_cast<char*>(ih),16);
 if(ih[0]!=0x47524935 || ih[1]!=64 || ih[2]!=32 || ih[3]!=4)return 3;
 out.write(reinterpret_cast<char*>(header),20);
 for(int batch=0;batch<4;++batch){
  int n=32,c=16;
  auto pl=read_f(in,c*(n+1)),ta=read_f(in,c*n),wa=read_f(in,c*n),oa=read_f(in,c*n),fc=read_f(in,c*n);
  auto cosz=read_f(in,c),tb=read_f(in,c),ts=read_f(in,c),em=read_f(in,c*10),al=read_f(in,c);
  int32_t ict[16],icb[16],mask[16];
  in.read(reinterpret_cast<char*>(ict),64);in.read(reinterpret_cast<char*>(icb),64);in.read(reinterpret_cast<char*>(mask),64);
  auto ct=read_f(in,c*n*11),cs=read_f(in,c*n*11),cg=read_f(in,c*n*11),lt=read_f(in,c*n*10),ls=read_f(in,c*n*10),lg=read_f(in,c*n*10);
  std::vector<std::vector<float>> result(7);
  for(int f=0;f<7;++f)result[f].resize(16*(f==3?4:33));
  for(int col=0;col<c;++col){
   auto p=column(pl,col),t=column(ta,col),w=column(wa,col),o=column(oa,col),cloud=column(fc,col),e=column(em,col);
   auto tc=column(ct,col),sc=column(cs,col),gc=column(cg,col),tl=column(lt,col),sl=column(ls,col),gl=column(lg,col);
   std::vector<GReal> zero(n*11,0);
   std::vector<std::vector<float>> flux(7);
   for(int f=0;f<7;++f)flux[f].resize(f==3?4:33);
   if(mask[col])g_swrad(n,GReal(cosz[col]),a1(p,n+1),a1(t,n),a1(w,n),a1(o,n),a1(cloud,n),ict[col],icb[col],
    a2(tc,n,11),a2(sc,n,11),a2(gc,n,11),a2(zero,n,11),a2(zero,n,11),a2(zero,n,11),
    GReal(al[col]),GReal(al[col]),GReal(al[col]),GReal(al[col]),o1(flux[0],33),o1(flux[1],33),o1(flux[2],33),o1(flux[3],4),true,1);
   g_lwrad(n,a1(e,10),GReal(tb[col]),GReal(ts[col]),ict[col],icb[col],a1(p,n+1),a1(t,n),a1(w,n),a1(o,n),a1(cloud,n),
    a2(tl,n,10),a2(sl,n,10),a2(gl,n,10),a2(zero,n,10),a2(zero,n,10),a2(zero,n,10),o1(flux[4],33),o1(flux[5],33),o1(flux[6],33),1);
   for(int f=0;f<7;++f)for(unsigned k=0;k<flux[f].size();++k)result[f][col+16*k]=flux[f][k];
  }
  for(const auto& v:result)out.write(reinterpret_cast<const char*>(v.data()),v.size()*4);
  for(const auto* v:{&ct,&cs,&cg,&lt,&ls,&lg})out.write(reinterpret_cast<const char*>(v->data()),v->size()*4);
 }
 if(in.peek()!=EOF)return 4;
 return out?0:5;
}
