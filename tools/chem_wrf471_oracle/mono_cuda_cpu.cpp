// CPU arithmetic check of the actual CUDA source, without a CUDA runtime.
// This cannot prove device compilation, launch addressing or device arithmetic.
// Compile -O0 -ffp-contract=off -fno-fast-math; IEEE float, round-to-nearest.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define __device__
#define __global__
#define __forceinline__ inline
struct Dim { int x=0,y=0,z=0; } blockIdx,blockDim,threadIdx;
inline float __fadd_rn(float a,float b) { volatile float r=a+b; return r; }
inline float __fsub_rn(float a,float b) { volatile float r=a-b; return r; }
inline float __fmul_rn(float a,float b) { volatile float r=a*b; return r; }
inline float __fdiv_rn(float a,float b) { volatile float r=a/b; return r; }
#ifndef MONO_SOURCE
#define MONO_SOURCE "../../gpuwm/core/kernels/mono_advection.cu"
#endif
#include MONO_SOURCE
namespace fs=std::filesystem;
using Vec=std::vector<float>;
Vec read(const fs::path& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("cannot read "+path.string());
    auto n=f.tellg(); if(n%4) throw std::runtime_error("partial float word");
    Vec a(size_t(n)/4); f.seekg(0); f.read(reinterpret_cast<char*>(a.data()),n); return a;
}
int flag(const fs::path& path) {
    std::ifstream f(path,std::ios::binary); int32_t n=0;
    f.read(reinterpret_cast<char*>(&n),4); if(!f) throw std::runtime_error("missing flag"); return n;
}
Vec volume(const fs::path& path,int nz,int ny,int nx) {
    Vec a=read(path),b(a.size());
    if(a.size()!=size_t(nz)*ny*nx) throw std::runtime_error("volume size mismatch");
    for(int j=0;j<ny;j++) for(int k=0;k<nz;k++) for(int i=0;i<nx;i++)
        b[(size_t(k)*ny+j)*nx+i]=a[(size_t(j)*nz+k)*nx+i];
    return b;
}
void put(const fs::path& dir,std::ofstream& manifest,const std::string& name,const Vec& a,
         int nz,int ny,int nx,int ox=0,int oy=0) {
    int sx=ox?1:0,sy=oy?1:0,ex=nx-sx,ey=ny-sy;
    std::ofstream f(dir/(name+".bin"),std::ios::binary);
    for(int j=sy;j<ey;j++) for(int k=0;k<nz;k++) for(int i=sx;i<ex;i++)
        f.write(reinterpret_cast<const char*>(&a[(size_t(k)*ny+j)*nx+i]),4);
    manifest<<name<<" f4 3 "<<ex-sx<<" "<<nz<<" "<<ey-sy<<"\n";
}
int main(int argc,char** argv) {
    if(argc!=3) { std::cerr<<"usage: mono_cuda_cpu fixtures output\n"; return 2; }
    static_assert(sizeof(float)==4);
    uint32_t endian=1; if(*reinterpret_cast<char*>(&endian)!=1) return 3;
    for(auto entry:fs::directory_iterator(argv[1])) {
        fs::path dir=entry.path(); if(!fs::exists(dir/"MANIFEST.txt")) continue;
        int nx=12,ny=10,nz=8,b=flag(dir/"boundary.bin"),vo=flag(dir/"vorder.bin");
        auto vol=[&](const char* name,int z,int y,int x) { return volume(dir/(std::string(name)+".bin"),z,y,x); };
        auto arr=[&](const char* name) { return read(dir/(std::string(name)+".bin")); };
        Vec q=vol("q",nz,ny,nx),q0=vol("q0",nz,ny,nx),ru=vol("ru",nz,ny,nx+1),rv=vol("rv",nz,ny+1,nx);
        Vec rw=vol("ww",nz+1,ny,nx),wi=vol("wi",nz+1,ny,nx),t=vol("initial_tendency",nz,ny,nx);
        Vec mut=arr("mut"),mub=arr("mub"),mu0=arr("mu0"),mx=arr("mx"),my=arr("my");
        Vec c1=arr("c1"),c2=arr("c2"),rd=arr("rd"),fnm=arr("fnm"),fnp=arr("fnp");
        Vec xl(ru.size()),xc(ru.size()),yl(rv.size()),yc(rv.size()),zl(rw.size()),zc(rw.size());
        Vec qmin(q.size()),qmax(q.size()),si(q.size()),so(q.size()),ht(q.size()),zt(q.size());
        Mono p={q.data(),q0.data(),ru.data(),rv.data(),rw.data(),mut.data(),c1.data(),c2.data(),rd.data(),
            fnm.data(),fnp.data(),mx.data(),my.data(),wi.data(),mu0.data(),mub.data(),
            xl.data(),xc.data(),yl.data(),yc.data(),zl.data(),zc.data(),qmin.data(),qmax.data(),si.data(),so.data(),
            t.data(),ht.data(),zt.data(),1.0f/arr("dx")[0],1.0f/arr("dy")[0],arr("dt")[0],nz,ny,nx,b!=0,b!=0,b==2,vo};
        for(int k=0;k<=nz;k++) for(int j=0;j<=ny;j++) for(int i=0;i<=nx;i++) mono_flux_at(p,k,j,i);
        for(int k=0;k<nz;k++) for(int j=0;j<ny;j++) for(int i=0;i<nx;i++) mono_scale_at(p,k,j,i);
        for(int k=0;k<nz;k++) for(int j=0;j<ny;j++) for(int i=0;i<nx;i++) mono_apply_at(p,k,j,i);
        fs::path out=fs::path(argv[2])/dir.filename(); fs::create_directories(out);
        std::ofstream manifest(out/"MANIFEST.txt");
        put(out,manifest,"tendency",t,nz,ny,nx);
        put(out,manifest,"h_tendency",ht,nz,ny,nx,b!=0,b!=0);
        put(out,manifest,"z_tendency",zt,nz,ny,nx);
        put(out,manifest,"qmin",qmin,nz,ny,nx); put(out,manifest,"qmax",qmax,nz,ny,nx);
        put(out,manifest,"scale_in",si,nz,ny,nx); put(out,manifest,"scale_out",so,nz,ny,nx);
        std::cout<<dir.filename().string()<<"\n";
    }
}
