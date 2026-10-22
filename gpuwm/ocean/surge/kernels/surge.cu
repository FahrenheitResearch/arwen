// SPDX-License-Identifier: Apache-2.0
// Original regular-grid finite-volume SWE implementation. No ADCIRC code.
// MUSCL hydrostatic reconstruction, HLL flux, SSPRK2, spherical metrics,
// wet/dry fronts, air stress, atmospheric pressure, Coriolis, Manning drag.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>

struct Parameters {
    int nx,ny,fx,fy,nt,ng,boundary_mask;
    double west,south,dlon,dlat,fwest,fsouth,fdlon,fdlat,duration,max_dt,output_interval;
    float dry,manning,cfl,reference_pressure,initial_level;
};
struct State { double h; float x,y; };
struct Flux { double h,x,y,pl,pr; };
struct Riemann { double h,x,y; };
struct Clock { double t,stop; float dt,wave; unsigned long long steps; int bad; float worst_negative; double boundary,roundoff; };
struct Statistics { double volume; float max_eta,max_speed,min_h; unsigned long long wet_land; };
constexpr float G=9.80665f,RHO=1025.f,AIR=1.225f;
constexpr double RAD=0.017453292519943295,EARTH=6371000.;
#define CU(call) do { cudaError_t rc=(call);if(rc!=cudaSuccess)throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(rc)); } while(0)
__device__ float mm(float a,float b){return a*b<=0.f?0.f:copysignf(fminf(fabsf(a),fabsf(b)),a);}
__device__ State primitive(State a,float b,float dry){float inv=a.h>dry?1.f/a.h:0.f;return {a.h+b,a.x*inv,a.y*inv};}
__device__ double latitude(Parameters p,int j){return (p.south+j*p.dlat)*RAD;}
__device__ double dx(Parameters p,int j){return EARTH*p.dlon*RAD*cos(latitude(p,j));}
__device__ float tide(Parameters p,const double*times,const double*values,double t){
    int lo=0,hi=p.nt-1;while(hi-lo>1){int mid=(hi+lo)/2;if(times[mid]<=t)lo=mid;else hi=mid;}
    float w=float(fmax(0.,fmin(1.,(t-times[lo])/(times[hi]-times[lo]))));return float(values[lo]*(1-w)+values[hi]*w);
}
__device__ float sample(Parameters p,const float*f,const double*times,int component,double lon,double lat,double t){
    double xx=fmax(0.,fmin(p.fx-1.000001,(lon-p.fwest)/p.fdlon));
    double yy=fmax(0.,fmin(p.fy-1.000001,(lat-p.fsouth)/p.fdlat));int i=int(xx),j=int(yy);float x=xx-i,y=yy-j;
    int lo=0,hi=p.nt-1;while(hi-lo>1){int mid=(hi+lo)/2;if(times[mid]<=t)lo=mid;else hi=mid;}
    float w=float(fmax(0.,fmin(1.,(t-times[lo])/(times[hi]-times[lo]))));
    int n=p.fx*p.fy;const float*a=f+(lo*3+component)*n,*b=f+(hi*3+component)*n;
    int k=j*p.fx+i;
    float va=(1-y)*((1-x)*a[k]+x*a[k+1])+y*((1-x)*a[k+p.fx]+x*a[k+p.fx+1]);
    float vb=(1-y)*((1-x)*b[k]+x*b[k+1])+y*((1-x)*b[k+p.fx]+x*b[k+p.fx+1]);return (1-w)*va+w*vb;
}
__global__ void init(Parameters p,const float*bed,State*u,float*maxeta,float*maxdepth){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=p.nx*p.ny)return;
    double h=fmax(double(p.initial_level)-bed[k],0.);u[k]={h,0,0};maxeta[k]=h>p.dry?0.f:-INFINITY;
    maxdepth[k]=bed[k]>=p.initial_level?0.f:NAN;
}
__global__ void reset_wave(Clock*c){c->wave=0;}
__global__ void wave(Parameters p,const State*u,Clock*c){
    __shared__ float v[256];int k=blockIdx.x*blockDim.x+threadIdx.x;float x=0;
    if(k<p.nx*p.ny && c->t<c->stop){State a=u[k];float inv=a.h>p.dry?1.f/a.h:0.f;float speed=sqrtf(G*fmaxf(a.h,0.f));
      x=(fabsf(a.x)*inv+speed)/float(dx(p,k/p.nx))+(fabsf(a.y)*inv+speed)/float(EARTH*p.dlat*RAD);}
    v[threadIdx.x]=x;__syncthreads();for(int s=128;s;s>>=1){if(threadIdx.x<s)v[threadIdx.x]=fmaxf(v[threadIdx.x],v[threadIdx.x+s]);__syncthreads();}
    if(threadIdx.x==0)atomicMax(reinterpret_cast<int*>(&c->wave),__float_as_int(v[0]));
}
__global__ void timestep(Parameters p,Clock*c){c->dt=float(fmin(c->stop-c->t,fmin(p.max_dt,double(p.cfl/fmaxf(c->wave,1e-12f)))));}
__device__ State reconstruct(Parameters p,const State*u,const float*bed,int k,int normal,float side){
    State a=primitive(u[k],bed[k],p.dry);int i=k%p.nx,j=k/p.nx;
    if((normal==0&&(i==0||i==p.nx-1))||(normal==1&&(j==0||j==p.ny-1))||u[k].h<=p.dry)return a;
    int stride=normal==0?1:p.nx; if(u[k-stride].h<=p.dry||u[k+stride].h<=p.dry)return a;
    State l=primitive(u[k-stride],bed[k-stride],p.dry),r=primitive(u[k+stride],bed[k+stride],p.dry);
    float se=mm(a.h-l.h,r.h-a.h);se=copysignf(fminf(fabsf(se),2*u[k].h),se);
    return {a.h+side*0.5f*se,a.x+side*0.5f*mm(a.x-l.x,r.x-a.x),a.y+side*0.5f*mm(a.y-l.y,r.y-a.y)};
}
__global__ void faces(Parameters p,const State*u,const float*bed,const float*forcing,const double*times,const double*tides,Clock*c,Flux*flux,int normal,int stage){
    if(c->dt<=0)return;int id=blockIdx.x*blockDim.x+threadIdx.x;
    int count=normal==0?(p.nx+1)*p.ny:p.nx*(p.ny+1);if(id>=count)return;
    int i=normal==0?id%(p.nx+1):id%p.nx,j=normal==0?id/(p.nx+1):id/p.nx;
    int lk=normal==0?j*p.nx+i-1:(j-1)*p.nx+i,rk=normal==0?j*p.nx+i:j*p.nx+i;
    bool left=normal==0?i==0:j==0,right=normal==0?i==p.nx:j==p.ny;
    if(left)lk=rk;if(right)rk=lk;
    State l=reconstruct(p,u,bed,lk,normal,1),r=reconstruct(p,u,bed,rk,normal,-1);
    float bl=bed[lk],br=bed[rk];
    if(left||right){
        State inner=primitive(u[left?rk:lk],bed[left?rk:lk],p.dry);l=r=inner;
        int bit=normal==0?(left?1:2):(left?4:8);
        State ghost=inner;
        if((p.boundary_mask&bit)&&u[left?rk:lk].h>p.dry){
            double t=c->t+(stage?c->dt:0);int cell=left?rk:lk;
            double lat=p.south+(cell/p.nx)*p.dlat,lon=p.west+(cell%p.nx)*p.dlon;
            float pressure=sample(p,forcing,times,2,lon,lat,t);
            float ext=p.initial_level+tide(p,times,tides,t)+(p.reference_pressure-pressure)/(RHO*G);
            float outward=(left?-1.f:1.f)*sqrtf(G/fmaxf(u[cell].h,p.dry))*(inner.h-ext);
            ghost.h=ext;
            if(normal==0)ghost.x=outward;else ghost.y=outward;
        }else{if(normal==0)ghost.x=-inner.x;else ghost.y=-inner.y;}
        if(left)l=ghost;else r=ghost;
    }
    float top=fmaxf(bl,br);double hl=fmax(l.h-bl,0.),hr=fmax(r.h-br,0.);
    double hL=fmax(l.h-top,0.),hR=fmax(r.h-top,0.);
    float unL=normal==0?l.x:l.y,unR=normal==0?r.x:r.y;
    if(hL<=p.dry)l.x=l.y=unL=0;if(hR<=p.dry)r.x=r.y=unR=0;
    Riemann a={hL,hL*l.x,hL*l.y},b={hR,hR*r.x,hR*r.y};
    Riemann fa={hL*unL,hL*unL*l.x+(normal==0?0.5f*G*hL*hL:0),hL*unL*l.y+(normal==1?0.5f*G*hL*hL:0)};
    Riemann fb={hR*unR,hR*unR*r.x+(normal==0?0.5f*G*hR*hR:0),hR*unR*r.y+(normal==1?0.5f*G*hR*hR:0)};
    float sl=fminf(unL-sqrtf(G*hL),unR-sqrtf(G*hR)),sr=fmaxf(unL+sqrtf(G*hL),unR+sqrtf(G*hR));
    Riemann f={0,0,0};if(sl>=0)f=fa;else if(sr<=0)f=fb;else {double inv=1./(double(sr)-sl);f={(sr*fa.h-sl*fb.h+sl*sr*(b.h-a.h))*inv,(sr*fa.x-sl*fb.x+sl*sr*(b.x-a.x))*inv,(sr*fa.y-sl*fb.y+sl*sr*(b.y-a.y))*inv};}
    flux[id]={f.h,f.x,f.y,0.5f*G*(hl*hl-hL*hL),0.5f*G*(hr*hr-hR*hR)};
}
// Donor-cell draining limiter keeps both RK stages positive at moving shores.
// One shared face flux is scaled, preserving mass exchanged between cells.
__global__ void draining(Parameters p,const State*u,const Flux*xf,const Flux*yf,Clock*c,float*theta){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=p.nx*p.ny||c->dt<=0)return;int i=k%p.nx,j=k/p.nx;
    double lat=latitude(p,j);float n=cos(lat+0.5*p.dlat*RAD)/cos(lat),s=cos(lat-0.5*p.dlat*RAD)/cos(lat);
    float leaving=(fmaxf(0,xf[j*(p.nx+1)+i+1].h)+fmaxf(0,-xf[j*(p.nx+1)+i].h))/float(dx(p,j))+
       (n*fmaxf(0,yf[(j+1)*p.nx+i].h)+s*fmaxf(0,-yf[j*p.nx+i].h))/float(EARTH*p.dlat*RAD);
    theta[k]=leaving>0?fminf(1.f,fmaxf(u[k].h,0.f)/(c->dt*leaving)*(1.f-2e-7f)):1.f;
}
__global__ void limit_faces(Parameters p,Flux*f,const float*theta,Clock*c,int normal){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(c->dt<=0)return;
    int count=normal==0?(p.nx+1)*p.ny:p.nx*(p.ny+1);if(id>=count)return;
    int i=normal==0?id%(p.nx+1):id%p.nx,j=normal==0?id/(p.nx+1):id/p.nx;
    int donor=normal==0?(f[id].h>=0?i-1:i):(f[id].h>=0?j-1:j);
    int length=normal==0?p.nx:p.ny;if(donor<0||donor>=length)return;
    int k=normal==0?j*p.nx+donor:donor*p.nx+i;float scale=theta[k];
    f[id].h*=scale;f[id].x*=scale;f[id].y*=scale;f[id].pl*=scale;f[id].pr*=scale;
}
__global__ void advance(Parameters p,const State*u,const State*initial,State*out,const float*bed,const float*forcing,const double*times,const double*tides,Clock*c,const Flux*xf,const Flux*yf,int stage,float*maxeta,float*maxdepth){
    if(c->dt<=0)return;int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=p.nx*p.ny)return;int i=k%p.nx,j=k/p.nx;
    State a=u[k];float inv=a.h>p.dry?1.f/a.h:0.f;float ux=a.x*inv,uy=a.y*inv;
    double lat=latitude(p,j),cosc=cos(lat),cn=cos(lat+0.5*p.dlat*RAD),cs=cos(lat-0.5*p.dlat*RAD);
    float ddx=float(dx(p,j)),ddy=float(EARTH*p.dlat*RAD),n=float(cn/cosc),s=float(cs/cosc);
    Flux w=xf[j*(p.nx+1)+i],e=xf[j*(p.nx+1)+i+1],b=yf[j*p.nx+i],t=yf[(j+1)*p.nx+i];
    double dh=-(e.h-w.h)/ddx-(n*t.h-s*b.h)/ddy;
    float qx=-(e.x+e.pl-w.x-w.pr)/ddx-(n*t.x-s*b.x)/ddy;
    float qy=-(e.y-w.y)/ddx-(n*(t.y+t.pl)-s*(b.y+b.pr))/ddy+0.5f*G*a.h*a.h*(n-s)/ddy;
    double now=c->t+(stage?c->dt:0);double lon=p.west+i*p.dlon,latdeg=p.south+j*p.dlat;
    float wu=sample(p,forcing,times,0,lon,latdeg,now),wv=sample(p,forcing,times,1,lon,latdeg,now);
    float speed=hypotf(wu,wv),cd=fminf(0.0025f,(0.75f+0.067f*speed)*0.001f);
    float px=(sample(p,forcing,times,2,lon+p.dlon,latdeg,now)-sample(p,forcing,times,2,lon-p.dlon,latdeg,now))/(2*ddx);
    float py=(sample(p,forcing,times,2,lon,latdeg+p.dlat,now)-sample(p,forcing,times,2,lon,latdeg-p.dlat,now))/(2*ddy);
    float coriolis=float(1.458423e-4*sin(lat)),curvature=float(tan(lat)/EARTH);
    if(a.h>p.dry){qx+=AIR/RHO*cd*speed*wu-a.h/RHO*px+coriolis*a.y+a.h*ux*uy*curvature;
                   qy+=AIR/RHO*cd*speed*wv-a.h/RHO*py-coriolis*a.x-a.h*ux*ux*curvature;}
    State v={a.h+c->dt*dh,a.x+c->dt*qx,a.y+c->dt*qy};
    if(stage){State old=initial[k];v={(old.h+v.h)*0.5f,(old.x+v.x)*0.5f,(old.y+v.y)*0.5f};}
    if(!isfinite(v.h)||!isfinite(v.x)||!isfinite(v.y)){atomicAdd(&c->bad,1);v={0,0,0};}
    if(v.h<0){atomicMax(reinterpret_cast<int*>(&c->worst_negative),__float_as_int(-v.h));
        if(v.h < -1e-4f)atomicAdd(&c->bad,1);atomicAdd(&c->roundoff,(stage?1.:0.5)*double(-v.h)*ddx*ddy);v.h=0;}
    if(v.h<=p.dry)v.x=v.y=0;
    if(stage&&v.h>p.dry){float damping=1+c->dt*G*p.manning*p.manning*hypotf(v.x,v.y)/powf(v.h,7.f/3.f);v.x/=damping;v.y/=damping;}
    out[k]=v;
    double bv=0;if(i==0)bv-=w.h*ddy;if(i==p.nx-1)bv+=e.h*ddy;
    if(j==0)bv-=b.h*ddx*s;if(j==p.ny-1)bv+=t.h*ddx*n;
    if(bv!=0)atomicAdd(&c->boundary,0.5*c->dt*bv);
    if(stage){if(v.h>p.dry)maxeta[k]=fmaxf(maxeta[k],v.h+bed[k]-p.initial_level-tide(p,times,tides,c->t+c->dt));
        if(bed[k]>=p.initial_level)maxdepth[k]=fmaxf(maxdepth[k],v.h);}
}
__global__ void tick(Clock*c){if(c->dt>0){c->t=fmin(c->stop,c->t+c->dt);++c->steps;}}
__global__ void extract(Parameters p,const State*u,const float*bed,float*eta){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<p.nx*p.ny)eta[k]=u[k].h>p.dry?u[k].h+bed[k]:NAN;}
__global__ void gauge(Parameters p,const State*u,const float*bed,const int*cells,const double*times,const double*tides,Clock*c,float*values){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i<p.ng){int k=cells[i];values[i]=u[k].h>p.dry?u[k].h+bed[k]-p.initial_level-tide(p,times,tides,c->t):NAN;}}
__global__ void statistics(Parameters p,const State*u,const float*bed,Statistics*stats){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=p.nx*p.ny)return;State a=u[k];
    atomicAdd(&stats->volume,double(a.h)*dx(p,k/p.nx)*EARTH*p.dlat*RAD);
    atomicMin(reinterpret_cast<int*>(&stats->min_h),__float_as_int(a.h));
    if(a.h>p.dry){atomicMax(reinterpret_cast<int*>(&stats->max_speed),__float_as_int(hypotf(a.x,a.y)/a.h));
        // float atomic max ordered via integer only for nonnegative anomaly here.
        atomicMax(reinterpret_cast<int*>(&stats->max_eta),__float_as_int(fmaxf(0,a.h+bed[k]-p.initial_level)));}
    if(bed[k]>=p.initial_level&&a.h>p.dry)atomicAdd(&stats->wet_land,1ULL);
}
__global__ void mask_maxima(Parameters p,float*maxeta){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<p.nx*p.ny&&!isfinite(maxeta[k]))maxeta[k]=NAN;}

struct Buffers {
    std::vector<void*> list;size_t bytes=0;
    template<class T>T*get(size_t n){T*p;CU(cudaMalloc(&p,n*sizeof(T)));list.push_back(p);bytes+=n*sizeof(T);return p;}
    ~Buffers(){for(void*p:list)cudaFree(p);}
};
static void dump(const std::string&path,const std::vector<float>&v){std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open "+path);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));if(!f)throw std::runtime_error("write failed "+path);}
extern "C" int surge_abi(){return 1;}
extern "C" int surge_run(const Parameters*params,const float*bed_host,const float*forcing_host,const double*times_host,const double*tides_host,const int*cells_host,const char*folder,char*error,int error_size){
    try{
        auto begin=std::chrono::steady_clock::now();Parameters p=*params;
        static std::once_flag scheduling;std::call_once(scheduling,[](){CU(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));});
        if(p.nx<3||p.ny<3||p.nt<2||p.fx<2||p.fy<2||p.ng<0||p.ng>100000||p.duration<=0||p.output_interval<=0)throw std::runtime_error("invalid native surge dimensions/times");
        int count=p.nx*p.ny,blocks=(count+255)/256;Buffers mem;
        auto bed=mem.get<float>(count),forcing=mem.get<float>(size_t(p.nt)*3*p.fx*p.fy);
        auto times=mem.get<double>(p.nt),tides=mem.get<double>(p.nt);auto cells=mem.get<int>(std::max(p.ng,1));
        auto u=mem.get<State>(count),first=mem.get<State>(count),second=mem.get<State>(count);
        auto xf=mem.get<Flux>(size_t(p.nx+1)*p.ny),yf=mem.get<Flux>(size_t(p.nx)*(p.ny+1));
        auto maxeta=mem.get<float>(count),maxdepth=mem.get<float>(count),eta=mem.get<float>(count),gauges=mem.get<float>(std::max(p.ng,1));
        auto theta=mem.get<float>(count);auto clock=mem.get<Clock>(1);auto stats=mem.get<Statistics>(1);
        CU(cudaMemcpy(bed,bed_host,count*sizeof(float),cudaMemcpyHostToDevice));
        CU(cudaMemcpy(forcing,forcing_host,size_t(p.nt)*3*p.fx*p.fy*sizeof(float),cudaMemcpyHostToDevice));
        CU(cudaMemcpy(times,times_host,p.nt*sizeof(double),cudaMemcpyHostToDevice));CU(cudaMemcpy(tides,tides_host,p.nt*sizeof(double),cudaMemcpyHostToDevice));
        CU(cudaMemcpy(cells,cells_host,std::max(p.ng,1)*sizeof(int),cudaMemcpyHostToDevice));
        Clock host={};CU(cudaMemcpy(clock,&host,sizeof(host),cudaMemcpyHostToDevice));
        init<<<blocks,256>>>(p,bed,u,maxeta,maxdepth);CU(cudaDeviceSynchronize());
        auto getstats=[&](){Statistics zero={};zero.min_h=INFINITY;CU(cudaMemcpy(stats,&zero,sizeof(zero),cudaMemcpyHostToDevice));
            statistics<<<blocks,256>>>(p,u,bed,stats);CU(cudaMemcpy(&zero,stats,sizeof(zero),cudaMemcpyDeviceToHost));return zero;};
        Statistics initial=getstats();cudaStream_t stream;CU(cudaStreamCreate(&stream));
        auto advance_batch=[&](){
        for(int batch=0;batch<64;++batch){
            reset_wave<<<1,1,0,stream>>>(clock);wave<<<blocks,256,0,stream>>>(p,u,clock);timestep<<<1,1,0,stream>>>(p,clock);
            int xb=((p.nx+1)*p.ny+255)/256,yb=(p.nx*(p.ny+1)+255)/256;
            faces<<<xb,256,0,stream>>>(p,u,bed,forcing,times,tides,clock,xf,0,0);faces<<<yb,256,0,stream>>>(p,u,bed,forcing,times,tides,clock,yf,1,0);
            draining<<<blocks,256,0,stream>>>(p,u,xf,yf,clock,theta);
            limit_faces<<<xb,256,0,stream>>>(p,xf,theta,clock,0);limit_faces<<<yb,256,0,stream>>>(p,yf,theta,clock,1);
            advance<<<blocks,256,0,stream>>>(p,u,u,first,bed,forcing,times,tides,clock,xf,yf,0,maxeta,maxdepth);
            faces<<<xb,256,0,stream>>>(p,first,bed,forcing,times,tides,clock,xf,0,1);faces<<<yb,256,0,stream>>>(p,first,bed,forcing,times,tides,clock,yf,1,1);
            draining<<<blocks,256,0,stream>>>(p,first,xf,yf,clock,theta);
            limit_faces<<<xb,256,0,stream>>>(p,xf,theta,clock,0);limit_faces<<<yb,256,0,stream>>>(p,yf,theta,clock,1);
            advance<<<blocks,256,0,stream>>>(p,first,u,second,bed,forcing,times,tides,clock,xf,yf,1,maxeta,maxdepth);
            CU(cudaMemcpyAsync(u,second,count*sizeof(State),cudaMemcpyDeviceToDevice,stream));tick<<<1,1,0,stream>>>(clock);
        }
        };
        auto initialized=std::chrono::steady_clock::now();
        std::string root=folder;std::ofstream series(root+"/gauges.csv"),diagnostics(root+"/diagnostics.csv");
        series<<std::setprecision(12)<<"seconds";for(int i=0;i<p.ng;++i)series<<",gauge_"<<i;series<<"\n";
        diagnostics<<std::setprecision(15)<<"seconds,volume_m3,boundary_outflow_m3,roundoff_added_m3,mass_balance_error_m3,max_eta_m,max_speed_ms,min_depth_m,wet_land_cells,steps\n";
        std::vector<float> buffer(count),gv(p.ng);std::vector<double> frames;
        double next_frame=0,next_gauge=0,max_eta=0,max_speed=0,min_h=INFINITY,max_mass_relative=0;unsigned long long peak_wet_land=0;
        auto frame=[&](){extract<<<blocks,256>>>(p,u,bed,eta);CU(cudaMemcpy(buffer.data(),eta,count*sizeof(float),cudaMemcpyDeviceToHost));
            std::ostringstream name;name<<root<<"/water-level-"<<std::setfill('0')<<std::setw(4)<<frames.size()<<".f32";dump(name.str(),buffer);frames.push_back(host.t);};
        frame();
        while(true){
            if(host.t>=next_gauge-1e-7||host.t>=p.duration-1e-7){
                if(p.ng){gauge<<<(p.ng+255)/256,256>>>(p,u,bed,cells,times,tides,clock,gauges);CU(cudaMemcpy(gv.data(),gauges,p.ng*sizeof(float),cudaMemcpyDeviceToHost));}
                series<<host.t;for(float x:gv)series<<","<<x;series<<"\n";series.flush();
                Statistics s=getstats();double balance=s.volume-initial.volume+host.boundary-host.roundoff;
                max_eta=std::max(max_eta,double(s.max_eta));max_speed=std::max(max_speed,double(s.max_speed));min_h=std::min(min_h,double(s.min_h));peak_wet_land=std::max(peak_wet_land,s.wet_land);
                max_mass_relative=std::max(max_mass_relative,std::abs(balance)/std::max(initial.volume,1.));
                diagnostics<<host.t<<","<<s.volume<<","<<host.boundary<<","<<host.roundoff<<","<<balance<<","<<s.max_eta<<","<<s.max_speed<<","<<s.min_h<<","<<s.wet_land<<","<<host.steps<<"\n";diagnostics.flush();
                std::fprintf(stdout,"SURGE t=%.0f steps=%llu max_eta=%.3f max_speed=%.3f wet_land=%llu mass_rel=%.3g\n",host.t,host.steps,s.max_eta,s.max_speed,s.wet_land,std::abs(balance)/std::max(initial.volume,1.));std::fflush(stdout);
                next_gauge+=600.;
            }
            if(host.t>=p.duration-1e-7)break;
            next_frame=std::min(p.duration,(std::floor(host.t/p.output_interval+1e-8)+1)*p.output_interval);
            host.stop=std::min(p.duration,std::min(next_frame,next_gauge));
            CU(cudaMemcpy(clock,&host,sizeof(host),cudaMemcpyHostToDevice));
            while(host.t<host.stop-1e-8){advance_batch();CU(cudaMemcpyAsync(&host,clock,sizeof(host),cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
                if(host.bad)throw std::runtime_error("nonfinite state or negative depth >1e-4 m; worst="+std::to_string(host.worst_negative));}
            if(host.t>=next_frame-1e-7)frame();
        }
        mask_maxima<<<blocks,256>>>(p,maxeta);CU(cudaMemcpy(buffer.data(),maxeta,count*sizeof(float),cudaMemcpyDeviceToHost));dump(root+"/max-surge.f32",buffer);
        CU(cudaMemcpy(buffer.data(),maxdepth,count*sizeof(float),cudaMemcpyDeviceToHost));dump(root+"/max-inundation.f32",buffer);
        std::ofstream fj(root+"/frames.json");fj<<"[";for(size_t i=0;i<frames.size();++i){if(i)fj<<",";fj<<frames[i];}fj<<"]\n";
        cudaDeviceProp prop;int device;CU(cudaGetDevice(&device));CU(cudaGetDeviceProperties(&prop,device));
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        double setup=std::chrono::duration<double>(initialized-begin).count();
        std::ofstream receipt(root+"/native-receipt.json");receipt<<std::setprecision(12)
          <<"{\"schema\":\"gpuwm.surge-native.v1\",\"solver\":\"MUSCL hydrostatic HLL SSPRK2\",\"gpu\":\""<<prop.name
          <<"\",\"allocated_device_bytes\":"<<mem.bytes<<",\"duration_seconds\":"<<host.t<<",\"steps\":"<<host.steps
          <<",\"frames\":"<<frames.size()<<",\"native_wall_seconds\":"<<wall<<",\"setup_seconds\":"<<setup
          <<",\"max_water_level_anomaly_m\":"<<max_eta<<",\"max_speed_ms\":"<<max_speed<<",\"min_depth_m\":"<<min_h<<",\"max_relative_mass_error\":"<<max_mass_relative
          <<",\"roundoff_added_m3\":"<<host.roundoff<<",\"worst_negative_depth_m\":"<<host.worst_negative
          <<",\"peak_wet_land_cells\":"<<peak_wet_land<<",\"cuda_graphs\":false,\"parallelism\":\"GPU CUDA stream, 64 explicit native steps per batch; host I/O every 600 simulated seconds; graphs disabled\"}\n";
        CU(cudaStreamDestroy(stream));return 0;
    }catch(const std::exception&e){std::snprintf(error,error_size,"%s",e.what());return 1;}
}
