// SPDX-License-Identifier: Apache-2.0
// Incremental device ABI. Reuse the standalone solver's hydrostatic MUSCL
// reconstruction, HLL fluxes and shared-face positive draining limiter.
#include "surge.cu"

struct DeviceParameters {
    int nx,ny,boundary_mask;
    double dx,dy;
    float dry,manning,initial_level,boundary_level,reference_pressure,cfl;
};
struct DevicePlanes {
    const float *bed,*wet,*lat,*ca,*sa,*pressure,*tx,*ty,*hs,*tp,*direction;
    State *u,*first,*second;
    Flux *xf,*yf;
    Clock*clock;
    float *theta,*stress,*eta,*inundation;
    int*bad;
};
__global__ void device_init(DeviceParameters p,DevicePlanes a){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=p.nx*p.ny)return;
    double h=a.wet[k]>.5f?fmax(double(p.initial_level)-a.bed[k],0.):0.;
    a.u[k]={h,0,0};a.eta[k]=float(h+a.bed[k]);a.inundation[k]=a.wet[k]>.5f?0.f:float(h);
}
__global__ void device_clock(Clock*c,float dt){c->dt=dt;}
// Radiation stress divided by water density, in grid coordinates. Linear
// finite-depth dispersion and group/phase-speed ratio retain shallow-water
// setup; Hs is depth limited by the 0.78 breaking index.
__global__ void device_stress(DeviceParameters p,DevicePlanes a,const State*u){
    int k=blockIdx.x*blockDim.x+threadIdx.x,n=p.nx*p.ny;if(k>=n)return;
    float h=float(u[k].h),height=a.hs[k],period=a.tp[k];
    float xx=0,xy=0,yy=0;
    if(!isfinite(height)||!isfinite(period)||!isfinite(a.direction[k])||height<0||period<0){atomicAdd(a.bad+3,1);}
    else if(h>p.dry&&height>0&&period>0){
        height=fminf(height,.78f*h);float om=6.28318530718f/period;
        float w=fmaxf(om*om/G,om/sqrtf(G*h));
        for(int j=0;j<8;++j){float th=tanhf(w*h);w=fmaxf(1e-10f,w-(G*w*th-om*om)/(G*(th+w*h*(1-th*th))));}
        float kh=w*h,ratio=kh>20?.5f:.5f*(1+2*kh/fmaxf(sinhf(2*kh),1e-20f));
        float east=sinf(a.direction[k]*RAD),north=cosf(a.direction[k]*RAD);
        float x=a.ca[k]*east+a.sa[k]*north,y=-a.sa[k]*east+a.ca[k]*north;
        float energy=G*height*height/16;
        xx=energy*(ratio*(1+x*x)-.5f);xy=energy*ratio*x*y;yy=energy*(ratio*(1+y*y)-.5f);
    }
    a.stress[k]=xx;a.stress[n+k]=xy;a.stress[2*n+k]=yy;
}
__device__ Flux boundary_flux(State l,State r,float bed,int normal,float dry){
    double hl=fmax(l.h-bed,0.),hr=fmax(r.h-bed,0.);
    float unL=normal==0?l.x:l.y,unR=normal==0?r.x:r.y;
    if(hl<=dry)l.x=l.y=unL=0;if(hr<=dry)r.x=r.y=unR=0;
    Riemann a={hl,hl*l.x,hl*l.y},b={hr,hr*r.x,hr*r.y};
    Riemann fa={hl*unL,hl*unL*l.x+(normal==0?.5*G*hl*hl:0),hl*unL*l.y+(normal==1?.5*G*hl*hl:0)};
    Riemann fb={hr*unR,hr*unR*r.x+(normal==0?.5*G*hr*hr:0),hr*unR*r.y+(normal==1?.5*G*hr*hr:0)};
    float sl=fminf(unL-sqrtf(G*hl),unR-sqrtf(G*hr)),sr=fmaxf(unL+sqrtf(G*hl),unR+sqrtf(G*hr));
    Riemann f={0,0,0};if(sl>=0)f=fa;else if(sr<=0)f=fb;else{double inv=1./(double(sr)-sl);
        f={(sr*fa.h-sl*fb.h+sl*sr*(b.h-a.h))*inv,(sr*fa.x-sl*fb.x+sl*sr*(b.x-a.x))*inv,(sr*fa.y-sl*fb.y+sl*sr*(b.y-a.y))*inv};}
    return {f.h,f.x,f.y,0,0};
}
__global__ void device_boundary(DeviceParameters p,DevicePlanes a,const State*u){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=2*(p.nx+p.ny))return;
    bool normal=id>=2*p.ny;int q=normal?id-2*p.ny:id;
    int len=normal?p.nx:p.ny;bool left=q<len;int index=q%len;
    int k=normal?(left?0:p.ny-1)*p.nx+index:index*p.nx+(left?0:p.nx-1);
    int bit=normal?(left?4:8):(left?1:2);
    if(!(p.boundary_mask&bit)||a.wet[k]<=.5f||u[k].h<=p.dry)return;
    State inner=primitive(u[k],a.bed[k],p.dry),ghost=inner;
    float ext=p.boundary_level+(p.reference_pressure-a.pressure[k])/(RHO*G);
    ghost.h=ext;float velocity=(left?-1.f:1.f)*sqrtf(G/u[k].h)*(inner.h-ext);
    if(normal)ghost.y=velocity;else ghost.x=velocity;
    Flux f=boundary_flux(left?ghost:inner,left?inner:ghost,a.bed[k],normal,p.dry);
    if(normal)a.yf[(left?0:p.ny)*p.nx+index]=f;else a.xf[index*(p.nx+1)+(left?0:p.nx)]=f;
}
__global__ void device_draining(DeviceParameters p,DevicePlanes a,const State*u){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=p.nx*p.ny)return;int i=k%p.nx,j=k/p.nx;
    double leaving=(fmax(0.,a.xf[j*(p.nx+1)+i+1].h)+fmax(0.,-a.xf[j*(p.nx+1)+i].h))/p.dx
                  +(fmax(0.,a.yf[(j+1)*p.nx+i].h)+fmax(0.,-a.yf[j*p.nx+i].h))/p.dy;
    a.theta[k]=leaving>0?float(fmin(1.,fmax(u[k].h,0.)/(a.clock->dt*leaving)*(1-2e-7))):1.f;
}
__device__ float derivative(const float*f,int k,int stride,int index,int length,double spacing){
    int lo=index>0?k-stride:k,hi=index+1<length?k+stride:k;
    return float((f[hi]-f[lo])/((index>0&&index+1<length?2:1)*spacing));
}
__global__ void device_advance(DeviceParameters p,DevicePlanes a,const State*u,State*out,int stage){
    int k=blockIdx.x*blockDim.x+threadIdx.x,n=p.nx*p.ny;if(k>=n)return;int i=k%p.nx,j=k/p.nx;
    State v=u[k];float dt=a.clock->dt;
    Flux w=a.xf[j*(p.nx+1)+i],e=a.xf[j*(p.nx+1)+i+1],b=a.yf[j*p.nx+i],t=a.yf[(j+1)*p.nx+i];
    double dh=-(e.h-w.h)/p.dx-(t.h-b.h)/p.dy;
    double qx=-(e.x+e.pl-w.x-w.pr)/p.dx-(t.x-b.x)/p.dy;
    double qy=-(e.y-w.y)/p.dx-(t.y+t.pl-b.y-b.pr)/p.dy;
    if(v.h>p.dry){
        float px=derivative(a.pressure,k,1,i,p.nx,p.dx),py=derivative(a.pressure,k,p.nx,j,p.ny,p.dy);
        float tx=a.ca[k]*a.tx[k]+a.sa[k]*a.ty[k],ty=-a.sa[k]*a.tx[k]+a.ca[k]*a.ty[k];
        if(!isfinite(tx)||!isfinite(ty)||!isfinite(px)||!isfinite(py)){atomicAdd(a.bad+3,1);tx=ty=px=py=0;}
        float f=1.458423e-4f*sinf(a.lat[k]*RAD);
        qx+=tx/RHO-v.h/RHO*px+f*v.y-derivative(a.stress,k,1,i,p.nx,p.dx)-derivative(a.stress+n,k,p.nx,j,p.ny,p.dy);
        qy+=ty/RHO-v.h/RHO*py-f*v.x-derivative(a.stress+n,k,1,i,p.nx,p.dx)-derivative(a.stress+2*n,k,p.nx,j,p.ny,p.dy);
        double c=sqrt(G*v.h),rate=(fabs(v.x)/v.h+c)/p.dx+(fabs(v.y)/v.h+c)/p.dy;
        if(dt*rate>p.cfl*(1+1e-5))atomicAdd(a.bad+2,1);
    }
    v={v.h+dt*dh,float(v.x+dt*qx),float(v.y+dt*qy)};
    if(stage){State old=a.u[k];v={(old.h+v.h)*.5,(old.x+v.x)*.5f,(old.y+v.y)*.5f};}
    if(!isfinite(v.h)||!isfinite(v.x)||!isfinite(v.y)){atomicAdd(a.bad,1);v={0,0,0};}
    if(v.h<0){if(v.h < -1e-6)atomicAdd(a.bad+1,1);v.h=0;}
    if(v.h<=p.dry)v.x=v.y=0;
    if(stage&&v.h>p.dry){float damping=1+dt*G*p.manning*p.manning*hypotf(v.x,v.y)/powf(v.h,7.f/3.f);v.x/=damping;v.y/=damping;}
    out[k]=v;
    if(stage){a.eta[k]=float(v.h+a.bed[k]);a.inundation[k]=a.wet[k]>.5f?0.f:float(v.h);}
}
// Pointer table contains only device addresses; its host lifetime ends after
// launch arguments have been copied by the CUDA runtime. No allocation,
// synchronization, graph operation, or host field transfer occurs here.
static DevicePlanes bind(void**a){
    return {(float*)a[0],(float*)a[1],(float*)a[2],(float*)a[3],(float*)a[4],(float*)a[5],
        (float*)a[6],(float*)a[7],(float*)a[8],(float*)a[9],(float*)a[10],
        (State*)a[11],(State*)a[12],(State*)a[13],(Flux*)a[14],(Flux*)a[15],
        (Clock*)a[16],(float*)a[17],(float*)a[18],(float*)a[19],(float*)a[20],(int*)a[21]};
}
extern "C" int surge_device_init(const DeviceParameters*p,void**pointers,cudaStream_t stream){
    auto a=bind(pointers);device_init<<<(p->nx*p->ny+255)/256,256,0,stream>>>(*p,a);return cudaGetLastError();
}
extern "C" int surge_device_step(const DeviceParameters*p,void**pointers,cudaStream_t stream,float dt,int pieces){
    auto a=bind(pointers);Parameters base={};base.nx=p->nx;base.ny=p->ny;base.dry=p->dry;
    int n=p->nx*p->ny,blocks=(n+255)/256,xb=((p->nx+1)*p->ny+255)/256,yb=(p->nx*(p->ny+1)+255)/256;
    device_clock<<<1,1,0,stream>>>(a.clock,dt);
    for(int sub=0;sub<pieces;++sub){
        for(int stage=0;stage<2;++stage){
            const State*u=stage?a.first:a.u;State*out=stage?a.second:a.first;
            device_stress<<<blocks,256,0,stream>>>(*p,a,u);
            faces<<<xb,256,0,stream>>>(base,u,a.bed,nullptr,nullptr,nullptr,a.clock,a.xf,0,stage);
            faces<<<yb,256,0,stream>>>(base,u,a.bed,nullptr,nullptr,nullptr,a.clock,a.yf,1,stage);
            device_boundary<<<(2*(p->nx+p->ny)+255)/256,256,0,stream>>>(*p,a,u);
            device_draining<<<blocks,256,0,stream>>>(*p,a,u);
            limit_faces<<<xb,256,0,stream>>>(base,a.xf,a.theta,a.clock,0);
            limit_faces<<<yb,256,0,stream>>>(base,a.yf,a.theta,a.clock,1);
            device_advance<<<blocks,256,0,stream>>>(*p,a,u,out,stage);
        }
        cudaError_t rc=cudaMemcpyAsync(a.u,a.second,n*sizeof(State),cudaMemcpyDeviceToDevice,stream);
        if(rc!=cudaSuccess)return rc;
    }
    return cudaGetLastError();
}
