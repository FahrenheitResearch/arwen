// SPDX-License-Identifier: Apache-2.0
// Original terrain discretization from published equations, not a software port.
// Derivation, assumptions and citations: docs/technical/NEXTGEN-TERRAIN.md.

__device__ __forceinline__ int adjacent(int p, int d, int nx, int ny) {
    const int ox[8] = {-1,1,0,0,-1,1,-1,1};
    const int oy[8] = {0,0,-1,1,-1,-1,1,1};
    int x=p%nx+ox[d], y=p/nx+oy[d];
    return x>=0 && x<nx && y>=0 && y<ny ? y*nx+x : -1;
}

// Recharge first fills the unconfined aquifer. Saturation excess returns to
// the surface. Inputs are nonnegative water-equivalent depths/rates in SI.
__device__ __forceinline__ void stores(
    int p, float dt, const float* h, const float* w, const float* soil,
    const float* sy, const float* rain, const float* recharge,
    float &surface, float &ground, float &excess) {
    float supplied=w[p]+dt*recharge[p];
    float capacity=soil[p]*sy[p];
    excess=fmaxf(supplied-capacity,0.f);
    ground=fminf(supplied,capacity);
    surface=h[p]+dt*rain[p]+excess;
}

extern "C" __global__ void terrain_flux(
    int nx, int ny, float dx, float dt,
    const float* h, const float* w, const float* z, const float* soil,
    const float* sy, const float* manning, const float* conductivity,
    const int* active, const float* rain, const float* recharge,
    const float* outlet_stage, const float* outlet_width,
    float* flux, unsigned int* health) {
    int p=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny;
    if(p>=n) return;
    #pragma unroll
    for(int d=0;d<14;++d) flux[d*n+p]=0.f;
    if(active[p]==0) return;
    if(active[p]!=1 || !isfinite(h[p]) || h[p]<0.f ||
       !isfinite(w[p]) || w[p]<0.f || !isfinite(z[p]) ||
       !isfinite(soil[p]) || soil[p]<=0.f || !isfinite(sy[p]) ||
       sy[p]<=0.f || sy[p]>1.f || !isfinite(manning[p]) ||
       manning[p]<=0.f || !isfinite(conductivity[p]) || conductivity[p]<0.f ||
       !isfinite(rain[p]) || rain[p]<0.f || !isfinite(recharge[p]) ||
       recharge[p]<0.f || !isfinite(outlet_width[p]) || outlet_width[p]<0.f ||
       !isfinite(outlet_stage[p])) {
        atomicOr(health,1u); return;
    }
    float hs,ws,ex;
    stores(p,dt,h,w,soil,sy,rain,recharge,hs,ws,ex);
    float surface_head=z[p]+hs, bottom=z[p]-soil[p];
    float ground_head=bottom+ws/sy[p];
    float q[14]={0.f};
    // Surface D4 face flux. q is donor water-depth loss rate [m/s].
    #pragma unroll
    for(int d=0;d<4;++d) {
        int j=adjacent(p,d,nx,ny);
        if(j<0 || active[j]!=1) continue; // sealed exterior / inactive cells
        float hj,wj,ej;
        stores(j,dt,h,w,soil,sy,rain,recharge,hj,wj,ej);
        float drop=surface_head-(z[j]+hj);
        float depth=fmaxf(surface_head-fmaxf(z[p],z[j]),0.f);
        if(drop>0.f && depth>0.f) {
            float nf=0.5f*(manning[p]+manning[j]);
            q[d]=(powf(depth,5.f/3.f)/nf)*sqrtf(drop/dx)/dx;
        }
    }
    // Subsurface D8: route to the largest positive hydraulic gradient.
    // T=K*b, harmonic K at the interface; b cannot cross either bedrock face.
    int selected=-1;
    float best=0.f, transmissivity=0.f;
    #pragma unroll
    for(int d=0;d<8;++d) {
        int j=adjacent(p,d,nx,ny);
        if(j<0 || active[j]!=1) continue;
        float hj,wj,ej;
        stores(j,dt,h,w,soil,sy,rain,recharge,hj,wj,ej);
        float btm=z[j]-soil[j];
        float headj=btm+wj/sy[j];
        float distance=dx*(d<4 ? 1.f : 1.4142135623730951f);
        float gradient=(ground_head-headj)/distance;
        float thickness=fmaxf(fminf(ground_head,z[p])-fmaxf(bottom,btm),0.f);
        float ki=conductivity[p], kj=conductivity[j];
        float k=ki>0.f && kj>0.f ? (2.f*ki*kj)/(ki+kj) : 0.f;
        if(gradient>best && thickness>0.f && k>0.f) {
            selected=d; best=gradient; transmissivity=k*thickness;
        }
    }
    if(selected>=0) q[5+selected]=transmissivity*best/dx;
    // Explicit one-way channel/drain boundary, length dx/2, width in metres.
    // The receiving stage is supplied by the coupled channel component.
    float width=outlet_width[p], stage=outlet_stage[p];
    if(width>0.f) {
        float depth=fmaxf(surface_head-fmaxf(z[p],stage),0.f);
        if(depth>0.f)
            q[4]=(width/(dx*dx))*(powf(depth,5.f/3.f)/manning[p])*
                 sqrtf((surface_head-stage)/(0.5f*dx));
        float thickness=fmaxf(fminf(ground_head,z[p])-bottom,0.f);
        if(ground_head>stage)
            q[13]=(width/(dx*dx))*conductivity[p]*thickness*
                  ((ground_head-stage)/(0.5f*dx));
    }
    // Double accumulation and downward float conversion make the total exported
    // depth <= available storage, even when the donor limiter is active.
    double sum_s=0.,sum_g=0.;
    #pragma unroll
    for(int d=0;d<5;++d) sum_s+=(double)q[d];
    #pragma unroll
    for(int d=5;d<14;++d) sum_g+=(double)q[d];
    double fs=sum_s>0. ? fmin(1.,(double)hs/((double)dt*sum_s)) : 1.;
    double fg=sum_g>0. ? fmin(1.,(double)ws/((double)dt*sum_g)) : 1.;
    #pragma unroll
    for(int d=0;d<14;++d) {
        double limited=(double)q[d]*(d<5 ? fs : fg);
        if(!isfinite(limited)) { atomicOr(health,2u); limited=0.; }
        flux[d*n+p]=__double2float_rz(limited);
    }
}

extern "C" __global__ void terrain_update(
    int nx, int ny, float dx, float dt, float* h, float* w,
    const float* soil, const float* sy, const int* active,
    const float* rain, const float* recharge, const float* flux,
    double* surface_outlet_volume, double* ground_outlet_volume,
    double* exfiltration_volume, double* surface_interval_volume,
    double* ground_interval_volume, int reset_interval, unsigned int* health) {
    int p=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny;
    if(p>=n) return;
    if(reset_interval) {
        surface_interval_volume[p]=0.;
        ground_interval_volume[p]=0.;
    }
    if(active[p]!=1) return;
    float hs,ws,ex;
    stores(p,dt,h,w,soil,sy,rain,recharge,hs,ws,ex);
    const int opposite[8]={1,0,3,2,7,6,5,4};
    double outgoing_s=0.,outgoing_g=0.,incoming_s=0.,incoming_g=0.;
    #pragma unroll
    for(int d=0;d<5;++d) outgoing_s+=(double)flux[d*n+p];
    #pragma unroll
    for(int d=5;d<14;++d) outgoing_g+=(double)flux[d*n+p];
    #pragma unroll
    for(int d=0;d<8;++d) {
        int j=adjacent(p,d,nx,ny);
        if(j<0 || active[j]!=1) continue;
        if(d<4) incoming_s+=(double)flux[opposite[d]*n+j];
        incoming_g+=(double)flux[(5+opposite[d])*n+j];
    }
    double next_s=((double)hs-(double)dt*outgoing_s)+(double)dt*incoming_s;
    double next_g=((double)ws-(double)dt*outgoing_g)+(double)dt*incoming_g;
    double capacity=(double)(soil[p]*sy[p]);
    double spill=fmax(next_g-capacity,0.);
    next_g-=spill; next_s+=spill;
    if(next_s<0. || next_g<0. || !isfinite(next_s) || !isfinite(next_g))
        atomicOr(health,4u);
    h[p]=(float)next_s; w[p]=(float)next_g;
    double area=(double)dx*(double)dx;
    double surface_export=(double)dt*(double)flux[4*n+p]*area;
    double ground_export=(double)dt*(double)flux[13*n+p]*area;
    surface_outlet_volume[p]+=surface_export;
    ground_outlet_volume[p]+=ground_export;
    surface_interval_volume[p]+=surface_export;
    ground_interval_volume[p]+=ground_export;
    exfiltration_volume[p]+=((double)ex+spill)*area;
}

// Hydrofabric owns the sealed inverse channel-cell map. One thread per node
// sums ascending flattened cells in f64, then rounds the interval mean to f32.
// This produces an isolated contribution; the receiver owns composition and
// reservoir inflow policy. It never reads cumulative diagnostics.
extern "C" __global__ void terrain_lateral(
    int nodes, int cells, long long entries, double period,
    const long long* offsets, const int* indices,
    const double* surface_interval, const double* ground_interval,
    float* lateral, unsigned int* health) {
    int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=nodes) return;
    long long begin=offsets[node], end=offsets[node+1];
    if(begin<0 || end<begin || end>entries ||
       (node==0 && begin!=0) || (node==nodes-1 && end!=entries)) {
        atomicOr(health,8u); lateral[node]=0.f; return;
    }
    double volume=0.;
    int previous=-1;
    for(long long k=begin;k<end;++k) {
        int p=indices[k];
        if(p<=previous || p>=cells) {
            atomicOr(health,8u); lateral[node]=0.f; return;
        }
        previous=p;
        volume+=surface_interval[p];
        volume+=ground_interval[p];
    }
    float rate=(float)(volume/period);
    if(!isfinite(rate) || rate<0.f) atomicOr(health,16u);
    lateral[node]=rate;
}
