// SPDX-License-Identifier: Apache-2.0
// Derived from NOAA-OWP/t-route at 12a8eae0cdfed437143c590659fa7077605a5e70,
// src/kernel/muskingum/MCsingleSegStime_f2py_NOLOOP.f90.
// Modified for CUDA by ArWen contributors, 2026-10-17.
// Adjacent t-route-LICENSE.txt and t-route-TERMS.txt retain upstream terms.
// No withdrawn WRF-Hydro source or port is used here.
// GPUWM_MC_STRICT uses the existing glibc float32 power implementation.
#ifdef GPUWM_MC_STRICT
#define mc_pow gfk_pow
#else
#define mc_pow powf
#endif

struct MCGeometry { float twl, radius, area, flood_area, perimeter, flood_perimeter, below, above; };
__device__ MCGeometry mc_geometry(float h,float bfd,float bw,float twcc,float z) {
    MCGeometry g;
    g.twl=bw+2.0f*z*h;
    g.above=fmaxf(h-bfd,0.0f); g.below=fminf(bfd,h);
    if(g.above>0.0f && twcc<=0.0f) { g.above=0.0f; g.below=h; }
    g.area=(bw+g.below*z)*g.below;
    g.perimeter=bw+2.0f*g.below*sqrtf(1.0f+z*z);
    g.flood_area=twcc*g.above;
    g.flood_perimeter=g.above>0.0f ? twcc+2.0f*g.above : 0.0f;
    g.radius=(g.area+g.flood_area)/(g.perimeter+g.flood_perimeter);
    return g;
}
__device__ float mc_channel_celerity(float s0,float n,float r,float z,float bw,float h) {
    return (sqrtf(s0)/n)*((5.0f/3.0f)*mc_pow(r,2.0f/3.0f)
        -(2.0f/3.0f)*mc_pow(r,5.0f/3.0f)*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h*z)));
}
__device__ float mc_flux(float c1,float c2,float c3,float c4,float qup,float quc,float qdp) {
    return ((c1*qup)+(c2*quc)+(c3*qdp)+c4);
}
__device__ void mc_secant(float z,float bw,float bfd,float twcc,float s0,float n,float ncc,
    float dt,float dx,float qdp,float ql,float qup,float quc,float h,int interval,
    float &qj,float &c1,float &c2,float &c3,float &c4,float &x) {
    MCGeometry g=mc_geometry(h,bfd,bw,twcc,z);
    float ck=0.0f;
    if(h>bfd && twcc>0.0f && ncc>0.0f) {
        ck=fmaxf(0.0f,(mc_channel_celerity(s0,n,g.radius,z,bw,bfd)*g.area
            +((sqrtf(s0)/ncc)*(5.0f/3.0f)*mc_pow(h-bfd,2.0f/3.0f))*g.flood_area)
            /(g.area+g.flood_area));
    } else if(h>0.0f) ck=fmaxf(0.0f,mc_channel_celerity(s0,n,g.radius,z,bw,h));
    float km=ck>0.0f ? fmaxf(dt,dx/ck) : dt;
    x=0.0f;
    if(ck>0.0f) {
        float width=(h>bfd && twcc>0.0f && ncc>0.0f) ? twcc : g.twl;
        if(interval==1) x=fminf(0.5f,fmaxf(0.0f,0.5f*(1.0f-qj/(2.0f*width*s0*ck*dx))));
        else x=fminf(0.5f,fmaxf(0.25f,0.5f*(1.0f-mc_flux(c1,c2,c3,c4,qup,quc,qdp)/(2.0f*width*s0*ck*dx))));
    } else x=0.5f;
    float d=km*(1.0f-x)+dt/2.0f;
    c1=(km*x+dt/2.0f)/d;
    c2=(dt/2.0f-km*x)/d;
    c3=(km*(1.0f-x)-dt/2.0f)/d;
    c4=(ql*dt)/d;
    if(interval==2 && c4<0.0f && fabsf(c4)>(c1*qup)+(c2*quc)+(c3*qdp))
        c4=-((c1*qup)+(c2*quc)+(c3*qdp));
    if(g.perimeter+g.flood_perimeter>0.0f)
        qj=mc_flux(c1,c2,c3,c4,qup,quc,qdp)
            -((1.0f/(((g.perimeter*n)+(g.flood_perimeter*ncc))/(g.perimeter+g.flood_perimeter)))
            *(g.area+g.flood_area)*mc_pow(g.radius,2.0f/3.0f)*sqrtf(s0));
    else qj=0.0f;
}
__device__ void mc_solve(const float *v,float *out) {
    float dt=v[0],qup=v[1],quc=v[2],qdp=v[3],ql=v[4],dx=v[5],bw=v[6],tw=v[7],twcc=v[8];
    float n=v[9],ncc=v[10],cs=v[11],s0=v[12],depthp=v[14];
    float z=cs==0.0f ? 1.0f : 1.0f/cs;
    float bfd=bw>tw ? bw/0.00001f : (bw==tw ? bw/(2.0f*z) : (tw-bw)/(2.0f*z));
    float depth=fmaxf(depthp,0.0f),h=depth*1.33f+0.01f,h0=depth*0.67f;
    float qdc=0.0f,vel=0.0f,x=0.0f;
    // Explicit zero matches the pinned native-tool initialization patch.
    // Untouched upstream reads an undefined first upper-interval Qj value.
    float qj0=0.0f,qj=0.0f,c1=0.0f,c2=0.0f,c3=0.0f,c4=0.0f;
    float aerror=0.01f,rerror=1.0f;
    int maxiter=100,tries=0;
    if(ql>0.0f || qup>0.0f || quc>0.0f || qdp>0.0f || qdc>0.0f) {
        for(;;) {
            int iter=0;
            while(rerror>0.01f && aerror>=0.01f && iter<=maxiter) {
                mc_secant(z,bw,bfd,twcc,s0,n,ncc,dt,dx,qdp,ql,qup,quc,h0,1,qj0,c1,c2,c3,c4,x);
                mc_secant(z,bw,bfd,twcc,s0,n,ncc,dt,dx,qdp,ql,qup,quc,h,2,qj,c1,c2,c3,c4,x);
                float h1=h;
                if(qj0-qj!=0.0f) { h1=h-((qj*(h0-h))/(qj0-qj)); if(h1<0.0f) h1=h; }
                if(h>0.0f) { rerror=fabsf((h1-h)/h); aerror=fabsf(h1-h); }
                else { rerror=0.0f; aerror=0.9f; }
                h0=fmaxf(0.0f,h); h=fmaxf(0.0f,h1); ++iter;
                if(h<0.01f) break;
            }
            if(iter>=maxiter && ++tries<=4) { h=h*1.33f; h0=h0*0.67f; maxiter+=25; continue; }
            break;
        }
        float flux=mc_flux(c1,c2,c3,c4,qup,quc,qdp);
        if(flux<0.0f) {
            if(c4<0.0f && fabsf(c4)>(c1*qup)+(c2*quc)+(c3*qdp)) qdc=0.0f;
            else qdc=fmaxf((c1*qup)+(c2*quc)+c4,(c1*qup)+(c3*qdp)+c4);
        } else qdc=flux;
        MCGeometry g=mc_geometry(h,bfd,bw,twcc,z);
        float half=(g.twl-bw)/2.0f;
        // The pinned strict native build calls powf for **0.5, not sqrtf.
        // These differ at actual captured call 68066; preserve that operation.
        float radius=(h*(bw+g.twl)/2.0f)/(bw+2.0f*mc_pow(half*half+h*h,0.5f));
        vel=(1.0f/n)*mc_pow(radius,2.0f/3.0f)*sqrtf(s0);
        depth=h;
    } else depth=0.0f;
    // Native computes Courant from the initial h even on its dry branch.
    MCGeometry g=mc_geometry(h,bfd,bw,twcc,z);
    float ck=fmaxf(0.0f,(mc_channel_celerity(s0,n,g.radius,z,bw,g.below)*g.area
        +((sqrtf(s0)/ncc)*(5.0f/3.0f)*mc_pow(g.above,2.0f/3.0f))*g.flood_area)
        /(g.area+g.flood_area));
    out[0]=qdc; out[1]=vel; out[2]=depth; out[3]=ck; out[4]=ck*(dt/dx); out[5]=x;
}
extern "C" __global__ void troute_mc_replay(int count,const float *inputs,float *outputs) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<count) mc_solve(inputs+15*i,outputs+6*i);
}
extern "C" __global__ void troute_mc_snapshot(int count,const float *state,float *previous) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<3*count) previous[i]=state[i];
}
extern "C" __global__ void troute_mc_publish(int count,const float *state,float *output,
    int channel_count,const int *channel_nodes,float *channel_output) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<count) {
        output[i]=state[3*i]; output[count+i]=state[3*i+1]; output[2*count+i]=state[3*i+2];
    }
    if(i<channel_count) {
        int node=channel_nodes[i];
        channel_output[i]=state[3*node];
        channel_output[channel_count+i]=state[3*node+1];
        channel_output[2*channel_count+i]=state[3*node+2];
    }
}
// Channel-state loads. Plain for the per-step kernels; the whole-replay
// kernel (routed_replay.cu) reads state other SMs published this launch and
// passes an L2 (ld.cg) load. Loads only: the arithmetic is the same either way.
struct McLoad { __device__ __forceinline__ float operator()(const float *p) const { return *p; } };
// Every launch path calls the same per-node arithmetic and upstream order.
template <class Load>
__device__ __forceinline__ void troute_mc_node_with(const Load &load,int i,float dt,int short_ts,
    const int *offsets,const int *upstream,const float *parameters,
    const float *lateral,const float *boundary_previous,const float *boundary_current,
    const float *previous,float *state,float *diagnostics) {
    float qup=boundary_previous[i],quc=short_ts ? boundary_previous[i] : boundary_current[i];
    for(int k=offsets[i];k<offsets[i+1];++k) {
        int u=upstream[k]; qup+=load(previous+3*u); quc+=short_ts ? load(previous+3*u) : load(state+3*u);
    }
    const float *p=parameters+8*i;
    float v[15]={dt,qup,quc,load(previous+3*i),lateral[i],p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],load(previous+3*i+1),load(previous+3*i+2)};
    float out[6]; mc_solve(v,out);
    for(int k=0;k<3;++k) { state[3*i+k]=out[k]; diagnostics[3*i+k]=out[3+k]; }
}
__device__ void troute_mc_node(int i,float dt,int short_ts,
    const int *offsets,const int *upstream,const float *parameters,
    const float *lateral,const float *boundary_previous,const float *boundary_current,
    const float *previous,float *state,float *diagnostics) {
    troute_mc_node_with(McLoad(),i,dt,short_ts,offsets,upstream,parameters,lateral,
        boundary_previous,boundary_current,previous,state,diagnostics);
}
extern "C" __global__ void troute_mc_level(int start,int count,float dt,int short_ts,
    const int *nodes,const int *offsets,const int *upstream,const float *parameters,
    const float *lateral,const float *boundary_previous,const float *boundary_current,
    const float *previous,float *state,float *diagnostics) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j<count) troute_mc_node(nodes[start+j],dt,short_ts,offsets,upstream,parameters,
        lateral,boundary_previous,boundary_current,previous,state,diagnostics);
}
// Consecutive narrow levels fit in ONE block. A block barrier publishes each
// level's global state before its downstream level. Never crosses a reservoir
// hook; wide levels keep the original multi-block launch. No cooperative grid,
// polling, altered cadence, or CUDA graph is involved.
extern "C" __global__ void troute_mc_thin_levels(
    int a,int b,int c,int d,int e,int f,int g,int h,int z,
    float dt,int short_ts,const int *nodes,const int *offsets,const int *upstream,
    const float *parameters,const float *lateral,const float *boundary_previous,
    const float *boundary_current,const float *previous,float *state,float *diagnostics) {
    int edge[9]={a,b,c,d,e,f,g,h,z};
    for(int level=0;level<8;++level) {
        int j=edge[level]+threadIdx.x;
        if(j<edge[level+1]) troute_mc_node(nodes[j],dt,short_ts,offsets,upstream,parameters,
            lateral,boundary_previous,boundary_current,previous,state,diagnostics);
        __syncthreads();
    }
}

// Original ArWen network glue. Native t-route mc_reach.pyx passes the summed
// current upstream flow and zero lateral into run_lp_c; bind_lp.f90 supplies
// that same inflow as both endpoints. Short-timestep mode first selects old q.
extern "C" __global__ void troute_mc_reservoir_gather(int count,int short_ts,
    const int *rows,const int *node_indices,const int *offsets,const int *upstream,
    const float *boundary_previous,const float *boundary_current,
    const float *previous,const float *state,
    float *previous_inflow,float *inflow,float *lateral) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=count) return;
    int r=rows[j],node=node_indices[r];
    float q=short_ts ? boundary_previous[node] : boundary_current[node];
    for(int k=offsets[node];k<offsets[node+1];++k) {
        int u=upstream[k]; q+=short_ts ? previous[3*u] : state[3*u];
    }
    previous_inflow[r]=q; inflow[r]=q; lateral[r]=0.0f;
}
// Connected parent runoff already uses the collapsed routing-node axis.
// Feed each lake's own catchment discharge to the native lateral inlet once.
// One reservoir row; the kernel below and the whole-replay kernel
// (routed_replay.cu) share it.
template <class Load>
__device__ __forceinline__ void troute_mc_reservoir_gather_routed_row(const Load &load,int r,int short_ts,
    const int *node_indices,const int *offsets,const int *upstream,
    const float *boundary_previous,const float *boundary_current,
    const float *previous,const float *state,const float *routing_lateral,
    float *previous_inflow,float *inflow,float *lateral) {
    int node=node_indices[r];
    float q=short_ts ? boundary_previous[node] : boundary_current[node];
    for(int k=offsets[node];k<offsets[node+1];++k) {
        int u=upstream[k]; q+=short_ts ? load(previous+3*u) : load(state+3*u);
    }
    previous_inflow[r]=q; inflow[r]=q; lateral[r]=routing_lateral[node];
}
extern "C" __global__ void troute_mc_reservoir_gather_routed(int count,int short_ts,
    const int *rows,const int *node_indices,const int *offsets,const int *upstream,
    const float *boundary_previous,const float *boundary_current,
    const float *previous,const float *state,const float *routing_lateral,
    float *previous_inflow,float *inflow,float *lateral) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=count) return;
    troute_mc_reservoir_gather_routed_row(McLoad(),rows[j],short_ts,node_indices,offsets,upstream,
        boundary_previous,boundary_current,previous,state,routing_lateral,
        previous_inflow,inflow,lateral);
}
__device__ __forceinline__ void troute_mc_reservoir_scatter_row(int r,
    const int *node_indices,const double *release,const float *height,float *state) {
    int node=node_indices[r];
    state[3*node]=(float)release[r];
    state[3*node+1]=0.0f; state[3*node+2]=height[r];
}
extern "C" __global__ void troute_mc_reservoir_scatter(int count,
    const int *rows,const int *node_indices,const double *release,
    const float *height,float *state) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=count) return;
    troute_mc_reservoir_scatter_row(rows[j],node_indices,release,height,state);
}
