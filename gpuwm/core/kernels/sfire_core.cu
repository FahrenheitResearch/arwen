// WRF v4.7.1 phys/module_fr_fire_core.F, pinned tag v4.7.1.
// https://github.com/wrf-model/WRF/blob/v4.7.1/phys/module_fr_fire_core.F
// The driver prepends glibc_flt32.cuh and sfire_phys.cuh.

__device__ __forceinline__ float sf_sq(float x) { return x*x; }
extern "C" __global__ void sfire_cfl_reciprocal(const float *x,float *y) {
    *y=__fdiv_rn(1.f,*x+100.f*0x1p-23f);
}
__device__ __forceinline__ float sf_eno(float l, float r) {
    // select_eno, lines 2172-2194.
    if (!(l>0.f) && !(r>0.f)) return r;
    if (!(l<0.f) && !(r<0.f)) return l;
    if (!(l<0.f) && !(r>0.f)) return !(fabsf(r)<fabsf(l)) ? r : l;
    return 0.f;
}
__device__ __forceinline__ float sf_upwind(float l,float r) {
    float d=0.f;
    if(l>0.f && r>0.f) d=l;
    if(l<0.f && r<0.f) d=r;
    return d;
}
__device__ __forceinline__ float sf_godunov(float l,float r) {
    float d=0.f, c=r+l;
    if(l>0.f && !(c<0.f)) d=l;
    if(r<0.f && c<0.f) d=r;
    return d;
}
__device__ __forceinline__ float sf_second(float dx,float z,float m,float p) {
    return __fdiv_rn(p+z,2.f*dx)-__fdiv_rn(z+m,2.f*dx);
}
__device__ __forceinline__ float sf_fourth(float dx,float z,float m,float mm,float p,float pp) {
    return __fdiv_rn(7.f*p+7.f*z-pp-m,12.f*dx)
         - __fdiv_rn(7.f*z+7.f*m-p-mm,12.f*dx);
}
__device__ __forceinline__ float sf_weno3_face(float mm,float m,float z,float p) {
    float fh1=-0.5f*m+1.5f*z, fh2=0.5f*z+0.5f*p;
    float beta1=sf_sq(z-m), beta2=sf_sq(p-z);
    float w1t=__fdiv_rn(__fdiv_rn(1.f,3.f),sf_sq(beta1+1.e-6f));
    float w2t=__fdiv_rn(__fdiv_rn(2.f,3.f),sf_sq(beta2+1.e-6f));
    float w1=__fdiv_rn(w1t,w1t+w2t), w2=__fdiv_rn(w2t,w1t+w2t);
    return w1*fh1+w2*fh2;
}
__device__ __forceinline__ float sf_weno3(float dx,float z,float m,float mm,float p,float pp,float uf) {
    // select_weno3, lines 2241-2294. Mirrored stencil for uf<0.
    if(uf<0.f) { float t=m; m=p; p=t; mm=pp; }
    float fluxp=sf_weno3_face(mm,m,z,p);
    float fluxm=sf_weno3_face(0.f,mm,m,z);
    return uf>=0.f ? __fdiv_rn(fluxp-fluxm,dx) : __fdiv_rn(fluxm-fluxp,dx);
}
__device__ __forceinline__ float sf_weno5_face(float mm,float m,float z,float p,float pp) {
    // select_weno5, lines 2340-2355; preserve expression order.
    float fh1=__fdiv_rn(2.f*mm-7.f*m+11.f*z,6.f);
    float fh2=__fdiv_rn(-1.f*m+5.f*z+2.f*p,6.f);
    float fh3=__fdiv_rn(2.f*z+5.f*p-1.f*pp,6.f);
    float c=__fdiv_rn(13.f,12.f);
    float beta1=c*sf_sq(mm-2.f*m+z)+0.25f*sf_sq(mm-4.f*m+3.f*z);
    float beta2=c*sf_sq(m-2.f*z+p)+0.25f*sf_sq(m-p);
    float beta3=c*sf_sq(z-2.f*p+pp)+0.25f*sf_sq(3.f*z-4.f*p+pp);
    float w1t=__fdiv_rn(__fdiv_rn(1.f,10.f),sf_sq(beta1+1.e-6f));
    float w2t=__fdiv_rn(__fdiv_rn(3.f,5.f),sf_sq(beta2+1.e-6f));
    float w3t=__fdiv_rn(__fdiv_rn(3.f,10.f),sf_sq(beta3+1.e-6f));
    float sum=w1t+w2t+w3t;
    float w1=__fdiv_rn(w1t,sum), w2=__fdiv_rn(w2t,sum), w3=__fdiv_rn(w3t,sum);
    return w1*fh1+w2*fh2+w3*fh3;
}
__device__ __forceinline__ float sf_weno5(float dx,float z,float m,float mm,float mmm,float p,float pp,float ppp,float uf) {
    // select_weno5, lines 2301-2366.
    if(uf<0.f) { float t=m; m=p; p=t; t=mm; mm=pp; pp=t; mmm=ppp; }
    float fluxp=sf_weno5_face(mm,m,z,p,pp);
    float fluxm=sf_weno5_face(mmm,mm,m,z,p);
    return uf>=0.f ? __fdiv_rn(fluxp-fluxm,dx) : __fdiv_rn(fluxm-fluxp,dx);
}
__device__ __forceinline__ float sf_extrapolate(float a,float b,float bias) {
    // module_fr_fire_util.F:383-389.
    return (1.f-bias)*(2.f*a-b)+bias*fmaxf(fmaxf(2.f*a-b,a),b);
}
extern "C" __global__ void sfire_boundary(float *a,int nx,int xlo,int xhi,int ylo,int yhi,float bias) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int width=xhi-xlo+1, height=yhi-ylo+1;
    if(q<height) {
        int j=ylo+q;
        a[j*nx+xlo-1]=sf_extrapolate(a[j*nx+xlo],a[j*nx+xlo+1],bias);
        a[j*nx+xhi+1]=sf_extrapolate(a[j*nx+xhi],a[j*nx+xhi-1],bias);
    }
    if(q<width) {
        int i=xlo+q;
        a[(ylo-1)*nx+i]=sf_extrapolate(a[ylo*nx+i],a[(ylo+1)*nx+i],bias);
        a[(yhi+1)*nx+i]=sf_extrapolate(a[yhi*nx+i],a[(yhi-1)*nx+i],bias);
    }
    if(q==0) {
        a[(ylo-1)*nx+xlo-1]=sf_extrapolate(a[ylo*nx+xlo],a[(ylo+1)*nx+xlo+1],bias);
        a[(yhi+1)*nx+xlo-1]=sf_extrapolate(a[yhi*nx+xlo],a[(yhi-1)*nx+xlo+1],bias);
        a[(ylo-1)*nx+xhi+1]=sf_extrapolate(a[ylo*nx+xhi],a[(ylo+1)*nx+xhi-1],bias);
        a[(yhi+1)*nx+xhi+1]=sf_extrapolate(a[yhi*nx+xhi],a[(yhi-1)*nx+xhi-1],bias);
    }
}
__device__ __forceinline__ void sf_nearest(float &d,float &t,float ax,float ay,float sx,float sy,float st,float ex,float ey,float et,float cx2,float cy2) {
    // nearest, lines 260-334. Retain midpoint construction.
    float mx=(sx+ex)*0.5f, my=(sy+ey)*0.5f;
    float dam2=sf_sq(ax-mx)*cx2+sf_sq(ay-my)*cy2;
    float des2=sf_sq(ex-sx)*cx2+sf_sq(ey-sy)*cy2;
    float dames=dam2*des2;
    float am_es=(ax-mx)*(ex-sx)*cx2+(ay-my)*(ey-sy)*cy2;
    float cos2=dames>0.f ? __fdiv_rn(am_es*am_es,dames) : 0.f;
    float dmc2=dam2*cos2, mcrel;
    if(4.f*dmc2<des2) mcrel=copysignf(sqrtf(__fdiv_rn(4.f*dmc2,des2)),am_es);
    else mcrel=am_es>0.f ? 1.f : -1.f;
    float cx=(ex+sx)*0.5f+mcrel*(ex-sx)*0.5f;
    float cy=(ey+sy)*0.5f+mcrel*(ey-sy)*0.5f;
    d=sqrtf(sf_sq(ax-cx)*cx2+sf_sq(ay-cy)*cy2);
    t=(et+st)*0.5f+mcrel*(et-st)*0.5f;
}
extern "C" __global__ void sfire_nearest(const float *ax,const float *ay,float *d,float *t,int n,
    float sx,float sy,float st,float ex,float ey,float et,float cx2,float cy2) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<n) sf_nearest(d[q],t[q],ax[q],ay[q],sx,sy,st,ex,ey,et,cx2,cy2);
}
extern "C" __global__ void sfire_no_fire(float *fuel,float *area,float *lfn,float *tign,int nx,
    int xlo,int xhi,int ylo,int yhi,float dx,float dy,float time) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i;
    fuel[k]=1.f; area[k]=0.f;
    lfn[k]=2.f*fmaxf(width*dx,(yhi-ylo+1)*dy);
    tign[k]=time+fmaxf(time,1.f)*0x1p-23f;
}
extern "C" __global__ void sfire_ignite(float *lfn,float *tign,const float *ax,const float *ay,int *count,
    int nx,int xlo,int xhi,int ylo,int yhi,float start_ts,float end_ts,
    float sx,float sy,float start_time,float ex,float ey,float end_time,float radius,float ros,float ux,float uy) {
    // ignite_fire, lines 85-252. Fix early return's uninitialized ignited count in driver.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    float tos=__fdiv_rn(radius,ros), et=fminf(end_ts,end_time), st=start_time;
    if(start_ts>et+tos || end_ts<st) return;
    if(start_time<end_time) {
        float rele=__fdiv_rn(et-start_time,end_time-start_time);
        ex=sx+rele*(ex-sx); ey=sy+rele*(ey-sy);
    }
    int k=j*nx+i; float d,time_ign;
    sf_nearest(d,time_ign,ax[k],ay[k],sx,sy,st,ex,ey,et,ux*ux,uy*uy);
    float lfn_new=d-fminf(radius,ros*(end_ts-time_ign));
    if(!(lfn_new>0.f)) atomicAdd(count,1);
    if(lfn[k]>0.f && !(lfn_new>0.f))
        tign[k]=fminf(fmaxf(time_ign+__fdiv_rn(d,ros),start_ts),end_ts);
    lfn[k]=fminf(lfn[k],lfn_new);
}
__device__ __forceinline__ void sf_fuel_cell(float &fuel,float &area,const float *l,const float *t,float now,float tau) {
    // fuel_left_cell_1, lines 589-729.
    float ts[4];
    for(int v=0;v<4;v++) { ts[v]=t[v]-now; if(l[v]>0.f || ts[v]>0.f) ts[v]=0.f; }
    float ps=l[0]+l[1]+l[2]+l[3];
    float aps=fabsf(l[0])+fabsf(l[1])+fabsf(l[2])+fabsf(l[3]);
    aps=fmaxf(aps,0x1p-126f);
    area=__fdiv_rn(-__fdiv_rn(ps,aps)+1.f,2.f);
    area=fminf(fmaxf(area,0.f),1.f);
    float ta=0.25f*(ts[0]+ts[1]+ts[2]+ts[3]);
    fuel=area>0.f ? area*gfk_exp(__fdiv_rn(ta,tau))+(1.f-area) : 1.f;
}
extern "C" __global__ void sfire_fuel_cell(const float *l,const float *t,const float *tau,float now,float *fuel,float *area,int n) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<n) sf_fuel_cell(fuel[q],area[q],l+4*q,t+4*q,now,tau[q]);
}
__device__ __forceinline__ float sf_bilinear(const float *a,int k,int nx,float tx,float ty) {
    return (1.f-tx)*(1.f-ty)*a[k]+(1.f-tx)*ty*a[k+nx]
         +tx*(1.f-ty)*a[k+1]+tx*ty*a[k+nx+1];
}
extern "C" __global__ void sfire_fuel_left(const float *l,const float *t,const float *tau,float now,
    float *fuel,float *area,int nx,int xlo,int xhi,int ylo,int yhi) {
    // fuel_left, lines 343-581. Exactly four subcells, column loop outermost.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i; float fs=0.f,as=0.f;
    for(int is=1;is<=2;is++) for(int js=1;js<=2;js++) {
        int k2=(j+(js==1?-1:0))*nx+i+(is==1?-1:0);
        float tx=is==1?0.5f:0.f, ty=js==1?0.5f:0.f;
        float txx=is==1?1.f:0.5f, tyy=js==1?1.f:0.5f;
        float lf[4]={sf_bilinear(l,k2,nx,tx,ty),sf_bilinear(l,k2,nx,tx,tyy),
                     sf_bilinear(l,k2,nx,txx,ty),sf_bilinear(l,k2,nx,txx,tyy)};
        float tf[4]={sf_bilinear(t,k2,nx,tx,ty),sf_bilinear(t,k2,nx,tx,tyy),
                     sf_bilinear(t,k2,nx,txx,ty),sf_bilinear(t,k2,nx,txx,tyy)};
        float ff,af; sf_fuel_cell(ff,af,lf,tf,now,tau[k]); fs=fs+ff; as=as+af;
    }
    fuel[k]=__fdiv_rn(fs,4.f); area[k]=__fdiv_rn(as,4.f);
}
extern "C" __global__ void sfire_select(const float *v,const float *dx,const float *uf,float *out,int n,int method) {
    // Stencil order: mmm,mm,m,z,p,pp,ppp.
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=n) return;
    const float *a=v+7*q; float h=dx[q];
    float l=__fdiv_rn(a[3]-a[2],h),r=__fdiv_rn(a[4]-a[3],h);
    float x=0.f;
    if(method==1) x=sf_upwind(l,r);
    if(method==2) x=sf_godunov(l,r);
    if(method==3) x=sf_eno(l,r);
    if(method==5) x=sf_second(h,a[3],a[2],a[4]);
    if(method==6) x=sf_weno3(h,a[3],a[2],a[1],a[4],a[5],uf[q]);
    if(method==7) x=sf_weno5(h,a[3],a[2],a[1],a[0],a[4],a[5],a[6],uf[q]);
    if(method==10) x=sf_fourth(h,a[3],a[2],a[1],a[4],a[5]);
    out[q]=x;
}
extern "C" __global__ void sfire_select_sided(const float *left,const float *right,float *out,int n,int method) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=n) return;
    if(method==1) out[q]=sf_upwind(left[q],right[q]);
    if(method==2) out[q]=sf_godunov(left[q],right[q]);
    if(method==3) out[q]=sf_eno(left[q],right[q]);
}
__device__ __forceinline__ float sf_high_diff(const float *a,int k,int stride,float dx,float uf,int order) {
    if(order==3) return sf_weno3(dx,a[k],a[k-stride],a[k-2*stride],a[k+stride],a[k+2*stride],uf);
    return sf_weno5(dx,a[k],a[k-stride],a[k-2*stride],a[k-3*stride],a[k+stride],a[k+2*stride],a[k+3*stride],uf);
}
extern "C" __global__ void sfire_tend(float *lfn,const float *vx,const float *vy,const float *zx,const float *zy,
    const float *bbb,const float *betafl,const float *phiwc,const float *r0,const float *ischap,
    float *tend,float *ros,float *cfl,int nx,int xlo,int xhi,int ylo,int yhi,int txlo,int txhi,int tylo,int tyhi,
    float dx,float dy,int upwinding,int split,int grows,int advection,float slope_factor,float band,
    float viscosity,float viscosity_bg,float viscosity_band,float viscosity_ngp) {
    // tend_ls, lines 1886-2122. One node per thread, stable max reduction in driver.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=txhi-txlo+1;
    int i=txlo+q%width,j=tylo+q/width;
    if(j>tyhi) return;
    int k=j*nx+i;
    float diffRx=__fdiv_rn(lfn[k+1]-lfn[k],dx), diffLx=__fdiv_rn(lfn[k]-lfn[k-1],dx);
    float diffRy=__fdiv_rn(lfn[k+nx]-lfn[k],dy), diffLy=__fdiv_rn(lfn[k]-lfn[k-nx],dy);
    float cx=diffLx+diffRx,cy=diffLy+diffRy,dx2,dy2,grad;
    bool edge=i<xlo+10 || i>xhi-10 || j<ylo+10 || j>yhi-10;
    int scheme=edge?3:upwinding;
    if(scheme==8 || scheme==9) scheme=fabsf(lfn[k])<band*dx?(scheme==8?6:7):3;
    dx2=0.f;dy2=0.f;
    if(scheme==0) { dx2=cx;dy2=cy; }
    if(scheme==1) { dx2=sf_upwind(diffLx,diffRx);dy2=sf_upwind(diffLy,diffRy); }
    if(scheme==2) { dx2=sf_godunov(diffLx,diffRx);dy2=sf_godunov(diffLy,diffRy); }
    if(scheme==3) { dx2=sf_eno(diffLx,diffRx);dy2=sf_eno(diffLy,diffRy); }
    if(scheme==4) {
        // WRF omitted normal components. Signed Sethian components remove
        // uninitialized normals while preserving its gradient magnitude.
        dx2=copysignf(sqrtf(sf_sq(fmaxf(diffLx,0.f))+sf_sq(fminf(diffRx,0.f))),cx);
        dy2=copysignf(sqrtf(sf_sq(fmaxf(diffLy,0.f))+sf_sq(fminf(diffRy,0.f))),cy);
    }
    if(scheme==5) { dx2=sf_second(dx,lfn[k],lfn[k-1],lfn[k+1]);dy2=sf_second(dy,lfn[k],lfn[k-nx],lfn[k+nx]); }
    if(scheme==6 || scheme==7) {
        float d4x=sf_fourth(dx,lfn[k],lfn[k-1],lfn[k-2],lfn[k+1],lfn[k+2]);
        float d4y=sf_fourth(dy,lfn[k],lfn[k-nx],lfn[k-2*nx],lfn[k+nx],lfn[k+2*nx]);
        float aval=d4x*vx[k]+d4y*vy[k];
        dx2=sf_high_diff(lfn,k,1,dx,aval*d4x,scheme==6?3:5);
        dy2=sf_high_diff(lfn,k,nx,dy,aval*d4y,scheme==6?3:5);
    }
    if(scheme==4) grad=sqrtf(sf_sq(fmaxf(diffLx,0.f))+sf_sq(fminf(diffRx,0.f))+sf_sq(fmaxf(diffLy,0.f))+sf_sq(fminf(diffRy,0.f)));
    else grad=sqrtf(dx2*dx2+dy2*dy2);
    // REAL exponent 2.0 is a scalar powf call in the native WRF build.
    float scale=sqrtf(gfk_pow(grad,2.f)+0x1p-23f);
    float nvx=__fdiv_rn(dx2,scale), nvy=__fdiv_rn(dy2,scale);
    float speed=vx[k]*nvx+vy[k]*nvy, rb,rw,rs;
    sfire_ros_device(rb,rw,rs,nvx,nvy,vx[k],vy[k],zx[k],zy[k],bbb[k],betafl[k],phiwc[k],r0[k],ischap[k],advection);
    float rr=rb+rw+slope_factor*rs;
    if(grows>0) rr=fmaxf(rr,0.f);
    ros[k]=rr;
    float te;
    if(split==0) te=-rr*grad;
    else {
        te=-rb*grad;
        float advx=0.f,advy=0.f;
        if(fabsf(speed)>0x1p-23f) { advx=__fdiv_rn(vx[k]*rw,speed);advy=__fdiv_rn(vy[k]*rw,speed); }
        float tanphi=zx[k]*nvx+zy[k]*nvy;
        if(fabsf(tanphi)>0x1p-23f) { advx=advx+__fdiv_rn(zx[k]*rs,tanphi);advy=advy+__fdiv_rn(zy[k]*rs,tanphi); }
        // Correct WRF's negative x term (diffRy) to its x derivative.
        te=te-fmaxf(advx,0.f)*diffLx-fminf(advx,0.f)*diffRx-fmaxf(advy,0.f)*diffLy-fminf(advy,0.f)*diffRy;
    }
    cfl[k]=grad>0.f ? __fdiv_rn(rr*(__fdiv_rn(fabsf(dx2),dx)+__fdiv_rn(fabsf(dy2),dy)),grad) : 0.f;
    float threshold=viscosity_ngp*dx,vis;
    bool inside=i>xlo+10 && i<xhi-10 && j>ylo+10 && j<yhi-10;
    if(fabsf(lfn[k])<threshold && inside) vis=viscosity_bg;
    else if(fabsf(lfn[k])>=threshold && fabsf(lfn[k])<threshold*(1.f+viscosity_band) && inside)
        vis=fminf(viscosity_bg+__fdiv_rn((viscosity-viscosity_bg)*(fabsf(lfn[k])-threshold),viscosity_band*threshold),viscosity);
    else vis=viscosity;
    tend[k]=te+vis*fabsf(rr)*((diffRx-diffLx)+(diffRy-diffLy));
}
extern "C" __global__ void sfire_rk_stage(const float *initial,const float *tend,float *out,int nx,
    int xlo,int xhi,int ylo,int yhi,float dt,int stage) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i;
    float factor=stage==1?__fdiv_rn(dt,3.f):(stage==2?__fdiv_rn(dt,2.f):dt);
    out[k]=initial[k]+factor*tend[k];
}
extern "C" __global__ void sfire_reinit_sign(const float *lfn,float *sign,int nx,
    int xlo,int xhi,int ylo,int yhi,float dx) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i;
    // Keep REAL**REAL powers; replacing them with multiplies loses native
    // lfn_s0 words even when the final growth-limited level set is unchanged.
    sign[k]=__fdiv_rn(lfn[k],sqrtf(gfk_pow(lfn[k],2.f)+gfk_pow(dx,2.f)));
}
extern "C" __global__ void sfire_advance_reinit(const float *sign,const float *initial,const float *current,float *out,
    int nx,int xlo,int xhi,int ylo,int yhi,int txlo,int txhi,int tylo,int tyhi,
    float dx,float dy,float dt_s,float threshold,int method,int bdy,float coeff) {
    // advance_ls_reinit, lines 1673-1768.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=txhi-txlo+1;
    int i=txlo+q%width,j=tylo+q/width;
    if(j>tyhi) return;
    int k=j*nx+i;
    bool edge=i<xlo+bdy || i>xhi-bdy || j<ylo+bdy || j>yhi-bdy;
    bool high=method==1 || method==2 || current[k]<threshold;
    float d2x,d2y;
    if(!edge && high) {
        float dx4=sf_fourth(dx,current[k],current[k-1],current[k-2],current[k+1],current[k+2]);
        float dy4=sf_fourth(dy,current[k],current[k-nx],current[k-2*nx],current[k+nx],current[k+2*nx]);
        int order=method==1||method==3?3:5;
        d2x=sf_high_diff(current,k,1,dx,sign[k]*dx4,order);
        d2y=sf_high_diff(current,k,nx,dy,sign[k]*dy4,order);
    } else {
        float lx=__fdiv_rn(current[k]-current[k-1],dx),rx=__fdiv_rn(current[k+1]-current[k],dx);
        float ly=__fdiv_rn(current[k]-current[k-nx],dy),ry=__fdiv_rn(current[k+nx]-current[k],dy);
        d2x=sf_eno(lx,rx);d2y=sf_eno(ly,ry);
    }
    float grad=sqrtf(d2x*d2x+d2y*d2y);
    float tendency=sign[k]*(1.f-grad);
    out[k]=initial[k]+(dt_s*coeff)*tendency;
}
extern "C" __global__ void sfire_growth_limit(const float *before,float *after,int nx,
    int xlo,int xhi,int ylo,int yhi) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j<=yhi) { int k=j*nx+i;after[k]=fminf(after[k],before[k]); }
}
extern "C" __global__ void sfire_tign_update(const float *before,const float *after,float *tign,int *hit,
    int nx,int xlo,int xhi,int ylo,int yhi,int txlo,int txhi,int tylo,int tyhi,float ts,float dt,int guard) {
    // tign_update, lines 1775-1845. Boundary coordinates relative to domain start.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=txhi-txlo+1;
    int i=txlo+q%width,j=tylo+q/width;
    if(j>tyhi) return;
    int k=j*nx+i;float now=ts+dt;now=now+fabsf(now)*0x1p-23f*2.f;
    if(!(after[k]>0.f) && before[k]>0.f) tign[k]=ts+__fdiv_rn(dt*before[k],before[k]-after[k]);
    if(after[k]>0.f) tign[k]=now;
    if(guard>=0 && after[k]<0.f && (i-xlo+1<=guard || i>xhi-guard || j-ylo+1<=guard || j>yhi-guard))
        atomicExch(hit,1);
}
extern "C" __global__ void sfire_flame(const float *ros,const float *iboros,const float *area,float *length,float *front,float *intensity,
    int nx,int xlo,int xhi,int ylo,int yhi) {
    // calc_flame_length, lines 1853-1879; fire-line intensity follows iboros*ros.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i;
    if(area[k]>0.f && area[k]<1.f) {
        intensity[k]=iboros[k]*ros[k];length[k]=0.0775f*gfk_pow(intensity[k],0.46f);front[k]=ros[k];
    } else { intensity[k]=0.f;length[k]=0.f;front[k]=0.f; }
}
extern "C" __global__ void sfire_ideal_coords(float *x,float *y,int nx,int ny,int xlo,int ylo,float dx,float dy) {
    // module_fr_fire_util.F:260-286; initialize all halos consistently.
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=nx*ny) return;
    int i=q%nx,j=q/nx;
    x[q]=(i-xlo+0.5f)*dx;y[q]=(j-ylo+0.5f)*dy;
}
extern "C" __global__ void sfire_driver_burn(float *old,const float *next,float *burnt,int nx,
    int xlo,int xhi,int ylo,int yhi) {
    // module_fr_fire_model.F:447-453.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i;
    float future=next[k];
    // Rounded native fuel quadrature can increase fuel by one ULP even
    // while the fire grows. Preserve consumed fuel before computing both
    // heat and smoke. Invalid quadrature values still propagate.
    if(isfinite(future) && future>old[k]) future=old[k];
    burnt[k]=old[k]-future;old[k]=future;
}
extern "C" __global__ void sfire_assimilate_perimeter(float *lfn,float *tign,const float *historic,const float *historic_tign,
    int use_tign,int *count,int nx,int xlo,int xhi,int ylo,int yhi,float ts) {
    // module_fr_fire_model.F:390-406. Optional observed ignition times are
    // supplied explicitly rather than inferred from a perimeter polygon.
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=xhi-xlo+1;
    int i=xlo+q%width,j=ylo+q/width;
    if(j>yhi) return;
    int k=j*nx+i;lfn[k]=historic[k];
    if(use_tign) tign[k]=historic_tign[k];
    if(fabsf(lfn[k])<0.00001f) { if(!use_tign) tign[k]=ts;atomicAdd(count,1); }
}
extern "C" __global__ void sfire_flux_accumulate(const float *h,const float *q,const float *mass,
    const float *burnt,float *hsum,float *qsum,float *msum,float *bsum,int n,float dt) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=n) return;
    hsum[k]=hsum[k]+h[k]*dt;qsum[k]=qsum[k]+q[k]*dt;
    msum[k]=msum[k]+mass[k];bsum[k]=bsum[k]+burnt[k];
}
extern "C" __global__ void sfire_flux_average(float *h,float *q,int n,float dt) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k<n) { h[k]=__fdiv_rn(h[k],dt);q[k]=__fdiv_rn(q[k],dt); }
}
