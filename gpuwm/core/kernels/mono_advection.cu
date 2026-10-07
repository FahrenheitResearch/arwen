// WRF v4.7.1 dyn_em/module_advect_em.F:9495-10560.
// Float32 statement-function coefficients and expression associations are
// retained, including the SIGN convention for zero and eta fluxes. No FMA.
// The CPU shim compiles these same functions; device parity remains separate.
typedef unsigned long long mono_index;

__device__ __forceinline__ float ma(float a,float b) { return __fadd_rn(a,b); }
__device__ __forceinline__ float ms(float a,float b) { return __fsub_rn(a,b); }
__device__ __forceinline__ float mm(float a,float b) { return __fmul_rn(a,b); }
__device__ __forceinline__ float md(float a,float b) { return __fdiv_rn(a,b); }
__device__ __forceinline__ float sign1(float v) { return copysignf(1.0f,v); }
__device__ float mf3(float a,float b,float c,float d,float v) {
    float centered=ms(mm(7.0f/12.0f,ma(c,b)),mm(1.0f/12.0f,ma(d,a)));
    return ma(centered,mm(mm(sign1(v),1.0f/12.0f),ms(ms(d,a),mm(3.0f,ms(c,b)))));
}
__device__ float mf5(float a,float b,float c,float d,float e,float f,float v) {
    float centered=ma(ms(mm(37.0f/60.0f,ma(d,c)),mm(2.0f/15.0f,ma(e,b))),mm(1.0f/60.0f,ma(f,a)));
    return ms(centered,mm(mm(sign1(v),1.0f/60.0f),ma(ms(ms(f,a),mm(5.0f,ms(e,b))),mm(10.0f,ms(d,c)))));
}
__device__ float mup(float a,float b,float v) {
    return ma(mm(mm(0.5f,ma(1.0f,sign1(v))),a),mm(mm(0.5f,ms(1.0f,sign1(v))),b));
}

struct Mono {
    const float *q,*q0,*ru,*rv,*rw,*mut,*c1,*c2,*rd,*fnm,*fnp,*mx,*my,*wi,*mu0,*mub;
    float *xl,*xc,*yl,*yc,*zl,*zc,*qmin,*qmax,*si,*so,*t,*ht,*zt;
    float rdx,rdy,dt;
    int nz,ny,nx,ox,oy,radiation,vo;
    __device__ mono_index cell(int k,int j,int i) const { return (mono_index(k)*ny+j)*nx+i; }
    __device__ mono_index xf(int k,int j,int i) const { return (mono_index(k)*ny+j)*(nx+1)+i; }
    __device__ mono_index yf(int k,int j,int i) const { return (mono_index(k)*(ny+1)+j)*nx+i; }
    __device__ int wrap(int i,int n) const { return (i%n+n)%n; }
    __device__ float sample(const float* a,int k,int j,int i) const { return a[cell(k,wrap(j,ny),wrap(i,nx))]; }
    __device__ bool ring(int j,int i) const { return (ox&&(i==0||i==nx-1))||(oy&&(j==0||j==ny-1)); }
};

// Same argument order for the decomposition, scale, and apply kernels.
#define MONO_ARGS const float *q,const float *q0,const float *ru,const float *rv,const float *rw, \
    const float *mut,const float *c1,const float *c2,const float *rd,const float *fnm,const float *fnp, \
    const float *mx,const float *my,const float *wi,const float *mu0,const float *mub, \
    float *xl,float *xc,float *yl,float *yc,float *zl,float *zc, \
    float *qmin,float *qmax,float *si,float *so,float *t,float *ht,float *zt, \
    float rdx,float rdy,float dt,int nz,int ny,int nx,int ox,int oy,int radiation,int vo
#define MONO_INIT Mono p={q,q0,ru,rv,rw,mut,c1,c2,rd,fnm,fnp,mx,my,wi,mu0,mub, \
    xl,xc,yl,yc,zl,zc,qmin,qmax,si,so,t,ht,zt,rdx,rdy,dt,nz,ny,nx,ox,oy,radiation,vo}

__device__ void mono_flux_at(Mono p,int k,int j,int i) {
    if(k<p.nz && j<p.ny && i<=p.nx) {
        mono_index f=p.xf(k,j,i);
        if(p.ox && (i==0||i==p.nx)) { p.xl[f]=0.0f; p.xc[f]=0.0f; }
        else {
            float v=p.ru[f];
            float a=p.sample(p.q,k,j,i-3),b=p.sample(p.q,k,j,i-2),c=p.sample(p.q,k,j,i-1);
            float d=p.sample(p.q,k,j,i),e=p.sample(p.q,k,j,i+1),ff=p.sample(p.q,k,j,i+2);
            float high=mm(v,mf5(a,b,c,d,e,ff,v));
            int dist=i<p.nx-i?i:p.nx-i;
            if(p.ox && dist==1) high=mm(mm(0.5f,v),ma(d,c));
            if(p.ox && dist==2) high=mm(v,mf3(b,c,d,e,v));
            float low=mm(v,mup(p.sample(p.q0,k,j,i-1),p.sample(p.q0,k,j,i),v));
            p.xl[f]=low; p.xc[f]=ms(high,low);
        }
    }
    if(k<p.nz && j<=p.ny && i<p.nx) {
        mono_index f=p.yf(k,j,i);
        if(p.oy && (j==0||j==p.ny)) { p.yl[f]=0.0f; p.yc[f]=0.0f; }
        else {
            float v=p.rv[f];
            float a=p.sample(p.q,k,j-3,i),b=p.sample(p.q,k,j-2,i),c=p.sample(p.q,k,j-1,i);
            float d=p.sample(p.q,k,j,i),e=p.sample(p.q,k,j+1,i),ff=p.sample(p.q,k,j+2,i);
            float high=mm(v,mf5(a,b,c,d,e,ff,v));
            int dist=j<p.ny-j?j:p.ny-j;
            if(p.oy && dist==1) high=mm(mm(0.5f,v),ma(d,c));
            if(p.oy && dist==2) high=mm(v,mf3(b,c,d,e,v));
            float low=mm(v,mup(p.sample(p.q0,k,j-1,i),p.sample(p.q0,k,j,i),v));
            p.yl[f]=low; p.yc[f]=ms(high,low);
        }
    }
    if(k<=p.nz && j<p.ny && i<p.nx) {
        mono_index f=p.cell(k,j,i); float low=0.0f,high=0.0f;
        if(k>0 && k<p.nz) {
            float v=p.rw[f],c=p.q[p.cell(k-1,j,i)],d=p.q[f];
            low=mm(v,mup(p.q0[p.cell(k-1,j,i)],p.q0[f],-v));
            if(k==1||k==p.nz-1) high=mm(v,ma(mm(p.fnm[k],d),mm(p.fnp[k],c)));
            else if(p.vo==5 && k>=3 && k<=p.nz-3)
                high=mm(v,mf5(p.q[p.cell(k-3,j,i)],p.q[p.cell(k-2,j,i)],c,d,
                              p.q[p.cell(k+1,j,i)],p.q[p.cell(k+2,j,i)],-v));
            else high=mm(v,mf3(p.q[p.cell(k-2,j,i)],c,d,p.q[p.cell(k+1,j,i)],-v));
        }
        p.zl[f]=low; p.zc[f]=ms(high,low);
    }
    // module_advect_em.F:9755-9761 and corresponding x/z donor updates.
    if(k<p.nz && j<p.ny && i<p.nx) {
        mono_index c=p.cell(k,j,i); float mn=p.q0[c],mxv=p.q0[c];
#define EXT(v) { float vv=(v); mn=fminf(mn,vv); mxv=fmaxf(mxv,vv); }
        if((!p.ox||i>0) && p.ru[p.xf(k,j,i)]>0.0f) EXT(p.sample(p.q0,k,j,i-1));
        if((!p.ox||i<p.nx-1) && p.ru[p.xf(k,j,i+1)]<=0.0f) EXT(p.sample(p.q0,k,j,i+1));
        if((!p.oy||j>0) && p.rv[p.yf(k,j,i)]>0.0f) EXT(p.sample(p.q0,k,j-1,i));
        if((!p.oy||j<p.ny-1) && p.rv[p.yf(k,j+1,i)]<=0.0f) EXT(p.sample(p.q0,k,j+1,i));
        if(k>0 && -p.rw[c]>0.0f) EXT(p.q0[p.cell(k-1,j,i)]);
        if(k<p.nz-1 && -p.rw[p.cell(k+1,j,i)]<=0.0f) EXT(p.q0[p.cell(k+1,j,i)]);
#undef EXT
        p.qmin[c]=mn; p.qmax[c]=mxv;
    }
}

__device__ void mono_scale_at(Mono p,int k,int j,int i) {
    mono_index c=p.cell(k,j,i),xy=mono_index(j)*p.nx+i;
    p.si[c]=1.0f; p.so[c]=1.0f;
    if(p.ring(j,i)) return;
    float mx=p.mx[xy],my=p.my[xy],rd=p.rd[k];
    float ieva=ma(ma(mm(p.c1[k],p.mut[xy]),p.c2[k]),mm(mm(mm(p.dt,my),rd),ms(p.wi[p.cell(k+1,j,i)],p.wi[c])));
    float mass=ma(ma(mm(p.c1[k],p.mub[xy]),p.c2[k]),mm(p.c1[k],p.mu0[xy]));
    float ph=ms(mm(mass,p.q0[c]),mm(p.dt,ma(mm(mm(mx,my),ma(
        mm(p.rdx,ms(p.xl[p.xf(k,j,i+1)],p.xl[p.xf(k,j,i)])),
        mm(p.rdy,ms(p.yl[p.yf(k,j+1,i)],p.yl[p.yf(k,j,i)])))),
        mm(mm(my,rd),ms(p.zl[p.cell(k+1,j,i)],p.zl[c])))));
    float xr=p.xc[p.xf(k,j,i+1)],xl=p.xc[p.xf(k,j,i)];
    float yr=p.yc[p.yf(k,j+1,i)],yl=p.yc[p.yf(k,j,i)];
    float zr=p.zc[p.cell(k+1,j,i)],zl=p.zc[c];
    float fi=mm(-p.dt,ma(mm(mm(mx,my),ma(mm(p.rdx,ms(fminf(0.0f,xr),fmaxf(0.0f,xl))),
        mm(p.rdy,ms(fminf(0.0f,yr),fmaxf(0.0f,yl))))),mm(mm(my,rd),ms(fmaxf(0.0f,zr),fminf(0.0f,zl)))));
    float fo=mm(p.dt,ma(mm(mm(mx,my),ma(mm(p.rdx,ms(fmaxf(0.0f,xr),fminf(0.0f,xl))),
        mm(p.rdy,ms(fmaxf(0.0f,yr),fminf(0.0f,yl))))),mm(mm(my,rd),ms(fminf(0.0f,zr),fmaxf(0.0f,zl)))));
    float hi=ms(mm(ieva,p.qmax[c]),ph),lo=ms(ph,mm(ieva,p.qmin[c]));
    if(fi>hi) p.si[c]=fmaxf(0.0f,md(hi,ma(fi,1.0e-20f)));
    if(fo>lo) p.so[c]=fmaxf(0.0f,md(lo,ma(fo,1.0e-20f)));
}

// module_advect_em.F:10413-10447: every face uses both cells' budgets.
__device__ float mcx(Mono p,int k,int j,int i) {
    float f=p.xc[p.xf(k,j,i)];
    if(p.oy && (j==0||j==p.ny-1)) return f;
    mono_index r=p.cell(k,j,p.wrap(i,p.nx)),l=p.cell(k,j,p.wrap(i-1,p.nx));
    return mm(f>0.0f?fminf(p.si[r],p.so[l]):fminf(p.so[r],p.si[l]),f);
}
__device__ float mcy(Mono p,int k,int j,int i) {
    float f=p.yc[p.yf(k,j,i)];
    if(p.ox && (i==0||i==p.nx-1)) return f;
    mono_index r=p.cell(k,p.wrap(j,p.ny),i),l=p.cell(k,p.wrap(j-1,p.ny),i);
    return mm(f>0.0f?fminf(p.si[r],p.so[l]):fminf(p.so[r],p.si[l]),f);
}
__device__ float mcz(Mono p,int k,int j,int i) {
    float f=p.zc[p.cell(k,j,i)];
    if(k==0||k==p.nz||p.ring(j,i)) return f;
    mono_index r=p.cell(k,j,i),l=p.cell(k-1,j,i);
    return mm(f<0.0f?fminf(p.si[r],p.so[l]):fminf(p.so[r],p.si[l]),f);
}

__device__ void mono_apply_at(Mono p,int k,int j,int i) {
    mono_index c=p.cell(k,j,i); float t=p.t[c];
    // module_advect_em.F:9998-10084. Radiation precedes z, x, y.
    if(p.radiation && p.ox && (i==0||i==p.nx-1)) {
        int f=i==0?0:p.nx-1;
        float va=p.ru[p.xf(k,j,f)],vb=p.ru[p.xf(k,j,f+1)];
        float avg=mm(0.5f,ma(va,vb));
        float u=i==0?fminf(avg,0.0f):fmaxf(avg,0.0f);
        float dq=i==0?ms(p.q0[p.cell(k,j,1)],p.q0[c]):ms(p.q0[c],p.q0[p.cell(k,j,i-1)]);
        t=ms(t,mm(p.rdx,ma(mm(u,dq),mm(p.q[c],ms(vb,va)))));
    }
    if(p.radiation && p.oy && (j==0||j==p.ny-1)) {
        int f=j==0?0:p.ny-1;
        float va=p.rv[p.yf(k,f,i)],vb=p.rv[p.yf(k,f+1,i)];
        float avg=mm(0.5f,ma(va,vb));
        float v=j==0?fminf(avg,0.0f):fmaxf(avg,0.0f);
        float dq=j==0?ms(p.q0[p.cell(k,1,i)],p.q0[c]):ms(p.q0[c],p.q0[p.cell(k,j-1,i)]);
        t=ms(t,mm(p.rdy,ma(mm(v,dq),mm(p.q[c],ms(vb,va)))));
    }
    float dz=mm(p.rd[k],ms(ma(ms(mcz(p,k+1,j,i),mcz(p,k,j,i)),p.zl[p.cell(k+1,j,i)]),p.zl[c]));
    t=ms(t,dz); p.zt[c]=ms(0.0f,dz); p.ht[c]=0.0f;
    bool xactive=!p.ox||(i>0&&i<p.nx-1),yactive=!p.oy||(j>0&&j<p.ny-1);
    float h=0.0f;
    if(xactive) {
        float dd=mm(p.mx[mono_index(j)*p.nx+i],mm(p.rdx,ms(ma(ms(mcx(p,k,j,i+1),mcx(p,k,j,i)),p.xl[p.xf(k,j,i+1)]),p.xl[p.xf(k,j,i)])));
        t=ms(t,dd); h=ms(0.0f,dd);
    }
    if(yactive) {
        float dd=mm(p.mx[mono_index(j)*p.nx+i],mm(p.rdy,ms(ma(ms(mcy(p,k,j+1,i),mcy(p,k,j,i)),p.yl[p.yf(k,j+1,i)]),p.yl[p.yf(k,j,i)])));
        t=ms(t,dd); h=ms(h,dd);
    }
    if(xactive&&yactive) p.ht[c]=h;
    p.t[c]=t;
}

extern "C" __global__ void mono_fluxes(MONO_ARGS) {
    MONO_INIT;
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    if(i<=nx && j<=ny && k<=nz) mono_flux_at(p,k,j,i);
}
extern "C" __global__ void mono_scales(MONO_ARGS) {
    MONO_INIT;
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    if(i<nx && j<ny && k<nz) mono_scale_at(p,k,j,i);
}
extern "C" __global__ void mono_apply(MONO_ARGS) {
    MONO_INIT;
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    if(i<nx && j<ny && k<nz) mono_apply_at(p,k,j,i);
}
