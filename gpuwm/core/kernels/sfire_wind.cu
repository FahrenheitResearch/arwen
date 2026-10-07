// WRF v4.7.1 interpolate_atm2fire, including staggered terrain heights.
__device__ float sfire_face_alt(const float *ph,const float *phb,int p,int k,int n,int stride) {
    float a=__fdiv_rn(ph[k*n+p-stride]+phb[k*n+p-stride],9.81f);
    float b=__fdiv_rn(ph[k*n+p]+phb[k*n+p],9.81f);
    return 0.5f*(a+b);
}

extern "C" __global__ void sfire_wind_vertical(const float *wind,const float *ph,
        const float *phb,const float *z0,float *out,int nx,int ny,int nz,
        int ml,int mb,int tx,int ty,int axis,int xl,int xh,int yl,int yh,float target) {
    int q=blockDim.x*blockIdx.x+threadIdx.x;
    if(q>=tx*ty) return;
    int i=q%tx+ml,j=q/tx+mb;
    if(i<xl||i>xh||j<yl||j>yh) return;
    int n=nx*ny,p=j*nx+i,s=axis ? nx : 1;
    float zr=axis ? 0.5f*(z0[p-s]+z0[p]) : 0.5f*(z0[p]+z0[p-s]);
    if(!(target>zr)) {out[q]=0.0f; return;}
    float logtarget=gfk_log(target),ground=sfire_face_alt(ph,phb,p,0,n,s),lastheight=0.0f;
    for(int k=0;k<nz;++k) {
        float ht=0.5f*(sfire_face_alt(ph,phb,p,k,n,s)+sfire_face_alt(ph,phb,p,k+1,n,s))-ground;
        if(!(ht<target)) {
            float loght=gfk_log(ht);
            if(k==0) {
                float logz=gfk_log(zr);
                out[q]=__fdiv_rn(wind[p]*(logtarget-logz),loght-logz);
            } else {
                float loglast=gfk_log(lastheight),last=wind[(k-1)*n+p];
                out[q]=last+__fdiv_rn((wind[k*n+p]-last)*(logtarget-loglast),loght-loglast);
            }
            return;
        }
        lastheight=ht;
    }
    out[q]=wind[(nz-1)*n+p];
}

extern "C" __global__ void sfire_wind_continue(float *a,int ml,int mb,int nx,int ny,
        int xl,int xh,int yl,int yh,int dl,int dh,int bl,int bh,int direction) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=nx*ny) return;
    int i=p%nx+ml,j=p/nx+mb;
    if(direction==0 && j>=yl && j<=yh) {
        if(xl==dl && i==dl-1) a[p]=2.0f*a[p+1]-a[p+2];
        if(xh==dh && i==dh+1) a[p]=2.0f*a[p-1]-a[p-2];
    }
    if(direction==1 && i>=xl && i<=xh) {
        if(yl==bl && j==bl-1) a[p]=2.0f*a[p+nx]-a[p+2*nx];
        if(yh==bh && j==bh+1) a[p]=2.0f*a[p-nx]-a[p-2*nx];
    }
    if(direction==2) {
        if(xl==dl && yl==bl && i==dl-1 && j==bl-1) a[p]=2.0f*a[p+nx+1]-a[p+2*nx+2];
        if(xl==dl && yh==bh && i==dl-1 && j==bh+1) a[p]=2.0f*a[p-nx+1]-a[p-2*nx+2];
        if(xh==dh && yl==bl && i==dh+1 && j==bl-1) a[p]=2.0f*a[p+nx-1]-a[p+2*nx-2];
        if(xh==dh && yh==bh && i==dh+1 && j==bh+1) a[p]=2.0f*a[p-nx-1]-a[p-2*nx-2];
    }
}

extern "C" __global__ void sfire_wind_diagnostics(const float *ua,const float *va,
        float *uah,float *vah,int nx,int ny,int ml,int mb,int tx,int ty,int xl,int xh,int yl,int yh) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=tx*ty) return;
    int i=p%tx+ml,j=p/tx+mb;
    if(i<xl||i>xh||j<yl||j>yh) return;
    uah[j*nx+i]=ua[p]; vah[j*nx+i]=va[p];
}

extern "C" __global__ void sfire_wind_refine(const float *coarse,float *fine,
        int ml,int mb,int nx,int ny,int fx,int rx,int ry,int ax,int ay,int fxl,int fyl,
        int fl,int fh,int fb,int ft,int axis,int xl,int xh,int yl,int yh) {
    int p=blockDim.x*blockIdx.x+threadIdx.x,fw=fh-fl+1;
    if(p>=fw*(ft-fb+1)) return;
    int i=p%fw+fl,j=p/fw+fb;
    float ox=(float)fxl+(axis ? (float)(rx-1)*0.5f : -0.5f);
    float oy=(float)fyl+(axis ? -0.5f : (float)(ry-1)*0.5f);
    int ic=min(max((int)floorf((float)ax+__fdiv_rn((float)i-ox,(float)rx)),xl),xh-1);
    int jc=min(max((int)floorf((float)ay+__fdiv_rn((float)j-oy,(float)ry)),yl),yh-1);
    float rio=ox+(float)rx*((float)ic-(float)ax),rjo=oy+(float)ry*((float)jc-(float)ay);
    float tx=((float)i-rio)*__fdiv_rn(1.0f,(float)rx),ty=((float)j-rjo)*__fdiv_rn(1.0f,(float)ry);
    int q=(jc-mb)*nx+ic-ml;
    fine[j*fx+i]=(1.0f-tx)*(1.0f-ty)*coarse[q]
                 +(1.0f-tx)*ty*coarse[q+nx]
                 +tx*(1.0f-ty)*coarse[q+1]
                 +tx*ty*coarse[q+nx+1];
}

extern "C" __global__ void sfire_wind_zcoupling(float *u,float *v,const float *z0,
        int nx,int xl,int xh,int yl,int yh,float height,float reference) {
    int q=blockDim.x*blockIdx.x+threadIdx.x,w=xh-xl+1;
    if(q>=w*(yh-yl+1)) return;
    int p=(q/w+yl)*nx+q%w+xl;
    // Real exponent 2.0 in the native -O0 routine calls scalar powf.
    float a=u[p],b=v[p],speed=fmaxf(__fsqrt_rn(gfk_pow(a,2.0f)+gfk_pow(b,2.0f)),0.1f),z=z0[p];
    float friction=__fdiv_rn(speed*0.4f,gfk_log(__fdiv_rn(reference,z)));
    float changed=__fdiv_rn(friction,0.4f)*gfk_log(__fdiv_rn(height+z,z));
    u[p]=__fdiv_rn(changed*a,speed); v[p]=__fdiv_rn(changed*b,speed);
}

extern "C" __global__ void sfire_wind_pack_native(const float *u,const float *v,
        const float *ph,const float *phb,const float *z0,const float *zs,
        float *pu,float *pv,float *pph,float *pphb,float *pz0,float *pzs,
        int nx,int ny,int nz,int profile) {
    int q=blockDim.x*blockIdx.x+threadIdx.x,hx=nx+3,hy=ny+3,n=hx*hy;
    if(q>=(nz+1)*n) return;
    int k=q/n,p=q%n,i=p%hx-1,j=p/hx-1;
    int mi=min(max(i,0),nx-1),mj=min(max(j,0),ny-1),mp=mj*nx+mi;
    if(k<nz) {
        pu[q]=u[k*ny*(nx+1)+mj*(nx+1)+min(max(i,0),nx)];
        pv[q]=v[k*(ny+1)*nx+min(max(j,0),ny)*nx+mi];
    }
    pph[q]=ph[k*nx*ny+mp];
    pphb[q]=profile ? phb[k] : phb[k*nx*ny+mp];
    if(k==0) {pz0[p]=z0[mp];pzs[p]=zs[mp];}
}
