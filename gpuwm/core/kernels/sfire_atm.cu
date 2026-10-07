// WRF v4.7.1 module_fr_fire_atm.F and module_fr_fire_util.F.
// Atmospheric arrays are (k,j,i); horizontal bounds are inclusive.
// Corrected TG height and sensible crown flux, and complete smoke columns.

__device__ float sfire_tg_cdf(float z, float peak, float width) {
    const float a=__fdiv_rn(167.0f,148.0f), b=__fdiv_rn(11.0f,109.0f);
    float x=__fdiv_rn(z-peak,width);
    return 0.5f*(1.0f+sfire_tanhf(a*x+b*((x*x)*x)));
}

__device__ float sfire_tg_layer(float lo,float hi,float peak,float width,float upper) {
    lo=fminf(fmaxf(lo,0.0f),upper);
    hi=fminf(fmaxf(hi,0.0f),upper);
    return fmaxf(sfire_tg_cdf(hi,peak,width)-sfire_tg_cdf(lo,peak,width),0.0f);
}

extern "C" __global__ void sfire_sum_cells(const float *fine, float *coarse,
        int nx, int ny, int rx, int ry, int mean) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= nx * ny) return;
    int i = p % nx, j = p / nx;
    float t = 0.0f;
    for (int jy=0; jy<ry; ++jy)
        for (int ix=0; ix<rx; ++ix)
            t = t + fine[(j*ry+jy)*(nx*rx)+i*rx+ix];
    coarse[p] = mean ? t * __fdiv_rn(1.0f, (float)(rx*ry)) : t;
}

extern "C" __global__ void sfire_interpolate_2d(const float *coarse, float *fine,
        int nx, int ny, int fx, int fy, int rx, int ry, float cx, float cy,
        float ox, float oy) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= fx*fy) return;
    int i = p%fx, j = p/fx;
    // Choose the last coarse cell at a shared node, as the Fortran loops do.
    int ic = min(max((int)floorf(cx+__fdiv_rn((float)i-ox,(float)rx)),0),nx-2);
    int jc = min(max((int)floorf(cy+__fdiv_rn((float)j-oy,(float)ry)),0),ny-2);
    float rio = ox + (float)rx*((float)ic-cx);
    float rjo = oy + (float)ry*((float)jc-cy);
    float tx = ((float)i-rio)*__fdiv_rn(1.0f,(float)rx);
    float ty = ((float)j-rjo)*__fdiv_rn(1.0f,(float)ry);
    fine[p] = (1.0f-tx)*(1.0f-ty)*coarse[jc*nx+ic]
             +(1.0f-tx)*ty*coarse[(jc+1)*nx+ic]
             +tx*(1.0f-ty)*coarse[jc*nx+ic+1]
             +tx*ty*coarse[(jc+1)*nx+ic+1];
}

extern "C" __global__ void sfire_log_wind(const float *wind, const float *height,
        const float *rough, float *out, int n, int nz, float target) {
    int p = blockDim.x*blockIdx.x+threadIdx.x;
    if (p>=n) return;
    float zr = rough[p];
    if (target <= zr) { out[p] = 0.0f; return; }
    float logtarget = gfk_log(target);
    for (int k=0; k<nz; ++k) {
        float ht = height[k*n+p];
        if (!(ht < target)) {
            float loght=gfk_log(ht);
            if (k==0) {
                float logz=gfk_log(zr);
                out[p]=__fdiv_rn(wind[p]*(logtarget-logz),loght-logz);
            } else {
                float loglast=gfk_log(height[(k-1)*n+p]);
                float last=wind[(k-1)*n+p];
                out[p]=last+__fdiv_rn((wind[k*n+p]-last)*(logtarget-loglast),loght-loglast);
            }
            return;
        }
    }
    out[p]=wind[(nz-1)*n+p];
}

extern "C" __global__ void sfire_tg_dist(const float *dz, const float *z,
        const float *terrain, float *prop, int nx, int ny, int nz,
        float peak, float upper, float ext, int active, int *status,
        int xl, int xh, int yl, int yh) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    int n=nx*ny;
    if(p>=n) return;
    int i=p%nx,j=p/nx;
    if(i<xl||i>xh||j<yl||j>yh) return;
    float width=0.5f*ext;
    float total=0.0f;
    for(int k=0;k<active;++k) {
        float lo=z[k*n+p]-terrain[p], hi=z[(k+1)*n+p]-terrain[p];
        float fraction=sfire_tg_layer(lo,hi,peak,width,upper);
        prop[k*n+p]=fraction;
        total=total+fraction;
    }
    if (!(total>0.0f) || !isfinite(total)) { atomicExch(status,1); return; }
    for(int k=0;k<active;++k) {
        prop[k*n+p]=__fdiv_rn(prop[k*n+p],total);
    }
}

__device__ void sfire_interface_flux(int k,int p,int n,const float *z,
        const float *terrain,const float *prop,float hg,float qg,float hc,float qc,
        float gi,float ci,float crown,int scheme,float *h,float *q) {
    const float cp_i=__fdiv_rn(1.0f,1004.5f), xlv_i=__fdiv_rn(1.0f,2500000.0f);
    float zw=z[k*n+p]-terrain[p];
    float fg,fc;
    fg=cp_i*gfk_exp(-gi*zw);
    fc=(zw<crown) ? cp_i : cp_i*gfk_exp(-ci*(zw-crown));
    *h=fg*hg+fc*hc;
    fg=xlv_i*gfk_exp(-gi*zw);
    fc=(zw<crown) ? xlv_i : xlv_i*gfk_exp(-ci*(zw-crown));
    *q=fg*qg+fc*qc;
}

extern "C" __global__ void sfire_fire_tendency(const float *hg,const float *qg,
        const float *hc,const float *qc,const float *terrain,const float *z,
        const float *dz,const float *mu,const float *c1,const float *c2,
        const float *rho,const float *prop,const float *prop_crown,float *th,float *qv,int nx,int ny,int nz,
        float extg,float extc,float crown,int scheme,int xl,int xh,int yl,int yh) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    int n=nx*ny;
    if(p>=n) return;
    int i=p%nx,j=p/nx;
    if(i<xl||i>xh||j<yl||j>yh) return;
    if (scheme==1) {
        const float cp_i=__fdiv_rn(1.0f,1004.5f), xlv_i=__fdiv_rn(1.0f,2500000.0f);
        for (int k=0;k<nz-1;++k) {
            float f=prop[k*n+p], c=prop_crown[k*n+p];
            float ri=__fdiv_rn(1.0f,rho[k*n+p]);
            float h=(f*hg[p]+c*hc[p])*cp_i;
            float q=(f*qg[p]+c*qc[p])*xlv_i;
            th[k*n+p]=__fdiv_rn((c1[k]*mu[p]+c2[k])*ri*h,dz[k*n+p]);
            qv[k*n+p]=__fdiv_rn((c1[k]*mu[p]+c2[k])*ri*q,dz[k*n+p]);
        }
        return;
    }
    float gi=__fdiv_rn(1.0f,extg),ci=__fdiv_rn(1.0f,extc);
    float hlo,qlo;
    sfire_interface_flux(0,p,n,z,terrain,prop,hg[p],qg[p],hc[p],qc[p],gi,ci,crown,scheme,&hlo,&qlo);
    for(int k=0;k<nz-1;++k) {
        float hhi,qhi;
        sfire_interface_flux(k+1,p,n,z,terrain,prop,hg[p],qg[p],hc[p],qc[p],gi,ci,crown,scheme,&hhi,&qhi);
        float ri=__fdiv_rn(1.0f,rho[k*n+p]);
        th[k*n+p]=__fdiv_rn(-(c1[k]*mu[p]+c2[k])*ri*(hhi-hlo),dz[k*n+p]);
        qv[k*n+p]=__fdiv_rn(-(c1[k]*mu[p]+c2[k])*ri*(qhi-qlo),dz[k*n+p]);
        hlo=hhi; qlo=qhi;
    }
}

extern "C" __global__ void sfire_smoke_emissions(float *tracer,const float *burnt,
        const float *fuel,const float *rho,const float *dz,const float *prop,
        int nx,int ny,int nz,int rx,int ry,float yield,int scheme,
        int xl,int xh,int yl,int yh) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    int n=nx*ny;
    if(p>=n) return;
    int i=p%nx,j=p/nx;
    if(i<xl||i>xh||j<yl||j>yh) return;
    float avgw=__fdiv_rn(1.0f,(float)(rx*ry));
    int nk=scheme ? nz : 1;
    for(int jy=0;jy<ry;++jy) {
        for(int ix=0;ix<rx;++ix) {
            int f=(j*ry+jy)*(nx*rx)+i*rx+ix;
            for(int k=0;k<nk;++k) {
                float numerator=scheme ? prop[k*n+p]*avgw*yield : avgw*yield;
                numerator=numerator*burnt[f]*fuel[f]*1000.0f;
                float emis=__fdiv_rn(numerator,rho[k*n+p]*dz[k*n+p]);
                tracer[k*n+p]=tracer[k*n+p]+emis;
            }
        }
    }
}
