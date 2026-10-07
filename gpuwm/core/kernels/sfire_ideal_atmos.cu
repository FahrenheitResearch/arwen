// Atmospheric initialization in WRF v4.7.1 module_initialize_fire.F.
// Each column and sounding preserves the native loop and operation order.
__device__ float sfia_interp(const float *v,const float *z,float target,int n) {
    int a,b;
    bool inc=z[n-1]>z[0];
    if((inc&&target>z[n-1])||(!inc&&target<z[n-1])) {a=n-1;b=n-2;}
    else if((inc&&target<z[0])||(!inc&&target>z[0])) {a=1;b=0;}
    else {
        a=n-1;b=n-2;
        for(int k=n-1;k>=1;--k) {
            if((inc&&z[k]>=target&&z[k-1]<=target)||(!inc&&z[k]<=target&&z[k-1]>=target)) {
                a=k;b=k-1;break;
            }
        }
    }
    float w2=__fdiv_rn(target-z[a],z[b]-z[a]);
    float w1=1.f-w2;
    return w1*v[a]+w2*v[b];
}
extern "C" __global__ void sfire_ideal_interp(const float *v,const float *z,const float *target,
    float *out,int n,int m) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<m)out[q]=sfia_interp(v,z,target[q],n);
}
extern "C" __global__ void sfire_ideal_prepare_sounding(const float *height,const float *theta,
    const float *qv_gkg,const float *surface,float *pm,float *pd,float *rho,float *qv,int n,int dry) {
    if(blockIdx.x||threadIdx.x)return;
    const float cvpm=-__fdiv_rn(CV,CP),rcp=__fdiv_rn(RD,CP),rp0=__fdiv_rn(RD,P0);
    for(int k=0;k<n;++k)qv[k]=.001f*(dry?0.f:qv_gkg[k]);
    float ps=100.f*surface[0];
    float qvf=1.f+RVOVRD*qv[0];
    float rh_s=__fdiv_rn(1.f,rp0*surface[1]*qvf*gfk_pow(__fdiv_rn(ps,P0),cvpm));
    rho[0]=rh_s;
    float dz=height[0],qvf1=1.f+qv[0];
    for(int it=0;it<10;++it) {
        pm[0]=ps-.5f*dz*(rh_s+rho[0])*G*qvf1;
        rho[0]=__fdiv_rn(1.f,rp0*theta[0]*qvf*gfk_pow(__fdiv_rn(pm[0],P0),cvpm));
    }
    for(int k=1;k<n;++k) {
        rho[k]=rho[k-1];dz=height[k]-height[k-1];
        qvf1=.5f*(2.f+(qv[k-1]+qv[k]));qvf=1.f+RVOVRD*qv[k];
        for(int it=0;it<10;++it) {
            pm[k]=pm[k-1]-.5f*dz*(rho[k]+rho[k-1])*G*qvf1;
            rho[k]=__fdiv_rn(1.f,rp0*theta[k]*qvf*gfk_pow(__fdiv_rn(pm[k],P0),cvpm));
        }
    }
    pd[n-1]=pm[n-1];
    for(int k=n-2;k>=0;--k) {
        dz=height[k+1]-height[k];
        pd[k]=pd[k+1]+.5f*dz*(rho[k]+rho[k+1])*G;
    }
}
extern "C" __global__ void sfire_ideal_columns(const float *terrain,const float *zd,const float *pd,
    const float *td,const float *zm,const float *pm,const float *pdm,const float *tm,const float *qvm,
    const float *dnw,const float *rdnw,const float *rdn,const float *c1f,const float *c2f,
    const float *c1h,const float *c2h,const float *c3h,const float *c4h,const float *weights,
    const float *ptop,float *phb,float *phi,float *mub,float *mu,float *pb,float *p,
    float *tinit,float *t,float *tnative,float *qv,float *alb,float *alt,float *al,float *skin,
    float *soil,int *status,int nx,int ny,int nz,int nd,int nm,float dx,float dy,
    float delt,float xr,float yr,float zr,float height,int theta_m) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,nxy=nx*ny;
    if(q>=nxy)return;
    const float rp0=__fdiv_rn(RD,P0),cvpm=-__fdiv_rn(CV,CP),rcp=__fdiv_rn(RD,CP);
    phb[q]=G*terrain[q];phi[q]=0.f;
    float ps=sfia_interp(pd,zd,__fdiv_rn(phb[q],G),nd),pt=ptop[0];
    mub[q]=ps-pt;
    for(int k=0;k<nz;++k) {
        int a=k*nxy+q;
        float level=c3h[k]*(ps-pt)+c4h[k]+pt;
        pb[a]=level;tinit[a]=sfia_interp(td,pd,level,nd)-T0;
        alb[a]=rp0*(tinit[a]+T0)*gfk_pow(__fdiv_rn(pb[a],P0),cvpm);
        phb[a+nxy]=phb[a]-dnw[k]*(c1h[k]*mub[q]+c2h[k])*alb[a];
    }
    float pds=sfia_interp(pdm,zm,__fdiv_rn(phb[q],G),nm);
    mu[q]=pds-pt-mub[q];
    for(int k=0;k<nz;++k) {
        int a=k*nxy+q;float level=c3h[k]*(pds-pt)+c4h[k]+pt;
        qv[a]=sfia_interp(qvm,pdm,level,nm);t[a]=sfia_interp(tm,pdm,level,nm)-T0;
    }
    for(int k=nz-1;k>=0;--k) {
        int a=k*nxy+q;
        float q1=.5f*(qv[a]+qv[k==nz-1?a:a+nxy]);
        float q2=__fdiv_rn(1.f,1.f+q1);q1=q1*q2;
        float term=(c1f[k+1]*mu[q])+q1*(c1f[k+1]*mub[q]+c2f[k+1]);
        if(k==nz-1)p[a]=__fdiv_rn(__fdiv_rn(-.5f*term,rdnw[k]),q2);
        else p[a]=p[a+nxy]-__fdiv_rn(__fdiv_rn(term,q2),rdn[k+1]);
        float qvf=1.f+RVOVRD*qv[a];
        alt[a]=rp0*(t[a]+T0)*qvf*gfk_pow(__fdiv_rn(p[a]+pb[a],P0),cvpm);
        al[a]=alt[a]-alb[a];
    }
    for(int k=1;k<=nz;++k) {
        int a=(k-1)*nxy+q;float dm=c1h[k-1]*mu[q];
        phi[a+nxy]=phi[a]-dnw[k-1]*(((c1h[k-1]*mub[q]+c2h[k-1])+dm)*al[a]+dm*alb[a]);
    }
    if(delt!=0.f&&xr>0.f&&yr>0.f&&zr>0.f) {
        int i=q%nx,j=q/nx;
        float xrad=__fdiv_rn(dx*(float)(i+1-(nx>>1)),xr);
        float yrad=__fdiv_rn(dy*(float)(j+1-(ny>>1)),yr);
        const float pi=2.f*glibc_asinf(1.f);
        for(int k=0;k<nz;++k) {
            int a=k*nxy+q;
            float zrad=__fdiv_rn(.5f*(phi[a]+phi[a+nxy]+phb[a]+phb[a+nxy]),G);
            zrad=__fdiv_rn(zrad-height,zr);
            float rad=__fsqrt_rn(xrad*xrad+yrad*yrad+zrad*zrad);
            if(rad<=1.f) {
                float cs=glibc_cosf(.5f*pi*rad);
                t[a]=t[a]+delt*(cs*cs);
                float qvf=1.f+RVOVRD*qv[a];
                alt[a]=rp0*(t[a]+T0)*qvf*gfk_pow(__fdiv_rn(p[a]+pb[a],P0),cvpm);
                al[a]=alt[a]-alb[a];
            }
        }
        for(int k=1;k<=nz;++k) {
            int a=(k-1)*nxy+q;float dm=c1h[k-1]*mu[q];
            phi[a+nxy]=phi[a]-dnw[k-1]*(((c1h[k-1]*mub[q]+c2h[k-1])+dm)*al[a]+dm*alb[a]);
        }
    }
    float temp[3];
    for(int k=0;k<3;++k) {
        int a=k*nxy+q;
        temp[k]=(t[a]+T0)*gfk_pow(__fdiv_rn(p[a]+pb[a],P0),rcp);
    }
    // Native original reuses the final column's temp outside its loop.
    // The corrected default computes each column's own skin temperature.
    skin[q]=weights[0]*temp[0]+weights[1]*temp[1]+weights[2]*temp[2];soil[q]=skin[q]-.5f;
    int bad=0;
    if(!isfinite(mub[q])||mub[q]<=0.f||!isfinite(mu[q]))bad=1;
    for(int k=0;k<nz;++k) {
        int a=k*nxy+q;
        tnative[a]=theta_m?((t[a]+T0)*(1.f+RVOVRD*qv[a])-T0):t[a];
        if(!isfinite(p[a])||p[a]+pb[a]<=0.f||!isfinite(alt[a])||alt[a]<=0.f||!isfinite(tnative[a]))bad=1;
    }
    status[q]=bad;
}
extern "C" __global__ void sfire_ideal_wind(const float *phb,const float *z,const float *pm,
    const float *wind,const float *c3h,const float *c4h,const float *ptop,float *out,
    int nx,int ny,int nz,int n,int axis) {
    int width=nx+(axis==0),height=ny+(axis==1),q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=width*height)return;
    int i=q%width,j=q/width;float zv;
    if(axis==0) {
        if(i==0)zv=__fdiv_rn(phb[j*nx],G);
        else if(i==nx)zv=__fdiv_rn(phb[j*nx+nx-1],G);
        else zv=__fdiv_rn(.5f*(phb[j*nx+i]+phb[j*nx+i-1]),G);
    } else {
        if(j==0)zv=__fdiv_rn(phb[i],G);
        else if(j==ny)zv=__fdiv_rn(phb[(ny-1)*nx+i],G);
        else zv=__fdiv_rn(.5f*(phb[j*nx+i]+phb[(j-1)*nx+i]),G);
    }
    float ps=sfia_interp(pm,z,zv,n),pt=ptop[0];
    for(int k=0;k<nz;++k) {
        float level=c3h[k]*(ps-pt)+c4h[k]+pt;
        out[k*width*height+q]=sfia_interp(wind,pm,level,n);
    }
}
