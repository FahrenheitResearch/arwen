// WRF v4.7.1 em_fire initialization, binary32 operation ordering.
#define IA __fadd_rn
#define IS __fsub_rn
#define IM __fmul_rn
#define ID __fdiv_rn

extern "C" __global__ void sfire_ideal_spacing(float *out,float dx,float dy,int rx,int ry) {
    if(blockIdx.x||threadIdx.x)return;
    out[0]=ID(dx,(float)rx);out[1]=ID(dy,(float)ry);
}

extern "C" __global__ void sfire_ideal_vertical(float *out,float *scalar,
        const float *eta,const float *ptop,int nz,int stretch,int hyperbolic,
        int explicit_eta,int hybrid,float scale,float etac,float dx,float dy) {
    if(blockIdx.x||threadIdx.x) return;
    int stride=nz+1;
    float *znw=out,*znu=out+stride,*dnw=out+2*stride,*rdnw=out+3*stride;
    float *dn=out+4*stride,*rdn=out+5*stride,*fnp=out+6*stride,*fnm=out+7*stride;
    float *c1f=out+8*stride,*c2f=out+9*stride,*c3f=out+10*stride,*c4f=out+11*stride;
    float *c1h=out+12*stride,*c2h=out+13*stride,*c3h=out+14*stride,*c4h=out+15*stride;
    for(int k=0;k<=nz;++k) {
        float fraction=ID((float)k,(float)nz);
        if(explicit_eta) znw[k]=eta[k];
        else if(stretch) {
            if(hyperbolic) znw[k]=ID(-sfire_tanhf(IM(scale,IS(fraction,1.0f))),sfire_tanhf(scale));
            else {
                float top=gfk_exp(-ID(1.0f,scale));
                znw[k]=ID(IS(gfk_exp(ID(-fraction,scale)),top),IS(1.0f,top));
            }
        } else znw[k]=IS(1.0f,fraction);
    }
    for(int k=0;k<nz;++k) {
        dnw[k]=IS(znw[k+1],znw[k]);
        rdnw[k]=ID(1.0f,dnw[k]);
        znu[k]=IM(0.5f,IA(znw[k+1],znw[k]));
    }
    for(int k=1;k<nz;++k) {
        dn[k]=IM(0.5f,IA(dnw[k],dnw[k-1]));
        rdn[k]=ID(1.0f,dn[k]);
        fnp[k]=ID(IM(0.5f,dnw[k]),dn[k]);
        fnm[k]=ID(IM(0.5f,dnw[k-1]),dn[k]);
    }
    float cof1=ID(IM(ID(IA(IM(2.0f,dn[1]),dn[2]),IA(dn[1],dn[2])),dnw[0]),dn[1]);
    float cof2=ID(IM(ID(dn[1],IA(dn[1],dn[2])),dnw[0]),dn[2]);
    scalar[0]=IA(fnp[1],cof1);
    scalar[1]=IS(IS(fnm[1],cof1),cof2);
    scalar[2]=cof2;
    scalar[3]=ID(IA(IM(0.5f,dnw[nz-1]),dn[nz-1]),dn[nz-1]);
    scalar[4]=ID(IM(-0.5f,dnw[nz-1]),dn[nz-1]);
    scalar[5]=ID(1.0f,dx);scalar[6]=ID(1.0f,dy);
    float pressure=IS(100000.0f,ptop[0]);
    for(int k=0;k<=nz;++k) {
        float z=znw[k];
        if(hybrid==0||hybrid==1) c3f[k]=z;
        else if(hybrid==2) {
            float e2=IM(etac,etac),e3=IM(e2,etac),one=IS(1.0f,etac);
            float b1=IM(IM(2.0f,e2),one);
            float b2=IM(-etac,IS(IS(4.0f,IM(3.0f,etac)),e3));
            float b3=IM(2.0f,IS(1.0f,e3)),b4=-IS(1.0f,e2);
            float one2=IM(one,one),b5=IM(one2,one2);
            float z2=IM(z,z),z3=IM(z2,z);
            c3f[k]=ID(IA(IA(IA(b1,IM(b2,z)),IM(b3,z2)),IM(b4,z3)),b5);
            if(z<etac) c3f[k]=0.0f;
            if(k==0)c3f[k]=1.0f;
            else if(k==nz)c3f[k]=0.0f;
        } else {
            float sine=glibc_sinf(IM(IM(0.5f,3.14159f),z));
            c3f[k]=IM(z,IM(sine,sine));
            if(k==0)c3f[k]=1.0f;
        }
        c4f[k]=IM(IS(z,c3f[k]),pressure);
    }
    for(int k=0;k<nz;++k) {
        znu[k]=IM(IA(znw[k+1],znw[k]),0.5f);
        c3h[k]=IM(IA(c3f[k+1],c3f[k]),0.5f);
        c4h[k]=IM(IS(znu[k],c3h[k]),pressure);
    }
    for(int k=1;k<nz;++k)c1f[k]=ID(IS(c3h[k],c3h[k-1]),IS(znu[k],znu[k-1]));
    c1f[0]=1.0f;c1f[nz]=(hybrid==0||hybrid==1)?1.0f:0.0f;
    for(int k=0;k<=nz;++k)c2f[k]=IM(IS(1.0f,c1f[k]),pressure);
    for(int k=0;k<nz;++k) {
        c1h[k]=ID(IS(c3f[k+1],c3f[k]),IS(znw[k+1],znw[k]));
        c2h[k]=IM(IS(1.0f,c1h[k]),pressure);
    }
}

extern "C" __global__ void sfire_ideal_mountain(float *height,int nx,int ny,
        float dx,float dy,int kind,float peak,float xs,float ys,float xe,float ye) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=nx*ny)return;
    int i=p%nx+1,j=p/nx+1;
    if(kind==0){height[p]=0.0f;return;}
    const float pi=0x1.921fb6p+1f;
    float ax=IS(IA(ID(xs,dx),1.0f),0.5f),bx=IS(IA(ID(xe,dx),1.0f),0.5f);
    float ay=IS(IA(ID(ys,dy),1.0f),0.5f),by=IS(IA(ID(ye,dy),1.0f),0.5f);
    float x=IA(pi,IM(IM(2.0f,pi),fmaxf(0.0f,fminf(ID(IS((float)i,ax),IS(bx,ax)),1.0f))));
    float y=IA(pi,IM(IM(2.0f,pi),fmaxf(0.0f,fminf(ID(IS((float)j,ay),IS(by,ay)),1.0f))));
    if(kind==1)height[p]=IM(IM(IM(peak,0.25f),IA(1.0f,glibc_cosf(x))),IA(1.0f,glibc_cosf(y)));
    else if(kind==2)height[p]=IM(IM(peak,0.5f),IA(1.0f,glibc_cosf(y)));
    else if(kind==3)height[p]=IM(IM(peak,0.5f),IA(1.0f,glibc_cosf(x)));
    else height[p]=0.0f;
}

extern "C" __global__ void sfire_ideal_gradient(const float *height,float *gx,float *gy,
        int nx,int ny,float dx,float dy) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=nx*ny)return;
    int i=p%nx+1,j=p/nx+1,pr=j*(nx+2)+i;
    gx[p]=ID(IS(height[pr+1],height[pr-1]),IM(2.0f,dx));
    gy[p]=ID(IS(height[pr+nx+2],height[pr-nx-2]),IM(2.0f,dy));
}

extern "C" __global__ void sfire_ideal_soil_depth(float *zs,float *dzs,int layers,int scheme) {
    if(blockIdx.x||threadIdx.x)return;
    if(scheme==1) {
        dzs[0]=0.01f;zs[0]=IM(0.5f,dzs[0]);
        for(int l=1;l<layers;++l) {dzs[l]=IM(2.0f,dzs[l-1]);zs[l]=IA(IA(zs[l-1],IM(0.5f,dzs[l-1])),IM(0.5f,dzs[l]));}
    } else if(scheme==2||scheme==4) {
        dzs[0]=0.1f;dzs[1]=0.3f;dzs[2]=0.6f;dzs[3]=1.0f;
        zs[0]=IM(0.5f,dzs[0]);
        for(int l=1;l<layers;++l)zs[l]=IA(IA(zs[l-1],IM(0.5f,dzs[l-1])),IM(0.5f,dzs[l]));
    } else if(scheme==3) {
        const float six[]={0.0f,0.05f,0.2f,0.4f,1.6f,3.0f};
        const float nine[]={0.0f,0.01f,0.04f,0.1f,0.3f,0.6f,1.0f,1.6f,3.0f};
        for(int l=0;l<layers;++l)zs[l]=(layers==6?six[l]:nine[l]);
        float old=0.0f,next=IM(IA(zs[1],zs[0]),0.5f);
        dzs[0]=IS(next,old);old=next;
        for(int l=1;l<layers-1;++l){next=IM(IA(zs[l+1],zs[l]),0.5f);dzs[l]=IS(next,old);old=next;}
        dzs[layers-1]=IS(zs[layers-1],old);
    }
}

extern "C" __global__ void sfire_ideal_soil(const float *tsk,const float *tmn,
        const float *zs,float *tslb,float *smois,int n,int layers,int scheme) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    if(scheme==1)for(int l=0;l<layers;++l)tslb[l*n+p]=ID(IA(IM(tsk[p],IS(zs[layers-1],zs[l])),IM(tmn[p],IS(zs[l],zs[0]))),IS(zs[layers-1],zs[0]));
    else if(scheme==2||scheme==4)for(int l=0;l<4;++l){tslb[l*n+p]=290.0f;smois[l*n+p]=0.30f;}
}

extern "C" __global__ void sfire_ideal_coordinates(float *x,float *y,int nx,int ny,float dx,float dy) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=nx*ny)return;
    x[p]=IM(IA((float)(p%nx),0.5f),dx);
    y[p]=IM(IA((float)(p/nx),0.5f),dy);
}

extern "C" __global__ void sfire_ideal_state_fields(const float *tinit,const float *theta,
        const float *pb,const float *perturbation,float *base_theta,float *thp,float *pressure,int n) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    base_theta[p]=IA(tinit[p],300.0f);
    thp[p]=IS(theta[p],tinit[p]);
    pressure[p]=IA(pb[p],perturbation[p]);
}

extern "C" __global__ void sfire_ideal_state_coordinate_drops(const float *c3f,const float *c4f,
        float *dc3f,float *dc4f,int nz) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=nz)return;
    dc3f[k]=IS(c3f[k],c3f[k+1]);dc4f[k]=IS(c4f[k],c4f[k+1]);
}

extern "C" __global__ void sfire_ideal_column_spacing(const float *phb,float *spacing,int n,int nz) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float old=ID(IM(IA(phb[p],phb[n+p]),0.5f),9.81f),minimum=0x1.fffffep+127f;
    for(int k=1;k<nz;++k){float next=ID(IM(IA(phb[k*n+p],phb[(k+1)*n+p]),0.5f),9.81f);minimum=fminf(minimum,IS(next,old));old=next;}
    spacing[p]=minimum;
}

extern "C" __global__ void sfire_ideal_landuse(const float *lu,const float *snowc,const float *xice,
        const float *snoalb,const float *initial_albbck,const float *table,float *out,int *ivg,int *status,
        int n,int cats,int seas,int season,int iswater,int isice,int fractional,int monalb,int nodata) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float value=lu[p];
    int category=value>=0.0f?(int)floorf(IA(value,0.5f)):(int)ceilf(IS(value,0.5f));
    ivg[p]=category;status[p]=0;
    if(category==0){category=iswater>0?iswater:nodata;if(iswater==0&&nodata>0)ivg[p]=nodata;}
    if(category<1||category>cats||isice<1||isice>cats){status[p]=1;return;}
    int r=((season-1)*cats+category-1)*7;
    float albbck=monalb?initial_albbck[p]:ID(table[r],100.0f);
    float albedo=albbck;
    if(snowc[p]>0.5f)albedo=monalb?snoalb[p]:IM(albbck,IA(1.0f,table[((seas-1)*cats+category-1)*7+5]));
    float thc=ID(table[r+4],100.0f),z0=ID(table[r+3],100.0f),embck=table[r+2],emiss=embck,mavail=table[r+1];
    float land=category==iswater?2.0f:1.0f;
    float threshold=fractional?0.02f:0.5f;
    if(xice[p]>=threshold) {
        land=1.0f;r=((season-1)*cats+isice-1)*7;
        albbck=ID(table[r],100.0f);embck=table[r+2];
        if(fractional){albedo=IA(IM(xice[p],albbck),IM(IS(1.0f,xice[p]),0.08f));emiss=IA(IM(xice[p],embck),IM(IS(1.0f,xice[p]),0.98f));}
        else {albedo=albbck;emiss=embck;}
        thc=ID(table[r+4],100.0f);z0=ID(table[r+3],100.0f);mavail=table[r+1];
    }
    out[p]=albedo;out[n+p]=albbck;out[2*n+p]=mavail;out[3*n+p]=emiss;out[4*n+p]=embck;
    out[5*n+p]=z0;out[6*n+p]=z0;out[7*n+p]=thc;out[8*n+p]=land;out[9*n+p]=xice[p];
    out[10*n+p]=land<1.5f?1.0f:0.0f;
}

extern "C" __global__ void sfire_moisture_surface_diagnostics(const float *temperature,const float *theta,
        const float *qv,const float *pressure,const float *coeff,float *t2,float *th2,float *q2,float *psfc,int n) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    t2[p]=IA(IA(IM(coeff[0],temperature[p]),IM(coeff[1],temperature[n+p])),IM(coeff[2],temperature[2*n+p]));
    th2[p]=IA(IA(IM(coeff[0],theta[p]),IM(coeff[1],theta[n+p])),IM(coeff[2],theta[2*n+p]));
    q2[p]=qv[p];psfc[p]=pressure[p];
}

#undef IA
#undef IS
#undef IM
#undef ID
