// Atmospheric SFIRE adapter operations. Every field operation stays on device.
extern "C" __global__ void sfire_geographic_ignition_units(const float *latitude,float *out,int n) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=n)return;
    // module_model_constants REAL parameters and fire_ignition_convert order.
    float pi2=__fmul_rn(2.0f,3.1415926f);
    float reradius=__fdiv_rn(1.0f,6370.0e3f);
    float unit_y=__fdiv_rn(pi2,__fmul_rn(360.0f,reradius));
    float angle=__fdiv_rn(__fmul_rn(latitude[p],pi2),360.0f);
    out[p]=__fmul_rn(glibc_cosf(angle),unit_y);
    out[n+p]=unit_y;
}
extern "C" __global__ void sfire_pad_plane(const float *src,float *out,int nx,int ny,int linear) {
    int p=blockDim.x*blockIdx.x+threadIdx.x,px=nx+2,py=ny+2;
    if(p>=px*py)return;
    int x=p%px-1,y=p/px-1,cx=max(0,min(nx-1,x)),cy=max(0,min(ny-1,y));
    float a=src[cy*nx+cx];
    if(linear && (x<0 || x>=nx || y<0 || y>=ny)) {
        int dx=x<0?1:(x>=nx?-1:0),dy=y<0?1:(y>=ny?-1:0);
        if(nx==1)dx=0;if(ny==1)dy=0;
        a=2.0f*a-src[(cy+dy)*nx+cx+dx];
    }
    out[p]=a;
}
extern "C" __global__ void sfire_plane_floor(float *a,int n,float lo) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p<n)a[p]=fmaxf(a[p],lo);
}
extern "C" __global__ void sfire_static_gradient(const float *z,float *gx,float *gy,int nx,int ny,float dx,float dy) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=nx*ny)return;
    int x=p%nx,y=p/nx,l=max(0,x-1),r=min(nx-1,x+1),b=max(0,y-1),t=min(ny-1,y+1);
    gx[p]=__fdiv_rn(z[y*nx+r]-z[y*nx+l],float(r-l)*dx);
    gy[p]=__fdiv_rn(z[t*nx+x]-z[b*nx+x],float(t-b)*dy);
}
extern "C" __global__ void sfire_set_nfuel(float *cat,const float *z,int n,int read,int uniform) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=n)return;
    if(read==0)cat[p]=float(uniform);
    if(read==1)cat[p]=z[p]<=1524.0f?3.0f:(z[p]<=2073.0f?2.0f:(z[p]<=2438.0f?8.0f:(z[p]<=3354.0f?10.0f:(z[p]<=3658.0f?1.0f:14.0f))));
}
extern "C" __global__ void sfire_feedback(const float *heat,const float *water,const float *fuel,
    float *gh,float *gq,float *ghfu,float *gqfu,float *af,int n,int refinement,float feedback) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=n)return;
    float s=__fdiv_rn(1.0f,float(refinement));
    ghfu[p]=heat[p]*s;gqfu[p]=water[p]*s;af[p]=fuel[p]*s;
    gh[p]=(feedback*heat[p])*s;gq[p]=(feedback*water[p])*s;
}
extern "C" __global__ void sfire_stage_tendencies(const float *th,const float *qv,
    const float *msft,float *rt,float *rq,int n,int plane,int mapped) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=n)return;
    // rk_addtend_dry divides thermal forcing by MSFTY. Moist scalar
    // tendencies retain coupled mass units for rk_update_scalar.
    rt[p]=mapped?__fdiv_rn(th[p],msft[p%plane]):th[p];rq[p]=qv[p];
}
