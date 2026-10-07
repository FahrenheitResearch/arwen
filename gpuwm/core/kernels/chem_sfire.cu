// WRF-SFIRE native g/kg-air increment -> existing AQ ug/kg-dry state.
// The native emission increment is computed by the pinned sfire_atm port.
extern "C" __global__ void chem_sfire_units(
    float *out, const float *value, const float *alt, const float *qv,
    int count, int to_native) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    if(p>=count) return;
    float dryrho=__fdiv_rn(1.f,alt[p]);
    float rho=__fmul_rn(dryrho,__fadd_rn(1.f,qv[p]));
    float ratio=__fdiv_rn(rho,dryrho);
    out[p]=to_native ? __fdiv_rn(__fmul_rn(value[p],1.e-6f),ratio)
                     : __fmul_rn(__fmul_rn(value[p],1.e6f),ratio);
}

extern "C" __global__ void chem_sfire_apply(
    float *q, const float *native_increment, const float *rho,
    const float *dryrho, const float *dz, const float *burnt,
    const float *fuel, const float *mx, const float *my,
    double *expected_kg, double *injected_kg, float *emitted_areal,
    int nx, int ny, int nz, int rx, int ry, float yield, double dxdy,
    int xl, int xh, int yl, int yh) {
    int p=blockDim.x*blockIdx.x+threadIdx.x;
    int nc=nx*ny;
    if(p>=nc) return;
    int i=p%nx,j=p/nx;
    if(i<xl||i>xh||j<yl||j>yh) return;
    double introduced_areal=0.0;
    for(int k=0;k<nz;++k) {
        int x=k*nc+p;
        float increment=__fmul_rn(__fmul_rn(native_increment[x],1.e6f),
                                  __fdiv_rn(rho[x],dryrho[x]));
        float old=q[x];
        q[x]=__fadd_rn(old,increment);
        introduced_areal+=((double)q[x]-(double)old)*(double)dryrho[x]*(double)dz[x]*1.e-9;
    }
    double area=__ddiv_rn(dxdy,(double)mx[p]*(double)my[p]);
    double consumed=0.0;
    for(int jy=0;jy<ry;++jy) for(int ix=0;ix<rx;++ix) {
        int f=(j*ry+jy)*(nx*rx)+i*rx+ix;
        consumed+=(double)burnt[f]*(double)fuel[f];
    }
    expected_kg[p]+=__ddiv_rn(consumed*(double)yield,(double)(rx*ry))*area;
    injected_kg[p]+=introduced_areal*area;
    emitted_areal[p]=__fadd_rn(emitted_areal[p],(float)(introduced_areal*1.e9));
}
