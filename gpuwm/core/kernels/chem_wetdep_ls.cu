// WRF v4.7.1 chem/module_wetdep_ls.F:49-91. One thread per column.
// chem_driver.F:778 supplies the mass-level count; sum nz-1, remove nz-2.
// All sums start at zero. No unwritten source scratch is read.
// Removed mass is an added dry-air diagnostic; it does not alter WRF arithmetic.
extern "C" __global__ void chem_wetdep_ls(
    float* var, const float* rain, const float* qc, const float* rho,
    const float* dryrho, const float* dz, const float* w, const float* alpha,
    float* removed, float dt, int nz, int nc, int nr) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= nc) return;
    for (int r=0; r<nr; ++r) {
        float a=alpha[r];
        if (a==0.0f || !(rain[c]>1.e-10f)) continue;
        float clw=0.0f, total=0.0f;
        for (int k=0; k<nz-1; ++k) {
            int x=k*nc+c, v=r*nz*nc+x;
            float d=fmaxf(0.0f,__fmul_rn(__fmul_rn(__fmul_rn(qc[x],rho[x]),w[x]),dz[x]));
            clw=__fadd_rn(clw,d);
            total=__fadd_rn(total,__fmul_rn(var[v],rho[x]));
        }
        if (!(total>1.e-10f && clw>1.e-10f)) continue;
        float frc=fmaxf(1.e-6f,fminf(__fdiv_rn(__fdiv_rn(rain[c],dt),clw),.005f));
        float mass=0.0f;
        for (int k=0; k<nz-2; ++k) {
            int x=k*nc+c, v=r*nz*nc+x;
            float old=var[v];
            if (old>1.e-16f && qc[x]>0.0f) {
                float fac=fmaxf(0.0f,__fmul_rn(__fmul_rn(__fmul_rn(frc,rho[x]),dz[x]),w[x]));
                float d=__fmul_rn(__fdiv_rn(__fmul_rn(a,fac),__fadd_rn(1.0f,fac)),old);
                var[v]=fmaxf(1.e-16f,__fsub_rn(old,d));
                mass=__fadd_rn(mass,__fmul_rn(__fmul_rn(__fsub_rn(old,var[v]),dryrho[x]),dz[x]));
            }
        }
        removed[r*nc+c]=__fadd_rn(removed[r*nc+c],mass);
    }
}
