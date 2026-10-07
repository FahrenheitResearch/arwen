// WRF module_dep_simple.F:1587-1614, aer_res_def arm (zr=2).
// rmol is a local INOUT copy. The model field is never modified.
__device__ float chem_aer_res(float &rmol, float z0, float ust) {
    if (fabsf(rmol) < 1.e-6f) rmol=0.0f;
    float polint;
    if (rmol<0.0f) {
        float ar0 = FADD(gfk_pow(FSUB(1.0f,FMUL(FMUL(9.0f,2.0f),rmol)),0.25f),0.001f);
        float ao0 = FADD(gfk_pow(FSUB(1.0f,FMUL(FMUL(9.0f,z0),rmol)),0.25f),0.001f);
        float ar = FMUL(ar0,ar0), ao = FMUL(ao0,ao0);
        polint = FMUL(0.74f,FSUB(gfk_log(FDIV(FSUB(ar,1.0f),FADD(ar,1.0f))),
                                     gfk_log(FDIV(FSUB(ao,1.0f),FADD(ao,1.0f)))));
    } else if (rmol==0.0f) {
        polint = FMUL(0.74f,gfk_log(FDIV(2.0f,z0)));
    } else {
        polint = FADD(FMUL(0.74f,gfk_log(FDIV(2.0f,z0))),FMUL(FMUL(4.7f,rmol),FSUB(2.0f,z0)));
    }
    return FDIV(polint,FMUL(0.4f,fmaxf(ust,1.e-4f)));
}

// WRF module_gocart_drydep.F:141-347, scalar column calculation.
__device__ double chem_depvel(double ust, double z0, double pbl, float rmol, float aer, int dust_arm) {
    double obk = rmol!=0.0f ? DDIV(1.0,(double)rmol) : 1.e5;
    double frac = fmin(DDIV(2.0,obk),1.0);
    double psi;
    if (frac>0.0 && frac<=1.0) psi=DMUL(-5.0,frac);
    else {
        double lm=log(fmin(1.0,-frac));
        psi=exp(DSUB(DADD(0.598,DMUL(0.39,lm)),DMUL(0.09,DMUL(lm,lm))));
    }
    double ra=DDIV(DSUB(log(DDIV(2.0,z0)),psi),DMUL(0.4,ust));
    double vds=DMUL(0.002,ust);
    if (obk<0.0) vds=DMUL(vds,DADD(1.0,pow(DDIV(-300.0,obk),0.6667)));
    double czh=DDIV(pbl,obk);
    if (czh < -30.0) vds=DMUL(DMUL(0.0009,ust),pow(-czh,0.6667));
    if (!dust_arm) ra=(double)aer;
    double rs=fmax(1.0,fmin(DDIV(1.0,fmin(vds,0.002)),9999.0));
    double rttl=DADD(DADD(ra,0.0),rs);
    return fmax(DADD(0.0,DDIV(1.0,rttl)),1.e-4);
}

extern "C" __global__ void chem_drydep_gocart(
    const unsigned long long *rows, const int *is_dust, const unsigned long long *accum,
    const float *rho, const float *rmol, const float *ust, const float *znt,
    const float *pbl, float *ddvel, float *aer_out, float *rmol_used,
    int nx, int ny, int nr, int x0, int y0, int ids, int ide, int jds, int jde,
    int dust_arm, float dt) {
    int c=blockIdx.x*blockDim.x+threadIdx.x, nc=nx*ny;
    if (c>=nc) return;
    int i=x0+c%nx, j=y0+c/nx;
    float rm=rmol[c];
    // Wesely precedes GOCART_SIMPLE only. DUST reads the original rmol.
    float aer=0.0f;
    if (!dust_arm) aer=chem_aer_res(rm,znt[c],ust[c]);
    aer_out[c]=aer; rmol_used[c]=rm;
    ddvel[c]=0.0f;
    if (i<=ids || i>=ide || j<=jds || j>=jde) return;
    double vd=chem_depvel(fmax(0.1,(double)ust[c]),(double)znt[c],(double)pbl[c],rm,aer,dust_arm);
    ddvel[c]=__double2float_rn(vd);
    // First multiply is REAL: 1.e-9*dtstep, before meeting dvel REAL*8.
    for (int r=0; r<nr; ++r) if (is_dust[r]) {
        float *a=(float*)accum[r];
        const float *q=(const float*)rows[r];
        double add=DMUL(DMUL(DMUL((double)FMUL(1.e-9f,dt),vd),(double)q[c]),(double)rho[c]);
        a[c]=__double2float_rn(DADD((double)a[c],add));
    }
}
