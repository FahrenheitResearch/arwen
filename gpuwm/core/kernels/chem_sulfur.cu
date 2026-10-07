// module_gocart_chem.F: GOCART_SIMPLE only. One thread per column.
// STUB until the engine's glibc float32 trig header lands
__device__ float chem_sinf(float x) { return sinf(x); }
__device__ float chem_cosf(float x) { return cosf(x); }
__device__ float chem_acosf(float x) { return acosf(x); }

// szangle:753-823. sza is unused, so its acos and conversion are omitted.
__device__ float chem_cossza(int doy, float hour, float lon, float lat) {
    float r = FDIV(FMUL(FMUL(2.f, 3.14f), (float)(doy-1)), 365.f);
    float dec = FSUB(0.006918f, FMUL(0.399912f, chem_cosf(r)));
    dec = FADD(dec, FMUL(0.070257f, chem_sinf(r)));
    dec = FSUB(dec, FMUL(0.006758f, chem_cosf(FMUL(2.f,r))));
    dec = FADD(dec, FMUL(0.000907f, chem_sinf(FMUL(2.f,r))));
    dec = FSUB(dec, FMUL(0.002697f, chem_cosf(FMUL(3.f,r))));
    dec = FADD(dec, FMUL(0.000148f, chem_sinf(FMUL(3.f,r))));
    float local = FADD(hour, FDIV(lon,15.f));
    if (local > 24.f) local = FSUB(local,24.f);
    float ahr = FDIV(FMUL(FMUL(fabsf(FSUB(local,12.f)),15.f),3.14f),180.f);
    float c = FADD(FMUL(chem_sinf(lat),chem_sinf(dec)),
                   FMUL(FMUL(chem_cosf(lat),chem_cosf(dec)),chem_cosf(ahr)));
    if (c < 0.f) c = 0.f;
    return c;
}
extern "C" __global__ void chem_sulfur_solar_init(
    const float* lat, const float* lon, float* tcosz, float* ttday,
    int nc, float dt, float gmt, int julday) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (col >= nc) return;
    float rlat = FDIV(FMUL(lat[col],3.1415926535590f),180.f);
    float sum = 0.f, day = 0.f;
    int ndystep = 86400 / (int)dt;
    for (int n=1; n<=ndystep; ++n) {
        float xtime = FDIV(FMUL((float)n,dt),60.f);
        int ixhour = (int)FADD(gmt,.01f) + (int)FDIV(xtime,60.f);
        float xhour = (float)ixhour;
        float xmin = FADD(FMUL(60.f,gmt),FSUB(xtime,FMUL(xhour,60.f)));
        float gmtp = fmodf(xhour,24.f);
        gmtp = FADD(gmtp,FDIV(xmin,60.f));
        float c = chem_cossza(julday,gmtp,lon[col],rlat);
        sum = FADD(sum,c);
        if (c > 0.f) day = FADD(day,dt);
    }
    tcosz[col]=sum; ttday[col]=day;
}
extern "C" __global__ void chem_sulfur_solar_step(
    const float* lat, const float* lon, float* cossza,
    int nc, double curr_secs, float gmt, int julday) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (col >= nc) return;
    double xtime = DDIV(curr_secs,60.0);
    long long ixhour = (long long)FADD(gmt,.01f) + (long long)DDIV(xtime,60.0);
    double xhour = (double)ixhour;
    float xmin = __double2float_rn(DADD((double)FMUL(60.f,gmt),
                                      DSUB(xtime,DMUL(xhour,60.0))));
    float gmtp = __double2float_rn(fmod(xhour,24.0));
    gmtp = FADD(gmtp,FDIV(xmin,60.f));
    float rlat = FDIV(FMUL(lat[col],3.1415926535590f),180.f);
    cossza[col] = chem_cossza(julday,gmtp,lon[col],rlat);
}

// chem_dms:267-438. Diagnostics do not feed any species.
__device__ void chem_dms(double* tc, double tk, double rho, double oh,
                         double no3, double cossza, float seconds,
                         double* pso2, double* pmsa) {
    // airmw is REAL, so the first quotient of this PARAMETER is float32.
    const double f = DMUL(DMUL((double)(1000.f/28.97f),6.022e23),1.e-6);
    double o2 = DMUL(DMUL(rho,f),(double).21f);
    double dms0 = tc[0], rk1=0.0, rk2=0.0, rk3=0.0;
    if (oh > 0.0) {
        rk1 = DDIV(DMUL(DMUL(1.7e-42,exp(DDIV(7810.0,tk))),o2),
                   DADD(1.0,DMUL(DMUL(5.5e-31,exp(DDIV(7460.0,tk))),o2)));
        rk1 = DMUL(DMUL(DMUL(rk1,oh),rho),f);
        rk2 = DMUL(DMUL(DMUL(DMUL(1.2e-11,exp(DDIV(-260.0,tk))),oh),rho),f);
    }
    if (cossza <= 0.0)
        rk3 = DMUL(DMUL(DMUL(DMUL(1.9e-13,exp(DDIV(500.0,tk))),no3),rho),f);
    double dms_oh = DMUL(dms0,exp(DMUL(DMUL(-DADD(rk1,rk2),1.0),(double)seconds)));
    double dms = DMUL(dms_oh,exp(DMUL(DMUL(-rk3,1.0),(double)seconds)));
    dms = fmax(dms,1.e-32); tc[0]=dms;
    if (DADD(rk1,rk2) == 0.0) *pmsa=0.0;
    else *pmsa = fmax(0.0,DMUL(DDIV(DMUL(DMUL(DSUB(dms0,dms_oh),.25),rk1),
                                      DMUL(DADD(rk1,rk2),1.0)),1.0));
    *pso2 = fmax(0.0,DSUB(DSUB(dms0,dms),*pmsa));
}
// chem_so2:442-604. The prescribed H2O2 is not written back by the driver.
__device__ double chem_so2(double* tc, double tk, double rho, double oh,
                           double h2o2, double cloud, float seconds, double pso2) {
    const double f = DMUL(DMUL((double)(1000.f/28.97f),6.022e23),1.e-6);
    double so20=tc[1];
    double k0=DMUL(3.e-31,pow(DDIV(300.0,tk),(double)3.3f));
    double m=DMUL(rho,f), kk=DDIV(DMUL(k0,m),1.5e-12);
    double lg=log10(kk);
    double f1=DDIV(1.0,DADD(1.0,DMUL(lg,lg)));
    double rk1=DMUL(DDIV(DMUL(k0,m),DADD(1.0,kk)),pow((double).6f,f1));
    rk1=DMUL(DMUL(DMUL(rk1,oh),rho),f);
    double rk=DADD(rk1,0.0), rkt=DMUL(rk,(double)seconds);
    double cd,l1;
    if (rk > 0.0) {
        cd=DADD(DMUL(so20,exp(-rkt)),DDIV(DMUL(pso2,DSUB(1.0,exp(-rkt))),rkt));
        l1=DDIV(DMUL(DADD(DSUB(so20,cd),pso2),rk1),rk);
    } else { cd=so20; l1=0.0; }
    double fc=cloud, so2,l2;
    if (fc > 0.0 && cd > 0.0 && tk > 258.0) {
        if (cd > h2o2) fc=DMUL(fc,DDIV(h2o2,cd));
        so2=DMUL(cd,DSUB(1.0,fc)); l2=DMUL(cd,fc);
    } else { so2=cd; l2=0.0; }
    tc[1]=fmax(so2,1.e-32);
    return fmax(0.0,DADD(l1,l2));
}
extern "C" __global__ void chem_sulfur_gocart(
    const unsigned long long* fields, const int* role_row,
    const float* temp, const float* rho, const float* qc, const float* qi,
    const float* gd_cldf, const float* backg_oh, const float* backg_h2o2,
    const float* backg_no3, const float* cossza, const float* tcosz,
    const float* ttday, int nz, int nc, float dt, int has_qc, int has_qi) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col >= nc) return;
    double cosz=(double)cossza[col];
    for (int k=0;k<nz-1;++k) {
        int idx=k*nc+col;
        double tc[4];
        for (int r=0;r<4;++r) tc[r] = role_row[r]<0 ? 0.0 :
            DMUL((double)((float*)fields[role_row[r]])[idx],1.e-6);
        double cloud=gd_cldf ? (double)gd_cldf[idx] : 0.0;
        if (has_qc && has_qi) {
            if (qc[idx]>0.f || qi[idx]>0.f) cloud=1.0;
        } else if (has_qc && qc[idx]>0.f) cloud=1.0;
        double oh=DDIV(DMUL(DMUL((double)FDIV(86400.f,dt),cosz),
                                (double)backg_oh[idx]),(double)tcosz[col]);
        double no3=cosz>0.0 ? 0.0 : (double)FDIV(backg_no3[idx],
                                 FSUB(1.f,FDIV(ttday[col],86400.f)));
        double pso2,pmsa;
        chem_dms(tc,(double)temp[idx],(double)rho[idx],oh,no3,cosz,
                 (float)((int)dt),&pso2,&pmsa);
        double pso4=chem_so2(tc,(double)temp[idx],(double)rho[idx],oh,
                           (double)backg_h2o2[idx],cloud,(float)((int)dt),pso2);
        // chem_so4:608-679 and chem_msa:683-752.
        tc[2]=fmax(DADD(tc[2],pso4),1.e-32);
        tc[3]=fmax(DADD(tc[3],pmsa),1.e-32);
        for (int r=0;r<4;++r) if (role_row[r]>=0)
            ((float*)fields[role_row[r]])[idx]=__double2float_rn(DMUL(tc[r],(double)1.e6f));
    }
}
