// WRF 4.7.1 module_gocart_settling.F:12-367, chem_opt 300/401.
// A thread owns a column and a group with one growth arm. No local arrays.
extern "C" __global__ void chem_settling_gocart(
    const unsigned long long *rows, const double *radius, const double *density,
    const unsigned long long *accum, const unsigned long long *velocity,
    const float *temp, const float *pressure, const float *dz, const float *rho,
    const float *qv, double *work, int *nsteps, int nz, int nc, int nr,
    int gerber, float dt, float gravity, float dyn_visc) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double dzmin = dz[c];
    for (int k=1; k<nz; ++k) dzmin = fmin(dzmin, (double)dz[k*nc+c]);
    const double growth = gerber ? 3.0 : 1.0;
    for (int r=0; r<nr; ++r) {
        float *field = (float*)rows[r];
        // tc remains REAL*8 through all substeps; only the driver rounds it.
        for (int k=0; k<nz; ++k) {
            double x = field[k*nc+c];
            work[k*nc+c] = x < 0.0 ? 1.0e-32 : x;
        }
        double rr = radius[r], den = density[r];
        double gr = DMUL(growth, rr);
        double vs = DDIV(DMUL(DMUL((double)FMUL(4.0f/9.0f,gravity),den),
                                  DMUL(gr,gr)), (double)dyn_visc);
        int nt = (int)dt;
        int ns = max(1, (int)DDIV((double)nt, DDIV(dzmin,vs)));
        ns = min(ns,12);
        nsteps[r*nc+c] = ns;
        double ds = (double)FDIV((float)nt,(float)ns);
        double flux = 0.0, v0 = 0.0;
        for (int n=0; n<ns; ++n) {
            double transfer = 0.0;
            for (int k=nz-1; k>=0; --k) {
                int idx = k*nc+c;
                double t = temp[idx], wet = rr, rd = den;
                if (gerber) {
                    float tf = temp[idx];
                    float exponent = FDIV(FMUL(17.27f,FSUB(tf,273.0f)),FSUB(tf,36.0f));
                    float sat = FDIV(FMUL(3.80f,gfk_exp(exponent)),FMUL(0.01f,pressure[idx]));
                    double rh = fmax(0.1, (double)fminf(0.95f,FDIV(qv[idx],sat)));
                    double rcm = DMUL(rr,100.0);
                    // Parameters c1..c4 are single literals widened to double.
                    double a = DDIV(DMUL((double)0.7674f,pow(rcm,(double)3.079f)),
                        DSUB(DMUL((double)2.573e-11f,pow(rcm,(double)-1.424f)),log10(fmin(0.99,rh))));
                    wet = DMUL((double)0.01f,pow(DADD(a,DMUL(rcm,DMUL(rcm,rcm))), (double)0.33f));
                    double ratio = pow(DDIV(rr,wet),3.0);
                    rd = DADD(DMUL(ratio,den),DMUL(DSUB(1.0,ratio),1000.0));
                }
                double stokes = DDIV(DMUL((double)1.458e-6f,pow(t,1.5)),DADD(t,(double)110.4f));
                // p_mid is reversed on entry, then indexed by l2. Both reversals cancel.
                double pmid = (double)FMUL(0.01f,pressure[idx]);
                double path = DDIV(DDIV((double)1.1e-3f,pmid),sqrt(t));
                double slip = DADD(1.0,DMUL(DDIV(path,wet),DADD((double)1.257f,
                    DMUL((double)0.4f,exp(DDIV(DMUL((double)-1.1f,wet),path))))));
                double visc = DDIV(stokes,slip);
                double vd = DDIV(DMUL(DMUL((double)FMUL(2.0f/9.0f,gravity),rd),DMUL(wet,wet)),visc);
                double old = work[idx];
                double frac = DDIV(DMUL(ds,vd),(double)dz[idx]);
                work[idx] = DADD(DMUL(old,DSUB(1.0,frac)),transfer);
                if (k==0) {
                    flux = DADD(flux,DDIV(DMUL(vd,old),(double)ns));
                    v0 = vd;
                } else {
                    double massratio = DDIV(DMUL((double)dz[idx],(double)rho[idx]),
                        DMUL((double)dz[idx-nc],(double)rho[idx-nc]));
                    transfer = DMUL(DMUL(old,frac),massratio);
                }
            }
        }
        for (int k=0; k<nz; ++k) field[k*nc+c] = __double2float_rn(work[k*nc+c]);
        if (accum[r]) {
            float *a = (float*)accum[r];
            double increment = DMUL(DMUL(DMUL((double)1.e-9f,flux),(double)rho[c]),(double)dt);
            a[c] = __double2float_rn(DADD((double)a[c],increment));
        }
        if (velocity[r]) ((float*)velocity[r])[c] = __double2float_rn(v0);
    }
}
