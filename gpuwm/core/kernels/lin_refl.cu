// WRF v4.6.1 module_mp_lin.F:2757-2927. Shared radar scattering helpers.
extern "C" __global__ void refl10cm_lin_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ qs, const real* __restrict__ qg,
    const real* __restrict__ t, const real* __restrict__ p,
    const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im, const double xam_g,
    const double n0_g, real* __restrict__ refl,
    const int nz, const int ny, const int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx, i = col - j * nx;
    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* xxdg = tables + 2 * REFL_NRBINS;
    const double* xdtg = tables + 3 * REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const cd mw = cd_make(m_w0_re, m_w0_im);
    const cd mi = cd_make(m_i0_re, m_i0_im);
    const double xam_r = 523.59832763671875;
    const double xam_s = 52.359832763671875;
    const double xobm = 1.0 / 3.0;
    const double xocms = pow(1.0 / xam_s, xobm);
    const double xocmg = pow(1.0 / xam_g, xobm);
    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    double n0s[REFL_KMAX];
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];
    for (int k = 0; k < nz; ++k) {
        size_t idx = I3(k, j, i, ny, nx);
        double tk = (double)t[idx];
        double qvk = fmax(1.0e-10, (double)qv[idx]);
        float rho = (0.622f * p[idx]) / (287.0f * t[idx] * (fmaxf(1.0e-10f, qv[idx]) + 0.622f));
        rr[k] = rs[k] = rg[k] = 1.0e-12f;
        lqr[k] = lqs[k] = lqg[k] = false;
        if (qr[idx] > 1.0e-9f) {
            rr[k] = qr[idx] * rho;
            double lam = pow(xam_r * RGAMMA_4 * 8.0e6 / rr[k], 0.25);
            ilamr[k] = 1.0 / lam; lqr[k] = true;
        }
        double tc = fmin(-0.001, tk - 273.15);
        n0s[k] = 3.0e6;
        if (qs[idx] > 1.0e-9f) {
            rs[k] = qs[idx] * rho;
            double lam = pow(xam_s * RGAMMA_4 * n0s[k] / rs[k], 0.25);
            ilams[k] = 1.0 / lam; lqs[k] = true;
        }
        if (qg[idx] > 1.0e-9f) {
            rg[k] = qg[idx] * rho;
            double lam = pow(xam_g * RGAMMA_4 * n0_g / rg[k], 0.25);
            ilamg[k] = 1.0 / lam; lqg[k] = true;
        }
    }
    bool melti = false; int k0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (t[I3(k,j,i,ny,nx)] > 273.15f && lqr[k]
            && (lqs[k+1] || lqg[k+1])) {
            k0 = max(k + 1, k0); melti = true; break;
        }
    }
    for (int k = 0; k < nz; ++k) {
        zer[k] = zes[k] = zeg[k] = 1.0e-22f;
        if (lqr[k]) zer[k] = (float)(8.0e6 * RGAMMA_7 * pow(ilamr[k],7.0));
        if (lqs[k]) zes[k] = (float)((double)((0.176f/0.93f)*(6.0f/3.1415926536f)*(6.0f/3.1415926536f))
            *pow(xam_s/900.0,2.0)*n0s[k]*RGAMMA_7*pow(ilams[k],7.0));
        if (lqg[k]) zeg[k] = (float)((double)((0.176f/0.93f)*(6.0f/3.1415926536f)*(6.0f/3.1415926536f))
            *pow(xam_g/900.0,2.0)*n0_g*RGAMMA_7*pow(ilamg[k],7.0));
    }
    if (melti && k0 >= 1) {
        for (int k = k0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k0]) {
                double fm = fmax(0.005, fmin(1.0-(double)(rs[k]/rs[k0]),0.99));
                double eta=0.0, lam=1.0/ilams[k];
                for(int n=0;n<REFL_NRBINS;++n){
                    double x=xam_s*xxds[n]*xxds[n]*xxds[n];
                    double cb=refl_rayleigh_soak_wetgraupel(
                        x,xocms,xobm,fm,RMELT_OUTSIDE,mw,mi);
                    eta += n0s[k]*exp(-lam*xxds[n])*cb*simpson[n]*xdts[n];
                }
                zes[k]=(float)(RLAMDA4/(RPI5*k_w)*eta);
            }
            if (lqg[k] && lqg[k0]) {
                double fm=fmax(0.005,fmin(1.0-(double)(rg[k]/rg[k0]),0.99));
                double eta=0.0, lam=1.0/ilamg[k];
                for(int n=0;n<REFL_NRBINS;++n){
                    double x=xam_g*xxdg[n]*xxdg[n]*xxdg[n];
                    double cb=refl_rayleigh_soak_wetgraupel(
                        x,xocmg,xobm,fm,RMELT_OUTSIDE,mw,mi);
                    eta += n0_g*exp(-lam*xxdg[n])*cb*simpson[n]*xdtg[n];
                }
                zeg[k]=(float)(RLAMDA4/(RPI5*k_w)*eta);
            }
        }
    }
    for(int k=0;k<nz;++k){
        float zsum=zer[k]+zes[k]+zeg[k];
        refl[I3(k,j,i,ny,nx)]=fmaxf(-35.0f,(float)(10.0*log10((double)zsum*1e18)));
    }
}
