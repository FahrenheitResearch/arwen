// Six-class single-moment Purdue Lin column. Native arrays use one-based levels.
extern "C" __global__ void lin_column(
    float* th, float* qv, float* qc, float* qr, float* qi, float* qs, float* qg,
    const float* rho, const float* pii, const float* p, const float* z,
    const float* dz, const float* ht,
    float* rainnc, float* rainncv, float* snownc, float* snowncv,
    float* graupelnc, float* graupelncv, float* sr,
    float dt, int nz, int ncol) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float qvz[LIN_KMAX+1], qlz[LIN_KMAX+1], qrz[LIN_KMAX+1];
    float qiz[LIN_KMAX+1], qsz[LIN_KMAX+1], qgz[LIN_KMAX+1];
    float thz[LIN_KMAX+1], tothz[LIN_KMAX+1], rhoz[LIN_KMAX+1];
    float orhoz[LIN_KMAX+1], sqrhoz[LIN_KMAX+1], prez[LIN_KMAX+1];
    float zz[LIN_KMAX+1], dzw[LIN_KMAX+1], qndropz[LIN_KMAX+1];
    float preclw[LIN_KMAX+1], precrz[LIN_KMAX+1], preciz[LIN_KMAX+1];
    float precsz[LIN_KMAX+1], precgz[LIN_KMAX+1];
    for (int k=1; k<=nz; ++k) {
        int at = (k-1)*ncol+col;
        qvz[k]=qv[at]; qlz[k]=qc[at]; qrz[k]=qr[at];
        qiz[k]=qi[at]; qsz[k]=qs[at]; qgz[k]=qg[at]; thz[k]=th[at];
        rhoz[k]=rho[at]; orhoz[k]=1.0f/rhoz[k];
        sqrhoz[k]=sqrtf(1.29f*orhoz[k]);
        tothz[k]=pii[at]; prez[k]=p[at]; zz[k]=z[at]; dzw[k]=dz[at];
        qndropz[k]=100.e6f;
    }
    float pptrain=0.f, pptsnow=0.f, pptgraul=0.f, pptice=0.f;
    float grav=9.81f, cp=1004.5f, rair=287.f, rvapor=461.6f;
    float xls=2.85e6f, xlv=2.5e6f, xlf=3.50e5f;
    float rhowater=1000.f, rhosnow=100.f, gindex=1.f;
    float ep2=287.f/461.6f, svp1=.6112f, svp2=17.67f, svp3=29.65f, svpt0=273.15f;
    float zsfc=ht[col]; int kts=1, kte=nz, i=1, j=1; bool flag_qndrop=false;
    lin_clphy1d(&dt,qvz,qlz,qrz,qiz,qsz,qgz,qndropz,&flag_qndrop,
        thz,tothz,rhoz,orhoz,sqrhoz,prez,zz,dzw,&zsfc,preclw,
        precrz,preciz,precsz,precgz,&pptrain,&pptsnow,&pptgraul,&pptice,
        &grav,&cp,&rair,&rvapor,&gindex,&xls,&xlv,&xlf,&rhowater,&rhosnow,
        &ep2,&svp1,&svp2,&svp3,&svpt0,&kts,&kte,&i,&j);
    sr[col]=(pptice+pptsnow+pptgraul)/(pptice+pptsnow+pptgraul+pptrain+1.e-12f);
    rainncv[col]=pptrain+pptsnow+pptgraul+pptice;
    rainnc[col]=rainnc[col]+pptrain+pptsnow+pptgraul+pptice;
    snowncv[col]=pptsnow+pptice; snownc[col]=snownc[col]+pptsnow+pptice;
    graupelncv[col]=pptgraul; graupelnc[col]=graupelnc[col]+pptgraul;
    for (int k=1; k<=nz; ++k) {
        int at=(k-1)*ncol+col;
        th[at]=thz[k]; qv[at]=qvz[k]; qc[at]=qlz[k]; qr[at]=qrz[k];
        qi[at]=qiz[k]; qs[at]=qsz[k]; qg[at]=qgz[k];
    }
}
