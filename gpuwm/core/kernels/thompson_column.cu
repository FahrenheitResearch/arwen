// One full, generated Thompson column. The registered generation supplies
// its statement program, init coefficients and table layout.
extern "C" __global__ void thompson_column_step(
    float* qv, float* qc, float* qi, float* qr, float* qs, float* qg,
    float* ni, float* nr, float* temperature,
    const float* pressure, const float* w, const float* dz,
    float* rainnc, float* rainncv, float* snownc, float* snowncv,
    float* graupelnc, float* graupelncv, float* sr,
    float* effc, float* effi, float* effs, float* refl, int diagnostic_due, int nz, int columns, float dt,
    const double* tcg_racg, const double* tmr_racg, const double* tcr_gacr,
    const double* tnr_racg, const double* tnr_gacr,
    const double* tcs_racs1, const double* tmr_racs1,
    const double* tcs_racs2, const double* tmr_racs2,
    const double* tcr_sacr1, const double* tms_sacr1,
    const double* tcr_sacr2, const double* tms_sacr2,
    const double* tnr_racs1, const double* tnr_racs2,
    const double* tnr_sacr1, const double* tnr_sacr2,
    const double* tpi_qrfz, const double* tni_qrfz,
    const double* tpg_qrfz, const double* tnr_qrfz,
    const double* tpi_qcfz, const double* tni_qcfz,
    const double* tps_iaus, const double* tni_iaus, const double* tpi_ide,
    const double* t_Efrw, const double* t_Efsw)
{
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    ThompsonColumn m;
    m.tcg_racg=tcg_racg;m.tmr_racg=tmr_racg;m.tcr_gacr=tcr_gacr;
    m.tnr_racg=tnr_racg;m.tnr_gacr=tnr_gacr;
    m.tcs_racs1=tcs_racs1;m.tmr_racs1=tmr_racs1;
    m.tcs_racs2=tcs_racs2;m.tmr_racs2=tmr_racs2;
    m.tcr_sacr1=tcr_sacr1;m.tms_sacr1=tms_sacr1;
    m.tcr_sacr2=tcr_sacr2;m.tms_sacr2=tms_sacr2;
    m.tnr_racs1=tnr_racs1;m.tnr_racs2=tnr_racs2;
    m.tnr_sacr1=tnr_sacr1;m.tnr_sacr2=tnr_sacr2;
    m.tpi_qrfz=tpi_qrfz;m.tni_qrfz=tni_qrfz;
    m.tpg_qrfz=tpg_qrfz;m.tnr_qrfz=tnr_qrfz;
    m.tpi_qcfz=tpi_qcfz;m.tni_qcfz=tni_qcfz;
    m.tps_iaus=tps_iaus;m.tni_iaus=tni_iaus;m.tpi_ide=tpi_ide;
    m.t_efrw=t_Efrw;m.t_efsw=t_Efsw;
    // Aerosol-only tables have no read in a classic generation.
    m.tnr_rev=nullptr;m.tpc_wev=nullptr;m.tnc_wev=nullptr;m.tnccn_act=nullptr;
    float qv1d[THOMPSON_COLUMN_KMAX],qc1d[THOMPSON_COLUMN_KMAX],qi1d[THOMPSON_COLUMN_KMAX],qr1d[THOMPSON_COLUMN_KMAX],qs1d[THOMPSON_COLUMN_KMAX],qg1d[THOMPSON_COLUMN_KMAX];
    float ni1d[THOMPSON_COLUMN_KMAX],nr1d[THOMPSON_COLUMN_KMAX],nc1d[THOMPSON_COLUMN_KMAX],nwfa1d[THOMPSON_COLUMN_KMAX],nifa1d[THOMPSON_COLUMN_KMAX],nbca1d[THOMPSON_COLUMN_KMAX];
    float t1d[THOMPSON_COLUMN_KMAX],p1d[THOMPSON_COLUMN_KMAX],w1d[THOMPSON_COLUMN_KMAX],dzq[THOMPSON_COLUMN_KMAX],re_c[THOMPSON_COLUMN_KMAX],re_i[THOMPSON_COLUMN_KMAX],re_s[THOMPSON_COLUMN_KMAX],dbz[THOMPSON_COLUMN_KMAX];
    for(int k=0;k<nz;k++) {
        const int idx=k*columns+col;
        qv1d[k]=qv[idx];qc1d[k]=qc[idx];qi1d[k]=qi[idx];
        qr1d[k]=qr[idx];qs1d[k]=qs[idx];qg1d[k]=qg[idx];
        ni1d[k]=ni[idx];nr1d[k]=nr[idx];
        t1d[k]=temperature[idx];p1d[k]=pressure[idx];
        w1d[k]=w[idx];dzq[k]=dz[idx];
        const float rho=0.622f*p1d[k]/(287.04f*t1d[k]*(qv1d[k]+0.622f));
        nc1d[k]=100.0e6f/rho;nwfa1d[k]=11.1e6f/rho;
        nifa1d[k]=(0.5e6f*0.01f)/rho;nbca1d[k]=5.55e6f/rho;
        re_c[k]=2.49e-6f;re_i[k]=4.99e-6f;re_s[k]=9.99e-6f;
    }
    float rain=0.0f,snow=0.0f,graupel=0.0f,ice=0.0f;
    m.mp_thompson(0,0,qv1d,qc1d,qi1d,qr1d,qs1d,qg1d,ni1d,nr1d,
        nc1d,nwfa1d,nifa1d,nbca1d,t1d,p1d,w1d,dzq,
        rain,snow,graupel,ice,1,nz,dt,1,1);
    m.calc_effectrad(t1d,p1d,qv1d,qc1d,nc1d,qi1d,ni1d,qs1d,
        re_c,re_i,re_s,1,nz);
    if(diagnostic_due) m.calc_refl10cm(qv1d,qc1d,qr1d,nr1d,qs1d,qg1d,t1d,p1d,dbz,1,nz,1,1,nz);
    rainncv[col]=rain+snow+graupel+ice;
    rainnc[col]=rainnc[col]+rain+snow+graupel+ice;
    snowncv[col]=snow+ice;snownc[col]=snownc[col]+snow+ice;
    graupelncv[col]=graupel;graupelnc[col]=graupelnc[col]+graupel;
    sr[col]=(snow+graupel+ice)/(rainncv[col]+1.0e-12f);
    for(int k=0;k<nz;k++) {
        const int idx=k*columns+col;
        qv[idx]=qv1d[k];qc[idx]=qc1d[k];qi[idx]=qi1d[k];
        qr[idx]=qr1d[k];qs[idx]=qs1d[k];qg[idx]=qg1d[k];
        ni[idx]=ni1d[k];nr[idx]=nr1d[k];temperature[idx]=t1d[k];
        effc[idx]=tc_max(2.49e-6f,tc_min(re_c[k],50.0e-6f))*1.0e6f;
        effi[idx]=tc_max(4.99e-6f,tc_min(re_i[k],125.0e-6f))*1.0e6f;
        effs[idx]=tc_max(9.99e-6f,tc_min(re_s[k],999.0e-6f))*1.0e6f;
        if(diagnostic_due)refl[idx]=tc_max(-35.0f,dbz[k]);
    }
}
