// WRF v4.7.1 phys/module_fr_fire_phys.F:454-733. One thread owns a
// surface cell and updates the classes in WRF's order.
extern "C" __global__ void sfire_advance_moisture(
    const float *rainc, const float *rainnc, const float *t2, const float *q2,
    const float *psfc, float *rain_old, float *t2_old, float *q2_old,
    float *psfc_old, float *fmc_gc, float *fmep, float *equi_out,
    float *lag_out, float *rh_fire, const float *parameters,
    const int *initialization, int *status, int n, int nc, int initialize,
    float dt, float decay_lag, float fuelmc_g)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx>=n) return;
    if (!(t2[idx]>0.0f && psfc[idx]>0.0f && !(q2[idx]<0.0f))) {
        atomicExch(status,1);
        return;
    }
    float rain_sum=rainc[idx]+rainnc[idx];
    if (initialize) {
        rain_old[idx]=rain_sum; t2_old[idx]=t2[idx];
        q2_old[idx]=q2[idx]; psfc_old[idx]=psfc[idx];
    }
    float rain_diff=rain_sum-rain_old[idx];
    float rain_int=dt>0.0f?__fdiv_rn(3600.0f*rain_diff,dt):0.0f;
    float T=0.5f*(t2_old[idx]+t2[idx]);
    float P=0.5f*(psfc_old[idx]+psfc[idx]);
    float Q=0.5f*(q2_old[idx]+q2[idx]);
    float Pw=__fdiv_rn(Q*P,0.622f+(1.0f-0.622f)*Q);
    float Pws=gfk_exp(54.842763f-__fdiv_rn(6763.22f,T)-4.210f*gfk_log(T)+0.000367f*T
        +sfire_tanhf(0.0415f*(T-218.8f))*(53.878f-__fdiv_rn(1331.22f,T)-9.44523f*gfk_log(T)+0.014025f*T));
    float RH=__fdiv_rn(Pw,Pws);
    rh_fire[idx]=RH;
    RH=fminf(RH,1.0f);
    float deltaE=fmep[idx], deltaS=fmep[n+idx];
    for (int k=0;k<nc;k++) {
        const float *p=parameters+k*6;
        float R=rain_int-p[4];
        float EMC_w,EMC_d,rlag;
        if (R>0.0f) {
            EMC_w=p[2]+deltaS;
            EMC_d=p[2]+deltaS;
            rlag=__fdiv_rn(1.0f,3600.0f*p[1])*(1.0f-gfk_exp(__fdiv_rn(-R,p[3])));
        } else {
            float H=RH*100.0f;
            float d=0.942f*gfk_pow(H,0.679f)+0.000499f*gfk_exp(0.1f*H)
                +0.18f*(21.1f+273.15f-T)*(1.0f-gfk_exp(-0.115f*H));
            float w=0.618f*gfk_pow(H,0.753f)+0.000454f*gfk_exp(0.1f*H)
                +0.18f*(21.1f+273.15f-T)*(1.0f-gfk_exp(-0.115f*H));
            if (isnan(d)||isnan(w)) { atomicExch(status,2); return; }
            d=d*0.01f; w=w*0.01f;
            EMC_d=fmaxf(fmaxf(d,w)+deltaE,0.0f);
            EMC_w=fmaxf(fminf(d,w)+deltaE,0.0f);
            rlag=__fdiv_rn(1.0f,3600.0f*p[0]);
        }
        if (rlag>0.0f) {
            float old;
            int initial=initialization[k];
            if (!initialize || initial==0) old=fmc_gc[k*n+idx];
            else if (initial==1) old=fuelmc_g;
            else if (initial==2) old=0.5f*(EMC_d+EMC_w);
            else old=p[5];
            float equi=fmaxf(fminf(old,EMC_d),EMC_w);
            float change=dt*rlag;
            float fmc;
            if (change<1.0e-2f) fmc=old+(equi-old)*change*(1.0f-0.5f*change);
            else fmc=old+(equi-old)*(1.0f-gfk_exp(-change));
            fmc_gc[k*n+idx]=fmc;
            equi_out[k*n+idx]=equi;
            lag_out[k*n+idx]=__fdiv_rn(1.0f,3600.0f*rlag);
        }
    }
    float change=__fdiv_rn(dt,decay_lag*3600.0f);
    for (int k=0;k<2;k++) {
        if (change<1.0e-2f) fmep[k*n+idx]=fmep[k*n+idx]*(1.0f-change*(1.0f-0.5f*change));
        else fmep[k*n+idx]=fmep[k*n+idx]*gfk_exp(-change);
    }
    rain_old[idx]=rain_sum; t2_old[idx]=t2[idx];
    q2_old[idx]=q2[idx]; psfc_old[idx]=psfc[idx];
}

