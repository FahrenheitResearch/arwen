// SFIRE physical routines from WRF v4.7.1 module_fr_fire_phys.F.
// Input fuel rows use 1-based internal categories, with 60 rows total.
extern "C" __global__ void sfire_set_params(
    const float *nfuel_cat, float *fmc_g, const float *table,
    float *fgip, float *ischap, float *betafl, float *bbb,
    float *fuel_time, float *phiwc, float *r0, float *iboros,
    int n, int fallback, int no_fuel_cat, int nfuelcats, int fmc_read, int *status,
    float fuelmc_g, float fuelmc_g_lh, float fuelheat)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    int k = sfire_category((int)nfuel_cat[idx], fallback);
    const float *t = table + (k-1)*10;
    if (k == no_fuel_cat) {
        fgip[idx]=0.0f; ischap[idx]=0.0f; betafl[idx]=1.0f; bbb[idx]=1.0f;
        fuel_time[idx]=__fdiv_rn(7.0f,0.85f); phiwc[idx]=0.0f;
        r0[idx]=0.0f; iboros[idx]=0.0f;
        return;
    }
    if (k > nfuelcats) { atomicExch(status,1); return; }
    if (fmc_read == 1) fmc_g[idx] = fuelmc_g;
    float fgi=t[0], fgi_lh=t[1], depth=t[2], savr=t[3], mce=t[4];
    float density=t[5], st=t[6], se=t[7], weight=t[8];
    fuel_time[idx]=__fdiv_rn(weight,0.85f);
    ischap[idx]=t[9];
    if (fuelmc_g_lh > 0.3f && fuelmc_g_lh < 1.2f)
        fgip[idx]=fgi+(1.0f-__fdiv_rn(fuelmc_g_lh-0.3f,0.9f))*fgi_lh;
    else if (fuelmc_g_lh <= 0.3f) fgip[idx]=fgi+fgi_lh;
    else fgip[idx]=fgi;
    if (fgip[idx] == 0.0f) {
        ischap[idx]=0.0f; betafl[idx]=1.0f; bbb[idx]=1.0f;
        phiwc[idx]=0.0f; r0[idx]=0.0f; iboros[idx]=0.0f;
        return;
    }
    float moisture=fmc_g[idx];
    float bmst=__fdiv_rn(moisture,1.0f+moisture);
    float fuelloadm=(1.0f-bmst)*fgip[idx];
    float fuelload=fuelloadm*(0.3048f*0.3048f)*2.205f;
    float fueldepth=__fdiv_rn(depth,0.3048f);
    betafl[idx]=__fdiv_rn(fuelload,fueldepth*density);
    float betaop=3.348f*gfk_pow(savr,-0.8189f);
    float qig=250.0f+1116.0f*moisture;
    float epsilon=gfk_exp(__fdiv_rn(-138.0f,savr));
    float rhob=__fdiv_rn(fuelload,fueldepth);
    float c=7.47f*gfk_exp(-0.133f*gfk_pow(savr,0.55f));
    bbb[idx]=0.02526f*gfk_pow(savr,0.54f);
    float e=0.715f*gfk_exp(-3.59e-4f*savr);
    phiwc[idx]=c*gfk_pow(__fdiv_rn(betafl[idx],betaop),-e);
    float rtemp2=gfk_pow(savr,1.5f);
    float gammax=__fdiv_rn(rtemp2,495.0f+0.0594f*rtemp2);
    float a=__fdiv_rn(1.0f,4.774f*gfk_pow(savr,0.1f)-7.27f);
    float ratio=__fdiv_rn(betafl[idx],betaop);
    float gamma=gammax*gfk_pow(ratio,a)*gfk_exp(a*(1.0f-ratio));
    float wn=__fdiv_rn(fuelload,1.0f+st);
    float rtemp1=__fdiv_rn(moisture,mce);
    float etam=1.0f-2.59f*rtemp1+5.11f*(rtemp1*rtemp1)-3.52f*((rtemp1*rtemp1)*rtemp1);
    // Extinguished fuel cannot spread backwards or release negative intensity.
    // WRF omits the extinction bound and its polynomial becomes negative.
    if (moisture >= mce) etam=0.0f;
    else etam=fmaxf(etam,0.0f);
    float etas=0.174f*gfk_pow(se,-0.19f);
    float ir=gamma*wn*fuelheat*etam*etas;
    iboros[idx]=__fdiv_rn(ir*1055.0f,(0.3048f*0.3048f)*60.0f)*1.0e-3f*__fdiv_rn(60.0f*12.6f,savr);
    float xifr=__fdiv_rn(gfk_exp((0.792f+0.681f*gfk_pow(savr,0.5f))*(betafl[idx]+0.1f)),192.0f+0.2595f*savr);
    r0[idx]=__fdiv_rn(ir*xifr,rhob*epsilon*qig);
}

extern "C" __global__ void sfire_ros(
    const float *px, const float *py, const float *vx, const float *vy,
    const float *dzdx, const float *dzdy, const float *bbb,
    const float *betafl, const float *phiwc, const float *r0, const float *ischap,
    float *base, float *wind, float *slope, int n, int advection)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx>=n) return;
    sfire_ros_device(base[idx],wind[idx],slope[idx],px[idx],py[idx],vx[idx],vy[idx],
                     dzdx[idx],dzdy[idx],bbb[idx],betafl[idx],phiwc[idx],r0[idx],ischap[idx],advection);
}

extern "C" __global__ void sfire_heat_fluxes(
    const float *fgip, const float *burnt, const float *fmc,
    float *hfx, float *qfx, float *dry_burnt, int n, float dt, float cmbcnst, float xlv)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx>=n) return;
    float dmass=fgip[idx]*burnt[idx];
    float bmst=__fdiv_rn(fmc[idx],1.0f+fmc[idx]);
    hfx[idx]=__fdiv_rn(dmass,dt)*(1.0f-bmst)*cmbcnst;
    qfx[idx]=(bmst+(1.0f-bmst)*0.56f)*__fdiv_rn(dmass,dt)*xlv;
    dry_burnt[idx]=dmass*(1.0f-bmst);
}

extern "C" __global__ void sfire_moisture_weights(
    const float *loads, float *weights, int n)
{
    int k=blockDim.x*blockIdx.x+threadIdx.x;
    if (k>=n) return;
    float total=0.0f;
    for (int c=0;c<5;c++) total=total+loads[k*5+c];
    for (int c=0;c<5;c++) weights[k*5+c]=total>0.0f?__fdiv_rn(loads[k*5+c],total):0.0f;
}

extern "C" __global__ void sfire_weighted_moisture(
    const float *classes, const float *cat, const float *weights,
    float *fmc, int n, int nc, int fallback)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx>=n) return;
    int k=sfire_category((int)cat[idx],fallback)-1;
    float value=0.0f;
    for (int c=0;c<nc;c++) value=value+weights[k*5+c]*classes[c*n+idx];
    fmc[idx]=value;
}
