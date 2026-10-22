// WRF v4.6.1 phys/module_sf_oml.F, public domain, OML1D and omlinit.
// WOOF's existing FSQRT supplies the float32 square root. No FTZ, no FMA.
extern "C" __global__ void oml_init(
    float *tml, float *t0ml, float *hml, float *h0ml, float *huml,
    float *hvml, float *tmoml, const float *tsk, float hml0, int start, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n || !start) return;
    tml[i]=tsk[i]; t0ml[i]=tsk[i];
    if(hml0>0.f) {
        hml[i]=hml0; h0ml[i]=hml[i]; huml[i]=0.f; hvml[i]=0.f;
        tmoml[i]=tsk[i]-5.f;
    } else if(hml0==0.f) {
        hml[i]=h0ml[i]; huml[i]=0.f; hvml[i]=0.f; tmoml[i]=tsk[i]-5.f;
    } else {
        hml[i]=h0ml[i];
        if(tmoml[i]>200.f && tmoml[i]<=201.f) tmoml[i]=tsk[i];
    }
}
extern "C" __global__ void oml_step(
    float *tmlp, const float *t0mlp, float *hp, const float *h0p,
    float *humlp, float *hvmlp, const float *tmomlp, float *tskp,
    const float *hfx, const float *lh, const float *gsw, const float *glw,
    const float *uair, const float *vair, const float *ust, const float *f,
    const float *emiss, const float *xland,
    const float *ice_heat, const float *ice_mask, int coupled_ice,
    float dt, float gam, float relaxation, float rhowater, float cwater,
    float stbolt, float gravity, int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n || !(xland[i]>1.5f || (coupled_ice && ice_mask[i]>0.f))) return;
    float tml=tmlp[i], t0ml=t0mlp[i], h=hp[i], h0=h0p[i];
    float hu1=humlp[i], hv1=hvmlp[i], tmoml=tmomlp[i];
    float alp=fmaxf((tml-273.15f)*1.e-5f,1.e-6f);
    float bv2=alp*gravity*gam;
    float thp=t0ml-gam*(h-h0);
    float a1=(tml-thp)*h-0.5f*gam*h*h;
    float q=(-hfx[i]-lh[i]+gsw[i]+glw[i]*emiss[i]-stbolt*emiss[i]*tml*tml*tml*tml)/(rhowater*cwater);
    if(coupled_ice && ice_mask[i]>0.f) q=ice_heat[i]/(rhowater*cwater);
    float wspd=FSQRT(uair[i]*uair[i]+vair[i]*vair[i]);
    if(wspd<1.e-10f) wspd=1.e-10f;
    float ust2=0.5f*ust[i];
    float tauxair=ust2*ust2*uair[i]/wspd;
    float taux=1.f/rhowater*tauxair;
    float tauyair=ust2*ust2*vair[i]/wspd;
    float tauy=1.f/rhowater*tauyair;
    float hu2=hu1+dt*(f[i]*hv1+taux);
    float hv2=hv1+dt*(-f[i]*hu2+tauy);
    float a2=a1+q*dt;
    float huml=hu2, hvml=hv2, hold=h;
    float b2=hu2*hu2+hv2*hv2;
    float hsqrd=-a2/gam+FSQRT(a2*a2/(gam*gam)+2.f*b2/bv2);
    h=FSQRT(fmaxf(hsqrd,0.f));
    if(h<hold) h=hold;
    if(tml>=tmoml && h!=0.f) {
        tml=fmaxf(t0ml-gam*(h-h0)+0.5f*gam*h+a2/h,tmoml);
    } else { tml=t0ml; }
    if(coupled_ice && ice_mask[i]>0.f && h>0.f) {
        // Polar liquid enthalpy: entrained seawater cannot be colder than
        // freezing. Retain surface cooling below Tf for the ice owner to
        // convert into latent heat. The warm-water OML's deep-temperature
        // floor must not discard that cooling or entrain supercooled water.
        float deep=fmaxf(271.35f,t0ml-gam*(h-h0));
        tml=(tmlp[i]*hold+deep*(h-hold)+q*dt)/h;
    }
    if(relaxation>0.f) {
        tml=tml-(tml-t0ml)*dt/relaxation;
        h=h-(h-h0)*dt/relaxation;
        huml=huml-huml*dt/relaxation;
        hvml=hvml-hvml*dt/relaxation;
    }
    tmlp[i]=tml; hp[i]=h; humlp[i]=huml; hvmlp[i]=hvml; tskp[i]=tml;
}
