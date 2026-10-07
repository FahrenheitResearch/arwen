// WRF v4.7.1 chem/module_gocart_dust.F and module_gocart_dust_afwa.F.
// Every arithmetic operation retains the source precision and order.
__device__ double dust_threshold(double density, double radius, float gravity,
                                  double airden) {
    double den=DMUL(density,1.e-3);
    double diam=DMUL(DMUL(2.f,radius),1.e2);
    float g=FMUL(gravity,100.f);
    float rhoa=__double2float_rn(DMUL(airden,1.e-3));
    double first=DMUL(DMUL((double)0.13f,1.e-2),sqrt(DDIV(DMUL(DMUL(den,g),diam),rhoa)));
    double second=sqrt(DADD(1.f,DDIV(DDIV(DDIV((double)0.006f,den),g),pow(diam,(double)2.5f))));
    double third=sqrt(DSUB(DMUL((double)1.928f,pow(DADD(DMUL((double)1331.f,pow(diam,(double)1.56f)),(double)0.38f),(double)0.092f)),1.f));
    return DDIV(DMUL(first,second),third);
}
// phys/module_data_gocart_dust.F:9-12, process soil porosity, REAL.
__device__ const float dust_porosity[19]={.339f,.421f,.434f,.476f,.476f,.439f,.404f,.464f,.465f,.406f,.468f,.468f,.439f,1.f,.2f,.421f,.468f,.2f,.339f};
extern "C" __global__ void chem_dust_gocart(
    const unsigned long long* species,const unsigned long long* edust,int rows,int nc,
    const double* density,const double* radius,const float* fraction,const int* ipoint,
    const float* ch,const float* xland,const float* u10,const float* v10,
    const float* smois,const int* soil,const float* erod,const float* rho,
    const float* dz,const float* u,const float* v,float dt,float gravity) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc || !(xland[c]<1.5f)) return;
    double wind=sqrtf(FADD(FMUL(u10[c],u10[c]),FMUL(v10[c],v10[c])));
    if(dz[c]<12.f) wind=sqrtf(FADD(FMUL(u[c],u[c]),FMUL(v[c],v[c])));
    double wet=FDIV(smois[c],dust_porosity[soil[c]-1]);
    for(int n=0;n<rows;n++) {
        double threshold=dust_threshold(density[n],radius[n],gravity,rho[c]);
        threshold=wet<.5 ? fmax(0.,DMUL(threshold,DADD(1.2,DMUL(.2,log10(fmax(1.e-3,wet)))))):100.;
        double source=DMUL(fraction[n],(double)erod[(ipoint[n]-1)*nc+c]);
        double mass=DMUL(DMUL(DMUL(DMUL(ch[n],source),DMUL(wind,wind)),DSUB(wind,threshold)),dt);
        if(mass<0.) mass=0.;
        float* q=(float*)species[n]; float* e=(float*)edust[n];
        double tc=DMUL(q[c],(double)1.e-9f);
        tc=DADD(tc,DDIV(DDIV(mass,dz[c]),rho[c]));
        q[c]=__double2float_rn(DMUL(tc,(double)1.e9f));
        e[c]=__double2float_rn(DADD(e[c],mass));
    }
}
// phys/module_data_gocart_dust.F:23-26, internal saltation bins.
__device__ const double salt_radius[9]={.71e-6,1.37e-6,2.63e-6,5.e-6,9.5e-6,18.1e-6,34.5e-6,65.5e-6,125.e-6};
__device__ const double salt_density[9]={2500.,2650.,2650.,2650.,2650.,2650.,2650.,2650.,2650.};
__device__ const int salt_class[9]={0,1,1,1,1,1,2,2,2};
__device__ const double salt_fraction[9]={1.,(double).2f,(double).2f,(double).2f,(double).2f,(double).2f,(double).333f,(double).333f,(double).333f};
extern "C" __global__ void chem_dust_afwa(
    const unsigned long long* species,const unsigned long long* edust,int rows,int nc,int nz,
    const double* density,const double* radius,const double* distribution,const float* extinction,
    const float* xland,const float* u10,const float* v10,const float* smois,
    const int* soil,const float* erod,const float* erod_dri,const float* rho,const float* dz,
    const float* snow,const float* veg,const float* lai,const float* ust,const float* znt,
    const float* clay_wrf,const float* sand_wrf,const float* clay_nga,const float* sand_nga,
    float* loft,float* total,float* total_e,float* visibility,
    int dsr,int vegetation,int soils,int moisture,int surface,
    float alpha,float gamma,float smtune,float ustune,float dt,float gravity) {
    int c=blockDim.x*blockIdx.x+threadIdx.x; if(c>=nc) return;
    // density and radius are row data. The pinned active AFWA source only
    // reads distribution; its radius/density distribution calculation is commented out.
    loft[c]=-99.f;
    if(xland[c]<1.5f) {
        double ustar=DMUL((double)ust[c],ustune);
        const float* er=(dsr==1 && erod_dri[c]>=0.f)?erod_dri:erod;
        double erodtot=FADD(FADD(FADD(0.f,er[c]),er[nc+c]),er[2*nc+c]);
        double mask=vegetation==1 ? (veg[c]>=5.f?0.:1.) : vegetation==2 ? (double)lai[c] : (erod[c]==0.f?0.:1.);
        erodtot=DMUL(erodtot,mask);
        bool nga=soils==1 && clay_nga[c]>=0.f;
        float clay=nga?clay_nga[c]:clay_wrf[c],sand=nga?sand_nga[c]:sand_wrf[c];
        double massfrac[3]={clay,FSUB(1.f,FADD(clay,sand)),sand};
        bool land=!(znt[c]>.2f || soil[c]==15 || soil[c]==16 || soil[c]==18 || snow[c]>.01f);
        double volsm=fmaxf(FMUL(smois[c],smtune),0.f);
        float denom=FMUL(FSUB(1.f,dust_porosity[soil[c]-1]),FADD(FMUL(2.65f,FSUB(1.f,clay)),FMUL(2.5f,clay)));
        double gravsm=DDIV(DMUL(100.,volsm),denom),drylimit;
        bool volumetric=moisture==1 && (surface==2 || surface==3 || surface==7);
        if(moisture==1 && (surface==3 || surface==7))
            drylimit=FMUL(.035f,gfk_pow(FADD(FMUL(13.52f,clay),3.53f),2.68f));
        else if(moisture==1 && surface==2)
            drylimit=FMUL(.0756f,gfk_pow(FADD(FMUL(15.127f,clay),3.09f),2.3211f));
        else drylimit=FADD(FMUL(FMUL(14.f,clay),clay),FMUL(17.f,clay));
        double ds[9],stotal=0.;
        for(int n=0;n<9;n++) {
            double dmass=DMUL(massfrac[salt_class[n]],salt_fraction[n]);
            ds[n]=DDIV(DMUL(.75f,dmass),DMUL(salt_density[n],salt_radius[n]));
            stotal=DADD(stotal,ds[n]);
        }
        double emit=0.; float ustart=0.f;
        for(int n=0;n<9;n++) {
            double threshold=dust_threshold(salt_density[n],salt_radius[n],gravity,rho[c]);
            double water=volumetric?DMUL(100.f,volsm):gravsm;
            if(water>drylimit) threshold=fmax(0.,DMUL(threshold,sqrt(DADD(1.f,DMUL((double)1.21f,pow(DSUB(water,drylimit),(double).68f))))));
            if(n==6) ustart=__double2float_rn(threshold);
            double salt=0.;
            if(ustar>threshold && erodtot>0. && land) {
                salt=DMUL(1.f,DDIV(ds[n],stotal));
                salt=DMUL(salt,DDIV((double)rho[c],gravity));
                salt=DMUL(salt,DMUL(ustar,DMUL(ustar,ustar)));
                salt=DMUL(salt,DADD(1.f,DDIV(threshold,ustar)));
                salt=DMUL(salt,DSUB(1.f,DDIV(DMUL(threshold,threshold),DMUL(ustar,ustar))));
            }
            double beta=pow(10.,DSUB(DMUL((double)13.6f,massfrac[0]),6.f));
            if(beta>(double)5.25e-4f) beta=(double)5.25e-4f;
            emit=DADD(emit,DMUL(DMUL(DMUL(salt,pow(erodtot,(double)gamma)),alpha),beta));
        }
        double sum_mass=0.;
        for(int n=0;n<rows;n++) {
            double mass=DMUL(DMUL(emit,distribution[n]),dt); if(mass<0.) mass=0.;
            float* q=(float*)species[n]; float* e=(float*)edust[n];
            double tc=FMUL(q[c],1.e-9f);
            q[c]=__double2float_rn(DMUL(DADD(tc,DDIV(DDIV(mass,dz[c]),rho[c])),1.e9f));
            e[c]=__double2float_rn(DADD(e[c],mass)); sum_mass=DADD(sum_mass,mass);
        }
        total_e[c]=__double2float_rn(DADD(total_e[c],sum_mass));
        float w10=gfk_pow(FADD(gfk_pow(u10[c],2.f),gfk_pow(v10[c],2.f)),.5f);
        float psi=0.f;
        if(ustar!=0. && znt[c]!=0.f)
            psi=__double2float_rn(DSUB(DDIV((double)FMUL(.4f,w10),ustar),gfk_log(FDIV(10.f,znt[c]))));
        if(erodtot>0.) loft[c]=FSUB(FMUL(ustune,w10),FDIV(FMUL(ustart,FADD(gfk_log(FDIV(10.f,znt[c])),psi)),.4f));
    }
    // module_gocart_dust_afwa.F:351-366, all mass levels including kte.
    for(int k=0;k<nz;k++) {
        int p=k*nc+c; float sum=0.f,ext=0.f;
        for(int n=0;n<rows;n++) {
            float q=((float*)species[n])[p];
            sum=n==0?q:FADD(sum,q);
            float term=FMUL(extinction[n],q); ext=n==0?term:FADD(ext,term);
        }
        total[p]=FMUL(sum,rho[p]);
        visibility[p]=total[p]>0.f?fminf(FDIV(3.912f,FMUL(ext,rho[p])),999999.f):999999.f;
    }
}
