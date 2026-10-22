// Original Apache-2.0 implementation. No third-party source is transcribed.
// Zero-layer ice/snow conduction and surface Stefan balance, SI throughout.
// Semtner (1976), doi:10.1175/1520-0485(1976)006<0379:AMFTTG>2.0.CO;2.
// Areal ice volume is conserved during area remapping. No dynamics/ridging.
static constexpr double rhoi=917., rhos=330., lf=333700., lv=2500000.;
static constexpr double tf=271.35, tm=273.15, sigma=5.670374419e-8;
__device__ double qsat(double t,double p,bool ice) {
    double e=611.2*exp((ice?22.46:17.67)*(t-273.15)/(t-(ice?.55:29.65)));
    return .622*e/(p-e);
}
__device__ double net(double t,double ta,double qa,double p,double ch,
                      double sw,double lw,double alb,double maxevap,double *h,double *e) {
    double rho=p/(287.05*ta*(1.+.61*qa));
    *h=rho*1004.5*ch*(t-ta);
    *e=fmin(maxevap,rho*ch*(qsat(t,p,true)-qa));
    return (1.-alb)*sw+.98*(lw-sigma*t*t*t*t)-*h-(lv+lf)*(*e);
}
extern "C" __global__ void ice_initialize(
    const float *a,const float *land,const float *lake,const float *snow,
    const float *tsk,float *sst,float *hi,float *hs,float *v,float *s,float *skin,float *mask,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
    mask[i]=(lake[i]<.5f && (land[i]>1.5f || a[i]>0.f))?1.f:0.f;
    double frac=mask[i]?fmin(1.,fmax(0.,(double)a[i])):0.;
    // Analysis SST may be masked to zero over sea ice classified as land.
    // Seed the liquid owner at freezing, never from ice-skin temperature.
    if(mask[i]) sst[i]=fmax((double)sst[i],tf);
    v[i]=frac*hi[i]; s[i]=frac*fmax(0.,(double)snow[i]);
    hs[i]=frac>0?s[i]/(frac*rhos):0.;
    skin[i]=fmin(tm,(double)tsk[i]);
    if(frac==0) hi[i]=0;
}
extern "C" __global__ void ice_precipitation(
    const float *rain,const float *pending,const float *ta,const float *ratio,float *out,float *snow,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
    out[i]=fmaxf(0.f,rain[i]+pending[i]);
    snow[i]=fminf(1.f,fmaxf(0.f,ratio[i]))*out[i];
}
extern "C" __global__ void ice_step(
    float *ap,float *vp,float *sp,float *tp,const float *mask,
    const float *rain,const float *snowfall,float *growth,float *melt,float *energy,
    float *error,float *ocean,float *ocean_heat,float *dhi,float *da,float *hiout,float *hsout,float *tsk,
    const float *sst,const float *ta,const float *qa,const float *p,const float *ch,
    const float *sw,const float *lw,float *hfx,float *lh,float *qfx,
    float *qsfc,float *albedo,float *emiss,float *snow,float *snowh,float *znt,
    float dt0,float qo0,float hnew0,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n || mask[i]==0)return;
    double a=ap[i], v=vp[i], s=sp[i], dt=dt0, qo=qo0;
    double oldv=v, olds=s, olda=a, oldenergy=-lf*(rhoi*v+s);
    // A warm ice-free ocean remains the ocean/LSM owner's column. New ice
    // forms only when that owner supplies water at/below the freezing point.
    double water_t=fmax(tf,(double)sst[i]);
    double rho=p[i]/(287.05*ta[i]*(1.+.61*qa[i]));
    double hw=rho*1004.5*fmax(0.,(double)ch[i])*(water_t-ta[i]);
    double ew=rho*fmax(0.,(double)ch[i])*(qsat(water_t,p[i],false)-qa[i]);
    double qw=.94*sw[i]+.98*(lw[i]-sigma*water_t*water_t*water_t*water_t)-hw-lv*ew;
    ocean_heat[i]=(1.-a)*qw-a*qo;
    if(a<=0. && sst[i]>tf) return;
    double snowfall_mass=a*snowfall[i]; s+=snowfall_mass;
    double h=0,e=0,ts=tm,q=0,c=0;
    double alb=s>1.e-8?.80:.65;
    double atm=0, sublim=0, runoff=0, basal=0, surf=0;
    if(a>0. && v>0.) {
        double hi=v/a, hs=s/(a*rhos);
        double conductance=1./fmax(hi/2.03+hs/.31,1.e-8);
        double maxevap=(s+rhoi*v)/(a*dt);
        // Monotone nonlinear surface balance; fixed iterations, no host sync.
        double lo=150.,up=tm;
        for(int k=0;k<36;k++) {
            ts=.5*(lo+up);
            q=net(ts,ta[i],qa[i],p[i],fmax(0.,(double)ch[i]),sw[i],lw[i],alb,maxevap,&h,&e);
            if(q+conductance*(tf-ts)>0)lo=ts; else up=ts;
        }
        ts=.5*(lo+up);
        q=net(ts,ta[i],qa[i],p[i],fmax(0.,(double)ch[i]),sw[i],lw[i],alb,maxevap,&h,&e);
        c=conductance*(tf-ts);
        double surface=fmax(0.,q+c)*a*dt;
        // Sublimation/deposition acts on snow first, then on ice.
        sublim=e*a*dt;
        if(sublim>0) {
            sublim=fmin(sublim,s+rhoi*v);
            double ds=fmin(s,sublim); s-=ds; v-=(sublim-ds)/rhoi;
        } else s-=sublim;
        double ds=fmin(s,surface/lf); s-=ds; surface-=ds*lf;
        runoff+=ds;
        double dv=fmin(v,surface/(rhoi*lf)); v-=dv; surf+=dv;
        runoff+=dv*rhoi;
        // Bottom heat flux positive from ocean to ice. Conduction is upward.
        basal=a*dt*(c-qo)/(rhoi*lf);
        double basal_excess=fmax(0.,-basal-v)*rhoi*lf;
        if(basal>=0) {v+=basal; growth[i]+=basal;}
        else { double dm=fmin(v,-basal);v-=dm;melt[i]+=dm; }
        melt[i]+=surf;
        // Any energy remaining after complete melt returns to the ocean.
        double excess=fmax(0.,surface-dv*rhoi*lf)
                     +basal_excess;
        ocean[i]+=excess;
        ocean_heat[i]+=excess/dt;
        // Atmospheric energy uses vapour referenced to liquid water. The
        // extra fusion heat in Lsub is balanced by solid mass sublimation.
        atm=a*dt*(q+lf*e+qo);
        if(excess>0)atm-=excess;
        // A snow load over fully melted ice enters the ocean as solid and
        // consumes its fusion enthalpy there, never disappearing from ledger.
        if(v<=0. && s>0.) {ocean[i]-=s*lf;ocean_heat[i]-=s*lf/dt;atm+=s*lf;s=0.;}
    }
    // Open-water freezing at the supplied ocean freezing temperature.
    double ow=1.-olda;
    if(ow>0. && sst[i]<=tf) {
        double freeze=fmax(0.,-qw-qo)*ow*dt/(rhoi*lf);
        v+=freeze; growth[i]+=freeze;
        ocean_heat[i]+=freeze*rhoi*lf/dt;
        a=fmin(1.,a+freeze/hnew0);
        atm-=freeze*rhoi*lf;
    }
    // Thermodynamic area loss: dA/A = 0.5 dV/V. Growth thickens old ice;
    // new ice occupies the open-water fraction at hnew0.
    if(olda==0. && v>0)ts=tf;
    if(v<oldv && oldv>0) a=olda*sqrt(fmax(0.,v/oldv));
    if(v<=0) {v=0;a=0;}
    if(v>0 && a<=0) a=fmin(1.,v/hnew0);
    dhi[i]+=(a>0?v/a:0)-(olda>0?oldv/olda:0);
    da[i]+=a-olda;
    double newenergy=-lf*(rhoi*(double)(float)v+(double)(float)s);
    double expected=-lf*snowfall_mass+atm;
    energy[i]+=expected;
    error[i]+=(newenergy-oldenergy)-expected;
    ap[i]=a;vp[i]=v;sp[i]=s;tp[i]=ts;
    hiout[i]=a>0?v/a:0;hsout[i]=a>0?s/(rhos*a):0;
    // Fractional tile fluxes feed the existing PBL and surface ledger.
    // Keep the LSM's open-water component, compute the ice component anew.
    if(olda>0 || a>0) {
        hfx[i]=olda*h+(1.-olda)*hw;
        qfx[i]=olda*e+(1.-olda)*ew;
        lh[i]=lv*qfx[i];
        qsfc[i]=a*qsat(ts,p[i],true)+(1.-a)*qsat(water_t,p[i],false);
        tsk[i]=a*ts+(1.-a)*water_t;
        albedo[i]=a*alb+(1.-a)*.06;emiss[i]=.98;
        snow[i]=a>0?s/a:0;snowh[i]=a>0?s/(rhos*a):0;
        znt[i]=a*.001+(1.-a)*.0001;
    }
}

extern "C" __global__ void ice_blend_ocean(const float *mask,const float *a,
    const float *skin,const float *sst,float *tsk,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n && mask[i]>0 && a[i]>0)
        tsk[i]=a[i]*skin[i]+(1.f-a[i])*fmaxf(271.35f,sst[i]);
}

// Mixed-layer supercooling freezes into the column, conserving the heat of
// water plus solid fusion enthalpy. The water owner never remains below Tf.
extern "C" __global__ void ice_ocean_freeze(
    const float *mask,float *a,float *v,float *skin,float *hi,const float *s,
    float *hs,float *growth,float *energy,float *error,float *ocean,
    float *dhi,float *da,float *snow,float *snowh,float *albedo,float *znt,
    float *sst,float *tml,const float *hml,
    float hnew,float water_capacity,int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n || mask[i]==0 || tml[i]>=271.35f)return;
    double oldv=v[i], olda=a[i], oldhi=hi[i];
    double heat=(271.35f-(double)tml[i])*water_capacity*hml[i];
    double vol=heat/(rhoi*lf), nv=oldv+vol;
    double na=fmin(1.,olda+vol/hnew);
    v[i]=nv;a[i]=na;hi[i]=nv/na;hs[i]=s[i]/(rhos*na);
    snow[i]=s[i]/na;snowh[i]=hs[i];
    albedo[i]=na*(s[i]>1.e-8?.80:.65)+(1.-na)*.06;
    znt[i]=na*.001+(1.-na)*.0001;
    if(olda==0)skin[i]=tf;
    growth[i]+=vol;energy[i]-=heat;ocean[i]+=heat;
    error[i]+=-lf*rhoi*((double)v[i]-oldv)+heat;
    dhi[i]+=(double)hi[i]-oldhi;da[i]+=(double)a[i]-olda;
    sst[i]=271.35f;tml[i]=271.35f;
}
