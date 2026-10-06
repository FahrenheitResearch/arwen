
// Full-width RUC driver.  Scratch arrays keep soil columns out of local frames.
#ifndef NAN
#define NAN __int_as_float(0x7fc00000)
#endif
__device__ __forceinline__ float d_min(float a,float b) {
    return (isnan(a) || isnan(b)) ? NAN : fminf(a,b);
}
__device__ __forceinline__ float d_max(float a,float b) {
    return (isnan(a) || isnan(b)) ? NAN : fmaxf(a,b);
}
__device__ __forceinline__ float d_npmin(float a,float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a<b ? a:b));
}
__device__ __forceinline__ float d_npmax(float a,float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a>b ? a:b));
}
__device__ __forceinline__ void d_flag(unsigned* flags,int bit) {
    atomicOr(flags+bit/32,1u<<(bit%32));
}
#define A(a,b) __fadd_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define B(a,b) __fsub_rn((a),(b))
#define Q(a,b) __fdiv_rn((a),(b))
__device__ __forceinline__ float d_qsn(float temperature,const float* tbq,
                                     unsigned* flags,int first,bool* admitted=nullptr) {
    if(!isfinite(temperature)) {
        d_flag(flags,first);
        if(admitted) *admitted=false;
    }
    float raw=A(Q(B(temperature,173.15f),0.05f),1.0f);
    if(!(fabsf(raw)<2147483648.0f)) {
        d_flag(flags,first+1);
        if(admitted) *admitted=false;
        return tbq[0];
    }
    return ruc_qsn_lookup(temperature,tbq);
}

extern "C" __global__ void ruc_driver_prologue(
    const unsigned long long* ip, const unsigned long long* sp,
    int* integer, bool* run, unsigned* flags,
    const int* ifortbl, const float* z0tbl, const float* lemitbl,
    const float* pctbl, const float* laitbl, const float* bb,
    const float* drysmc, const float* hc, const float* maxsmc,
    const float* refsmc, const float* satpsi, const float* satdk,
    const float* wltsmc, const float* qtz, const float* tbq,
    int n,int ktau,float dt,int iswater,int isice,int nv,int ns,
    float icealbedo,float cn,
    const float* landusef,const float* soilctop,int nlcat,int nscat,
    int mosaic_lu,int mosaic_soil,int lakemodel,
    int qvg_air,int rdlai2d,float xice_threshold) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    D_DECLARE_SCRATCH
    D_COPY_INPUT
    // xice_threshold: module_surface_driver.F:1365-1368, 0.5 or 0.02 by
    // fractional_seaice; the host hands the run's value to every ice test.
    bool component=D_F(xice)>=xice_threshold && D_F(xice)<=1.0f;
    if(component) {
        D_F(albbck)=icealbedo;
        D_F(alb)=Q(B(D_F(alb),M(B(1.0f,D_F(xice)),0.08f)),D_F(xice));
        D_F(emiss)=Q(B(D_F(emiss),M(B(1.0f,D_F(xice)),0.98f)),D_F(xice));
        D_F(soilt)=D_F(tsk_save);
    }
    bool admitted=true;
    D_ADMIT
    int veg=((const int*)ip[D_CAT_INPUT])[i];
    int soil=((const int*)ip[D_CAT_INPUT+1])[i];
    if(veg<1 || veg>nv) { d_flag(flags,D_CAT_FLAG); admitted=false; veg=1; }
    if(soil<1 || soil>ns) { d_flag(flags,D_CAT_FLAG+1); admitted=false; soil=1; }
    integer[i]=veg; integer[n+i]=soil;
    if(ktau==1) {
        for(int k=0;k<RUC_NZS;++k) D_P(keepfr3dflag,k)=0.0f;
        float blended=M(0.5f,A(D_F(soilt),D_P(tso,0)));
        // Snow by lineage (gpuwm.core.ruc_tier RUC_SNOW_FORMS).  wrf_45, the
        // operational RAP/HRRR branch's LSMRUC:459-465: snow water with no
        // cover starts at min(1, snow/32), written as the exact snow*2**-5,
        // and the inside-snow repair blends above 32 mm of snow water.
        // GPUWM_SNOW_WRF461: WRF v4.6.1 blends wherever snowc > 0.
#ifdef GPUWM_SNOW_WRF461
        bool inside_snow=D_F(snowc)>0.0f;
#else
        if(D_F(snow)>0.0f && D_F(snowc)<=0.0f) D_F(snowc)=d_min(1.0f,M(D_F(snow),0.03125f));
        bool inside_snow=D_F(snow)>32.0f;
#endif
        if(D_F(soilt1)<170.0f || D_F(soilt1)>400.0f)
            D_F(soilt1)=inside_snow ? blended:D_P(tso,0);
        D_F(tsnav)=B(blended,273.15f);
        D_F(qsg)=Q(d_qsn(D_F(soilt),tbq,flags,80,&admitted),M(D_F(p8w),1.0e-2f));
        // QVG/QCG cold start by lineage (gpuwm.core.ruc_tier
        // RUC_QVG_COLD_START_FORMS).  qvg_air: the operational RAP/HRRR branch's
        // LSMRUC:479-483, the lowest-level vapour with no ground condensate.
        // Otherwise public WRF v4.6.1 LSMRUC:505-514, saturation at the
        // skin times moisture availability, condensate from the air.
        if(qvg_air) {
            if(D_F(qvg)<=0.0f || D_F(qvg)>0.1f) { D_F(qvg)=D_F(qv3d); D_F(qcg)=0.0f; }
        } else {
            if(D_F(qcg)<0.0f || D_F(qcg)>0.1f) D_F(qcg)=D_F(qc3d);
            if(D_F(qvg)<=0.0f || D_F(qvg)>0.1f) D_F(qvg)=M(D_F(qsg),D_F(mavail));
        }
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg)));
        D_F(snom)=D_F(snowfallac)=D_F(precipfr)=D_F(dew)=0.0f;
        D_F(sfcrunoff)=D_F(udrunoff)=D_F(acrunoff)=0.0f;
        D_F(rhosnf)=-1000.0f; D_F(chklowq)=1.0f;
    }
    D_F(patm)=M(D_F(p8w),1.0e-5f);
    D_F(conflx)=M(D_F(z3d),0.5f);
    float resolved_liquid=M(D_F(rainncv),B(1.0f,D_F(frzfrac)));
    float resolved_frozen=M(D_F(rainncv),D_F(frzfrac));
    float convective=B(D_F(rainbl),D_F(rainncv));
    bool cold=D_F(t3d)<273.0f;
    bool mixed_cold=D_F(frzfrac)>0.0f && cold;
    float convective_liquid=mixed_cold ? d_max(0.0f,M(convective,B(1.0f,D_F(frzfrac)))) : (cold ? 0.0f:d_max(0.0f,convective));
    float convective_frozen=mixed_cold ? d_max(0.0f,M(convective,D_F(frzfrac))) : (cold ? d_max(0.0f,convective):0.0f);
    D_F(prcpms)=M(Q(A(resolved_liquid,convective_liquid),dt),1.0e-3f);
    D_F(newsnms)=M(Q(A(resolved_frozen,convective_frozen),dt),1.0e-3f);
    float frozen_total=A(resolved_frozen,convective_frozen);
    bool falling=frozen_total>0.0f;
    float denominator=falling ? frozen_total:1.0f;
    D_F(snowrat)=falling ? d_min(1.0f,d_max(0.0f,Q(D_F(snowncv),denominator))):0.0f;
    D_F(grauprat)=falling ? d_min(1.0f,d_max(0.0f,Q(D_F(graupelncv),denominator))):0.0f;
    D_F(icerat)=falling ? d_min(1.0f,d_max(0.0f,Q(B(B(resolved_frozen,D_F(snowncv)),D_F(graupelncv)),denominator))):0.0f;
    D_F(curat)=falling ? d_min(1.0f,d_max(0.0f,Q(convective_frozen,denominator))):0.0f;
    D_F(precipfr)=M(M(D_F(newsnms),dt),1000.0f);
    D_F(qkms)=Q(Q(D_F(flqc),D_F(rho3d)),D_F(mavail));
    D_F(tkms)=Q(Q(D_F(flhc),D_F(rho3d)),M(1004.5f,A(1.0f,M(0.84f,D_F(qv3d)))));
    D_F(snwe)=M(D_F(snow),1.0e-3f); D_F(snhei)=D_F(snowh);
    D_F(canwatr)=M(D_F(canwat),1.0e-3f); D_F(snowfrac)=D_F(snowc);
    D_F(rhosnfall)=D_F(rhosnf);
    D_F(rhosn)=D_F(snow)>0.0f && D_F(snowh)>0.0f ? Q(D_F(snow),D_F(snowh)):300.0f;
    // The driver version of SOILVEGIN.  The old leaf's fmin/fmax omit NaN
    // propagation.  These operands have been admitted, but derived NaNs must
    // still match the driver's CuPy minimum and maximum.
    int forest=ifortbl[veg-1];
    float range=B(D_F(shdmax),D_F(shdmin));
    float ratio=Q(B(D_F(vegfra),D_F(shdmin)),d_max(1.0f,range));
    float bounded=d_max(0.0f,d_min(1.0f,ratio));
    float factor=range<1.0f ? 1.0f:B(1.0f,bounded);
    float scaled=M(0.8f,laitbl[veg-1]), delta=0.0f;
    if(forest==1) delta=d_min(0.2f,scaled);
    if(forest==2 || forest==7) delta=d_min(0.5f,scaled);
    if(forest==3) delta=d_min(0.45f,scaled);
    if(forest==4) delta=d_min(0.75f,scaled);
    if(forest==5) delta=d_min(0.86f,scaled);
    float incoming_znt=D_F(znt);
    // module_sf_ruclsm.F:7075 ``if(.not.rdlai2d) LAI = LAItoday(IVGTYP)``:
    // under rdlai2d the prescribed 2-D LAI (the monthly field) stays.
    if(!rdlai2d) D_F(lai)=veg==iswater ? laitbl[veg-1]:B(laitbl[veg-1],M(delta,factor));
    if(veg!=iswater) D_F(znt)=forest==7 ? B(z0tbl[veg-1],M(0.125f,factor)):z0tbl[veg-1];
    D_F(emissl)=lemitbl[veg-1]; D_F(pc)=pctbl[veg-1];
    D_F(qwrtz)=D_F(rhocs)=D_F(bclh)=D_F(dqm)=D_F(ksat)=D_F(psis)=D_F(qmin)=D_F(ref)=D_F(wilt)=0.0f;
    if(soil!=14) {
        D_F(qwrtz)=qtz[soil-1]; D_F(rhocs)=M(hc[soil-1],1.0e6f);
        D_F(bclh)=bb[soil-1]; D_F(dqm)=B(maxsmc[soil-1],drysmc[soil-1]);
        D_F(ksat)=satdk[soil-1]; D_F(psis)=-satpsi[soil-1];
        D_F(qmin)=drysmc[soil-1]; D_F(ref)=refsmc[soil-1]; D_F(wilt)=wltsmc[soil-1];
    }
    ruc_mosaic_parameters(i,n,nlcat,nscat,mosaic_lu,mosaic_soil,
        landusef,soilctop,soil,iswater,rdlai2d!=0,factor,incoming_znt,
        ifortbl,z0tbl,lemitbl,pctbl,laitbl,bb,drysmc,hc,maxsmc,
        refsmc,satpsi,satdk,wltsmc,qtz,
        D_F(emissl),D_F(pc),D_F(znt),D_F(lai),D_F(qwrtz),
        D_F(rhocs),D_F(bclh),D_F(dqm),D_F(ksat),D_F(psis),
        D_F(qmin),D_F(ref),D_F(wilt));
    bool forested=forest>2;
    D_F(meltfactor)=forested ? 2.0f:0.85f;
    integer[3*n+i]=4;
    for(int k=1;k<RUC_NZS;++k) if(ruc_soil_layer_depth[k]>=(forested ? 0.4f:1.1f)) {
        integer[3*n+i]=k+1; break;
    }
    bool lake=lakemodel==1 && D_F(lakemask)==1.0f;
    bool water=B(D_F(xland),1.5f)>=0.0f && !lake;
    bool land=!(B(D_F(xland),1.5f)>=0.0f || lake);
    bool ice=land && D_F(xice)>=xice_threshold;
    D_F(seaice)=D_F(xice)>=xice_threshold ? 1.0f:0.0f;
    integer[2*n+i]=ice ? isice:veg; integer[4*n+i]=1;
    if(water) {
        D_F(smavail)=D_F(smmax)=1.0f; D_F(snow)=D_F(snowh)=D_F(snowc)=0.0f;
        D_F(qvg)=Q(d_qsn(D_F(soilt),tbq,flags,82,&admitted),M(D_F(p8w),1.0e-2f));
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg))); D_F(chklowq)=1.0f;
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilmois,k)=D_P(sh2o,k)=1.0f; D_P(tso,k)=D_F(soilt);
        }
    }
    if(ice) {
        D_F(znt)=0.011f; D_F(snoalb)=0.75f; D_F(dqm)=D_F(ref)=1.0f;
        D_F(qmin)=D_F(wilt)=0.0f; D_F(emissl)=0.98f;
        D_F(qvg)=D_F(qsg)=Q(d_qsn(D_F(soilt),tbq,flags,84,&admitted),M(D_F(p8w),1.0e-2f));
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg)));
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilmois,k)=D_P(smfr3d,k)=1.0f;
            D_P(sh2o,k)=D_P(keepfr3dflag,k)=0.0f;
            D_P(tso,k)=d_min(271.4f,D_P(tso,k));
        }
    }
    for(int k=0;k<RUC_NZS;++k) {
        D_P(soilm1d,k)=d_min(d_max(0.0f,B(D_P(soilmois,k),D_F(qmin))),D_F(dqm));
        D_P(tso1d,k)=D_P(tso,k);
        D_P(soiliqw,k)=d_min(d_max(0.0f,B(D_P(sh2o,k),D_F(qmin))),D_P(soilm1d,k));
        D_P(soilice,k)=Q(B(D_P(soilm1d,k),D_P(soiliqw,k)),0.9f);
        D_P(smfrkeep,k)=D_P(smfr3d,k); D_P(keepfr,k)=D_P(keepfr3dflag,k);
    }
    D_F(lmavail)=land ? d_max(0.00001f,d_min(1.0f,Q(D_P(soilm1d,0),B(D_F(ref),D_F(qmin))))):0.0f;
    D_F(sat)=5.0e-4f; D_F(cn)=cn;
    D_F(snoh)=D_F(snflx)=D_F(s)=D_F(sublim)=D_F(evapl)=0.0f;
    D_F(infiltr)=D_F(smelt)=D_F(runoff1)=D_F(runoff2)=0.0f;
    run[i]=land && admitted;
    atomicAdd(flags+36,land && !ice ? 1u:0u);
    atomicAdd(flags+37,water ? 1u:0u);
    atomicAdd(flags+38,lake ? 1u:0u);
    atomicAdd(flags+39,ice ? 1u:0u);
}

// NumPy float32 arithmetic retains an operand NaN's payload and quiets it.
// Invalid arithmetic without NaN operands produces its negative quiet NaN.
// CUDA arithmetic canonicalizes both, so retain these words explicitly in
// the host diagnostic transcription.  Finite arithmetic remains one IEEE op.
__device__ __forceinline__ float d_np_result(float a,float b,float result) {
    if(isnan(a)) return __int_as_float(__float_as_int(a) | 0x00400000);
    if(isnan(b)) return __int_as_float(__float_as_int(b) | 0x00400000);
    return isnan(result) ? __int_as_float(0xffc00000):result;
}
__device__ __forceinline__ float d_npadd(float a,float b) { return d_np_result(a,b,__fadd_rn(a,b)); }
__device__ __forceinline__ float d_npsub(float a,float b) { return d_np_result(a,b,__fsub_rn(a,b)); }
__device__ __forceinline__ float d_npmul(float a,float b) { return d_np_result(a,b,__fmul_rn(a,b)); }
__device__ __forceinline__ float d_npdiv(float a,float b) { return d_np_result(a,b,__fdiv_rn(a,b)); }
#define P_A(a,b) d_npadd((a),(b))
#define P_B(a,b) d_npsub((a),(b))
#define P_M(a,b) d_npmul((a),(b))
#define P_Q(a,b) d_npdiv((a),(b))
__device__ __forceinline__ float d_saturation(float pressure,float temperature) {
    float x=d_npmax(-80.0f,P_B(temperature,273.16f));
    float liquid=-.321582393e-13f;
    liquid=P_A(.379534310e-11f,P_M(x,liquid));
    liquid=P_A(.702620698e-8f,P_M(x,liquid));
    liquid=P_A(.203154182e-5f,P_M(x,liquid));
    liquid=P_A(.299291081e-3f,P_M(x,liquid));
    liquid=P_A(.264224321e-1f,P_M(x,liquid));
    liquid=P_A(.143177157e01f,P_M(x,liquid));
    liquid=P_A(.444606896e02f,P_M(x,liquid));
    liquid=P_A(.611583699e03f,P_M(x,liquid));
    float ice=.161444444e-12f;
    ice=P_A(.105785160e-9f,P_M(x,ice));
    ice=P_A(.307839583e-7f,P_M(x,ice));
    ice=P_A(.521693933e-5f,P_M(x,ice));
    ice=P_A(.565392987e-3f,P_M(x,ice));
    ice=P_A(.402737184e-1f,P_M(x,ice));
    ice=P_A(.184672631e01f,P_M(x,ice));
    ice=P_A(.499320233e02f,P_M(x,ice));
    ice=P_A(.609868993e03f,P_M(x,ice));
    return P_B(temperature,273.15f)<=0.0f ? P_Q(P_M(.622f,ice),P_B(pressure,ice)) : P_Q(P_M(.622f,liquid),P_B(pressure,liquid));
}
#define REBLEND(name,sea) if(component) D_F(name)=A(M(D_F(name),fraction),M(B(1.0f,fraction),sea))
#define FROM(name,out) D_F(name)=o_##out[i]
extern "C" __global__ void ruc_driver_epilogue(
    const unsigned long long* sp,const unsigned long long* op,
    const int* integer,const bool* run,unsigned* flags,
    const float* tbq,const float* lemitbl,const float* half,float dt,int n,
    const float* landusef,int nlcat,int mosaic_lu,int crop,int natural,
    int irrigation,int log_profile,float xice_threshold) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    D_DECLARE_SCRATCH
    D_DECLARE_OUTPUT
    float qsat=Q(d_qsn(D_F(t3d),tbq,flags,1024),M(D_F(p8w),1.0e-2f));
    if(run[i]) {
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilm1d,k)=o_soilm1d[k*n+i]; D_P(tso1d,k)=o_ts1d[k*n+i];
            D_P(smfrkeep,k)=o_smfrkeep[k*n+i]; D_P(keepfr,k)=o_keepfr[k*n+i];
            D_P(soiliqw,k)=o_soiliqw[k*n+i];
        }
        FROM(soilt,soilt); FROM(soilt1,soilt1); FROM(tsnav,tsnav); FROM(dew,dew);
        FROM(qvg,qvg); FROM(qsg,qsg); FROM(qcg,qcg); FROM(snom,snom);
        FROM(snowfallac,snowfallac); FROM(acsnow,acsnow); FROM(alb,alb);
        FROM(znt,znt); FROM(emissl,emiss); FROM(snwe,snwe); FROM(snhei,snhei);
        FROM(snowfrac,snowfrac); FROM(rhosnfall,rhosnfall); FROM(canwatr,cst);
        FROM(lmavail,mavail); FROM(smelt,smelt); FROM(runoff1,runoff1);
        FROM(runoff2,runoff2); FROM(infiltr,infiltr); FROM(qfx,eeta);
        FROM(lh,qfx); FROM(hfx,hfx); FROM(s,s);
        // Irrigation after SFCTMP and before soil diagnostics, by WRF
        // lineage (gpuwm.core.ruc_mosaic.IRRIGATION_FORMS; the host twin is
        // ruc_mosaic.irrigate).  irrigation==1: WRF v4.6.1 LSMRUC:985-1009
        // under mosaic_lu, a per-step relaxation.  irrigation==0: WRF v4.5.2
        // LSMRUC:970-999, a crop-fraction-scaled hard floor with no mosaic
        // gate.  It reads the LANDUSEF fractions when the run carries them
        // (nlcat>0); a run without them (mosaic_lu=0) gives the dominant
        // category the whole cell, which is WRF's arithmetic on a one-hot
        // LANDUSEF.  Its gates read the leaf area index SOILVEGIN left in
        // the scratch, table or 2-D, and the dominant category.
        if(irrigation==1) {
            if(mosaic_lu) {
                float croparea=landusef[(crop-1)*n+i];
                float naturalarea=landusef[(natural-1)*n+i];
                float factor=d_max(0.0f,d_min(1.0f,Q(B(D_F(vegfra),D_F(shdmin)),d_max(1.0f,B(D_F(shdmax),D_F(shdmin))))));
                if((croparea>0.0f || naturalarea>0.0f) && factor>0.75f) {
                    float cropsm=B(M(1.1f,D_F(wilt)),D_F(qmin));
                    float cropfr=d_min(1.0f,A(croparea,M(0.4f,naturalarea)));
                    for(int k=0;k<integer[3*n+i];++k) {
                        float newsm=A(M(cropsm,cropfr),M(B(1.0f,cropfr),D_P(soilm1d,k)));
                        if(D_P(soilm1d,k)<newsm) D_P(soilm1d,k)=newsm;
                    }
                }
            }
        } else {
            float croparea=nlcat>0 ? landusef[(crop-1)*n+i]:(integer[i]==crop ? 1.0f:0.0f);
            float naturalarea=nlcat>0 ? landusef[(natural-1)*n+i]:(integer[i]==natural ? 1.0f:0.0f);
            if(croparea>0.0f && D_F(lai)>1.1f) {
                float cropsm=B(M(1.1f,D_F(wilt)),D_F(qmin));
                float floor=M(cropsm,croparea);
                for(int k=0;k<integer[3*n+i];++k) {
                    if(D_P(soilm1d,k)<floor) D_P(soilm1d,k)=floor;
                }
            } else if(integer[i]==natural && D_F(lai)>0.7f) {
                float cropsm=B(M(1.2f,D_F(wilt)),D_F(qmin));
                float floor=M(M(cropsm,naturalarea),0.4f);
                for(int k=0;k<integer[3*n+i];++k) {
                    if(D_P(soilm1d,k)<floor) D_P(soilm1d,k)=floor;
                }
            }
        }
        float available=0.0f,maximum=0.0f;
        for(int k=0;k<RUC_NZS-1;++k) {
            float thickness=B(half[k+1],half[k]);
            available=A(available,M(A(D_F(qmin),D_P(soilm1d,k)),thickness));
            maximum=A(maximum,M(A(D_F(qmin),D_F(dqm)),thickness));
        }
        float bottom=B(ruc_soil_layer_depth[RUC_NZS-1],half[RUC_NZS-1]);
        available=A(available,M(A(D_F(qmin),D_P(soilm1d,RUC_NZS-1)),bottom));
        maximum=A(maximum,M(A(D_F(qmin),D_F(dqm)),bottom));
        float sr=M(M(D_F(runoff1),dt),1000.0f),ur=M(M(D_F(runoff2),dt),1000.0f);
        D_F(sfcrunoff)=A(D_F(sfcrunoff),sr); D_F(udrunoff)=A(D_F(udrunoff),ur);
        D_F(acrunoff)=A(D_F(acrunoff),sr); D_F(smavail)=M(available,1000.0f);
        D_F(smmax)=M(maximum,1000.0f);
        for(int k=0;k<RUC_NZS;++k) {
            D_P(soilmois,k)=A(D_P(soilm1d,k),D_F(qmin));
            D_P(sh2o,k)=d_min(A(D_P(soiliqw,k),D_F(qmin)),D_P(soilmois,k));
            D_P(tso,k)=k==RUC_NZS-1 ? D_F(tbot):D_P(tso1d,k);
            D_P(smfr3d,k)=D_P(smfrkeep,k); D_P(keepfr3dflag,k)=D_P(keepfr,k);
        }
        D_F(z0)=D_F(znt); D_F(sfcexc)=D_F(tkms);
        D_F(qsfc)=Q(D_F(qvg),A(1.0f,D_F(qvg)));
        D_F(chklowq)=D_F(qv3d)>=M(qsat,0.95f) && D_F(qv3d)<D_F(qvg) ? 0.0f:1.0f;
        D_F(emiss)=D_F(snow)==0.0f ? lemitbl[integer[i]-1]:D_F(emissl);
        D_F(snow)=M(D_F(snwe),1000.0f); D_F(snowh)=D_F(snhei);
        D_F(canwat)=M(D_F(canwatr),1000.0f); D_F(mavail)=D_F(lmavail);
        D_F(sfcevp)=A(D_F(sfcevp),M(D_F(qfx),dt)); D_F(grdflx)=M(-1.0f,D_F(s));
        D_F(snowc)=D_F(snowfrac)>0.0f && D_F(xice)>=xice_threshold ? M(D_F(snowfrac),D_F(xice)):D_F(snowfrac);
        D_F(rhosnf)=D_F(rhosnfall);
        D_F(sfcevp)=A(D_F(sfcevp),M(D_F(qfx),dt));
    }
    D_CHECK_OUTPUT
    float fraction=D_F(xice);
    bool component=fraction>=xice_threshold && fraction<=1.0f;
    REBLEND(alb,0.08f); REBLEND(emiss,0.98f);
    REBLEND(flhc,D_F(flhc_sea)); REBLEND(flqc,D_F(flqc_sea));
    REBLEND(cpm,D_F(cpm_sea)); REBLEND(cqs2,D_F(cqs2_sea));
    REBLEND(chs2,D_F(chs2_sea)); REBLEND(chs,D_F(chs_sea));
    REBLEND(qsfc,D_F(qsfc_sea)); REBLEND(qgh,D_F(qgh_sea));
    REBLEND(hfx,D_F(hfx_sea)); REBLEND(qfx,D_F(qfx_sea)); REBLEND(lh,D_F(lh_sea));
    if(component) D_F(tsk_save)=D_F(soilt);
    REBLEND(soilt,D_F(tsk_sea));
    float cqs=Q(D_F(flqc),M(D_F(mavail),D_F(rho3d)));
    D_F(chs)=Q(D_F(flhc),M(D_F(cpm),D_F(rho3d)));
    float th2=D_F(chs2)<1.0e-5f ? P_M(D_F(t3d),D_F(scale)) : P_B(P_M(D_F(soilt),D_F(scale)),P_Q(D_F(hfx),P_M(P_M(D_F(rho3d),1004.5f),D_F(chs2))));
    float t2=P_M(th2,D_F(inverse));
    float lower=d_npmin(D_F(soilt),D_F(t3d)),upper=d_npmax(D_F(soilt),D_F(t3d));
    t2=d_npmin(upper,d_npmax(lower,t2));
    D_F(t2)=t2; D_F(th2)=P_M(t2,D_F(scale));
    float qlev=d_npmin(d_saturation(D_F(p8w),D_F(t3d)),D_F(qv3d));
    float prox=P_A(qlev,P_Q(D_F(qfx),P_M(D_F(rho3d),cqs)));
    float qsfcmr=P_Q(D_F(qsfc),P_B(1.0f,D_F(qsfc)));
    float q2=D_F(cqs2)<1.0e-5f ? qlev:P_B(prox,P_Q(D_F(qfx),P_M(D_F(rho3d),D_F(cqs2))));
    q2=d_npmin(d_npmax(qsfcmr,qlev),d_npmax(d_npmin(qsfcmr,qlev),q2));
    D_F(q2)=d_npmin(d_saturation(D_F(psfc),t2),q2);
    // ruc_2m_diagnostic = log_profile: the operational RAP/HRRR branch's
    // module_sf_sfcdiags_ruclsm.F:150-179 (gpuwm.core.ruc_tier
    // RUC_2M_DIAGNOSTIC_FORMS; host twin ruc_runtime._sfcdiags_ruclsm).
    // dz1 is half the lowest layer, LSMRUC's conflx.
    if(log_profile) {
        float dz1=D_F(conflx);
        float dT=P_B(D_F(t3d),D_F(soilt));
        float dQ=P_B(qlev,qsfcmr);
        if(dT>0.0f) {
            float fh=d_npmin(d_npmax(P_B(1.0f,P_Q(dT,10.0f)),0.01f),1.0f);
            float fac=P_Q(gfk_log(P_Q(P_A(2.0f,0.05f),P_A(0.05f,fh))),
                          gfk_log(P_Q(P_A(dz1,0.05f),P_A(0.05f,fh))));
            float t2a=P_A(D_F(soilt),P_M(fac,P_B(D_F(t3d),D_F(soilt))));
            D_F(t2)=t2a; D_F(th2)=P_M(t2a,D_F(scale));
        }
        if(dQ>0.0f) {
            float fh=d_npmin(d_npmax(P_B(1.0f,P_Q(dQ,0.003f)),0.01f),1.0f);
            float fac=P_Q(gfk_log(P_Q(P_A(2.0f,0.05f),P_A(0.05f,fh))),
                          gfk_log(P_Q(P_A(dz1,0.05f),P_A(0.05f,fh))));
            D_F(q2)=P_A(qsfcmr,P_M(fac,P_B(qlev,qsfcmr)));
        }
    }
}

// The fields are written only when no check anywhere in the call failed:
// the driver's own words and the fused sfctmp's bit words (sflags).
extern "C" __global__ void ruc_driver_commit(
    const unsigned long long* sp,const unsigned long long* cp,const unsigned* flags,
    const unsigned long long* sflags,int swords,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int word=0;word<36;++word) if(flags[word]) return;
    for(int word=0;word<swords;++word) if(sflags[word]) return;
    D_DECLARE_SCRATCH
    D_COMMIT
}
#undef A
#undef M
#undef B
#undef Q
#undef REBLEND
#undef FROM

#undef P_A
#undef P_B
#undef P_M
#undef P_Q
