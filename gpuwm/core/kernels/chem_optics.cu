// WRF-Chem v4.7.1 module_optical_averaging.F, GOCART volume approximation.
// All arrays use engine k,y,x order. One thread visits every level in a column.
// These functions are required stubs; this path currently calls none of them.
__device__ float chem_log10f(float x) { // STUB until the engine's glibc float32 trig header lands
    return log10f(x);
}
__device__ float chem_sinf(float x) { // STUB until the engine's glibc float32 trig header lands
    return sinf(x);
}
__device__ float chem_cosf(float x) { // STUB until the engine's glibc float32 trig header lands
    return cosf(x);
}
__device__ float chem_acosf(float x) { // STUB until the engine's glibc float32 trig header lands
    return acosf(x);
}

// binterp, 5171:5228. The terminal grid entry has zero interpolation weight.
__device__ void chem_bweights(float x,float y,const float *xt,const float *yt,
                             int &ix,int &iy,float *w) {
    int i=0,j=0;
    while(i<7 && !(x<xt[i])) ++i;
    while(j<7 && !(y<yt[j])) ++j;
    ix=max(i-1,0); iy=max(j-1,0);
    int ip=min(ix+1,6),jp=min(iy+1,6);
    float dx=FSUB(xt[ip],xt[ix]),dy=FSUB(yt[jp],yt[iy]);
    float t=fabsf(dx)>1.e-20f ? FDIV(FSUB(x,xt[ix]),dx):0.f;
    float u=fabsf(dy)>1.e-20f ? FDIV(FSUB(y,yt[iy]),dy):0.f;
    float tu=FMUL(t,u),tuc=FSUB(t,tu);
    w[0]=FSUB(FSUB(1.f,tuc),u);w[1]=tuc;w[2]=tu;w[3]=FSUB(u,tu);
}
__device__ float chem_coef(const float *a,int nc,int ix,int iy,const float *w) {
    int ip=min(ix+1,6),jp=min(iy+1,6);
    float v=FADD(FMUL(w[0],a[nc+50*(ix+7*iy)]),FMUL(w[1],a[nc+50*(ip+7*iy)]));
    v=FADD(v,FMUL(w[2],a[nc+50*(ip+7*jp)]));
    return FADD(v,FMUL(w[3],a[nc+50*(ix+7*jp)]));
}
__device__ float chem_cheb(const float *a,const float *ch,int ix,int iy,const float *w,bool logarithmic) {
    float v=FMUL(0.5f,chem_coef(a,0,ix,iy,w));
    for(int nc=1;nc<50;++nc) v=FADD(v,FMUL(ch[nc],chem_coef(a,nc,ix,iy,w)));
    return logarithmic ? gfk_exp(v):v;
}

// p: densities [so4,nh4,oin,oc,bc,na,cl,dust,msa,water], followed by
// hygroscopicities in source order [so4,oc,nh4,cl,na,msa,oin,bc2,dust].
// Then RH cap and molecular masses [sulfate,dry_air,Na,Cl,NaCl].
// ri: 20 bands, 11 components in complex-sum source order, real/imag fastest.
__device__ void chem_prep(const unsigned long long *ptrs,int nr,const int *ip,const float *rp,
 const float *frac,const float *mi,const float *mj,const float *mc,const float *diam,
 const float *p,const float *ri,int cell,float alt,float rh,float *rad,float *num,float *ridx) {
    float ai[8]={},aj[8]={};
    float conv=FMUL(FDIV(1.f,alt),1.e-12f);
    float convs=FDIV(FMUL(FMUL(FDIV(1.f,alt),1.e-9f),p[20]),p[21]);
    for(int r=0;r<nr;++r) {
        int target=ip[3*r],mode=ip[3*r+1],conversion=ip[3*r+2];
        if(mode==0) {
            float q=((const float*)ptrs[r])[cell],cv=conversion ? convs:conv;
            float a=FMUL(FMUL(rp[2*r],q),cv),b=FMUL(FMUL(FSUB(1.f,rp[2*r]),q),cv);
            if(rp[2*r+1]!=1.f){a=FMUL(a,rp[2*r+1]);b=FMUL(b,rp[2*r+1]);}
            ai[target]=FADD(ai[target],a);aj[target]=FADD(aj[target],b);
        }
    }
    float previous_soil=0.f;
    for(int s=0;s<9;++s) {
        float mass[11]={};
        mass[0]=FADD(FMUL(ai[0],mi[s]),FMUL(aj[0],mj[s]));
        mass[1]=0.f;
        mass[2]=FADD(FMUL(ai[1],mi[s]),FMUL(aj[1],mj[s]));
        mass[3]=FADD(FADD(FADD(FMUL(ai[2],mi[s]),FMUL(aj[2],mj[s])),FMUL(previous_soil,mc[s])),FMUL(0.f,mc[s]));
        float oc1=FADD(FMUL(aj[3],mj[s]),FMUL(ai[3],mi[s]));
        float oc2=FADD(FMUL(aj[4],mj[s]),FMUL(ai[4],mi[s]));
        float bc1=FADD(FMUL(ai[5],mi[s]),FMUL(aj[5],mj[s]));
        float bc2=FADD(FMUL(ai[6],mi[s]),FMUL(aj[6],mj[s]));
        mass[5]=FADD(oc1,oc2);mass[6]=FADD(bc1,bc2);
        mass[9]=FADD(FMUL(ai[7],mi[s]),FMUL(aj[7],mj[s]));
        float salt=0.f,soil=0.f;
        for(int r=0;r<nr;++r) {
            int mode=ip[3*r+1];
            if(mode!=0) {
                float q=((const float*)ptrs[r])[cell];
                if(mode==1) salt=FADD(salt,FMUL(frac[r*9+s],q));
                if(mode==2) soil=FADD(soil,FMUL(frac[r*9+s],q));
            }
        }
        mass[7]=FDIV(FMUL(FMUL(salt,conv),p[22]),p[24]);
        mass[8]=FDIV(FMUL(FMUL(salt,conv),p[23]),p[24]);
        mass[4]=FMUL(soil,conv);previous_soil=mass[4];
        float vso=FDIV(mass[0],p[0]),vnh=FDIV(mass[2],p[1]),voin=FDIV(mass[3],p[2]);
        float voc=FDIV(mass[5],p[3]),vbc=FDIV(mass[6],p[4]),vna=FDIV(mass[7],p[5]);
        float vcl=FDIV(mass[8],p[6]),vdust=FDIV(mass[4],p[7]),vmsa=FDIV(mass[9],p[8]);
        float voc2=FDIV(oc2,p[3]),vbc2=FDIV(bc2,p[4]);
        float vh=FADD(FMUL(vso,p[10]),FMUL(voc2,p[11]));
        vh=FADD(vh,FMUL(vnh,p[12]));vh=FADD(vh,FMUL(vcl,p[13]));vh=FADD(vh,FMUL(vna,p[14]));
        vh=FADD(vh,FMUL(vmsa,p[15]));vh=FADD(vh,FMUL(voin,p[16]));vh=FADD(vh,FMUL(vbc2,p[17]));vh=FADD(vh,FMUL(vdust,p[18]));
        float hr=fminf(p[19],rh);vh=FDIV(FMUL(hr,vh),FSUB(1.f,hr));
        mass[10]=FMUL(vh,p[9]);
        float vd=FADD(vso,0.f);vd=FADD(vd,vnh);vd=FADD(vd,voin);vd=FADD(vd,voc);
        vd=FADD(vd,vbc);vd=FADD(vd,vna);vd=FADD(vd,vcl);vd=FADD(vd,vdust);
        float vw=FADD(vd,vh),shell=FSUB(vw,vbc);
        float den=FMUL(FMUL(FMUL(0.52359877f,diam[s]),diam[s]),diam[s]);
        num[s]=FDIV(vw,den);
        float dp=diam[s];
        if(!(num[s]<1.e-20f || vw<1.e-20f)) dp=gfk_pow(FDIV(FMUL(1.90985f,vw),num[s]),0.3333333f);
        float half=FDIV(dp,2.f),lower=FDIV(FMUL(0.0390625f,1.e-4f),2.f);
        bool fallback=half<lower || num[s]<1.e-20f || shell<1.e-20f;
        rad[s]=fallback ? lower:half;
        // Complex arithmetic retains multiplication then division per component.
        for(int b=0;b<20;++b)for(int z=0;z<2;++z) {
            float value=0.f;
            for(int t=0;t<11;++t) {
                float term=FDIV(FMUL(ri[(b*11+t)*3+z],mass[t]),ri[(b*11+t)*3+2]);
                value=t==0 ? term:FADD(value,term);
            }
            ridx[(b*9+s)*2+z]=fallback ? (z==0 ? 1.5f:0.f):FDIV(value,vw);
        }
    }
}

extern "C" __global__ void chem_optics_gocart(const unsigned long long *ptrs,int nr,const int *ip,const float *rp,
 const float *frac,const float *mi,const float *mj,const float *mc,const float *diam,const float *p,const float *ri,
 const float *alt,const float *rh,const float *dz,const float *tables,const float *grids,const float *constants,
 float *sw_tau,float *sw_ext,float *sw_ssa,float *sw_asm,float *sw_back,float *lw_tau,float *lw_ext,
 float *moment,float *radius,float *number,float *refindex,int *invalid_refindex,int nz,int ncol) {
 // Every output is optional: a null pointer skips its store, never its
 // arithmetic, so a caller that needs four bands of tau does not hold 77
 // fields.  sw_* are (4,nz,ncol), lw_* (16,nz,ncol), moment (6,4,nz,ncol).
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=ncol)return;
 invalid_refindex[col]=0;
 int cells=nz*ncol;
 for(int k=0;k<nz;++k) {
  int cell=k*ncol+col;float rad[9],original_rad[9],num[9],idx[360];
  chem_prep(ptrs,nr,ip,rp,frac,mi,mj,mc,diam,p,ri,cell,alt[cell],rh[cell],rad,num,idx);
  float total=0.f;for(int m=0;m<9;++m){total=FADD(total,num[m]);original_rad[m]=rad[m];if(radius)radius[m*cells+cell]=rad[m];if(number)number[m*cells+cell]=num[m];}
  if(refindex)for(int t=0;t<360;++t)refindex[t*cells+cell]=idx[t];
  for(int b=0;b<20;++b) {
   float tau=0.f,ssa=0.f,asmv=0.f,back=0.f,mm[6]={};
   if(total>1.e-21f)for(int m=0;m<9;++m) {
    if(b<4)rad[m]=fminf(fmaxf(rad[m],constants[0]),constants[1]);
    float x=gfk_log(rad[m]);
    float xr=FDIV(FSUB(FSUB(FMUL(2.f,x),constants[3]),constants[2]),FSUB(constants[3],constants[2]));
    const float *xt=grids+b*14,*yt=xt+7;
    float re=idx[(b*9+m)*2],im=-idx[(b*9+m)*2+1];
    // module_optical_averaging.F:4690-4702 and 4965-4977.
    // The source aborts before grid clipping when these diagnostics fail.
    if(fabsf(re)>10.f || fabsf(re)<=0.001f || fabsf(im)>10.f)invalid_refindex[col]|=1<<b;
    re=fminf(fmaxf(re,xt[0]),xt[6]);im=fminf(fmaxf(im,yt[0]),yt[6]);
    int ix,iy;float w[4],ch[50];chem_bweights(re,im,xt,yt,ix,iy,w);
    ch[0]=1.f;ch[1]=xr;for(int c=2;c<50;++c)ch[c]=FSUB(FMUL(FMUL(2.f,xr),ch[c-1]),ch[c-2]);
    const float *a=tables+b*11*2450;
    float q=chem_cheb(a,ch,ix,iy,w,true),er=gfk_exp(x),area=FMUL(er,er);
    if(b<4) {
     float qs=fminf(chem_cheb(a+2450,ch,ix,iy,w,true),q);
     float g=chem_cheb(a+2*2450,ch,ix,iy,w,true),bs=chem_cheb(a+3*2450,ch,ix,iy,w,true);
     if(bs<=0.f)bs=0.f;
     float we=FMUL(FMUL(q,constants[4]),area),ws=FMUL(FMUL(qs,constants[4]),area);
     tau=FADD(tau,FMUL(we,num[m]));ssa=FADD(ssa,FMUL(ws,num[m]));
     asmv=FADD(asmv,FMUL(FMUL(g,ws),num[m]));
     back=FADD(back,FMUL(FMUL(FMUL(constants[4],area),bs),num[m]));
     for(int t=0;t<6;++t) {
      float pm=chem_cheb(a+(4+t)*2450,ch,ix,iy,w,false);
      // sizem is captured before the first SW wavelength clamps radius.
      float sizem=(b==0 ? original_rad[m]:rad[m]);
      if(pm<=0.f || (t>=2 && sizem<=0.03e-4f))pm=0.f;
      mm[t]=FADD(mm[t],FMUL(FMUL(ws,pm),num[m]));
     }
    }else {
     float wa=FMUL(FMUL(q,constants[4]),area);tau=FADD(tau,FMUL(wa,num[m]));
    }
   }
   float ext=FMUL(tau,1.e5f),optical=FMUL(FMUL(tau,dz[cell]),100.f);
   if(b<4) {
    if(total>1.e-21f){asmv=FDIV(asmv,ssa);for(int t=0;t<6;++t)mm[t]=FDIV(mm[t],ssa);ssa=FDIV(ssa,tau);}
    if(sw_tau)sw_tau[b*cells+cell]=fmaxf(optical,1.e-20f);
    if(sw_ext)sw_ext[b*cells+cell]=fmaxf(ext,1.e-20f);
    if(sw_ssa)sw_ssa[b*cells+cell]=fmaxf(fminf(ssa,FSUB(1.f,1.e-8f)),1.e-20f);
    if(sw_asm)sw_asm[b*cells+cell]=fmaxf(fminf(asmv,FSUB(1.f,1.e-8f)),1.e-20f);
    if(sw_back)sw_back[b*cells+cell]=fmaxf(FMUL(back,1.e5f),1.e-20f);
    if(moment)for(int t=0;t<6;++t)moment[(t*4+b)*cells+cell]=mm[t];
   }else {
    if(lw_tau)lw_tau[(b-4)*cells+cell]=fmaxf(optical,1.e-20f);
    if(lw_ext)lw_ext[(b-4)*cells+cell]=fmaxf(ext,1.e-20f);
   }
  }
 }
}

extern "C" __global__ void chem_optics_opt_out(const float *tau,const float *dz,float *ext,float *aod,int nz,int ncol) {
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=ncol)return;int cells=nz*ncol;float sum=0.f;
 for(int k=0;k<nz;++k) {
  int i=k*ncol+col;
  float ang=FDIV(gfk_log(FDIV(tau[i],tau[3*cells+i])),gfk_log(999.f/300.f));
  float value=FDIV(FMUL(FMUL(tau[cells+i],1.e3f),gfk_pow(0.4f/0.55f,ang)),dz[i]);
  ext[i]=value;
  // WRF-Chem has no AOD5502D field. This is its EXTCOF55 column integral.
  sum=FADD(sum,FMUL(FMUL(value,dz[i]),1.e-3f));
 }aod[col]=sum;
}

// module_ra_rrtmg_sw.F:11355-11430, includes threshold and column rescale.
__device__ void chem_rrtmg_sw_aer_bands(const float *tau,const float *ssa,const float *asmv,
 const float *bounds,float *bt,float *bs,float *ba,int *negative,int col,int nz,int ncol) {
 int cells=nz*ncol;
 negative[col]=0;
 for(int b=0;b<14;++b) {
  float mid=FMUL(0.5f,FADD(bounds[b],bounds[14+b])),sum=0.f;
  for(int k=0;k<nz;++k) {
   int i=k*ncol+col,o=(k*14+b)*ncol+col;
   float t=0.f,s=1.f,a=0.f;
   if(tau[i]>1.e-9f && tau[3*cells+i]>1.e-9f) {
    float ang=FDIV(gfk_log(FDIV(tau[i],tau[3*cells+i])),gfk_log(999.f/300.f));
    t=FMUL(tau[cells+i],gfk_pow(FDIV(0.4f,mid),ang));
    float slope=FDIV(FSUB(ssa[2*cells+i],ssa[cells+i]),0.2f);
    s=FADD(FMUL(slope,FSUB(mid,0.6f)),ssa[2*cells+i]);
    if(s<0.4f)s=0.4f;if(s>=1.f)s=1.f;
    slope=FDIV(FSUB(asmv[2*cells+i],asmv[cells+i]),0.2f);
    a=FADD(FMUL(slope,FSUB(mid,0.6f)),asmv[2*cells+i]);
    if(a<0.5f)a=0.5f;if(a>=1.f)a=1.f;
   }bt[o]=t;bs[o]=s;ba[o]=a;sum=FADD(sum,t);
  }
  // WRF calls wrf_error_fatal for a negative column sum. Return its band
  // mask so the launcher can report the fatal condition on the host.
  if(sum<0.f)negative[col]|=1<<b;
  else if(sum>6.f)for(int k=0;k<nz;++k){int o=(k*14+b)*ncol+col;bt[o]=FDIV(FMUL(bt[o],6.f),sum);}
 }
}
extern "C" __global__ void chem_optics_rrtmg_sw(const float *tau,const float *ssa,const float *asmv,
 const float *bounds,float *bt,float *bs,float *ba,int *negative,int nz,int ncol) {
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col<ncol)chem_rrtmg_sw_aer_bands(tau,ssa,asmv,bounds,bt,bs,ba,negative,col,nz,ncol);
}
