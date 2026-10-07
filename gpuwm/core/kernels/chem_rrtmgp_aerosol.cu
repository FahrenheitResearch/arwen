// BSD-3-Clause. Port of pinned RTE-RRTMGP MERRA aerosol optics.
// One thread per column; all mass levels, including the top mass level.
// Packed table: [type-1, band, size bin, RH, (ext,ssa,g)].
__device__ void merra_moments(int type, float size, float mass, float rh,
 const float *table, const float *lims, const float *rhs, int nrh, int nb,
 int band, float &t, float &ts, float &tsg) {
 t=ts=tsg=0.f;
 if(type==0) return;
 int bin=-1;
 for(int i=0;i<5;i++) if(size>=lims[2*i] && size<=lims[2*i+1]) bin=i;
 int hi=0;
 while(hi<nrh && rh>rhs[hi]) ++hi;
 int lo=max(0,hi-1); hi=min(nrh-1,hi);
 float drh0=FSUB(rhs[hi],rhs[lo]);
 float drh1=FSUB(rh,rhs[lo]);
 float weight=(lo==hi)?0.f:FDIV(drh1,drh0);
 if(type<1 || type>7 || bin<0) return;
 float v[3];
 for(int j=0;j<3;j++) {
  int base=((((type-1)*nb+band)*5+bin)*nrh)*3+j;
  float a=table[base+lo*3], b=table[base+hi*3];
  // Dry tables have no interpolation in the source.
  v[j]=(type==1 || type==5 || type==7)?a:FADD(a,FMUL(weight,FSUB(b,a)));
 }
 t=FMUL(mass,v[0]); ts=FMUL(t,v[1]); tsg=FMUL(ts,v[2]);
}
__device__ void merra_store(float t,float ts,float tsg,int out,
 float *tau,float *ssa,float *g,float *absorption) {
 tau[out]=t;
 ssa[out]=FDIV(ts,fmaxf(0x1p-23f,t));
 g[out]=FDIV(tsg,fmaxf(0x1p-23f,ts));
 absorption[out]=FSUB(t,ts);
}
extern "C" __global__ void merra_optics(const int *types,const float *sizes,
 const float *mass,const float *rh,const float *table,const float *lims,
 const float *rhs,int nrh,int nb,int nz,int nc,float *tau,float *ssa,float *g,
 float *absorption,int *status) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc) return;
 for(int k=0;k<nz;k++) {
  int i=k*nc+c, type=types[i];
  int err=(type<0 || type>7)?1:0;
  if(type>0 && (sizes[i]<lims[0] || sizes[i]>lims[9])) err|=2;
  if(type>0 && (rh[i]<0.f || rh[i]>1.f)) err|=4;
  status[i]=err;
  for(int b=0;b<nb;b++) {
   float t,ts,tsg;
   if(err) {t=ts=tsg=0.f;} else
    merra_moments(type,sizes[i],mass[i],rh[i],table,lims,rhs,nrh,nb,b,t,ts,tsg);
   merra_store(t,ts,tsg,(c*nz+k)*nb+b,tau,ssa,g,absorption);
  }
 }
}
extern "C" __global__ void merra_rows(const unsigned long long *fields,
 const int *types,const float *sizes,const float *scales,int nrow,
 const float *dry_dp,const float *rh,float gravity,const float *table,
 const float *lims,const float *rhs,int nrh,int nb,int nz,int nc,
 float *tau,float *ssa,float *g,float *absorption,int *status) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc) return;
 for(int k=0;k<nz;k++) {
  int i=k*nc+c;
  int err=(!isfinite(dry_dp[i]) || dry_dp[i]<=0.f)?8:0;
  if(!isfinite(rh[i]) || rh[i]<0.f || rh[i]>1.f) err|=4;
  for(int r=0;r<nrow;r++) {
   float q=((const float*)fields[r])[i];
   if(!isfinite(q) || q<0.f) err|=16;
  }
  status[i]=err;
  for(int b=0;b<nb;b++) {
   float total=0.f, totalw=0.f, totalg=0.f, absorb=0.f;
   if(!err) for(int r=0;r<nrow;r++) {
    float q=((const float*)fields[r])[i];
    float mass=FMUL(FMUL(q,scales[r]),FDIV(dry_dp[i],gravity));
    float t,ts,tsg;
    merra_moments(types[r],sizes[r],mass,rh[i],table,lims,rhs,nrh,nb,b,t,ts,tsg);
    // Normalize each species exactly as aerosol_optics does, then increment.
    float w=FDIV(ts,fmaxf(0x1p-23f,t));
    float gg=FDIV(tsg,fmaxf(0x1p-23f,ts));
    float s=FMUL(t,w);
    float newt=FADD(total,t), news=FADD(FMUL(total,totalw),s);
    float newg=FDIV(FADD(FMUL(FMUL(total,totalw),totalg),FMUL(s,gg)),fmaxf(3.f*0x1p-126f,news));
    float neww=FDIV(news,fmaxf(3.f*0x1p-126f,newt));
    total=newt; totalw=neww; totalg=newg;
    absorb=FADD(absorb,FSUB(t,ts));
   }
   int o=(c*nz+k)*nb+b;
   tau[o]=total; ssa[o]=totalw; g[o]=totalg;
   absorption[o]=absorb;
  }
 }
}
