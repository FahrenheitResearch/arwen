// WRF 4.6.1 aerosol_in, aer_time_int, aer_p_int and iaer=6.
// Tables are caller-owned. Every float operation rounds separately in both arms.
__device__ void aer1_bracket(const float *v,int n,float x,int &a,int &b,float &w1,float &w2) {
  a=-1; b=-1; float lo=0,hi=0;
  for(int k=0;k<n-1;k++) if(v[k]<=x && v[k+1]>x) {a=k;b=k+1;lo=v[k];hi=v[k+1];break;}
  if(a<0) {a=n-1;b=0;lo=v[n-1];hi=v[0]; if(x<v[0]) lo=__fsub_rn(lo,360.f);else hi=__fadd_rn(hi,360.f);}
  w2=__fdiv_rn(__fsub_rn(x,lo),__fsub_rn(hi,lo)); w1=__fsub_rn(1.f,w2);
}
extern "C" __global__ void rrtmg_aer1_monthly(int nc,const float *lat,const float *lon,
 const float *clat,const float *clon,const float *clim,float *monthly) {
  int q=blockIdx.x*blockDim.x+threadIdx.x; if(q>=nc*864)return;
  int s=q%6,k=(q/6)%12,m=(q/72)%12,c=q/864;
  int a,b,x,y; float la,lb,xa,xb;
  aer1_bracket(clat,46,lat[c],a,b,la,lb); aer1_bracket(clon,72,lon[c],x,y,xa,xb);
  int base=((s*12+m)*46)*12*72;
  float u=__fadd_rn(__fmul_rn(la,clim[base+(a*12+k)*72+x]),__fmul_rn(lb,clim[base+(b*12+k)*72+x]));
  float v=__fadd_rn(__fmul_rn(la,clim[base+(a*12+k)*72+y]),__fmul_rn(lb,clim[base+(b*12+k)*72+y]));
  monthly[q]=__fadd_rn(__fmul_rn(xa,u),__fmul_rn(xb,v));
}
extern "C" __global__ void rrtmg_aer1_time(int nc,float julian,const float *monthly,const int *dates,float *timed) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nc*72)return;
  float day=__fadd_rn(julian,1.f);int whole=(int)day;
  day=__fsub_rn(day,(float)whole);whole=whole%365;if(whole==0)whole=365;day=__fadd_rn(day,(float)whole);
  int np=0;bool found=false;
  for(int m=0;m<12;m++)if((float)dates[m]>day && !found){np=m;found=true;}
  int nm=np>0?np-1:11;float dp=(float)dates[np],dm=(float)dates[nm],dt,f1,f2;
  if(np==0){dt=__fsub_rn(__fadd_rn(dp,365.f),dm);
    if(day>dp){f1=__fdiv_rn(__fsub_rn(__fadd_rn(dp,365.f),day),dt);f2=__fdiv_rn(__fsub_rn(day,dm),dt);}
    else{f1=__fdiv_rn(__fsub_rn(dp,day),dt);f2=__fdiv_rn(__fsub_rn(__fadd_rn(day,365.f),dm),dt);}
  }else{dt=__fsub_rn(dp,dm);f1=__fdiv_rn(__fsub_rn(dp,day),dt);f2=__fdiv_rn(__fsub_rn(day,dm),dt);}
  int c=q/72,within=q%72;
  timed[q]=__fadd_rn(__fmul_rn(monthly[(c*12+nm)*72+within],f1),__fmul_rn(monthly[(c*12+np)*72+within],f2));
}
extern "C" __global__ void rrtmg_aer1_pressure(int nc,int nz,const float *p,const float *pf,
 const float *pin,const float *timed,float *aerod) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nc*nz*6)return;
  int s=q%6,k=(q/6)%nz,c=q/(6*nz);
  float mid=__fmul_rn(p[c*nz+k],0.01f);
  float dpm=__fsub_rn(pf[c*(nz+1)+k],pf[c*(nz+1)+k+1]);
  int base=c*72+s,upper=0;float value;
  if(mid<pin[0]) value=__fdiv_rn(__fmul_rn(timed[base],mid),pin[0]);
  else if(mid>pin[11]) value=timed[base+11*6];
  else {for(int h=0;h<11;h++)if(pin[h]<mid && mid<=pin[h+1])upper=h;
    float dpu=__fsub_rn(mid,pin[upper]),dpl=__fsub_rn(pin[upper+1],mid);
    value=__fdiv_rn(__fadd_rn(__fmul_rn(timed[base+upper*6],dpl),__fmul_rn(timed[base+(upper+1)*6],dpu)),__fadd_rn(dpl,dpu));}
  aerod[q]=__fmul_rn(value,dpm);
}
extern "C" __global__ void rrtmg_aer1_total(int nc,int nz,const float *aerod,float *aodtot) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;float a=0.f;
  for(int s=0;s<6;s++)for(int k=0;k<nz;k++)a=__fadd_rn(a,aerod[(c*nz+k)*6+s]);aodtot[c]=a;
}
extern "C" __global__ void rrtmg_aer1_optics(int nc,int nz,int nlayers,const float *aerod,
 const float *tau,const float *ssa,const float *asy,float *ztaua,float *zasya,float *zomga) {
  int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=nc*14*nlayers)return;
  int k=q%nlayers,b=(q/nlayers)%14,c=q/(14*nlayers);float t=0.f,w=0.f,g=0.f;
  if(k<nz)for(int s=0;s<6;s++){
    float e=aerod[(c*nz+k)*6+s],te=__fmul_rn(tau[b*6+s],e);
    t=__fadd_rn(t,te);w=__fadd_rn(w,__fmul_rn(te,ssa[b*6+s]));
    g=__fadd_rn(g,__fmul_rn(__fmul_rn(te,ssa[b*6+s]),asy[b*6+s]));
  }
  if(w!=0.f)g=__fdiv_rn(g,w);if(t!=0.f)w=__fdiv_rn(w,t);
  ztaua[q]=t;zasya[q]=g;zomga[q]=w;
}
