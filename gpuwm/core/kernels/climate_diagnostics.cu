// NOAA-EMC/HRRR 40ee6058, module_diag_cl.F. All arithmetic is REAL(4).
// Retain the source's strict extrema comparisons and untouched reset times.
__device__ inline void cl_finish(float &mean,float &std,int n) {
 mean=mean/n; std=sqrtf(fmaxf(0.f,std/n-mean*mean));
}
__device__ inline void cl_six(float x,float tt,bool reset,bool finish,int n,
 float &lo,float &hi,float &tlo,float &thi,float &mean,float &std) {
 if(reset) {lo=x;hi=x;mean=x;std=x*x;}
 else {
  if(x<lo){lo=x;tlo=tt;} if(x>hi){hi=x;thi=tt;}
  mean=mean+x;std=std+x*x;
  if(finish)cl_finish(mean,std,n);
 }
}
__device__ inline void cl_four(float x,float tt,bool reset,bool finish,int n,
 float &hi,float &thi,float &mean,float &std) {
 if(reset){hi=x;mean=x;std=x*x;}
 else {if(x>hi){hi=x;thi=tt;} mean=mean+x;std=std+x*x;
  if(finish)cl_finish(mean,std,n);}
}
extern "C" __global__ void clwrf_output_calc(
 const float *t2,const float *q2,const float *tsk,const float *swdown,
 const float *swddni,const float *u10,const float *v10,
 const float *raincv,const float *rainncv,
 float *t2min,
 float *t2max,
 float *tt2min,
 float *tt2max,
 float *t2mean,
 float *t2std,
 float *q2min,
 float *q2max,
 float *tq2min,
 float *tq2max,
 float *q2mean,
 float *q2std,
 float *skintempmin,
 float *skintempmax,
 float *tskintempmin,
 float *tskintempmax,
 float *skintempmean,
 float *skintempstd,
 float *swradmin,
 float *swradmax,
 float *tswradmin,
 float *tswradmax,
 float *swradmean,
 float *swradstd,
 float *swnormmin,
 float *swnormmax,
 float *tswnormmin,
 float *tswnormmax,
 float *swnormmean,
 float *swnormstd,
 float *u10max,
 float *v10max,
 float *spduv10max,
 float *tspduv10max,
 float *u10mean,
 float *v10mean,
 float *spduv10mean,
 float *u10std,
 float *v10std,
 float *spduv10std,
 float *raincvmax,
 float *traincvmax,
 float *raincvmean,
 float *raincvstd,
 float *rainncvmax,
 float *trainncvmax,
 float *rainncvmean,
 float *rainncvstd,
 float *cl_nsteps,
 float *cl_nsteps_wind,
 int count,float dt,float xtime,float secs,int diag_int,int wind_int,int restart) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float period=diag_int*60.f/dt,wp=wind_int*60.f/dt;
 bool reset=fabsf(fmodf(secs/dt,period))<.01f && !restart;
 bool finish=fabsf(fmodf((secs+dt)/dt,period))<.01f;
 bool wr=fabsf(fmodf(secs/dt,wp))<.01f && !restart;
 bool wf=fabsf(fmodf((secs+dt)/dt,wp))<.01f;
 int n=(int)period,nw=(int)wp;
 cl_nsteps[i]=(float)n;cl_nsteps_wind[i]=(float)nw;
 float tt=xtime+__fdiv_rn(dt,60.f);
 cl_six(t2[i],tt,reset,finish,n,t2min[i],t2max[i],tt2min[i],tt2max[i],t2mean[i],t2std[i]);
 cl_six(q2[i],tt,reset,finish,n,q2min[i],q2max[i],tq2min[i],tq2max[i],q2mean[i],q2std[i]);
 cl_six(tsk[i],tt,reset,finish,n,skintempmin[i],skintempmax[i],tskintempmin[i],tskintempmax[i],skintempmean[i],skintempstd[i]);
 cl_six(swdown[i],tt,reset,finish,n,swradmin[i],swradmax[i],tswradmin[i],tswradmax[i],swradmean[i],swradstd[i]);
 cl_six(swddni[i],tt,reset,finish,n,swnormmin[i],swnormmax[i],tswnormmin[i],tswnormmax[i],swnormmean[i],swnormstd[i]);
 cl_four(raincv[i]/dt,tt,reset,finish,n,raincvmax[i],traincvmax[i],raincvmean[i],raincvstd[i]);
 cl_four(rainncv[i]/dt,tt,reset,finish,n,rainncvmax[i],trainncvmax[i],rainncvmean[i],rainncvstd[i]);
 float u=u10[i],v=v10[i],speed2=u*u+v*v,speed=sqrtf(speed2);
 if(wr){
  spduv10max[i]=speed;
  u10mean[i]=u;v10mean[i]=v;spduv10mean[i]=speed;
  u10std[i]=u*u;v10std[i]=v*v;spduv10std[i]=speed2;
 }else{
  if(speed>spduv10max[i]){
   u10max[i]=u;v10max[i]=v;spduv10max[i]=speed;tspduv10max[i]=tt;
  }
  spduv10mean[i]=spduv10mean[i]+speed;
  spduv10std[i]=spduv10std[i]+speed*speed;
  u10mean[i]=u10mean[i]+u;v10mean[i]=v10mean[i]+v;
  u10std[i]=u10std[i]+u*u;v10std[i]=v10std[i]+v*v;
  if(wf){cl_finish(u10mean[i],u10std[i],nw);
   cl_finish(v10mean[i],v10std[i],nw);
   cl_finish(spduv10mean[i],spduv10std[i],nw);}
 }
}
