// Apache-2.0. Column entry point and WOOF table construction.
__device__ void sm_run_column(int column,int nc,int km,int ntrac,double dt,const double* input,double* output,int* output_kpbl,double* scratch,const double* table,int* highwater){
int capacity=(220+16*ntrac)*km+4096; SMWorkspace ws(scratch,table,capacity);int offset=0;
double* u1=ws.alloc<double>(km);for(int j=0;j<km;++j)u1[j]=input[(offset+j)*nc+column];offset+=km;
double* v1=ws.alloc<double>(km);for(int j=0;j<km;++j)v1[j]=input[(offset+j)*nc+column];offset+=km;
double* t1=ws.alloc<double>(km);for(int j=0;j<km;++j)t1[j]=input[(offset+j)*nc+column];offset+=km;
double* q1=ws.alloc<double>(ntrac*km);for(int j=0;j<ntrac*km;++j)q1[j]=input[(offset+j)*nc+column];offset+=ntrac*km;
double* swh=ws.alloc<double>(km);for(int j=0;j<km;++j)swh[j]=input[(offset+j)*nc+column];offset+=km;
double* hlw=ws.alloc<double>(km);for(int j=0;j<km;++j)hlw[j]=input[(offset+j)*nc+column];offset+=km;
double* prsi=ws.alloc<double>(km+1);for(int j=0;j<km+1;++j)prsi[j]=input[(offset+j)*nc+column];offset+=km+1;
double* prsl=ws.alloc<double>(km);for(int j=0;j<km;++j)prsl[j]=input[(offset+j)*nc+column];offset+=km;
double* prslk=ws.alloc<double>(km);for(int j=0;j<km;++j)prslk[j]=input[(offset+j)*nc+column];offset+=km;
double* phii=ws.alloc<double>(km+1);for(int j=0;j<km+1;++j)phii[j]=input[(offset+j)*nc+column];offset+=km+1;
double* phil=ws.alloc<double>(km);for(int j=0;j<km;++j)phil[j]=input[(offset+j)*nc+column];offset+=km;
double* psk=ws.alloc<double>(1);psk[0]=input[(offset+0)*nc+column];
double* rbsoil=ws.alloc<double>(1);rbsoil[0]=input[(offset+1)*nc+column];
double* zorl=ws.alloc<double>(1);zorl[0]=input[(offset+2)*nc+column];
double* tsea=ws.alloc<double>(1);tsea[0]=input[(offset+3)*nc+column];
double* heat=ws.alloc<double>(1);heat[0]=input[(offset+4)*nc+column];
double* evap=ws.alloc<double>(1);evap[0]=input[(offset+5)*nc+column];
double* stress=ws.alloc<double>(1);stress[0]=input[(offset+6)*nc+column];
double* spd1=ws.alloc<double>(1);spd1[0]=input[(offset+7)*nc+column];
double* u10m=ws.alloc<double>(1);u10m[0]=input[(offset+8)*nc+column];
double* v10m=ws.alloc<double>(1);v10m[0]=input[(offset+9)*nc+column];
double* fm=ws.alloc<double>(1);fm[0]=input[(offset+10)*nc+column];
double* fh=ws.alloc<double>(1);fh[0]=input[(offset+11)*nc+column];
double* xmu=ws.alloc<double>(1);xmu[0]=input[(offset+12)*nc+column];
double* garea=ws.alloc<double>(1);garea[0]=input[(offset+13)*nc+column];
double* zvfun=ws.alloc<double>(1);zvfun[0]=input[(offset+14)*nc+column];
double* sigmaf=ws.alloc<double>(1);sigmaf[0]=input[(offset+15)*nc+column];
offset+=16;
double* du=ws.alloc<double>(km);for(int j=0;j<km;++j)du[j]=0.;
double* dv=ws.alloc<double>(km);for(int j=0;j<km;++j)dv[j]=0.;
double* tdt=ws.alloc<double>(km);for(int j=0;j<km;++j)tdt[j]=0.;
double* rtg=ws.alloc<double>(ntrac*km);for(int j=0;j<ntrac*km;++j)rtg[j]=0.;
double* hpbl=ws.alloc<double>(1);for(int j=0;j<1;++j)hpbl[j]=0.;
double* dkt=ws.alloc<double>(km);for(int j=0;j<km;++j)dkt[j]=0.;
double* dku=ws.alloc<double>(km);for(int j=0;j<km;++j)dku[j]=0.;
double* del=ws.alloc<double>(km);for(int j=0;j<km;++j)del[j]=0;
double* usfco=ws.alloc<double>(1);for(int j=0;j<1;++j)usfco[j]=0;
double* vsfco=ws.alloc<double>(1);for(int j=0;j<1;++j)vsfco[j]=0;
double* dusfc=ws.alloc<double>(1);for(int j=0;j<1;++j)dusfc[j]=0;
double* dvsfc=ws.alloc<double>(1);for(int j=0;j<1;++j)dvsfc[j]=0;
double* dtsfc=ws.alloc<double>(1);for(int j=0;j<1;++j)dtsfc[j]=0;
double* dqsfc=ws.alloc<double>(1);for(int j=0;j<1;++j)dqsfc[j]=0;
int* kpbl=ws.alloc<int>(1);for(int j=0;j<1;++j)kpbl[j]=0;
int* kinver=ws.alloc<int>(1);for(int j=0;j<1;++j)kinver[j]=0;
int* dtidx=ws.alloc<int>(100+ntrac);for(int j=0;j<100+ntrac;++j)dtidx[j]=0;
double* dtend=ws.alloc<double>(ntrac*km);for(int j=0;j<ntrac*km;++j)dtend[j]=0;
kinver[0]=km;for(int j=0;j<km;++j)del[j]=sm_sub(prsi[j],prsi[j+1]);
char errmsg[1];int errflg=0;
sm_satmedmfvdifq_run(1,km,ntrac,2,3,4,ntrac,sm_g,sm_rd,sm_cp,sm_rv,sm_hvap,sm_hfus,sm_fvirt,sm_eps,sm_epsm1,dv,du,tdt,rtg,u1,v1,t1,q1,usfco,vsfco,0,swh,hlw,xmu,garea,zvfun,sigmaf,psk,rbsoil,zorl,u10m,v10m,fm,fh,tsea,heat,evap,stress,spd1,kpbl,prsi,del,prsl,prslk,phii,phil,dt,false,dusfc,dvsfc,dtsfc,dqsfc,hpbl,dkt,dku,kinver,1.,0.01,0.2,1.,0.13,0.13,300.,300.,1,0,1,dtend,dtidx,1,2,3,1,false,false,errmsg,errflg,ws);
offset=0;
for(int j=0;j<km;++j)output[(offset+j)*nc+column]=du[j];offset+=km;
for(int j=0;j<km;++j)output[(offset+j)*nc+column]=dv[j];offset+=km;
for(int j=0;j<km;++j)output[(offset+j)*nc+column]=tdt[j];offset+=km;
for(int j=0;j<ntrac*km;++j)output[(offset+j)*nc+column]=rtg[j];offset+=ntrac*km;
for(int j=0;j<1;++j)output[(offset+j)*nc+column]=hpbl[j];offset+=1;
for(int j=0;j<km;++j)output[(offset+j)*nc+column]=dkt[j];offset+=km;
for(int j=0;j<km;++j)output[(offset+j)*nc+column]=dku[j];offset+=km;
output[offset*nc+column]=dusfc[0];++offset;
output[offset*nc+column]=dvsfc[0];++offset;
output[offset*nc+column]=dtsfc[0];++offset;
output[offset*nc+column]=dqsfc[0];++offset;
output_kpbl[column]=kpbl[0];highwater[column]=ws.highwater;
}
#ifdef GPUWM_SATMEDMF_HOST
extern "C" void satmedmf_host(int nc,int km,int ntrac,double dt,const double* input,double* output,int* kpbl,const double* table,int* highwater){
std::vector<double> scratch((220+16*ntrac)*km+4096,0.);for(int c=0;c<nc;++c)sm_run_column(c,nc,km,ntrac,dt,input,output,kpbl,scratch.data(),table,highwater);}
#else
extern "C" __global__ void satmedmf_column(int nc,int km,int ntrac,double dt,const double* input,double* output,int* kpbl,double* scratch,const double* table,int* highwater){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;sm_run_column(c,nc,km,ntrac,dt,input,output,kpbl,scratch+((220+16*ntrac)*km+4096)*c,table,highwater);}
#endif

#ifndef GPUWM_SATMEDMF_HOST
// funcphys.f90:841-888 and :1044-1074. Generated with WOOF's math routines.
extern "C" __global__ void satmedmf_fpvs_table(double* table) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=7501)return;
 double xinc=sm_div(sm_sub(330.,180.),7500.);
 double t=sm_add(180.,sm_mul((double)j,xinc));
 double tr=sm_div(273.16,t);
 double dl=sm_sub(1846.,4185.5),di=sm_sub(1846.,2106.);
 double al=sm_div(-dl,sm_rv),ai=sm_div(-di,sm_rv);
 double bl=sm_add(al,sm_div(sm_hvap,sm_mul(sm_rv,273.16)));
 double bi=sm_add(ai,sm_div(sm_add(sm_hvap,sm_hfus),sm_mul(sm_rv,273.16)));
 double pvl=sm_mul(sm_mul(610.78,sm_pow(tr,al)),sm_exp(sm_mul(bl,sm_sub(1.,tr))));
 double pvi=sm_mul(sm_mul(610.78,sm_pow(tr,ai)),sm_exp(sm_mul(bi,sm_sub(1.,tr))));
 if(t>=273.16)table[j]=pvl;
 else if(t<sm_sub(273.16,20.))table[j]=pvi;
 else{
  double tice=sm_sub(273.16,20.);
  double w=sm_div(sm_sub(t,tice),sm_sub(273.16,tice));
  table[j]=sm_add(sm_mul(w,pvl),sm_mul(sm_sub(1.,w),pvi));
 }
}
#endif

#ifndef GPUWM_SATMEDMF_HOST
extern "C" __global__ void satmedmf_exner(const double* p,const double* pi,double* exner,double* psk,int nc,int km){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<nc*km)exner[i]=sm_pow(sm_div(p[i],100000.),sm_div(sm_rd,sm_cp));
 if(i<nc)psk[i]=sm_pow(sm_div(pi[i],100000.),sm_div(sm_rd,sm_cp));
}
#endif
