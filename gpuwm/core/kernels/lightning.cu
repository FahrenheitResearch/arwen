// WRF v4.6.1 module_lightning_driver and module_ltng_*.
// Float32 operations in source order, using WOOF's own float32 math.
// A single thread defines patch/domain reduction order. No reduction atomics.
extern "C" __global__ void lightning_patch(
 const float* refl,const float* w,const float* t,const float* z,
 const float* ht,const float* land,int nz,int ny,int nx,int option,
 int i0,int i1,int j0,int j1,float* summary) {
 if(blockIdx.x || threadIdx.x) return;
 int n=ny*nx,kmax=0,ktop=0; float maxcount=0.f,wmax=__int_as_float(0xff800000u);
 for(int k=0;k<nz;k++) {
   float count=0.f;
   for(int j=j0;j<j1;j++) for(int i=i0;i<i1;i++) {
     int p=k*n+j*nx+i;
     if(refl[p]>20.f) count=count+1.f;
     if(w[p]>wmax) wmax=w[p];
   }
   if(count>0.f) ktop=k;
   if(count>maxcount) {maxcount=count;kmax=k;}
 }
 float total=0.f;
 if(maxcount!=0.f) {
   if(option==1) total=5.7e-6f*gfk_pow(wmax,4.5f);
   else {
     float zmax=0.f,mostlyland=0.f; int count=0;
     // WRF's loop is I then J here, unlike countCells' J,K,I.
     for(int i=i0;i<i1;i++) for(int j=j0;j<j1;j++) {
       int c=j*nx+i,p=ktop*n+c;
       if(refl[p]>20.f && t[p]<273.15f) {
         if(z[p]-ht[c]>zmax) zmax=z[p]-ht[c];
         count=count+1;mostlyland=mostlyland+land[c];
       }
     }
     mostlyland=mostlyland/(float)count;
     zmax=zmax*1.e-3f;
     total=(mostlyland<1.5f ? 3.44e-5f:6.57e-6f)*gfk_pow(zmax,4.9f);
   }
   total=total/maxcount/60.f;
 }
 summary[0]=total;summary[1]=(float)kmax;
}
extern "C" __global__ void lightning_columns(
 const float* refl,const float* w,const float* t,const float* z,
 const float* dz,const float* theta,const float* pi,
 const float* qc,const float* qr,const float* qi,const float* qs,const float* qg,
 const float* ht,const float* land,const float* lat,const float* lon,
 const int* top,const float* inputnum,const float* inputden,const float* summary,
 int nz,int ny,int nx,int option,int method,int month,
 float dx,float dy,float adjust,float num,float den,float factor,float interval,
 int i0,int i1,int j0,int j1,
 float* ic,float* cg,float* icc,float* cgc,float* lpi,int* invalid) {
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=ny*nx;
 if(c>=n) return;
 int j=c/nx,i=c%nx;
 if(i<i0||i>=i1||j<j0||j>=j1) return;
 if(option==3) {
   // calclpi builds cumulative heights, then calc_lpi subtracts them.
   float height=0.f,ave=0.f,totaldz=0.f;
   for(int k=0;k<nz-1;k++) {
     int p=k*n+c;
     float oldheight=height;height=height+dz[p];
     float delz=height-oldheight;
     float temperature=theta[p]*pi[p]-273.16f;
     if(temperature<=0.f && temperature>-20.f) {
       float deni=qi[p]+qg[p],dens=qs[p]+qg[p],ni=0.f,ns=0.f;
       if(qs[p]<=1.e-10f||qg[p]<=1.e-10f) dens=10000.f;
       else ns=sqrtf(qs[p]*qg[p]);
       if(qi[p]<=1.e-10f||qg[p]<=1.e-10f) deni=10000.f;
       else ni=sqrtf(qi[p]*qg[p]);
       float ice=qg[p]*(ni/deni+ns/dens);
       float liquid=qc[p]+qr[p],a=0.f,b=10000.f;
       if(liquid>1.e-10f && ice!=0.f) {a=sqrtf(liquid*ice);b=liquid+ice;}
       float lower=w[p],upper=(k<nz-2 ? w[p+n]:0.f);
       float wave=0.5f*(upper+lower);
       totaldz=totaldz+delz;
       if(a>0.f) ave=ave+((delz*(2.f*a/b))*(wave*wave));
     }
   }
   if(totaldz==0.f) totaldz=100000.f;
   lpi[c]=ave/totaldz;
   return; // WRF LPI does not update any flash field.
 }
 float total=0.f; int kt=0;
 if(option==11) {
   kt=top[c]-1; // WRF cloud top uses one-based levels.
   if(kt < -1 || kt>=nz) {atomicExch(invalid,1);return;}
   if(kt>=0 && t[kt*n+c]<273.15f) {
     float zkm=(z[kt*n+c]-ht[c])/1.e3f+adjust;
     if(zkm>0.f) total=(land[c]<1.5f ? 3.44e-5f:6.57e-6f)*gfk_pow(zkm,4.9f)/60.f;
   }
   float area=dx*dy/1.e6f;
   total=total*area/1296.f;
 } else {
   kt=(int)summary[1];
   if(refl[kt*n+c]>20.f) total=summary[0];
 }
 float ir=0.f,cr=0.f;
 if(method==1) {
   if(den==0.f) ir=total;
   else if(num==0.f) cr=total;
   else if(total!=0.f) {float ratio=num/den;cr=total*(1.f/(ratio+1.f));ir=total-cr;}
 } else if(method==0 || method==2) {
   if(total>0.f) {
     if(lat[c]<25.f||lat[c]>55.f||lon[c]<-120.f||lon[c]>-70.f) {
       if(den!=0.f) {float ratio=num/den;cr=total*(1.f/(ratio+1.f));}
     } else cr=total*((lon[c]>-90.f||lon[c]<-105.f) ? .4f:.22f);
     ir=total-cr;
   }
 } else if(method==3) {
   if(total>0.f) {
     if(option!=11) {
       kt=nz-1;
       while(refl[kt*n+c]<20.f && kt>0) kt--;
     }
     int kfreeze=kt;
     while(t[kfreeze*n+c]<273.15f && kfreeze>0) kfreeze--;
     if(option==11 && t[kfreeze*n+c]<273.15f) {atomicExch(invalid,2);return;}
     float depth=(z[kt*n+c]-z[kfreeze*n+c])*1.e-3f;
     if(option==11) depth=depth+adjust;
     depth=fmaxf(5.5f,fminf(14.f,depth));
     float ratio=(((.021f*depth-.648f)*depth+7.493f)*depth-36.54f)*depth+63.09f;
     cr=total*(1.f/(ratio+1.f));ir=total-cr;
   }
 } else {
   int p=(month-1)*n+c;
   // WRF method 4 overwrites its nominal 0/0 fallback with all CG.
   if(inputden[p]==0.f) cr=total;
   else {
     float ratio=inputnum[p]/inputden[p];
     if(ratio==-1.f) {atomicExch(invalid,3);return;}
     cr=total*(1.f/(ratio+1.f));
   }
   ir=total-cr;
 }
 ir=ir*factor;cr=cr*factor;ic[c]=ir;cg[c]=cr;
 icc[c]=icc[c]+ir*interval;cgc[c]=cgc[c]+cr*interval;
}
