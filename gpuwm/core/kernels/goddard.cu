// Column entry point. The driver supplies top-down pressure in hPa.
// Inputs and outputs use column-contiguous storage: column + columns * level.
// Band optics use column + columns * (level + levels * band).
extern "C" __global__ void goddard_bands_columns(
 const float* input,const int* flags,const float* optics,
 float* output,int np,int columns) {
 int col=blockIdx.x*blockDim.x+threadIdx.x;
 if(col>=columns || np>GDD_MAX_LEVELS)return;
 GReal p[GDD_MAX_LEVELS+1],t[GDD_MAX_LEVELS],w[GDD_MAX_LEVELS],o[GDD_MAX_LEVELS],fc[GDD_MAX_LEVELS];
 GReal e[10],tc[GDD_MAX_LEVELS*11],sc[GDD_MAX_LEVELS*11],gc[GDD_MAX_LEVELS*11];
 GReal tl[GDD_MAX_LEVELS*10],sl[GDD_MAX_LEVELS*10],gl[GDD_MAX_LEVELS*10],zero[GDD_MAX_LEVELS*11];
 int at=col;
 for(int k=0;k<=np;++k)p[k]=GReal(input[at+k*columns]);at+=columns*(np+1);
 for(int k=0;k<np;++k)t[k]=GReal(input[at+k*columns]);at+=columns*np;
 for(int k=0;k<np;++k)w[k]=GReal(input[at+k*columns]);at+=columns*np;
 for(int k=0;k<np;++k)o[k]=GReal(input[at+k*columns]);at+=columns*np;
 for(int k=0;k<np;++k)fc[k]=GReal(input[at+k*columns]);at+=columns*np;
 GReal cosine=GReal(input[at]);at+=columns;
 GReal tb=GReal(input[at]);at+=columns;
 GReal ts=GReal(input[at]);at+=columns;
 for(int b=0;b<10;++b)e[b]=GReal(input[at+b*columns]);at+=10*columns;
 GReal albedo=GReal(input[at]);
 at=col;
 for(int k=0;k<np*11;++k)tc[k]=GReal(optics[at+k*columns]);at+=columns*np*11;
 for(int k=0;k<np*11;++k)sc[k]=GReal(optics[at+k*columns]);at+=columns*np*11;
 for(int k=0;k<np*11;++k)gc[k]=GReal(optics[at+k*columns]);at+=columns*np*11;
 for(int k=0;k<np*10;++k)tl[k]=GReal(optics[at+k*columns]);at+=columns*np*10;
 for(int k=0;k<np*10;++k)sl[k]=GReal(optics[at+k*columns]);at+=columns*np*10;
 for(int k=0;k<np*10;++k)gl[k]=GReal(optics[at+k*columns]);
 for(int k=0;k<np*11;++k)zero[k]=GReal(0);
 GArray<GReal,1> pp={p,{1},{np+1}},tt={t,{1},{np}},ww={w,{1},{np}},oo={o,{1},{np}},ff={fc,{1},{np}},ee={e,{1},{10}};
 GArray<GReal,2> tcs={tc,{1,1},{np,11}},scs={sc,{1,1},{np,11}},gcs={gc,{1,1},{np,11}},zs={zero,{1,1},{np,11}};
 GArray<GReal,2> tls={tl,{1,1},{np,10}},sls={sl,{1,1},{np,10}},gls={gl,{1,1},{np,10}},zl={zero,{1,1},{np,10}};
 float sw[GDD_MAX_LEVELS+1]={0},sd[GDD_MAX_LEVELS+1]={0},su[GDD_MAX_LEVELS+1]={0},sf[4]={0};
 float lw[GDD_MAX_LEVELS+1],ld[GDD_MAX_LEVELS+1],lu[GDD_MAX_LEVELS+1];
 GArray<float,1> swn={sw,{1},{np+1}},swd={sd,{1},{np+1}},swu={su,{1},{np+1}},sfc={sf,{1},{4}};
 GArray<float,1> lwn={lw,{1},{np+1}},lwd={ld,{1},{np+1}},lwu={lu,{1},{np+1}};
 int ict=flags[col],icb=flags[col+columns];
 if(flags[col+2*columns])g_swrad(np,cosine,pp,tt,ww,oo,ff,ict,icb,tcs,scs,gcs,zs,zs,zs,
  albedo,albedo,albedo,albedo,swn,swd,swu,sfc,true,1);
 g_lwrad(np,ee,tb,ts,ict,icb,pp,tt,ww,oo,ff,tls,sls,gls,zl,zl,zl,lwn,lwd,lwu,1);

#ifdef GDD_MUTATE_BAND
 if(col==0) {
  if(GDD_MUTATE_BAND==1) sw[np]=__uint_as_float(__float_as_uint(sw[np]) ^ 1u);
  if(GDD_MUTATE_BAND==2) lw[np]=__uint_as_float(__float_as_uint(lw[np]) ^ 1u);
 }
#endif
 at=col;
 for(int k=0;k<=np;++k)output[at+k*columns]=sw[k];at+=columns*(np+1);
 for(int k=0;k<=np;++k)output[at+k*columns]=sd[k];at+=columns*(np+1);
 for(int k=0;k<=np;++k)output[at+k*columns]=su[k];at+=columns*(np+1);
 for(int k=0;k<4;++k)output[at+k*columns]=sf[k];at+=columns*4;
 for(int k=0;k<=np;++k)output[at+k*columns]=lw[k];at+=columns*(np+1);
 for(int k=0;k<=np;++k)output[at+k*columns]=ld[k];at+=columns*(np+1);
 for(int k=0;k<=np;++k)output[at+k*columns]=lu[k];
}

__device__ const float gdd_radius_points[109]={1.0e-2f,1.5e-2f,2.0e-2f,2.5e-2f,3.0e-2f,3.5e-2f,4.0e-2f,4.5e-2f,5.0e-2f,5.5e-2f,6.0e-2f,6.5e-2f,7.0e-2f,7.5e-2f,8.0e-2f,8.5e-2f,9.0e-2f,9.5e-2f,1.0e-1f,1.5e-1f,2.0e-1f,2.5e-1f,3.0e-1f,3.5e-1f,4.0e-1f,4.5e-1f,5.0e-1f,5.5e-1f,6.0e-1f,6.5e-1f,7.0e-1f,7.5e-1f,8.0e-1f,8.5e-1f,9.0e-1f,9.5e-1f,1.0e-0f,1.5e-0f,2.0e-0f,2.5e-0f,3.0e-0f,3.5e-0f,4.0e-0f,4.5e-0f,5.0e-0f,5.5e-0f,6.0e-0f,6.5e-0f,7.0e-0f,7.5e-0f,8.0e-0f,8.5e-0f,9.0e-0f,9.5e-0f,1.0e+1f,1.5e+1f,2.0e+1f,2.5e+1f,3.0e+1f,3.5e+1f,4.0e+1f,4.5e+1f,5.0e+1f,5.5e+1f,6.0e+1f,6.5e+1f,7.0e+1f,7.5e+1f,8.0e+1f,8.5e+1f,9.0e+1f,9.5e+1f,1.0e+2f,1.5e+2f,2.0e+2f,2.5e+2f,3.0e+2f,3.5e+2f,4.0e+2f,4.5e+2f,5.0e+2f,5.5e+2f,6.0e+2f,6.5e+2f,7.0e+2f,7.5e+2f,8.0e+2f,8.5e+2f,9.0e+2f,9.5e+2f,1.0e+3f,1.5e+3f,2.0e+3f,2.5e+3f,3.0e+3f,3.5e+3f,4.0e+3f,4.5e+3f,5.0e+3f,5.5e+3f,6.0e+3f,6.5e+3f,7.0e+3f,7.5e+3f,8.0e+3f,8.5e+3f,9.0e+3f,9.5e+3f,1e+4f};

// WRF module_ra_goddard.F:12295-12415. Input q is g/m3 and re is micrometres.
// The Rust decoder supplies the original LUT words in little-endian order.
extern "C" __global__ void goddard_cloud_columns(
 const float *dz,const float *q,const float *re,const float *lut,
 float *output,int np,int columns) {
 int cell=blockIdx.x*blockDim.x+threadIdx.x;
 if(cell>=np*columns)return;
 int col=cell%columns,k=cell/columns;
 constexpr int species_map[6]={2,0,1,3,4,5};
 constexpr int record=109*6*11*2;
 for(int arm=0;arm<2;++arm) {
  int bands=arm==0?11:10;
  int base=arm==0?0:3*np*columns*11;
  for(int b=0;b<bands;++b) {
   float tau[6],ssa[6],asy[6];
   for(int t=0;t<6;++t) {
    int qi=col+columns*(k+np*t);
    float radius=re[qi];
    if(radius<gdd_radius_points[0])radius=gdd_radius_points[0];
    if(radius>gdd_radius_points[108])radius=gdd_radius_points[108];
    int ir=0;
    for(;ir<108;++ir)if(radius>=gdd_radius_points[ir]&&radius<=gdd_radius_points[ir+1])break;
    // A NaN radius matches no WRF interval. Its initialized species optics stay zero.
    if(ir==108){tau[t]=0.f;ssa[t]=0.f;asy[t]=0.f;continue;}
    float w2=(radius-gdd_radius_points[ir])/(gdd_radius_points[ir+1]-gdd_radius_points[ir]);
    float w1=1.0f-w2;
    int index=ir+109*(species_map[t]+6*(b+11*arm));
    tau[t]=((w1*lut[index]+w2*lut[index+1])*q[qi])*dz[cell]*1.e-3f;
    ssa[t]=w1*lut[record+index]+w2*lut[record+index+1];
    asy[t]=w1*lut[2*record+index]+w2*lut[2*record+index+1];
   }
   float total_tau=0.f,total_ssa=0.f,total_asy=0.f;
   for(int t=0;t<6;++t)total_tau+=tau[t];
   for(int t=0;t<6;++t)total_ssa+=tau[t]*ssa[t];
   total_ssa=total_ssa/(total_tau>1.e-8f?total_tau:1.e-8f);
   for(int t=0;t<6;++t)total_asy+=(tau[t]*ssa[t])*asy[t];
   float denom=total_tau*total_ssa;
   total_asy=total_asy/(denom>1.e-8f?denom:1.e-8f);
   int at=base+col+columns*(k+np*b);
   output[at]=total_tau;
   output[at+np*columns*bands]=total_ssa;
   output[at+2*np*columns*bands]=total_asy;
  }
 }
}

// WRF sounding_interp:2427-2505 and ozone_interp:2510-2581.
__device__ float gdd_ozone(float latitude,int julday,float pressure) {
 float summer,winter;
 if(julday<30){summer=__fdiv_rn(float(30-julday),183.f);winter=1.f-summer;}
 else if(julday<=212){summer=__fdiv_rn(float(julday-30),183.f);winter=1.f-summer;}
 else{winter=__fdiv_rn(float(julday-212),183.f);summer=1.f-winter;}
 if(latitude<0.f){summer=winter;winter=1.f-summer;}
 float lat=g_abs(latitude);
 // The equator uses the endpoint sounding. WRF otherwise indexes past mcdat.
 int north=2*int(__fdiv_rn((90.f-lat),22.5f));if(north>6)north=6;
 int south=north+2;
 float lat_north=90.f-float(north/2)*22.5f;
 float wn=__fdiv_rn((lat-(lat_north-22.5f)),22.5f),ws=1.f-wn;
 float sounding[33][3]; constexpr int field[3]={1,4,5};
 for(int k=0;k<33;++k)for(int f=0;f<3;++f){
  float a=summer*gdd_sounding[north][k][field[f]]+winter*gdd_sounding[north+1][k][field[f]];
  float b=summer*gdd_sounding[south][k][field[f]]+winter*gdd_sounding[south+1][k][field[f]];
  sounding[k][f]=wn*a+ws*b;
 }
 float pa=pressure*100.f,o,r;
 if(pa<=sounding[32][0]){o=sounding[32][1];r=sounding[32][2];}
 else if(pa>=sounding[0][0]){o=sounding[0][1];r=sounding[0][2];}
 else{
  o=0.f;r=1.f;
  for(int kk=31;kk>=0;--kk)if(pa>=sounding[kk+1][0]&&pa<=sounding[kk][0]){
   float w=(pa-sounding[kk+1][0])/(sounding[kk][0]-sounding[kk+1][0]);
   o=sounding[kk][1]*w+sounding[kk+1][1]*(1.f-w);
   r=sounding[kk][2]*w+sounding[kk+1][2]*(1.f-w);break;
  }
 }
 return o/r;
}

// Raw bottom-up grids: pw,t,p,qv,fc,dz,rho,q[6],re[6],
// lat,cosz,tb,ts,emiss,albedo,pi. band and cloud output are top-down.
extern "C" __global__ void goddard_prepare_columns(const float* raw,
 float* band,int* flags,float* dz,float* q,float* re,int np,int columns,int julday){
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=columns)return;
 int n=np*columns,base=(np+1)*columns,surface=base+18*n;
 int ict=1,icb=1;float p400=1.e5f,p700=1.e5f;
 for(int k=0;k<=np;++k)band[col+k*columns]=(k==0?0.f:raw[col+(np-k)*columns]*.01f);
 for(int k=0;k<np;++k){
  int src=col+(np-1-k)*columns,dst=col+k*columns;
  float pm=raw[base+n+src]*.01f;
  band[base+dst]=raw[base+src];
  float vapor=g_max(0.f,raw[base+2*n+src]);
  band[base+n+dst]=g_max(0.f,vapor/(1.f+vapor));
  band[base+2*n+dst]=gdd_ozone(raw[surface+col],julday,pm);
  band[base+3*n+dst]=raw[base+3*n+src];
  dz[dst]=raw[base+4*n+src];
  for(int s=0;s<6;++s){
   q[dst+s*n]=(g_max(0.f,raw[base+(6+s)*n+src])*raw[base+5*n+src])*1.e3f;
   re[dst+s*n]=raw[base+(12+s)*n+src];
  }
  if(g_abs(pm-400.f)<p400){p400=g_abs(pm-400.f);ict=k+1;}
  if(g_abs(pm-700.f)<p700){p700=g_abs(pm-700.f);icb=k+1;}
 }
 int bs=base+4*n;
 for(int s=0;s<3;++s)band[bs+s*columns+col]=raw[surface+(s+1)*columns+col];
 for(int b=0;b<10;++b)band[bs+(3+b)*columns+col]=raw[surface+4*columns+col];
 band[bs+13*columns+col]=raw[surface+5*columns+col];
 flags[col]=ict;flags[col+columns]=icb;
 flags[col+2*columns]=raw[surface+columns+col]>1.e-4f;
}

// WRF goddardrad:2091-2230,2320-2408, radiation_driver:2265,2852.
extern "C" __global__ void goddard_publish_columns(const float* band,
 const float* flux,const float* raw,float* output,int np,int columns,float solcon){
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=columns)return;
 int n=np*columns,iface=(np+1)*columns,base=iface,surface=base+18*n;
 float cosine=band[iface+4*n+col],albedo=raw[surface+5*columns+col];
 bool day=cosine>1.e-4f;
 int lw=3*iface+4*columns;
 float fac=(.01f*9.81f)/1004.5f;
 for(int k=0;k<np;++k){
  int at=col+k*columns;
  float a=((flux[at]*solcon)*cosine),b=((flux[at+columns]*solcon)*cosine);
  float dp=band[at]-band[at+columns];
  float sw=(-fac*(a-b))/dp;if(isnan(sw)||sw<0.f||!day)sw=0.f;
  float lwh=(-fac*(flux[lw+at]-flux[lw+at+columns]))/dp;
  int dst=col+(np-1-k)*columns;
  float pi=raw[surface+6*columns+dst];
  output[dst]=lwh/pi;
  // WRF subtracts held LW from the accumulated total theta heating.
  output[n+dst]=(output[dst]+sw/pi)-output[dst];
 }
 int out=2*n+col;
 float sd=day?((flux[iface+np*columns+col]*solcon)*cosine):0.f;
 float glw=flux[lw+iface+np*columns+col];
 output[out]=sd;output[out+columns]=glw;
 output[out+2*columns]=day?(((1.f-albedo)*flux[iface+np*columns+col])*solcon)*cosine:0.f;
 output[out]=output[out+2*columns]/(1.f-albedo);
 output[out+3*columns]=-flux[lw+2*iface+col];
 float dni=day?(flux[3*iface+col]+flux[3*iface+2*columns+col])*solcon:0.f;
 output[out+4*columns]=dni;output[out+5*columns]=day?dni*cosine:0.f;
 output[out+6*columns]=day?((flux[3*iface+columns+col]+flux[3*iface+3*columns+col])*solcon)*cosine:0.f;
 output[out+7*columns]=flux[lw+iface+col];output[out+8*columns]=flux[lw+2*iface+col];
 output[out+9*columns]=glw;output[out+10*columns]=flux[lw+2*iface+np*columns+col];
 output[out+11*columns]=day?(flux[iface+col]*solcon)*cosine:0.f;
 output[out+12*columns]=day?(flux[2*iface+col]*solcon)*cosine:0.f;
 output[out+13*columns]=sd;
 output[out+14*columns]=day?(flux[2*iface+np*columns+col]*solcon)*cosine:0.f;
}

// Shared WRF phy_prep temperature interpolation, bottom-up grid order.
extern "C" __global__ void goddard_interface_temperature(const float* t,const float* z,
 const float* fnm,const float* fnp,float* tw,int nz,int columns){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(nz+1)*columns)return;
 int col=i%columns,k=i/columns;
 if(k>0&&k<nz)tw[i]=fnm[k]*t[col+k*columns]+fnp[k]*t[col+(k-1)*columns];
 else if(k==0){
  float z0=.5f*(z[col]+z[col+columns]),z1=.5f*(z[col+columns]+z[col+2*columns]);
  float w1=(z[col]-z1)/(z0-z1),w2=1.f-w1;
  tw[i]=w1*t[col]+w2*t[col+columns];
 }else{
  float za=.5f*(z[col+(nz-2)*columns]+z[col+(nz-1)*columns]);
  float zb=.5f*(z[col+(nz-1)*columns]+z[col+nz*columns]);
  float w1=(z[col+nz*columns]-za)/(zb-za),w2=1.f-w1;
  tw[i]=w1*t[col+(nz-1)*columns]+w2*t[col+(nz-2)*columns];
 }
}
