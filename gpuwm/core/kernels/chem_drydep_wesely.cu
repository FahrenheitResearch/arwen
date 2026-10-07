// Float32 storage and operations with explicit rounding and no FTZ modifier.
// chem/module_dep_simple.F:223-257,946-1287,1495-1614 uses scalar REAL(4).
// CuPy cuda/compiler.py appends -ftz=true, overriding a user --ftz=false.
// Plain float is used only as the storage word; asm controls every operation.
struct WFloat {
    float v;
    __device__ WFloat() {}
    __device__ WFloat(float value):v(value) {}
};
static_assert(sizeof(WFloat)==sizeof(float), "float32 table layout");
__device__ WFloat operator+(WFloat a,WFloat b) {
    float v;asm("add.rn.f32 %0,%1,%2;":"=f"(v):"f"(a.v),"f"(b.v));return WFloat(v);
}
__device__ WFloat operator+(float a,WFloat b) {return WFloat(a)+b;}
__device__ WFloat operator+(WFloat a,float b) {return a+WFloat(b);}
__device__ WFloat operator-(WFloat a,WFloat b) {
    float v;asm("sub.rn.f32 %0,%1,%2;":"=f"(v):"f"(a.v),"f"(b.v));return WFloat(v);
}
__device__ WFloat operator-(float a,WFloat b) {return WFloat(a)-b;}
__device__ WFloat operator-(WFloat a,float b) {return a-WFloat(b);}
__device__ WFloat operator*(WFloat a,WFloat b) {
    float v;asm("mul.rn.f32 %0,%1,%2;":"=f"(v):"f"(a.v),"f"(b.v));return WFloat(v);
}
__device__ WFloat operator*(float a,WFloat b) {return WFloat(a)*b;}
__device__ WFloat operator*(WFloat a,float b) {return a*WFloat(b);}
__device__ WFloat operator/(WFloat a,WFloat b) {
    float v;asm("div.rn.f32 %0,%1,%2;":"=f"(v):"f"(a.v),"f"(b.v));return WFloat(v);
}
__device__ WFloat operator/(float a,WFloat b) {return WFloat(a)/b;}
__device__ WFloat operator/(WFloat a,float b) {return a/WFloat(b);}
// Named entry for constant denominators, using the native operation above.
__device__ WFloat wd_div(WFloat a,WFloat b) {return a/b;}
__device__ bool operator<(WFloat a,WFloat b) {
    unsigned value;
    asm("{ .reg .pred p; setp.lt.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
        :"=r"(value):"f"(a.v),"f"(b.v));return value!=0;
}
__device__ bool operator<(float a,WFloat b) {return WFloat(a)<b;}
__device__ bool operator<(WFloat a,float b) {return a<WFloat(b);}
__device__ bool operator>(WFloat a,WFloat b) {
    unsigned value;
    asm("{ .reg .pred p; setp.gt.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
        :"=r"(value):"f"(a.v),"f"(b.v));return value!=0;
}
__device__ bool operator>(float a,WFloat b) {return WFloat(a)>b;}
__device__ bool operator>(WFloat a,float b) {return a>WFloat(b);}
__device__ bool operator<=(WFloat a,WFloat b) {
    unsigned value;
    asm("{ .reg .pred p; setp.le.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
        :"=r"(value):"f"(a.v),"f"(b.v));return value!=0;
}
__device__ bool operator<=(float a,WFloat b) {return WFloat(a)<=b;}
__device__ bool operator<=(WFloat a,float b) {return a<=WFloat(b);}
__device__ bool operator>=(WFloat a,WFloat b) {
    unsigned value;
    asm("{ .reg .pred p; setp.ge.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
        :"=r"(value):"f"(a.v),"f"(b.v));return value!=0;
}
__device__ bool operator>=(float a,WFloat b) {return WFloat(a)>=b;}
__device__ bool operator>=(WFloat a,float b) {return a>=WFloat(b);}
__device__ bool operator==(WFloat a,WFloat b) {
    unsigned value;
    asm("{ .reg .pred p; setp.eq.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
        :"=r"(value):"f"(a.v),"f"(b.v));return value!=0;
}
__device__ bool operator==(float a,WFloat b) {return WFloat(a)==b;}
__device__ bool operator==(WFloat a,float b) {return a==WFloat(b);}
__device__ bool operator!=(WFloat a,WFloat b) {
    unsigned value;
    asm("{ .reg .pred p; setp.ne.f32 p,%1,%2; selp.u32 %0,1,0,p; }"
        :"=r"(value):"f"(a.v),"f"(b.v));return value!=0;
}
__device__ bool operator!=(float a,WFloat b) {return WFloat(a)!=b;}
__device__ bool operator!=(WFloat a,float b) {return a!=WFloat(b);}
__device__ WFloat operator-(WFloat a) {
    return WFloat(__uint_as_float(__float_as_uint(a.v)^0x80000000u));
}
__device__ WFloat wd_abs(WFloat a) {
    return WFloat(__uint_as_float(__float_as_uint(a.v)&0x7fffffffu));
}
__device__ WFloat wd_min(WFloat a,WFloat b) {return a<b?a:b;}
__device__ WFloat wd_max(WFloat a,WFloat b) {return a>b?a:b;}
__device__ WFloat wd_exp(WFloat a) {return WFloat(gfk_exp(a.v));}
__device__ WFloat wd_log(WFloat a) {return WFloat(gfk_log(a.v));}
__device__ WFloat wd_pow(WFloat a,WFloat b) {return WFloat(gfk_pow(a.v,b.v));}
__device__ WFloat wd_sqrt(WFloat a) {
    return WFloat(FSQRT(a.v));
}

// Gas path from chem/module_dep_simple.F:178-257,888-1616 (WRF v4.7.1).
// One thread per column. No FMA contraction. Float32 operators preserve subnormals.
// glibc_flt32.cuh is prepended by the shared kernel loader.
__device__ WFloat wesely_polint(WFloat mol, WFloat zr, WFloat z0) {
    // chem/module_dep_simple.F:1592-1603. Integer powers are multiplies.
    if (mol < 0.f) {
        WFloat ar = wd_pow(1.f-9.f*zr*mol,0.25f)+0.001f;
        ar = ar*ar;
        WFloat ao = wd_pow(1.f-9.f*z0*mol,0.25f)+0.001f;
        ao = ao*ao;
        return 0.74f*(wd_log((ar-1.f)/(ar+1.f))-wd_log((ao-1.f)/(ao+1.f)));
    }
    WFloat value = 0.74f*wd_log(zr/z0);
    if (mol > 0.f) value = value+4.7f*mol*(zr-z0);
    return value;
}
// chem/module_dep_simple.F:1314-1341. Particle/fog outputs are unused
// by gas rows; retained here for the source subtree and later aerosol use.
__device__ void wesely_deppart(WFloat mol,WFloat ustar,WFloat rh,WFloat clw,
                              WFloat kpart,int lu,WFloat& dvpart,WFloat& dvfog) {
    dvpart=ustar/kpart;
    if(mol<0.f)dvpart=dvpart*(1.f+wd_pow(-300.f*mol,0.66667f));
    if(rh>80.f)dvpart=dvpart*(1.f+0.37f*wd_exp(wd_div(rh-80.f,WFloat(20.f))));
    dvfog=0.06f*clw;
    if(lu==5)dvfog=dvfog+0.195f*ustar*ustar;
}
extern "C" __global__ void chem_wesely_ddvel(
 const WFloat* t_phy_k1,const WFloat* p_phy_k1,const WFloat* p8w_k1,
 const WFloat* qv_k1,const WFloat* qc_k1,const WFloat* qr_k1,const WFloat* dz1,
 const WFloat* tsk,const WFloat* gsw,const WFloat* vegfra,const WFloat* ust,
 const WFloat* rmol,const WFloat* znt,const WFloat* raincv,const int* ivgtyp,
 const WFloat* params,const int* arms,const WFloat* diffusion,
 const WFloat* tables,const int* classes,const int* landmap,
 const WFloat* chem_k1,int julday,int nrow,int nh3_row,int so2_row,int ncol,
 WFloat* ddvel,WFloat* aer_res_def,WFloat* aer_res_zcen,int water,int ice) {
 int col=blockIdx.x*blockDim.x+threadIdx.x;
 if(col>=ncol)return;
 // Host validates land indices. Tables are seven Fortran-order blocks of 125.
 int land=landmap[ivgtyp[col]-1];
 int s=(julday<90 || julday>270)?1:0;
 int ix=(land-1)+25*s;
 WFloat ri=tables[ix],rlu=tables[125+ix],rac=tables[250+ix];
 WFloat rgss=tables[375+ix],rgso=tables[500+ix];
 WFloat rcls=tables[625+ix],rclo=tables[750+ix];
 int lu=classes[land-1];
 WFloat t=tsk[col],rad=gsw[col],tc=t-273.15f;
 WFloat pa=0.01f*p_phy_k1[col];
 // chem/module_dep_simple.F:223-226, all associations retained.
 WFloat rh=wd_min(100.f,100.f*qv_k1[col]/
   (3.80f*wd_exp(17.27f*(t_phy_k1[col]-273.f)/(t_phy_k1[col]-36.f))/pa));
 rh=wd_max(5.f,rh);
 bool wet=rh>=95.f,rain=qr_k1[col]>1.e-18f || raincv[col]>0.f;
 bool high=false;
 if(nh3_row>=0 && so2_row>=0)
   high=chem_k1[nh3_row*ncol+col]>2.f*chem_k1[so2_row*ncol+col];
 // chem/module_dep_simple.F:946-964.
 WFloat z=200.f/(rad+0.1f),rs=9999.f;
 if(tc>0.f && tc<40.f)rs=ri*(1.f+z*z)*(400.f/(tc*(40.f-tc)));
 WFloat rdc=100.f*(1.f+1000.f/(rad+10.f))/(1.f+1000.f*0.f);
 WFloat rluo1=1.f/(1.f/3000.f+3.f/rlu);
 WFloat rluo2=1.f/(1.f/1000.f+3.f/rlu);
 WFloat resice=1000.f*wd_exp(-(tc+4.f));
 WFloat wrk=(298.f-t)/(298.f*t);
 // chem/module_dep_simple.F:1415-1430,1592: local snap, no input mutation.
 // KARMAN=0.4: share/module_model_constants.F:82.
 WFloat dvpart,dvfog;
 wesely_deppart(rmol[col],ust[col],rh,qc_k1[col],tables[875+land-1],lu,dvpart,dvfog);
 WFloat mol=rmol[col];
 if(wd_abs(mol)<1.e-6f)mol=0.f;
 WFloat pol=wesely_polint(mol,2.f,znt[col]);
 WFloat cen=wesely_polint(mol,dz1[col]*0.5f,znt[col]);
 if(aer_res_def)aer_res_def[col]=pol/(0.4f*wd_max(ust[col],1.e-4f));
 if(aer_res_zcen)aer_res_zcen[col]=cen/(0.4f*wd_max(ust[col],1.e-4f));
 // chem/module_dep_simple.F:1495-1507, common cell-average factor.
 WFloat dz=dz1[col],a;
 if(mol<0.f){
   WFloat pdz=wd_sqrt(1.f-9.f*dz*mol),pzr=wd_sqrt(1.f-9.f*2.f*mol);
   WFloat fac=((pdz-1.f)/(pzr-1.f))*((pzr+1.f)/(pdz+1.f));
   a=0.74f*dz*wd_log(fac)+(0.164f/mol)*(pdz-pzr);
 }else{
   a=0.74f*(dz*wd_log(dz/2.f)-dz+2.f);
   if(mol>0.f){WFloat d=dz-2.f;a=a+(2.35f*mol)*(d*d);}
 }
 for(int row=0;row<nrow;++row){
   WFloat hs=params[4*row],dhr=params[4*row+1];
   WFloat f0=params[4*row+2],ratio=params[4*row+3];
   WFloat rc=1.f,rsmx,rlux,rclx,rgsx;
   int arm=arms[row];
   // chem/module_dep_simple.F:965-1038, general gas arm.
   if(hs!=0.f){
     WFloat hy=hs*wd_exp(dhr*wrk);
     WFloat rmx=1.f/(wd_div(hy,WFloat(3000.f))+100.f*f0);
     rsmx=rs*ratio+rmx;
     rclx=1.f/(1.e-5f*hy/rcls+f0/rclo)+resice;
     rgsx=1.f/(1.e-5f*hy/rgss+f0/rgso)+resice;
     rlux=rlu/(1.e-5f*hy+f0)+resice;
     if(wet)rlux=1.f/(1.f/(3.f*rlu)+1.e-7f*hy+f0/rluo1);
     if(rain)rlux=1.f/(1.f/(3.f*rlu)+1.e-7f*hy+f0/rluo2);
     rc=1.f/(1.f/rsmx+1.f/rlux+1.f/(rdc+rclx)+1.f/(rac+rgsx));
     rc=wd_max(1.f,rc);
   }
   // chem/module_dep_simple.F:1044-1057. Multiplication before division.
   if(arm==1){
     WFloat hy=hs*wd_exp(dhr*(298.f-t)/(298.f*t));
     WFloat rmx=1.f/(wd_div(hy,WFloat(3000.f))+100.f*f0);
     rsmx=rs*ratio+rmx;
     rlux=rlu/(1.e-5f*hy+f0)+resice;
     rclx=rclo+resice;rgsx=rgso+resice;
     if(wet)rlux=rluo1;
     if(rain)rlux=rluo2;
     rc=1.f/(1.f/rsmx+1.f/rlux+1.f/(rdc+rclx)+1.f/(wd_min(100.f,rac)+rgsx));
     rc=wd_max(1.f,rc);
   }
   // chem/module_dep_simple.F:1081-1150.
   if(arm==2){
     rsmx=rs*ratio;
     if(tc>-1.f){
       if(rh<81.3f)rlux=25000.f*wd_exp(-0.0693f*rh);
       else rlux=0.58e12f*wd_exp(-0.278f*rh);
     }
     if((wet||rain)&&tc>-1.f)rlux=1.f;
     if(tc>=-5.f && tc<=-1.f)rlux=200.f;
     if(tc<-5.f)rlux=500.f;
     rclx=rcls;rgsx=1000.f;
     if(wet||rain)rgsx=high?0.f:500.f;
     if(land==water)rgsx=0.f;
     if(land==ice){
       if(tc>2.f)rgsx=0.f;
       else if(tc>=-1.f && tc<=2.f)rgsx=70.f*(2.f-tc);
       else if(tc<-1.f)rgsx=500.f;
     }
     if(lu!=1 && land!=water && land!=ice)
       rc=1.f/(1.f/rsmx+1.f/rlux+1.f/(rclx+rdc+rgsx));
     else rc=rgsx;
     rc=wd_max(1.f,rc);
   }
   // chem/module_dep_simple.F:1154-1287. Non-MOZART seasons are 1 or 2.
   if(arm==3){
     rsmx=rs*ratio;
     if(lu==3){
       if(s==0)rc=1000.f;
       else{
         if(tc>-1.f){rc=rad!=0.f?50.f:100.f;if(wet||rain)rc=20.f;}
         if(tc>=-5.f && tc<=-1.f)rc=200.f;
         if(tc<-5.f)rc=500.f;
       }
     }
     if(lu==2){
       if(s==0){rc=rad!=0.f?rsmx:200.f;if(wet||rain)rc=50.f;}
       else{
         if(tc>-1.f){rc=rad!=0.f?rsmx:300.f;if(wet||rain)rc=100.f;}
         if(tc>=-5.f && tc<=-1.f)rc=200.f;
         if(tc<-5.f)rc=500.f;
       }
     }
     if(lu==4 || lu==5 || lu==6){
       rc=rad!=0.f?500.f:1000.f;
       if(wet||rain)rc=high?100.f:0.f;
       if(s==1){if(tc>=-5.f && tc<=-1.f)rc=200.f;if(tc<-5.f)rc=500.f;}
     }
     if(land==water)rc=0.f;
     if(lu==1)rc=wet?0.f:50.f;
     if(land==ice){
       if(tc>2.f)rc=0.f;
       if(tc>=-1.f && tc<=2.f)rc=70.f*(2.f-tc);
       if(tc<-1.f)rc=500.f;
     }
     rc=wd_max(1.f,rc);
   }
   // depvel, chem/module_dep_simple.F:1611, with scpr23 as dep_init derived
   // it (:3478-3489); the row carries the Fortran's float32 value.
   WFloat scpr=diffusion[row];
   WFloat v=ust[col]*0.4f/(2.f*scpr+pol);
   v=1.f/(1.f/v+rc);
   ddvel[row*ncol+col]=v/(1.f+v*a/(0.4f*ust[col]*(dz-2.f)));
 }
}
