// WRF 4.8 column and existing Noah / Noah-MP handover.
struct Ucm480Col : UcmCol {
float tvg,tt,xxxvg,tvgl[4],smg[4],cmcg,flxhumvg,flxhumt;
float declin,cosz,xlat,ssgd,ssgq; bool lsolar;
};
__device__ int ucm480_urban(const float* tab,const float* glob,const int* sw,Ucm480Col& c) {
const U480Env env{tab,glob,sw};int status=0;const int layers=4;
float zero=0.0f;float lf[4]={};
u480_urban(c.lsolar,layers,layers,layers,glob+0,glob+4,glob+8,c.utype,c.ta,c.qa,c.ua,c.u1,c.v1,c.ssg,c.ssgd,c.ssgq,c.llg,c.rain,c.rhoo,c.za,c.declin,c.cosz,c.omg,c.xlat,c.delt,c.znt,c.chs,c.chs2,c.tr,c.tb,c.tg,c.tc,c.qc,c.uc,c.trl,c.tbl,c.tgl,c.xxxr,c.xxxb,c.xxxg,c.xxxc,c.ts,c.qs,c.sh,c.lh,c.lh_kin,c.sw,c.alb,c.lw,c.g,c.rn,c.psim,c.psih,c.gz1oz0,c.cmr,c.chr,c.cmc,c.chc,c.u10,c.v10,c.th2,c.q2,c.ust,zero,zero,lf,zero,zero,zero,zero,zero,c.cmcr,c.tgr,c.tgrl,c.smr,c.cmgr,c.chgr,c.jmonth,c.drelr,c.drelb,c.drelg,c.flxhumr,c.flxhumb,c.flxhumg,c.tvg,c.tt,c.xxxvg,c.tvgl,c.smg,c.cmcg,c.flxhumvg,c.flxhumt,zero,zero,zero,env,status);
return status;
}
__device__ void ucm480_load_state(const unsigned long long* sp,size_t idx,size_t plane,Ucm480Col& c) {
ucm_load_state(sp,idx,plane,c);
c.tvg=PLANE(sp,43)[idx];
c.tt=PLANE(sp,44)[idx];
c.xxxvg=PLANE(sp,45)[idx];
for(int k=0;k<4;++k) c.tvgl[k]=PLANE(sp,46)[k*plane+idx];
for(int k=0;k<4;++k) c.smg[k]=PLANE(sp,47)[k*plane+idx];
c.cmcg=PLANE(sp,48)[idx];
c.flxhumvg=PLANE(sp,49)[idx];
c.flxhumt=PLANE(sp,50)[idx];
}
__device__ void ucm480_store_state(const unsigned long long* sp,size_t idx,size_t plane,Ucm480Col& c) {
ucm_store_state(sp,idx,plane,c);
PLANE(sp,43)[idx]=c.tvg;
PLANE(sp,44)[idx]=c.tt;
PLANE(sp,45)[idx]=c.xxxvg;
for(int k=0;k<4;++k) PLANE(sp,46)[k*plane+idx]=c.tvgl[k];
for(int k=0;k<4;++k) PLANE(sp,47)[k*plane+idx]=c.smg[k];
PLANE(sp,48)[idx]=c.cmcg;
PLANE(sp,49)[idx]=c.flxhumvg;
PLANE(sp,50)[idx]=c.flxhumt;
}
extern "C" __global__ void ucm480_noah_after_lsm(
    const int* __restrict__ urban, const int* __restrict__ utype,
    const float* __restrict__ frc, const float* __restrict__ u1,
    const float* __restrict__ v1, const float* __restrict__ hrang,const float* __restrict__ cosz,const float* __restrict__ xlat,float declin,
    const unsigned long long* __restrict__ rural,
    const unsigned long long* __restrict__ fields,
    const unsigned long long* __restrict__ state,
    const float* __restrict__ tab, const float* __restrict__ glob,
    const int* __restrict__ isw, float dt, int jmonth, unsigned int* err,
    int ny, int nx)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = (size_t)ny * (size_t)nx;
    if (idx >= plane || !urban[idx]) return;
    float* chs = PLANE(fields, GF_CHS);
    float* chs2 = PLANE(fields, GF_CHS2);
    float* cqs2 = PLANE(fields, GF_CQS2);
    Ucm480Col c{};
    c.utype = utype[idx];
    c.jmonth = jmonth;
    c.ta = PLANE(rural, RH_SFCTMP)[idx];
    c.qa = PLANE(rural, RH_Q2K)[idx];
    c.ua = ucm_wind(u1[idx], v1[idx]);
    c.u1 = u1[idx];
    c.v1 = v1[idx];
    float soldn = PLANE(rural, RH_SOLDN)[idx];
    c.ssg = soldn;
    c.llg = PLANE(fields, GF_GLW)[idx];
    c.rain = FMUL(FDIV(PLANE(rural, RH_RAINBL)[idx], dt), 3600.0f);
    float sfcprs = PLANE(rural, RH_SFCPRS)[idx];
    c.rhoo = FDIV(sfcprs, FMUL(FMUL(287.04f, c.ta), FADD(1.0f, FMUL(0.61f, c.qa))));
    c.za = PLANE(rural, RH_ZLVL)[idx];
    c.delt = dt;
    c.omg = hrang[idx];c.cosz=cosz[idx];c.xlat=xlat[idx];c.declin=declin;c.lsolar=false;c.ssgd=FMUL(0.8f,c.ssg);c.ssgq=FSUB(c.ssg,c.ssgd);
    c.znt = PLANE(fields, GF_ZNT)[idx];
    ucm_floor_exchange(chs, chs2, cqs2, idx);
    c.chs = chs[idx];
    c.chs2 = chs2[idx];
    ucm480_load_state(state, idx, plane, c);
    int code = ucm480_urban(tab, glob, isw, c);
    if (code != UCM_OK) { ucm_flag(err, code); return; }
    float f = frc[idx];
    float omf = FSUB(1.0f, f);
    float* albedo = PLANE(fields, GF_ALBEDO);
    albedo[idx] = FADD(FMUL(f, c.alb), FMUL(omf, PLANE(rural, RH_ALBEDOK)[idx]));
    PLANE(fields, GF_HFX)[idx] = FADD(FMUL(f, c.sh), FMUL(omf, PLANE(rural, RH_SHEAT)[idx]));
    PLANE(fields, GF_QFX)[idx] = FADD(FMUL(f, c.lh_kin), FMUL(omf, PLANE(rural, RH_ETA_KIN)[idx]));
    PLANE(fields, GF_LH)[idx] = FADD(FMUL(f, c.lh), FMUL(omf, PLANE(rural, RH_ETA)[idx]));
    PLANE(fields, GF_GRDFLX)[idx] = FADD(FMUL(f, c.g), FMUL(omf, PLANE(rural, RH_SSOIL)[idx]));
    PLANE(fields, GF_TSK)[idx] = FADD(FMUL(f, c.ts), FMUL(omf, PLANE(rural, RH_T1)[idx]));
    float q1 = FADD(FMUL(f, c.qs), FMUL(omf, PLANE(rural, RH_Q1)[idx]));
    PLANE(fields, GF_QSFC)[idx] = FDIV(q1, FSUB(1.0f, q1));
    float* ust = PLANE(fields, GF_UST);
    ust[idx] = FADD(FMUL(f, c.ust), FMUL(omf, ust[idx]));
    ucm480_store_state(state, idx, plane, c);
}

// ---------------------------------------------------------------------------
// Noah-MP (sf_surface_physics = 4): noahmp_urban's option-1 arm,
// module_sf_noahmpdrv.F 3374-3598.  The rural values are the grid fields as
// noahmplsm left them; the forcing is read from the lowest model level.
// ---------------------------------------------------------------------------
extern "C" __global__ void ucm480_noahmp_after_lsm(
    const int* __restrict__ urban, const int* __restrict__ utype,
    const float* __restrict__ frc, const float* __restrict__ u1,
    const float* __restrict__ v1, const float* __restrict__ t3d1,
    const float* __restrict__ qv1, const float* __restrict__ p8w1,
    const float* __restrict__ p8w2, const float* __restrict__ dz8w1,
    const float* __restrict__ hrang,const float* __restrict__ cosz,const float* __restrict__ xlat,float declin,
    const unsigned long long* __restrict__ fields,
    const unsigned long long* __restrict__ state,
    const float* __restrict__ tab, const float* __restrict__ glob,
    const int* __restrict__ isw, float dt, int jmonth, unsigned int* err,
    int ny, int nx)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = (size_t)ny * (size_t)nx;
    if (idx >= plane || !urban[idx]) return;
    float* chs = PLANE(fields, GF_CHS);
    float* chs2 = PLANE(fields, GF_CHS2);
    float* cqs2 = PLANE(fields, GF_CQS2);
    Ucm480Col c{};
    c.utype = utype[idx];
    c.jmonth = jmonth;
    c.ta = t3d1[idx];
    float qv = qv1[idx];
    c.qa = FDIV(qv, FADD(1.0f, qv));
    c.ua = ucm_wind(u1[idx], v1[idx]);
    c.u1 = u1[idx];
    c.v1 = v1[idx];
    float swdown = PLANE(fields, GF_SWDOWN)[idx];
    c.ssg = swdown;
    c.llg = PLANE(fields, GF_GLW)[idx];
    c.rain = FMUL(FDIV(PLANE(fields, GF_RAINBL)[idx], dt), 3600.0f);
    c.rhoo = FDIV(FMUL(FADD(p8w2[idx], p8w1[idx]), 0.5f),
                  FMUL(FMUL(287.04f, c.ta), FADD(1.0f, FMUL(0.61f, c.qa))));
    c.za = FMUL(0.5f, dz8w1[idx]);
    c.delt = dt;
    c.omg = hrang[idx];c.cosz=cosz[idx];c.xlat=xlat[idx];c.declin=declin;c.lsolar=false;c.ssgd=FMUL(0.8f,c.ssg);c.ssgq=FSUB(c.ssg,c.ssgd);
    c.znt = PLANE(fields, GF_ZNT)[idx];
    ucm_floor_exchange(chs, chs2, cqs2, idx);
    c.chs = chs[idx];
    chs2[idx] = cqs2[idx];                         // :3445 CHS2(I,J)= CQS2(I,J)
    c.chs2 = chs2[idx];
    ucm480_load_state(state, idx, plane, c);
    int code = ucm480_urban(tab, glob, isw, c);
    if (code != UCM_OK) { ucm_flag(err, code); return; }
    float f = frc[idx];
    float omf = FSUB(1.0f, f);
    float* albedo = PLANE(fields, GF_ALBEDO);
    float* hfx = PLANE(fields, GF_HFX);
    float* qfx = PLANE(fields, GF_QFX);
    float* lh = PLANE(fields, GF_LH);
    float* grdflx = PLANE(fields, GF_GRDFLX);
    float* tsk = PLANE(fields, GF_TSK);
    float* qsfc = PLANE(fields, GF_QSFC);
    float* ust = PLANE(fields, GF_UST);
    albedo[idx] = FADD(FMUL(f, c.alb), FMUL(omf, albedo[idx]));
    hfx[idx] = FADD(FMUL(f, c.sh), FMUL(omf, hfx[idx]));
    qfx[idx] = FADD(FMUL(f, c.lh_kin), FMUL(omf, qfx[idx]));
    lh[idx] = FADD(FMUL(f, c.lh), FMUL(omf, lh[idx]));
    grdflx[idx] = FADD(FMUL(f, FMUL(c.g, -1.0f)), FMUL(omf, grdflx[idx]));
    tsk[idx] = FADD(FMUL(f, c.ts), FMUL(omf, tsk[idx]));
    qsfc[idx] = FADD(FMUL(f, c.qs), FMUL(omf, qsfc[idx]));
    ust[idx] = FADD(FMUL(f, c.ust), FMUL(omf, ust[idx]));
    ucm480_store_state(state, idx, plane, c);
}

// ---------------------------------------------------------------------------

extern "C" __global__ void urban480_columns(const unsigned long long* args,const float* table,const float* globals,const int* switches,int* codes,int n) {
int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;int status=0;U480Env env{table,globals,switches};
bool lsolar=((const int*)args[0])[row];
int num_roof_layers=((const int*)args[1])[row];
int num_wall_layers=((const int*)args[2])[row];
int num_road_layers=((const int*)args[3])[row];
float dzr[4];for(int k=0;k<4;++k)dzr[k]=((const float*)args[4])[row+k*n];
float dzb[4];for(int k=0;k<4;++k)dzb[k]=((const float*)args[5])[row+k*n];
float dzg[4];for(int k=0;k<4;++k)dzg[k]=((const float*)args[6])[row+k*n];
int utype=((const int*)args[7])[row];
float ta=((const float*)args[8])[row];
float qa=((const float*)args[9])[row];
float ua=((const float*)args[10])[row];
float u1=((const float*)args[11])[row];
float v1=((const float*)args[12])[row];
float ssg=((const float*)args[13])[row];
float ssgd=((const float*)args[14])[row];
float ssgq=((const float*)args[15])[row];
float llg=((const float*)args[16])[row];
float rain=((const float*)args[17])[row];
float rhoo=((const float*)args[18])[row];
float za=((const float*)args[19])[row];
float declin=((const float*)args[20])[row];
float cosz=((const float*)args[21])[row];
float omg=((const float*)args[22])[row];
float xlat=((const float*)args[23])[row];
float delt=((const float*)args[24])[row];
float znt=((const float*)args[25])[row];
float chs=((const float*)args[26])[row];
float chs2=((const float*)args[27])[row];
float tr=((const float*)args[28])[row];
float tb=((const float*)args[29])[row];
float tg=((const float*)args[30])[row];
float tc=((const float*)args[31])[row];
float qc=((const float*)args[32])[row];
float uc=((const float*)args[33])[row];
float trl[4];for(int k=0;k<4;++k)trl[k]=((const float*)args[34])[row+k*n];
float tbl[4];for(int k=0;k<4;++k)tbl[k]=((const float*)args[35])[row+k*n];
float tgl[4];for(int k=0;k<4;++k)tgl[k]=((const float*)args[36])[row+k*n];
float xxxr=((const float*)args[37])[row];
float xxxb=((const float*)args[38])[row];
float xxxg=((const float*)args[39])[row];
float xxxc=((const float*)args[40])[row];
float ts=((const float*)args[41])[row];
float qs=((const float*)args[42])[row];
float sh=((const float*)args[43])[row];
float lh=((const float*)args[44])[row];
float lh_kinematic=((const float*)args[45])[row];
float sw=((const float*)args[46])[row];
float alb=((const float*)args[47])[row];
float lw=((const float*)args[48])[row];
float g=((const float*)args[49])[row];
float rn=((const float*)args[50])[row];
float psim=((const float*)args[51])[row];
float psih=((const float*)args[52])[row];
float gz1oz0=((const float*)args[53])[row];
float cmr_urb=((const float*)args[54])[row];
float chr_urb=((const float*)args[55])[row];
float cmc_urb=((const float*)args[56])[row];
float chc_urb=((const float*)args[57])[row];
float u10=((const float*)args[58])[row];
float v10=((const float*)args[59])[row];
float th2=((const float*)args[60])[row];
float q2=((const float*)args[61])[row];
float ust=((const float*)args[62])[row];
float mh_urb=((const float*)args[63])[row];
float stdh_urb=((const float*)args[64])[row];
float lf_urb[4];for(int k=0;k<4;++k)lf_urb[k]=((const float*)args[65])[row+k*n];
float lp_urb=((const float*)args[66])[row];
float hgt_urb=((const float*)args[67])[row];
float frc_urb=((const float*)args[68])[row];
float lb_urb=((const float*)args[69])[row];
float zo_check=((const float*)args[70])[row];
float cmcr=((const float*)args[71])[row];
float tgr=((const float*)args[72])[row];
float tgrl[4];for(int k=0;k<4;++k)tgrl[k]=((const float*)args[73])[row+k*n];
float smr[4];for(int k=0;k<4;++k)smr[k]=((const float*)args[74])[row+k*n];
float cmgr_urb=((const float*)args[75])[row];
float chgr_urb=((const float*)args[76])[row];
int jmonth=((const int*)args[77])[row];
float drelr=((const float*)args[78])[row];
float drelb=((const float*)args[79])[row];
float drelg=((const float*)args[80])[row];
float flxhumr=((const float*)args[81])[row];
float flxhumb=((const float*)args[82])[row];
float flxhumg=((const float*)args[83])[row];
float tvg=((const float*)args[84])[row];
float tt=((const float*)args[85])[row];
float xxxvg=((const float*)args[86])[row];
float tvgl[4];for(int k=0;k<4;++k)tvgl[k]=((const float*)args[87])[row+k*n];
float smg[4];for(int k=0;k<4;++k)smg[k]=((const float*)args[88])[row+k*n];
float cmcg=((const float*)args[89])[row];
float flxhumvg=((const float*)args[90])[row];
float flxhumt=((const float*)args[91])[row];
float lf_urb_s=((const float*)args[92])[row];
float z0_urb=((const float*)args[93])[row];
float vegfrac_in=((const float*)args[94])[row];
u480_urban(lsolar,num_roof_layers,num_wall_layers,num_road_layers,dzr,dzb,dzg,utype,ta,qa,ua,u1,v1,ssg,ssgd,ssgq,llg,rain,rhoo,za,declin,cosz,omg,xlat,delt,znt,chs,chs2,tr,tb,tg,tc,qc,uc,trl,tbl,tgl,xxxr,xxxb,xxxg,xxxc,ts,qs,sh,lh,lh_kinematic,sw,alb,lw,g,rn,psim,psih,gz1oz0,cmr_urb,chr_urb,cmc_urb,chc_urb,u10,v10,th2,q2,ust,mh_urb,stdh_urb,lf_urb,lp_urb,hgt_urb,frc_urb,lb_urb,zo_check,cmcr,tgr,tgrl,smr,cmgr_urb,chgr_urb,jmonth,drelr,drelb,drelg,flxhumr,flxhumb,flxhumg,tvg,tt,xxxvg,tvgl,smg,cmcg,flxhumvg,flxhumt,lf_urb_s,z0_urb,vegfrac_in,env,status);
((int*)args[0])[row]=lsolar;
((int*)args[1])[row]=num_roof_layers;
((int*)args[2])[row]=num_wall_layers;
((int*)args[3])[row]=num_road_layers;
for(int k=0;k<4;++k)((float*)args[4])[row+k*n]=dzr[k];
for(int k=0;k<4;++k)((float*)args[5])[row+k*n]=dzb[k];
for(int k=0;k<4;++k)((float*)args[6])[row+k*n]=dzg[k];
((int*)args[7])[row]=utype;
((float*)args[8])[row]=ta;
((float*)args[9])[row]=qa;
((float*)args[10])[row]=ua;
((float*)args[11])[row]=u1;
((float*)args[12])[row]=v1;
((float*)args[13])[row]=ssg;
((float*)args[14])[row]=ssgd;
((float*)args[15])[row]=ssgq;
((float*)args[16])[row]=llg;
((float*)args[17])[row]=rain;
((float*)args[18])[row]=rhoo;
((float*)args[19])[row]=za;
((float*)args[20])[row]=declin;
((float*)args[21])[row]=cosz;
((float*)args[22])[row]=omg;
((float*)args[23])[row]=xlat;
((float*)args[24])[row]=delt;
((float*)args[25])[row]=znt;
((float*)args[26])[row]=chs;
((float*)args[27])[row]=chs2;
((float*)args[28])[row]=tr;
((float*)args[29])[row]=tb;
((float*)args[30])[row]=tg;
((float*)args[31])[row]=tc;
((float*)args[32])[row]=qc;
((float*)args[33])[row]=uc;
for(int k=0;k<4;++k)((float*)args[34])[row+k*n]=trl[k];
for(int k=0;k<4;++k)((float*)args[35])[row+k*n]=tbl[k];
for(int k=0;k<4;++k)((float*)args[36])[row+k*n]=tgl[k];
((float*)args[37])[row]=xxxr;
((float*)args[38])[row]=xxxb;
((float*)args[39])[row]=xxxg;
((float*)args[40])[row]=xxxc;
((float*)args[41])[row]=ts;
((float*)args[42])[row]=qs;
((float*)args[43])[row]=sh;
((float*)args[44])[row]=lh;
((float*)args[45])[row]=lh_kinematic;
((float*)args[46])[row]=sw;
((float*)args[47])[row]=alb;
((float*)args[48])[row]=lw;
((float*)args[49])[row]=g;
((float*)args[50])[row]=rn;
((float*)args[51])[row]=psim;
((float*)args[52])[row]=psih;
((float*)args[53])[row]=gz1oz0;
((float*)args[54])[row]=cmr_urb;
((float*)args[55])[row]=chr_urb;
((float*)args[56])[row]=cmc_urb;
((float*)args[57])[row]=chc_urb;
((float*)args[58])[row]=u10;
((float*)args[59])[row]=v10;
((float*)args[60])[row]=th2;
((float*)args[61])[row]=q2;
((float*)args[62])[row]=ust;
((float*)args[63])[row]=mh_urb;
((float*)args[64])[row]=stdh_urb;
for(int k=0;k<4;++k)((float*)args[65])[row+k*n]=lf_urb[k];
((float*)args[66])[row]=lp_urb;
((float*)args[67])[row]=hgt_urb;
((float*)args[68])[row]=frc_urb;
((float*)args[69])[row]=lb_urb;
((float*)args[70])[row]=zo_check;
((float*)args[71])[row]=cmcr;
((float*)args[72])[row]=tgr;
for(int k=0;k<4;++k)((float*)args[73])[row+k*n]=tgrl[k];
for(int k=0;k<4;++k)((float*)args[74])[row+k*n]=smr[k];
((float*)args[75])[row]=cmgr_urb;
((float*)args[76])[row]=chgr_urb;
((int*)args[77])[row]=jmonth;
((float*)args[78])[row]=drelr;
((float*)args[79])[row]=drelb;
((float*)args[80])[row]=drelg;
((float*)args[81])[row]=flxhumr;
((float*)args[82])[row]=flxhumb;
((float*)args[83])[row]=flxhumg;
((float*)args[84])[row]=tvg;
((float*)args[85])[row]=tt;
((float*)args[86])[row]=xxxvg;
for(int k=0;k<4;++k)((float*)args[87])[row+k*n]=tvgl[k];
for(int k=0;k<4;++k)((float*)args[88])[row+k*n]=smg[k];
((float*)args[89])[row]=cmcg;
((float*)args[90])[row]=flxhumvg;
((float*)args[91])[row]=flxhumt;
((float*)args[92])[row]=lf_urb_s;
((float*)args[93])[row]=z0_urb;
((float*)args[94])[row]=vegfrac_in;
codes[row]=status;
}
