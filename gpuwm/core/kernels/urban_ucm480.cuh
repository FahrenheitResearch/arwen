// Generated from the pinned WRF v4.8 numerical statements.
struct U480Env {const float* tab; const float* glob; const int* sw;};
__device__ inline double u480_mind(double a,double b) {return a<b?a:b;}
__device__ inline double u480_maxd(double a,double b) {return a>b?a:b;}
__device__ inline double u480_powi64(double x,int n) {unsigned int m=n<0?0u-(unsigned int)n:(unsigned int)n;double y=m&1?x:1.0;while(m>>=1) {x=__dmul_rn(x,x);if(m&1)y=__dmul_rn(y,x);}return n<0?__ddiv_rn(1.0,y):y;}
__device__ void u480_urban(const bool& lsolar,const int& num_roof_layers,const int& num_wall_layers,const int& num_road_layers,const float* dzr,const float* dzb,const float* dzg,const int& utype,const float& ta,const float& qa,const float& ua,const float& u1,const float& v1,const float& ssg,float& ssgd,float& ssgq,const float& llg,const float& rain,const float& rhoo,const float& za,const float& declin,const float& cosz,const float& omg,const float& xlat,const float& delt,float& znt,float& chs,float& chs2,float& tr,float& tb,float& tg,float& tc,float& qc,float& uc,float* trl,float* tbl,float* tgl,float& xxxr,float& xxxb,float& xxxg,float& xxxc,float& ts,float& qs,float& sh,float& lh,float& lh_kinematic,float& sw,float& alb,float& lw,float& g,float& rn,float& psim,float& psih,float& gz1oz0,float& cmr_urb,float& chr_urb,float& cmc_urb,float& chc_urb,float& u10,float& v10,float& th2,float& q2,float& ust,float& mh_urb,float& stdh_urb,float* lf_urb,float& lp_urb,float& hgt_urb,float& frc_urb,float& lb_urb,float& zo_check,float& cmcr,float& tgr,float* tgrl,float* smr,float& cmgr_urb,float& chgr_urb,const int& jmonth,float& drelr,float& drelb,float& drelg,float& flxhumr,float& flxhumb,float& flxhumg,float& tvg,float& tt,float& xxxvg,float* tvgl,float* smg,float& cmcg,float& flxhumvg,float& flxhumt,const float& lf_urb_s,const float& z0_urb,const float& vegfrac_in,const U480Env& env,int& status);
__device__ void u480_mos(float& xxx,float& alpha,float& cd,const float& b1,float& rib,const float& z,const float& z0,const float& ua,const float& ta,const float& tsf,const float& rho,const U480Env& env,int& status);
__device__ void u480_multi_layer(const int& km,const int& bound,const float& g0,const float& cap,const float& aks,float* tsl,const float* dz,const float& delt,const float& tslend,const U480Env& env,int& status);
__device__ void u480_force_restore(const float& cap,const float& aks,const float& delt,const float& s,const float& r,const float& h,const float& le,const float& tslend,const float& tsp,float& ts,const U480Env& env,int& status);
__device__ void u480_sfcdif_urb(const float& zlm,const float& z0,const float& thz0,const float& thlm,const float& sfcspd,const float& akanda,float& akms,float& akhs,float& rlmo,float& cd,float& zt_out,const float& vegfrac,bool optional_present,const U480Env& env,int& status);
__device__ void u480_direvap(float& edir,const float& etp,const float& smc,const float& shdfac,const float& smcmax,const float& smcdry,const float& fxexp,const U480Env& env,int& status);
__device__ void u480_transp(float& ett,float* et,float& ec,const float& shdfac,const float& etp1,const float& cmc,const float& cfactr,const float& cmcmax,const float& lai,const float& rsmin,const float& rsmax,const float& rgl,const float& sx,const float& ts,const float& ta,const float& qa,const float* smc,const float& smcwlt,const float& smcref,const float& cpp,const float& ps,const float& ch,const float& epsv,const float& delt,const int& nroot,const int& nsoil,const float* dzvr,const float* zsoil,const float& hs,const U480Env& env,int& status);
__device__ void u480_smflx(const float* smcp,float* smc,const int& nsoil,const float& cmcp,float& cmc,const float& dt,const float& prcp1,const float* zsoil,const float& smcmax,const float& bexp,const float& smcwlt,const float* sroot,const float& dksat,const float& dwsat,const float& shdfac,const float& cmcmax,float& runoff1,float& runoff2,float& runoff3,const float& edir,const float& ec,const float* et,float& drip,const U480Env& env,int& status);
__device__ void u480_srt(float* rhstt,const float& edir,const float* et,const float* smcp,const int& nsoil,const float& pcpdrp,const float* zsoil,const float& dwsat,const float& dksat,const float& smcmax,const float& bexp,float& runoff1,float& runoff2,const float& dt,const float& smcwlt,const float* sroot,float* ai,float* bi,float* ci,const U480Env& env,int& status);
__device__ void u480_sstep(const float* smcp,float* smc,const float& cmcp,float& cmc,float* rhstt,const float& rhsct,const float& dt,const int& nsoil,const float& smcmax,const float& cmcmax,float& runoff3,const float* zsoil,float* ai,float* bi,float* ci,const U480Env& env,int& status);
__device__ void u480_wdfcnd(float& wdf,float& wcnd,const float& smc,const float& smcmax,const float& bexp,const float& dksat,const float& dwsat,const U480Env& env,int& status);
__device__ void u480_rosr12(float* p,const float* a,const float* b,float* c,const float* d,float* delta,const int& nsoil,const U480Env& env,int& status);
__device__ void u480_shflx(float& ssoil,float* stc,const float* smc,const float& smcmax,const int& nsoil,float& t1,const float& dt,const float& yy,const float& zz1,const float* zsoil,const float& tbot,const float& zbot,const float& smcwlt,const float& df1,const float& quartz,const float& csoil,const float& capr,const U480Env& env,int& status);
__device__ void u480_hrt(float* rhsts,const float* stc,const float* smc,const float& smcmax,const int& nsoil,const float* zsoil,const float& yy,const float& zz1,const float& tbot,const float& zbot,const float& dt,const float& df1,float* ai,float* bi,float* ci,const float& quartz,const float& csoil,const float& capr,const U480Env& env,int& status);
__device__ void u480_hstep(float* stcout,const float* stcin,float* rhsts,const float& dt,const int& nsoil,float* ai,float* bi,float* ci,const U480Env& env,int& status);
__device__ void u480_tbnd(const float& tu,const float& tb,const float* zsoil,const float& zbot,const int& k,const int& nsoil,float& tbnd1,const U480Env& env,int& status);
__device__ void u480_tdfcnd(float& df,const float& smc,const float& qz,const float& smcmax,const U480Env& env,int& status);
__device__ void u480_shadow_tree(const float& xi,const float& h,const float& w,const float& htree,const float& rtree,const float& dtree,float& chi_shaded,float& eta_shaded,float& chi_shaded_tree,float& eta_shaded_tree,const U480Env& env,int& status);
__device__ void u480_stomatal(float& rs,const float& rsmin,const float& sd_facet,const float& lai,const float& w2,const float& wr,const float& ws,const float& ta,const float& ts,const float& qa,const float& qsat,const U480Env& env,int& status);
__device__ void u480_leleaf(float& le,const float& sw,const float& lw,const float& ra_leaf,const float& rs_leaf,const float& ra,const float& ta,const float& qa,const float& qsat,const U480Env& env,int& status);
__device__ void u480_force_restore_tree(const float& delt,const float& s,const float& r,const float& h,const float& le,const float& lai,const float& tsp,float& ts,const U480Env& env,int& status);
__device__ void u480_root_uptake(float* sroot,const int& nlayer,const float* smc,const float& smcmax,const float& smcdry,const float& rtree,const float& lai_tree,const float* dzg,const float* zsoil,const float& ele,const float& fvg,const float& rw,const U480Env& env,int& status);
__device__ float u480_cdf2d_vf(const float& theta,const U480Env& env,int& status);
__device__ void u480_vf_tree_analytical(const float& w,const float& h,const float& ht,const float& rt,const float& trans,float& fgs,float& fgw,float& fgt,float& fws,float& fwg,float& fww,float& fwt,float& fts,float& ftg,float& ftw,const U480Env& env,int& status);

__device__ void u480_urban(const bool& lsolar,const int& num_roof_layers,const int& num_wall_layers,const int& num_road_layers,const float* dzr,const float* dzb,const float* dzg,const int& utype,const float& ta,const float& qa,const float& ua,const float& u1,const float& v1,const float& ssg,float& ssgd,float& ssgq,const float& llg,const float& rain,const float& rhoo,const float& za,const float& declin,const float& cosz,const float& omg,const float& xlat,const float& delt,float& znt,float& chs,float& chs2,float& tr,float& tb,float& tg,float& tc,float& qc,float& uc,float* trl,float* tbl,float* tgl,float& xxxr,float& xxxb,float& xxxg,float& xxxc,float& ts,float& qs,float& sh,float& lh,float& lh_kinematic,float& sw,float& alb,float& lw,float& g,float& rn,float& psim,float& psih,float& gz1oz0,float& cmr_urb,float& chr_urb,float& cmc_urb,float& chc_urb,float& u10,float& v10,float& th2,float& q2,float& ust,float& mh_urb,float& stdh_urb,float* lf_urb,float& lp_urb,float& hgt_urb,float& frc_urb,float& lb_urb,float& zo_check,float& cmcr,float& tgr,float* tgrl,float* smr,float& cmgr_urb,float& chgr_urb,const int& jmonth,float& drelr,float& drelb,float& drelg,float& flxhumr,float& flxhumb,float& flxhumg,float& tvg,float& tt,float& xxxvg,float* tvgl,float* smg,float& cmcg,float& flxhumvg,float& flxhumt,const float& lf_urb_s,const float& z0_urb,const float& vegfrac_in,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  const float cp=__uint_as_float(0x3e75c28fu);
  const float el=__uint_as_float(0x4411c000u);
  const float sig=__uint_as_float(0x2eb3a903u);
  const float sig_si=__uint_as_float(0x3373864fu);
  const float ak=__uint_as_float(0x3ecccccdu);
  const float pi=__uint_as_float(0x40490fd0u);
  const float tetena=__uint_as_float(0x40f00000u);
  const float tetenb=__uint_as_float(0x436d4ccdu);
  const float sratio=__uint_as_float(0x3f400000u);
  const float ell=__uint_as_float(0x4a150c40u);
  const float xka=__uint_as_float(0x37c9539cu);
  float tvgp={};
  float ttp={};
  float albge={};
  float xi={};
  float angle={};
  float tan1={};
  float tan2={};
  float sdt={};
  float sdg={};
  float sdb={};
  float tau={};
  float svg={};
  float svg1={};
  float svg2={};
  float st={};
  float st1={};
  float st2={};
  float sleaf={};
  float chi_shaded={};
  float chi_shaded_tree={};
  float eta_shaded={};
  float eta_shaded_tree={};
  float vfws_tree={};
  float vfww_tree={};
  float vfts_tree={};
  float vfwt_tree={};
  float vfgs_tree={};
  float vfwg_tree={};
  float vfgw_tree={};
  float vfgt_tree={};
  float vftw_tree={};
  float vftg_tree={};
  float bhvg={};
  float ribvg={};
  float alphavg={};
  float chvg={};
  float cdvg={};
  float srun1={};
  float srun2={};
  float srun3={};
  float zsoilg[4]={};
  float smgp[4]={};
  float etg[4]={};
  float dzvg[4]={};
  float qs0vg={};
  float dqs0vgdtvg={};
  float qs0t={};
  float dqs0tdtt={};
  float epvg={};
  float edirg={};
  float ettg={};
  float ecg={};
  float etag={};
  float dripg={};
  float betvg={};
  float vfac={};
  float emitw={};
  float emitg={};
  float emitt={};
  float epsge={};
  float rvg1={};
  float rvg2={};
  float rvg={};
  float rt1={};
  float rt2={};
  float rt={};
  float rleaf={};
  double drbdtvg1={};
  double drbdtvg2={};
  double drbdtt1={};
  double drbdtt2={};
  double drgdtvg1={};
  double drgdtvg2={};
  double drgdtt1={};
  double drgdtt2={};
  double drvgdtb1={};
  double drvgdtb2={};
  double drvgdtvg1={};
  double drvgdtvg2={};
  double drvgdtg1={};
  double drvgdtg2={};
  double drvgdtt1={};
  double drvgdtt2={};
  double drtdtb1={};
  double drtdtb2={};
  double drtdtg1={};
  double drtdtg2={};
  double drtdtvg1={};
  double drtdtvg2={};
  double drtdtt1={};
  double drtdtt2={};
  double drbdtvg={};
  double drbdtt={};
  double drgdtvg={};
  double drgdtt={};
  double drvgdtb={};
  double drvgdtg={};
  double drvgdtvg={};
  double drvgdtt={};
  double drtdtb={};
  double drtdtg={};
  double drtdtvg={};
  double drtdtt={};
  double dhbdtvg={};
  double dhbdtt={};
  double dhgdtvg={};
  double dhgdtt={};
  double dhvgdtb={};
  double dhvgdtg={};
  double dhvgdtvg={};
  double dhvgdtt={};
  double dhtdtb={};
  double dhtdtg={};
  double dhtdtvg={};
  double dhtdtt={};
  float hvg={};
  float ht={};
  float ra_leaf={};
  float alphat={};
  float w2={};
  float rs_leaf={};
  float elet={};
  float available_water_sum={};
  float potential_elet={};
  float qs0c={};
  float sroot[4]={};
  float srootp[4]={};
  float des={};
  float mix={};
  float dmix={};
  float qsat={};
  float dqsat={};
  float esat={};
  float desat={};
  float delebdtvg={};
  float delebdtt={};
  float delegdtvg={};
  float delegdtt={};
  float delevgdtb={};
  float delevgdtg={};
  double drs_leaf={};
  double m={};
  double d={};
  double dd={};
  double fe={};
  double dfe={};
  double s={};
  double dm={};
  double kk={};
  double l={};
  double n={};
  double delevgdtvg={};
  double delevgdtt={};
  double deletdtb={};
  double deletdtg={};
  double deletdtvg={};
  double deletdtt={};
  double dtcdtvg={};
  double dtcdtt={};
  float dqcdtvg={};
  float dqcdtt={};
  float elevg={};
  float kvg={};
  float g0vg={};
  float dg0bdtvg={};
  float dg0bdtt={};
  float dg0gdtvg={};
  float dg0gdtt={};
  float dg0vgdtb={};
  float dg0vgdtg={};
  float dg0vgdtvg={};
  float dg0vgdtt={};
  double wf={};
  double a={};
  double b={};
  double c={};
  double e={};
  double ff={};
  double gg={};
  double h={};
  double vgf={};
  double i={};
  double j={};
  double o={};
  double p={};
  double chara={};
  double det={};
  double det_tol={};
  double det_scale={};
  double a_use={};
  double ff_use={};
  double kk_use={};
  double f_lambda={};
  double relax={};
  double w2_num={};
  double w2_den={};
  double den_qc={};
  float max_step={};
  float ssoilg={};
  float tge={};
  float tgep={};
  float qs0ge={};
  float alphage={};
  float flxthvg={};
  float flxtht={};
  float cmcgp={};
  const int nvg=4;
  const float cpp=__uint_as_float(0x447b2000u);
  const float dleaf=__uint_as_float(0x3dcccccdu);
  const float tref=__uint_as_float(0x43950000u);
  const float eref=__uint_as_float(0x4545f000u);
  float fvg={};
  float z0vg={};
  float z0hvg={};
  float capvg={};
  float epsvg={};
  float epst={};
  float rtree={};
  float rtree_m={};
  float rw_m={};
  float dtree={};
  float htree={};
  float albt={};
  float albvg={};
  float lai_tree={};
  float lai_veg={};
  float zr={};
  float z0c={};
  float z0hc={};
  float zdc={};
  float svf={};
  float r={};
  float rw={};
  float hgt={};
  float ah={};
  float alh={};
  float sigma_zed={};
  float capr={};
  float capb={};
  float capg={};
  float aksr={};
  float aksb={};
  float aksg={};
  float albr={};
  float albb={};
  float albg={};
  float epsr={};
  float epsb={};
  float epsg={};
  float z0r={};
  float z0b={};
  float z0g={};
  float z0hb={};
  float z0hg={};
  float trlend={};
  float tblend={};
  float tglend={};
  float t1vr={};
  float t1vc={};
  float th2v={};
  float rlmo_urb={};
  float akanda_urban={};
  float th2x={};
  int boundr={};
  int boundb={};
  int boundg={};
  int ch_scheme={};
  int ts_scheme={};
  bool shadow={};
  int numdir={};
  float street_direction[3]={};
  float street_width[3]={};
  float building_width[3]={};
  int numhgt={};
  float height_bin[50]={};
  float hpercent_bin[50]={};
  float betr={};
  float betb={};
  float betg={};
  float sx={};
  float sd={};
  float sq={};
  float rx={};
  float ur={};
  float zc={};
  float xlb={};
  float bb={};
  float z={};
  float ribb={};
  float ribg={};
  float ribc={};
  float bhr={};
  float bhb={};
  float bhg={};
  float bhc={};
  float tsc={};
  float lnet={};
  float snet={};
  float flxuv={};
  float thg={};
  float flxth={};
  float flxhum={};
  float flxg={};
  float w={};
  float vfgs={};
  float vfgw={};
  float vfwg={};
  float vfws={};
  float vfww={};
  float houi1={};
  float houi2={};
  float houi3={};
  float houi4={};
  float houi5={};
  float houi6={};
  float houi7={};
  float houi8={};
  float slx={};
  float slx1={};
  float slx2={};
  float slx3={};
  float slx4={};
  float slx5={};
  float slx6={};
  float slx7={};
  float slx8={};
  float flxthr={};
  float flxthb={};
  float flxthg={};
  float sr={};
  float sb={};
  float sg={};
  float rr={};
  float rb={};
  float rg={};
  float sr1={};
  float sr2={};
  float sb1={};
  float sb2={};
  float sg1={};
  float sg2={};
  float rr1={};
  float rr2={};
  float rb1={};
  float rb2={};
  float rg1={};
  float rg2={};
  float hr={};
  float hb={};
  float hg={};
  float eler={};
  float eleb={};
  float eleg={};
  float g0r={};
  float g0b={};
  float g0g={};
  float alphac={};
  float alphar={};
  float alphab={};
  float alphag={};
  float chc={};
  float chr={};
  float chb={};
  float chg={};
  float cdc={};
  float cdr={};
  float cdb={};
  float cdg={};
  float cdgr={};
  float c1r={};
  float c1b={};
  float c1g={};
  float te={};
  float tc1={};
  float tc2={};
  float qc1={};
  float qc2={};
  float qs0r={};
  float qs0b={};
  float qs0g={};
  float rho={};
  float es={};
  float desdt={};
  float f={};
  float dqs0rdtr={};
  float drrdtr={};
  float dhrdtr={};
  float delerdtr={};
  float dg0rdtr={};
  float dtr={};
  float dfdt={};
  float fx={};
  float fy={};
  float gf={};
  float gx={};
  float gy={};
  float dtcdtb={};
  float dtcdtg={};
  float dqcdtb={};
  float dqcdtg={};
  float drbdtb1={};
  float drbdtg1={};
  float drbdtb2={};
  float drbdtg2={};
  float drgdtb1={};
  float drgdtg1={};
  float drgdtb2={};
  float drgdtg2={};
  float drbdtb={};
  float drbdtg={};
  float drgdtb={};
  float drgdtg={};
  float dhbdtb={};
  float dhbdtg={};
  float dhgdtb={};
  float dhgdtg={};
  float delebdtb={};
  float delebdtg={};
  float delegdtg={};
  float delegdtb={};
  float dg0bdtb={};
  float dg0bdtg={};
  float dg0gdtg={};
  float dg0gdtb={};
  float dqs0bdtb={};
  float dqs0gdtg={};
  float dtb={};
  float dtg={};
  float dtc={};
  float dtvg={};
  float dtt={};
  float theataz={};
  float theatas={};
  float fai={};
  float cnt={};
  float snt={};
  float ps={};
  float tav={};
  float xxx={};
  float x={};
  float z0={};
  float z0h={};
  float cd={};
  float ch={};
  float xxx2={};
  float psim2={};
  float psih2={};
  float xxx10={};
  float psim10={};
  float psih10={};
  float psix={};
  float psit={};
  float psix2={};
  float psit2={};
  float psix10={};
  float psit10={};
  float trp={};
  float tbp={};
  float tgp={};
  float tcp={};
  float qcp={};
  float tst={};
  float qst={};
  float tsp={};
  float chs_local={};
  float chs2_local={};
  float wdr={};
  float hgt2={};
  float bw={};
  float dhgt={};
  const float vonk=__uint_as_float(0x3ecccccdu);
  float lambda_f={};
  float alpha_macd={};
  float beta_macd={};
  float lambda_fr={};
  float lambda_p={};
  float vegfrac={};
  int iteration={};
  int k={};
  int nudapt={};
  int tloc={};
  int tloc2={};
  int kalh={};
  float flxhumrp={};
  float flxhumbp={};
  float flxhumgp={};
  float drelrp={};
  float drelbp={};
  float drelgp={};
  float tgrp={};
  float cmcrp={};
  float zsoilr[4]={};
  float etr[4]={};
  float smrp[4]={};
  int kz={};
  float runoff1={};
  float runoff2={};
  float runoff3={};
  float sgr={};
  float sgr1={};
  float t1vgr={};
  float chgr={};
  float alphagr={};
  float flxthgr={};
  float flxhumgr={};
  float hgr={};
  float elegr={};
  float g0gr={};
  float qs0gr={};
  float epgr={};
  float edir={};
  float ettr={};
  float fv={};
  float dtgr={};
  float drip={};
  float dqs0grdtgr={};
  float ecr={};
  float rain1={};
  float raindr={};
  float dew={};
  float etar={};
  float betgr={};
  float df1={};
  float rgr={};
  float rgrr={};
  float rch={};
  float yy={};
  float zz1={};
  float ssoilr={};
  float rvgt={};
  float drrdtgr={};
  float dhrdtgr={};
  float delerdtgr={};
  float dg0rdtgr={};
  float dfdvt={};
  const float shdfac=__uint_as_float(0x3f4ccccdu);
  const float albv=__uint_as_float(0x3e4ccccdu);
  const float epsv=__uint_as_float(0x3f6e147bu);
  const float lai=__uint_as_float(0x3fc00000u);
  const float cmcmax=__uint_as_float(0x3a03126fu);
  const float smcref=__uint_as_float(0x3ea872b0u);
  const float smcdry=__uint_as_float(0x3d872b02u);
  const float smcwlt=__uint_as_float(0x3dac0831u);
  const float smcmax=__uint_as_float(0x3ee0c49cu);
  const float rsmax=__uint_as_float(0x459c4000u);
  const float rsmin=__uint_as_float(0x42c80000u);
  const float rgl=__uint_as_float(0x42c80000u);
  const float cfactr=__uint_as_float(0x3f000000u);
  const float dwsat=__uint_as_float(0x376fea08u);
  const float dksat=__uint_as_float(0x3662d3f5u);
  const float bexp=__uint_as_float(0x40a80000u);
  const float fxexp=__uint_as_float(0x40000000u);
  const float zbot=__uint_as_float(0xc0000000u);
  const float quartz=__uint_as_float(0x3ecccccdu);
  const float csoil=__uint_as_float(0x49f42400u);
  const float hs=__uint_as_float(0x42100000u);
  const int nroot=2;
  const int ngr=4;
  const int impr=1;
  const int impb=2;
  const int impg=3;
  const float* dzgr=env.glob+12;
  const float* porimp=env.glob+16;
  const float* dengimp=env.glob+19;
  const float* ahdiuprf=env.glob+22;
  const float* alhseason=env.glob+46;
  const float* alhdiuprf=env.glob+50;
  const float fgr=env.glob[98];
  const int ahoption=env.sw[5],alhoption=env.sw[6],imp_scheme=env.sw[7],iri_scheme=env.sw[8],groption=env.sw[9];
  const int treeoption=env.sw[11],ttscheme=env.sw[12];
  const bool distributed_aerodynamics_option=false;
  float optional_zt=0.0f,optional_veg=0.0f;
  if(mh_urb>0.0f) {status=2;return;}
  if(utype<1 || utype>env.sw[10]) {status=3;return;}
  {
  }
  // source line 739
  if ((((ahoption==1)) || ((iri_scheme==1)))) {
  // source line 740
  tloc=u480_mod(u480_int(FADD(FADD(FDIV(FMUL(FDIV(omg,__uint_as_float(0x40490fd0u)),__uint_as_float(0x43340000u)),__uint_as_float(0x41700000u)),__uint_as_float(0x41400000u)),__uint_as_float(0x3f000000u))),24);
  // source line 741
  if (((tloc<0))) {
  tloc=(tloc+24);
  }
  // source line 742
  if (((tloc==0))) {
  tloc=24;
  }
  // source line 743
  }
  // source line 745
  if (((alhoption==1))) {
  // source line 746
  tloc2=u480_mod(u480_int(FADD(FMUL(FADD(FDIV(FMUL(FDIV(omg,__uint_as_float(0x40490fd0u)),__uint_as_float(0x43340000u)),__uint_as_float(0x41700000u)),__uint_as_float(0x41400000u)),__uint_as_float(0x40000000u)),__uint_as_float(0x3f000000u))),48);
  // source line 747
  if (((tloc2<0))) {
  tloc2=(tloc2+48);
  }
  // source line 748
  if (((tloc2==0))) {
  tloc2=48;
  }
  // source line 749
  }
  // source line 751
  zr=env.tab[(utype-1)*49+0];
  z0c=env.tab[(utype-1)*49+1];
  z0hc=env.tab[(utype-1)*49+2];
  zdc=env.tab[(utype-1)*49+3];
  svf=env.tab[(utype-1)*49+4];
  r=env.tab[(utype-1)*49+5];
  rw=env.tab[(utype-1)*49+6];
  hgt=env.tab[(utype-1)*49+7];
  ah=env.tab[(utype-1)*49+8];
  alh=env.tab[(utype-1)*49+9];
  betr=env.tab[(utype-1)*49+10];
  betb=env.tab[(utype-1)*49+11];
  betg=env.tab[(utype-1)*49+12];
  capr=env.tab[(utype-1)*49+13];
  capb=env.tab[(utype-1)*49+14];
  capg=env.tab[(utype-1)*49+15];
  aksr=env.tab[(utype-1)*49+16];
  aksb=env.tab[(utype-1)*49+17];
  aksg=env.tab[(utype-1)*49+18];
  albr=env.tab[(utype-1)*49+19];
  albb=env.tab[(utype-1)*49+20];
  albg=env.tab[(utype-1)*49+21];
  epsr=env.tab[(utype-1)*49+22];
  epsb=env.tab[(utype-1)*49+23];
  epsg=env.tab[(utype-1)*49+24];
  z0r=env.tab[(utype-1)*49+25];
  z0b=env.tab[(utype-1)*49+26];
  z0g=env.tab[(utype-1)*49+27];
  z0hb=env.tab[(utype-1)*49+28];
  z0hg=env.tab[(utype-1)*49+29];
  trlend=env.tab[(utype-1)*49+30];
  tblend=env.tab[(utype-1)*49+31];
  tglend=env.tab[(utype-1)*49+32];
  akanda_urban=env.tab[(utype-1)*49+33];
  albvg=env.tab[(utype-1)*49+34];
  albt=env.tab[(utype-1)*49+35];
  fvg=env.tab[(utype-1)*49+36];
  rtree=env.tab[(utype-1)*49+37];
  rtree_m=env.tab[(utype-1)*49+38];
  rw_m=env.tab[(utype-1)*49+39];
  htree=env.tab[(utype-1)*49+40];
  dtree=env.tab[(utype-1)*49+41];
  lai_tree=env.tab[(utype-1)*49+42];
  lai_veg=env.tab[(utype-1)*49+43];
  z0vg=env.tab[(utype-1)*49+44];
  z0hvg=env.tab[(utype-1)*49+45];
  capvg=env.tab[(utype-1)*49+46];
  epsvg=env.tab[(utype-1)*49+47];
  epst=env.tab[(utype-1)*49+48];
  boundr=env.sw[0];boundb=env.sw[1];boundg=env.sw[2];ch_scheme=env.sw[3];ts_scheme=env.sw[4];
  {
  }
  // source line 912
  if (((ahoption==1))) {
  ah=FMUL(ah,ahdiuprf[(tloc)-1]);
  }
  // source line 915
  kalh=0;
  // source line 916
  if (((alhoption==1))) {
  // source line 917
  if ((((jmonth==3)) || ((jmonth==4)) || ((jmonth==5)))) {
  kalh=1;
  }
  // source line 918
  if ((((jmonth==6)) || ((jmonth==7)) || ((jmonth==8)))) {
  kalh=2;
  }
  // source line 919
  if ((((jmonth==9)) || ((jmonth==10)) || ((jmonth==11)))) {
  kalh=3;
  }
  // source line 920
  if ((((jmonth==12)) || ((jmonth==1)) || ((jmonth==2)))) {
  kalh=4;
  }
  // source line 921
  }
  // source line 922
  if (((alhoption==1))) {
  alh=FMUL(FMUL(alh,alhdiuprf[(tloc2)-1]),alhseason[(kalh)-1]);
  }
  {
  }
  // source line 943
  if (((FADD(FADD(zdc,z0c),__uint_as_float(0x40000000u))>=za))) {
  // source line 944
  status=1; return;
  // source line 945
  }
  // source line 947
  if ((!lsolar)) {
  // source line 948
  ssgd=FMUL(__uint_as_float(0x3f400000u),ssg);
  // source line 949
  ssgq=u480_sub(ssg,ssgd);
  // source line 950
  }
  // source line 951
  ssgd=FMUL(__uint_as_float(0x3f400000u),ssg);
  // source line 952
  ssgq=u480_sub(ssg,ssgd);
  // source line 954
  w=FMUL(__uint_as_float(0x40000000u),hgt);
  // source line 955
  vfgs=svf;
  // source line 956
  vfgw=u480_sub(__uint_as_float(0x3f800000u),svf);
  // source line 957
  vfwg=FDIV(FMUL(u480_sub(__uint_as_float(0x3f800000u),svf),u480_sub(__uint_as_float(0x3f800000u),r)),w);
  // source line 958
  vfws=vfwg;
  // source line 959
  vfww=u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfwg));
  // source line 966
  sx=FDIV(FDIV(FADD(ssgd,ssgq),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 967
  sd=FDIV(FDIV(ssgd,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 968
  sq=FDIV(FDIV(ssgq,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 969
  rx=FDIV(FDIV(llg,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 970
  rho=FMUL(rhoo,__uint_as_float(0x3a83126fu));
  // source line 972
  trp=tr;
  // source line 973
  tbp=tb;
  // source line 974
  tgp=tg;
  // source line 975
  tcp=tc;
  // source line 976
  qcp=qc;
  // source line 978
  tsp=FDIV(FADD(FADD(FMUL(tr,r),FMUL(tb,w)),FMUL(tg,rw)),FADD(FADD(r,rw),w));
  // source line 982
  flxhumrp=flxhumr;
  // source line 983
  flxhumbp=flxhumb;
  // source line 984
  flxhumgp=flxhumg;
  // source line 985
  drelrp=drelr;
  // source line 986
  drelbp=drelb;
  // source line 987
  drelgp=drelg;
  // source line 988
  tgrp=tgr;
  // source line 989
  cmcrp=cmcr;
  // source line 990
  for(int vi=0;vi<4;++vi) smrp[vi]=smr[vi];
  // source line 993
  if (((iri_scheme==1))) {
  // source line 994
  if ((((tloc==21)) || ((tloc==22)))) {
  // source line 995
  if ((((jmonth==5)) || ((jmonth==6)) || ((jmonth==7)) || ((jmonth==8)) || ((jmonth==9)))) {
  // source line 997
  for (kz=1;kz<=2;kz+=1) {
  // source line 998
  smrp[(kz)-1]=__uint_as_float(0x3ea872b0u);
  // source line 999
  }
  // source line 1000
  }
  // source line 1001
  }
  // source line 1002
  }
  // source line 1004
  tav=FMUL(ta,FADD(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x3f1c28f6u),qa)));
  // source line 1005
  ps=FDIV(FMUL(FMUL(rhoo,__uint_as_float(0x438f8000u)),tav),__uint_as_float(0x42c80000u));
  // source line 1008
  for(int vi=0;vi<4;++vi) sroot[vi]=__uint_as_float(0x00000000u);
  for(int vi=0;vi<4;++vi) etr[vi]=__uint_as_float(0x00000000u);
  for(int vi=0;vi<4;++vi) etg[vi]=__uint_as_float(0x00000000u);
  tge=FADD(FMUL(u480_sub(__uint_as_float(0x3f800000u),fvg),tg),FMUL(fvg,tvg));
  hvg=__uint_as_float(0x00000000u);
  elevg=__uint_as_float(0x00000000u);
  ht=__uint_as_float(0x00000000u);
  elet=__uint_as_float(0x00000000u);
  sleaf=__uint_as_float(0x00000000u);
  svg1=__uint_as_float(0x00000000u);
  svg2=__uint_as_float(0x00000000u);
  tvgp=tvg;
  // source line 1009
  tgep=tge;
  // source line 1010
  cmcgp=cmcg;
  // source line 1011
  for(int vi=0;vi<4;++vi) smgp[vi]=smg[vi];
  // source line 1012
  ttp=tt;
  // source line 1013
  for(int vi=0;vi<4;++vi) srootp[vi]=sroot[vi];
  // source line 1018
  if (((FADD(zr,__uint_as_float(0x40000000u))<za))) {
  // source line 1019
  ur=FDIV(FMUL(ua,gfk_log(FDIV(u480_sub(zr,zdc),z0c))),gfk_log(FDIV(u480_sub(za,zdc),z0c)));
  // source line 1020
  zc=FMUL(__uint_as_float(0x3f333333u),zr);
  // source line 1021
  xlb=FMUL(__uint_as_float(0x3ecccccdu),u480_sub(zr,zdc));
  // source line 1026
  bb=FDIV(FMUL(__uint_as_float(0x3ecccccdu),zr),FMUL(xlb,gfk_log(FDIV(u480_sub(zr,zdc),z0c))));
  // source line 1028
  uc=FMUL(ur,gfk_exp(FMUL((-bb),u480_sub(__uint_as_float(0x3f800000u),FDIV(zc,zr)))));
  // source line 1029
  } else {
  // source line 1031
  zc=FDIV(za,__uint_as_float(0x40000000u));
  // source line 1032
  uc=FDIV(ua,__uint_as_float(0x40000000u));
  // source line 1033
  }
  // source line 1039
  shadow=true;
  // source line 1043
  tau=gfk_exp(FMUL(__uint_as_float(0xbf1c28f6u),lai_tree));
  // source line 1044
  if (((treeoption==1))) {
  // source line 1045
  u480_vf_tree_analytical(rw,hgt,htree,rtree,tau,vfgs_tree,vfgw_tree,vfgt_tree,vfws_tree,vfwg_tree,vfww_tree,vfwt_tree,vfts_tree,vftg_tree,vftw_tree,env,status);
  // source line 1046
  }
  // source line 1048
  if (((ssg>__uint_as_float(0x00000000u)))) {
  // source line 1050
  if (shadow) {
  // source line 1051
  fai=FDIV(FMUL(xlat,__uint_as_float(0x40490fd0u)),__uint_as_float(0x43340000u));
  // source line 1052
  theatas=fabsf(glibc_asinf(cosz));
  // source line 1053
  theataz=fabsf(glibc_acosf(cosz));
  // source line 1055
  snt=FDIV(FMUL(glibc_cosf(declin),glibc_sinf(omg)),glibc_cosf(theatas));
  // source line 1056
  cnt=FDIV(FDIV(u480_sub(FMUL(cosz,glibc_sinf(fai)),glibc_sinf(declin)),glibc_cosf(theatas)),glibc_cosf(fai));
  // source line 1058
  houi1=u480_sub(FMUL(snt,__uint_as_float(0x3f6c8361u)),FMUL(cnt,__uint_as_float(0x3ec3ef0bu)));
  // source line 1059
  houi2=u480_sub(FMUL(snt,__uint_as_float(0x3f3504fbu)),FMUL(cnt,__uint_as_float(0x3f3504ecu)));
  // source line 1060
  houi3=u480_sub(FMUL(snt,__uint_as_float(0x3ec3ef33u)),FMUL(cnt,__uint_as_float(0x3f6c8358u)));
  // source line 1061
  houi4=u480_sub(FMUL(snt,__uint_as_float(0x35aa2217u)),FMUL(cnt,__uint_as_float(0x3f800000u)));
  // source line 1062
  houi5=u480_sub(FMUL(snt,__uint_as_float(0xbec3eee4u)),FMUL(cnt,__uint_as_float(0x3f6c8369u)));
  // source line 1063
  houi6=u480_sub(FMUL(snt,__uint_as_float(0xbf3504ddu)),FMUL(cnt,__uint_as_float(0x3f35050au)));
  // source line 1064
  houi7=u480_sub(FMUL(snt,__uint_as_float(0xbf6c8350u)),FMUL(cnt,__uint_as_float(0x3ec3ef5au)));
  // source line 1065
  houi8=u480_sub(FMUL(snt,__uint_as_float(0xbf800000u)),FMUL(cnt,__uint_as_float(0x362a2217u)));
  // source line 1067
  slx1=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi1));
  // source line 1068
  slx2=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi2));
  // source line 1069
  slx3=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi3));
  // source line 1070
  slx4=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi4));
  // source line 1071
  slx5=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi5));
  // source line 1072
  slx6=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi6));
  // source line 1073
  slx7=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi7));
  // source line 1074
  slx8=FMUL(FMUL(hgt,fabsf(glibc_tanf(theataz))),fabsf(houi8));
  // source line 1076
  if (((slx1>rw))) {
  slx1=rw;
  }
  // source line 1077
  if (((slx2>rw))) {
  slx2=rw;
  }
  // source line 1078
  if (((slx3>rw))) {
  slx3=rw;
  }
  // source line 1079
  if (((slx4>rw))) {
  slx4=rw;
  }
  // source line 1080
  if (((slx5>rw))) {
  slx5=rw;
  }
  // source line 1081
  if (((slx6>rw))) {
  slx6=rw;
  }
  // source line 1082
  if (((slx7>rw))) {
  slx7=rw;
  }
  // source line 1083
  if (((slx8>rw))) {
  slx8=rw;
  }
  // source line 1085
  slx=FDIV(FADD(FADD(FADD(FADD(FADD(FADD(FADD(slx1,slx2),slx3),slx4),slx5),slx6),slx7),slx8),__uint_as_float(0x41000000u));
  // source line 1086
  }
  // source line 1088
  if ((!shadow)) {
  // source line 1090
  sr1=FMUL(sx,u480_sub(__uint_as_float(0x3f800000u),albr));
  // source line 1091
  sgr1=FMUL(sx,__uint_as_float(0x3f4ccccdu));
  // source line 1092
  sg1=FMUL(FMUL(sx,vfgs),u480_sub(__uint_as_float(0x3f800000u),albg));
  // source line 1093
  sb1=FMUL(FMUL(sx,vfws),u480_sub(__uint_as_float(0x3f800000u),albb));
  // source line 1094
  sg2=FMUL(FMUL(FDIV(FMUL(sb1,albb),u480_sub(__uint_as_float(0x3f800000u),albb)),vfgw),u480_sub(__uint_as_float(0x3f800000u),albg));
  // source line 1095
  sb2=FADD(FMUL(FMUL(FDIV(FMUL(sg1,albg),u480_sub(__uint_as_float(0x3f800000u),albg)),vfwg),u480_sub(__uint_as_float(0x3f800000u),albb)),FMUL(FMUL(sb1,albb),vfww));
  // source line 1097
  } else if (((treeoption!=1))) {
  // source line 1099
  sr1=FADD(FMUL(sd,u480_sub(__uint_as_float(0x3f800000u),albr)),FMUL(sq,u480_sub(__uint_as_float(0x3f800000u),albr)));
  // source line 1100
  sgr1=FADD(FMUL(sd,__uint_as_float(0x3f4ccccdu)),FMUL(sq,__uint_as_float(0x3f4ccccdu)));
  // source line 1101
  sg1=FADD(FMUL(FDIV(FMUL(sd,u480_sub(rw,slx)),rw),u480_sub(__uint_as_float(0x3f800000u),albg)),FMUL(FMUL(sq,vfgs),u480_sub(__uint_as_float(0x3f800000u),albg)));
  // source line 1102
  sb1=FADD(FMUL(FDIV(FMUL(sd,slx),w),u480_sub(__uint_as_float(0x3f800000u),albb)),FMUL(FMUL(sq,vfws),u480_sub(__uint_as_float(0x3f800000u),albb)));
  // source line 1103
  sg2=FMUL(FMUL(FDIV(FMUL(sb1,albb),u480_sub(__uint_as_float(0x3f800000u),albb)),vfgw),u480_sub(__uint_as_float(0x3f800000u),albg));
  // source line 1104
  sb2=FADD(FMUL(FMUL(FDIV(FMUL(sg1,albg),u480_sub(__uint_as_float(0x3f800000u),albg)),vfwg),u480_sub(__uint_as_float(0x3f800000u),albb)),FMUL(FMUL(sb1,albb),vfww));
  // source line 1105
  } else {
  // source line 1109
  if (((FADD(htree,rtree)>zr))) {
  // source line 1110
  status=1; return;
  // source line 1111
  }
  // source line 1113
  albge=FADD(FMUL(u480_sub(__uint_as_float(0x3f800000u),fvg),albg),FMUL(fvg,albvg));
  // source line 1114
  xi=FDIV(slx,hgt);
  // source line 1116
  angle=u480_sub(FADD(gfk_pow(dtree,__uint_as_float(0x40000000u)),gfk_pow(u480_sub(hgt,htree),__uint_as_float(0x40000000u))),gfk_pow(rtree,__uint_as_float(0x40000000u)));
  // source line 1117
  tan1=FDIV(FADD(FMUL(rtree,u480_sub(hgt,htree)),FMUL(dtree,FSQRT(angle))),u480_sub(FMUL(u480_sub(hgt,htree),FSQRT(angle)),FMUL(rtree,dtree)));
  // source line 1118
  tan2=FDIV(FADD(FMUL((-rtree),u480_sub(hgt,htree)),FMUL(dtree,FSQRT(angle))),FADD(FMUL(u480_sub(hgt,htree),FSQRT(angle)),FMUL(rtree,dtree)));
  // source line 1120
  if (((theataz>__uint_as_float(0x3fc90fd0u)))) {
  // source line 1121
  sdt=0;
  // source line 1122
  sdg=0;
  // source line 1123
  sdb=0;
  // source line 1124
  } else if (((xi>=tan1))) {
  // source line 1125
  sdt=0;
  // source line 1126
  } else if ((((xi>=tan2)) && ((xi<tan1)))) {
  // source line 1127
  sdt=FDIV(FMUL(sd,u480_sub(FADD(FMUL(rtree,FSQRT(FADD(1,gfk_pow(xi,__uint_as_float(0x40000000u))))),dtree),FMUL(u480_sub(hgt,htree),xi))),FMUL(__uint_as_float(0x40c90fd0u),rtree));
  // source line 1128
  } else {
  // source line 1129
  sdt=FDIV(FMUL(sd,FMUL(FMUL(__uint_as_float(0x40000000u),rtree),FSQRT(FADD(__uint_as_float(0x3f800000u),gfk_pow(xi,__uint_as_float(0x40000000u)))))),FMUL(__uint_as_float(0x40c90fd0u),rtree));
  // source line 1130
  }
  // source line 1131
  // source line 1134
  u480_shadow_tree(xi,hgt,rw,htree,rtree,dtree,chi_shaded,eta_shaded,chi_shaded_tree,eta_shaded_tree,env,status);
  // source line 1137
  sdg=FDIV(FMUL(sd,FADD(u480_sub(rw,chi_shaded),FMUL(chi_shaded_tree,tau))),rw);
  // source line 1138
  sdb=FDIV(FMUL(FMUL(sd,xi),FADD(u480_sub(hgt,eta_shaded),FMUL(eta_shaded_tree,tau))),w);
  // source line 1140
  sr1=FADD(FMUL(sd,u480_sub(__uint_as_float(0x3f800000u),albr)),FMUL(sq,u480_sub(__uint_as_float(0x3f800000u),albr)));
  // source line 1141
  sgr1=FADD(FMUL(sd,__uint_as_float(0x3f4ccccdu)),FMUL(sq,__uint_as_float(0x3f4ccccdu)));
  // source line 1142
  sb1=FMUL(FADD(sdb,FMUL(sq,vfws_tree)),u480_sub(__uint_as_float(0x3f800000u),albb));
  // source line 1143
  sb2=FADD(FADD(FMUL(FMUL(FMUL(FADD(sdb,FMUL(sq,vfws_tree)),albb),vfww_tree),u480_sub(1,albb)),FMUL(FMUL(FMUL(FADD(sdt,FMUL(sq,vfts_tree)),albt),vfwt_tree),u480_sub(1,albb))),FMUL(FMUL(FMUL(FADD(sdg,FMUL(sq,vfgs_tree)),albge),vfwg_tree),u480_sub(1,albb)));
  // source line 1146
  sg1=FMUL(FADD(sdg,FMUL(sq,vfgs_tree)),u480_sub(1,albg));
  // source line 1147
  sg2=FADD(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40000000u),FADD(sdb,FMUL(sq,vfws_tree))),albb),vfgw_tree),u480_sub(1,albg)),FMUL(FMUL(FMUL(FADD(sdt,FMUL(sq,vfts_tree)),albt),vfgt_tree),u480_sub(1,albg)));
  // source line 1149
  svg1=FMUL(FADD(sdg,FMUL(sq,vfgs_tree)),u480_sub(1,albvg));
  // source line 1150
  svg2=FADD(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40000000u),FADD(sdb,FMUL(sq,vfws_tree))),albb),vfgw_tree),u480_sub(1,albvg)),FMUL(FMUL(FMUL(FADD(sdt,FMUL(sq,vfts_tree)),albt),vfgt_tree),u480_sub(1,albvg)));
  // source line 1152
  st1=FMUL(FADD(sdt,FMUL(sq,vfts_tree)),u480_sub(1,albt));
  // source line 1153
  st2=FADD(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40000000u),FADD(sdb,FMUL(sq,vfws_tree))),albb),vftw_tree),u480_sub(1,albt)),FMUL(FMUL(FMUL(FADD(sdg,FMUL(sq,vfgs_tree)),albg),vftg_tree),u480_sub(1,albt)));
  // source line 1155
  }
  // source line 1157
  sr=sr1;
  // source line 1158
  sgr=sgr1;
  // source line 1159
  sg=FADD(sg1,sg2);
  // source line 1160
  svg=FADD(svg1,svg2);
  // source line 1161
  sb=FADD(sb1,sb2);
  // source line 1163
  if (((treeoption==1))) {
  // source line 1164
  st=FMUL(FMUL(FMUL(FMUL(FADD(st1,st2),__uint_as_float(0x40000000u)),__uint_as_float(0x40490fd0u)),rtree),u480_sub(1,tau));
  // source line 1165
  sleaf=FMUL(FMUL(FDIV(st,FMUL(FMUL(__uint_as_float(0x40000000u),rtree),lai_tree)),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 1166
  } else {
  // source line 1167
  st=0;
  // source line 1168
  }
  // source line 1171
  if ((((groption==1)) && ((treeoption!=1)))) {
  // source line 1172
  snet=FADD(FADD(FADD(FADD(FMUL(FMUL(r,fgr),sgr),FMUL(FMUL(r,u480_sub(__uint_as_float(0x3f800000u),fgr)),sr)),FMUL(w,sb)),FMUL(rw,sg)),st);
  // source line 1173
  } else if ((((treeoption==1)) && ((groption!=1)))) {
  // source line 1174
  snet=FADD(FADD(FADD(FADD(FMUL(r,sr),FMUL(w,sb)),FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fvg),rw),sg)),FMUL(FMUL(fvg,rw),svg)),st);
  // source line 1175
  } else if ((((treeoption==1)) && ((groption==1)))) {
  // source line 1176
  snet=FADD(FADD(FADD(FADD(FADD(FMUL(FMUL(r,fgr),sgr),FMUL(FMUL(r,u480_sub(__uint_as_float(0x3f800000u),fgr)),sr)),FMUL(w,sb)),FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fvg),rw),sg)),FMUL(FMUL(fvg,rw),svg)),st);
  // source line 1177
  } else {
  // source line 1178
  snet=FADD(FADD(FADD(FMUL(r,sr),FMUL(w,sb)),FMUL(rw,sg)),st);
  // source line 1179
  }
  // source line 1182
  } else {
  // source line 1184
  sr=__uint_as_float(0x00000000u);
  // source line 1185
  sg=__uint_as_float(0x00000000u);
  // source line 1186
  sgr=__uint_as_float(0x00000000u);
  // source line 1187
  svg=__uint_as_float(0x00000000u);
  // source line 1188
  sb=__uint_as_float(0x00000000u);
  // source line 1189
  st=__uint_as_float(0x00000000u);
  // source line 1190
  snet=__uint_as_float(0x00000000u);
  // source line 1192
  }
  // source line 1209
  t1vr=FMUL(trp,FADD(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x3f1c28f6u),qa)));
  // source line 1210
  th2v=FMUL(FADD(ta,FMUL(__uint_as_float(0x3c20902eu),za)),FADD(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x3f1c28f6u),qa)));
  // source line 1213
  rlmo_urb=__uint_as_float(0x00000000u);
  {
  // source line 1226
  u480_sfcdif_urb(za,z0r,t1vr,th2v,ua,akanda_urban,cmr_urb,chr_urb,rlmo_urb,cdr,optional_zt,optional_veg,false,env,status);
  // source line 1227
  alphar=FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chr_urb);
  // source line 1228
  chr=FDIV(FDIV(FDIV(alphar,rho),__uint_as_float(0x3e75c28fu)),ua);
  // source line 1229
  }
  // source line 1232
  rain1=FDIV(FMUL(rain,__uint_as_float(0x3a83126fu)),3600);
  // source line 1233
  if (((imp_scheme==1))) {
  // source line 1234
  if (((rain>__uint_as_float(0x3f800000u)))) {
  betr=__uint_as_float(0x3f333333u);
  }
  // source line 1235
  }
  // source line 1237
  if (((imp_scheme==2))) {
  // source line 1238
  if (((flxhumrp<=__uint_as_float(0x00000000u)))) {
  flxhumrp=__uint_as_float(0x00000000u);
  }
  // source line 1241
  drelr=FADD(drelrp,FDIV(FMUL(u480_sub(rain1,FDIV(FMUL(flxhumrp,rhoo),__uint_as_float(0x447a0000u))),delt),porimp[(1)-1]));
  if(env.sw[13]==0 && rain>0.0f && drelr<drelrp) drelr=drelrp;
  // source line 1244
  if (((drelr<=__uint_as_float(0x00000000u)))) {
  // source line 1245
  drelr=__uint_as_float(0x00000000u);
  // source line 1246
  betr=__uint_as_float(0x00000000u);
  // source line 1247
  } else if (((drelr<=dengimp[(1)-1]))) {
  // source line 1248
  betr=FMUL(FDIV(drelr,dengimp[(1)-1]),porimp[(1)-1]);
  // source line 1249
  } else {
  // source line 1250
  drelr=dengimp[(1)-1];
  // source line 1251
  betr=porimp[(1)-1];
  // source line 1252
  }
  // source line 1254
  if (((betr<__uint_as_float(0x3727c5acu)))) {
  betr=__uint_as_float(0x00000000u);
  }
  // source line 1255
  }
  // source line 1257
  if (((ts_scheme==1))) {
  // source line 1276
  for (iteration=1;iteration<=20;iteration+=1) {
  // source line 1278
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(trp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),trp))));
  // source line 1279
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),gfk_pow(trp,__uint_as_float(0x40000000u)));
  // source line 1280
  qs0r=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1281
  dqs0rdtr=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),gfk_pow(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),__uint_as_float(0x40000000u)));
  // source line 1283
  rr=FMUL(epsr,u480_sub(rx,FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(trp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1284
  hr=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chr),ua),u480_sub(trp,ta)),__uint_as_float(0x42c80000u));
  // source line 1285
  eler=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chr),ua),betr),u480_sub(qs0r,qa)),__uint_as_float(0x42c80000u));
  // source line 1286
  g0r=FDIV(FMUL(aksr,u480_sub(trp,trl[(1)-1])),FDIV(dzr[(1)-1],__uint_as_float(0x40000000u)));
  // source line 1288
  f=u480_sub(u480_sub(u480_sub(FADD(sr,rr),hr),eler),g0r);
  // source line 1290
  drrdtr=FDIV(FMUL(FMUL(FMUL(__uint_as_float(0xc0800000u),epsr),__uint_as_float(0x2eb3a903u)),gfk_pow(trp,__uint_as_float(0x40400000u))),__uint_as_float(0x42700000u));
  // source line 1291
  dhrdtr=FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chr),ua),__uint_as_float(0x42c80000u));
  // source line 1292
  delerdtr=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chr),ua),betr),dqs0rdtr),__uint_as_float(0x42c80000u));
  // source line 1293
  dg0rdtr=FDIV(FMUL(__uint_as_float(0x40000000u),aksr),dzr[(1)-1]);
  // source line 1295
  dfdt=u480_sub(u480_sub(u480_sub(drrdtr,dhrdtr),delerdtr),dg0rdtr);
  // source line 1296
  dtr=FDIV(f,dfdt);
  // source line 1298
  tr=u480_sub(trp,dtr);
  // source line 1299
  trp=tr;
  // source line 1301
  if ((((fabsf(f)<__uint_as_float(0x358637bdu))) && ((fabsf(dtr)<__uint_as_float(0x358637bdu))))) {
  break;
  }
  // source line 1303
  }
  // source line 1307
  u480_multi_layer(num_roof_layers,boundr,g0r,capr,aksr,trl,dzr,delt,trlend,env,status);
  // source line 1309
  } else {
  // source line 1311
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(trp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),trp))));
  // source line 1312
  qs0r=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1314
  rr=FMUL(epsr,u480_sub(rx,FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(trp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1315
  hr=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chr),ua),u480_sub(trp,ta)),__uint_as_float(0x42c80000u));
  // source line 1316
  eler=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chr),ua),betr),u480_sub(qs0r,qa)),__uint_as_float(0x42c80000u));
  // source line 1317
  g0r=u480_sub(u480_sub(FADD(sr,rr),hr),eler);
  // source line 1319
  u480_force_restore(capr,aksr,delt,sr,rr,hr,eler,trlend,trp,tr,env,status);
  // source line 1321
  trp=tr;
  // source line 1323
  }
  // source line 1325
  flxthr=FDIV(FDIV(FDIV(hr,rho),__uint_as_float(0x3e75c28fu)),__uint_as_float(0x42c80000u));
  // source line 1326
  flxhumr=FDIV(FDIV(FDIV(eler,rho),__uint_as_float(0x4411c000u)),__uint_as_float(0x42c80000u));
  // source line 1332
  if (((groption==1))) {
  // source line 1333
  t1vgr=FMUL(tgrp,FADD(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x3f1c28f6u),qa)));
  // source line 1334
  rlmo_urb=__uint_as_float(0x00000000u);
  // source line 1335
  u480_sfcdif_urb(za,z0r,t1vgr,th2v,ua,akanda_urban,cmgr_urb,chgr_urb,rlmo_urb,cdgr,optional_zt,optional_veg,false,env,status);
  // source line 1336
  alphagr=FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chgr_urb);
  // source line 1337
  chgr=FDIV(FDIV(FDIV(alphagr,rho),__uint_as_float(0x3e75c28fu)),ua);
  // source line 1338
  runoff1=__uint_as_float(0x00000000u);
  // source line 1339
  runoff2=__uint_as_float(0x00000000u);
  // source line 1340
  runoff3=__uint_as_float(0x00000000u);
  // source line 1341
  for(int vi=0;vi<4;++vi) sroot[vi]=__uint_as_float(0x00000000u);
  // source line 1343
  kz=1;
  // source line 1344
  zsoilr[(kz)-1]=(-dzgr[(kz)-1]);
  // source line 1345
  for (kz=2;kz<=4;kz+=1) {
  // source line 1346
  zsoilr[(kz)-1]=FADD((-dzgr[(kz)-1]),zsoilr[((kz-1))-1]);
  // source line 1347
  }
  // source line 1349
  for (iteration=1;iteration<=100;iteration+=1) {
  // source line 1350
  kz=1;
  // source line 1351
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tgrp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tgrp))));
  // source line 1352
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),gfk_pow(tgrp,__uint_as_float(0x40000000u)));
  // source line 1353
  qs0gr=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1354
  dqs0grdtgr=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),gfk_pow(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),__uint_as_float(0x40000000u)));
  // source line 1355
  epgr=FMUL(FMUL(FMUL(rhoo,chgr),ua),u480_sub(qs0gr,qa));
  // source line 1357
  if (((epgr>__uint_as_float(0x00000000u)))) {
  // source line 1359
  u480_direvap(edir,epgr,smrp[(kz)-1],__uint_as_float(0x3f4ccccdu),__uint_as_float(0x3ee0c49cu),__uint_as_float(0x3d872b02u),__uint_as_float(0x40000000u),env,status);
  // source line 1361
  u480_transp(ettr,etr,ecr,__uint_as_float(0x3f4ccccdu),epgr,cmcrp,__uint_as_float(0x3f000000u),__uint_as_float(0x3a03126fu),__uint_as_float(0x3fc00000u),__uint_as_float(0x42c80000u),__uint_as_float(0x459c4000u),__uint_as_float(0x42c80000u),sx,tgrp,ta,qa,smrp,__uint_as_float(0x3dac0831u),__uint_as_float(0x3ea872b0u),__uint_as_float(0x447b2000u),ps,chgr,__uint_as_float(0x3f6e147bu),delt,2,4,dzgr,zsoilr,__uint_as_float(0x42100000u),env,status);
  // source line 1365
  u480_smflx(smrp,smr,4,cmcrp,cmcr,delt,rain,zsoilr,__uint_as_float(0x3ee0c49cu),__uint_as_float(0x40a80000u),__uint_as_float(0x3dac0831u),sroot,__uint_as_float(0x3662d3f5u),__uint_as_float(0x376fea08u),__uint_as_float(0x3f4ccccdu),__uint_as_float(0x3a03126fu),runoff1,runoff2,runoff3,edir,ecr,etr,drip,env,status);
  // source line 1367
  } else {
  // source line 1368
  dew=(-epgr);
  // source line 1369
  raindr=FADD(rain,FMUL(dew,__uint_as_float(0x45610000u)));
  // source line 1370
  edir=__uint_as_float(0x00000000u);
  // source line 1371
  ecr=__uint_as_float(0x00000000u);
  // source line 1372
  ettr=__uint_as_float(0x00000000u);
  // source line 1373
  u480_smflx(smrp,smr,4,cmcrp,cmcr,delt,raindr,zsoilr,__uint_as_float(0x3ee0c49cu),__uint_as_float(0x40a80000u),__uint_as_float(0x3dac0831u),sroot,__uint_as_float(0x3662d3f5u),__uint_as_float(0x376fea08u),__uint_as_float(0x3f4ccccdu),__uint_as_float(0x3a03126fu),runoff1,runoff2,runoff3,edir,ecr,etr,drip,env,status);
  // source line 1375
  }
  // source line 1379
  edir=FMUL(edir,__uint_as_float(0x447a0000u));
  // source line 1380
  ettr=FMUL(ettr,__uint_as_float(0x447a0000u));
  // source line 1381
  ecr=FMUL(ecr,__uint_as_float(0x447a0000u));
  // source line 1382
  etar=FADD(FADD(edir,ettr),ecr);
  // source line 1383
  if (((etar<__uint_as_float(0x1e3ce508u)))) {
  etar=__uint_as_float(0x00000000u);
  }
  // source line 1385
  if (((epgr<=__uint_as_float(0x00000000u)))) {
  // source line 1386
  betgr=__uint_as_float(0x00000000u);
  // source line 1387
  } else {
  // source line 1388
  betgr=FDIV(etar,epgr);
  // source line 1389
  }
  // source line 1390
  elegr=FMUL(FDIV(FMUL(FMUL(etar,rho),__uint_as_float(0x4411c000u)),rhoo),100);
  // source line 1392
  u480_tdfcnd(df1,smr[(kz)-1],__uint_as_float(0x3ecccccdu),__uint_as_float(0x3ee0c49cu),env,status);
  // source line 1393
  df1=FMUL(df1,__uint_as_float(0x3e4ebdf6u));
  // source line 1394
  rgr=FMUL(__uint_as_float(0x3f6e147bu),u480_sub(rx,FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tgrp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1395
  rgrr=FMUL(FMUL(FADD(sgr,rgr),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 1396
  rch=FMUL(FMUL(rhoo,__uint_as_float(0x447b2000u)),chgr);
  // source line 1397
  rr1=FADD(FDIV(FMUL(FMUL(__uint_as_float(0x3f6e147bu),u_powi(ta,4)),__uint_as_float(0x338b282du)),FMUL(ps,chgr)),__uint_as_float(0x3f800000u));
  // source line 1398
  if (((rain>__uint_as_float(0x00000000u)))) {
  // source line 1399
  rr2=FADD(rr1,FDIV(FMUL(FDIV(rain,3600),__uint_as_float(0x4583d000u)),rch));
  // source line 1400
  } else {
  // source line 1401
  rr2=rr1;
  // source line 1402
  }
  // source line 1403
  yy=FADD(ta,FDIV(u480_sub(FDIV(rgrr,rch),FDIV(FMUL(FMUL(betgr,epgr),__uint_as_float(0x4a150c40u)),rch)),rr2));
  // source line 1404
  zz1=FADD(FDIV(df1,FMUL(FMUL(FMUL(__uint_as_float(0xbf000000u),zsoilr[(kz)-1]),rch),rr2)),__uint_as_float(0x3f800000u));
  // source line 1407
  hgr=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chgr),ua),u480_sub(tgrp,ta)),__uint_as_float(0x42c80000u));
  // source line 1408
  runoff3=FDIV(runoff3,delt);
  // source line 1409
  runoff2=FADD(runoff2,runoff3);
  // source line 1410
  g0gr=FDIV(FDIV(FDIV(FMUL(df1,u480_sub(tgrp,tgrl[(1)-1])),FDIV(dzgr[(1)-1],__uint_as_float(0x40000000u))),__uint_as_float(0x442e6ccdu)),60);
  // source line 1412
  fv=u480_sub(u480_sub(u480_sub(FADD(sgr,rgr),hgr),elegr),g0gr);
  // source line 1413
  drrdtgr=FDIV(FMUL(__uint_as_float(0xafa71580u),gfk_pow(tgrp,__uint_as_float(0x40400000u))),__uint_as_float(0x42700000u));
  // source line 1414
  dhrdtgr=FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chgr),ua),__uint_as_float(0x42c80000u));
  // source line 1415
  delerdtgr=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chgr),ua),betgr),dqs0grdtgr),__uint_as_float(0x42c80000u));
  // source line 1416
  dg0rdtgr=FMUL(FMUL(FDIV(FMUL(__uint_as_float(0x40000000u),df1),dzgr[(kz)-1]),__uint_as_float(0x3e749405u)),__uint_as_float(0x38d1b717u));
  // source line 1417
  dfdvt=u480_sub(u480_sub(u480_sub(drrdtgr,dhrdtgr),delerdtgr),dg0rdtgr);
  // source line 1418
  dtgr=FDIV(FDIV(fv,dfdvt),6);
  // source line 1419
  tgr=u480_sub(tgrp,dtgr);
  // source line 1420
  tgrp=tgr;
  // source line 1422
  if ((((fabsf(fv)<__uint_as_float(0x38d1b717u))) && ((fabsf(dtgr)<__uint_as_float(0x3a83126fu))))) {
  // source line 1423
  break;
  // source line 1424
  }
  // source line 1425
  }
  // source line 1427
  u480_shflx(ssoilr,tgrl,smr,__uint_as_float(0x3ee0c49cu),4,tgrp,delt,yy,zz1,zsoilr,trlend,__uint_as_float(0xc0000000u),__uint_as_float(0x3dac0831u),df1,__uint_as_float(0x3ecccccdu),__uint_as_float(0x49f42400u),capr,env,status);
  // source line 1429
  flxthgr=FDIV(FDIV(FDIV(hgr,rho),__uint_as_float(0x3e75c28fu)),__uint_as_float(0x42c80000u));
  // source line 1430
  flxhumgr=FDIV(FDIV(FDIV(elegr,rho),__uint_as_float(0x4411c000u)),__uint_as_float(0x42c80000u));
  // source line 1431
  } else {
  // source line 1432
  flxthgr=__uint_as_float(0x00000000u);
  // source line 1433
  flxhumgr=__uint_as_float(0x00000000u);
  // source line 1434
  }
  // source line 1451
  if ((!distributed_aerodynamics_option)) {
  // source line 1453
  t1vc=FMUL(tcp,FADD(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x3f1c28f6u),qa)));
  // source line 1454
  rlmo_urb=__uint_as_float(0x00000000u);
  // source line 1455
  u480_sfcdif_urb(za,z0c,t1vc,th2v,ua,akanda_urban,cmc_urb,chc_urb,rlmo_urb,cdc,optional_zt,optional_veg,false,env,status);
  // source line 1456
  alphac=FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chc_urb);
  // source line 1458
  if (((ch_scheme==1))) {
  // source line 1460
  z=zdc;
  // source line 1461
  bhb=FDIV(gfk_log(FDIV(z0b,z0hb)),__uint_as_float(0x3ecccccdu));
  // source line 1462
  bhg=FDIV(gfk_log(FDIV(z0g,z0hg)),__uint_as_float(0x3ecccccdu));
  // source line 1463
  ribb=FDIV(FMUL(FMUL(FDIV(__uint_as_float(0x419ccccdu),FADD(tcp,tbp)),u480_sub(tcp,tbp)),FADD(z,z0b)),FMUL(uc,uc));
  // source line 1464
  ribg=FDIV(FMUL(FMUL(FDIV(__uint_as_float(0x419ccccdu),FADD(tcp,tgp)),u480_sub(tcp,tgp)),FADD(z,z0g)),FMUL(uc,uc));
  // source line 1466
  u480_mos(xxxb,alphab,cdb,bhb,ribb,z,z0b,uc,tcp,tbp,rho,env,status);
  // source line 1467
  u480_mos(xxxg,alphag,cdg,bhg,ribg,z,z0g,uc,tcp,tgp,rho,env,status);
  // source line 1469
  } else {
  // source line 1471
  alphab=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),FADD(__uint_as_float(0x40c4cccdu),FMUL(__uint_as_float(0x4085c28fu),uc))),__uint_as_float(0x44960000u));
  // source line 1472
  if (((uc>__uint_as_float(0x40a00000u)))) {
  alphab=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),FMUL(__uint_as_float(0x40f051ecu),gfk_pow(uc,__uint_as_float(0x3f47ae14u)))),__uint_as_float(0x44960000u));
  }
  // source line 1473
  alphag=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),FADD(__uint_as_float(0x40c4cccdu),FMUL(__uint_as_float(0x4085c28fu),uc))),__uint_as_float(0x44960000u));
  // source line 1474
  if (((uc>__uint_as_float(0x40a00000u)))) {
  alphag=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),FMUL(__uint_as_float(0x40f051ecu),gfk_pow(uc,__uint_as_float(0x3f47ae14u)))),__uint_as_float(0x44960000u));
  }
  // source line 1476
  }
  // source line 1478
  chc=FDIV(FDIV(FDIV(alphac,rho),__uint_as_float(0x3e75c28fu)),ua);
  // source line 1479
  chb=FDIV(FDIV(FDIV(alphab,rho),__uint_as_float(0x3e75c28fu)),uc);
  // source line 1480
  chg=FDIV(FDIV(FDIV(alphag,rho),__uint_as_float(0x3e75c28fu)),uc);
  // source line 1481
  }
  // source line 1484
  if (((imp_scheme==1))) {
  // source line 1485
  betb=__uint_as_float(0x00000000u);
  // source line 1486
  if (((rain>__uint_as_float(0x3f800000u)))) {
  betg=__uint_as_float(0x3f333333u);
  }
  // source line 1487
  }
  // source line 1489
  if (((imp_scheme==2))) {
  // source line 1490
  if (((flxhumbp<=__uint_as_float(0x00000000u)))) {
  flxhumbp=__uint_as_float(0x00000000u);
  }
  // source line 1491
  if (((flxhumgp<=__uint_as_float(0x00000000u)))) {
  flxhumgp=__uint_as_float(0x00000000u);
  }
  // source line 1493
  drelb=FADD(drelbp,FDIV(FMUL(u480_sub(rain1,FDIV(FMUL(flxhumbp,rhoo),__uint_as_float(0x447a0000u))),delt),porimp[(2)-1]));
  if(env.sw[13]==0 && rain>0.0f && drelb<drelbp) drelb=drelbp;
  // source line 1495
  drelg=FADD(drelgp,FDIV(FMUL(u480_sub(rain1,FDIV(FMUL(flxhumgp,rhoo),__uint_as_float(0x447a0000u))),delt),porimp[(3)-1]));
  if(env.sw[13]==0 && rain>0.0f && drelg<drelgp) drelg=drelgp;
  // source line 1498
  if (((drelb<=__uint_as_float(0x00000000u)))) {
  // source line 1499
  drelb=__uint_as_float(0x00000000u);
  // source line 1500
  betb=__uint_as_float(0x00000000u);
  // source line 1501
  } else if (((drelb<=dengimp[(2)-1]))) {
  // source line 1502
  betb=FMUL(FDIV(drelb,dengimp[(2)-1]),porimp[(2)-1]);
  // source line 1503
  } else {
  // source line 1504
  drelb=dengimp[(2)-1];
  // source line 1505
  betb=porimp[(2)-1];
  // source line 1506
  }
  // source line 1508
  if (((drelg<=__uint_as_float(0x00000000u)))) {
  // source line 1509
  drelg=__uint_as_float(0x00000000u);
  // source line 1510
  betg=__uint_as_float(0x00000000u);
  // source line 1511
  } else if (((drelg<=dengimp[(3)-1]))) {
  // source line 1512
  betg=FMUL(FDIV(drelg,dengimp[(3)-1]),porimp[(3)-1]);
  // source line 1513
  } else {
  // source line 1514
  drelg=dengimp[(3)-1];
  // source line 1515
  betg=porimp[(3)-1];
  // source line 1516
  }
  // source line 1518
  if (((betg<__uint_as_float(0x3727c5acu)))) {
  betg=__uint_as_float(0x00000000u);
  }
  // source line 1519
  if (((betb<__uint_as_float(0x3727c5acu)))) {
  betb=__uint_as_float(0x00000000u);
  }
  // source line 1521
  }
  // source line 1527
  if (((treeoption!=1))) {
  // source line 1529
  if (((ts_scheme==1))) {
  // source line 1534
  for (iteration=1;iteration<=20;iteration+=1) {
  // source line 1536
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tbp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tbp))));
  // source line 1537
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),gfk_pow(tbp,__uint_as_float(0x40000000u)));
  // source line 1538
  qs0b=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1539
  dqs0bdtb=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),gfk_pow(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),__uint_as_float(0x40000000u)));
  // source line 1541
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tgp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tgp))));
  // source line 1542
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),gfk_pow(tgp,__uint_as_float(0x40000000u)));
  // source line 1543
  qs0g=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1544
  dqs0gdtg=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),gfk_pow(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),__uint_as_float(0x40000000u)));
  // source line 1546
  rg1=FMUL(epsg,u480_sub(FADD(FMUL(rx,vfgs),FDIV(FMUL(FMUL(FMUL(epsb,vfgw),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1550
  rb1=FMUL(epsb,u480_sub(FADD(FADD(FMUL(rx,vfws),FDIV(FMUL(FMUL(FMUL(epsg,vfwg),__uint_as_float(0x2eb3a903u)),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(FMUL(FMUL(epsb,vfww),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1555
  rg2=FMUL(epsg,FADD(FADD(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),u480_sub(__uint_as_float(0x3f800000u),svf)),vfws),rx),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),u480_sub(__uint_as_float(0x3f800000u),svf)),vfwg),epsg),__uint_as_float(0x2eb3a903u)),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb,u480_sub(__uint_as_float(0x3f800000u),epsb)),u480_sub(__uint_as_float(0x3f800000u),svf)),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1559
  rb2=FMUL(epsb,FADD(FADD(FADD(FADD(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsg),vfwg),vfgs),rx),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsg),epsb),vfgw),vfwg),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),vfws),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),rx)),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),vfwg),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),__uint_as_float(0x2eb3a903u)),epsg),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb,u480_sub(__uint_as_float(0x3f800000u),epsb)),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1565
  rg=FADD(rg1,rg2);
  // source line 1566
  rb=FADD(rb1,rb2);
  // source line 1568
  drbdtb1=FDIV(FMUL(epsb,u480_sub(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),epsb),__uint_as_float(0x2eb3a903u)),gfk_pow(tb,__uint_as_float(0x40400000u))),vfww),FMUL(__uint_as_float(0x2fb3a903u),gfk_pow(tb,__uint_as_float(0x40400000u))))),__uint_as_float(0x42700000u));
  // source line 1569
  drbdtg1=FDIV(FMUL(epsb,FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),epsg),__uint_as_float(0x2eb3a903u)),gfk_pow(tg,__uint_as_float(0x40400000u))),vfwg)),__uint_as_float(0x42700000u));
  // source line 1570
  drbdtb2=FDIV(FMUL(epsb,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),u480_sub(__uint_as_float(0x3f800000u),epsg)),epsb),__uint_as_float(0x2eb3a903u)),gfk_pow(tb,__uint_as_float(0x40400000u))),vfgw),vfwg),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),epsb),u480_sub(__uint_as_float(0x3f800000u),epsb)),__uint_as_float(0x2eb3a903u)),gfk_pow(tb,__uint_as_float(0x40400000u))),vfww),vfww))),__uint_as_float(0x42700000u));
  // source line 1572
  drbdtg2=FDIV(FMUL(epsb,FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),u480_sub(__uint_as_float(0x3f800000u),epsb)),epsg),__uint_as_float(0x2eb3a903u)),gfk_pow(tg,__uint_as_float(0x40400000u))),vfwg),vfww)),__uint_as_float(0x42700000u));
  // source line 1574
  drgdtb1=FDIV(FMUL(epsg,FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),epsb),__uint_as_float(0x2eb3a903u)),gfk_pow(tb,__uint_as_float(0x40400000u))),vfgw)),__uint_as_float(0x42700000u));
  // source line 1575
  drgdtg1=FDIV(FMUL(epsg,FMUL(__uint_as_float(0xafb3a903u),gfk_pow(tg,__uint_as_float(0x40400000u)))),__uint_as_float(0x42700000u));
  // source line 1576
  drgdtb2=FDIV(FMUL(epsg,FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),epsb),u480_sub(__uint_as_float(0x3f800000u),epsb)),__uint_as_float(0x2eb3a903u)),gfk_pow(tb,__uint_as_float(0x40400000u))),vfww),vfgw)),__uint_as_float(0x42700000u));
  // source line 1577
  drgdtg2=FDIV(FMUL(epsg,FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),u480_sub(__uint_as_float(0x3f800000u),epsb)),epsg),__uint_as_float(0x2eb3a903u)),gfk_pow(tg,__uint_as_float(0x40400000u))),vfwg),vfgw)),__uint_as_float(0x42700000u));
  // source line 1579
  drbdtb=FADD(drbdtb1,drbdtb2);
  // source line 1580
  drbdtg=FADD(drbdtg1,drbdtg2);
  // source line 1581
  drgdtb=FADD(drgdtb1,drgdtb2);
  // source line 1582
  drgdtg=FADD(drgdtg1,drgdtg2);
  {
  // source line 1601
  hb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(tbp,tcp)),__uint_as_float(0x42c80000u));
  // source line 1602
  hg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(tgp,tcp)),__uint_as_float(0x42c80000u));
  // source line 1604
  dtcdtb=FDIV(FMUL(w,alphab),FADD(FADD(FMUL(rw,alphac),FMUL(rw,alphag)),FMUL(w,alphab)));
  // source line 1605
  dtcdtg=FDIV(FMUL(rw,alphag),FADD(FADD(FMUL(rw,alphac),FMUL(rw,alphag)),FMUL(w,alphab)));
  // source line 1607
  dhbdtb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(__uint_as_float(0x3f800000u),dtcdtb)),__uint_as_float(0x42c80000u));
  // source line 1608
  dhbdtg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(__uint_as_float(0x00000000u),dtcdtg)),__uint_as_float(0x42c80000u));
  // source line 1609
  dhgdtg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(__uint_as_float(0x3f800000u),dtcdtg)),__uint_as_float(0x42c80000u));
  // source line 1610
  dhgdtb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(__uint_as_float(0x00000000u),dtcdtb)),__uint_as_float(0x42c80000u));
  // source line 1612
  eleb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(qs0b,qcp)),__uint_as_float(0x42c80000u));
  // source line 1613
  eleg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(qs0g,qcp)),__uint_as_float(0x42c80000u));
  // source line 1615
  dqcdtb=FDIV(FMUL(FMUL(FMUL(w,alphab),betb),dqs0bdtb),FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,alphag),betg)),FMUL(FMUL(w,alphab),betb)));
  // source line 1616
  dqcdtg=FDIV(FMUL(FMUL(FMUL(rw,alphag),betg),dqs0gdtg),FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,alphag),betg)),FMUL(FMUL(w,alphab),betb)));
  // source line 1618
  delebdtb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(dqs0bdtb,dqcdtb)),__uint_as_float(0x42c80000u));
  // source line 1619
  delebdtg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(__uint_as_float(0x00000000u),dqcdtg)),__uint_as_float(0x42c80000u));
  // source line 1620
  delegdtg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(dqs0gdtg,dqcdtg)),__uint_as_float(0x42c80000u));
  // source line 1621
  delegdtb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(__uint_as_float(0x00000000u),dqcdtb)),__uint_as_float(0x42c80000u));
  // source line 1622
  }
  // source line 1624
  g0b=FDIV(FMUL(aksb,u480_sub(tbp,tbl[(1)-1])),FDIV(dzb[(1)-1],__uint_as_float(0x40000000u)));
  // source line 1625
  g0g=FDIV(FMUL(aksg,u480_sub(tgp,tgl[(1)-1])),FDIV(dzg[(1)-1],__uint_as_float(0x40000000u)));
  // source line 1627
  dg0bdtb=FDIV(FMUL(__uint_as_float(0x40000000u),aksb),dzb[(1)-1]);
  // source line 1628
  dg0bdtg=__uint_as_float(0x00000000u);
  // source line 1629
  dg0gdtg=FDIV(FMUL(__uint_as_float(0x40000000u),aksg),dzg[(1)-1]);
  // source line 1630
  dg0gdtb=__uint_as_float(0x00000000u);
  // source line 1632
  f=u480_sub(u480_sub(u480_sub(FADD(sb,rb),hb),eleb),g0b);
  // source line 1633
  fx=u480_sub(u480_sub(u480_sub(drbdtb,dhbdtb),delebdtb),dg0bdtb);
  // source line 1634
  fy=u480_sub(u480_sub(u480_sub(drbdtg,dhbdtg),delebdtg),dg0bdtg);
  // source line 1636
  gf=u480_sub(u480_sub(u480_sub(FADD(sg,rg),hg),eleg),g0g);
  // source line 1637
  gx=u480_sub(u480_sub(u480_sub(drgdtb,dhgdtb),delegdtb),dg0gdtb);
  // source line 1638
  gy=u480_sub(u480_sub(u480_sub(drgdtg,dhgdtg),delegdtg),dg0gdtg);
  // source line 1640
  dtb=FDIV(u480_sub(FMUL(gf,fy),FMUL(f,gy)),u480_sub(FMUL(fx,gy),FMUL(gx,fy)));
  // source line 1641
  dtg=FDIV((-FADD(gf,FMUL(gx,dtb))),gy);
  // source line 1643
  tb=FADD(tbp,dtb);
  // source line 1644
  tg=FADD(tgp,dtg);
  // source line 1646
  tbp=tb;
  // source line 1647
  tgp=tg;
  {
  // source line 1652
  tc1=FADD(FADD(FMUL(rw,alphac),FMUL(rw,alphag)),FMUL(w,alphab));
  // source line 1653
  tc2=FADD(FADD(FMUL(FMUL(rw,alphac),ta),FMUL(FMUL(rw,alphag),tgp)),FMUL(FMUL(w,alphab),tbp));
  // source line 1654
  tc=FDIV(tc2,tc1);
  // source line 1656
  qc1=FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,alphag),betg)),FMUL(FMUL(w,alphab),betb));
  // source line 1657
  qc2=FADD(FADD(FMUL(FMUL(rw,alphac),qa),FMUL(FMUL(FMUL(rw,alphag),betg),qs0g)),FMUL(FMUL(FMUL(w,alphab),betb),qs0b));
  // source line 1658
  qc=FDIV(qc2,qc1);
  // source line 1660
  dtc=u480_sub(tcp,tc);
  // source line 1661
  tcp=tc;
  // source line 1662
  qcp=qc;
  // source line 1663
  }
  // source line 1665
  if ((((fabsf(f)<__uint_as_float(0x358637bdu))) && ((fabsf(dtb)<__uint_as_float(0x358637bdu))) && ((fabsf(gf)<__uint_as_float(0x358637bdu))) && ((fabsf(dtg)<__uint_as_float(0x358637bdu))) && ((fabsf(dtc)<__uint_as_float(0x358637bdu))))) {
  break;
  }
  // source line 1669
  }
  // source line 1671
  u480_multi_layer(num_wall_layers,boundb,g0b,capb,aksb,tbl,dzb,delt,tblend,env,status);
  // source line 1673
  u480_multi_layer(num_road_layers,boundg,g0g,capg,aksg,tgl,dzg,delt,tglend,env,status);
  // source line 1674
  } else {
  // source line 1680
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tbp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tbp))));
  // source line 1681
  qs0b=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1683
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tgp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tgp))));
  // source line 1684
  qs0g=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1686
  rg1=FMUL(epsg,u480_sub(FADD(FMUL(rx,vfgs),FDIV(FMUL(FMUL(FMUL(epsb,vfgw),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1690
  rb1=FMUL(epsb,u480_sub(FADD(FADD(FMUL(rx,vfws),FDIV(FMUL(FMUL(FMUL(epsg,vfwg),__uint_as_float(0x2eb3a903u)),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(FMUL(FMUL(epsb,vfww),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1695
  rg2=FMUL(epsg,FADD(FADD(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),u480_sub(__uint_as_float(0x3f800000u),svf)),vfws),rx),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),u480_sub(__uint_as_float(0x3f800000u),svf)),vfwg),epsg),__uint_as_float(0x2eb3a903u)),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb,u480_sub(__uint_as_float(0x3f800000u),epsb)),u480_sub(__uint_as_float(0x3f800000u),svf)),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1699
  rb2=FMUL(epsb,FADD(FADD(FADD(FADD(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsg),vfwg),vfgs),rx),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsg),epsb),vfgw),vfwg),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),vfws),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),rx)),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),epsb),vfwg),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),__uint_as_float(0x2eb3a903u)),epsg),gfk_pow(tgp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))),FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb,u480_sub(__uint_as_float(0x3f800000u),epsb)),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40000000u),vfws))),__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),__uint_as_float(0x42700000u))));
  // source line 1705
  rg=FADD(rg1,rg2);
  // source line 1706
  rb=FADD(rb1,rb2);
  // source line 1708
  hb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(tbp,tcp)),__uint_as_float(0x42c80000u));
  // source line 1709
  eleb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(qs0b,qcp)),__uint_as_float(0x42c80000u));
  // source line 1710
  g0b=u480_sub(u480_sub(FADD(sb,rb),hb),eleb);
  // source line 1712
  hg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(tgp,tcp)),__uint_as_float(0x42c80000u));
  // source line 1713
  eleg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(qs0g,qcp)),__uint_as_float(0x42c80000u));
  // source line 1714
  g0g=u480_sub(u480_sub(FADD(sg,rg),hg),eleg);
  // source line 1716
  u480_force_restore(capb,aksb,delt,sb,rb,hb,eleb,tblend,tbp,tb,env,status);
  // source line 1717
  u480_force_restore(capg,aksg,delt,sg,rg,hg,eleg,tglend,tgp,tg,env,status);
  // source line 1719
  tbp=tb;
  // source line 1720
  tgp=tg;
  // source line 1722
  tc1=FADD(FADD(FMUL(rw,alphac),FMUL(rw,alphag)),FMUL(w,alphab));
  // source line 1723
  tc2=FADD(FADD(FMUL(FMUL(rw,alphac),ta),FMUL(FMUL(rw,alphag),tgp)),FMUL(FMUL(w,alphab),tbp));
  // source line 1724
  tc=FDIV(tc2,tc1);
  // source line 1726
  qc1=FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,alphag),betg)),FMUL(FMUL(w,alphab),betb));
  // source line 1727
  qc2=FADD(FADD(FMUL(FMUL(rw,alphac),qa),FMUL(FMUL(FMUL(rw,alphag),betg),qs0g)),FMUL(FMUL(FMUL(w,alphab),betb),qs0b));
  // source line 1728
  qc=FDIV(qc2,qc1);
  // source line 1730
  tcp=tc;
  // source line 1731
  qcp=qc;
  // source line 1733
  }
  // source line 1735
  } else {
  // source line 1754
  if (((ts_scheme==1))) {
  // source line 1757
  if (((ch_scheme==1))) {
  // source line 1758
  z=zdc;
  // source line 1759
  bhvg=FDIV(gfk_log(FDIV(z0vg,z0hvg)),__uint_as_float(0x3ecccccdu));
  // source line 1760
  ribvg=FDIV(FMUL(FMUL(FDIV(__uint_as_float(0x419ccccdu),FADD(tcp,tvgp)),u480_sub(tcp,tvgp)),FADD(z,z0vg)),FMUL(uc,uc));
  // source line 1761
  u480_mos(xxxvg,alphavg,cdvg,bhvg,ribvg,z,z0vg,uc,tcp,tvgp,rho,env,status);
  // source line 1762
  } else {
  // source line 1763
  alphavg=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),FADD(__uint_as_float(0x40c4cccdu),FMUL(__uint_as_float(0x4085c28fu),uc))),1200);
  // source line 1764
  if (((uc>5))) {
  alphavg=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),FMUL(__uint_as_float(0x40f051ecu),gfk_pow(uc,__uint_as_float(0x3f47ae14u)))),1200);
  }
  // source line 1765
  }
  // source line 1766
  chvg=FDIV(FDIV(FDIV(alphavg,rho),__uint_as_float(0x3e75c28fu)),uc);
  // source line 1768
  srun1=__uint_as_float(0x00000000u);
  // source line 1769
  srun2=__uint_as_float(0x00000000u);
  // source line 1770
  srun3=__uint_as_float(0x00000000u);
  // source line 1772
  for(int vi=0;vi<4;++vi) dzvg[vi]=dzgr[vi];
  // source line 1773
  kz=1;
  // source line 1774
  zsoilg[(kz)-1]=(-dzvg[(kz)-1]);
  // source line 1775
  for (kz=2;kz<=4;kz+=1) {
  // source line 1776
  zsoilg[(kz)-1]=FADD((-dzvg[(kz)-1]),zsoilg[((kz-1))-1]);
  // source line 1777
  }
  // source line 1779
  if (((ttscheme==0))) {
  // source line 1780
  ttp=tcp;
  // source line 1781
  } else if (((ttscheme==1))) {
  // source line 1782
  ttp=tt;
  // source line 1783
  }
  // source line 1785
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(ttp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),ttp))));
  // source line 1786
  qs0t=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1787
  ra_leaf=FMUL(FMUL(58,gfk_pow(lai_tree,__uint_as_float(0x3f0f5c29u))),FSQRT(FDIV(__uint_as_float(0x3dcccccdu),uc)));
  // source line 1793
  w2_num=__uint_as_float(0x00000000u);
  // source line 1794
  w2_den=__uint_as_float(0x00000000u);
  // source line 1795
  for (k=1;k<=3;k+=1) {
  // source line 1796
  w2_num=__dadd_rn(w2_num,FMUL(smg[(k)-1],dzvg[(k)-1]));
  // source line 1797
  w2_den=__dadd_rn(w2_den,dzvg[(k)-1]);
  // source line 1798
  }
  // source line 1799
  w2=__ddiv_rn(w2_num,w2_den);
  // source line 1801
  u480_stomatal(rs_leaf,__uint_as_float(0x42c80000u),sleaf,lai_tree,w2,__uint_as_float(0x3dac0831u),__uint_as_float(0x3ee0c49cu),tcp,ttp,qcp,qs0t,env,status);
  // source line 1803
  for (iteration=1;iteration<=100;iteration+=1) {
  // source line 1805
  tbp=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tbp));
  // source line 1806
  tgp=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tgp));
  // source line 1807
  tvgp=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tvgp));
  // source line 1809
  kz=1;
  // source line 1810
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tvgp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tvgp))));
  // source line 1811
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),u_powi(tvgp,2));
  // source line 1812
  qs0vg=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1813
  dqs0vgdtvg=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),u_powi(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),2));
  // source line 1814
  epvg=FMUL(FMUL(FMUL(rhoo,chvg),uc),u480_sub(qs0vg,qcp));
  // source line 1816
  for(int vi=0;vi<4;++vi) smgp[vi]=u_max(smgp[vi],__uint_as_float(0x3dac0831u));
  // source line 1817
  for(int vi=0;vi<4;++vi) smgp[vi]=u_min(smgp[vi],__uint_as_float(0x3ee0c49cu));
  // source line 1819
  vfac=u480_sub(__uint_as_float(0x3f800000u),gfk_exp(FMUL(__uint_as_float(0xbf051eb8u),lai_veg)));
  // source line 1820
  if (((epvg>__uint_as_float(0x00000000u)))) {
  // source line 1821
  u480_direvap(edirg,epvg,smgp[(kz)-1],vfac,__uint_as_float(0x3ee0c49cu),__uint_as_float(0x3d872b02u),__uint_as_float(0x40000000u),env,status);
  // source line 1822
  u480_transp(ettg,etg,ecg,vfac,epvg,cmcgp,__uint_as_float(0x3f000000u),__uint_as_float(0x3a03126fu),lai_veg,__uint_as_float(0x42c80000u),__uint_as_float(0x459c4000u),__uint_as_float(0x42c80000u),sx,tvgp,tcp,qcp,smgp,__uint_as_float(0x3dac0831u),__uint_as_float(0x3ea872b0u),__uint_as_float(0x447b2000u),ps,chvg,epsvg,delt,2,4,dzvg,zsoilg,__uint_as_float(0x42100000u),env,status);
  // source line 1825
  u480_smflx(smgp,smg,4,cmcgp,cmcg,delt,rain,zsoilg,__uint_as_float(0x3ee0c49cu),__uint_as_float(0x40a80000u),__uint_as_float(0x3dac0831u),srootp,__uint_as_float(0x3662d3f5u),__uint_as_float(0x376fea08u),vfac,__uint_as_float(0x3a03126fu),srun1,srun2,srun3,edirg,ecg,etg,dripg,env,status);
  // source line 1827
  } else {
  // source line 1828
  dew=(-epvg);
  // source line 1829
  raindr=FADD(rain,FMUL(dew,3600));
  // source line 1830
  edirg=__uint_as_float(0x00000000u);
  // source line 1831
  ecg=__uint_as_float(0x00000000u);
  // source line 1832
  ettg=__uint_as_float(0x00000000u);
  // source line 1833
  u480_smflx(smgp,smg,4,cmcgp,cmcg,delt,raindr,zsoilg,__uint_as_float(0x3ee0c49cu),__uint_as_float(0x40a80000u),__uint_as_float(0x3dac0831u),srootp,__uint_as_float(0x3662d3f5u),__uint_as_float(0x376fea08u),vfac,__uint_as_float(0x3a03126fu),srun1,srun2,srun3,edirg,ecg,etg,dripg,env,status);
  // source line 1835
  }
  // source line 1837
  for(int vi=0;vi<4;++vi) smg[vi]=u_max(smg[vi],__uint_as_float(0x3dac0831u));
  // source line 1838
  for(int vi=0;vi<4;++vi) smg[vi]=u_min(smg[vi],__uint_as_float(0x3ee0c49cu));
  // source line 1840
  edirg=FMUL(edirg,1000);
  // source line 1841
  ettg=FMUL(ettg,1000);
  // source line 1842
  ecg=FMUL(ecg,1000);
  // source line 1843
  etag=FADD(FADD(edirg,ettg),ecg);
  // source line 1844
  if (((etag<__uint_as_float(0x1e3ce508u)))) {
  etag=0;
  }
  // source line 1845
  if (((epvg<=0))) {
  // source line 1846
  betvg=0;
  // source line 1847
  } else {
  // source line 1848
  betvg=FDIV(etag,epvg);
  // source line 1849
  }
  // source line 1852
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tbp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tbp))));
  // source line 1853
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),u_powi(tbp,2));
  // source line 1854
  qs0b=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1855
  dqs0bdtb=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),u_powi(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),2));
  // source line 1857
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tgp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tgp))));
  // source line 1858
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),u_powi(tgp,2));
  // source line 1859
  qs0g=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1860
  dqs0gdtg=FDIV(FMUL(FMUL(desdt,__uint_as_float(0x3f1f3b64u)),ps),u_powi(u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)),2));
  // source line 1862
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(ttp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),ttp))));
  // source line 1863
  qs0t=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 1864
  epsge=FADD(FMUL(u480_sub(1,fvg),epsg),FMUL(fvg,epsvg));
  // source line 1867
  emitw=FDIV(FMUL(FMUL(epsb,__uint_as_float(0x2eb3a903u)),gfk_pow(tbp,__uint_as_float(0x40800000u))),60);
  // source line 1868
  emitg=FDIV(FMUL(FMUL(epsge,__uint_as_float(0x2eb3a903u)),gfk_pow(tgep,__uint_as_float(0x40800000u))),60);
  // source line 1869
  emitt=FDIV(FMUL(FMUL(epst,__uint_as_float(0x2eb3a903u)),gfk_pow(ttp,__uint_as_float(0x40800000u))),60);
  // source line 1871
  rg1=FMUL(epsg,u480_sub(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tgp,__uint_as_float(0x40800000u))),60)));
  // source line 1875
  rvg1=FMUL(epsvg,u480_sub(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tvgp,__uint_as_float(0x40800000u))),60)));
  // source line 1879
  rb1=FMUL(epsb,u480_sub(FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree)),FDIV(FMUL(__uint_as_float(0x2eb3a903u),gfk_pow(tbp,__uint_as_float(0x40800000u))),60)));
  // source line 1883
  rg2=FMUL(epsg,FADD(FMUL(FMUL(FMUL(2,FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree))),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree)),u480_sub(1,epst)),vfgt_tree)));
  // source line 1888
  rvg2=FMUL(epsvg,FADD(FMUL(FMUL(FMUL(2,FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree))),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree)),u480_sub(1,epst)),vfgt_tree)));
  // source line 1893
  rb2=FMUL(epsb,FADD(FADD(FMUL(FMUL(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),u480_sub(1,epsge)),vfwg_tree),FMUL(FMUL(FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree)),u480_sub(1,epsb)),vfww_tree)),FMUL(FMUL(FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree)),u480_sub(1,epst)),vfwt_tree)));
  // source line 1900
  rt1=u480_sub(FMUL(epst,FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree))),emitt);
  // source line 1903
  rt2=FMUL(epst,FADD(FMUL(FMUL(FMUL(2,FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree))),u480_sub(1,epsb)),vftw_tree),FMUL(FMUL(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),u480_sub(1,epsge)),vftg_tree)));
  // source line 1908
  rg=FADD(rg1,rg2);
  // source line 1909
  rb=FADD(rb1,rb2);
  // source line 1910
  rvg=FADD(rvg1,rvg2);
  // source line 1911
  rt=FADD(rt1,rt2);
  // source line 1912
  rleaf=FMUL(FMUL(FDIV(FMUL(rt,FMUL(FMUL(__uint_as_float(0x40c90fd0u),rtree),u480_sub(1,tau))),FMUL(FMUL(2,rtree),lai_tree)),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 1915
  drbdtb1=FDIV(FMUL(epsb,u480_sub(FMUL(FMUL(FMUL(FMUL(4,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfww_tree),FMUL(__uint_as_float(0x2fb3a903u),u_powi(tb,3)))),60);
  // source line 1916
  drbdtg1=FDIV(FMUL(epsb,FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),vfwg_tree),u480_sub(1,fvg))),60);
  // source line 1917
  drbdtvg1=FDIV(FMUL(epsb,FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),vfwg_tree),fvg)),60);
  // source line 1918
  drbdtb2=FDIV(FMUL(epsb,FADD(FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfgw_tree),u480_sub(1,epsge)),vfwg_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfww_tree),u480_sub(1,epsb)),vfww_tree)),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vftw_tree),u480_sub(1,epst)),vfwt_tree))),60);
  // source line 1921
  drbdtg2=FDIV(FMUL(epsb,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),vfwg_tree),u480_sub(1,fvg)),u480_sub(1,epsb)),vfww_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),vftg_tree),u480_sub(1,fvg)),u480_sub(1,epst)),vfwt_tree))),60);
  // source line 1923
  drbdtvg2=FDIV(FMUL(epsb,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),vfwg_tree),fvg),u480_sub(1,epsb)),vfww_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),vftg_tree),fvg),u480_sub(1,epst)),vfwt_tree))),60);
  // source line 1927
  drgdtb1=FDIV(FMUL(epsg,FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfgw_tree)),60);
  // source line 1928
  drgdtg1=FDIV(FMUL(epsg,FMUL(__uint_as_float(0xafb3a903u),u_powi(tg,3))),60);
  // source line 1929
  drgdtvg1=0;
  // source line 1930
  drgdtb2=FDIV(FMUL(epsg,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfww_tree),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vftw_tree),u480_sub(1,epst)),vfgt_tree))),60);
  // source line 1932
  drgdtg2=FDIV(FMUL(epsg,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),u480_sub(1,fvg)),vfwg_tree),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),u480_sub(1,fvg)),vftg_tree),u480_sub(1,epst)),vfgt_tree))),60);
  // source line 1934
  drgdtvg2=FDIV(FMUL(epsg,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),fvg),vfwg_tree),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),fvg),vftg_tree),u480_sub(1,epst)),vfgt_tree))),60);
  // source line 1938
  drvgdtb1=FDIV(FMUL(epsvg,FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfgw_tree)),60);
  // source line 1939
  drvgdtg1=0;
  // source line 1940
  drvgdtvg1=FDIV(FMUL(epsvg,FMUL(__uint_as_float(0xafb3a903u),u_powi(tvg,3))),60);
  // source line 1941
  drvgdtb2=FDIV(FMUL(epsvg,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vfww_tree),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsb),__uint_as_float(0x2eb3a903u)),u_powi(tb,3)),vftw_tree),u480_sub(1,epst)),vfgt_tree))),60);
  // source line 1943
  drvgdtg2=FDIV(FMUL(epsvg,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),u480_sub(1,fvg)),vfwg_tree),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),u480_sub(1,fvg)),vftg_tree),u480_sub(1,epst)),vfgt_tree))),60);
  // source line 1945
  drvgdtvg2=FDIV(FMUL(epsvg,FADD(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(8,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),fvg),vfwg_tree),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4,epsge),__uint_as_float(0x2eb3a903u)),u_powi(tge,3)),fvg),vftg_tree),u480_sub(1,epst)),vfgt_tree))),60);
  // source line 1949
  drbdtb=FADD(drbdtb1,drbdtb2);
  // source line 1950
  drbdtg=FADD(drbdtg1,drbdtg2);
  // source line 1951
  drbdtvg=__dadd_rn(drbdtvg1,drbdtvg2);
  // source line 1952
  drgdtb=FADD(drgdtb1,drgdtb2);
  // source line 1953
  drgdtg=FADD(drgdtg1,drgdtg2);
  // source line 1954
  drgdtvg=__dadd_rn(drgdtvg1,drgdtvg2);
  // source line 1955
  drvgdtb=__dadd_rn(drvgdtb1,drvgdtb2);
  // source line 1956
  drvgdtg=__dadd_rn(drvgdtg1,drvgdtg2);
  // source line 1957
  drvgdtvg=__dadd_rn(drvgdtvg1,drvgdtvg2);
  // source line 1959
  hb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(tbp,tcp)),100);
  // source line 1960
  hg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(tgp,tcp)),100);
  // source line 1961
  hvg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chvg),uc),u480_sub(tvgp,tcp)),100);
  // source line 1964
  eleb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(qs0b,qcp)),100);
  // source line 1965
  eleg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(qs0g,qcp)),100);
  // source line 1968
  u480_leleaf(elet,sleaf,rleaf,ra_leaf,rs_leaf,rhoo,tcp,qcp,qs0t,env,status);
  // source line 1970
  available_water_sum=__uint_as_float(0x00000000u);
  // source line 1971
  for (k=1;k<=3;k+=1) {
  // source line 1972
  available_water_sum=FADD(available_water_sum,FMUL(u480_sub(smg[(k)-1],__uint_as_float(0x3dac0831u)),dzvg[(k)-1]));
  // source line 1973
  }
  // source line 1975
  if ((((rtree>__uint_as_float(0x00000000u))) && ((lai_tree>__uint_as_float(0x00000000u))))) {
  // source line 1976
  potential_elet=FDIV(FDIV(FMUL(FMUL(FMUL(__uint_as_float(0x4f118df6u),available_water_sum),fvg),rw),delt),FMUL(FMUL(__uint_as_float(0x40000000u),rtree),lai_tree));
  // source line 1978
  } else {
  // source line 1979
  potential_elet=__uint_as_float(0x00000000u);
  // source line 1980
  }
  // source line 1981
  elet=u_min(potential_elet,elet);
  // source line 1983
  u480_root_uptake(sroot,4,smg,__uint_as_float(0x3ee0c49cu),__uint_as_float(0x3d872b02u),rtree_m,lai_tree,dzvg,zsoilg,elet,fvg,rw_m,env,status);
  // source line 1984
  for(int vi=0;vi<4;++vi) srootp[vi]=sroot[vi];
  // source line 1987
  if (((ttscheme==0))) {
  // source line 1988
  ht=FMUL(u480_sub(u480_sub(FADD(sleaf,rleaf),elet),FMUL(FDIV(u480_sub(tt,ttp),delt),640)),__uint_as_float(0x37c8631eu));
  // source line 1989
  } else if (((ttscheme==1))) {
  // source line 1990
  alphat=FDIV(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),100),FMUL(__uint_as_float(0x3fa3126fu),ra_leaf));
  // source line 1991
  ht=FMUL(alphat,u480_sub(ttp,tcp));
  // source line 1992
  }
  // source line 1995
  dtcdtb=FDIV(FMUL(w,alphab),FADD(FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,fvg),alphavg)),FMUL(FMUL(rw,u480_sub(1,fvg)),alphag)),FMUL(w,alphab)));
  // source line 1996
  dtcdtg=FDIV(FMUL(FMUL(rw,alphag),u480_sub(1,fvg)),FADD(FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,fvg),alphavg)),FMUL(FMUL(rw,u480_sub(1,fvg)),alphag)),FMUL(w,alphab)));
  // source line 1997
  dtcdtvg=FDIV(FMUL(FMUL(rw,alphavg),fvg),FADD(FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,fvg),alphavg)),FMUL(FMUL(rw,u480_sub(1,fvg)),alphag)),FMUL(w,alphab)));
  // source line 2001
  dhbdtb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(1,dtcdtb)),100);
  // source line 2002
  dhbdtg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(0,dtcdtg)),100);
  // source line 2003
  dhbdtvg=__dmul_rn(__dmul_rn(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),__dsub_rn(0,dtcdtvg)),100);
  // source line 2004
  dhgdtb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(0,dtcdtb)),100);
  // source line 2005
  dhgdtg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(1,dtcdtg)),100);
  // source line 2006
  dhgdtvg=__dmul_rn(__dmul_rn(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),__dsub_rn(0,dtcdtvg)),100);
  // source line 2007
  dhvgdtb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chvg),uc),u480_sub(0,dtcdtb)),100);
  // source line 2008
  dhvgdtg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chvg),uc),u480_sub(0,dtcdtg)),100);
  // source line 2009
  dhvgdtvg=__dmul_rn(__dmul_rn(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chvg),uc),__dsub_rn(1,dtcdtvg)),100);
  // source line 2011
  den_qc=FADD(FADD(FADD(FMUL(FMUL(w,alphab),betb),FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),betg)),FMUL(FMUL(FMUL(rw,fvg),alphavg),betvg)),FMUL(rw,alphac));
  // source line 2012
  dqcdtb=__ddiv_rn(FMUL(FMUL(FMUL(w,alphab),betb),dqs0bdtb),den_qc);
  // source line 2013
  dqcdtg=__ddiv_rn(FMUL(FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),betg),dqs0gdtg),den_qc);
  // source line 2015
  dqcdtvg=__ddiv_rn(FMUL(FMUL(FMUL(FMUL(rw,fvg),alphavg),betvg),dqs0vgdtvg),den_qc);
  // source line 2017
  delebdtb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(dqs0bdtb,dqcdtb)),100);
  // source line 2018
  delebdtg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(0,dqcdtg)),100);
  // source line 2019
  delebdtvg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(0,dqcdtvg)),100);
  // source line 2020
  delegdtb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(0,dqcdtb)),100);
  // source line 2021
  delegdtg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(dqs0gdtg,dqcdtg)),100);
  // source line 2022
  delegdtvg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(0,dqcdtvg)),100);
  // source line 2023
  delevgdtb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chvg),uc),betvg),u480_sub(0,dqcdtb)),100);
  // source line 2024
  delevgdtg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chvg),uc),betvg),u480_sub(0,dqcdtg)),100);
  // source line 2025
  delevgdtvg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chvg),uc),betvg),u480_sub(dqs0vgdtvg,dqcdtvg)),100);
  // source line 2028
  elevg=FMUL(FMUL(etag,__uint_as_float(0x4411c000u)),__uint_as_float(0x3dcccccdu));
  // source line 2030
  g0b=FDIV(FMUL(aksb,u480_sub(tbp,tbl[(1)-1])),FDIV(dzb[(1)-1],2));
  // source line 2031
  g0g=FDIV(FMUL(aksg,u480_sub(tgp,tgl[(1)-1])),FDIV(dzg[(1)-1],2));
  // source line 2033
  u480_tdfcnd(kvg,smg[(kz)-1],__uint_as_float(0x3ecccccdu),__uint_as_float(0x3ee0c49cu),env,status);
  // source line 2035
  g0vg=FDIV(FDIV(FDIV(FMUL(kvg,u480_sub(tvgp,tvgl[(1)-1])),FDIV(dzvg[(1)-1],__uint_as_float(0x40000000u))),__uint_as_float(0x442e6ccdu)),60);
  // source line 2038
  dg0bdtb=FDIV(FMUL(2,aksb),dzb[(1)-1]);
  // source line 2039
  dg0bdtg=0;
  // source line 2040
  dg0bdtvg=0;
  // source line 2041
  dg0gdtb=0;
  // source line 2042
  dg0gdtg=FDIV(FMUL(2,aksg),dzg[(1)-1]);
  // source line 2043
  dg0gdtvg=0;
  // source line 2044
  dg0vgdtb=0;
  // source line 2045
  dg0vgdtg=0;
  // source line 2046
  dg0vgdtvg=FDIV(FDIV(FDIV(FMUL(2,kvg),dzvg[(1)-1]),__uint_as_float(0x442e6ccdu)),60);
  // source line 2048
  wf=u480_sub(u480_sub(u480_sub(FADD(sb,rb),hb),eleb),g0b);
  // source line 2049
  a=u480_sub(u480_sub(u480_sub(drbdtb,dhbdtb),delebdtb),dg0bdtb);
  // source line 2050
  b=u480_sub(u480_sub(u480_sub(drbdtg,dhbdtg),delebdtg),dg0bdtg);
  // source line 2051
  c=__dsub_rn(__dsub_rn(__dsub_rn(drbdtvg,dhbdtvg),delebdtvg),dg0bdtvg);
  // source line 2053
  gf=u480_sub(u480_sub(u480_sub(FADD(sg,rg),hg),eleg),g0g);
  // source line 2054
  e=u480_sub(u480_sub(u480_sub(drgdtb,dhgdtb),delegdtb),dg0gdtb);
  // source line 2055
  ff=u480_sub(u480_sub(u480_sub(drgdtg,dhgdtg),delegdtg),dg0gdtg);
  // source line 2056
  gg=__dsub_rn(__dsub_rn(__dsub_rn(drgdtvg,dhgdtvg),delegdtvg),dg0gdtvg);
  // source line 2058
  vgf=u480_sub(u480_sub(u480_sub(FADD(svg,rvg),hvg),elevg),g0vg);
  // source line 2059
  i=__dsub_rn(__dsub_rn(__dsub_rn(drvgdtb,dhvgdtb),delevgdtb),dg0vgdtb);
  // source line 2060
  j=__dsub_rn(__dsub_rn(__dsub_rn(drvgdtg,dhvgdtg),delevgdtg),dg0vgdtg);
  // source line 2061
  kk=__dsub_rn(__dsub_rn(__dsub_rn(drvgdtvg,dhvgdtvg),delevgdtvg),dg0vgdtvg);
  // source line 2064
  det_scale=u480_maxd(u480_maxd(u480_maxd(u480_maxd(u480_maxd(u480_maxd(u480_maxd(u480_maxd(u480_maxd(__longlong_as_double(0x3ff0000000000000ull),fabs(a)),fabs(b)),fabs(c)),fabs(e)),fabs(ff)),fabs(gg)),fabs(i)),fabs(j)),fabs(kk));
  // source line 2066
  det_tol=__dmul_rn(__longlong_as_double(0x3d719799812dea11ull),u480_powi64(det_scale,3));
  // source line 2067
  det=__dadd_rn(__dsub_rn(__dmul_rn(a,__dsub_rn(__dmul_rn(ff,kk),__dmul_rn(gg,j))),__dmul_rn(b,__dsub_rn(__dmul_rn(e,kk),__dmul_rn(gg,i)))),__dmul_rn(c,__dsub_rn(__dmul_rn(e,j),__dmul_rn(ff,i))));
  // source line 2068
  a_use=a;
  // source line 2069
  ff_use=ff;
  // source line 2070
  kk_use=kk;
  // source line 2071
  relax=__longlong_as_double(0x3ff0000000000000ull);
  // source line 2072
  max_step=__uint_as_float(0x40a00000u);
  // source line 2075
  if (((fabs(det)<det_tol))) {
  // source line 2076
  f_lambda=__dmul_rn(__longlong_as_double(0x3eb0c6f7a0b5ed8dull),det_scale);
  // source line 2077
  a_use=__dadd_rn(a,f_lambda);
  // source line 2078
  ff_use=__dadd_rn(ff,f_lambda);
  // source line 2079
  kk_use=__dadd_rn(kk,f_lambda);
  // source line 2080
  det=__dadd_rn(__dsub_rn(__dmul_rn(a_use,__dsub_rn(__dmul_rn(ff_use,kk_use),__dmul_rn(gg,j))),__dmul_rn(b,__dsub_rn(__dmul_rn(e,kk_use),__dmul_rn(gg,i)))),__dmul_rn(c,__dsub_rn(__dmul_rn(e,j),__dmul_rn(ff_use,i))));
  // source line 2081
  relax=__longlong_as_double(0x3fe0000000000000ull);
  // source line 2082
  if (((fabs(det)<det_tol))) {
  // source line 2083
  det=copysign(det_tol,det);
  // source line 2084
  }
  // source line 2085
  // source line 2086
  }
  // source line 2088
  chara=__ddiv_rn(__longlong_as_double(0x3ff0000000000000ull),det);
  // source line 2090
  dtb=__dmul_rn((-chara),__dadd_rn(__dadd_rn(__dmul_rn(wf,__dsub_rn(__dmul_rn(ff_use,kk_use),__dmul_rn(gg,j))),__dmul_rn(gf,__dsub_rn(__dmul_rn(c,j),__dmul_rn(b,kk_use)))),__dmul_rn(vgf,__dsub_rn(__dmul_rn(b,gg),__dmul_rn(c,ff_use)))));
  // source line 2091
  dtg=__dmul_rn((-chara),__dadd_rn(__dadd_rn(__dmul_rn(wf,__dsub_rn(__dmul_rn(gg,i),__dmul_rn(e,kk_use))),__dmul_rn(gf,__dsub_rn(__dmul_rn(a_use,kk_use),__dmul_rn(c,i)))),__dmul_rn(vgf,__dsub_rn(__dmul_rn(c,e),__dmul_rn(a_use,gg)))));
  // source line 2092
  dtvg=__dmul_rn((-chara),__dadd_rn(__dadd_rn(__dmul_rn(wf,__dsub_rn(__dmul_rn(e,j),__dmul_rn(ff_use,i))),__dmul_rn(gf,__dsub_rn(__dmul_rn(b,i),__dmul_rn(a_use,j)))),__dmul_rn(vgf,__dsub_rn(__dmul_rn(a_use,ff_use),__dmul_rn(b,e)))));
  // source line 2093
  dtb=__dmul_rn(dtb,relax);
  // source line 2094
  dtg=__dmul_rn(dtg,relax);
  // source line 2095
  dtvg=__dmul_rn(dtvg,relax);
  // source line 2096
  dtb=u480_sign(u_min(fabsf(dtb),max_step),dtb);
  // source line 2097
  dtg=u480_sign(u_min(fabsf(dtg),max_step),dtg);
  // source line 2098
  dtvg=u480_sign(u_min(fabsf(dtvg),max_step),dtvg);
  // source line 2101
  tb=FADD(tbp,dtb);
  // source line 2102
  tg=FADD(tgp,dtg);
  // source line 2103
  tvg=FADD(tvgp,dtvg);
  // source line 2104
  tge=FADD(FMUL(u480_sub(1,fvg),tg),FMUL(fvg,tvg));
  // source line 2106
  tbp=tb;
  // source line 2107
  tgp=tg;
  // source line 2108
  tvgp=tvg;
  // source line 2109
  tgep=tge;
  // source line 2112
  tbp=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tbp));
  // source line 2113
  tgp=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tgp));
  // source line 2114
  tvgp=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tvgp));
  // source line 2121
  tc1=FADD(FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,fvg),alphavg)),FMUL(FMUL(rw,u480_sub(1,fvg)),alphag)),FMUL(w,alphab));
  // source line 2122
  tc2=FADD(FADD(FADD(FADD(FMUL(FMUL(rw,alphac),ta),FMUL(FMUL(FMUL(rw,fvg),alphavg),tvgp)),FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),tgp)),FMUL(FMUL(w,alphab),tbp)),FMUL(FMUL(FMUL(2,rtree),lai_tree),FDIV(ht,100)));
  // source line 2123
  tc=FDIV(tc2,tc1);
  // source line 2124
  tc=u_max(__uint_as_float(0x435f2666u),u_min(__uint_as_float(0x43b09333u),tc));
  // source line 2125
  tcp=tc;
  // source line 2129
  qc1=FADD(FADD(FMUL(FMUL(w,alphab),betb),FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),betg)),FMUL(rw,alphac));
  // source line 2130
  qc2=FADD(FADD(FADD(FADD(FMUL(FMUL(FMUL(w,alphab),betb),qs0b),FMUL(FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),betg),qs0g)),FDIV(FMUL(FMUL(FMUL(rw,fvg),etag),__uint_as_float(0x3e75c28fu)),1000)),FMUL(FMUL(rw,alphac),qa)),FDIV(FDIV(FMUL(FMUL(FMUL(FMUL(2,rtree),lai_tree),__uint_as_float(0x3e75c28fu)),FDIV(FDIV(elet,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u))),100),__uint_as_float(0x4411c000u)));
  // source line 2131
  qc=FDIV(qc2,qc1);
  // source line 2135
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tc,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tc))));
  // source line 2136
  desdt=FDIV(FMUL(__uint_as_float(0x45a94801u),es),u_powi(tc,2));
  // source line 2137
  qs0c=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 2139
  qc=u_min(qc,qs0c);
  // source line 2140
  qc=u_max(qc,__uint_as_float(0x00000000u));
  // source line 2141
  if ((((qc<=__uint_as_float(0x3089705fu))) && ((qcp>__uint_as_float(0x3089705fu))))) {
  // source line 2142
  qc=qcp;
  // source line 2143
  } else if ((((qc<=__uint_as_float(0x3089705fu))) && ((qcp<=__uint_as_float(0x3089705fu))))) {
  // source line 2144
  // source line 2145
  // source line 2146
  }
  // source line 2149
  if (((ttscheme==0))) {
  // source line 2150
  tt=tc;
  // source line 2151
  } else if (((ttscheme==1))) {
  // source line 2152
  tt=FADD(ttp,FMUL(__uint_as_float(0x3dcccccdu),u480_sub(tc,ttp)));
  // source line 2153
  }
  // source line 2155
  dtc=u480_sub(tcp,tc);
  // source line 2156
  qcp=qc;
  // source line 2159
  rvgt=FMUL(FMUL(FADD(svg,rvg),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 2160
  rch=FMUL(FMUL(rhoo,__uint_as_float(0x447b2000u)),chvg);
  // source line 2161
  rr1=FADD(FDIV(FMUL(FMUL(epsvg,u_powi(tvgp,4)),__uint_as_float(0x338b282du)),FMUL(ps,chvg)),__uint_as_float(0x3f800000u));
  // source line 2162
  if (((rain>__uint_as_float(0x00000000u)))) {
  // source line 2163
  rr2=FADD(rr1,FDIV(FMUL(FDIV(rain,3600),__uint_as_float(0x4583d000u)),rch));
  // source line 2164
  } else {
  // source line 2165
  rr2=rr1;
  // source line 2166
  }
  // source line 2167
  yy=FADD(tvg,FDIV(u480_sub(FDIV(rvgt,rch),FDIV(FMUL(FMUL(betvg,epvg),__uint_as_float(0x4a150c40u)),rch)),rr2));
  // source line 2168
  zz1=FADD(FDIV(kvg,FMUL(FMUL(FMUL(__uint_as_float(0xbf000000u),zsoilg[(kz)-1]),rch),rr2)),__uint_as_float(0x3f800000u));
  // source line 2169
  srun3=FDIV(srun3,delt);
  // source line 2170
  srun2=FADD(srun2,srun3);
  // source line 2173
  if ((((fabs(wf)<__uint_as_float(0x358637bdu))) && ((fabsf(dtb)<__uint_as_float(0x358637bdu))) && ((fabsf(gf)<__uint_as_float(0x358637bdu))) && ((fabsf(dtg)<__uint_as_float(0x358637bdu))) && ((fabs(vgf)<__uint_as_float(0x358637bdu))) && ((fabsf(dtvg)<__uint_as_float(0x358637bdu))) && ((fabsf(dtc)<__uint_as_float(0x358637bdu))))) {
  break;
  }
  // source line 2179
  }
  // source line 2182
  u480_multi_layer(num_wall_layers,boundb,g0b,capb,aksb,tbl,dzb,delt,tblend,env,status);
  // source line 2183
  u480_multi_layer(num_road_layers,boundg,g0g,capg,aksg,tgl,dzg,delt,tglend,env,status);
  // source line 2185
  u480_shflx(ssoilg,tvgl,smg,__uint_as_float(0x3ee0c49cu),4,tvgp,delt,yy,zz1,zsoilg,tglend,__uint_as_float(0xc0000000u),__uint_as_float(0x3dac0831u),kvg,__uint_as_float(0x3ecccccdu),__uint_as_float(0x49f42400u),capvg,env,status);
  // source line 2188
  } else {
  // source line 2194
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tbp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tbp))));
  // source line 2195
  qs0b=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 2197
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tgp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tgp))));
  // source line 2198
  qs0g=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 2200
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(ttp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),ttp))));
  // source line 2201
  qs0t=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 2203
  es=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(tvgp,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),tvgp))));
  // source line 2204
  qs0vg=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),es),u480_sub(ps,FMUL(__uint_as_float(0x3ec18937u),es)));
  // source line 2206
  epsge=FADD(FMUL(u480_sub(1,fvg),epsg),FMUL(fvg,epsvg));
  // source line 2207
  tgep=FADD(FMUL(u480_sub(1,fvg),tgp),FMUL(fvg,tvgp));
  // source line 2208
  qs0ge=FADD(FMUL(u480_sub(1,fvg),qs0g),FMUL(fvg,qs0vg));
  // source line 2209
  alphage=FADD(FMUL(u480_sub(1,fvg),alphag),FMUL(fvg,alphavg));
  // source line 2211
  emitw=FDIV(FMUL(FMUL(epsb,__uint_as_float(0x2eb3a903u)),u_powi(tbp,4)),60);
  // source line 2212
  emitg=FDIV(FMUL(FMUL(epsge,__uint_as_float(0x2eb3a903u)),u_powi(tgep,4)),60);
  // source line 2213
  emitt=FDIV(FMUL(FMUL(epst,__uint_as_float(0x2eb3a903u)),u_powi(ttp,4)),60);
  // source line 2215
  rg1=FMUL(epsg,u480_sub(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),FDIV(FMUL(__uint_as_float(0x2eb3a903u),u_powi(tgp,4)),60)));
  // source line 2219
  rvg1=FMUL(epsvg,u480_sub(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),FDIV(FMUL(__uint_as_float(0x2eb3a903u),u_powi(tvgp,4)),60)));
  // source line 2223
  rb1=FMUL(epsb,u480_sub(FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree)),FDIV(FMUL(__uint_as_float(0x2eb3a903u),u_powi(tbp,4)),60)));
  // source line 2227
  rg2=FMUL(epsg,FADD(FMUL(FMUL(FMUL(2,FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree))),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree)),u480_sub(1,epst)),vfgt_tree)));
  // source line 2232
  rvg2=FMUL(epsvg,FADD(FMUL(FMUL(FMUL(2,FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree))),u480_sub(1,epsb)),vfgw_tree),FMUL(FMUL(FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree)),u480_sub(1,epst)),vfgt_tree)));
  // source line 2237
  rb2=FMUL(epsb,FADD(FADD(FMUL(FMUL(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),u480_sub(1,epsge)),vfwg_tree),FMUL(FMUL(FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree)),u480_sub(1,epsb)),vfww_tree)),FMUL(FMUL(FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree)),u480_sub(1,epst)),vfwt_tree)));
  // source line 2244
  rt1=u480_sub(FMUL(epst,FADD(FADD(FMUL(rx,vfts_tree),FMUL(FMUL(2,emitw),vftw_tree)),FMUL(emitg,vftg_tree))),emitt);
  // source line 2247
  rt2=FMUL(epst,FADD(FMUL(FMUL(FMUL(2,FADD(FADD(FADD(FMUL(rx,vfws_tree),FMUL(emitw,vfww_tree)),FMUL(emitg,vfwg_tree)),FMUL(emitt,vfwt_tree))),u480_sub(1,epsb)),vftw_tree),FMUL(FMUL(FADD(FADD(FMUL(rx,vfgs_tree),FMUL(FMUL(2,emitw),vfgw_tree)),FMUL(emitt,vfgt_tree)),u480_sub(1,epsge)),vftg_tree)));
  // source line 2252
  rg=FADD(rg1,rg2);
  // source line 2253
  rb=FADD(rb1,rb2);
  // source line 2254
  rvg=FADD(rvg1,rvg2);
  // source line 2255
  rt=FADD(rt1,rt2);
  // source line 2257
  hb=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chb),uc),u480_sub(tbp,tcp)),100);
  // source line 2258
  eleb=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chb),uc),betb),u480_sub(qs0b,qcp)),100);
  // source line 2259
  g0b=u480_sub(u480_sub(FADD(sb,rb),hb),eleb);
  // source line 2261
  hg=FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),chg),uc),u480_sub(tgp,tcp)),100);
  // source line 2262
  eleg=FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho,__uint_as_float(0x4411c000u)),chg),uc),betg),u480_sub(qs0g,qcp)),100);
  // source line 2263
  g0g=u480_sub(u480_sub(FADD(sg,rg),hg),eleg);
  // source line 2265
  ra_leaf=FMUL(FMUL(58,gfk_pow(lai_tree,__uint_as_float(0x3f0f5c29u))),FSQRT(FDIV(__uint_as_float(0x3dcccccdu),uc)));
  // source line 2266
  ht=FDIV(FDIV(u480_sub(u480_sub(FADD(sleaf,rleaf),elet),FMUL(FDIV(u480_sub(tt,ttp),delt),640)),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 2267
  w2=FMUL(__uint_as_float(0x3f000000u),FADD(smgp[(1)-1],smgp[(2)-1]));
  // source line 2268
  u480_stomatal(rs_leaf,__uint_as_float(0x42c80000u),sleaf,lai_tree,w2,__uint_as_float(0x3ea872b0u),__uint_as_float(0x3ee0c49cu),tc,tt,qc,qs0t,env,status);
  // source line 2269
  u480_leleaf(elet,sleaf,rleaf,ra_leaf,rs_leaf,rhoo,tc,qc,qs0t,env,status);
  // source line 2272
  u480_force_restore(capb,aksb,delt,sb,rb,hb,eleb,tblend,tbp,tb,env,status);
  // source line 2273
  u480_force_restore(capg,aksg,delt,sg,rg,hg,eleg,tglend,tgp,tg,env,status);
  // source line 2274
  u480_force_restore_tree(delt,st,rt,ht,FDIV(elet,__uint_as_float(0x47238600u)),lai_tree,ttp,tt,env,status);
  // source line 2276
  tbp=tb;
  // source line 2277
  tgp=tg;
  // source line 2278
  ttp=tt;
  // source line 2279
  tc1=FADD(FADD(FADD(FMUL(rw,alphac),FMUL(FMUL(rw,fvg),alphavg)),FMUL(FMUL(rw,u480_sub(1,fvg)),alphag)),FMUL(w,alphab));
  // source line 2280
  tc2=FADD(FADD(FADD(FADD(FMUL(FMUL(rw,alphac),ta),FMUL(FMUL(FMUL(rw,fvg),alphavg),tvgp)),FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),tgp)),FMUL(FMUL(w,alphab),tbp)),FMUL(FMUL(FMUL(2,rtree),lai_tree),FDIV(ht,100)));
  // source line 2281
  tc=FDIV(tc2,tc1);
  // source line 2284
  qc1=FADD(FADD(FMUL(FMUL(w,alphab),betb),FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),betg)),FMUL(rw,alphac));
  // source line 2285
  qc2=FADD(FADD(FADD(FADD(FMUL(FMUL(FMUL(w,alphab),betb),qs0b),FMUL(FMUL(FMUL(FMUL(rw,u480_sub(1,fvg)),alphag),betg),qs0g)),FDIV(FMUL(FMUL(FMUL(rw,fvg),etag),__uint_as_float(0x3e75c28fu)),1000)),FMUL(FMUL(rw,alphac),qa)),FDIV(FDIV(FMUL(FMUL(FMUL(FMUL(2,rtree),lai_tree),__uint_as_float(0x3e75c28fu)),FDIV(FDIV(elet,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u))),100),__uint_as_float(0x4411c000u)));
  // source line 2286
  qc=FDIV(qc2,qc1);
  // source line 2289
  tcp=tc;
  // source line 2290
  qcp=qc;
  // source line 2292
  }
  // source line 2294
  }
  // source line 2297
  flxthb=FDIV(FDIV(FDIV(hb,rho),__uint_as_float(0x3e75c28fu)),__uint_as_float(0x42c80000u));
  // source line 2298
  flxhumb=FDIV(FDIV(FDIV(eleb,rho),__uint_as_float(0x4411c000u)),__uint_as_float(0x42c80000u));
  // source line 2299
  flxthg=FDIV(FDIV(FDIV(hg,rho),__uint_as_float(0x3e75c28fu)),__uint_as_float(0x42c80000u));
  // source line 2300
  flxhumg=FDIV(FDIV(FDIV(eleg,rho),__uint_as_float(0x4411c000u)),__uint_as_float(0x42c80000u));
  // source line 2301
  flxthvg=FDIV(FDIV(FDIV(hvg,rho),__uint_as_float(0x3e75c28fu)),__uint_as_float(0x42c80000u));
  // source line 2302
  flxhumvg=FDIV(FDIV(FDIV(elevg,rho),__uint_as_float(0x4411c000u)),__uint_as_float(0x42c80000u));
  // source line 2303
  flxtht=FDIV(FDIV(FDIV(ht,rho),__uint_as_float(0x3e75c28fu)),__uint_as_float(0x42c80000u));
  // source line 2304
  flxhumt=FDIV(FDIV(FDIV(FMUL(elet,__uint_as_float(0x37c8631eu)),rho),__uint_as_float(0x4411c000u)),__uint_as_float(0x42c80000u));
  // source line 2309
  if ((((groption==1)) && ((treeoption!=1)))) {
  // source line 2310
  if (((ahoption==1))) {
  // source line 2311
  flxth=FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),flxthr),FMUL(FMUL(fgr,r),flxthgr)),FMUL(w,flxthb)),FMUL(rw,flxthg)),FDIV(FDIV(ah,rhoo),__uint_as_float(0x447b2000u)));
  // source line 2312
  } else {
  // source line 2313
  flxth=FADD(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),flxthr),FMUL(FMUL(fgr,r),flxthgr)),FMUL(w,flxthb)),FMUL(rw,flxthg));
  // source line 2314
  }
  // source line 2315
  if (((alhoption==1))) {
  // source line 2316
  flxhum=FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),flxhumr),FMUL(FMUL(fgr,r),flxhumgr)),FMUL(w,flxhumb)),FMUL(rw,flxhumg)),FDIV(FDIV(alh,rhoo),__uint_as_float(0x4a150c40u)));
  // source line 2317
  } else {
  // source line 2318
  flxhum=FADD(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),flxhumr),FMUL(FMUL(fgr,r),flxhumgr)),FMUL(w,flxhumb)),FMUL(rw,flxhumg));
  // source line 2319
  }
  // source line 2320
  flxuv=FMUL(FMUL(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),cdr),FMUL(FMUL(fgr,r),cdgr)),FMUL(rw,cdc)),ua),ua);
  // source line 2321
  flxg=FADD(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),g0r),FMUL(FMUL(fgr,r),g0gr)),FMUL(w,g0b)),FMUL(rw,g0g));
  // source line 2322
  lnet=FADD(FADD(FADD(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),fgr),r),rr),FMUL(FMUL(fgr,r),rgr)),FMUL(w,rb)),FMUL(rw,rg));
  // source line 2324
  } else if ((((groption!=1)) && ((treeoption==1)))) {
  // source line 2325
  if (((ahoption==1))) {
  // source line 2326
  flxth=FADD(FADD(FADD(FADD(FADD(FMUL(r,flxthr),FMUL(w,flxthb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxthg)),FMUL(FMUL(fvg,rw),flxthvg)),FDIV(FDIV(ah,rhoo),__uint_as_float(0x447b2000u))),FMUL(FMUL(FMUL(flxtht,lai_tree),2),rtree));
  // source line 2327
  } else {
  // source line 2328
  flxth=FADD(FADD(FADD(FADD(FMUL(r,flxthr),FMUL(w,flxthb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxthg)),FMUL(FMUL(fvg,rw),flxthvg)),FMUL(FMUL(FMUL(flxtht,lai_tree),2),rtree));
  // source line 2329
  }
  // source line 2330
  if (((alhoption==1))) {
  // source line 2331
  flxhum=FADD(FADD(FADD(FADD(FADD(FMUL(r,flxhumr),FMUL(w,flxhumb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxhumg)),FMUL(FMUL(fvg,rw),flxhumvg)),FDIV(FDIV(alh,rhoo),__uint_as_float(0x4a150c40u))),FMUL(FMUL(FMUL(flxhumt,lai_tree),2),rtree));
  // source line 2332
  } else {
  // source line 2333
  flxhum=FADD(FADD(FADD(FADD(FMUL(r,flxhumr),FMUL(w,flxhumb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxhumg)),FMUL(FMUL(fvg,rw),flxhumvg)),FMUL(FMUL(FMUL(flxhumt,lai_tree),2),rtree));
  // source line 2334
  }
  {
  // source line 2338
  flxuv=FMUL(FMUL(FADD(FMUL(r,cdr),FMUL(rw,cdc)),ua),ua);
  // source line 2339
  }
  // source line 2340
  flxg=FADD(FADD(FADD(FMUL(r,g0r),FMUL(w,g0b)),FMUL(FMUL(u480_sub(1,fvg),rw),g0g)),FMUL(FMUL(fvg,rw),g0vg));
  // source line 2341
  lnet=FADD(FADD(FADD(FADD(FMUL(r,rr),FMUL(w,rb)),FMUL(FMUL(u480_sub(1,fvg),rw),rg)),FMUL(FMUL(fvg,rw),rvg)),FMUL(FMUL(FMUL(FMUL(rt,2),__uint_as_float(0x40490fd0u)),rtree),u480_sub(1,tau)));
  // source line 2343
  } else if ((((groption==1)) && ((treeoption==1)))) {
  // source line 2344
  if (((ahoption==1))) {
  // source line 2345
  flxth=FADD(FADD(FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),flxthr),FMUL(FMUL(fgr,r),flxthgr)),FMUL(w,flxthb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxthg)),FMUL(FMUL(fvg,rw),flxthvg)),FDIV(FDIV(ah,rhoo),__uint_as_float(0x447b2000u))),FMUL(FMUL(FMUL(flxtht,lai_tree),2),rtree));
  // source line 2346
  } else {
  // source line 2347
  flxth=FADD(FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),flxthr),FMUL(FMUL(fgr,r),flxthgr)),FMUL(w,flxthb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxthg)),FMUL(FMUL(fvg,rw),flxthvg)),FMUL(FMUL(FMUL(flxtht,lai_tree),2),rtree));
  // source line 2348
  }
  // source line 2349
  if (((alhoption==1))) {
  // source line 2350
  flxhum=FADD(FADD(FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),flxhumr),FMUL(FMUL(fgr,r),flxhumgr)),FMUL(w,flxhumb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxhumg)),FMUL(FMUL(fvg,rw),flxhumvg)),FDIV(FDIV(alh,rhoo),__uint_as_float(0x4a150c40u))),FMUL(FMUL(FMUL(flxhumt,lai_tree),2),rtree));
  // source line 2351
  } else {
  // source line 2352
  flxhum=FADD(FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),flxhumr),FMUL(FMUL(fgr,r),flxhumgr)),FMUL(w,flxhumb)),FMUL(FMUL(u480_sub(1,fvg),rw),flxhumg)),FMUL(FMUL(fvg,rw),flxhumvg)),FMUL(FMUL(FMUL(flxhumt,lai_tree),2),rtree));
  // source line 2353
  }
  // source line 2354
  flxuv=FMUL(FMUL(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),cdr),FMUL(FMUL(fgr,r),cdgr)),FMUL(rw,cdc)),ua),ua);
  // source line 2355
  flxg=FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),g0r),FMUL(FMUL(fgr,r),g0gr)),FMUL(w,g0b)),FMUL(FMUL(u480_sub(1,fvg),rw),g0g)),FMUL(FMUL(fvg,rw),g0vg));
  // source line 2356
  lnet=FADD(FADD(FADD(FADD(FADD(FMUL(FMUL(u480_sub(1,fgr),r),rr),FMUL(FMUL(fgr,r),rgr)),FMUL(w,rb)),FMUL(FMUL(u480_sub(1,fvg),rw),rg)),FMUL(FMUL(fvg,rw),rvg)),FMUL(FMUL(FMUL(FMUL(rt,2),__uint_as_float(0x40490fd0u)),rtree),u480_sub(1,tau)));
  // source line 2358
  } else {
  // source line 2359
  if (((ahoption==1))) {
  // source line 2360
  flxth=FADD(FADD(FADD(FMUL(r,flxthr),FMUL(w,flxthb)),FMUL(rw,flxthg)),FDIV(FDIV(ah,rhoo),__uint_as_float(0x447b2000u)));
  // source line 2361
  flxhum=FADD(FADD(FADD(FMUL(r,flxhumr),FMUL(w,flxhumb)),FMUL(rw,flxhumg)),FDIV(FDIV(alh,rhoo),__uint_as_float(0x4a150c40u)));
  // source line 2362
  } else {
  // source line 2363
  flxth=FADD(FADD(FMUL(r,flxthr),FMUL(w,flxthb)),FMUL(rw,flxthg));
  // source line 2364
  flxhum=FADD(FADD(FMUL(r,flxhumr),FMUL(w,flxhumb)),FMUL(rw,flxhumg));
  // source line 2365
  }
  {
  // source line 2369
  flxuv=FMUL(FMUL(FADD(FMUL(r,cdr),FMUL(rw,cdc)),ua),ua);
  // source line 2370
  }
  // source line 2371
  flxg=FADD(FADD(FMUL(r,g0r),FMUL(w,g0b)),FMUL(rw,g0g));
  // source line 2372
  lnet=FADD(FADD(FMUL(r,rr),FMUL(w,rb)),FMUL(rw,rg));
  // source line 2373
  }
  // source line 2379
  sh=FMUL(FMUL(flxth,rhoo),__uint_as_float(0x447b2000u));
  // source line 2380
  lh=FMUL(FMUL(flxhum,rhoo),__uint_as_float(0x4a150c40u));
  // source line 2381
  lh_kinematic=FMUL(flxhum,rhoo);
  // source line 2382
  lw=u480_sub(llg,FMUL(FMUL(lnet,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u)));
  // source line 2383
  sw=u480_sub(ssg,FMUL(FMUL(snet,__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u)));
  // source line 2384
  alb=__uint_as_float(0x00000000u);
  // source line 2385
  if (((fabsf(ssg)>__uint_as_float(0x38d1b717u)))) {
  alb=FDIV(sw,ssg);
  }
  // source line 2386
  g=FMUL(FMUL((-flxg),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 2387
  rn=FMUL(FMUL(FADD(snet,lnet),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u));
  // source line 2389
  ust=FSQRT(flxuv);
  // source line 2390
  tst=FDIV((-flxth),ust);
  // source line 2391
  qst=FDIV((-flxhum),ust);
  // source line 2397
  z0=z0c;
  // source line 2398
  z0h=z0hc;
  // source line 2399
  z=u480_sub(za,zdc);
  // source line 2400
  znt=z0;
  // source line 2402
  xxx=FDIV(FDIV(FDIV(FMUL(FMUL(__uint_as_float(0x407b22d2u),z),tst),ta),ust),ust);
  // source line 2404
  if (((xxx>=__uint_as_float(0x3f800000u)))) {
  xxx=__uint_as_float(0x3f800000u);
  }
  // source line 2405
  if (((xxx<=__uint_as_float(0xc0a00000u)))) {
  xxx=__uint_as_float(0xc0a00000u);
  }
  // source line 2407
  if (((xxx>0))) {
  // source line 2408
  psim=FMUL(__uint_as_float(0xc0a00000u),xxx);
  // source line 2409
  psih=FMUL(__uint_as_float(0xc0a00000u),xxx);
  // source line 2410
  } else {
  // source line 2411
  x=gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx)),__uint_as_float(0x3e800000u));
  // source line 2417
  psim=FADD(u480_sub(FADD(FMUL(__uint_as_float(0x40000000u),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),x),__uint_as_float(0x40000000u)))),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),FMUL(x,x)),__uint_as_float(0x40000000u)))),FMUL(__uint_as_float(0x40000000u),glibc_atanf(x))),__uint_as_float(0x3fc90fd0u));
  // source line 2418
  psih=FMUL(__uint_as_float(0x40000000u),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),FMUL(x,x)),__uint_as_float(0x40000000u))));
  // source line 2420
  }
  // source line 2430
  gz1oz0=gfk_log(FDIV(z,z0));
  // source line 2431
  cd=FDIV(__uint_as_float(0x3e23d70bu),gfk_pow(u480_sub(gfk_log(FDIV(z,z0)),psim),__uint_as_float(0x40000000u)));
  // source line 2432
  chs_local=FDIV(FMUL(__uint_as_float(0x3ecccccdu),ust),u480_sub(gfk_log(FDIV(z,z0h)),psih));
  // source line 2446
  ts=FADD(ta,FDIV(flxth,chs));
  // source line 2447
  qs=FADD(qa,FDIV(flxhum,chs));
  // source line 2454
  xxx2=FMUL(FDIV(__uint_as_float(0x40000000u),z),xxx);
  // source line 2455
  if (((xxx2>=__uint_as_float(0x3f800000u)))) {
  xxx2=__uint_as_float(0x3f800000u);
  }
  // source line 2456
  if (((xxx2<=__uint_as_float(0xc0a00000u)))) {
  xxx2=__uint_as_float(0xc0a00000u);
  }
  // source line 2458
  if (((xxx2>0))) {
  // source line 2459
  psim2=FMUL(__uint_as_float(0xc0a00000u),xxx2);
  // source line 2460
  psih2=FMUL(__uint_as_float(0xc0a00000u),xxx2);
  // source line 2461
  } else {
  // source line 2462
  x=gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx2)),__uint_as_float(0x3e800000u));
  // source line 2468
  psim2=FADD(u480_sub(FADD(FMUL(__uint_as_float(0x40000000u),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),x),__uint_as_float(0x40000000u)))),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),FMUL(x,x)),__uint_as_float(0x40000000u)))),FMUL(__uint_as_float(0x40000000u),glibc_atanf(x))),__uint_as_float(0x3fc90fdbu));
  // source line 2469
  psih2=FMUL(__uint_as_float(0x40000000u),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),FMUL(x,x)),__uint_as_float(0x40000000u))));
  // source line 2472
  }
  // source line 2483
  xxx10=FMUL(FDIV(__uint_as_float(0x41200000u),z),xxx);
  // source line 2484
  if (((xxx10>=__uint_as_float(0x3f800000u)))) {
  xxx10=__uint_as_float(0x3f800000u);
  }
  // source line 2485
  if (((xxx10<=__uint_as_float(0xc0a00000u)))) {
  xxx10=__uint_as_float(0xc0a00000u);
  }
  // source line 2487
  if (((xxx10>0))) {
  // source line 2488
  psim10=FMUL(__uint_as_float(0xc0a00000u),xxx10);
  // source line 2489
  psih10=FMUL(__uint_as_float(0xc0a00000u),xxx10);
  // source line 2490
  } else {
  // source line 2491
  x=gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx10)),__uint_as_float(0x3e800000u));
  // source line 2496
  psim10=FADD(u480_sub(FADD(FMUL(__uint_as_float(0x40000000u),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),x),__uint_as_float(0x40000000u)))),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),FMUL(x,x)),__uint_as_float(0x40000000u)))),FMUL(__uint_as_float(0x40000000u),glibc_atanf(x))),__uint_as_float(0x3fc90fdbu));
  // source line 2497
  psih10=FMUL(__uint_as_float(0x40000000u),gfk_log(FDIV(FADD(__uint_as_float(0x3f800000u),FMUL(x,x)),__uint_as_float(0x40000000u))));
  // source line 2500
  }
  // source line 2512
  psix=u480_sub(gfk_log(FDIV(z,z0)),psim);
  // source line 2513
  psit=u480_sub(gfk_log(FDIV(z,z0h)),psih);
  // source line 2515
  psix2=u480_sub(gfk_log(FDIV(__uint_as_float(0x40000000u),z0)),psim2);
  // source line 2516
  psit2=u480_sub(gfk_log(FDIV(__uint_as_float(0x40000000u),z0h)),psih2);
  // source line 2518
  psix10=u480_sub(gfk_log(FDIV(__uint_as_float(0x41200000u),z0)),psim10);
  // source line 2519
  psit10=u480_sub(gfk_log(FDIV(__uint_as_float(0x41200000u),z0h)),psih10);
  // source line 2522
  u10=FMUL(u1,FDIV(psix10,psix));
  // source line 2523
  v10=FMUL(v1,FDIV(psix10,psix));
  {
  // source line 2533
  th2=FADD(ts,FMUL(u480_sub(ta,ts),FDIV(chs,chs2)));
  // source line 2534
  q2=FADD(qs,FMUL(u480_sub(qa,qs),FDIV(psit2,psit)));
  // source line 2535
  }
}

__device__ void u480_mos(float& xxx,float& alpha,float& cd,const float& b1,float& rib,const float& z,const float& z0,const float& ua,const float& ta,const float& tsf,const float& rho,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  const float cp=__uint_as_float(0x3e75c28fu);
  float xxx0={};
  float x={};
  float x0={};
  float faih={};
  float dpsim={};
  float dpsih={};
  float f={};
  float df={};
  float xxxp={};
  float us={};
  float ts={};
  float al={};
  float xkb={};
  float dd={};
  float psim={};
  float psih={};
  int newt={};
  const int newt_end=10;
  // source line 2563
  if (((rib<=__uint_as_float(0xc1700000u)))) {
  rib=__uint_as_float(0xc1700000u);
  }
  // source line 2565
  if (((rib<__uint_as_float(0x00000000u)))) {
  // source line 2567
  for (newt=1;newt<=10;newt+=1) {
  // source line 2569
  if (((xxx>=__uint_as_float(0x00000000u)))) {
  xxx=__uint_as_float(0xba83126fu);
  }
  // source line 2571
  xxx0=FDIV(FMUL(xxx,z0),FADD(z,z0));
  // source line 2573
  x=gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx)),__uint_as_float(0x3e800000u));
  // source line 2574
  x0=gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx0)),__uint_as_float(0x3e800000u));
  // source line 2587
  psim=u480_sub(FADD(FADD(u480_sub(gfk_log(FDIV(FADD(z,z0),z0)),gfk_log(FMUL(gfk_pow(FADD(x,__uint_as_float(0x3f800000u)),__uint_as_float(0x40000000u)),FADD(gfk_pow(x,__uint_as_float(0x40000000u)),__uint_as_float(0x3f800000u))))),FMUL(__uint_as_float(0x40000000u),glibc_atanf(x))),gfk_log(FMUL(gfk_pow(FADD(x0,__uint_as_float(0x3f800000u)),__uint_as_float(0x40000000u)),FADD(gfk_pow(x0,__uint_as_float(0x40000000u)),__uint_as_float(0x3f800000u))))),FMUL(__uint_as_float(0x40000000u),glibc_atanf(x0)));
  // source line 2592
  faih=FDIV(__uint_as_float(0x3f800000u),FSQRT(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx))));
  // source line 2593
  psih=FADD(u480_sub(FADD(gfk_log(FDIV(FADD(z,z0),z0)),FMUL(__uint_as_float(0x3ecccccdu),b1)),FMUL(__uint_as_float(0x40000000u),gfk_log(FADD(FSQRT(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx))),__uint_as_float(0x3f800000u))))),FMUL(__uint_as_float(0x40000000u),gfk_log(FADD(FSQRT(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx0))),__uint_as_float(0x3f800000u)))));
  // source line 2598
  dpsim=u480_sub(FDIV(gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx)),__uint_as_float(0xbe800000u)),xxx),FDIV(gfk_pow(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx0)),__uint_as_float(0xbe800000u)),xxx));
  // source line 2600
  dpsih=u480_sub(FDIV(FDIV(__uint_as_float(0x3f800000u),FSQRT(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx)))),xxx),FDIV(FDIV(__uint_as_float(0x3f800000u),FSQRT(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),xxx0)))),xxx));
  // source line 2603
  f=u480_sub(FDIV(FMUL(rib,gfk_pow(psim,__uint_as_float(0x40000000u))),psih),xxx);
  // source line 2605
  df=u480_sub(FDIV(FMUL(rib,u480_sub(FMUL(FMUL(FMUL(__uint_as_float(0x40000000u),dpsim),psim),psih),FMUL(dpsih,gfk_pow(psim,__uint_as_float(0x40000000u))))),gfk_pow(psih,__uint_as_float(0x40000000u))),__uint_as_float(0x3f800000u));
  // source line 2608
  xxxp=xxx;
  // source line 2609
  xxx=u480_sub(xxxp,FDIV(f,df));
  // source line 2610
  if (((xxx<=__uint_as_float(0xc1200000u)))) {
  xxx=__uint_as_float(0xc1200000u);
  }
  // source line 2612
  }
  // source line 2614
  } else if (((rib>=__uint_as_float(0x3e12491bu)))) {
  // source line 2616
  xxx=__uint_as_float(0x3f36c8b4u);
  // source line 2620
  psim=FADD(gfk_log(FDIV(FADD(z,z0),z0)),FMUL(__uint_as_float(0x40e00000u),xxx));
  // source line 2622
  psih=FADD(psim,FMUL(__uint_as_float(0x3ecccccdu),b1));
  // source line 2624
  } else {
  // source line 2628
  al=gfk_log(FDIV(FADD(z,z0),z0));
  // source line 2630
  xkb=FMUL(__uint_as_float(0x3ecccccdu),b1);
  // source line 2631
  dd=FADD(FMUL(FMUL(FMUL(FMUL(__uint_as_float(0xc0800000u),rib),__uint_as_float(0x40e00000u)),xkb),al),gfk_pow(FADD(al,xkb),__uint_as_float(0x40000000u)));
  // source line 2632
  if (((dd<=__uint_as_float(0x00000000u)))) {
  dd=__uint_as_float(0x00000000u);
  }
  // source line 2633
  xxx=FDIV(u480_sub(u480_sub(FADD(al,xkb),FMUL(FMUL(FMUL(__uint_as_float(0x40000000u),rib),__uint_as_float(0x40e00000u)),al)),FSQRT(dd)),FMUL(__uint_as_float(0x40000000u),u480_sub(FMUL(rib,__uint_as_float(0x42440000u)),__uint_as_float(0x40e00000u))));
  // source line 2638
  psim=FADD(gfk_log(FDIV(FADD(z,z0),z0)),FMUL(__uint_as_float(0x40e00000u),u_min(xxx,__uint_as_float(0x3f36c8b4u))));
  // source line 2641
  psih=FADD(psim,FMUL(__uint_as_float(0x3ecccccdu),b1));
  // source line 2643
  }
  // source line 2645
  us=FDIV(FMUL(__uint_as_float(0x3ecccccdu),ua),psim);
  // source line 2646
  if (((us<=__uint_as_float(0x3c23d70au)))) {
  us=__uint_as_float(0x3c23d70au);
  }
  // source line 2647
  ts=FDIV(FMUL(__uint_as_float(0x3ecccccdu),u480_sub(ta,tsf)),psih);
  // source line 2649
  cd=FDIV(FMUL(us,us),gfk_pow(ua,__uint_as_float(0x40000000u)));
  // source line 2650
  alpha=FDIV(FMUL(FMUL(FMUL(rho,__uint_as_float(0x3e75c28fu)),__uint_as_float(0x3ecccccdu)),us),psih);
  // source line 2652
  return;
}

__device__ void u480_multi_layer(const int& km,const int& bound,const float& g0,const float& cap,const float& aks,float* tsl,const float* dz,const float& delt,const float& tslend,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float a[4]={};
  float b[4]={};
  float c[4]={};
  float d[4]={};
  float x[4]={};
  float p[4]={};
  float q[4]={};
  float dzend={};
  int k={};
  // source line 2769
  dzend=dz[(km)-1];
  // source line 2771
  a[(1)-1]=__uint_as_float(0x00000000u);
  // source line 2773
  b[(1)-1]=FADD(FDIV(FMUL(cap,dz[(1)-1]),delt),FDIV(FMUL(__uint_as_float(0x40000000u),aks),FADD(dz[(1)-1],dz[(2)-1])));
  // source line 2775
  c[(1)-1]=FDIV(FMUL(__uint_as_float(0xc0000000u),aks),FADD(dz[(1)-1],dz[(2)-1]));
  // source line 2776
  d[(1)-1]=FADD(FMUL(FDIV(FMUL(cap,dz[(1)-1]),delt),tsl[(1)-1]),g0);
  // source line 2778
  for (k=2;k<=(km-1);k+=1) {
  // source line 2779
  a[(k)-1]=FDIV(FMUL(__uint_as_float(0xc0000000u),aks),FADD(dz[((k-1))-1],dz[(k)-1]));
  // source line 2780
  b[(k)-1]=FADD(FADD(FDIV(FMUL(cap,dz[(k)-1]),delt),FDIV(FMUL(__uint_as_float(0x40000000u),aks),FADD(dz[((k-1))-1],dz[(k)-1]))),FDIV(FMUL(__uint_as_float(0x40000000u),aks),FADD(dz[(k)-1],dz[((k+1))-1])));
  // source line 2781
  c[(k)-1]=FDIV(FMUL(__uint_as_float(0xc0000000u),aks),FADD(dz[(k)-1],dz[((k+1))-1]));
  // source line 2782
  d[(k)-1]=FMUL(FDIV(FMUL(cap,dz[(k)-1]),delt),tsl[(k)-1]);
  // source line 2783
  }
  // source line 2785
  if (((bound==1))) {
  // source line 2786
  a[(km)-1]=FDIV(FMUL(__uint_as_float(0xc0000000u),aks),FADD(dz[((km-1))-1],dz[(km)-1]));
  // source line 2787
  b[(km)-1]=FADD(FDIV(FMUL(cap,dz[(km)-1]),delt),FDIV(FMUL(__uint_as_float(0x40000000u),aks),FADD(dz[((km-1))-1],dz[(km)-1])));
  // source line 2788
  c[(km)-1]=__uint_as_float(0x00000000u);
  // source line 2789
  d[(km)-1]=FMUL(FDIV(FMUL(cap,dz[(km)-1]),delt),tsl[(km)-1]);
  // source line 2790
  } else {
  // source line 2791
  a[(km)-1]=FDIV(FMUL(__uint_as_float(0xc0000000u),aks),FADD(dz[((km-1))-1],dz[(km)-1]));
  // source line 2792
  b[(km)-1]=FADD(FADD(FDIV(FMUL(cap,dz[(km)-1]),delt),FDIV(FMUL(__uint_as_float(0x40000000u),aks),FADD(dz[((km-1))-1],dz[(km)-1]))),FDIV(FMUL(__uint_as_float(0x40000000u),aks),FADD(dz[(km)-1],dzend)));
  // source line 2793
  c[(km)-1]=__uint_as_float(0x00000000u);
  // source line 2794
  d[(km)-1]=FADD(FMUL(FDIV(FMUL(cap,dz[(km)-1]),delt),tsl[(km)-1]),FDIV(FMUL(FMUL(__uint_as_float(0x40000000u),aks),tslend),FADD(dz[(km)-1],dzend)));
  // source line 2795
  }
  // source line 2797
  p[(1)-1]=FDIV((-c[(1)-1]),b[(1)-1]);
  // source line 2798
  q[(1)-1]=FDIV(d[(1)-1],b[(1)-1]);
  // source line 2800
  for (k=2;k<=km;k+=1) {
  // source line 2801
  p[(k)-1]=FDIV((-c[(k)-1]),FADD(FMUL(a[(k)-1],p[((k-1))-1]),b[(k)-1]));
  // source line 2802
  q[(k)-1]=FDIV(FADD(FMUL((-a[(k)-1]),q[((k-1))-1]),d[(k)-1]),FADD(FMUL(a[(k)-1],p[((k-1))-1]),b[(k)-1]));
  // source line 2803
  }
  // source line 2805
  x[(km)-1]=q[(km)-1];
  // source line 2807
  for (k=(km-1);k>=1;k+=-1) {
  // source line 2808
  x[(k)-1]=FADD(FMUL(p[(k)-1],x[((k+1))-1]),q[(k)-1]);
  // source line 2809
  }
  // source line 2811
  for (k=1;k<=km;k+=1) {
  // source line 2812
  tsl[(k)-1]=x[(k)-1];
  // source line 2813
  }
  // source line 2815
  return;
}

__device__ void u480_force_restore(const float& cap,const float& aks,const float& delt,const float& s,const float& r,const float& h,const float& le,const float& tslend,const float& tsp,float& ts,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float c1={};
  float c2={};
  // source line 4056
  c2=__uint_as_float(0x4656dbfeu);
  // source line 4057
  c1=FSQRT(FMUL(FMUL(FMUL(__uint_as_float(0x3f000000u),c2),cap),aks));
  // source line 4059
  ts=FADD(tsp,FMUL(delt,u480_sub(FDIV(u480_sub(u480_sub(FADD(s,r),h),le),c1),FDIV(u480_sub(tsp,tslend),c2))));
}

__device__ void u480_sfcdif_urb(const float& zlm,const float& z0,const float& thz0,const float& thlm,const float& sfcspd,const float& akanda,float& akms,float& akhs,float& rlmo,float& cd,float& zt_out,const float& vegfrac,bool optional_present,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  const float wwst=__uint_as_float(0x3f99999au);
  const float wwst2=__uint_as_float(0x3fb851ecu);
  const float g=__uint_as_float(0x411ccccdu);
  const float vkrm=__uint_as_float(0x3ecccccdu);
  const float excm=__uint_as_float(0x3a83126fu);
  const float beta=__uint_as_float(0x3b72b9d6u);
  const float btg=__uint_as_float(0x3d14ab6du);
  const float elfc=__uint_as_float(0x3c6ddf15u);
  const float wold=__uint_as_float(0x3e19999au);
  const float wnew=__uint_as_float(0x3f59999au);
  const float pihf=__uint_as_float(0x3fc90fdbu);
  const float epsu2=__uint_as_float(0x38d1b717u);
  const float epsust=__uint_as_float(0x3d8f5c29u);
  const float epsit=__uint_as_float(0x38d1b717u);
  const float epsa=__uint_as_float(0x322bcc77u);
  const float ztmin=__uint_as_float(0xc0a00000u);
  const float ztmax=__uint_as_float(0x3f800000u);
  const float hpbl=__uint_as_float(0x447a0000u);
  const float sqvisc=__uint_as_float(0x4381199au);
  const float ric=__uint_as_float(0x3e3b645au);
  const float rric=__uint_as_float(0x40aedd07u);
  const float fhneu=__uint_as_float(0x3f4ccccdu);
  const float rfc=__uint_as_float(0x3e439581u);
  const float rlmo_thr=__uint_as_float(0x3a83126fu);
  const float rfac=__uint_as_float(0x40c8a703u);
  float zz={};
  float xx={};
  float yy={};
  float zu={};
  float zt={};
  float rdz={};
  float cxch={};
  float dthv={};
  float du2={};
  float btgh={};
  float wstar2={};
  float ustar={};
  float zslu={};
  float zslt={};
  float rlogu={};
  float rlogt={};
  float zetalt={};
  float zetalu={};
  float zetau={};
  float zetat={};
  float xlu4={};
  float xlt4={};
  float xu4={};
  float xt4={};
  float xlu={};
  float xlt={};
  float xu={};
  float xt={};
  float psmz={};
  float simm={};
  float pshz={};
  float simh={};
  float ustark={};
  float rlmn={};
  float rlma={};
  const int itrmx=5;
  int ilech={};
  int itr={};
  // source line 4158
  auto pslmu=[&](float zz) {return FMUL(__uint_as_float(0xbf75c28fu),gfk_log(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40900000u),zz))));};
  // source line 4159
  auto pslms=[&](float zz) {return u480_sub(FDIV(zz,__uint_as_float(0x3e439581u)),FMUL(__uint_as_float(0x4004dd2fu),u480_sub(__uint_as_float(0x3f800000u),FDIV(__uint_as_float(0x3f800000u),FADD(zz,__uint_as_float(0x3f800000u))))));};
  // source line 4160
  auto pslhu=[&](float zz) {return FMUL(__uint_as_float(0xbf75c28fu),gfk_log(u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x40900000u),zz))));};
  // source line 4161
  auto pslhs=[&](float zz) {return u480_sub(FMUL(zz,__uint_as_float(0x40c8a703u)),FMUL(__uint_as_float(0x4004dd2fu),u480_sub(__uint_as_float(0x3f800000u),gfk_exp(FMUL(__uint_as_float(0xbf99999au),zz)))));};
  // source line 4165
  auto pspmu=[&](float xx) {return u480_sub(FADD(u480_sub(FMUL(__uint_as_float(0xc0000000u),gfk_log(FMUL(FADD(xx,__uint_as_float(0x3f800000u)),__uint_as_float(0x3f000000u)))),gfk_log(FMUL(FADD(FMUL(xx,xx),__uint_as_float(0x3f800000u)),__uint_as_float(0x3f000000u)))),FMUL(__uint_as_float(0x40000000u),glibc_atanf(xx))),__uint_as_float(0x3fc90fdbu));};
  // source line 4166
  auto pspms=[&](float yy) {return FMUL(__uint_as_float(0x40a00000u),yy);};
  // source line 4167
  auto psphu=[&](float xx) {return FMUL(__uint_as_float(0xc0000000u),gfk_log(FMUL(FADD(FMUL(xx,xx),__uint_as_float(0x3f800000u)),__uint_as_float(0x3f000000u))));};
  // source line 4168
  auto psphs=[&](float yy) {return FMUL(__uint_as_float(0x40a00000u),yy);};
  // source line 4175
  ilech=0;
  // source line 4180
  zu=z0;
  // source line 4181
  rdz=FDIV(__uint_as_float(0x3f800000u),zlm);
  // source line 4182
  cxch=FMUL(__uint_as_float(0x3a83126fu),rdz);
  // source line 4183
  dthv=u480_sub(thlm,thz0);
  // source line 4188
  du2=u_max(FMUL(sfcspd,sfcspd),__uint_as_float(0x38d1b717u));
  // source line 4190
  btgh=__uint_as_float(0x42112f68u);
  // source line 4191
  if (((FMUL(FMUL(btgh,akhs),dthv)!=__uint_as_float(0x00000000u)))) {
  // source line 4192
  wstar2=FMUL(__uint_as_float(0x3fb851ecu),gfk_pow(fabsf(FMUL(FMUL(btgh,akhs),dthv)),__uint_as_float(0x3f2aaaabu)));
  // source line 4193
  } else {
  // source line 4194
  wstar2=__uint_as_float(0x00000000u);
  // source line 4195
  }
  // source line 4200
  ustar=u_max(FSQRT(FMUL(akms,FSQRT(FADD(du2,wstar2)))),__uint_as_float(0x3d8f5c29u));
  // source line 4205
  if (optional_present) {
  // source line 4207
  zt=FMUL(gfk_exp(u480_sub(__uint_as_float(0x40000000u),FMUL(u480_sub(akanda,FMUL(__uint_as_float(0x3f666666u),gfk_pow(vegfrac,__uint_as_float(0x3e947ae1u)))),gfk_pow(FMUL(FMUL(__uint_as_float(0x478235a0u),ustar),z0),__uint_as_float(0x3e800000u))))),z0);
  // source line 4208
  } else {
  // source line 4209
  zt=FMUL(gfk_exp(u480_sub(__uint_as_float(0x40000000u),FMUL(akanda,gfk_pow(FMUL(FMUL(__uint_as_float(0x478235a0u),ustar),z0),__uint_as_float(0x3e800000u))))),z0);
  // source line 4210
  }
  // source line 4212
  zslu=FADD(zlm,zu);
  // source line 4214
  zslt=FADD(zlm,zt);
  // source line 4215
  rlogu=gfk_log(FDIV(zslu,zu));
  // source line 4217
  rlogt=gfk_log(FDIV(zslt,zt));
  // source line 4219
  rlmo=FDIV(FMUL(FMUL(__uint_as_float(0x3c6ddf15u),akhs),dthv),u_powi(ustar,3));
  // source line 4223
  for (itr=1;itr<=5;itr+=1) {
  // source line 4224
  zetalt=u_max(FMUL(zslt,rlmo),__uint_as_float(0xc0a00000u));
  // source line 4225
  rlmo=FDIV(zetalt,zslt);
  // source line 4226
  zetalu=FMUL(zslu,rlmo);
  // source line 4227
  zetau=FMUL(zu,rlmo);
  // source line 4228
  zetat=FMUL(zt,rlmo);
  // source line 4231
  if (((ilech==0))) {
  // source line 4232
  if (((rlmo<__uint_as_float(0x00000000u)))) {
  // source line 4233
  xlu4=u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),zetalu));
  // source line 4234
  xlt4=u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),zetalt));
  // source line 4235
  xu4=u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),zetau));
  // source line 4237
  xt4=u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x41800000u),zetat));
  // source line 4238
  xlu=FSQRT(FSQRT(xlu4));
  // source line 4239
  xlt=FSQRT(FSQRT(xlt4));
  // source line 4240
  xu=FSQRT(FSQRT(xu4));
  // source line 4242
  xt=FSQRT(FSQRT(xt4));
  // source line 4244
  psmz=pspmu(xu);
  // source line 4245
  simm=FADD(u480_sub(pspmu(xlu),psmz),rlogu);
  // source line 4246
  pshz=psphu(xt);
  // source line 4247
  simh=FADD(u480_sub(psphu(xlt),pshz),rlogt);
  // source line 4248
  } else {
  // source line 4249
  zetalu=u_min(zetalu,__uint_as_float(0x3f800000u));
  // source line 4250
  zetalt=u_min(zetalt,__uint_as_float(0x3f800000u));
  // source line 4251
  psmz=pspms(zetau);
  // source line 4252
  simm=FADD(u480_sub(pspms(zetalu),psmz),rlogu);
  // source line 4253
  pshz=psphs(zetat);
  // source line 4254
  simh=FADD(u480_sub(psphs(zetalt),pshz),rlogt);
  // source line 4255
  }
  // source line 4259
  } else {
  // source line 4260
  if (((rlmo<__uint_as_float(0x00000000u)))) {
  // source line 4261
  psmz=pslmu(zetau);
  // source line 4262
  simm=FADD(u480_sub(pslmu(zetalu),psmz),rlogu);
  // source line 4263
  pshz=pslhu(zetat);
  // source line 4264
  simh=FADD(u480_sub(pslhu(zetalt),pshz),rlogt);
  // source line 4265
  } else {
  // source line 4266
  zetalu=u_min(zetalu,__uint_as_float(0x3f800000u));
  // source line 4267
  zetalt=u_min(zetalt,__uint_as_float(0x3f800000u));
  // source line 4268
  psmz=pslms(zetau);
  // source line 4269
  simm=FADD(u480_sub(pslms(zetalu),psmz),rlogu);
  // source line 4270
  pshz=pslhs(zetat);
  // source line 4271
  simh=FADD(u480_sub(pslhs(zetalt),pshz),rlogt);
  // source line 4272
  }
  // source line 4273
  }
  // source line 4278
  ustar=u_max(FSQRT(FMUL(akms,FSQRT(FADD(du2,wstar2)))),__uint_as_float(0x3d8f5c29u));
  // source line 4281
  if (optional_present) {
  // source line 4283
  zt=FMUL(gfk_exp(u480_sub(__uint_as_float(0x40000000u),FMUL(u480_sub(akanda,FMUL(__uint_as_float(0x3f666666u),gfk_pow(vegfrac,__uint_as_float(0x3e947ae1u)))),gfk_pow(FMUL(FMUL(__uint_as_float(0x478235a0u),ustar),z0),__uint_as_float(0x3e800000u))))),z0);
  // source line 4284
  } else {
  // source line 4285
  zt=FMUL(gfk_exp(u480_sub(__uint_as_float(0x40000000u),FMUL(akanda,gfk_pow(FMUL(FMUL(__uint_as_float(0x478235a0u),ustar),z0),__uint_as_float(0x3e800000u))))),z0);
  // source line 4286
  }
  // source line 4287
  zslt=FADD(zlm,zt);
  // source line 4288
  rlogt=gfk_log(FDIV(zslt,zt));
  // source line 4289
  ustark=FMUL(ustar,__uint_as_float(0x3ecccccdu));
  // source line 4290
  akms=u_max(FDIV(ustark,simm),cxch);
  // source line 4291
  akhs=u_max(FDIV(ustark,simh),cxch);
  // source line 4293
  if (((FMUL(FMUL(btgh,akhs),dthv)!=__uint_as_float(0x00000000u)))) {
  // source line 4294
  wstar2=FMUL(__uint_as_float(0x3fb851ecu),gfk_pow(fabsf(FMUL(FMUL(btgh,akhs),dthv)),__uint_as_float(0x3f2aaaabu)));
  // source line 4295
  } else {
  // source line 4296
  wstar2=__uint_as_float(0x00000000u);
  // source line 4297
  }
  // source line 4299
  rlmn=FDIV(FMUL(FMUL(__uint_as_float(0x3c6ddf15u),akhs),dthv),u_powi(ustar,3));
  // source line 4303
  rlma=FADD(FMUL(rlmo,__uint_as_float(0x3e19999au)),FMUL(rlmn,__uint_as_float(0x3f59999au)));
  // source line 4305
  rlmo=rlma;
  // source line 4307
  }
  // source line 4309
  cd=FDIV(FMUL(ustar,ustar),u_powi(sfcspd,2));
  // source line 4310
  if (optional_present) {
  zt_out=zt;
  }
}

__device__ void u480_direvap(float& edir,const float& etp,const float& smc,const float& shdfac,const float& smcmax,const float& smcdry,const float& fxexp,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float fx={};
  float sratio={};
  // source line 4328
  sratio=FDIV(u480_sub(smc,smcdry),u480_sub(smcmax,smcdry));
  // source line 4329
  if (((sratio>__uint_as_float(0x00000000u)))) {
  // source line 4330
  fx=gfk_pow(sratio,fxexp);
  // source line 4331
  fx=u_max(u_min(fx,__uint_as_float(0x3f800000u)),__uint_as_float(0x00000000u));
  // source line 4332
  } else {
  // source line 4333
  fx=__uint_as_float(0x00000000u);
  // source line 4334
  }
  // source line 4335
  edir=FMUL(FMUL(FMUL(fx,u480_sub(__uint_as_float(0x3f800000u),shdfac)),etp),__uint_as_float(0x3a83126fu));
}

__device__ void u480_transp(float& ett,float* et,float& ec,const float& shdfac,const float& etp1,const float& cmc,const float& cfactr,const float& cmcmax,const float& lai,const float& rsmin,const float& rsmax,const float& rgl,const float& sx,const float& ts,const float& ta,const float& qa,const float* smc,const float& smcwlt,const float& smcref,const float& cpp,const float& ps,const float& ch,const float& epsv,const float& delt,const int& nroot,const int& nsoil,const float* dzvr,const float* zsoil,const float& hs,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float rc={};
  float rcs={};
  float rct={};
  float rcq={};
  float rcsoil={};
  float ff={};
  float ws={};
  float slv={};
  float desdt={};
  float sigma={};
  float pc={};
  float cmc2ms={};
  float sgx={};
  float denom={};
  float rtx={};
  float ett1={};
  int k={};
  float part[4]={};
  float gx[4]={};
  float delta={};
  float ea={};
  float rr={};
  // source line 4356
  slv=__uint_as_float(0x4a18a620u);
  // source line 4357
  sigma=__uint_as_float(0x3373864fu);
  // source line 4358
  ett=__uint_as_float(0x00000000u);
  // source line 4359
  for (k=1;k<=nsoil;k+=1) {
  // source line 4360
  et[(k)-1]=__uint_as_float(0x00000000u);
  // source line 4361
  }
  // source line 4366
  rcs=__uint_as_float(0x00000000u);
  // source line 4367
  rct=__uint_as_float(0x00000000u);
  // source line 4368
  rcq=__uint_as_float(0x00000000u);
  // source line 4369
  rcsoil=__uint_as_float(0x00000000u);
  // source line 4374
  ff=FDIV(FMUL(FMUL(FMUL(__uint_as_float(0x3f8ccccdu),sx),__uint_as_float(0x442e6ccdu)),__uint_as_float(0x42700000u)),FMUL(rgl,lai));
  // source line 4375
  rcs=FDIV(FADD(ff,FDIV(rsmin,rsmax)),FADD(__uint_as_float(0x3f800000u),ff));
  // source line 4376
  rcs=u_max(rcs,__uint_as_float(0x38d1b717u));
  // source line 4381
  rct=u480_sub(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0x3ad1b717u),gfk_pow(u480_sub(298,ta),__uint_as_float(0x40000000u))));
  // source line 4382
  rct=u_max(rct,__uint_as_float(0x38d1b717u));
  // source line 4387
  ea=FMUL(__uint_as_float(0x40c3851fu),gfk_exp(FDIV(FMUL(__uint_as_float(0x45a94801u),u480_sub(ta,__uint_as_float(0x43889333u))),FMUL(__uint_as_float(0x43889333u),ta))));
  // source line 4388
  ws=FDIV(FMUL(__uint_as_float(0x3f1f3b64u),ea),1013);
  // source line 4389
  rcq=FDIV(__uint_as_float(0x3f800000u),FADD(__uint_as_float(0x3f800000u),FMUL(hs,u480_sub(ws,qa))));
  // source line 4390
  rcq=u_max(rcq,__uint_as_float(0x3c23d70au));
  // source line 4395
  for (k=1;k<=nroot;k+=1) {
  // source line 4396
  gx[(k)-1]=FDIV(u480_sub(smc[(k)-1],smcwlt),u480_sub(smcref,smcwlt));
  // source line 4397
  if (((gx[(k)-1]>__uint_as_float(0x3f800000u)))) {
  gx[(k)-1]=__uint_as_float(0x3f800000u);
  }
  // source line 4398
  if (((gx[(k)-1]<__uint_as_float(0x00000000u)))) {
  gx[(k)-1]=__uint_as_float(0x00000000u);
  }
  // source line 4399
  part[(k)-1]=FMUL(FDIV((-dzvr[(k)-1]),zsoil[(3)-1]),gx[(k)-1]);
  // source line 4400
  }
  // source line 4402
  sgx=__uint_as_float(0x00000000u);
  // source line 4403
  for (k=1;k<=nroot;k+=1) {
  // source line 4404
  sgx=FADD(sgx,gx[(k)-1]);
  // source line 4405
  rcsoil=FADD(rcsoil,part[(k)-1]);
  // source line 4406
  }
  // source line 4407
  sgx=FDIV(sgx,nroot);
  // source line 4409
  rcsoil=u_max(rcsoil,__uint_as_float(0x38d1b717u));
  // source line 4411
  rc=FDIV(rsmin,FMUL(FMUL(FMUL(FMUL(lai,rcs),rct),rcq),rcsoil));
  // source line 4412
  desdt=FDIV(FDIV(FDIV(FDIV(FMUL(FMUL(__uint_as_float(0x3f1f3b64u),slv),ea),__uint_as_float(0x43e6c148u)),ta),ta),1013);
  // source line 4413
  delta=FMUL(FDIV(slv,cpp),desdt);
  // source line 4414
  rr=FADD(FDIV(FMUL(FDIV(FMUL(FMUL(FMUL(__uint_as_float(0x40800000u),epsv),sigma),__uint_as_float(0x438f851fu)),cpp),gfk_pow(ta,__uint_as_float(0x40800000u))),FMUL(ts,ch)),__uint_as_float(0x3f800000u));
  // source line 4415
  pc=FDIV(FADD(rr,delta),FADD(FMUL(rr,FADD(__uint_as_float(0x3f800000u),FMUL(rc,ch))),delta));
  // source line 4417
  if (((cmc!=__uint_as_float(0x00000000u)))) {
  // source line 4418
  ett1=FMUL(FMUL(FMUL(FMUL(shdfac,pc),etp1),u480_sub(__uint_as_float(0x3f800000u),gfk_pow(FDIV(cmc,cmcmax),cfactr))),__uint_as_float(0x3a83126fu));
  // source line 4419
  } else {
  // source line 4420
  ett1=FMUL(FMUL(FMUL(shdfac,pc),etp1),__uint_as_float(0x3a83126fu));
  // source line 4421
  }
  // source line 4423
  denom=__uint_as_float(0x00000000u);
  // source line 4424
  for (k=1;k<=nroot;k+=1) {
  // source line 4425
  rtx=u480_sub(FADD(FDIV((-dzvr[(k)-1]),zsoil[(3)-1]),gx[(k)-1]),sgx);
  // source line 4426
  gx[(k)-1]=FMUL(gx[(k)-1],u_max(rtx,__uint_as_float(0x00000000u)));
  // source line 4427
  denom=FADD(denom,gx[(k)-1]);
  // source line 4428
  }
  // source line 4429
  if (((denom<=__uint_as_float(0x00000000u)))) {
  denom=__uint_as_float(0x3f800000u);
  }
  // source line 4431
  for (k=1;k<=nroot;k+=1) {
  // source line 4432
  et[(k)-1]=FDIV(FMUL(ett1,gx[(k)-1]),denom);
  // source line 4433
  ett=FADD(ett,et[(k)-1]);
  // source line 4434
  }
  // source line 4437
  if (((cmc>__uint_as_float(0x00000000u)))) {
  // source line 4438
  ec=FMUL(FMUL(FMUL(shdfac,gfk_pow(FDIV(cmc,cmcmax),cfactr)),etp1),__uint_as_float(0x3a83126fu));
  // source line 4439
  } else {
  // source line 4440
  ec=__uint_as_float(0x00000000u);
  // source line 4441
  }
  // source line 4442
  cmc2ms=FDIV(cmc,delt);
  // source line 4443
  ec=u_min(cmc2ms,ec);
}

__device__ void u480_smflx(const float* smcp,float* smc,const int& nsoil,const float& cmcp,float& cmc,const float& dt,const float& prcp1,const float* zsoil,const float& smcmax,const float& bexp,const float& smcwlt,const float* sroot,const float& dksat,const float& dwsat,const float& shdfac,const float& cmcmax,float& runoff1,float& runoff2,float& runoff3,const float& edir,const float& ec,const float* et,float& drip,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  int i={};
  int k={};
  float ai[4]={};
  float bi[4]={};
  float ci[4]={};
  float stcf[4]={};
  float rhsts[4]={};
  float rhstt[4]={};
  float excess={};
  float pcpdrp={};
  float rhsct={};
  float trhsct={};
  // source line 4478
  rhsct=u480_sub(FDIV(FMUL(FMUL(shdfac,prcp1),__uint_as_float(0x3a83126fu)),__uint_as_float(0x45610000u)),ec);
  // source line 4479
  drip=__uint_as_float(0x00000000u);
  // source line 4480
  trhsct=FMUL(dt,rhsct);
  // source line 4481
  excess=FADD(cmcp,trhsct);
  // source line 4487
  if (((excess>cmcmax))) {
  drip=u480_sub(excess,cmcmax);
  }
  // source line 4488
  pcpdrp=FADD(FDIV(FMUL(FMUL(u480_sub(__uint_as_float(0x3f800000u),shdfac),prcp1),__uint_as_float(0x3a83126fu)),__uint_as_float(0x45610000u)),FDIV(drip,dt));
  // source line 4494
  u480_srt(rhstt,edir,et,smcp,nsoil,pcpdrp,zsoil,dwsat,dksat,smcmax,bexp,runoff1,runoff2,dt,smcwlt,sroot,ai,bi,ci,env,status);
  // source line 4497
  u480_sstep(smcp,smc,cmcp,cmc,rhstt,rhsct,dt,nsoil,smcmax,cmcmax,runoff3,zsoil,ai,bi,ci,env,status);
}

__device__ void u480_srt(float* rhstt,const float& edir,const float* et,const float* smcp,const int& nsoil,const float& pcpdrp,const float* zsoil,const float& dwsat,const float& dksat,const float& smcmax,const float& bexp,float& runoff1,float& runoff2,const float& dt,const float& smcwlt,const float* sroot,float* ai,float* bi,float* ci,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  int k={};
  int ks={};
  float ddmax[4]={};
  float dd={};
  float ddt={};
  float ddz={};
  float ddz2={};
  float denom={};
  float denom2={};
  float dsmdz={};
  float dsmdz2={};
  float dt1={};
  float infmax={};
  float mxsmc={};
  float mxsmc2={};
  float numer={};
  float pddum={};
  float px={};
  float smcav={};
  float sstt={};
  float par={};
  float val={};
  float wcnd={};
  float wcnd2={};
  float wdf={};
  float wdf2={};
  float kdt={};
  // source line 4535
  pddum=pcpdrp;
  // source line 4536
  runoff1=__uint_as_float(0x00000000u);
  // source line 4537
  par=__uint_as_float(0x360637bdu);
  // source line 4539
  if (((pcpdrp!=__uint_as_float(0x00000000u)))) {
  // source line 4540
  smcav=u480_sub(smcmax,smcwlt);
  // source line 4541
  ddmax[(1)-1]=FMUL((-zsoil[(1)-1]),smcav);
  // source line 4542
  ddmax[(1)-1]=FMUL(ddmax[(1)-1],u480_sub(__uint_as_float(0x3f800000u),FDIV(u480_sub(smcp[(1)-1],smcwlt),smcav)));
  // source line 4543
  ddmax[(2)-1]=FMUL(u480_sub(zsoil[(1)-1],zsoil[(2)-1]),smcav);
  // source line 4544
  ddmax[(2)-1]=FMUL(ddmax[(2)-1],u480_sub(__uint_as_float(0x3f800000u),FDIV(u480_sub(smcp[(2)-1],smcwlt),smcav)));
  // source line 4545
  ddmax[(3)-1]=FMUL(u480_sub(zsoil[(2)-1],zsoil[(3)-1]),smcav);
  // source line 4546
  ddmax[(3)-1]=FMUL(ddmax[(3)-1],u480_sub(__uint_as_float(0x3f800000u),FDIV(u480_sub(smcp[(3)-1],smcwlt),smcav)));
  // source line 4548
  dd=FADD(FADD(ddmax[(1)-1],ddmax[(2)-1]),ddmax[(3)-1]);
  // source line 4549
  dt1=FDIV(dt,86400);
  // source line 4550
  kdt=FDIV(FMUL(__uint_as_float(0x40400000u),dksat),par);
  // source line 4551
  val=u480_sub(__uint_as_float(0x3f800000u),gfk_exp(FMUL((-kdt),dt1)));
  // source line 4552
  ddt=FMUL(dd,val);
  // source line 4553
  px=FMUL(pcpdrp,dt);
  // source line 4554
  if (((px<__uint_as_float(0x00000000u)))) {
  px=__uint_as_float(0x00000000u);
  }
  // source line 4556
  infmax=FDIV(FMUL(px,FDIV(ddt,FADD(px,ddt))),dt);
  // source line 4557
  mxsmc=smcp[(1)-1];
  // source line 4558
  u480_wdfcnd(wdf,wcnd,mxsmc,smcmax,bexp,dksat,dwsat,env,status);
  // source line 4559
  infmax=u_max(infmax,wcnd);
  // source line 4560
  infmax=u_min(infmax,FDIV(px,dt));
  // source line 4563
  if (((pcpdrp>infmax))) {
  // source line 4564
  runoff1=u480_sub(pcpdrp,infmax);
  // source line 4565
  pddum=infmax;
  // source line 4566
  }
  // source line 4567
  }
  // source line 4571
  u480_wdfcnd(wdf,wcnd,smcp[(1)-1],smcmax,bexp,dksat,dwsat,env,status);
  // source line 4572
  ddz=FDIV(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0xbf000000u),zsoil[(2)-1]));
  // source line 4573
  ai[(1)-1]=__uint_as_float(0x00000000u);
  // source line 4574
  bi[(1)-1]=FDIV(FMUL(wdf,ddz),(-zsoil[(1)-1]));
  // source line 4575
  ci[(1)-1]=(-bi[(1)-1]);
  // source line 4576
  dsmdz=FDIV(u480_sub(smcp[(1)-1],smcp[(2)-1]),FMUL(__uint_as_float(0xbf000000u),zsoil[(2)-1]));
  // source line 4577
  rhstt[(1)-1]=FDIV(FADD(FADD(FADD(u480_sub(FADD(FMUL(wdf,dsmdz),wcnd),pddum),edir),et[(1)-1]),sroot[(1)-1]),zsoil[(1)-1]);
  // source line 4578
  sstt=FADD(FADD(FADD(FMUL(wdf,dsmdz),wcnd),edir),et[(1)-1]);
  // source line 4583
  ddz2=__uint_as_float(0x00000000u);
  // source line 4584
  for (k=2;k<=(nsoil-1);k+=1) {
  // source line 4585
  denom2=u480_sub(zsoil[((k-1))-1],zsoil[(k)-1]);
  // source line 4586
  if (((k!=(nsoil-1)))) {
  // source line 4587
  mxsmc2=smcp[(k)-1];
  // source line 4588
  u480_wdfcnd(wdf2,wcnd2,mxsmc2,smcmax,bexp,dksat,dwsat,env,status);
  // source line 4589
  denom=u480_sub(zsoil[((k-1))-1],zsoil[((k+1))-1]);
  // source line 4590
  dsmdz2=FDIV(u480_sub(smcp[(k)-1],smcp[((k+1))-1]),FMUL(denom,__uint_as_float(0x3f000000u)));
  // source line 4591
  ddz2=FDIV(__uint_as_float(0x40000000u),denom);
  // source line 4592
  ci[(k)-1]=FDIV(FMUL((-wdf2),ddz2),denom2);
  // source line 4593
  } else {
  // source line 4594
  u480_wdfcnd(wdf2,wcnd2,smcp[((nsoil-1))-1],smcmax,bexp,dksat,dwsat,env,status);
  // source line 4595
  dsmdz2=__uint_as_float(0x00000000u);
  // source line 4596
  ci[(k)-1]=__uint_as_float(0x00000000u);
  // source line 4597
  }
  // source line 4598
  numer=FADD(FADD(u480_sub(u480_sub(FMUL(wdf2,dsmdz2),FMUL(wdf,dsmdz)),wcnd),et[(k)-1]),sroot[(k)-1]);
  // source line 4600
  rhstt[(k)-1]=FDIV(numer,(-denom2));
  // source line 4601
  ai[(k)-1]=FDIV(FMUL((-wdf),ddz),denom2);
  // source line 4602
  bi[(k)-1]=(-FADD(ai[(k)-1],ci[(k)-1]));
  // source line 4603
  if (((k==(nsoil-1)))) {
  // source line 4604
  runoff2=__uint_as_float(0x00000000u);
  // source line 4605
  }
  // source line 4606
  if (((k!=(nsoil-1)))) {
  // source line 4607
  wdf=wdf2;
  // source line 4608
  wcnd=wcnd2;
  // source line 4609
  dsmdz=dsmdz2;
  // source line 4610
  ddz=ddz2;
  // source line 4611
  }
  // source line 4612
  }
}

__device__ void u480_sstep(const float* smcp,float* smc,const float& cmcp,float& cmc,float* rhstt,const float& rhsct,const float& dt,const int& nsoil,const float& smcmax,const float& cmcmax,float& runoff3,const float* zsoil,float* ai,float* bi,float* ci,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  int i={};
  int k={};
  int kk11={};
  float rhsttin[4]={};
  float smcout[4]={};
  float smcin[4]={};
  float ciin[4]={};
  float ddz={};
  float wplus={};
  float stot={};
  // source line 4647
  for (k=1;k<=(nsoil-1);k+=1) {
  // source line 4648
  rhstt[(k)-1]=FMUL(rhstt[(k)-1],dt);
  // source line 4649
  ai[(k)-1]=FMUL(ai[(k)-1],dt);
  // source line 4650
  bi[(k)-1]=FADD(__uint_as_float(0x3f800000u),FMUL(bi[(k)-1],dt));
  // source line 4651
  ci[(k)-1]=FMUL(ci[(k)-1],dt);
  // source line 4652
  }
  // source line 4656
  for (k=1;k<=(nsoil-1);k+=1) {
  // source line 4657
  rhsttin[(k)-1]=rhstt[(k)-1];
  // source line 4658
  }
  // source line 4659
  for (k=1;k<=(nsoil-1);k+=1) {
  // source line 4660
  ciin[(k)-1]=ci[(k)-1];
  // source line 4661
  }
  // source line 4665
  u480_rosr12(ci,ai,bi,ciin,rhsttin,rhstt,(nsoil-1),env,status);
  // source line 4671
  smc[(nsoil)-1]=smcp[(nsoil)-1];
  wplus=__uint_as_float(0x00000000u);
  // source line 4672
  runoff3=__uint_as_float(0x00000000u);
  // source line 4674
  ddz=(-zsoil[(1)-1]);
  // source line 4675
  for (k=1;k<=(nsoil-1);k+=1) {
  // source line 4676
  if (((k!=1))) {
  ddz=u480_sub(zsoil[((k-1))-1],zsoil[(k)-1]);
  }
  // source line 4677
  smcout[(k)-1]=FADD(FADD(smcp[(k)-1],ci[(k)-1]),FDIV(wplus,ddz));
  // source line 4678
  stot=smcout[(k)-1];
  // source line 4679
  if (((stot>smcmax))) {
  // source line 4680
  if (((k==1))) {
  // source line 4681
  ddz=(-zsoil[(1)-1]);
  // source line 4682
  } else {
  // source line 4683
  kk11=(k-1);
  // source line 4684
  ddz=FADD((-zsoil[(k)-1]),zsoil[(kk11)-1]);
  // source line 4685
  }
  // source line 4686
  wplus=FMUL(u480_sub(stot,smcmax),ddz);
  // source line 4687
  } else {
  // source line 4688
  smc[(nsoil)-1]=smcp[(nsoil)-1];
  wplus=__uint_as_float(0x00000000u);
  // source line 4689
  }
  // source line 4690
  smc[(k)-1]=u_max(u_min(stot,smcmax),__uint_as_float(0x3d8f5c29u));
  // source line 4691
  }
  // source line 4697
  runoff3=wplus;
  // source line 4698
  cmc=FADD(cmcp,FMUL(dt,rhsct));
  // source line 4699
  if (((cmc<__uint_as_float(0x1e3ce508u)))) {
  cmc=__uint_as_float(0x00000000u);
  }
  // source line 4700
  cmc=u_min(cmc,cmcmax);
}

__device__ void u480_wdfcnd(float& wdf,float& wcnd,const float& smc,const float& smcmax,const float& bexp,const float& dksat,const float& dwsat,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float expon={};
  float factr1={};
  float factr2={};
  // source line 4728
  factr1=FDIV(__uint_as_float(0x3d4ccccdu),smcmax);
  // source line 4733
  factr2=FDIV(smc,smcmax);
  // source line 4734
  factr1=u_min(factr1,factr2);
  // source line 4735
  expon=FADD(bexp,__uint_as_float(0x40000000u));
  // source line 4736
  wdf=FMUL(dwsat,gfk_pow(factr2,expon));
  // source line 4737
  expon=FADD(FMUL(__uint_as_float(0x40000000u),bexp),__uint_as_float(0x40400000u));
  // source line 4738
  wcnd=FMUL(dksat,gfk_pow(factr2,expon));
}

__device__ void u480_rosr12(float* p,const float* a,const float* b,float* c,const float* d,float* delta,const int& nsoil,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  int k={};
  int kk={};
  // source line 4773
  c[(nsoil)-1]=__uint_as_float(0x00000000u);
  // source line 4774
  p[(1)-1]=FDIV((-c[(1)-1]),b[(1)-1]);
  // source line 4775
  delta[(1)-1]=FDIV(d[(1)-1],b[(1)-1]);
  // source line 4776
  for (k=2;k<=nsoil;k+=1) {
  // source line 4777
  p[(k)-1]=FMUL((-c[(k)-1]),FDIV(__uint_as_float(0x3f800000u),FADD(b[(k)-1],FMUL(a[(k)-1],p[((k-1))-1]))));
  // source line 4778
  delta[(k)-1]=FMUL(u480_sub(d[(k)-1],FMUL(a[(k)-1],delta[((k-1))-1])),FDIV(__uint_as_float(0x3f800000u),FADD(b[(k)-1],FMUL(a[(k)-1],p[((k-1))-1]))));
  // source line 4780
  }
  // source line 4784
  p[(nsoil)-1]=delta[(nsoil)-1];
  // source line 4789
  for (k=2;k<=nsoil;k+=1) {
  // source line 4790
  kk=((nsoil-k)+1);
  // source line 4791
  p[(kk)-1]=FADD(FMUL(p[(kk)-1],p[((kk+1))-1]),delta[(kk)-1]);
  // source line 4792
  }
}

__device__ void u480_shflx(float& ssoil,float* stc,const float* smc,const float& smcmax,const int& nsoil,float& t1,const float& dt,const float& yy,const float& zz1,const float* zsoil,const float& tbot,const float& zbot,const float& smcwlt,const float& df1,const float& quartz,const float& csoil,const float& capr,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  int i={};
  float ai[4]={};
  float bi[4]={};
  float ci[4]={};
  float stcf[4]={};
  float rhsts[4]={};
  // source line 4826
  u480_hrt(rhsts,stc,smc,smcmax,nsoil,zsoil,yy,zz1,tbot,zbot,dt,df1,ai,bi,ci,quartz,csoil,capr,env,status);
  // source line 4829
  u480_hstep(stcf,stc,rhsts,dt,nsoil,ai,bi,ci,env,status);
  // source line 4831
  for (i=1;i<=nsoil;i+=1) {
  // source line 4832
  stc[(i)-1]=stcf[(i)-1];
  // source line 4833
  }
  // source line 4838
  t1=FDIV(FADD(yy,FMUL(u480_sub(zz1,__uint_as_float(0x3f800000u)),stc[(1)-1])),zz1);
  // source line 4839
  ssoil=FDIV(FMUL(df1,u480_sub(stc[(1)-1],t1)),FMUL(__uint_as_float(0x3f000000u),zsoil[(1)-1]));
}

__device__ void u480_hrt(float* rhsts,const float* stc,const float* smc,const float& smcmax,const int& nsoil,const float* zsoil,const float& yy,const float& zz1,const float& tbot,const float& zbot,const float& dt,const float& df1,float* ai,float* bi,float* ci,const float& quartz,const float& csoil,const float& capr,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  bool itavg={};
  int i={};
  int k={};
  float ddz={};
  float ddz2={};
  float denom={};
  float df1k={};
  float dtsdz={};
  float df1n={};
  float dtsdz2={};
  float hcpct={};
  float qtot={};
  float ssoil={};
  float sice={};
  float tavg={};
  float tbk={};
  float tbk1={};
  float tsnsr={};
  float tsurf={};
  const float cair=__uint_as_float(0x447b0000u);
  const float ch2o=__uint_as_float(0x4a802c80u);
  // source line 4872
  itavg=true;
  // source line 4877
  hcpct=FADD(FADD(FMUL(smc[(1)-1],__uint_as_float(0x4a802c80u)),FMUL(u480_sub(__uint_as_float(0x3f800000u),smcmax),csoil)),FMUL(u480_sub(smcmax,smc[(1)-1]),__uint_as_float(0x447b0000u)));
  // source line 4879
  ddz=FDIV(__uint_as_float(0x3f800000u),FMUL(__uint_as_float(0xbf000000u),zsoil[(2)-1]));
  // source line 4880
  ai[(1)-1]=__uint_as_float(0x00000000u);
  // source line 4881
  ci[(1)-1]=FDIV(FMUL(df1,ddz),FMUL(zsoil[(1)-1],hcpct));
  // source line 4887
  bi[(1)-1]=FADD((-ci[(1)-1]),FDIV(df1,FMUL(FMUL(FMUL(FMUL(__uint_as_float(0x3f000000u),zsoil[(1)-1]),zsoil[(1)-1]),hcpct),zz1)));
  // source line 4889
  dtsdz=FDIV(u480_sub(stc[(1)-1],stc[(2)-1]),FMUL(__uint_as_float(0xbf000000u),zsoil[(2)-1]));
  // source line 4890
  ssoil=FDIV(FMUL(df1,u480_sub(stc[(1)-1],yy)),FMUL(FMUL(__uint_as_float(0x3f000000u),zsoil[(1)-1]),zz1));
  // source line 4891
  denom=FMUL(zsoil[(1)-1],hcpct);
  // source line 4897
  rhsts[(1)-1]=FDIV(u480_sub(FMUL(df1,dtsdz),ssoil),denom);
  // source line 4898
  qtot=FMUL(FMUL(__uint_as_float(0xbf800000u),rhsts[(1)-1]),denom);
  // source line 4899
  if (itavg) {
  // source line 4900
  tsurf=FDIV(FADD(yy,FMUL(u480_sub(zz1,1),stc[(1)-1])),zz1);
  // source line 4901
  u480_tbnd(stc[(1)-1],stc[(2)-1],zsoil,zbot,1,nsoil,tbk,env,status);
  // source line 4902
  }
  // source line 4903
  ddz2=__uint_as_float(0x00000000u);
  // source line 4904
  df1n=df1;
  // source line 4910
  for (k=2;k<=nsoil;k+=1) {
  // source line 4914
  if (((k<(nsoil-1)))) {
  // source line 4915
  hcpct=FADD(FADD(FMUL(smc[(k)-1],__uint_as_float(0x4a802c80u)),FMUL(u480_sub(__uint_as_float(0x3f800000u),smcmax),csoil)),FMUL(u480_sub(smcmax,smc[(k)-1]),__uint_as_float(0x447b0000u)));
  // source line 4917
  u480_tdfcnd(df1k,smc[(k)-1],quartz,smcmax,env,status);
  // source line 4918
  denom=FMUL(__uint_as_float(0x3f000000u),u480_sub(zsoil[((k-1))-1],zsoil[((k+1))-1]));
  // source line 4919
  dtsdz2=FDIV(u480_sub(stc[(k)-1],stc[((k+1))-1]),denom);
  // source line 4920
  ddz2=FDIV(__uint_as_float(0x40000000u),u480_sub(zsoil[((k-1))-1],zsoil[((k+1))-1]));
  // source line 4926
  ci[(k)-1]=FDIV(FMUL((-df1k),ddz2),FMUL(u480_sub(zsoil[((k-1))-1],zsoil[(k)-1]),hcpct));
  // source line 4928
  if (itavg) {
  // source line 4929
  u480_tbnd(stc[(k)-1],stc[((k+1))-1],zsoil,zbot,k,nsoil,tbk1,env,status);
  // source line 4930
  }
  // source line 4932
  } else if (((k==(nsoil-1)))) {
  // source line 4934
  hcpct=FADD(FADD(FMUL(smc[(k)-1],__uint_as_float(0x4a802c80u)),FMUL(u480_sub(__uint_as_float(0x3f800000u),smcmax),csoil)),FMUL(u480_sub(smcmax,smc[(k)-1]),__uint_as_float(0x447b0000u)));
  // source line 4936
  u480_tdfcnd(df1k,smc[(k)-1],quartz,smcmax,env,status);
  // source line 4937
  denom=FMUL(__uint_as_float(0x3f000000u),u480_sub(zsoil[((k-1))-1],zsoil[((k+1))-1]));
  // source line 4938
  dtsdz2=FDIV(u480_sub(stc[(k)-1],stc[((k+1))-1]),denom);
  // source line 4939
  ddz2=FDIV(__uint_as_float(0x40000000u),u480_sub(zsoil[((k-1))-1],zsoil[((k+1))-1]));
  // source line 4944
  ci[(k)-1]=FDIV(FMUL((-df1k),ddz2),FMUL(u480_sub(zsoil[((k-1))-1],zsoil[(k)-1]),hcpct));
  // source line 4946
  if (itavg) {
  // source line 4947
  u480_tbnd(stc[(k)-1],tbot,zsoil,zbot,k,nsoil,tbk1,env,status);
  // source line 4948
  }
  // source line 4949
  } else {
  // source line 4953
  hcpct=FMUL(FMUL(capr,__uint_as_float(0x4085fa44u)),__uint_as_float(0x49742400u));
  // source line 4954
  df1k=__uint_as_float(0x404f5c29u);
  // source line 4958
  denom=u480_sub(FMUL(__uint_as_float(0x3f000000u),FADD(zsoil[((k-1))-1],zsoil[(k)-1])),zbot);
  // source line 4959
  dtsdz2=FDIV(u480_sub(stc[(k)-1],tbot),denom);
  // source line 4964
  ci[(k)-1]=__uint_as_float(0x00000000u);
  // source line 4965
  if (itavg) {
  // source line 4966
  u480_tbnd(stc[(k)-1],tbot,zsoil,zbot,k,nsoil,tbk1,env,status);
  // source line 4967
  }
  // source line 4970
  }
  // source line 4974
  denom=FMUL(u480_sub(zsoil[(k)-1],zsoil[((k-1))-1]),hcpct);
  // source line 4975
  rhsts[(k)-1]=FDIV(u480_sub(FMUL(df1k,dtsdz2),FMUL(df1n,dtsdz)),denom);
  // source line 4976
  qtot=FMUL(FMUL(__uint_as_float(0xbf800000u),denom),rhsts[(k)-1]);
  // source line 4981
  ai[(k)-1]=FDIV(FMUL((-df1n),ddz),FMUL(u480_sub(zsoil[((k-1))-1],zsoil[(k)-1]),hcpct));
  // source line 4986
  bi[(k)-1]=(-FADD(ai[(k)-1],ci[(k)-1]));
  // source line 4987
  tbk=tbk1;
  // source line 4988
  df1n=df1k;
  // source line 4989
  dtsdz=dtsdz2;
  // source line 4990
  ddz=ddz2;
  // source line 4991
  }
}

__device__ void u480_hstep(float* stcout,const float* stcin,float* rhsts,const float& dt,const int& nsoil,float* ai,float* bi,float* ci,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  int k={};
  float rhstsin[4]={};
  float ciin[4]={};
  // source line 5014
  for (k=1;k<=nsoil;k+=1) {
  // source line 5015
  rhsts[(k)-1]=FMUL(rhsts[(k)-1],dt);
  // source line 5016
  ai[(k)-1]=FMUL(ai[(k)-1],dt);
  // source line 5017
  bi[(k)-1]=FADD(__uint_as_float(0x3f800000u),FMUL(bi[(k)-1],dt));
  // source line 5018
  ci[(k)-1]=FMUL(ci[(k)-1],dt);
  // source line 5019
  }
  // source line 5023
  for (k=1;k<=nsoil;k+=1) {
  // source line 5024
  rhstsin[(k)-1]=rhsts[(k)-1];
  // source line 5025
  }
  // source line 5026
  for (k=1;k<=nsoil;k+=1) {
  // source line 5027
  ciin[(k)-1]=ci[(k)-1];
  // source line 5028
  }
  // source line 5032
  u480_rosr12(ci,ai,bi,ciin,rhstsin,rhsts,nsoil,env,status);
  // source line 5036
  for (k=1;k<=nsoil;k+=1) {
  // source line 5037
  stcout[(k)-1]=FADD(stcin[(k)-1],ci[(k)-1]);
  // source line 5038
  }
}

__device__ void u480_tbnd(const float& tu,const float& tb,const float* zsoil,const float& zbot,const int& k,const int& nsoil,float& tbnd1,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float zb={};
  float zup={};
  // source line 5063
  if (((k==1))) {
  // source line 5064
  zup=__uint_as_float(0x00000000u);
  // source line 5065
  } else {
  // source line 5066
  zup=zsoil[((k-1))-1];
  // source line 5067
  }
  // source line 5072
  if (((k==nsoil))) {
  // source line 5073
  zb=u480_sub(FMUL(__uint_as_float(0x40000000u),zbot),zsoil[(k)-1]);
  // source line 5074
  } else {
  // source line 5075
  zb=zsoil[((k+1))-1];
  // source line 5076
  }
  // source line 5081
  tbnd1=FADD(tu,FDIV(FMUL(u480_sub(tb,tu),u480_sub(zup,zsoil[(k)-1])),u480_sub(zup,zb)));
}

__device__ void u480_tdfcnd(float& df,const float& smc,const float& qz,const float& smcmax,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float ake={};
  float gammd={};
  float thkdry={};
  float thko={};
  float thkqtz={};
  float thksat={};
  float thks={};
  float thkw={};
  float satratio={};
  // source line 5126
  satratio=FDIV(smc,smcmax);
  // source line 5128
  thkw=__uint_as_float(0x3f11eb85u);
  // source line 5131
  thko=__uint_as_float(0x40000000u);
  // source line 5133
  thkqtz=__uint_as_float(0x40f66666u);
  // source line 5135
  thks=FMUL(gfk_pow(thkqtz,qz),gfk_pow(thko,u480_sub(__uint_as_float(0x3f800000u),qz)));
  // source line 5138
  thksat=FMUL(gfk_pow(thks,u480_sub(__uint_as_float(0x3f800000u),smcmax)),gfk_pow(thkw,smcmax));
  // source line 5141
  gammd=FMUL(u480_sub(__uint_as_float(0x3f800000u),smcmax),__uint_as_float(0x4528c000u));
  // source line 5144
  thkdry=FDIV(FADD(FMUL(__uint_as_float(0x3e0a3d71u),gammd),__uint_as_float(0x42816666u)),u480_sub(__uint_as_float(0x4528c000u),FMUL(__uint_as_float(0x3f726e98u),gammd)));
  // source line 5150
  if (((satratio>__uint_as_float(0x3dcccccdu)))) {
  // source line 5152
  ake=FADD(u_log10(satratio),__uint_as_float(0x3f800000u));
  // source line 5155
  } else {
  // source line 5157
  ake=__uint_as_float(0x00000000u);
  // source line 5158
  }
  // source line 5161
  df=FADD(FMUL(ake,u480_sub(thksat,thkdry)),thkdry);
}

__device__ void u480_shadow_tree(const float& xi,const float& h,const float& w,const float& htree,const float& rtree,const float& dtree,float& chi_shaded,float& eta_shaded,float& chi_shaded_tree,float& eta_shaded_tree,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float xi_a={};
  float xi_b={};
  float x0={};
  float y0={};
  float x1={};
  float x2={};
  float y1={};
  float y2={};
  float chi_tree={};
  float eta_tree={};
  // source line 5192
  xi_a=FSQRT(FADD(1,u_powi(xi,2)));
  // source line 5193
  xi_b=FSQRT(FADD(1,u_powi(xi,-2)));
  // source line 5195
  x0=u_max(__uint_as_float(0x00000000u),u480_sub(w,FMUL(h,xi)));
  // source line 5196
  y0=u_max(__uint_as_float(0x00000000u),u480_sub(h,FDIV(w,xi)));
  // source line 5198
  x1=u_max(__uint_as_float(0x00000000u),u480_sub(u480_sub(dtree,FMUL(htree,xi)),FMUL(rtree,xi_a)));
  // source line 5199
  x2=u_max(__uint_as_float(0x00000000u),FADD(u480_sub(dtree,FMUL(htree,xi)),FMUL(rtree,xi_a)));
  // source line 5201
  y1=u_max(__uint_as_float(0x00000000u),u480_sub(u480_sub(htree,FDIV(dtree,xi)),FMUL(rtree,xi_b)));
  // source line 5202
  y2=u_max(__uint_as_float(0x00000000u),FADD(u480_sub(htree,FDIV(dtree,xi)),FMUL(rtree,xi_b)));
  // source line 5204
  chi_tree=u480_sub(x2,x1);
  // source line 5205
  eta_tree=u480_sub(y2,y1);
  // source line 5210
  if (((x2<=x0))) {
  // source line 5211
  chi_shaded=FADD(u480_sub(w,x0),chi_tree);
  // source line 5212
  chi_shaded_tree=chi_tree;
  // source line 5213
  } else if ((((x1<=x0)) && ((x0<x2)))) {
  // source line 5214
  chi_shaded=u480_sub(FADD(u480_sub(w,x0),chi_tree),u480_sub(x2,x0));
  // source line 5215
  chi_shaded_tree=u480_sub(chi_tree,u480_sub(x2,x0));
  // source line 5216
  } else {
  // source line 5217
  chi_shaded=u480_sub(w,x0);
  // source line 5218
  chi_shaded_tree=0;
  // source line 5219
  }
  // source line 5224
  if (((y2<=y0))) {
  // source line 5225
  eta_shaded=y0;
  // source line 5226
  eta_shaded_tree=0;
  // source line 5227
  } else if ((((y1<=y0)) && ((y0<y2)))) {
  // source line 5228
  eta_shaded=y2;
  // source line 5229
  eta_shaded_tree=u480_sub(y2,y0);
  // source line 5230
  } else {
  // source line 5231
  eta_shaded=FADD(y0,eta_tree);
  // source line 5232
  eta_shaded_tree=eta_tree;
  // source line 5233
  }
}

__device__ void u480_stomatal(float& rs,const float& rsmin,const float& sd_facet,const float& lai,const float& w2,const float& wr,const float& ws,const float& ta,const float& ts,const float& qa,const float& qsat,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float f={};
  float fsr={};
  float f_theta={};
  float tsc={};
  float fe={};
  float ft={};
  float lai_eff={};
  const float sd_lim=__uint_as_float(0x41f00000u);
  const float rsmax=__uint_as_float(0x459c4000u);
  const float hs=__uint_as_float(0x425a1eb8u);
  // source line 5253
  lai_eff=lai;
  // source line 5254
  if (((lai_eff<__uint_as_float(0x3c23d70au)))) {
  // source line 5255
  lai_eff=__uint_as_float(0x3c23d70au);
  // source line 5256
  }
  // source line 5258
  f=FDIV(FMUL(FMUL(__uint_as_float(0x3f0ccccdu),FDIV(sd_facet,__uint_as_float(0x41f00000u))),2),lai_eff);
  // source line 5259
  fsr=FDIV(FADD(f,FDIV(rsmin,__uint_as_float(0x459c4000u))),FADD(1,f));
  // source line 5260
  fsr=u_max(fsr,__uint_as_float(0x38d1b717u));
  // source line 5262
  if (((w2>FMUL(__uint_as_float(0x3f400000u),ws)))) {
  // source line 5263
  f_theta=1;
  // source line 5264
  } else if ((((w2>=wr)) && ((w2<=FMUL(__uint_as_float(0x3f400000u),ws))))) {
  // source line 5265
  f_theta=FDIV(u480_sub(w2,wr),u480_sub(FMUL(__uint_as_float(0x3f400000u),ws),wr));
  // source line 5266
  } else if (((w2<wr))) {
  // source line 5267
  f_theta=0;
  // source line 5268
  }
  // source line 5270
  f_theta=u_max(f_theta,__uint_as_float(0x38d1b717u));
  // source line 5272
  tsc=u480_sub(ts,__uint_as_float(0x43889333u));
  // source line 5274
  fe=FDIV(1,FADD(1,FMUL(__uint_as_float(0x425a1eb8u),u480_sub(qsat,qa))));
  // source line 5275
  fe=u_max(fe,__uint_as_float(0x3c23d70au));
  // source line 5277
  ft=u480_sub(1,FMUL(__uint_as_float(0x3ad1b717u),u_powi(u480_sub(298,ta),2)));
  // source line 5278
  ft=u_max(ft,__uint_as_float(0x38d1b717u));
  // source line 5280
  rs=FDIV(FDIV(FDIV(FDIV(FDIV(rsmin,lai),fsr),f_theta),fe),ft);
}

__device__ void u480_leleaf(float& le,const float& sw,const float& lw,const float& ra_leaf,const float& rs_leaf,const float& ra,const float& ta,const float& qa,const float& qsat,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float rn={};
  float tc={};
  float s={};
  float esat={};
  float rh={};
  float ea={};
  float d={};
  const float gamma=__uint_as_float(0x42843333u);
  const float cp=__uint_as_float(0x447b4000u);
  // source line 5301
  rn=FADD(sw,lw);
  // source line 5302
  tc=u480_sub(ta,__uint_as_float(0x43889333u));
  // source line 5304
  s=FMUL(FDIV(FDIV(__uint_as_float(0x4a191830u),FADD(tc,__uint_as_float(0x436d4ccdu))),FADD(tc,__uint_as_float(0x436d4ccdu))),gfk_exp(FDIV(FMUL(__uint_as_float(0x418a6666u),tc),FADD(tc,__uint_as_float(0x436d4ccdu)))));
  // source line 5305
  esat=FMUL(611,gfk_exp(FDIV(FMUL(__uint_as_float(0x418a6666u),tc),FADD(tc,__uint_as_float(0x436d4ccdu)))));
  // source line 5306
  rh=FDIV(qa,qsat);
  // source line 5307
  ea=FMUL(esat,rh);
  // source line 5308
  d=u480_sub(esat,ea);
  // source line 5310
  le=FDIV(FADD(FMUL(s,rn),FDIV(FMUL(FMUL(FMUL(__uint_as_float(0x3f6e147bu),ra),__uint_as_float(0x447b4000u)),d),ra_leaf)),FADD(s,FMUL(__uint_as_float(0x4275e45au),FADD(2,FDIV(rs_leaf,ra_leaf)))));
}

__device__ void u480_force_restore_tree(const float& delt,const float& s,const float& r,const float& h,const float& le,const float& lai,const float& tsp,float& ts,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float c={};
  // source line 5330
  c=640;
  // source line 5332
  ts=FADD(tsp,FDIV(FMUL(delt,u480_sub(u480_sub(FADD(s,r),h),le)),c));
}

__device__ void u480_root_uptake(float* sroot,const int& nlayer,const float* smc,const float& smcmax,const float& smcdry,const float& rtree,const float& lai_tree,const float* dzg,const float* zsoil,const float& ele,const float& fvg,const float& rw,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float normq[4]={};
  float alphai[4]={};
  float root[4]={};
  float part2[4]={};
  float alphabar={};
  float part1={};
  const float qc1=__uint_as_float(0x3f19999au);
  const float qc2=__uint_as_float(0x3f800000u);
  const float lv=__uint_as_float(0x4a09f080u);
  const float rwd=__uint_as_float(0x447a0000u);
  int k={};
  // source line 5357
  if ((((fvg<=__uint_as_float(0x00000000u))) || ((rw<=__uint_as_float(0x00000000u))) || ((rtree<=__uint_as_float(0x00000000u))) || ((lai_tree<=__uint_as_float(0x00000000u))))) {
  // source line 5358
  for(int vi=0;vi<4;++vi) sroot[vi]=__uint_as_float(0x00000000u);
  // source line 5359
  return;
  // source line 5360
  }
  // source line 5362
  for (k=1;k<=nlayer;k+=1) {
  // source line 5363
  normq[(k)-1]=FDIV(u480_sub(smc[(k)-1],smcdry),u480_sub(smcmax,smcdry));
  // source line 5364
  if (((normq[(k)-1]>__uint_as_float(0x3f800000u)))) {
  // source line 5365
  alphai[(k)-1]=__uint_as_float(0x3f800000u);
  // source line 5366
  } else if ((((normq[(k)-1]>=__uint_as_float(0x3f19999au))) && ((normq[(k)-1]<=__uint_as_float(0x3f800000u))))) {
  // source line 5367
  alphai[(k)-1]=__uint_as_float(0x3f800000u);
  // source line 5368
  } else if ((((normq[(k)-1]<__uint_as_float(0x3f19999au))) && ((normq[(k)-1]>=0)))) {
  // source line 5369
  alphai[(k)-1]=FDIV(normq[(k)-1],__uint_as_float(0x3f19999au));
  // source line 5370
  } else {
  // source line 5371
  alphai[(k)-1]=__uint_as_float(0x00000000u);
  // source line 5372
  }
  // source line 5373
  }
  // source line 5375
  for (k=1;k<=nlayer;k+=1) {
  // source line 5376
  root[(k)-1]=FDIV(dzg[(k)-1],(-zsoil[(nlayer)-1]));
  // source line 5377
  }
  // source line 5379
  alphabar=0;
  // source line 5380
  for (k=1;k<=nlayer;k+=1) {
  // source line 5381
  alphabar=FADD(alphabar,FMUL(root[(k)-1],alphai[(k)-1]));
  // source line 5382
  }
  // source line 5384
  if (((alphabar<=__uint_as_float(0x2b8cbcccu)))) {
  // source line 5385
  for(int vi=0;vi<4;++vi) sroot[vi]=__uint_as_float(0x00000000u);
  // source line 5386
  return;
  // source line 5387
  }
  // source line 5389
  for (k=1;k<=nlayer;k+=1) {
  // source line 5390
  part1=FDIV(FMUL(FMUL(FMUL(2,rtree),lai_tree),FDIV(ele,__uint_as_float(0x4a09f080u))),__uint_as_float(0x447a0000u));
  // source line 5391
  part2[(k)-1]=FDIV(FMUL(root[(k)-1],alphai[(k)-1]),alphabar);
  // source line 5392
  sroot[(k)-1]=FDIV(FMUL(part1,part2[(k)-1]),FMUL(fvg,rw));
  // source line 5393
  }
}

__device__ float u480_cdf2d_vf(const float& theta,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float anglecdf={};
  // source line 5613
  anglecdf=FMUL(__uint_as_float(0x3f000000u),u480_sub(__uint_as_float(0x3f800000u),glibc_cosf(theta)));
  return anglecdf;
}

__device__ void u480_vf_tree_analytical(const float& w,const float& h,const float& ht,const float& rt,const float& trans,float& fgs,float& fgw,float& fgt,float& fws,float& fwg,float& fww,float& fwt,float& fts,float& ftg,float& ftw,const U480Env& env,int& status) {
  const float piconst=__uint_as_float(0x40490fdbu);
  const int maxdirs=3;
  const int maxhgts=50;
  float dx={};
  float dy={};
  float tau={};
  float fww_emp={};
  float fwg_emp={};
  float fws_emp={};
  float fsg_emp={};
  float fsw_emp={};
  float fgs_emp={};
  float fgw_emp={};
  float fsw={};
  float fsg={};
  float fst={};
  float fww_cf1={};
  float fws_cf1={};
  float fwg_cf1={};
  float fww_cf2={};
  float fws_cf2={};
  float fwg_cf2={};
  float fgs_cf1={};
  float fgw_cf1={};
  float fgs_cf2={};
  float fgw_cf2={};
  float fsw_cf1={};
  float fsg_cf1={};
  float fsw_cf2={};
  float fsg_cf2={};
  float y={};
  float d1={};
  float theta_blocked_ww={};
  float theta_range_ww={};
  float theta_blocked_ws={};
  float theta_range_ws={};
  float theta_blocked_wg={};
  float theta_range_wg={};
  float x={};
  float d2={};
  float theta_blocked_gs={};
  float theta_range_gs={};
  float theta_blocked_gw={};
  float theta_range_gw={};
  float theta_blocked_sw={};
  float theta_range_sw={};
  float theta_blocked_sg={};
  float theta_range_sg={};
  float a_s={};
  float a_g={};
  float a_w={};
  float a_t={};
  float sum_t={};
  float resid_t={};
  float a_sum_t={};
  float sum_w={};
  float resid_w={};
  float a_sum_w={};
  float sum_g={};
  float resid_g={};
  float a_sum_g={};
  float sum_s={};
  float resid_s={};
  float a_sum_s={};
  const float pi=__uint_as_float(0x40490fdbu);
  int i={};
  // source line 5432
  if ((((FMUL(__uint_as_float(0x40000000u),rt)>w)) || ((FADD(ht,rt)>h)))) {
  // source line 5433
  status=1; return;
  // source line 5434
  }
  // source line 5437
  dx=FDIV(w,__uint_as_float(0x41a00000u));
  // source line 5438
  dy=FDIV(h,__uint_as_float(0x41a00000u));
  // source line 5440
  tau=trans;
  // source line 5443
  fww_emp=u480_sub(FSQRT(FADD(__uint_as_float(0x3f800000u),u_powi(FDIV(w,h),2))),FDIV(w,h));
  // source line 5444
  fwg_emp=FDIV(u480_sub(__uint_as_float(0x3f800000u),fww_emp),__uint_as_float(0x40000000u));
  // source line 5445
  fws_emp=fwg_emp;
  // source line 5446
  fsg_emp=u480_sub(FSQRT(FADD(__uint_as_float(0x3f800000u),u_powi(FDIV(h,w),2))),FDIV(h,w));
  // source line 5447
  fsw_emp=FDIV(u480_sub(__uint_as_float(0x3f800000u),fsg_emp),__uint_as_float(0x40000000u));
  // source line 5448
  fgs_emp=fsg_emp;
  // source line 5449
  fgw_emp=FDIV(u480_sub(__uint_as_float(0x3f800000u),fgs_emp),__uint_as_float(0x40000000u));
  // source line 5453
  ftw=FMUL(__uint_as_float(0x3ea2f983u),glibc_atanf(FDIV(h,w)));
  // source line 5454
  fts=FMUL(__uint_as_float(0x3ea2f983u),glibc_atanf(FDIV(w,FMUL(__uint_as_float(0x40000000u),u480_sub(h,ht)))));
  // source line 5455
  ftg=FMUL(__uint_as_float(0x3ea2f983u),glibc_atanf(FDIV(w,FMUL(__uint_as_float(0x40000000u),ht))));
  // source line 5458
  fwt=FMUL(FDIV(FMUL(__uint_as_float(0x40000000u),rt),h),glibc_atanf(FDIV(h,w)));
  // source line 5459
  fgt=FMUL(FDIV(FMUL(__uint_as_float(0x40000000u),rt),w),glibc_atanf(FDIV(w,FMUL(__uint_as_float(0x40000000u),ht))));
  // source line 5460
  fst=FMUL(FDIV(FMUL(__uint_as_float(0x40000000u),rt),w),glibc_atanf(FDIV(w,FMUL(__uint_as_float(0x40000000u),u480_sub(h,ht)))));
  // source line 5463
  fwt=FMUL(fwt,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5464
  fst=FMUL(fst,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5465
  fgt=FMUL(fgt,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5469
  fww_cf1=__uint_as_float(0x00000000u);
  // source line 5469
  fws_cf1=__uint_as_float(0x00000000u);
  // source line 5469
  fwg_cf1=__uint_as_float(0x00000000u);
  // source line 5470
  fww_cf2=__uint_as_float(0x00000000u);
  // source line 5470
  fws_cf2=__uint_as_float(0x00000000u);
  // source line 5470
  fwg_cf2=__uint_as_float(0x00000000u);
  // source line 5472
  for (i=1;i<=u480_nint(FDIV(h,dy));i+=1) {
  // source line 5473
  y=u480_sub(FMUL(dy,i),FDIV(dy,__uint_as_float(0x40000000u)));
  // source line 5474
  d1=FSQRT(FADD(u_powi(FDIV(w,__uint_as_float(0x40000000u)),2),u_powi(u480_sub(y,ht),2)));
  // source line 5475
  theta_blocked_ww=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(ht,y),FDIV(w,__uint_as_float(0x40000000u))))),glibc_asinf(FDIV(rt,d1))),FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(h,y),w)))),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(ht,y),FDIV(w,__uint_as_float(0x40000000u))))),glibc_asinf(FDIV(rt,d1))),glibc_atanf(FDIV(w,y))),env,status)));
  // source line 5479
  theta_range_ww=u480_sub(u480_cdf2d_vf(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(h,y),w))),env,status),u480_cdf2d_vf(glibc_atanf(FDIV(w,y)),env,status));
  // source line 5480
  theta_blocked_ww=FMUL(theta_blocked_ww,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5481
  fww_cf1=FADD(fww_cf1,FMUL(dy,u480_sub(theta_range_ww,theta_blocked_ww)));
  // source line 5482
  fww_cf2=FADD(fww_cf2,FMUL(dy,theta_range_ww));
  // source line 5484
  theta_blocked_ws=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(ht,y),FDIV(w,__uint_as_float(0x40000000u))))),glibc_asinf(FDIV(rt,d1))),__uint_as_float(0x40490fdbu)),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(ht,y),FDIV(w,__uint_as_float(0x40000000u))))),glibc_asinf(FDIV(rt,d1))),FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(h,y),w)))),env,status)));
  // source line 5488
  theta_range_ws=u480_sub(u480_cdf2d_vf(__uint_as_float(0x40490fdbu),env,status),u480_cdf2d_vf(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(h,y),w))),env,status));
  // source line 5489
  theta_blocked_ws=FMUL(theta_blocked_ws,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5490
  fws_cf1=FADD(fws_cf1,FMUL(dy,u480_sub(theta_range_ws,theta_blocked_ws)));
  // source line 5491
  fws_cf2=FADD(fws_cf2,FMUL(dy,theta_range_ws));
  // source line 5493
  theta_blocked_wg=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(ht,y),FDIV(w,__uint_as_float(0x40000000u))))),glibc_asinf(FDIV(rt,d1))),glibc_atanf(FDIV(w,y))),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(ht,y),FDIV(w,__uint_as_float(0x40000000u))))),glibc_asinf(FDIV(rt,d1))),__uint_as_float(0x00000000u)),env,status)));
  // source line 5497
  theta_range_wg=u480_sub(u480_cdf2d_vf(glibc_atanf(FDIV(w,y)),env,status),u480_cdf2d_vf(__uint_as_float(0x00000000u),env,status));
  // source line 5498
  theta_blocked_wg=FMUL(theta_blocked_wg,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5499
  fwg_cf1=FADD(fwg_cf1,FMUL(dy,u480_sub(theta_range_wg,theta_blocked_wg)));
  // source line 5500
  fwg_cf2=FADD(fwg_cf2,FMUL(dy,theta_range_wg));
  // source line 5501
  }
  // source line 5504
  if (((fww_cf2>__uint_as_float(0x00000000u)))) {
  fww=FDIV(FMUL(fww_emp,fww_cf1),fww_cf2);
  }
  // source line 5505
  if (((fws_cf2>__uint_as_float(0x00000000u)))) {
  fws=FDIV(FMUL(fws_emp,fws_cf1),fws_cf2);
  }
  // source line 5506
  if (((fwg_cf2>__uint_as_float(0x00000000u)))) {
  fwg=FDIV(FMUL(fwg_emp,fwg_cf1),fwg_cf2);
  }
  // source line 5509
  fgs_cf1=__uint_as_float(0x00000000u);
  // source line 5509
  fgw_cf1=__uint_as_float(0x00000000u);
  // source line 5510
  fgs_cf2=__uint_as_float(0x00000000u);
  // source line 5510
  fgw_cf2=__uint_as_float(0x00000000u);
  // source line 5512
  for (i=1;i<=u480_nint(FDIV(w,dx));i+=1) {
  // source line 5513
  x=u480_sub(FMUL(dx,i),FDIV(dx,__uint_as_float(0x40000000u)));
  // source line 5514
  d2=FSQRT(FADD(u_powi(u480_sub(FDIV(w,__uint_as_float(0x40000000u)),x),2),u_powi(ht,2)));
  // source line 5515
  theta_blocked_gs=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),ht))),glibc_asinf(FDIV(rt,d2))),FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(x,h)))),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),ht))),glibc_asinf(FDIV(rt,d2))),glibc_atanf(FDIV(h,u480_sub(w,x)))),env,status)));
  // source line 5519
  theta_range_gs=u480_sub(u480_cdf2d_vf(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(x,h))),env,status),u480_cdf2d_vf(glibc_atanf(FDIV(h,u480_sub(w,x))),env,status));
  // source line 5520
  theta_blocked_gs=FMUL(theta_blocked_gs,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5521
  fgs_cf1=FADD(fgs_cf1,FMUL(dx,u480_sub(theta_range_gs,theta_blocked_gs)));
  // source line 5522
  fgs_cf2=FADD(fgs_cf2,FMUL(dx,theta_range_gs));
  // source line 5524
  theta_blocked_gw=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),ht))),glibc_asinf(FDIV(rt,d2))),glibc_atanf(FDIV(h,u480_sub(w,x)))),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),ht))),glibc_asinf(FDIV(rt,d2))),__uint_as_float(0x00000000u)),env,status)));
  // source line 5528
  theta_range_gw=u480_sub(u480_cdf2d_vf(glibc_atanf(FDIV(h,u480_sub(w,x))),env,status),u480_cdf2d_vf(__uint_as_float(0x00000000u),env,status));
  // source line 5529
  theta_blocked_gw=FMUL(theta_blocked_gw,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5530
  fgw_cf1=FADD(fgw_cf1,FMUL(dx,u480_sub(theta_range_gw,theta_blocked_gw)));
  // source line 5531
  fgw_cf2=FADD(fgw_cf2,FMUL(dx,theta_range_gw));
  // source line 5532
  }
  // source line 5535
  if (((fgw_cf2>__uint_as_float(0x00000000u)))) {
  fgw=FDIV(FMUL(fgw_emp,fgw_cf1),fgw_cf2);
  }
  // source line 5536
  if (((fgs_cf2>__uint_as_float(0x00000000u)))) {
  fgs=FDIV(FMUL(fgs_emp,fgs_cf1),fgs_cf2);
  }
  // source line 5539
  fsw_cf1=__uint_as_float(0x00000000u);
  // source line 5539
  fsg_cf1=__uint_as_float(0x00000000u);
  // source line 5540
  fsw_cf2=__uint_as_float(0x00000000u);
  // source line 5540
  fsg_cf2=__uint_as_float(0x00000000u);
  // source line 5542
  for (i=1;i<=u480_nint(FDIV(w,dx));i+=1) {
  // source line 5543
  x=u480_sub(FMUL(dx,i),FDIV(dx,__uint_as_float(0x40000000u)));
  // source line 5544
  d2=FSQRT(FADD(u_powi(u480_sub(FDIV(w,__uint_as_float(0x40000000u)),x),2),u_powi(u480_sub(h,ht),2)));
  // source line 5545
  theta_blocked_sw=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),u480_sub(h,ht)))),glibc_asinf(FDIV(rt,d2))),glibc_atanf(FDIV(h,u480_sub(w,x)))),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),u480_sub(h,ht)))),glibc_asinf(FDIV(rt,d2))),__uint_as_float(0x00000000u)),env,status)));
  // source line 5549
  theta_range_sw=u480_sub(u480_cdf2d_vf(glibc_atanf(FDIV(h,u480_sub(w,x))),env,status),u480_cdf2d_vf(__uint_as_float(0x00000000u),env,status));
  // source line 5550
  theta_blocked_sw=FMUL(theta_blocked_sw,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5551
  fsw_cf1=FADD(fsw_cf1,FMUL(dx,u480_sub(theta_range_sw,theta_blocked_sw)));
  // source line 5552
  fsw_cf2=FADD(fsw_cf2,FMUL(dx,theta_range_sw));
  // source line 5554
  theta_blocked_sg=u_max(__uint_as_float(0x00000000u),u480_sub(u480_cdf2d_vf(u_min(FADD(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),u480_sub(h,ht)))),glibc_asinf(FDIV(rt,d2))),FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(x,h)))),env,status),u480_cdf2d_vf(u_max(u480_sub(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(u480_sub(x,FDIV(w,__uint_as_float(0x40000000u))),u480_sub(h,ht)))),glibc_asinf(FDIV(rt,d2))),glibc_atanf(FDIV(h,u480_sub(w,x)))),env,status)));
  // source line 5559
  theta_range_sg=u480_sub(u480_cdf2d_vf(FADD(__uint_as_float(0x3fc90fdbu),glibc_atanf(FDIV(x,h))),env,status),u480_cdf2d_vf(glibc_atanf(FDIV(h,u480_sub(w,x))),env,status));
  // source line 5560
  theta_blocked_sg=FMUL(theta_blocked_sg,u480_sub(__uint_as_float(0x3f800000u),tau));
  // source line 5561
  fsg_cf1=FADD(fsg_cf1,FMUL(dx,u480_sub(theta_range_sg,theta_blocked_sg)));
  // source line 5562
  fsg_cf2=FADD(fsg_cf2,FMUL(dx,theta_range_sg));
  // source line 5564
  }
  // source line 5566
  if (((fsw_cf2>__uint_as_float(0x00000000u)))) {
  fsw=FDIV(FMUL(fsw_emp,fsw_cf1),fsw_cf2);
  }
  // source line 5567
  if (((fsg_cf2>__uint_as_float(0x00000000u)))) {
  fsg=FDIV(FMUL(fsg_emp,fsg_cf1),fsg_cf2);
  }
  // source line 5571
  a_s=w;
  // source line 5572
  a_g=w;
  // source line 5573
  a_w=h;
  // source line 5574
  a_t=FMUL(__uint_as_float(0x40c90fdbu),rt);
  // source line 5577
  sum_t=FADD(FADD(fts,ftg),FMUL(__uint_as_float(0x40000000u),ftw));
  // source line 5578
  resid_t=u480_sub(__uint_as_float(0x3f800000u),sum_t);
  // source line 5579
  a_sum_t=FADD(FADD(a_s,a_g),FMUL(__uint_as_float(0x40000000u),a_w));
  // source line 5580
  fts=FADD(fts,FMUL(resid_t,FDIV(a_s,a_sum_t)));
  // source line 5581
  ftg=FADD(ftg,FMUL(resid_t,FDIV(a_g,a_sum_t)));
  // source line 5582
  ftw=FADD(ftw,FMUL(resid_t,FDIV(a_w,a_sum_t)));
  // source line 5585
  sum_w=FADD(FADD(FADD(fws,fwg),fww),fwt);
  // source line 5586
  resid_w=u480_sub(__uint_as_float(0x3f800000u),sum_w);
  // source line 5587
  a_sum_w=FADD(FADD(FADD(a_s,a_g),a_w),a_t);
  // source line 5588
  fws=FADD(fws,FMUL(resid_w,FDIV(a_s,a_sum_w)));
  // source line 5589
  fwg=FADD(fwg,FMUL(resid_w,FDIV(a_g,a_sum_w)));
  // source line 5590
  fww=FADD(fww,FMUL(resid_w,FDIV(a_w,a_sum_w)));
  // source line 5591
  fwt=FADD(fwt,FMUL(resid_w,FDIV(a_t,a_sum_w)));
  // source line 5594
  sum_g=FADD(FADD(fgs,FMUL(__uint_as_float(0x40000000u),fgw)),fgt);
  // source line 5595
  resid_g=u480_sub(__uint_as_float(0x3f800000u),sum_g);
  // source line 5596
  a_sum_g=FADD(FADD(a_s,FMUL(__uint_as_float(0x40000000u),a_w)),a_t);
  // source line 5597
  fgs=FADD(fgs,FMUL(resid_g,FDIV(a_s,a_sum_g)));
  // source line 5598
  fgw=FADD(fgw,FMUL(resid_g,FDIV(a_w,a_sum_g)));
  // source line 5599
  fgt=FADD(fgt,FMUL(resid_g,FDIV(a_t,a_sum_g)));
  // source line 5602
  sum_s=FADD(FADD(FMUL(__uint_as_float(0x40000000u),fsw),fsg),fst);
  // source line 5603
  resid_s=u480_sub(__uint_as_float(0x3f800000u),sum_s);
  // source line 5604
  a_sum_s=FADD(FADD(FMUL(__uint_as_float(0x40000000u),a_w),a_g),a_t);
  // source line 5605
  fsw=FADD(fsw,FMUL(resid_s,FDIV(a_w,a_sum_s)));
  // source line 5606
  fsg=FADD(fsg,FMUL(resid_s,FDIV(a_g,a_sum_s)));
  // source line 5607
  fst=FADD(fst,FMUL(resid_s,FDIV(a_t,a_sum_s)));
}
